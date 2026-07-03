"""Signal scorecard + simulated paper P&L — forward-tests the dashboard's calls.

Each DISTINCT signal the dashboard emits (de-duplicated: one per *new* signal,
not every poll) is logged to ``data/signal_log.csv`` and fed into a per-source
paper portfolio that starts at $10,000:

* BUY / bullish  -> go long;  SELL / bearish -> go short (or flat if shorting is
  off);  HOLD -> hold the current position.
* Fixed notional per trade (``trade_pct`` of the starting equity).
* Flip / close when the next *directional* signal differs; marked-to-market on
  every price update.
* Realistic COSTS are subtracted on every entry AND exit (round-trip bps) —
  without them the result is fiction, so they are mandatory.

Separately, every directional signal is scored correct / wrong / pending after a
fixed ``horizon`` (default 15 min) using only price observations at-or-after the
horizon (no lookahead) — that drives the hit-rate.

State persists to ``data/scorecard_state.json`` so it survives restarts and
accumulates overnight. Everything is clock- and dir-injectable for testing.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional


def _pid_alive(pid: int) -> bool:
    """True if process `pid` is running. Safe (never signals/kills the process)."""
    if pid == os.getpid():
        return True
    try:
        if os.name == "nt":
            import ctypes
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            h = ctypes.windll.kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not h:
                return False
            code = ctypes.c_ulong()
            ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
            ctypes.windll.kernel32.CloseHandle(h)
            return code.value == STILL_ACTIVE
        os.kill(pid, 0)          # POSIX: signal 0 only checks existence
        return True
    except (OSError, ValueError, Exception):
        return False

_LONG = {"buy", "bullish", "long", "up"}
_SHORT = {"sell", "bearish", "short", "down"}

# Per-market agents, each a single $10k book. All SIX run continuously and accumulate paper
# trades — kept live as honest DEMONSTRATIONS for stakeholders, each with a one-line technique
# + verdict (``desc``). The five PREDICTOR agents (US/crypto ML, TA chart, crypto order-book)
# were rigorously tested and found to have NO validated edge after costs, so they carry the
# ``retired`` flag (= "demo, no validated edge"); they are NOT stopped or hidden. The one real
# edge here is the delta-neutral funding CARRY. The validated/forward strategies (diversified
# trend core, frozen momentum satellite, lead-lag, PEAD, desk) have their own panels above.
# (US order-book is absent — free Alpaca is top-of-book only.)
DEFAULT_CONFIGS = {
    "us-ml":     {"label": "US ML model",        "cost_bps_round": 10.0, "allow_short": True, "retired": True,
                  "desc": "predicts next move via machine learning — tested: no edge after costs"},
    "us-ta":     {"label": "US TA chart",        "cost_bps_round": 10.0, "allow_short": True, "retired": True,
                  "desc": "candlestick/chart patterns — tested: ~50% hit-rate, costs win"},
    "crypto-ml": {"label": "Crypto ML model",    "cost_bps_round": 10.0, "allow_short": True, "retired": True,
                  "desc": "predicts next move via machine learning — tested: no edge after costs"},
    "crypto-ob": {"label": "Crypto order-book",  "cost_bps_round": 10.0, "allow_short": True, "retired": True,
                  "desc": "order-book imbalance — tested: churns to bust on costs"},
    "crypto-ta": {"label": "Crypto TA chart",    "cost_bps_round": 10.0, "allow_short": True, "retired": True,
                  "desc": "candlestick/chart patterns — tested: ~50% hit-rate, costs win"},
    # Not a prediction agent: a delta-neutral CARRY (long spot + short perp) that
    # accrues live 8h funding minus costs. The one real edge — held continuously
    # with hysteresis, it sits flat-to-slightly-positive while the predictors bleed.
    "funding-carry": {"label": "Funding carry (BTC/ETH)", "cost_bps_round": 10.0,
                      "allow_short": True, "carry": True,
                      "desc": "delta-neutral funding capture — small real edge"},
}


def _by_tag(recs) -> dict:
    """Per-tag (e.g. per candlestick pattern) hit-rate breakdown for a source.

    -> {tag: {total, correct, wrong, pending, hit_rate}}. Only tagged records
    (patterns) are broken out, so we can see if ANY single pattern beats 50%.
    """
    out: dict = {}
    for r in recs:
        tag = r.get("tag") or ""
        if not tag:
            continue
        d = out.setdefault(tag, {"total": 0, "correct": 0, "wrong": 0, "pending": 0})
        d["total"] += 1
        st = r.get("status")
        if st in ("correct", "wrong", "pending"):
            d[st] += 1
    for d in out.values():
        decided = d["correct"] + d["wrong"]
        d["hit_rate"] = round(100.0 * d["correct"] / decided, 1) if decided else None
    return out


def side_of(direction) -> int:
    """+1 long, -1 short, 0 flat/hold — from any direction label."""
    d = str(direction).strip().lower()
    if d in _LONG:
        return 1
    if d in _SHORT:
        return -1
    return 0


class PaperBook:
    """One simulated portfolio for a source. Fixed notional per trade, mandatory
    round-trip costs, mark-to-market equity. Gross P&L and costs tracked apart so
    ``equity = start + realized_gross - costs + unrealized_gross``."""

    def __init__(self, source: str, start: float = 10_000.0, trade_pct: float = 1.0,
                 cost_bps_round: float = 10.0, allow_short: bool = True):
        self.source = source
        self.start = float(start)
        # notional is a FRACTION of the book and never exceeds it (cap trade_pct at 1.0),
        # so a position is sized to the $10k base, not at full external (e.g. BTC) notional.
        self.trade_pct = min(1.0, float(trade_pct))
        self.cost_side = float(cost_bps_round) / 2.0 / 10_000.0   # fraction, per side
        self.allow_short = bool(allow_short)
        self.realized = 0.0
        self.costs = 0.0
        self.trades = 0
        self.positions: dict = {}   # symbol -> {side, qty, entry, notional}

    def _notional(self) -> float:
        return self.trade_pct * self.start

    def realized_equity(self) -> float:
        """Settled equity (no open marks). A book is BUST when this hits <= 0."""
        return self.start + self.realized - self.costs

    def set_target(self, symbol: str, side: int, price: float) -> None:
        """Move the position in ``symbol`` to ``side`` (closing the old one first),
        paying entry/exit costs. ``side==-1`` becomes flat when shorting is off. A BUST
        book (realized equity <= 0) stops opening new positions, so cumulative costs and
        losses can never exceed the $10k base by multiples (no unbounded churn)."""
        price = float(price)
        if side == -1 and not self.allow_short:
            side = 0
        cur = self.positions.get(symbol)
        if cur and cur["side"] != side:                     # close existing leg
            self.realized += cur["qty"] * (price - cur["entry"])
            self.costs += abs(cur["qty"]) * price * self.cost_side
            del self.positions[symbol]
        if side != 0 and symbol not in self.positions and self.realized_equity() > 0:
            notional = self._notional()                     # <= start (cap above); bust -> no open
            self.costs += notional * self.cost_side
            self.trades += 1
            self.positions[symbol] = {"side": side, "qty": side * notional / price,
                                      "entry": price, "notional": notional}

    def unrealized(self, last_price: dict) -> float:
        u = 0.0
        for sym, p in self.positions.items():
            lp = last_price.get(sym)
            if lp is not None:
                u += p["qty"] * (float(lp[0]) - p["entry"])
        return u

    def equity(self, last_price: dict) -> float:
        return self.start + self.realized - self.costs + self.unrealized(last_price)

    def to_dict(self) -> dict:
        return {"realized": self.realized, "costs": self.costs,
                "trades": self.trades, "positions": self.positions}

    def load(self, d: dict) -> None:
        self.realized = float(d.get("realized", 0.0))
        self.costs = float(d.get("costs", 0.0))
        self.trades = int(d.get("trades", 0))
        self.positions = d.get("positions", {}) or {}


class Scorecard:
    """Logs distinct signals, runs the per-source paper books, scores hit-rate."""

    CSV_FIELDS = ["timestamp", "source", "symbol", "direction", "price"]

    def __init__(self, data_dir="data", horizon_s: int = 900, trade_pct: float = 1.0,
                 clock=None, configs: Optional[dict] = None, reset: bool = False,
                 single_instance: bool = True, funding_enter_bps: float = 1.0,
                 funding_band_bps: float = 4.0, funding_cost_bps: float = 5.0):
        self.dir = Path(data_dir)
        self.csv_path = self.dir / "signal_log.csv"
        self.state_path = self.dir / "scorecard_state.json"
        self.lock_path = self.dir / "scorecard.lock"
        self.horizon = int(horizon_s)
        self.trade_pct = float(trade_pct)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.configs = configs or DEFAULT_CONFIGS
        self._lock = threading.RLock()
        self.books: dict = {}            # (source, symbol) -> PaperBook ($10k each)
        self.last_side: dict = {}
        self.scores: list = []
        self.last_price: dict = {}
        self.carry_state: dict = {}      # symbol -> {in_carry, last_ft} for funding-carry
        self.funding_enter_bps = float(funding_enter_bps)
        self.funding_band_bps = float(funding_band_bps)
        self.funding_cost_bps = float(funding_cost_bps)
        self.dir.mkdir(parents=True, exist_ok=True)
        if reset:                         # ONLY when explicitly asked
            self.reset_state()
        # Single-instance ownership: a second LIVE process won't write/clobber the
        # state or append the log. State is always RESUMED from disk on startup.
        self.owner = self._acquire_lock() if single_instance else True
        self._load()

    # ------------------------------------------------------------------ #
    def _acquire_lock(self) -> bool:
        if self.lock_path.exists():
            try:
                old = int(self.lock_path.read_text().strip())
            except Exception:
                old = None
            if old and old != os.getpid() and _pid_alive(old):
                sys.stderr.write(
                    f"[scorecard] another instance (pid {old}) owns {self.dir}; "
                    f"this process will read but NOT write state.\n")
                return False
        try:
            self.lock_path.write_text(str(os.getpid()))
        except Exception:
            return False
        return True

    def reset_state(self) -> None:
        """Wipe persisted state + log (only call on an explicit reset request)."""
        for p in (self.state_path, self.csv_path):
            try:
                p.unlink()
            except FileNotFoundError:
                pass
        self.books, self.last_side, self.scores, self.last_price = {}, {}, [], {}
        self.carry_state = {}

    # ------------------------------------------------------------------ #
    def _book(self, source: str, symbol: str, tag: str = "") -> PaperBook:
        # tag isolates a sub-strategy's own $10k book (e.g. each candlestick
        # pattern under us-ta / crypto-ta); untagged signals keep tag="".
        key = (source, symbol, tag)
        if key not in self.books:
            cfg = self.configs.get(source, {})
            self.books[key] = PaperBook(
                source, trade_pct=self.trade_pct,
                cost_bps_round=cfg.get("cost_bps_round", 10.0),
                allow_short=cfg.get("allow_short", True))
        return self.books[key]

    def _mark(self, symbol: str, price: float, ts: datetime) -> bool:
        """Record a price and resolve any pending scores past their horizon. Only
        prices at-or-after ts0+horizon resolve a record (no lookahead)."""
        self.last_price[symbol] = [float(price), ts.isoformat()]
        changed = False
        for r in self.scores:
            if r["status"] != "pending" or r["symbol"] != symbol:
                continue
            if ts >= datetime.fromisoformat(r["ts"]) + timedelta(seconds=self.horizon):
                moved = r["side"] * (float(price) - r["entry_price"])
                r["status"] = "correct" if moved > 0 else "wrong"
                r["resolved_price"] = float(price)
                r["resolved_ts"] = ts.isoformat()
                changed = True
        return changed

    def observe(self, source: str, symbol: str, direction, price, ts=None,
                tag: str = "") -> Optional[dict]:
        """Feed one poll's recommendation. Always marks the price (resolving scores
        + mark-to-market); logs a NEW row + opens/flips only when the directional
        signal changes (de-dup). ``tag`` makes an independent signal stream + book
        (e.g. one per candlestick pattern). Returns the logged event, or None."""
        with self._lock:
            ts = ts or self._clock().replace(microsecond=0)
            symbol = str(symbol).upper()
            price = float(price)
            changed = self._mark(symbol, price, ts)

            side = side_of(direction)
            key = f"{source}|{symbol}" + (f"|{tag}" if tag else "")
            event = None
            if side != 0 and self.last_side.get(key) != side:      # distinct new signal
                self.last_side[key] = side
                self._append_csv(ts, source, symbol, direction, price)
                self.scores.append({
                    "ts": ts.isoformat(), "source": source, "symbol": symbol,
                    "direction": str(direction), "side": side, "entry_price": price,
                    "status": "pending", "resolved_price": None, "resolved_ts": None,
                    "tag": tag})
                self._book(source, symbol, tag).set_target(symbol, side, price)
                event = self.scores[-1]
                changed = True

            if changed:
                self._save()
            return event

    def accrue_funding(self, symbol: str, funding_rate: float, funding_time,
                       ts=None) -> Optional[dict]:
        """Mark the delta-neutral funding-carry book for ``symbol`` once per NEW 8h
        funding interval (deduped by ``funding_time``):

        * if we held INTO this interval, accrue the 8h ``funding_rate`` (dollars on
          a fixed $10k notional; positive funding pays the short, negative costs it);
        * then update the HOLD decision with hysteresis — enter when funding clears
          the hurdle, exit only when it's sustainedly negative (below
          hurdle − band), paying a one-off both-legs cost on each entry/exit.

        Delta-neutral, so price moves cancel — equity = $10k + funding − costs.
        Repeated calls within the same interval do nothing (no double accrual, no
        churn). Returns the logged carry record, or None if the interval is stale.
        """
        with self._lock:
            symbol = str(symbol).upper()
            funding_rate = float(funding_rate)
            ft = int(funding_time)
            ts = ts or self._clock().replace(microsecond=0)
            st = self.carry_state.setdefault(symbol, {"in_carry": False, "last_ft": None})
            if st["last_ft"] is not None and ft <= st["last_ft"]:
                return None                       # same/old interval -> no double accrual
            st["last_ft"] = ft

            book = self._book("funding-carry", symbol)
            notional = book._notional()
            enter = self.funding_enter_bps / 1e4
            exit_ = (self.funding_enter_bps - self.funding_band_bps) / 1e4
            cost = self.funding_cost_bps / 1e4 * notional

            if st["in_carry"]:                    # held through this interval -> collect funding
                book.realized += funding_rate * notional
            if not st["in_carry"] and funding_rate >= enter:
                st["in_carry"] = True             # open long-spot + short-perp
                book.costs += cost
                book.trades += 1
            elif st["in_carry"] and funding_rate < exit_:
                st["in_carry"] = False            # sustained negative funding -> close
                book.costs += cost

            rec = {"ts": ts.isoformat(), "source": "funding-carry", "symbol": symbol,
                   "direction": "carry", "side": 1 if st["in_carry"] else 0,
                   "entry_price": round(funding_rate * 1e4, 4), "status": "carry",
                   "resolved_price": None, "resolved_ts": None}
            self.scores.append(rec)
            self._save()
            return rec

    # ------------------------------------------------------------------ #
    def symbols(self) -> list:
        """Distinct symbols seen (for the dashboard's per-symbol filter)."""
        with self._lock:
            return sorted({r["symbol"] for r in self.scores})

    def scorecard(self, symbol: Optional[str] = None) -> dict:
        """Per-source stats. ``symbol`` filters to one symbol's $10k account per
        source; ``None``/"all" aggregates that source across its symbols
        (equal-weight $10k each). The source split is kept either way."""
        with self._lock:
            sym = None if (symbol is None or str(symbol).lower() == "all") else str(symbol).upper()
            out = {}
            for source, cfg in self.configs.items():
                keys = [k for k in self.books
                        if k[0] == source and (sym is None or k[1] == sym)]
                recs = [r for r in self.scores
                        if r["source"] == source and (sym is None or r["symbol"] == sym)]
                correct = sum(r["status"] == "correct" for r in recs)
                wrong = sum(r["status"] == "wrong" for r in recs)
                pending = sum(r["status"] == "pending" for r in recs)
                decided = correct + wrong
                # ONE $10,000 account per source: its sub-books (per symbol / per pattern
                # tag) equal-weight the SAME $10k, so dollar figures are on a $10k base —
                # not summed into N x $10k (which made cards read "from $70,000"). Equity is
                # floored at 0 and costs capped at the base, so a churned-out agent shows a
                # bust (-100%) rather than impossible losses/costs that dwarf the base.
                start = 10_000.0
                if keys:
                    w = 1.0 / len(keys)
                    realized = sum(self.books[k].realized for k in keys) * w
                    raw_costs = sum(self.books[k].costs for k in keys) * w
                    trades = sum(self.books[k].trades for k in keys)
                    unreal = sum(self.books[k].unrealized(self.last_price) for k in keys) * w
                    openp = sum(len(self.books[k].positions) for k in keys)
                else:                       # no activity yet -> a fresh $10k account
                    realized = raw_costs = unreal = 0.0
                    trades = openp = 0
                raw_eq = start + realized - raw_costs + unreal
                eq = max(0.0, raw_eq)                         # lose at most the base
                costs = min(raw_costs, start)                # bounded by the $10k base
                bust = raw_eq <= 0.0
                out[source] = {
                    "source": source, "label": cfg.get("label", source),
                    "retired": bool(cfg.get("retired", False)),
                    # demo = a live technique kept running for demonstration with NO validated
                    # edge (the rigorously-killed predictors). The funding carry is a real edge.
                    "demo": bool(cfg.get("retired", False)),
                    "desc": cfg.get("desc", ""),
                    "total": len(recs), "correct": correct, "wrong": wrong,
                    "pending": pending, "bust": bust,
                    "hit_rate": round(100.0 * correct / decided, 1) if decided else None,
                    "equity_start": round(start, 2), "equity": round(eq, 2),
                    "pct_change": round((eq - start) / start * 100.0, 2),
                    "realized_pnl": round(realized, 2),
                    "unrealized_pnl": round(unreal, 2),
                    "costs": round(costs, 2), "trades": trades,
                    "open_positions": openp,
                    "recent": [{
                        "ts": r["ts"], "symbol": r["symbol"], "direction": r["direction"],
                        "side": r["side"], "status": r["status"], "tag": r.get("tag", ""),
                        "entry": round(r["entry_price"], 4),
                        "resolved": (round(r["resolved_price"], 4)
                                     if r["resolved_price"] is not None else None),
                    } for r in reversed(recs)][:8],
                    "by_pattern": _by_tag(recs),
                }
            return {"symbol": sym or "all", "symbols": sorted({r["symbol"] for r in self.scores}),
                    "horizon_min": self.horizon // 60, "trade_pct": self.trade_pct,
                    "sources": out}

    # ------------------------------------------------------------------ #
    def _append_csv(self, ts: datetime, source, symbol, direction, price) -> None:
        if not self.owner:                # single owner appends the shared log
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        new = not self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(self.CSV_FIELDS)
            w.writerow([ts.isoformat(), source, symbol, str(direction), price])

    def _save(self) -> None:
        if not self.owner:                # never clobber the owning instance's state
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        state = {
            "horizon": self.horizon, "trade_pct": self.trade_pct,
            "books": {f"{s}|{sym}|{tag}": b.to_dict() for (s, sym, tag), b in self.books.items()},
            "last_side": self.last_side, "scores": self.scores,
            "last_price": self.last_price, "carry_state": self.carry_state,
        }
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        tmp.replace(self.state_path)

    def _load(self) -> None:
        """Always RESUME from disk on startup (never silently reset to $10k)."""
        if not self.state_path.exists():
            return
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return
        self.last_side = state.get("last_side", {}) or {}
        self.scores = state.get("scores", []) or []
        self.last_price = state.get("last_price", {}) or {}
        self.carry_state = state.get("carry_state", {}) or {}
        for key, d in (state.get("books") or {}).items():
            parts = key.split("|")
            source = parts[0]
            symbol = parts[1] if len(parts) > 1 else ""
            tag = parts[2] if len(parts) > 2 else ""        # back-compat with 2-part keys
            self._book(source, symbol, tag).load(d)
