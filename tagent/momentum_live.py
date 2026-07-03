"""Live PAPER runner for the deployable KR momentum strategy.

The one strategy that survived the full validation: **point-in-time Korean 12-1
momentum + market-regime filter, LONG-ONLY**. This operationalizes it as a
monthly-rebalanced paper book (the basis for a Kiwoom deployment later).

Each monthly rebalance:
  1. take the CURRENT point-in-time top-N KR universe (tagent.kr_universe),
  2. rank it by 12-1 momentum (252d formation skipping the last 21d),
  3. apply the MARKET-REGIME filter — full exposure when the universe basket is
     above its 200-day moving average, **go to cash** when below,
  4. hold the LONG-ONLY top quantile, equal-weight, capped at 1x.

It marks the book to market over the month, charges realistic KR costs (commission
+ sell tax + spread) on turnover, tracks equity vs an equal-weight basket, and
persists so it resumes across restarts (data/momentum_live_state.json +
data/momentum_live.csv).

STRICTLY NO-LOOKAHEAD: every signal (momentum, regime) is computed from data dated
<= the rebalance date — :meth:`MomentumLiveTrader.step` slices each price series to
``asof`` before doing anything. The selection/regime/weight logic are pure
functions, and prices are injected, so the whole thing is unit-tested with mock
data and no network.

PAPER only: this module places no orders and constructs no broker client.
"""

from __future__ import annotations

import csv
import json
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from tagent.config import DATA_DIR
from tagent.xs_momentum import align_close, momentum_signal


@dataclass
class MomentumLiveConfig:
    top_n: int = 100              # point-in-time universe size (top-N by market cap)
    lookback: int = 252          # 12-month formation
    skip_recent: int = 21        # skip the most recent ~1 month (12-1 momentum)
    top_q: float = 0.2           # long the top quantile
    regime_ma: int = 200         # market-regime moving-average window (trading days)
    capital: float = 10_000.0    # paper capital
    cost_bps: float = 15.0       # KR commission + 0.18% sell tax (amortised) + spread
    slippage_bps: float = 5.0    # additional slippage per unit turnover
    market: str = "kr"


# --------------------------------------------------------------------------- #
# pure, no-lookahead strategy functions (data must be sliced to the rebalance date)
# --------------------------------------------------------------------------- #
def momentum_scores(panel: Dict[str, pd.DataFrame], cfg: MomentumLiveConfig,
                    asof=None) -> pd.Series:
    """12-1 momentum per symbol at the latest bar (<= ``asof``). Uses only past
    prices (close.shift(skip)/close.shift(skip+lookback) - 1), so no lookahead."""
    close = align_close(panel)
    if asof is not None:
        close = close.loc[close.index <= pd.Timestamp(asof)]
    if close.empty:
        return pd.Series(dtype=float)
    sig = momentum_signal(close, cfg.lookback, cfg.skip_recent)
    return sig.iloc[-1].dropna() if len(sig) else pd.Series(dtype=float)


def basket_nav(panel: Dict[str, pd.DataFrame], asof=None,
               universe: Optional[Sequence[str]] = None) -> pd.Series:
    """Equal-weight basket NAV from daily returns of the (universe) names, through
    the latest bar <= ``asof``. The market-regime proxy."""
    close = align_close(panel)
    if asof is not None:
        close = close.loc[close.index <= pd.Timestamp(asof)]
    if universe is not None:
        keep = [c for c in close.columns if c in set(universe)]
        close = close[keep]
    if close.shape[1] == 0 or close.empty:
        return pd.Series(dtype=float)
    basket = close.pct_change().mean(axis=1)
    return (1.0 + basket.fillna(0.0)).cumprod()


def regime_is_on(panel: Dict[str, pd.DataFrame], cfg: MomentumLiveConfig, asof=None,
                 universe: Optional[Sequence[str]] = None) -> bool:
    """True (full exposure) when the basket NAV at the rebalance date is at/above its
    ``regime_ma`` moving average; False (to cash) when below. During the MA warmup
    there is no confirmed downtrend yet -> full exposure (matches the backtest overlay).
    """
    nav = basket_nav(panel, asof=asof, universe=universe)
    if len(nav) < cfg.regime_ma:
        return True
    sma = nav.rolling(cfg.regime_ma, min_periods=cfg.regime_ma).mean()
    cur, cur_sma = nav.iloc[-1], sma.iloc[-1]
    if pd.isna(cur_sma):
        return True
    return bool(cur >= cur_sma)


def target_weights(scores: pd.Series, universe: Sequence[str], regime_on: bool,
                   cfg: MomentumLiveConfig) -> Dict[str, float]:
    """Long-only top-quantile equal-weight target (sums to 1, capped 1x). Empty
    (all cash) when the regime filter is off."""
    if not regime_on:
        return {}
    uni = set(universe)
    cand = scores[[s for s in scores.index if s in uni]].dropna()
    if cand.empty:
        return {}
    n = max(1, int(round(len(cand) * cfg.top_q)))
    top = cand.sort_values(ascending=False).head(n)
    w = 1.0 / len(top)
    return {str(s): w for s in top.index}


def should_rebalance(asof, last_month: Optional[str]) -> bool:
    """Monthly cadence: rebalance once per calendar month."""
    ym = pd.Timestamp(asof).strftime("%Y-%m")
    return ym != last_month


def normalize_ticker(sym) -> str:
    """Canonical KR code so the SAME stock compares equal across universe sources.

    Strips an exchange suffix (``005930.KS`` -> ``005930``) and a Kiwoom letter
    prefix (``A005930`` -> ``005930``), then zero-pads a 6-digit code. Without this,
    a switch of universe source (cached snapshot vs live pykrx) would make an
    unchanged holding look SOLD-then-BOUGHT purely from a formatting difference."""
    s = str(sym).strip().upper()
    if "." in s:
        s = s.split(".", 1)[0]                       # drop .KS / .KQ / .KRX suffix
    if len(s) == 7 and s[0].isalpha() and s[1:].isdigit():
        s = s[1:]                                    # drop Kiwoom 'A' prefix
    return s.zfill(6) if s.isdigit() else s


def diff_holdings(prev, target) -> Dict[str, List[str]]:
    """Compare the previous holdings to the new target list and return the actions:
    ``to_buy`` (new names), ``to_sell`` (dropped names), ``to_hold`` (unchanged).
    First run (no prior holdings) -> everything is a BUY. Going to cash (empty
    target) -> everything is a SELL. Both sides are NORMALIZED first, so identical
    economic holdings in different ticker formats register as HOLD, not rotation."""
    prev_set = {normalize_ticker(x) for x in (prev or [])}
    tgt_set = {normalize_ticker(x) for x in (target or [])}
    return {"to_buy": sorted(tgt_set - prev_set),
            "to_sell": sorted(prev_set - tgt_set),
            "to_hold": sorted(tgt_set & prev_set)}


def holdings_turnover(prev, target) -> float:
    """Fraction of names that changed between the previous and target books (Jaccard
    distance over normalized tickers): 0.0 = identical, 1.0 = fully rotated."""
    prev_set = {normalize_ticker(x) for x in (prev or [])}
    tgt_set = {normalize_ticker(x) for x in (target or [])}
    union = prev_set | tgt_set
    return len(prev_set ^ tgt_set) / len(union) if union else 0.0


# --------------------------------------------------------------------------- #
# persistent paper trader
# --------------------------------------------------------------------------- #
class MomentumLiveTrader:
    """Monthly long-only KR momentum + regime-filter paper book (persistent)."""

    CSV_FIELDS = ["ts", "cycle", "equity", "basket_equity", "regime", "exposure_pct",
                  "n_held", "costs", "ann_return_pct", "held"]

    def __init__(self, cfg: Optional[MomentumLiveConfig] = None, data_dir=None, clock=None):
        self.cfg = cfg or MomentumLiveConfig()
        self.dir = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
        self.state_path = self.dir / "momentum_live_state.json"
        self.csv_path = self.dir / "momentum_live.csv"
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.weights: Dict[str, float] = {}
        self.entry_prices: Dict[str, float] = {}
        self.basket_entry: Dict[str, float] = {}
        self.equity = self.cfg.capital
        self.basket_equity = self.cfg.capital
        self.costs = 0.0
        self.regime_on = True
        self.cycle = 0
        self.in_market_cycles = 0
        self.universe_size = 0
        self.first_ts: Optional[str] = None
        self.last_month: Optional[str] = None
        self.last_rebalanced: Optional[str] = None
        self.last_actions: Dict[str, List[str]] = {"to_buy": [], "to_sell": [], "to_hold": []}
        self.last_turnover: float = 0.0
        self._load()

    # ------------------------------------------------------------------ #
    def step(self, panel: Dict[str, pd.DataFrame], universe: Sequence[str],
             asof=None, force: bool = False) -> dict:
        """Run one monthly rebalance. ``panel`` = {symbol: df with 'close'} (any
        history; sliced to ``asof`` internally), ``universe`` = current point-in-time
        members. Deduped to one rebalance per calendar month unless ``force``."""
        asof = pd.Timestamp(asof) if asof is not None else pd.Timestamp(self._clock()).normalize()
        ym = asof.strftime("%Y-%m")
        if not force and not should_rebalance(asof, self.last_month):
            return self.status()

        # no-lookahead: only data dated <= asof is visible to the rebalance
        panel = {s: df.loc[df.index <= asof] for s, df in panel.items()}
        panel = {s: df for s, df in panel.items() if len(df)}
        price_now = {s: float(df["close"].iloc[-1]) for s, df in panel.items()}

        # 1) mark the existing book to market over the elapsed month
        realized = sum(w * (price_now[c] / self.entry_prices[c] - 1.0)
                       for c, w in self.weights.items()
                       if c in price_now and self.entry_prices.get(c))
        self.equity *= (1.0 + realized)

        # equal-weight basket comparison over the same period
        b_rets = [(price_now[c] / ep - 1.0) for c, ep in self.basket_entry.items()
                  if c in price_now and ep]
        if b_rets:
            self.basket_equity *= (1.0 + sum(b_rets) / len(b_rets))

        # 2) new signal: momentum ranks + regime filter -> long-only target
        prev_holdings = list(self.weights)              # before we overwrite the book
        scores = momentum_scores(panel, self.cfg, asof=asof)
        on = regime_is_on(panel, self.cfg, asof=asof, universe=universe)
        new_w = target_weights(scores, universe, on, self.cfg)
        self.last_actions = diff_holdings(prev_holdings, list(new_w))   # BUY/SELL/HOLD
        self.last_turnover = holdings_turnover(prev_holdings, list(new_w))
        if prev_holdings and self.last_turnover > 0.60:                 # warn on heavy rotation
            warnings.warn(f"[momentum_live] {ym} rebalance turnover "
                          f"{self.last_turnover*100:.0f}% exceeds 60%", stacklevel=2)

        # 3) realistic KR costs on turnover (charged on the marked equity)
        names = set(self.weights) | set(new_w)
        turnover = sum(abs(new_w.get(c, 0.0) - self.weights.get(c, 0.0)) for c in names)
        cost_frac = turnover * (self.cfg.cost_bps + self.cfg.slippage_bps) / 1e4
        self.costs += self.equity * cost_frac
        self.equity *= (1.0 - cost_frac)

        # 4) roll positions: entry prices for the new book; reset basket basis
        self.entry_prices = {c: price_now[c] for c in new_w if c in price_now}
        self.basket_entry = {c: price_now[c] for c in universe if c in price_now}
        self.weights = new_w
        self.regime_on = on
        self.universe_size = len(universe)
        self.cycle += 1
        self.in_market_cycles += int(on)
        self.last_month = ym
        self.last_rebalanced = asof.isoformat()
        if self.first_ts is None:
            self.first_ts = asof.isoformat()

        self._save()
        self._append_csv(asof)
        return self.status()

    # ------------------------------------------------------------------ #
    def status(self) -> dict:
        cap = self.cfg.capital
        eq = self.equity
        held = [{"symbol": c, "weight": round(w, 4), "notional": round(w * eq, 2)}
                for c, w in sorted(self.weights.items(), key=lambda kv: -kv[1])]
        ann = ((eq / cap - 1.0) * (12.0 / self.cycle) * 100.0) if self.cycle else 0.0
        next_rebalance = (str(pd.Period(self.last_month, "M") + 1) if self.last_month else None)
        return {
            "enabled": True, "strategy": "KR 12-1 momentum + regime filter (long-only)",
            "cycle": self.cycle, "capital": cap,
            "equity": round(eq, 2), "pct_change": round((eq / cap - 1.0) * 100.0, 3),
            "basket_equity": round(self.basket_equity, 2),
            "basket_pct": round((self.basket_equity / cap - 1.0) * 100.0, 3),
            "regime": "in-market" if self.regime_on else "cash",
            "exposure_pct": 100.0 if self.regime_on else 0.0,
            "n_held": len(held), "held": held,
            "costs": round(self.costs, 2),
            "ann_return_pct": round(ann, 2),
            "universe_size": self.universe_size,
            "frac_in_market_pct": round(self.in_market_cycles / self.cycle * 100.0, 1) if self.cycle else 0.0,
            "first_ts": self.first_ts, "last_month": self.last_month,
            "last_rebalanced": self.last_rebalanced, "next_rebalance": next_rebalance,
            "actions": dict(self.last_actions),
            "turnover_pct": round(self.last_turnover * 100.0, 1),
            "high_turnover": bool(self.last_turnover > 0.60),
        }

    # ------------------------------------------------------------------ #
    def _save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        state = {"cfg": asdict(self.cfg), "weights": self.weights,
                 "entry_prices": self.entry_prices, "basket_entry": self.basket_entry,
                 "equity": self.equity, "basket_equity": self.basket_equity,
                 "costs": self.costs, "regime_on": self.regime_on, "cycle": self.cycle,
                 "in_market_cycles": self.in_market_cycles, "universe_size": self.universe_size,
                 "first_ts": self.first_ts, "last_month": self.last_month,
                 "last_rebalanced": self.last_rebalanced, "last_actions": self.last_actions,
                 "last_turnover": self.last_turnover}
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        tmp.replace(self.state_path)

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            s = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return
        self.weights = {k: float(v) for k, v in (s.get("weights") or {}).items()}
        self.entry_prices = {k: float(v) for k, v in (s.get("entry_prices") or {}).items()}
        self.basket_entry = {k: float(v) for k, v in (s.get("basket_entry") or {}).items()}
        self.equity = float(s.get("equity", self.cfg.capital))
        self.basket_equity = float(s.get("basket_equity", self.cfg.capital))
        self.costs = float(s.get("costs", 0.0))
        self.regime_on = bool(s.get("regime_on", True))
        self.cycle = int(s.get("cycle", 0))
        self.in_market_cycles = int(s.get("in_market_cycles", 0))
        self.universe_size = int(s.get("universe_size", 0))
        self.first_ts = s.get("first_ts")
        self.last_month = s.get("last_month")
        self.last_rebalanced = s.get("last_rebalanced")
        la = s.get("last_actions") or {}
        self.last_actions = {k: list(la.get(k, [])) for k in ("to_buy", "to_sell", "to_hold")}
        self.last_turnover = float(s.get("last_turnover", 0.0))

    def _append_csv(self, ts) -> None:
        st = self.status()
        self.dir.mkdir(parents=True, exist_ok=True)
        new = not self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(self.CSV_FIELDS)
            w.writerow([pd.Timestamp(ts).isoformat(), st["cycle"], st["equity"],
                        st["basket_equity"], st["regime"], st["exposure_pct"],
                        st["n_held"], st["costs"], st["ann_return_pct"],
                        "|".join(f"{h['symbol']}:{h['weight']}" for h in st["held"])])


class MonthlyRebalanceScheduler:
    """Fire a monthly rebalance automatically — at most once per calendar month.

    Designed to be ticked frequently (e.g. hourly) from a background thread. It
    throttles real attempts to ``min_interval_s`` apart, and only triggers when the
    current calendar month hasn't been rebalanced yet — determined from BOTH an
    in-process guard (``_last_run_month``) and ``month_done_fn`` (which reads the
    persisted ``last_month`` from disk, so a fresh process / the manual script can't
    cause a double-rebalance). ``rebalance_fn`` is a zero-arg callable that performs
    one rebalance; everything is injected so this is unit-tested with no network.
    """

    def __init__(self, rebalance_fn, month_done_fn=None, clock=None,
                 min_interval_s: float = 86400.0):
        self.rebalance_fn = rebalance_fn
        self.month_done_fn = month_done_fn
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.min_interval_s = float(min_interval_s)
        self._last_run_month: Optional[str] = None
        self._last_tick = None

    def due(self, now=None) -> bool:
        """True if the current calendar month still needs a rebalance."""
        ym = pd.Timestamp(now or self._clock()).strftime("%Y-%m")
        if ym == self._last_run_month:
            return False
        done = self.month_done_fn() if self.month_done_fn else None
        return ym != done

    def tick(self, now=None) -> bool:
        """One scheduler tick. Runs the rebalance iff a new month is due (and the
        throttle interval has elapsed). Returns True iff it triggered a rebalance."""
        now = pd.Timestamp(now or self._clock())
        if (self._last_tick is not None and self.min_interval_s > 0
                and (now - pd.Timestamp(self._last_tick)).total_seconds() < self.min_interval_s):
            return False
        self._last_tick = now
        if not self.due(now):
            return False
        self.rebalance_fn()
        self._last_run_month = now.strftime("%Y-%m")
        return True


def load_live_status(data_dir=None) -> dict:
    """Read the persisted live state for the dashboard (no stepping). Returns
    ``{"enabled": False}`` if the runner hasn't produced state yet."""
    path = (Path(data_dir) if data_dir is not None else Path(DATA_DIR)) / "momentum_live_state.json"
    if not path.exists():
        return {"enabled": False, "held": []}
    try:
        return MomentumLiveTrader(data_dir=data_dir).status()
    except Exception:
        return {"enabled": False, "held": []}
