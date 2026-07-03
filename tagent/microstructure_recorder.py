"""Forward-recorder for KR's NEW microstructure sessions — append-only, deduped, no-lookahead.

Two venues opened recently and so have almost no history anywhere: the **NXT (Nextrade)
pre-market call, 08:00–08:50 KST**, and the **KRX night futures session (~18:00–익일 05:00
KST)**. Because these are < ~2 years old, data we forward-collect ourselves quickly becomes a
*meaningful fraction of all that exists*. This module just ACCRUES that out-of-sample dataset
from the live feed — there is deliberately NO strategy here.

What it records (per in-session order-book snapshot) is the **depth-weighted** quote — the whole
visible book, not just the last trade: qty-weighted VWAP per side, the top-of-book microprice,
total depth and depth-imbalance, plus best bid/ask and spread.

Guarantees (all unit-tested, no network):
  * **append-only** — rows are only ever appended; existing rows are never rewritten/deleted.
  * **deduped** — keyed on (session, symbol, snapshot-timestamp); the same observation is never
    stored twice, even across process restarts (the seen-set is reloaded from the file).
  * **no-lookahead** — a snapshot is recorded ONLY if its timestamp is inside an open session
    window AND not after the recorder's ``now`` clock; future-dated observations are refused.

Pure / injectable: ``record(ob, now=...)`` takes a normalized
:class:`tagent.feeds.base.OrderBookSnapshot` and a clock, so tests drive it with canned books.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

from tagent.config import DATA_DIR
from tagent.feeds.base import OrderBookSnapshot

_KST = timezone(timedelta(hours=9))     # KR bars/sessions are reckoned in KST (project convention)


# --------------------------------------------------------------------------- #
# session calendar — the two new microstructure windows (KST, time-of-day)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Session:
    name: str
    start: time          # inclusive, KST
    end: time            # exclusive, KST
    label: str = ""

    def contains(self, t: time) -> bool:
        """Is KST time-of-day ``t`` inside this window? Handles windows that wrap midnight."""
        if self.start <= self.end:                       # same calendar day
            return self.start <= t < self.end
        return t >= self.start or t < self.end           # wraps past midnight (night session)


# NXT pre-market call auction and the KRX (CME-linked) night derivatives session. Times are the
# documented windows; the recorder gates purely on these, so widening them later just collects more.
SESSIONS: List[Session] = [
    Session("nxt-premarket", time(8, 0), time(8, 50), "NXT (Nextrade) pre-market 08:00–08:50 KST"),
    Session("krx-night", time(18, 0), time(5, 0), "KRX night futures ~18:00–익일 05:00 KST"),
]
SESSION_NAMES = [s.name for s in SESSIONS]


def _to_kst(ts: datetime) -> datetime:
    """A snapshot timestamp as KST. Naive timestamps are assumed KST (the KR-bar convention);
    tz-aware ones (e.g. the Kiwoom feed's UTC) are converted."""
    return ts.replace(tzinfo=_KST) if ts.tzinfo is None else ts.astimezone(_KST)


def _to_utc(ts: datetime) -> datetime:
    return ts.replace(tzinfo=_KST).astimezone(timezone.utc) if ts.tzinfo is None else ts.astimezone(timezone.utc)


def session_for(kst_dt: datetime) -> Optional[str]:
    """Name of the session containing this KST datetime, or None (out of session)."""
    t = kst_dt.time()
    for s in SESSIONS:
        if s.contains(t):
            return s.name
    return None


# --------------------------------------------------------------------------- #
# depth-weighted quote — the whole visible book, not just the last trade
# --------------------------------------------------------------------------- #
def depth_weighted_quote(ob: OrderBookSnapshot) -> Optional[dict]:
    """Depth-weighted features of a two-sided book, or None if a side is empty / non-positive.

    * ``dw_bid_price`` / ``dw_ask_price`` — qty-weighted VWAP across ALL visible levels per side.
    * ``microprice`` — top-of-book imbalance-weighted fair price
      ``(bid·ask_size + ask·bid_size)/(bid_size+ask_size)`` (leans toward the heavier side).
    * ``depth_imbalance`` — ``(bid_depth − ask_depth)/(bid_depth + ask_depth)`` ∈ [−1, 1].
    """
    if not ob.bids or not ob.asks:
        return None
    best_bid, best_bid_sz = ob.bids[0]
    best_ask, best_ask_sz = ob.asks[0]
    if best_bid <= 0 or best_ask <= 0:
        return None
    bid_depth = float(sum(q for _, q in ob.bids))
    ask_depth = float(sum(q for _, q in ob.asks))
    dw_bid = (sum(p * q for p, q in ob.bids) / bid_depth) if bid_depth > 0 else best_bid
    dw_ask = (sum(p * q for p, q in ob.asks) / ask_depth) if ask_depth > 0 else best_ask
    mid = (best_bid + best_ask) / 2.0
    tob = best_bid_sz + best_ask_sz
    microprice = ((best_bid * best_ask_sz + best_ask * best_bid_sz) / tob) if tob > 0 else mid
    tot = bid_depth + ask_depth
    return {
        "best_bid": best_bid, "best_ask": best_ask,
        "best_bid_size": best_bid_sz, "best_ask_size": best_ask_sz,
        "mid": round(mid, 6), "microprice": round(microprice, 6),
        "dw_bid_price": round(dw_bid, 6), "dw_ask_price": round(dw_ask, 6),
        "depth_weighted_mid": round((dw_bid + dw_ask) / 2.0, 6),
        "bid_depth": round(bid_depth, 6), "ask_depth": round(ask_depth, 6),
        "depth_imbalance": round((bid_depth - ask_depth) / tot, 6) if tot > 0 else 0.0,
        "spread": round(best_ask - best_bid, 6),
        "levels": min(len(ob.bids), len(ob.asks)),
    }


# --------------------------------------------------------------------------- #
# the recorder: append-only, deduped, no-lookahead store
# --------------------------------------------------------------------------- #
class MicrostructureRecorder:
    COLS = ["session", "symbol", "ts", "best_bid", "best_ask", "best_bid_size", "best_ask_size",
            "mid", "microprice", "dw_bid_price", "dw_ask_price", "depth_weighted_mid",
            "bid_depth", "ask_depth", "depth_imbalance", "spread", "levels"]

    def __init__(self, data_dir=None, path=None):
        self.path = Path(path) if path is not None else (
            (Path(data_dir) if data_dir is not None else Path(DATA_DIR)) / "microstructure_sessions.csv")
        self._seen: set = set()              # (session, symbol, ts_iso) already persisted
        self._rows_added = 0
        self._fh = None
        self._writer = None
        self._load_seen()

    # -------- dedup state reloaded from disk (so restarts don't double-count) -------- #
    def _load_seen(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open("r", newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    self._seen.add((row.get("session"), row.get("symbol"), row.get("ts")))
        except Exception:
            pass

    def _ensure_writer(self):
        if self._fh is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            new = (not self.path.exists()) or self.path.stat().st_size == 0
            self._fh = self.path.open("a", newline="", encoding="utf-8")
            self._writer = csv.writer(self._fh)
            if new:
                self._writer.writerow(self.COLS)
        return self._writer

    def record(self, ob: OrderBookSnapshot, now: Optional[datetime] = None) -> bool:
        """Record one order-book snapshot iff it is in-session, not future-dated, two-sided, and
        new. Returns True iff a row was appended. Pure aside from the append to ``self.path``."""
        now = now or datetime.now(timezone.utc)
        ts = getattr(ob, "timestamp", None)
        if ts is None:
            return False
        # no-lookahead: never store an observation timestamped after the clock
        if _to_utc(ts) > _to_utc(now):
            return False
        kst = _to_kst(ts)
        session = session_for(kst)
        if session is None:                              # only the new microstructure windows
            return False
        feat = depth_weighted_quote(ob)
        if feat is None:                                 # need a valid two-sided book
            return False
        ts_iso = kst.replace(tzinfo=None).isoformat()    # canonical naive-KST stamp
        key = (session, ob.symbol, ts_iso)
        if key in self._seen:                            # dedup — same observation already stored
            return False
        self._ensure_writer().writerow(
            [session, ob.symbol, ts_iso] + [feat[c] for c in self.COLS[3:]])
        self._fh.flush()
        self._seen.add(key)
        self._rows_added += 1
        return True

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.flush()
                self._fh.close()
            finally:
                self._fh, self._writer = None, None

    def summary(self) -> dict:
        """Coverage of the accrued dataset (no strategy) — rows per session/symbol."""
        by_session: Dict[str, Dict[str, int]] = {}
        for sess, sym, _ in self._seen:
            by_session.setdefault(sess, {})
            by_session[sess][sym] = by_session[sess].get(sym, 0) + 1
        return {
            "path": str(self.path), "total_rows": len(self._seen),
            "rows_added_this_run": self._rows_added,
            "sessions": sorted(k for k in by_session),
            "by_session": {s: dict(sorted(v.items())) for s, v in sorted(by_session.items())},
        }


def load_recorder_summary(data_dir=None, path=None) -> dict:
    """Read-only coverage summary for the dashboard/CLI — never writes."""
    return MicrostructureRecorder(data_dir=data_dir, path=path).summary()
