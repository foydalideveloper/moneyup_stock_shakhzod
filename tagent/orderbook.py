"""Level-2 order-book depth memory.

The live state in :mod:`tagent.state` only remembers *top-of-book* (best bid /
best ask). This module remembers the *full visible depth* and — crucially —
*why* price levels disappear between two consecutive snapshots. A level can
leave the visible top-10 window for three very different reasons:

    * "filled"    — liquidity was consumed: there were trades at/through that
                    price since the last snapshot (or, absent trade data, price
                    advanced past it). Real, aggressive flow.
    * "cancelled" — the resting size was pulled with *no* trade at that price.
                    The order was withdrawn while still inside the visible
                    window. A burst of these is a classic spoofing signature.
    * "scrolled"  — the level only fell out of the window because price moved
                    and the 10-level window shifted over it (it is now beyond
                    the far edge). The order itself was neither hit nor pulled.

Distinguishing *consumed* liquidity from *pulled* liquidity is the whole point:
absorption (lots of fills) vs. spoofing (lots of cancels) are opposite signals.

All updates are incremental (compare new snapshot to the previous one) and
timestamps are passed in explicitly, so the logic is deterministic and unit
testable with no network.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Deque, Iterable, List, Optional, Tuple

from tagent.feeds.base import OrderBookSnapshot

# Vanished-level record stored in the memory deques.
VanishedLevel = Tuple[float, float, str, datetime]  # (price, last_qty, reason, ts)

_EPS = 1e-9


def _trade_prices(recent_trades: Optional[Iterable]) -> List[float]:
    """Normalize whatever the caller passed into a flat list of trade prices.

    Accepts a list of floats, of (price, size) tuples, or of objects exposing a
    ``.price`` attribute (e.g. :class:`tagent.feeds.base.Trade`).
    """
    if not recent_trades:
        return []
    prices: List[float] = []
    for t in recent_trades:
        if hasattr(t, "price"):
            prices.append(float(t.price))
        elif isinstance(t, (tuple, list)):
            prices.append(float(t[0]))
        else:
            prices.append(float(t))
    return prices


def _depth(levels: Iterable[Tuple[float, float]]) -> float:
    return float(sum(qty for _, qty in levels))


class OrderBookMemory:
    """Remembers vanished depth levels per symbol and derives flow features."""

    def __init__(self, symbol: str, maxlen: int = 10):
        self.symbol = symbol
        self.vanished_asks: Deque[VanishedLevel] = deque(maxlen=maxlen)
        self.vanished_bids: Deque[VanishedLevel] = deque(maxlen=maxlen)
        self._prev: Optional[OrderBookSnapshot] = None
        # Latest depth, cached for the imbalance feature.
        self._bid_depth: float = 0.0
        self._ask_depth: float = 0.0

    # ------------------------------------------------------------------ #
    # classification
    # ------------------------------------------------------------------ #
    @staticmethod
    def _classify_ask(price: float, best: Optional[float], worst: Optional[float],
                      trades: List[float]) -> str:
        """Why did an *ask* level at `price` leave the window?

        Asks sit *above* mid and ascend. `best` is the new lowest ask, `worst`
        the new highest visible ask.
        """
        # A trade at/through an ask lifts it: trade price >= the ask price.
        if any(tp >= price - _EPS for tp in trades):
            return "filled"
        if best is not None and price < best - _EPS:
            return "filled"      # price advanced up past it (heuristic fallback)
        if worst is not None and price > worst + _EPS:
            return "scrolled"    # window shifted down (price dropped) past it
        return "cancelled"       # vanished inside the live window, no trade

    @staticmethod
    def _classify_bid(price: float, best: Optional[float], worst: Optional[float],
                      trades: List[float]) -> str:
        """Why did a *bid* level at `price` leave the window?

        Bids sit *below* mid and descend. `best` is the new highest bid, `worst`
        the new lowest visible bid.
        """
        # A trade at/through a bid hits it: trade price <= the bid price.
        if any(tp <= price + _EPS for tp in trades):
            return "filled"
        if best is not None and price > best + _EPS:
            return "filled"      # price advanced down past it (heuristic fallback)
        if worst is not None and price < worst - _EPS:
            return "scrolled"    # window shifted up (price rose) past it
        return "cancelled"       # vanished inside the live window, no trade

    # ------------------------------------------------------------------ #
    # update
    # ------------------------------------------------------------------ #
    def update(self, snapshot: OrderBookSnapshot,
               recent_trades: Optional[Iterable] = None) -> None:
        """Fold one new snapshot in, recording any vanished levels.

        `recent_trades` are the trades that printed *since the previous
        snapshot* (any iterable of prices / (price, size) tuples / Trade-like
        objects). When omitted, classification falls back to the documented
        price-movement heuristic only.
        """
        trades = _trade_prices(recent_trades)
        self._bid_depth = _depth(snapshot.bids)
        self._ask_depth = _depth(snapshot.asks)

        prev = self._prev
        self._prev = snapshot
        if prev is None:
            return  # nothing to diff against on the very first snapshot

        cur_asks = {p: q for p, q in snapshot.asks}
        cur_bids = {p: q for p, q in snapshot.bids}
        best_ask = snapshot.asks[0][0] if snapshot.asks else None
        worst_ask = snapshot.asks[-1][0] if snapshot.asks else None
        best_bid = snapshot.bids[0][0] if snapshot.bids else None
        worst_bid = snapshot.bids[-1][0] if snapshot.bids else None
        ts = snapshot.timestamp

        for price, last_qty in prev.asks:
            if cur_asks.get(price, 0.0) > _EPS:
                continue  # still resting in the window
            reason = self._classify_ask(price, best_ask, worst_ask, trades)
            self.vanished_asks.append((price, float(last_qty), reason, ts))

        for price, last_qty in prev.bids:
            if cur_bids.get(price, 0.0) > _EPS:
                continue
            reason = self._classify_bid(price, best_bid, worst_bid, trades)
            self.vanished_bids.append((price, float(last_qty), reason, ts))

    # ------------------------------------------------------------------ #
    # features
    # ------------------------------------------------------------------ #
    def _qty_by_reason(self, reason: str) -> float:
        total = 0.0
        for _, qty, r, _ts in self.vanished_asks:
            if r == reason:
                total += qty
        for _, qty, r, _ts in self.vanished_bids:
            if r == reason:
                total += qty
        return total

    def snapshot(self) -> dict:
        """Derived Level-2 flow features for downstream rules / agents.

        * filled_qty / cancelled_qty / scrolled_qty — consumed vs pulled vs
          shifted-out resting size remembered across the last 10 events/side.
        * absorption_ratio = filled / (filled + cancelled): near 1.0 means the
          book is *absorbing* aggressive flow; near 0.0 means liquidity is
          being pulled.
        * spoof_ratio = cancelled / (filled + cancelled + scrolled): a high
          value flags pulled/possibly-spoofed liquidity.
        * depth_imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth),
          in [-1, 1]; positive means heavier resting bids (buy-side pressure).
        """
        filled = self._qty_by_reason("filled")
        cancelled = self._qty_by_reason("cancelled")
        scrolled = self._qty_by_reason("scrolled")

        consumed = filled + cancelled
        absorption = filled / consumed if consumed > 0 else 0.0
        total = filled + cancelled + scrolled
        spoof = cancelled / total if total > 0 else 0.0

        depth_sum = self._bid_depth + self._ask_depth
        imbalance = ((self._bid_depth - self._ask_depth) / depth_sum
                     if depth_sum > 0 else 0.0)

        return {
            "symbol": self.symbol,
            "filled_qty": round(filled, 6),
            "cancelled_qty": round(cancelled, 6),
            "scrolled_qty": round(scrolled, 6),
            "absorption_ratio": round(absorption, 6),
            "spoof_ratio": round(spoof, 6),
            "bid_depth": round(self._bid_depth, 6),
            "ask_depth": round(self._ask_depth, 6),
            "depth_imbalance": round(imbalance, 6),
            "vanished_asks": len(self.vanished_asks),
            "vanished_bids": len(self.vanished_bids),
        }
