"""Per-symbol live state.

All updates are *incremental* on each tick (no recomputing from scratch), which
keeps the monitor fast even at many updates per second. Timestamps are passed in
explicitly (not read from the clock inside update methods) so the logic is fully
unit-testable and deterministic.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque, Optional, Tuple


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class SymbolState:
    symbol: str
    window_seconds: int = 60

    # latest quote (best prices in the order book)
    bid_price: float = 0.0
    ask_price: float = 0.0
    bid_size: float = 0.0
    ask_size: float = 0.0

    # latest trade
    last_price: float = 0.0
    last_trade_size: float = 0.0

    # remembered session extremes
    session_high: float = 0.0
    session_low: float = 0.0

    # window extremes as they were BEFORE the latest trade (excludes the
    # current price) — used for breakout detection without self-reference.
    prev_window_high: float = 0.0
    prev_window_low: float = 0.0

    last_update: Optional[datetime] = None
    _recent: Deque[Tuple[datetime, float]] = field(default_factory=deque)

    # ---------------- derived values ---------------- #
    @property
    def spread(self) -> float:
        if self.bid_price > 0 and self.ask_price > 0:
            return self.ask_price - self.bid_price
        return 0.0

    @property
    def spread_pct(self) -> float:
        if self.ask_price > 0:
            return self.spread / self.ask_price * 100.0
        return 0.0

    @property
    def mid_price(self) -> float:
        if self.bid_price > 0 and self.ask_price > 0:
            return (self.bid_price + self.ask_price) / 2.0
        return self.last_price

    def recent_high(self) -> float:
        return max((p for _, p in self._recent), default=self.last_price)

    def recent_low(self) -> float:
        return min((p for _, p in self._recent), default=self.last_price)

    def age_seconds(self, now: Optional[datetime] = None) -> float:
        if self.last_update is None:
            return float("inf")
        now = now or _utcnow()
        return (now - self.last_update).total_seconds()

    def is_stale(self, max_age: float = 30.0, now: Optional[datetime] = None) -> bool:
        return self.age_seconds(now) > max_age

    def has_quote(self) -> bool:
        return self.bid_price > 0 and self.ask_price > 0

    # ---------------- updates ---------------- #
    def update_quote(self, bid: float, ask: float, bid_size: float,
                     ask_size: float, ts: datetime) -> None:
        # Guard against malformed/crossed quotes.
        if bid > 0:
            self.bid_price = float(bid)
            self.bid_size = float(bid_size or 0.0)
        if ask > 0:
            self.ask_price = float(ask)
            self.ask_size = float(ask_size or 0.0)
        self.last_update = ts

    def update_trade(self, price: float, size: float, ts: datetime) -> None:
        if price <= 0:
            return
        price = float(price)

        # Snapshot the window extremes BEFORE this trade enters the window.
        if self._recent:
            self.prev_window_high = max(p for _, p in self._recent)
            self.prev_window_low = min(p for _, p in self._recent)

        self.last_price = price
        self.last_trade_size = float(size or 0.0)
        self.last_update = ts

        if self.session_high == 0.0 or price > self.session_high:
            self.session_high = price
        if self.session_low == 0.0 or price < self.session_low:
            self.session_low = price

        self._recent.append((ts, price))
        self._trim_window(ts)

    def _trim_window(self, now: datetime) -> None:
        cutoff = now.timestamp() - self.window_seconds
        while self._recent and self._recent[0][0].timestamp() < cutoff:
            self._recent.popleft()

    def snapshot(self) -> dict:
        return {
            "symbol": self.symbol,
            "last": self.last_price,
            "bid": self.bid_price,
            "ask": self.ask_price,
            "spread_pct": round(self.spread_pct, 4),
            "session_high": self.session_high,
            "session_low": self.session_low,
            "recent_high": self.recent_high(),
            "recent_low": self.recent_low(),
        }
