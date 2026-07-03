"""Normalized market-data interface.

Every feed adapter converts its broker's raw messages into these neutral Quote,
Trade and OrderBookSnapshot objects and invokes the registered callbacks.
Swapping Alpaca for Kiwoom later means writing one new subclass — nothing else
changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, List, Optional, Tuple


@dataclass
class Quote:
    symbol: str
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    timestamp: datetime


@dataclass
class Trade:
    symbol: str
    price: float
    size: float
    timestamp: datetime


# One Level-2 price level: (price, quantity).
PriceLevel = Tuple[float, float]


@dataclass
class OrderBookSnapshot:
    """A Level-2 depth snapshot: up to 10 levels per side.

    `asks` are sorted ascending by price (best/lowest ask first); `bids` are
    sorted descending by price (best/highest bid first). Each level is a
    (price, qty) tuple. The lists mirror what a venue exposes as the visible
    top-of-book window.
    """

    symbol: str
    timestamp: datetime
    asks: List[PriceLevel]  # 10 levels, ascending (best first)
    bids: List[PriceLevel]  # 10 levels, descending (best first)


QuoteHandler = Callable[[Quote], None]
TradeHandler = Callable[[Trade], None]
OrderBookHandler = Callable[[OrderBookSnapshot], None]


class MarketDataFeed(ABC):
    """Base class for all live data feeds."""

    def __init__(self, symbols: List[str]):
        self.symbols = [s.strip().upper() for s in symbols if s.strip()]
        self._on_quote: Optional[QuoteHandler] = None
        self._on_trade: Optional[TradeHandler] = None
        self._on_orderbook: Optional[OrderBookHandler] = None

    def on_quote(self, handler: QuoteHandler) -> None:
        self._on_quote = handler

    def on_trade(self, handler: TradeHandler) -> None:
        self._on_trade = handler

    def on_orderbook(self, handler: OrderBookHandler) -> None:
        self._on_orderbook = handler

    # Subclasses call these after normalizing a raw message.
    def _emit_quote(self, q: Quote) -> None:
        if self._on_quote is not None:
            self._on_quote(q)

    def _emit_trade(self, t: Trade) -> None:
        if self._on_trade is not None:
            self._on_trade(t)

    def _emit_orderbook(self, ob: OrderBookSnapshot) -> None:
        if self._on_orderbook is not None:
            self._on_orderbook(ob)

    @abstractmethod
    def run(self) -> None:
        """Connect, subscribe, and block while streaming. Handles reconnection."""
        raise NotImplementedError
