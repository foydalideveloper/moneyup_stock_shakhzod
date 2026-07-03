"""Alpaca live data feed adapter.

Wraps alpaca-py's StockDataStream and normalizes its messages into our neutral
Quote/Trade objects. The free Alpaca plan uses the IEX feed (real-time).

`alpaca-py` is imported lazily inside __init__ so the rest of the package (and
the test suite) imports cleanly even when alpaca-py isn't installed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List

from tagent.feeds.base import MarketDataFeed, Quote, Trade


class AlpacaFeed(MarketDataFeed):
    def __init__(self, symbols: List[str], api_key: str, secret_key: str,
                 feed: str = "iex"):
        super().__init__(symbols)
        if not api_key or not secret_key:
            raise ValueError("Alpaca API key and secret are required.")
        try:
            from alpaca.data.live import StockDataStream
            from alpaca.data.enums import DataFeed
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "alpaca-py not installed. Run: pip install -r requirements.txt"
            ) from e

        feed_enum = DataFeed.SIP if feed.lower() == "sip" else DataFeed.IEX
        self._stream = StockDataStream(api_key, secret_key, feed=feed_enum)

    async def _handle_quote(self, q) -> None:
        ts = getattr(q, "timestamp", None) or datetime.now(timezone.utc)
        self._emit_quote(Quote(
            symbol=q.symbol,
            bid_price=float(getattr(q, "bid_price", 0) or 0),
            ask_price=float(getattr(q, "ask_price", 0) or 0),
            bid_size=float(getattr(q, "bid_size", 0) or 0),
            ask_size=float(getattr(q, "ask_size", 0) or 0),
            timestamp=ts,
        ))

    async def _handle_trade(self, t) -> None:
        ts = getattr(t, "timestamp", None) or datetime.now(timezone.utc)
        self._emit_trade(Trade(
            symbol=t.symbol,
            price=float(getattr(t, "price", 0) or 0),
            size=float(getattr(t, "size", 0) or 0),
            timestamp=ts,
        ))

    def run(self) -> None:
        self._stream.subscribe_quotes(self._handle_quote, *self.symbols)
        self._stream.subscribe_trades(self._handle_trade, *self.symbols)
        # StockDataStream.run() manages its own reconnection loop.
        self._stream.run()
