"""Free crypto order-book feed (Binance public WebSocket — no API key).

Implements the existing :class:`tagent.feeds.base.MarketDataFeed` interface, so
``Monitor.on_quote`` / ``on_trade`` / ``on_orderbook`` fire just like Alpaca /
Kiwoom and :class:`tagent.orderbook.OrderBookMemory` ingests unchanged. The feed
also keeps **one OrderBookMemory per subscribed symbol**, so the vanished-level
tagging + absorption / spoof / depth-imbalance features run live per coin.

Binance combined stream (``wss://stream.binance.com:9443/stream``):

  * partial depth: ``<sym>@depth10@100ms`` -> ``{"lastUpdateId":..,
    "bids":[["p","q"],..10], "asks":[["p","q"],..10]}`` (bids descending, asks
    ascending — already our convention).
  * trades:        ``<sym>@trade`` -> ``{"e":"trade","s":"BTCUSDT","p":"..","q":".."}``.
  * combined wrapper: ``{"stream":"<sym>@depth10@100ms","data":{..}}``.
  * runtime control: ``{"method":"SUBSCRIBE"|"UNSUBSCRIBE","params":[streams],"id":n}``
    over the SAME socket — so the dashboard can add/switch coins on the fly.

Message normalization (:meth:`feed_message`) and symbol handling are pure and
synchronous, so tests drive them with canned payloads and never touch the
network. ``asyncio`` / ``websockets`` are imported lazily inside :meth:`run`.
"""

from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional

from tagent.feeds.base import MarketDataFeed, OrderBookSnapshot, Quote, Trade
from tagent.orderbook import OrderBookMemory

BINANCE_WS_URL = "wss://stream.binance.com:9443/stream"

# Quote assets we recognize so "btc" -> BTCUSDT but "ethbtc" stays ETHBTC.
_QUOTE_ASSETS = ("USDT", "FDUSD", "USDC", "BUSD", "TUSD", "USD",
                 "BTC", "ETH", "BNB", "EUR", "TRY", "DAI")

_DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT")
_MAX_BACKOFF_SECONDS = 30.0


# --------------------------------------------------------------------------- #
# symbol validation / normalization
# --------------------------------------------------------------------------- #
def normalize_symbol(sym: str, default_quote: str = "USDT") -> str:
    """Normalize a typed symbol to a Binance pair, or raise ValueError.

    "btc" -> "BTCUSDT", "ETH/USDT" -> "ETHUSDT", "ethbtc" -> "ETHBTC".
    Empty / non-alphanumeric / too-short inputs are rejected cleanly.
    """
    if not isinstance(sym, str):
        raise ValueError(f"symbol must be a string, got {type(sym).__name__}")
    cleaned = re.sub(r"[^A-Za-z0-9]", "", sym).upper()
    if not cleaned:
        raise ValueError(f"empty or invalid symbol: {sym!r}")
    # Already a full pair (ends with a known quote, with a base in front)?
    has_quote = any(cleaned.endswith(q) and len(cleaned) > len(q)
                    for q in _QUOTE_ASSETS)
    pair = cleaned if has_quote else cleaned + default_quote.upper()
    if not re.fullmatch(r"[A-Z0-9]{6,20}", pair):
        raise ValueError(f"invalid symbol: {sym!r} (normalized {pair!r})")
    return pair


def _to_float(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


# --------------------------------------------------------------------------- #
# pure parsers
# --------------------------------------------------------------------------- #
def parse_depth(symbol: str, data: dict, now: datetime) -> OrderBookSnapshot:
    """Binance partial-depth `data` -> OrderBookSnapshot (asks ↑, bids ↓)."""
    asks = [(_to_float(p), _to_float(q)) for p, q in (data.get("asks") or [])[:10]
            if _to_float(p) > 0]
    bids = [(_to_float(p), _to_float(q)) for p, q in (data.get("bids") or [])[:10]
            if _to_float(p) > 0]
    return OrderBookSnapshot(symbol=symbol, timestamp=now, asks=asks, bids=bids)


def quote_from_orderbook(ob: OrderBookSnapshot) -> Optional[Quote]:
    """Best bid/ask of a snapshot -> Quote (None if a side is empty)."""
    if not ob.asks or not ob.bids:
        return None
    bid_px, bid_qty = ob.bids[0]
    ask_px, ask_qty = ob.asks[0]
    return Quote(symbol=ob.symbol, bid_price=bid_px, ask_price=ask_px,
                bid_size=bid_qty, ask_size=ask_qty, timestamp=ob.timestamp)


def parse_trade(symbol: str, data: dict, now: datetime) -> Optional[Trade]:
    """Binance `@trade` `data` -> Trade (None if no valid price)."""
    price = _to_float(data.get("p"))
    if price <= 0:
        return None
    return Trade(symbol=(data.get("s") or symbol).upper(),
                 price=price, size=_to_float(data.get("q")), timestamp=now)


def _symbol_from_stream(stream: Optional[str]) -> str:
    return stream.split("@", 1)[0].upper() if stream else ""


class CryptoFeed(MarketDataFeed):
    def __init__(self, symbols=_DEFAULT_SYMBOLS, ws_url: Optional[str] = None,
                 depth: int = 10, depth_speed: str = "100ms",
                 default_quote: str = "USDT"):
        norm = [normalize_symbol(s, default_quote) for s in symbols]
        super().__init__(norm)
        self._ws_url = ws_url or BINANCE_WS_URL
        self.depth = depth
        self.speed = depth_speed
        self.default_quote = default_quote
        self._stop = False
        self._ws = None

        # One OrderBookMemory per coin + a buffer of trades since the last depth
        # snapshot (for the memory's fill-vs-cancel attribution).
        self._subs = set(self.symbols)
        self.books: Dict[str, OrderBookMemory] = {s: OrderBookMemory(s) for s in self._subs}
        self._trades_since_depth: Dict[str, List[float]] = {s: [] for s in self._subs}

        # Runtime control messages (SUBSCRIBE/UNSUBSCRIBE) drained by the WS loop.
        self._ctrl_lock = threading.Lock()
        self._ctrl_pending: List[dict] = []
        self._ctrl_id = 0

    # ---------------- subscriptions (runtime, thread-safe) ---------------- #
    @property
    def subscriptions(self) -> List[str]:
        return sorted(self._subs)

    def _streams_for(self, sym: str) -> List[str]:
        s = sym.lower()
        return [f"{s}@depth{self.depth}@{self.speed}", f"{s}@trade"]

    def _all_streams(self) -> List[str]:
        streams: List[str] = []
        for s in sorted(self._subs):
            streams.extend(self._streams_for(s))
        return streams

    def _control_message(self, method: str, streams: List[str]) -> dict:
        self._ctrl_id += 1
        return {"method": method, "params": streams, "id": self._ctrl_id}

    def _enqueue_control(self, msg: dict) -> None:
        with self._ctrl_lock:
            self._ctrl_pending.append(msg)

    def _drain_control(self) -> List[dict]:
        with self._ctrl_lock:
            msgs = self._ctrl_pending[:]
            self._ctrl_pending.clear()
            return msgs

    def pending_control(self) -> List[dict]:
        """Test/inspection helper: queued (not-yet-sent) control messages."""
        with self._ctrl_lock:
            return list(self._ctrl_pending)

    def subscribe_symbol(self, sym: str) -> str:
        """Add a coin at runtime (dashboard search/switch). Returns the pair.

        Idempotent; raises ValueError on an invalid symbol.
        """
        pair = normalize_symbol(sym, self.default_quote)
        if pair not in self._subs:
            self._subs.add(pair)
            self.books.setdefault(pair, OrderBookMemory(pair))
            self._trades_since_depth.setdefault(pair, [])
            self.symbols = sorted(self._subs)
            self._enqueue_control(self._control_message("SUBSCRIBE", self._streams_for(pair)))
        return pair

    def unsubscribe_symbol(self, sym: str) -> bool:
        """Remove a coin at runtime. Returns True if it was subscribed."""
        pair = normalize_symbol(sym, self.default_quote)
        if pair not in self._subs:
            return False
        self._subs.discard(pair)
        self.books.pop(pair, None)
        self._trades_since_depth.pop(pair, None)
        self.symbols = sorted(self._subs)
        self._enqueue_control(self._control_message("UNSUBSCRIBE", self._streams_for(pair)))
        return True

    # ---------------- per-coin features ---------------- #
    def book_features(self, sym: str) -> Optional[dict]:
        pair = normalize_symbol(sym, self.default_quote)
        mem = self.books.get(pair)
        return mem.snapshot() if mem is not None else None

    # ---------------- message handling (pure, testable) ---------------- #
    def feed_message(self, payload: dict, now: Optional[datetime] = None) -> None:
        """Process one decoded Binance message (combined or raw, or a sub ack)."""
        now = now or datetime.now(timezone.utc)
        if not isinstance(payload, dict):
            return
        if "stream" in payload and "data" in payload:
            stream, data = payload["stream"], payload["data"]
        elif "result" in payload or ("id" in payload and "e" not in payload):
            return  # SUBSCRIBE/UNSUBSCRIBE acknowledgement
        else:
            stream, data = None, payload
        if not isinstance(data, dict):
            return

        if data.get("e") == "trade" or ("p" in data and "q" in data and "bids" not in data):
            sym = (data.get("s") or _symbol_from_stream(stream)).upper()
            t = parse_trade(sym, data, now)
            if t is not None:
                self._trades_since_depth.setdefault(t.symbol, []).append(t.price)
                self._emit_trade(t)
            return

        if "bids" in data and "asks" in data:
            sym = _symbol_from_stream(stream) or (data.get("s") or "").upper()
            if not sym:
                return
            ob = parse_depth(sym, data, now)
            self._ingest_depth(ob)
            self._emit_orderbook(ob)
            q = quote_from_orderbook(ob)
            if q is not None:
                self._emit_quote(q)

    def _ingest_depth(self, ob: OrderBookSnapshot) -> None:
        mem = self.books.get(ob.symbol)
        if mem is None:
            mem = self.books.setdefault(ob.symbol, OrderBookMemory(ob.symbol))
        trades = self._trades_since_depth.setdefault(ob.symbol, [])
        mem.update(ob, recent_trades=trades)
        trades.clear()

    # ---------------- lifecycle ---------------- #
    def run(self) -> None:
        import asyncio
        try:
            asyncio.run(self._run_async())
        except KeyboardInterrupt:  # pragma: no cover - manual stop
            self._stop = True

    def stop(self) -> None:
        self._stop = True

    async def _run_async(self) -> None:  # pragma: no cover - network path
        import asyncio
        try:
            import websockets
        except ImportError as e:
            raise ImportError(
                "websockets not installed. Run: pip install -r requirements.txt") from e

        backoff = 1.0
        while not self._stop:
            try:
                async with websockets.connect(self._ws_url, ping_interval=20) as ws:
                    self._ws = ws
                    # Authoritative (re)subscribe to the full current set; drop any
                    # stale incremental control messages.
                    self._drain_control()
                    streams = self._all_streams()
                    if streams:
                        await ws.send(json.dumps(self._control_message("SUBSCRIBE", streams)))
                    backoff = 1.0
                    while not self._stop:
                        for msg in self._drain_control():
                            await ws.send(json.dumps(msg))
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        except asyncio.TimeoutError:
                            continue
                        self.feed_message(json.loads(raw))
            except Exception as e:
                if self._stop:
                    break
                print(f"[crypto] connection error: {e}; reconnecting in {backoff:.0f}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, _MAX_BACKOFF_SECONDS)
        self._ws = None
