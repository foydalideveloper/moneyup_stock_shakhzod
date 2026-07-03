"""Kiwoom live data feed (KR) over the REST API's real-time WebSocket.

Implements the existing :class:`tagent.feeds.base.MarketDataFeed` interface, so
``Monitor.on_quote`` / ``on_trade`` / ``on_orderbook`` fire exactly as with
Alpaca and :class:`tagent.orderbook.OrderBookMemory` ingests unchanged.

Protocol (openapi.kiwoom.com → API 가이드, 실시간시세):

  * Connect: ``wss://{api|mockapi}.kiwoom.com:10000/api/dostk/websocket``
  * Login:   send ``{"trnm":"LOGIN","token":<access_token>}``; server replies
             ``{"trnm":"LOGIN","return_code":0,...}``.
  * Register: ``{"trnm":"REG","grp_no":"1","refresh":"1",
                 "data":[{"item":[codes],"type":["0D","0B"]}]}``
  * Keepalive: server sends ``{"trnm":"PING"}``; echo it back verbatim.
  * Data:    ``{"trnm":"REAL","data":[{"type":"0D"|"0B","item":code,
                 "values":{<FID>:<str>}}]}``

Real-time type codes (confirmed in the docs): ``0D`` = 주식호가잔량 (order book),
``0B`` = 주식체결 (executions).

FID field numbers for 0D/0B follow Kiwoom's long-standing real-time spec (the
docs confirm 41 = best ask, 51 = best bid; the contiguous 10-level ranges below
follow from that — see ``_ASK_PX`` etc.). Prices/quantities arrive as strings,
sometimes sign-prefixed ("+74100", "-73900"); :func:`_to_number` strips the sign
and parses the Korean integer magnitude.

The message *normalization* is pure and synchronous (see :meth:`feed_message`),
so tests drive it with canned payloads and never touch the network. ``asyncio`` /
``websockets`` are imported lazily inside :meth:`run`.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import List, Optional

from tagent.config import KIWOOM_WS_URLS
from tagent.feeds.base import MarketDataFeed, OrderBookSnapshot, Quote, Trade
from tagent.feeds.kiwoom_auth import KiwoomAuth

# Real-time type codes.
TYPE_ORDERBOOK = "0D"   # 주식호가잔량
TYPE_TRADE = "0B"       # 주식체결

# 호가 (0D) FID layout — 10 levels each, contiguous from the confirmed anchors.
_ASK_PX = [str(40 + i) for i in range(1, 11)]   # 41..50  매도호가1..10 (ascending)
_ASK_QTY = [str(60 + i) for i in range(1, 11)]  # 61..70  매도호가수량1..10
_BID_PX = [str(50 + i) for i in range(1, 11)]   # 51..60  매수호가1..10 (descending)
_BID_QTY = [str(70 + i) for i in range(1, 11)]  # 71..80  매수호가수량1..10

# 체결 (0B) FIDs.
_TRADE_PRICE = "10"     # 현재가 (sign-prefixed)
_TRADE_QTY = "15"       # 체결거래량 (signed: + buy / - sell)

_MAX_BACKOFF_SECONDS = 30.0


def _to_number(raw) -> float:
    """Parse a Kiwoom numeric string, stripping sign prefixes and commas.

    "+74100" -> 74100.0, "-73900" -> 73900.0, "" / None -> 0.0. Kiwoom signs
    denote up/down vs. previous close (and buy/sell for trade qty), not negative
    magnitudes, so we return the absolute value.
    """
    if raw is None:
        return 0.0
    s = str(raw).strip().replace(",", "")
    if not s:
        return 0.0
    s = s.lstrip("+-")
    if not s:
        return 0.0
    try:
        return abs(float(s))
    except ValueError:
        return 0.0


def parse_orderbook(item: str, values: dict, now: datetime) -> OrderBookSnapshot:
    """0D values -> OrderBookSnapshot (asks ascending, bids descending)."""
    asks = []
    for px_fid, qty_fid in zip(_ASK_PX, _ASK_QTY):
        px = _to_number(values.get(px_fid))
        if px > 0:
            asks.append((px, _to_number(values.get(qty_fid))))
    bids = []
    for px_fid, qty_fid in zip(_BID_PX, _BID_QTY):
        px = _to_number(values.get(px_fid))
        if px > 0:
            bids.append((px, _to_number(values.get(qty_fid))))
    return OrderBookSnapshot(symbol=item, timestamp=now, asks=asks, bids=bids)


def quote_from_orderbook(ob: OrderBookSnapshot) -> Optional[Quote]:
    """Best bid/ask of a snapshot -> Quote (None if a side is empty)."""
    if not ob.asks or not ob.bids:
        return None
    bid_px, bid_qty = ob.bids[0]
    ask_px, ask_qty = ob.asks[0]
    return Quote(symbol=ob.symbol, bid_price=bid_px, ask_price=ask_px,
                bid_size=bid_qty, ask_size=ask_qty, timestamp=ob.timestamp)


def parse_trade(item: str, values: dict, now: datetime) -> Optional[Trade]:
    """0B values -> Trade (None if there's no valid price)."""
    price = _to_number(values.get(_TRADE_PRICE))
    if price <= 0:
        return None
    return Trade(symbol=item, price=price,
                 size=_to_number(values.get(_TRADE_QTY)), timestamp=now)


class KiwoomFeed(MarketDataFeed):
    def __init__(self, symbols: List[str], app_key: str = "", secret_key: str = "",
                 env: str = "mock", auth: Optional[KiwoomAuth] = None,
                 ws_url: Optional[str] = None):
        super().__init__(symbols)
        self.env = env
        self._ws_url = ws_url or KIWOOM_WS_URLS.get(env.lower(), KIWOOM_WS_URLS["mock"])
        # Auth is built lazily-ish: constructed now (cheap) but only contacts the
        # network at login time, so tests can construct a feed with no creds.
        self._auth = auth
        self._app_key = app_key
        self._secret_key = secret_key
        self._stop = False

    # ---------------- auth ---------------- #
    def _get_auth(self) -> KiwoomAuth:
        if self._auth is None:
            self._auth = KiwoomAuth(self._app_key, self._secret_key, env=self.env)
        return self._auth

    # ---------------- message handling (pure, testable) ---------------- #
    def _reg_message(self) -> dict:
        return {
            "trnm": "REG",
            "grp_no": "1",
            "refresh": "1",
            "data": [{"item": list(self.symbols),
                      "type": [TYPE_ORDERBOOK, TYPE_TRADE]}],
        }

    def feed_message(self, payload: dict, now: Optional[datetime] = None) -> List[dict]:
        """Process one decoded server message; return any messages to send back.

        * LOGIN ok  -> [REG message]      (register the watchlist)
        * LOGIN fail -> raises KiwoomAuthError
        * PING      -> [echo]             (keepalive)
        * REAL      -> dispatch to on_orderbook/on_trade/on_quote, return []
        """
        now = now or datetime.now(timezone.utc)
        trnm = payload.get("trnm")
        if trnm == "LOGIN":
            code = payload.get("return_code")
            if code not in (0, "0", None):
                from tagent.feeds.kiwoom_auth import KiwoomAuthError
                raise KiwoomAuthError(
                    f"Kiwoom WS login rejected (return_code={code}): "
                    f"{payload.get('return_msg')}")
            return [self._reg_message()]
        if trnm == "PING":
            return [payload]
        if trnm == "REAL":
            for entry in payload.get("data") or []:
                self._dispatch_real(entry, now)
        return []

    def _dispatch_real(self, entry: dict, now: datetime) -> None:
        typ = entry.get("type")
        item = entry.get("item") or entry.get("name")
        values = entry.get("values") or {}
        if typ == TYPE_ORDERBOOK:
            ob = parse_orderbook(item, values, now)
            self._emit_orderbook(ob)
            q = quote_from_orderbook(ob)
            if q is not None:
                self._emit_quote(q)
        elif typ == TYPE_TRADE:
            t = parse_trade(item, values, now)
            if t is not None:
                self._emit_trade(t)

    # ---------------- lifecycle ---------------- #
    def run(self) -> None:
        import asyncio
        try:
            asyncio.run(self._run_async())
        except KeyboardInterrupt:  # pragma: no cover - manual stop
            self._stop = True

    def stop(self) -> None:
        self._stop = True

    async def _run_async(self) -> None:
        import asyncio

        try:
            import websockets
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "websockets not installed. Run: pip install -r requirements.txt"
            ) from e

        backoff = 1.0
        while not self._stop:
            try:
                token = self._get_auth().get_token()
                # We handle keepalive via Kiwoom's PING frames ourselves.
                async with websockets.connect(self._ws_url, ping_interval=None) as ws:
                    await ws.send(json.dumps({"trnm": "LOGIN", "token": token}))
                    backoff = 1.0  # connected + login sent -> reset backoff
                    async for raw in ws:
                        for reply in self.feed_message(json.loads(raw)):
                            await ws.send(json.dumps(reply))
                        if self._stop:
                            break
            except Exception as e:  # pragma: no cover - network paths
                if self._stop:
                    break
                print(f"[kiwoom] connection error: {e}; reconnecting in {backoff:.0f}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, _MAX_BACKOFF_SECONDS)
