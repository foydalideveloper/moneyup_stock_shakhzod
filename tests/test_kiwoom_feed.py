"""Kiwoom feed + auth tests — fully offline (no live or mock API calls).

Auth's HTTP call is mocked with a fake session; the feed is driven with canned
WebSocket payloads shaped like the documented Kiwoom messages, asserting they
normalize into OrderBookSnapshot / Trade / Quote and that OrderBookMemory ingests
them. Nothing here touches the network.
"""

from datetime import datetime, timedelta, timezone

import pytest

from tagent.feeds.base import OrderBookSnapshot, Quote, Trade
from tagent.feeds.kiwoom_auth import KiwoomAuth, KiwoomAuthError
from tagent.feeds.kiwoom_feed import (
    TYPE_ORDERBOOK,
    TYPE_TRADE,
    KiwoomFeed,
    _to_number,
    parse_orderbook,
)
from tagent.orderbook import OrderBookMemory

NOW = datetime(2026, 6, 2, 1, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# canned payloads (documented Kiwoom shapes)
# --------------------------------------------------------------------------- #
def _orderbook_values():
    """10 ask + 10 bid levels with Kiwoom sign-prefixed string prices."""
    values = {}
    # Asks 41..50 ascending from 74100; quantities 61..70.
    for lvl in range(1, 11):
        values[str(40 + lvl)] = f"+{74100 + (lvl - 1) * 100}"
        values[str(60 + lvl)] = str(lvl * 10)
    # Bids 51..60 descending from 74000; quantities 71..80.
    for lvl in range(1, 11):
        values[str(50 + lvl)] = f"-{74000 - (lvl - 1) * 100}"
        values[str(70 + lvl)] = str(lvl * 5)
    return values


def _ob_payload(item="005930"):
    return {"trnm": "REAL", "data": [
        {"type": TYPE_ORDERBOOK, "name": "주식호가잔량",
         "item": item, "values": _orderbook_values()}]}


def _trade_payload(item="005930", price="+74050", qty="-7"):
    return {"trnm": "REAL", "data": [
        {"type": TYPE_TRADE, "name": "주식체결", "item": item,
         "values": {"20": "090501", "10": price, "15": qty}}]}


def _collect_feed():
    feed = KiwoomFeed(["005930"], app_key="x", secret_key="y", env="mock")
    obs, trades, quotes = [], [], []
    feed.on_orderbook(obs.append)
    feed.on_trade(trades.append)
    feed.on_quote(quotes.append)
    return feed, obs, trades, quotes


# --------------------------------------------------------------------------- #
# number parsing
# --------------------------------------------------------------------------- #
def test_to_number_strips_signs_and_commas():
    assert _to_number("+74100") == 74100.0
    assert _to_number("-73900") == 73900.0   # sign = direction, not negative price
    assert _to_number("1,234") == 1234.0
    assert _to_number("") == 0.0
    assert _to_number(None) == 0.0
    assert _to_number("abc") == 0.0


# --------------------------------------------------------------------------- #
# order book normalization
# --------------------------------------------------------------------------- #
def test_orderbook_normalizes_to_snapshot():
    feed, obs, _, quotes = _collect_feed()
    feed.feed_message(_ob_payload(), now=NOW)

    assert len(obs) == 1
    ob = obs[0]
    assert isinstance(ob, OrderBookSnapshot)
    assert ob.symbol == "005930"
    assert ob.timestamp == NOW
    assert len(ob.asks) == 10 and len(ob.bids) == 10

    # Asks ascending (best/lowest first), bids descending (best/highest first).
    ask_prices = [p for p, _ in ob.asks]
    bid_prices = [p for p, _ in ob.bids]
    assert ask_prices == sorted(ask_prices)
    assert bid_prices == sorted(bid_prices, reverse=True)
    assert ob.asks[0] == (74100.0, 10.0)   # best ask price + qty
    assert ob.bids[0] == (74000.0, 5.0)    # best bid price + qty

    # Best bid/ask also surfaced as a Quote.
    assert len(quotes) == 1
    q = quotes[0]
    assert isinstance(q, Quote)
    assert q.bid_price == 74000.0 and q.ask_price == 74100.0
    assert q.bid_size == 5.0 and q.ask_size == 10.0


def test_orderbook_skips_empty_levels():
    values = _orderbook_values()
    # Blank out the 10th ask level entirely -> should be dropped, not zero-filled.
    values["50"] = ""
    values["70"] = ""
    ob = parse_orderbook("005930", values, NOW)
    assert len(ob.asks) == 9
    assert all(p > 0 for p, _ in ob.asks)


# --------------------------------------------------------------------------- #
# trade normalization
# --------------------------------------------------------------------------- #
def test_trade_normalizes_to_trade():
    feed, _, trades, _ = _collect_feed()
    feed.feed_message(_trade_payload(price="+74050", qty="-7"), now=NOW)
    assert len(trades) == 1
    t = trades[0]
    assert isinstance(t, Trade)
    assert t.symbol == "005930"
    assert t.price == 74050.0
    assert t.size == 7.0          # signed qty -> magnitude
    assert t.timestamp == NOW


def test_zero_price_trade_dropped():
    feed, _, trades, _ = _collect_feed()
    feed.feed_message(_trade_payload(price="0"), now=NOW)
    assert trades == []


# --------------------------------------------------------------------------- #
# control messages: LOGIN / REG / PING
# --------------------------------------------------------------------------- #
def test_login_ok_returns_registration():
    feed, *_ = _collect_feed()
    replies = feed.feed_message({"trnm": "LOGIN", "return_code": 0, "return_msg": ""})
    assert len(replies) == 1
    reg = replies[0]
    assert reg["trnm"] == "REG"
    assert reg["data"][0]["item"] == ["005930"]
    assert reg["data"][0]["type"] == [TYPE_ORDERBOOK, TYPE_TRADE]


def test_login_failure_raises():
    feed, *_ = _collect_feed()
    with pytest.raises(KiwoomAuthError):
        feed.feed_message({"trnm": "LOGIN", "return_code": 1,
                           "return_msg": "invalid token"})


def test_ping_is_echoed():
    feed, *_ = _collect_feed()
    ping = {"trnm": "PING", "data": "keepalive"}
    assert feed.feed_message(ping) == [ping]


# --------------------------------------------------------------------------- #
# integration: OrderBookMemory ingests Kiwoom-normalized snapshots
# --------------------------------------------------------------------------- #
def test_orderbook_memory_ingests_feed_output():
    feed, obs, _, _ = _collect_feed()
    mem = OrderBookMemory("005930")
    feed.on_orderbook(lambda ob: mem.update(ob))

    feed.feed_message(_ob_payload(), now=NOW)
    # Second snapshot: the best ask (74100) is gone after a trade lifts it.
    values2 = _orderbook_values()
    del values2["41"], values2["61"]            # drop level-1 ask
    values2["50"] = "+75100"                     # backfill a new top level
    values2["70"] = "11"
    feed.feed_message(
        {"trnm": "REAL", "data": [{"type": TYPE_ORDERBOOK, "item": "005930",
                                   "values": values2}]},
        now=NOW + timedelta(seconds=1))

    snap = mem.snapshot()
    assert snap["symbol"] == "005930"
    assert snap["bid_depth"] > 0 and snap["ask_depth"] > 0
    # The vanished 74100 ask was recorded as a vanished level.
    assert snap["vanished_asks"] >= 1


# --------------------------------------------------------------------------- #
# auth (mocked HTTP)
# --------------------------------------------------------------------------- #
class _FakeResp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        return _FakeResp(self.payload)


def _expires(dt: datetime) -> str:
    # KiwoomAuth treats expires_dt as KST; build a KST string from a UTC instant.
    kst = dt.astimezone(timezone(timedelta(hours=9)))
    return kst.strftime("%Y%m%d%H%M%S")


def test_auth_issues_and_caches_token():
    clock = {"t": NOW}
    far = _expires(NOW + timedelta(hours=6))
    session = _FakeSession({"token": "TOK123", "token_type": "bearer",
                            "expires_dt": far, "return_code": 0, "return_msg": ""})
    auth = KiwoomAuth("APPKEY", "SUPERSECRET", env="mock",
                      session=session, clock=lambda: clock["t"])

    assert auth.get_token() == "TOK123"
    assert len(session.calls) == 1
    # Posts to the mock token endpoint with the documented body.
    body = session.calls[0]["json"]
    assert session.calls[0]["url"] == "https://mockapi.kiwoom.com/oauth2/token"
    assert body["grant_type"] == "client_credentials"
    assert body["appkey"] == "APPKEY" and body["secretkey"] == "SUPERSECRET"

    # Cached: a second call within validity does NOT re-post.
    assert auth.get_token() == "TOK123"
    assert len(session.calls) == 1


def test_auth_refreshes_after_expiry():
    clock = {"t": NOW}
    soon = _expires(NOW + timedelta(seconds=30))   # within the 60s refresh margin
    session = _FakeSession({"token": "TOK1", "expires_dt": soon, "return_code": 0})
    auth = KiwoomAuth("A", "B", env="mock", session=session, clock=lambda: clock["t"])
    assert auth.get_token() == "TOK1"

    session.payload = {"token": "TOK2", "expires_dt": _expires(NOW + timedelta(hours=6)),
                       "return_code": 0}
    # Still "now": token is within the refresh margin -> re-issues.
    assert auth.get_token() == "TOK2"
    assert len(session.calls) == 2


def test_auth_failure_is_clear_and_hides_secret():
    session = _FakeSession({"return_code": 3, "return_msg": "IP not allowlisted"})
    auth = KiwoomAuth("APPKEY", "SUPERSECRET", env="mock", session=session,
                      clock=lambda: NOW)
    with pytest.raises(KiwoomAuthError) as exc:
        auth.get_token()
    msg = str(exc.value)
    assert "IP not allowlisted" in msg
    assert "SUPERSECRET" not in msg     # never leak the secret in errors


def test_auth_requires_credentials():
    with pytest.raises(KiwoomAuthError):
        KiwoomAuth("", "", env="mock")
