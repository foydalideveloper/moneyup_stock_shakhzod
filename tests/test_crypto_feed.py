"""Crypto feed tests — canned Binance payloads, no network."""

from datetime import datetime, timezone

import pytest

from tagent.feeds.base import OrderBookSnapshot, Quote, Trade
from tagent.feeds.crypto_feed import (
    CryptoFeed,
    normalize_symbol,
    parse_depth,
    parse_trade,
)
from tagent.orderbook import OrderBookMemory

NOW = datetime(2026, 6, 4, 12, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# symbol validation / normalization
# --------------------------------------------------------------------------- #
def test_normalize_bare_base_gets_default_quote():
    assert normalize_symbol("btc") == "BTCUSDT"
    assert normalize_symbol("ETH") == "ETHUSDT"
    assert normalize_symbol("doge") == "DOGEUSDT"


def test_normalize_keeps_full_pairs():
    assert normalize_symbol("ETHUSDT") == "ETHUSDT"
    assert normalize_symbol("ethbtc") == "ETHBTC"        # BTC is a known quote
    assert normalize_symbol("BTC/USDT") == "BTCUSDT"     # strips punctuation
    assert normalize_symbol("sol-usdc") == "SOLUSDC"


@pytest.mark.parametrize("bad", ["", "   ", "!!", "@@@", "x", "1", None, 123])
def test_normalize_rejects_invalid(bad):
    with pytest.raises(ValueError):
        normalize_symbol(bad)


def test_default_quote_override():
    assert normalize_symbol("btc", default_quote="USDC") == "BTCUSDC"


# --------------------------------------------------------------------------- #
# pure parsers (Binance partial depth + trade shapes)
# --------------------------------------------------------------------------- #
def _depth_data():
    # Binance: bids descending, asks ascending; values are strings.
    return {
        "lastUpdateId": 1,
        "bids": [["100.0", "1.0"], ["99.5", "2.0"], ["99.0", "3.0"]],
        "asks": [["100.5", "1.5"], ["101.0", "2.5"], ["101.5", "3.5"]],
    }


def test_parse_depth_ordering_and_types():
    ob = parse_depth("BTCUSDT", _depth_data(), NOW)
    assert isinstance(ob, OrderBookSnapshot)
    assert ob.symbol == "BTCUSDT" and ob.timestamp == NOW
    assert ob.asks == [(100.5, 1.5), (101.0, 2.5), (101.5, 3.5)]   # ascending
    assert ob.bids == [(100.0, 1.0), (99.5, 2.0), (99.0, 3.0)]     # descending
    assert ob.asks[0][0] < ob.asks[-1][0]
    assert ob.bids[0][0] > ob.bids[-1][0]


def test_parse_trade():
    data = {"e": "trade", "s": "BTCUSDT", "p": "100.25", "q": "0.7", "T": 1}
    t = parse_trade("BTCUSDT", data, NOW)
    assert isinstance(t, Trade)
    assert t.symbol == "BTCUSDT" and t.price == 100.25 and t.size == 0.7


# --------------------------------------------------------------------------- #
# feed_message dispatch (combined-stream wrapper)
# --------------------------------------------------------------------------- #
def _feed():
    feed = CryptoFeed(symbols=["BTCUSDT"])
    obs, trades, quotes = [], [], []
    feed.on_orderbook(obs.append)
    feed.on_trade(trades.append)
    feed.on_quote(quotes.append)
    return feed, obs, trades, quotes


def _depth_msg(symbol="btcusdt", data=None):
    return {"stream": f"{symbol}@depth10@100ms", "data": data or _depth_data()}


def _trade_msg(symbol="btcusdt", price="100.25", qty="0.7"):
    return {"stream": f"{symbol}@trade",
            "data": {"e": "trade", "s": symbol.upper(), "p": price, "q": qty, "T": 1}}


def test_feed_message_emits_orderbook_and_quote():
    feed, obs, _, quotes = _feed()
    feed.feed_message(_depth_msg(), now=NOW)
    assert len(obs) == 1 and obs[0].symbol == "BTCUSDT"
    assert len(quotes) == 1
    q = quotes[0]
    assert isinstance(q, Quote)
    assert q.bid_price == 100.0 and q.ask_price == 100.5      # best of each side
    assert q.bid_size == 1.0 and q.ask_size == 1.5


def test_feed_message_emits_trade():
    feed, _, trades, _ = _feed()
    feed.feed_message(_trade_msg(price="100.25", qty="0.7"), now=NOW)
    assert len(trades) == 1
    assert trades[0].symbol == "BTCUSDT" and trades[0].price == 100.25


def test_subscribe_ack_is_ignored():
    feed, obs, trades, quotes = _feed()
    feed.feed_message({"result": None, "id": 1}, now=NOW)
    assert obs == [] and trades == [] and quotes == []


# --------------------------------------------------------------------------- #
# per-coin OrderBookMemory updates (fill vs cancel attribution)
# --------------------------------------------------------------------------- #
def test_per_symbol_memory_updates_and_features():
    feed, _, _, _ = _feed()
    # First snapshot establishes the book.
    feed.feed_message(_depth_msg(), now=NOW)
    # A trade lifts the best ask (100.5), then it vanishes -> "filled".
    feed.feed_message(_trade_msg(price="100.5", qty="1.5"), now=NOW)
    data2 = {
        "bids": [["100.0", "1.0"], ["99.5", "2.0"], ["99.0", "3.0"]],
        "asks": [["101.0", "2.5"], ["101.5", "3.5"], ["102.0", "1.0"]],  # 100.5 gone
    }
    feed.feed_message(_depth_msg(data=data2), now=NOW)

    feats = feed.book_features("BTCUSDT")
    assert feats["symbol"] == "BTCUSDT"
    assert feats["bid_depth"] > 0 and feats["ask_depth"] > 0
    assert feats["vanished_asks"] >= 1
    assert feats["filled_qty"] >= 1.5            # the lifted 100.5 ask, qty 1.5


def test_each_coin_has_its_own_memory():
    feed = CryptoFeed(symbols=["BTCUSDT", "ETHUSDT"])
    assert set(feed.books) == {"BTCUSDT", "ETHUSDT"}
    assert feed.books["BTCUSDT"] is not feed.books["ETHUSDT"]
    feed.feed_message(_depth_msg("btcusdt"), now=NOW)
    # ETH memory untouched (no snapshot yet); BTC got one.
    assert feed.books["BTCUSDT"]._prev is not None
    assert feed.books["ETHUSDT"]._prev is None


# --------------------------------------------------------------------------- #
# dynamic subscribe / unsubscribe at runtime
# --------------------------------------------------------------------------- #
def test_subscribe_symbol_adds_book_and_control_message():
    feed = CryptoFeed(symbols=["BTCUSDT"])
    pair = feed.subscribe_symbol("sol")
    assert pair == "SOLUSDT"
    assert "SOLUSDT" in feed.subscriptions
    assert isinstance(feed.books["SOLUSDT"], OrderBookMemory)
    # A SUBSCRIBE control message was queued for the WS loop.
    ctrl = feed.pending_control()
    assert any(m["method"] == "SUBSCRIBE" and "solusdt@trade" in m["params"]
               for m in ctrl)


def test_subscribe_is_idempotent():
    feed = CryptoFeed(symbols=["BTCUSDT"])
    feed.subscribe_symbol("eth")
    n_before = len(feed.pending_control())
    feed.subscribe_symbol("ETHUSDT")   # same coin again
    assert feed.subscriptions.count("ETHUSDT") == 1
    assert len(feed.pending_control()) == n_before   # no extra control msg


def test_unsubscribe_symbol_removes_book_and_queues_message():
    feed = CryptoFeed(symbols=["BTCUSDT", "ETHUSDT"])
    assert feed.unsubscribe_symbol("eth") is True
    assert "ETHUSDT" not in feed.subscriptions
    assert "ETHUSDT" not in feed.books
    assert any(m["method"] == "UNSUBSCRIBE" and "ethusdt@trade" in m["params"]
               for m in feed.pending_control())
    # Unsubscribing something not subscribed is a clean no-op.
    assert feed.unsubscribe_symbol("xrp") is False


def test_subscribe_rejects_invalid_symbol():
    feed = CryptoFeed(symbols=["BTCUSDT"])
    with pytest.raises(ValueError):
        feed.subscribe_symbol("!!")
    assert feed.subscriptions == ["BTCUSDT"]   # unchanged


def test_constructor_normalizes_symbols():
    feed = CryptoFeed(symbols=["btc", "eth"])
    assert feed.subscriptions == ["BTCUSDT", "ETHUSDT"]
