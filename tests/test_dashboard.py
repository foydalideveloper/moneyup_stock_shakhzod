"""Dashboard endpoint tests — fake data sources, FastAPI TestClient (no network)."""

import math
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from tagent.dashboard import (
    AlertLog, CryptoSource, TaTracker, candle_type, create_app, crypto_decision,
    poll_scorecard, stock_signal,
)
from tagent.scorecard import Scorecard
from tagent.feeds.base import OrderBookSnapshot
from tagent.feeds.crypto_feed import CryptoFeed
from tagent.risk import RiskManager, RiskParams

NOW = datetime(2026, 6, 4, 12, 0, tzinfo=timezone.utc)


def _risk():
    return RiskManager(RiskParams(account_equity=100_000.0))


# --------------------------------------------------------------------------- #
# pure signal mapping
# --------------------------------------------------------------------------- #
def test_stock_signal_buy_hold_sell():
    buy = stock_signal("AAPL", 0.80, 100.0, _risk())
    assert buy["action"] == "BUY" and buy["qty"] > 0 and buy["stop"] > 0
    hold = stock_signal("AAPL", 0.50, 100.0, _risk())
    assert hold["action"] == "HOLD" and hold["qty"] == 0
    sell = stock_signal("AAPL", 0.20, 100.0, _risk())
    assert sell["action"] == "SELL" and sell["qty"] == 0
    assert buy["confidence"] == 0.80


# --------------------------------------------------------------------------- #
# fake sources
# --------------------------------------------------------------------------- #
class FakeStockSource:
    def __init__(self):
        self.last_symbols = None

    def signals(self, symbols):
        self.last_symbols = list(symbols)
        return [{"symbol": s, "price": 100.0, "action": "BUY", "confidence": 0.7,
                 "qty": 10, "stop": 98.0, "target": 104.0} for s in symbols]

    def account(self):
        return {"equity": 100000.0, "cash": 50000.0, "day_pnl": 123.0,
                "day_pnl_pct": 0.12, "positions": [{"symbol": "AAPL", "qty": 5,
                "market_value": 1500.0, "unrealized_pl": 20.0}]}

    def candles(self, symbol, interval="1m"):
        return {"symbol": symbol.upper(), "market": "us", "interval": interval,
                "candles": [{"t": NOW.isoformat(), "o": 100, "h": 101, "l": 99,
                             "c": 100.5, "v": 1000}]}

    def analysis(self, symbol):
        return {"symbol": symbol.upper(), "market": "us", "price": 100.5,
                "candle": {"type": "bullish", "body_pct": 50.0, "up": True},
                "rsi": 55.0, "momentum": 1.2, "probability": 0.7, "action": "BUY",
                "confidence": 0.7, "qty": 10, "stop": 98.0, "target": 104.0,
                "decision_source": "ML model"}

    def ta(self, symbol, interval="1m"):
        return {"symbol": symbol.upper(), "market": "us", "interval": interval,
                "lines": [{"type": "upper", "points": [[0, 101.0, "t0"], [9, 105.0, "t9"]],
                           "pivots": [[0, 101.0, "t0"], [5, 103.0, "t5"], [9, 105.0, "t9"]]}],
                "levels": [{"type": "resistance", "price": 105.0, "touches": 3}],
                "pivots": [{"type": "high", "index": 9, "price": 105.0, "t": "t9"}],
                "pattern": "ascending channel", "breakout": None,
                "updated_at": "2026-06-04T12:00:00+00:00", "reason": "new swing high confirmed",
                "explanation": "Drew an upper trendline — an ascending channel."}

    def alerts(self, limit=50, symbol=None):
        rows = [{"ts": "2026-06-04T12:00:05+00:00", "symbol": "AAPL", "market": "us",
                 "kind": "breakout", "text": "AAPL broke above the upper trendline",
                 "meta": {"direction": "up"}},
                {"ts": "2026-06-04T12:00:06+00:00", "symbol": "NVDA", "market": "us",
                 "kind": "breakout", "text": "NVDA broke below the lower trendline",
                 "meta": {"direction": "down"}}]
        if symbol and str(symbol).lower() != "all":
            rows = [r for r in rows if r["symbol"] == str(symbol).upper()]
        return rows


class FakeCryptoSource:
    def __init__(self):
        self.subscribed = []

    def orderbook(self, symbol):
        self.subscribed.append(symbol)
        return {"symbol": symbol.upper(), "status": "live",
                "asks": [[101.0, 1.0], [102.0, 2.0]], "bids": [[100.0, 1.5], [99.0, 2.5]],
                "best_ask": 101.0, "best_bid": 100.0, "ts": NOW.isoformat()}

    def memory(self, symbol):
        return {"symbol": symbol.upper(), "absorption_ratio": 0.9, "spoof_ratio": 0.1,
                "depth_imbalance": 0.2, "vanished": [
                    {"side": "ask", "price": 101.0, "qty": 1.0, "reason": "filled"}]}

    def candles(self, symbol, interval="1m"):
        return {"symbol": symbol.upper(), "market": "crypto", "interval": interval,
                "candles": [{"t": NOW.isoformat(), "o": 100, "h": 101, "l": 99,
                             "c": 100.5, "v": 5}]}

    def analysis(self, symbol):
        return {"symbol": symbol.upper(), "market": "crypto", "price": 100.5,
                "candle": {"type": "bearish", "body_pct": 40.0, "up": False},
                "rsi": 48.0, "momentum": -0.5, "imbalance": 0.3, "absorption": 0.9,
                "spoof": 0.1, "action": "BUY", "confidence": 0.3, "probability": None,
                "decision_source": "order-book memory"}

    def ta(self, symbol, interval="1m"):
        return {"symbol": symbol.upper(), "market": "crypto", "interval": interval,
                "lines": [], "levels": [], "pattern": "n/a", "breakout": None,
                "pivots": [], "updated_at": None, "reason": "first analysis",
                "explanation": "Not enough bars for analysis yet."}

    def ml(self, symbol):
        return {"symbol": symbol.upper(), "market": "crypto", "price": 100.5,
                "action": "BUY", "probability": 0.7, "confidence": 0.7,
                "decision_source": "ML model (kline tech + order-book micro)"}

    def funding(self, symbol):
        return {"symbol": symbol.upper(), "market": "crypto", "funding_rate": 0.0002,
                "funding_time": 1_700_000_000_000, "mark": 100.0}

    def alerts(self, limit=50, symbol=None):
        return []


@pytest.fixture
def client():
    return TestClient(create_app(FakeStockSource(), FakeCryptoSource()))


# --------------------------------------------------------------------------- #
# endpoints
# --------------------------------------------------------------------------- #
def test_signals_endpoint_uses_requested_symbols(client):
    r = client.get("/signals?symbols=AAPL,NVDA")
    assert r.status_code == 200
    sigs = r.json()["signals"]
    assert [s["symbol"] for s in sigs] == ["AAPL", "NVDA"]
    assert sigs[0]["action"] == "BUY"


def test_signals_empty_when_no_symbols(client):
    assert client.get("/signals").json()["signals"] == []


def test_account_endpoint(client):
    a = client.get("/account").json()
    assert a["equity"] == 100000.0 and a["day_pnl"] == 123.0
    assert a["positions"][0]["symbol"] == "AAPL"


def test_orderbook_endpoint(client):
    d = client.get("/orderbook?symbol=btcusdt").json()
    assert d["symbol"] == "BTCUSDT" and d["status"] == "live"
    assert d["asks"][0] == [101.0, 1.0] and d["bids"][0] == [100.0, 1.5]


def test_memory_endpoint(client):
    m = client.get("/memory?symbol=ethusdt").json()
    assert m["absorption_ratio"] == 0.9 and m["spoof_ratio"] == 0.1
    assert m["vanished"][0]["reason"] == "filled"


def test_index_serves_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "trading-agent" in r.text and "live analysis" in r.text


def test_endpoint_errors_are_graceful_not_500():
    class Boom:
        def signals(self, s): raise RuntimeError("alpaca down")
        def account(self): raise RuntimeError("alpaca down")
    c = TestClient(create_app(Boom(), FakeCryptoSource()))
    r = c.get("/signals?symbols=AAPL")
    assert r.status_code == 200 and "error" in r.json()      # graceful, not 500
    assert "error" in c.get("/account").json()


# --------------------------------------------------------------------------- #
# CryptoSource wiring: requesting an unsubscribed coin subscribes the feed
# --------------------------------------------------------------------------- #
def test_crypto_source_subscribes_on_request():
    feed = CryptoFeed(symbols=["BTCUSDT"])
    src = CryptoSource(feed, autostart=False)            # don't hit the network
    assert "ETHUSDT" not in feed.subscriptions
    src.orderbook("eth")                                 # request a new coin
    assert "ETHUSDT" in feed.subscriptions               # feed was subscribed
    # No data yet -> 'subscribing'; memory returns the (empty) feature dict.
    assert src.orderbook("eth")["status"] == "subscribing"
    assert src.memory("eth")["symbol"] == "ETHUSDT"


def test_crypto_source_serves_live_book_from_memory():
    feed = CryptoFeed(symbols=["BTCUSDT"])
    src = CryptoSource(feed, autostart=False)
    # Feed a depth snapshot through the feed's parser -> stored as memory._prev.
    feed.feed_message({"stream": "btcusdt@depth10@100ms", "data": {
        "bids": [["100.0", "1.0"], ["99.0", "2.0"]],
        "asks": [["101.0", "1.5"], ["102.0", "2.5"]]}}, now=NOW)
    d = src.orderbook("BTCUSDT")
    assert d["status"] == "live"
    assert d["best_bid"] == 100.0 and d["best_ask"] == 101.0
    assert d["asks"][0] == [101.0, 1.5]


def test_crypto_source_rejects_invalid_symbol():
    src = CryptoSource(CryptoFeed(symbols=["BTCUSDT"]), autostart=False)
    assert "error" in src.orderbook("!!")
    assert "error" in src.memory("@@")


# --------------------------------------------------------------------------- #
# pure helpers: candle classification + crypto microstructure decision
# --------------------------------------------------------------------------- #
def test_candle_type_classifies():
    assert candle_type(100, 105, 99, 104)["type"] == "bullish"
    assert candle_type(104, 105, 99, 100)["type"] == "bearish"
    assert candle_type(100, 110, 90, 100.3)["type"] == "doji"   # tiny body vs range
    assert candle_type(100, 105, 99, 104)["up"] is True


def test_crypto_decision_rules():
    assert crypto_decision(0.30, 0.9, 0.1)["action"] == "BUY"    # bid-heavy, absorbed
    assert crypto_decision(-0.30, 0.5, 0.2)["action"] == "SELL"  # ask-heavy
    assert crypto_decision(0.10, 0.9, 0.8)["action"] == "SELL"   # lots of spoof
    assert crypto_decision(0.05, 0.9, 0.1)["action"] == "HOLD"   # balanced
    assert crypto_decision(0.30, 0.9, 0.1)["confidence"] == 0.30


# --------------------------------------------------------------------------- #
# /candles and /analysis dispatch by market
# --------------------------------------------------------------------------- #
def test_candles_dispatches_by_market(client):
    us = client.get("/candles?market=us&symbol=AAPL").json()
    assert us["market"] == "us" and us["interval"] == "1m" and us["candles"]
    cx = client.get("/candles?market=crypto&symbol=btc").json()
    assert cx["market"] == "crypto" and cx["symbol"] == "BTC"


def test_candles_defaults_to_us(client):
    assert client.get("/candles?symbol=AAPL").json()["market"] == "us"


def test_candles_dispatches_timeframe(client):
    for tf in ("1m", "10m", "30m", "1h", "1d"):
        d = client.get(f"/candles?market=us&symbol=AAPL&interval={tf}").json()
        assert d["interval"] == tf
    cx = client.get("/candles?market=crypto&symbol=btc&interval=1h").json()
    assert cx["interval"] == "1h"


def test_ta_dispatches_timeframe(client):
    assert client.get("/ta?market=us&symbol=AAPL&interval=30m").json()["interval"] == "30m"
    assert client.get("/ta?market=crypto&symbol=btc&interval=1d").json()["interval"] == "1d"


def test_normalize_interval_falls_back():
    from tagent.dashboard import normalize_interval, TIMEFRAMES
    assert normalize_interval("1h") == "1h"
    assert normalize_interval("bogus") == "1m"          # unknown -> default
    assert set(TIMEFRAMES) == {"1m", "10m", "30m", "1h", "1d"}


def test_analysis_dispatches_by_market(client):
    us = client.get("/analysis?market=us&symbol=AAPL").json()
    assert us["market"] == "us" and us["action"] == "BUY"
    assert us["probability"] == 0.7 and us["candle"]["type"] == "bullish"
    cx = client.get("/analysis?market=crypto&symbol=eth").json()
    assert cx["market"] == "crypto" and cx["probability"] is None
    assert "imbalance" in cx and cx["decision_source"].startswith("order-book")


def test_ta_dispatches_by_market(client):
    us = client.get("/ta?market=us&symbol=AAPL").json()
    assert us["market"] == "us" and us["pattern"] == "ascending channel"
    assert us["lines"][0]["type"] == "upper" and "trendline" in us["explanation"]
    assert us["levels"][0]["type"] == "resistance"
    cx = client.get("/ta?market=crypto&symbol=btc").json()
    assert cx["market"] == "crypto" and cx["lines"] == []


def test_ta_defaults_to_us(client):
    assert client.get("/ta?symbol=AAPL").json()["market"] == "us"


def test_ta_error_is_graceful_not_500():
    class Boom:
        def candles(self, s): raise RuntimeError("alpaca down")
        def ta(self, s): raise RuntimeError("alpaca down")
    c = TestClient(create_app(Boom(), FakeCryptoSource()))
    r = c.get("/ta?market=us&symbol=AAPL")
    assert r.status_code == 200 and "error" in r.json()


def test_ta_from_candles_wires_analyze():
    # the dashboard helper turns chart candles into the TA overlay structure
    from tagent.dashboard import _ta_from_candles
    import numpy as np
    t = np.arange(60)
    close = 100 + 0.4 * t + 2.0 * np.sin(t / 2.0)        # rising channel
    candles = [{"t": NOW.isoformat(), "o": float(c), "h": float(c) + 0.5,
                "l": float(c) - 0.5, "c": float(c), "v": 1000.0} for c in close]
    out = _ta_from_candles(candles, "AAPL", "us")
    assert out["symbol"] == "AAPL" and out["market"] == "us"
    assert out["n_bars"] == 60 and out["lines"]
    assert any(l["type"] == "upper" for l in out["lines"])


def test_ta_from_candles_handles_too_few():
    from tagent.dashboard import _ta_from_candles
    out = _ta_from_candles([{"t": NOW.isoformat(), "o": 1, "h": 1, "l": 1, "c": 1, "v": 1}],
                           "BTC", "crypto")
    assert out["lines"] == [] and "explanation" in out


def test_ta_endpoint_exposes_pivots_and_line_details(client):
    d = client.get("/ta?market=us&symbol=AAPL").json()
    assert d["pivots"][0]["type"] == "high"            # marked dots for verification
    assert d["lines"][0]["pivots"][0] == [0, 101.0, "t0"]
    assert d["reason"] and d["updated_at"]             # why/when it last recomputed


# --------------------------------------------------------------------------- #
# /alerts endpoint + AlertLog + TaTracker
# --------------------------------------------------------------------------- #
def test_alerts_endpoint(client):
    d = client.get("/alerts?market=us").json()
    assert d["alerts"][0]["symbol"] == "AAPL"
    assert "broke above" in d["alerts"][0]["text"]


def test_alerts_endpoint_filters_by_symbol(client):
    d = client.get("/alerts?market=us&symbol=NVDA").json()
    assert [a["symbol"] for a in d["alerts"]] == ["NVDA"]      # only NVDA
    alld = client.get("/alerts?market=us&symbol=all").json()
    assert {a["symbol"] for a in alld["alerts"]} == {"AAPL", "NVDA"}


def test_scorecard_endpoint_filters_by_symbol(tmp_path):
    sc = Scorecard(data_dir=tmp_path, horizon_s=900)
    sc.observe("crypto-ob", "BTCUSDT", "BUY", 100.0, ts=NOW)
    sc.observe("crypto-ob", "ETHUSDT", "SELL", 50.0, ts=NOW)
    app = create_app(FakeStockSource(), FakeCryptoSource(), scorecard=sc)
    c = TestClient(app)
    btc = c.get("/scorecard?symbol=BTCUSDT").json()
    assert btc["symbol"] == "BTCUSDT"
    assert btc["sources"]["crypto-ob"]["total"] == 1                  # only BTC
    assert btc["sources"]["crypto-ob"]["recent"][0]["symbol"] == "BTCUSDT"
    assert btc["sources"]["crypto-ob"]["equity_start"] == 10000.0     # one $10k account
    allc = c.get("/scorecard?symbol=all").json()
    assert allc["sources"]["crypto-ob"]["total"] == 2                 # both coins
    assert allc["sources"]["crypto-ob"]["equity_start"] == 10000.0    # ONE $10k base (not summed)
    assert set(allc["symbols"]) == {"BTCUSDT", "ETHUSDT"}


def test_alerts_endpoint_graceful_error():
    class Boom:
        def alerts(self, limit=50, symbol=None): raise RuntimeError("boom")
    c = TestClient(create_app(Boom(), FakeCryptoSource()))
    assert "error" in c.get("/alerts?market=us").json()


def test_alertlog_dedups_and_orders_newest_first():
    log = AlertLog()
    a = dict(symbol="BTCUSDT", market="crypto", kind="breakout", ts="t1",
             meta={"t": "bar1", "direction": "up"})
    assert log.add(text="first", **a) is True
    assert log.add(text="dup", **a) is False           # same (symbol,kind,t,dir) -> ignored
    log.add(symbol="ETHUSDT", market="crypto", kind="breakout", text="second",
            ts="t2", meta={"t": "bar2", "direction": "down"})
    recent = log.recent()
    assert [x["symbol"] for x in recent] == ["ETHUSDT", "BTCUSDT"]   # newest first


def test_alertlog_respects_capacity_and_limit():
    log = AlertLog(capacity=3)
    for i in range(5):
        log.add(symbol="X", market="us", kind="breakout", text=f"a{i}", ts=str(i),
                meta={"t": f"bar{i}", "direction": "up"})
    assert len(log._items) == 3                          # trimmed to capacity
    assert [x["text"] for x in log.recent(2)] == ["a4", "a3"]


def _candle_series(closes, vols=None):
    vols = vols or [1000.0] * len(closes)
    return [{"t": f"2026-06-04T07:{i:02d}:00+00:00", "o": c, "h": c + 0.5,
             "l": c - 0.5, "c": c, "v": v} for i, (c, v) in enumerate(zip(closes, vols))]


def _fixed_clock(dt):
    return lambda: dt


def test_tatracker_enriches_pivots_and_lines_with_timestamps():
    tr = TaTracker(clock=_fixed_clock(NOW))
    closes = [100 + 0.4 * i + 2.0 * math.sin(i / 2.0) for i in range(48)]
    out = tr.analyze(_candle_series(closes), "AAPL", "us")
    assert out["pivots"] and all("t" in p for p in out["pivots"])
    for line in out["lines"]:
        assert all(len(pt) == 3 for pt in line["points"])      # [index, price, t]
        assert all(len(pv) == 3 for pv in line["pivots"])
    assert out["updated_at"] == NOW.isoformat() and out["reason"] == "first analysis"


def test_tatracker_fires_breakout_alert():
    log = AlertLog()
    tr = TaTracker(alert_log=log, clock=_fixed_clock(NOW))
    closes = [100 + (0.5 if i % 2 else -0.5) for i in range(30)] + [106.0]
    vols = [1000.0] * 30 + [6000.0]
    out = tr.analyze(_candle_series(closes, vols), "BTCUSDT", "crypto")
    assert out["breakout"]["direction"] == "up"
    assert out["reason"] == "up breakout"
    feed = log.recent()
    assert len(feed) == 1 and "broke above the upper trendline" in feed[0]["text"]
    assert "6.0x volume -> bullish" in feed[0]["text"]
    # re-analyzing the same bars does NOT duplicate the alert
    tr.analyze(_candle_series(closes, vols), "BTCUSDT", "crypto")
    assert len(log.recent()) == 1


def test_scorecard_endpoint_disabled_when_absent(client):
    d = client.get("/scorecard").json()
    assert d["enabled"] is False and d["sources"] == {}


def test_funding_live_endpoint_disabled_by_default(client):
    assert client.get("/funding_live").json() == {"enabled": False, "held": []}


def test_funding_live_endpoint_returns_status():
    status = {"enabled": True, "cycle": 4, "capital": 10000.0, "equity": 10020.0,
              "accrued": 25.0, "costs": 5.0, "pct_change": 0.2, "ann_yield_pct": 3.1,
              "n_held": 2,
              "held": [{"coin": "BTCUSDT", "weight": 0.6, "notional": 6000, "funding_bps": 2.1},
                       {"coin": "ETHUSDT", "weight": 0.4, "notional": 4000, "funding_bps": 1.4}],
              "margin": {"min_margin_ratio": 0.33, "headroom_pct": 32.5,
                         "leverage": 3.0, "maint_margin_rate": 0.005}}
    app = create_app(FakeStockSource(), FakeCryptoSource(), funding_live=lambda: status)
    d = TestClient(app).get("/funding_live").json()
    assert d["enabled"] is True and d["n_held"] == 2
    assert d["held"][0]["coin"] == "BTCUSDT" and d["margin"]["leverage"] == 3.0


def test_momentum_live_endpoint_disabled_by_default(client):
    assert client.get("/momentum_live").json() == {"enabled": False, "held": []}


def test_momentum_live_endpoint_returns_status():
    status = {"enabled": True, "strategy": "KR 12-1 momentum + regime filter (long-only)",
              "cycle": 3, "capital": 10000.0, "equity": 10500.0, "pct_change": 5.0,
              "basket_equity": 10200.0, "basket_pct": 2.0, "regime": "in-market",
              "exposure_pct": 100.0, "n_held": 2, "costs": 12.0, "ann_return_pct": 20.0,
              "universe_size": 100, "frac_in_market_pct": 67.0,
              "held": [{"symbol": "005930", "weight": 0.5, "notional": 5250.0},
                       {"symbol": "000660", "weight": 0.5, "notional": 5250.0}]}
    app = create_app(FakeStockSource(), FakeCryptoSource(), momentum_live=lambda: status)
    d = TestClient(app).get("/momentum_live").json()
    assert d["enabled"] is True and d["regime"] == "in-market" and d["n_held"] == 2
    assert d["held"][0]["symbol"] == "005930" and d["exposure_pct"] == 100.0


def test_trend_core_endpoint():
    status = {"enabled": True, "combined_exposure": 1.0, "size_cap_x": 0.68,
              "markets": {"KOSPI200": {"in_market": True, "weight": 0.4},
                          "S&P500": {"in_market": True, "weight": 0.6}},
              "drawdown_pct": -8.2, "dd_budget_pct": -22.0}
    app = create_app(FakeStockSource(), FakeCryptoSource(), trend_core=lambda: status)
    d = TestClient(app).get("/trend_core").json()
    assert d["enabled"] is True and d["size_cap_x"] == 0.68
    assert d["markets"]["S&P500"]["weight"] == 0.6
    # absent runner -> disabled, never 500s
    d2 = TestClient(create_app(FakeStockSource(), FakeCryptoSource())).get("/trend_core").json()
    assert d2 == {"enabled": False}


def test_forward_ops_endpoint():
    payload = {"enabled": True, "as_of": "2026-06-10", "candidates": [
        {"name": "vrp", "label": "Options VRP", "status": "tracking", "forward_days": 1}],
        "triggers": {"any_triggered": False, "n_triggers": 0, "checked": 6, "triggers": []}}
    app = create_app(FakeStockSource(), FakeCryptoSource(), forward_ops=lambda: payload)
    d = TestClient(app).get("/forward_ops").json()
    assert d["enabled"] is True and d["candidates"][0]["name"] == "vrp"
    assert d["triggers"]["any_triggered"] is False
    d2 = TestClient(create_app(FakeStockSource(), FakeCryptoSource())).get("/forward_ops").json()
    assert d2 == {"enabled": False, "candidates": []}


def test_intraday_demo_endpoint_disabled_by_default(client):
    assert client.get("/intraday_demo").json() == {"enabled": False}


def test_intraday_demo_endpoint_computes_from_ticks():
    from datetime import datetime, timezone
    from tagent.dashboard import IntradayDemoSource

    def ticks(symbol):
        T = lambda s: datetime(2026, 6, 9, 10, 0, s, tzinfo=timezone.utc)
        return [(T(5), 100, 1), (T(20), 102, 2), (T(40), 99, 1), (T(55), 101, 1)]

    src = IntradayDemoSource(tick_provider=ticks,
                             linked_provider=lambda s: {"SPY": -0.05})  # crash -> HALT
    app = create_app(FakeStockSource(), FakeCryptoSource(), intraday_source=src)
    d = TestClient(app).get("/intraday_demo?symbol=005930&interval=1min").json()
    assert d["enabled"] is True and d["symbol"] == "005930"
    assert d["n_candles"] == 1 and d["candles"][0]["color"] == "red"   # close>open
    assert d["candles"][0]["body"] == 1 and d["safety"] == "HALT"


def test_intraday_demo_endpoint_honors_interval_param():
    from datetime import datetime, timezone
    from tagent.dashboard import IntradayDemoSource

    def ticks(symbol):
        # ticks spanning three distinct minutes
        return [(datetime(2026, 6, 9, 10, mnt, sec, tzinfo=timezone.utc), 100 + mnt, 1)
                for mnt in (0, 1, 2) for sec in (5, 35)]

    src = IntradayDemoSource(tick_provider=ticks, interval="1s")
    c = TestClient(create_app(FakeStockSource(), FakeCryptoSource(), intraday_source=src))
    d1 = c.get("/intraday_demo?symbol=X&interval=1min").json()
    d5 = c.get("/intraday_demo?symbol=X&interval=5min").json()
    assert d1["interval"] == "1min" and d1["n_candles"] == 3      # one bar per minute
    assert d5["interval"] == "5min" and d5["n_candles"] == 1      # all three minutes in one bar


def test_news_endpoint_disabled_by_default(client):
    assert client.get("/news").json() == {"enabled": False, "alerts": []}


def test_news_endpoint_returns_payload():
    payload = {"enabled": True, "n_kr": 1, "n_us": 1,
               "safety": {"state": "HALT", "halt": True, "reasons": ["linked US news sentiment"],
                          "overnight_return": -0.04, "linked_sentiment": -0.6},
               "alerts": [{"ts": 200, "market": "us", "symbol": "NVDA", "title": "plunge",
                           "sentiment": "bearish", "type": "news"},
                          {"ts": 100, "market": "kr", "symbol": "005930", "title": "유상증자",
                           "sentiment": "bearish", "type": "dilution"}]}
    app = create_app(FakeStockSource(), FakeCryptoSource(), news=lambda: payload)
    d = TestClient(app).get("/news").json()
    assert d["enabled"] is True and d["safety"]["state"] == "HALT"
    assert d["alerts"][0]["symbol"] == "NVDA" and d["alerts"][0]["ts"] == 200


def test_media_endpoint_disabled_by_default(client):
    # absent -> disabled, and explicitly flagged not-a-signal
    assert client.get("/media").json() == {"enabled": False, "items": [], "signal": False}


def test_media_endpoint_returns_grounded_display_only_payload():
    payload = {"enabled": True, "label": "awareness only — not a trading signal", "signal": False,
               "grounded": True, "n": 1, "items": [{"stock": "000660", "stock_name": "SK하이닉스",
                   "channel": "한국경제TV", "video_title": "증시 브리핑",
                   "quote": "SK하이닉스가 엔비디아와 HBM 공급계약을 체결",
                   "timestamp_mmss": "01:23", "deeplink": "https://www.youtube.com/watch?v=v1&t=83s",
                   "source_link": "https://www.youtube.com/watch?v=v1&t=83s"}]}
    app = create_app(FakeStockSource(), FakeCryptoSource(), media=lambda: payload)
    d = TestClient(app).get("/media").json()
    assert d["enabled"] is True and d["signal"] is False and d["grounded"] is True
    it = d["items"][0]
    assert it["stock"] == "000660" and it["deeplink"].endswith("&t=83s") and it["quote"]
    # the media endpoint must never carry trading/halt fields
    assert "safety" not in d and "halt" not in d


def test_briefing_endpoint_disabled_by_default(client):
    assert client.get("/briefing").json() == {"enabled": False, "reports": {}, "breaking": []}


def test_briefing_endpoint_returns_persisted_payload():
    payload = {"enabled": True, "date": "2026-06-11", "breaking": [
                   {"breaking_category": "macro", "title": "CPI shock", "url": "https://x/cpi"}],
               "reports": {"recommendation": {"items": [
                   {"symbol": "005930", "agreement": "동의", "distinct_from_kiwoom": True,
                    "houses": ["미래에셋증권"], "cite": "https://mk.co.kr/s", "our_view": {"side": "buy"}}]}}}
    app = create_app(FakeStockSource(), FakeCryptoSource(), briefing=lambda: payload)
    d = TestClient(app).get("/briefing").json()
    assert d["enabled"] is True and d["breaking"][0]["breaking_category"] == "macro"
    assert d["reports"]["recommendation"]["items"][0]["cite"].startswith("http")


def test_orders_endpoint_present_and_absent(client):
    assert client.get("/orders").json() == {"enabled": False, "orders": [], "n": 0}
    payload = {"enabled": True, "n": 1, "orders": [
        {"time_kst": "06-12 17:05", "stock": "005930", "side": "BUY", "qty": 1, "price": 284000,
         "ord_no": "0001234", "status": "ACCEPTED", "return_code": 0, "return_msg": "ok",
         "env": "mock", "live": False}]}
    d = TestClient(create_app(FakeStockSource(), FakeCryptoSource(), orders=lambda: payload)).get("/orders").json()
    assert d["enabled"] is True and d["orders"][0]["ord_no"] == "0001234" and d["orders"][0]["live"] is False


def test_us_stock_source_degrades_gracefully_not_crash():
    from scripts.run_dashboard import _BrokenStockSource
    bs = _BrokenStockSource("No trained model")
    # every endpoint method the dashboard calls returns a clean payload — no AttributeError/crash
    assert bs.analysis("AAPL")["unavailable"] is True and "unavailable" in bs.analysis("AAPL")["note"].lower()
    assert bs.candles("AAPL")["candles"] == [] and bs.ta("AAPL")["breakout"] is None
    assert bs.signals(["AAPL"])[0]["action"] == "UNAVAILABLE" and bs.account()["unavailable"] is True
    # end-to-end through the dashboard endpoints -> graceful, never the old '_BrokenStockSource ...analysis'
    c = TestClient(create_app(bs, FakeCryptoSource()))
    assert c.get("/analysis?market=us&symbol=AAPL").json()["unavailable"] is True
    assert c.get("/signals?symbols=AAPL").json()["signals"][0]["action"] == "UNAVAILABLE"
    assert c.get("/candles?market=us&symbol=AAPL").json()["candles"] == []


def test_feed_endpoint_disabled_by_default(client):
    assert client.get("/feed").json() == {"enabled": False, "items": [], "n": 0,
                                          "note": "feed not wired"}


def test_feed_endpoint_passes_since_and_marks_new():
    captured = {}

    def fake_feed(since=None):
        captured["since"] = since
        return {"enabled": True, "date": "2026-06-11", "n": 1, "n_new": 1,
                "latest_detected_at": "2026-06-11T10:20:00",
                "items": [{"id": "u:1", "link": "https://x/1", "source": "naver",
                           "detected_at": "2026-06-11T10:20:00",
                           "published_at": "2026-06-11T09:00:00", "is_new": True}]}
    app = create_app(FakeStockSource(), FakeCryptoSource(), feed=fake_feed)
    d = TestClient(app).get("/feed?since=2026-06-11T10:10:00").json()
    assert captured["since"] == "2026-06-11T10:10:00"        # the client's watermark reaches the view
    assert d["enabled"] is True and d["items"][0]["is_new"] is True
    assert d["items"][0]["link"].startswith("http")          # cited; dual timestamps present
    assert d["items"][0]["detected_at"] and d["items"][0]["published_at"]


def test_leadlag_endpoint_disabled_by_default(client):
    assert client.get("/leadlag").json() == {"enabled": False}


def test_leadlag_endpoint_returns_status():
    status = {"enabled": True, "state": "ARMED", "armed": True, "threshold_pct": -2.0,
              "last_date": "2024-01-03", "last_us_overnight_pct": -3.1, "capital": 10000.0,
              "equity": 10120.0, "pct_change": 1.2, "n_trades": 5, "win_rate_pct": 60.0,
              "expectancy_pct": 0.18, "round_trip_cost_pct": 0.51, "first_ts": "2016-01-01"}
    app = create_app(FakeStockSource(), FakeCryptoSource(), leadlag=lambda: status)
    d = TestClient(app).get("/leadlag").json()
    assert d["enabled"] is True and d["state"] == "ARMED" and d["n_trades"] == 5


def test_pead_endpoint_disabled_by_default(client):
    assert client.get("/pead").json() == {"enabled": False}


def test_pead_endpoint_returns_status():
    status = {"enabled": True, "strategy": "PEAD", "hold": 20, "capital": 10000.0,
              "equity": 10350.0, "pct_change": 3.5, "n_trades": 40, "win_rate_pct": 52.0,
              "expectancy_pct": 0.6, "round_trip_cost_pct": 0.51, "n_open": 3, "state": "ARMED",
              "open_positions": [{"symbol": "005930", "entry": "2026-06-01"}], "first_ts": "2016-02-01"}
    app = create_app(FakeStockSource(), FakeCryptoSource(), pead=lambda: status)
    d = TestClient(app).get("/pead").json()
    assert d["enabled"] is True and d["state"] == "ARMED" and d["n_trades"] == 40
    assert d["open_positions"][0]["symbol"] == "005930"


def test_desk_endpoints_disabled_by_default(client):
    assert client.get("/desk").json() == {"enabled": False}
    assert client.get("/desk_session").json() == {"enabled": False, "rows": []}


def test_desk_endpoints_return_payloads():
    snap = {"enabled": True, "generated": "2026-06-10", "regime": "in-market",
            "buys": [{"symbol": "000660", "price": 200000, "stop": 184000, "reason": "momentum rank #1"}],
            "sells": [], "briefing": [{"symbol": "005930", "price": 60000, "change": 0.01}],
            "track": {"n_trades": 30, "win_rate_pct": 55.0, "expectancy_pct": 0.4, "equity": 10500.0,
                      "round_trip_cost_pct": 0.51},
            "labels": {"recommendations": "validated momentum edge"}}
    sess = lambda syms: [{"symbol": s, "price": 100.0, "rsi": 55.0, "vol_vs_avg": 1.2,
                          "us_shock": "OK", "label": "decision-support"} for s in syms]
    app = create_app(FakeStockSource(), FakeCryptoSource(), desk=lambda: snap, desk_session=sess)
    c = TestClient(app)
    d = c.get("/desk").json()
    assert d["enabled"] and d["buys"][0]["symbol"] == "000660" and d["regime"] == "in-market"
    ds = c.get("/desk_session?symbols=000660,005930").json()
    assert ds["enabled"] and [r["symbol"] for r in ds["rows"]] == ["000660", "005930"]


def test_scorecard_endpoint_reports_sources(tmp_path):
    sc = Scorecard(data_dir=tmp_path, horizon_s=900)
    sc.observe("us-ml", "AAPL", "BUY", 100.0, ts=NOW)
    app = create_app(FakeStockSource(), FakeCryptoSource(), scorecard=sc)
    d = TestClient(app).get("/scorecard").json()
    assert d["enabled"] is True and d["horizon_min"] == 15
    us = d["sources"]["us-ml"]
    assert us["equity_start"] == 10000.0 and us["trades"] == 1
    assert us["total"] == 1 and us["recent"][0]["symbol"] == "AAPL"


def test_poll_scorecard_feeds_signals_and_dedups(tmp_path):
    sc = Scorecard(data_dir=tmp_path, horizon_s=900)
    stock, crypto = FakeStockSource(), FakeCryptoSource()
    poll_scorecard(sc, stock, crypto, ["AAPL"], ["BTCUSDT"])
    # FakeStockSource.signals -> BUY AAPL ; crypto analysis -> BUY ; crypto ml -> BUY
    sources = {r["source"] for r in sc.scores}
    assert {"us-ml", "crypto-ob", "crypto-ml"} <= sources
    n_before = len(sc.scores)
    poll_scorecard(sc, stock, crypto, ["AAPL"], ["BTCUSDT"])   # same calls -> deduped
    assert len(sc.scores) == n_before


def test_poll_scorecard_scores_us_ta_and_crypto_ml_per_market(tmp_path):
    class StockTA:
        def signals(self, syms): return []
        def ta(self, sym, interval="1m"):
            return {"symbol": sym.upper(), "breakout": {"direction": "up", "price": 150.0}}

    class CryptoMLOnly:
        def analysis(self, sym): return {"error": "no book"}
        def ta(self, sym, interval="1m"):
            return {"symbol": sym.upper(),
                    "breakout": {"direction": "down", "price": 90.0}}
        def ml(self, sym):
            return {"symbol": sym.upper(), "market": "crypto", "action": "BUY", "price": 200.0}

    sc = Scorecard(data_dir=tmp_path, horizon_s=900)
    poll_scorecard(sc, StockTA(), CryptoMLOnly(), ["AAPL"], ["BTCUSDT"])
    srcs = {r["source"] for r in sc.scores}
    assert {"us-ta", "crypto-ml", "crypto-ta"} <= srcs      # per-market TA + crypto ML
    card = sc.scorecard()["sources"]
    assert card["us-ta"]["total"] == 1 and card["us-ta"]["trades"] == 1
    assert card["crypto-ml"]["total"] == 1 and card["crypto-ta"]["total"] == 1
    # us-ta gets its own $10k book per (source, symbol)
    assert card["us-ta"]["equity_start"] == 10000.0


def test_poll_scorecard_feeds_candle_patterns_with_breakdown(tmp_path):
    class StockTA:
        def signals(self, syms): return []
        def ta(self, sym, interval="1m"):
            return {"symbol": sym.upper(),
                    "breakout": {"direction": "up", "price": 150.0},
                    "candle_patterns": [
                        {"name": "engulfing", "direction": "bullish", "index": 99, "price": 150.0},
                        {"name": "pin_bar", "direction": "bearish", "index": 99, "price": 150.0}]}

    class CryptoNone:
        def analysis(self, sym): return {"error": "x"}
        def ta(self, sym, interval="1m"): return {"symbol": sym.upper(), "breakout": None,
                                                  "candle_patterns": []}
        def ml(self, sym): return {"error": "x"}
        def funding(self, sym): return {"error": "x"}

    sc = Scorecard(data_dir=tmp_path, horizon_s=900)
    poll_scorecard(sc, StockTA(), CryptoNone(), ["AAPL"], ["BTCUSDT"])
    tags = {r.get("tag") for r in sc.scores if r["source"] == "us-ta"}
    assert {"breakout", "engulfing", "pin_bar"} <= tags          # each is its own stream
    bp = sc.scorecard()["sources"]["us-ta"]["by_pattern"]
    assert "engulfing" in bp and "pin_bar" in bp and bp["engulfing"]["total"] == 1


def test_poll_scorecard_accrues_funding_carry_per_coin(tmp_path):
    sc = Scorecard(data_dir=tmp_path, horizon_s=900, funding_enter_bps=1.0)
    poll_scorecard(sc, FakeStockSource(), FakeCryptoSource(), ["AAPL"], ["BTCUSDT", "ETHUSDT"])
    srcs = {r["source"] for r in sc.scores}
    assert "funding-carry" in srcs
    card = sc.scorecard()["sources"]["funding-carry"]
    assert card["trades"] == 2 and card["equity_start"] == 10000.0   # BTC + ETH share ONE $10k base
    n = len(sc.scores)
    poll_scorecard(sc, FakeStockSource(), FakeCryptoSource(), ["AAPL"], ["BTCUSDT", "ETHUSDT"])
    assert len(sc.scores) == n                                       # same funding interval -> no churn


def test_all_scorecard_agents_stay_active_and_expose_demo_fields(tmp_path):
    """Every technique agent — the five no-edge predictors AND funding carry — keeps running
    and accumulating paper trades (none stopped or hidden), and the /scorecard payload carries
    the honest DEMO framing (demo flag + one-line desc) the dashboard renders."""
    class AllStock:                                      # emits both us-ml and us-ta signals
        def signals(self, syms):
            return [{"symbol": s, "price": 100.0, "action": "BUY"} for s in syms]
        def ta(self, sym, interval="1m"):
            return {"symbol": sym.upper(), "breakout": {"direction": "up", "price": 150.0}}

    class AllCrypto:                                     # emits crypto-ob, crypto-ml, crypto-ta, funding
        def analysis(self, sym):
            return {"symbol": sym.upper(), "price": 100.5, "action": "BUY"}
        def ml(self, sym):
            return {"symbol": sym.upper(), "action": "BUY", "price": 200.0}
        def ta(self, sym, interval="1m"):
            return {"symbol": sym.upper(), "breakout": {"direction": "down", "price": 90.0}}
        def funding(self, sym):
            return {"symbol": sym.upper(), "funding_rate": 0.0005, "funding_time": 1_700_000_000_000}

    sc = Scorecard(data_dir=tmp_path, horizon_s=900, funding_enter_bps=1.0)
    poll_scorecard(sc, AllStock(), AllCrypto(), ["AAPL"], ["BTCUSDT"])
    src = TestClient(create_app(AllStock(), AllCrypto(), scorecard=sc)).get("/scorecard").json()["sources"]

    # ALL SIX agents took at least one paper trade this pass — proof none is stopped/hidden
    for name in ("us-ml", "us-ta", "crypto-ob", "crypto-ml", "crypto-ta", "funding-carry"):
        assert src[name]["trades"] >= 1, f"{name} not active"

    # the no-edge predictors are flagged demo with an honest verdict; carry is the real edge
    for name in ("us-ml", "us-ta", "crypto-ob", "crypto-ml", "crypto-ta"):
        assert src[name]["demo"] is True and src[name]["desc"]
    assert src["funding-carry"]["demo"] is False
    assert src["funding-carry"]["desc"] == "delta-neutral funding capture — small real edge"


def test_tatracker_updated_at_only_advances_on_change():
    t0 = datetime(2026, 6, 4, 12, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 6, 4, 12, 0, 30, tzinfo=timezone.utc)
    clock = {"t": t0}
    tr = TaTracker(clock=lambda: clock["t"])
    closes = [100 + 0.4 * i + 2.0 * math.sin(i / 2.0) for i in range(48)]
    first = tr.analyze(_candle_series(closes), "AAPL", "us")
    clock["t"] = t1
    same = tr.analyze(_candle_series(closes), "AAPL", "us")   # identical bars
    assert same["updated_at"] == first["updated_at"]          # stamp held
    assert same["updated_at"] == t0.isoformat()


def test_analysis_error_is_graceful_not_500():
    class Boom:
        def candles(self, s): raise RuntimeError("alpaca down")
        def analysis(self, s): raise RuntimeError("alpaca down")
    c = TestClient(create_app(Boom(), FakeCryptoSource()))
    assert c.get("/analysis?market=us&symbol=AAPL").status_code == 200
    assert "error" in c.get("/analysis?market=us&symbol=AAPL").json()


def test_crypto_source_analysis_rejects_invalid_symbol():
    src = CryptoSource(CryptoFeed(symbols=["BTCUSDT"]), autostart=False)
    assert "error" in src.analysis("!!")       # invalid -> graceful, no network
    assert "error" in src.candles("@@")
