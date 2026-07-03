"""Paper-trader decision/risk/order wiring — fully mocked, no network."""

import csv

import numpy as np
import pandas as pd
import pytest

from tagent.paper import Decision, PaperTrader, decide_long
from tagent.risk import RiskManager, RiskParams


def _risk(equity=100_000.0, **kw):
    return RiskManager(RiskParams(account_equity=equity, **kw))


# --------------------------------------------------------------------------- #
# pure decision logic
# --------------------------------------------------------------------------- #
def test_decide_buy_sizes_with_risk():
    r = _risk()
    d = decide_long("AAPL", proba=0.80, price=100.0, threshold=0.55,
                    held=set(), risk=r)
    assert d.action == "buy"
    # qty matches the risk layer's own sizing for the same stop.
    stop, take = r.stop_take_prices(100.0, "buy")
    assert d.qty == r.position_size(100.0, stop) > 0
    assert d.stop == stop and d.take == take


def test_decide_below_threshold_holds():
    d = decide_long("AAPL", 0.40, 100.0, 0.55, set(), _risk())
    assert d.action == "hold_low_proba" and d.qty == 0


def test_decide_skips_when_already_held():
    d = decide_long("AAPL", 0.90, 100.0, 0.55, {"AAPL"}, _risk())
    assert d.action == "skip_held" and d.qty == 0


def test_decide_blocked_when_risk_halted():
    r = _risk()
    r.trip_kill_switch("daily loss limit")
    d = decide_long("AAPL", 0.99, 100.0, 0.55, set(), r)
    assert d.action == "halted" and d.qty == 0


def test_decide_no_size_when_equity_tiny():
    # Equity too small for even 1 share within max_position_pct -> no_size.
    r = _risk(equity=10.0, max_position_pct=1.0)
    d = decide_long("AAPL", 0.99, 100.0, 0.55, set(), r)
    assert d.action == "no_size" and d.qty == 0


# --------------------------------------------------------------------------- #
# fakes for run_once
# --------------------------------------------------------------------------- #
class _Clock:
    def __init__(self, is_open): self.is_open = is_open


class _Pos:
    def __init__(self, symbol): self.symbol = symbol


class FakeTrading:
    def __init__(self, is_open=True, positions=()):
        self._open = is_open
        self._positions = [_Pos(s) for s in positions]
        self.orders = []

    def get_clock(self): return _Clock(self._open)
    def get_all_positions(self): return list(self._positions)
    def submit_order(self, req):
        self.orders.append(req); return {"id": "paper-order", "req": req}


class FakeData:
    """Not used directly — recent_bars/latest_price are monkeypatched per test."""


class FakePredictor:
    def __init__(self, proba): self.proba = proba; self.metrics = {}
    def predict_proba_latest(self, ohlcv): return self.proba


def _bars(n=60, seed=0):
    idx = pd.date_range("2026-01-01", periods=n, freq="D")
    rng = np.random.default_rng(seed)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)), index=idx).clip(lower=1)
    return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                         "close": close, "volume": 1e6}, index=idx)


def _make_trader(tmp_path, proba=0.9, is_open=True, positions=(), threshold=0.55,
                 price=100.0, monkeypatch=None):
    trading = FakeTrading(is_open=is_open, positions=positions)
    trader = PaperTrader(FakePredictor(proba), trading, FakeData(),
                         watchlist=["AAPL", "MSFT"], threshold=threshold,
                         log_path=str(tmp_path / "paper_trades.csv"))
    monkeypatch.setattr(trader, "recent_bars", lambda s: _bars())
    monkeypatch.setattr(trader, "latest_price", lambda s: price)
    return trader, trading


# --------------------------------------------------------------------------- #
# run_once wiring
# --------------------------------------------------------------------------- #
def test_run_once_places_paper_buys(tmp_path, monkeypatch):
    trader, trading = _make_trader(tmp_path, proba=0.9, monkeypatch=monkeypatch)
    decisions = trader.run_once()
    assert [d.action for d in decisions] == ["buy", "buy"]
    assert len(trading.orders) == 2
    o = trading.orders[0]
    assert o.symbol == "AAPL" and o.qty > 0
    # Order is a BUY (paper); side enum stringifies with BUY.
    assert "BUY" in str(o.side).upper()


def test_run_once_idle_when_market_closed(tmp_path, monkeypatch):
    trader, trading = _make_trader(tmp_path, is_open=False, monkeypatch=monkeypatch)
    assert trader.run_once() == []
    assert trading.orders == []


def test_run_once_skips_held_no_duplicate(tmp_path, monkeypatch):
    trader, trading = _make_trader(tmp_path, positions=("AAPL",), monkeypatch=monkeypatch)
    decisions = trader.run_once()
    actions = {d.symbol: d.action for d in decisions}
    assert actions["AAPL"] == "skip_held"
    assert actions["MSFT"] == "buy"
    assert [o.symbol for o in trading.orders] == ["MSFT"]   # only the unheld name


def test_run_once_halted_places_nothing(tmp_path, monkeypatch):
    trader, trading = _make_trader(tmp_path, monkeypatch=monkeypatch)
    trader.risk.trip_kill_switch("test halt")
    decisions = trader.run_once()
    assert all(d.action == "halted" for d in decisions)
    assert trading.orders == []


def test_acts_once_per_bar(tmp_path, monkeypatch):
    trader, trading = _make_trader(tmp_path, monkeypatch=monkeypatch)
    trader.run_once()
    assert len(trading.orders) == 2
    # Same bar again -> no new orders (one action per new bar).
    again = trader.run_once()
    assert again == [] and len(trading.orders) == 2


def test_decisions_logged_to_csv(tmp_path, monkeypatch):
    trader, _ = _make_trader(tmp_path, monkeypatch=monkeypatch)
    trader.run_once()
    trader.close()
    rows = list(csv.DictReader(open(tmp_path / "paper_trades.csv")))
    assert {r["symbol"] for r in rows} == {"AAPL", "MSFT"}
    assert all(r["action"] == "buy" for r in rows)
    assert all(float(r["qty"]) > 0 for r in rows)


def test_order_error_is_caught_and_logged(tmp_path, monkeypatch):
    trader, trading = _make_trader(tmp_path, monkeypatch=monkeypatch)
    def boom(req): raise RuntimeError("broker down")
    monkeypatch.setattr(trading, "submit_order", boom)
    decisions = trader.run_once()
    assert all(d.action == "order_error" for d in decisions)   # graceful, not a crash
