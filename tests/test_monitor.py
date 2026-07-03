"""Integration test: drive the Monitor with fake events (no network)."""

from datetime import datetime, timezone

from tagent.alerts import Alerter
from tagent.config import Settings
from tagent.feeds.base import OrderBookSnapshot, Quote, Trade
from tagent.monitor import Monitor


def test_pipeline_fires_alert(tmp_path):
    settings = Settings()
    settings.watchlist = ["X"]

    captured = []
    alerter = Alerter(cooldown_seconds=0.0, printer=captured.append)
    mon = Monitor(settings=settings, feed=None, alerter=alerter,
                  log_path=str(tmp_path / "ticks.csv"))

    now = datetime.now(timezone.utc)  # current time so the feed isn't "stale"
    mon.on_quote(Quote("X", 99.99, 100.0, 1, 1, now))  # tight spread
    mon.on_trade(Trade("X", 100.0, 1, now))            # at session low -> buy signal
    mon.close()

    assert any("X" in line for line in captured), captured


def test_kill_switch_blocks_alerts(tmp_path):
    settings = Settings()
    settings.watchlist = ["X"]

    captured = []
    alerter = Alerter(cooldown_seconds=0.0, printer=captured.append)
    mon = Monitor(settings=settings, feed=None, alerter=alerter,
                  log_path=str(tmp_path / "ticks.csv"))
    mon.risk.trip_kill_switch("test halt")

    now = datetime.now(timezone.utc)
    mon.on_quote(Quote("X", 99.99, 100.0, 1, 1, now))
    mon.on_trade(Trade("X", 100.0, 1, now))
    mon.close()

    assert captured == []  # halted -> nothing dispatched


def test_orderbook_events_feed_depth_memory(tmp_path):
    """Order-book events update the per-symbol memory without breaking ticks."""
    settings = Settings()
    settings.watchlist = ["X"]

    captured = []
    alerter = Alerter(cooldown_seconds=0.0, printer=captured.append)
    mon = Monitor(settings=settings, feed=None, alerter=alerter,
                  log_path=str(tmp_path / "ticks.csv"))

    now = datetime.now(timezone.utc)
    asks0 = [(100.0, 5), (100.1, 5), (100.2, 5)]
    bids0 = [(99.9, 5), (99.8, 5), (99.7, 5)]
    mon.on_orderbook(OrderBookSnapshot("X", now, asks0, bids0))

    # A print lifts the 100.0 ask, then the book steps up -> 'filled'.
    mon.on_trade(Trade("X", 100.0, 1, now))
    asks1 = [(100.1, 5), (100.2, 5), (100.3, 5)]
    bids1 = [(100.0, 5), (99.9, 5), (99.8, 5)]
    mon.on_orderbook(OrderBookSnapshot("X", now, asks1, bids1))
    mon.close()

    snap = mon.orderbook_snapshot("X")
    assert snap is not None
    assert snap["filled_qty"] == 5.0
    # tick handling still works in parallel: the trade updated the state.
    assert mon.states["X"].last_price == 100.0
