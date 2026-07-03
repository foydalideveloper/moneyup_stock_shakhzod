from datetime import datetime, timedelta, timezone

from tagent.state import SymbolState

BASE = datetime(2024, 1, 1, tzinfo=timezone.utc)


def t(seconds):
    return BASE + timedelta(seconds=seconds)


def test_quote_spread_and_mid():
    s = SymbolState("AAPL")
    s.update_quote(100.0, 101.0, 5, 7, t(0))
    assert s.spread == 1.0
    assert abs(s.spread_pct - (1.0 / 101.0 * 100.0)) < 1e-9
    assert s.mid_price == 100.5
    assert s.has_quote()


def test_session_and_window_extremes():
    s = SymbolState("AAPL", window_seconds=60)
    s.update_quote(99.0, 101.0, 1, 1, t(0))
    s.update_trade(100.0, 10, t(1))
    s.update_trade(105.0, 10, t(2))
    s.update_trade(95.0, 10, t(3))

    assert s.session_high == 105.0
    assert s.session_low == 95.0
    assert s.last_price == 95.0
    assert s.recent_high() == 105.0
    assert s.recent_low() == 95.0


def test_prev_window_high_excludes_current():
    s = SymbolState("AAPL", window_seconds=60)
    s.update_trade(100.0, 1, t(1))
    s.update_trade(101.0, 1, t(2))
    s.update_trade(100.5, 1, t(3))
    # Before this 4th trade, the window high was 101.
    s.update_trade(101.5, 1, t(4))
    assert s.prev_window_high == 101.0


def test_window_trimming():
    s = SymbolState("AAPL", window_seconds=60)
    s.update_trade(100.0, 1, t(1))
    s.update_trade(105.0, 1, t(2))
    # Far in the future -> older points drop out of the 60s window.
    s.update_trade(200.0, 1, t(200))
    assert s.recent_high() == 200.0
    assert s.recent_low() == 200.0
    assert s.session_low == 100.0  # session memory is not trimmed


def test_staleness():
    s = SymbolState("AAPL")
    s.update_trade(100.0, 1, t(10))
    assert not s.is_stale(30, now=t(10) + timedelta(seconds=5))
    assert s.is_stale(30, now=t(10) + timedelta(seconds=40))
