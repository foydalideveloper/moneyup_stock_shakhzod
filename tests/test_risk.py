import pytest

from tagent.risk import RiskManager, RiskParams


def make_rm():
    return RiskManager(RiskParams(
        account_equity=100_000.0,
        risk_per_trade_pct=1.0,     # risk $1,000/trade
        max_position_pct=10.0,      # max $10,000/position
        stop_loss_pct=2.0,
        take_profit_pct=4.0,
        max_daily_loss_pct=3.0,     # halt after -$3,000
    ))


def test_position_size_capped_by_value():
    rm = make_rm()
    # risk: $1000 / $2 per share = 500 shares; value cap: $10000 / $100 = 100.
    assert rm.position_size(entry_price=100.0, stop_price=98.0) == 100


def test_position_size_uses_default_stop_when_none():
    rm = make_rm()
    # default stop 2% -> $2 per-share risk -> same 100 (value-capped).
    assert rm.position_size(entry_price=100.0) == 100


def test_stop_take_prices_long_and_short():
    rm = make_rm()
    stop, take = rm.stop_take_prices(100.0, "buy")
    assert stop == pytest.approx(98.0)
    assert take == pytest.approx(104.0)
    stop, take = rm.stop_take_prices(100.0, "sell")
    assert stop == pytest.approx(102.0)
    assert take == pytest.approx(96.0)


def test_daily_loss_trips_kill_switch():
    rm = make_rm()
    assert rm.can_trade()
    rm.register_trade_result(-3000.0)          # hits the -3% limit exactly
    assert not rm.can_trade()
    assert rm.position_size(100.0, 98.0) == 0   # no sizing while halted
    rm.reset_day()
    assert rm.can_trade()


def test_kill_switch_manual():
    rm = make_rm()
    rm.trip_kill_switch("stale feed")
    assert not rm.can_trade()
    assert "stale feed" in rm.halt_reason


def test_invalid_side_raises():
    rm = make_rm()
    with pytest.raises(ValueError):
        rm.stop_take_prices(100.0, "hold")
