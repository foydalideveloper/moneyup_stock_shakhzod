"""US-shock bounce live paper trader — mock data, no network."""

import pandas as pd

from tagent.intraday_backtest import CostModel
from tagent.leadlag_live import LeadlagLiveConfig, LeadlagLiveTrader, is_armed

CFG = LeadlagLiveConfig(threshold=-0.02, capital=10_000.0, slippage_bps=15.0)   # round trip 0.51%


def test_is_armed_threshold():
    assert is_armed(-0.03, -0.02) is True
    assert is_armed(-0.02, -0.02) is True            # at the threshold
    assert is_armed(-0.01, -0.02) is False
    assert is_armed(None, -0.02) is False
    assert is_armed(float("nan"), -0.02) is False


def test_settle_books_only_on_armed_days_net_of_cost(tmp_path):
    tr = LeadlagLiveTrader(CFG, data_dir=tmp_path)
    cf = CFG.cost().round_trip_frac()
    # quiet US night -> OK, flat, no trade
    st = tr.settle("2024-01-02", us_overnight=-0.005, kr_intraday_ret=0.02)
    assert st["state"] == "OK" and st["n_trades"] == 0 and st["equity"] == 10_000.0
    # sharp US down-night -> ARMED, books EW KR intraday return net of cost
    st = tr.settle("2024-01-03", us_overnight=-0.03, kr_intraday_ret=0.015)
    assert st["state"] == "ARMED" and st["n_trades"] == 1
    expected_net = 0.015 - cf
    assert abs(tr.equity - 10_000.0 * (1 + expected_net)) < 1e-6
    assert st["win_rate_pct"] == 100.0
    assert abs(st["expectancy_pct"] - expected_net * 100) < 1e-9


def test_settle_dedups_by_date(tmp_path):
    tr = LeadlagLiveTrader(CFG, data_dir=tmp_path)
    tr.settle("2024-01-03", -0.03, 0.01)
    eq, n = tr.equity, tr.trades
    tr.settle("2024-01-03", -0.03, 0.01)             # same date replayed -> no double book
    assert tr.equity == eq and tr.trades == n


def test_persistence_across_restart(tmp_path):
    a = LeadlagLiveTrader(CFG, data_dir=tmp_path)
    a.settle("2024-01-03", -0.03, 0.02)
    a.settle("2024-01-04", -0.025, -0.01)
    b = LeadlagLiveTrader(CFG, data_dir=tmp_path)     # fresh instance reads state
    assert b.trades == a.trades == 2
    assert abs(b.equity - a.equity) < 1e-9
    assert b.status()["n_trades"] == 2
    assert (tmp_path / "leadlag_live_state.json").exists()
    assert (tmp_path / "leadlag_live.csv").exists()


def test_no_lookahead_armed_day_pending_until_return_known(tmp_path):
    # pre-close: armed but the day's KR return is not yet known -> no booking, still ARMED
    tr = LeadlagLiveTrader(CFG, data_dir=tmp_path)
    st = tr.settle("2024-01-03", us_overnight=-0.03, kr_intraday_ret=None)
    assert st["state"] == "ARMED" and st["n_trades"] == 0
    # post-close on the SAME day, the realised return is booked exactly once
    st = tr.settle("2024-01-03", us_overnight=-0.03, kr_intraday_ret=0.02)
    assert st["n_trades"] == 1
