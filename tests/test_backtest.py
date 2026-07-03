import pandas as pd
import pytest

from tagent.backtest import position_from_proba, run_backtest


def test_costs_reduce_returns():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    close = pd.Series([100.0, 110.0, 121.0], index=idx)  # +10% each step
    pos = pd.Series([1, 1, 1], index=idx)

    free = run_backtest(close, pos, cost_bps=0, slippage_bps=0)
    costed = run_backtest(close, pos, cost_bps=5, slippage_bps=2)

    assert free.stats["final_equity"] == pytest.approx(1.21, rel=1e-6)
    assert free.stats["final_equity"] > costed.stats["final_equity"]
    assert costed.stats["n_trades"] == 1
    assert free.stats["max_drawdown"] <= 0.0


def test_flat_position_makes_no_money_and_no_trades():
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    close = pd.Series([100.0, 105.0, 95.0, 110.0], index=idx)
    pos = pd.Series([0, 0, 0, 0], index=idx)
    res = run_backtest(close, pos, cost_bps=5, slippage_bps=2)
    assert res.stats["final_equity"] == pytest.approx(1.0)
    assert res.stats["n_trades"] == 0


def test_position_from_proba_threshold():
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    proba = pd.Series([0.40, 0.60, 0.55, 0.70], index=idx)
    pos = position_from_proba(proba, threshold=0.55)
    assert list(pos) == [0, 1, 1, 1]


def test_stats_keys_present():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    close = pd.Series([100, 101, 102, 101, 103], index=idx, dtype=float)
    pos = pd.Series([0, 1, 1, 0, 1], index=idx)
    res = run_backtest(close, pos)
    for key in ("total_return", "sharpe", "max_drawdown", "n_trades", "hit_rate"):
        assert key in res.stats
