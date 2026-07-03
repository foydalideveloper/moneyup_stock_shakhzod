"""Validation of cross-sectional momentum — synthetic panels, no network."""

import numpy as np
import pandas as pd

from tagent.xs_momentum import XSMomConfig, backtest
from tagent.xs_momentum_validate import (
    by_regime, by_year, robustness, walk_forward,
)


def _coin(prices, start="2022-01-01"):
    idx = pd.date_range(start, periods=len(prices), freq="D", tz="UTC")
    return pd.DataFrame({"close": np.asarray(prices, float)}, index=idx)


def _trending_panel(n=500, k=10, seed=0, persist=True, start="2022-01-01"):
    rng = np.random.default_rng(seed)
    drifts = np.linspace(-0.012, 0.012, k)
    out = {}
    for i in range(k):
        steps = rng.normal(drifts[i], 0.01, n)
        if not persist:
            steps *= np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
        out[f"C{i}"] = _coin(100 * np.cumprod(1 + np.r_[0.0, steps[:-1]]), start)
    return out


# --------------------------------------------------------------------------- #
# walk-forward
# --------------------------------------------------------------------------- #
def test_walk_forward_folds_are_chronological_and_disjoint():
    wf = walk_forward(_trending_panel(), lookbacks=(20, 40, 60),
                      train_bars=150, test_bars=60, cfg=XSMomConfig())
    assert wf["n_folds"] >= 2
    prev_end = None
    seen = []
    for f in wf["folds"]:
        assert f["train_end"] < f["test_start"]          # train strictly precedes test
        assert f["test_start"] <= f["test_end"]
        if prev_end is not None:
            assert f["test_start"] > prev_end            # folds don't overlap, roll forward
        prev_end = f["test_end"]
        seen.append(f["chosen_lookback"])
    assert all(lb in (20, 40, 60) for lb in seen)        # only the offered lookbacks chosen


def test_walk_forward_oos_is_concatenated_test_windows():
    wf = walk_forward(_trending_panel(), lookbacks=(20, 40, 60), train_bars=150, test_bars=60)
    assert isinstance(wf["oos"].index, pd.DatetimeIndex)
    assert wf["oos"].index.is_monotonic_increasing and not wf["oos"].index.has_duplicates


def test_walk_forward_no_lookahead_future_does_not_change_early_folds():
    panel = _trending_panel(n=400, seed=3)
    wf1 = walk_forward(panel, lookbacks=(20, 40), train_bars=120, test_bars=60)
    ext = {c: pd.concat([df, _coin([df["close"].iloc[-1] * 3] * 40,
                                   start="2023-12-01")]) for c, df in panel.items()}
    wf2 = walk_forward(ext, lookbacks=(20, 40), train_bars=120, test_bars=60)
    # the first fold's choice + realised OOS return are unchanged by appended future bars
    assert wf1["folds"][0]["chosen_lookback"] == wf2["folds"][0]["chosen_lookback"]
    assert abs(wf1["folds"][0]["test_return"] - wf2["folds"][0]["test_return"]) < 1e-9


def test_walk_forward_generalizes_on_persistent_trends():
    wf = walk_forward(_trending_panel(persist=True, seed=1, n=600),
                      lookbacks=(20, 40, 60), train_bars=200, test_bars=80,
                      cfg=XSMomConfig(cost_bps=0, slippage_bps=0))
    assert wf["stats"]["total_return"] > 0               # the edge survives OOS


# --------------------------------------------------------------------------- #
# robustness
# --------------------------------------------------------------------------- #
def test_robustness_axes_present_and_cost_monotone():
    panel = _trending_panel(persist=True, seed=2, n=400, k=12)
    rob = robustness(panel, XSMomConfig(lookback=40),
                     universes=(8, 12), quantiles=(0.2, 0.3),
                     rebalances=(7, 30), skips=(0, 7), costs=(5.0, 20.0))
    assert set(rob) == {"universe", "quantile", "rebalance", "skip_recent", "cost_bps"}
    assert all("sharpe" in r and "max_drawdown" in r for axis in rob.values() for r in axis)
    cost_rows = sorted(rob["cost_bps"], key=lambda r: r["value"])
    assert cost_rows[-1]["total_return"] <= cost_rows[0]["total_return"]   # more cost -> not better


def test_robustness_universe_changes_coin_count():
    panel = _trending_panel(persist=True, n=300, k=20)
    rob = robustness(panel, XSMomConfig(lookback=30), universes=(10, 20),
                     quantiles=(0.2,), rebalances=(7,), skips=(7,), costs=(5.0,))
    sizes = [r["value"] for r in rob["universe"]]
    assert sizes == [10, 20]


# --------------------------------------------------------------------------- #
# sub-period split
# --------------------------------------------------------------------------- #
def test_by_year_splits_calendar_years():
    panel = _trending_panel(persist=True, n=500, start="2022-06-01")   # spans 2022..2023
    r = backtest(panel, XSMomConfig(lookback=30))
    yr = by_year(r["net"])
    assert set(yr) >= {"2022", "2023"}
    assert all("sharpe" in v and "max_drawdown" in v for v in yr.values())


def test_by_regime_returns_bull_quiet_bear():
    panel = _trending_panel(persist=True, n=400)
    r = backtest(panel, XSMomConfig(lookback=30))
    nav = (1.0 + r["basket"]).cumprod()
    reg = by_regime(r["net"], nav, window=20)
    assert set(reg) == {"bull", "quiet", "bear"}
    assert sum(v["n"] for v in reg.values()) <= len(r["net"])   # partition, no double count
