"""Cross-sectional long-short tests — synthetic panels, no network/ML."""

import numpy as np
import pandas as pd

from tagent.cross_sectional import (
    benchmark_returns,
    cross_sectional_normalize,
    long_short_returns,
    make_rank_labels,
    purged_walk_forward_days,
    stats,
)


def _panel(days, symbols, seed=0):
    """A (day, stock) panel with a known feature 'sig' and ret_1d/fwd_ret."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in days:
        for s in symbols:
            rows.append({"date": pd.Timestamp(d), "symbol": s,
                         "sig": rng.normal(), "ret_1d": rng.normal(0, 0.02),
                         "fwd_ret": rng.normal(0, 0.05)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# labels
# --------------------------------------------------------------------------- #
def test_rank_labels_top_and_bottom_tercile():
    # One day, 6 stocks with distinct fwd returns -> top2=1, bottom2=0, mid2=NaN.
    df = pd.DataFrame({
        "date": [pd.Timestamp("2024-01-01")] * 6,
        "symbol": list("ABCDEF"),
        "fwd_ret": [-0.05, -0.02, 0.00, 0.01, 0.03, 0.10],
    })
    lab = make_rank_labels(df, q=3)
    assert lab.tolist() == [0.0, 0.0, np.nan, np.nan, 1.0, 1.0] or (
        # qcut tie-handling: bottom third 0, top third 1, middle NaN
        [0.0, 0.0] == lab.iloc[:2].tolist() and [1.0, 1.0] == lab.iloc[-2:].tolist())
    assert np.isnan(lab.iloc[2]) and np.isnan(lab.iloc[3])


def test_rank_labels_skip_thin_days():
    df = pd.DataFrame({"date": [pd.Timestamp("2024-01-01")] * 2,
                       "symbol": ["A", "B"], "fwd_ret": [0.1, -0.1]})
    lab = make_rank_labels(df, q=3)        # < q stocks -> all NaN
    assert lab.isna().all()


# --------------------------------------------------------------------------- #
# cross-sectional normalization
# --------------------------------------------------------------------------- #
def test_cross_sectional_normalize_zero_mean_per_day():
    df = _panel(["2024-01-01", "2024-01-02"], list("ABCDE"))
    out = cross_sectional_normalize(df, ["sig"])
    means = out.groupby("date")["sig"].mean()
    assert np.allclose(means.values, 0.0, atol=1e-9)
    stds = out.groupby("date")["sig"].std()
    assert np.allclose(stds.values, 1.0, atol=1e-6)


# --------------------------------------------------------------------------- #
# purged + embargoed walk-forward
# --------------------------------------------------------------------------- #
def test_purged_walk_forward_has_gap_no_leak():
    days = pd.date_range("2024-01-01", periods=120, freq="D")
    horizon, embargo = 5, 2
    folds = list(purged_walk_forward_days(days, n_splits=4, horizon=horizon, embargo=embargo))
    assert folds, "expected folds"
    for train_days, test_days in folds:
        # Train strictly precedes test with a >= horizon+embargo day gap (purge).
        gap_days = (pd.Timestamp(min(test_days)) - pd.Timestamp(max(train_days))).days
        assert gap_days >= horizon + embargo
        assert pd.Timestamp(max(train_days)) < pd.Timestamp(min(test_days))


# --------------------------------------------------------------------------- #
# long-short portfolio mechanics
# --------------------------------------------------------------------------- #
def _signal_panel(days, symbols):
    """ret_1d is monotone in a known score so the long-short MUST be positive."""
    rows = []
    for d in days:
        for i, s in enumerate(symbols):
            r = (i - (len(symbols) - 1) / 2) * 0.01     # spread of returns
            rows.append({"date": pd.Timestamp(d), "symbol": s,
                         "score": float(i), "ret_1d": r, "fwd_ret": r})
    return pd.DataFrame(rows)


def test_long_short_profits_from_perfect_ranking():
    days = pd.date_range("2024-01-01", periods=10, freq="D")
    p = _signal_panel(days, list("ABCDEFGHIJ"))   # 10 names -> decile = 1 each leg
    net = long_short_returns(p, "score", top_q=0.1, allow_short=True,
                             cost_bps=0.0, slippage_bps=0.0)
    # Long the highest-return name, short the lowest -> positive every day.
    assert (net > 0).all()
    assert stats(net).total_return > 0


def test_costs_reduce_return():
    days = pd.date_range("2024-01-01", periods=10, freq="D")
    p = _signal_panel(days, list("ABCDEFGHIJ"))
    free = long_short_returns(p, "score", top_q=0.1, allow_short=True,
                              cost_bps=0.0, slippage_bps=0.0)
    costed = long_short_returns(p, "score", top_q=0.1, allow_short=True,
                                cost_bps=50.0, slippage_bps=50.0)
    assert stats(costed).total_return < stats(free).total_return


def test_long_short_market_neutral_to_common_shift():
    """Adding a constant to every stock's return must NOT change a $-neutral L/S."""
    days = pd.date_range("2024-01-01", periods=8, freq="D")
    p = _signal_panel(days, list("ABCDEFGHIJ"))
    base = long_short_returns(p, "score", top_q=0.1, allow_short=True,
                              cost_bps=0.0, slippage_bps=0.0)
    p2 = p.copy()
    p2["ret_1d"] = p2["ret_1d"] + 0.05            # market-wide move
    shifted = long_short_returns(p2, "score", top_q=0.1, allow_short=True,
                                 cost_bps=0.0, slippage_bps=0.0)
    assert np.allclose(base.values, shifted.values, atol=1e-12)


def test_long_only_is_not_neutral():
    days = pd.date_range("2024-01-01", periods=8, freq="D")
    p = _signal_panel(days, list("ABCDEFGHIJ"))
    base = long_short_returns(p, "score", top_q=0.1, allow_short=False,
                              cost_bps=0.0, slippage_bps=0.0)
    p2 = p.copy(); p2["ret_1d"] = p2["ret_1d"] + 0.05
    shifted = long_short_returns(p2, "score", top_q=0.1, allow_short=False,
                                 cost_bps=0.0, slippage_bps=0.0)
    # Long-only is exposed to the market move (+0.05 each day).
    assert np.allclose((shifted - base).values, 0.05, atol=1e-12)


# --------------------------------------------------------------------------- #
# benchmarks + stats
# --------------------------------------------------------------------------- #
def test_benchmarks_present_and_aligned():
    days = pd.date_range("2024-01-01", periods=10, freq="D")
    p = _signal_panel(days, list("ABCDE"))
    b = benchmark_returns(p)
    assert set(b) == {"index", "buy_hold"}
    assert len(b["index"]) == 10 and len(b["buy_hold"]) == 10


def test_stats_drawdown_and_sharpe_signs():
    net = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
    s = stats(net)
    assert s.n_days == 5
    assert s.max_drawdown <= 0.0
    assert isinstance(s.sharpe, float)


def test_normalize_constant_feature_becomes_zero_not_nan():
    """A feature constant across the day's cross-section -> 0 (so rows survive)."""
    df = pd.DataFrame({
        "date": [pd.Timestamp("2024-01-01")] * 4,
        "symbol": list("ABCD"),
        "varies": [1.0, 2.0, 3.0, 4.0],
        "constant": [5.0, 5.0, 5.0, 5.0],   # zero cross-sectional std
    })
    out = cross_sectional_normalize(df, ["varies", "constant"])
    assert out["constant"].notna().all()           # not NaN
    assert (out["constant"] == 0.0).all()           # mapped to 0
    assert abs(out["varies"].mean()) < 1e-9         # varying feature still z-scored
