"""YouTuber 공매도 signal trial — signal construction, no-lookahead, cost, calendar-time t.
Synthetic wide frames, no network."""

import numpy as np
import pandas as pd

from tagent.short_signal import (
    backtest, by_year, calendar_time_t, evaluate, is_degenerate, long_short_weights,
    short_ratio_wide, signal_change,
)


# --------------------------------------------------------------------------- #
# 1) signal construction — ratio, release-delayed 3-day change, dollar-neutral weights
# --------------------------------------------------------------------------- #
def test_short_ratio_and_release_delayed_change():
    idx = pd.date_range("2022-01-03", periods=8, freq="B")
    sd = {"A": pd.DataFrame({"short_volume": [10, 20, 30, 40, 50, 60, 70, 80],
                             "volume": [100] * 8}, index=idx)}
    sr = short_ratio_wide(sd)
    assert np.allclose(sr["A"].to_numpy(), [.1, .2, .3, .4, .5, .6, .7, .8])
    chg = signal_change(sr, lookback=3, release_delay=1)
    # release-delay 1 then 3-day change: chg[4] = sr[3]-sr[0] = 0.4-0.1 = 0.3
    assert abs(chg["A"].iloc[4] - 0.3) < 1e-9
    assert chg["A"].iloc[:4].isna().all()                     # 1 release + 3 lookback = no signal yet


def test_long_short_weights_dollar_neutral_overhang_and_ban_zeroed():
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    sig = pd.DataFrame({"A": [1.0, 1.0], "B": [-1.0, -1.0], "C": [0.0, 0.0]}, index=idx)
    w = long_short_weights(sig, overhang=True)
    assert abs(w.iloc[0].sum()) < 1e-12                       # dollar-neutral
    assert abs(w.iloc[0].abs().sum() - 1.0) < 1e-9            # gross = 1
    assert w.iloc[0]["A"] == -0.5 and w.iloc[0]["B"] == 0.5   # overhang SHORTS high-short A
    ban = pd.Series([False, True], index=idx)
    wb = long_short_weights(sig, overhang=True, ban_mask=ban)
    assert (wb.iloc[1] == 0).all() and not (wb.iloc[0] == 0).all()


def test_is_degenerate_flags_no_cross_section():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    assert is_degenerate(pd.DataFrame({"A": [1.] * 5, "B": [1.] * 5}, index=idx)) is True
    assert is_degenerate(pd.DataFrame({"A": [1., 2, 3, 4, 5], "B": [5., 4, 3, 2, 1]}, index=idx)) is False


# --------------------------------------------------------------------------- #
# 2) no-lookahead — a signal at t earns only from the NEXT session
# --------------------------------------------------------------------------- #
def test_backtest_no_lookahead_and_hand_computed_value():
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    close = pd.DataFrame({"A": 100 * (1.01 ** np.arange(10)), "B": 100 * np.ones(10)}, index=idx)
    sig = pd.DataFrame(0.0, index=idx, columns=["A", "B"])
    sig.iloc[5] = [1.0, -1.0]                                 # signal only on day index 5
    net = backtest(sig, close, hold=2, enter_lag=1, cost_round_trip=0.0)
    # the day-5 signal cannot affect any return on day <= 5; first held day is 6
    assert (net.index <= idx[5]).sum() == 0 and net.index.min() == idx[6]
    # hand value at day 6: active = mean(w5, w4)=mean([-.5,.5],0)=[-.25,.25]; retA=0.01,retB=0
    assert abs(net.loc[idx[6]] - (-0.25 * 0.01)) < 1e-12


# --------------------------------------------------------------------------- #
# 3) cost — a higher round trip lowers net by exactly the extra turnover cost
# --------------------------------------------------------------------------- #
def test_backtest_cost_reduces_net_by_turnover():
    idx = pd.date_range("2024-01-01", periods=12, freq="B")
    rng = np.random.default_rng(3)
    close = pd.DataFrame({s: 100 * np.cumprod(1 + rng.normal(0, 0.01, 12)) for s in ("A", "B", "C")},
                         index=idx)
    sig = pd.DataFrame(rng.normal(0, 1, (12, 3)), index=idx, columns=["A", "B", "C"])
    n1 = backtest(sig, close, hold=3, cost_round_trip=0.0020)
    n2 = backtest(sig, close, hold=3, cost_round_trip=0.0040)
    diff = (n1 - n2).dropna()
    assert (diff >= -1e-12).all() and diff.sum() > 0          # extra cost only ever lowers net
    gross = backtest(sig, close, hold=3, cost_round_trip=0.0)
    assert (gross >= n1 - 1e-12).all()                        # zero-cost dominates costed


# --------------------------------------------------------------------------- #
# 4) calendar-time Newey-West t + by-year
# --------------------------------------------------------------------------- #
def test_calendar_time_t_strong_vs_noise():
    idx = pd.date_range("2022-01-03", periods=400, freq="B")
    rng = np.random.default_rng(7)
    strong = pd.Series(0.001 + rng.normal(0, 0.0005, 400), index=idx)
    t, n = calendar_time_t(strong, lag=5)
    assert n == 400 and t > 5                                 # persistent positive mean -> high t
    noise = pd.Series(rng.normal(0, 0.01, 400), index=idx)
    t2, _ = calendar_time_t(noise, lag=5)
    assert abs(t2) < 2.5                                      # zero-mean noise -> insignificant


def test_evaluate_clears_only_with_t_survival_and_stability():
    idx = pd.date_range("2022-01-03", periods=520, freq="B")  # ~2 calendar years
    rng = np.random.default_rng(1)
    good = pd.Series(0.001 + rng.normal(0, 0.003, 520), index=idx)
    ev = evaluate(good, good, t_bar=2.5)
    assert ev["clears"] is True and ev["t_2x"] >= 2.5 and ev["survives_cost"] is True
    assert ev["stable_by_year"] is True and len(ev["by_year"]) >= 2
    flat = pd.Series(rng.normal(0, 0.01, 520), index=idx)     # zero-mean -> no edge
    assert evaluate(flat, flat, t_bar=2.5)["clears"] is False
    # a book that loses net of cost does NOT clear even if |t| is large
    losing = pd.Series(-0.001 + rng.normal(0, 0.002, 520), index=idx)
    assert evaluate(losing, losing, t_bar=2.5)["clears"] is False
