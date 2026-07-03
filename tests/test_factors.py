"""Value/quality/multi cross-sectional factors — mock data, no network."""

import numpy as np
import pandas as pd

from tagent.data.kr_fundamental import metric_panel
from tagent.strategies.factor_utils import avg_panels, xs_zscore
from tagent.strategies.multi_factor import combine_signals
from tagent.strategies.quality_factor import quality_signal, roe_panel
from tagent.strategies.value_factor import value_signal
from tagent.xs_momentum import XSMomConfig, backtest
from tagent.xs_momentum_validate import walk_forward_signal


# --------------------------------------------------------------------------- #
# factor ranking
# --------------------------------------------------------------------------- #
def test_value_signal_ranks_cheap_highest_and_drops_nonpositive():
    d = pd.to_datetime(["2020-01-01", "2020-02-01"])
    pbr = pd.DataFrame({"A": [1.0, 1.0], "B": [2.0, 2.0], "C": [3.0, 3.0]}, index=d)
    sig = value_signal(pbr=pbr)
    assert sig.loc[d[0]].idxmax() == "A" and sig.loc[d[0]].idxmin() == "C"   # cheapest -> highest
    # a non-positive PBR (negative book) is dropped, not treated as 'cheap'
    pbr2 = pd.DataFrame({"A": [1.0], "B": [2.0], "C": [-5.0]}, index=d[:1])
    s2 = value_signal(pbr=pbr2)
    assert pd.isna(s2.loc[d[0], "C"]) and s2.loc[d[0], "A"] > s2.loc[d[0], "B"]


def test_value_signal_blends_pbr_and_per():
    d = pd.to_datetime(["2020-01-01"])
    pbr = pd.DataFrame({"A": [1.0], "B": [2.0], "C": [3.0]}, index=d)
    per = pd.DataFrame({"A": [30.0], "B": [10.0], "C": [5.0]}, index=d)   # C cheapest on PER
    sig = value_signal(pbr=pbr, per=per)
    assert sig.notna().loc[d[0]].all()                                   # blended for all three


def test_quality_signal_ranks_high_roe():
    d = pd.to_datetime(["2020-01-01"])
    eps = pd.DataFrame({"A": [2.0], "B": [1.0], "C": [0.5]}, index=d)
    bps = pd.DataFrame({"A": [10.0], "B": [10.0], "C": [10.0]}, index=d)  # ROE 0.20/0.10/0.05
    q = quality_signal(eps, bps)
    assert q.loc[d[0]].idxmax() == "A" and q.loc[d[0]].idxmin() == "C"
    assert np.allclose(roe_panel(eps, bps).loc[d[0]].to_numpy(), [0.2, 0.1, 0.05])


def test_xs_zscore_and_avg_panels():
    d = pd.to_datetime(["2020-01-01"])
    p = pd.DataFrame({"A": [1.0], "B": [2.0], "C": [3.0]}, index=d)
    z = xs_zscore(p)
    assert abs(z.loc[d[0]].mean()) < 1e-9 and z.loc[d[0], "C"] > z.loc[d[0], "A"]
    # avg_panels skips NaN: B missing in one panel -> averaged from the other
    q = pd.DataFrame({"A": [10.0], "B": [np.nan], "C": [30.0]}, index=d)
    a = avg_panels([p, q])
    assert a.loc[d[0], "B"] == 2.0 and a.loc[d[0], "A"] == (1.0 + 10.0) / 2


def test_combine_signals_blends_zscores():
    d = pd.to_datetime(["2020-01-01"])
    v = pd.DataFrame({"A": [2.0], "B": [-1.0], "C": [-1.0]}, index=d)
    qf = pd.DataFrame({"A": [-1.0], "B": [2.0], "C": [-1.0]}, index=d)
    c = combine_signals([v, qf])
    assert set(c.columns) == {"A", "B", "C"} and c.notna().loc[d[0]].all()


# --------------------------------------------------------------------------- #
# point-in-time / no-lookahead fundamentals panel
# --------------------------------------------------------------------------- #
def test_metric_panel_asof_no_lookahead():
    long = pd.DataFrame({"date": ["2020-01-01", "2020-01-01", "2020-02-01"],
                         "ticker": ["A", "B", "A"], "PBR": [1.0, 2.0, 1.5]})
    daily = pd.date_range("2020-01-01", "2020-02-28", freq="D")
    p = metric_panel(long, "PBR", daily, symbols=["A", "B"])
    assert p.loc["2020-01-15", "A"] == 1.0                  # Jan uses the Jan snapshot
    assert p.loc["2020-02-15", "A"] == 1.5                  # Feb uses the Feb snapshot
    # a FUTURE snapshot must not change a past day's value
    long2 = pd.concat([long, pd.DataFrame({"date": ["2020-03-01"], "ticker": ["A"], "PBR": [9.0]})])
    p2 = metric_panel(long2, "PBR", daily, symbols=["A", "B"])
    assert p2.loc["2020-01-15", "A"] == 1.0 and p2.loc["2020-02-15", "A"] == 1.5


# --------------------------------------------------------------------------- #
# engine ranks on the injected signal (long-only top quantile, members only)
# --------------------------------------------------------------------------- #
def _price_panel(n=80, k=4, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return {f"S{i}": pd.DataFrame({"close": 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))},
                                  index=idx) for i in range(k)}


def _const_signal(panel, values):
    idx = next(iter(panel.values())).index
    return pd.DataFrame({c: [values[c]] * len(idx) for c in panel}, index=idx)


def test_backtest_ranks_on_signal_long_only_top_quantile():
    panel = _price_panel()
    sig = _const_signal(panel, {"S0": 3.0, "S1": 2.0, "S2": 1.0, "S3": 0.0})   # S0,S1 best
    cfg = XSMomConfig(lookback=5, rebalance=5, top_q=0.4, allow_short=False,
                      cost_bps=0, slippage_bps=0)
    r = backtest(panel, cfg, periods_per_year=252, signal=sig)
    w = r["weights"]
    assert np.allclose(w[["S2", "S3"]].to_numpy(), 0.0)        # bottom half never held
    invested = w[(w != 0).any(axis=1)]
    assert (invested.to_numpy() >= -1e-12).all()              # long-only
    assert np.allclose(invested.sum(axis=1).to_numpy(), 1.0)  # fully invested, equal-weight


def test_backtest_signal_respects_membership():
    panel = _price_panel()
    idx = next(iter(panel.values())).index
    sig = _const_signal(panel, {"S0": 3.0, "S1": 2.0, "S2": 1.0, "S3": 0.0})
    memb = pd.DataFrame(False, index=idx, columns=list(panel))
    memb[["S2", "S3"]] = True                                 # only S2,S3 eligible
    cfg = XSMomConfig(lookback=5, rebalance=5, top_q=0.5, allow_short=False)
    w = backtest(panel, cfg, periods_per_year=252, signal=sig, membership=memb)["weights"]
    assert np.allclose(w[["S0", "S1"]].to_numpy(), 0.0)       # top-signal names excluded by PIT mask


def test_walk_forward_signal_chronological():
    panel = _price_panel(n=120)
    sig = _const_signal(panel, {"S0": 3.0, "S1": 2.0, "S2": 1.0, "S3": 0.0})
    cfg = XSMomConfig(lookback=5, rebalance=5, allow_short=False, cost_bps=0, slippage_bps=0)
    wf = walk_forward_signal(panel, sig, top_qs=(0.25, 0.5), train_bars=40, test_bars=20,
                             cfg=cfg, periods_per_year=252)
    assert wf["n_folds"] >= 2
    for f in wf["folds"]:
        assert f["train_end"] < f["test_start"]               # train precedes test
        assert f["chosen_top_q"] in (0.25, 0.5)
