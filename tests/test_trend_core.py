"""Trend-core lockdown — DD attribution, execution lag, continuous v2, sizing, spec file."""

import pathlib

import numpy as np
import pandas as pd

from tagent.trend_core import (
    ContinuousV2Spec, binary_trend_net, continuous_exposure, continuous_trend_net,
    drawdown_episode, execution_lag_compare, size_from_maxdd,
)


# --------------------------------------------------------------------------- #
# 1) drawdown episode attribution
# --------------------------------------------------------------------------- #
def test_drawdown_episode_dates_and_depth():
    idx = pd.date_range("2000-01-03", periods=8, freq="B")
    # equity 1.0 ->1.2 (peak) ->0.6 (trough, -50%) ->1.3 (recovers past peak)
    eq = pd.Series([1.0, 1.1, 1.2, 0.9, 0.6, 0.8, 1.25, 1.3], index=idx)
    net = eq.pct_change().dropna()
    ep = drawdown_episode(net)
    assert abs(ep["depth"] - (0.6 / 1.2 - 1.0)) < 1e-9          # -50%
    assert ep["peak_date"] == idx[2] and ep["trough_date"] == idx[4]
    assert ep["recovery_date"] == idx[6]                        # first bar back above the peak


def test_drawdown_episode_no_recovery():
    idx = pd.date_range("2000-01-03", periods=4, freq="B")
    net = pd.Series([1.0, 1.0, 1.0, 1.0], index=idx).pct_change().fillna(0.0)
    net.iloc[1:] = [-0.1, -0.1, -0.1]                            # monotone down
    ep = drawdown_episode(net)
    assert ep["recovery_date"] is None and ep["depth"] < 0


# --------------------------------------------------------------------------- #
# 2) execution lag — realistic next-bar is no-lookahead and (here) worse
# --------------------------------------------------------------------------- #
def _wavy_close(n=120, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2001-01-01", periods=n, freq="B")
    steps = rng.normal(0.0003, 0.02, n)
    return pd.Series(100 * np.cumprod(1 + np.r_[0.0, steps[:-1]]), index=idx)


def test_binary_next_bar_no_lookahead():
    close = _wavy_close()
    net = binary_trend_net(close, regime_ma=10, mode="next_bar")
    ext = pd.concat([close, pd.Series(
        float(close.iloc[-1]) * np.cumprod(1 + np.full(20, 0.002)),
        index=pd.date_range(close.index[-1] + pd.offsets.BDay(1), periods=20, freq="B"))])
    net2 = binary_trend_net(ext, regime_ma=10, mode="next_bar")
    common = net.index[:-2]
    assert np.allclose(net.reindex(common).to_numpy(), net2.reindex(common).to_numpy())


def test_execution_lag_compare_has_both_modes():
    close = _wavy_close()
    cmp = execution_lag_compare(close, regime_ma=10)
    assert set(cmp) == {"same_bar", "next_bar"}
    for m in cmp:
        assert "stats" in cmp[m] and "episode" in cmp[m]
    # same-bar credits the signal bar itself (lookahead) -> differs from realistic next-bar
    assert not np.allclose(cmp["same_bar"]["net"].reindex(cmp["next_bar"]["net"].index).fillna(0).to_numpy(),
                           cmp["next_bar"]["net"].fillna(0).to_numpy())


# --------------------------------------------------------------------------- #
# 3) continuous-exposure v2 — bounds, trend response, no-lookahead
# --------------------------------------------------------------------------- #
def _spec():
    return ContinuousV2Spec(regime_ma=10, std_window=10, vol_window=5)


def test_continuous_exposure_bounds_and_trend_response():
    n = 80
    idx = pd.date_range("2010-01-01", periods=n, freq="B")
    up = pd.Series(100 * np.cumprod(1 + np.full(n, 0.003)), index=idx)        # steady uptrend
    down = pd.Series(100 * np.cumprod(1 + np.full(n, -0.003)), index=idx)     # steady downtrend
    eu = continuous_exposure(up, _spec())
    ed = continuous_exposure(down, _spec())
    assert ((eu >= 0) & (eu <= 1)).all() and ((ed >= 0) & (ed <= 1)).all()
    assert eu.iloc[-1] > 0.5 and ed.iloc[-1] < 0.5            # long uptrend high, downtrend low
    assert eu.iloc[-1] > ed.iloc[-1]


def test_continuous_exposure_no_lookahead():
    close = _wavy_close()
    exp = continuous_exposure(close, _spec())
    ext = pd.concat([close, pd.Series(
        float(close.iloc[-1]) * np.cumprod(1 + np.full(15, -0.004)),
        index=pd.date_range(close.index[-1] + pd.offsets.BDay(1), periods=15, freq="B"))])
    exp2 = continuous_exposure(ext, _spec())
    assert np.allclose(exp.to_numpy(), exp2.reindex(exp.index).to_numpy())


def test_continuous_trend_net_runs_and_is_finite():
    close = _wavy_close()
    net = continuous_trend_net(close, _spec())
    assert len(net) > 0 and np.isfinite(net.to_numpy()).all()


# --------------------------------------------------------------------------- #
# 4) sizing off the real drawdown
# --------------------------------------------------------------------------- #
def test_size_from_maxdd():
    assert abs(size_from_maxdd(-0.497, -0.15) - 0.15 / 0.497) < 1e-9   # ~0.30x
    assert size_from_maxdd(-0.10, -0.15) == 1.0                        # capped at 1x (no leverage)
    assert size_from_maxdd(0.0, -0.15) == 1.0


# --------------------------------------------------------------------------- #
# 5) deploy spec file is present with the three pre-commitments
# --------------------------------------------------------------------------- #
def test_deploy_spec_present_with_precommitments():
    spec = pathlib.Path(__file__).resolve().parents[1] / "deploy_spec.md"
    assert spec.exists()
    text = spec.read_text(encoding="utf-8")
    assert "49.7%" in text                                    # sizes off the REAL drawdown
    assert "No discretionary override" in text
    assert "decade table" in text.lower() and ("3–5" in text or "3-5" in text)
    assert "do not deploy" in text.lower()
