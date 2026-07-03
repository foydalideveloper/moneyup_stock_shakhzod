"""Microstructure study maths — synthetic data, no network."""

import numpy as np
import pandas as pd

from tagent.microstructure import (
    forward_returns,
    information_coefficient,
    oos_eval,
    study,
    time_split,
    to_epoch_seconds,
    verdict,
)


# --------------------------------------------------------------------------- #
# forward returns
# --------------------------------------------------------------------------- #
def test_forward_returns_constant_drift():
    # 1 row/sec, mid grows 0.1% each second -> 5s forward return ~ +0.5%.
    ts = np.arange(0, 60, 1.0)
    mid = 100.0 * (1.001 ** np.arange(60))
    fwd = forward_returns(ts, mid, horizon_s=5)
    # first 55 rows have a +5s point; last 5 are NaN (no future row).
    assert np.isnan(fwd[-1])
    valid = fwd[:55]
    assert np.all(valid > 0)
    assert abs(np.nanmean(valid) - (1.001 ** 5 - 1)) < 1e-6


def test_forward_returns_nan_on_gap():
    # A 100s gap means the +5s target has no nearby row -> NaN.
    ts = np.array([0.0, 1.0, 2.0, 102.0, 103.0])
    mid = np.array([100, 100, 100, 100, 100], float)
    fwd = forward_returns(ts, mid, horizon_s=5, tol_s=3)
    assert np.isnan(fwd[2])          # row at t=2 -> target 7, next row is 102 (gap)


def test_forward_returns_sign():
    ts = np.arange(0, 10, 1.0)
    mid = np.array([100, 100, 100, 100, 99, 99, 99, 99, 99, 99], float)
    fwd = forward_returns(ts, mid, horizon_s=4)
    assert fwd[0] < 0                # 100 -> 99 over 4s


# --------------------------------------------------------------------------- #
# information coefficient (Spearman)
# --------------------------------------------------------------------------- #
def test_ic_perfect_monotonic():
    f = np.arange(100, dtype=float)
    r = f * 3.0 + 1.0                 # strictly increasing -> Spearman = +1
    assert abs(information_coefficient(f, r) - 1.0) < 1e-9


def test_ic_perfect_inverse():
    f = np.arange(100, dtype=float)
    r = -f
    assert abs(information_coefficient(f, r) + 1.0) < 1e-9


def test_ic_random_near_zero():
    rng = np.random.default_rng(0)
    f = rng.normal(size=2000)
    r = rng.normal(size=2000)
    assert abs(information_coefficient(f, r)) < 0.1


def test_ic_nan_when_too_few():
    assert np.isnan(information_coefficient([1, 2, 3], [1, 2, 3]))


# --------------------------------------------------------------------------- #
# no-lookahead chronological split
# --------------------------------------------------------------------------- #
def test_time_split_is_chronological_and_disjoint():
    tr, te = time_split(100, train_frac=0.6)
    assert list(tr) == list(range(60)) and list(te) == list(range(60, 100))
    assert set(tr).isdisjoint(set(te))
    assert tr.max() < te.min()       # train strictly precedes test (no shuffle)


def test_oos_eval_detects_real_signal_out_of_sample():
    # Feature linearly drives the forward return -> OOS IC high, hit rate > 0.5.
    rng = np.random.default_rng(1)
    f = rng.normal(size=400)
    r = f * 0.001 + rng.normal(0, 0.0002, 400)
    res = oos_eval(f, r, train_frac=0.6)
    assert res["oos_ic"] > 0.5 and res["oos_hit"] > 0.7
    assert res["n_test"] == 160


def test_oos_eval_no_signal_is_chance():
    rng = np.random.default_rng(2)
    f = rng.normal(size=600)
    r = rng.normal(size=600)         # independent -> OOS IC ~0, hit ~0.5
    res = oos_eval(f, r)
    assert abs(res["oos_ic"]) < 0.15
    assert 0.4 < res["oos_hit"] < 0.6


# --------------------------------------------------------------------------- #
# end-to-end study + verdict on a synthetic frame
# --------------------------------------------------------------------------- #
def _frame(n=300, signal=True, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-06-04T00:00:00Z", periods=n, freq="s")
    imb = rng.normal(0, 0.3, n)
    drift = rng.normal(0, 0.0001, n)
    if signal:                       # imbalance at t drives the NEXT move (t -> t+1)
        drift[1:] += imb[:-1] * 0.0008
    mid = 100.0 * np.cumprod(1 + drift)
    return pd.DataFrame({
        "ts": ts.astype(str), "mid": mid,
        "bid_size": rng.uniform(1, 5, n), "ask_size": rng.uniform(1, 5, n),
        "depth_imbalance": imb, "absorption_ratio": rng.uniform(0, 1, n),
        "spoof_ratio": rng.uniform(0, 1, n), "filled_qty": rng.uniform(0, 2, n),
        "cancelled_qty": rng.uniform(0, 2, n),
    })


def test_study_returns_rows_for_each_feature_horizon():
    rows = study(_frame(300), horizons=[5, 30])
    feats = {r["feature"] for r in rows}
    assert {"depth_imbalance", "size_imbalance", "net_flow"}.issubset(feats)
    assert {r["horizon_s"] for r in rows} == {5, 30}
    # derived size_imbalance present
    assert any(r["feature"] == "size_imbalance" for r in rows)


def test_verdict_flags_edge_when_feature_drives_returns():
    rows = study(_frame(400, signal=True, seed=3), horizons=[5])
    v = verdict(rows, cost_bps=0.0, ic_thresh=0.05, hit_thresh=0.52)
    assert v["has_edge"]
    assert any(e["feature"] == "depth_imbalance" for e in v["edges"])


def test_verdict_no_edge_on_pure_noise():
    rows = study(_frame(400, signal=False, seed=4), horizons=[5, 30, 60])
    v = verdict(rows, cost_bps=10.0)        # realistic cost hurdle
    assert not v["has_edge"]


def test_verdict_statistical_but_not_tradable():
    # A predictive feature but a tiny move -> clears IC+hit, fails the cost hurdle.
    rows = study(_frame(400, signal=True, seed=3), horizons=[5])
    v = verdict(rows, cost_bps=10_000.0, ic_thresh=0.05, hit_thresh=0.52)  # impossible cost
    assert not v["has_edge"]
    assert any(e["feature"] == "depth_imbalance" for e in v["statistical_only"])


def test_to_epoch_seconds_monotonic():
    ts = pd.date_range("2026-01-01", periods=5, freq="s").astype(str)
    s = to_epoch_seconds(ts)
    assert np.all(np.diff(s) == 1.0)
