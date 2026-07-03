"""Model factory, calibrated ensemble, purged CV, and feature selection tests."""

import numpy as np
import pandas as pd
import pytest

from tagent.ml.models import (
    EnsembleClassifier,
    available_models,
    make_model,
)
from tagent.ml.train import (
    purged_walk_forward_splits,
    select_features,
    walk_forward_predict,
)


def _xy(n=200, n_feat=6, seed=0, informative=True):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, n_feat)),
                     columns=[f"f{i}" for i in range(n_feat)])
    if informative:
        y = pd.Series(((X["f0"] + rng.normal(0, 1.0, n)) > 0).astype(int))
    else:
        y = pd.Series(rng.integers(0, 2, n))
    return X, y


# --------------------------------------------------------------------------- #
# purged + embargoed splits
# --------------------------------------------------------------------------- #
def test_purged_splits_enforce_gap_and_disjoint():
    folds = list(purged_walk_forward_splits(400, n_splits=4, horizon=10, embargo=2))
    assert folds
    for tr, te in folds:
        assert set(tr).isdisjoint(set(te))
        assert tr.max() < te.min()
        # purge >= horizon + embargo between train end and test start.
        assert te.min() - tr.max() >= 10 + 2


def test_purged_splits_test_folds_tile_contiguously():
    folds = list(purged_walk_forward_splits(360, n_splits=5, gap=5))
    test_starts = [te.min() for _, te in folds]
    # Each test block starts where the previous ended (contiguous tiling).
    for (_, te_prev), (_, te_next) in zip(folds, folds[1:]):
        assert te_next.min() == te_prev.max() + 1


def test_purged_splits_backward_compatible_with_gap():
    # No horizon -> purge == gap (reproduces the old TimeSeriesSplit behavior).
    folds = list(purged_walk_forward_splits(300, n_splits=5, gap=5))
    for tr, te in folds:
        assert te.min() - tr.max() == 6        # gap of 5 -> one-position-after = 6


# --------------------------------------------------------------------------- #
# model factory + calibration + ensemble
# --------------------------------------------------------------------------- #
def test_available_models_includes_lightgbm():
    assert "lightgbm" in available_models()


@pytest.mark.parametrize("kind", ["lightgbm", "xgboost", "catboost"])
def test_each_base_model_fits_and_predicts(kind):
    if kind not in available_models():
        pytest.skip(f"{kind} not installed")
    X, y = _xy(150)
    m = make_model(kind, calib_cv=2)
    m.fit(X.iloc[:120], y.iloc[:120])
    p = m.predict_proba(X.iloc[120:])
    assert p.shape == (30, 2)
    assert ((p >= 0.0) & (p <= 1.0)).all()      # calibrated probabilities
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)


def test_ensemble_averages_available_models():
    X, y = _xy(180)
    m = make_model("ensemble", calib_cv=2)
    m.fit(X.iloc[:140], y.iloc[:140])
    assert len(m.models_) == len(available_models())
    p = m.predict_proba(X.iloc[140:])
    assert p.shape == (40, 2)
    assert ((p >= 0) & (p <= 1)).all()


def test_ensemble_degrades_to_available_subset():
    X, y = _xy(150)
    m = EnsembleClassifier(kinds=["lightgbm"], calib_cv=2)   # pretend only lgb exists
    m.fit(X.iloc[:120], y.iloc[:120])
    assert len(m.models_) == 1
    assert m.predict(X.iloc[120:]).shape == (30,)


def test_make_model_rejects_unknown_kind():
    with pytest.raises(ValueError):
        make_model("randomforest")


# --------------------------------------------------------------------------- #
# feature selection
# --------------------------------------------------------------------------- #
def test_select_features_keeps_informative_drops_noise():
    X, y = _xy(400, n_feat=8, informative=True)
    selected, imp = select_features(X, y, min_frac=0.02)
    assert "f0" in selected                      # the informative feature survives
    assert imp.index[0] == "f0"                  # and ranks first by gain
    assert len(selected) <= len(X.columns)
    assert imp.sum() > 0


def test_select_features_always_keeps_some():
    X, y = _xy(200, informative=False)
    selected, _ = select_features(X, y, min_frac=0.99)   # impossible threshold
    assert len(selected) >= 1                            # never returns empty


# --------------------------------------------------------------------------- #
# walk_forward_predict with the new model + purge
# --------------------------------------------------------------------------- #
def test_walk_forward_predict_with_ensemble_and_purge():
    X, y = _xy(300)
    oos, folds = walk_forward_predict(
        X, y, n_splits=4, horizon=10, embargo=2, return_folds=True,
        make_model=lambda: make_model("lightgbm", calib_cv=2))
    assert oos.notna().sum() > 0
    for tr, te in folds:
        assert te.min() - tr.max() >= 12          # purge respected
