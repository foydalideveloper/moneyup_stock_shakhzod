"""Out-of-sample prediction guarantees: no bar is predicted by a model that was
trained on that same bar (no leakage), and predictions are time-ordered."""

import numpy as np
import pandas as pd

from tagent.ml.train import walk_forward_predict


def _synthetic(n=300, seed=0):
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"), index=idx)
    y = pd.Series(rng.integers(0, 2, size=n), index=idx)
    return X, y


def test_no_bar_predicted_by_model_trained_on_it():
    X, y = _synthetic()
    oos, folds = walk_forward_predict(X, y, n_splits=5, gap=5, return_folds=True)

    predicted = set(np.flatnonzero(oos.notna().to_numpy()).tolist())
    assert predicted, "expected some out-of-sample predictions"

    all_test = set()
    for tr, te in folds:
        tr_set, te_set = set(tr.tolist()), set(te.tolist())
        # The model for this fold trained on `tr` and predicted `te`.
        assert tr_set.isdisjoint(te_set)        # no bar both trained on and predicted
        assert max(tr) < min(te)                # train strictly precedes its test bars
        all_test |= te_set

    # Every out-of-sample probability lands exactly on a held-out test bar.
    assert predicted == all_test


def test_pre_first_fold_bars_have_no_oos_prediction():
    X, y = _synthetic()
    oos, folds = walk_forward_predict(X, y, n_splits=5, gap=5, return_folds=True)
    first_test_start = min(int(te.min()) for _, te in folds)
    # Bars before any test fold are only ever training data -> must stay NaN.
    assert oos.iloc[:first_test_start].isna().all()
    assert oos.iloc[first_test_start:].notna().all()
