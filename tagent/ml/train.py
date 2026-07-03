"""Build the training dataset and train a LightGBM model with walk-forward
validation (the honest, time-aware way to evaluate financial models).
"""

from __future__ import annotations

import pickle
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from tagent.config import MODEL_DIR
from tagent.features import make_features
from tagent.features_short import merge_short_features
from tagent.labels import triple_barrier_labels

# Tightened (regularized) LightGBM defaults: shallow trees, large min-child, L1/L2
# — overfits less on noisy financial features than the old num_leaves=31 config.
DEFAULT_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.03,
    num_leaves=15,
    max_depth=4,
    min_child_samples=60,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)


def purged_walk_forward_splits(n_samples: int, n_splits: int = 5, gap: int = 5,
                               horizon: Optional[int] = None, embargo: int = 0):
    """PURGED + EMBARGOED expanding-window splits (replaces plain TimeSeriesSplit).

    Test folds tile the tail of the series in ``n_splits`` contiguous blocks (same
    as ``TimeSeriesSplit``). Training is everything strictly before each test
    block MINUS a purge gap, so no training sample's label window (``horizon`` bars
    ahead) overlaps the test fold, plus an ``embargo``. The purge is
    ``horizon + embargo`` when ``horizon`` is given, else ``gap`` (so the existing
    ``gap``-based call is reproduced exactly). Yields ``(train_idx, test_idx)``.
    """
    purge = (horizon + embargo) if horizon is not None else gap
    fold = n_samples // (n_splits + 1)
    if fold == 0:
        return
    for k in range(1, n_splits + 1):
        te_start = k * fold
        te_end = (k + 1) * fold if k < n_splits else n_samples
        train_end = te_start - purge
        if train_end <= 0 or te_start >= n_samples:
            continue
        yield np.arange(0, train_end), np.arange(te_start, te_end)


def _fit_predict_proba(estimator, X_tr, y_tr, X_te) -> np.ndarray:
    estimator.fit(X_tr, y_tr)
    return estimator.predict_proba(X_te)[:, 1]


def _default_estimator(params: Optional[dict]):
    import lightgbm as lgb
    return lgb.LGBMClassifier(**(params or DEFAULT_PARAMS))


def select_features(X: pd.DataFrame, y: pd.Series, min_frac: float = 0.005,
                    importance_type: str = "gain") -> Tuple[List[str], pd.Series]:
    """Fit a regularized LightGBM and keep features that earn their place.

    Returns (selected_columns, importance Series sorted desc). Features whose
    normalized importance is below ``min_frac`` of the total are dropped (zero /
    near-zero importance). Always keeps at least the top 5.
    """
    import lightgbm as lgb
    model = lgb.LGBMClassifier(importance_type=importance_type, **DEFAULT_PARAMS)
    model.fit(X, y)
    imp = pd.Series(model.feature_importances_, index=X.columns, dtype=float)
    imp = imp.sort_values(ascending=False)
    total = imp.sum()
    frac = imp / total if total > 0 else imp
    selected = list(frac[frac >= min_frac].index)
    if not selected:
        selected = list(imp.index[:5])
    return selected, imp


def build_dataset(history: Dict[str, pd.DataFrame], tp_pct: float = 0.04,
                  sl_pct: float = 0.02, horizon: int = 10,
                  short_data: Optional[Dict[str, pd.DataFrame]] = None
                  ) -> Tuple[pd.DataFrame, pd.Series]:
    """Combine features + triple-barrier labels across all symbols.

    Returns (X, y) sorted by time. Pooling several symbols is a deliberate
    baseline simplification; rows are time-sorted so walk-forward stays roughly
    chronological.

    Pass ``short_data`` = {symbol: canonical short-selling DataFrame} to merge
    leak-free Korean short-selling features (see :mod:`tagent.features_short`).
    Omitting it (or passing an empty dict) leaves the pipeline exactly as before.
    """
    X_parts, y_parts = [], []
    for sym, df in history.items():
        feats = make_features(df, dropna=False)
        if short_data and sym in short_data:
            feats = merge_short_features(feats, short_data[sym])
        labels = triple_barrier_labels(df, tp_pct=tp_pct, sl_pct=sl_pct, horizon=horizon)
        data = feats.copy()
        data["label"] = labels
        data = data.dropna()
        if data.empty:
            continue
        X_parts.append(data.drop(columns=["label"]))
        y_parts.append(data["label"].astype(int))

    if not X_parts:
        raise ValueError("No usable training rows. Need more history.")

    X = pd.concat(X_parts)
    y = pd.concat(y_parts)
    order = np.argsort(X.index.values, kind="stable")
    return X.iloc[order], y.iloc[order]


def walk_forward_train(X: pd.DataFrame, y: pd.Series, n_splits: int = 5,
                       gap: int = 5, params: dict | None = None,
                       horizon: Optional[int] = None, embargo: int = 0,
                       make_model: Optional[Callable[[], object]] = None):
    """Train + evaluate with PURGED + EMBARGOED walk-forward CV.

    ``make_model`` is a no-arg factory returning a fresh classifier (default: the
    regularized LightGBM). Returns (final_model, metrics)."""
    try:
        from sklearn.metrics import accuracy_score, roc_auc_score
    except ImportError as e:  # pragma: no cover
        raise ImportError("Install ML deps: pip install -r requirements.txt") from e
    factory = make_model or (lambda: _default_estimator(params))

    accs, aucs = [], []
    for tr, te in purged_walk_forward_splits(len(X), n_splits, gap, horizon, embargo):
        model = factory()
        model.fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[te])
        accs.append(accuracy_score(y.iloc[te], pred))
        if len(np.unique(y.iloc[te])) == 2:
            proba = model.predict_proba(X.iloc[te])[:, 1]
            aucs.append(roc_auc_score(y.iloc[te], proba))

    final_model = factory().fit(X, y)
    metrics = {
        "cv_accuracy_mean": float(np.mean(accs)) if accs else float("nan"),
        "cv_accuracy_std": float(np.std(accs)) if accs else float("nan"),
        "cv_auc_mean": float(np.mean(aucs)) if aucs else float("nan"),
        "n_splits": n_splits,
        "n_samples": int(len(X)),
        "positive_rate": float(y.mean()),
    }
    return final_model, metrics


def walk_forward_predict(X: pd.DataFrame, y: pd.Series, n_splits: int = 5,
                         gap: int = 5, params: dict | None = None,
                         return_folds: bool = False, horizon: Optional[int] = None,
                         embargo: int = 0,
                         make_model: Optional[Callable[[], object]] = None):
    """Produce TRUE out-of-sample probabilities (no leakage), PURGED + EMBARGOED.

    For each fold a fresh model (``make_model`` factory, default regularized
    LightGBM) trains on the purged train slice and predicts ONLY its test slice,
    so every probability comes from a model that never saw that bar AND whose
    training labels never overlapped the test window (purge = horizon + embargo).

    Returns a ``pd.Series`` aligned to ``X.index``; pre-first-fold bars stay NaN.
    With ``return_folds=True`` also returns the ``(train_idx, test_idx)`` per fold.
    """
    factory = make_model or (lambda: _default_estimator(params))
    oos = pd.Series(np.nan, index=X.index, name="oos_proba")
    folds = []
    for tr, te in purged_walk_forward_splits(len(X), n_splits, gap, horizon, embargo):
        model = factory()
        model.fit(X.iloc[tr], y.iloc[tr])
        oos.iloc[te] = model.predict_proba(X.iloc[te])[:, 1]
        folds.append((tr, te))

    if return_folds:
        return oos, folds
    return oos


def save_model(model, feature_names, metrics: dict, path=None) -> str:
    path = str(path or (MODEL_DIR / "model.pkl"))
    bundle = {"model": model, "features": list(feature_names), "metrics": metrics}
    with open(path, "wb") as fh:
        pickle.dump(bundle, fh)
    return path
