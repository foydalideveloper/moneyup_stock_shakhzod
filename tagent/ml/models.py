"""Model factory: regularized LightGBM / XGBoost / CatBoost, calibrated, + ensemble.

Three boosters with tightened regularization (shallow trees, large min-child,
L1/L2) so they overfit less on noisy financial features; each wrapped in
probability calibration (isotonic / Platt) so a threshold like 0.55 is
*meaningful*; and an ensemble that averages the calibrated probabilities. Missing
libraries degrade gracefully (the ensemble uses whatever is installed).

``make_model(kind=...)`` returns a fresh sklearn-compatible classifier
(``fit`` / ``predict_proba`` / ``predict``).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

# --- tightened (regularized) hyper-parameters --- #
LIGHTGBM_PARAMS = dict(
    n_estimators=250, learning_rate=0.03, num_leaves=15, max_depth=4,
    min_child_samples=60, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    reg_alpha=0.5, reg_lambda=1.0, random_state=42, n_jobs=-1, verbose=-1)

XGBOOST_PARAMS = dict(
    n_estimators=250, learning_rate=0.03, max_depth=4, min_child_weight=5.0,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=1.0,
    gamma=0.0, tree_method="hist", eval_metric="logloss", random_state=42, n_jobs=-1)

CATBOOST_PARAMS = dict(
    iterations=200, learning_rate=0.03, depth=4, l2_leaf_reg=6.0, subsample=0.8,
    random_seed=42, verbose=False, allow_writing_files=False, thread_count=-1)


def _lightgbm(params: Optional[dict] = None):
    import lightgbm as lgb
    return lgb.LGBMClassifier(**{**LIGHTGBM_PARAMS, **(params or {})})


def _xgboost(params: Optional[dict] = None):
    import xgboost as xgb
    return xgb.XGBClassifier(**{**XGBOOST_PARAMS, **(params or {})})


def _catboost(params: Optional[dict] = None):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(**{**CATBOOST_PARAMS, **(params or {})})


_BASE = {"lightgbm": _lightgbm, "xgboost": _xgboost, "catboost": _catboost}


def available_models() -> List[str]:
    """Which base boosters are importable in this environment (in priority order)."""
    out = []
    for k, f in _BASE.items():
        try:
            f()
            out.append(k)
        except Exception:
            pass
    return out


def _calibrated(estimator, method: Optional[str], cv: int):
    if not method:
        return estimator
    from sklearn.calibration import CalibratedClassifierCV
    return CalibratedClassifierCV(estimator, method=method, cv=cv)


class EnsembleClassifier:
    """Average of calibrated base models' P(class=1). sklearn-compatible.

    Each base (lightgbm/xgboost/catboost) is wrapped in CalibratedClassifierCV
    (isotonic/Platt) and the calibrated probabilities are averaged. If calibration
    can't fit (too few samples / one class in an inner fold) the base falls back
    to uncalibrated; a base that fails entirely is dropped.
    """

    def __init__(self, kinds: Optional[List[str]] = None, calibrate: str = "isotonic",
                 calib_cv: int = 3, params: Optional[Dict[str, dict]] = None):
        self.kinds = kinds if kinds is not None else available_models()
        self.calibrate = calibrate
        self.calib_cv = calib_cv
        self.params = params or {}
        self.models_: list = []

    def fit(self, X, y):
        self.classes_ = np.unique(np.asarray(y))
        self.models_ = []
        for k in self.kinds:
            base = _BASE.get(k)
            if base is None:
                continue
            try:
                m = _calibrated(base(self.params.get(k)), self.calibrate, self.calib_cv)
                m.fit(X, y)
            except Exception:
                try:                              # fall back to uncalibrated base
                    m = base(self.params.get(k))
                    m.fit(X, y)
                except Exception:
                    continue
            self.models_.append(m)
        if not self.models_:                      # last-resort plain LightGBM
            m = _lightgbm()
            m.fit(X, y)
            self.models_ = [m]
        return self

    def predict_proba(self, X):
        probs = [m.predict_proba(X) for m in self.models_]
        return np.mean(probs, axis=0)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def make_model(kind: str = "ensemble", calibrate: str = "isotonic", calib_cv: int = 3,
               params: Optional[Dict[str, dict]] = None):
    """Fresh classifier. kind: 'lightgbm'|'xgboost'|'catboost'|'ensemble'.

    'ensemble' averages the calibrated probabilities of all *available* boosters.
    """
    if kind == "ensemble":
        return EnsembleClassifier(available_models(), calibrate=calibrate,
                                  calib_cv=calib_cv, params=params)
    if kind not in _BASE:
        raise ValueError(f"unknown model kind: {kind!r} "
                         f"(expected one of {list(_BASE)} or 'ensemble')")
    p = (params or {}).get(kind)
    return _calibrated(_BASE[kind](p), calibrate, calib_cv)
