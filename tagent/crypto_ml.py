"""Crypto ML agent — LightGBM on a coin's historical Binance klines.

Mirrors the US ML agent so the two markets are consistent: technical features
from :mod:`tagent.features` (returns / RSI / MACD / momentum / volatility /
volume), PLUS order-book microstructure features (depth imbalance / absorption /
spoof) where a recording overlaps the bars, and triple-barrier labels
(:mod:`tagent.labels`) on the kline horizon. Training uses the same
purged/embargoed walk-forward as US (:mod:`tagent.ml.train`).

Per-coin models are saved to ``models/crypto_<SYM>.pkl``; :class:`CryptoMLPredictor`
loads one and turns a recent kline window (plus optional live microstructure) into
P(good long entry), mapped to BUY/HOLD/SELL. Pure pandas/numpy + LightGBM;
unit-tested on synthetic klines (no network).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from tagent.config import MODEL_DIR
from tagent.features import make_features
from tagent.labels import triple_barrier_labels

MICRO_FEATURES = ["depth_imbalance", "absorption_ratio", "spoof_ratio"]
_BINANCE = "https://api.binance.com"
_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000,
       "4h": 14_400_000, "1d": 86_400_000}


def fetch_klines(symbol: str, interval: str = "1h", years: float = 2.0,
                 session=None, start_ms: Optional[int] = None, now_ms: Optional[int] = None,
                 max_pages: int = 60) -> pd.DataFrame:
    """Public Binance spot klines (no key), paged FORWARD to cover ``years``."""
    import time
    import requests
    s = session or requests
    step = _MS.get(interval, _MS["1h"])
    now = int(now_ms if now_ms is not None else time.time() * 1000)
    cur = int(start_ms) if start_ms is not None else now - int(years * 365 * 86_400_000)
    rows: list = []
    for _ in range(max_pages):
        page = s.get(f"{_BINANCE}/api/v3/klines",
                     params={"symbol": symbol, "interval": interval,
                             "startTime": cur, "limit": 1000}, timeout=10).json()
        if not isinstance(page, list) or not page:
            break
        rows += page
        if len(page) < 1000:
            break
        cur = int(page[-1][0]) + step
    if not rows:
        raise ValueError(f"Binance klines error for {symbol}: empty")
    df = pd.DataFrame({
        "open": [float(k[1]) for k in rows], "high": [float(k[2]) for k in rows],
        "low": [float(k[3]) for k in rows], "close": [float(k[4]) for k in rows],
        "volume": [float(k[5]) for k in rows],
    }, index=pd.to_datetime([k[0] for k in rows], unit="ms", utc=True))
    return df[~df.index.duplicated()].sort_index()


def merge_micro_features(feats: pd.DataFrame, micro_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """As-of merge order-book microstructure features onto kline features.

    For each bar, take the most recent microstructure sample at/just before the
    bar time (no lookahead). Missing where no recording overlaps -> left NaN.
    """
    if micro_df is None or micro_df.empty:
        return feats
    m = micro_df.copy()
    if "ts" in m.columns:
        m = m.set_index(pd.to_datetime(m["ts"], utc=True))
    m = m[[c for c in MICRO_FEATURES if c in m.columns]].apply(pd.to_numeric, errors="coerce")
    m = m[~m.index.duplicated(keep="last")].sort_index()
    if m.empty:
        return feats
    joined = pd.merge_asof(feats.sort_index(), m, left_index=True, right_index=True,
                           direction="backward")
    return joined.reindex(feats.index)


def build_crypto_dataset(klines: pd.DataFrame, micro_df: Optional[pd.DataFrame] = None,
                         tp_pct: float = 0.04, sl_pct: float = 0.02, horizon: int = 10):
    """(X, y) for one coin: kline tech features (+ micro where available) and
    triple-barrier labels on the kline horizon. Time-sorted, NaNs dropped."""
    feats = make_features(klines, dropna=False)
    tech_cols = list(feats.columns)
    feats = merge_micro_features(feats, micro_df)         # micro cols may be partly NaN
    labels = triple_barrier_labels(klines, tp_pct=tp_pct, sl_pct=sl_pct, horizon=horizon)
    data = feats.copy()
    data["label"] = labels
    # Require the tech features + label; leave microstructure NaN where no recording
    # overlaps (LightGBM handles NaN) so a short micro capture doesn't gut the dataset.
    data = data.dropna(subset=tech_cols + ["label"])
    if data.empty:
        raise ValueError("No usable crypto training rows. Need more klines.")
    X = data.drop(columns=["label"])
    y = data["label"].astype(int)
    order = np.argsort(X.index.values, kind="stable")
    return X.iloc[order], y.iloc[order]


def train_crypto_model(klines: pd.DataFrame, micro_df: Optional[pd.DataFrame] = None,
                       tp_pct: float = 0.04, sl_pct: float = 0.02, horizon: int = 10,
                       n_splits: int = 4, params: Optional[dict] = None):
    """Train a per-coin LightGBM with purged walk-forward CV. Returns
    (model, feature_names, metrics)."""
    from tagent.ml.train import walk_forward_train
    X, y = build_crypto_dataset(klines, micro_df, tp_pct, sl_pct, horizon)
    model, metrics = walk_forward_train(X, y, n_splits=n_splits, params=params,
                                        horizon=horizon)
    metrics["features"] = list(X.columns)
    return model, list(X.columns), metrics


def crypto_model_path(symbol: str, model_dir: Optional[Path] = None) -> Path:
    base = Path(model_dir) if model_dir is not None else Path(MODEL_DIR)
    return base / f"crypto_{symbol.upper()}.pkl"


def save_crypto_model(model, feature_names, metrics: dict, symbol: str,
                      model_dir: Optional[Path] = None) -> str:
    path = crypto_model_path(symbol, model_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump({"model": model, "features": list(feature_names),
                     "metrics": metrics, "symbol": symbol.upper()}, fh)
    return str(path)


def crypto_ml_decision(proba: float, buy_threshold: float = 0.55,
                       sell_threshold: float = 0.45) -> dict:
    """Map P(good long entry) -> BUY / HOLD / SELL (same shape as the other agents)."""
    if proba >= buy_threshold:
        action = "BUY"
    elif proba <= sell_threshold:
        action = "SELL"
    else:
        action = "HOLD"
    return {"action": action, "probability": round(float(proba), 4),
            "confidence": round(float(proba), 4)}


class CryptoMLPredictor:
    """Loads a per-coin crypto model and predicts from recent klines (+ live micro)."""

    def __init__(self, bundle: dict):
        self.model = bundle["model"]
        self.features = list(bundle["features"])
        self.metrics = bundle.get("metrics", {})
        self.symbol = bundle.get("symbol", "")

    @classmethod
    def load(cls, symbol: str, model_dir: Optional[Path] = None) -> "CryptoMLPredictor":
        with open(crypto_model_path(symbol, model_dir), "rb") as fh:
            return cls(pickle.load(fh))

    def predict_proba_latest(self, klines: pd.DataFrame,
                             micro_row: Optional[Dict[str, float]] = None) -> float:
        feats = make_features(klines, dropna=True)
        if feats.empty:
            raise ValueError("Not enough klines to compute features.")
        row = feats.iloc[[-1]].copy()
        for c in MICRO_FEATURES:                          # live order-book features
            if c in self.features:
                row[c] = float((micro_row or {}).get(c, np.nan))
        row = row.reindex(columns=self.features)
        return float(self.model.predict_proba(row)[:, 1][0])
