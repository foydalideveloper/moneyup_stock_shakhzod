"""Crypto ML agent — synthetic klines, no network."""

import numpy as np
import pandas as pd

from tagent.crypto_ml import (
    CryptoMLPredictor,
    MICRO_FEATURES,
    build_crypto_dataset,
    crypto_ml_decision,
    crypto_model_path,
    fetch_klines,
    merge_micro_features,
    save_crypto_model,
    train_crypto_model,
)


def _klines(n=500, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    ret = rng.normal(0.0002, 0.012, n)
    close = 100 * np.cumprod(1 + ret)
    high = close * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.004, n)))
    open_ = np.r_[close[0], close[:-1]]
    vol = rng.uniform(100, 300, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": vol}, index=idx)


def _micro(idx, vals=0.3):
    return pd.DataFrame({"ts": [t.isoformat() for t in idx],
                         "depth_imbalance": vals, "absorption_ratio": 0.5,
                         "spoof_ratio": 0.1})


# --------------------------------------------------------------------------- #
# dataset + features
# --------------------------------------------------------------------------- #
def test_build_dataset_has_tech_features_and_two_classes():
    X, y = build_crypto_dataset(_klines(), tp_pct=0.02, sl_pct=0.02, horizon=8)
    assert {"rsi_14", "macd_hist", "mom_5", "vol_ratio"}.issubset(X.columns)
    assert len(X) == len(y) and y.nunique() == 2          # both labels present
    assert X.index.is_monotonic_increasing                # time-sorted, no shuffle


def test_micro_features_merged_when_available():
    k = _klines(120)
    X, _ = build_crypto_dataset(k, micro_df=_micro(k.index), tp_pct=0.02, sl_pct=0.02, horizon=6)
    assert all(c in X.columns for c in MICRO_FEATURES)     # micro columns present


def test_merge_micro_is_asof_no_lookahead():
    feats = pd.DataFrame({"x": [1.0, 2.0, 3.0]},
                         index=pd.to_datetime(["2024-01-01T01:00", "2024-01-01T02:00",
                                               "2024-01-01T03:00"], utc=True))
    micro = pd.DataFrame({
        "ts": ["2024-01-01T00:30", "2024-01-01T02:30"],     # samples before bars 1 and 3
        "depth_imbalance": [0.2, 0.9], "absorption_ratio": [0.5, 0.5], "spoof_ratio": [0.0, 0.0]})
    out = merge_micro_features(feats, micro)
    # bar @01:00 sees the 00:30 sample; @02:00 still 00:30; @03:00 sees 02:30 (past only)
    assert list(out["depth_imbalance"]) == [0.2, 0.2, 0.9]


# --------------------------------------------------------------------------- #
# train / save / load / predict
# --------------------------------------------------------------------------- #
def test_train_save_load_predict(tmp_path):
    k = _klines(500)
    model, feats, metrics = train_crypto_model(k, tp_pct=0.02, sl_pct=0.02,
                                               horizon=8, n_splits=3)
    assert feats and "cv_accuracy_mean" in metrics and metrics["n_samples"] > 0
    path = save_crypto_model(model, feats, metrics, "BTCUSDT", model_dir=tmp_path)
    assert crypto_model_path("BTCUSDT", tmp_path).exists()

    pred = CryptoMLPredictor.load("BTCUSDT", model_dir=tmp_path)
    p = pred.predict_proba_latest(_klines(120, seed=9))
    assert 0.0 <= p <= 1.0 and pred.features == feats


def test_predictor_uses_micro_when_model_trained_with_it(tmp_path):
    k = _klines(500)
    model, feats, metrics = train_crypto_model(k, micro_df=_micro(k.index),
                                               tp_pct=0.02, sl_pct=0.02, horizon=8, n_splits=3)
    assert any(c in feats for c in MICRO_FEATURES)
    save_crypto_model(model, feats, metrics, "ETHUSDT", model_dir=tmp_path)
    pred = CryptoMLPredictor.load("ETHUSDT", model_dir=tmp_path)
    # supplying a live micro row works (and is required-shaped, not crashing)
    p = pred.predict_proba_latest(_klines(120, seed=3),
                                  micro_row={"depth_imbalance": 0.4, "absorption_ratio": 0.6,
                                             "spoof_ratio": 0.0})
    assert 0.0 <= p <= 1.0


# --------------------------------------------------------------------------- #
# decision mapping
# --------------------------------------------------------------------------- #
def test_crypto_ml_decision_thresholds():
    assert crypto_ml_decision(0.80)["action"] == "BUY"
    assert crypto_ml_decision(0.20)["action"] == "SELL"
    assert crypto_ml_decision(0.50)["action"] == "HOLD"
    assert crypto_ml_decision(0.80)["probability"] == 0.8


# --------------------------------------------------------------------------- #
# kline fetch (mocked HTTP — no network)
# --------------------------------------------------------------------------- #
def test_fetch_klines_parses_and_pages(monkeypatch_free=None):
    rows = [[i * 3_600_000, "100", "101", "99", f"{100 + i*0.1:.2f}", "5.0",
             0, "0", 0, "0", "0", "0"] for i in range(5)]

    class Resp:
        def __init__(self, d): self._d = d
        def json(self): return self._d

    class Sess:
        def __init__(self): self.calls = 0
        def get(self, url, params=None, timeout=None):
            self.calls += 1
            return Resp(rows if self.calls == 1 else [])    # one page then done

    df = fetch_klines("BTCUSDT", interval="1h", session=Sess(), start_ms=0)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 5 and df["close"].iloc[-1] == 100.4
