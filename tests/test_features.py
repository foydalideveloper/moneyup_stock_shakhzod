import numpy as np
import pandas as pd

from tagent.features import make_features


def sample_ohlcv(n=80, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    price = 100 + np.cumsum(rng.normal(0, 1, n))
    price = np.maximum(price, 1.0)
    close = pd.Series(price, index=idx)
    df = pd.DataFrame({
        "open": close.shift(1).fillna(close),
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": pd.Series(rng.integers(1_000_000, 2_000_000, n), index=idx).astype(float),
    })
    return df


def test_make_features_basic():
    df = sample_ohlcv()
    feats = make_features(df)
    assert not feats.empty
    expected = {"ret_1", "vol_10", "mom_5", "close_sma20",
                "rsi_14", "macd", "macd_hist", "pos_in_range", "vol_ratio"}
    assert expected.issubset(set(feats.columns))


def test_no_nan_or_inf_after_dropna():
    feats = make_features(sample_ohlcv())
    assert np.isfinite(feats.to_numpy()).all()


def test_rsi_bounds():
    feats = make_features(sample_ohlcv())
    assert (feats["rsi_14"] >= 0).all()
    assert (feats["rsi_14"] <= 100).all()


def test_requires_ohlcv_columns():
    bad = pd.DataFrame({"close": [1, 2, 3]})
    try:
        make_features(bad)
        assert False, "expected ValueError"
    except ValueError:
        pass
