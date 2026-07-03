"""Leak-free feature tests for the gated KR signals — synthetic, no network."""

import numpy as np
import pandas as pd
import pytest

from tagent.data.krx_signals import SIGNALS
from tagent.features import make_features
from tagent.features_signals import (
    make_signal_features,
    merge_signal_features,
    signal_feature_columns,
)

ALL_SIGNALS = list(SIGNALS)


def _signal_frame(value_cols, n=30, seed=0):
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    idx.name = "date"
    rng = np.random.default_rng(seed)
    return pd.DataFrame({c: rng.normal(0, 1e8, n) for c in value_cols}, index=idx)


@pytest.mark.parametrize("sig", ALL_SIGNALS)
def test_feature_columns_present_for_each_signal(sig):
    cols = SIGNALS[sig].value_cols
    idx = pd.date_range("2024-01-01", periods=30, freq="D")
    feats = make_signal_features(_signal_frame(cols, 30), idx, value_cols=cols, z_window=5)
    assert list(feats.columns) == signal_feature_columns(cols)
    assert feats.index.equals(idx)


@pytest.mark.parametrize("sig", ALL_SIGNALS)
def test_no_lookahead_for_each_signal(sig):
    """A spike on day D must surface only on the price bar for D+1, never D."""
    cols = SIGNALS[sig].value_cols
    col = cols[0]
    idx = pd.date_range("2024-01-01", periods=8, freq="D")
    df = _signal_frame(cols, 8, seed=1)
    df[col] = [1, 1, 1, 1, 999, 1, 1, 1]            # spike at idx[4]
    feats = make_signal_features(df, idx, value_cols=cols, z_window=3)
    assert feats.loc[idx[4], col] == 1              # spike NOT visible on its own bar
    assert feats.loc[idx[5], col] == 999            # surfaces exactly one day later


@pytest.mark.parametrize("sig", ALL_SIGNALS)
def test_first_bar_has_no_value(sig):
    cols = SIGNALS[sig].value_cols
    idx = pd.date_range("2024-02-01", periods=10, freq="D")
    df = _signal_frame(cols, 10); df.index = idx
    feats = make_signal_features(df, idx, value_cols=cols, z_window=3)
    assert np.isnan(feats[cols[0]].iloc[0])          # no prior day -> NaN (no leak)
    expected_prev = df[cols[0]].shift(1)
    for t in idx[1:]:
        assert np.isclose(feats.loc[t, cols[0]], expected_prev.loc[t])


def _ohlcv(idx, seed=0):
    rng = np.random.default_rng(seed)
    n = len(idx)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)), index=idx).clip(lower=1)
    return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                         "close": close, "volume": rng.integers(1e6, 2e6, n).astype(float)})


def test_merge_is_additive_and_nonmutating():
    cols = SIGNALS["investor_detail"].value_cols
    idx = pd.date_range("2024-01-01", periods=60, freq="D")
    base = make_features(_ohlcv(idx), dropna=False)
    before = list(base.columns)
    sig = _signal_frame(cols, 60); sig.index = idx

    merged = merge_signal_features(base, sig, value_cols=cols)
    assert list(base.columns) == before                       # input untouched
    assert len(merged) == len(base)                           # no rows added/dropped
    assert set(signal_feature_columns(cols)).issubset(merged.columns)
    assert len(merged.columns) == len(base.columns) + len(signal_feature_columns(cols))
