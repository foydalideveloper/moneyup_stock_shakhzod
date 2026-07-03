import numpy as np
import pandas as pd

from tagent.labels import triple_barrier_labels


def test_triple_barrier_known_outcomes():
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    df = pd.DataFrame({
        "close": [100, 102, 106, 100, 96, 95],
        "high":  [100, 103, 107, 101, 97, 96],
        "low":   [100, 101, 104, 99, 95, 94],
    }, index=idx)

    labels = triple_barrier_labels(df, tp_pct=0.05, sl_pct=0.03, horizon=3)

    # t0: up-barrier (105) hit at t2 high=107 -> 1
    assert labels.iloc[0] == 1
    # t1: down-barrier hit first within horizon -> 0
    assert labels.iloc[1] == 0
    # t2: down-barrier hit at t3 low=99 -> 0
    assert labels.iloc[2] == 0
    # last `horizon` rows can't be evaluated -> NaN
    assert np.isnan(labels.iloc[3])
    assert np.isnan(labels.iloc[4])
    assert np.isnan(labels.iloc[5])


def test_labels_are_binary_where_defined():
    idx = pd.date_range("2024-01-01", periods=30, freq="D")
    close = pd.Series(np.linspace(100, 130, 30), index=idx)
    df = pd.DataFrame({"close": close, "high": close * 1.01, "low": close * 0.99})
    labels = triple_barrier_labels(df, horizon=5).dropna()
    assert set(labels.unique()).issubset({0, 1})
