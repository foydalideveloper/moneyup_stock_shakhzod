"""Generic leak-free features for the gated KR signals (:mod:`tagent.data.krx_signals`).

Same no-lookahead discipline as the short/flow features: these are DAILY / EOD /
delayed, so for a price bar dated T we may only use signal data dated **before**
T. Enforced by:

1. **Delay** every signal series by ``shift_days`` rows (default 1).
2. **As-of align** the delayed series onto the price index (backward fill), so
   each bar takes the most recent prior-day value — no future leakage.

For each canonical value column ``c`` the features are:
    <c>          — delayed daily level / net
    <c>_change   — day-over-day change
    <c>_z        — rolling z-score over ``z_window`` days

Works for any signal frame; pass ``value_cols`` (defaults to all numeric columns).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


def signal_feature_columns(value_cols: List[str]) -> List[str]:
    cols: List[str] = []
    for c in value_cols:
        cols += [c, f"{c}_change", f"{c}_z"]
    return cols


def _value_cols(signal_df: pd.DataFrame, value_cols: Optional[List[str]]) -> List[str]:
    if value_cols is not None:
        return [c for c in value_cols if c in signal_df.columns]
    return [c for c in signal_df.columns
            if pd.api.types.is_numeric_dtype(signal_df[c])]


def _raw_features(signal_df: pd.DataFrame, value_cols: List[str], z_window: int) -> pd.DataFrame:
    f = pd.DataFrame(index=signal_df.index)
    for c in value_cols:
        s = pd.to_numeric(signal_df[c], errors="coerce").astype(float)
        f[c] = s
        f[f"{c}_change"] = s.diff()
        std = s.rolling(z_window).std().replace(0.0, np.nan)
        f[f"{c}_z"] = (s - s.rolling(z_window).mean()) / std
    return f[signal_feature_columns(value_cols)]


def align_to_bars(feat: pd.DataFrame, price_index: pd.Index,
                  shift_days: int = 1) -> pd.DataFrame:
    """Delay by ``shift_days`` then as-of align onto ``price_index`` (no leakage)."""
    delayed = feat.shift(shift_days)
    combined = delayed.index.union(price_index)
    aligned = delayed.reindex(combined).sort_index().ffill()
    return aligned.reindex(price_index)


def make_signal_features(signal_df: pd.DataFrame, price_index: pd.Index,
                         value_cols: Optional[List[str]] = None,
                         z_window: int = 20, shift_days: int = 1) -> pd.DataFrame:
    """Build leak-free features for a signal frame, aligned to ``price_index``."""
    cols = _value_cols(signal_df, value_cols)
    if not cols:
        raise ValueError("no numeric value columns to build features from")
    raw = _raw_features(signal_df, cols, z_window=z_window)
    aligned = align_to_bars(raw, price_index, shift_days=shift_days)
    aligned.index.name = getattr(price_index, "name", None) or "timestamp"
    return aligned


def merge_signal_features(features: pd.DataFrame, signal_df: pd.DataFrame,
                          value_cols: Optional[List[str]] = None,
                          z_window: int = 20, shift_days: int = 1) -> pd.DataFrame:
    """Join leak-free signal features onto an existing feature DataFrame.

    Alignment uses ``features.index`` as the price bars. Returns a new DataFrame;
    the input is not mutated. Signal columns overwrite same-named existing ones.
    """
    sig = make_signal_features(signal_df, features.index, value_cols=value_cols,
                               z_window=z_window, shift_days=shift_days)
    out = features.drop(columns=[c for c in sig.columns if c in features.columns],
                        errors="ignore")
    return out.join(sig)
