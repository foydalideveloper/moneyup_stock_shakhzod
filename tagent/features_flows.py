"""수급 (investor-flow) features, aligned to price bars with NO lookahead.

KRX investor net-buying is DAILY / end-of-day, so a price bar dated T may only
use flow data dated **strictly before T**. This module enforces that exactly like
:mod:`tagent.features_short`:

1. **Delay** every flow-derived series by ``shift_days`` rows (default 1).
2. **As-of align** the delayed series onto the price index (backward fill), so
   each bar takes the most recent prior-day value — no future leakage.

Features produced for each of ``foreign_net`` and ``inst_net``:
    <flow>           — delayed daily net buying
    <flow>_sum<W>    — rolling sum over ``sum_window`` days (accumulated pressure)
    <flow>_change    — day-over-day change
    <flow>_z         — rolling z-score over ``z_window`` days
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_FLOWS = ["foreign_net", "inst_net"]


def flow_feature_columns(sum_window: int = 5) -> list:
    cols = []
    for base in _FLOWS:
        cols += [base, f"{base}_sum{sum_window}", f"{base}_change", f"{base}_z"]
    return cols


# Default feature set (sum_window=5) for callers that need a static list.
FLOW_FEATURE_COLS = flow_feature_columns(5)


def _raw_flow_features(flow_df: pd.DataFrame, sum_window: int, z_window: int) -> pd.DataFrame:
    """Compute features on the flow data's own daily index (pre-delay)."""
    f = pd.DataFrame(index=flow_df.index)
    for base in _FLOWS:
        s = pd.to_numeric(flow_df[base], errors="coerce").astype(float)
        f[base] = s
        f[f"{base}_sum{sum_window}"] = s.rolling(sum_window).sum()
        f[f"{base}_change"] = s.diff()
        std = s.rolling(z_window).std().replace(0.0, np.nan)
        f[f"{base}_z"] = (s - s.rolling(z_window).mean()) / std
    return f[flow_feature_columns(sum_window)]


def align_flows_to_bars(flow_feat: pd.DataFrame, price_index: pd.Index,
                        shift_days: int = 1) -> pd.DataFrame:
    """Delay by ``shift_days`` then as-of align onto ``price_index`` (no leakage)."""
    delayed = flow_feat.shift(shift_days)
    combined = delayed.index.union(price_index)
    aligned = delayed.reindex(combined).sort_index().ffill()
    return aligned.reindex(price_index)


def make_flow_features(flow_df: pd.DataFrame, price_index: pd.Index,
                       sum_window: int = 5, z_window: int = 20,
                       shift_days: int = 1) -> pd.DataFrame:
    """Build leak-free investor-flow features aligned to ``price_index``.

    `flow_df` is the canonical daily frame from
    :func:`tagent.data.flows_source.load_flows`.
    """
    raw = _raw_flow_features(flow_df, sum_window=sum_window, z_window=z_window)
    aligned = align_flows_to_bars(raw, price_index, shift_days=shift_days)
    aligned.index.name = getattr(price_index, "name", None) or "timestamp"
    return aligned


def merge_flow_features(features: pd.DataFrame, flow_df: pd.DataFrame,
                        sum_window: int = 5, z_window: int = 20,
                        shift_days: int = 1) -> pd.DataFrame:
    """Join leak-free flow features onto an existing feature DataFrame.

    Alignment uses ``features.index`` as the price bars. Returns a new DataFrame;
    the input is not mutated. Flow columns overwrite any pre-existing columns of
    the same name.
    """
    flow_feats = make_flow_features(
        flow_df, features.index, sum_window=sum_window, z_window=z_window,
        shift_days=shift_days)
    out = features.drop(columns=[c for c in flow_feats.columns if c in features.columns],
                        errors="ignore")
    return out.join(flow_feats)
