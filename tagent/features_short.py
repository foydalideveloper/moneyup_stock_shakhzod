"""Short-selling (공매도) features, aligned to price bars with NO lookahead.

KRX short-selling data is DAILY / end-of-day and *delayed*: the figures for
trading day D are not knowable until after D (typically the next business day).
So when we attach a short metric to a price bar dated T, we may only use short
data dated **strictly before T**.

This module enforces that with two steps:

1. **Delay** every short-derived series by ``shift_days`` rows (default 1). A
   daily series shifted by one row means "the value as of the previous trading
   day" — i.e. the most recent figure actually published by bar T.
2. **As-of align** the delayed series onto the price index with a backward fill
   (each price bar takes the last delayed short value dated on/before it). Because
   the series was already shifted, even a same-date match carries only prior-day
   information — no future leakage.

Features produced (all aligned to the supplied price index):
    short_ratio          — delayed daily short_volume / volume
    short_ratio_change    — day-over-day change in short_ratio
    short_balance_change  — day-over-day change in short_balance (NaN if no balance)
    short_ratio_z         — rolling z-score of short_ratio (z_window, default 20)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SHORT_FEATURE_COLS = [
    "short_ratio",
    "short_ratio_change",
    "short_balance_change",
    "short_ratio_z",
]

# KR short-selling BAN windows: during these, 공매도 volume is suppressed to ~0
# by regulation, so short features are a regulatory artifact, not market signal.
# Studies should flag/exclude these bars. (Inclusive date ranges.)
SHORT_BAN_WINDOWS = [
    ("2020-03-16", "2021-05-02"),   # COVID full-market ban (partial lift 2021-05-03)
    ("2023-11-06", "2025-03-31"),   # 2023 ban, extended; lifted 2025-03-31
]


def in_short_ban(index) -> "pd.Series":
    """Boolean mask (True = bar falls inside a short-selling-ban window)."""
    idx = pd.DatetimeIndex(index)
    mask = pd.Series(False, index=idx)
    for lo, hi in SHORT_BAN_WINDOWS:
        mask |= (idx >= pd.Timestamp(lo)) & (idx <= pd.Timestamp(hi))
    return mask


def _raw_short_features(short_df: pd.DataFrame, z_window: int) -> pd.DataFrame:
    """Compute the features on the short data's own daily index (pre-delay)."""
    sr = short_df["short_ratio"].astype(float)

    f = pd.DataFrame(index=short_df.index)
    f["short_ratio"] = sr
    f["short_ratio_change"] = sr.diff()

    if "short_balance" in short_df.columns and short_df["short_balance"].notna().any():
        bal = short_df["short_balance"].astype(float)
        f["short_balance_change"] = bal.diff()
    else:
        # Balance not available in this source (e.g. a universe pull that skipped
        # the 공매도잔고 call). Use a constant 0 — a benign no-information feature —
        # rather than NaN, so these rows survive the study's dropna().
        f["short_balance_change"] = 0.0

    roll = sr.rolling(z_window)
    std = roll.std().replace(0.0, np.nan)
    f["short_ratio_z"] = (sr - roll.mean()) / std

    return f[SHORT_FEATURE_COLS]


def align_short_to_bars(short_feat: pd.DataFrame, price_index: pd.Index,
                        shift_days: int = 1) -> pd.DataFrame:
    """Delay by ``shift_days`` then as-of align onto ``price_index`` (no leakage).

    Each price bar receives the most recent *delayed* short value dated on/before
    it. Bars before the first available (delayed) short date stay NaN.
    """
    delayed = short_feat.shift(shift_days)
    # As-of backward fill: union the indexes, forward-fill, then keep price bars.
    combined = delayed.index.union(price_index)
    aligned = delayed.reindex(combined).sort_index().ffill()
    return aligned.reindex(price_index)


def make_short_features(short_df: pd.DataFrame, price_index: pd.Index,
                        z_window: int = 20, shift_days: int = 1) -> pd.DataFrame:
    """Build leak-free short-selling features aligned to ``price_index``.

    `short_df` is the canonical daily frame from
    :func:`tagent.data.short_selling.load_short_selling`. The result is indexed
    by ``price_index`` so it joins straight onto the price-based feature matrix.
    """
    raw = _raw_short_features(short_df, z_window=z_window)
    aligned = align_short_to_bars(raw, price_index, shift_days=shift_days)
    aligned.index.name = getattr(price_index, "name", None) or "timestamp"
    return aligned


def merge_short_features(features: pd.DataFrame, short_df: pd.DataFrame,
                         z_window: int = 20, shift_days: int = 1) -> pd.DataFrame:
    """Join leak-free short features onto an existing price-feature DataFrame.

    Alignment uses ``features.index`` as the price bars. Returns a new DataFrame;
    the input is not mutated. Short feature columns overwrite any pre-existing
    columns of the same name.
    """
    short_feats = make_short_features(
        short_df, features.index, z_window=z_window, shift_days=shift_days)
    out = features.drop(columns=[c for c in SHORT_FEATURE_COLS if c in features.columns],
                        errors="ignore")
    return out.join(short_feats)
