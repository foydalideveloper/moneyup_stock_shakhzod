"""Quality factor — long the PROFITABLE names (high ROE).

pykrx fundamentals give EPS (trailing earnings per share) and BPS (book value per
share); ROE ~= EPS / BPS. Signal (higher = higher quality) = the cross-sectional
z-score of ROE, so the engine longs the most profitable top quantile, equal-weight,
monthly.
"""

from __future__ import annotations

import pandas as pd

from tagent.strategies.factor_utils import xs_zscore


def roe_panel(eps: pd.DataFrame, bps: pd.DataFrame) -> pd.DataFrame:
    """ROE proxy = EPS / BPS (per-share earnings over per-share book)."""
    return eps.div(bps.where(bps != 0))


def quality_signal(eps: pd.DataFrame, bps: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional quality signal (higher ROE = more attractive)."""
    return xs_zscore(roe_panel(eps, bps))
