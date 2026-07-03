"""Value factor — long the CHEAP names (low PBR and/or low PER).

Signal (higher = cheaper = more attractive) = the cross-sectional z-score of -PBR
(and optionally -PER), so the xs_momentum engine longs the top quantile, equal-
weight, monthly. Non-positive PBR/PER (negative book/earnings) are dropped — a low
PER from negative earnings is not 'cheap'.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from tagent.strategies.factor_utils import avg_panels, xs_zscore


def value_signal(pbr: Optional[pd.DataFrame] = None,
                 per: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Cross-sectional cheapness signal from PBR and/or PER panels (higher = cheaper)."""
    legs = []
    if pbr is not None:
        legs.append(-xs_zscore(pbr.where(pbr > 0)))      # low PBR -> high signal
    if per is not None:
        legs.append(-xs_zscore(per.where(per > 0)))      # low (positive) PER -> high signal
    if not legs:
        raise ValueError("value_signal needs a pbr and/or per panel")
    return legs[0] if len(legs) == 1 else avg_panels(legs)
