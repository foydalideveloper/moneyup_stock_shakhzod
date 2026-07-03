"""Shared cross-sectional factor helpers (z-scoring + NaN-robust blending).

A factor SIGNAL is a [time x symbol] panel where HIGHER = more attractive (long the
top quantile). These helpers standardize raw fundamentals into comparable, sign-
aligned signals that the xs_momentum engine can rank via its ``signal=`` injection.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def xs_zscore(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-date (cross-sectional) z-score: (x - row_mean) / row_std. Constant/empty
    rows -> NaN (no spurious ranking)."""
    mu = panel.mean(axis=1)
    sd = panel.std(axis=1)
    return panel.sub(mu, axis=0).div(sd.where(sd > 0), axis=0)


def avg_panels(panels: Sequence[pd.DataFrame], weights: Optional[Sequence[float]] = None) -> pd.DataFrame:
    """Element-wise mean of aligned panels, SKIPPING NaNs (so a name missing one
    factor still gets the average of the others). Optional per-panel weights."""
    panels = [p for p in panels if p is not None]
    if not panels:
        raise ValueError("avg_panels needs at least one panel")
    weights = list(weights) if weights is not None else [1.0] * len(panels)
    idx, cols = panels[0].index, panels[0].columns
    num = pd.DataFrame(0.0, index=idx, columns=cols)
    den = pd.DataFrame(0.0, index=idx, columns=cols)
    for p, w in zip(panels, weights):
        pa = p.reindex(index=idx, columns=cols)
        mask = pa.notna()
        num = num.add((pa.fillna(0.0) * w).where(mask, 0.0), fill_value=0.0)
        den = den.add(mask.astype(float) * abs(w), fill_value=0.0)
    return num.div(den.where(den > 0))
