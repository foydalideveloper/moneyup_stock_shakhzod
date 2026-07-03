"""Multi-factor blend — combine value + quality + momentum into one signal.

Each input is re-z-scored cross-sectionally (so the factors are on a comparable
scale) and averaged (NaN-skipping, optionally weighted). The result is a single
[time x symbol] signal the engine longs the top quantile of — the standard
'composite factor' construction.
"""

from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd

from tagent.strategies.factor_utils import avg_panels, xs_zscore


def momentum_signal_panel(close_wide: pd.DataFrame, lookback: int = 252,
                          skip_recent: int = 21) -> pd.DataFrame:
    """12-1 momentum as a rank-able signal panel (reuses the engine's formula)."""
    from tagent.xs_momentum import momentum_signal
    return momentum_signal(close_wide, lookback, skip_recent)


def combine_signals(signals: Sequence[pd.DataFrame],
                    weights: Optional[Sequence[float]] = None) -> pd.DataFrame:
    """Blend factor signals: z-score each, then NaN-skip weighted-average."""
    zs = [xs_zscore(s) for s in signals]
    return avg_panels(zs, weights=weights)
