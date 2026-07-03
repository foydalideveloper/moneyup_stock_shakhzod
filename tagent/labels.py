"""Triple-barrier labeling (López de Prado).

For each bar we look forward up to `horizon` bars and ask which barrier is hit
first: the upper (take-profit) or lower (stop-loss). This produces a realistic
"was this a good long entry?" target instead of a naive next-bar return.

Label: 1 = good (hit the profit target first, or positive at timeout),
        0 = bad (hit the stop first, or non-positive at timeout).
If both barriers are touched in the same bar we conservatively assume the stop
hit first (label 0). The final `horizon` rows are NaN (not enough future data).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def triple_barrier_labels(df: pd.DataFrame, tp_pct: float = 0.04,
                          sl_pct: float = 0.02, horizon: int = 10,
                          timeout: str = "sign") -> pd.Series:
    if "close" not in df.columns:
        raise ValueError("df must contain a 'close' column")

    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float) if "high" in df.columns else close
    low = df["low"].to_numpy(dtype=float) if "low" in df.columns else close

    n = len(close)
    labels = np.full(n, np.nan)

    for t in range(n):
        end = t + horizon
        if end >= n:  # not enough future bars from here on
            break
        entry = close[t]
        if entry <= 0:
            continue
        up = entry * (1.0 + tp_pct)
        dn = entry * (1.0 - sl_pct)

        outcome = None
        for k in range(t + 1, end + 1):
            hit_up = high[k] >= up
            hit_dn = low[k] <= dn
            if hit_up and hit_dn:
                outcome = 0  # conservative: assume stop first
                break
            if hit_up:
                outcome = 1
                break
            if hit_dn:
                outcome = 0
                break

        if outcome is None:  # timeout
            if timeout == "sign":
                outcome = 1 if close[end] > entry else 0
            else:
                outcome = 0
        labels[t] = outcome

    return pd.Series(labels, index=df.index, name="label")
