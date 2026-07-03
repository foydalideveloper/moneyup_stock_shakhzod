"""Classic candlestick pattern detectors (pure OHLC, no lookahead).

Each detector inspects bars at/just before an index ``i`` (never after — the
signal fires on bar ``i``'s close and is scored forward), and returns
``"bullish"`` / ``"bearish"`` / ``None``. :func:`detect_patterns` runs them all at
the most recent bar and returns ``[{"name", "direction", "index"}, ...]`` for the
TA agent to draw + forward-test on the scorecard, one signal per pattern.

Patterns: engulfing, pin bar (hammer / shooting star), inside bar, morning /
evening star, three white soldiers / three black crows, rising / falling three
methods. Thresholds are deliberately simple and unit-tested on synthetic bars.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

# every detector and its minimum number of bars of history it needs
PATTERN_BARS = {
    "engulfing": 2, "pin_bar": 1, "inside_bar": 2, "star": 3,
    "three_soldiers": 3, "three_methods": 5,
}


def _ohlc(df: pd.DataFrame):
    return (df["open"].to_numpy(float), df["high"].to_numpy(float),
            df["low"].to_numpy(float), df["close"].to_numpy(float))


def _body(o, c):
    return abs(c - o)


def _bull(o, c):
    return c > o


def engulfing(o, h, l, c, i: int) -> Optional[str]:
    """Current real body fully engulfs the prior real body, opposite colour."""
    if i < 1:
        return None
    pb = _body(o[i - 1], c[i - 1])
    cb = _body(o[i], c[i])
    if cb <= pb or pb <= 0:
        return None
    top, bot = max(o[i], c[i]), min(o[i], c[i])
    ptop, pbot = max(o[i - 1], c[i - 1]), min(o[i - 1], c[i - 1])
    if not (top >= ptop and bot <= pbot):
        return None
    if not _bull(o[i - 1], c[i - 1]) and _bull(o[i], c[i]):
        return "bullish"
    if _bull(o[i - 1], c[i - 1]) and not _bull(o[i], c[i]):
        return "bearish"
    return None


def pin_bar(o, h, l, c, i: int) -> Optional[str]:
    """Hammer (long lower wick -> bullish) / shooting star (long upper wick -> bearish)."""
    rng = h[i] - l[i]
    body = _body(o[i], c[i])
    if rng <= 0 or body > 0.35 * rng:
        return None
    upper = h[i] - max(o[i], c[i])
    lower = min(o[i], c[i]) - l[i]
    if lower >= 2.0 * body and upper <= body:
        return "bullish"                                  # hammer
    if upper >= 2.0 * body and lower <= body:
        return "bearish"                                  # shooting star
    return None


def inside_bar(o, h, l, c, i: int) -> Optional[str]:
    """Current bar's range is inside the prior (mother) bar; direction = mother's."""
    if i < 1:
        return None
    if h[i] <= h[i - 1] and l[i] >= l[i - 1] and (h[i] - l[i]) < (h[i - 1] - l[i - 1]):
        return "bullish" if _bull(o[i - 1], c[i - 1]) else "bearish"
    return None


def star(o, h, l, c, i: int) -> Optional[str]:
    """Morning star (bullish) / evening star (bearish): big bar, small-body star,
    big opposite bar reversing into the first bar's body."""
    if i < 2:
        return None
    b0 = _body(o[i - 2], c[i - 2]); b1 = _body(o[i - 1], c[i - 1]); b2 = _body(o[i], c[i])
    if b0 <= 0 or b2 <= 0 or b1 > 0.5 * b0 or b1 > 0.5 * b2:
        return None                                        # middle must be a small star
    mid0 = (o[i - 2] + c[i - 2]) / 2.0
    # morning: bar0 bearish, bar2 bullish closing back above bar0 midpoint
    if (not _bull(o[i - 2], c[i - 2]) and _bull(o[i], c[i]) and c[i] > mid0
            and max(o[i - 1], c[i - 1]) < c[i - 2]):
        return "bullish"
    if (_bull(o[i - 2], c[i - 2]) and not _bull(o[i], c[i]) and c[i] < mid0
            and min(o[i - 1], c[i - 1]) > c[i - 2]):
        return "bearish"
    return None


def three_soldiers(o, h, l, c, i: int) -> Optional[str]:
    """Three white soldiers (bullish) / three black crows (bearish): three strong
    same-colour bars each closing further in the trend, opening within the prior body."""
    if i < 2:
        return None
    idx = (i - 2, i - 1, i)
    bulls = [_bull(o[k], c[k]) for k in idx]
    bodies = [_body(o[k], c[k]) for k in idx]
    if any(b <= 0 for b in bodies):
        return None
    strong = all(bodies[j] > 0.6 * (h[idx[j]] - l[idx[j]] or 1e-12) for j in range(3))
    if all(bulls) and strong and c[i] > c[i - 1] > c[i - 2] \
            and o[i - 1] <= c[i - 2] and o[i] <= c[i - 1] \
            and o[i - 1] >= o[i - 2] and o[i] >= o[i - 1]:
        return "bullish"
    if not any(bulls) and strong and c[i] < c[i - 1] < c[i - 2] \
            and o[i - 1] >= c[i - 2] and o[i] >= c[i - 1] \
            and o[i - 1] <= o[i - 2] and o[i] <= o[i - 1]:
        return "bearish"
    return None


def three_methods(o, h, l, c, i: int) -> Optional[str]:
    """Rising (bullish) / falling (bearish) three methods: a long bar, three small
    counter bars holding inside its range, then a long bar closing past the first."""
    if i < 4:
        return None
    a, mids, e = i - 4, (i - 3, i - 2, i - 1), i
    big0 = _body(o[a], c[a]); big2 = _body(o[e], c[e])
    if big0 <= 0 or big2 <= 0:
        return None
    small = all(_body(o[m], c[m]) < 0.6 * big0 for m in mids)
    within = all(h[m] <= h[a] and l[m] >= l[a] for m in mids)
    if not (small and within):
        return None
    # rising: bar0 bullish, middles drift down, final bullish closes above bar0 close
    if _bull(o[a], c[a]) and _bull(o[e], c[e]) and c[e] > c[a] \
            and all(not _bull(o[m], c[m]) for m in mids):
        return "bullish"
    if not _bull(o[a], c[a]) and not _bull(o[e], c[e]) and c[e] < c[a] \
            and all(_bull(o[m], c[m]) for m in mids):
        return "bearish"
    return None


_DETECTORS = {
    "engulfing": engulfing, "pin_bar": pin_bar, "inside_bar": inside_bar,
    "star": star, "three_soldiers": three_soldiers, "three_methods": three_methods,
}

# friendly per-direction display names for the chart/scorecard
PATTERN_LABELS = {
    ("engulfing", "bullish"): "Bull Engulf", ("engulfing", "bearish"): "Bear Engulf",
    ("pin_bar", "bullish"): "Hammer", ("pin_bar", "bearish"): "Shooting Star",
    ("inside_bar", "bullish"): "Inside Bar↑", ("inside_bar", "bearish"): "Inside Bar↓",
    ("star", "bullish"): "Morning Star", ("star", "bearish"): "Evening Star",
    ("three_soldiers", "bullish"): "3 White Soldiers", ("three_soldiers", "bearish"): "3 Black Crows",
    ("three_methods", "bullish"): "Rising 3 Methods", ("three_methods", "bearish"): "Falling 3 Methods",
}


def detect_at(df: pd.DataFrame, i: int) -> List[dict]:
    """All patterns whose final bar is index ``i`` (uses only bars <= i)."""
    o, h, l, c = _ohlc(df)
    n = len(c)
    if not (0 <= i < n):
        return []
    out = []
    for name, fn in _DETECTORS.items():
        d = fn(o, h, l, c, i)
        if d:
            out.append({"name": name, "direction": d, "index": int(i),
                        "label": PATTERN_LABELS.get((name, d), name)})
    return out


def detect_patterns(df: pd.DataFrame, lookback: int = 1) -> List[dict]:
    """Patterns ending within the last ``lookback`` bars (default just the last bar)
    — the TA agent's fresh triggers. No lookahead: each uses only past/current bars."""
    n = len(df)
    if n == 0:
        return []
    out = []
    for i in range(max(0, n - lookback), n):
        out.extend(detect_at(df, i))
    return out
