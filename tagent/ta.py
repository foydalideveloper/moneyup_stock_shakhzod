"""Rule-based technical analysis overlay (swing pivots, trendlines, S/R, patterns).

Pure OHLCV maths used by the dashboard's chart overlay: find swing pivots, fit
upper/lower **trendlines** (a channel), cluster horizontal **support/resistance**
levels, classify a simple **pattern** (ascending/descending channel, triangle,
wedge), and flag a **breakout** (price crossing a trendline on a volume spike) —
with a 1–2 sentence rule-based explanation of what was drawn and why.

Everything is positional (line endpoints reference bar indices 0..n-1 into the
same window the chart draws), so it overlays directly. No network; unit-tested on
synthetic bars.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


def swing_pivots(high, low, left: int = 3, right: int = 3):
    """Indices of swing-high and swing-low pivots (local extrema with `left`/`right`
    bars lower on each side). The last `right` bars can't be confirmed pivots yet."""
    high = np.asarray(high, float)
    low = np.asarray(low, float)
    n = len(high)
    ph, pl = [], []
    for i in range(left, n - right):
        win = slice(i - left, i + right + 1)
        if high[i] >= high[win].max() - 1e-12 and high[i] >= high[i - 1] and high[i] >= high[i + 1]:
            ph.append(i)
        if low[i] <= low[win].min() + 1e-12 and low[i] <= low[i - 1] and low[i] <= low[i + 1]:
            pl.append(i)
    return ph, pl


def fit_line(xs, ys):
    """Least-squares line through (xs, ys) -> (slope, intercept), or None."""
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    if len(xs) < 2:
        return None
    slope, intercept = np.polyfit(xs, ys, 1)
    return float(slope), float(intercept)


def bounding_line(points, upper: bool):
    """A real trendline: the rightmost edge of the convex hull of ``points``
    (``[(index, price), ...]``). For ``upper=True`` it rides the swing HIGHS and
    stays at/above every high (resistance); for ``upper=False`` it rides the
    swing LOWS and stays at/below every low (support). Unlike a regression it
    passes through two ACTUAL pivots and never cuts through the price.

    Returns ``{"a","b","slope","intercept"}`` (a,b are the two touched pivots),
    or None if fewer than two distinct points.
    """
    pts = sorted({(int(i), float(p)) for i, p in points})
    if len(pts) < 2:
        return None

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    chain = []
    for p in pts:
        # upper hull keeps clockwise turns (boundary above); lower keeps ccw (below)
        while len(chain) >= 2 and (
                (cross(chain[-2], chain[-1], p) >= 0) if upper
                else (cross(chain[-2], chain[-1], p) <= 0)):
            chain.pop()
        chain.append(p)
    a, b = chain[-2], chain[-1]
    slope = (b[1] - a[1]) / (b[0] - a[0]) if b[0] != a[0] else 0.0
    intercept = a[1] - slope * a[0]
    return {"a": a, "b": b, "slope": float(slope), "intercept": float(intercept)}


def _cluster_levels(prices: List[float], tol_frac: float = 0.004,
                    max_levels: int = 2) -> List[dict]:
    """Cluster nearby prices into horizontal levels (mean + #touches)."""
    if not prices:
        return []
    ps = sorted(prices)
    clusters, cur = [], [ps[0]]
    for p in ps[1:]:
        if abs(p - cur[-1]) <= tol_frac * max(cur[-1], 1e-9):
            cur.append(p)
        else:
            clusters.append(cur); cur = [p]
    clusters.append(cur)
    levels = [{"price": float(np.mean(c)), "touches": len(c)} for c in clusters]
    levels.sort(key=lambda d: (-d["touches"], -d["price"]))
    return levels[:max_levels]


def _slope_sign(slope: Optional[tuple], typ_price: float, eps_frac: float = 0.0004) -> int:
    """+1 / 0 / -1 for a fitted line's slope, thresholded relative to price/bar."""
    if slope is None:
        return 0
    eps = eps_frac * max(typ_price, 1e-9)
    if slope[0] > eps:
        return 1
    if slope[0] < -eps:
        return -1
    return 0


def detect_pattern(upper, lower, typ_price: float) -> str:
    us, ls = _slope_sign(upper, typ_price), _slope_sign(lower, typ_price)
    if upper is None and lower is None:
        return "no clear structure"
    if us > 0 and ls > 0:
        return "ascending channel"
    if us < 0 and ls < 0:
        return "descending channel"
    if us < 0 and ls > 0:
        return "symmetrical triangle"
    if us == 0 and ls > 0:
        return "ascending triangle"
    if us < 0 and ls == 0:
        return "descending triangle"
    if upper is not None and lower is not None and (upper[0] - lower[0]) < 0:
        return "converging wedge"
    return "horizontal range"


def detect_breakout(upper, lower, df: pd.DataFrame, vol_mult: float = 1.5) -> Optional[dict]:
    """Last close crossing a trendline on a volume spike -> breakout event."""
    close = df["close"].to_numpy(float)
    vol = df["volume"].to_numpy(float)
    last = len(df) - 1
    if last < 21:
        return None
    avg_vol = float(np.mean(vol[last - 20:last]))
    vol_spike = avg_vol > 0 and vol[last] > vol_mult * avg_vol
    if upper is not None:
        uy = upper[0] * last + upper[1]
        if close[last] > uy and vol_spike:
            return {"direction": "up", "index": last, "price": float(close[last]),
                    "vol_x": round(float(vol[last] / avg_vol), 2)}
    if lower is not None:
        ly = lower[0] * last + lower[1]
        if close[last] < ly and vol_spike:
            return {"direction": "down", "index": last, "price": float(close[last]),
                    "vol_x": round(float(vol[last] / avg_vol), 2)}
    return None


def _explain(upper, lower, pattern, breakout, n_used, candle_patterns=None) -> str:
    drew = []
    if upper is not None:
        drew.append(f"an upper trendline through the last {n_used} swing highs")
    if lower is not None:
        drew.append("a lower trendline through the swing lows")
    s = ("Drew " + " and ".join(drew) + f" — a {pattern}.") if drew \
        else "Not enough confirmed swing pivots to draw trendlines yet."
    if breakout:
        side = "upper" if breakout["direction"] == "up" else "lower"
        bull = "bullish" if breakout["direction"] == "up" else "bearish"
        s += (f" Price broke through the {side} trendline on a "
              f"{breakout['vol_x']}× volume spike → {bull} breakout.")
    if candle_patterns:
        names = ", ".join(f"{p['label']} ({p['direction']})" for p in candle_patterns)
        s += f" Candlestick pattern{'s' if len(candle_patterns) > 1 else ''}: {names}."
    return s


def analyze(df: pd.DataFrame, left: int = 3, right: int = 3, n_pivots: int = 3,
            vol_mult: float = 1.5) -> dict:
    """Full overlay: trendlines, S/R levels, pattern, breakout, explanation.

    `df` needs open/high/low/close/volume; line endpoints are bar **indices**
    (0..n-1) into `df`, so they map straight onto the candle chart.
    """
    from tagent.candle_patterns import detect_patterns
    n = len(df)
    candle_patterns = detect_patterns(df) if {"open", "high", "low", "close"}.issubset(df.columns) else []
    out = {"lines": [], "levels": [], "pattern": "n/a", "breakout": None,
           "pivots": [], "candle_patterns": candle_patterns, "n_bars": int(n),
           "explanation": "Not enough bars for analysis yet."}
    if n < left + right + 6:
        return out

    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    typ = float(np.mean(df["close"].to_numpy(float)))
    ph, pl = swing_pivots(high, low, left, right)

    # every confirmed swing pivot, so the chart can mark the dots the lines connect
    pivots = ([{"type": "high", "index": int(i), "price": float(high[i])} for i in ph]
              + [{"type": "low", "index": int(i), "price": float(low[i])} for i in pl])

    # Trendlines are tangents through the swing extremes (hull edges), drawn over
    # the most recent pivots so the channel hugs current price.
    upper = lower = None
    lines, n_used = [], 0
    recent = max(2, n_pivots + 3)
    uh = bounding_line([(i, high[i]) for i in ph[-recent:]], upper=True) if len(ph) >= 2 else None
    if uh:
        upper = (uh["slope"], uh["intercept"])
        a, b = uh["a"], uh["b"]; n_used = 2
        lines.append({"type": "upper",
                      "points": [[a[0], upper[0] * a[0] + upper[1]],
                                 [int(n - 1), upper[0] * (n - 1) + upper[1]]],
                      "pivots": [[a[0], a[1]], [b[0], b[1]]]})
    lo = bounding_line([(i, low[i]) for i in pl[-recent:]], upper=False) if len(pl) >= 2 else None
    if lo:
        lower = (lo["slope"], lo["intercept"])
        a, b = lo["a"], lo["b"]
        lines.append({"type": "lower",
                      "points": [[a[0], lower[0] * a[0] + lower[1]],
                                 [int(n - 1), lower[0] * (n - 1) + lower[1]]],
                      "pivots": [[a[0], a[1]], [b[0], b[1]]]})

    res = _cluster_levels([float(high[i]) for i in ph[-6:]])
    sup = _cluster_levels([float(low[i]) for i in pl[-6:]])
    levels = ([{"type": "resistance", "price": r["price"], "touches": r["touches"]} for r in res]
              + [{"type": "support", "price": s["price"], "touches": s["touches"]} for s in sup])

    pattern = detect_pattern(upper, lower, typ)
    breakout = detect_breakout(upper, lower, df, vol_mult)
    out.update(lines=lines, levels=levels, pattern=pattern, breakout=breakout,
               pivots=pivots, candle_patterns=candle_patterns,
               explanation=_explain(upper, lower, pattern, breakout, n_used, candle_patterns))
    return out


def stabilize_pattern(state: dict, raw: str, min_confirm: int = 2):
    """Hysteresis for the pattern label so it doesn't flicker between polls.

    `state` is ``{"shown", "cand", "n"}``. The displayed pattern only switches to
    a new value after that value has appeared ``min_confirm`` polls in a row.
    Returns ``(new_state, changed)``.
    """
    shown = state.get("shown")
    if shown is None:
        return {"shown": raw, "cand": raw, "n": 0}, True
    if raw == shown:
        return {"shown": shown, "cand": shown, "n": 0}, False
    n = (state.get("n", 0) + 1) if raw == state.get("cand") else 1
    if n >= min_confirm:
        return {"shown": raw, "cand": raw, "n": 0}, True
    return {"shown": shown, "cand": raw, "n": n}, False
