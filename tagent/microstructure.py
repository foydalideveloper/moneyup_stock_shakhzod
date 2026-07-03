"""Microstructure edge experiment: does the order book predict short-term moves?

ONE question, answered honestly: do the live `OrderBookMemory` features (depth
imbalance, absorption, spoof, size imbalance, net filled-vs-cancelled flow)
predict the FORWARD mid-price return over short horizons (5s / 30s / 60s)?

We measure it two ways:
  * **Information Coefficient (IC)** — Spearman rank-correlation of a feature at
    time t vs the forward return from t to t+H. |IC| ~0 means no monotonic signal.
  * **Strict out-of-sample test** — split the recording chronologically (train =
    earlier, test = later; no shuffling, no lookahead). Learn only the *sign* of
    the relationship on train, then on the held-out later part measure the OOS IC
    and the directional **hit rate**. Compare the typical move size to trading
    costs — a signal that can't clear the spread + fees is not an edge.

All pure pandas/numpy, so the maths is unit-tested on synthetic data with no
network and no recorded file.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Microstructure features tested as predictors (derived columns added by `study`).
DEFAULT_FEATURES = ["depth_imbalance", "absorption_ratio", "spoof_ratio",
                    "size_imbalance", "net_flow"]
DEFAULT_HORIZONS = [5, 30, 60]            # seconds


def to_epoch_seconds(ts) -> np.ndarray:
    """ISO timestamps / datetimes -> float seconds since epoch (UTC).

    ``format="ISO8601"`` so mixed fractional-second precision (e.g. trade prints
    at ms vs whole-second depth rows) parses without pandas' strict-format error.
    """
    s = pd.Series(ts)
    try:
        dt = pd.to_datetime(s.values, utc=True, format="ISO8601")
    except (ValueError, TypeError):
        dt = pd.to_datetime(s.values, utc=True)               # datetimes / non-string fallback
    return dt.astype("int64").to_numpy() / 1e9


def forward_returns(ts_seconds, mid, horizon_s: float,
                    tol_s: Optional[float] = None) -> np.ndarray:
    """Forward mid return from each row to the first row >= t + horizon_s.

    Returns NaN where no future row lands within ``tol_s`` of the target time (a
    data gap). ``ts_seconds`` must be ascending. The forward price uses FUTURE
    data on purpose — it is the prediction *target*, not a feature.
    """
    ts = np.asarray(ts_seconds, dtype=float)
    mid = np.asarray(mid, dtype=float)
    n = len(ts)
    out = np.full(n, np.nan)
    if n == 0:
        return out
    if tol_s is None:
        tol_s = max(2.0, 0.5 * horizon_s)
    target = ts + horizon_s
    j = np.searchsorted(ts, target, side="left")     # first index with ts[j] >= target
    ok = j < n
    jj = np.where(ok, j, 0)
    within = ok & (ts[jj] <= target + tol_s) & (mid > 0)
    out[within] = mid[jj[within]] / mid[within] - 1.0
    return out


def information_coefficient(feature, fwd_ret) -> float:
    """Spearman rank-correlation (IC) of feature vs forward return."""
    df = pd.DataFrame({"f": np.asarray(feature, float),
                       "r": np.asarray(fwd_ret, float)})
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < 10 or df["f"].nunique() < 3 or df["r"].nunique() < 3:
        return float("nan")
    return float(df["f"].rank().corr(df["r"].rank()))


def time_split(n: int, train_frac: float = 0.6):
    """Chronological split (no shuffling): (train_idx, test_idx), train precedes test."""
    cut = int(round(n * train_frac))
    cut = max(1, min(n - 1, cut)) if n >= 2 else n
    return np.arange(0, cut), np.arange(cut, n)


def oos_eval(feature, fwd_ret, train_frac: float = 0.6) -> Dict[str, float]:
    """Strict OOS test for one feature: learn the relationship's sign on the
    earlier part, evaluate IC + directional hit rate on the held-out later part."""
    f = np.asarray(feature, float)
    r = np.asarray(fwd_ret, float)
    n = len(f)
    tr, te = time_split(n, train_frac)
    in_ic = information_coefficient(f[tr], r[tr])
    oos_ic = information_coefficient(f[te], r[te])
    med = np.nanmedian(f[tr]) if np.isfinite(f[tr]).any() else 0.0
    sgn = np.sign(in_ic) if np.isfinite(in_ic) else 0.0
    fte, rte = f[te], r[te]
    pred = np.sign(fte - med) * sgn                  # predicted direction from train
    m = np.isfinite(fte) & np.isfinite(rte) & (pred != 0) & (rte != 0)
    hit = float(np.mean(pred[m] == np.sign(rte[m]))) if m.sum() >= 10 else float("nan")
    return {"in_ic": in_ic, "oos_ic": oos_ic, "oos_hit": hit,
            "n_test": int(np.isfinite(fte).sum())}


def _add_derived(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if {"bid_size", "ask_size"}.issubset(out.columns):
        denom = (out["bid_size"] + out["ask_size"]).replace(0.0, np.nan)
        out["size_imbalance"] = (out["bid_size"] - out["ask_size"]) / denom
    if {"filled_qty", "cancelled_qty"}.issubset(out.columns):
        out["net_flow"] = out["filled_qty"] - out["cancelled_qty"]
    return out


def study(df: pd.DataFrame, horizons: List[int] = None,
          features: List[str] = None, train_frac: float = 0.6) -> List[dict]:
    """Per (feature, horizon): in-sample IC, OOS IC, OOS hit rate, mean |fwd ret|.

    `df` needs columns ``ts``, ``mid`` and the feature columns; derived features
    (``size_imbalance``, ``net_flow``) are added automatically.
    """
    horizons = horizons or DEFAULT_HORIZONS
    df = _add_derived(df).copy()
    df = df.dropna(subset=["ts", "mid"]).sort_values("ts").reset_index(drop=True)
    feats = [f for f in (features or DEFAULT_FEATURES) if f in df.columns]
    ts_s = to_epoch_seconds(df["ts"])
    mid = pd.to_numeric(df["mid"], errors="coerce").to_numpy()

    rows: List[dict] = []
    for h in horizons:
        fwd = forward_returns(ts_s, mid, h)
        mean_abs_bps = float(np.nanmean(np.abs(fwd)) * 1e4) if np.isfinite(fwd).any() else float("nan")
        for feat in feats:
            fv = pd.to_numeric(df[feat], errors="coerce").to_numpy()
            res = oos_eval(fv, fwd, train_frac)
            rows.append({"feature": feat, "horizon_s": h,
                         "ic": information_coefficient(fv, fwd),
                         "oos_ic": res["oos_ic"], "oos_hit": res["oos_hit"],
                         "n_test": res["n_test"], "mean_abs_bps": round(mean_abs_bps, 2)})
    return rows


def verdict(rows: List[dict], cost_bps: float = 10.0, ic_thresh: float = 0.03,
            hit_thresh: float = 0.52) -> dict:
    """Decide edge / no edge, distinguishing *statistical* from *tradable* signal.

    A candidate **edge** needs a non-trivial OOS IC, a hit rate above chance, AND a
    typical move bigger than round-trip costs. A feature that clears the IC + hit
    bars but whose move is *below* costs is a real **statistical** signal that is
    **not tradable** — reported separately, honestly.
    """
    def is_stat(r):
        return (np.isfinite(r["oos_ic"]) and abs(r["oos_ic"]) >= ic_thresh
                and np.isfinite(r["oos_hit"]) and r["oos_hit"] >= hit_thresh)
    stat = [r for r in rows if is_stat(r)]
    edges = [r for r in stat
             if np.isfinite(r["mean_abs_bps"]) and r["mean_abs_bps"] > cost_bps]
    statistical_only = [r for r in stat if r not in edges]
    return {"has_edge": bool(edges), "edges": edges,
            "statistical_only": statistical_only, "cost_bps": cost_bps,
            "ic_thresh": ic_thresh, "hit_thresh": hit_thresh}
