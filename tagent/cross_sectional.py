"""Cross-sectional long-short research on the KR universe.

Per-stock daily up/down classification showed no edge vs buy-and-hold. This
module reframes the problem **cross-sectionally**: each row is a (day, stock);
the label is the stock's forward N-day return ranked *relative to the universe
that day* (top vs bottom tercile); ONE model is trained across ALL stocks to
predict that relative rank; and each day we go long the top-decile predicted
names and short (or stay flat on) the bottom decile — a market-neutral book.

Leakage controls:
* Features are the existing leak-free ones (technicals + 공매도 + 수급, each
  already 1-day EOD-shifted) and are additionally **cross-sectionally normalized**
  (z-scored within each day) so the model learns *relative* position.
* The label uses an N-day forward return, so a training row "knows" data up to
  t+N. Walk-forward CV therefore **purges** training days whose label window
  overlaps the test block and adds an **embargo** — a gap of (horizon + embargo)
  days between train end and test start. Every test prediction is out-of-sample.

Pure pandas/numpy (+ lazy LightGBM), so the logic is unit-testable with no
network and no real data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from tagent.data.short_selling import complete_short_ratio
from tagent.features import make_features
from tagent.features_flows import flow_feature_columns, merge_flow_features
from tagent.features_short import (
    SHORT_FEATURE_COLS, in_short_ban, merge_short_features,
)

_META_COLS = {"date", "symbol", "ret_1d", "fwd_ret"}

_LGB_PARAMS = dict(n_estimators=200, learning_rate=0.05, num_leaves=31,
                   subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)


# --------------------------------------------------------------------------- #
# panel construction
# --------------------------------------------------------------------------- #
def build_panel(history: Dict[str, pd.DataFrame],
                short_data: Optional[Dict[str, pd.DataFrame]] = None,
                flow_data: Optional[Dict[str, pd.DataFrame]] = None,
                horizon: int = 5, exclude_ban: bool = True
                ) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Stack per-stock features into one (day, stock) panel.

    Returns (panel, cols) where cols maps 'all'/'tech'/'short'/'flow' to column
    lists. `ret_1d` is the next-day return (portfolio P&L), `fwd_ret` the N-day
    forward return (ranking label basis).
    """
    frames = []
    for sym, df in history.items():
        df = df.sort_index()
        feats = make_features(df, dropna=False)
        if short_data and sym in short_data:
            feats = merge_short_features(
                feats, complete_short_ratio(short_data[sym], df["volume"]))
        if flow_data and sym in flow_data:
            feats = merge_flow_features(feats, flow_data[sym])
        close = df["close"].astype(float)
        feats = feats.copy()
        feats["ret_1d"] = close.shift(-1) / close - 1.0
        feats["fwd_ret"] = close.shift(-horizon) / close - 1.0
        feats["symbol"] = sym
        feats["date"] = feats.index
        frames.append(feats.reset_index(drop=True))

    panel = pd.concat(frames, ignore_index=True)
    if exclude_ban:
        panel = panel[~in_short_ban(panel["date"]).values].reset_index(drop=True)

    feat_cols = [c for c in panel.columns if c not in _META_COLS]
    short_cols = [c for c in feat_cols if c in SHORT_FEATURE_COLS]
    flow_cols = [c for c in feat_cols if c in flow_feature_columns()]
    tech_cols = [c for c in feat_cols if c not in short_cols and c not in flow_cols]
    cols = {"all": feat_cols, "tech": tech_cols, "short": short_cols, "flow": flow_cols}
    return panel.sort_values(["date", "symbol"]).reset_index(drop=True), cols


def cross_sectional_normalize(panel: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Z-score each feature within each day (contemporaneous, no leakage).

    A feature that is constant across the day's cross-section (std 0 — e.g. a
    degenerate column like short_balance_change when balance wasn't fetched)
    carries no relative information and is mapped to 0, NOT NaN, so those rows
    aren't dropped downstream. Genuinely missing values stay NaN.
    """
    out = panel.copy()
    g = out.groupby("date")
    mean = g[cols].transform("mean")
    std = g[cols].transform("std").replace(0.0, np.nan)
    z = (out[cols] - mean) / std
    # present-but-NaN-from-zero-std (constant feature) -> 0
    present = out[cols].notna()
    z = z.mask(present & z.isna(), 0.0)
    out[cols] = z
    return out


def make_rank_labels(panel: pd.DataFrame, q: int = 3) -> pd.Series:
    """Per-day q-quantile of fwd_ret: 1.0 = top group, 0.0 = bottom, NaN = middle.

    Middle quantiles are NaN so the model trains only on clear winners vs losers.
    """
    lab = pd.Series(np.nan, index=panel.index)
    for _day, g in panel.groupby("date"):
        fr = g["fwd_ret"].dropna()
        if len(fr) < q:
            continue
        try:
            qc = pd.qcut(fr.rank(method="first"), q, labels=False)
        except ValueError:
            continue
        lab.loc[fr.index[qc == q - 1]] = 1.0
        lab.loc[fr.index[qc == 0]] = 0.0
    return lab


# --------------------------------------------------------------------------- #
# purged + embargoed walk-forward
# --------------------------------------------------------------------------- #
def purged_walk_forward_days(days, n_splits: int, horizon: int, embargo: int):
    """Yield (train_days, test_days) over sorted unique days, expanding window,
    with a (horizon + embargo)-day gap purged between train end and test start."""
    days = list(pd.unique(pd.Series(days).sort_values()))
    n = len(days)
    fold = n // (n_splits + 1)
    if fold == 0:
        return
    gap = horizon + embargo
    for k in range(1, n_splits + 1):
        te_lo = k * fold
        te_hi = (k + 1) * fold if k < n_splits else n
        cut = te_lo - gap
        if cut <= 0 or te_lo >= n:
            continue
        yield days[:cut], days[te_lo:te_hi]


def purged_xs_predict(panel: pd.DataFrame, feat_cols: List[str], label: pd.Series,
                      n_splits: int = 5, horizon: int = 5, embargo: int = 2,
                      params: Optional[dict] = None) -> pd.Series:
    """Out-of-sample P(top-group) per (day, stock) via purged walk-forward."""
    try:
        import lightgbm as lgb
    except ImportError as e:  # pragma: no cover
        raise ImportError("Install ML deps: pip install -r requirements.txt") from e
    params = params or _LGB_PARAMS

    scores = pd.Series(np.nan, index=panel.index)
    date = panel["date"]
    feats_ok = panel[feat_cols].notna().all(axis=1)
    for train_days, test_days in purged_walk_forward_days(
            date.values, n_splits, horizon, embargo):
        tr = date.isin(train_days).values & feats_ok.values & label.notna().values
        if tr.sum() < 50 or label[tr].nunique() < 2:
            continue
        te = date.isin(test_days).values & feats_ok.values
        if te.sum() == 0:
            continue
        model = lgb.LGBMClassifier(**params)
        model.fit(panel.loc[tr, feat_cols], label[tr].astype(int))
        proba = model.predict_proba(panel.loc[te, feat_cols])[:, 1]
        scores.loc[panel.index[te]] = proba
    return scores


# --------------------------------------------------------------------------- #
# portfolio backtest
# --------------------------------------------------------------------------- #
def _decile_weights(group: pd.DataFrame, score_col: str, top_q: float,
                    allow_short: bool) -> pd.Series:
    n = len(group)
    k = max(1, int(round(n * top_q)))
    g = group.sort_values(score_col)
    w = pd.Series(0.0, index=g.index)
    w.iloc[-k:] = 1.0 / k                      # long top-k (highest score)
    if allow_short:
        w.iloc[:k] = -1.0 / k                  # short bottom-k
    return w


def long_short_returns(panel: pd.DataFrame, score_col: str = "score",
                       top_q: float = 0.1, allow_short: bool = True,
                       cost_bps: float = 5.0, slippage_bps: float = 2.0) -> pd.Series:
    """Daily net return of the decile portfolio (costs on turnover).

    allow_short=True -> dollar-neutral long-short; False -> long-only top decile.
    """
    df = panel[panel[score_col].notna() & panel["ret_1d"].notna()].copy()
    if df.empty:
        return pd.Series(dtype=float)
    parts = [_decile_weights(g, score_col, top_q, allow_short)
             for _d, g in df.groupby("date")]
    df["w"] = pd.concat(parts)

    wp = df.pivot_table(index="date", columns="symbol", values="w", fill_value=0.0)
    rp = df.pivot_table(index="date", columns="symbol", values="ret_1d", fill_value=0.0)
    gross = (wp * rp).sum(axis=1)
    turnover = wp.diff().abs().sum(axis=1)
    turnover.iloc[0] = wp.iloc[0].abs().sum()  # initial entry
    cost = turnover * (cost_bps + slippage_bps) / 10_000.0
    return (gross - cost).rename(score_col)


def benchmark_returns(panel: pd.DataFrame, days=None) -> Dict[str, pd.Series]:
    """Equal-weight index (daily-rebalanced) and buy-and-hold universe (drift),
    over the panel's days (optionally restricted to ``days``)."""
    df = panel[panel["ret_1d"].notna()]
    if days is not None:
        df = df[df["date"].isin(set(days))]
    rp = df.pivot_table(index="date", columns="symbol", values="ret_1d")
    index_ret = rp.mean(axis=1)                                   # EW, rebalanced daily
    eq = (1.0 + rp.fillna(0.0)).cumprod()
    bh_equity = eq.mean(axis=1)                                   # EW buy & hold (drift)
    bh_ret = bh_equity.pct_change()
    bh_ret.iloc[0] = bh_equity.iloc[0] - 1.0
    return {"index": index_ret, "buy_hold": bh_ret}


# --------------------------------------------------------------------------- #
# stats
# --------------------------------------------------------------------------- #
@dataclass
class PortStats:
    total_return: float
    sharpe: float
    max_drawdown: float
    ann_return: float
    n_days: int


def stats(net: pd.Series, periods_per_year: int = 252) -> PortStats:
    net = net.dropna()
    n = len(net)
    if n == 0:
        return PortStats(0.0, 0.0, 0.0, 0.0, 0)
    equity = (1.0 + net).cumprod()
    final = float(equity.iloc[-1])
    total = final - 1.0
    std = float(net.std())
    sharpe = float(net.mean() / std * np.sqrt(periods_per_year)) if std > 0 else 0.0
    dd = float((equity / equity.cummax() - 1.0).min())
    ann = (final ** (periods_per_year / n) - 1.0) if final > 0 else float("nan")
    return PortStats(total, sharpe, dd, ann, n)
