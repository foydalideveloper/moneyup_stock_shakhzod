"""Decisive decompositions: is the edge ALPHA or just BETA?

Two studies that separate a *skill* claim from a *market-exposure* claim:

PART 1 — Momentum: filter-beta vs ranking-alpha. The survivorship-corrected 12-1
book mixes two things: a market-timing trend filter (200d-MA regime) and a
cross-sectional stock RANKING. We benchmark the full book against two filter-only
controls that do NO ranking — a regime-filtered equal-weight basket and a
regime-filtered KOSPI-200 index (futures cost) — and measure how much the ranking
adds on top. If ranking adds < ~3%/yr over the filtered basket, the real product is
"trend-filtered index exposure (one mini-futures contract)," not a stock-picking edge.

PART 2 — PEAD: alpha vs beta. Re-run the surviving positive-reaction 20d drift on
BETA-ADJUSTED returns (event return minus pre-event beta x KOSPI-200 return over the
same window). If the net expectancy + t-stat survive, the drift is genuine, hedgeable
alpha; if they collapse it was market beta in disguise.

All causal/no-lookahead: regime exposure is lagged one bar (tagent.risk_managed),
event betas use only the pre-event window, and the held-window market return is the
realised index move over the same bars the drift is measured. Pure numpy/pandas;
unit-tested on synthetic panels (no network).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Optional

import numpy as np
import pandas as pd

from tagent.report import perf_stats
from tagent.risk_managed import RiskOverlayConfig, regime_exposure
from tagent.strategies.us_leadlag_daily import _wide
from tagent.xs_momentum import XSMomConfig, align_close, backtest

PPY = 252

# Corrected 2026 round-trip costs (fraction of notional).
CASH_ROUND_TRIP = 0.0020        # KR cash equity: ~0.15% tax + commission + slippage
FUT_ROUND_TRIP = 0.0005         # index / single-stock futures: no tax, thin spread


def _one_way(round_trip: float) -> float:
    """A full position round trip costs ``round_trip``; one side is half of it."""
    return round_trip / 2.0


def cost_bps_for(round_trip: float) -> float:
    """Convert a round-trip fraction to the engine's per-turnover-unit bps
    (``cost_bps + slippage_bps``): turnover 2.0 over a full in-and-out pays
    ``round_trip``."""
    return _one_way(round_trip) * 1e4


# --------------------------------------------------------------------------- #
# PART 1 — momentum: regime-timed series + the ranking-vs-filter decomposition
# --------------------------------------------------------------------------- #
def regime_timed(ret: pd.Series, nav: pd.Series, regime_ma: int = 200,
                 one_way_cost: float = 0.0) -> pd.Series:
    """Apply the 200d-MA regime gate to a FORWARD-indexed return series.

    ``ret[t]`` is the t->t+1 return; ``nav`` is the market level whose trend gates
    exposure (lagged one bar inside :func:`regime_exposure`, so the decision for
    ``ret[t]`` uses only data through t — no-lookahead). Exposure is 1.0 in an
    uptrend, 0.0 (cash) below the MA. ``one_way_cost`` is charged on each exposure
    switch (entering or exiting the whole position), including the initial entry.
    Returns the net (exposure-scaled, cost-charged) return series on ``ret``'s index.
    """
    ret = pd.Series(ret, dtype=float)
    cfg = RiskOverlayConfig(regime_ma=regime_ma, regime_off=0.0)
    exp = regime_exposure(nav, cfg).reindex(ret.index).fillna(1.0)
    switch = exp.diff()
    switch.iloc[0] = exp.iloc[0]                  # establishing the initial position
    cost = switch.abs() * one_way_cost
    return exp * ret - cost


def _series_stats(net: pd.Series, ppy: int = PPY) -> dict:
    st = perf_stats(pd.Series(net).dropna(), ppy)
    return {"cagr": st["cagr"], "sharpe": st["sharpe"], "max_drawdown": st["max_drawdown"],
            "total_return": st["total_return"], "ann_vol": st["ann_vol"], "n": st["n"]}


def by_year_returns(net: pd.Series, ppy: int = PPY) -> Dict[str, dict]:
    """Per-calendar-year {total_return, cagr, sharpe, maxDD} for a return series."""
    net = pd.Series(net).dropna()
    out: Dict[str, dict] = {}
    if net.empty or not isinstance(net.index, pd.DatetimeIndex):
        return out
    for yr, seg in net.groupby(net.index.year):
        out[str(int(yr))] = _series_stats(seg, ppy)
    return out


def _delta(c: dict, base: dict) -> dict:
    return {"cagr": c["cagr"] - base["cagr"], "sharpe": c["sharpe"] - base["sharpe"],
            "max_drawdown": c["max_drawdown"] - base["max_drawdown"]}


def momentum_decomposition(panel, membership, index_close: pd.Series,
                           cfg: Optional[XSMomConfig] = None, regime_ma: int = 200,
                           cash_round_trip: float = CASH_ROUND_TRIP,
                           fut_round_trip: float = FUT_ROUND_TRIP,
                           ppy: int = PPY, exclude_year: Optional[int] = 2025) -> dict:
    """Decompose the momentum book into filter-beta vs ranking-alpha.

    Returns net series + stats for three regime-filtered books on the SAME PIT
    universe and SAME 200d-MA rule, net of the corrected costs:
      a) equal-weight basket  (no ranking, cash cost),
      b) KOSPI-200 index      (200d-MA timing on the index alone, futures cost),
      c) full momentum book   (ranking + filter, cash cost),
    plus the ranking-alpha = (c) - (a) and (c) - (b), by-year, and an ex-``exclude_year``
    stress split. ``cfg`` defaults to classic 12-1 with the corrected cash cost.
    """
    cfg = cfg or XSMomConfig(lookback=252, skip_recent=21, rebalance=21, top_q=0.2,
                             hysteresis=0.1, allow_short=False)
    cfg = replace(cfg, cost_bps=cost_bps_for(cash_round_trip), slippage_bps=0.0)
    cash_ow, fut_ow = _one_way(cash_round_trip), _one_way(fut_round_trip)

    bt = backtest(panel, cfg, ppy, membership=membership)
    basket = bt["basket"]                                  # EW forward return over PIT members
    plain_net = bt["net"]                                  # ranking, net of cash ranking-turnover
    basket_nav = (1.0 + basket.fillna(0.0)).cumprod()

    full_idx = align_close(panel).index
    idx_px = pd.Series(index_close, dtype=float).reindex(full_idx).ffill()
    idx_fwd = idx_px.pct_change(fill_method=None).shift(-1).reindex(basket.index)  # t->t+1 on the book's index

    net_a = regime_timed(basket, basket_nav, regime_ma, cash_ow)            # filtered basket
    net_b = regime_timed(idx_fwd, idx_px, regime_ma, fut_ow)               # filtered index/futures
    net_c = regime_timed(plain_net, basket_nav, regime_ma, cash_ow)        # ranking + filter

    books = {"filtered_basket": net_a, "filtered_index": net_b, "momentum_book": net_c}
    stats = {k: _series_stats(v, ppy) for k, v in books.items()}
    by_year = {k: by_year_returns(v, ppy) for k, v in books.items()}

    ex = {}
    if exclude_year is not None:
        for k, v in books.items():
            v = pd.Series(v).dropna()
            ex[k] = _series_stats(v[v.index.year != exclude_year], ppy)

    return {
        "nets": books, "stats": stats, "by_year": by_year, "ex_year": ex,
        "exclude_year": exclude_year,
        "ranking_alpha_vs_basket": _delta(stats["momentum_book"], stats["filtered_basket"]),
        "ranking_alpha_vs_index": _delta(stats["momentum_book"], stats["filtered_index"]),
        "ranking_alpha_vs_basket_ex": (_delta(ex["momentum_book"], ex["filtered_basket"])
                                       if ex else None),
        "costs": {"cash_round_trip": cash_round_trip, "fut_round_trip": fut_round_trip},
    }


# --------------------------------------------------------------------------- #
# PART 2 — PEAD: beta-adjust the surviving drift
# --------------------------------------------------------------------------- #
def beta_of(stock_ret, mkt_ret, min_obs: int = 30) -> float:
    """OLS slope of ``stock_ret`` on ``mkt_ret`` (market beta). Pairs are aligned and
    dropna-ed; returns 0.0 with fewer than ``min_obs`` overlapping points or zero
    market variance."""
    s = pd.Series(stock_ret, dtype=float)
    m = pd.Series(mkt_ret, dtype=float)
    df = pd.concat([s, m], axis=1).dropna()
    if len(df) < min_obs:
        return 0.0
    x = df.iloc[:, 1].to_numpy()
    y = df.iloc[:, 0].to_numpy()
    var = float(((x - x.mean()) ** 2).sum())
    if var <= 0:
        return 0.0
    return float(((x - x.mean()) * (y - y.mean())).sum() / var)


def pead_beta_adjust(trades: pd.DataFrame, panel, index_close: pd.Series, hold: int,
                     beta_window: int = 120, min_obs: int = 30) -> pd.DataFrame:
    """Add ``beta``, ``mkt`` and ``abnormal`` columns to PEAD ``trades``.

    For each event: ``beta`` is the stock's pre-event beta vs the KOSPI-200 over the
    ``beta_window`` daily returns ending the bar BEFORE entry (strictly no-lookahead);
    ``mkt`` is the realised index return over the same held bars (entry close ->
    exit close, ``hold`` days); ``abnormal = ret - beta*mkt`` is the market-hedged
    drift. Trades whose window falls off the data are dropped.
    """
    closes = _wide(panel, "close").where(lambda x: x > 0)
    idx = closes.index
    idx_px = pd.Series(index_close, dtype=float).reindex(idx).ffill()
    idx_ret = idx_px.pct_change(fill_method=None)
    out = []
    for t in trades.itertuples():
        sym = t.symbol
        if sym not in closes.columns or t.entry not in idx:
            continue
        ei = idx.get_loc(t.entry)
        xi = ei + hold
        lo = ei - 1 - beta_window
        if lo < 0 or xi >= len(idx):
            continue
        sret = closes[sym].pct_change(fill_method=None)
        beta = beta_of(sret.iloc[lo:ei - 1], idx_ret.iloc[lo:ei - 1], min_obs=min_obs)
        mkt = float(idx_px.iloc[xi] / idx_px.iloc[ei] - 1.0)
        rec = t._asdict()
        rec.pop("Index", None)
        rec["beta"] = beta
        rec["mkt"] = mkt
        rec["abnormal"] = float(t.ret) - beta * mkt
        out.append(rec)
    cols = list(trades.columns) + ["beta", "mkt", "abnormal"]
    return pd.DataFrame(out, columns=cols)
