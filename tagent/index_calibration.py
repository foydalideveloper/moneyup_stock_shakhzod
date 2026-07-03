"""Finalize the momentum/index analysis — long-history calibration + satellite alpha.

PRIORITY 1 — Calibrate the deployable product (200d-MA trend-filtered KOSPI-200) on
the LONGEST available index history (1990->) so the honest expectation includes the
1997 IMF crash, 2000-02 dot-com, 2008 GFC, and the 2011-2016 "Boxpi" range years
(trend-following's known failure mode). Reported BY DECADE and per crisis/Boxpi
sub-period, net of a continuous futures roll friction (~10-15 bps/yr).

PRIORITY 2 — Is there a real RANKING alpha (satellite sleeve), or just a semi
overweight? A cap-weighted PIT basket sanity cell, ONE pre-registered sector-capped
momentum cell (no sector > 30%), a spanning regression of filtered-momentum on the
filtered-EW-basket (intercept = ranking alpha, Newey-West t + CI), and an itemization
of how much of the book-vs-index gap is pure instrument cost (cash 0.20% vs futures
0.05%) rather than the EW-vs-CW + ranking signal.

Pure numpy/pandas, no-lookahead (regime lagged one bar, momentum uses only past bars);
unit-tested on synthetic panels. Real runs are driven by scripts/run_index_calibration.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from tagent.config import DATA_DIR
from tagent.decomposition import (
    CASH_ROUND_TRIP, FUT_ROUND_TRIP, PPY, _one_way, _series_stats,
)
from tagent.risk_managed import RiskOverlayConfig, regime_exposure
from tagent.xs_momentum import _apply_membership, align_close, momentum_signal

# Named crisis / range sub-periods (trend-following's stress tests).
DEFAULT_SUBPERIODS: Dict[str, Tuple[str, str]] = {
    "IMF 1997-98": ("1997-07-01", "1998-12-31"),
    "Dotcom 2000-02": ("2000-01-01", "2002-12-31"),
    "GFC 2008": ("2008-01-01", "2008-12-31"),
    "Boxpi 2011-16": ("2011-01-01", "2016-12-31"),
    "COVID 2020": ("2020-01-01", "2020-12-31"),
    "AI rally 2025-26": ("2025-01-01", "2026-12-31"),
}
ROLL_FRICTION_ANNUAL = 0.0012        # ~12 bps/yr futures roll/financing (mid of 10-15)


# --------------------------------------------------------------------------- #
# PRIORITY 1 — long-history filtered index
# --------------------------------------------------------------------------- #
def load_index_close(name: str = "kospi200_long", data_dir=None) -> pd.Series:
    base = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
    p = base / f"{name}_1d.csv"
    if not p.exists():
        return pd.Series(dtype=float)
    df = pd.read_csv(p, index_col="timestamp", parse_dates=True).sort_index()
    return df["close"].astype(float)


def filtered_index_net(close, regime_ma: int = 200, roll_annual: float = ROLL_FRICTION_ANNUAL,
                       switch_cost: Optional[float] = None, ppy: int = PPY) -> pd.Series:
    """200d-MA trend-filtered index daily NET forward return.

    Exposure is 1.0 when the (lagged) index is at/above its ``regime_ma`` MA, else 0
    (cash) — strictly no-lookahead. Costs: ``switch_cost`` (one-way futures cost,
    default half of FUT_ROUND_TRIP) on each regime switch, plus a continuous
    ``roll_annual`` roll/financing friction prorated per trading day while exposed.
    """
    switch_cost = _one_way(FUT_ROUND_TRIP) if switch_cost is None else switch_cost
    close = pd.Series(close, dtype=float).dropna()
    fwd = close.pct_change(fill_method=None).shift(-1)
    ix = fwd.dropna().index
    fwd = fwd.reindex(ix)
    exp = regime_exposure(close, RiskOverlayConfig(regime_ma=regime_ma, regime_off=0.0))
    exp = exp.reindex(ix).fillna(1.0)
    switch = exp.diff()
    if len(switch):
        switch.iloc[0] = exp.iloc[0]
    roll = exp * (roll_annual / ppy)                       # prorated roll while in-market
    return exp * fwd - switch.abs() * switch_cost - roll


def raw_index_net(close) -> pd.Series:
    """Buy-and-hold index forward return (no filter) — the benchmark the filter is
    judged against (what the trend filter gives up in range years)."""
    close = pd.Series(close, dtype=float).dropna()
    fwd = close.pct_change(fill_method=None).shift(-1)
    return fwd.dropna()


def by_decade(net, ppy: int = PPY) -> Dict[str, dict]:
    """Per-decade {cagr, sharpe, maxDD, ...} of a daily return series."""
    net = pd.Series(net).dropna()
    out: Dict[str, dict] = {}
    if net.empty or not isinstance(net.index, pd.DatetimeIndex):
        return out
    for dec, seg in net.groupby((net.index.year // 10) * 10):
        out[f"{int(dec)}s"] = _series_stats(seg, ppy)
    return out


def sub_periods(net, periods: Optional[Dict[str, Tuple[str, str]]] = None,
                ppy: int = PPY) -> Dict[str, dict]:
    """Stats over named [start, end] sub-periods (crisis / Boxpi range years)."""
    periods = periods or DEFAULT_SUBPERIODS
    net = pd.Series(net).dropna()
    out: Dict[str, dict] = {}
    for name, (a, b) in periods.items():
        seg = net.loc[(net.index >= pd.Timestamp(a)) & (net.index <= pd.Timestamp(b))]
        out[name] = _series_stats(seg, ppy)
    return out


# --------------------------------------------------------------------------- #
# PRIORITY 2 — loaders + ranking-alpha satellite tests
# --------------------------------------------------------------------------- #
def load_shares(data_dir=None) -> Dict[str, float]:
    base = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
    p = base / "kr_shares.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p, dtype={"ticker": str})
    return {str(t).zfill(6): float(s) for t, s in zip(df["ticker"], df["shares"])}


def load_sectors(data_dir=None) -> Dict[str, str]:
    base = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
    p = base / "kr_sectors.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p, dtype={"ticker": str})
    return {str(t).zfill(6): str(s) for t, s in zip(df["ticker"], df["sector"])}


def _fwd(close: pd.DataFrame) -> pd.DataFrame:
    return close.pct_change(fill_method=None).shift(-1)


def _book_net(close: pd.DataFrame, W: pd.DataFrame, one_way: float) -> Tuple[pd.Series, pd.Series]:
    """Net forward return + turnover of a weight book (mirrors the engine's accounting)."""
    fwd = _fwd(close)
    Wn = W.to_numpy(float)
    Fn = np.nan_to_num(fwd.to_numpy(float), nan=0.0)
    gross = pd.Series((Wn * Fn).sum(axis=1), index=close.index)
    turn = pd.Series(np.abs(np.diff(Wn, axis=0, prepend=np.zeros((1, Wn.shape[1])))).sum(axis=1),
                     index=close.index)
    net = gross - turn * one_way
    valid = fwd.notna().any(axis=1)
    return net[valid], turn[valid]


def cap_weighted_basket(panel, membership, shares: Dict[str, float]) -> pd.Series:
    """Cap-weighted PIT-member basket forward return (weights = shares_i x close_i(t),
    renormalised over current members). A cap-weighted basket floats with price, so it
    needs no drift-rebalancing — this is the index sanity cell (compare to the index)."""
    close = align_close(panel)
    sh = pd.Series({c: float(shares.get(c, np.nan)) for c in close.columns})
    cap = _apply_membership(close.mul(sh, axis=1), membership)
    w = cap.div(cap.sum(axis=1), axis=0).fillna(0.0)       # non-members -> 0 weight (not NaN)
    fwd = _fwd(close)
    gross = pd.Series((w.to_numpy(float) * np.nan_to_num(fwd.to_numpy(float), nan=0.0)).sum(axis=1),
                      index=close.index)
    valid = fwd.notna().any(axis=1)
    return gross[valid]


def momentum_book_weights(close: pd.DataFrame, membership, sectors: Optional[Dict[str, str]] = None,
                          top_q: float = 0.2, rebalance: int = 21, skip: int = 21,
                          lookback: int = 252, sector_cap: Optional[float] = None) -> pd.DataFrame:
    """Long-only top-quantile 12-1 momentum weights, equal-weight within the book,
    monthly rebalance. If ``sector_cap`` is set (e.g. 0.30), a pre-registered cap limits
    each sector to at most ``floor(sector_cap * book_size)`` names — under equal weights
    that caps each sector's book WEIGHT at ``sector_cap`` (drops the weakest-momentum
    names in an over-cap sector, back-fills with the next-best other-sector names)."""
    mom = _apply_membership(momentum_signal(close, lookback, skip), membership)
    names = list(close.columns)
    sec = {n: (sectors or {}).get(n, "_") for n in names}
    W = pd.DataFrame(0.0, index=close.index, columns=names)
    prev = pd.Series(0.0, index=names)
    for ti, t in enumerate(close.index):
        if ti % rebalance == 0:
            row = mom.loc[t].dropna()
            if len(row) >= 5:
                k = max(1, int(round(top_q * len(row))))
                ranked = row.sort_values(ascending=False)
                if sector_cap is not None:
                    cap_n = max(1, int(np.floor(sector_cap * k)))
                    chosen, scount = [], {}
                    for nm in ranked.index:
                        s = sec[nm]
                        if scount.get(s, 0) >= cap_n:
                            continue
                        chosen.append(nm)
                        scount[s] = scount.get(s, 0) + 1
                        if len(chosen) >= k:
                            break
                else:
                    chosen = list(ranked.index[:k])
                w = pd.Series(0.0, index=names)
                if chosen:
                    w[chosen] = 1.0 / len(chosen)
                prev = w
        W.iloc[ti] = prev.to_numpy()
    return W


def sector_exposure(W: pd.DataFrame, sectors: Dict[str, str]) -> pd.Series:
    """Max single-sector weight per rebalance date (to verify the cap holds)."""
    sec = pd.Series({c: sectors.get(c, "_") for c in W.columns})
    by_sec = W.T.groupby(sec).sum().T
    return by_sec.max(axis=1)


def spanning_regression(book_net, basket_net, nw_lag: int = 6,
                        ppy_months: int = 12) -> dict:
    """Regress book MONTHLY returns on EW-basket monthly returns. Intercept = ranking
    alpha; reported with a Newey-West (HAC) t-stat and a 95% CI (monthly + annualised).
    Real satellite alpha requires alpha>0 with the CI excluding (or nearly excluding) 0."""
    bm = (1.0 + pd.Series(book_net).dropna()).resample("ME").prod() - 1.0
    am = (1.0 + pd.Series(basket_net).dropna()).resample("ME").prod() - 1.0
    df = pd.concat([bm, am], axis=1, keys=["book", "basket"]).dropna()
    n = len(df)
    if n < 12:
        return {"n_months": n, "alpha_m": 0.0, "alpha_ann": 0.0, "beta": 0.0,
                "t_alpha": 0.0, "se_m": 0.0, "ci_m": (0.0, 0.0), "ci_ann": (0.0, 0.0)}
    y = df["book"].to_numpy()
    x = df["basket"].to_numpy()
    X = np.column_stack([np.ones(n), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    XtX_inv = np.linalg.inv(X.T @ X)
    S = X * resid[:, None]
    meat = S.T @ S
    for l in range(1, min(nw_lag, n - 1) + 1):
        w = 1.0 - l / (nw_lag + 1.0)
        G = S[l:].T @ S[:-l]
        meat += w * (G + G.T)
    cov = XtX_inv @ meat @ XtX_inv
    se = float(np.sqrt(np.diag(cov))[0])
    alpha = float(b[0])
    t_alpha = alpha / se if se > 0 else 0.0
    ci_m = (alpha - 1.96 * se, alpha + 1.96 * se)
    return {"n_months": n, "alpha_m": alpha, "alpha_ann": alpha * ppy_months,
            "beta": float(b[1]), "t_alpha": t_alpha, "se_m": se,
            "ci_m": ci_m, "ci_ann": (ci_m[0] * ppy_months, ci_m[1] * ppy_months)}


def turnover_cost_itemization(turnover, n_years: float, cash_rt: float = CASH_ROUND_TRIP,
                              fut_rt: float = FUT_ROUND_TRIP) -> dict:
    """Annual turnover + the cash-vs-futures cost differential (the instrument-cost share
    of the book-vs-index gap). ``turnover`` is the engine's per-bar sum|dW| (two-way)."""
    two_way = float(pd.Series(turnover).sum()) / max(n_years, 1e-9)
    cash_drag = two_way * _one_way(cash_rt)
    fut_drag = two_way * _one_way(fut_rt)
    return {"turnover_twoway_annual": two_way, "turnover_oneway_annual": two_way / 2.0,
            "cash_cost_annual": cash_drag, "fut_cost_annual": fut_drag,
            "instrument_cost_diff_annual": cash_drag - fut_drag}
