"""Rigorous validation for the classic cross-sectional momentum edge.

Single-pass backtests overstate edges (best-of-sweep bias, regime luck). This adds:

* **walk_forward** — rolling train/test: pick the lookback on each TRAIN window,
  apply it on the next UNSEEN test window, roll, concatenate the out-of-sample
  returns. The OOS result has no best-of-sweep bias.
* **robustness** — vary universe size / quantile / rebalance / skip-recent / cost
  one axis at a time; is the edge stable or does it hinge on one setting?
* **by_year / by_regime** — per-calendar-year and per bull/quiet/bear sub-periods,
  to see if it's broad or one bull run plus unwinds.

Pure pandas/numpy (reuses tagent.xs_momentum, tagent.report, tagent.funding_portfolio);
unit-tested on synthetic panels. quantstats/Monte-Carlo risk is driven from
scripts/validate_xs_momentum.py via tagent.report.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from tagent.funding_portfolio import classify_regime
from tagent.report import perf_stats
from tagent.xs_momentum import XSMomConfig, backtest


def _nets_by_lookback(per_coin, lookbacks, cfg, ppy, membership=None):
    """net return Series per candidate lookback (each no-lookahead, shared index)."""
    return {lb: backtest(per_coin, replace(cfg, lookback=int(lb)), ppy, membership=membership)["net"]
            for lb in lookbacks}


def walk_forward_signal(per_coin: Dict[str, pd.DataFrame], signal,
                        top_qs=(0.1, 0.2, 0.3), train_bars: int = 756, test_bars: int = 252,
                        cfg: Optional[XSMomConfig] = None, periods_per_year: int = 252,
                        select: str = "sharpe", membership=None) -> dict:
    """Walk-forward for an injected cross-sectional factor ``signal`` (value/quality).

    A factor has no lookback to tune, so the swept hyper-parameter is the long
    quantile ``top_q``: on each train window pick the ``top_q`` with the best
    in-sample ``select`` metric, realise the next unseen test window with it, roll,
    concatenate OOS. No lookahead — the choice uses only train performance and the
    signal is itself as-of (no-lookahead)."""
    cfg = cfg or XSMomConfig()
    nets = {q: backtest(per_coin, replace(cfg, top_q=q), periods_per_year,
                        membership=membership, signal=signal)["net"] for q in top_qs}
    dates = list(nets[list(top_qs)[0]].index)
    n = len(dates)
    folds: List[dict] = []
    oos_parts = []
    start = train_bars
    while start < n:
        tr = dates[max(0, start - train_bars):start]
        te = dates[start:start + test_bars]
        if not te:
            break
        scored = {q: perf_stats(nets[q].reindex(tr).dropna(), periods_per_year)[select]
                  for q in top_qs}
        best = max(top_qs, key=lambda q: (scored[q] if np.isfinite(scored[q]) else -1e9))
        oos_parts.append(nets[best].reindex(te))
        folds.append({"train_end": str(tr[-1]), "test_start": str(te[0]),
                      "test_end": str(te[-1]), "chosen_top_q": float(best),
                      "train_sharpe": round(float(scored[best]), 3)})
        start += test_bars
    oos = pd.concat(oos_parts) if oos_parts else pd.Series(dtype=float)
    return {"oos": oos, "folds": folds, "n_folds": len(folds),
            "stats": perf_stats(oos, periods_per_year),
            "top_qs_used": sorted({f["chosen_top_q"] for f in folds})}


def walk_forward(per_coin: Dict[str, pd.DataFrame], lookbacks=(30, 60, 90),
                 train_bars: int = 365, test_bars: int = 90,
                 cfg: Optional[XSMomConfig] = None, periods_per_year: int = 365,
                 select: str = "sharpe", membership=None) -> dict:
    """Rolling walk-forward. On each train window pick the lookback with the best
    in-sample ``select`` metric, then realise the NEXT (unseen) test window with it;
    roll by ``test_bars``. Returns the combined OOS return series + per-fold log.

    No lookahead: the chosen lookback uses only train-window performance, and each
    lookback's returns are themselves built from past-only weights.
    """
    cfg = cfg or XSMomConfig()
    nets = _nets_by_lookback(per_coin, lookbacks, cfg, periods_per_year, membership)
    dates = list(nets[list(lookbacks)[0]].index)
    n = len(dates)
    folds: List[dict] = []
    oos_parts = []
    start = train_bars
    while start + 1 <= n and start < n:                       # need >=1 unseen bar
        tr = dates[max(0, start - train_bars):start]
        te = dates[start:start + test_bars]
        if not te:
            break
        scored = {lb: perf_stats(nets[lb].reindex(tr).dropna(), periods_per_year)[select]
                  for lb in lookbacks}
        best = max(lookbacks, key=lambda lb: (scored[lb] if np.isfinite(scored[lb]) else -1e9))
        seg = nets[best].reindex(te)
        oos_parts.append(seg)
        folds.append({"train_start": str(tr[0]), "train_end": str(tr[-1]),
                      "test_start": str(te[0]), "test_end": str(te[-1]),
                      "chosen_lookback": int(best), "train_sharpe": round(float(scored[best]), 3),
                      "test_return": round(float(seg.sum()), 4)})
        start += test_bars
    oos = pd.concat(oos_parts) if oos_parts else pd.Series(dtype=float)
    return {"oos": oos, "folds": folds, "n_folds": len(folds),
            "stats": perf_stats(oos, periods_per_year),
            "lookbacks_used": sorted({f["chosen_lookback"] for f in folds})}


# --------------------------------------------------------------------------- #
# robustness sweeps
# --------------------------------------------------------------------------- #
def _row(per_coin, cfg, ppy, axis, value, membership=None):
    r = backtest(per_coin, cfg, ppy, membership=membership)
    st = r["stats"]
    return {"axis": axis, "value": value, "sharpe": round(st["sharpe"], 2),
            "ann_return": round(st["cagr"], 4), "max_drawdown": round(st["max_drawdown"], 4),
            "total_return": round(st["total_return"], 4), "avg_turnover": round(r["avg_turnover"], 3),
            "n_bars": r["n_bars"]}


def robustness(per_coin: Dict[str, pd.DataFrame], cfg: Optional[XSMomConfig] = None,
               periods_per_year: int = 365, universes=(10, 20, 30), quantiles=(0.2, 0.3),
               rebalances=(7, 14, 30), skips=(0, 7, 14), costs=(5.0, 10.0, 20.0),
               membership=None) -> Dict[str, list]:
    """Vary each setting one axis at a time from ``cfg`` and report stability.

    Universe size takes the first N coins (sorted), so 'is it BTC/ETH-only or broad?'.
    When a point-in-time ``membership`` panel is given it is carried through every
    sweep (and subset to the same names on the universe axis).
    """
    cfg = cfg or XSMomConfig()
    coins = sorted(per_coin)
    out: Dict[str, list] = {"universe": [], "quantile": [], "rebalance": [],
                            "skip_recent": [], "cost_bps": []}
    for u in universes:
        sub = {c: per_coin[c] for c in coins[:u]} if u <= len(coins) else per_coin
        sub_m = (membership[[c for c in membership.columns if c in sub]]
                 if membership is not None else None)
        if len(sub) >= 4:
            out["universe"].append(_row(sub, cfg, periods_per_year, "universe", len(sub), sub_m))
    for q in quantiles:
        out["quantile"].append(_row(per_coin, replace(cfg, top_q=q), periods_per_year, "quantile", q, membership))
    for rb in rebalances:
        out["rebalance"].append(_row(per_coin, replace(cfg, rebalance=rb), periods_per_year, "rebalance", rb, membership))
    for sk in skips:
        out["skip_recent"].append(_row(per_coin, replace(cfg, skip_recent=sk), periods_per_year, "skip_recent", sk, membership))
    for c in costs:
        out["cost_bps"].append(_row(per_coin, replace(cfg, cost_bps=c, slippage_bps=0.0),
                                    periods_per_year, "cost_bps", c, membership))
    return out


# --------------------------------------------------------------------------- #
# sub-period: per year + per regime
# --------------------------------------------------------------------------- #
def by_year(net: pd.Series, periods_per_year: int = 365) -> Dict[str, dict]:
    """Per-calendar-year stats of a return series."""
    net = pd.Series(net).dropna()
    if net.empty or not isinstance(net.index, pd.DatetimeIndex):
        return {}
    out = {}
    for yr, seg in net.groupby(net.index.year):
        st = perf_stats(seg, periods_per_year)
        out[str(int(yr))] = {"n": st["n"], "total_return": round(st["total_return"], 4),
                             "sharpe": round(st["sharpe"], 2),
                             "max_drawdown": round(st["max_drawdown"], 4)}
    return out


def by_regime(net: pd.Series, benchmark_nav: pd.Series, periods_per_year: int = 365,
              window: int = 30) -> Dict[str, dict]:
    """Per bull/quiet/bear regime (regime from the basket NAV trend). Aligns the
    strategy returns to the regime label of each bar."""
    net = pd.Series(net).dropna()
    nav = pd.Series(benchmark_nav).reindex(net.index).ffill()
    reg = classify_regime(nav, window=window, periods_per_year=periods_per_year)
    out = {}
    for name in ("bull", "quiet", "bear"):
        seg = net[reg.reindex(net.index) == name].dropna()
        st = perf_stats(seg, periods_per_year)
        out[name] = {"n": st["n"], "total_return": round(st["total_return"], 4),
                     "sharpe": round(st["sharpe"], 2),
                     "ann_return": round(st["cagr"], 4) if st["n"] else 0.0}
    return out
