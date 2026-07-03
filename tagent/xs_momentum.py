"""Cross-sectional crypto momentum — rank coins AGAINST each other, long/short.

NOT per-coin prediction: each rebalance, rank the universe by trailing return,
go LONG the top quantile and SHORT the bottom (dollar-neutral, market-neutral),
hold over the next bar, pay costs on turnover, and use hysteresis to limit churn.
Backtest is strictly no-lookahead: the weight at bar t uses the trailing return
known AT t and earns the t->t+1 return.

Pure numpy/pandas; unit-tested on synthetic panels (no network). The runner
(scripts/run_xs_momentum.py) feeds real Binance klines and sweeps 1-7 day
lookbacks, comparing net-of-cost long-short vs simply holding the basket.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from tagent.report import perf_stats


@dataclass
class XSMomConfig:
    lookback: int = 3            # formation-period length in bars (e.g. 30/60/90 days)
    skip_recent: int = 0         # skip the most recent N bars (classic momentum: ~7 days)
    rebalance: int = 1           # recompute weights every N bars (1=daily, 7=weekly, 30=monthly)
    reverse: bool = False        # True = REVERSAL (long losers / short winners)
    top_q: float = 0.2           # top / bottom quantile traded
    allow_short: bool = True     # True = dollar-neutral long-short; False = long-only top
    cost_bps: float = 5.0        # fee per unit turnover
    slippage_bps: float = 2.0    # slippage per unit turnover
    hysteresis: float = 0.0      # widen the exit band by this (fraction of rank) to cut turnover


def align_close(per_coin: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """{coin: df with 'close'} -> wide close DataFrame [time x coin] on the union index."""
    cols = {c: pd.to_numeric(df["close"], errors="coerce") for c, df in per_coin.items()}
    wide = pd.DataFrame(cols).sort_index()
    return wide


def momentum_signal(close_wide: pd.DataFrame, lookback: int, skip_recent: int = 0) -> pd.DataFrame:
    """Formation-period return over ``lookback`` bars, optionally SKIPPING the most
    recent ``skip_recent`` bars (classic momentum skips ~1 week to avoid short-term
    reversal contamination). Value at t = ret over [t-skip-lookback, t-skip]; uses
    only past bars (no lookahead)."""
    recent = close_wide.shift(skip_recent)
    return recent / recent.shift(lookback) - 1.0


def _hold_masks(pct: np.ndarray, top_q: float, band: float):
    """Per-coin hysteresis on cross-sectional rank percentile.

    A coin ENTERS the long book when its percentile >= 1-top_q and only EXITS when
    it falls below 1-top_q-band; symmetric for the short book. ``band`` (the
    hysteresis) keeps borderline names in place to limit turnover.
    """
    T, C = pct.shape
    enter_l, exit_l = 1.0 - top_q, 1.0 - top_q - band
    enter_s, exit_s = top_q, top_q + band
    longm = np.zeros((T, C), dtype=bool)
    shortm = np.zeros((T, C), dtype=bool)
    lstate = np.zeros(C, dtype=bool)
    sstate = np.zeros(C, dtype=bool)
    for t in range(T):
        for c in range(C):
            p = pct[t, c]
            if not np.isfinite(p):
                lstate[c] = sstate[c] = False           # no rank this bar -> flat
            else:
                if not lstate[c] and p >= enter_l:
                    lstate[c] = True
                elif lstate[c] and p < exit_l:
                    lstate[c] = False
                if not sstate[c] and p <= enter_s:
                    sstate[c] = True
                elif sstate[c] and p > exit_s:
                    sstate[c] = False
                if lstate[c] and sstate[c]:              # mutually exclusive safety
                    sstate[c] = False
            longm[t, c] = lstate[c]
            shortm[t, c] = sstate[c]
    return longm, shortm


def _apply_membership(frame: pd.DataFrame, membership: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Mask ``frame`` to NaN wherever ``membership`` is not True (point-in-time
    eligibility). ``membership`` is a [time x name] boolean panel; it is reindexed
    onto ``frame`` and missing cells are treated as not-a-member. Returns ``frame``
    unchanged when ``membership`` is None (the default — no behaviour change)."""
    if membership is None:
        return frame
    m = membership.reindex(index=frame.index, columns=frame.columns).fillna(False)
    return frame.where(m.to_numpy(dtype=bool))


def _weights(close_wide: pd.DataFrame, cfg: XSMomConfig,
             membership: Optional[pd.DataFrame] = None,
             signal: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Per-bar target weights [time x coin]: +1/nL across the long book, -1/nS
    across the short book (each leg normalised -> dollar-neutral when shorting).

    ``reverse`` flips the sign of the ranking (long losers / short winners).
    ``rebalance`` > 1 holds weights between rebalances (turnover only on rebalance
    bars), which is what makes low-frequency momentum cheap to run.

    ``membership`` (optional point-in-time mask) makes a name ELIGIBLE to be ranked
    only on bars where it was a universe member; non-members get NaN momentum and so
    are never selected. Formation still uses the name's real trailing prices — only
    selection eligibility is gated, which is what corrects survivorship bias.

    ``signal`` (optional [time x name] panel) RANKS on an injected cross-sectional
    factor (higher = more attractive) instead of trailing-return momentum — so value
    / quality / multi-factor books reuse this exact engine. Default None = momentum.
    """
    if signal is not None:
        mom = signal.reindex(index=close_wide.index, columns=close_wide.columns)
    else:
        mom = momentum_signal(close_wide, cfg.lookback, cfg.skip_recent)
    if cfg.reverse:
        mom = -mom                                       # rank losers highest -> long them
    mom = _apply_membership(mom, membership)             # point-in-time eligibility
    pct = mom.rank(axis=1, pct=True)                     # 1.0 = strongest signal
    longm, shortm = _hold_masks(pct.to_numpy(float), cfg.top_q, cfg.hysteresis)
    T, C = longm.shape
    target = np.zeros((T, C))
    for t in range(T):
        nL = longm[t].sum()
        if nL:
            target[t, longm[t]] += 1.0 / nL
        if cfg.allow_short:
            nS = shortm[t].sum()
            if nS:
                target[t, shortm[t]] -= 1.0 / nS
    if cfg.rebalance and cfg.rebalance > 1:              # hold between rebalance bars
        W = np.zeros((T, C))
        prev = np.zeros(C)
        for t in range(T):
            if t % cfg.rebalance == 0:
                prev = target[t]
            W[t] = prev
    else:
        W = target
    return pd.DataFrame(W, index=close_wide.index, columns=close_wide.columns)


def backtest(per_coin: Dict[str, pd.DataFrame], cfg: Optional[XSMomConfig] = None,
             periods_per_year: int = 365,
             membership: Optional[pd.DataFrame] = None,
             signal: Optional[pd.DataFrame] = None) -> dict:
    """Net-of-cost cross-sectional momentum backtest vs holding the basket.

    Returns net/gross/turnover/weights Series + perf stats for the strategy and an
    equal-weight basket benchmark over the same bars.

    ``membership`` (optional, default None) is a point-in-time [time x name] boolean
    eligibility panel: only members are ranked/selected, and the equal-weight basket
    benchmark is taken over members too — so both the strategy and its benchmark see
    the survivorship-corrected universe as it was at each bar.
    """
    cfg = cfg or XSMomConfig()
    close = align_close(per_coin)
    min_bars = 3 if signal is not None else cfg.lookback + 3   # signal has its own warmup
    if close.shape[0] < min_bars or close.shape[1] < 2:
        empty = pd.Series(dtype=float)
        return {"net": empty, "gross": empty, "turnover": empty, "basket": empty,
                "stats": perf_stats([], periods_per_year),
                "basket_stats": perf_stats([], periods_per_year),
                "avg_turnover": 0.0, "n_bars": 0, "lookback": cfg.lookback}

    W = _weights(close, cfg, membership, signal)
    fwd = close.pct_change().shift(-1)                   # fwd[t] = return from t -> t+1
    Wn = W.to_numpy(float)
    Fn = np.nan_to_num(fwd.to_numpy(float), nan=0.0)
    gross = pd.Series((Wn * Fn).sum(axis=1), index=close.index)
    turnover = pd.Series(np.abs(np.diff(Wn, axis=0, prepend=np.zeros((1, Wn.shape[1])))).sum(axis=1),
                         index=close.index)
    cost = turnover * (cfg.cost_bps + cfg.slippage_bps) / 1e4
    net = (gross - cost)
    # drop the final bar (no forward return) so stats aren't diluted by a 0
    valid = fwd.notna().any(axis=1)
    net, gross, turnover = net[valid], gross[valid], turnover[valid]
    basket_fwd = _apply_membership(fwd, membership)      # index = members only, when masked
    basket = basket_fwd.mean(axis=1)[valid]              # equal-weight basket (NaN-safe)
    return {
        "net": net, "gross": gross, "turnover": turnover, "weights": W, "basket": basket,
        "stats": perf_stats(net, periods_per_year),
        "basket_stats": perf_stats(basket, periods_per_year),
        "avg_turnover": float(turnover.mean()),
        "gross_after_cost_bps": float(net.sum() * 1e4),
        "n_bars": int(len(net)), "lookback": cfg.lookback,
    }


def sweep_lookbacks(per_coin: Dict[str, pd.DataFrame], lookbacks=range(1, 8),
                    cfg: Optional[XSMomConfig] = None, periods_per_year: int = 365) -> List[dict]:
    """Backtest each trailing-return lookback (1..7) and return a summary row each."""
    from dataclasses import replace
    cfg = cfg or XSMomConfig()
    rows = []
    for lb in lookbacks:
        r = backtest(per_coin, replace(cfg, lookback=int(lb)), periods_per_year)
        st = r["stats"]
        rows.append({
            "lookback": int(lb), "sharpe": round(st["sharpe"], 2),
            "ann_return": round(st["cagr"], 4), "max_drawdown": round(st["max_drawdown"], 4),
            "total_return": round(st["total_return"], 4),
            "avg_turnover": round(r["avg_turnover"], 3), "n_bars": r["n_bars"],
        })
    return rows
