"""Cash-and-carry simulation: long spot + short perp to harvest funding.

The trade is **delta-neutral**: long 1 unit of spot and short 1 unit of the
perpetual. Price moves on the two legs cancel (up to the small basis residual),
and every 8 hours the position **collects the funding rate** — positive funding
means perp longs pay shorts, so a short-perp earns it. This is a *structural
premium*, not a forecast.

Per 8h interval, while in the carry:
    gross = funding_rate            (received by the short when rate > 0)
          + (spot_ret − perp_ret)   (delta-neutral residual = basis P&L)
Entering or exiting pays round-trip costs on **both** legs (fees + spread).

We only carry when recent funding clears the cost hurdle — and the decision uses
**past** funding only (no lookahead). All pure pandas/numpy, unit-tested on
synthetic series.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

INTERVALS_PER_YEAR = 3 * 365          # funding paid every 8h


def carry_returns(df: pd.DataFrame, fee_bps: float = 7.0, spread_bps: float = 1.0,
                  min_funding_bps: float = 0.0, lookback: int = 3,
                  hysteresis_bps: float = 3.0) -> pd.DataFrame:
    """Per-interval net carry returns, with hysteresis to avoid churn.

    `df` has ``funding_rate`` (per-8h), ``perp``, ``spot_close``. `fee_bps` is
    per leg; a round-trip leg cost (open OR close) is ``2*fee + spread`` bps.

    We ENTER the carry when the trailing-`lookback` mean funding (lagged one step
    — past only, no lookahead) is >= `min_funding_bps`, and only EXIT when it
    falls below `min_funding_bps - hysteresis_bps` (sustained negative funding).
    The hysteresis band is what stops the strategy from churning in/out around
    zero funding and bleeding round-trip costs — funding carry must be HELD.
    """
    out = df.copy().sort_index()
    funding = out["funding_rate"].astype(float)
    spot_ret = out["spot_close"].astype(float).pct_change()
    perp_ret = out["perp"].astype(float).pct_change()
    basis = spot_ret - perp_ret                       # long spot − short perp price P&L

    # Decide with PAST funding only: trailing mean, lagged by one interval.
    signal = funding.rolling(lookback, min_periods=1).mean().shift(1).to_numpy()
    enter, exit_ = min_funding_bps / 1e4, (min_funding_bps - hysteresis_bps) / 1e4
    state, states = 0, np.zeros(len(signal), dtype=int)
    for i, sig in enumerate(signal):
        if np.isfinite(sig):
            if state == 0 and sig >= enter:
                state = 1
            elif state == 1 and sig < exit_:
                state = 0
        states[i] = state
    in_carry = pd.Series(states, index=out.index)

    gross = np.where(in_carry == 1, funding + basis.fillna(0.0), 0.0)

    leg_cost = (2.0 * fee_bps + spread_bps) / 1e4      # open or close (both legs + spread)
    turn = in_carry.diff().abs()
    turn.iloc[0] = in_carry.iloc[0]                    # initial entry counts
    cost = turn * leg_cost

    out["funding"] = funding
    out["basis"] = basis
    out["in_carry"] = in_carry
    out["gross"] = gross
    out["cost"] = cost
    out["net"] = out["gross"] - out["cost"]
    return out


def carry_stats(net: pd.Series, in_carry: Optional[pd.Series] = None,
                periods_per_year: int = INTERVALS_PER_YEAR) -> Dict[str, float]:
    net = pd.Series(net).dropna()
    n = len(net)
    if n == 0:
        return {"ann_return": 0.0, "ann_vol": 0.0, "sharpe": 0.0, "max_drawdown": 0.0,
                "total_return": 0.0, "n_intervals": 0, "in_carry_pct": 0.0}
    equity = (1.0 + net).cumprod()
    final = float(equity.iloc[-1])
    total = final - 1.0
    ann = (final ** (periods_per_year / n) - 1.0) if final > 0 else float("nan")
    std = float(net.std())
    vol = std * np.sqrt(periods_per_year)
    sharpe = float(net.mean() / std * np.sqrt(periods_per_year)) if std > 0 else 0.0
    mdd = float((equity / equity.cummax() - 1.0).min())
    carry_pct = float(np.mean(np.asarray(in_carry))) * 100.0 if in_carry is not None else float("nan")
    return {"ann_return": ann, "ann_vol": vol, "sharpe": sharpe, "max_drawdown": mdd,
            "total_return": total, "n_intervals": n, "in_carry_pct": carry_pct}


def funding_flips(df: pd.DataFrame, n: int = 5) -> Dict[str, object]:
    """Worst funding-flip events: most-negative rates and longest negative streak."""
    fr = df["funding_rate"].astype(float)
    neg = fr < 0
    worst = fr.nsmallest(n)
    # longest run of consecutive negative-funding intervals
    longest = cur = 0
    for v in neg:
        cur = cur + 1 if v else 0
        longest = max(longest, cur)
    return {
        "neg_pct": float(neg.mean()) * 100.0,
        "min_funding_bps": float(fr.min()) * 1e4,
        "longest_neg_streak": int(longest),
        "worst_events": [{"time": str(t), "funding_bps": round(float(v) * 1e4, 2)}
                         for t, v in worst.items()],
    }


def simulate(df: pd.DataFrame, **kw) -> Dict[str, object]:
    r = carry_returns(df, **kw)
    return {"returns": r, "stats": carry_stats(r["net"], r["in_carry"]),
            "flips": funding_flips(df)}


def basket(per_coin: Dict[str, pd.DataFrame]) -> Dict[str, object]:
    """Equal-weight basket of per-coin net return series (aligned by time)."""
    nets = pd.DataFrame({c: r["net"] for c, r in per_coin.items()}).dropna(how="all")
    if nets.empty:
        return {"net": pd.Series(dtype=float), "stats": carry_stats(pd.Series(dtype=float))}
    net = nets.mean(axis=1)
    carry = pd.DataFrame({c: r["in_carry"] for c, r in per_coin.items()}).reindex(net.index).mean(axis=1)
    return {"net": net, "stats": carry_stats(net, carry)}
