"""Cross-sectional funding-carry PORTFOLIO: optimise the one real edge.

Builds on :mod:`tagent.funding_study` (single-coin carry) with the pieces a real
deployment needs:

* **Cross-sectional selection** — each 8h interval, rank the universe by recent
  funding and allocate to the richest positive-funding coins (equal-weight,
  funding-weighted, or top-N), with a per-coin cap and hysteresis so we only hold
  when funding clears costs and don't churn.
* **Cost / execution** — maker-vs-taker fees, post-only (no spread paid), and a
  rebalance-only-on-change rule so turnover (the carry killer) is minimised.
* **Risk** — isolated-margin liquidation modelling for the short-perp leg
  (margin ratio, funding spikes + sharp rallies) with a max-leverage knob and a
  yield-vs-tail-risk sweep.
* **Honest reporting** — cycle-average net yield split by bull/quiet/bear regime,
  Sharpe, maxDD, worst tails, and a simple capacity curve.

Everything is pure numpy/pandas and unit-tested on synthetic panels (no network).
Decisions use PAST funding only (lagged) — no lookahead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from tagent.funding_study import INTERVALS_PER_YEAR, carry_stats


# --------------------------------------------------------------------------- #
# cost / execution model
# --------------------------------------------------------------------------- #
@dataclass
class CostModel:
    """Trading cost per unit of |weight change| (turnover), both legs.

    Post-only (maker) pays the maker fee on each leg and crosses no spread; a
    taker pays the taker fee plus the spread. Slippage is added on top.
    """
    maker_fee_bps: float = 1.0
    taker_fee_bps: float = 4.5
    spread_bps: float = 1.0
    slippage_bps: float = 0.5
    post_only: bool = True

    def turnover_cost_bps(self) -> float:
        leg_fee = self.maker_fee_bps if self.post_only else self.taker_fee_bps
        spread = 0.0 if self.post_only else self.spread_bps
        return 2.0 * leg_fee + spread + self.slippage_bps     # two legs

    def frac(self) -> float:
        return self.turnover_cost_bps() / 1e4


# --------------------------------------------------------------------------- #
# panel alignment + signal
# --------------------------------------------------------------------------- #
def align_panel(per_coin: Dict[str, pd.DataFrame]):
    """-> (funding_wide, basis_wide) DataFrames [time x coin] on the union index.

    ``basis`` is the delta-neutral residual (spot_ret − perp_ret); missing coins
    (listed later) are NaN and simply not held until they have data.
    """
    funding = pd.DataFrame({c: df["funding_rate"].astype(float) for c, df in per_coin.items()})
    spot_ret = pd.DataFrame({c: df["spot_close"].astype(float).pct_change() for c, df in per_coin.items()})
    perp_ret = pd.DataFrame({c: df["perp"].astype(float).pct_change() for c, df in per_coin.items()})
    basis = (spot_ret - perp_ret)
    idx = funding.sort_index().index
    return funding.reindex(idx), basis.reindex(idx)


def _hysteresis_hold(signal: np.ndarray, enter: float, exit_: float) -> np.ndarray:
    """Per-coin hold mask: enter when signal >= enter, exit only below exit_."""
    held = np.zeros(len(signal), dtype=bool)
    state = False
    for i, s in enumerate(signal):
        if np.isfinite(s):
            if not state and s >= enter:
                state = True
            elif state and s < exit_:
                state = False
        held[i] = state
    return held


def _row_weights(sig: np.ndarray, held: np.ndarray, scheme: str,
                 top_n: int, max_weight: float) -> np.ndarray:
    """Target weights for one interval given the held set and a scheme."""
    idx = [i for i in np.where(held)[0] if np.isfinite(sig[i]) and sig[i] > 0]
    w = np.zeros(len(sig))
    if not idx:
        return w
    if scheme == "topN":
        idx = sorted(idx, key=lambda i: -sig[i])[:max(1, top_n)]
        raw = np.ones(len(idx))
    elif scheme == "funding":
        raw = np.array([sig[i] for i in idx], dtype=float)
    else:                                                   # equal
        raw = np.ones(len(idx))
    wv = raw / raw.sum()
    for _ in range(20):                                     # cap with redistribution
        over = wv > max_weight + 1e-12
        if not over.any():
            break
        excess = (wv[over] - max_weight).sum()
        wv[over] = max_weight
        under = ~over
        room = (max_weight - wv[under]).sum()
        if under.any() and room > 1e-12:
            wv[under] += excess * (max_weight - wv[under]) / room
        else:
            break                                          # fully capped -> partial deploy
    wv = np.minimum(wv, max_weight)
    for k, i in enumerate(idx):
        w[i] = wv[k]
    return w


def portfolio_returns(per_coin: Dict[str, pd.DataFrame], scheme: str = "funding",
                      hurdle_bps: float = 2.0, band_bps: float = 2.0, lookback: int = 3,
                      top_n: int = 5, max_weight: float = 0.25,
                      cost: Optional[CostModel] = None,
                      rebalance_on_change_only: bool = True) -> pd.DataFrame:
    """Per-interval cross-sectional carry returns (gross, cost, net, turnover).

    Weights are decided from the trailing-``lookback`` mean funding LAGGED one
    interval (no lookahead). Coins are held with hysteresis (enter at
    ``hurdle_bps``, exit only below ``hurdle_bps − band_bps``). With
    ``rebalance_on_change_only`` weights are frozen until the held set changes, so
    turnover (and its cost) only occurs on real entries/exits.
    """
    cost = cost or CostModel()
    funding, basis = align_panel(per_coin)
    coins = list(funding.columns)
    if not coins:
        return pd.DataFrame(columns=["gross", "cost", "net", "turnover", "n_held", "invested"])
    sig = funding.rolling(lookback, min_periods=1).mean().shift(1)
    enter, exit_ = hurdle_bps / 1e4, (hurdle_bps - band_bps) / 1e4
    held = np.column_stack([_hysteresis_hold(sig[c].to_numpy(float), enter, exit_) for c in coins])
    sigv = sig.to_numpy(float)
    fund = funding.to_numpy(float)
    bas = basis.to_numpy(float)
    bas = np.nan_to_num(bas, nan=0.0)
    fund0 = np.nan_to_num(fund, nan=0.0)
    T, C = fund.shape
    W = np.zeros((T, C))
    prev = np.zeros(C)
    prev_mask = np.zeros(C, dtype=bool)
    for t in range(T):
        mask = held[t]
        if rebalance_on_change_only and t > 0 and np.array_equal(mask, prev_mask):
            W[t] = prev
        else:
            W[t] = _row_weights(sigv[t], mask, scheme, top_n, max_weight)
        prev, prev_mask = W[t], mask
    gross = (W * (fund0 + bas)).sum(axis=1)
    turnover = np.abs(np.diff(W, axis=0, prepend=np.zeros((1, C)))).sum(axis=1)
    cst = turnover * cost.frac()
    out = pd.DataFrame({
        "gross": gross, "cost": cst, "net": gross - cst,
        "turnover": turnover, "n_held": held.sum(axis=1),
        "invested": W.sum(axis=1),
    }, index=funding.index)
    return out


def compare_schemes(per_coin: Dict[str, pd.DataFrame], schemes=("equal", "funding", "topN"),
                    **kw) -> Dict[str, dict]:
    """Run several allocation schemes and return their stats side by side."""
    out = {}
    for sch in schemes:
        r = portfolio_returns(per_coin, scheme=sch, **kw)
        st = carry_stats(r["net"], (r["invested"] > 0).astype(float))
        st["avg_turnover_bps"] = round(float(r["turnover"].mean()) * 1e4, 2)
        st["avg_n_held"] = round(float(r["n_held"].mean()), 2)
        st["total_cost"] = round(float(r["cost"].sum()), 6)
        out[sch] = st
    return out


# --------------------------------------------------------------------------- #
# risk: isolated-margin liquidation of the short-perp leg
# --------------------------------------------------------------------------- #
def margin_path(perp, leverage: float, maint_margin_rate: float = 0.005,
                funding=None, cross_margin: bool = False) -> dict:
    """Margin-ratio path of a SHORT-perp position (isolated by default).

    equity_ratio = 1/leverage + short_pnl_frac + cumulative_funding, where the
    short loses when price rises. Liquidation when the ratio falls to the
    maintenance margin rate. In ISOLATED margin the spot-leg gains don't help the
    perp account, so a sharp rally can liquidate the short before you can
    rebalance — that's the real tail. ``cross_margin=True`` adds the spot offset.
    """
    p = np.asarray(perp, dtype=float)
    if len(p) == 0:
        return {"min_ratio": float("nan"), "liquidated": False, "liq_index": -1,
                "leverage": leverage, "maint": maint_margin_rate}
    p0 = p[0]
    short_pnl = -(p / p0 - 1.0)
    fund = np.zeros(len(p)) if funding is None else np.asarray(funding, dtype=float)
    cum_fund = np.cumsum(np.nan_to_num(fund))
    ratio = 1.0 / leverage + short_pnl + cum_fund
    if cross_margin:
        ratio = ratio + (p / p0 - 1.0)                      # spot long offsets the short
    liq = ratio <= maint_margin_rate
    liq_index = int(np.argmax(liq)) if liq.any() else -1
    return {
        "ratio": pd.Series(ratio, index=getattr(perp, "index", None)),
        "min_ratio": float(ratio.min()),
        "liquidated": bool(liq.any()),
        "liq_index": liq_index,
        "leverage": float(leverage),
        "maint": float(maint_margin_rate),
    }


def stress_test(leverage: float, rally_pct: float, funding_spike_bps: float,
                n_intervals: int = 9, maint_margin_rate: float = 0.005,
                entry_price: float = 100.0) -> dict:
    """Stress the short leg: a linear rally of ``rally_pct`` over ``n_intervals``
    while funding spikes (``funding_spike_bps`` per 8h, negative = short pays)."""
    p = entry_price * (1.0 + np.linspace(0.0, rally_pct, n_intervals + 1))
    fund = np.full(len(p), funding_spike_bps / 1e4)
    mp = margin_path(p, leverage, maint_margin_rate, funding=fund)
    mp.pop("ratio", None)
    return {**mp, "rally_pct": rally_pct, "funding_spike_bps": funding_spike_bps,
            "n_intervals": n_intervals}


def leverage_tradeoff(funding_per_interval: float, leverages: List[float],
                      rally_pct: float = 0.30, funding_spike_bps: float = -30.0,
                      n_intervals: int = 9, maint_margin_rate: float = 0.005) -> List[dict]:
    """Yield-on-margin vs tail risk across leverages. Carry yield scales ~linearly
    with leverage; the stress scenario shows where liquidation kicks in."""
    rows = []
    for L in leverages:
        ann = funding_per_interval * INTERVALS_PER_YEAR * L
        st = stress_test(L, rally_pct, funding_spike_bps, n_intervals, maint_margin_rate)
        rows.append({
            "leverage": float(L),
            "ann_yield_on_margin": round(ann, 4),
            "worst_margin_ratio": round(st["min_ratio"], 4),
            "liquidated": st["liquidated"],
        })
    return rows


# --------------------------------------------------------------------------- #
# regime split + capacity
# --------------------------------------------------------------------------- #
def classify_regime(price, window: int = 90, up_ann: float = 0.5, dn_ann: float = -0.2,
                    periods_per_year: int = INTERVALS_PER_YEAR) -> pd.Series:
    """Label each interval bull / quiet / bear from a trailing annualised return
    of ``price`` (use BTC or the basket NAV). Funding pays far more in bull."""
    p = pd.Series(price).astype(float)
    roll = p.pct_change().rolling(window, min_periods=max(2, window // 3)).mean() * periods_per_year
    reg = pd.Series("quiet", index=p.index)
    reg[roll >= up_ann] = "bull"
    reg[roll <= dn_ann] = "bear"
    return reg


def yield_by_regime(net, regime, periods_per_year: int = INTERVALS_PER_YEAR) -> Dict[str, dict]:
    """Cycle-average net yield (annualised) within each regime."""
    net = pd.Series(net)
    regime = pd.Series(regime).reindex(net.index)
    out = {}
    for r in ("bull", "quiet", "bear"):
        seg = net[regime == r].dropna()
        out[r] = {
            "n_intervals": int(len(seg)),
            "ann_return": float(seg.mean() * periods_per_year) if len(seg) else 0.0,
            "mean_bps": round(float(seg.mean()) * 1e4, 3) if len(seg) else 0.0,
        }
    return out


def capacity_curve(gross_ann_yield: float, depth_usd: float, sizes_usd: List[float],
                   impact_coef: float = 0.5) -> dict:
    """Net yield vs deployed size: bigger size compresses funding / adds impact,
    modelled as a linear yield drag ``impact_coef * size / depth``. Capacity =
    the size at which net yield hits zero."""
    curve = [{"size_usd": float(s),
              "net_yield": gross_ann_yield - impact_coef * (s / depth_usd)}
             for s in sizes_usd]
    cap = depth_usd * gross_ann_yield / impact_coef if impact_coef > 0 else float("inf")
    return {"curve": curve, "capacity_usd": float(cap)}
