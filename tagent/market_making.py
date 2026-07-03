"""Offline market-making simulator — the one way the order-book signal could pay.

Instead of PAYING the spread to act on the (real but sub-cost) order-book signal,
a maker EARNS it: post a resting bid below mid and ask above mid, collect the
half-spread (+ maker rebate) when they fill — and use the depth-IMBALANCE signal
to SKEW quotes (tighter/more size on the side imbalance favours, wider on the
other) to lean into the predicted move and dodge the toxic side.

Model (simplified, snapshot-based — see the honest caveats in the runner):
* a resting level fills only when price TRADES THROUGH it (next mid crosses the
  quote) — so each entry fill is locally adverse, and profit comes from capturing
  the spread on the exit when price reverts;
* inventory accumulates per fill; inventory risk via a hard position LIMIT and a
  reservation-price SKEW-to-flatten;
* a maker fee/rebate model (negative fee = rebate);
* adverse selection measured explicitly (markout on just-filled size).

P&L is decomposed EXACTLY into  net = spread_capture + rebate + adverse_selection
+ inventory_carry  (the last two are usually negative). Pure numpy/pandas;
unit-tested on synthetic books (no network). Decisions use only info known at the
quote time — the future move is the fill TRIGGER, never an input to the quote.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

_SECONDS_PER_YEAR = 365.0 * 24 * 3600


@dataclass
class MMConfig:
    base_half_spread_bps: float = 1.5    # distance from mid to each quote
    quote_size: float = 1.0              # size posted per side
    fee_bps: float = -1.0                # maker fee; NEGATIVE = rebate received
    skew: float = 0.0                    # imbalance skew strength (0 = naive symmetric)
    inv_skew_bps: float = 0.5            # reservation-price shift per unit inventory
    max_inventory: float = 5.0           # hard position limit (abs)


def quote(mid: float, imbalance: float, inventory: float, cfg: MMConfig):
    """Resting (bid_px, ask_px, post_bid, post_ask) for one step.

    Uses ONLY current mid / imbalance / inventory (no lookahead). Imbalance > 0
    (bid-heavy, predicts up) tightens the bid and widens the ask so we lean long
    and avoid selling into the rise; inventory shifts the reservation price to
    flatten; the position limit stops quoting the side that would breach it.
    """
    base = cfg.base_half_spread_bps / 1e4 * mid
    sk = cfg.skew * float(np.clip(imbalance, -1.0, 1.0))
    r = mid - cfg.inv_skew_bps / 1e4 * mid * inventory      # reservation price
    hs_bid = max(base * (1.0 - sk), 0.0)                    # tighter bid when imb>0
    hs_ask = max(base * (1.0 + sk), 0.0)                    # wider ask when imb>0
    bid_px = r - hs_bid
    ask_px = r + hs_ask
    post_bid = inventory < cfg.max_inventory - 1e-12
    post_ask = inventory > -cfg.max_inventory + 1e-12
    return bid_px, ask_px, post_bid, post_ask


def _ann_factor(ts) -> float:
    try:
        t = pd.to_datetime(pd.Series(ts), utc=True).astype("int64").to_numpy() / 1e9
        span = float(t[-1] - t[0])
        n = len(t) - 1
        if span > 0 and n > 0:
            return math.sqrt(n / span * _SECONDS_PER_YEAR)
    except Exception:
        pass
    return math.sqrt(max(len(ts) - 1, 1))


def simulate_market_making(df: pd.DataFrame, cfg: Optional[MMConfig] = None) -> Dict:
    """Backtest the maker over a recorded book. Returns the P&L decomposition,
    Sharpe, fill count, and inventory stats. ``df`` needs ``mid`` (and optionally
    ``depth_imbalance``, ``ts``)."""
    cfg = cfg or MMConfig()
    d = df.dropna(subset=["mid"]).reset_index(drop=True)
    mid = pd.to_numeric(d["mid"], errors="coerce").to_numpy(float)
    imb = pd.to_numeric(d.get("depth_imbalance", 0.0), errors="coerce").fillna(0.0).to_numpy(float)
    n = len(mid)
    q = cfg.quote_size
    fee = cfg.fee_bps / 1e4

    inv = 0.0
    spread = rebate = adverse = carry = 0.0
    notional = 0.0
    fills = 0
    max_inv = 0.0
    pnl_steps = []
    for t in range(n - 1):
        m, m1 = mid[t], mid[t + 1]
        dm = m1 - m
        inv_pre = inv
        bid_px, ask_px, pb, pa = quote(m, imb[t], inv, cfg)
        s_t = r_t = newfill = 0.0
        if pb and m1 <= bid_px:                 # bid filled: bought q at bid_px
            s_t = (m - bid_px) * q
            r_t = -fee * bid_px * q
            newfill = q
            inv += q
            fills += 1
            notional += bid_px * q
        elif pa and m1 >= ask_px:               # ask filled: sold q at ask_px
            s_t = (ask_px - m) * q
            r_t = -fee * ask_px * q
            newfill = -q
            inv -= q
            fills += 1
            notional += ask_px * q
        a_t = newfill * dm                       # adverse selection (markout on new fill)
        c_t = inv_pre * dm                       # carry on inventory held into the step
        spread += s_t
        rebate += r_t
        adverse += a_t
        carry += c_t
        pnl_steps.append(s_t + r_t + a_t + c_t)  # = step equity change
        max_inv = max(max_inv, abs(inv))

    net = spread + rebate + adverse + carry
    pnl = np.asarray(pnl_steps, float)
    std = float(pnl.std())
    sharpe = float(pnl.mean() / std * _ann_factor(d.get("ts", range(n)))) if std > 0 else 0.0
    # normalise the $ P&L to bps of traded notional, so the numbers are interpretable
    bps = (lambda v: v / notional * 1e4) if notional > 0 else (lambda v: 0.0)
    return {
        # absolute ($, price x size units)
        "net": net, "spread_capture": spread, "rebate": rebate,
        "adverse_selection": adverse, "inventory_carry": carry,
        "notional_traded": notional,
        # normalised (bps of notional traded)
        "net_bps": bps(net), "spread_bps": bps(spread), "rebate_bps": bps(rebate),
        "adverse_bps": bps(adverse), "inventory_bps": bps(carry),
        # risk / activity
        "fills": fills, "max_inventory": max_inv, "final_inventory": inv,
        "sharpe": sharpe, "n_steps": n - 1,
    }


def compare_makers(df: pd.DataFrame, base: Optional[MMConfig] = None,
                   skew: float = 0.6) -> Dict[str, Dict]:
    """Naive symmetric maker (skew=0) vs imbalance-skewed maker, same everything else."""
    base = base or MMConfig()
    from dataclasses import replace
    naive = simulate_market_making(df, replace(base, skew=0.0))
    skewed = simulate_market_making(df, replace(base, skew=skew))
    return {"naive": naive, "skewed": skewed,
            "skew_helps_net": skewed["net"] > naive["net"],
            "skew_cuts_adverse": skewed["adverse_selection"] > naive["adverse_selection"]}


# =========================================================================== #
# DEEPENED, more-realistic sim: queue position + latency + ACTUAL trade prints
# =========================================================================== #
from tagent.microstructure import to_epoch_seconds  # noqa: E402


@dataclass
class RealMMConfig:
    """Config for the realistic maker (touch-joining, queue + latency + trades)."""
    quote_size: float = 1.0
    fee_bps: float = -1.0          # maker fee; NEGATIVE = rebate
    skew: float = 0.0              # imbalance skew (steps the toxic side BACK in queue)
    max_inventory: float = 20.0
    latency_ms: float = 100.0      # quote-update latency: skew reacts late to imbalance
    queue: bool = True             # model queue position (False = front of queue)
    ahead_mult: float = 1.0        # displayed size treated as queue resting AHEAD of you


def _empty_real_result() -> Dict:
    return {"net": 0.0, "spread_capture": 0.0, "rebate": 0.0, "adverse_selection": 0.0,
            "inventory_carry": 0.0, "notional_traded": 0.0, "net_bps": 0.0,
            "spread_bps": 0.0, "rebate_bps": 0.0, "adverse_bps": 0.0, "inventory_bps": 0.0,
            "fills": 0, "max_inventory": 0.0, "final_inventory": 0.0, "n_epochs": 0}


def simulate_mm_realistic(depth: pd.DataFrame, trades: pd.DataFrame,
                          cfg: Optional[RealMMConfig] = None) -> Dict:
    """Touch-joining maker filled by ACTUAL trade prints, behind a modelled queue.

    Each depth snapshot is a quote epoch; the maker rests at the touch
    (best_bid / best_ask). It sits at the BACK of the queue: its order only fills
    after the trade volume on its side exceeds the size resting ahead (the
    displayed depth at the time it posted). Queue priority PERSISTS while the best
    price is unchanged and RESETS when the price moves. The imbalance SKEW steps
    the toxic side deeper in the queue, and reacts to imbalance with a configurable
    LATENCY (so on fast imbalance flips it protects the wrong side).

    `depth`: ts, mid, best_bid, best_ask, bid_size, ask_size, depth_imbalance.
    `trades`: ts, price, qty, side (+1 buy-aggressor / -1 sell-aggressor).
    P&L decomposes exactly: net = spread_capture + rebate + adverse + inventory.
    """
    cfg = cfg or RealMMConfig()
    d = depth.dropna(subset=["mid"]).reset_index(drop=True)
    n = len(d)
    if n < 2:
        return _empty_real_result()
    ts = to_epoch_seconds(d["ts"])
    mid = pd.to_numeric(d["mid"], errors="coerce").to_numpy(float)
    bb = pd.to_numeric(d["best_bid"], errors="coerce").to_numpy(float)
    ba = pd.to_numeric(d["best_ask"], errors="coerce").to_numpy(float)
    bs = pd.to_numeric(d["bid_size"], errors="coerce").fillna(0.0).to_numpy(float)
    asz = pd.to_numeric(d["ask_size"], errors="coerce").fillna(0.0).to_numpy(float)
    imb = pd.to_numeric(d.get("depth_imbalance", 0.0), errors="coerce").fillna(0.0).to_numpy(float)

    # latency: the skew at epoch t reacts to the imbalance from `latency` ago
    lat = max(cfg.latency_ms, 0.0) / 1000.0
    react = np.clip(np.searchsorted(ts + lat, ts, side="right") - 1, 0, n - 1)

    # bucket trade volume that hits each side, per epoch (the fill driver)
    V_sell = np.zeros(n)   # sell-aggressor volume reaching the bid
    V_buy = np.zeros(n)    # buy-aggressor volume reaching the ask
    if trades is not None and len(trades):
        tr = trades.dropna(subset=["price"]) if "price" in trades else trades
        if len(tr):
            tt = to_epoch_seconds(tr["ts"])
            tp = pd.to_numeric(tr["price"], errors="coerce").to_numpy(float)
            tq = pd.to_numeric(tr["qty"], errors="coerce").fillna(0.0).to_numpy(float)
            tsd = pd.to_numeric(tr["side"], errors="coerce").fillna(0.0).to_numpy(float)
            ep = np.searchsorted(ts, tt, side="right") - 1          # owning epoch
            ok = (ep >= 0) & (ep < n)
            ep = ep[ok]; tp = tp[ok]; tq = tq[ok]; tsd = tsd[ok]
            sell = (tsd < 0) & (tp <= bb[ep] + 1e-12)
            buy = (tsd > 0) & (tp >= ba[ep] - 1e-12)
            np.add.at(V_sell, ep[sell], tq[sell])
            np.add.at(V_buy, ep[buy], tq[buy])

    fee = cfg.fee_bps / 1e4
    inv = 0.0
    spread = rebate = adverse = carry = notional = 0.0
    fills = 0
    max_inv = 0.0
    ahead_b = ahead_a = bid_rem = ask_rem = 0.0
    for t in range(n - 1):
        im = imb[react[t]]
        # (re)post at the touch: reset queue when price moved or our order emptied
        if t == 0 or bb[t] != bb[t - 1] or bid_rem <= 0:
            ahead_b = max(0.0, bs[t] * cfg.ahead_mult * (1.0 - cfg.skew * im)) if cfg.queue else 0.0
            bid_rem = cfg.quote_size
        if t == 0 or ba[t] != ba[t - 1] or ask_rem <= 0:
            ahead_a = max(0.0, asz[t] * cfg.ahead_mult * (1.0 + cfg.skew * im)) if cfg.queue else 0.0
            ask_rem = cfg.quote_size

        # consume queue ahead, then our resting size, from this epoch's trade volume
        cb = min(V_sell[t], ahead_b); ahead_b -= cb
        bid_fill = min(max(V_sell[t] - cb, 0.0), bid_rem)
        ca = min(V_buy[t], ahead_a); ahead_a -= ca
        ask_fill = min(max(V_buy[t] - ca, 0.0), ask_rem)

        # position limit
        bid_fill = min(bid_fill, max(0.0, cfg.max_inventory - inv))
        ask_fill = min(ask_fill, max(0.0, cfg.max_inventory + inv))
        bid_rem -= bid_fill
        ask_rem -= ask_fill

        m, m1 = mid[t], mid[t + 1]
        dm = m1 - m
        inv_pre = inv
        spread += bid_fill * (m - bb[t]) + ask_fill * (ba[t] - m)
        rebate += -fee * (bid_fill * bb[t] + ask_fill * ba[t])
        newfill = bid_fill - ask_fill
        inv += newfill
        adverse += newfill * dm
        carry += inv_pre * dm
        notional += bid_fill * bb[t] + ask_fill * ba[t]
        fills += int(bid_fill > 0) + int(ask_fill > 0)
        max_inv = max(max_inv, abs(inv))

    net = spread + rebate + adverse + carry
    bps = (lambda v: v / notional * 1e4) if notional > 0 else (lambda v: 0.0)
    return {
        "net": net, "spread_capture": spread, "rebate": rebate,
        "adverse_selection": adverse, "inventory_carry": carry, "notional_traded": notional,
        "net_bps": bps(net), "spread_bps": bps(spread), "rebate_bps": bps(rebate),
        "adverse_bps": bps(adverse), "inventory_bps": bps(carry),
        "fills": fills, "max_inventory": max_inv, "final_inventory": inv, "n_epochs": n - 1,
    }


def compare_makers_realistic(depth: pd.DataFrame, trades: pd.DataFrame,
                             base: Optional[RealMMConfig] = None, skew: float = 0.6) -> Dict:
    """Naive vs imbalance-skewed maker under the realistic (queue+latency+trades) sim."""
    from dataclasses import replace
    base = base or RealMMConfig()
    naive = simulate_mm_realistic(depth, trades, replace(base, skew=0.0))
    skewed = simulate_mm_realistic(depth, trades, replace(base, skew=skew))
    return {"naive": naive, "skewed": skewed,
            "skew_helps_net": skewed["net"] > naive["net"],
            "skew_cuts_adverse": skewed["adverse_selection"] > naive["adverse_selection"]}


def latency_rebate_sweep(depth: pd.DataFrame, trades: pd.DataFrame,
                         latencies_ms=(0, 50, 100, 200), rebates_bps=(0.0, 0.5, 1.0, 2.0),
                         skew: float = 0.6, base: Optional[RealMMConfig] = None) -> list:
    """Grid of net_bps over (latency, rebate) for the skewed maker — how the edge
    degrades with latency and what rebate it needs."""
    from dataclasses import replace
    base = base or RealMMConfig()
    rows = []
    for lat in latencies_ms:
        for reb in rebates_bps:
            r = simulate_mm_realistic(depth, trades,
                                      replace(base, skew=skew, latency_ms=lat, fee_bps=-reb))
            rows.append({"latency_ms": lat, "rebate_bps": reb, "net_bps": round(r["net_bps"], 3),
                         "adverse_bps": round(r["adverse_bps"], 3), "fills": r["fills"],
                         "max_inventory": round(r["max_inventory"], 2)})
    return rows


def breakeven_rebate(depth: pd.DataFrame, trades: pd.DataFrame, latency_ms: float = 100.0,
                     skew: float = 0.6, base: Optional[RealMMConfig] = None,
                     lo: float = 0.0, hi: float = 5.0) -> Optional[float]:
    """Rebate (bps) at which the skewed maker breaks even net at a given latency.

    net_bps rises ~linearly with the rebate (each bp of rebate adds ~1bp/notional
    on the filled volume), so we bisect for net_bps = 0. Returns None if it's
    already profitable at ``lo`` or still negative at ``hi``."""
    from dataclasses import replace
    base = base or RealMMConfig()

    def net_at(reb):
        return simulate_mm_realistic(depth, trades,
                                     replace(base, skew=skew, latency_ms=latency_ms,
                                             fee_bps=-reb))["net_bps"]
    if net_at(lo) >= 0:
        return lo
    if net_at(hi) < 0:
        return None
    for _ in range(40):
        mid_r = 0.5 * (lo + hi)
        if net_at(mid_r) >= 0:
            hi = mid_r
        else:
            lo = mid_r
    return round(0.5 * (lo + hi), 3)
