"""Market-making simulator — synthetic books, no network."""

import numpy as np
import pandas as pd

from tagent.market_making import (
    MMConfig, RealMMConfig, breakeven_rebate, compare_makers,
    compare_makers_realistic, latency_rebate_sweep, quote,
    simulate_market_making, simulate_mm_realistic,
)


def _df(mid, imb=0.0, ts_start="2026-01-01"):
    n = len(mid)
    ts = pd.date_range(ts_start, periods=n, freq="s", tz="UTC").astype(str)
    imb = np.full(n, imb, float) if np.isscalar(imb) else np.asarray(imb, float)
    return pd.DataFrame({"ts": ts, "mid": np.asarray(mid, float), "depth_imbalance": imb})


# --------------------------------------------------------------------------- #
# quote logic (pure, no lookahead)
# --------------------------------------------------------------------------- #
def test_quote_symmetric_when_no_skew_no_inventory():
    cfg = MMConfig(base_half_spread_bps=2.0, skew=0.0, inv_skew_bps=0.0)
    bid, ask, pb, pa = quote(100.0, 0.5, 0.0, cfg)
    assert abs((100.0 - bid) - (ask - 100.0)) < 1e-9        # symmetric around mid
    assert abs((100.0 - bid) - 2.0 / 1e4 * 100.0) < 1e-12   # 2 bps each side
    assert pb and pa


def test_imbalance_skew_tightens_favoured_side():
    cfg = MMConfig(base_half_spread_bps=2.0, skew=0.5, inv_skew_bps=0.0)
    bid, ask, _, _ = quote(100.0, 1.0, 0.0, cfg)            # imbalance up
    assert (100.0 - bid) < (ask - 100.0)                    # bid tighter, ask wider


def test_inventory_skew_leans_to_flatten():
    cfg = MMConfig(base_half_spread_bps=2.0, skew=0.0, inv_skew_bps=1.0)
    _, ask_flat, _, _ = quote(100.0, 0.0, 0.0, cfg)
    _, ask_long, _, _ = quote(100.0, 0.0, 3.0, cfg)         # long inventory
    assert ask_long < ask_flat                              # ask drops -> sell sooner to flatten


def test_position_limit_stops_quoting_the_breaching_side():
    cfg = MMConfig(max_inventory=2.0)
    _, _, pb, pa = quote(100.0, 0.0, 2.0, cfg)              # at long limit
    assert not pb and pa                                    # no more buying; can still sell
    _, _, pb2, pa2 = quote(100.0, 0.0, -2.0, cfg)
    assert pb2 and not pa2


# --------------------------------------------------------------------------- #
# fill logic + spread / rebate / adverse accounting
# --------------------------------------------------------------------------- #
def test_bid_fills_when_price_trades_through_and_captures_spread():
    cfg = MMConfig(base_half_spread_bps=1.5, quote_size=1.0, fee_bps=-1.0,
                   skew=0.0, inv_skew_bps=0.0)
    r = simulate_market_making(_df([100.0, 99.90]), cfg)    # mid drops 10 bps -> bid fills
    assert r["fills"] == 1 and r["final_inventory"] == 1.0
    # spread captured = quoted half-spread (1.5 bps of 100 = 0.015) on size 1
    assert abs(r["spread_capture"] - 1.5 / 1e4 * 100.0) < 1e-9
    # maker rebate is positive (fee_bps negative)
    assert r["rebate"] > 0


def test_no_fill_when_price_does_not_reach_quote():
    cfg = MMConfig(base_half_spread_bps=5.0)                # quotes far from mid
    r = simulate_market_making(_df([100.0, 100.001, 99.999]), cfg)  # tiny moves
    assert r["fills"] == 0 and r["final_inventory"] == 0.0


def test_adverse_selection_is_negative_and_marks_to_next_mid():
    cfg = MMConfig(base_half_spread_bps=1.5, quote_size=1.0, fee_bps=0.0,
                   skew=0.0, inv_skew_bps=0.0)
    r = simulate_market_making(_df([100.0, 99.90]), cfg)
    # bought as price fell -> adverse = qty * (m1 - m) = 1 * (99.90 - 100.0)
    assert abs(r["adverse_selection"] - (99.90 - 100.0)) < 1e-9
    assert r["adverse_selection"] < 0


def test_rebate_scales_and_costs_when_taker_fee_positive():
    base = MMConfig(base_half_spread_bps=1.5, fee_bps=-2.0, skew=0, inv_skew_bps=0)
    reb = simulate_market_making(_df([100.0, 99.9]), base)["rebate"]
    cost = simulate_market_making(_df([100.0, 99.9]),
                                  MMConfig(base_half_spread_bps=1.5, fee_bps=2.0,
                                           skew=0, inv_skew_bps=0))["rebate"]
    assert reb > 0 > cost                                   # rebate positive, positive fee = cost


# --------------------------------------------------------------------------- #
# P&L decomposition is exact
# --------------------------------------------------------------------------- #
def test_pnl_decomposition_sums_to_net():
    rng = np.random.default_rng(0)
    mid = 100 + np.cumsum(rng.normal(0, 0.02, 300))
    r = simulate_market_making(_df(mid, imb=rng.uniform(-1, 1, 300)),
                               MMConfig(skew=0.4))
    parts = r["spread_capture"] + r["rebate"] + r["adverse_selection"] + r["inventory_carry"]
    assert abs(parts - r["net"]) < 1e-6                     # components reconcile to net


# --------------------------------------------------------------------------- #
# inventory limit holds through a sustained trend
# --------------------------------------------------------------------------- #
def test_inventory_never_exceeds_limit():
    mid = list(100 - np.arange(40) * 0.05)                  # steady decline -> bid fills repeatedly
    r = simulate_market_making(_df(mid), MMConfig(max_inventory=3.0, base_half_spread_bps=1.0,
                                                  skew=0, inv_skew_bps=0.0))
    assert r["max_inventory"] <= 3.0 + 1e-9


# --------------------------------------------------------------------------- #
# imbalance skew reduces adverse selection on a signalled trend
# --------------------------------------------------------------------------- #
def test_skew_cuts_adverse_when_imbalance_predicts_trend():
    # imbalance strongly positive while price trends UP: the naive maker keeps
    # selling into the rise (toxic asks); the skewed maker widens the ask to avoid it.
    n = 120
    mid = 100 + np.arange(n) * 0.03                         # steady up-trend
    df = _df(mid, imb=0.9)
    cmp = compare_makers(df, MMConfig(base_half_spread_bps=1.0, inv_skew_bps=0.2,
                                      max_inventory=10.0), skew=0.8)
    assert cmp["skew_cuts_adverse"]                         # skewed adverse > naive (less negative)
    assert cmp["skewed"]["net"] > cmp["naive"]["net"]       # and nets better


def test_no_lookahead_quote_uses_current_not_future():
    # the quote at t depends only on mid[t]/imb[t]; the fill is triggered by mid[t+1].
    cfg = MMConfig(base_half_spread_bps=1.5, quote_size=1.0, fee_bps=0.0,
                   skew=0.5, inv_skew_bps=0.0)
    # quote at step 0 is computed from mid[0]/imb[0] only -> independent of later rows
    bid0, ask0, pb0, _ = quote(100.0, 0.2, 0.0, cfg)
    for tail in ([99.90, 101.0], [99.90, 99.0], [99.90, 100.05]):
        r = simulate_market_making(_df([100.0] + tail, imb=[0.2, 0.2, 0.2]), cfg)
        assert r["fills"] >= 1 and pb0 and tail[0] <= bid0  # step-0 bid fills regardless of the future


# =========================================================================== #
# Deepened realistic sim: queue position + latency + trade prints
# =========================================================================== #
def _depth(mids, bs=1.0, asz=1.0, imb=0.0, half=0.5, dt=0.05):
    n = len(mids)
    mids = np.asarray(mids, float)
    ts = pd.date_range("2026-01-01", periods=n, freq=f"{int(dt*1000)}ms", tz="UTC").astype(str)
    imb = np.full(n, imb, float) if np.isscalar(imb) else np.asarray(imb, float)
    return pd.DataFrame({"ts": ts, "mid": mids, "best_bid": mids - half,
                         "best_ask": mids + half,
                         "bid_size": np.full(n, bs, float), "ask_size": np.full(n, asz, float),
                         "depth_imbalance": imb})


def _trades(rows):
    return pd.DataFrame(rows, columns=["ts", "price", "qty", "side"])


# --------------------------------------------------------------------------- #
# trade-print fills + queue position
# --------------------------------------------------------------------------- #
def test_trade_print_fills_directionally():
    # front of queue: a sell-aggressor hitting the bid -> we BUY; buy-aggressor -> we SELL
    depth = _depth([100.0, 100.0, 100.0])
    trades = _trades([("2026-01-01T00:00:00.010Z", 99.5, 1.0, -1),   # epoch 0: hits bid
                      ("2026-01-01T00:00:00.060Z", 100.5, 1.0, 1)])  # epoch 1: lifts ask
    r = simulate_mm_realistic(depth, trades, RealMMConfig(queue=False, skew=0.0, fee_bps=0.0))
    assert r["fills"] == 2 and abs(r["final_inventory"]) < 1e-9   # bought then sold
    assert r["spread_capture"] > 0                                # captured the half-spread


def test_queue_blocks_fill_until_ahead_cleared():
    depth = _depth([100.0, 100.0], bs=2.0)                        # 2.0 resting ahead at the bid
    trades = _trades([("2026-01-01T00:00:00.010Z", 99.5, 1.0, -1)])  # only 1.0 traded
    q = simulate_mm_realistic(depth, trades, RealMMConfig(queue=True, ahead_mult=1.0, skew=0.0))
    front = simulate_mm_realistic(depth, trades, RealMMConfig(queue=False, skew=0.0))
    assert q["fills"] == 0                                        # queue ahead not cleared -> no fill
    assert front["fills"] == 1 and front["final_inventory"] == 1.0  # front of queue fills


def test_queue_fills_once_volume_exceeds_ahead():
    depth = _depth([100.0, 100.0], bs=2.0)
    trades = _trades([("2026-01-01T00:00:00.010Z", 99.5, 3.0, -1)])  # 3 > 2 ahead -> 1 reaches us
    r = simulate_mm_realistic(depth, trades, RealMMConfig(queue=True, ahead_mult=1.0,
                                                          skew=0.0, quote_size=1.0))
    assert r["fills"] == 1 and r["final_inventory"] == 1.0


def test_spread_and_rebate_accounting():
    depth = _depth([100.0, 100.0], half=0.5)
    trades = _trades([("2026-01-01T00:00:00.010Z", 99.5, 1.0, -1)])
    r = simulate_mm_realistic(depth, trades, RealMMConfig(queue=False, fee_bps=-1.0))
    assert abs(r["spread_capture"] - 0.5) < 1e-9                  # mid 100 - bid 99.5
    assert r["rebate"] > 0                                        # 1bp maker rebate on notional


# --------------------------------------------------------------------------- #
# latency degrades the imbalance skew
# --------------------------------------------------------------------------- #
def _toxic_dataset(n=60, seed=0, q=1.5, half=0.005, move=0.02, dt=0.05):
    """Realistic-ish toxic book: ~1bp spread, ~2bp moves. Imbalance leads the next
    move; aggressive trades hit the side the price moves TOWARD (so filling is
    adverse). Trade size q > displayed size so a naive maker fills but a maker that
    steps the toxic side BACK in the queue avoids it. Best price moves every epoch
    (forces reposts, so the skew — and its latency — actually bite)."""
    rng = np.random.default_rng(seed)
    moves = rng.choice([-1.0, 1.0], size=n) * move
    mid = 100 + np.cumsum(moves)
    imb = np.sign(moves)                                          # imb[t] predicts move[t]->[t+1]
    depth = _depth(mid, bs=1.0, asz=1.0, imb=imb, half=half, dt=dt)
    rows = []
    for t in range(n):
        tstamp = (pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(seconds=t * dt + 0.001)).isoformat()
        if moves[t] < 0:
            rows.append((tstamp, mid[t] - half, q, -1))           # sell hits bid (we'd buy a drop)
        else:
            rows.append((tstamp, mid[t] + half, q, 1))            # buy lifts ask (we'd sell a rise)
    return depth, _trades(rows)


def test_latency_degrades_skewed_maker():
    depth, trades = _toxic_dataset()
    base = RealMMConfig(skew=0.8, fee_bps=0.0, ahead_mult=1.0, quote_size=1.0)
    from dataclasses import replace
    fast = simulate_mm_realistic(depth, trades, replace(base, latency_ms=0.0))
    slow = simulate_mm_realistic(depth, trades, replace(base, latency_ms=200.0))
    assert fast["net"] >= slow["net"]                            # latency erodes the edge
    assert fast["adverse_selection"] >= slow["adverse_selection"]  # and worsens adverse selection


def test_zero_latency_skew_cuts_adverse_vs_naive():
    depth, trades = _toxic_dataset()
    cmp = compare_makers_realistic(depth, trades,
                                   RealMMConfig(fee_bps=0.0, latency_ms=0.0, ahead_mult=1.0),
                                   skew=0.8)
    assert cmp["skew_cuts_adverse"]                              # skew avoids the toxic side


# --------------------------------------------------------------------------- #
# sensitivity sweep + break-even rebate
# --------------------------------------------------------------------------- #
def test_sensitivity_sweep_monotone_in_rebate():
    depth, trades = _toxic_dataset()
    grid = latency_rebate_sweep(depth, trades, latencies_ms=(0, 200),
                                rebates_bps=(0.0, 1.0, 2.0), skew=0.8,
                                base=RealMMConfig(ahead_mult=1.0))
    by_lat = {}
    for row in grid:
        by_lat.setdefault(row["latency_ms"], []).append(row["net_bps"])
    for nets in by_lat.values():
        assert nets[0] <= nets[1] <= nets[2]                     # more rebate -> higher net


def test_breakeven_rebate_found():
    depth, trades = _toxic_dataset()
    from dataclasses import replace
    base = RealMMConfig(ahead_mult=1.0)

    def net_at(reb):
        return simulate_mm_realistic(depth, trades,
                                     replace(base, skew=0.8, latency_ms=200.0,
                                             fee_bps=-reb))["net_bps"]
    assert net_at(0.0) <= net_at(5.0)                            # net rises with the rebate
    be = breakeven_rebate(depth, trades, latency_ms=200.0, skew=0.8, base=base, lo=0.0, hi=20.0)
    if be is None:
        assert net_at(20.0) < 0                                  # never breaks even in range
    elif be == 0.0:
        assert net_at(0.0) >= 0                                  # already profitable, no rebate needed
    else:
        assert net_at(be) >= -0.5 and net_at(max(0.0, be - 0.5)) < net_at(be)  # zero crossing


# --------------------------------------------------------------------------- #
# no-lookahead: future trades don't change earlier P&L
# --------------------------------------------------------------------------- #
def test_no_lookahead_future_trades_do_not_change_pnl():
    depth = _depth([100.0, 100.0, 100.0, 100.0])
    early = [("2026-01-01T00:00:00.010Z", 99.5, 1.0, -1)]        # epoch 0 fill
    a = simulate_mm_realistic(depth, _trades(early), RealMMConfig(queue=False, fee_bps=0.0))
    # add a trade in the final (no-decision) epoch -> must not change anything
    b = simulate_mm_realistic(depth, _trades(early + [("2026-01-01T00:00:00.160Z", 100.5, 5.0, 1)]),
                              RealMMConfig(queue=False, fee_bps=0.0))
    assert a["net"] == b["net"] and a["final_inventory"] == b["final_inventory"]
