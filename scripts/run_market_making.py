"""Offline market-making backtest on recorded order books.

Two modes:
* DEFAULT (snapshot): resting quotes around mid, filled when price crosses them.
* --realistic: touch-joining maker filled by ACTUAL trade prints behind a modelled
  QUEUE, with quote-update LATENCY so the imbalance skew reacts late. Reports the
  net-P&L decomposition (spread + rebate - adverse - inventory), fills, inventory,
  and a sensitivity sweep over latency x rebate + the break-even rebate.

Usage:
    python scripts/run_market_making.py                         # snapshot mode
    python scripts/run_market_making.py --realistic             # queue + latency + trades
    python scripts/run_market_making.py --realistic --latency-ms 100 --rebate-bps 1.0 --skew 0.8
"""

import argparse
import glob
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd  # noqa: E402

from tagent.config import DATA_DIR  # noqa: E402
from tagent.market_making import (  # noqa: E402
    MMConfig, RealMMConfig, breakeven_rebate, compare_makers,
    compare_makers_realistic, latency_rebate_sweep)


def _discover(symbols):
    if symbols:
        return [(s, os.path.join(DATA_DIR, f"microstructure_{s.upper()}.csv")) for s in symbols]
    return [(os.path.basename(p)[len("microstructure_"):-len(".csv")], p)
            for p in sorted(glob.glob(os.path.join(DATA_DIR, "microstructure_*.csv")))]


def _line(tag, r):
    # all P&L shown in bps of traded notional (so spread ~ the quoted half-spread)
    extra = f"Sharpe {r['sharpe']:+6.2f}  " if "sharpe" in r else ""
    return (f"  {tag:8s} net {r['net_bps']:+7.2f}bps = spread {r['spread_bps']:+6.2f} "
            f"+ rebate {r['rebate_bps']:+5.2f} + adverse {r['adverse_bps']:+7.2f} "
            f"+ inv {r['inventory_bps']:+6.2f}  | {extra}fills {r['fills']:4d}  "
            f"maxInv {r['max_inventory']:.1f}")


def _run_realistic(args, files):
    base = RealMMConfig(quote_size=args.size, fee_bps=-args.rebate_bps,
                        max_inventory=args.max_inventory, latency_ms=args.latency_ms,
                        queue=not args.no_queue, ahead_mult=args.ahead_mult)
    print(f"\nMarket-making backtest (REALISTIC: queue + {args.latency_ms:.0f}ms latency + trade prints) "
          f"— rebate {args.rebate_bps}bps, skew {args.skew}\n")
    any_profit = False
    for sym, path in files:
        tpath = os.path.join(DATA_DIR, f"microstructure_trades_{sym.upper()}.csv")
        if not os.path.exists(tpath):
            print(f"{sym}: no trade prints (record with the updated record_orderbook.py) — skipping\n")
            continue
        try:
            depth = pd.read_csv(path); trades = pd.read_csv(tpath)
        except Exception as e:
            print(f"{sym}: read failed ({str(e)[:50]})"); continue
        if len(depth) < 50 or len(trades) < 10:
            print(f"{sym}: too little data (depth {len(depth)}, trades {len(trades)})\n"); continue
        cmp = compare_makers_realistic(depth, trades, base, skew=args.skew)
        print(f"{sym}  ({len(depth)} depth, {len(trades)} trades)")
        print(_line("NAIVE", cmp["naive"]))
        print(_line("SKEWED", cmp["skewed"]))
        # sensitivity sweep over latency x rebate (skewed maker)
        print("  latency x rebate -> net bps (skewed):")
        grid = latency_rebate_sweep(depth, trades, latencies_ms=(0, 50, 100, 200),
                                    rebates_bps=(0.0, 0.5, 1.0, 2.0), skew=args.skew, base=base)
        lats = sorted({g["latency_ms"] for g in grid}); rebs = sorted({g["rebate_bps"] for g in grid})
        print("        rebate:  " + "  ".join(f"{r:4.1f}bp" for r in rebs))
        for lat in lats:
            row = {g["rebate_bps"]: g["net_bps"] for g in grid if g["latency_ms"] == lat}
            print(f"    {lat:4.0f}ms:   " + "  ".join(f"{row[r]:+6.2f}" for r in rebs))
        be = breakeven_rebate(depth, trades, latency_ms=args.latency_ms, skew=args.skew, base=base)
        be_s = "already profitable @0" if be == 0.0 else (f"{be:.2f} bps" if be is not None else ">5 bps (no break-even)")
        print(f"  break-even rebate @ {args.latency_ms:.0f}ms latency: {be_s}\n")
        any_profit = any_profit or cmp["skewed"]["net"] > 0

    print("================ VERDICT (realistic) ================")
    if any_profit:
        print("  The faint edge SURVIVES here with queue + latency: skewed net is positive")
        print("  on the recorded book. Check the break-even rebate above — it's typically")
        print("  small but NON-ZERO, i.e. it leans on a maker rebate. Latency erodes it")
        print("  (top row vs bottom row of the sweep).")
    else:
        print("  With realistic QUEUE position + latency the edge does NOT survive net of")
        print("  costs: queue priority kills most benign fills and latency lets toxic ones")
        print("  through. It only turns positive at a high enough rebate (see the sweep).")
    print("  Closer to reality, but STILL NOT LIVE:")
    print("   * queue model approximates 'ahead = displayed size' and resets on price")
    print("     change — real queues need full L3 / order-by-order data;")
    print("   * latency only delays the skew at snapshot granularity (record depth at")
    print("     high hz, e.g. --hz 10, for fine latency resolution);")
    print("   * no cancel/replace cost, no partial-tick price improvement, no fees on")
    print("     the maker-takes-taker edge cases; inventory marked to mid.")
    print("  Honest read: spread-earning + skew + rebate is the ONLY plausibly-positive")
    print("  path, and even here it's a few tenths of a bp riding on rebate and queue luck.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--realistic", action="store_true", help="queue + latency + trade-print fills")
    ap.add_argument("--half-spread-bps", type=float, default=1.5)
    ap.add_argument("--rebate-bps", type=float, default=1.0, help="maker rebate (>0 = you receive)")
    ap.add_argument("--skew", type=float, default=0.8)
    ap.add_argument("--inv-skew-bps", type=float, default=0.5)
    ap.add_argument("--max-inventory", type=float, default=5.0)
    ap.add_argument("--size", type=float, default=1.0)
    ap.add_argument("--latency-ms", type=float, default=100.0)
    ap.add_argument("--ahead-mult", type=float, default=1.0)
    ap.add_argument("--no-queue", action="store_true", help="front of queue (no queue model)")
    args = ap.parse_args()

    files = _discover(args.symbols)
    if not files:
        print(f"No microstructure_*.csv in {DATA_DIR}. Record with scripts/record_orderbook.py first.")
        sys.exit(0)

    if args.realistic:
        _run_realistic(args, files)
        sys.exit(0)

    cfg = MMConfig(base_half_spread_bps=args.half_spread_bps, quote_size=args.size,
                   fee_bps=-args.rebate_bps, inv_skew_bps=args.inv_skew_bps,
                   max_inventory=args.max_inventory)
    print(f"\nMarket-making backtest — half-spread {args.half_spread_bps}bps, "
          f"rebate {args.rebate_bps}bps, skew {args.skew}, max inv {args.max_inventory}\n")
    any_profit = any_ex_rebate = False
    for sym, path in files:
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"{sym}: read failed ({str(e)[:50]})")
            continue
        if len(df) < 50:
            print(f"{sym}: too few rows ({len(df)})")
            continue
        cmp = compare_makers(df, cfg, skew=args.skew)
        print(f"{sym}  ({len(df)} snapshots)")
        print(_line("NAIVE", cmp["naive"]))
        print(_line("SKEWED", cmp["skewed"]))
        ex = cmp["skewed"]["net_bps"] - cmp["skewed"]["rebate_bps"]   # spread capture vs adverse, no rebate
        print(f"  -> skew {'HELPS' if cmp['skew_helps_net'] else 'does NOT help'} net; "
              f"{'cuts' if cmp['skew_cuts_adverse'] else 'does NOT cut'} adverse; "
              f"ex-rebate net {ex:+.2f}bps\n")
        any_profit = any_profit or cmp["skewed"]["net"] > 0
        any_ex_rebate = any_ex_rebate or ex > 0

    print("================ VERDICT ================")
    if any_profit and any_ex_rebate:
        print("  Earning the spread (with imbalance skew) is net-POSITIVE here EVEN before")
        print("  the rebate — spread capture beats adverse selection. The skew is what")
        print("  does it (widening the toxic side). BUT this is a SIMPLIFIED sim:")
    elif any_profit:
        print("  Net is POSITIVE here, but it HINGES ON THE MAKER REBATE: ex-rebate (raw")
        print("  spread capture vs adverse selection) is NEGATIVE. Imbalance skew cuts the")
        print("  adverse-selection bleed but doesn't beat it on spread alone — the edge is")
        print("  rebate + skew, not spread capture. AND this is a SIMPLIFIED sim:")
    else:
        print("  Even earning the spread, net is around/below zero: adverse selection eats")
        print("  the spread + rebate. Imbalance skew reduces the bleed but doesn't")
        print("  manufacture an edge. AND this is a SIMPLIFIED sim:")
    print("   * fills are modelled as 'price trades through the quote' on 1s snapshots")
    print("     (no real trade prints) -> every fill looks locally adverse; reality has")
    print("     non-toxic flow we under-count, and intrabar reverts we can't see.")
    print("   * NO queue position / NO latency: real making lives or dies on being at")
    print("     the front of the queue and reacting in microseconds — we model neither.")
    print("   * inventory marked to mid, no funding/borrow on the short side.")
    print("  So treat this as a DIRECTIONAL sanity check, not a P&L promise. The honest")
    print("  read: spread-earning + skew is the only way the order-book signal could pay,")
    print("  but the edge is thin and dominated by microstructure we can't fully simulate.")
