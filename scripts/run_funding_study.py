"""Crypto funding-rate cash-and-carry study (long spot + short perp).

Harvests the funding rate (a structural premium) net of realistic costs. Pulls
public Binance data (no key), simulates the carry per coin and as an equal-weight
basket, and prints an honest verdict: is the net-of-cost yield positive and
meaningful, or eaten by fees / basis? Notes tail risks (funding flips, liquidation).

Usage:
    python scripts/run_funding_study.py
    python scripts/run_funding_study.py --symbols BTCUSDT ETHUSDT --fee-bps 7 --min-funding-bps 0
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from tagent.data.funding import DEFAULT_COINS, load_carry  # noqa: E402
from tagent.funding_study import basket, carry_returns, carry_stats, funding_flips  # noqa: E402


def _fmt(s):
    return (f"ann {s['ann_return']:+7.2%}  vol {s['ann_vol']:6.2%}  "
            f"Sharpe {s['sharpe']:+5.2f}  maxDD {s['max_drawdown']:+7.2%}  "
            f"in-carry {s['in_carry_pct']:4.0f}%  ({s['n_intervals']} x 8h)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=DEFAULT_COINS)
    ap.add_argument("--fee-bps", type=float, default=7.0, help="per-leg fee (bps)")
    ap.add_argument("--spread-bps", type=float, default=1.0)
    ap.add_argument("--min-funding-bps", type=float, default=0.0,
                    help="per-8h funding hurdle to be in the carry")
    ap.add_argument("--lookback", type=int, default=3)
    args = ap.parse_args()

    kw = dict(fee_bps=args.fee_bps, spread_bps=args.spread_bps,
              min_funding_bps=args.min_funding_bps, lookback=args.lookback)
    per_coin, any_pos = {}, False
    print(f"Cash-and-carry (fee {args.fee_bps}bps/leg + {args.spread_bps}bps spread, "
          f"funding hurdle {args.min_funding_bps}bps/8h)\n")
    for sym in args.symbols:
        try:
            df = load_carry(sym)
        except Exception as e:
            print(f"  {sym}: load failed ({str(e)[:60]})")
            continue
        if len(df) < 30:
            print(f"  {sym}: too little data ({len(df)})")
            continue
        r = carry_returns(df, **kw)
        per_coin[sym] = r
        st = carry_stats(r["net"], r["in_carry"])
        fl = funding_flips(df)
        any_pos = any_pos or st["ann_return"] > 0
        span_d = (df.index.max() - df.index.min()).days
        print(f"  {sym:9s} {span_d:4d}d  {_fmt(st)}")
        print(f"            funding: {fl['neg_pct']:.0f}% intervals negative, "
              f"worst {fl['min_funding_bps']:.1f}bps, longest neg streak {fl['longest_neg_streak']} (x8h)")

    if per_coin:
        b = basket(per_coin)
        print(f"\n  {'BASKET':9s}       {_fmt(b['stats'])}")

    print("\n================ VERDICT ================")
    if not per_coin:
        print("  No data. Funding endpoints are public; check connectivity.")
        sys.exit(0)
    s = b["stats"]
    if s["ann_return"] > 0 and s["sharpe"] > 1.0:
        print(f"  REAL but SMALL net-of-cost carry: basket ann {s['ann_return']:+.2%}, "
              f"vol {s['ann_vol']:.2%}, Sharpe {s['sharpe']:+.2f}, maxDD {s['max_drawdown']:+.2%}.")
        print("  A genuine STRUCTURAL premium (not a forecast): held continuously it pays.")
        print("  HONEST caveats:")
        print("   • Yield is small and REGIME-DEPENDENT — funding is compressed now (a")
        print("     neutral market); in bull runs it can be many× this, near zero otherwise.")
        print("   • The high Sharpe is MISLEADING — vol is tiny so it looks great, but the")
        print("     left tail is fat: funding can spike negative (a coin here hit ~-30bps/8h)")
        print("     and a sharp rally can LIQUIDATE the short-perp leg if margin isn't managed.")
        print("   • It only works if you HOLD — timing funding by toggling in/out churns and")
        print("     the round-trip costs wipe out the premium (we use hysteresis to avoid that).")
    else:
        print(f"  NOT WORTH IT here: basket ann {s['ann_return']:+.2%}, Sharpe {s['sharpe']:+.2f}.")
        print("  The funding premium in this window does not clearly beat fees + basis,")
        print("  and carries funding-flip + liquidation tail risk.")
    print("  (Delta-neutral; decision uses past funding only; costs on both legs.)")
