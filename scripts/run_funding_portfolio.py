"""Multi-year cross-sectional funding-carry study — optimise the one real edge.

Pulls multi-year funding + spot/perp for a broad liquid universe (public Binance,
no key), then:
  * compares equal-weight / funding-weighted / top-N selection,
  * under maker (post-only) vs taker cost assumptions,
  * splits cycle-average net yield by bull / quiet / bear regime,
  * models short-perp liquidation risk (margin ratio, funding spikes + rallies)
    and a yield-vs-leverage tail-risk sweep,
  * estimates capacity,
and prints an HONEST verdict (small, regime-dependent, tail-risky).

Usage:
    python scripts/run_funding_portfolio.py
    python scripts/run_funding_portfolio.py --years 3 --top-n 5 --max-weight 0.25
    python scripts/run_funding_portfolio.py --coins BTCUSDT ETHUSDT SOLUSDT
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:                                     # so unicode prints on any console codepage
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd  # noqa: E402

from tagent.data.funding import LIQUID_COINS, load_carry_history  # noqa: E402
from tagent.funding_portfolio import (  # noqa: E402
    CostModel, capacity_curve, classify_regime, compare_schemes,
    leverage_tradeoff, portfolio_returns, stress_test, yield_by_regime,
)
from tagent.funding_study import INTERVALS_PER_YEAR, carry_stats  # noqa: E402


def _load_panel(coins, years):
    panel = {}
    for sym in coins:
        try:
            df = load_carry_history(sym, years=years)
        except Exception as e:
            print(f"  {sym}: load failed ({str(e)[:50]})")
            continue
        if len(df) >= 60:
            panel[sym] = df
            span = (df.index.max() - df.index.min()).days
            print(f"  {sym:10s} {len(df):5d} intervals  {span:5d}d")
        else:
            print(f"  {sym}: too little data ({len(df)})")
    return panel


def _fmt(st):
    return (f"ann {st['ann_return']:+7.2%}  vol {st['ann_vol']:6.2%}  "
            f"Sharpe {st['sharpe']:+5.2f}  maxDD {st['max_drawdown']:+7.2%}  "
            f"invested {st.get('in_carry_pct', float('nan')):4.0f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--coins", nargs="+", default=LIQUID_COINS)
    ap.add_argument("--years", type=float, default=3.0)
    ap.add_argument("--hurdle-bps", type=float, default=2.0)
    ap.add_argument("--band-bps", type=float, default=2.0)
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--max-weight", type=float, default=0.25)
    ap.add_argument("--leverages", nargs="+", type=float, default=[1, 2, 3, 5, 10])
    ap.add_argument("--capital", type=float, default=1e7, help="deploy size for capacity")
    args = ap.parse_args()

    print(f"\nLoading {len(args.coins)} coins, ~{args.years}y of 8h funding data...")
    panel = _load_panel(args.coins, args.years)
    if not panel:
        print("\nNo data (funding endpoints are public; check connectivity).")
        sys.exit(0)

    kw = dict(hurdle_bps=args.hurdle_bps, band_bps=args.band_bps,
              top_n=args.top_n, max_weight=args.max_weight)

    # 1) scheme comparison under maker vs taker costs
    for label, cost in (("MAKER (post-only)", CostModel(post_only=True)),
                        ("TAKER", CostModel(post_only=False))):
        print(f"\n================ schemes — {label}, {cost.turnover_cost_bps():.1f}bps/turn ===")
        cmp = compare_schemes(panel, cost=cost, **kw)
        for sch, st in cmp.items():
            print(f"  {sch:8s} {_fmt(st)}  turn {st['avg_turnover_bps']:.2f}bps  "
                  f"~{st['avg_n_held']:.1f} coins")

    # 2) regime split (funding-weighted, maker) using BTC trend
    r = portfolio_returns(panel, scheme="funding", cost=CostModel(post_only=True), **kw)
    btc = panel.get("BTCUSDT")
    if btc is not None:
        price = btc["spot_close"].reindex(r.index).ffill()
        regime = classify_regime(price)
        byreg = yield_by_regime(r["net"], regime)
        print("\n================ cycle-average net yield BY REGIME (funding-wtd, maker) ===")
        for reg in ("bull", "quiet", "bear"):
            d = byreg[reg]
            print(f"  {reg:6s} ann {d['ann_return']:+7.2%}  ({d['mean_bps']:+.2f} bps/8h, "
                  f"{d['n_intervals']} intervals)")
    full = carry_stats(r["net"], (r["invested"] > 0).astype(float))
    print(f"\n  FULL SAMPLE (funding-wtd, maker): {_fmt(full)}")

    # 3) risk: yield vs leverage tail
    per_int = float(r["net"][r["net"] != 0].mean() or 0.0)
    print("\n================ RISK — short-perp leverage vs liquidation tail ===")
    print("  (stress: +30% rally over 3 days while funding spikes to -30bps/8h)")
    for row in leverage_tradeoff(per_int, args.leverages, rally_pct=0.30, funding_spike_bps=-30):
        flag = "LIQUIDATED" if row["liquidated"] else f"min margin {row['worst_margin_ratio']:+.3f}"
        print(f"  {row['leverage']:4.0f}x  yield-on-margin {row['ann_yield_on_margin']:+7.2%}  -> {flag}")

    # 4) capacity
    gross_ann = full["ann_return"]
    cap = capacity_curve(max(gross_ann, 0.0), depth_usd=1e9, sizes_usd=[1e6, 1e7, 1e8, 5e8])
    print("\n================ CAPACITY (linear impact, depth ~$1B) ===")
    for pt in cap["curve"]:
        print(f"  ${pt['size_usd']/1e6:6.0f}M -> net {pt['net_yield']:+.2%}")
    print(f"  approx capacity (net->0): ${cap['capacity_usd']/1e6:.0f}M")

    # 5) honest verdict
    print("\n================ VERDICT ================")
    if full["ann_return"] > 0 and full["sharpe"] > 0.8:
        print(f"  REAL but SMALL structural carry: full-sample {full['ann_return']:+.2%} ann, "
              f"Sharpe {full['sharpe']:+.2f}, maxDD {full['max_drawdown']:+.2%}.")
        print("  HONEST caveats:")
        print("   • REGIME-DEPENDENT — bull funding pays many× the quiet/bear yield above;")
        print("     a quiet-window backtest understates AND a bull-only one overstates it.")
        print("   • The high Sharpe is misleading: tiny vol, FAT LEFT TAIL — the short-perp")
        print("     leg can be LIQUIDATED in a sharp rally (see the leverage table); keep")
        print("     leverage low and use cross-margin, or the carry is wiped by one event.")
        print("   • Post-only (maker) execution + holding (not churning) are REQUIRED — the")
        print("     taker rows above lose materially more to costs.")
        print("   • Capacity is finite: size compresses funding and adds impact.")
    else:
        print(f"  NOT WORTH IT in this sample: {full['ann_return']:+.2%} ann, Sharpe {full['sharpe']:+.2f}.")
    print("  (Delta-neutral; decisions use PAST funding only; costs on both legs.)")
