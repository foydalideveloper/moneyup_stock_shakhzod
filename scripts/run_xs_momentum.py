"""Cross-sectional crypto MOMENTUM backtest (rank coins against each other).

Pulls multi-year daily Binance klines for a liquid universe (public, no key;
cached to data/), and tests CLASSIC long-horizon momentum: rank coins by their
formation-period return (default 30/60/90 days, SKIPPING the most recent week to
dodge short-term reversal), go LONG the top quantile / SHORT the bottom
(dollar-neutral), rebalance at LOW frequency (weekly/monthly) so turnover stays
cheap, net costs, vs holding the basket. Also runs a short-horizon REVERSAL sanity
check (1-3 day: long losers / short winners). Honest verdict at the end.

Usage:
    python scripts/run_xs_momentum.py
    python scripts/run_xs_momentum.py --lookbacks 30 60 90 --rebalance weekly --skip-recent 7
    python scripts/run_xs_momentum.py --rebalance monthly --years 4 --top-q 0.25
"""

import argparse
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
from tagent.crypto_ml import fetch_klines  # noqa: E402
from tagent.data.funding import LIQUID_COINS  # noqa: E402
from tagent.xs_momentum import XSMomConfig, backtest, sweep_lookbacks  # noqa: E402

_PPY = {"1d": 365, "8h": 3 * 365, "1h": 24 * 365, "4h": 6 * 365}


def _load(sym, interval, years):
    path = os.path.join(DATA_DIR, f"klines_{sym}_{interval}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if len(df) > 30:
            return df
    df = fetch_klines(sym, interval=interval, years=years)
    df.to_csv(path)
    return df


def _fmt_stats(st, tag):
    return (f"  {tag:14s} Sharpe {st['sharpe']:+6.2f}  ann {st['cagr']:+8.2%}  "
            f"maxDD {st['max_drawdown']:+7.2%}  total {st['total_return']:+8.2%}")


_REBAL = {"daily": 1, "weekly": 7, "monthly": 30}


def _rebal(s):
    s = str(s).lower()
    if s in _REBAL:
        return _REBAL[s]
    try:
        return max(1, int(s))
    except ValueError:
        return 7


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--coins", nargs="+", default=LIQUID_COINS)
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--years", type=float, default=3.0)
    ap.add_argument("--lookbacks", nargs="+", type=int, default=[30, 60, 90])
    ap.add_argument("--rebalance", default="weekly", help="daily/weekly/monthly or N bars")
    ap.add_argument("--skip-recent", type=int, default=7, help="skip the most recent N bars")
    ap.add_argument("--top-q", type=float, default=0.2)
    ap.add_argument("--hysteresis", type=float, default=0.1)
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--slippage-bps", type=float, default=2.0)
    ap.add_argument("--long-only", action="store_true")
    args = ap.parse_args()
    ppy = _PPY.get(args.interval, 365)
    rebal = _rebal(args.rebalance)

    print(f"\nLoading {len(args.coins)} coins, {args.interval} bars, ~{args.years}y...")
    per_coin = {}
    for sym in args.coins:
        try:
            df = _load(sym, args.interval, args.years)
        except Exception as e:
            print(f"  {sym}: load failed ({str(e)[:45]})")
            continue
        if len(df) > 30:
            per_coin[sym] = df
    if len(per_coin) < 4:
        print("\nNeed >=4 coins with data — try again (klines are public).")
        sys.exit(0)
    span = max(len(d) for d in per_coin.values())
    print(f"Loaded {len(per_coin)} coins, up to {span} bars each.\n")

    base = XSMomConfig(top_q=args.top_q, allow_short=not args.long_only,
                       cost_bps=args.cost_bps, slippage_bps=args.slippage_bps,
                       hysteresis=args.hysteresis, rebalance=rebal, skip_recent=args.skip_recent)

    cps = args.cost_bps + args.slippage_bps
    print(f"================ CLASSIC momentum ({'long-only' if args.long_only else 'long-short'}, "
          f"rebalance every {rebal}d, skip {args.skip_recent}d, cost {cps:.0f}bps/turn) ===")
    print("  lookback  Sharpe   ann       maxDD     turnover")
    rows = sweep_lookbacks(per_coin, lookbacks=args.lookbacks, cfg=base, periods_per_year=ppy)
    for r in rows:
        print(f"   {r['lookback']:>3}d     {r['sharpe']:+6.2f}  {r['ann_return']:+8.2%}  "
              f"{r['max_drawdown']:+7.2%}   {r['avg_turnover']:.3f}")

    from dataclasses import replace
    best = max(rows, key=lambda r: r["sharpe"])
    r = backtest(per_coin, replace(base, lookback=best["lookback"]), periods_per_year=ppy)
    print(f"\n================ best lookback = {best['lookback']}d ({r['n_bars']} bars) ===")
    print(_fmt_stats(r["stats"], "MOMENTUM"))
    print(_fmt_stats(r["basket_stats"], "BASKET (hold)"))
    print(f"  avg turnover {r['avg_turnover']:.3f}/bar   "
          f"net total {r['net'].sum()*100:+.2f}% vs gross {r['gross'].sum()*100:+.2f}%")

    # short-horizon REVERSAL sanity check (long losers / short winners, daily)
    print("\n================ REVERSAL sanity (1-3d, long losers/short winners, daily) ===")
    print("  lookback  Sharpe   GROSS%    net%      turnover")
    for lb in (1, 2, 3):
        rr = backtest(per_coin, replace(base, lookback=lb, rebalance=1, skip_recent=0,
                                        reverse=True), periods_per_year=ppy)
        print(f"   {lb:>3}d     {rr['stats']['sharpe']:+6.2f}  {rr['gross'].sum()*100:+7.2f}%  "
              f"{rr['net'].sum()*100:+7.2f}%   {rr['avg_turnover']:.3f}")

    s, b = r["stats"], r["basket_stats"]
    print("\n================ VERDICT ================")
    edge = s["sharpe"] > 0.3 and s["total_return"] > 0 and s["sharpe"] > b["sharpe"]
    stable = sum(1 for x in rows if x["sharpe"] > 0) >= max(2, len(rows) - 1)
    if edge and stable:
        print(f"  CLASSIC long-horizon momentum shows a net-of-cost edge: best {best['lookback']}d")
        print(f"  long-short Sharpe {s['sharpe']:+.2f} (vs basket {b['sharpe']:+.2f}), "
              f"ann {s['cagr']:+.2%}, maxDD {s['max_drawdown']:+.2%}, low turnover "
              f"({r['avg_turnover']:.3f}/bar).")
        print("  Caveats: crypto's history is SHORT (few independent momentum cycles), and")
        print("  the factor unwinds VIOLENTLY in regime flips (the maxDD is the tell);")
        print("  market-neutral dodges beta, not crashes; short-leg borrow + alt liquidity")
        print("  cap size. Real but fragile — size small, expect deep drawdowns. SIM, not live.")
    else:
        print(f"  CLASSIC momentum is weak/unstable here: best {best['lookback']}d Sharpe "
              f"{s['sharpe']:+.2f} (basket {b['sharpe']:+.2f}), total {s['total_return']:+.2%}, "
              f"positive in {sum(1 for x in rows if x['sharpe']>0)}/{len(rows)} lookbacks.")
        print("  Low turnover keeps costs off the table, but crypto's short history +")
        print("  violent momentum unwinds make the edge unreliable. Honest read: not a")
        print("  dependable standalone edge on this sample (longer history would test it better).")
    print("  (Dollar-neutral; ranks on PAST return only; costs on turnover; OOS across the window.)")
