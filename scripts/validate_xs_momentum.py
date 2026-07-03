"""Rigorously validate the classic cross-sectional momentum edge before trusting it.

Walk-forward OOS (kills best-of-sweep bias) + robustness sweeps + per-year /
per-regime split + a quantstats tear sheet and Monte-Carlo bust probability.
Cached Binance daily klines (public, no key). Honest verdict at the end.

Usage:
    python scripts/validate_xs_momentum.py
    python scripts/validate_xs_momentum.py --train 365 --test 90 --rebalance weekly
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
from tagent.report import monte_carlo, tear_sheet  # noqa: E402
from tagent.xs_momentum import XSMomConfig, backtest  # noqa: E402
from tagent.xs_momentum_validate import by_regime, by_year, robustness, walk_forward  # noqa: E402

_REBAL = {"daily": 1, "weekly": 7, "monthly": 30}


def _load(sym, interval, years):
    path = os.path.join(DATA_DIR, f"klines_{sym}_{interval}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if len(df) > 30:
            return df
    df = fetch_klines(sym, interval=interval, years=years)
    df.to_csv(path)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--coins", nargs="+", default=LIQUID_COINS)
    ap.add_argument("--years", type=float, default=3.0)
    ap.add_argument("--lookbacks", nargs="+", type=int, default=[30, 60, 90])
    ap.add_argument("--train", type=int, default=365)
    ap.add_argument("--test", type=int, default=90)
    ap.add_argument("--rebalance", default="weekly")
    ap.add_argument("--skip-recent", type=int, default=7)
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--slippage-bps", type=float, default=2.0)
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()
    ppy = 365
    rebal = _REBAL.get(str(args.rebalance).lower(), 7)

    print(f"\nLoading {len(args.coins)} coins, 1d bars, ~{args.years}y (cached)...")
    per_coin = {}
    for sym in args.coins:
        try:
            df = _load(sym, "1d", args.years)
        except Exception as e:
            print(f"  {sym}: load failed ({str(e)[:40]})")
            continue
        if len(df) > 30:
            per_coin[sym] = df
    if len(per_coin) < 6:
        print("\nNeed >=6 coins with data.")
        sys.exit(0)
    print(f"Loaded {len(per_coin)} coins.\n")

    cfg = XSMomConfig(rebalance=rebal, skip_recent=args.skip_recent,
                      cost_bps=args.cost_bps, slippage_bps=args.slippage_bps)

    # 1) WALK-FORWARD out-of-sample
    wf = walk_forward(per_coin, lookbacks=args.lookbacks, train_bars=args.train,
                      test_bars=args.test, cfg=cfg, periods_per_year=ppy)
    s = wf["stats"]
    print(f"================ WALK-FORWARD OOS ({wf['n_folds']} folds, "
          f"train {args.train}d / test {args.test}d) ===")
    print(f"  combined OOS: Sharpe {s['sharpe']:+.2f}  ann {s['cagr']:+.2%}  "
          f"maxDD {s['max_drawdown']:+.2%}  total {s['total_return']:+.2%}  n={s['n']}")
    print(f"  lookbacks chosen across folds: {wf['lookbacks_used']}")
    for f in wf["folds"]:
        print(f"    {f['test_start'][:10]}..{f['test_end'][:10]}  lb={f['chosen_lookback']:>3}d  "
              f"train Sharpe {f['train_sharpe']:+.2f}  -> test {f['test_return']:+.2%}")

    oos = wf["oos"]

    # 2) ROBUSTNESS sweeps (one axis at a time, fixed mid lookback)
    base = XSMomConfig(lookback=args.lookbacks[len(args.lookbacks) // 2], rebalance=rebal,
                       skip_recent=args.skip_recent, cost_bps=args.cost_bps,
                       slippage_bps=args.slippage_bps)
    rob = robustness(per_coin, base, periods_per_year=ppy)
    print("\n================ ROBUSTNESS (vary one axis; lookback "
          f"{base.lookback}d) ===")
    for axis, rows in rob.items():
        cells = "  ".join(f"{r['value']}:Sh{r['sharpe']:+.2f}/dd{r['max_drawdown']:+.0%}" for r in rows)
        print(f"  {axis:11s} {cells}")

    # 3) sub-period: per year + per regime (on the OOS series)
    nav = (1.0 + backtest(per_coin, base, ppy)["basket"]).cumprod()
    print("\n================ PER YEAR (walk-forward OOS) ===")
    for yr, v in by_year(oos, ppy).items():
        print(f"  {yr}: total {v['total_return']:+.2%}  Sharpe {v['sharpe']:+.2f}  maxDD {v['max_drawdown']:+.2%}")
    print("================ PER REGIME (bull/quiet/bear) ===")
    for name, v in by_regime(oos, nav, ppy).items():
        print(f"  {name:6s}: n={v['n']:4d}  total {v['total_return']:+.2%}  ann {v['ann_return']:+.2%}")

    # 4) RISK: Monte Carlo bust probability + tear sheet
    mc = {"bust_prob": float("nan")}
    if not oos.empty:
        mc = monte_carlo(oos, n_paths=5000, block=10, ruin_drawdown=0.5, seed=0)
        print(f"\n================ RISK (Monte Carlo, {mc['n_paths']} paths, block 10) ===")
        print(f"  BUST prob (>=50% DD): {mc['bust_prob']*100:.2f}%   "
              f"median worst-DD {mc['worst_drawdown_med']*100:+.1f}%")
        print(f"  final equity median {mc['median_final']:,.0f}  p5 {mc['p5_final']:,.0f}  "
              f"p95 {mc['p95_final']:,.0f}  (start {mc['start_equity']:,.0f})")
        out = pathlib.Path(args.out)
        try:
            bench = backtest(per_coin, base, ppy)["basket"].reindex(oos.index).fillna(0.0)
            tear_sheet(oos, out / "xs_momentum_oos.html", benchmark=bench,
                       title="XS momentum (walk-forward OOS)")
            print(f"  tear sheet -> {out / 'xs_momentum_oos.html'}")
        except Exception as e:
            print(f"  (tear sheet skipped: {str(e)[:60]})")

    # 5) VERDICT
    rob_pos = sum(1 for rows in rob.values() for r in rows if r["sharpe"] > 0)
    rob_tot = sum(len(rows) for rows in rob.values())
    yrs = by_year(oos, ppy)
    yr_pos = sum(1 for v in yrs.values() if v["total_return"] > 0)
    bust = mc.get("bust_prob", float("nan"))
    print("\n================ VERDICT ================")
    robust = (s["sharpe"] > 0.3 and s["total_return"] > 0 and rob_pos >= 0.7 * rob_tot
              and yr_pos >= max(1, len(yrs) - 1))
    print(f"  Walk-forward OOS (no best-of-sweep): Sharpe {s['sharpe']:+.2f}, ann {s['cagr']:+.2%}, "
          f"maxDD {s['max_drawdown']:+.2%}, total {s['total_return']:+.2%}.")
    print(f"  Robustness: sign stable in {rob_pos}/{rob_tot} settings. "
          f"Breadth: positive in {yr_pos}/{len(yrs)} years. MC bust(>=50%DD): {bust*100:.0f}%.")
    if robust:
        print("  -> HOLDS UP: out-of-sample, setting-robust AND broad across years. Still")
        print("     fragile in the tail (deep DDs, fat MC) and crypto's history is short —")
        print("     real signal, size small, not live.")
    else:
        why = []
        if rob_pos < 0.7 * rob_tot:
            why.append("it hinges on specific settings")
        if yr_pos < max(1, len(yrs) - 1):
            why.append("the OOS gain is CONCENTRATED in one period/regime, not broad")
        if s["sharpe"] <= 0.3 or s["total_return"] <= 0:
            why.append("the OOS edge is weak once best-of-sweep bias is removed")
        print("  -> FRAGILE / not convincing: " + "; ".join(why) + ".")
        print("     The single-pass Sharpe overstated it; out-of-sample it shrinks and leans")
        print(f"     on a single window, with a {bust*100:.0f}% chance of a >=50% drawdown in the MC.")
        print("     Crypto's SHORT history (few independent momentum cycles) can't support")
        print("     confidence. Treat as a hypothesis, not a deployable edge.")
    print("  (Walk-forward selects on TRAIN only; costs on turnover; OOS = unseen test windows.)")
