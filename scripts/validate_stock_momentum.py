"""Validate classic 12-1 cross-sectional momentum on STOCKS (KR 50 + US watchlist).

The documented equity momentum factor, on ~10 years of daily data, run through the
SAME rigour that failed crypto: walk-forward OOS, robustness sweep, per-year /
per-regime split, quantstats tear sheet + Monte-Carlo bust probability. Reports
LONG-ONLY top-quantile (the deployable KR version — shorting restricted) vs the
index, with realistic stock costs. Cached daily CSVs, no network.

Usage:
    python scripts/validate_stock_momentum.py
    python scripts/validate_stock_momentum.py --markets kr us --train 756 --test 252
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd  # noqa: E402

from tagent.report import monte_carlo, perf_stats, tear_sheet  # noqa: E402
from tagent.stock_momentum import (  # noqa: E402
    STOCK_PPY, classic_config, load_benchmark, load_stock_panel,
)
from tagent.xs_momentum import backtest  # noqa: E402
from tagent.xs_momentum_validate import by_regime, by_year, robustness, walk_forward  # noqa: E402

PPY = STOCK_PPY
LOOKBACKS = [126, 189, 252]              # 6 / 9 / 12-month formation


def _wf_line(tag, st, n_folds=None):
    extra = f"  ({n_folds} folds)" if n_folds is not None else ""
    return (f"  {tag:16s} Sharpe {st['sharpe']:+6.2f}  ann {st['cagr']:+8.2%}  "
            f"maxDD {st['max_drawdown']:+7.2%}  total {st['total_return']:+8.2%}  n={st['n']}{extra}")


def run_market(market, args):
    panel = load_stock_panel(market, min_bars=400)
    if len(panel) < 6:
        print(f"\n[{market.upper()}] only {len(panel)} symbols with data — skipping.")
        return
    span = max(len(d) for d in panel.values())
    print(f"\n############### {market.upper()} — {len(panel)} stocks, up to {span} daily bars ###############")

    out = pathlib.Path(args.out)
    summary = {}
    for short in (False, True):
        label = "LONG-ONLY" if not short else "long-short"
        cfg = classic_config(market, allow_short=short)
        wf = walk_forward(panel, lookbacks=LOOKBACKS, train_bars=args.train,
                          test_bars=args.test, cfg=cfg, periods_per_year=PPY)
        print(f"\n=== {market.upper()} {label}: walk-forward 12-1 momentum "
              f"(train {args.train}d / test {args.test}d) ===")
        print(_wf_line(label, wf["stats"], wf["n_folds"]))
        print(f"   lookbacks chosen: {wf['lookbacks_used']}")
        summary[label] = wf

    # headline = the deployable long-only book
    wf = summary["LONG-ONLY"]
    oos = wf["oos"]
    base = classic_config(market, allow_short=False)
    full = backtest(panel, base, PPY)
    print(f"\n=== {market.upper()} LONG-ONLY vs index ===")
    print(_wf_line("MOMENTUM (OOS)", wf["stats"]))
    print(_wf_line("BASKET (EW hold)", full["basket_stats"]))
    spy = load_benchmark("SPY") if market == "us" else None
    if spy is not None:
        print(_wf_line("SPY (index)", perf_stats(spy.reindex(oos.index).dropna(), PPY)))

    # robustness (long-only)
    rob = robustness(panel, base, periods_per_year=PPY, universes=(10, 20, min(30, len(panel))),
                     quantiles=(0.2, 0.3), rebalances=(21, 42, 63), skips=(0, 21, 42),
                     costs=(5.0, 10.0, 20.0))
    print(f"\n=== {market.upper()} robustness (long-only, vary one axis) ===")
    for axis, rows in rob.items():
        print(f"  {axis:11s} " + "  ".join(
            f"{r['value']}:Sh{r['sharpe']:+.2f}/dd{r['max_drawdown']:+.0%}" for r in rows))
    rob_pos = sum(1 for rows in rob.values() for r in rows if r["sharpe"] > 0)
    rob_tot = sum(len(rows) for rows in rob.values())

    # per year + regime on the OOS
    nav = (1.0 + full["basket"]).cumprod()
    yrs = by_year(oos, PPY)
    print(f"\n=== {market.upper()} per year (long-only OOS) ===")
    for yr, v in yrs.items():
        print(f"  {yr}: total {v['total_return']:+8.2%}  Sharpe {v['sharpe']:+5.2f}  maxDD {v['max_drawdown']:+7.2%}")
    print(f"=== {market.upper()} per regime ===")
    for name, v in by_regime(oos, nav, PPY, window=63).items():
        print(f"  {name:6s}: n={v['n']:4d}  total {v['total_return']:+8.2%}  ann {v['ann_return']:+8.2%}")
    yr_pos = sum(1 for v in yrs.values() if v["total_return"] > 0)

    # risk
    bust = float("nan")
    if not oos.empty:
        mc = monte_carlo(oos, n_paths=5000, block=21, ruin_drawdown=0.5, seed=0)
        bust = mc["bust_prob"]
        print(f"\n=== {market.upper()} risk (Monte Carlo, {mc['n_paths']} paths, block 21) ===")
        print(f"  BUST prob (>=50% DD): {bust*100:.2f}%   median worst-DD {mc['worst_drawdown_med']*100:+.1f}%")
        print(f"  final equity median {mc['median_final']:,.0f}  p5 {mc['p5_final']:,.0f}  p95 {mc['p95_final']:,.0f}")
        try:
            bench = full["basket"].reindex(oos.index).fillna(0.0)
            tear_sheet(oos, out / f"stock_momentum_{market}.html", benchmark=bench,
                       title=f"{market.upper()} 12-1 momentum (long-only OOS)")
            print(f"  tear sheet -> {out / f'stock_momentum_{market}.html'}")
        except Exception as e:
            print(f"  (tear sheet skipped: {str(e)[:50]})")

    # verdict — an EDGE must (a) be positive OOS, (b) be setting-stable, (c) be broad
    # across years, AND (d) BEAT the equal-weight basket (selection adds value, not
    # just beta), on (e) an adequate universe + history.
    s, bk = wf["stats"], full["basket_stats"]
    beats_basket = s["sharpe"] > bk["sharpe"] and s["total_return"] > bk["total_return"]
    broad = yr_pos >= max(1, len(yrs) - 1) and len(yrs) >= 3
    stable = rob_pos >= 0.7 * rob_tot
    adequate = len(panel) >= 15 and span >= 252 * 4      # >=15 names, >=4y of bars
    print(f"\n=== {market.upper()} VERDICT ===")
    print(f"  Long-only walk-forward OOS: Sharpe {s['sharpe']:+.2f} (basket {bk['sharpe']:+.2f}), "
          f"ann {s['cagr']:+.2%}, maxDD {s['max_drawdown']:+.2%}.")
    print(f"  Beats basket: {beats_basket}.  Robust {rob_pos}/{rob_tot}.  "
          f"Positive {yr_pos}/{len(yrs)} years.  MC bust {bust*100:.0f}%.  "
          f"Universe {len(panel)} names / {span//252}y.")
    if s["sharpe"] > 0.3 and s["total_return"] > 0 and stable and broad and beats_basket and adequate:
        print("  -> ROBUST + BROAD edge: 12-1 momentum BEATS the equal-weight basket out-of-")
        print("     sample, stable across settings and years/regimes — the documented equity")
        print("     factor confirms here (unlike crypto). Caveats: SURVIVORSHIP bias (fixed")
        print("     today's-liquid universe overstates absolute returns), momentum CRASHES in")
        print("     reversals (the maxDD / MC tail), KR shorting limits force long-only.")
    elif not adequate:
        print("  -> INCONCLUSIVE: the universe is too small / history too short to validate a")
        print(f"     cross-sectional factor ({len(panel)} names, {span//252}y). The OOS Sharpe")
        print(f"     {'EVEN UNDERPERFORMS' if not beats_basket else 'barely beats'} the basket — with")
        print("     ~9 correlated mega-caps it's basket beta, not a selection edge. Need a broad")
        print("     universe (e.g. S&P 500) + more history to test momentum properly here.")
    elif not beats_basket:
        print("  -> NOT A SELECTION EDGE: positive OOS but it does NOT beat the equal-weight")
        print("     basket — the cross-sectional ranking adds no value over just holding the")
        print("     universe. That's beta, not momentum alpha.")
    elif s["sharpe"] > 0.3 and s["total_return"] > 0:
        print("  -> PARTIAL: beats the basket OOS but " + ("setting-stable" if stable else "broad")
              + f"; {'concentrated in a few years' if not broad else 'setting-sensitive'}. "
              "Survivorship + reversal tail still apply.")
    else:
        print("  -> WEAK OOS once best-of-sweep is removed; not a convincing standalone edge.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", nargs="+", default=["kr", "us"])
    ap.add_argument("--train", type=int, default=756)
    ap.add_argument("--test", type=int, default=252)
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()
    pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)
    for m in args.markets:
        run_market(m, args)
    print("\n(12-1 = 252d formation skipping the last 21d; monthly rebalance; walk-forward")
    print(" selects lookback on TRAIN only; costs on turnover. SURVIVORSHIP: fixed liquid")
    print(" universe -> absolute returns optimistic. Long-only is the deployable KR book.)")
