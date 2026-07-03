"""Survivorship-corrected KR momentum: point-in-time universe vs the 50-name set.

The 50-name study inflates momentum because the universe is *today's* liquid
names. Here we re-run the SAME classic 12-1 long-only momentum on the
point-in-time top-100-by-market-cap universe (313 distinct names ever in the
top-100, 2016->today, including since-delisted), gating selection with the
as-of-date membership mask, and COMPARE the two head to head with the full rigour
(walk-forward OOS, robustness, by-year/regime, quantstats + Monte-Carlo).

Needs data/kr_pit_members.csv + the union OHLCV (scripts/download_kr_pit_universe.py).
Cached CSVs, no network.

Usage:
    python scripts/validate_kr_pit_momentum.py
    python scripts/validate_kr_pit_momentum.py --train 756 --test 252 --top-n 100
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

from tagent.kr_universe import load_members, membership_panel, universe_symbols  # noqa: E402
from tagent.report import monte_carlo, perf_stats, tear_sheet  # noqa: E402
from tagent.stock_momentum import classic_config, load_stock_panel  # noqa: E402
from tagent.xs_momentum import align_close, backtest  # noqa: E402
from tagent.xs_momentum_validate import by_regime, by_year, robustness, walk_forward  # noqa: E402

PPY = 252
LOOKBACKS = [126, 189, 252]              # 6 / 9 / 12-month formation

# today's liquid 50 (the inflated baseline universe)
from scripts.download_krx_universe import DEFAULT_TICKERS as FIFTY  # noqa: E402


def _wf_line(tag, st):
    return (f"  {tag:22s} Sharpe {st['sharpe']:+6.2f}  ann {st['cagr']:+8.2%}  "
            f"maxDD {st['max_drawdown']:+7.2%}  total {st['total_return']:+9.2%}  n={st['n']}")


def _run_wf(panel, membership, train, test):
    cfg = classic_config("kr", allow_short=False)
    return walk_forward(panel, lookbacks=LOOKBACKS, train_bars=train, test_bars=test,
                        cfg=cfg, periods_per_year=PPY, membership=membership)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=756)
    ap.add_argument("--test", type=int, default=252)
    ap.add_argument("--top-n", type=int, default=100)
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # ---- 1) INFLATED baseline: today's 50 liquid names, no membership ----------
    base_panel = load_stock_panel("kr", symbols=FIFTY, min_bars=400)
    wf_base = _run_wf(base_panel, None, args.train, args.test)
    full_base = backtest(base_panel, classic_config("kr", allow_short=False), PPY)

    # ---- 2) CORRECTED: point-in-time universe + as-of membership mask ----------
    members = load_members()
    if not members:
        print("No data/kr_pit_members.csv — run scripts/download_kr_pit_universe.py first.")
        return 1
    syms = universe_symbols(members)
    pit_panel = load_stock_panel("kr", symbols=syms, min_bars=60)   # keep short-lived/delisted
    idx = align_close(pit_panel).index
    memb = membership_panel(members, idx, symbols=list(pit_panel))
    # only names that actually appear as members AND have prices contribute
    print(f"PIT universe: {len(pit_panel)} names with OHLCV of {len(syms)} ever-members; "
          f"{memb.any(axis=0).sum()} ever eligible over {len(idx)} bars.")

    wf_pit = _run_wf(pit_panel, memb, args.train, args.test)
    full_pit = backtest(pit_panel, classic_config("kr", allow_short=False), PPY, membership=memb)

    # ---- 3) head-to-head -------------------------------------------------------
    print("\n################ SURVIVORSHIP: inflated 50-name vs point-in-time ################")
    print("Long-only 12-1 momentum, walk-forward OOS (train "
          f"{args.train}d / test {args.test}d), identical method:")
    print(_wf_line("INFLATED (today's 50)", wf_base["stats"]))
    print(_wf_line("CORRECTED (PIT top-N)", wf_pit["stats"]))
    print(_wf_line("  PIT basket (EW memb)", full_pit["basket_stats"]))
    print(_wf_line("  50 basket (EW hold)", full_base["basket_stats"]))
    sb, sp = wf_base["stats"], wf_pit["stats"]
    keep = (sp["sharpe"] / sb["sharpe"]) if sb["sharpe"] else float("nan")
    print(f"\n  Survivorship haircut: Sharpe {sb['sharpe']:+.2f} -> {sp['sharpe']:+.2f} "
          f"({keep*100:.0f}% retained); ann {sb['cagr']:+.2%} -> {sp['cagr']:+.2%}; "
          f"total {sb['total_return']:+.2%} -> {sp['total_return']:+.2%}.")

    # ---- 4) full rigour on the CORRECTED book ---------------------------------
    oos = wf_pit["oos"]
    base_cfg = classic_config("kr", allow_short=False)
    rob = robustness(pit_panel, base_cfg, periods_per_year=PPY,
                     universes=(20, 50, min(100, len(pit_panel))), quantiles=(0.2, 0.3),
                     rebalances=(21, 42, 63), skips=(0, 21, 42), costs=(5.0, 10.0, 20.0),
                     membership=memb)
    print("\n=== CORRECTED robustness (long-only, vary one axis) ===")
    for axis, rows in rob.items():
        print(f"  {axis:11s} " + "  ".join(
            f"{r['value']}:Sh{r['sharpe']:+.2f}/dd{r['max_drawdown']:+.0%}" for r in rows))
    rob_pos = sum(1 for rows in rob.values() for r in rows if r["sharpe"] > 0)
    rob_tot = sum(len(rows) for rows in rob.values())

    nav = (1.0 + full_pit["basket"]).cumprod()
    yrs = by_year(oos, PPY)
    print("\n=== CORRECTED per year (long-only OOS) ===")
    for yr, v in yrs.items():
        print(f"  {yr}: total {v['total_return']:+8.2%}  Sharpe {v['sharpe']:+5.2f}  maxDD {v['max_drawdown']:+7.2%}")
    print("=== CORRECTED per regime ===")
    for name, v in by_regime(oos, nav, PPY, window=63).items():
        print(f"  {name:6s}: n={v['n']:4d}  total {v['total_return']:+8.2%}  ann {v['ann_return']:+8.2%}")
    yr_pos = sum(1 for v in yrs.values() if v["total_return"] > 0)

    bust = float("nan")
    if not oos.empty:
        mc = monte_carlo(oos, n_paths=5000, block=21, ruin_drawdown=0.5, seed=0)
        bust = mc["bust_prob"]
        print(f"\n=== CORRECTED risk (Monte Carlo, {mc['n_paths']} paths, block 21) ===")
        print(f"  BUST prob (>=50% DD): {bust*100:.2f}%   median worst-DD {mc['worst_drawdown_med']*100:+.1f}%")
        print(f"  final equity median {mc['median_final']:,.0f}  p5 {mc['p5_final']:,.0f}  p95 {mc['p95_final']:,.0f}")
        try:
            bench = full_pit["basket"].reindex(oos.index).fillna(0.0)
            tear_sheet(oos, out / "kr_pit_momentum.html", benchmark=bench,
                       title="KR 12-1 momentum (point-in-time, long-only OOS)")
            print(f"  tear sheet -> {out / 'kr_pit_momentum.html'}")
        except Exception as e:
            print(f"  (tear sheet skipped: {str(e)[:50]})")

    # ---- 5) verdict ------------------------------------------------------------
    bk = full_pit["basket_stats"]
    beats_basket = sp["sharpe"] > bk["sharpe"] and sp["total_return"] > bk["total_return"]
    broad = yr_pos >= max(1, len(yrs) - 1) and len(yrs) >= 3
    stable = rob_pos >= 0.7 * rob_tot
    print("\n=== VERDICT (survivorship-corrected, deployable long-only) ===")
    print(_wf_line("DEPLOYABLE (PIT OOS)", sp))
    print(f"  Beats PIT basket: {beats_basket} (basket Sharpe {bk['sharpe']:+.2f}). "
          f"Robust {rob_pos}/{rob_tot}. Positive {yr_pos}/{len(yrs)} yrs. MC bust {bust*100:.0f}%.")
    print(f"  Survivorship-corrected Sharpe {sp['sharpe']:+.2f} / ann {sp['cagr']:+.2%} is the "
          f"realistic number ({keep*100:.0f}% of the inflated {sb['sharpe']:+.2f}).")
    if sp["sharpe"] > 0.3 and sp["total_return"] > 0 and stable and broad and beats_basket:
        print("  -> EDGE SURVIVES the correction: still beats the (point-in-time) basket OOS,")
        print("     stable across settings and broad across years on a 300+ name universe — a")
        print("     genuine, broad equity-momentum factor, not a 50-name artifact. Smaller than")
        print("     the inflated headline; deploy at the corrected size, long-only (KR shorting).")
    elif sp["sharpe"] > 0.3 and sp["total_return"] > 0:
        print("  -> PARTIAL after correction: positive OOS but " +
              ("not setting-stable" if not stable else "narrow across years" if not broad
               else "does NOT beat the point-in-time basket") + " — much of the headline was")
        print("     survivorship. Treat the corrected number as the ceiling.")
    else:
        print("  -> MOSTLY SURVIVORSHIP: once the point-in-time universe includes the names")
        print("     that fell out / delisted, the edge largely collapses. Not deployable as-is.")
    print("\n(Caveat: pykrx OHLCV ends at a name's last trading day, so the final delisting")
    print(" crash isn't booked — the corrected number is still mildly optimistic on that tail.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
