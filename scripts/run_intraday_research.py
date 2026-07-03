"""Intraday research — Hypothesis A: overnight-overreaction gap-down reversal.

Runs the gap-reversal strategy across symbols and years on minute bars, NET of
realistic KR round-trip costs, and reports win rate, avg net per trade, expectancy,
profit factor, max drawdown, trades/day, and a by-year breakdown — then states
plainly whether expectancy beats the cost hurdle (if not, move to Hypothesis B).

Data: cached data/<SYMBOL>_1m.csv minute bars (real KR minute data is Kiwoom-gated,
so this env may have none) + the US overnight return from cached SPY_1d.csv.
``--synthetic`` runs a LABELLED random-walk demo to exercise the pipeline (no edge
by construction — it shows the reporting + the honest negative verdict path).

Usage:
    python scripts/run_intraday_research.py
    python scripts/run_intraday_research.py --synthetic
    python scripts/run_intraday_research.py --gap 0.02 --take 0.01 --stop 0.01
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tagent.data.intraday_history import (  # noqa: E402
    available_minute_symbols, load_intraday, load_spy_daily_returns,
    make_synthetic_minutes, overnight_for_dates,
)
from tagent.intraday_backtest import (  # noqa: E402
    CostModel, by_year, metrics, run, walk_forward,
)
from tagent.strategies.gap_reversal import GapReversalParams, default_grid, make_strategy
from tagent.strategies import opening_range as orb  # noqa: E402
from tagent.strategies import vwap_reversion as vwr  # noqa: E402


def _all_dates(panels):
    ds = set()
    for df in panels.values():
        ds |= {ts.date() for ts in df.index.normalize().unique()}
    return sorted(ds)


def _synthetic_panels(n_symbols=8, n_days=140, seed=0):
    """Labelled synthetic data: random-walk minutes with overnight-driven gaps."""
    rng = np.random.default_rng(seed)
    days = [d.date() for d in pd.bdate_range("2023-01-02", periods=n_days)]
    overnight = {d: float(rng.normal(0.0, 0.012)) for d in days}     # synthetic US overnight
    panels = {f"SYN{i:02d}": make_synthetic_minutes(n_days=n_days, seed=seed + i,
                                                    overnight=overnight)
              for i in range(n_symbols)}
    return panels, overnight


def _run_pooled(panels, strat, cost, overnight):
    trades = []
    for s, df in panels.items():
        trades += run(df, strat, cost=cost, overnight=overnight, symbol=s)
    return trades


def _compare(panels, overnight, cost, hurdle, n_days, synthetic) -> int:
    """Run Hypotheses A/B/C side by side, net of cost, over the same symbols/dates."""
    strategies = [
        ("A gap_reversal", make_strategy(GapReversalParams())),
        ("B opening_range", orb.make_strategy(orb.OpeningRangeParams(open_minutes=30))),
        ("C vwap_reversion", vwr.make_strategy(vwr.VwapReversionParams(dev_pct=0.01))),
    ]
    print(f"\n################ A/B/C SIDE-BY-SIDE (net of {hurdle*100:.3f}% cost) ################")
    print(f"  {'strategy':18s} {'trades':>6s} {'win':>6s} {'gross/t':>9s} "
          f"{'NET/t':>9s} {'PF':>5s} {'total':>9s}  beats?")
    rows = []
    for name, strat in strategies:
        m = metrics(_run_pooled(panels, strat, cost, overnight), n_days=n_days)
        beats = "YES" if (m["n_trades"] and m["expectancy"] > 0) else "no"
        rows.append((name, m, beats))
        print(f"  {name:18s} {m['n_trades']:6d} {m['win_rate']:6.0%} "
              f"{m['gross_expectancy']*100:+8.3f}% {m['expectancy']*100:+8.3f}% "
              f"{m['profit_factor']:5.2f} {m['total_net']*100:+8.2f}%  {beats}")

    traded = [r for r in rows if r[1]["n_trades"] > 0]
    print("\n=== VERDICT (A/B/C) ===")
    if synthetic:
        print("  SYNTHETIC data — signs are noise; this only checks the machinery + cost hurdle.")
    if not traded:
        print("  No qualifying trades for any hypothesis on this sample.")
        return 0
    best = max(traded, key=lambda r: r[1]["expectancy"])
    bm = best[1]
    total_trades = sum(r[1]["n_trades"] for r in rows)
    if total_trades < 20 or n_days < 20:
        print(f"  ⚠ SMALL SAMPLE: {total_trades} trades across {n_days} days — DIRECTIONAL ONLY,")
        print("  NOT a verdict. A real conclusion needs multi-year vendor data.")
    verb = "BEATS" if bm["expectancy"] > 0 else "is closest to clearing"
    print(f"  Most promising: {best[0]} — net {bm['expectancy']*100:+.3f}%/trade "
          f"(gross {bm['gross_expectancy']*100:+.3f}% vs {hurdle*100:.3f}% hurdle), "
          f"win {bm['win_rate']:.0%}, {bm['n_trades']} trades; it {verb} the hurdle.")
    print("  -> pursue this hypothesis FIRST with multi-year vendor data; re-test the others too.")
    print("\n(No-lookahead: signals use only past/intra bars; exits simulated forward. "
          "Costs: 0.18% KR sell tax + fees + slippage.)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--gap", type=float, default=0.02)
    ap.add_argument("--take", type=float, default=0.01)
    ap.add_argument("--stop", type=float, default=0.01)
    ap.add_argument("--entry-minute", type=int, default=5)
    ap.add_argument("--overnight-thresh", type=float, default=-0.005)
    ap.add_argument("--slippage-bps", type=float, default=10.0)
    ap.add_argument("--train-months", type=int, default=6)
    ap.add_argument("--test-months", type=int, default=1)
    ap.add_argument("--list-trades", action="store_true", help="print every qualifying trade")
    ap.add_argument("--compare", action="store_true",
                    help="run A/B/C (gap_reversal / opening_range / vwap_reversion) side by side")
    args = ap.parse_args()

    cost = CostModel(slippage_bps=args.slippage_bps)
    hurdle = cost.round_trip_frac()
    params = GapReversalParams(gap_pct=args.gap, take_pct=args.take, stop_pct=args.stop,
                               entry_minute=args.entry_minute,
                               overnight_thresh=args.overnight_thresh)

    if args.synthetic:
        print("### SYNTHETIC random-walk demo — NOT a real edge (pipeline check only) ###")
        panels, overnight = _synthetic_panels()
    else:
        syms = args.symbols or available_minute_symbols()
        if not syms:
            print("No data/<SYMBOL>_1m.csv minute bars found.")
            print("Real KR minute data is Kiwoom/vendor-gated; drop CSVs in data/ or use --synthetic.")
            return 0
        panels = {s: load_intraday(s) for s in syms}
        panels = {s: df for s, df in panels.items() if not df.empty}
        if not panels:
            print("Minute CSVs found but empty. Nothing to backtest.")
            return 0
        overnight = overnight_for_dates(load_spy_daily_returns(), _all_dates(panels))

    n_days = len(_all_dates(panels))
    print(f"\nUniverse: {len(panels)} symbols, {n_days} trading days. "
          f"Round-trip cost hurdle: {hurdle*100:.3f}% "
          f"(fees {cost.fee_bps}bps×2 + slip {cost.slippage_bps}bps×2 + tax {cost.sell_tax_bps}bps).")

    if args.compare:
        return _compare(panels, overnight, cost, hurdle, n_days, args.synthetic)

    print(f"Params: gap<=-{params.gap_pct:.1%}, overnight<={params.overnight_thresh:.2%}, "
          f"+{params.take_pct:.1%} take / -{params.stop_pct:.1%} stop, entry +{params.entry_minute}min.")

    # 1) fixed-params, pooled across symbols (+ per-symbol qualifying-trade counts)
    strat = make_strategy(params)
    all_trades = []
    per_symbol = {}
    for s, df in panels.items():
        ts = run(df, strat, cost=cost, overnight=overnight, symbol=s)
        per_symbol[s] = ts
        all_trades += ts
    m = metrics(all_trades, n_days=n_days)
    print(f"\n=== FIXED-PARAMS (pooled, net of costs) ===")
    _print_metrics(m, hurdle)
    print("\n  qualifying trades per symbol:")
    for s in panels:
        sm = metrics(per_symbol[s])
        print(f"    {s}: {sm['n_trades']:3d} trades  "
              f"{'win '+format(sm['win_rate'],'.0%')+'  exp '+format(sm['expectancy']*100,'+.3f')+'%' if sm['n_trades'] else ''}")
    if args.list_trades and all_trades:
        print("\n  per-trade list (date  symbol  entry->exit  gross / NET  reason):")
        for t in sorted(all_trades, key=lambda x: (x.date, x.symbol)):
            print(f"    {t.date}  {t.symbol}  {t.entry_time[11:16]}->{t.exit_time[11:16]}  "
                  f"{t.entry_price:,.0f}->{t.exit_price:,.0f}  "
                  f"gross {t.gross*100:+.3f}% / net {t.net*100:+.3f}%  {t.reason}")
    print("\n  by year:")
    for yr, v in by_year(all_trades).items():
        print(f"    {yr}: n={v['n_trades']:4d}  win {v['win_rate']:5.1%}  "
              f"exp {v['expectancy']*100:+.3f}%  PF {v['profit_factor']:.2f}  "
              f"maxDD {v['max_drawdown']:+.1%}")

    # 2) walk-forward (params chosen on train months only)
    grid = default_grid()
    wf_trades = []
    for s, df in panels.items():
        wf = walk_forward(df, make_strategy, grid, cost=cost, overnight=overnight,
                          train_months=args.train_months, test_months=args.test_months, symbol=s)
        wf_trades += wf["oos"]
    wm = metrics(wf_trades, n_days=n_days)
    print(f"\n=== WALK-FORWARD OOS (params selected on train months) ===")
    _print_metrics(wm, hurdle)

    # 3) honest verdict
    print("\n=== VERDICT (Hypothesis A: overnight-overreaction gap reversal) ===")
    fixed_ok = m["n_trades"] > 0 and m["expectancy"] > 0
    wf_ok = wm["n_trades"] > 0 and wm["expectancy"] > 0
    SMALL = 20
    if 0 < m["n_trades"] < SMALL:
        print(f"  ⚠ SMALL SAMPLE: only {m['n_trades']} qualifying trades over {n_days} days — this is")
        print("  DIRECTIONAL ONLY, NOT a verdict. A real conclusion needs multi-year vendor data.")
    if args.synthetic:
        print("  SYNTHETIC data — any sign here is noise; this only proves the machinery + the")
        print("  cost hurdle is enforced. Use real minute bars for a real verdict.")
    if not all_trades:
        print("  No qualifying gap-down/overnight-driven days — hypothesis untestable on this sample.")
    elif wf_ok and fixed_ok:
        print(f"  Net expectancy BEATS the cost hurdle: fixed {m['expectancy']*100:+.3f}%/trade, "
              f"walk-forward {wm['expectancy']*100:+.3f}%/trade. Worth deeper validation "
              "(more symbols/years, robustness, slippage stress) before trusting.")
    else:
        print(f"  Net expectancy does NOT beat the cost hurdle "
              f"(fixed {m['expectancy']*100:+.3f}%/trade, walk-forward {wm['expectancy']*100:+.3f}%/trade).")
        print(f"  Gross was {m['gross_expectancy']*100:+.3f}%/trade vs a {hurdle*100:.3f}% hurdle — "
              "costs eat it. Hypothesis A is NOT a net edge here; move to Hypothesis B.")
    print("\n(No-lookahead: gap/overnight known pre-entry; exit simulated forward; walk-forward")
    print(" selects params on train months only. Costs: 0.18% KR sell tax + fees + slippage.)")
    return 0


def _print_metrics(m, hurdle):
    print(f"  trades {m['n_trades']}  ·  win rate {m['win_rate']:.1%}  ·  trades/day {m['trades_per_day']:.2f}")
    print(f"  avg NET/trade (expectancy) {m['expectancy']*100:+.4f}%   "
          f"(gross {m['gross_expectancy']*100:+.4f}%, hurdle {hurdle*100:.3f}%)")
    print(f"  profit factor {m['profit_factor']:.2f}   max drawdown {m['max_drawdown']:+.2%}   "
          f"total net {m['total_net']*100:+.2f}%")
    verdict = "BEATS" if m["expectancy"] > 0 else "below"
    print(f"  -> expectancy {verdict} the cost hurdle")


if __name__ == "__main__":
    raise SystemExit(main())
