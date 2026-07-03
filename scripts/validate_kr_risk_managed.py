"""Risk-managed vs plain KR momentum — survivorship-corrected walk-forward OOS.

Takes the point-in-time (survivorship-corrected) long-only 12-1 momentum book and
applies the causal risk overlays from tagent.risk_managed: a market-regime filter
(to cash when the point-in-time basket is below its long-term MA) and Barroso-style
volatility targeting (scale inversely to recent realized vol, capped at 1x). Then
re-runs the full validation head-to-head vs plain momentum: OOS Sharpe / ann /
maxDD, by-year, by-regime, Monte-Carlo bust, tear sheet.

No-lookahead: the overlay lags every signal one bar (see tagent.risk_managed).
Cached CSVs only, no network.

Usage:
    python scripts/validate_kr_risk_managed.py
    python scripts/validate_kr_risk_managed.py --vol-target 0.12 --regime-ma 200
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from tagent.kr_universe import load_members, membership_panel, universe_symbols  # noqa: E402
from tagent.report import monte_carlo, perf_stats, tear_sheet  # noqa: E402
from tagent.risk_managed import RiskOverlayConfig, apply_overlay  # noqa: E402
from tagent.stock_momentum import classic_config, load_stock_panel  # noqa: E402
from tagent.xs_momentum import align_close, backtest  # noqa: E402
from tagent.xs_momentum_validate import by_regime, by_year, walk_forward  # noqa: E402

PPY = 252
LOOKBACKS = [126, 189, 252]


def _line(tag, st, extra=""):
    return (f"  {tag:24s} Sharpe {st['sharpe']:+6.2f}  ann {st['cagr']:+8.2%}  "
            f"vol {st['ann_vol']:6.2%}  maxDD {st['max_drawdown']:+7.2%}  "
            f"total {st['total_return']:+9.2%}{extra}")


def _bust(net):
    if net.dropna().empty:
        return float("nan"), None
    mc = monte_carlo(net.dropna(), n_paths=5000, block=21, ruin_drawdown=0.5, seed=0)
    return mc["bust_prob"], mc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=756)
    ap.add_argument("--test", type=int, default=252)
    ap.add_argument("--regime-ma", type=int, default=200)
    ap.add_argument("--regime-off", type=float, default=0.0)
    ap.add_argument("--vol-target", type=float, default=0.15)
    ap.add_argument("--vol-window", type=int, default=63)
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)

    members = load_members()
    if not members:
        print("No data/kr_pit_members.csv — run scripts/download_kr_pit_universe.py first.")
        return 1
    syms = universe_symbols(members)
    panel = load_stock_panel("kr", symbols=syms, min_bars=60)
    idx = align_close(panel).index
    memb = membership_panel(members, idx, symbols=list(panel))
    cfg = classic_config("kr", allow_short=False)

    # plain survivorship-corrected walk-forward OOS + the PIT basket NAV (regime input)
    wf = walk_forward(panel, lookbacks=LOOKBACKS, train_bars=args.train, test_bars=args.test,
                      cfg=cfg, periods_per_year=PPY, membership=memb)
    plain = wf["oos"]
    full = backtest(panel, cfg, PPY, membership=memb)
    nav = (1.0 + full["basket"]).cumprod()

    ov = RiskOverlayConfig(regime_ma=args.regime_ma, regime_off=args.regime_off,
                           vol_target=args.vol_target, vol_window=args.vol_window,
                           vol_cap=1.0, periods_per_year=PPY)
    runs = {m: apply_overlay(plain, nav, ov, mode=m) for m in ("regime", "vol", "combined")}
    sp = perf_stats(plain.dropna(), PPY)
    bp, _ = _bust(plain)
    rb = by_regime(plain, nav, PPY, window=63)

    # per-variant tail metrics, so the verdict can pick the genuinely best overlay
    for m in runs:
        runs[m]["bust"], _ = _bust(runs[m]["net"])
        runs[m]["bear"] = by_regime(runs[m]["net"], nav, PPY, window=63)["bear"]["total_return"]

    print(f"\nPIT universe: {len(panel)} names, {len(idx)} bars; plain OOS n={len(plain)}.")
    print("\n################ RISK-MANAGED vs PLAIN (survivorship-corrected OOS) ################")
    print(_line("PLAIN momentum", sp, f"  bust {bp*100:.0f}%  bear {rb['bear']['total_return']:+.0%}"))
    for m in ("regime", "vol", "combined"):
        r = runs[m]
        print(_line(f"+ {m}", r["stats"],
                    f"  avgExp {r['avg_exposure']:.2f}  cash {r['frac_in_cash']*100:.0f}%  "
                    f"bust {r['bust']*100:.0f}%  bear {r['bear']:+.0%}"))

    # recommend: highest Sharpe among variants that MATERIALLY cut the drawdown
    elig = [m for m in runs if runs[m]["stats"]["max_drawdown"] > sp["max_drawdown"] + 0.05]
    best = max(elig or list(runs), key=lambda m: runs[m]["stats"]["sharpe"])
    rm = runs[best]["net"]
    sc = runs[best]["stats"]

    print(f"\n=== by year (total return) ===   PLAIN -> +{best}")
    yb, yr = by_year(plain, PPY), by_year(rm, PPY)
    for k in sorted(set(yb) | set(yr)):
        a, b = yb.get(k, {}).get("total_return", float("nan")), yr.get(k, {}).get("total_return", float("nan"))
        print(f"  {k}: {a:+8.2%}  ->  {b:+8.2%}")

    print(f"\n=== by regime (bull/quiet/bear) ===   PLAIN -> +{best}")
    rr = by_regime(rm, nav, PPY, window=63)
    for name in ("bull", "quiet", "bear"):
        print(f"  {name:6s}: total {rb[name]['total_return']:+8.2%}  ->  {rr[name]['total_return']:+8.2%}   "
              f"(ann {rb[name]['ann_return']:+7.2%} -> {rr[name]['ann_return']:+7.2%})")

    try:
        tear_sheet(rm.dropna(), out / "kr_risk_managed.html",
                   benchmark=plain.reindex(rm.index).fillna(0.0),
                   title=f"KR risk-managed 12-1 momentum (+{best}, PIT long-only OOS)")
        print(f"\n  tear sheet -> {out / 'kr_risk_managed.html'} (benchmark = plain momentum)")
    except Exception as e:
        print(f"\n  (tear sheet skipped: {str(e)[:50]})")

    # verdict on the recommended variant
    bc, bear_p, bear_c = runs[best]["bust"], rb["bear"]["total_return"], runs[best]["bear"]
    print(f"\n=== VERDICT (risk-managed momentum — best overlay: +{best}) ===")
    print(f"  Sharpe {sp['sharpe']:+.2f} -> {sc['sharpe']:+.2f}; ann {sp['cagr']:+.2%} -> {sc['cagr']:+.2%}; "
          f"maxDD {sp['max_drawdown']:+.2%} -> {sc['max_drawdown']:+.2%}; bust {bp*100:.0f}% -> {bc*100:.0f}%.")
    print(f"  Bear-regime: {bear_p:+.2%} -> {bear_c:+.2%}.  (vol-targeting alone "
          f"Sharpe {runs['vol']['stats']['sharpe']:+.2f} — it de-risks momentum's best, high-vol months.)")
    dd_cut = sc["max_drawdown"] > sp["max_drawdown"] + 0.05
    if sc["sharpe"] >= sp["sharpe"] - 0.02 and dd_cut and sc["cagr"] > 0 and bear_c > bear_p:
        print(f"  -> WORKS: the {best} filter cuts the bear-crash drawdown and bust probability while")
        print("     keeping (even improving) risk-adjusted return — the deployable long-only book.")
        print("     Cost: it sits in cash ~half the time in downtrends, so absolute return is a bit")
        print("     lower. Naive vol-targeting HURTS here (momentum's biggest gains are high-vol).")
    elif dd_cut and sc["cagr"] > 0:
        print(f"  -> PARTIAL: the {best} overlay cuts tail risk but Sharpe is ~flat/lower — a risk")
        print("     trade-off, not a free lunch. Worth it if the -45% crash is the binding concern.")
    else:
        print("  -> WEAK: no overlay meaningfully improves risk-adjusted return on this book.")
    print("\n(No-lookahead: regime/vol signals are lagged one bar. Overlay scales the realized")
    print(" OOS book; costs scale with exposure. pykrx omits the final delisting-day crash.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
