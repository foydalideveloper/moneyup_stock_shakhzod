"""Value / Quality / Momentum / Multi-factor on KR stocks — same rigor as momentum.

Survivorship-corrected (point-in-time membership), NET of realistic KR costs, with
the market-regime filter (cash in downtrends), walk-forward OOS, by-year breakdown,
and Monte-Carlo bust probability — reported SIDE BY SIDE vs momentum and the
equal-weight basket.

Needs the cached PIT universe (download_kr_pit_universe.py), its OHLCV, and the
fundamentals (download_kr_fundamentals.py). Cached CSVs, no network.

Usage:
    python scripts/validate_kr_factors.py
    python scripts/validate_kr_factors.py --train 756 --test 252
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from tagent.data.kr_fundamental import load_fundamentals, metric_panel  # noqa: E402
from tagent.kr_universe import load_members, membership_panel, universe_symbols  # noqa: E402
from tagent.report import monte_carlo, perf_stats  # noqa: E402
from tagent.risk_managed import RiskOverlayConfig, apply_overlay  # noqa: E402
from tagent.stock_momentum import classic_config, load_stock_panel  # noqa: E402
from tagent.strategies.multi_factor import combine_signals, momentum_signal_panel  # noqa: E402
from tagent.strategies.quality_factor import quality_signal  # noqa: E402
from tagent.strategies.value_factor import value_signal  # noqa: E402
from tagent.xs_momentum import align_close, backtest  # noqa: E402
from tagent.xs_momentum_validate import by_year, walk_forward_signal  # noqa: E402

PPY = 252


def _bust(net):
    if net.dropna().empty:
        return float("nan")
    return monte_carlo(net.dropna(), n_paths=5000, block=21, ruin_drawdown=0.5, seed=0)["bust_prob"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=756)
    ap.add_argument("--test", type=int, default=252)
    ap.add_argument("--regime-ma", type=int, default=200)
    args = ap.parse_args()

    members = load_members()
    if not members:
        print("No PIT membership — run scripts/download_kr_pit_universe.py first.")
        return 1
    syms = universe_symbols(members)
    panel = load_stock_panel("kr", symbols=syms, min_bars=60)
    idx = align_close(panel).index
    memb = membership_panel(members, idx, symbols=list(panel))
    cols = list(panel)

    fund = load_fundamentals()
    if fund.empty:
        print("No data/kr_fundamentals.csv — run scripts/download_kr_fundamentals.py first.")
        return 1
    pbr = metric_panel(fund, "PBR", idx, symbols=cols)
    per = metric_panel(fund, "PER", idx, symbols=cols)
    eps = metric_panel(fund, "EPS", idx, symbols=cols)
    bps = metric_panel(fund, "BPS", idx, symbols=cols)
    close = align_close(panel)

    val = value_signal(pbr=pbr, per=per)
    qual = quality_signal(eps, bps)
    mom = momentum_signal_panel(close, 252, 21)
    multi = combine_signals([val, qual, mom])

    cfg = classic_config("kr", allow_short=False)            # KR costs, long-only, monthly, top 20%
    ov = RiskOverlayConfig(regime_ma=args.regime_ma, regime_off=0.0, vol_cap=1.0,
                           periods_per_year=PPY)

    # the survivorship-corrected EW basket (same for all — members only)
    base = backtest(panel, cfg, PPY, membership=memb)
    bstats = base["basket_stats"]
    nav = (1.0 + base["basket"]).cumprod()

    strategies = {"VALUE": val, "QUALITY": qual, "MOMENTUM": mom, "MULTI(V+Q+M)": multi}
    print(f"\nUniverse: {len(panel)} PIT names, {len(idx)} bars. Long-only top-20%, monthly, "
          f"NET of KR costs (sell tax+fees+slippage), regime filter (cash < {args.regime_ma}d MA).")
    print(f"\n{'strategy':14s} {'Sharpe':>7s} {'CAGR':>8s} {'maxDD':>8s} {'bust':>6s}  "
          f"{'+regime Sharpe':>14s} {'CAGR':>8s} {'maxDD':>8s} {'bust':>6s}  beats basket?")

    results = {}
    for name, sig in strategies.items():
        r = backtest(panel, cfg, PPY, membership=memb, signal=sig)
        net = r["net"]
        s = r["stats"]
        rm = apply_overlay(net, nav, ov, mode="regime")
        sr = rm["stats"]
        beats = "YES" if s["sharpe"] > bstats["sharpe"] and s["total_return"] > bstats["total_return"] else "no"
        results[name] = {"plain": s, "regime": sr, "net": net, "rm_net": rm["net"]}
        print(f"{name:14s} {s['sharpe']:+7.2f} {s['cagr']:+8.2%} {s['max_drawdown']:+8.2%} "
              f"{_bust(net)*100:5.0f}%  {sr['sharpe']:+14.2f} {sr['cagr']:+8.2%} {sr['max_drawdown']:+8.2%} "
              f"{_bust(rm['net'])*100:5.0f}%  {beats}")
    print(f"{'BASKET(EW)':14s} {bstats['sharpe']:+7.2f} {bstats['cagr']:+8.2%} "
          f"{bstats['max_drawdown']:+8.2%} {_bust(base['basket'])*100:5.0f}%")

    # walk-forward OOS (sweep top_q on train) for the fundamental factors + multi
    print(f"\n=== WALK-FORWARD OOS (top_q chosen on train {args.train}d / test {args.test}d) ===")
    wf_stats = {}
    for name in ("VALUE", "QUALITY", "MULTI(V+Q+M)"):
        wf = walk_forward_signal(panel, strategies[name], top_qs=(0.1, 0.2, 0.3),
                                 train_bars=args.train, test_bars=args.test, cfg=cfg,
                                 periods_per_year=PPY, membership=memb)
        wf_stats[name] = wf["stats"]
        s = wf["stats"]
        print(f"  {name:14s} Sharpe {s['sharpe']:+.2f}  CAGR {s['cagr']:+.2%}  "
              f"maxDD {s['max_drawdown']:+.2%}  n={s['n']}  top_qs={wf['top_qs_used']}")

    # by-year on the regime-managed books
    print("\n=== by year (regime-managed, total return) ===")
    yrs = sorted({d.year for d in idx})
    hdr = "  year   " + "".join(f"{n[:9]:>11s}" for n in strategies)
    print(hdr)
    by = {n: by_year(results[n]["rm_net"], PPY) for n in strategies}
    for y in yrs:
        cells = "".join(f"{by[n].get(str(y), {}).get('total_return', float('nan')):>+11.1%}"
                        for n in strategies)
        print(f"  {y}  {cells}")

    # verdict
    print("\n=== VERDICT (KR cross-sectional factors, survivorship-corrected, net of cost) ===")
    mom_s = results["MOMENTUM"]["plain"]
    ranked = sorted(strategies, key=lambda n: results[n]["plain"]["sharpe"], reverse=True)
    print(f"  Sharpe ranking: " + " > ".join(f"{n} {results[n]['plain']['sharpe']:+.2f}" for n in ranked))
    cleared = [n for n in ("VALUE", "QUALITY") if results[n]["plain"]["sharpe"] > 0.3
               and results[n]["plain"]["total_return"] > bstats["total_return"]]
    print(f"  Factors clearing the bar (Sharpe>0.3 AND beat the basket): {cleared or 'NONE'}.")
    multi_s = results["MULTI(V+Q+M)"]["plain"]
    if multi_s["sharpe"] > mom_s["sharpe"] + 0.05:
        print(f"  MULTI ({multi_s['sharpe']:+.2f}) BEATS momentum alone ({mom_s['sharpe']:+.2f}) "
              "— the blend adds value.")
    else:
        print(f"  MULTI ({multi_s['sharpe']:+.2f}) does NOT beat momentum alone "
              f"({mom_s['sharpe']:+.2f}) on this sample — momentum dominates the blend.")
    print("\n(No-lookahead: fundamentals are as-of monthly snapshots; signals z-scored "
          "cross-sectionally; PIT membership; regime/costs as in the momentum study.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
