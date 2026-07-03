"""Deep US-lead-lag study — KR after a sharp US down-night, LONG history (2001+).

Uses data/leadlag/ (download_leadlag_history.py): long-lived KR large-caps + US
signals (SPY broad, SOXX semis). Sweeps US-drop thresholds and reports, per threshold:
event counts, win rate, NET expectancy/trade, t-stat, EW maxDD — both EQUAL-WEIGHT
(one trade per event day) and PER-STOCK (one obs per stock-event, far more N) — plus a
by-year breakdown and a CHIP variant (KR semis after a SOXX down-night). Net of the
~0.41% KR round-trip cost. Loud about sample size.

Survivorship caveat: a FIXED basket of today's long-lived large caps (no point-in-time
membership pre-2016), so absolute levels are optimistic; the bounce PATTERN is the focus.

Usage:
    python scripts/validate_kr_leadlag_deep.py
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from tagent.data.intraday_history import load_spy_daily_returns  # noqa: E402
from tagent.intraday_backtest import CostModel  # noqa: E402
from tagent.report import perf_stats  # noqa: E402
from tagent.stock_momentum import load_stock_panel  # noqa: E402
from tagent.strategies.us_leadlag_daily import (  # noqa: E402
    event_day_returns, event_stock_returns, summarize,
)
from tagent.kr_universe import load_members, membership_panel, universe_symbols  # noqa: E402
from tagent.xs_momentum import align_close  # noqa: E402
from scripts.download_leadlag_history import KR_BASKET, KR_CHIP, LEADLAG_DIR  # noqa: E402

# Conservative round-trip cost scenarios via slippage (round trip = 21 + 2*slip bps).
COST_SCENARIOS = {"0.41%": CostModel(slippage_bps=10), "0.51%": CostModel(slippage_bps=15),
                  "0.71%": CostModel(slippage_bps=25)}

THRESHOLDS = (-0.01, -0.015, -0.02, -0.025, -0.03)
TARGET_N = 50                     # "meaningful sample" floor for per-stock observations


def _variant(title, panel, signal, sig_name, cf):
    span = align_close(panel).index
    print(f"\n=== {title}  (signal: {sig_name}; {len(panel)} names; "
          f"{span.min().date()} -> {span.max().date()}) ===")
    print(f"  {'thr':>6s} | {'EWdays':>6s} {'win':>5s} {'net/day':>9s} {'maxDD':>7s} | "
          f"{'PSn':>5s} {'win':>5s} {'net/obs':>9s} {'tstat':>6s}  clears?")
    rows = {}
    for x in THRESHOLDS:
        ew = event_day_returns(panel, signal, threshold=x) - cf
        ps = event_stock_returns(panel, signal, threshold=x)
        pss = summarize(ps, cf)
        ewdd = perf_stats(ew.dropna(), 252)["max_drawdown"] if len(ew.dropna()) else 0.0
        clears = "YES" if (pss["net_exp"] > 0 and pss["n"] >= TARGET_N and pss["tstat"] >= 1.5) else \
                 ("thin" if pss["net_exp"] > 0 else "no")
        rows[x] = pss
        ewwin = float((ew > 0).mean()) if len(ew.dropna()) else 0.0
        print(f"  {x*100:+5.1f}% | {len(ew.dropna()):6d} {ewwin:5.0%} {ew.mean()*100:+8.3f}% "
              f"{ewdd:+7.1%} | {pss['n']:5d} {pss['win']:5.0%} {pss['net_exp']*100:+8.3f}% "
              f"{pss['tstat']:+6.2f}  {clears}")
    return rows


def _by_year(panel, signal, threshold, cf):
    ps = event_stock_returns(panel, signal, threshold=threshold)
    if ps.empty:
        return
    net = ps - cf
    print(f"\n  by year (per-stock net, threshold {threshold*100:+.1f}%):")
    yrs = sorted({ts.year for ts, _ in net.index})
    for y in yrs:
        seg = net[[ts.year == y for ts, _ in net.index]]
        if len(seg):
            tag = " <-shock" if y in (2008, 2011, 2018, 2020, 2022) else ""
            print(f"    {y}: n={len(seg):4d}  net/obs {seg.mean()*100:+.3f}%  win {(seg>0).mean():.0%}{tag}")


def _pit_cost_sweep(spy):
    """Survivorship-corrected (point-in-time, 2016+) lead-lag under CONSERVATIVE cost.
    Reports per-stock net expectancy/trade by threshold under each round-trip cost."""
    members = load_members()
    if not members:
        print("\n(no PIT membership cached — skip the survivorship-corrected sweep)")
        return
    syms = universe_symbols(members)
    panel = load_stock_panel("kr", symbols=syms, fields=["open", "close"], min_bars=60)
    if not panel:
        print("\n(no PIT OHLCV cached — skip)")
        return
    idx = align_close(panel).index
    memb = membership_panel(members, idx, symbols=list(panel))
    print(f"\n=== PIT SURVIVORSHIP-CORRECTED ({len(panel)} names, {idx.min().date()}->{idx.max().date()}) "
          "x CONSERVATIVE SLIPPAGE — per-stock NET expectancy/trade ===")
    print(f"  {'thr':>6s} {'events':>6s} {'obs':>6s} {'win':>5s}  "
          + "  ".join(f"net@{k:>5s}" for k in COST_SCENARIOS) + "   clears@0.71%? (t)")
    for x in THRESHOLDS:
        ps = event_stock_returns(panel, spy, threshold=x, membership=memb)
        ew = event_day_returns(panel, spy, threshold=x, membership=memb)
        if ps.empty:
            print(f"  {x*100:+5.1f}%  (no events)")
            continue
        gross = float(ps.mean())
        cells = "  ".join(f"{(gross - c.round_trip_frac())*100:+7.3f}%" for c in COST_SCENARIOS.values())
        cons = summarize(ps, COST_SCENARIOS["0.71%"].round_trip_frac())
        flag = "YES" if (cons["net_exp"] > 0 and cons["n"] >= TARGET_N and cons["tstat"] >= 1.5) else \
               ("thin" if cons["net_exp"] > 0 else "no")
        print(f"  {x*100:+5.1f}% {len(ew.dropna()):6d} {len(ps):6d} {(ps>0).mean():5.0%}  "
              f"{cells}   {flag} (t={cons['tstat']:+.2f})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep-threshold", type=float, default=-0.02)
    args = ap.parse_args()
    if not LEADLAG_DIR.exists():
        print("No data/leadlag/ — run scripts/download_leadlag_history.py first.")
        return 1

    d = str(LEADLAG_DIR)
    panel = load_stock_panel("kr", data_dir=d, symbols=KR_BASKET, fields=["open", "close"], min_bars=200)
    chip = {k: v for k, v in panel.items() if k in KR_CHIP}
    spy = load_spy_daily_returns(data_dir=d, symbol="SPY")
    soxx = load_spy_daily_returns(data_dir=d, symbol="SOXX")
    cf = CostModel().round_trip_frac()
    if panel == {} or spy.empty:
        print("Missing KR panel or SPY in data/leadlag/. Re-run the downloader.")
        return 1

    print(f"Deep US-lead-lag: 'buy KR at open, sell at close, the day after a US down-night'. "
          f"Round-trip hurdle {cf*100:.3f}%. SURVIVORSHIP caveat: fixed long-lived large-cap basket.")
    broad = _variant("BROAD: KR large-caps after a down SPY night", panel, spy, "SPY", cf)
    _by_year(panel, spy, args.deep_threshold, cf)
    _pit_cost_sweep(spy)            # survivorship-corrected + conservative slippage
    chip_rows = None
    if chip and not soxx.empty:
        chip_rows = _variant("CHIP: KR semis after a down SOXX night", chip, soxx, "SOXX", cf)
        _by_year(chip, soxx, args.deep_threshold, cf)

    # verdict
    print("\n=== VERDICT (deepened US-lead-lag, 2001+) ===")

    def best_clearing(rows):
        ok = {x: r for x, r in rows.items() if r["net_exp"] > 0 and r["n"] >= TARGET_N and r["tstat"] >= 1.5}
        return (max(ok, key=lambda x: ok[x]["tstat"]), ok) if ok else (None, {})

    bx, _ = best_clearing(broad)
    if bx is not None:
        r = broad[bx]
        print(f"  BROAD: DEPLOYABLE-grade signal at US<= {bx*100:+.1f}% — per-stock net "
              f"{r['net_exp']*100:+.3f}%/trade on n={r['n']} obs, win {r['win']:.0%}, t={r['tstat']:.2f} "
              f"(>=50 obs AND t>=1.5).")
    else:
        pos = {x: r for x, r in broad.items() if r["net_exp"] > 0}
        if pos:
            x = max(pos, key=lambda k: pos[k]["tstat"])
            r = pos[x]
            print(f"  BROAD: positive net at US<= {x*100:+.1f}% ({r['net_exp']*100:+.3f}%/trade, "
                  f"n={r['n']}, t={r['tstat']:.2f}) but NOT a meaningful sample / significance "
                  "(need n>=50 AND t>=1.5). Still a promising HYPOTHESIS, not deployable.")
        else:
            print("  BROAD: no threshold clears the cost hurdle net of cost.")
    if chip_rows:
        cx, _ = best_clearing(chip_rows)
        if cx is not None:
            r = chip_rows[cx]
            print(f"  CHIP : stronger linkage confirmed — US<= {cx*100:+.1f}% per-stock net "
                  f"{r['net_exp']*100:+.3f}%/trade, n={r['n']}, win {r['win']:.0%}, t={r['tstat']:.2f}.")
        else:
            print("  CHIP : semis-after-SOXX did not reach the meaningful-sample bar either.")
    print("\n(No-lookahead: US overnight is the last US session strictly before the KR date; "
          "buy KR open / sell close same day; net of KR round-trip cost. Fixed survivor basket.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
