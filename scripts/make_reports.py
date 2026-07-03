"""Generate quantstats HTML tear sheets (+ a funding-carry Monte Carlo) to reports/.

Builds, from whatever is cached locally (no network), tear sheets for:
  * funding-carry — the offline cross-sectional backtest if funding history is
    cached, else the live runner (data/funding_live.csv), else the scorecard state
    — PLUS a bootstrap Monte Carlo giving the bust (ruin) probability + tail;
  * the live funding runner equity curve;
  * each prediction agent (us-ml / crypto-ml / crypto-ob / TA ...) reconstructed
    from data/signal_log.csv.

Benchmark for funding is buy-and-hold BTC where its price history is cached.

Usage:
    python scripts/make_reports.py
    python scripts/make_reports.py --out reports --mc-paths 5000 --ruin 0.5
"""

import argparse
import glob
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
from tagent.report import (  # noqa: E402
    INTERVALS_8H_PER_YEAR, monte_carlo, perf_stats, returns_from_funding_live,
    returns_from_signal_log, tear_sheet, funding_carry_returns_from_scorecard,
)


def _funding_backtest_returns():
    """Net per-8h returns of the cached cross-sectional funding-carry backtest, and
    a buy-and-hold BTC benchmark — or (None, None) if no funding history is cached."""
    paths = glob.glob(os.path.join(DATA_DIR, "funding_hist_*.csv")) or \
        glob.glob(os.path.join(DATA_DIR, "funding_*.csv"))
    per_coin, btc = {}, None
    for p in paths:
        name = os.path.basename(p)
        sym = name.replace("funding_hist_", "").replace("funding_", "").replace(".csv", "")
        try:
            df = pd.read_csv(p, index_col="time", parse_dates=True)
        except Exception:
            continue
        if {"funding_rate", "perp", "spot_close"}.issubset(df.columns) and len(df) > 30:
            per_coin[sym] = df
            if sym.upper().startswith("BTC"):
                btc = df["spot_close"].astype(float).pct_change().dropna()
    if not per_coin:
        return None, None
    from tagent.funding_portfolio import portfolio_returns
    r = portfolio_returns(per_coin, scheme="funding")["net"]
    return r, btc


def _summary(name, returns, ppy):
    st = perf_stats(returns, periods_per_year=ppy)
    return (f"  {name:22s} n={st['n']:5d}  Sharpe {st['sharpe']:+6.2f}  Sortino {st['sortino']:+6.2f}  "
            f"CAGR {st['cagr']:+7.2%}  maxDD {st['max_drawdown']:+7.2%}  "
            f"VaR95 {st['var']*100:5.2f}%  CVaR95 {st['cvar']*100:5.2f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports")
    ap.add_argument("--mc-paths", type=int, default=5000)
    ap.add_argument("--ruin", type=float, default=0.5, help="drawdown that counts as a bust")
    ap.add_argument("--cost-bps", type=float, default=10.0, help="round-trip cost for signal-log agents")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting tear sheets to {out}/ (cached data only, no network)\n")
    made = []

    # 1) funding-carry: backtest -> live -> scorecard (first available)
    fr, bench = _funding_backtest_returns()
    src = "backtest"
    if fr is None or fr.dropna().empty:
        live = os.path.join(DATA_DIR, "funding_live.csv")
        state = os.path.join(DATA_DIR, "scorecard_state.json")
        if os.path.exists(live):
            fr, src = returns_from_funding_live(live), "live runner"
        elif os.path.exists(state):
            fr, src = funding_carry_returns_from_scorecard(state), "scorecard"

    if fr is not None and not pd.Series(fr).dropna().empty:
        print(_summary(f"funding-carry [{src}]", fr, INTERVALS_8H_PER_YEAR))
        try:
            tear_sheet(fr, out / "funding_carry.html", benchmark=bench, title="Funding carry")
            made.append("funding_carry.html")
        except Exception as e:
            print(f"    (tear sheet skipped: {str(e)[:60]})")
        # 2) THE honest number: Monte Carlo bust probability + tail (block bootstrap)
        mc = monte_carlo(fr, n_paths=args.mc_paths, block=9, ruin_drawdown=args.ruin, seed=0)
        print(f"    Monte Carlo ({mc['n_paths']} paths, {mc['horizon']}x8h, block=9):")
        print(f"      BUST prob (>= {int(args.ruin*100)}% drawdown): {mc['bust_prob']*100:.2f}%")
        print(f"      final equity  median {mc['median_final']:,.0f}  "
              f"p5 {mc['p5_final']:,.0f}  p95 {mc['p95_final']:,.0f}  "
              f"(start {mc['start_equity']:,.0f})")
        print(f"      95% VaR on final {mc['var_final']:,.0f}  CVaR {mc['cvar_final']:,.0f}  "
              f"median worst-DD {mc['worst_drawdown_med']*100:+.1f}%")
        import json
        (out / "funding_carry_montecarlo.json").write_text(json.dumps(mc, indent=2), encoding="utf-8")
        made.append("funding_carry_montecarlo.json")
    else:
        print("  funding-carry: no cached funding history / live state — skipped")

    # 3) live funding runner equity curve (if separate from above)
    live = os.path.join(DATA_DIR, "funding_live.csv")
    if os.path.exists(live) and src != "live runner":
        try:
            r = returns_from_funding_live(live)
            if not r.empty:
                print(_summary("funding-carry [live]", r, INTERVALS_8H_PER_YEAR))
                tear_sheet(r, out / "funding_live.html", title="Funding carry (live runner)")
                made.append("funding_live.html")
        except Exception as e:
            print(f"  funding_live: skipped ({str(e)[:50]})")

    # 4) prediction agents from the signal log
    slog = os.path.join(DATA_DIR, "signal_log.csv")
    if os.path.exists(slog):
        try:
            sources = sorted(pd.read_csv(slog)["source"].dropna().unique())
        except Exception:
            sources = []
        for s in sources:
            r = returns_from_signal_log(slog, source=s, cost_bps=args.cost_bps)
            if r.dropna().shape[0] < 5:
                continue
            print(_summary(f"agent [{s}]", r, 365))
            try:
                tear_sheet(r, out / f"agent_{s}.html", title=f"Agent: {s}")
                made.append(f"agent_{s}.html")
            except Exception as e:
                print(f"    ({s} tear sheet skipped: {str(e)[:50]})")
    else:
        print("  signal_log.csv not found — no per-agent tear sheets")

    print(f"\nWrote {len(made)} file(s) to {out}/: " + (", ".join(made) if made else "(none)"))
    print("Honest note: tear-sheet annualization uses a synthetic daily index for")
    print("rendering; cadence-correct stats are in the summary above. The Monte Carlo")
    print("bust probability is the key tail number — funding carry's fat left tail")
    print("(funding flips + short-leg liquidation) is what it captures.")
