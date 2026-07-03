"""Short-horizon DAILY KR trading edges (free daily data only) — same rigor.

Tests three candidates on the point-in-time, survivorship-corrected KR universe, NET
of realistic KR round-trip cost (~0.41%):
  * OVERNIGHT  — buy at close, sell at next open (close->open, EW universe);
  * REVERSAL   — long the recent losers (trailing 1-5d), hold N days;
  * US LEAD-LAG — after a sharp US overnight drop, intraday-long KR the next day.

Reports win rate, expectancy/trade, CAGR, Sharpe, max drawdown, MC bust + by-year,
and states plainly whether each clears the cost hurdle. Cached CSVs, no network.

Usage:
    python scripts/validate_kr_daily_edges.py
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
from tagent.kr_universe import load_members, membership_panel, universe_symbols  # noqa: E402
from tagent.report import monte_carlo, perf_stats  # noqa: E402
from tagent.stock_momentum import classic_config, load_stock_panel  # noqa: E402
from tagent.strategies.overnight_effect import overnight_backtest  # noqa: E402
from tagent.strategies.short_term_reversal import reversal_backtest, reversal_signal  # noqa: E402
from tagent.strategies.us_leadlag_daily import leadlag_backtest  # noqa: E402
from tagent.xs_momentum import align_close, backtest  # noqa: E402
from tagent.xs_momentum_validate import by_year, walk_forward_signal  # noqa: E402

PPY = 252


def _bust(net):
    return monte_carlo(net.dropna(), n_paths=5000, block=21, ruin_drawdown=0.5,
                       seed=0)["bust_prob"] if not net.dropna().empty else float("nan")


def _line(net, hurdle, per="day"):
    n = net.dropna()
    s = perf_stats(n, PPY)
    win = float((n > 0).mean()) if len(n) else 0.0
    exp = float(n.mean()) if len(n) else 0.0
    verdict = "BEATS" if exp > 0 else "below"
    return (f"n={len(n):4d}  win {win:5.1%}  exp/{per} {exp*100:+.4f}%  "
            f"ann {s['cagr']:+7.2%}  Sharpe {s['sharpe']:+5.2f}  maxDD {s['max_drawdown']:+6.1%}  "
            f"bust {_bust(n)*100:3.0f}%  -> {verdict} hurdle")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=756)
    ap.add_argument("--test", type=int, default=252)
    args = ap.parse_args()

    members = load_members()
    if not members:
        print("No PIT membership — run scripts/download_kr_pit_universe.py first.")
        return 1
    syms = universe_symbols(members)
    panel = load_stock_panel("kr", symbols=syms, min_bars=60, fields=["open", "close"])
    idx = align_close(panel).index
    memb = membership_panel(members, idx, symbols=list(panel))
    spy = load_spy_daily_returns()
    cost = CostModel()
    hurdle = cost.round_trip_frac()

    print(f"\nUniverse: {len(panel)} PIT names, {len(idx)} bars. NET of KR cost; "
          f"round-trip hurdle {hurdle*100:.3f}%. Survivorship-corrected (PIT membership).")

    # 1) OVERNIGHT
    onb = overnight_backtest(panel, membership=memb, cost=cost)
    print("\n=== 1) OVERNIGHT (buy close, sell next open, EW universe) ===")
    print("  " + _line(onb["net"], hurdle, per="day"))
    print(f"     gross/day {onb['gross'].mean()*100:+.4f}% vs {hurdle*100:.3f}% hurdle "
          f"(a full round trip EVERY day).")

    # 2) SHORT-TERM REVERSAL — sweep lookback (hold = lookback), top 20%
    print("\n=== 2) SHORT-TERM REVERSAL (long losers, hold = lookback, top 20%) ===")
    base = backtest(panel, classic_config("kr", allow_short=False), PPY, membership=memb)
    bs = base["basket_stats"]
    best = None
    for lb in (1, 2, 3, 5):
        r = reversal_backtest(panel, lookback=lb, hold=lb, top_q=0.2, membership=memb,
                              periods_per_year=PPY)
        s = r["stats"]
        flag = "beats basket" if s["sharpe"] > bs["sharpe"] and s["total_return"] > bs["total_return"] else ""
        print(f"  lookback {lb}d: Sharpe {s['sharpe']:+5.2f}  ann {s['cagr']:+7.2%}  "
              f"maxDD {s['max_drawdown']:+6.1%}  turnover {r['avg_turnover']:.2f}  {flag}")
        if best is None or s["sharpe"] > best[1]["sharpe"]:
            best = (lb, s, r)
    print(f"  BASKET(EW): Sharpe {bs['sharpe']:+5.2f}  ann {bs['cagr']:+7.2%}")
    lb, _, _ = best
    close = align_close(panel)
    wf = walk_forward_signal(panel, reversal_signal(close, lb), top_qs=(0.1, 0.2, 0.3),
                             train_bars=args.train, test_bars=args.test, cfg=classic_config("kr", allow_short=False),
                             periods_per_year=PPY, membership=memb)
    ws = wf["stats"]
    print(f"  walk-forward OOS (lookback {lb}d, top_q on train): Sharpe {ws['sharpe']:+.2f}  "
          f"ann {ws['cagr']:+.2%}  maxDD {ws['max_drawdown']:+.2%}  n={ws['n']}")

    # 3) US LEAD-LAG — sweep the US-drop threshold
    print("\n=== 3) US LEAD-LAG (US overnight <= -X% -> intraday long KR next day) ===")
    ll = {}
    if spy.empty:
        print("  no SPY_1d.csv — skipping (refresh via yfinance).")
    else:
        for x in (-0.01, -0.015, -0.02):
            r = leadlag_backtest(panel, spy, threshold=x, membership=memb, cost=cost)
            ll[x] = r
            if r["n_events"] == 0:
                print(f"  US<= {x*100:+.1f}%: no event days in sample.")
                continue
            print(f"  US<= {x*100:+.1f}%: " + _line(r["net"], hurdle, per="event")
                  + f"   gross/event {r['gross'].mean()*100:+.3f}%")

    # by-year on the deployable books (overnight net + best reversal net)
    print("\n=== by year (total return) ===   overnight  |  reversal(best)")
    yb_on = by_year(onb["net"], PPY)
    yb_rev = by_year(best[2]["net"], PPY)
    for y in sorted(set(yb_on) | set(yb_rev)):
        a = yb_on.get(y, {}).get("total_return", float("nan"))
        b = yb_rev.get(y, {}).get("total_return", float("nan"))
        print(f"  {y}: {a:+8.2%}   |   {b:+8.2%}")

    # verdict
    print("\n=== VERDICT (free daily-trading candidates) ===")
    on_exp = onb["net"].mean()
    rev_beats = best[1]["sharpe"] > bs["sharpe"] and best[1]["total_return"] > bs["total_return"]
    print(f"  OVERNIGHT: expectancy/day {on_exp*100:+.4f}% -> "
          f"{'CLEARS' if on_exp > 0 else 'FAILS'} the cost hurdle "
          f"(daily round trips make this very hard).")
    print(f"  REVERSAL : best lookback {best[0]}d Sharpe {best[1]['sharpe']:+.2f} vs basket {bs['sharpe']:+.2f} -> "
          f"{'beats the basket' if rev_beats else 'does NOT beat buy-and-hold'}.")
    pos = {x: r for x, r in ll.items() if r["n_events"] and r["net"].mean() > 0}
    if pos:
        x = min(pos)                                        # most extreme clearing threshold
        r = pos[x]
        print(f"  LEAD-LAG : clears the hurdle ONLY at US<= {x*100:+.1f}% "
              f"(+{r['net'].mean()*100:.3f}%/event, win {(r['net']>0).mean():.0%}) — but on just "
              f"{r['n_events']} events. DIRECTIONAL ONLY, not a verdict; gross rises with US-shock "
              "severity (economically sensible), worth more data.")
    else:
        print("  LEAD-LAG : no threshold clears the cost hurdle net of cost on this sample.")
    print("\n(No-lookahead: close->open / as-of US overnight / trailing returns use only past data; "
          "PIT membership; KR costs charged per round trip.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
