"""Cross-sectional long-short OOS study on the KR universe.

Per-stock daily classification had no edge vs buy-and-hold. This ranks stocks
*against each other* each day and trades a market-neutral book:

  * label  = forward N-day return, top vs bottom TERCILE within each day
  * model  = ONE LightGBM across ALL stocks, purged + embargoed walk-forward CV
  * trade  = long top-DECILE predicted names, short bottom decile (dollar-neutral),
             costs on turnover; plus a long-only top-decile variant
  * compare technicals-only vs technicals + 공매도 + 수급 on the SAME rows/folds
  * benchmark vs the equal-weight index and buy-and-hold the universe

Needs data/<code>_{1d,short,flows}.csv (run download_krx_universe.py first).

Usage:
    python scripts/run_cross_sectional.py --horizon 5 --top-q 0.1
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from tagent.config import SETTINGS  # noqa: E402
from tagent.cross_sectional import (  # noqa: E402
    benchmark_returns, build_panel, cross_sectional_normalize, long_short_returns,
    make_rank_labels, purged_xs_predict, stats,
)
from tagent.data.flows_source import load_flows  # noqa: E402
from tagent.data.historical import load_history  # noqa: E402
from tagent.data.short_selling import load_short_selling  # noqa: E402

try:
    from scripts.download_krx_universe import DEFAULT_TICKERS
except Exception:
    DEFAULT_TICKERS = ["005930", "000660", "035420"]


def _fmt(s) -> str:
    return (f"ret {s.total_return:+7.2%}  ann {s.ann_return:+6.2%}  "
            f"Sharpe {s.sharpe:+5.2f}  maxDD {s.max_drawdown:+6.2%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    ap.add_argument("--horizon", type=int, default=5, help="forward return horizon (days)")
    ap.add_argument("--top-q", type=float, default=0.1, help="decile fraction per leg")
    ap.add_argument("--label-q", type=int, default=3, help="label quantiles (3=terciles)")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--embargo", type=int, default=2)
    args = ap.parse_args()

    history, short_data, flow_data = {}, {}, {}
    for sym in args.tickers:
        try:
            history[sym] = load_history(sym)
        except FileNotFoundError:
            continue
        try:
            short_data[sym] = load_short_selling(sym)
        except FileNotFoundError:
            pass
        try:
            flow_data[sym] = load_flows(sym)
        except FileNotFoundError:
            pass
    print(f"Loaded {len(history)} names "
          f"({len(short_data)} with 공매도, {len(flow_data)} with 수급)")

    panel, cols = build_panel(history, short_data, flow_data,
                              horizon=args.horizon, exclude_ban=True)
    panel = cross_sectional_normalize(panel, cols["all"])
    # Fair comparison: BOTH feature sets on the exact same rows (all features present).
    panel = panel.dropna(subset=cols["all"]).reset_index(drop=True)
    label = make_rank_labels(panel, q=args.label_q)

    n_days = panel["date"].nunique()
    print(f"Panel: {len(panel)} rows, {n_days} days, "
          f"{panel['date'].min().date()} -> {panel['date'].max().date()} (ban-excluded)\n")

    feature_sets = [("technicals", cols["tech"]),
                    ("+공매도+수급", cols["all"])]

    score_days = None
    for name, fcols in feature_sets:
        scores = purged_xs_predict(panel, fcols, label, n_splits=args.n_splits,
                                   horizon=args.horizon, embargo=args.embargo)
        p = panel.assign(score=scores)
        ls = long_short_returns(p, "score", top_q=args.top_q, allow_short=True,
                                cost_bps=SETTINGS.transaction_cost_bps,
                                slippage_bps=SETTINGS.slippage_bps)
        lo = long_short_returns(p, "score", top_q=args.top_q, allow_short=False,
                                cost_bps=SETTINGS.transaction_cost_bps,
                                slippage_bps=SETTINGS.slippage_bps)
        if score_days is None:
            score_days = ls.index
        print(f"=== feature set: {name}  ({len(fcols)} features, {len(ls)} OOS days) ===")
        print(f"  long-short (decile, market-neutral): {_fmt(stats(ls))}")
        print(f"  long-only  (top decile)            : {_fmt(stats(lo))}")
        print()

    bench = benchmark_returns(panel, days=score_days)
    print(f"=== benchmarks (same {len(score_days)} OOS days) ===")
    print(f"  equal-weight index (daily reb.) : {_fmt(stats(bench['index']))}")
    print(f"  buy & hold the universe (drift)  : {_fmt(stats(bench['buy_hold']))}")
    print("\nLong-short is dollar-neutral (long top decile, short bottom), costs on "
          "turnover. Compare its Sharpe/maxDD to the index — neutrality should cut "
          "drawdown even if return trails a raging bull.")
