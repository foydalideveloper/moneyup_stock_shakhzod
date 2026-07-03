"""Foreign-futures-flow timing — one pre-registered trial (KRX derivatives investor flow).

Loads (or fetches) daily KOSPI200-futures foreign net flow, runs the locked z-score timing
rule, reports net Sharpe/CAGR/maxDD + calendar-time NW t + by-year + early/late DECAY, at
0.05% and 2x conservative cost. If the futures-investor data is unavailable (all-zero in
this environment), reports the data wall honestly instead of a fake verdict.

Usage: python scripts/run_futures_flow.py [--fetch-days 120]
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

from tagent.data.krx_deriv_investor import (  # noqa: E402
    fetch_kospi200_futures_flow, flow_csv_path, load_futures_flow,
)
from tagent.decomposition import PPY, _series_stats  # noqa: E402
from tagent.futures_flow import (  # noqa: E402
    FUT_ROUND_TRIP, Z_ENTER, by_year, calendar_time_t, decay_split, flow_backtest, is_degenerate,
)
from tagent.index_calibration import load_index_close  # noqa: E402


def _report_wall():
    print("\n=== DATA WALL (honest; re-diagnosed across sources) ===")
    print("  KOSPI200 FUTURES investor flow is NOT available in this environment. Re-sourcing tried:")
    print("   * KRX MDC MDCSTAT13101 (derivatives 투자자별) for products KRDRVFUK2I (KOSPI200 fut),")
    print("     KRDRVFUMKI (mini), and mktId KRDRV -> ALL return 외국인/기관 net = 0.")
    print("   * The ETF investor aggregate (069500) also returns all-zero net flows.")
    print("   * Kiwoom investor-trend API: the configured Kiwoom env is MOCK (mockapi.kiwoom.com),")
    print("     so it cannot supply real futures flows.")
    print("  Per-ISSUE STOCK flows (e.g. 005930 foreign net) ARE populated, so the env carries")
    print("  per-stock cash flows but NOT the index/futures investor aggregates the signal needs.")
    print("  -> No real verdict. Source + pipeline are correct/unit-tested; ready for a populated")
    print("     KRX. Per-stock CASH flow is NOT substituted (that is the already-killed cash 수급).")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch-days", type=int, default=120, help="trading days to probe if no cache")
    args = ap.parse_args()

    print("############ FOREIGN-FUTURES-FLOW TIMING — one pre-registered trial ############")
    print(f"Signal: z-score of trailing 5d FOREIGN KOSPI200-futures net flow -> long/flat/short "
          f"next day (|z|>={Z_ENTER}). Costs {FUT_ROUND_TRIP*100:.2f}% RT + roll, 2x stress. LOCKED.")

    flow_df = load_futures_flow()
    if flow_df.empty:
        print(f"\nNo cached flow — probing the last {args.fetch_days} trading days via KRX MDC ...")
        try:
            cal = load_index_close("kospi200_long").index[-args.fetch_days:]
            flow_df = fetch_kospi200_futures_flow(cal.min(), cal.max(), calendar=cal)
            if not flow_df.empty:
                flow_df.to_csv(flow_csv_path())
        except Exception as e:
            print(f"  fetch failed: {str(e)[:80]}")

    foreign = flow_df["foreign_net"] if "foreign_net" in flow_df else pd.Series(dtype=float)
    print(f"\nfetched flow rows: {len(foreign)}; non-zero: {int((foreign.fillna(0) != 0).sum())}")

    if is_degenerate(foreign):
        _report_wall()
        return 0

    # --- real trial (runs when KRX flow is populated) ---
    fut = load_index_close("kospi200_long")               # KOSPI200 index ~ futures return proxy
    for label, cost in (("0.05%", FUT_ROUND_TRIP), ("2x (0.10%)", FUT_ROUND_TRIP * 2)):
        net = flow_backtest(foreign, fut, cost_round_trip=cost)
        st = _series_stats(net, PPY)
        t, n = calendar_time_t(net)
        print(f"\n  cost {label}: CAGR {st['cagr']:+.2%}  Sharpe {st['sharpe']:+.2f}  "
              f"maxDD {st['max_drawdown']:+.2%}  calendar NW t {t:+.2f} (n={n})")
        if label == "0.05%":
            base_t = t
            by = by_year(net)
            print("  by year: " + "  ".join(f"{y}:{v['sharpe']:+.2f}" for y, v in sorted(by.items())))
            dec = decay_split(net)
            print(f"  DECAY (split {dec['mid']}): early Sharpe {dec['early']['sharpe']:+.2f} -> "
                  f"late {dec['late']['sharpe']:+.2f}")
            decayed = dec["early"]["sharpe"] > dec["late"]["sharpe"] + 0.3
        net2 = net
    st2 = _series_stats(net2, PPY)
    print("\n=== VERDICT (NW t>=2.5-3 AND survives 2x cost AND not early-sample-only) ===")
    clears = base_t >= 2.5 and st2["cagr"] > 0 and not decayed
    print(f"  -> {'CLEARS' if clears else 'does NOT clear'} the bar "
          f"(NW t {base_t:+.2f}, 2x-cost CAGR {st2['cagr']:+.2%}, decay={'yes' if decayed else 'no'}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
