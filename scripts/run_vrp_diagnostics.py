"""VRP deployment-gating diagnostics (FREE; diagnostics, not new trials).

1) Timing beta-trap: time-in-market % + spanning regression of the VRP strategy on
   buy-&-hold KOSPI200 -> the INTERCEPT (alpha) + Newey-West t is the DECIDING number
   (separates real timing signal from the equity premium in a costume).
2) Redundancy vs the KOSPI200 trend core: spanning BOTH ways (VRP on trend, trend on VRP).
3) VKOSPI archaeology: 2006-2009 provenance + the 2009-onward (real-data) verdict.

Usage: python scripts/run_vrp_diagnostics.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd  # noqa: E402

from tagent.data.vkospi_source import is_degenerate, load_vkospi  # noqa: E402
from tagent.decomposition import PPY, _series_stats  # noqa: E402
from tagent.index_calibration import load_index_close, spanning_regression  # noqa: E402
from tagent.trend_core import binary_trend_net  # noqa: E402
from tagent.vrp import (  # noqa: E402
    FUT_ROUND_TRIP, REALIZED_WINDOW, by_year, calendar_time_t, vrp_backtest, vrp_exposure,
)


def _span(label, book, basis):
    r = spanning_regression(book, basis)
    lo, hi = r["ci_ann"]
    print(f"  {label:34s} alpha {r['alpha_ann']:+.2%}/yr  NW t {r['t_alpha']:+.2f}  "
          f"beta {r['beta']:+.2f}  95%CI [{lo:+.2%},{hi:+.2%}]  n={r['n_months']}mo")
    return r


def main() -> int:
    vk = load_vkospi()
    idx_full = load_index_close("kospi200_long")
    if is_degenerate(vk) or idx_full.empty:
        print("Need real data/vkospi_1d.csv + kospi200_long. (VKOSPI unavailable -> no diagnostics.)")
        return 1
    idx = idx_full[idx_full.index >= vk.index.min()]
    print("############ VRP DEPLOYMENT-GATING DIAGNOSTICS ############")
    print(f"VKOSPI {vk.index.min().date()}..{vk.index.max().date()} ({len(vk)} rows); active idx "
          f"{idx.index.min().date()}..{idx.index.max().date()}.")

    net = vrp_backtest(vk, idx)
    exp = vrp_exposure(vk, idx).reindex(net.index)
    buyhold = idx.pct_change(fill_method=None).reindex(net.index)        # gross KOSPI200 long
    trend = binary_trend_net(idx, regime_ma=200, mode="next_bar").reindex(net.index)

    # 1) TIMING BETA-TRAP
    print("\n=== 1) TIMING BETA-TRAP (the deciding number) ===")
    print(f"  time-in-market (long fraction): {exp.mean():.1%}")
    st = _series_stats(net, PPY)
    ct, n = calendar_time_t(net)
    print(f"  VRP net: Sharpe {st['sharpe']:+.2f}  CAGR {st['cagr']:+.2%}  raw calendar NW t {ct:+.2f}")
    print("  spanning VRP ~ a + b*(buy&hold KOSPI200)  [alpha = timing skill beyond beta]:")
    bh = _span("VRP on buy&hold", net, buyhold)

    # 2) REDUNDANCY vs TREND CORE (mutual spanning)
    print("\n=== 2) REDUNDANCY vs KOSPI200 trend core (binary 200d) ===")
    _span("VRP on trend core", net, trend)
    _span("trend core on VRP", trend, net)

    # 3) VKOSPI ARCHAEOLOGY + 2009+ robustness
    print("\n=== 3) VKOSPI ARCHAEOLOGY + 2009+ robustness ===")
    print("  VKOSPI official KRX launch: 2009-04-13. KRX back-calculates history earlier, so the")
    print(f"  2006-2009 segment ({vk.index.min().date()}..2009-04-12) is BACK-CALCULATED, not")
    print("  live-traded implied vol. Real-data verdict = 2009-04-13 onward.")
    print(f"  REALIZED-VARIANCE WINDOW = {REALIZED_WINDOW}d: pre-registered/FIXED at the original")
    print("  VRP build (matches VKOSPI's ~1-month / 30-day implied horizon); NOT selected from a")
    print("  sweep -> VRP remains ONE trial (no extra windows tried).")
    real_idx = idx[idx.index >= pd.Timestamp("2009-04-13")]
    net09 = vrp_backtest(vk, real_idx)
    ct09, n09 = calendar_time_t(net09)
    st09 = _series_stats(net09, PPY)
    bh09 = spanning_regression(net09, real_idx.pct_change(fill_method=None).reindex(net09.index))
    yb = by_year(net09)
    pos = sum(1 for v in yb.values() if v["sharpe"] > 0)
    print(f"\n  2009+ (real-data) VRP: Sharpe {st09['sharpe']:+.2f}  CAGR {st09['cagr']:+.2%}  "
          f"calendar NW t {ct09:+.2f}  (n={n09})")
    print(f"  2009+ buy&hold spanning alpha {bh09['alpha_ann']:+.2%}/yr (NW t {bh09['t_alpha']:+.2f}); "
          f"positive {pos}/{len(yb)} yrs.")

    # VERDICT
    print("\n=== DECISION (buy&hold alpha is the gate, NOT the raw calendar t) ===")
    weak = bh["t_alpha"] < 2.0
    print(f"  buy&hold spanning alpha {bh['alpha_ann']:+.2%}/yr, NW t {bh['t_alpha']:+.2f} "
          f"(full) | {bh09['t_alpha']:+.2f} (2009+).")
    if weak:
        print("  -> WEAK alpha over buy&hold: VRP's return is mostly the EQUITY PREMIUM captured by")
        print(f"     being long ~{exp.mean():.0%} of the time, not skillful vol timing. FORWARD-TRACK")
        print("     ONLY; do NOT deploy as a standalone edge.")
    else:
        print("  -> Alpha survives buy&hold spanning -> genuine timing signal beyond beta.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
