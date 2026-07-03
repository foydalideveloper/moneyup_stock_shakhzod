"""Options VRP timing on KOSPI200 — one pre-registered trial (cached, no network).

Loads VKOSPI (implied vol) + KOSPI200 (realized vol), builds the locked VRP long/flat book,
reports net Sharpe/CAGR/maxDD + calendar-time NW t + by-year at 0.05% and 2x cost. If VKOSPI
is unavailable (this environment), reports the data wall honestly.

Usage: python scripts/run_vrp.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from tagent.data.vkospi_source import is_degenerate, load_vkospi  # noqa: E402
from tagent.decomposition import PPY, _series_stats  # noqa: E402
from tagent.index_calibration import load_index_close  # noqa: E402
from tagent.vrp import FUT_ROUND_TRIP, Z_ENTER, by_year, calendar_time_t, vrp_backtest  # noqa: E402


def _wall():
    print("\n=== DATA WALL (honest) ===")
    print("  VKOSPI (KOSPI200 implied-vol index) is NOT available in this environment:")
    print("  it is absent from pykrx's index universe (163 indices, none volatility), and the")
    print("  KRX MDC index endpoints return no VKOSPI series here. There is no substitute —")
    print("  the VRP signal requires real IMPLIED variance, not realized. So the trial renders")
    print("  NO verdict. The signal + pipeline are correct and unit-tested; drop a real VKOSPI")
    print("  series at data/vkospi_1d.csv (date,vkospi) to run the locked spec.")


def main() -> int:
    print("############ OPTIONS VRP TIMING — one pre-registered trial ############")
    print(f"Signal: VRP = VKOSPI^2 - trailing realized var; LONG KOSPI200 futures when VRP "
          f"z-score >= {Z_ENTER}, else FLAT; weekly. Cost {FUT_ROUND_TRIP*100:.2f}% + 2x. LOCKED.")
    vkospi = load_vkospi()
    idx = load_index_close("kospi200_long")
    print(f"\nVKOSPI rows: {len(vkospi)}; KOSPI200 rows: {len(idx)}")
    if is_degenerate(vkospi):
        _wall()
        return 0
    # the strategy can only run where VKOSPI exists -> restrict to the implied-vol period
    # (otherwise pre-VKOSPI years are spurious flat zeros that dilute the stats/by-year)
    idx = idx[idx.index >= vkospi.index.min()]
    print(f"  active period (VKOSPI available): {idx.index.min().date()} .. {idx.index.max().date()}")

    for label, cost in (("0.05%", FUT_ROUND_TRIP), ("2x (0.10%)", FUT_ROUND_TRIP * 2)):
        net = vrp_backtest(vkospi, idx, cost_round_trip=cost)
        st = _series_stats(net, PPY)
        t, n = calendar_time_t(net)
        print(f"\n  cost {label}: CAGR {st['cagr']:+.2%}  Sharpe {st['sharpe']:+.2f}  "
              f"maxDD {st['max_drawdown']:+.2%}  calendar NW t {t:+.2f} (n={n})")
        if label == "0.05%":
            base_t, base_net = t, net
    yb = by_year(base_net)
    print("  by year: " + "  ".join(f"{y}:{v['sharpe']:+.2f}" for y, v in sorted(yb.items())))
    pos = sum(1 for v in yb.values() if v["sharpe"] > 0)
    st2 = _series_stats(vrp_backtest(vkospi, idx, cost_round_trip=FUT_ROUND_TRIP * 2), PPY)
    print("\n=== VERDICT (calendar NW t>=2.5-3 AND survives 2x cost AND stable by-year) ===")
    clears = base_t >= 2.5 and st2["cagr"] > 0 and pos >= 0.6 * len(yb)
    print(f"  -> {'CLEARS' if clears else 'does NOT clear'} (NW t {base_t:+.2f}, 2x-cost CAGR "
          f"{st2['cagr']:+.2%}, positive {pos}/{len(yb)} yrs).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
