"""Index-rebalance FADE — one pre-registered trial (cached CSVs, no network).

KOSPI200 semi-annual reviews: SHORT adds / LONG deletes for HOLD days AFTER the effective
date, size-matched abnormal (vs KOSPI mid-cap index), net of SSF (adds) / cash (deletes)
costs + 2x stress. Calendar-time NW t + plain cross-sectional t + by-year. MSCI Korea
unavailable (separate provider) -> KOSPI200 only.

Usage: python scripts/run_index_rebalance.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd  # noqa: E402

from tagent.config import DATA_DIR  # noqa: E402
from tagent.index_calibration import load_index_close  # noqa: E402
from tagent.index_rebalance import (  # noqa: E402
    CASH_ROUND_TRIP, ENTRY_LAG, HOLD, SSF_ROUND_TRIP, calendar_time_excess, calendar_time_tstat,
    fade_event_returns, fade_net, plain_tstat,
)
from tagent.stock_momentum import load_stock_panel  # noqa: E402


def main() -> int:
    path = pathlib.Path(DATA_DIR) / "kospi200_changes.csv"
    if not path.exists():
        print("DATA WALL: data/kospi200_changes.csv not built — run download_kospi200_changes.py.")
        print("  (Historical KOSPI200 membership IS available via pykrx; MSCI Korea is NOT.)")
        return 1
    changes = pd.read_csv(path, dtype={"ticker": str})
    changes["ticker"] = changes["ticker"].str.zfill(6)
    changes["effective"] = pd.to_datetime(changes["effective"])
    bench = load_index_close("kospi_midcap")               # size-matched benchmark (mid-cap)
    if bench.empty:
        bench = load_index_close("kospi200_long")
    names = sorted(set(changes["ticker"]))
    panel = load_stock_panel("kr", symbols=names, fields=["close"], min_bars=30)

    print("############ INDEX-REBALANCE FADE — one pre-registered trial (KOSPI200) ############")
    print(f"DATA: KOSPI200 membership changes reconstructed from pykrx snapshots — "
          f"{len(changes)} events ({changes['action'].value_counts().to_dict()}), "
          f"{changes['effective'].dt.year.min()}-{changes['effective'].dt.year.max()}.")
    print("MSCI Korea: NOT available (separate provider, not in KRX/pykrx) -> KOSPI200 only.")
    print(f"Spec LOCKED: enter effective+{ENTRY_LAG}, hold {HOLD}d, SHORT adds / LONG deletes; "
          f"size-matched vs KOSPI mid-cap; SSF {SSF_ROUND_TRIP*100:.2f}% / cash {CASH_ROUND_TRIP*100:.2f}% + 2x.")

    ev = fade_event_returns(changes, panel, bench)
    if ev.empty:
        print("\nNo events with usable price windows — cannot run (price fetch incomplete?).")
        return 1
    adds = ev[ev["action"] == "add"]
    dels = ev[ev["action"] == "delete"]
    print(f"\n=== size-matched abnormal over the post-effective {HOLD}d window ===")
    print(f"  ADDS  (n={len(adds)}): mean abnormal {adds['abnormal'].mean()*100:+.2f}% "
          f"(t={plain_tstat(adds['abnormal']):+.2f})  [fade expects < 0: adds underperform]")
    print(f"  DELS  (n={len(dels)}): mean abnormal {dels['abnormal'].mean()*100:+.2f}% "
          f"(t={plain_tstat(dels['abnormal']):+.2f})  [fade expects > 0: deletes rebound]")

    # pooled FADE strategy (size-matched, pre-cost) + net at base / 2x
    strat = ev["strategy_ret"]
    print(f"\n=== pooled FADE strategy (SHORT adds / LONG deletes) ===")
    print(f"  pre-cost: mean {strat.mean()*100:+.3f}%/event  win {(strat>0).mean():.0%}  "
          f"plain t={plain_tstat(strat):+.2f}  (n={len(ev)})")
    for label, sm, cm in (("base", SSF_ROUND_TRIP, CASH_ROUND_TRIP),
                          ("2x", SSF_ROUND_TRIP * 2, CASH_ROUND_TRIP * 2)):
        net = fade_net(ev, sm, cm)
        print(f"  net@{label}: mean {net.mean()*100:+.3f}%/event  plain t={plain_tstat(net):+.2f}")
        if label == "base":
            base_net = net

    # calendar-time NW t (rigorous under event clustering)
    ex = calendar_time_excess(ev, panel, bench)
    ct, n = calendar_time_tstat(ex)
    print(f"\n  CALENDAR-TIME excess: mean/day {ex.mean()*1e4:+.2f}bp  Newey-West t={ct:+.2f}  ({n} days)")

    # by year (net base)
    yb = base_net.groupby(base_net.index.year)
    print("\n=== by year (net fade, base cost) ===")
    pos = ny = 0
    for y, seg in yb:
        ny += 1
        pos += int(seg.mean() > 0)
        print(f"  {y}: n={len(seg):3d}  mean {seg.mean()*100:+.3f}%/event")
    print(f"  positive years: {pos}/{ny}")

    # VERDICT
    sm_ok = strat.mean() > 0                                # size-matched abnormal (fade) positive
    t_ok = ct >= 2.5
    net2 = fade_net(ev, SSF_ROUND_TRIP * 2, CASH_ROUND_TRIP * 2)
    cost_ok = net2.mean() > 0
    print("\n=== VERDICT (size-matched abnormal>0 AND calendar t>=2.5-3 AND survives costs) ===")
    print(f"  abnormal>0: {'YES' if sm_ok else 'no'} ({strat.mean()*100:+.3f}%) | calendar t>=2.5: "
          f"{'YES' if t_ok else 'no'} ({ct:+.2f}) | survives 2x cost: {'YES' if cost_ok else 'no'} "
          f"({net2.mean()*100:+.3f}%)")
    clears = sm_ok and t_ok and cost_ok
    print(f"  -> {'CLEARS — tradable post-effective fade (slow sleeve).' if clears else 'does NOT clear (flat/negative is fine; low-frequency effect absent or too weak).'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
