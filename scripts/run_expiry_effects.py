"""KR derivatives expiry-calendar effects — one pre-registered trial (cached, no network).

Stoll-Whaley expiration reversal on KOSPI200 (1990->). Pre-scheduled, NON-OVERLAPPING
events -> clean plain t-stat. Reports event count, mean effect, t-stat, by-year, descriptive
reversal regression, at 0.05% and 2x conservative cost. Calendar-only (program/arbitrage
volumes unavailable here).

Usage: python scripts/run_expiry_effects.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from tagent.expiry_effects import (  # noqa: E402
    FUT_ROUND_TRIP, HOLD, PRE_WINDOW, by_year, event_returns, expiry_dates, plain_tstat,
    reversal_net, reversal_regression,
)
from tagent.index_calibration import load_index_close  # noqa: E402


def _report(name, events, close):
    print(f"\n================= {name}  ({len(events)} non-overlapping events) =================")
    if events.empty:
        print("  no events.")
        return None
    # descriptive Stoll-Whaley signatures (unconditional)
    print(f"  unconditional: expiry-day mean {events['expiry_ret'].mean()*100:+.3f}% "
          f"(t={plain_tstat(events['expiry_ret']):+.2f}) | post-{HOLD}d mean "
          f"{events['fwd_ret'].mean()*100:+.3f}% (t={plain_tstat(events['fwd_ret']):+.2f})")
    reg = reversal_regression(events)
    print(f"  reversal regression fwd_ret ~ a + b*pre_drift: b={reg['beta']:+.3f} "
          f"(t={reg['t_beta']:+.2f}, n={reg['n']})  [b<0 = reversal]")
    # tradable reversal, net of cost (0.05% and 2x)
    base_t = None
    for label, cost in (("0.05%", FUT_ROUND_TRIP), ("2x (0.10%)", FUT_ROUND_TRIP * 2)):
        net = reversal_net(events, cost)
        t = plain_tstat(net)
        wr = (net > 0).mean()
        print(f"  REVERSAL trade net@{label}: mean {net.mean()*100:+.3f}%/event  win {wr:.0%}  "
              f"plain t={t:+.2f}  (n={len(net)})")
        if label == "0.05%":
            base_t, base_net = t, net
    yb = by_year(base_net)
    pos_years = sum(1 for v in yb.values() if v["mean_pct"] > 0)
    print(f"  by-year: positive in {pos_years}/{len(yb)} years  "
          f"(mean/event range {min(v['mean_pct'] for v in yb.values()):+.2f}%..{max(v['mean_pct'] for v in yb.values()):+.2f}%)")
    return base_t, base_net


def main() -> int:
    close = load_index_close("kospi200_long")
    if close.empty:
        print("Need data/kospi200_long_1d.csv.")
        return 1
    idx = close.index
    print("############ KR EXPIRY-CALENDAR EFFECTS — one pre-registered trial (Stoll-Whaley) ############")
    print(f"KOSPI200 {idx.min().date()}..{idx.max().date()}. Spec LOCKED: pre-window {PRE_WINDOW}d, "
          f"hold {HOLD}d, reversal = -sign(pre-expiry drift). Cost {FUT_ROUND_TRIP*100:.2f}% RT + 2x.")
    print("PROGRAM-TRADING/ARBITRAGE VOLUMES: NOT available in this environment (not in pykrx; KRX")
    print("MDC aggregates return zero here) -> CALENDAR-ONLY trial (no flow conditioning).")

    monthly = expiry_dates(idx.min(), idx.max(), idx, quarterly_only=False)
    quarterly = expiry_dates(idx.min(), idx.max(), idx, quarterly_only=True)
    ev_m = event_returns(close, monthly)
    ev_q = event_returns(close, quarterly)
    res_m = _report("MONTHLY option expiry (2nd Thursday)", ev_m, close)
    res_q = _report("QUARTERLY futures expiry (witching)", ev_q, close)

    print("\n=== VERDICT (pre-registered: plain t>=2.5-3 on non-overlapping events AND survives 2x cost AND stable by-year) ===")
    for name, res, events in (("monthly", res_m, ev_m), ("quarterly", res_q, ev_q)):
        if res is None:
            continue
        t, net = res
        net2 = reversal_net(events, FUT_ROUND_TRIP * 2)
        yb = by_year(net)
        pos_years = sum(1 for v in yb.values() if v["mean_pct"] > 0)
        stable = len(yb) > 0 and pos_years >= 0.6 * len(yb)
        clears = t >= 2.5 and net2.mean() > 0 and stable
        print(f"  {name:10s}: plain t {t:+.2f} | 2x-cost mean {net2.mean()*100:+.3f}%/event | "
              f"stable {pos_years}/{len(yb)}yr -> {'CLEARS' if clears else 'does NOT clear'}")
    print("\n  (Expect a modest effect; realistic use is a timing OVERLAY, not a standalone trade. "
          "Flat/negative is fine. Calendar-only — program/arbitrage flow conditioning unavailable here.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
