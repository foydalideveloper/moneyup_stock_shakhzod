"""DART event studies — cancellation / acquisition / contract drift (cached, no network).

Runs the pre-registered methodology (dart_event_studies_spec.md): size-matched abnormal
returns + calendar-time portfolio Newey-West t + beta autopsy + gap-entry slippage, holds
20/40/60d. NO raw headline (raw is a footnote). Verdict per the locked bar.

Usage: python scripts/run_dart_events.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tagent.config import DATA_DIR  # noqa: E402
from tagent.dart_events import (  # noqa: E402
    BASE_COST, GAP_EXTRA_SLIP, GAP_THRESHOLD, HOLDS, STRESS_COSTS, add_size_matched,
    calendar_time_excess, calendar_time_tstat, event_trades, events_by_symbol,
    gap_decomposition, net_with_gap_slippage, quintile_daily_returns, size_quintiles,
)
from tagent.decomposition import beta_of  # noqa: E402
from tagent.index_calibration import load_index_close, load_shares  # noqa: E402
from tagent.kr_universe import load_members, membership_panel, universe_symbols  # noqa: E402
from tagent.stock_momentum import load_stock_panel  # noqa: E402
from tagent.strategies.us_leadlag_daily import summarize  # noqa: E402
from tagent.xs_momentum import align_close  # noqa: E402


def _beta_adjust(trades, panel, idx_close, hold, beta_window=120):
    """Event return minus pre-event beta x KOSPI-200 over the held window (the autopsy)."""
    close = align_close(panel)
    ix = close.index
    idxpx = pd.Series(idx_close, dtype=float).reindex(ix).ffill()
    idxret = idxpx.pct_change(fill_method=None)
    out = []
    for t in trades.itertuples():
        if t.symbol not in close.columns or t.entry not in ix:
            continue
        ei = ix.get_loc(t.entry)
        xi, lo = ei + hold, ei - 1 - beta_window
        if lo < 0 or xi >= len(ix):
            continue
        sret = close[t.symbol].pct_change(fill_method=None)
        b = beta_of(sret.iloc[lo:ei - 1], idxret.iloc[lo:ei - 1])
        mkt = float(idxpx.iloc[xi] / idxpx.iloc[ei] - 1.0)
        out.append(float(t.ret) - b * mkt)
    return pd.Series(out, dtype=float)


def run_event_type(name, label, ev, panel, memb, quintiles, qret, idx_close):
    print(f"\n{'='*78}\n{label}  ({sum(len(v) for v in ev.values())} filings, "
          f"{len(ev)} names)\n{'='*78}")
    clears = []
    for hold in HOLDS:
        tr = event_trades(panel, ev, hold=hold, membership=memb, non_overlapping=True)
        if len(tr) < 20:
            print(f"  hold {hold}d: only {len(tr)} non-overlapping trades — too few, skip.")
            continue
        sm = add_size_matched(tr, panel, quintiles, qret, hold)
        if len(sm) < 20:
            print(f"  hold {hold}d: only {len(sm)} size-matched trades — skip.")
            continue
        # primary: size-matched abnormal, net of base + stress costs (+ gap slippage)
        abn = sm["abnormal"]
        raw = sm["ret"]
        s_base = summarize(net_with_gap_slippage(sm, BASE_COST), 0.0)
        s_051 = summarize(net_with_gap_slippage(sm, STRESS_COSTS[0]), 0.0)
        s_071 = summarize(net_with_gap_slippage(sm, STRESS_COSTS[1]), 0.0)
        # calendar-time portfolio NW t (the rigorous significance test)
        ex = calendar_time_excess(sm, panel, quintiles, qret, hold)
        ct_t, ct_n = calendar_time_tstat(ex, hold)
        ct_mean = float(ex.mean()) if len(ex) else float("nan")
        # beta autopsy
        ba = _beta_adjust(sm, panel, idx_close, hold)
        gd = gap_decomposition(sm)

        print(f"\n  --- hold {hold}d  (n={len(sm)} size-matched, non-overlapping) ---")
        print(f"    ABNORMAL (size-matched) per event: mean {abn.mean()*100:+.3f}%  "
              f"win {(abn>0).mean():.0%}   [raw footnote {raw.mean()*100:+.3f}%]")
        print(f"    net abnormal: base0.20% {s_base['net_exp']*100:+.3f}% (t={s_base['tstat']:+.2f}) | "
              f"0.51% {s_051['net_exp']*100:+.3f}% | 0.71%+gap {s_071['net_exp']*100:+.3f}%")
        print(f"    CALENDAR-TIME excess: mean/day {ct_mean*1e4:+.2f}bp  Newey-West t={ct_t:+.2f}  "
              f"({ct_n} days)")
        print(f"    BETA-ADJUSTED abnormal: mean {ba.mean()*100:+.3f}%  win {(ba>0).mean():.0%}  "
              f"(autopsy: is it just KOSPI-200 beta?)")
        print(f"    GAP decomposition: gap r0 {gd['mean_gap_r0']*100:+.3f}% (MISSED, entering next open) "
              f"vs post-open {gd['mean_post_open']*100:+.3f}% (earned); {gd['frac_gap_up']:.0%} gap-up entries")
        # by size quintile (1=smallest .. 5=largest)
        qmean = sm.groupby("quintile")["abnormal"].agg(["mean", "size"])
        qstr = "  ".join(f"Q{int(q)} {row['mean']*100:+.2f}%(n{int(row['size'])})"
                         for q, row in qmean.iterrows())
        print(f"    by size quintile (abnormal): {qstr}")

        # pre-registered bar: abnormal>0 AND calendar NW t>3 AND survives 0.71%+gap AND beta-adj>0
        ok = (abn.mean() > 0 and ct_t > 3.0 and s_071["net_exp"] > 0 and ba.mean() > 0)
        clears.append((hold, ok))
        print(f"    -> clears bar at {hold}d: {'YES' if ok else 'no'} "
              f"(abn>0:{abn.mean()>0}, NWt>3:{ct_t>3.0}, net@0.71%>0:{s_071['net_exp']>0}, "
              f"beta-adj>0:{ba.mean()>0})")
    n_ok = sum(1 for _, ok in clears if ok)
    verdict = "CLEARS" if n_ok >= 2 else "does NOT clear"
    print(f"\n  === VERDICT [{name}]: {verdict} the bar ({n_ok}/{len(clears)} holds clear). ===")
    return n_ok >= 2


def main() -> int:
    path = pathlib.Path(DATA_DIR) / "kr_corporate_disclosures.csv"
    if not path.exists():
        print("Need data/kr_corporate_disclosures.csv — run download_kr_corporate_disclosures.py.")
        return 1
    disc = pd.read_csv(path, dtype={"symbol": str})
    disc["symbol"] = disc["symbol"].str.zfill(6)
    members = load_members()
    shares = load_shares()
    idx_close = load_index_close("kospi200_long")
    if idx_close.empty:
        idx_close = load_index_close("kospi200")
    syms = universe_symbols(members)
    panel = load_stock_panel("kr", symbols=syms, fields=["open", "close"], min_bars=60)
    idx = align_close(panel).index
    memb = membership_panel(members, idx, symbols=list(panel))
    quintiles = size_quintiles(panel, memb, shares)
    qret = quintile_daily_returns(panel, quintiles)

    from collections import Counter
    print("DART EVENT STUDIES (pre-registered; size-matched + calendar-time NW t + beta autopsy)")
    print(f"PIT {len(panel)} names; disclosures {dict(Counter(disc['type']))}.")
    print("NOTE: PIT universe is large-cap KOSPI — KOSDAQ contract-pump small-caps are under-represented.")

    results = {}
    for name, label in (("cancellation", "STUDY 1a — CANCELLATION (소각) [PRIMARY]"),
                        ("acquisition", "STUDY 1b — ACQUISITION (취득) [SECONDARY]"),
                        ("contract", "STUDY 2 — SUPPLY CONTRACT (단일판매·공급계약)")):
        ev = events_by_symbol(disc, name)
        if not ev:
            print(f"\n{label}: no events found — skip.")
            continue
        results[name] = run_event_type(name, label, ev, panel, memb, quintiles, qret, idx_close)

    print(f"\n{'#'*78}\nFINAL (separate trials, bar = abnormal>0 AND calendar NW t>3 AND survives "
          f"0.71%+gap AND beta-adj>0):")
    for k in ("cancellation", "acquisition", "contract"):
        if k in results:
            print(f"  {k:14s}: {'CLEARS' if results[k] else 'flat/negative (does not clear)'}")
    print("Honest baseline expectation: positive-event drift is mostly bull-market beta (PEAD trap).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
