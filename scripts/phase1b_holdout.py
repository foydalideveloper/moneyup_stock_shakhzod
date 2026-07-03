# -*- coding: utf-8 -*-
"""Task 4 — ONE-SHOT sealed-holdout OOS confirmation (read-only, no tuning).

Runs the UNCHANGED corrected long-side scorer (phase1b.score_call) on the sealed 152-video holdout to
confirm or deny the in-sample LOSS out-of-sample, and tests the CONTRARIAN sign (mirror = short his buy
calls) as a PAPER signal net of the short's round-trip cost — shortability/borrow flagged UNRESOLVED, so
it is explicitly NOT a tradeable edge. Confirmation only: the scorer is not modified, nothing is tuned on
the holdout, no peek-then-adjust. The exploratory 70% long-side is recomputed only as the comparison
baseline. Writes to data/_moneyup_advisor/qa_audit/.
"""
from __future__ import annotations
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
from moneyup_advisor import config, phase1b as P
from moneyup_advisor.phase1b import (_ohlcv5, _to_kst, _agg, INDEX_PROXY, BETA_LOOKBACK, KST,
                                     SCORED_LONG, COST_ROUNDTRIP, TSTAT_BAR)

OUT = config.DATA_DIR / "qa_audit"


def _block(scored):
    """Long-side (pooled SCORED_LONG entered) + contrarian mirror, as (n, mean net %, t)."""
    longs = [s for s in scored if s.get("play") in SCORED_LONG and s.get("status") == "entered"
             and s.get("net_abnormal") is not None]
    nets = [s["net_abnormal"] for s in longs]
    # contrarian mirror: short the same entry->exit, beta+size adj, net of the SHORT's own round-trip
    # cost (= -long_net - 2*round-trip). borrow + shortability are UNRESOLVED (flagged) -> paper only.
    mirror = [-n - 2 * COST_ROUNDTRIP for n in nets]
    seg = {}
    for s_name in SCORED_LONG:
        rows = [s["net_abnormal"] for s in longs if s["play"] == s_name]
        seg[s_name] = _agg(rows)
    return {"long_pooled": _agg(nets), "contrarian_mirror": _agg(mirror),
            "by_segment": seg, "n_entered": len(longs)}


def main():
    man = json.loads((OUT / "diagnostic" / "holdout_manifest.json").read_text(encoding="utf-8"))
    holdout = set(man["holdout_video_ids"])
    explore = set(man["exploratory_video_ids"])
    calls = P.load_calls()
    pubs = [_to_kst(c["publish"]).date() for c in calls]
    start = (min(pubs) - dt.timedelta(days=BETA_LOOKBACK + 180)).strftime("%Y%m%d")
    end = dt.datetime.now(KST).strftime("%Y%m%d")
    idx_s = _ohlcv5(INDEX_PROXY, start, end)
    dates = sorted(idx_s)
    last_date = dates[-1] if dates else dt.datetime.now(KST).date()
    print(f"[holdout] scoring {len(calls)} calls (UNCHANGED scorer) …", flush=True)

    scored = []
    for i, c in enumerate(calls):
        scored.append({**c, **P.score_call(c, _ohlcv5(c["ticker"], start, end), idx_s, dates, last_date)})
        if i % 100 == 0:
            print(f"  {i}/{len(calls)}", flush=True)

    hd = _block([s for s in scored if s["video_id"] in holdout])     # OOS
    ex = _block([s for s in scored if s["video_id"] in explore])     # in-sample baseline (already seen)

    def verdict(b):
        lp = b["long_pooled"]
        if lp["n"] < 30:
            return f"INCONCLUSIVE (n={lp['n']}<30)"
        if lp["mean_pct"] is not None and lp["mean_pct"] < 0 and lp["t"] is not None and lp["t"] <= -TSTAT_BAR:
            return f"LOSS CONFIRMED — significantly negative (mean {lp['mean_pct']}%, t {lp['t']} <= -{TSTAT_BAR})"
        if lp["mean_pct"] is not None and lp["mean_pct"] < 0:
            return f"negative but |t| below base bar (mean {lp['mean_pct']}%, t {lp['t']})"
        return f"NOT a loss OOS (mean {lp['mean_pct']}%, t {lp['t']})"

    res = {"test": "Task 4 — one-shot sealed-holdout OOS confirmation (no tuning)",
           "scorer": "corrected phase1b.score_call (UNCHANGED)", "cost_roundtrip_pct": COST_ROUNDTRIP * 100,
           "base_tstat_bar": TSTAT_BAR,
           "holdout_OOS": hd, "exploratory_in_sample": ex,
           "holdout_long_verdict": verdict(hd), "exploratory_long_verdict": verdict(ex),
           "contrarian_note": ("PAPER signal only — net of the short's 0.50% round-trip; KRX shortability "
                               "(borrowable universe) + borrow fee + 공매도 과열종목 status are UNRESOLVED, so "
                               "this is NOT a tradeable edge. Most of his picks are mid/small theme names "
                               "likely outside the retail-borrowable universe."),
           "held_oos": None}
    # did the in-sample loss hold OOS?
    res["held_oos"] = bool(hd["long_pooled"]["n"] >= 30 and hd["long_pooled"]["mean_pct"] is not None
                           and hd["long_pooled"]["mean_pct"] < 0)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "phase1b_holdout.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    def show(name, b, v):
        lp, mr = b["long_pooled"], b["contrarian_mirror"]
        print(f"\n=== {name} ===")
        print(f"  LONG pooled:  n={lp['n']:>4} mean net%={lp['mean_pct']} t={lp['t']}  -> {v}")
        print(f"  CONTRARIAN (mirror, paper, net cost): n={mr['n']:>4} mean net%={mr['mean_pct']} t={mr['t']}")
        for s in SCORED_LONG:
            a = b["by_segment"][s]
            if a["n"]:
                print(f"     {s:13} n={a['n']:>4} mean%={a['mean_pct']} t={a['t']}")
    show("HOLDOUT (OOS, 152 videos — sealed until now)", hd, res["holdout_long_verdict"])
    show("EXPLORATORY (in-sample baseline, 355 videos)", ex, res["exploratory_long_verdict"])
    print(f"\nIN-SAMPLE LOSS HELD OUT-OF-SAMPLE: {res['held_oos']}")
    print(f"saved -> {OUT/'phase1b_holdout.json'}")


if __name__ == "__main__":
    main()
