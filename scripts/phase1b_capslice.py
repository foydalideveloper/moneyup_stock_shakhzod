# -*- coding: utf-8 -*-
"""Task 3 — market-cap / per-ticker slice of the corrected Phase 1B BUY calls (read-only, no re-extraction).

Reuses phase1b's scoring (classify_call + score_call + _ohlcv5 + _nw_tstat) and pykrx prices — returns are
NOT re-implemented. Slices net-vs-market by ticker and by market-cap tier (large/mid/small via FDR caps),
with the n>=30 floor. Runs on the EXPLORATORY 70% only (the sealed 152-video holdout stays untouched for
Task 4's OOS test). Writes to data/_moneyup_advisor/qa_audit/. No edge claim (in-sample descriptive).
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
from moneyup_advisor import config, tickers, phase1b as P
from moneyup_advisor.phase1b import _ohlcv5, _to_kst, _agg, INDEX_PROXY, BETA_LOOKBACK, KST, SCORED_LONG

OUT = config.DATA_DIR / "qa_audit"
# env-inflated market scale (this dataset's 삼성전자 ~1,953조) — tiers chosen for THIS scale; relative
# ordering (large > mid > small) is what matters. Reported explicitly.
LARGE, MID = 10e12, 1e12          # large >=10조, mid 1-10조, small <1조
MAJORS = {"005930": "Samsung", "000660": "SK Hynix", "035420": "NAVER"}


def _caps():
    import FinanceDataReader as fdr
    df = fdr.StockListing("KRX").set_index("Code")
    return {str(ix): float(df.loc[ix, "Marcap"]) for ix in df.index if str(df.loc[ix, "Marcap"]).replace(".", "").isdigit()}


def _tier(cap):
    if cap is None:
        return "unknown"
    return "large" if cap >= LARGE else "mid" if cap >= MID else "small"


def main():
    man = json.loads((OUT / "diagnostic" / "holdout_manifest.json").read_text(encoding="utf-8"))
    explore = set(man["exploratory_video_ids"])
    calls = [c for c in P.load_calls() if c["video_id"] in explore]
    pubs = [_to_kst(c["publish"]).date() for c in calls]
    start = (min(pubs) - dt.timedelta(days=BETA_LOOKBACK + 180)).strftime("%Y%m%d")
    end = dt.datetime.now(KST).strftime("%Y%m%d")
    idx_s = _ohlcv5(INDEX_PROXY, start, end)
    dates = sorted(idx_s)
    last_date = dates[-1] if dates else dt.datetime.now(KST).date()
    print(f"[capslice] exploratory videos={len(explore)} calls={len(calls)} — scoring (reuse phase1b)…", flush=True)

    caps = _caps()
    print(f"[capslice] FDR market caps loaded: {len(caps)} tickers", flush=True)

    scored = []
    for i, c in enumerate(calls):
        r = P.score_call(c, _ohlcv5(c["ticker"], start, end), idx_s, dates, last_date)
        scored.append({**c, **r})
        if i % 100 == 0:
            print(f"  scored {i}/{len(calls)}", flush=True)

    # BUY calls = corrected long-side classification (sign +1). Count per ticker (entered or not).
    buys = [s for s in scored if s.get("play") in SCORED_LONG]
    from collections import defaultdict
    per = defaultdict(lambda: {"n_calls": 0, "entered": [], "nets": []})
    for s in buys:
        tk = s["ticker"]
        per[tk]["n_calls"] += 1
        per[tk]["name"] = tickers.display_name(tk) or tk
        per[tk]["cap"] = caps.get(tk)
        per[tk]["tier"] = _tier(caps.get(tk))
        if s.get("status") == "entered" and s.get("net_abnormal") is not None:
            per[tk]["entered"].append(s)
            per[tk]["nets"].append(s["net_abnormal"])

    # per-ticker table
    rows = []
    for tk, d in per.items():
        a = _agg(d["nets"])
        rows.append({"ticker": tk, "name": d.get("name"), "n_buy_calls": d["n_calls"],
                     "n_entered": len(d["nets"]), "mean_net_pct": a["mean_pct"], "t": a["t"],
                     "cap_won": d.get("cap"), "tier": d.get("tier")})
    rows.sort(key=lambda r: -r["n_buy_calls"])

    # cap-tier breakdown (over ENTERED buy calls)
    tier_nets = defaultdict(list)
    for s in buys:
        if s.get("status") == "entered" and s.get("net_abnormal") is not None:
            tier_nets[_tier(caps.get(s["ticker"]))].append(s["net_abnormal"])
    tiers = {}
    for t in ("large", "mid", "small", "unknown"):
        a = _agg(tier_nets.get(t, []))
        tiers[t] = {"n_entered": a["n"], "mean_net_pct": a["mean_pct"], "t": a["t"],
                    "real": a["n"] >= 30, "verdict": ("powered (n>=30)" if a["n"] >= 30 else "UNDERPOWERED (n<30)")}

    # honest verdict
    powered = {t: tiers[t] for t in ("large", "mid", "small") if tiers[t]["real"]}
    if not powered:
        verdict = ("UNDERPOWERED — no cap tier reaches n>=30 entered buy calls; cannot say majors are "
                   "better or worse. (His ex-ante BUY calls concentrate in mid/small theme names; the "
                   "majors are watchlist context, rarely the primary call.)")
    else:
        best = max(powered, key=lambda t: powered[t]["mean_net_pct"])
        worst = min(powered, key=lambda t: powered[t]["mean_net_pct"])
        spread = powered[best]["mean_net_pct"] - powered[worst]["mean_net_pct"]
        if "large" in powered and powered["large"]["mean_net_pct"] > max(
                powered.get(t, {"mean_net_pct": -1e9})["mean_net_pct"] for t in ("mid", "small") if t in powered) and spread > 2:
            verdict = f"MAJORS BETTER — large tier mean net {powered['large']['mean_net_pct']}% beats mid/small by >2pp"
        elif spread <= 2:
            verdict = f"SAME across powered tiers (spread {round(spread,2)}pp) — cap tier does not separate the result"
        else:
            verdict = f"NON-majors better — {best} tier leads; large is not the strongest"

    n_majors = {tk: per.get(tk, {}).get("n_calls", 0) for tk in MAJORS}
    res = {"scope": "exploratory 70% (sealed 152-video holdout untouched)", "n_explore_videos": len(explore),
           "n_buy_calls": len(buys), "n_entered_buy": sum(len(d["nets"]) for d in per.values()),
           "cap_tiers_won": {"large>=": LARGE, "mid>=": MID, "small<": MID},
           "note_scale": "market caps are this dataset's (env-inflated ~4x vs real KR market); tiers chosen for this scale",
           "majors_buy_calls": {f"{tk} ({MAJORS[tk]})": n for tk, n in n_majors.items()},
           "cap_tier_breakdown": tiers, "verdict": verdict,
           "per_ticker_top40": rows[:40]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "phase1b_capslice.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    # console report
    print("\n=== MAJORS — ex-ante BUY calls ===")
    for tk in MAJORS:
        print(f"  {tk} {MAJORS[tk]:9} n_buy_calls={n_majors[tk]}")
    print("\n=== CAP-TIER breakdown (entered buy calls) ===")
    for t in ("large", "mid", "small", "unknown"):
        x = tiers[t]
        print(f"  {t:7} n={x['n_entered']:>4} mean_net%={x['mean_net_pct']} t={x['t']} -> {x['verdict']}")
    print("\n=== TOP tickers by #buy calls ===")
    for r in rows[:12]:
        print(f"  {r['ticker']} {str(r['name'])[:12]:12} calls={r['n_buy_calls']:>3} entered={r['n_entered']:>3} "
              f"mean%={r['mean_net_pct']} t={r['t']} tier={r['tier']} cap={('%.2e'%r['cap_won']) if r['cap_won'] else 'NA'}")
    print(f"\nVERDICT: {verdict}")
    print(f"saved -> {OUT/'phase1b_capslice.json'}")


if __name__ == "__main__":
    main()
