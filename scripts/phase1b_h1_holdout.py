# -*- coding: utf-8 -*-
"""Task 5 close — ONE-SHOT holdout test, H1 (mega-cap) ONLY. H2 NOT run (no shortability data).

Verifies the locked pre-registration is intact (sha256), then runs the UNCHANGED phase1b.score_call once
on the SEALED holdout's mega-cap (005930/035420/000660) buy calls. Decision rule is fixed by the lock:
mega-cap long edge IFF n>=30 AND mean_net>0 AND t>=+3.92 (Bonferroni K=2). No tuning, no re-run.
"""
from __future__ import annotations
import datetime as dt
import hashlib
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
from moneyup_advisor.phase1b import _ohlcv5, _to_kst, _agg, INDEX_PROXY, BETA_LOOKBACK, KST, SCORED_LONG

OUT = config.DATA_DIR / "qa_audit"
MEGA = {"005930": "Samsung", "035420": "NAVER", "000660": "SK Hynix"}
BONF_BAR = 3.92                                   # locked: Bonferroni two-sided, K=2 hypotheses
LOCK_SHA = "445d1745bc5779afc3103778f5ecdb8b57d3034c9b9db8b290db29cb52e6ca57"


def main():
    # 1) integrity: pre-registration unchanged since lock
    pre = OUT / "phase1b_prereg.md"
    sha = hashlib.sha256(pre.read_bytes()).hexdigest()
    lock_ok = (sha == LOCK_SHA)
    print(f"[H1] pre-reg sha256={sha}\n     lock intact={lock_ok}")
    if not lock_ok:
        print("ABORT — pre-registration changed since lock; refuse to run."); return

    man = json.loads((OUT / "diagnostic" / "holdout_manifest.json").read_text(encoding="utf-8"))
    holdout = set(man["holdout_video_ids"])
    calls = [c for c in P.load_calls() if c["video_id"] in holdout and c["ticker"] in MEGA]
    print(f"[H1] holdout mega-cap calls: {len(calls)}")
    if not calls:
        print("H1 RESULT: n=0 — no mega-cap buy calls in the holdout -> FAIL (n<30)."); return

    allpubs = [_to_kst(c["publish"]).date() for c in P.load_calls()]
    start = (min(allpubs) - dt.timedelta(days=BETA_LOOKBACK + 180)).strftime("%Y%m%d")
    end = dt.datetime.now(KST).strftime("%Y%m%d")
    idx_s = _ohlcv5(INDEX_PROXY, start, end)
    dates = sorted(idx_s)
    last_date = dates[-1] if dates else dt.datetime.now(KST).date()

    nets, per = [], {}
    for c in calls:
        r = P.score_call(c, _ohlcv5(c["ticker"], start, end), idx_s, dates, last_date)
        per.setdefault(c["ticker"], {"entered": 0, "nets": []})
        if r.get("play") in SCORED_LONG and r.get("status") == "entered" and r.get("net_abnormal") is not None:
            nets.append(r["net_abnormal"]); per[c["ticker"]]["entered"] += 1; per[c["ticker"]]["nets"].append(r["net_abnormal"])

    a = _agg(nets)
    edge = bool(a["n"] >= 30 and (a["mean_pct"] or 0) > 0 and (a["t"] or 0) >= BONF_BAR)
    verdict = ("EDGE (rule fired)" if edge else
               f"NO EDGE — rule not met (need n>=30 & mean>0 & t>=+{BONF_BAR})")
    res = {"test": "H1 mega-cap one-shot holdout (LOCKED)", "lock_sha256_ok": lock_ok,
           "H2_run": False, "H2_reason": "no shortability data authorized",
           "tickers": MEGA, "n_entered": a["n"], "mean_net_pct": a["mean_pct"], "t": a["t"],
           "bonferroni_bar": BONF_BAR, "decision_rule": "edge iff n>=30 & mean>0 & t>=+3.92",
           "edge": edge, "verdict": verdict,
           "per_ticker": {k: {"entered": v["entered"], **_agg(v["nets"])} for k, v in per.items()},
           "stated_prior": "exploratory mega-cap -2.46% t-1.96 (NS) -> expected FAIL"}
    (OUT / "phase1b_h1_holdout.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== H1 mega-cap — ONE-SHOT HOLDOUT (sealed) ===")
    print(f"  n_entered={a['n']}  mean_net%={a['mean_pct']}  t={a['t']}  vs bar +{BONF_BAR}")
    for k, v in per.items():
        aa = _agg(v["nets"])
        print(f"    {k} {MEGA[k]:9} entered={v['entered']} mean%={aa['mean_pct']} t={aa['t']}")
    print(f"\n  DECISION: {verdict}")
    print(f"saved -> {OUT/'phase1b_h1_holdout.json'}")


if __name__ == "__main__":
    main()
