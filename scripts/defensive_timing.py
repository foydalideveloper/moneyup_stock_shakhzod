# -*- coding: utf-8 -*-
"""TEST 1 — defensive-timing: do 머니업's 'go to cash / step aside' (HOLD-WAIT-CASH) calls predict
market drops? Read-only, exploratory 70% (holdout noted); no GPU; no re-extraction.

Forward KOSPI/KOSDAQ returns (ETF proxies 069500 / 229200 — pykrx index endpoints fail here) over
h=1/5/10/20 trading days AFTER each market-level defensive call, vs (a) the unconditional baseline and
(b) a random-timing bootstrap null. Confound control: also report the BEFORE return (reactive vs
predictive). Secondary: forward index returns after BUY vs after DEFENSIVE calls. Deflated bar across the
horizons; n>=30 floor.
"""
from __future__ import annotations
import datetime as dt
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
import numpy as np
from moneyup_advisor import config, phase1b as P
from moneyup_advisor.phase1b import _ohlcv5, _to_kst, _nw_tstat, KST

OUT = config.DATA_DIR / "qa_audit"
RNG = np.random.RandomState(20260702)
H = [1, 5, 10, 20]
INDICES = {"KOSPI~069500": "069500", "KOSDAQ~229200": "229200"}
NBOOT = 5000

MKT = ["시장", "증시", "지수", "코스피", "코스닥", "전체", "대세", "인덱스", "전반", "뉴욕", "나스닥", "매크로"]
CAUTION = ["보수적", "관망", "현금", "비중 축소", "비중을 줄", "비중 줄", "대기", "쉬", "지켜보", "조심",
           "방어", "기다리", "신중", "리스크 관리", "리스크관리", "위험 관리"]


def _has(q, kws):
    q = q or ""
    return any(k in q for k in kws)


def _idx_series(code, start, end):
    ss = _ohlcv5(code, start, end)
    dates = sorted(ss)
    closes = np.array([ss[d][3] for d in dates], float)
    return dates, closes


def _ref(dates, pubdate):
    for i, d in enumerate(dates):
        if d >= pubdate:
            return i
    return None


def _fwd(closes, i, h):
    return float(closes[i + h] / closes[i] - 1) if (i is not None and i + h < len(closes)) else None


def _bwd(closes, i, h):
    return float(closes[i] / closes[i - h] - 1) if (i is not None and i - h >= 0) else None


def _t_for_p(p):
    from moneyup_advisor.phase1b import _t_for_p as f
    return f(p)


def _block(dates, closes, refs, label):
    """Forward/before returns for a set of reference indices; vs baseline + random-timing bootstrap."""
    n_valid = sum(1 for r in refs if r is not None)
    res = {"label": label, "n": n_valid, "horizons": {}}
    for h in H:
        fwd = [_fwd(closes, r, h) for r in refs]
        fwd = [x for x in fwd if x is not None]
        bwd = [_bwd(closes, r, h) for r in refs]
        bwd = [x for x in bwd if x is not None]
        # baseline = unconditional mean forward h-day return across all sample days
        allf = [_fwd(closes, i, h) for i in range(len(closes) - h)]
        allf = np.array([x for x in allf if x is not None], float)
        base = float(allf.mean())
        obs = float(np.mean(fwd)) if fwd else None
        # random-timing bootstrap: mean forward return of n random days
        pboot = None
        if fwd:
            valid_i = np.arange(len(closes) - h)
            nulls = np.array([allf[RNG.choice(len(allf), size=len(fwd), replace=True)].mean() for _ in range(NBOOT)])
            pboot = float((nulls <= obs).mean())        # one-sided: obs BELOW random timing?
        abn = [x - base for x in fwd]
        t_abn = _nw_tstat(abn, 0) if len(abn) >= 5 else None
        res["horizons"][str(h)] = {
            "n_fwd": len(fwd), "fwd_mean_pct": round(obs * 100, 3) if obs is not None else None,
            "baseline_mean_pct": round(base * 100, 3),
            "abnormal_pct": round((obs - base) * 100, 3) if obs is not None else None,
            "t_abnormal": t_abn, "boot_p_below_random": round(pboot, 4) if pboot is not None else None,
            "before_mean_pct": round(float(np.mean(bwd)) * 100, 3) if bwd else None}
    return res


def main():
    man = json.loads((OUT / "diagnostic" / "holdout_manifest.json").read_text(encoding="utf-8"))
    ex = set(man["exploratory_video_ids"])
    calls = [c for c in P.load_calls() if c["video_id"] in ex]
    defensive = [c for c in calls if P.classify_call(c).get("bucket") == "HOLD-WAIT-CASH"]
    buys = [c for c in calls if P.classify_call(c).get("segment") in P.SCORED_LONG]
    loose = [c for c in defensive if _has(c.get("quote"), MKT)]
    strict = [c for c in defensive if _has(c.get("quote"), MKT) and _has(c.get("quote"), CAUTION)]
    stocklvl = [c for c in defensive if not _has(c.get("quote"), MKT)]
    print(f"[def-timing] exploratory: defensive(HOLD-WAIT-CASH)={len(defensive)} "
          f"market-loose={len(loose)} market-strict={len(strict)} stock-level={len(stocklvl)} buys={len(buys)}", flush=True)

    pubs = [_to_kst(c["publish"]).date() for c in calls]
    start = (min(pubs) - dt.timedelta(days=150)).strftime("%Y%m%d")
    end = dt.datetime.now(KST).strftime("%Y%m%d")

    K = len(H) * len(INDICES)                                   # deflated across horizons×indices
    base_p = 2 * (0.5 * math.erfc(3.5 / math.sqrt(2)))
    deflated_bar = round(_t_for_p(base_p / K), 3)

    out = {"test": "TEST 1 defensive-timing", "scope": "exploratory 70% (holdout noted)",
           "index_proxies": INDICES, "note": "pykrx index endpoints fail here → KODEX ETF proxies",
           "n": {"defensive": len(defensive), "market_loose": len(loose), "market_strict": len(strict),
                 "stock_level": len(stocklvl), "buys": len(buys)},
           "K_deflated": K, "deflated_tstat_bar": deflated_bar, "bootstrap_n": NBOOT,
           "market_loose": {}, "market_strict": {}, "secondary_buy_vs_defensive": {}}

    for iname, code in INDICES.items():
        dates, closes = _idx_series(code, start, end)
        for setname, cset in (("market_loose", loose), ("market_strict", strict)):
            refs = [_ref(dates, _to_kst(c["publish"]).date()) for c in cset]
            out[setname][iname] = _block(dates, closes, refs, f"{setname}@{iname}")
        # secondary: forward mean after BUY vs after DEFENSIVE (all defensive)
        rb = [_ref(dates, _to_kst(c["publish"]).date()) for c in buys]
        rd = [_ref(dates, _to_kst(c["publish"]).date()) for c in defensive]
        sec = {}
        for h in H:
            fb = [x for x in (_fwd(closes, r, h) for r in rb) if x is not None]
            fd = [x for x in (_fwd(closes, r, h) for r in rd) if x is not None]
            sec[str(h)] = {"after_buy_mean_pct": round(float(np.mean(fb)) * 100, 3) if fb else None,
                           "after_defensive_mean_pct": round(float(np.mean(fd)) * 100, 3) if fd else None,
                           "def_minus_buy_pp": round((np.mean(fd) - np.mean(fb)) * 100, 3) if (fb and fd) else None}
        out["secondary_buy_vs_defensive"][iname] = sec

    # verdict: does a market-level set show forward returns significantly BELOW baseline (deflated), with
    # a NEGATIVE forward beyond the before-drift (predictive not merely reactive)?
    survived = []
    for setname in ("market_strict", "market_loose"):
        for iname in INDICES:
            b = out[setname][iname]
            if b["n"] < 30:
                continue
            for h, hh in b["horizons"].items():
                if (hh["abnormal_pct"] or 0) < 0 and (hh["boot_p_below_random"] or 1) < base_p / K \
                        and (hh["t_abnormal"] or 0) <= -deflated_bar:
                    survived.append(f"{setname}/{iname}/h{h}")
    out["survivors_deflated"] = survived
    if not any(out[s][i]["n"] >= 30 for s in ("market_strict", "market_loose") for i in INDICES):
        out["verdict"] = "UNDERPOWERED — market-level defensive n<30; cannot conclude"
    elif survived:
        out["verdict"] = f"SURVIVES exploratory (forward underperformance, deflated): {survived} — propose holdout"
    else:
        out["verdict"] = ("NO defensive-timing edge — forward index returns after his defensive calls are "
                          "NOT significantly below random timing at the deflated bar")
    (OUT / "defensive_timing.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== TEST 1 defensive-timing (deflated bar t={deflated_bar}, K={K}, boot={NBOOT}) ===")
    for setname in ("market_strict", "market_loose"):
        for iname in INDICES:
            b = out[setname][iname]
            print(f"\n[{setname} · {iname}]  n={b['n']}")
            for h in H:
                x = b["horizons"][str(h)]
                print(f"   h{h:>2}d fwd={x['fwd_mean_pct']}% base={x['baseline_mean_pct']}% "
                      f"abn={x['abnormal_pct']}% t={x['t_abnormal']} boot_p={x['boot_p_below_random']} "
                      f"| before={x['before_mean_pct']}%")
    print("\n=== secondary: forward after BUY vs after DEFENSIVE ===")
    for iname in INDICES:
        for h in H:
            s = out["secondary_buy_vs_defensive"][iname][str(h)]
            print(f"   {iname} h{h:>2}d  buy={s['after_buy_mean_pct']}%  def={s['after_defensive_mean_pct']}%  "
                  f"def-buy={s['def_minus_buy_pp']}pp")
    print(f"\nVERDICT: {out['verdict']}")
    print(f"saved -> {OUT/'defensive_timing.json'}")


if __name__ == "__main__":
    main()
