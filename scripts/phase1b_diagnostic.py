# -*- coding: utf-8 -*-
"""머니업 Phase 1B — Loss-Decomposition Diagnostic (EXPLORATORY; NOT a verdict).

Discipline (enforced in code):
  * Seals the most-recent 30% of videos by publish date as a HOLDOUT *before* any return is computed;
    analysis runs ONLY on the older 70%. The holdout manifest is written first.
  * Reuses phase1b.py's entry/exit/return/cost/beta machinery (score_call + _ohlcv5 + the exact net
    formula) — returns are NOT re-implemented.
  * Records EVERY slice tried (good and bad) + the total count, so the multiple-testing surface is visible.
  * Deterministic (no LLM in any number). Read-only. Writes ONLY to qa_audit/diagnostic/.

Outputs: diagnostic/phase1b_diagnostic.json (+ holdout_manifest.json). The narrative .md (ranked
hypotheses + draft pre-registrations) is authored from these numbers afterwards. No confirmatory test is run.
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
import numpy as np
from moneyup_advisor import config, tickers
from moneyup_advisor import phase1b as P
from moneyup_advisor.phase1b import _ohlcv5, _to_kst, _agg, COST_ROUNDTRIP, INDEX_PROXY, BETA_LOOKBACK, KST

OUT = config.DATA_DIR / "qa_audit" / "diagnostic"
OUT.mkdir(parents=True, exist_ok=True)
HOLDOUT_FRAC = 0.30
SLICES = []                      # every slice tried: (name, n, mean_pct, t)


def rec(name, vals):
    a = _agg([v for v in vals if v is not None])
    SLICES.append({"slice": name, "n": a["n"], "mean_pct": a["mean_pct"], "t": a["t"]})
    return SLICES[-1]


# 2차전지 / battery regime tag (deterministic code set) — sector is not in metadata
_BATTERY = {"086520", "247540", "005490", "066970", "373220", "006400", "096770", "003670",
            "278280", "020150", "457190", "137400", "121600", "051910", "051900"}


# --------------------------------------------------------------------------- #
# 0. SEAL THE HOLDOUT  (before any return is computed)
# --------------------------------------------------------------------------- #
def seal_holdout():
    vids = []
    for p in sorted(config.SHEET_DIR.glob("*.json")):
        try:
            s = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        pub = s.get("publish_datetime") or (s.get("publish_date") + "T23:59:59Z" if s.get("publish_date") else None)
        if s.get("video_id") and pub:
            vids.append((s["video_id"], pub))
    vids.sort(key=lambda x: (x[1], x[0]))
    n_hold = int(round(len(vids) * HOLDOUT_FRAC))
    holdout = vids[len(vids) - n_hold:]                 # most-recent 30%
    explore = vids[:len(vids) - n_hold]
    cutoff = holdout[0][1] if holdout else None
    manifest = {"sealed_at_rule": "most-recent 30% of videos by publish datetime",
                "n_videos_total": len(vids), "n_holdout_sealed": len(holdout),
                "n_exploratory": len(explore), "holdout_cutoff_publish": cutoff,
                "holdout_video_ids": sorted(v for v, _ in holdout),
                "exploratory_video_ids": sorted(v for v, _ in explore)}
    (OUT / "holdout_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[seal] HOLDOUT sealed: {len(holdout)} videos (>= {cutoff}) | exploratory: {len(explore)} videos")
    return set(v for v, _ in explore), set(v for v, _ in holdout), cutoff


# --------------------------------------------------------------------------- #
# return helpers — REUSE phase1b's exact formula
# --------------------------------------------------------------------------- #
def net_at(ss, idx_s, dates, i0, entry_px, beta, h, sign=1):
    """score_call's exact fixed-horizon net: sign*(gs - beta*gi) - cost. Market+beta adjusted."""
    j = i0 + h
    if j >= len(dates):
        return None
    ed, e0 = dates[j], dates[i0]
    if ed not in ss or e0 not in idx_s or ed not in idx_s or entry_px in (None, 0):
        return None
    gs = ss[ed][3] / entry_px - 1.0
    gi = idx_s[ed][3] / idx_s[e0][3] - 1.0
    return sign * (gs - beta * gi) - COST_ROUNDTRIP


def main():
    explore_vids, holdout_vids, cutoff = seal_holdout()

    # exploratory scored calls only — NEVER touch holdout
    calls = [c for c in P.load_calls() if c["video_id"] in explore_vids]
    pubs = [_to_kst(c["publish"]).date() for c in calls]
    start = (min(pubs) - dt.timedelta(days=BETA_LOOKBACK + 180)).strftime("%Y%m%d") if pubs else "20230101"
    end = dt.datetime.now(KST).strftime("%Y%m%d")
    idx_s = _ohlcv5(INDEX_PROXY, start, end)
    dates = sorted(idx_s)
    last_date = dates[-1] if dates else dt.datetime.now(KST).date()

    rows = []
    occ = {}                                            # ticker -> running mention count (publish order)
    for c in sorted(calls, key=lambda x: (x["publish"], x["video_id"])):
        ss = _ohlcv5(c["ticker"], start, end)
        r = P.score_call(c, ss, idx_s, dates, last_date)
        play = r.get("play")
        scored = r.get("status") == "entered"
        occ[c["ticker"]] = occ.get(c["ticker"], 0) + 1   # count ALL scored-eligible mentions in publish order
        if not scored or play not in ("LONG-GENERIC", "DIP-BUY", "SCHEDULE", "ROTATION", "BREAKOUT", "BEARISH"):
            continue
        ed = dt.date.fromisoformat(r["entry_date"]); i0 = dates.index(ed)
        entry_px = r["entry_px"]; beta = r.get("beta", 1.0) or 1.0
        nopen = ss[ed][0]                                # next-open anchor for D1/D2
        prevc = ss[dates[i0 - 1]][3] if i0 >= 1 and dates[i0 - 1] in ss else None
        # prior 20d run-up (trailing return BEFORE the call)
        run20 = None
        if i0 >= 21 and dates[i0 - 1] in ss and dates[i0 - 21] in ss:
            run20 = ss[dates[i0 - 1]][3] / ss[dates[i0 - 21]][3] - 1.0
        rows.append({
            "video": c["video_id"], "ticker": c["ticker"], "name": tickers.display_name(c["ticker"]) or c["ticker"],
            "play": play, "pub": _to_kst(c["publish"]).date().isoformat(), "month": _to_kst(c["publish"]).strftime("%Y-%m"),
            "occ": occ[c["ticker"]], "entry_date": r["entry_date"], "i0": i0,
            "entry_px": entry_px, "nopen": nopen, "beta": beta, "prevc": prevc, "run20": run20,
            "net_barrier": r.get("net_abnormal"),
            "gap_prevclose_to_open": (nopen / prevc - 1.0) if prevc else None,
            "ss_has": True,
        })

    LG = [r for r in rows if r["play"] == "LONG-GENERIC"]
    DB = [r for r in rows if r["play"] == "DIP-BUY"]
    print(f"[explore] scored rows: {len(rows)} (LONG-GENERIC {len(LG)}, DIP-BUY {len(DB)})")

    res = {"holdout_cutoff": cutoff, "n_holdout_videos": len(holdout_vids), "n_explore_videos": len(explore_vids),
           "n_scored_rows": len(rows), "n_long_generic": len(LG), "n_dip_buy": len(DB)}

    def reget(r):
        return _ohlcv5(r["ticker"], start, end)

    # ---- D1: entry timing — gap vs drift, + entry variants ----
    d1 = {}
    d1["mean_overnight_gap_LG"] = rec("D1 LG overnight gap (prevclose→open) %", [r["gap_prevclose_to_open"] for r in LG])
    d1["drift5_open_LG"] = rec("D1 LG post-entry drift open→5d (mkt-adj) %", [net_at(reget(r), idx_s, dates, r["i0"], r["nopen"], r["beta"], 5) for r in LG])
    # entry variants, measured to 5 trading days, market+beta adjusted (LONG-GENERIC)
    def var_open(r): return net_at(reget(r), idx_s, dates, r["i0"], r["nopen"], r["beta"], 5)
    def var_close(r):
        ss = reget(r); ed = dates[r["i0"]]
        return net_at(ss, idx_s, dates, r["i0"], ss[ed][3] if ed in ss else None, r["beta"], 5)
    def var_open1(r):
        ss = reget(r); j = r["i0"] + 1
        if j >= len(dates) or dates[j] not in ss: return None
        return net_at(ss, idx_s, dates, j, ss[dates[j]][0], r["beta"], 5)
    def var_pubclose(r):
        ss = reget(r); pub = _to_kst(next(c["publish"] for c in calls if c["video_id"] == r["video"] and c["ticker"] == r["ticker"]))
        pd = pub.date()
        if pub.hour >= 16 or pd not in ss or pd not in dates: return None    # only if published during/after a trading session it reacts to
        ip = dates.index(pd)
        return net_at(ss, idx_s, dates, ip, ss[pd][3], r["beta"], 5)
    d1["var_a_next_open_5d_LG"] = rec("D1 entry (a) next-open → 5d %", [var_open(r) for r in LG])
    d1["var_b_next_close_5d_LG"] = rec("D1 entry (b) next-close → 5d %", [var_close(r) for r in LG])
    d1["var_c_open_plus1_5d_LG"] = rec("D1 entry (c) open+1 → 5d %", [var_open1(r) for r in LG])
    d1["var_d_pubclose_5d_LG"] = rec("D1 entry (d) publish-day close → 5d %", [var_pubclose(r) for r in LG])

    # ---- D2: horizon curve (mkt-adj, ignore the ±8% barrier) ----
    d2 = {"LONG-GENERIC": {}, "DIP-BUY": {}}
    for hz in (1, 2, 3, 5, 10, 20):
        d2["LONG-GENERIC"][str(hz)] = rec(f"D2 LG h={hz} %", [net_at(reget(r), idx_s, dates, r["i0"], r["nopen"], r["beta"], hz) for r in LG])
        d2["DIP-BUY"][str(hz)] = rec(f"D2 DIP h={hz} %", [net_at(reget(r), idx_s, dates, r["i0"], r["nopen"], r["beta"], hz) for r in DB])

    # ---- D3: first-mention vs repeat + prior run-up regression ----
    first = [r for r in rows if r["occ"] == 1]
    repeat = [r for r in rows if r["occ"] >= 2]
    d3 = {"first_mention_net_barrier": rec("D3 first-mention net (barrier) %", [r["net_barrier"] for r in first]),
          "repeat_net_barrier": rec("D3 repeat(2+) net (barrier) %", [r["net_barrier"] for r in repeat]),
          "first_mention_LG_drift5": rec("D3 first-mention LG open→5d %", [net_at(reget(r), idx_s, dates, r["i0"], r["nopen"], r["beta"], 5) for r in first if r["play"] == "LONG-GENERIC"]),
          "repeat_LG_drift5": rec("D3 repeat LG open→5d %", [net_at(reget(r), idx_s, dates, r["i0"], r["nopen"], r["beta"], 5) for r in repeat if r["play"] == "LONG-GENERIC"])}
    # regress forward-5d (mkt-adj) on prior 20d run-up
    xs, ys = [], []
    for r in LG:
        f5 = net_at(reget(r), idx_s, dates, r["i0"], r["nopen"], r["beta"], 5)
        if r["run20"] is not None and f5 is not None:
            xs.append(r["run20"]); ys.append(f5)
    if len(xs) >= 10:
        xa, ya = np.array(xs), np.array(ys)
        slope, intercept = np.polyfit(xa, ya, 1)
        corr = float(np.corrcoef(xa, ya)[0, 1])
        # tercile by prior run-up
        order = np.argsort(xa); n = len(xa); t1, t2 = order[:n // 3], order[2 * n // 3:]
        d3["runup_regress"] = {"n": len(xs), "slope": round(float(slope), 4), "intercept_pct": round(float(intercept) * 100, 3), "corr": round(corr, 3)}
        d3["runup_low_tercile_fwd5"] = rec("D3 LG low prior-runup → fwd5 %", list(ya[t1]))
        d3["runup_high_tercile_fwd5"] = rec("D3 LG high prior-runup → fwd5 %", list(ya[t2]))
    else:
        d3["runup_regress"] = {"n": len(xs), "note": "insufficient run-up data"}

    # ---- D4: concentration & regime ----
    from collections import defaultdict
    by_t = defaultdict(list); by_m = defaultdict(list)
    for r in rows:
        if r["net_barrier"] is not None:
            by_t[r["ticker"]].append(r["net_barrier"]); by_m[r["month"]].append(r["net_barrier"])
    contrib = sorted(((tk, float(np.sum(v)), len(v), float(np.mean(v))) for tk, v in by_t.items()), key=lambda x: -abs(x[1]))
    top10 = [t[0] for t in contrib[:10]]; top3 = [t[0] for t in contrib[:3]]
    allnet = [r["net_barrier"] for r in rows if r["net_barrier"] is not None]
    d4 = {"headline_all": rec("D4 headline all scored (barrier) %", allnet),
          "ex_top3": rec("D4 excl top-3 contributors %", [r["net_barrier"] for r in rows if r["ticker"] not in top3 and r["net_barrier"] is not None]),
          "ex_top10": rec("D4 excl top-10 contributors %", [r["net_barrier"] for r in rows if r["ticker"] not in top10 and r["net_barrier"] is not None]),
          "battery_2cheonji": rec("D4 battery(2차전지) names %", [r["net_barrier"] for r in rows if r["ticker"] in _BATTERY and r["net_barrier"] is not None]),
          "non_battery": rec("D4 non-battery names %", [r["net_barrier"] for r in rows if r["ticker"] not in _BATTERY and r["net_barrier"] is not None]),
          "top10_contributors": [{"ticker": t[0], "name": tickers.display_name(t[0]) or t[0], "sum_net_pct": round(t[1] * 100, 1), "n": t[2], "mean_pct": round(t[3] * 100, 2)} for t in contrib[:10]],
          "by_month": {m: rec(f"D4 month {m} %", v) for m, v in sorted(by_m.items())}}
    # index regime over the exploratory sample
    e_dates = [d for d in dates if pubs and min(pubs) <= d <= max(pubs)]
    if len(e_dates) >= 2:
        d4["index_regime_return_pct"] = round((idx_s[e_dates[-1]][3] / idx_s[e_dates[0]][3] - 1.0) * 100, 2)

    # ---- D5: cost-aware contrarian (mirror) + shortability ----
    # market membership + market cap (one recent snapshot)
    kospi = kosdaq = set(); cap = {}
    try:
        import pykrx.stock as st
        snap = "20260626"
        kospi = set(st.get_market_ticker_list(snap, market="KOSPI"))
        kosdaq = set(st.get_market_ticker_list(snap, market="KOSDAQ"))
        capdf = st.get_market_cap_by_ticker(snap)
        cap = {ix: float(capdf.loc[ix, "시가총액"]) for ix in capdf.index}
    except Exception as e:
        print("[D5] market/cap snapshot unavailable:", str(e)[:80])
    BORROW_ANN = 0.05                                   # conservative annual borrow assumption (flagged)
    def mirror_net(r, hz=5):
        ss = reget(r)
        base = net_at(ss, idx_s, dates, r["i0"], r["nopen"], r["beta"], hz, sign=-1)   # short, beta-adj, incl 0.5% rt
        if base is None: return None
        return base - BORROW_ANN * hz / 252.0          # minus borrow over the holding window
    def shortable(tk):
        capv = cap.get(tk, 0.0)
        return (tk in kospi) and capv >= 1e12          # proxy: KOSPI + >=1조 market cap (liquid, generally borrowable)
    sub = [r for r in rows if shortable(r["ticker"])]
    d5 = {"mirror_all_gross_5d": rec("D5 mirror short all gross→5d (no cost) %", [(-1) * (net_at(reget(r), idx_s, dates, r["i0"], r["nopen"], r["beta"], 5) + COST_ROUNDTRIP) for r in rows]),
          "mirror_all_net_5d": rec("D5 mirror short all net→5d (0.5%rt) %", [net_at(reget(r), idx_s, dates, r["i0"], r["nopen"], r["beta"], 5, sign=-1) for r in rows]),
          "mirror_shortable_net_5d": rec("D5 mirror short on SHORTABLE subset net+borrow→5d %", [mirror_net(r) for r in sub]),
          "n_shortable_proxy": len(sub), "n_total": len(rows),
          "pct_shortable_proxy": round(100 * len(sub) / len(rows), 1) if rows else None,
          "shortability_proxy": "KOSPI-listed AND market_cap>=1조 (KOSDAQ small-caps treated NOT retail-shortable)",
          "borrow_assumption_annual": BORROW_ANN, "caveats": "exact retail shortability / borrow fee / 공매도 과열종목 list per call-date NOT fully determinable from pykrx — proxy + conservative borrow used; flagged."}

    res.update({"D1": d1, "D2": d2, "D3": d3, "D4": d4, "D5": d5,
                "slices_examined_total": len(SLICES), "all_slices": SLICES})
    (OUT / "phase1b_diagnostic.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[done] slices examined: {len(SLICES)} | wrote {OUT/'phase1b_diagnostic.json'}")
    # console summary (ascii-safe)
    print("D2 LONG-GENERIC horizon curve (mean%/t/n):")
    for hz in (1, 2, 3, 5, 10, 20):
        s = d2["LONG-GENERIC"][str(hz)]; print(f"   h={hz:>2} mean%={s['mean_pct']} t={s['t']} n={s['n']}")
    print("D2 DIP-BUY horizon curve:")
    for hz in (1, 2, 3, 5, 10, 20):
        s = d2["DIP-BUY"][str(hz)]; print(f"   h={hz:>2} mean%={s['mean_pct']} t={s['t']} n={s['n']}")
    print("D1 gap vs drift:", d1["mean_overnight_gap_LG"], d1["drift5_open_LG"])
    print("D1 entry variants 5d:", {k: (d1[k]["mean_pct"], d1[k]["t"], d1[k]["n"]) for k in ("var_a_next_open_5d_LG","var_b_next_close_5d_LG","var_c_open_plus1_5d_LG","var_d_pubclose_5d_LG")})
    print("D3 first vs repeat:", d3["first_mention_net_barrier"], d3["repeat_net_barrier"], "| runup:", d3.get("runup_regress"))
    print("D4 headline/ex3/ex10:", d4["headline_all"]["mean_pct"], d4["ex_top3"]["mean_pct"], d4["ex_top10"]["mean_pct"], "| battery:", d4["battery_2cheonji"], "| index regime%:", d4.get("index_regime_return_pct"))
    print("D5 mirror net all/shortable:", d5["mirror_all_net_5d"], d5["mirror_shortable_net_5d"], "| %shortable:", d5["pct_shortable_proxy"])


if __name__ == "__main__":
    main()
