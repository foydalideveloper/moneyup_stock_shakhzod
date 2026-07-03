# -*- coding: utf-8 -*-
"""TEST 2 — contrarian tradeability, best-effort PROXY (real KRX shortability/borrow data PENDING).

Does the contrarian (short his buy calls; +1.93% OOS on paper) survive shortability + borrow costs?
Real KRX 대차/공매도 과열 data is NOT available here — so the borrowable universe is PROXIED by FDR market
cap + exchange (KOSPI large-cap ≈ likely borrowable; small/mid KOSDAQ theme names ≈ likely not). Labelled
PROXY throughout; NO tradeable-edge claim. Read-only, exploratory 70% primary (holdout noted); no GPU.

Per-call mirror = short the same entry→exit, β/size-adj, net of the short's OWN 0.50% round-trip
(= -long_net - 2×round-trip). Borrow fee (annualized) is prorated over each call's actual holding days.
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
                                     SCORED_LONG, COST_ROUNDTRIP)

OUT = config.DATA_DIR / "qa_audit"
# env-inflated caps (삼성 ~1,953조 here vs ~470조 real ≈ 4.15×). KOSPI200-style large-cap ≈ ≥10조 env
# (≈ ~2.4조 real). Report the count at looser thresholds too.
LARGE_THR = 10e12
BORROW_RATES = [3, 6, 12, 20]                          # annualized % ; prorated over each call's hold days


def _caps_markets():
    import FinanceDataReader as fdr
    df = fdr.StockListing("KRX").set_index("Code")
    out = {}
    for ix in df.index:
        try:
            cap = float(df.loc[ix, "Marcap"])
        except Exception:
            cap = None
        out[str(ix)] = {"cap": cap, "market": str(df.loc[ix, "Market"])}
    return out


def _score_set(vids, calls, idx_s, dates, last_date, start, end):
    rows = []
    for c in calls:
        if c["video_id"] not in vids:
            continue
        r = P.score_call(c, _ohlcv5(c["ticker"], start, end), idx_s, dates, last_date)
        if r.get("play") not in SCORED_LONG or r.get("status") != "entered" or r.get("net_abnormal") is None:
            continue
        ed = dt.date.fromisoformat(r["entry_date"])
        xd = dt.date.fromisoformat(r["exit_date"])
        hold = max(1, (xd - ed).days)
        mirror = -r["net_abnormal"] - 2 * COST_ROUNDTRIP        # contrarian net of short's own round-trip
        rows.append({"ticker": c["ticker"], "hold_days": hold, "mirror": mirror})
    return rows


def _contrarian(rows, cm, label):
    def borrowable(r, kospi_only):
        info = cm.get(r["ticker"])
        if not info or info["cap"] is None:
            return None
        if info["cap"] < LARGE_THR:
            return False
        if kospi_only and info["market"] != "KOSPI":
            return False
        return True

    res = {"label": label, "n_all": len(rows)}
    for pname, kospi_only in (("proxyA_KOSPI_large", True), ("proxyB_any_large", False)):
        sub = [r for r in rows if borrowable(r, kospi_only)]
        a = _agg([r["mirror"] for r in sub])
        # borrow sensitivity (prorated over hold days)
        borrow = {}
        for rate in BORROW_RATES:
            net = [r["mirror"] - (rate / 100.0) * (r["hold_days"] / 365.0) for r in sub]
            b = _agg(net)
            borrow[f"{rate}pct"] = {"mean_net_pct": b["mean_pct"], "t": b["t"]}
        res[pname] = {"n": a["n"], "pct_of_all": round(100 * a["n"] / max(1, len(rows)), 1),
                      "mean_before_borrow_pct": a["mean_pct"], "t_before_borrow": a["t"],
                      "avg_hold_days": round(sum(r["hold_days"] for r in sub) / max(1, len(sub)), 1),
                      "after_borrow": borrow}
    return res


def main():
    man = json.loads((OUT / "diagnostic" / "holdout_manifest.json").read_text(encoding="utf-8"))
    ex, ho = set(man["exploratory_video_ids"]), set(man["holdout_video_ids"])
    calls = P.load_calls()
    pubs = [_to_kst(c["publish"]).date() for c in calls]
    start = (min(pubs) - dt.timedelta(days=BETA_LOOKBACK + 180)).strftime("%Y%m%d")
    end = dt.datetime.now(KST).strftime("%Y%m%d")
    idx_s = _ohlcv5(INDEX_PROXY, start, end)
    dates = sorted(idx_s)
    last_date = dates[-1] if dates else dt.datetime.now(KST).date()
    print("[contrarian-proxy] scoring buy calls + loading FDR caps/markets…", flush=True)
    cm = _caps_markets()

    ex_rows = _score_set(ex, calls, idx_s, dates, last_date, start, end)
    ho_rows = _score_set(ho, calls, idx_s, dates, last_date, start, end)

    # borrowable-universe split (what % of his buy calls are even plausibly shortable)
    def split_counts(rows):
        cnt = {"kospi_large": 0, "any_large": 0, "unknown_cap": 0, "total": len(rows)}
        for r in rows:
            info = cm.get(r["ticker"])
            if not info or info["cap"] is None:
                cnt["unknown_cap"] += 1; continue
            if info["cap"] >= LARGE_THR:
                cnt["any_large"] += 1
                if info["market"] == "KOSPI":
                    cnt["kospi_large"] += 1
        return cnt

    out = {"test": "TEST 2 contrarian tradeability — PROXY (real KRX shortability/borrow PENDING)",
           "label": "PROXY — NOT a tradeable-edge claim",
           "large_cap_threshold_env": LARGE_THR,
           "threshold_note": "≥10조 in this env's ~4× inflated scale ≈ ~2.4조 real ≈ KOSPI200-style large-cap",
           "borrowable_proxy": "A = KOSPI-listed & cap≥thr (task spec) ; B = any-exchange large-cap "
                               "(more realistic — 에코프로/알테오젠 are KOSDAQ large-caps that ARE shorted)",
           "full_set_contrarian_reference": {
               "exploratory": _agg([r["mirror"] for r in ex_rows]),
               "holdout": _agg([r["mirror"] for r in ho_rows])},
           "borrowable_split": {"exploratory": split_counts(ex_rows), "holdout": split_counts(ho_rows)},
           "exploratory": _contrarian(ex_rows, cm, "exploratory"),
           "holdout": _contrarian(ho_rows, cm, "holdout"),
           "gwamyeoldo_flag": "공매도 과열종목 status UNAVAILABLE here — some borrowable names may have been "
                              "short-banned on the call date; the cap/exchange proxy cannot see this.",
           "what_real_data_would_change": "real KRX 대차잔고/대차가능 + per-name borrow fee + 공매도 과열/금지 "
                                          "history would replace the cap proxy (exact borrowable set), replace "
                                          "the flat borrow-rate sweep with actual per-name fees, and drop "
                                          "short-banned dates — likely SHRINKING the tradeable subset further."}

    # honest verdict — check BOTH exploratory survival AND whether the (already-unsealed, noted) holdout confirms
    ea, eb = out["exploratory"]["proxyA_KOSPI_large"], out["exploratory"]["proxyB_any_large"]
    ha, hb = out["holdout"]["proxyA_KOSPI_large"], out["holdout"]["proxyB_any_large"]
    def surv6(blk):
        if blk["n"] < 30:
            return None
        m6 = blk["after_borrow"]["6pct"]
        return (m6["mean_net_pct"] or 0) > 0 and (m6["t"] or 0) >= 3.5
    exp_surv = bool(surv6(ea) or surv6(eb))
    hold_conf = bool(surv6(ha) or surv6(hb))
    head = (f"PROXY verdict — exploratory: KOSPI-large n={ea['n']} before {ea['mean_before_borrow_pct']}% "
            f"(t {ea['t_before_borrow']}) → @20% borrow {ea['after_borrow']['20pct']['mean_net_pct']}% "
            f"(t {ea['after_borrow']['20pct']['t']}); any-large n={eb['n']} before {eb['mean_before_borrow_pct']}% "
            f"→ @20% {eb['after_borrow']['20pct']['mean_net_pct']}%. "
            f"NOTED HOLDOUT: KOSPI-large n={ha['n']} before {ha['mean_before_borrow_pct']}% (t {ha['t_before_borrow']}); "
            f"any-large n={hb['n']} before {hb['mean_before_borrow_pct']}% (t {hb['t_before_borrow']}). ")
    if exp_surv and hold_conf:
        tail = ("Survives the borrow PROXY in-sample AND on the noted holdout — but it is a PROXY; a clean "
                "one-shot needs real KRX shortability/borrow data + a FRESH forward OOS. Propose pre-registration.")
    elif exp_surv and not hold_conf:
        tail = ("Survives the borrow PROXY IN-SAMPLE but the noted holdout does NOT confirm (collapses to ~0, "
                "not significant) → NOT robust. Only ~20–31% of his calls are even plausibly borrowable (the "
                "big-fade mid/small theme names are excluded). NO tradeable edge. The 152-video holdout is "
                "already spent on the contrarian, so a clean confirmation requires real KRX shortability/borrow/"
                "과열 data AND a fresh forward out-of-sample — not another pass on spent data.")
    else:
        tail = "Does not survive the borrow PROXY even in-sample. No edge; no holdout warranted."
    out["exploratory_survives_proxy"] = exp_surv
    out["noted_holdout_confirms"] = hold_conf
    out["verdict"] = head + tail + " Still a PROXY — no tradeable edge claimed."
    (OUT / "contrarian_proxy.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== TEST 2 contrarian PROXY (real KRX shortability PENDING) ===")
    print("full-set contrarian (reference): exploratory mean=%s%% (t %s, n %s) | holdout mean=%s%% (t %s, n %s)" % (
        out["full_set_contrarian_reference"]["exploratory"]["mean_pct"], out["full_set_contrarian_reference"]["exploratory"]["t"],
        out["full_set_contrarian_reference"]["exploratory"]["n"], out["full_set_contrarian_reference"]["holdout"]["mean_pct"],
        out["full_set_contrarian_reference"]["holdout"]["t"], out["full_set_contrarian_reference"]["holdout"]["n"]))
    sc = out["borrowable_split"]["exploratory"]
    print("borrowable split (exploratory): KOSPI-large=%d any-large=%d unknown=%d of %d (%.0f%% any-large)" % (
        sc["kospi_large"], sc["any_large"], sc["unknown_cap"], sc["total"], 100 * sc["any_large"] / max(1, sc["total"])))
    for scope in ("exploratory", "holdout"):
        for pn in ("proxyA_KOSPI_large", "proxyB_any_large"):
            b = out[scope][pn]
            print(f"\n[{scope} · {pn}] n={b['n']} ({b['pct_of_all']}% of calls) avg_hold={b['avg_hold_days']}d "
                  f"| before borrow mean={b['mean_before_borrow_pct']}% t={b['t_before_borrow']}")
            for rate in BORROW_RATES:
                x = b["after_borrow"][f"{rate}pct"]
                print(f"     borrow {rate:>2}%/yr -> net {x['mean_net_pct']}% (t {x['t']})")
    print(f"\nVERDICT: {out['verdict']}")
    print(f"saved -> {OUT/'contrarian_proxy.json'}")


if __name__ == "__main__":
    main()
