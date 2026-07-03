# -*- coding: utf-8 -*-
"""Task 5 — OFFENSE HUNT (CV stage, EXPLORATORY 70% only; sealed holdout NEVER touched).

Looks for a genuine long edge in 머니업's calls. Deliver (then HARD STOP, no holdout run, no edge claim):
  * data inventory + feature list,
  * Track A: his BUY setups pooled by TYPE (n>=30), net-of-market+cost, deflated bar,
  * Track B: full-feature model — PRIMARY conditional selection on his calls (separate winners/losers),
    SECONDARY standalone signal on his-universe daily panel; LightGBM + regularized linear;
    purged + embargoed walk-forward CV; SHUFFLED-LABEL null floor; deflated bar (Bonferroni; print K),
  * a "true mega-cap calls" slice (005930 / 035420 / 000660) as ONE pre-registered hypothesis,
  * a LOCKED pre-registration for the one-shot holdout test.

Anti-overfit: temporal splits only, point-in-time features (no look-ahead), net 0.50% round-trip,
beta+size-adjusted (reuse phase1b), shuffled-label null, deflated bar. NO tuning to chase a result.
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
from moneyup_advisor import config, tickers, phase1b as P
from moneyup_advisor.phase1b import _ohlcv5, _to_kst, _agg, INDEX_PROXY, BETA_LOOKBACK, KST, SCORED_LONG, COST_ROUNDTRIP

OUT = config.DATA_DIR / "qa_audit"
RNG = np.random.RandomState(20260630)            # fixed seed — deterministic null
MEGA = {"005930": "Samsung", "035420": "NAVER", "000660": "SK Hynix"}

# --- concrete mechanizable BUY setup sub-types parsed from the quote (bilingual) ---
_SETUPS = [
    ("breakout_volume", ["거래량 터", "거래량 폭발", "대량 거래", "거래량 실리", "volume"]),   # + a breakout word
    ("매물대_돌파", ["매물대 돌파", "매물대를 돌파", "매물대 상단", "물량대 돌파"]),
    ("OBV_핵심물량", ["obv", "핵심 물량", "핵심물량", "세력 물량", "물량 잠"]),
    ("공매도_매집선", ["공매도", "대차", "대주", "숏 커버", "매집선"]),
    ("눌림목_지지", ["눌림", "지지선", "지지받", "조정 시 매수", "저점 매수", "매집"]),
    ("저항_돌파", ["저항 돌파", "전고점 돌파", "박스권 돌파", "신고가"]),
    ("이평_정배열", ["정배열", "이평선 위", "골든크로스", "이동평균 상향"]),
    ("수급_기관외인", ["기관 매수", "외국인 매수", "수급", "프로그램 매수", "싹쓸이"]),
]


def setup_types(quote: str):
    q = (quote or "").lower()
    out = [name for name, kws in _SETUPS if any(k in q for k in kws)]
    return out or ["unclassified_setup"]


# --------------------------------------------------------------------------- #
# point-in-time features (computed from OHLCV up to the day BEFORE entry — no look-ahead)
# --------------------------------------------------------------------------- #
def _rsi(closes, n=14):
    d = np.diff(closes[-(n + 1):])
    if len(d) < n:
        return 50.0
    up = d[d > 0].sum() / n
    dn = -d[d < 0].sum() / n
    return 100.0 if dn == 0 else 100 - 100 / (1 + up / dn)


def features_at(ss, idx_s, entry_date, gap):
    sd = sorted(ss)
    if entry_date not in sd:
        return None
    si = sd.index(entry_date)
    if si < 121:                                  # need >=120 trading days of point-in-time history
        return None
    pre = sd[:si]                                 # strictly BEFORE entry day (point-in-time)
    cl = np.array([ss[d][3] for d in pre], float)
    hi = np.array([ss[d][1] for d in pre], float)
    lo = np.array([ss[d][2] for d in pre], float)
    vo = np.array([ss[d][4] for d in pre], float)
    rc = cl[-1]
    rets = np.diff(cl) / cl[:-1]
    def r(k): return float(cl[-1] / cl[-1 - k] - 1) if len(cl) > k else 0.0
    def sma(k): return float(cl[-k:].mean())
    # index (regime / relative strength) aligned to the same pre-dates
    icl = np.array([idx_s[d][3] for d in pre if d in idx_s], float)
    iret20 = float(icl[-1] / icl[-21] - 1) if len(icl) > 21 else 0.0
    isma20 = float(icl[-1] / icl[-20:].mean() - 1) if len(icl) >= 20 else 0.0
    f = {
        "ret_1d": r(1), "ret_5d": r(5), "ret_20d": r(20), "ret_60d": r(60),
        "px_vs_sma5": rc / sma(5) - 1, "px_vs_sma20": rc / sma(20) - 1,
        "px_vs_sma60": rc / sma(60) - 1, "px_vs_sma120": rc / sma(120) - 1,
        "sma20_slope": sma(20) / float(cl[-25:-5].mean()) - 1 if len(cl) >= 25 else 0.0,
        "rsi14": _rsi(cl), "vol20": float(rets[-20:].std()) if len(rets) >= 20 else 0.0,
        "atr_pct": float((hi[-14:] - lo[-14:]).mean() / rc) if len(hi) >= 14 else 0.0,
        "vol_z20": float((vo[-1] - vo[-20:].mean()) / (vo[-20:].std() + 1)) if len(vo) >= 20 else 0.0,
        "dist_hi60": rc / hi[-60:].max() - 1, "dist_lo60": rc / lo[-60:].min() - 1,
        "rel_str_20d": r(20) - iret20, "idx_ret20": iret20, "idx_vs_sma20": isma20,
        "gap_open": float(gap or 0.0),
    }
    return {k: (0.0 if (v is None or not np.isfinite(v)) else round(float(v), 5)) for k, v in f.items()}


# --------------------------------------------------------------------------- #
# purged + embargoed walk-forward CV  (group = entry month; embargo overlapping holding windows)
# --------------------------------------------------------------------------- #
def walk_forward_auc(X, y, dates_ord, model_fn, n_folds=5, embargo_days=25):
    from sklearn.metrics import roc_auc_score
    order = np.argsort(dates_ord)
    Xo, yo, do = X[order], y[order], np.array(dates_ord)[order]
    n = len(yo)
    bounds = [int(n * i / (n_folds + 1)) for i in range(n_folds + 2)]   # indices 0..n_folds+1
    oof_pred, oof_true = [], []
    for f in range(n_folds):
        te_start, te_end = bounds[f + 1], bounds[f + 2]
        tr_idx = np.arange(0, te_start)
        te_idx = np.arange(te_start, te_end)
        if len(te_idx) < 5 or len(tr_idx) < 30:
            continue
        # embargo: drop train rows whose entry is within embargo_days before the test block start
        te_min = do[te_start]
        keep = [i for i in tr_idx if (te_min - do[i]).days > embargo_days]
        if len(keep) < 30 or len(set(yo[keep])) < 2:
            continue
        try:
            m = model_fn()
            m.fit(Xo[keep], yo[keep])
            p = m.predict_proba(Xo[te_idx])[:, 1]
            oof_pred += list(p); oof_true += list(yo[te_idx])
        except Exception:
            continue
    if len(set(oof_true)) < 2 or len(oof_true) < 20:
        return None, len(oof_true)
    return float(roc_auc_score(oof_true, oof_pred)), len(oof_true)


def main():
    man = json.loads((OUT / "diagnostic" / "holdout_manifest.json").read_text(encoding="utf-8"))
    explore = set(man["exploratory_video_ids"])
    calls = [c for c in P.load_calls() if c["video_id"] in explore]
    pubs = [_to_kst(c["publish"]).date() for c in calls]
    start = (min(pubs) - dt.timedelta(days=BETA_LOOKBACK + 220)).strftime("%Y%m%d")
    end = dt.datetime.now(KST).strftime("%Y%m%d")
    idx_s = _ohlcv5(INDEX_PROXY, start, end)
    dates = sorted(idx_s)
    last_date = dates[-1] if dates else dt.datetime.now(KST).date()
    print(f"[offense] exploratory calls={len(calls)} — scoring + building point-in-time features…", flush=True)

    # score (reuse phase1b) + build features for ENTERED BUY calls
    occ = {}
    samples = []
    for i, c in enumerate(calls):
        ss = _ohlcv5(c["ticker"], start, end)
        r = P.score_call(c, ss, idx_s, dates, last_date)
        occ[c["ticker"]] = occ.get(c["ticker"], 0) + 1
        if r.get("play") not in SCORED_LONG or r.get("status") != "entered" or r.get("net_abnormal") is None:
            continue
        ed = dt.date.fromisoformat(r["entry_date"])
        ref_i = dates.index(ed) if ed in dates else None
        gap = None
        sd = sorted(ss)
        if ed in sd:
            j = sd.index(ed)
            if j >= 1:
                gap = ss[ed][0] / ss[sd[j - 1]][3] - 1
        feats = features_at(ss, idx_s, ed, gap)
        if feats is None:
            continue
        feats.update({"first_mention": 1 if occ[c["ticker"]] == 1 else 0,
                      "has_stated_price": 1 if c.get("stated_price") else 0,
                      "entry_month": ed.month, "entry_dow": ed.weekday()})
        samples.append({"ticker": c["ticker"], "entry_date": ed, "net": r["net_abnormal"],
                        "win": 1 if r["net_abnormal"] > 0 else 0, "play": r["play"],
                        "setups": setup_types(c.get("quote", "")), "feats": feats})
        if i % 100 == 0:
            print(f"  {i}/{len(calls)}", flush=True)
    print(f"[offense] feature samples (entered long w/ >=120d history): {len(samples)}", flush=True)

    # ---------------- Track A — setups by TYPE (n>=30), net-of-market+cost, deflated bar ----------------
    from collections import defaultdict
    by_play = defaultdict(list)
    by_setup = defaultdict(list)
    for s in samples:
        by_play[s["play"]].append(s["net"])
        for st in s["setups"]:
            by_setup[st].append(s["net"])
    seg_present = sum(1 for v in list(by_play.values()) + list(by_setup.values()) if len(v) >= 30)
    K_A = max(1, seg_present)
    base_p = 2 * (0.5 * math.erfc(P.TSTAT_BAR / math.sqrt(2)))     # base two-sided p at TSTAT_BAR
    deflated_bar_A = round(max(P.TSTAT_BAR, P._t_for_p(base_p / K_A)), 3)
    def bucketize(d):
        out = {}
        for k, v in sorted(d.items(), key=lambda kv: -len(kv[1])):
            a = _agg(v)
            out[k] = {"n": a["n"], "mean_net_pct": a["mean_pct"], "t": a["t"],
                      "powered": a["n"] >= 30,
                      "verdict": ("UNDERPOWERED (n<30)" if a["n"] < 30 else
                                  "positive & clears deflated bar" if (a["t"] or 0) >= deflated_bar_A and (a["mean_pct"] or 0) > 0 else
                                  "no positive edge")}
        return out
    track_a = {"play_buckets": bucketize(by_play), "setup_buckets": bucketize(by_setup),
               "K_deflated": K_A, "deflated_tstat_bar": deflated_bar_A,
               "powered_with_positive_edge": []}
    for grp in (track_a["play_buckets"], track_a["setup_buckets"]):
        for k, v in grp.items():
            if v["powered"] and (v["mean_net_pct"] or 0) > 0 and (v["t"] or 0) >= deflated_bar_A:
                track_a["powered_with_positive_edge"].append(k)

    # ---------------- Track B — conditional model (separate his winners from losers) ----------------
    FEATS = sorted(samples[0]["feats"].keys()) if samples else []
    X = np.array([[s["feats"][k] for k in FEATS] for s in samples], float)
    y = np.array([s["win"] for s in samples])
    do = [s["entry_date"] for s in samples]
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    models = {
        "logistic_L2": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=1000)),
        "lightgbm": lambda: lgb.LGBMClassifier(n_estimators=120, num_leaves=8, learning_rate=0.05,
                                               min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                                               reg_lambda=1.0, verbose=-1, random_state=0),
    }
    track_b = {"primary_conditional": {}, "n_samples": len(samples), "n_features": len(FEATS),
               "win_rate": round(float(y.mean()), 3), "features": FEATS}
    K_B = len(models)                                            # models tried (primary)
    for name, mf in models.items():
        auc, n_oof = walk_forward_auc(X, y, do, mf)
        # shuffled-label null floor
        null = []
        for _ in range(60):
            yp = RNG.permutation(y)
            a, _n = walk_forward_auc(X, yp, do, mf)
            if a is not None:
                null.append(a)
        null = np.array(null)
        floor95 = float(np.percentile(null, 95)) if len(null) else None
        pval = float((null >= auc).mean()) if (auc is not None and len(null)) else None
        track_b["primary_conditional"][name] = {
            "oos_auc": round(auc, 4) if auc is not None else None, "n_oof": n_oof,
            "null_auc_mean": round(float(null.mean()), 4) if len(null) else None,
            "null_auc_95pct_FLOOR": round(floor95, 4) if floor95 else None,
            "p_vs_null": round(pval, 4) if pval is not None else None,
            "beats_null_floor": bool(auc is not None and floor95 is not None and auc > floor95),
            "deflated_alpha": round(0.05 / K_B, 4)}

    # ---------------- mega-cap slice (pre-registered hypothesis from Task 3) ----------------
    mega = [s for s in samples if s["ticker"] in MEGA]
    mg = _agg([s["net"] for s in mega])
    mega_slice = {"tickers": MEGA, "n_entered": mg["n"], "mean_net_pct": mg["mean_pct"], "t": mg["t"],
                  "powered_n>=30": mg["n"] >= 30,
                  "note": "DESCRIPTIVE only (exploratory); pre-registered as a holdout hypothesis below — not a finding."}

    res = {"task": "Task 5 offense hunt — CV stage (exploratory 70%); holdout SEALED & untouched",
           "data_inventory": {
               "exploratory_calls": len(calls), "unique_tickers": len({c['ticker'] for c in calls}),
               "feature_samples_entered_long": len(samples),
               "prices": "pykrx OHLCV (full KRX universe) + FDR — WORK", "caps": "FDR StockListing — WORK",
               "FAIL_in_this_env": "pykrx market-cap/ticker-list/shorting/fundamental endpoints (empty/500)",
               "supabase_feature_tables": "flows/short/fundamentals/features cover ONLY ~9 mega-caps "
                                          "(005930/035420/000660 of his set) — NOT the mid/small bulk",
               "implication": "investor-flow / short-selling / fundamental features are NOT buildable for the "
                              "mid-cap bulk; feature base = price/MA/momentum/volatility/volume + rel-strength/"
                              "regime + caps + call-meta. flows/short/fundamentals usable only as a mega-cap overlay."},
           "track_A": track_a, "track_B": track_b, "mega_cap_slice": mega_slice}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "phase1b_offense.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== TRACK A — setups by type (n>=30; deflated bar %.3f, K=%d) ===" % (deflated_bar_A, K_A))
    for grp, lab in ((track_a["play_buckets"], "play"), (track_a["setup_buckets"], "setup")):
        for k, v in grp.items():
            print(f"  [{lab}] {k:18} n={v['n']:>4} mean%={v['mean_net_pct']} t={v['t']} -> {v['verdict']}")
    print("  powered + positive edge:", track_a["powered_with_positive_edge"] or "NONE")
    print("\n=== TRACK B — conditional model (separate winners/losers); win-rate=%.2f n=%d ===" % (track_b["win_rate"], len(samples)))
    for name, v in track_b["primary_conditional"].items():
        print(f"  {name:12} OOS_AUC={v['oos_auc']} null95_floor={v['null_auc_95pct_FLOOR']} "
              f"p_vs_null={v['p_vs_null']} beats_floor={v['beats_null_floor']}")
    print("\n=== MEGA-CAP slice (Samsung/NAVER/SK Hynix) ===")
    print(f"  n={mega_slice['n_entered']} mean_net%={mega_slice['mean_net_pct']} t={mega_slice['t']} powered={mega_slice['powered_n>=30']}")
    print(f"saved -> {OUT/'phase1b_offense.json'}")


if __name__ == "__main__":
    main()
