# -*- coding: utf-8 -*-
"""머니업 TAUGHT-RULES backtest — his concrete PRICE rules as GENERAL market strategies (not his picks).

STEP 1: 5 mechanizable price rules (below). Flow/schedule/discretionary rules EXCLUDED (no broad flow data;
Track A already tested them on his picks → negative).
STEP 2: backtest each on the KOSPI200+KOSDAQ150 proxy universe (top-200 KOSPI + top-150 KOSDAQ by FDR cap;
current membership — caveat), real pykrx OHLCV, enter next open, β-adjusted, net 0.50% round-trip.
STEP 3: temporal split (explore ~70% / sealed ~30% holdout — holdout NOT touched here); deflated bar over
K=#rules×#horizons; each rule must beat BOTH buy-and-hold AND a random-entry null; n>=30; point-in-time.
STEP 4: report every rule; propose a locked pre-registration for survivors. NO holdout run, NO edge claim.
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
import pandas as pd
from moneyup_advisor import config, phase1b as P
from moneyup_advisor.phase1b import _ohlcv5, _nw_tstat, _t_for_p, INDEX_PROXY, BETA_LOOKBACK, COST_ROUNDTRIP

OUT = config.DATA_DIR / "qa_audit"
START, END = "20230101", "20260702"
H = [5, 10, 20]
LB = BETA_LOOKBACK
COST = COST_ROUNDTRIP
WARM = 250                                   # need SMA120 + beta lookback before any trigger
NBOOT = 3000
N_KOSPI, N_KOSDAQ = 200, 150

# STEP 1 — mechanized rules {name: (sign, direction, definition)}
RULES = ["R1_50MA_break_fail(short)", "R2_reclaim_50MA(long)", "R3_jeongbaeyeol(long)",
         "R4_resistance_breakout_vol(long)", "R5_pullback_SMA20(long)"]
RULE_SIGN = {"R1_50MA_break_fail(short)": -1, "R2_reclaim_50MA(long)": 1, "R3_jeongbaeyeol(long)": 1,
             "R4_resistance_breakout_vol(long)": 1, "R5_pullback_SMA20(long)": 1}


def _universe():
    import FinanceDataReader as fdr
    df = fdr.StockListing("KRX").dropna(subset=["Marcap"])
    uni = []
    for mk, k in (("KOSPI", N_KOSPI), ("KOSDAQ", N_KOSDAQ)):
        sub = df[df["Market"] == mk].sort_values("Marcap", ascending=False).head(k)
        uni += [str(x) for x in sub["Code"].tolist()]
    return uni


def _sma(a, w):
    return pd.Series(a).rolling(w).mean().values


def _roll_beta(rs, ri, lb):
    s, i = pd.Series(rs), pd.Series(ri)
    return ((s.rolling(lb).cov(i)) / (i.rolling(lb).var())).shift(1).values   # data through t-1


def _triggers(name, c, o, h, l, v, s20, s50, s60, s120, vma20, hi60):
    """boolean mask of signal days t (entry is next open t+1)."""
    n = len(c)
    m = np.zeros(n, bool)
    for t in range(WARM, n - 1):
        if any(np.isnan(x) for x in (s120[t], s50[t], vma20[t])):
            continue
        if name == "R1_50MA_break_fail(short)":
            m[t] = (c[t] < s50[t]) and (h[t] >= s50[t]) and (t >= 5 and s50[t] < s50[t - 5])
        elif name == "R2_reclaim_50MA(long)":
            m[t] = (c[t] >= s50[t]) and (c[t - 1] < s50[t - 1])
        elif name == "R3_jeongbaeyeol(long)":
            al = lambda i: (c[i] > s20[i] > s60[i] > s120[i])
            m[t] = al(t) and not al(t - 1)
        elif name == "R4_resistance_breakout_vol(long)":
            m[t] = (not np.isnan(hi60[t])) and (c[t] > hi60[t]) and (v[t] >= 1.5 * vma20[t])
        elif name == "R5_pullback_SMA20(long)":
            m[t] = (t >= 5 and s20[t] > s20[t - 5]) and (l[t] <= s20[t] * 1.02) and \
                   (c[t] > s20[t]) and (c[t - 1] > s20[t - 1])
    return m


def main():
    uni = _universe()
    idx = _ohlcv5(INDEX_PROXY, START, END)
    idx_close = {d: idx[d][3] for d in idx}
    idx_dates = sorted(idx)
    split_date = idx_dates[int(len(idx_dates) * 0.70)]
    print(f"[taught] universe={len(uni)} (top {N_KOSPI} KOSPI + {N_KOSDAQ} KOSDAQ by cap) "
          f"split={split_date} (explore< / holdout>=)", flush=True)

    pooled = {h: [] for h in H}          # explore universe-day LONG β-adj net (random-null + baseline)
    raw_pooled = {h: [] for h in H}      # explore universe-day RAW net (buy-and-hold baseline)
    trig = {r: {h: {"explore": [], "holdout_n": 0} for h in H} for r in RULES}

    for n_done, tk in enumerate(uni):
        ss = _ohlcv5(tk, START, END)
        d = sorted(ss)
        if len(d) < 320:
            continue
        o = np.array([ss[x][0] for x in d], float); h = np.array([ss[x][1] for x in d], float)
        l = np.array([ss[x][2] for x in d], float); c = np.array([ss[x][3] for x in d], float)
        v = np.array([ss[x][4] for x in d], float)
        s20, s50, s60, s120 = _sma(c, 20), _sma(c, 50), _sma(c, 60), _sma(c, 120)
        vma20 = _sma(v, 20)
        hi60 = pd.Series(h).rolling(60).max().shift(1).values
        ic = np.array([idx_close.get(x, np.nan) for x in d], float)
        rs = np.concatenate([[np.nan], c[1:] / c[:-1] - 1])
        ri = np.concatenate([[np.nan], ic[1:] / ic[:-1] - 1])
        beta = _roll_beta(rs, ri, LB)
        n = len(d)
        # forward β-adj net per horizon (enter next open i+1, exit close i+1+h)
        netL = {}; rawL = {}
        for hz in H:
            fs = np.full(n, np.nan); fi = np.full(n, np.nan)
            for i in range(WARM, n - 1 - hz):
                ei, xi = i + 1, i + 1 + hz
                if np.isnan(o[ei]) or np.isnan(c[xi]) or np.isnan(ic[ei]) or np.isnan(ic[xi]) or np.isnan(beta[i]):
                    continue
                fs[i] = c[xi] / o[ei] - 1
                fi[i] = ic[xi] / ic[ei] - 1
            abn = fs - beta * fi
            netL[hz] = abn - COST
            rawL[hz] = fs - COST
            # pooled explore universe-day population
            for i in range(WARM, n - 1 - hz):
                if not np.isnan(netL[hz][i]) and d[i + 1] < split_date:
                    pooled[hz].append(netL[hz][i]); raw_pooled[hz].append(rawL[hz][i])
        # rule triggers
        for r in RULES:
            m = _triggers(r, c, o, h, l, v, s20, s50, s60, s120, vma20, hi60)
            sign = RULE_SIGN[r]
            for hz in H:
                for i in np.where(m)[0]:
                    if i >= n - 1 - hz or np.isnan(netL[hz][i]):
                        continue
                    nl = netL[hz][i]
                    net = nl if sign > 0 else (-(nl + COST) - COST)     # short = -abn - cost
                    raw = rawL[hz][i] if sign > 0 else (-(rawL[hz][i] + COST) - COST)
                    if d[i + 1] < split_date:
                        trig[r][hz]["explore"].append((net, raw))
                    else:
                        trig[r][hz]["holdout_n"] += 1
        if n_done % 50 == 0:
            print(f"  {n_done}/{len(uni)} tickers", flush=True)

    # ---- stats ----
    K = len(RULES) * len(H)
    base_p = 2 * (0.5 * math.erfc(3.5 / math.sqrt(2)))
    deflated_bar = round(_t_for_p(base_p / K), 3)
    alpha_bonf = 0.05 / K
    rng = np.random.RandomState(20260702)
    pooled = {h: np.array(pooled[h]) for h in H}
    raw_pooled = {h: np.array(raw_pooled[h]) for h in H}

    results = {}
    for r in RULES:
        sign = RULE_SIGN[r]
        results[r] = {}
        for hz in H:
            rows = trig[r][hz]["explore"]
            nets = np.array([x[0] for x in rows]) if rows else np.array([])
            raws = np.array([x[1] for x in rows]) if rows else np.array([])
            n = len(nets)
            if n < 30:
                results[r][str(hz)] = {"n": n, "holdout_n": trig[r][hz]["holdout_n"],
                                       "verdict": "UNDERPOWERED (n<30)"}
                continue
            mean_net = float(nets.mean()); t = _nw_tstat(list(nets), 0)
            # buy-and-hold baseline (direction-matched): long -> universe raw mean ; short -> -universe raw
            bnh_long = float(raw_pooled[hz].mean())
            bnh = bnh_long if sign > 0 else (-(bnh_long + COST) - COST)
            beat_bnh = float(raws.mean()) > bnh
            # random-entry null (direction-matched), same trade count
            base = pooled[hz] if sign > 0 else (-(pooled[hz] + COST) - COST)
            nulls = np.array([base[rng.randint(0, len(base), n)].mean() for _ in range(NBOOT)])
            p_rand = float((nulls >= mean_net).mean())
            passed = bool(mean_net > 0 and (t or 0) >= deflated_bar and p_rand < alpha_bonf and beat_bnh)
            results[r][str(hz)] = {
                "n": n, "holdout_n": trig[r][hz]["holdout_n"],
                "mean_net_pct": round(mean_net * 100, 3), "t": t,
                "raw_mean_pct": round(float(raws.mean()) * 100, 3), "bnh_baseline_pct": round(bnh * 100, 3),
                "beats_buy_hold": beat_bnh, "rand_null_mean_pct": round(float(nulls.mean()) * 100, 3),
                "p_vs_random": round(p_rand, 4), "beats_random_deflated": p_rand < alpha_bonf,
                "PASS_explore": passed,
                "verdict": ("PASS explore (beats B&H + random @ deflated bar)" if passed else
                            "no edge" + ("" if mean_net > 0 else " (negative)"))}

    survivors = [f"{r}@{hz}d" for r in RULES for hz in H
                 if isinstance(results[r].get(str(hz)), dict) and results[r][str(hz)].get("PASS_explore")]
    out = {"test": "taught-rules backtest — general strategies (STEP 1-3)",
           "scope": "explore ~70% (holdout sealed, NOT analyzed)",
           "universe": f"top {N_KOSPI} KOSPI + {N_KOSDAQ} KOSDAQ by FDR cap (current membership — caveat)",
           "n_universe": len(uni), "split_date": str(split_date), "horizons": H,
           "K_deflated": K, "deflated_tstat_bar": deflated_bar, "alpha_bonferroni": round(alpha_bonf, 5),
           "cost_roundtrip_pct": COST * 100, "beta_lookback": LB, "nboot": NBOOT,
           "excluded": "flow-based (수급/외국인/공매도/대차/program), schedule/일정매매, 순환매, CB/블록딜, 타점/감(vague)",
           "rules": results, "survivors_explore": survivors}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "taught_rules_backtest.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== TAUGHT-RULES explore results (deflated bar t={deflated_bar}, K={K}, α_bonf={alpha_bonf:.4f}) ===")
    for r in RULES:
        print(f"\n[{r}]")
        for hz in H:
            x = results[r][str(hz)]
            if "mean_net_pct" not in x:
                print(f"   h{hz:>2}d  n={x['n']}  {x['verdict']}")
            else:
                print(f"   h{hz:>2}d  n={x['n']:>5} mean_net={x['mean_net_pct']}% t={x['t']} "
                      f"| B&H {x['bnh_baseline_pct']}% beat={x['beats_buy_hold']} "
                      f"| rand {x['rand_null_mean_pct']}% p={x['p_vs_random']} -> {x['verdict']}")
    print(f"\nSURVIVORS (explore): {survivors or 'NONE'}")
    print(f"saved -> {OUT/'taught_rules_backtest.json'}")


if __name__ == "__main__":
    main()
