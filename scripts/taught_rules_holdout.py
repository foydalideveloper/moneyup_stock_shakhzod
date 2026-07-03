# -*- coding: utf-8 -*-
"""One-shot LOCKED holdout — R4 (resistance breakout on volume) @ 20d ONLY.

Verifies the pre-registration sha256, then scores R4@20d on the SEALED holdout (entry ≥ split_date) using
the SAME engine functions/constants as scripts.taught_rules_backtest (no re-implementation, no tuning).
Decision (locked): edge iff n>=30 AND mean_βadj_net>0 AND t>=4.166 AND beats buy-and-hold (raw) AND beats
random-entry null (p<0.0033). One-shot; no peek-then-adjust.
"""
from __future__ import annotations
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
import numpy as np
import pandas as pd
from moneyup_advisor import config
from moneyup_advisor.phase1b import _ohlcv5, _nw_tstat, INDEX_PROXY
from scripts.taught_rules_backtest import (_universe, _sma, _roll_beta, _triggers,
                                           START, END, LB, COST, WARM)

OUT = config.DATA_DIR / "qa_audit"
LOCK_SHA = "92a4ea4100a017576f4bbcffa108ad98d17010bfb3065969f600e249b98d240f"
BAR_T, ALPHA, NBOOT, HZ = 4.166, 0.0033, 3000, 20
RULE = "R4_resistance_breakout_vol(long)"


def main():
    pre = OUT / "taught_rules_prereg.md"
    sha = hashlib.sha256(pre.read_bytes()).hexdigest()
    print(f"[R4-holdout] pre-reg sha256={sha}\n            lock intact={sha == LOCK_SHA}")
    if sha != LOCK_SHA:
        print("ABORT — pre-registration changed since lock."); return

    uni = _universe()
    idx = _ohlcv5(INDEX_PROXY, START, END)
    idx_close = {d: idx[d][3] for d in idx}
    idx_dates = sorted(idx)
    split_date = idx_dates[int(len(idx_dates) * 0.70)]                 # same split as explore
    print(f"[R4-holdout] universe={len(uni)} split={split_date} (holdout = entry >= split)", flush=True)

    r4_net, r4_raw, pool_net, pool_raw = [], [], [], []
    for k, tk in enumerate(uni):
        ss = _ohlcv5(tk, START, END)
        d = sorted(ss)
        if len(d) < 320:
            continue
        o = np.array([ss[x][0] for x in d], float); h = np.array([ss[x][1] for x in d], float)
        l = np.array([ss[x][2] for x in d], float); c = np.array([ss[x][3] for x in d], float)
        v = np.array([ss[x][4] for x in d], float)
        s20, s50, s60, s120 = _sma(c, 20), _sma(c, 50), _sma(c, 60), _sma(c, 120)
        vma20 = _sma(v, 20); hi60 = pd.Series(h).rolling(60).max().shift(1).values
        ic = np.array([idx_close.get(x, np.nan) for x in d], float)
        rs = np.concatenate([[np.nan], c[1:] / c[:-1] - 1])
        ri = np.concatenate([[np.nan], ic[1:] / ic[:-1] - 1])
        beta = _roll_beta(rs, ri, LB)
        n = len(d)
        m = _triggers(RULE, c, o, h, l, v, s20, s50, s60, s120, vma20, hi60)
        for i in range(WARM, n - 1 - HZ):
            ei, xi = i + 1, i + 1 + HZ
            if np.isnan(o[ei]) or np.isnan(c[xi]) or np.isnan(ic[ei]) or np.isnan(ic[xi]) or np.isnan(beta[i]):
                continue
            if d[ei] < split_date:                                    # HOLDOUT only
                continue
            fs = c[xi] / o[ei] - 1
            fi = ic[xi] / ic[ei] - 1
            net = (fs - beta[i] * fi) - COST                          # β-adj net (long)
            raw = fs - COST
            pool_net.append(net); pool_raw.append(raw)                # holdout universe-day population
            if m[i]:
                r4_net.append(net); r4_raw.append(raw)
        if k % 50 == 0:
            print(f"  {k}/{len(uni)}", flush=True)

    r4_net = np.array(r4_net); pool_net = np.array(pool_net)
    n = len(r4_net)
    mean_net = float(r4_net.mean()); t = _nw_tstat(list(r4_net), 0)
    bnh = float(np.mean(pool_raw))                                    # buy-and-hold raw mean (long)
    beat_bnh = float(np.mean(r4_raw)) > bnh
    rng = np.random.RandomState(20260702)
    nulls = np.array([pool_net[rng.randint(0, len(pool_net), n)].mean() for _ in range(NBOOT)])
    p_rand = float((nulls >= mean_net).mean())
    edge = bool(n >= 30 and mean_net > 0 and (t or 0) >= BAR_T and beat_bnh and p_rand < ALPHA)
    res = {"test": "R4 resistance-breakout+vol @20d — ONE-SHOT holdout (locked)", "lock_ok": sha == LOCK_SHA,
           "n_holdout": n, "mean_betaadj_net_pct": round(mean_net * 100, 3), "t": t,
           "bar_t": BAR_T, "buy_hold_raw_pct": round(bnh * 100, 3), "rule_raw_pct": round(float(np.mean(r4_raw)) * 100, 3),
           "beats_buy_hold": beat_bnh, "rand_null_mean_pct": round(float(nulls.mean()) * 100, 3),
           "p_vs_random": round(p_rand, 4), "beats_random": p_rand < ALPHA, "alpha": ALPHA,
           "EDGE": edge,
           "decision": ("EDGE — all locked conditions met" if edge else
                        "NO EDGE — a locked condition not met"),
           "survivorship_caveat": ("Universe is CURRENT top-350-by-cap, not point-in-time. Even if this passes, "
                                   "a survivorship-clean confirmation with point-in-time KOSPI200/KOSDAQ150 "
                                   "membership is REQUIRED before calling it a real edge.")}
    (OUT / "taught_rules_h_holdout.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== R4@20d ONE-SHOT HOLDOUT (entry >= {split_date}) ===")
    print(f"  n={n}  mean_βadj_net={res['mean_betaadj_net_pct']}%  t={t}  (bar {BAR_T})")
    print(f"  buy-and-hold raw={res['buy_hold_raw_pct']}%  rule raw={res['rule_raw_pct']}%  beats_B&H={beat_bnh}")
    print(f"  random-null mean={res['rand_null_mean_pct']}%  p={p_rand}  beats_random={p_rand < ALPHA}")
    print(f"\n  DECISION: {res['decision']}  (EDGE={edge})")
    if edge:
        print("  ** Survivorship-clean confirmation (point-in-time index membership) STILL REQUIRED before it's an edge. **")
    print(f"saved -> {OUT/'taught_rules_h_holdout.json'}")


if __name__ == "__main__":
    main()
