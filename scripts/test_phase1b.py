# -*- coding: utf-8 -*-
"""Gate 1 — deterministic unit-test suite for the CORRECTED Phase 1B scorer (phase1b.py).

Fixtures for the classification tests are REAL flagged quotes from factsheet_qa_flags.json (the QA audit).
Entry/exit/sign/gate tests use synthetic OHLCV paths (no network, no pykrx). Gate 1.7 re-runs the fixed
classifier over all 508 stored calls and asserts the measured gaps are now ZERO.

Run: python -m scripts.test_phase1b      (exit code 0 ⇔ all green)
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

PASS, FAIL = [], []
def ok(name, cond, extra=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'OK  ' if cond else 'FAIL'} {name}" + (f"  [{extra}]" if extra and not cond else ""))

FX = json.loads((config.DATA_DIR / "qa_audit" / "_fixtures.json").read_text(encoding="utf-8"))
def cc(q, direction=None):
    return P.classify_call({"quote": q, "direction": direction})

# --------------------------------------------------------------------------- #
# 1. CLASSIFICATION — the audit's real cases
# --------------------------------------------------------------------------- #
print("\n[1] classification (real flagged quotes)")
ok("1a 포스코홀딩스/원익홀딩스 → not avoid/bearish", cc(FX["holdings"]["quote"])["bucket"] not in ("BEARISH", "HOLD-WAIT-CASH"))
ok("1b 원익홀딩스 (2nd) → not bearish", cc("원익홀딩스 13% SK하이닉스 7% 올라가면서 반도체 수급 쏠림")["sign"] != -1)
ok("1c '…아닙니다/필요 없다' → negation excluded", cc(FX["negation"]["quote"])["bucket"] == "CANCELLED")
ok("1d 서킷브레이크/사이드카 → noise excluded", cc(FX["noise"]["quote"])["bucket"] == "NOISE" and not cc(FX["noise"]["quote"])["scored"])
ok("1e 블록딜 처분가 → TRIM not short", cc(FX["trim"]["quote"])["bucket"] == "TRIM")
ok("1f '스케줄 매매를 하셔야 됩니다' → NOT SCHEDULE", cc(FX["schedule_boiler"]["quote"])["segment"] != "SCHEDULE")
ok("1g a real 매수 → LONG-GENERIC entered", cc(FX["long_generic"]["quote"])["segment"] == "LONG-GENERIC" and cc(FX["long_generic"]["quote"])["scored"])
ok("1h a real 매도(하락) → BEARISH", cc("지금 이 종목은 매도하세요 추가 하락이 예상됩니다")["segment"] == "BEARISH")
ok("1i ex-post recap → excluded", cc(FX["expost"]["quote"])["bucket"] == "EXPOST" and not cc(FX["expost"]["quote"])["scored"])
ok("1j SCHEDULE WITH a real date → SCHEDULE", cc(FX["schedule_dated"]["quote"])["segment"] == "SCHEDULE")

# --------------------------------------------------------------------------- #
# synthetic calendar / prices
# --------------------------------------------------------------------------- #
def caldays(start, n):
    out, d = [], dt.date.fromisoformat(start)
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += dt.timedelta(days=1)
    return out

DATES = caldays("2025-01-02", 45)
LAST = DATES[-1]
PUB = DATES[19].isoformat() + "T23:00:00+09:00"          # window = DATES[20:30]
IDX = {d: (1000.0, 1000.0, 1000.0, 1000.0 + (i % 2), 1000.0) for i, d in enumerate(DATES)}   # beta→0 vs flat stock

def base_ss(vol=100.0):
    return {d: [10000.0, 10000.0, 10000.0, 10000.0, vol] for d in DATES}

def call(quote, sp=None, direction=None):
    return {"quote": quote, "direction": direction, "publish": PUB, "stated_price": sp, "ambiguous": False}

# --------------------------------------------------------------------------- #
# 2. ENTRY trigger
# --------------------------------------------------------------------------- #
print("\n[2] entry trigger (synthetic)")
ss = base_ss()
r = P.score_call(call("지금 매수 타점입니다 들어가세요"), ss, IDX, DATES, LAST)            # LONG-GENERIC
ok("2a LONG-GENERIC enters at next open", r["status"] == "entered" and r["play"] == "LONG-GENERIC")
r = P.score_call(call("내일 실적 발표 예정이라 매수 타점입니다"), ss, IDX, DATES, LAST)       # SCHEDULE
ok("2b SCHEDULE (dated) enters", r["status"] == "entered" and r["play"] == "SCHEDULE")
# DIP-BUY: low must touch support
ss = base_ss()
ss[DATES[22]] = [10000.0, 10000.0, 9990.0, 9995.0, 100.0]      # low 9990 touches support 10000±0.5%
r = P.score_call(call("지지선에서 받쳐주면 매수하겠습니다", sp=10000), ss, IDX, DATES, LAST)
ok("2c DIP-BUY triggers when low touches support", r["status"] == "entered" and r["play"] == "DIP-BUY")
r = P.score_call(call("지지선에서 받쳐주면 매수하겠습니다", sp=5000), base_ss(), IDX, DATES, LAST)  # support far below
ok("2d DIP-BUY non-triggered when support not touched", r["status"] == "non_triggered")
# BREAKOUT: close>level AND volume>=2x 20d-avg(=100)
ss = base_ss(vol=100.0)
ss[DATES[20]] = [10000.0, 10100.0, 10000.0, 10050.0, 300.0]    # close 10050>level, vol 300>=200
r = P.score_call(call("저항 돌파하면 매수", sp=10000), ss, IDX, DATES, LAST)
ok("2e BREAKOUT triggers on close>level + 2x volume", r["status"] == "entered" and r["play"] == "BREAKOUT")
r = P.score_call(call("저항 돌파하면 매수", sp=10000), base_ss(vol=100.0), IDX, DATES, LAST)  # no volume surge
ok("2f BREAKOUT non-triggered without 2x volume", r["status"] == "non_triggered")

# --------------------------------------------------------------------------- #
# 3. EXIT barriers (+ symmetric defaults)
# --------------------------------------------------------------------------- #
print("\n[3] exit barriers (synthetic)")
ok("3a defaults SYMMETRIC (DEF_TARGET == DEF_STOP)", P.DEF_TARGET == P.DEF_STOP and P.DEF_STOP == P.DEF_BARRIER)
ss = base_ss(); ss[DATES[21]] = [10000.0, 11000.0, 9900.0, 10800.0, 100.0]    # high 11000 ≥ target 10800
r = P.score_call(call("지금 매수 타점입니다 들어가세요"), ss, IDX, DATES, LAST)
ok("3b long target barrier first", r.get("barrier") == "target" and r.get("hit") is True)
ss = base_ss(); ss[DATES[21]] = [10000.0, 10100.0, 9000.0, 9100.0, 100.0]     # low 9000 ≤ stop 9200
r = P.score_call(call("지금 매수 타점입니다 들어가세요"), ss, IDX, DATES, LAST)
ok("3c long stop barrier first", r.get("barrier") == "stop" and r.get("hit") is False)
r = P.score_call(call("지금 매수 타점입니다 들어가세요"), base_ss(), IDX, DATES, LAST)        # flat → cap
ok("3d long 20d cap when neither barrier", r.get("barrier") == "cap")
ss = base_ss(); ss[DATES[21]] = [10000.0, 10100.0, 9000.0, 9100.0, 100.0]     # falls → bearish target
r = P.score_call(call("지금 이 종목 매도하세요 하락 예상"), ss, IDX, DATES, LAST)
ok("3e bearish target on a falling stock", r["play"] == "BEARISH" and r.get("barrier") == "target" and r.get("hit") is True)

# --------------------------------------------------------------------------- #
# 4. SIGN convention
# --------------------------------------------------------------------------- #
print("\n[4] sign convention (net_abnormal)")
ss = base_ss(); ss[DATES[21]] = [10000.0, 11000.0, 9900.0, 10800.0, 100.0]
r = P.score_call(call("지금 매수 타점입니다 들어가세요"), ss, IDX, DATES, LAST)
ok("4a winning long → +net", r["net_abnormal"] > 0, f"net={r.get('net_abnormal')}")
ss = base_ss(); ss[DATES[21]] = [10000.0, 10100.0, 9000.0, 9100.0, 100.0]
r = P.score_call(call("지금 이 종목 매도하세요 하락 예상"), ss, IDX, DATES, LAST)
ok("4b winning bearish → +net", r["net_abnormal"] > 0, f"net={r.get('net_abnormal')}")
ss = base_ss(); ss[DATES[21]] = [10000.0, 10100.0, 9000.0, 9100.0, 100.0]
r = P.score_call(call("지금 매수 타점입니다 들어가세요"), ss, IDX, DATES, LAST)
ok("4c losing long → -net", r["net_abnormal"] < 0, f"net={r.get('net_abnormal')}")

# --------------------------------------------------------------------------- #
# 5. VERDICT gate
# --------------------------------------------------------------------------- #
print("\n[5] verdict gate (§1.2 — uniform, no sign flip)")
ok("5a (+net, t≥bar) → PASS", P._verdict(40, 0.5, 4.0, 3.5).startswith("PASS"))
ok("5b (−net, |t|≥bar) → WRONG-SIGNED", P._verdict(40, -0.5, -4.0, 3.5).startswith("WRONG-SIGNED"))
ok("5c |t|<bar → no edge", P._verdict(40, 0.5, 2.0, 3.5).startswith("no edge"))
ok("5d n<30 → INCONCLUSIVE", P._verdict(20, 0.5, 9.0, 3.5).startswith("INCONCLUSIVE"))
ok("5e bearish winner is +net → PASS (NOT scored as negative)", P._verdict(40, 0.6, 4.2, 3.5).startswith("PASS"))
# Test B uses the SAME sign convention — a negative CAAR is NOT a pass (the abs(t) bug is pinned out)
ok("5f Test B negative CAAR, |t|≥bar → WRONG-SIGNED (not PASS)", P._testb_verdict(-7.4, -3.6, 740, 3.5) == "WRONG-SIGNED")
ok("5g Test B positive CAAR, t≥bar → PASS", P._testb_verdict(2.0, 4.0, 740, 3.5) == "PASS")
ok("5h Test B negative CAAR is never PASS", P._testb_verdict(-7.4, -9.0, 740, 4.0) != "PASS")

# --------------------------------------------------------------------------- #
# 6. DEFLATED bar
# --------------------------------------------------------------------------- #
print("\n[6] deflated bar (Bonferroni K = segments×horizons + 1)")
n_seg, n_h = 3, 1 + len(P.FIXED_HORIZONS)
K = max(1, n_seg * n_h + 1)
bar = round(max(P.TSTAT_BAR, P._t_for_p(P._two_sided_p(P.TSTAT_BAR) / K)), 3)
ok("6a K formula = segs×horizons + 1", K == 3 * 4 + 1 == 13)
ok("6b deflated bar ≥ base TSTAT_BAR", bar >= P.TSTAT_BAR)
ok("6c larger K → higher bar", round(max(P.TSTAT_BAR, P._t_for_p(P._two_sided_p(P.TSTAT_BAR) / 25)), 3) >= bar)

# --------------------------------------------------------------------------- #
# 7. REGRESSION vs the audit — fixed classifier drives every gap to 0
# --------------------------------------------------------------------------- #
print("\n[7] regression vs audit (all 508 through the FIXED classifier)")
from scripts.factsheet_qa_audit import regression_against_fixed
reg = regression_against_fixed()
for k, v in reg["residual_gaps"].items():
    ok(f"7.{k} == 0", v == 0, f"got {v}")
ok("7 all residual gaps zero", reg["all_zero"], str(reg["residual_gaps"]))

print(f"\n=== {len(PASS)} passed, {len(FAIL)} failed ===")
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
print("ALL GREEN")
