"""Phase 1B — FAITHFUL re-test of 머니업's calls (play-type aware), additive to Test A.

LOCKED spec (pre-registered; do NOT re-tune on results). Reuses the Test A beta-adjusted, cost-aware
engine (β vs KODEX 200 / 069500, 252-day; 0.50% round-trip). Research/mock only — not a trading signal.

CORRECTED per the QA audit (§1.1-1.4; the fixes are pre-registered, not tuned on results):
  classify_call -> bucket. SCORED: SCHEDULE / ROTATION / DIP-BUY / BREAKOUT / LONG-GENERIC (sign +1) +
    BEARISH (sign -1). EXCLUDED-but-counted: TRIM / HOLD-WAIT-CASH / NOISE / EXPOST / CANCELLED /
    AMBIGUOUS / UNCLASSIFIED. (§1.1 AVOID split into BEARISH/HOLD/TRIM; §1.3 plain long -> LONG-GENERIC,
    never dropped; SCHEDULE is DATE-GATED; Tier-1 noise/expost/negation/holdings filtered.)
  ENTRY in a 10-trading-day window after publish:
    DIP-BUY  : low trades to stated support ±0.5% ; BREAKOUT : CLOSE > level on vol >= 2x 20d-avg ;
    SCHEDULE/ROTATION/LONG-GENERIC: next session open ; BEARISH: short at next session open.
    never triggers -> NON-TRIGGERED (counted, excluded; rate reported)
  EXIT: SCHEDULE -> hold to the parsed event date (D-N / relative); else triple-barrier {target/stop/20d}.
    His spoken target(목표가)/stop(손절) used when given; otherwise SYMMETRIC default ±8% (upper = |lower|,
    NEVER the biased -5/+8); inferred-level rows flagged + reported separately.
  OUTCOME: β- and size-adjusted return entry->exit, net 0.50% round-trip; net_abnormal = sign*abn - cost.
§1.2 UNIFORM verdict gate: a call RIGHT in its stated direction is POSITIVE for EVERY segment (long AND
  bearish) — PASS iff mean net > 0 AND t >= +deflated_bar; t <= -deflated_bar => significantly WRONG-SIGNED;
  no per-segment sign flip. DEFLATED bar K = segments x horizons (Bonferroni; PRINT K + threshold). Floor
  n>=30/segment else INCONCLUSIVE. Test B event study (CAAR t0..t+5, event-clustered SE, >=30 events).
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np

from moneyup_advisor import config, tickers
from moneyup_advisor import calls as _C            # READ-ONLY reuse of the keyword lists + parsers
from moneyup_advisor.phase1_callscore import (
    BETA_LOOKBACK, COST_ROUNDTRIP, INDEX_PROXY, KST, MARKET_OPEN, TSTAT_BAR,
    _beta, _nw_tstat, _to_kst,
)

ENTRY_WINDOW = 10          # trading-day window for the entry trigger
CAP_DAYS = 20             # triple-barrier time cap
DIP_TOL = 0.005           # support touch tolerance ±0.5%
DEF_BARRIER = 0.08        # §1.4 SYMMETRIC default barrier ±8% (upper = |lower|) — NEVER the biased -5/+8
DEF_TARGET = DEF_STOP = DEF_BARRIER          # symmetric aliases; pinned by test 3 (DEF_TARGET == DEF_STOP)
# Scored segments: long-side plays (sign +1) + BEARISH down-calls (sign -1). AVOID is gone — §1.1 splits
# it into BEARISH (scored) / HOLD-WAIT-CASH (excluded) / TRIM (excluded).
SEGMENTS = ("SCHEDULE", "ROTATION", "DIP-BUY", "BREAKOUT", "LONG-GENERIC", "BEARISH")
SCORED_LONG = ("SCHEDULE", "ROTATION", "DIP-BUY", "BREAKOUT", "LONG-GENERIC")
EXCLUDED_BUCKETS = ("TRIM", "HOLD-WAIT-CASH", "NOISE", "EXPOST", "CANCELLED", "AMBIGUOUS", "UNCLASSIFIED")
FIXED_HORIZONS = (1, 5, 20)

# long-side play keywords (SCHEDULE is now DATE-GATED — handled in _long_play, not here; 익절/비중축소/
# 목표가 removed from ROTATION → those are TRIM)
_KW = {
    "ROTATION": ["순환매", "교체", "갈아", "목표 달성", "돌리"],
    "DIP-BUY":  ["매집선", "지지", "하단", "눌림", "저가 매수", "저점", "조정 시", "빠지면", "받치", "반등"],
    "BREAKOUT": ["돌파", "저항 돌파", "박스권 상단", "거래량 터", "거래량 폭발", "신고가", "뚫"],
}

# ---- corrected-classification dictionaries (Tier-1 §1.1/§1.3 filters; replay on stored calls+quotes) ----
_MARKET_NOISE = ["사이드카", "서킷브레이커", "서킷 브레이크", "써킷브레이커", "반대매매", "동시호가",
                 "변동성완화장치", "vi 발동", "vi발동", "단일가"]
_NEG = re.compile(r"(필요\s*없|필요없|사지\s*마|팔지\s*마|아니다|아닙니다|않습니다|않는다|않다|없다|없습니다|마세요|불필요)")
_TRIM = ["익절", "차익실현", "차익 실현", "비중축소", "비중 축소", "일부 매도", "일부매도", "처분", "정리", "블록딜", "청산", "손절"]
# BEARISH = an actual down-CALL (recommendation/imperative), NOT a descriptive mention of 공매도/하락/숏
# (those saturate market commentary). §1.1: AVOID -> BEARISH means "he tells you to avoid/short it".
_BEARISH = ["매도하세", "매도하라", "매도해야", "매도 추천", "팔아라", "팔아야", "파세요", "파시는",
            "숏 치", "숏을 치", "숏을 잡으세", "숏 잡으세", "공매도 치", "공매도하세", "공매도 추천",
            "회피하세", "회피 추천", "비추천", "비추합", "사지 마", "사지말", "들어가지 마", "들어가면 안",
            "조심하세", "정리하세요", "손절하세요", "하방 베팅", "하락 베팅"]
_HOLD = ["관망", "보유", "지켜", "대기", "쉬어", "현금", "보수적", "기다리", "홀딩"]   # holdings-token-safe: 홀딩스 neutralised first
_REL_FUTURE = ["내일", "모레", "글피", "다음 주", "다음주", "담주", "한 달 후", "한달 후", "며칠 후", "조만간", "곧"]
_REL_OFFSET = {"내일": 1, "모레": 2, "글피": 3, "다음 주": 5, "다음주": 5, "담주": 5, "한 달 후": 20, "한달 후": 20}
_CATALYST = ["예정", "발표", "만기", "상장", "승인", "청구", "공모", "임상", "실적", "fomc", "이벤트",
             "컨퍼런스", "회의", "결정", "출시", "디데이"]
_DATE_ABS = re.compile(r"(d-\s*\d+|\d{1,2}\s*월\s*\d{1,2}\s*일)")
_STOP_RE = re.compile(r"(?:손절|스탑|스톱)\D{0,4}([0-9][0-9,]{2,})")


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def _tier3_set() -> set:
    """(video_id, call_index) of Tier-3 ambiguous calls (direction_mismatch + ticker_resolution) per the
    QA audit — excluded from scoring, surfaced for the §4 sample-audit (§E)."""
    out = set()
    try:
        fl = json.loads((config.DATA_DIR / "qa_audit" / "factsheet_qa_flags.json").read_text(encoding="utf-8"))
        for f in fl.get("track_a_flags", []):
            if f.get("category") in ("direction_mismatch", "ticker_resolution"):
                out.add((f.get("sheet"), f.get("call_index")))
    except Exception:
        pass
    return out


def load_calls() -> List[dict]:
    tier3 = _tier3_set()
    calls = []
    for p in sorted(config.SHEET_DIR.glob("*.json")):
        try:
            s = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        vid = s.get("video_id")
        prim = str(s.get("primary_ticker") or "").zfill(6) if s.get("primary_ticker") else None
        for ci, c in enumerate(s.get("exante_calls", [])):
            pub = c.get("publish_datetime") or ((c.get("publish_date") + "T23:59:59Z")
                                                if c.get("publish_date") else None)
            if not c.get("ticker") or not pub:
                continue
            calls.append({"video_id": vid, "call_index": ci, "ticker": str(c["ticker"]).zfill(6),
                          "name": c.get("name"), "direction": c.get("direction"), "publish": pub,
                          "primary": prim, "stated_price": c.get("stated_price"), "mmss": c.get("mmss"),
                          "quote": c.get("quote") or "", "degraded": bool(s.get("degraded")),
                          "ambiguous": (vid, ci) in tier3})
    return calls


@lru_cache(maxsize=512)
def _ohlcv5(ticker: str, start: str, end: str):
    """{date: (open, high, low, close, volume)} via pykrx. {} on failure."""
    from pykrx import stock
    try:
        df = stock.get_market_ohlcv_by_date(start, end, ticker)
    except Exception:
        return {}
    out = {}
    for i in range(len(df) if df is not None else 0):
        d = df.index[i].date()
        o, h, l, c, v = (float(df.iloc[i, 0]), float(df.iloc[i, 1]), float(df.iloc[i, 2]),
                         float(df.iloc[i, 3]), float(df.iloc[i, 4]))
        if o > 0 and c > 0:
            out[d] = (o, h, l, c, v)
    return out


def _strip_holdings(t: str) -> str:
    return t.replace("홀딩스", " ")                          # neutralise the company suffix (포스코홀딩스/원익홀딩스)


def _has(text: str, kws) -> bool:
    return any(k in text for k in kws)


def is_market_noise(q: str) -> bool:
    return _has(q, _MARKET_NOISE)


def is_expost(q: str) -> bool:
    return _has(q, _C._EXPOST)


def is_negated(q: str) -> bool:
    """An action keyword cancelled by an adjacent negation (…필요 없다/아니다/않)."""
    t = _C._strip_noise(_strip_holdings(q))
    for kw in _C._LONG + ["매도", "팔아", "파세요", "숏", "공매도"]:
        i = t.find(kw)
        while i >= 0:
            if _NEG.search(t[i:i + len(kw) + 9]):
                return True
            i = t.find(kw, i + 1)
    return False


def has_dated_event(q: str) -> bool:
    """§1.4 concrete dated event: a relative-future expression, a D-N countdown, or an absolute date
    coupled with a catalyst word. Bare '27일부터 매수' (past day numbers) is NOT a dated event."""
    t = q.lower()
    if _has(q, _REL_FUTURE):
        return True
    if re.search(r"d-\s*\d+", t):
        return True
    if _DATE_ABS.search(t) and _has(t, _CATALYST):
        return True
    return False


def corrected_dir(q: str):
    """Quote-authoritative direction → long / bearish / trim / hold / None. Holdings-token-safe; uses the
    noise filter so 매수가/매도가/매도세 don't fire; 공매도 checked pre-noise-strip so it survives."""
    base = _strip_holdings(q)
    clean = _C._strip_noise(base)
    has_long = _has(clean, _C._LONG)
    has_trim = _has(clean, _TRIM)
    has_hold = _has(clean, _HOLD)
    has_bear = _has(base, _BEARISH)                          # pre-noise-strip keeps 공매도
    # An explicit BUY keyword makes it long-side even if the quote also mentions 공매도/하락 descriptively
    # (a down-call almost never contains a buy verb). Only a quote with NO buy keyword can be BEARISH.
    if has_long:
        return "long"
    if has_bear:
        return "bearish"
    if has_trim:
        return "trim"
    if has_hold:
        return "hold"
    return None


def _long_play(q: str) -> str:
    """A long → SCHEDULE (only if a real dated event), ROTATION/DIP-BUY/BREAKOUT (keyword), else
    LONG-GENERIC (§1.3 — never dropped)."""
    if has_dated_event(q):
        return "SCHEDULE"
    t = q.lower()
    for seg in ("ROTATION", "DIP-BUY", "BREAKOUT"):
        if _has(t, _KW[seg]):
            return seg
    return "LONG-GENERIC"


def _disp(bucket, segment=None, sign=0, scored=False, reason=""):
    return {"bucket": bucket, "segment": segment, "sign": sign, "scored": scored, "reason": reason}


def classify_call(call: dict) -> dict:
    """The corrected, quote-authoritative classifier (single source of truth). Returns a disposition:
       {bucket, segment, sign, scored, reason}. Tier-1 exclusions first, then §1.1 split, then §1.3 long
       plays. Tier-3 ambiguous (direction/ticker) is handled at scoring LOAD (excluded), not here."""
    q = str(call.get("quote", "") or "")
    if is_market_noise(q):
        return _disp("NOISE", reason="market-structure commentary (사이드카/반대매매/서킷브레이커/동시호가)")
    if is_expost(q):
        return _disp("EXPOST", reason="ex-post recap markers (" + ",".join(k for k in _C._EXPOST if k in q)[:40] + ")")
    if is_negated(q):
        return _disp("CANCELLED", reason="action negated (…필요 없다/아니다/않)")
    cdir = corrected_dir(q)
    if cdir == "trim":
        return _disp("TRIM", reason="profit-take/reduce of a long (익절/처분/비중축소) — excluded, counted")
    if cdir == "hold":
        return _disp("HOLD-WAIT-CASH", reason="wait/cash, no directional bet — excluded, counted")
    if cdir == "bearish":
        return _disp("BEARISH", segment="BEARISH", sign=-1, scored=True, reason="down-call (scored, sign -1)")
    if cdir == "long":
        seg = _long_play(q)
        return _disp(seg, segment=seg, sign=1, scored=True, reason="long → " + seg)
    return _disp("UNCLASSIFIED", reason="no actionable direction")


def classify(call: dict) -> str:
    """Back-compat: the segment name for scored calls, else the excluded-bucket name (so the QA audit's
    `play = phase1b.classify(...)` now sees LONG-GENERIC / date-gated SCHEDULE — the gaps close)."""
    d = classify_call(call)
    return d["segment"] or d["bucket"]


def _avg20vol(ss: Dict, d, dates) -> float:
    prior = [x for x in dates if x < d][-20:]
    vols = [ss[x][4] for x in prior if x in ss and ss[x][4] > 0]
    return float(np.mean(vols)) if vols else 0.0


def _event_offset(quote: str) -> Optional[int]:
    """§1.4 trading-day offset to the spoken event: D-N, or a relative-future expression."""
    m = re.search(r"d-\s*(\d+)", quote.lower())
    if m:
        return int(m.group(1))
    for w, n in _REL_OFFSET.items():
        if w in quote:
            return n
    return None


def _spoken_levels(quote: str, entry_px: float, sign: int):
    """His explicit spoken target/stop as ABS-move fractions, if given (§1.4). Returns (tgt_frac, stop_frac,
    inferred). Target = 목표가; stop = 손절. Falls back to the SYMMETRIC default ±DEF_BARRIER."""
    tgt = stop = None
    mt = _C._TARGET.search(quote)
    if mt:
        tp = int(mt.group(1).replace(",", ""))
        if tp > 0:
            tgt = abs(tp / entry_px - 1.0)
    ms = _STOP_RE.search(quote)
    if ms:
        spv = int(ms.group(1).replace(",", ""))
        if spv > 0:
            stop = abs(spv / entry_px - 1.0)
    inferred = (tgt is None and stop is None)
    return (tgt or DEF_BARRIER), (stop or DEF_BARRIER), inferred


# --------------------------------------------------------------------------- #
# entry / exit / score one call
# --------------------------------------------------------------------------- #
def score_call(call: dict, ss: Dict, idx_s: Dict, dates: List, last_date) -> dict:
    if call.get("ambiguous"):                                # §E Tier-3: excluded, surfaced for review
        return {"play": "AMBIGUOUS", "status": "excluded", "inferred_levels": False}
    disp = classify_call(call)
    play = disp["segment"] or disp["bucket"]
    out = {"play": play, "bucket": disp["bucket"], "status": None, "inferred_levels": False}
    if not disp["scored"]:                                   # NOISE/EXPOST/CANCELLED/TRIM/HOLD/UNCLASSIFIED
        out["status"] = "excluded"
        return out
    pub = _to_kst(call["publish"])
    after = [d for d in dates if datetime.combine(d, MARKET_OPEN, KST) > pub]
    window = after[:ENTRY_WINDOW]
    if not window:
        out["status"] = "pending"
        return out
    sp = call.get("stated_price")
    sign = disp["sign"]                                      # +1 long-side, -1 BEARISH
    entry_date = entry_px = None
    if play in ("SCHEDULE", "ROTATION", "LONG-GENERIC", "BEARISH"):
        d = window[0]                                        # next session open (long or short-at-open)
        if d in ss:
            entry_date, entry_px = d, ss[d][0]
    elif play == "DIP-BUY":
        if sp:
            for d in window:
                if d in ss and ss[d][2] <= sp * (1 + DIP_TOL):       # low touches support ±0.5%
                    entry_date, entry_px = d, float(sp)
                    break
    elif play == "BREAKOUT":
        if sp:
            for d in window:
                if d in ss and ss[d][3] > sp and ss[d][4] >= 2 * _avg20vol(ss, d, dates):
                    entry_date, entry_px = d, ss[d][3]               # close above level on volume
                    break
    if entry_date is None:
        out["status"] = "non_triggered"
        return out
    if entry_date not in idx_s:
        out["status"] = "no_price"
        return out
    i0 = dates.index(entry_date)
    beta = _beta({d: (0, ss[d][3]) for d in ss}, {d: (0, idx_s[d][3]) for d in idx_s}, entry_date)
    out.update({"status": "entered", "entry_date": entry_date.isoformat(), "entry_px": round(entry_px, 1),
                "beta": round(beta, 3)})

    # ---- exit ----
    inferred = True
    barrier = "cap"
    exit_date = exit_px = None
    if play == "SCHEDULE":
        n = _event_offset(call.get("quote", ""))
        if n:
            j = min(i0 + n, len(dates) - 1)
            exit_date = dates[j]
            exit_px = ss[exit_date][3] if exit_date in ss else entry_px
            barrier, inferred = "event", False
    if exit_date is None:                                            # triple-barrier (SYMMETRIC default ±)
        tgt_f, stp_f, inferred = _spoken_levels(call.get("quote", ""), entry_px, sign)
        tgt = entry_px * (1 + tgt_f) if sign > 0 else entry_px * (1 - tgt_f)   # target = move IN his favour
        stp = entry_px * (1 - stp_f) if sign > 0 else entry_px * (1 + stp_f)   # stop   = move AGAINST
        hit_first = None
        for k in range(1, CAP_DAYS + 1):
            j = i0 + k
            if j >= len(dates):
                break
            d = dates[j]
            if d not in ss or d > last_date:
                break
            hi, lo = ss[d][1], ss[d][2]
            if sign > 0:
                if hi >= tgt:
                    exit_date, exit_px, barrier, hit_first = d, tgt, "target", True
                    break
                if lo <= stp:
                    exit_date, exit_px, barrier, hit_first = d, stp, "stop", False
                    break
            else:
                if lo <= tgt:
                    exit_date, exit_px, barrier, hit_first = d, tgt, "target", True
                    break
                if hi >= stp:
                    exit_date, exit_px, barrier, hit_first = d, stp, "stop", False
                    break
        if exit_date is None:                                       # time cap
            j = min(i0 + CAP_DAYS, len(dates) - 1)
            exit_date = dates[j]
            if exit_date not in ss or exit_date > last_date:
                out["status"] = "pending"
                return out
            exit_px, hit_first = ss[exit_date][3], False
        out["hit"] = bool(hit_first)
    else:
        out["hit"] = (exit_px / entry_px - 1) * sign > 0
    out["inferred_levels"] = inferred
    out["barrier"] = barrier
    out["exit_date"] = exit_date.isoformat()

    g_stock = exit_px / entry_px - 1.0
    g_idx = idx_s[exit_date][3] / idx_s[entry_date][3] - 1.0
    control = beta * g_idx                                          # size/market-matched control
    abn = g_stock - control
    out["net_abnormal"] = round(sign * abn - COST_ROUNDTRIP, 4)
    out["control"] = round(control, 4)
    out["gross"] = round(sign * g_stock - COST_ROUNDTRIP, 4)
    # fixed 1/5/20d β-adj net (for AVOID-as-short cut + horizon view)
    fixed = {}
    for h in FIXED_HORIZONS:
        j = i0 + h
        if j < len(dates) and dates[j] in ss and dates[j] in idx_s and dates[j] <= last_date:
            ed = dates[j]
            gs = ss[ed][3] / entry_px - 1.0
            gi = idx_s[ed][3] / idx_s[entry_date][3] - 1.0
            fixed[str(h)] = round(sign * (gs - beta * gi) - COST_ROUNDTRIP, 4)
    out["fixed"] = fixed
    return out


# --------------------------------------------------------------------------- #
# multiple-testing deflated threshold (Bonferroni over K = segments x horizons)
# --------------------------------------------------------------------------- #
def _two_sided_p(t):
    return 2 * (0.5 * math.erfc(abs(t) / math.sqrt(2)))


def _t_for_p(p):
    lo, hi = 0.0, 12.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if _two_sided_p(mid) > p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _agg(vals: List[float]) -> dict:
    a = np.asarray(vals, float)
    if a.size == 0:
        return {"n": 0, "mean_pct": None, "hit": None, "t": None}
    return {"n": int(a.size), "mean_pct": round(float(a.mean()) * 100, 3), "t": _nw_tstat(list(a), 0)}


def _verdict(n: int, mean_pct, t, bar: float) -> str:
    """§1.2 UNIFORM gate (no per-segment sign flip): net_abnormal already carries the call's sign, so a
    call RIGHT in its stated direction is POSITIVE for EVERY segment (long AND bearish)."""
    if n < 30:
        return "INCONCLUSIVE (n<30 floor)"
    if t is not None and mean_pct is not None and mean_pct > 0 and t >= bar:
        return "PASS (mean net > 0 and t >= +deflated bar)"
    if t is not None and t <= -bar:
        return "WRONG-SIGNED (significantly against the stated direction)"
    return "no edge (|t| below deflated bar)"


def _testb_verdict(caar_pct, t, n, bar: float) -> str:
    """§1.2 sign convention applied to Test B (CAAR): positive in his favour to PASS, never abs(t)."""
    if n < 30 or caar_pct is None or t is None:
        return "INCONCLUSIVE"
    if caar_pct > 0 and t >= bar:
        return "PASS"
    if caar_pct < 0 and t <= -bar:
        return "WRONG-SIGNED"
    return "no edge"


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def run(limit: Optional[int] = None) -> dict:
    calls = load_calls()
    if limit:
        calls = calls[:limit]
    pubs = [_to_kst(c["publish"]).date() for c in calls]
    start = (min(pubs) - timedelta(days=BETA_LOOKBACK + 180)).strftime("%Y%m%d") if pubs else "20240101"
    end = datetime.now(KST).strftime("%Y%m%d")
    idx_s = _ohlcv5(INDEX_PROXY, start, end)
    dates = sorted(idx_s)
    last_date = dates[-1] if dates else datetime.now(KST).date()

    scored = []
    for c in calls:
        ss = _ohlcv5(c["ticker"], start, end)
        scored.append({**c, **score_call(c, ss, idx_s, dates, last_date)})

    entered = [s for s in scored if s.get("status") == "entered"]
    n_window = sum(1 for s in scored if s.get("status") in ("entered", "non_triggered"))
    non_trig = sum(1 for s in scored if s.get("status") == "non_triggered")

    # K = segments-tested x horizons (primary TB + 1/5/20 + Test-B CAAR), Bonferroni-deflated bar
    seg_present = [seg for seg in SEGMENTS if sum(1 for s in entered if s["play"] == seg) > 0]
    n_horizons = 1 + len(FIXED_HORIZONS)                            # triple-barrier + 1/5/20
    K = max(1, len(seg_present) * n_horizons + 1)                   # +1 for Test B
    deflated_bar = round(max(TSTAT_BAR, _t_for_p(_two_sided_p(TSTAT_BAR) / K)), 3)

    segments = {}
    for seg in SEGMENTS:
        rows = [s for s in entered if s["play"] == seg]
        prim = _agg([s["net_abnormal"] for s in rows])
        prim["hit_rate"] = round(float(np.mean([1.0 if s.get("hit") else 0.0 for s in rows])), 3) if rows else None
        prim["control_mean_pct"] = round(float(np.mean([s["control"] for s in rows])) * 100, 3) if rows else None
        prim["inferred_rows"] = sum(1 for s in rows if s.get("inferred_levels"))
        verdict = _verdict(prim["n"], prim["mean_pct"], prim["t"], deflated_bar)
        fixed = {str(h): _agg([s["fixed"][str(h)] for s in rows if str(h) in s.get("fixed", {})])
                 for h in FIXED_HORIZONS}
        segments[seg] = {"hypothesis": ("negative-price/down-call" if seg == "BEARISH" else "positive"),
                         "n_entered": prim["n"], "primary": prim, "fixed_horizons": fixed, "verdict": verdict}

    # ---- Test B: event study CAAR t0..t+5 vs control, event-clustered (by ISO week) SE ----
    test_b = _event_study(entered, idx_s, dates)
    tb_c, tb_t, tb_n = test_b.get("caar_pct"), test_b.get("t"), test_b.get("n_events", 0)
    test_b["bar"] = deflated_bar
    test_b["verdict"] = _testb_verdict(tb_c, tb_t, tb_n, deflated_bar)
    test_b["pass"] = test_b["verdict"] == "PASS"
    test_b["wrong_signed"] = test_b["verdict"] == "WRONG-SIGNED"

    # ---- decision (§5 locked) ----
    seg_pass = [seg for seg in SEGMENTS if segments[seg]["verdict"].startswith("PASS")]
    wrong = [seg for seg in SEGMENTS if segments[seg]["verdict"].startswith("WRONG-SIGNED")]
    if test_b["wrong_signed"]:
        wrong = wrong + ["Test B CAAR(t+5)"]
    long_pass = [s for s in seg_pass if s in SCORED_LONG]
    if seg_pass or test_b["pass"]:
        if long_pass:
            decision = f"PASS — {', '.join(long_pass)}{' + Test B' if test_b['pass'] else ''} clears the deflated bar"
        elif "BEARISH" in seg_pass:
            decision = "BEARISH-only edge -> risk/'avoid' monitor, not a buy oracle"
        else:
            decision = "PASS — Test B clears the deflated bar"
    elif wrong:
        decision = (f"WRONG-SIGNED — {', '.join(wrong)} significantly AGAINST the stated direction "
                    "(contrarian lead; needs its own out-of-sample test; NOT traded from this run)")
    elif all(segments[seg]["n_entered"] < 30 for seg in SEGMENTS):
        decision = "INCONCLUSIVE — every segment below the n>=30 floor (corpus too small / not clean yet)"
    else:
        decision = "descriptive only — no segment or Test B clears the deflated bar"

    return {
        "generated": datetime.now(KST).strftime("%Y-%m-%d %H:%M KST"),
        "test": "Phase 1B — faithful, play-type-aware re-test",
        "banner": "research / mock only — NOT a trading signal; pre-registered spec, not re-tuned on results",
        "index_proxy": INDEX_PROXY, "data_through": last_date.isoformat(),
        "cost_roundtrip_pct": COST_ROUNDTRIP * 100, "beta_lookback_days": BETA_LOOKBACK,
        "base_tstat_bar": TSTAT_BAR, "K_tests": K, "deflated_tstat_bar": deflated_bar,
        "n_calls": len(calls), "n_entered": len(entered), "n_window_calls": n_window,
        "non_triggered": non_trig,
        "non_triggered_rate": round(non_trig / n_window, 3) if n_window else None,
        "n_inferred_levels": sum(1 for s in entered if s.get("inferred_levels")),
        "bucket_counts": {b: sum(1 for s in scored if s.get("bucket") == b or s.get("play") == b)
                          for b in list(SEGMENTS) + list(EXCLUDED_BUCKETS)},
        "n_excluded": sum(1 for s in scored if s.get("status") == "excluded"),
        "segments": segments, "test_b": test_b, "decision": decision,
        "n_degraded_calls": sum(1 for s in scored if s.get("degraded")),
    }


def _event_study(entered, idx_s, dates) -> dict:
    """CAAR over t0..t+5 (β-adjusted vs market control), event-clustered SE (cluster=ISO week of entry)."""
    rows = []
    for s in entered:
        ed = datetime.fromisoformat(s["entry_date"]).date()
        if ed not in dates:
            continue
        i0 = dates.index(ed)
        ss = _ohlcv5(s["ticker"], (ed - timedelta(days=400)).strftime("%Y%m%d"),
                     (ed + timedelta(days=30)).strftime("%Y%m%d"))
        beta = s.get("beta", 1.0) or 1.0
        sign = -1 if s["play"] == "BEARISH" else 1
        path = []
        ok = True
        for t in range(0, 6):
            j = i0 + t
            if j >= len(dates) or dates[j] not in ss or dates[j] not in idx_s:
                ok = False
                break
            ed_t = dates[j]
            gs = ss[ed_t][3] / ss[ed][3] - 1.0 if ed in ss else None
            gi = idx_s[ed_t][3] / idx_s[ed][3] - 1.0
            if gs is None:
                ok = False
                break
            path.append(sign * (gs - beta * gi))
        if ok and len(path) == 6:
            rows.append((ed.isocalendar()[:2], path[-1]))            # CAR to t+5, cluster key = (iso year,week)
    if len(rows) < 1:
        return {"n_events": 0, "caar_pct": None, "t": None, "pass": False, "note": "no complete event paths"}
    cars = np.array([r[1] for r in rows], float)
    caar = float(cars.mean())
    # event-clustered SE: average within cluster, then SE across clusters
    clusters = {}
    for k, v in rows:
        clusters.setdefault(k, []).append(v)
    cl_means = np.array([np.mean(v) for v in clusters.values()], float)
    se = float(cl_means.std(ddof=1) / math.sqrt(len(cl_means))) if len(cl_means) > 1 else None
    t = round(caar / se, 2) if se and se > 0 else None
    drift = "continuation" if caar > 0 else "reversal"
    # pass/wrong_signed are judged in run() against the DEFLATED bar with the §1.2 sign convention
    # (positive CAAR in his favour to PASS; negative ⇒ WRONG-SIGNED) — NOT abs(t).
    return {"n_events": len(rows), "n_clusters": len(clusters), "caar_pct": round(caar * 100, 3),
            "t": t, "drift": drift}


def main():
    import argparse
    import sys
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="score only the first N calls (smoke test)")
    ap.add_argument("--smoke", action="store_true", help="dry run on a small sample to prove the code works")
    a = ap.parse_args()
    rep = run(limit=(a.limit or (8 if a.smoke else None)))
    out = config.DATA_DIR / ("phase1b_smoke.json" if a.smoke else "phase1b_callscore.json")
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[phase1b] {'SMOKE ' if a.smoke else ''}calls={rep['n_calls']} entered={rep['n_entered']} "
          f"non-triggered={rep['non_triggered']} excluded={rep['n_excluded']}")
    print(f"[phase1b] K={rep['K_tests']} deflated t-bar={rep['deflated_tstat_bar']} (base {rep['base_tstat_bar']})")
    for seg in SEGMENTS:
        s = rep["segments"][seg]
        print(f"  {seg:11} n={s['n_entered']:>4} mean%={s['primary']['mean_pct']} "
              f"t={s['primary']['t']} -> {s['verdict']}")
    tb = rep["test_b"]
    print(f"  Test B (CAAR t+5): n={tb['n_events']} caar%={tb['caar_pct']} t={tb['t']} pass={tb['pass']}")
    print(f"[phase1b] DECISION: {rep['decision']}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
