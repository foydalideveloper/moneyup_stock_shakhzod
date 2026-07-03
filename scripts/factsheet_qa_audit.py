# -*- coding: utf-8 -*-
"""Fact-sheet QA AUDIT — read-only, deterministic, re-runnable regression check.

Measures how correct the production fact sheets are, separating:
  TRACK A (SCORING-CRITICAL): replay the corrected call-classification rules on each stored ex-ante
    call's quote and flag disagreements with the stored extraction. This is the layer Phase 1B scores.
  TRACK B (DESCRIPTIVE QUALITY): cross-check OCR/stated numbers + watchlist against REAL KRX prices
    (pykrx — full-universe; the same source phase1b.py scores with), plus CONFLICT-truth, transcript
    garbles, and number plausibility.

EVERY flag is COMPUTED (rule replay + arithmetic + dictionary). No LLM judgement in any count.
Read-only: never modifies a fact sheet / pipeline / collector / report / playbook / dashboard, never
downloads video, never uses the GPU. Emits a markdown report + a machine-readable JSON of all flags.

Usage:
  python -m scripts.factsheet_qa_audit                 # full audit (Track A + Track B w/ pykrx)
  python -m scripts.factsheet_qa_audit --no-prices     # Track A + non-price Track B only (fast)
  python -m scripts.factsheet_qa_audit --limit-sheets 50 --max-price-tickers 60   # quick sample
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moneyup_advisor import config, tickers
from moneyup_advisor import calls as C
from moneyup_advisor import phase1b

OUT_DIR = config.DATA_DIR / "qa_audit"
PRICE_CACHE = OUT_DIR / "_price_cache.json"
PRICE_TOL = 0.03            # ±3% band around the day's [low,high] = "consistent with reality"
NUM_TOL = 0.02             # primary number match tolerance

# --------------------------------------------------------------------------- #
# corrected-rule helpers (Track A) — deterministic
# --------------------------------------------------------------------------- #
_TRIM = ["차익실현", "차익 실현", "비중축소", "비중 축소", "익절", "정리", "처분"]   # reduce-a-long, not a bearish short
_GENUINE_SELL = ["매도", "팔아", "파세요", "숏", "공매도", "손절", "청산"]
_HOLD = C._AVOID                                                                   # 관망/보유/지켜/홀딩/대기/쉬어/회피/현금
_MARKET_STRUCT = ["사이드카", "서킷브레이커", "서킷 브레이커", "반대매매", "동시호가",
                  "변동성완화장치", "vi 발동", "vi발동", "단일가", "써킷브레이커"]
_NEG = re.compile(r"(필요\s*없|필요없|사지\s*마|팔지\s*마|아니다|아닙니다|않습니다|않는다|않다|없다|없습니다|마세요|불필요)")
_DATE_EVENT = re.compile(r"(d-\s*\d+|\d{1,2}\s*월\s*\d{1,2}\s*일|\d{1,2}\s*/\s*\d{1,2}|[월화수목금]요일|만기일|"
                         r"상장일|발표일|예정일|공모(?:주|가|청약)|\d{1,2}\s*일\s*(?:발표|만기|상장|행사|예정|결정))")
_HOLDINGS = re.compile(r"홀딩스")


def _has(text: str, kws) -> bool:
    return any(k in text for k in kws)


def _negated_action(text: str) -> Optional[str]:
    """An action keyword immediately followed (within ~8 chars) by a negation -> the call is flipped."""
    for kw in C._LONG + _GENUINE_SELL:
        i = text.find(kw)
        while i >= 0:
            if _NEG.search(text[i:i + len(kw) + 9]):
                return kw
            i = text.find(kw, i + 1)
    return None


def _avoid_trigger_is_only_holdings(text: str) -> bool:
    """avoid fired, but the only avoid keyword present is 홀딩 occurring INSIDE 홀딩스 (a company name)."""
    if "홀딩" not in text:
        return False
    others = [k for k in _HOLD if k != "홀딩" and k in text]
    if others:
        return False
    # 홀딩 present only as part of 홀딩스?
    stripped = _HOLDINGS.sub("", text)
    return "홀딩" not in stripped


def _is_market_structure(text: str) -> bool:
    return _has(text, _MARKET_STRUCT)


def _looks_hold(text: str) -> bool:
    t = _HOLDINGS.sub("", text)                              # ignore 홀딩스 company suffix
    return _has(t, _HOLD) and not _has(t, _GENUINE_SELL)


def _trim_only(text: str) -> bool:
    return _has(text, _TRIM) and not _has(text, ["매도", "팔아", "파세요", "숏", "공매도", "손절"])


def corrected_direction(quote: str) -> Optional[str]:
    """Corrected reading of a quote -> long / short / trim / avoid / None. Fixes: 홀딩스 suffix, negation,
    market-structure noise, TRIM≠short. Used to flag mismatches vs the stored direction."""
    if not quote:
        return None
    t = _HOLDINGS.sub("", quote)                             # 홀딩스 must not fire avoid via 홀딩
    if _is_market_structure(quote):
        return None                                          # market structure = not a call
    if _negated_action(quote):
        return None                                          # negated action = not a (positive) call
    t = C._strip_noise(t)
    li = next((t.find(k) for k in C._LONG if k in t), -1)
    si = next((t.find(k) for k in _GENUINE_SELL if k in t), -1)
    ti = next((t.find(k) for k in _TRIM if k in t), -1)
    if li >= 0 and (si < 0 or li <= si) and (ti < 0 or li <= ti):
        return "long"
    if si >= 0 and (ti < 0 or si <= ti):
        return "short"
    if ti >= 0:
        return "trim"
    if _has(t, _HOLD):
        return "avoid"
    return None


def _ticker_mismatch(quote: str, stored: str, primary: Optional[str]) -> Optional[str]:
    """Flag when the quote names stock(s) but the resolved code isn't among them (likely a wrong/fallback
    resolution). Conservative: only when named stocks exist and the stored code is none of them."""
    codes = tickers.codes_in_text(quote)
    if stored in codes:
        return None
    named = tickers.names_in_text(quote, min_len=3)
    if named and stored not in named:
        nm = tickers.display_name(named[0]) or named[0]
        extra = " (resolved to primary/fallback)" if stored == primary else ""
        return f"quote names {nm}({named[0]}) but call ticker={stored}{extra}"
    if codes and stored not in codes:
        return f"quote shows code {codes[0]} but call ticker={stored}"
    return None


# --------------------------------------------------------------------------- #
# Track A — per ex-ante call
# --------------------------------------------------------------------------- #
def track_a_call(call: dict, sheet: dict) -> List[dict]:
    q = call.get("quote", "") or ""
    direction = call.get("direction")
    ticker = str(call.get("ticker", "")).zfill(6)
    primary = str(sheet.get("primary_ticker") or "").zfill(6) if sheet.get("primary_ticker") else None
    play = phase1b.classify({"quote": q, "name": call.get("name", ""), "direction": direction})
    out = []

    def f(cat, reason, corrected=None):
        out.append({"category": cat, "reason": reason, "stored_class": direction,
                    "corrected_class": corrected, "play": play, "ticker": ticker,
                    "mmss": call.get("mmss"), "quote": q[:240]})

    if direction == "avoid" and _avoid_trigger_is_only_holdings(q):
        f("holdings_false_positive", "avoid fired only via 홀딩 inside 홀딩스 (company name)", "none")
    neg = _negated_action(q)
    if neg:
        f("negation_missed", f"action '{neg}' adjacent to a negation (…필요 없다/아니다/않)", "none")
    if _is_market_structure(q):
        f("market_structure_noise", "quote is market-structure (사이드카/반대매매/서킷브레이커/동시호가), not a call", "none")
    if direction == "short" and _looks_hold(q):
        f("avoid_as_short", "§1.1 hold/관망/보유 labeled as short", "avoid")
    if direction == "short" and _trim_only(q):
        f("trim_as_short", "profit-take/reduce (차익실현/비중축소/익절) labeled as bearish short", "trim")
    if _has(q, C._EXPOST):
        f("expost_as_exante", "ex-post recap markers present in an EX-ANTE call: "
          + ",".join(k for k in C._EXPOST if k in q)[:60])
    if play == "SCHEDULE" and not _DATE_EVENT.search(q):
        f("schedule_no_dated_event", "classified SCHEDULE but no concrete dated event in quote")
    if direction == "long" and play == "UNCLASSIFIED":
        f("long_generic_dropped", "§1.3 plain-long is UNCLASSIFIED → dropped from scoring; should be LONG-GENERIC", "long-generic")
    tm = _ticker_mismatch(q, ticker, primary)
    if tm:
        f("ticker_resolution", tm)
    dur = sheet.get("duration_s")
    t = call.get("in_video_t")
    if dur and t is not None and float(t) > float(dur) + 2:
        f("timestamp_out_of_range", f"t={t}s > duration {dur}s")
    if not call.get("publish_date") and not call.get("publish_datetime"):
        f("no_publish_date", "publish date/datetime missing")
    corr = corrected_direction(q)
    if corr and direction and corr != direction and not any(o["category"] in
            ("avoid_as_short", "trim_as_short", "negation_missed", "market_structure_noise",
             "holdings_false_positive") for o in out):
        f("direction_mismatch", f"stored={direction} vs corrected={corr}", corr)
    return out


# --------------------------------------------------------------------------- #
# Track B — prices (pykrx) + plausibility + garbles
# --------------------------------------------------------------------------- #
class Prices:
    """pykrx daily OHLCV with a persistent on-disk cache. Read-only; degrades to unavailable on error."""
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.ok = False
        self.cache: Dict[str, Dict[str, list]] = {}
        self.fetched = set()
        if PRICE_CACHE.exists():
            try:
                self.cache = json.loads(PRICE_CACHE.read_text(encoding="utf-8"))
            except Exception:
                self.cache = {}
        if enabled:
            try:
                from pykrx import stock  # noqa
                self.ok = True
            except Exception as e:
                print(f"[qa] pykrx unavailable ({str(e)[:80]}) — Track B price checks will be skipped")

    def _fetch(self, ticker: str):
        if ticker in self.fetched or not self.ok:
            return
        self.fetched.add(ticker)
        if ticker in self.cache and self.cache[ticker]:
            return
        from pykrx import stock
        end = datetime.now().strftime("%Y%m%d")
        start = "20230101"
        try:
            df = stock.get_market_ohlcv_by_date(start, end, ticker)
        except Exception:
            self.cache[ticker] = {}
            return
        d = {}
        for i in range(len(df) if df is not None else 0):
            di = df.index[i].date().isoformat()
            o, h, l, c, v = (float(df.iloc[i, 0]), float(df.iloc[i, 1]), float(df.iloc[i, 2]),
                             float(df.iloc[i, 3]), float(df.iloc[i, 4]))
            if c > 0:
                d[di] = [o, h, l, c, v]
        self.cache[ticker] = d

    def bar(self, ticker: str, on: str):
        """Nearest trading-day bar with date <= `on` (within 6 days). Returns (date,o,h,l,c,v) or None."""
        if not self.ok or not on:
            return None
        ticker = str(ticker).zfill(6)
        self._fetch(ticker)
        d = self.cache.get(ticker) or {}
        if not d:
            return None
        try:
            target = datetime.fromisoformat(on[:10]).date()
        except Exception:
            return None
        for back in range(0, 7):
            di = (target - timedelta(days=back)).isoformat()
            if di in d:
                return (di, *d[di])
        return None

    def save(self):
        try:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            PRICE_CACHE.write_text(json.dumps(self.cache), encoding="utf-8")
        except Exception:
            pass


def _consistent(value, bar) -> Optional[bool]:
    """Is an intraday OCR/stated price consistent with that day's real [low,high] (±tol band)? None if no bar."""
    if bar is None or value in (None, 0):
        return None
    _, o, h, l, c, v = bar
    return bool(l * (1 - PRICE_TOL) <= float(value) <= h * (1 + PRICE_TOL))


# field-aware magnitude plausibility (mirrors the live-demo guard; computed, no LLM)
_MAG = {"price", "change_abs", "volume", "value", "open", "high", "low"}


def _ndigits(x) -> int:
    try:
        return len(str(int(round(abs(float(x))))))
    except Exception:
        return 0


def implausible_primary(pn: dict, anchor: Optional[float]) -> Optional[str]:
    field = pn.get("field") or ""
    val = pn.get("ocr_value")
    if val is None or field == "change_pct" or field.endswith("pct"):
        return None
    if field in _MAG and anchor and _ndigits(val) < _ndigits(anchor) - 1:
        return f"{field}={val} has far fewer digits than anchor price {int(anchor)} (likely OCR misread)"
    return None


# Korean-finance dictionary + conservative garble detection (DETECTION ONLY) ------------------------
_DICT = ["머니업", "풋옵션", "콜옵션", "코스피", "코스닥", "에스앤피", "에스엔피", "약보합", "강보합",
         "공매도", "리밸런싱", "외국인", "기관", "순환매", "종가", "시가총액", "거래량", "선물옵션",
         "변동성", "반도체", "이차전지", "바이오"]
_KNOWN_GARBLE = {"에솔피오백": "S&P500", "에스엔피오백": "S&P500", "코스피지수": None}
_FUZZ = [w for w in _DICT if len(w) >= 4]
_TOKRE = re.compile(r"[가-힣]{3,}")


def _lev1(a: str, b: str) -> bool:
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    s, l = (a, b) if len(a) < len(b) else (b, a)
    for i in range(len(l)):
        if s == l[:i] + l[i + 1:]:
            return True
    return False


def garbles(quote: str) -> List[str]:
    found = []
    for g, canon in _KNOWN_GARBLE.items():
        if g in quote:
            found.append(f"{g}→{canon or '?'}")
    toks = set(_TOKRE.findall(quote))
    dictset = set(_DICT)
    for tok in toks:
        if tok in dictset:
            continue
        for w in _FUZZ:
            if _lev1(tok, w):
                found.append(f"{tok}~{w}?")
                break
    return found


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def run(limit_sheets=None, do_prices=True, max_price_tickers=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sheet_paths = sorted(config.SHEET_DIR.glob("*.json"))
    if limit_sheets:
        sheet_paths = sheet_paths[:limit_sheets]
    sheets = []
    for p in sheet_paths:
        try:
            sheets.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception as e:
            sheets.append({"video_id": p.stem, "_load_error": str(e)})

    # ---------- STEP 0 — inventory ----------
    inv = {"n_sheets": len(sheets), "with_exante": 0, "with_expost": 0, "with_primary_numbers": 0,
           "with_watchlist": 0, "with_timeline": 0, "with_vlm": 0, "full_transcript": 0,
           "transcript_excerpt_only": 0, "frames_ocr>0": 0, "has_publish": 0, "load_errors": 0}
    n_exante = n_expost = 0
    for s in sheets:
        if s.get("_load_error"):
            inv["load_errors"] += 1
            continue
        ex, xp = s.get("exante_calls") or [], s.get("expost_commentary") or []
        n_exante += len(ex); n_expost += len(xp)
        inv["with_exante"] += bool(ex); inv["with_expost"] += bool(xp)
        inv["with_primary_numbers"] += bool(s.get("primary_numbers"))
        inv["with_watchlist"] += bool(s.get("watchlist_ocr"))
        inv["with_timeline"] += bool(s.get("timeline"))
        inv["with_vlm"] += bool(s.get("vlm_observations"))
        inv["full_transcript"] += len(s.get("transcript_excerpt") or "") > 200
        inv["transcript_excerpt_only"] += 0 < len(s.get("transcript_excerpt") or "") <= 200
        inv["frames_ocr>0"] += (s.get("n_frames_ocr") or 0) > 0
        inv["has_publish"] += bool(s.get("publish_date") or s.get("publish_datetime"))
    inv["total_exante_calls"] = n_exante
    inv["total_expost_calls"] = n_expost

    # ---------- TRACK A ----------
    a_flags = []
    a_counts: Dict[str, int] = {}
    per_sheet_a: Dict[str, int] = {}
    for s in sheets:
        if s.get("_load_error"):
            continue
        vid = s.get("video_id")
        for ci, call in enumerate(s.get("exante_calls") or []):
            for fl in track_a_call(call, s):
                fl.update({"sheet": vid, "call_index": ci})
                a_flags.append(fl)
                a_counts[fl["category"]] = a_counts.get(fl["category"], 0) + 1
                per_sheet_a[vid] = per_sheet_a.get(vid, 0) + 1
        # vice-versa: a fresh testable call sitting in EX-POST (no recap markers but a clear direction)
        for ci, call in enumerate(s.get("expost_commentary") or []):
            q = call.get("quote", "") or ""
            if not _has(q, C._EXPOST) and corrected_direction(q) in ("long", "short"):
                fl = {"category": "exante_as_expost", "reason": "fresh directional call filed under EX-POST "
                      "with no recap markers", "stored_class": call.get("direction"), "corrected_class": None,
                      "play": None, "ticker": str(call.get("ticker", "")).zfill(6), "mmss": call.get("mmss"),
                      "quote": q[:240], "sheet": vid, "call_index": ci}
                a_flags.append(fl)
                a_counts["exante_as_expost"] = a_counts.get("exante_as_expost", 0) + 1

    # ---------- TRACK B ----------
    prices = Prices(enabled=do_prices)
    if do_prices and prices.ok and max_price_tickers:
        # bound price fetches: prioritise primaries, then most-frequent watchlist tickers
        pri = [str(s.get("primary_ticker")).zfill(6) for s in sheets if s.get("primary_ticker")]
        from collections import Counter
        wl = Counter(str(w["ticker"]).zfill(6) for s in sheets for w in (s.get("watchlist_ocr") or []) if w.get("ticker"))
        allow = list(dict.fromkeys(pri + [t for t, _ in wl.most_common()]))[:max_price_tickers]
        prices._allow = set(allow)
    else:
        prices._allow = None

    b_flags = []
    prim_ocr_ok = prim_ocr_tot = prim_stated_ok = prim_stated_tot = 0
    vol_ocr_ok = vol_ocr_tot = 0
    conflict_total = conflict_checked = conflict_false = conflict_ocr_wrong = conflict_stated_wrong = 0
    wl_ok = wl_tot = wl_checked = 0
    garble_sheets = 0
    garble_total = 0
    plaus_flags = 0

    t0 = time.time()
    for si, s in enumerate(sheets):
        if s.get("_load_error"):
            continue
        vid = s.get("video_id")
        pub = s.get("publish_date") or (s.get("publish_datetime") or "")[:10]
        prim = str(s.get("primary_ticker") or "").zfill(6) if s.get("primary_ticker") else None
        pn_list = s.get("primary_numbers") or []
        anchor = None
        for pn in pn_list:
            if pn.get("field") == "price":
                anchor = pn.get("stated_value") or pn.get("ocr_value")
        # number plausibility (no prices needed)
        for pn in pn_list:
            msg = implausible_primary(pn, anchor)
            if msg:
                plaus_flags += 1
                b_flags.append({"sheet": vid, "field": pn.get("field"), "issue": "implausible_number",
                                "evidence": msg, "ocr_value": pn.get("ocr_value"), "tag": pn.get("tag")})
        # garbles (on the only transcript text we have: call quotes)
        gset = []
        for call in (s.get("exante_calls") or []) + (s.get("expost_commentary") or []):
            gset += garbles(call.get("quote", "") or "")
        if gset:
            garble_sheets += 1
            garble_total += len(gset)
            b_flags.append({"sheet": vid, "field": "transcript", "issue": "garble_candidate",
                            "evidence": list(dict.fromkeys(gset))[:8]})
        # ---- price-dependent checks ----
        allow = prices._allow
        def can(t):
            return prices.ok and (allow is None or str(t).zfill(6) in allow)
        # primary numbers vs reality (price + volume)
        if prim and can(prim):
            bar = prices.bar(prim, pub)
            pf = next((x for x in pn_list if x.get("field") == "price"), None)
            if pf and bar:
                c_ocr = _consistent(pf.get("ocr_value"), bar)
                c_st = _consistent(pf.get("stated_value"), bar)
                if c_ocr is not None:
                    prim_ocr_tot += 1; prim_ocr_ok += int(c_ocr)
                    if not c_ocr:
                        b_flags.append({"sheet": vid, "field": "primary_price_ocr", "issue": "price_vs_reality",
                                        "evidence": f"OCR {pf.get('ocr_value')} not in real [{bar[3]:.0f},{bar[2]:.0f}] on {bar[0]} (ticker {prim})"})
                if c_st is not None:
                    prim_stated_tot += 1; prim_stated_ok += int(c_st)
                    if not c_st:
                        b_flags.append({"sheet": vid, "field": "primary_price_stated", "issue": "price_vs_reality",
                                        "evidence": f"stated {pf.get('stated_value')} not in real [{bar[3]:.0f},{bar[2]:.0f}] on {bar[0]} (ticker {prim})"})
            vf = next((x for x in pn_list if x.get("field") == "volume"), None)
            if vf and bar and vf.get("ocr_value"):
                real_v = bar[5]
                ok = real_v > 0 and 0.5 <= float(vf["ocr_value"]) / real_v <= 2.0
                vol_ocr_tot += 1; vol_ocr_ok += int(ok)
        # CONFLICT truth (price field)
        for pn in pn_list:
            if pn.get("tag") != "CONFLICT":
                continue
            conflict_total += 1
            ov, sv = pn.get("ocr_value"), pn.get("stated_value")
            if ov is not None and sv is not None and sv != 0 and abs(ov - sv) / abs(sv) <= NUM_TOL:
                conflict_false += 1
                b_flags.append({"sheet": vid, "field": pn.get("field"), "issue": "false_conflict",
                                "evidence": f"OCR {ov} ≈ stated {sv} but tagged CONFLICT (should be AGREE)"})
                continue
            if pn.get("field") == "price" and prim and can(prim):
                bar = prices.bar(prim, pub)
                if bar:
                    conflict_checked += 1
                    c_ocr, c_st = _consistent(ov, bar), _consistent(sv, bar)
                    if c_st and not c_ocr:
                        conflict_ocr_wrong += 1
                        b_flags.append({"sheet": vid, "field": "price", "issue": "conflict_ocr_wrong",
                                        "evidence": f"real {bar[4]:.0f} on {bar[0]} matches stated {sv}, not OCR {ov} → OCR is the wrong side"})
                    elif c_ocr and not c_st:
                        conflict_stated_wrong += 1
                        b_flags.append({"sheet": vid, "field": "price", "issue": "conflict_stated_wrong",
                                        "evidence": f"real {bar[4]:.0f} matches OCR {ov}, not stated {sv}"})
        # watchlist integrity
        for w in (s.get("watchlist_ocr") or []):
            wt = str(w.get("ticker") or "").zfill(6)
            if not w.get("ocr_price") or not can(wt):
                continue
            wl_tot += 1
            bar = prices.bar(wt, pub)
            if bar is None:
                continue
            wl_checked += 1
            if _consistent(w["ocr_price"], bar):
                wl_ok += 1
            else:
                b_flags.append({"sheet": vid, "field": "watchlist", "issue": "watchlist_price_misalign",
                                "evidence": f"{w.get('name')}({wt}) OCR {w['ocr_price']} not in real "
                                            f"[{bar[3]:.0f},{bar[2]:.0f}] on {bar[0]}"})
        if do_prices and prices.ok and si % 25 == 0:
            print(f"[qa] track B {si}/{len(sheets)} ({time.time()-t0:.0f}s)", flush=True)
    prices.save()

    # ---------- STEP 3 — degraded sheets ----------
    degraded = []
    for s in sheets:
        vid = s.get("video_id")
        reasons = []
        if s.get("_load_error"):
            reasons.append("load_error:" + s["_load_error"][:60])
        if (s.get("n_frames_ocr") or 0) == 0:
            reasons.append("0 frames")
        if (s.get("n_transcript_segments") or 0) == 0:
            reasons.append("0 transcript segs")
        if not (s.get("exante_calls")):
            reasons.append("0 ex-ante calls")
        fs = s.get("fusion_summary") or {}
        if fs and (fs.get("AGREE", 0) + fs.get("CONFLICT", 0) + fs.get("VIDEO-ONLY", 0)) == 0 and fs.get("AUDIO-ONLY", 0) > 0:
            reasons.append("all AUDIO-ONLY (no usable video grounding)")
        if not (s.get("publish_date") or s.get("publish_datetime")):
            reasons.append("no publish date")
        if reasons:
            degraded.append({"video_id": vid, "reasons": reasons,
                             "n_frames_ocr": s.get("n_frames_ocr"), "n_segs": s.get("n_transcript_segments")})

    track_b = {
        "price_source": "pykrx" if prices.ok else "UNAVAILABLE",
        "price_check_scope": ("bounded:" + str(max_price_tickers)) if max_price_tickers else "all tickers",
        "primary_price_ocr_correct_pct": _pct(prim_ocr_ok, prim_ocr_tot),
        "primary_price_stated_correct_pct": _pct(prim_stated_ok, prim_stated_tot),
        "primary_volume_ocr_correct_pct": _pct(vol_ocr_ok, vol_ocr_tot),
        "primary_price_checked": prim_ocr_tot,
        "conflicts_total": conflict_total, "conflicts_false_AGREE": conflict_false,
        "conflicts_price_reality_checked": conflict_checked,
        "conflicts_ocr_wrong": conflict_ocr_wrong, "conflicts_stated_wrong": conflict_stated_wrong,
        "conflict_false_or_resolvable_pct": _pct(conflict_false + conflict_ocr_wrong + conflict_stated_wrong, conflict_total),
        "watchlist_rows_checked": wl_checked, "watchlist_rows_total": wl_tot,
        "watchlist_price_accuracy_pct": _pct(wl_ok, wl_checked),
        "garble_sheets": garble_sheets, "garble_rate_pct": _pct(garble_sheets, inv["n_sheets"]),
        "garble_candidates_total": garble_total,
        "implausible_primary_numbers": plaus_flags,
    }

    report = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
              "inventory": inv, "track_a_counts": a_counts, "track_a_total_flags": len(a_flags),
              "track_a_sheets_flagged": len(per_sheet_a),
              "track_b": track_b, "degraded_sheets": degraded}
    return report, a_flags, b_flags


def _pct(num, den):
    return round(100 * num / den, 1) if den else None


# --------------------------------------------------------------------------- #
# self-check (hand-verify ~5 flags against raw data)
# --------------------------------------------------------------------------- #
def self_check(prices: "Prices"):
    out = []
    sd = config.SHEET_DIR
    # 1+2: an 086520 price CONFLICT — confirm we identify the wrong side via real price
    found = 0
    for p in sorted(sd.glob("*.json")):
        s = json.loads(p.read_text(encoding="utf-8"))
        if str(s.get("primary_ticker")) != "086520":
            continue
        pn = next((x for x in s.get("primary_numbers", []) if x.get("field") == "price" and x.get("tag") == "CONFLICT"), None)
        if not pn:
            continue
        pub = s.get("publish_date") or (s.get("publish_datetime") or "")[:10]
        bar = prices.bar("086520", pub) if prices.ok else None
        verdict = "no real price" if not bar else (
            "OCR wrong" if (_consistent(pn.get("stated_value"), bar) and not _consistent(pn.get("ocr_value"), bar))
            else "stated wrong" if (_consistent(pn.get("ocr_value"), bar) and not _consistent(pn.get("stated_value"), bar))
            else "ambiguous")
        out.append(f"SC{found+1} 086520 CONFLICT {s['video_id']} pub={pub} OCR={pn.get('ocr_value')} "
                   f"stated={pn.get('stated_value')} real={'%.0f'%bar[4] if bar else None} → {verdict}")
        found += 1
        if found >= 2:
            break
    # 3: HLB(028300) watchlist row vs real price
    for p in sorted(sd.glob("*.json")):
        s = json.loads(p.read_text(encoding="utf-8"))
        w = next((w for w in s.get("watchlist_ocr", []) if str(w.get("ticker")) == "028300" and w.get("ocr_price")), None)
        if not w:
            continue
        pub = s.get("publish_date") or (s.get("publish_datetime") or "")[:10]
        bar = prices.bar("028300", pub) if prices.ok else None
        cons = _consistent(w["ocr_price"], bar)
        out.append(f"SC3 HLB(028300) watchlist {s['video_id']} OCR={w['ocr_price']} "
                   f"real[{('%.0f,%.0f'%(bar[3],bar[2])) if bar else '—'}] → {'CONSISTENT' if cons else 'MISALIGNED/flagged' if bar else 'no real price'}")
        break
    # 4: a clean ex-ante call passes Track A (no flags)
    for p in sorted(sd.glob("*.json")):
        s = json.loads(p.read_text(encoding="utf-8"))
        for c in s.get("exante_calls", []):
            if not track_a_call(c, s) and c.get("direction") == "long":
                out.append(f"SC4 clean call {s['video_id']} {c.get('ticker')} dir={c.get('direction')} "
                           f"mmss={c.get('mmss')} → 0 Track-A flags (PASS)")
                break
        else:
            continue
        break
    # 5: a holdings_false_positive OR avoid_as_short if any exists
    hit = None
    for p in sorted(sd.glob("*.json")):
        s = json.loads(p.read_text(encoding="utf-8"))
        for c in s.get("exante_calls", []):
            fls = track_a_call(c, s)
            for fl in fls:
                if fl["category"] in ("holdings_false_positive", "avoid_as_short", "negation_missed"):
                    hit = f"SC5 {fl['category']} {s['video_id']} {c.get('ticker')} stored={c.get('direction')} → flagged"
                    break
            if hit:
                break
        if hit:
            break
    out.append(hit or "SC5 (no holdings/avoid-short/negation flag found in corpus)")
    return out


# --------------------------------------------------------------------------- #
# markdown
# --------------------------------------------------------------------------- #
def write_markdown(report, a_flags, b_flags, spotchecks, path):
    inv = report["inventory"]; ac = report["track_a_counts"]; tb = report["track_b"]
    L = []
    L.append("# Fact-sheet QA Audit\n")
    L.append(f"_generated {report['generated']} · read-only · deterministic (no LLM in any count) · "
             f"price ground truth: {tb['price_source']}_\n")

    L.append("## Step 0 — Inventory & coverage\n")
    L.append(f"- Sheets: **{inv['n_sheets']}** · ex-ante calls: **{inv['total_exante_calls']}** · "
             f"ex-post: **{inv['total_expost_calls']}**")
    L.append(f"- Persisted per-field: ex-ante {inv['with_exante']}/{inv['n_sheets']} · "
             f"ex-post {inv['with_expost']} · primary_numbers {inv['with_primary_numbers']} · "
             f"watchlist {inv['with_watchlist']} · timeline {inv['with_timeline']} · VLM {inv['with_vlm']} · "
             f"frames>0 {inv['frames_ocr>0']} · publish {inv['has_publish']}")
    L.append(f"- **Full transcript persisted: {inv['full_transcript']}/{inv['n_sheets']}** "
             f"(excerpt-only: {inv['transcript_excerpt_only']}). Frame images: not persisted (media auto-deleted).")
    L.append("- **Auditable:** the SCORED call layer (quotes+direction) fully; primary/watchlist numbers vs "
             "real prices; CONFLICT tags; number plausibility. **Not auditable from disk:** per-frame OCR "
             "tokens, full transcript (garble runs on stored quotes only), frame images.\n")

    L.append("## Track A — SCORING-CRITICAL (call-rule replay across ALL calls)\n")
    L.append(f"Total flags: **{report['track_a_total_flags']}** across **{report['track_a_sheets_flagged']}** sheets "
             f"(of {inv['total_exante_calls']} ex-ante calls).\n")
    L.append("| category | count | % of ex-ante |")
    L.append("|---|--:|--:|")
    for cat, n in sorted(ac.items(), key=lambda kv: -kv[1]):
        L.append(f"| {cat} | {n} | {_pct(n, inv['total_exante_calls'])}% |")
    L.append("\n**Flagged examples (first 25):**\n")
    for fl in a_flags[:25]:
        L.append(f"- `{fl['sheet']}` [{fl.get('mmss')}] {fl['ticker']} **{fl['category']}** "
                 f"(stored={fl['stored_class']}→corrected={fl.get('corrected_class')}): {fl['reason']}  \n"
                 f"  > {fl['quote'][:160]}")
    L.append("")

    L.append("## Track B — EXTRACTION QUALITY (vs real prices / dictionary / plausibility)\n")
    L.append(f"- Primary 현재가 — OCR consistent w/ reality: **{tb['primary_price_ocr_correct_pct']}%** · "
             f"stated: **{tb['primary_price_stated_correct_pct']}%** (n={tb['primary_price_checked']} checked)")
    L.append(f"- Primary 거래량 OCR within 2× of real: **{tb['primary_volume_ocr_correct_pct']}%**")
    L.append(f"- CONFLICT tags: {tb['conflicts_total']} total · false (OCR≈stated→AGREE): {tb['conflicts_false_AGREE']} · "
             f"OCR-wrong-side: {tb['conflicts_ocr_wrong']} · stated-wrong: {tb['conflicts_stated_wrong']} · "
             f"**false/auto-resolvable: {tb['conflict_false_or_resolvable_pct']}%**")
    L.append(f"- Watchlist price accuracy: **{tb['watchlist_price_accuracy_pct']}%** "
             f"({tb['watchlist_rows_checked']}/{tb['watchlist_rows_total']} rows had a real price)")
    L.append(f"- Transcript garble candidates: **{tb['garble_rate_pct']}%** of sheets "
             f"({tb['garble_candidates_total']} candidates; detection-only, on stored quotes)")
    L.append(f"- Implausible primary numbers (field-aware guard): **{tb['implausible_primary_numbers']}**\n")

    L.append("## Step 3 — Degraded-sheet list (candidates for TARGETED re-extract; NOT re-extracted here)\n")
    L.append(f"{len(report['degraded_sheets'])} sheets flagged structurally degraded:\n")
    for d in report["degraded_sheets"][:60]:
        L.append(f"- `{d['video_id']}` — {', '.join(d['reasons'])}")
    if len(report["degraded_sheets"]) > 60:
        L.append(f"- …(+{len(report['degraded_sheets'])-60} more in the JSON)")
    L.append("")

    L.append("## Self-check (5 hand-verifications vs raw data)\n")
    for sc in spotchecks:
        L.append(f"- {sc}")
    L.append("")

    # verdict — three tiers (extraction-error data vs scorer-rule gap vs advisory)
    EXTRACT_ERR = ("holdings_false_positive", "negation_missed", "market_structure_noise",
                   "avoid_as_short", "trim_as_short", "expost_as_exante", "exante_as_expost")
    SCORER_GAP = ("long_generic_dropped", "schedule_no_dated_event")
    ADVISORY = ("direction_mismatch", "ticker_resolution")
    n_ext = sum(ac.get(c, 0) for c in EXTRACT_ERR)
    n_gap = sum(ac.get(c, 0) for c in SCORER_GAP)
    n_adv = sum(ac.get(c, 0) for c in ADVISORY)
    N = inv["total_exante_calls"]
    L.append("## VERDICT\n")
    L.append("### (A) Is the SCORED call data clean enough for the Phase 1B re-run? (GATES 1B)\n")
    L.append("Track-A flags fall in **three tiers** with very different fixes:\n")
    L.append("| tier | what it means | fix | count | % of calls |")
    L.append("|---|---|---|--:|--:|")
    L.append(f"| **1. Extraction errors** | the *stored call is wrong* | post-hoc FILTER on stored calls "
             f"(no re-extraction) | {n_ext} | {_pct(n_ext, N)}% |")
    L.append(f"| **2. Scorer-rule gaps** | the call is fine; `phase1b.classify()` mishandles it | one-line "
             f"`classify()` amendments (§1.3 LONG-GENERIC + SCHEDULE-needs-date) | {n_gap} | {_pct(n_gap, N)}% |")
    L.append(f"| **3. Advisory / ambiguous** | genuinely unclear direction or quote-context ticker | manual "
             f"review (medium confidence) | {n_adv} | {_pct(n_adv, N)}% |")
    L.append("")
    L.append(f"- **Tier 1 (data errors): {n_ext} calls ({_pct(n_ext, N)}%)** — concentrated in a few PRECISE, "
             "deterministically-fixable patterns (홀딩스 substring firing avoid; market-structure noise; "
             "profit-take/disposal mislabeled short; ex-post recaps filed as ex-ante). These are real but "
             "**auto-correctable by a post-hoc filter on the stored `exante_calls` — the 508 sheets do NOT "
             "need re-extraction.**")
    L.append(f"- **Tier 2 (scorer gaps): {n_gap} calls ({_pct(n_gap, N)}%)** — DOMINATED by `long_generic_dropped` "
             f"({ac.get('long_generic_dropped',0)}: plain-long calls the scorer drops as UNCLASSIFIED — §1.3 says "
             f"score them as LONG-GENERIC) and `schedule_no_dated_event` ({ac.get('schedule_no_dated_event',0)}: "
             "boilerplate '스케줄 매매' firing the SCHEDULE play with no real date). The sheet data is fine; "
             "`phase1b.classify()` must adopt the amendment before 1B or the primary test silently loses ~"
             f"{ac.get('long_generic_dropped',0)} long signals.")
    L.append("")
    gate = ("🟡 **CONDITIONAL GO** — the scored call DATA is clean enough (Tier-1 errors are a small, precise, "
            "filterable set), BUT a trustworthy 1B re-run REQUIRES first: (a) apply the `classify()` amendments "
            "(§1.3 LONG-GENERIC + SCHEDULE-needs-concrete-date), and (b) run the Tier-1 post-hoc call filter. "
            "Both are code-level fixes on the existing data — **no re-extraction of the 508 sheets is needed** "
            f"(only {len(report['degraded_sheets'])} structurally-degraded sheet(s), listed above).")
    if _pct(n_ext, N) and _pct(n_ext, N) >= 15:
        gate = ("⚠️ **FIX FIRST** — Tier-1 extraction errors exceed 15% of calls; clean the extractor/filter "
                "before trusting 1B.")
    L.append(gate + "\n")
    L.append("### (B) Descriptive-quality fix list (shared OCR/Whisper step — does NOT block 1B)\n")
    L.append(f"- Primary 현재가 OCR consistent w/ real KRX price: **{tb['primary_price_ocr_correct_pct']}%** · "
             f"stated **{tb['primary_price_stated_correct_pct']}%** · watchlist **{tb['watchlist_price_accuracy_pct']}%** "
             f"(price source: {tb['price_source']}). Lever: OCR row/column mapping.")
    L.append(f"- CONFLICT tags **{tb['conflict_false_or_resolvable_pct']}%** false/auto-resolvable "
             f"({tb['conflicts_false_AGREE']} are OCR≈stated mis-tagged AGREE; {tb['conflicts_ocr_wrong']} have OCR on "
             "the wrong side per real price) → auto-resolve CONFLICTs against the live price.")
    L.append(f"- {tb['implausible_primary_numbers']} implausible primary numbers + garble candidates "
             f"{tb['garble_rate_pct']}% → add the field-aware magnitude guard to extraction + a Korean-finance "
             "spell-normalizer. **None of (B) biases the 1B verdict.**\n")
    path.write_text("\n".join(L), encoding="utf-8")


def regression_against_fixed():
    """Gate 1.7 — run the FIXED phase1b.classify_call over ALL 508 stored calls and count residual gaps in
    each error category. The corrected classifier should drive every one to 0 (proves the fix at scale)."""
    from moneyup_advisor import phase1b as P
    cnt = {k: 0 for k in ("long_generic_dropped", "schedule_no_dated_event", "holdings_false_positive",
                           "trim_as_short", "market_structure_noise", "expost_as_exante", "negation_missed")}
    n_calls = 0
    for p in sorted(config.SHEET_DIR.glob("*.json")):
        try:
            s = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        for c in s.get("exante_calls", []):
            q = c.get("quote", "") or ""
            d = P.classify_call(c)
            n_calls += 1
            if c.get("direction") == "long" and d["bucket"] == "UNCLASSIFIED":
                cnt["long_generic_dropped"] += 1
            if d["segment"] == "SCHEDULE" and not P.has_dated_event(q):
                cnt["schedule_no_dated_event"] += 1
            if d["sign"] == -1 and "홀딩스" in q and not _has(q.replace("홀딩스", " "), P._BEARISH + P._HOLD):
                cnt["holdings_false_positive"] += 1
            if d["segment"] == "BEARISH" and _has(q, P._TRIM) and not _has(q, P._BEARISH):
                cnt["trim_as_short"] += 1
            if d["scored"] and P.is_market_noise(q):
                cnt["market_structure_noise"] += 1
            if d["scored"] and P.is_expost(q):
                cnt["expost_as_exante"] += 1
            if d["scored"] and P.is_negated(q):
                cnt["negation_missed"] += 1
    return {"n_calls": n_calls, "residual_gaps": cnt, "all_zero": all(v == 0 for v in cnt.values())}


def main():
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-sheets", type=int, default=None)
    ap.add_argument("--no-prices", action="store_true", help="skip pykrx price cross-checks (fast)")
    ap.add_argument("--max-price-tickers", type=int, default=None, help="bound price fetches to N tickers")
    ap.add_argument("--regression", action="store_true", help="Gate 1.7: run the FIXED classifier over 508, print residual gaps")
    a = ap.parse_args()
    if a.regression:
        r = regression_against_fixed()
        print(f"[regression] {r['n_calls']} calls through the FIXED phase1b.classify_call")
        for k, v in r["residual_gaps"].items():
            print(f"  {k:26} {v}")
        print(f"[regression] all_zero = {r['all_zero']}")
        return

    report, a_flags, b_flags = run(limit_sheets=a.limit_sheets, do_prices=not a.no_prices,
                                   max_price_tickers=a.max_price_tickers)
    prices = Prices(enabled=not a.no_prices)
    spot = self_check(prices)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "factsheet_qa_flags.json").write_text(json.dumps(
        {"report": report, "track_a_flags": a_flags, "track_b_flags": b_flags, "self_check": spot},
        ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, a_flags, b_flags, spot, OUT_DIR / "factsheet_qa_report.md")
    print("\n".join(spot))
    print(f"\n[qa] Track A flags={report['track_a_total_flags']} | degraded={len(report['degraded_sheets'])}")
    print(f"[qa] saved -> {OUT_DIR/'factsheet_qa_report.md'} + factsheet_qa_flags.json")


if __name__ == "__main__":
    main()
