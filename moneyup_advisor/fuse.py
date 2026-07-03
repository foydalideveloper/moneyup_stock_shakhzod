"""Audio <-> video fusion on one timeline (base env).

Tags (per the brief):
  * AGREE       — audio and video say the same thing (same ticker on screen + spoken; or a spoken
                  number matches the OCR'd number within tolerance).
  * AUDIO-ONLY  — stated in the transcript, not corroborated on screen.
  * VIDEO-ONLY  — on screen (OCR / chart geometry / VLM), not spoken. VLM claims live HERE and are
                  never promoted to facts (Rule 1) — incl. the 호가 ladder Qwen hallucinated in Step 0.
  * CONFLICT    — audio asserts a number/trend that the OCR/geometry contradicts.

PaddleOCR numbers are the source of truth; audio is only cross-checked against them, never the reverse.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Optional

from moneyup_advisor import calls, ocr_parse, tickers
from tagent.news.youtube_source import mmss

_WINDOW_S = 9.0
_NUM_RE = re.compile(r"[0-9][0-9,]{2,}")
_MAN_RE = re.compile(r"([0-9]+)\s*만\s*(?:([0-9]+)\s*천)?")
_PCT_RE = re.compile(r"([0-9]+\.[0-9]+)\s*%")
_VOL_RE = re.compile(r"([0-9][0-9,]*)\s*만")


def stated_numbers(segments: List[dict]) -> Dict[str, dict]:
    """The PRIMARY ticker's numbers as SPOKEN in the opening narration (audio-grounded). These
    머니업 videos show a sector WATCHLIST (peers) on the HTS, not the primary — so the primary's
    현재가/등락률/거래량 come from speech, not the OCR'd panel. Each carries its quote + timestamp."""
    out: Dict[str, dict] = {}
    for s in segments[:18]:
        t, st = s.get("text", ""), float(s.get("start", 0.0) or 0.0)
        if "change_pct" not in out:
            m = _PCT_RE.search(t)
            if m and any(k in t for k in ("상승", "하락", "등락", "올라", "빠", "보합")):
                v = float(m.group(1))
                if ("하락" in t or "빠" in t or "마이너스" in t) and "상승" not in t:
                    v = -v
                out["change_pct"] = {"value": v, "quote": t.strip(), "t": st}
        if "price" not in out and "원" in t:
            p = calls.parse_korean_price(t)
            if p and p >= 1000:
                out["price"] = {"value": p, "quote": t.strip(), "t": st}
        if "volume" not in out and "거래량" in t:
            m = _VOL_RE.search(t)
            if m:
                out["volume"] = {"value": int(m.group(1).replace(",", "")) * 10000,
                                 "quote": t.strip(), "t": st}
    return out
_UP_WORDS = ["상승", "오를", "올라", "반등", "급등", "양봉", "돌파", "우상향"]
_DOWN_WORDS = ["하락", "떨어", "급락", "조정", "음봉", "빠질", "우하향", "흘러"]
# VLM structured fields surfaced as [chart] (new schema) with back-compat to the old keys
_VLM_FIELDS = ("instrument", "timeframe", "trend_structure", "moving_averages", "key_levels",
               "recent_event", "chart_pattern", "points_at")
_VLM_EMPTY = {"", "안 보임", "not legible", "none", "none visible", "n/a", "unclear", "unknown", "not visible"}


def _vlm_trend(p: dict) -> Optional[str]:
    """Map the VLM's structural read to up/down for the deterministic cross-check (None if ambiguous)."""
    s = " ".join(str(p.get(k, "")) for k in ("trend_structure", "stance", "chart_pattern", "recent_event")).lower()
    up = any(w in s for w in ("higher-high", "higher high", "higher-low", "uptrend", "bullish", "breakout", "상승", "우상향"))
    down = any(w in s for w in ("lower-high", "lower-low", "lower low", "downtrend", "bearish", "breakdown", "하락", "우하향"))
    return "up" if up and not down else "down" if down and not up else None


# Near-term technical FEATURES the speaker emphasizes (bilingual KO/EN) + their directional bias. These
# are the SPEAKER's near-term read — kept distinct from the chart's overall multi-month structure.
_AUDIO_FEATURES = [
    ("lower high / lower low", "down", ["lower high", "lower low", "고점을 낮추", "저점을 낮추", "고점 하향", "저점 하향", "고점이 낮아", "저점이 낮아"]),
    ("higher high / higher low", "up", ["higher high", "higher low", "고점을 높이", "저점을 높이", "고점이 높아", "저점이 높아", "신고가"]),
    ("below the MA", "down", ["below the 50", "below the 200", "below the moving average", "below its 50", "below the ma",
                              "50일선 아래", "200일선 아래", "20일선 아래", "60일선 아래", "이평선 아래", "이동평균선 아래", "이평선 하향", "이평선 이탈"]),
    ("above the MA", "up", ["above the 50", "above the 200", "above the moving average", "above the ma",
                            "이평선 위", "이동평균선 위", "이평선 상향", "정배열"]),
    ("gap down", "down", ["gapped below", "gap below", "gapped down", "gap down", "갭하락", "갭 하락", "갭다운"]),
    ("gap up", "up", ["gapped above", "gap up", "gapped up", "갭상승", "갭 상승", "갭업"]),
    ("breakdown / lost support", "down", ["broke below", "break below", "breakdown", "broke down", "lost support",
                                          "하향이탈", "지지 이탈", "지지선 이탈", "지지선을 깨", "지지를 깨"]),
    ("breakout / cleared resistance", "up", ["broke above", "break above", "breakout", "broke out", "cleared resistance",
                                             "저항 돌파", "박스권 돌파", "전고점 돌파"]),
    ("at support", "neutral", ["testing support", "at support", "지지선", "지지 테스트", "지지받"]),
    ("at resistance", "neutral", ["testing resistance", "at resistance", "저항선", "저항 테스트", "저항에"]),
]


def audio_features(segments: List[dict]) -> Dict:
    """The near-term technical features the SPEAKER emphasizes (with [mm:ss]) + the near-term direction
    they imply — reported DISTINCTLY from the chart's overall multi-month structure (never blended)."""
    found, bull, bear = [], 0, 0
    for s in segments:
        t = float(s.get("start", 0.0) or 0.0)
        txt = (s.get("text", "") or "").lower()
        for label, bias, kws in _AUDIO_FEATURES:
            if any(k in txt for k in kws):
                found.append({"feature": label, "bias": bias, "mmss": mmss(t), "t": round(t, 1)})
                bull += bias == "up"
                bear += bias == "down"
    # dedupe by feature label, keep earliest mention
    seen, uniq = set(), []
    for f in found:
        if f["feature"] not in seen:
            seen.add(f["feature"])
            uniq.append(f)
    direction = "down" if bear > bull else "up" if bull > bear else None
    return {"features": uniq, "near_term_direction": direction, "n_bull": bull, "n_bear": bear}


def _call_direction(call_data) -> Optional[str]:
    """The host's EXTRACTED recommendation direction, mapped to up/down for the audio↔chart comparison.
    long->up, short->down. 'avoid' is defensive (not a directional chart bet) -> None. None if no call."""
    dirs = [c.get("direction") for c in ((call_data or {}).get("exante") or [])]
    up = sum(d == "long" for d in dirs)
    down = sum(d == "short" for d in dirs)
    return "up" if up > down else "down" if down > up else None


def _spoken_numbers(text: str) -> List[int]:
    out = [int(m.replace(",", "")) for m in _NUM_RE.findall(text)]
    for m in _MAN_RE.finditer(text):
        v = int(m.group(1)) * 10000 + (int(m.group(2)) * 1000 if m.group(2) else 0)
        out.append(v)
    return out


def _audio_index(segments: List[dict], primary_code: Optional[str]):
    """Per-segment {t, codes(set), numbers, up, down, text}."""
    idx = []
    for s in segments:
        text = s.get("text", "")
        codes = set(tickers.codes_in_text(text)) | set(tickers.names_in_text(text, min_len=3))
        idx.append({"t": float(s.get("start", 0.0) or 0.0), "codes": codes,
                    "numbers": _spoken_numbers(text),
                    "up": any(w in text for w in _UP_WORDS),
                    "down": any(w in text for w in _DOWN_WORDS), "text": text})
    return idx


def _near(audio_idx, t, window=_WINDOW_S):
    return [a for a in audio_idx if abs(a["t"] - t) <= window]


def _price_match(spoken: int, ocr: float, tol=0.02) -> bool:
    return ocr and spoken and abs(spoken - ocr) <= max(2.0, ocr * tol)


def fuse(segments: List[dict], frames: List[dict], primary_code: Optional[str],
         vlm_obs: List[dict], call_data: Optional[Dict] = None) -> Dict:
    audio_idx = _audio_index(segments, primary_code)
    agg = ocr_parse.aggregate(frames, primary_code)
    stated = stated_numbers(segments)
    timeline: List[dict] = []

    # 1) PRIMARY ticker numbers — OCR (when the primary is on the OCR'd panel) vs STATED (audio).
    #    PaddleOCR stays the source of truth; audio fills in / cross-checks. Tagged per modality:
    #    AGREE (both, consistent) / CONFLICT (both, differ) / VIDEO-ONLY (OCR only) / AUDIO-ONLY (spoken only).
    def _agree(field, a, b):
        if field == "change_pct":
            return abs(a - b) <= 0.15
        if field == "volume":
            return abs(a - b) <= max(b * 0.15, 1)
        return abs(a - b) <= max(b * 0.02, 2)               # price-class
    primary_numbers = []
    for field in ("price", "change_pct", "change_abs", "volume", "open", "high", "low"):
        ocr = agg["fields"].get(field)
        st = stated.get(field)
        if not ocr and not st:
            continue
        ov = ocr["value"] if ocr else None
        sv = st["value"] if st else None
        tag = ("AGREE" if _agree(field, ov, sv) else "CONFLICT") if (ocr and st) else \
              ("VIDEO-ONLY" if ocr else "AUDIO-ONLY")
        t = (ocr["evidence"][0]["t"] if ocr and ocr.get("evidence") else
             (st["t"] if st else 0.0)) or 0.0
        primary_numbers.append({"field": field, "ocr_value": ov,
                                "ocr_confidence": ocr["score"] if ocr else None,
                                "ocr_frames": ocr["n_frames"] if ocr else None,
                                "stated_value": sv, "stated_quote": st["quote"] if st else None,
                                "tag": tag, "t": round(t, 1)})
        vid = (f"OCR {field}={ov:,}" if isinstance(ov, int) else f"OCR {field}={ov}") if ocr else None
        aud = (f"stated {field}={sv:,}" if isinstance(sv, int) else f"stated {field}={sv}") if st else None
        timeline.append({"t": t, "mmss": mmss(t), "ticker": primary_code, "kind": "number",
                         "field": field, "tag": tag, "video": vid, "audio": aud})

    # 2) per-ticker subject coverage (who's on screen vs who's spoken)
    spoken_codes = Counter(c for a in audio_idx for c in a["codes"])
    screen_codes = Counter(agg["codes_seen"])
    for code in sorted(set(spoken_codes) | set(screen_codes)):
        sp, sc = spoken_codes.get(code, 0), screen_codes.get(code, 0)
        if sp < 2 and sc == 0:                          # drop 1x-spoken noise (substring false hits)
            continue
        in_audio, in_video = sp > 0, sc > 0
        tag = "AGREE" if (in_audio and in_video) else ("AUDIO-ONLY" if in_audio else "VIDEO-ONLY")
        timeline.append({"t": 0.0, "mmss": "00:00", "ticker": code, "kind": "subject",
                         "tag": tag, "name": tickers.display_name(code),
                         "video": f"OCR'd on {screen_codes.get(code,0)} frame(s)" if in_video else None,
                         "audio": f"spoken {spoken_codes.get(code,0)}x" if in_audio else None})

    # 3) chart geometry (VIDEO-ONLY) + trend agreement with audio
    charts = [f["chart"] for f in frames if f.get("chart", {}).get("has_chart")]
    chart_trend = None
    if charts:
        chart_trend = Counter(c.get("trend_dir") for c in charts).most_common(1)[0][0]
        up_votes = sum(a["up"] for a in audio_idx)
        down_votes = sum(a["down"] for a in audio_idx)
        audio_trend = "up" if up_votes > down_votes else "down" if down_votes > up_votes else None
        tag = ("AGREE" if audio_trend == chart_trend else
               "CONFLICT" if audio_trend and audio_trend != chart_trend else "VIDEO-ONLY")
        timeline.append({"t": 0.0, "mmss": "00:00", "ticker": primary_code, "kind": "chart",
                         "tag": tag, "video": f"OpenCV candle trend: {chart_trend} "
                         f"({len(charts)} chart frames)",
                         "audio": f"audio trend words up={up_votes}/down={down_votes}" if audio_trend
                                  else None})

    # 3b) chart price-axis SCALE (round evenly-spaced gridline ticks) -> context ONLY, NOT price levels
    axis_seen = Counter()
    for f in frames:
        for lv in (f.get("axis_levels") or []):
            axis_seen[lv["text"]] += 1
    axis_top = [t for t, _ in axis_seen.most_common(8)]
    if len(axis_top) >= 3:
        timeline.append({"t": 0.0, "mmss": "00:00", "ticker": primary_code, "kind": "axis_scale",
                         "tag": "VIDEO-ONLY", "video": "chart AXIS SCALE / gridline ticks (OCR): "
                         + ", ".join(axis_top), "audio": None,
                         "note": "even-spaced axis SCALE — NOT price levels; never treat as support/resistance"})

    # 3c) on-screen EXACT PRICE per instrument (OCR header C / last-price label = SOURCE OF TRUTH),
    #     attributed to the on-screen symbol. Audio = cross-check / fallback only.
    # gate on a real OHLC header somewhere in the video (TradingView-style chart). A Korean HTS has none,
    # so this whole block is skipped for 머니업 -> no spurious on-screen-price entries (no regression).
    has_header = any((f.get("price_field") or {}).get("c") for f in frames)
    px_by_sym = {}
    if has_header:
        for f in frames:
            pf = f.get("price_field") or {}
            price = pf.get("c") or (f.get("axis_price") or {}).get("value")
            if price is None:
                continue
            key = pf.get("symbol") or "(on-chart)"
            px_by_sym.setdefault(key, Counter())[round(price, 2)] += 1
    price_fields = []
    sp = (stated.get("price") or {}).get("value")
    for sym, cnts in px_by_sym.items():
        val = cnts.most_common(1)[0][0]                   # modal on-screen price for this instrument
        agree = sp is not None and _price_match(int(sp), val)
        price_fields.append({"symbol": sym, "price": val, "n_frames": cnts.total(),
                             "audio_agrees": bool(agree)})
        timeline.append({"t": 0.0, "mmss": "00:00", "ticker": primary_code, "kind": "onscreen_price",
                         "tag": "AGREE" if agree else "VIDEO-ONLY",
                         "video": f"on-screen price (OCR header/last-label — source of truth): {sym} ~{val:,.2f}",
                         "audio": (f"speaker states ~{sp:,}" if agree else None),
                         "note": "OCR-exact on-screen price attributed to the instrument"})

    # 4) VLM observations — VIDEO-ONLY visual interpretation (never numbers), cross-checked vs the
    #    deterministic OpenCV trend so a hallucinated trend is CONFLICT-tagged, not trusted.
    for ob in vlm_obs or []:
        p = ob.get("parsed", {})
        parts = [f"{k}={p.get(k)}" for k in _VLM_FIELDS
                 if p.get(k) and str(p.get(k)).strip().lower() not in _VLM_EMPTY]
        if not parts:
            continue
        vt = _vlm_trend(p)
        tag = ("AGREE" if chart_trend and vt == chart_trend else
               "CONFLICT" if chart_trend and vt and vt != chart_trend else "VIDEO-ONLY")
        t = float(ob.get("t", 0.0))
        timeline.append({"t": t, "mmss": mmss(t), "ticker": primary_code, "kind": "vlm", "tag": tag,
                         "video": "Qwen3-VL (visual / unverified): " + "; ".join(parts),
                         "audio": (f"OpenCV trend={chart_trend}" if chart_trend and tag != "VIDEO-ONLY" else None),
                         "note": "VLM interpretation only — no numbers (Rule 1); trend cross-checked vs OpenCV"})

    # 5) NEAR-TERM speaker feature (distinct from the overall structure) + AUDIO↔CHART conflict (Task 1)
    af = audio_features(segments)
    vlm_stance_c = Counter(_vlm_trend(o.get("parsed", {})) for o in (vlm_obs or [])
                           if _vlm_trend(o.get("parsed", {})))
    vlm_dir = vlm_stance_c.most_common(1)[0][0] if vlm_stance_c else None
    overall_dir = vlm_dir or chart_trend                 # the chart's overall multi-month structure
    # near-term direction the speaker emphasizes; explicit technical feature wins, else fall back to audio
    # up/down sentiment ONLY with a clear >=2:1 margin (so a 1-vote sentiment diff can't fire a conflict)
    audio_up = sum(a["up"] for a in audio_idx)
    audio_down = sum(a["down"] for a in audio_idx)
    near_dir = af["near_term_direction"]
    if near_dir is None:
        if audio_down >= 2 * max(1, audio_up):
            near_dir = "down"
        elif audio_up >= 2 * max(1, audio_down):
            near_dir = "up"
    if af["features"]:
        feats = "; ".join(f"{f['feature']} [{f['mmss']}]" for f in af["features"][:6])
        timeline.append({"t": 0.0, "mmss": "00:00", "ticker": primary_code, "kind": "near_term",
                         "tag": "AUDIO-ONLY", "audio": f"speaker's NEAR-TERM emphasis ({near_dir or '?'}): " + feats,
                         "video": (f"chart OVERALL structure: {overall_dir}" if overall_dir else None),
                         "note": "near-term feature the speaker points at — reported separately from the multi-month structure"})

    # AUDIO↔CHART direction: his EXTRACTED CALL (actual recommendation) when the video has one; else the
    # near-term feature/sentiment. AGREE if it matches the chart's overall structure, CONFLICT if opposed.
    call_dir = _call_direction(call_data)
    if call_dir is not None:
        speaker_dir, basis = call_dir, "extracted call (recommendation)"
    elif near_dir is not None:
        speaker_dir, basis = near_dir, ("near-term feature" if af["features"] else "audio sentiment")
    else:
        speaker_dir, basis = None, None
    audio_chart = {"speaker_dir": speaker_dir, "basis": basis, "overall_dir": overall_dir, "tag": None}
    if speaker_dir and overall_dir:
        audio_chart["tag"] = "AGREE" if speaker_dir == overall_dir else "CONFLICT"
        timeline.append({"t": 0.0, "mmss": "00:00", "ticker": primary_code, "kind": "audio_vs_chart",
                         "tag": audio_chart["tag"], "audio": f"speaker ({basis}): {speaker_dir}",
                         "video": f"chart overall (VLM/geometry): {overall_dir}",
                         "note": ("recommendation matches the chart's overall structure" if audio_chart["tag"] == "AGREE"
                                  else "speaker thesis opposes the chart's overall structure — timeframe split / contrarian")})

    timeline.sort(key=lambda e: (e["t"], e["kind"]))
    summary = Counter(e["tag"] for e in timeline)
    return {"timeline": timeline, "summary": dict(summary), "ocr_aggregate": agg,
            "primary_numbers": primary_numbers, "audio_features": af, "price_fields": price_fields,
            "near_term_direction": near_dir, "overall_direction": overall_dir, "audio_chart": audio_chart}
