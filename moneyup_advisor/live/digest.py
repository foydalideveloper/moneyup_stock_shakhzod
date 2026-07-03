# -*- coding: utf-8 -*-
"""Grounded summary + raw extraction + comparison — all as Word .docx (Korean-capable font).

Outputs (saved per job, served on live AND replay, bundled in the static export):
  * raw_full.docx + raw_full.json — DETERMINISTIC dump (no LLM): full Whisper transcript, every OCR
    token per frame (text + confidence + frame time), all Qwen3-VL observations.
  * summary_full.docx — Gemini 3.1 Pro grounded digest from transcript+OCR+VLM, each point source-
    tagged [audio]/[screen]/[chart] + [mm:ss].
  * comparison.docx — (a) Whisper-ONLY extraction (transcript verbatim), (b) full extraction,
    (c) Whisper-only summary vs full summary in a 2-column table, (d) COMPUTED "what the visual
    reading added" delta.

INTEGRITY: the Whisper-only summary is generated from ONLY the transcript (OCR/VLM never enter that
prompt — input isolation). The delta is computed from the data, not the LLM. Generated quotes are
verified against their source and dropped if not found. No fabrication, no source mixing.

Isolated: imports only config + tickers + tagent.gemini (read-only), reads playbook.json read-only,
writes only into the job folder. Never touches the collector / report / playbook.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

from moneyup_advisor import config, tickers, ocr_parse   # ocr_parse: read-only reuse (watchlist_rows; base-env, no GPU)

GEMINI_MODEL = os.getenv("MONEYUP_LIVE_SUMMARY_MODEL", "gemini-3.1-pro-preview")
GEMINI_FALLBACKS = ["gemini-3-pro-preview", "gemini-pro-latest"]
FONT = os.getenv("MONEYUP_LIVE_DOCX_FONT", "Malgun Gothic")           # Korean-capable Word font
_NUM_RE = re.compile(r"[+\-▲▼]?\s*[\d][\d,]*(?:\.\d+)?%?")
_QUOTE_RE = re.compile(r"[\"“”'‘’「」『』](.+?)[\"“”'‘’「」『』]")


def _mmss(t):
    s = int(max(0.0, float(t or 0)))
    return f"{s // 60:02d}:{s % 60:02d}"


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s or ""))


# --------------------------------------------------------------------------- #
# docx helpers (Korean font everywhere)
# --------------------------------------------------------------------------- #
def _kfont_run(run):
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    run.font.name = FONT
    rpr = run._element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.append(rf)
    for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rf.set(qn(a), FONT)


def _new_doc():
    from docx import Document
    from docx.shared import Pt
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    doc = Document()
    for sname in ["Normal", "Heading 1", "Heading 2", "Heading 3", "Title", "List Bullet"]:
        try:
            st = doc.styles[sname]
            st.font.name = FONT
            rpr = st.element.get_or_add_rPr()
            rf = rpr.find(qn("w:rFonts"))
            if rf is None:
                rf = OxmlElement("w:rFonts")
                rpr.append(rf)
            for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
                rf.set(qn(a), FONT)
        except Exception:
            pass
    try:
        doc.styles["Normal"].font.size = Pt(10.5)
    except Exception:
        pass
    return doc


def _h(doc, text, level=1):
    h = doc.add_heading("", level=level)
    r = h.add_run(text)
    _kfont_run(r)
    return h


def _p(doc, text, bullet=False, bold=False, italic=False, size=None):
    from docx.shared import Pt
    p = doc.add_paragraph(style="List Bullet") if bullet else doc.add_paragraph()
    r = p.add_run(text)
    r.bold = bold
    r.italic = italic
    if size:
        r.font.size = Pt(size)
    _kfont_run(r)
    return p


def _cell_md(cell, md, header):
    """Render a (simplified) markdown summary into a table cell."""
    cell.paragraphs[0].text = ""
    hr = cell.paragraphs[0].add_run(header)
    hr.bold = True
    _kfont_run(hr)
    for raw in (md or "").splitlines():
        line = raw.rstrip()
        if not line:
            continue
        txt = _clean_inline(re.sub(r"^#{1,6}\s*", "", line))
        p = cell.add_paragraph()
        if re.match(r"^#{1,6}\s", line):
            r = p.add_run(txt); r.bold = True
        elif re.match(r"^[-*]\s", line):
            r = p.add_run("• " + _clean_inline(line[2:]))
        else:
            r = p.add_run(txt)
        _kfont_run(r)


def _clean_inline(s):
    return (s or "").replace("**", "").replace("`", "").strip()


def _md_to_doc(doc, md):
    for raw in (md or "").splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.startswith("### "):
            _h(doc, _clean_inline(line[4:]), 3)
        elif line.startswith("## "):
            _h(doc, _clean_inline(line[3:]), 2)
        elif line.startswith("# "):
            _h(doc, _clean_inline(line[2:]), 1)
        elif re.match(r"^[-*]\s", line):
            _p(doc, _clean_inline(line[2:]), bullet=True)
        elif line.startswith(">"):
            _p(doc, _clean_inline(line.lstrip("> ")), italic=True)
        else:
            _p(doc, _clean_inline(line))


# --------------------------------------------------------------------------- #
# Gemini
# --------------------------------------------------------------------------- #
def _gemini_key() -> str:
    k = os.getenv("GEMINI_API_KEY")
    if k:
        return k.strip()
    try:
        for line in (config.REPO_ROOT / ".env").read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("GEMINI_API_KEY="):
                return line.split("=", 1)[1].strip()
    except Exception:
        pass
    return ""


def _gen(prompt: str, system: str, temperature: float = 0.2, timeout: float = 150.0) -> str:
    key = _gemini_key()
    if not key:
        return ""
    from tagent.gemini import GeminiClient                  # READ-ONLY reuse (same client the playbook uses)
    for m in [GEMINI_MODEL] + GEMINI_FALLBACKS:
        try:
            out = GeminiClient(key, model=m, timeout=timeout).generate(
                prompt, system=system, temperature=temperature, json_out=False)
            if out and out.strip():
                return out
        except Exception:
            continue
    return ""


def _playbook_strategies() -> List[str]:
    out: List[str] = []
    try:
        a = json.loads((config.DATA_DIR / "playbook" / "playbook.json").read_text(encoding="utf-8"))
        for r in a.get("rules", []):
            if r.get("canonical"):
                out.append(r["canonical"])
        for v in a.get("vocabulary", []):
            if v.get("term"):
                out.append(v["term"])
    except Exception:
        pass
    if not out:
        out = ["일정매매(스케줄 매매)", "순환매", "공매도/수급 분석", "박스권 매매", "저항 돌파매매",
               "눌림목/지지 매수", "대차·대주 분석", "차익실현/비중조절"]
    seen, uniq = set(), []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq[:24]


# --------------------------------------------------------------------------- #
# data helpers
# --------------------------------------------------------------------------- #
def _transcript_text(segments) -> str:
    return " ".join(s.get("text", "") for s in segments)


def _ocr_text(frames_data) -> str:
    return " ".join(str(r.get("text", "")) for fr in frames_data for r in (fr.get("ocr") or []))


_VLM_KEYS = ("instrument", "timeframe", "trend_structure", "moving_averages", "key_levels",
             "recent_event", "stance", "chart_pattern", "points_at", "note")
_VLM_EMPTY = {"", "안 보임", "not legible", "none", "none visible", "n/a", "unclear", "unknown", "not visible"}


def _vlm_desc(parsed) -> str:
    """Non-empty structured VLM fields (new schema + old keys) as one readable line — visual structure only."""
    p = parsed or {}
    return " · ".join(f"{k}: {p.get(k)}" for k in _VLM_KEYS
                      if p.get(k) and str(p.get(k)).strip().lower() not in _VLM_EMPTY)


def _axis_levels_all(frames_data):
    """Distinct OCR'd chart price-axis levels across frames (GROUNDED numbers) -> [(text, value), …]."""
    seen = {}
    for fr in frames_data:
        for lv in (fr.get("axis_levels") or []):
            if lv.get("text") and lv["text"] not in seen:
                seen[lv["text"]] = lv.get("value")
    return list(seen.items())


def _vlm_text(vlm_obs) -> str:
    return " ".join(_vlm_desc(o.get("parsed", {})) for o in (vlm_obs or []))


def _nl(s) -> str:
    """normalize + lowercase (for case-insensitive name/alias matching incl. romanizations like LNF)."""
    return re.sub(r"\s+", "", str(s or "")).lower()


def _plausible(tok: str) -> bool:
    """A REAL KRX-screen number? percent / decimal / comma price-or-volume / Korean myriad / short int.
    Rejects OCR junk like long bare digit runs (110130, 12345678)."""
    t = tok.strip().replace(" ", "")
    if not t:
        return False
    return bool(
        re.fullmatch(r"[+\-▲▼]?\d{1,3}(?:\.\d{1,2})?%", t) or             # 등락률  -6.37%
        re.fullmatch(r"[+\-▲▼]?\d{1,4}\.\d{1,2}", t) or                  # 6.37 / 1.86 (등락·배율)
        re.fullmatch(r"[+\-]?\d{1,3}(?:,\d{3})+", t) or                  # 95,500 / 2,471,234 (가격·거래량)
        re.fullmatch(r"\d{1,4}(?:[.,]\d+)?[만억조천]", t) or             # 247만 / 3조
        re.fullmatch(r"[+\-]?\d{1,4}", t))                               # short bare int (<=4) — drop long junk


def _frame_numbers(fr) -> List[str]:
    """OCR numeric tokens on a frame, filtered to PLAUSIBLE real numbers (drops OCR junk tokens)."""
    return [t for t in (str(r.get("text", "")).strip() for r in (fr.get("ocr") or [])) if _plausible(t)]


# spoken-name variants (romanizations / common Whisper mis-transcriptions), keyed by normalized name
_TK_SUFFIX = re.compile(r"(홀딩스|지주회사|지주|우선주|[1-3]?우[bB]?|스팩\d*호?)$")
_NAME_ALIAS = {
    "포스코": ["포스콜딩스", "포스코홀딩스", "포스코홀딩", "posco"],
    "포스코홀딩스": ["포스콜딩스", "포스코", "posco"],
    "엘앤에프": ["lnf", "l&f", "엘엔에프", "에루엔에프"],
    "에코프로": ["ecopro"], "에코프로비엠": ["에코프로bm", "ecoprobm"],
    "엘지에너지솔루션": ["lg엔솔", "엘지엔솔", "lg에너지솔루션"],
    "lg에너지솔루션": ["lg엔솔", "엘지엔솔", "엘지에너지솔루션"],
    "삼성에스디아이": ["삼성sdi", "samsungsdi"], "에스케이하이닉스": ["sk하이닉스", "하이닉스"],
    "에스케이이노베이션": ["sk이노", "sk이노베이션"],
}


def _lev(a: str, b: str) -> int:
    n = len(b)
    prev = list(range(n + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if a[i - 1] == b[j - 1] else 1))
        prev = cur
    return prev[n]


def _near(pat: str, text: str, thr: int) -> bool:
    """True if `pat` is within edit distance `thr` of some ~same-length window of `text` (Whisper errors)."""
    L = len(pat)
    if L < 4 or len(text) < L - thr:
        return False
    for w in (L - 1, L, L + 1):
        if w < 3:
            continue
        for i in range(0, len(text) - w + 1):
            if _lev(pat, text[i:i + w]) <= thr:
                return True
    return False


def _name_variants(code) -> List[str]:
    nm = tickers.display_name(code) or ""
    vs = set()
    if nm:
        vs.add(nm)
        base = _TK_SUFFIX.sub("", nm)
        if base and base != nm:
            vs.add(base)
        for key in (_nl(nm), _nl(base)):
            vs.update(_NAME_ALIAS.get(key, []))
    return [v for v in vs if v]


def _named_in_audio(code, t_nl: str, t_digits: str) -> bool:
    """A screen ticker counts as NAMED only if its CODE or its NAME/aliases (incl. transcription
    variants: 포스코홀딩스/포스콜딩스, 엘앤에프/LNF/엘엔에프) appears in the transcript (fuzzy for Whisper errors)."""
    if str(code) in t_digits:
        return True
    variants = [_nl(v) for v in _name_variants(code)]
    if any(len(v) >= 2 and v in t_nl for v in variants):
        return True
    return any(_near(v, t_nl, 2 if len(v) >= 6 else 1) for v in variants)


def _ground_transcript(meta, segments) -> str:
    """ONLY the audio transcript — the Whisper-only summary sees nothing else (input isolation)."""
    L = [f"VIDEO: {meta.get('title','')}", "", "=== 음성 자막 [audio] (Whisper, 전체) ==="]
    for s in segments:
        L.append(f"[audio][{_mmss(s.get('start'))}] {s.get('text','')}")
    return "\n".join(L)


def _ground_full(meta, segments, frames_data, vlm_obs, call_data, fused) -> str:
    L = [f"VIDEO: {meta.get('title','')}", "", "=== 음성 자막 [audio] (Whisper) ==="]
    for s in segments:
        L.append(f"[audio][{_mmss(s.get('start'))}] {s.get('text','')}")
    L.append("")
    L.append("=== 추출된 콜 (ticker/방향/언급가격) — 출처는 audio+screen ===")
    for c in call_data.get("exante", []):
        L.append(f"[{c.get('mmss')}] {c.get('name')}({c.get('ticker')}) {c.get('direction')} "
                 f"price={c.get('stated_price')} :: \"{(c.get('quote') or '')[:140]}\"")
    if not call_data.get("exante"):
        L.append("(없음)")
    L.append("")
    # ON-SCREEN EXACT PRICE per instrument (OCR header C / last-price label = SOURCE OF TRUTH) — Task fix
    pfs = fused.get("price_fields") or []
    if pfs:
        L.append("=== 화면 실시간 가격 [screen] (OCR 헤더 C / 마지막가 라벨 = 숫자의 SOURCE OF TRUTH · 종목별) ===")
        for pf in pfs:
            L.append(f"[screen] {pf.get('symbol')} 현재가 ~{pf.get('price'):,.2f}"
                     + ("  (음성과 일치 AGREE)" if pf.get("audio_agrees") else "")
                     + "  ← 요약의 가격은 이 값을 쓸 것 (축 눈금 아님)")
        L.append("규칙: 위 종목별 현재가를 그 종목에 귀속해 보고. 가격 진술은 화면 헤더/마지막가와 일치해야 함.")
    L.append("")
    L.append("=== 화면 OCR 숫자 [screen] (PaddleOCR = 숫자의 출처) ===")
    for fr in frames_data:
        ns = _frame_numbers(fr)[:14]
        if ns:
            ap = (fr.get("axis_price") or {}).get("text")
            L.append(f"[screen][{_mmss(fr.get('t'))}] " + ", ".join(ns) + (f" (마지막가 라벨:{ap})" if ap else ""))
    axis = _axis_levels_all(frames_data)
    if axis:
        L.append("[screen] 차트 축 눈금/스케일 (OCR — 균등 간격 눈금): " + ", ".join(t for t, _ in axis[:10])
                 + "  ← 이것은 축 SCALE(눈금)이며 '주요 가격 레벨'이 아님 — 지지/저항으로 쓰지 말 것")
    L.append("")
    L.append("=== 차트 구조 [chart] (Qwen3-VL, 시각 해석 — 숫자 아님 · 미검증) ===")
    for o in vlm_obs or []:
        d = _vlm_desc(o.get("parsed", {}))
        if d:
            L.append(f"[chart][{_mmss(o.get('t'))}] {d}")
    # near-term speaker feature vs overall multi-month structure — reported SEPARATELY (Task 1.1), never blended
    af = fused.get("audio_features", {})
    nd, od = fused.get("near_term_direction"), fused.get("overall_direction")
    if af.get("features") or nd or od:
        L.append("")
        L.append("=== 화자의 단기(near-term) 강조 vs 차트 전체(multi-month) 구조 — 분리 보고(블렌딩 금지) ===")
        if af.get("features"):
            L.append("[audio] 화자 단기 강조 방향=" + str(nd) + " · 근거: "
                     + "; ".join(f"{f['feature']}[{f['mmss']}]" for f in af["features"][:8]))
        if od:
            L.append("[chart] 차트 전체 구조 방향=" + str(od))
        ac = fused.get("audio_chart") or {}
        if ac.get("tag") == "CONFLICT":
            L.append(f"[CONFLICT] 화자 방향({ac.get('basis')}: {ac.get('speaker_dir')}) ↔ 차트 전체({ac.get('overall_dir')}) "
                     "반대 — 요약에 둘을 분리해 명시(timeframe split / contrarian)")
        elif ac.get("tag") == "AGREE":
            L.append(f"[AGREE] 화자 추천({ac.get('basis')}: {ac.get('speaker_dir')})가 차트 전체 구조({ac.get('overall_dir')})와 일치")
    L.append("")
    L.append("=== 오디오↔화면 융합 태그 수 === " + json.dumps(fused.get("summary", {}), ensure_ascii=False))
    return "\n".join(L)


def _sections(is_moneyup: bool = True) -> str:
    """The 4-section summary template. Section 3 frames as a 머니업 플레이 for 머니업 videos, else a NEUTRAL
    'technical approach' for other creators (Task 1.4 — no forced 머니업 framing on non-머니업 channels)."""
    s3 = ("## 3. 사용한 머니업 플레이 (PLAYBOOK 매핑; 명확히 없으면 정확히 '명확히 사용된 전략 없음')\n" if is_moneyup
          else "## 3. 사용한 기술적 접근 (technical approach — 차트 패턴/지표/구조 기반; 명확히 없으면 '명확한 전략 없음')\n")
    return ("## 1. 핵심 발언 (SAID — 화자의 단기 vs 전체 구조가 다르면 분리해 명시)\n"
            "## 2. 추천/콜 (RECOMMENDED — 매수/매도/회피 + 가격대·목표가)\n" + s3 + "## 4. 핵심 인사이트\n")

_WONLY_SYS = (
    "당신은 한국 주식 유튜브 클립의 '음성(자막) 전용' 요약가입니다. 아래 '자막 텍스트'만 근거로 사용하세요. "
    "화면(OCR) 숫자나 차트(VLM) 패턴은 '제공되지 않았습니다' — 그것을 언급하거나 창작하지 마세요. 모든 항목 끝에 "
    "[audio] 와 [mm:ss] 를 표기. 자막에 근거가 없으면 해당 항목에 '정보 없음'이라고 적으세요. 절대 지어내지 말 것. "
    "한국어, 간결한 마크다운.")

_FULL_SYS = (
    "당신은 멀티모달 '근거 기반' 요약가입니다. 근거 출처는 세 가지: 음성 자막[audio], 화면 OCR 숫자/텍스트[screen], "
    "차트 관찰[chart]. 각 항목 끝에 해당 출처 태그([audio]/[screen]/[chart])와 [mm:ss]를 반드시 표기. 모든 숫자(가격/"
    "등락률/거래량)의 출처는 [screen](OCR)입니다. 제공된 데이터에 실제로 존재하는 것만 사용. 절대 지어내지 말 것. "
    "한국어, 간결한 마크다운.")


def _verify_drop(md: str, source_norm: str) -> Tuple[str, int]:
    """Drop any line whose VERBATIM quote (in quotes) does not appear in its source. Returns (md, dropped)."""
    kept, dropped = [], 0
    for line in (md or "").splitlines():
        bad = False
        for q in _QUOTE_RE.findall(line):
            qn_ = _norm(q)
            if len(qn_) >= 4 and qn_ not in source_norm:
                bad = True
                break
        if bad:
            dropped += 1
        else:
            kept.append(line)
    out = "\n".join(kept)
    if dropped:
        out += f"\n\n> (무결성 검증: 출처에서 확인되지 않은 인용 {dropped}건 제거됨)"
    return out, dropped


_TRANSLATE_SYS = (
    "You are a faithful translator for a GROUNDED Korean stock-video digest. Translate the Korean markdown "
    "to natural English, rendering EXACTLY the same points — add NO new facts, numbers, tickers or claims; "
    "drop or invent nothing; no interpretation. KEEP every source tag [audio]/[screen]/[chart] and every "
    "[mm:ss] citation verbatim and in place. For any quoted Korean phrase, KEEP the original Korean quote "
    "and add a short English gloss in parentheses right after, so it stays verifiable. Keep the markdown "
    "structure (headings/bullets). Output English markdown only.")


def _translate_to_en(ko_md: str) -> str:
    """Faithful EN rendering of the verified KO summary — same points, tags + [mm:ss] kept, KO quotes kept
    with an English gloss. A translation cannot introduce new facts, so grounding/verification carry over."""
    if not (ko_md or "").strip():
        return ""
    return _gen("Translate the following to English — same points only, keep all [audio]/[screen]/[chart] "
                "tags and [mm:ss] citations verbatim, and keep each Korean quote followed by a short English "
                "gloss in parentheses:\n\n" + ko_md, _TRANSLATE_SYS, temperature=0.0)


def build_summaries(meta, segments, frames_data, vlm_obs, call_data, fused):
    """Return (wonly_ko, wonly_en, full_ko, full_en). Whisper-only sees ONLY the transcript (input
    isolation). EN is a FAITHFUL translation of the verified KO — same points, no new facts/drift."""
    if not _gemini_key():
        return "", "", "", ""
    is_mu = bool(meta.get("is_moneyup", True))               # default True (the 508 corpus is 머니업)
    sections = _sections(is_mu)
    wonly = _gen("아래 자막만 근거로 다음 4개 섹션의 마크다운 요약을 작성하라. 각 항목 끝에 [audio]와 [mm:ss]를 표기하고, "
                 "근거가 된 자막의 짧은 원문을 큰따옴표(\"…\")로 1개 그대로 인용하라(검증용, 절대 변형 금지). "
                 "화자의 단기(near-term) 기술적 견해와 전체 추세가 다르면 분리해서 적어라.\n\n"
                 + sections + "\n=== 자막 시작 ===\n" + _ground_transcript(meta, segments) + "\n=== 자막 끝 ===\n",
                 _WONLY_SYS, temperature=0.2)
    wonly, _ = _verify_drop(wonly, _norm(_transcript_text(segments)))
    s3_rule = ("섹션 3은 아래 머니업 플레이북 목록에 매핑하라.\n\n" + "머니업 플레이북 전략 목록: "
               + " / ".join(_playbook_strategies()) if is_mu else
               "섹션 3은 차트 패턴/지표 기반의 일반 '기술적 접근'으로 기술하고, 머니업 플레이북에 억지로 매핑하지 말 것.")
    full = _gen("아래 데이터(자막+화면+차트)만 근거로 다음 4개 섹션의 마크다운 요약을 작성하라. 각 항목 끝에 출처태그"
                "([audio]/[screen]/[chart])와 [mm:ss]를 표기. [audio] 근거 항목은 자막 원문을 큰따옴표로 짧게 1개 "
                "그대로 인용(검증용, 변형 금지); 숫자는 [screen] OCR 값을 그대로 쓰라. 화자가 강조한 단기(near-term) 기술적 "
                "견해와 차트 전체(multi-month) 구조가 다르면 둘을 분리해 명시하라(블렌딩 금지; CONFLICT면 그렇게 표기). "
                + s3_rule + "\n\n"
                + sections + "\n=== 데이터 시작 ===\n" + _ground_full(meta, segments, frames_data, vlm_obs, call_data, fused)
                + "\n=== 데이터 끝 ===\n", _FULL_SYS, temperature=0.2)
    full, _ = _verify_drop(full, _norm(_transcript_text(segments) + " " + _ocr_text(frames_data)
                                       + " " + _vlm_text(vlm_obs)))
    return wonly, _translate_to_en(wonly), full, _translate_to_en(full)


# --------------------------------------------------------------------------- #
# COMPUTED delta (no LLM) — "what the visual reading added"
# --------------------------------------------------------------------------- #
_FIELD_KO = {"price": "현재가", "change_pct": "등락률", "change_abs": "전일대비", "volume": "거래량",
             "value": "거래대금", "open": "시가", "high": "고가", "low": "저가"}
_FIELD_EN = {"price": "current price", "change_pct": "change %", "change_abs": "day change",
             "volume": "volume", "value": "value traded", "open": "open", "high": "high", "low": "low"}
_MAG_FIELDS = {"price", "change_abs", "volume", "value", "open", "high", "low"}   # magnitude (vs ratio)


def _ndigits(x) -> int:
    """Number of digits in round(|x|)."""
    try:
        return len(str(int(round(abs(float(x))))))
    except Exception:
        return 0


def _focus_anchor_price(primary_numbers) -> Tuple[Optional[float], Optional[float]]:
    """(anchor_price, stated_price) for the focus ticker. anchor = stated price if known else the
    largest plausible OCR price (현재가/시가/고가/저가). stated_price = 현재가 stated value or None."""
    stated = None
    for pn in primary_numbers:
        if pn.get("field") == "price" and pn.get("stated_value") is not None:
            try:
                stated = abs(float(pn["stated_value"]))
                break
            except Exception:
                pass
    cands = []
    for pn in primary_numbers:
        if pn.get("field") in ("price", "open", "high", "low") and pn.get("ocr_value") is not None:
            try:
                cands.append(abs(float(pn["ocr_value"])))
            except Exception:
                pass
    anchor = stated if stated is not None else (max(cands) if cands else None)
    return anchor, stated


def _focus_number_plausible(pn, anchor, stated_price, pct) -> bool:
    """Field-aware sanity check on ONE focus-ticker VIDEO-ONLY number → True to KEEP. Drops OCR
    misreads (e.g. 전일대비 '2' for a 95,500원 stock) without ever dropping a plausible real value."""
    field = pn.get("field") or ""
    try:
        v = abs(float(pn.get("ocr_value")))
    except Exception:
        return False
    # Ratio fields (등락률 / any %/decimal): no magnitude guard — drop only the clearly-impossible.
    if field == "change_pct" or field.endswith("pct") or field == "ratio":
        return v <= 100.0
    # Magnitude fields: must not have far fewer digits than the anchor price.
    if anchor is not None and _ndigits(v) < _ndigits(anchor) - 1:
        return False
    # change_abs extra check when the stated price AND a % both exist: |Δ| ≈ price·|%|/100.
    if field == "change_abs" and stated_price is not None and pct is not None:
        expected = abs(stated_price) * abs(pct) / 100.0
        if expected > 0 and not (0.2 * expected <= v <= 5.0 * expected):
            return False
    return True


def compute_delta(segments, frames_data, vlm_obs, fused, primary=None) -> dict:
    transcript = _transcript_text(segments)
    t_digits = re.sub(r"[^\d]", "", transcript)
    t_nl = _nl(transcript)
    nums, seen = [], set()
    for fr in frames_data:
        for tok in _frame_numbers(fr):                  # plausibility-filtered: real numbers only, no junk
            d = re.sub(r"[^\d]", "", tok)
            if len(d) >= 3 and d not in t_digits and d not in seen:
                seen.add(d)
                nums.append({"num": tok, "mmss": _mmss(fr.get("t"))})
    codes = (fused.get("ocr_aggregate", {}) or {}).get("codes_seen", {}) or {}
    tickers_not_named = [{"code": code, "name": tickers.display_name(code) or ""}
                         for code in codes if not _named_in_audio(code, t_nl, t_digits)]
    conflicts = int((fused.get("summary", {}) or {}).get("CONFLICT", 0))
    vlm_trends = [{"mmss": _mmss(o.get("t")),
                   "pattern": (o.get("parsed", {}) or {}).get("trend_structure") or (o.get("parsed", {}) or {}).get("chart_pattern"),
                   "stance": (o.get("parsed", {}) or {}).get("stance")} for o in (vlm_obs or [])]
    # chart STRUCTURE the speaker SHOWS but may not say (VLM visual interpretation, per frame) — [chart]
    chart_structure = [{"mmss": _mmss(o.get("t")), "desc": _vlm_desc(o.get("parsed", {}))}
                       for o in (vlm_obs or []) if _vlm_desc(o.get("parsed", {}))]
    # chart PRICE-AXIS levels the screen shows (OCR'd) that the speaker doesn't say — grounded [screen]
    axis_levels = _axis_levels_all(frames_data)
    axis_not_spoken = [{"text": t, "value": v} for t, v in axis_levels
                       if re.sub(r"[^\d]", "", str(t)) and re.sub(r"[^\d]", "", str(t)) not in t_digits]
    # FOCUS: the primary ticker's OWN screen numbers (price/levels) that were NOT spoken — fuse tags
    # these VIDEO-ONLY (OCR'd, not stated). Apply a field-aware plausibility guard so an OCR misread
    # (e.g. 전일대비 '2' for a 95,500원 stock) never headlines. Separate from the watchlist bulk.
    pnums = fused.get("primary_numbers") or []
    anchor, stated_price = _focus_anchor_price(pnums)
    pct = None
    for pn in pnums:
        if pn.get("field") == "change_pct":
            pv = pn.get("stated_value") if pn.get("stated_value") is not None else pn.get("ocr_value")
            if pv is not None:
                try:
                    pct = abs(float(pv))
                except Exception:
                    pass
            break
    key, key_dropped = [], 0
    for pn in pnums:
        if pn.get("tag") != "VIDEO-ONLY" or pn.get("ocr_value") is None:
            continue
        if _focus_number_plausible(pn, anchor, stated_price, pct):
            key.append({"field": pn.get("field"), "label": _FIELD_KO.get(pn.get("field"), pn.get("field")),
                        "value": pn.get("ocr_value"), "mmss": _mmss(pn.get("t"))})
        else:
            key_dropped += 1
    # PEERS: the 관심종목 watchlist (code+name+price), aggregated per code across frames — the screen's
    # real addition. Exclude the focus ticker and any peer he NAMED in audio (reuse _named_in_audio).
    peer_px, peer_first = {}, {}
    for fr in frames_data:
        try:
            wrows = ocr_parse.watchlist_rows(fr)
        except Exception:
            wrows = []
        for row in wrows:
            code, price = row.get("code"), row.get("price")
            if not code or price is None or (primary and str(code) == str(primary)):
                continue
            try:
                pr = int(round(float(price)))
            except Exception:
                continue
            if pr < 50:                                    # implausible KRW price (e.g. 0) → OCR misread, skip
                continue
            peer_px.setdefault(str(code), Counter())[pr] += 1
            peer_first.setdefault(str(code), _mmss(fr.get("t")))
    peers = []
    for code, cnt in peer_px.items():
        if _named_in_audio(code, t_nl, t_digits):              # he named it → not "added by the screen"
            continue
        price = cnt.most_common(1)[0][0]
        peers.append({"code": code, "name": tickers.display_name(code) or code,
                      "price": price, "mmss": peer_first.get(code), "seen": cnt.total()})
    peers.sort(key=lambda p: (-p["seen"], p["code"]))          # most-persistently-shown peers first
    return {"key_numbers_not_spoken": key, "key_numbers_dropped": key_dropped,
            "peers_unspoken": peers, "peers_unspoken_count": len(peers),
            "primary": primary, "primary_name": (tickers.display_name(primary) if primary else None),
            "ocr_numbers_not_spoken": nums, "tickers_on_screen_not_named": tickers_not_named,
            "conflicts": conflicts, "vlm_trends": vlm_trends, "chart_structure": chart_structure,
            "axis_levels": [t for t, _ in axis_levels], "axis_levels_not_spoken": axis_not_spoken,
            "price_fields": fused.get("price_fields", []),
            "n_transcript": len(segments),
            "n_ocr_tokens": sum(len(fr.get("ocr") or []) for fr in frames_data),
            "n_frames": len(frames_data), "n_vlm": len(vlm_obs or [])}


# --------------------------------------------------------------------------- #
# writers
# --------------------------------------------------------------------------- #
_RAW_NOISE = re.compile(r"^\d{1,4}\.\d$")     # 999.9-style one-decimal OCR-grid artifact (axis garbage)


def _raw_token_ok(text) -> bool:
    """Plausibility filter for the RAW token dump (Task 1.3): keep all text + plausible numbers, DROP the
    OCR-grid garbage (999.9-style one-decimal runs, bare 0/00/000 tick artifacts)."""
    t = str(text or "").strip()
    if not t:
        return False
    if _RAW_NOISE.match(t):
        return False
    if re.fullmatch(r"\d{1,3}", t) and int(t) < 100:
        return False
    return True


def write_raw_full(jd: Path, meta, segments, frames_data, vlm_obs):
    n_dropped = 0
    fjson = []
    for fr in frames_data:
        kept = [r for r in (fr.get("ocr") or []) if _raw_token_ok(r.get("text"))]
        n_dropped += len(fr.get("ocr") or []) - len(kept)
        fjson.append({"frame": fr.get("frame"), "t": fr.get("t"), "w": fr.get("w"), "h": fr.get("h"),
                      "axis_price": fr.get("axis_price"), "axis_levels": fr.get("axis_levels"),
                      "chart": fr.get("chart"), "_noise_tokens_dropped": len(fr.get("ocr") or []) - len(kept),
                      "tokens": [{"text": r.get("text"), "score": r.get("score"), "bbox": r.get("bbox")} for r in kept]})
    raw = {
        "video_id": meta.get("video_id"), "title": meta.get("title"), "url": meta.get("url"),
        "noise_tokens_dropped": n_dropped,                   # OCR-grid garbage filtered (Task 1.3)
        "transcript": [{"start": s.get("start"), "end": s.get("end"), "mmss": _mmss(s.get("start")),
                        "text": s.get("text")} for s in segments],
        "frames_ocr": fjson,
        "vlm": [{"t": o.get("t"), "frame": o.get("frame"), "parsed": o.get("parsed")} for o in (vlm_obs or [])],
    }
    (jd / "raw_full.json").write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    doc = _new_doc()
    _h(doc, f"Raw extraction (full) — {meta.get('title','')}", 0)
    _p(doc, f"연구/모의용 · {meta.get('url','') or ''} · 요약 없는 전체 추출 (deterministic) · "
            f"OCR 노이즈 토큰 {n_dropped}개 필터(999.9 등)", italic=True)
    _h(doc, f"음성 자막 [audio] — Whisper ({len(segments)} segments)", 1)
    for s in segments:
        _p(doc, f"[audio][{_mmss(s.get('start'))}] {s.get('text','')}", bullet=True)
    _h(doc, f"화면 OCR [screen] — PaddleOCR (노이즈 제외 토큰/프레임 · {len(frames_data)} frames)", 1)
    for fr in frames_data:
        rows = [r for r in (fr.get("ocr") or []) if _raw_token_ok(r.get("text"))]
        _p(doc, f"[screen] frame @ {_mmss(fr.get('t'))}  ({fr.get('frame')}) — {len(rows)} tokens", bold=True)
        toks = " · ".join(f"{r.get('text')}({round((r.get('score') or 0)*100)}%)" for r in rows)
        _p(doc, toks or "(no tokens)")
    axis = _axis_levels_all(frames_data)
    if axis:
        _h(doc, f"차트 축 눈금/스케일 [screen] — OCR ({len(axis)} ticks · 가격 레벨 아님)", 1)
        _p(doc, ", ".join(t for t, _ in axis), bullet=True)
    _h(doc, f"차트 구조 [chart] — Qwen3-VL ({len(vlm_obs or [])}) · 시각 해석 · 미검증", 1)
    for o in (vlm_obs or []):
        d = _vlm_desc(o.get("parsed", {}))
        _p(doc, f"[chart][{_mmss(o.get('t'))}] {d or '(아무것도 읽지 못함)'}", bullet=True)
    doc.save(str(jd / "raw_full.docx"))


# --------------------------------------------------------------------------- #
# provenance — COMPUTED mapping of each summary citation → its source item (Task 2). Only items the
# summary ACTUALLY cites are listed; every extracted-but-unused item is excluded. No LLM here.
# --------------------------------------------------------------------------- #
_TS_RE = re.compile(r"\[(\d{1,2}):(\d{2})\]")
_TAG_RE = re.compile(r"\[(audio|screen|chart)\]", re.I)


def _parse_cited_points(md: str):
    """Each non-heading summary line that carries a source tag → its tags + cited [mm:ss] + the claim text."""
    pts = []
    for line in (md or "").splitlines():
        l = line.strip()
        if not l or l.lstrip().startswith("#"):
            continue
        tags = [t.lower() for t in _TAG_RE.findall(l)]
        if not tags:
            continue
        secs = [int(a) * 60 + int(b) for a, b in _TS_RE.findall(l)]
        text = _TAG_RE.sub("", _TS_RE.sub("", l)).strip(" -·•*\t>")
        pts.append({"text": text, "tags": set(tags), "secs": secs})
    return pts


def _nearest(items, sec, tkey):
    if sec is None or not items:
        return None
    return min(items, key=lambda x: abs(float(x.get(tkey, 0) or 0) - sec))


def _numkey(s: str) -> str:
    """Normalize a number for matching: strip commas + trailing .00/.0 (27,600.00 == 27,600 == 27600)."""
    s = re.sub(r"\.0+$", "", str(s).replace(",", ""))
    return re.sub(r"[^\d]", "", s)


def build_provenance(md, segments, frames_data, vlm_obs) -> dict:
    """Map every [audio]/[screen]/[chart]+[mm:ss] citation in the summary to the SOURCE item it draws
    from (nearest transcript segment / OCR frame / VLM observation). Returns only the cited items."""
    pts = _parse_cited_points(md)
    rows = {"audio": {}, "screen": {}, "chart": {}}
    cited = {"audio": 0, "screen": 0, "chart": 0}
    for pt in pts:
        for tag in pt["tags"]:
            for sec in (pt["secs"] or [None]):
                cited[tag] += 1
                mmss = _mmss(sec) if sec is not None else "—"
                if tag == "audio":
                    seg = _nearest(segments, sec, "start")
                    src = (seg.get("text") if seg else "(no matching transcript segment)")
                elif tag == "screen":
                    # match the cited NUMBER to its real OCR source: a chart price-axis level (aggregated)
                    # if it is one, else the nearest frame's OCR numbers — so the row shows the grounding.
                    pt_keys = {_numkey(n) for n in re.findall(r"\d[\d,]*\.?\d*", pt["text"])}
                    matched_ax = [t for t, _ in _axis_levels_all(frames_data) if _numkey(t) in pt_keys]
                    if matched_ax:
                        src = "chart price-axis level (OCR): " + ", ".join(list(dict.fromkeys(matched_ax))[:8])
                    else:
                        fr = _nearest(frames_data, sec, "t")
                        nums = _frame_numbers(fr)[:10] if fr else []
                        src = ("frame OCR: " + ", ".join(nums)) if nums else "(no OCR numbers at this frame)"
                else:  # chart
                    ob = _nearest(vlm_obs, sec, "t")
                    src = (_vlm_desc(ob.get("parsed", {})) if ob else "") or "(no VLM observation at this frame)"
                b = rows[tag].setdefault(src[:72], {"source": src, "cites": set(), "points": set()})
                b["cites"].add(mmss)
                b["points"].add(pt["text"][:140])
    return {"rows": {k: sorted(v.values(), key=lambda r: min(r["cites"])) for k, v in rows.items()},
            "cited": cited,
            "extracted": {"audio": len(segments), "screen": sum(1 for f in frames_data if _frame_numbers(f)),
                          "chart": len(vlm_obs or [])}}


def _write_provenance(doc, prov, en=False):
    _h(doc, ("Provenance — sources ACTUALLY cited (computed; unused items excluded)" if en
             else "출처 표 — 요약이 실제로 인용한 항목만 (계산식 매핑; 미사용 항목 제외)"), 1)
    specs = [("audio", "Whisper / audio  [audio]", "음성 Whisper  [audio]"),
             ("screen", "OCR / screen  [screen]", "화면 OCR  [screen]"),
             ("chart", "VLM / chart  [chart]", "차트 VLM  [chart]")]
    hdr = ["cited [mm:ss]", "source item (only if used)", "summary point it supports"] if en \
        else ["인용된 [mm:ss]", "출처 항목 (사용된 것만)", "지지하는 요약 포인트"]
    for k, en_t, ko_t in specs:
        rws = prov["rows"][k]
        ext, used = prov["extracted"][k], len(rws)
        _p(doc, (en_t if en else ko_t)
                + (f"  —  {used} source item(s) cited / {ext} extracted ({max(0, ext - used)} unused, excluded)"), bold=True)
        if not rws:
            _p(doc, "(none cited)" if en else "(인용된 항목 없음)")
            continue
        t = doc.add_table(rows=1, cols=3)
        try:
            t.style = "Table Grid"
        except Exception:
            pass
        for c, txt in zip(t.rows[0].cells, hdr):
            r = c.paragraphs[0].add_run(txt); r.bold = True; _kfont_run(r)
        for row in rws:
            cells = t.add_row().cells
            vals = [", ".join(sorted(row["cites"])), row["source"], " / ".join(sorted(row["points"]))[:240]]
            for c, txt in zip(cells, vals):
                rr = c.paragraphs[0].add_run(str(txt)); _kfont_run(rr)


def write_summary_full(jd: Path, meta, md, segments=None, frames_data=None, vlm_obs=None, en=False):
    doc = _new_doc()
    if en:
        _h(doc, f"Summary (Full: OCR+Whisper+VLM) — {meta.get('title','')}", 0)
        _p(doc, f"research / mock only · grounded · Gemini 3.1 Pro · source tags [audio]/[screen]/[chart] · "
                f"faithful English of the Korean (same points) · {meta.get('url','') or ''}", italic=True)
    else:
        _h(doc, f"요약 (Full: OCR+Whisper+VLM) — {meta.get('title','')}", 0)
        _p(doc, f"연구/모의용 · 근거 기반 · 생성: Gemini 3.1 Pro · 출처태그 [audio]/[screen]/[chart] · "
                f"{meta.get('url','') or ''}", italic=True)
    _md_to_doc(doc, md)
    prov = build_provenance(md, segments or [], frames_data or [], vlm_obs or [])
    _write_provenance(doc, prov, en=en)
    doc.save(str(jd / ("summary_full_en.docx" if en else "summary_full.docx")))
    return prov


def _render_delta_block(doc, delta, en=False):
    """Render section-(d) in KO or EN. ALL values are COMPUTED (no LLM); the EN block carries the SAME
    numbers / codes / K as the KO block. Order: focus-ticker line → LEAD (unspoken peers) → support."""
    key = delta.get("key_numbers_not_spoken", [])
    dropped = delta.get("key_numbers_dropped", 0)
    peers = delta.get("peers_unspoken", [])
    K = delta.get("peers_unspoken_count", len(peers))
    pname, pcode = delta.get("primary_name") or "?", delta.get("primary") or "?"
    nn = len(delta.get("ocr_numbers_not_spoken", []))
    conflicts = delta.get("conflicts", 0)
    vlm = delta.get("vlm_trends", [])
    # 1) focus-ticker line — its own screen-only key numbers that PASS the plausibility guard
    if key:
        _p(doc, ("Focus ticker %s(%s) — screen-only key numbers (not said aloud):" % (pname, pcode)) if en
                else ("포커스 종목 %s(%s) — 화면 단독 핵심 수치(음성 미언급):" % (pname, pcode)), bold=True)
        for k in key:
            lbl = _FIELD_EN.get(k.get("field"), k.get("field")) if en else k.get("label")
            _p(doc, f"   · {lbl} = {k['value']}  [{k['mmss']}]  [screen]", bullet=True)
    elif en:
        _p(doc, f"Focus ticker {pname}({pcode}): current price / change% / volume were stated in audio "
                f"(or CONFLICT) → no screen-only key number ({dropped} unrealistic day-change OCR value(s) "
                f"excluded).")
    else:
        _p(doc, f"포커스 종목 {pname}({pcode}): 현재가·등락률·거래량은 음성으로 직접 언급(또는 CONFLICT) "
                f"→ 화면 단독 핵심 수치 없음 (전일대비 OCR 비현실값 {dropped}개 제외).")
    # 2) LEAD — the screen's real addition: peers he never named (computed from the 관심종목 watchlist)
    _p(doc, (f"▶ What the screen added — peers he never named: {K}") if en
            else (f"▶ 화면이 더한 핵심 — 그가 호명하지 않은 관심종목(피어) {K}개:"), bold=True)
    for p in peers[:8]:
        _p(doc, f"   · {p['name']}({p['code']})  {p['price']:,}", bullet=True)
    if K > 8:
        _p(doc, ("   …(total %d)" % K) if en else ("   …(총 %d개)" % K))
    if K == 0:
        _p(doc, "(no unnamed peers on the watchlist)" if en else "(관심종목에서 미호명 피어 없음)")
    # 2b) chart price-AXIS levels + chart STRUCTURE he shows but doesn't say (the core visual add)
    axisns = delta.get("axis_levels_not_spoken", [])
    struct = delta.get("chart_structure", [])
    pfs = delta.get("price_fields", [])
    if pfs:
        lead = "▶ on-screen PRICE (OCR header/last-label — source of truth), by instrument:" if en \
            else "▶ 화면 실시간 가격(OCR 헤더/마지막가 — 숫자의 SOURCE OF TRUTH) · 종목별:"
        _p(doc, lead + " " + " · ".join(f"{p.get('symbol')} ~{p.get('price'):,.2f}"
                                        + ("✓audio" if p.get("audio_agrees") else "") for p in pfs), bold=True)
    if en:
        _p(doc, f"▶ chart AXIS SCALE / gridline ticks (OCR) — NOT price levels: {len(axisns)}", bold=True)
        if axisns:
            _p(doc, "   " + ", ".join(str(a["text"]) for a in axisns[:10]) + "   [screen · scale, not levels]")
        _p(doc, f"▶ chart STRUCTURE the speaker showed (VLM visual interpretation / unverified): {len(struct)}", bold=True)
        for c in struct[:5]:
            _p(doc, f"   [{c['mmss']}] {c['desc']}   [chart]", bullet=True)
    else:
        _p(doc, f"▶ 화면 축 눈금/스케일(OCR) — 가격 레벨 아님: {len(axisns)}개", bold=True)
        if axisns:
            _p(doc, "   " + ", ".join(str(a["text"]) for a in axisns[:10]) + "   [screen]")
        _p(doc, f"▶ 화자가 보여준 차트 구조(VLM 시각 해석 · 미검증): {len(struct)}건", bold=True)
        for c in struct[:5]:
            _p(doc, f"   [{c['mmss']}] {c['desc']}   [chart]", bullet=True)
    # 3) supporting detail — kept below the headline
    if en:
        _p(doc, f"• supporting — raw screen numbers not spoken (watchlist + live-tick superset; OCR noise "
                f"excluded): {nn}")
        _p(doc, f"• audio↔screen CONFLICT tags: {conflicts} · chart trends (VLM): {len(vlm)}")
        _p(doc, f"(totals: transcript {delta.get('n_transcript')} segs · OCR {delta.get('n_ocr_tokens')} "
                f"tokens / {delta.get('n_frames')} frames · VLM {delta.get('n_vlm')})", italic=True)
    else:
        _p(doc, f"• 참고 — 음성에 없던 화면 숫자 원시 집계(관심종목·실시간 호가 상위집합; OCR 노이즈 제외): {nn}개")
        _p(doc, f"• 오디오↔화면 CONFLICT(상충) 태그: {conflicts}개 · 차트 추세(VLM): {len(vlm)}건")
        _p(doc, f"(집계: 자막 {delta.get('n_transcript')}세그먼트 · OCR {delta.get('n_ocr_tokens')}토큰/"
                f"{delta.get('n_frames')}프레임 · VLM {delta.get('n_vlm')}관찰)", italic=True)


def write_comparison(jd: Path, meta, segments, frames_data, vlm_obs, wonly_ko, full_ko, wonly_en, full_en, delta):
    doc = _new_doc()
    _h(doc, f"비교: Whisper-only vs OCR+Whisper+VLM — {meta.get('title','')}", 0)
    _p(doc, f"연구/모의용 · {meta.get('url','') or ''}", italic=True)

    _h(doc, "(a) Whisper-ONLY 추출 — 음성 자막 단독 (verbatim)", 1)
    _p(doc, "오직 Whisper 자막. 화면/차트 데이터 일절 미포함.", italic=True)
    for s in segments:
        _p(doc, f"[audio][{_mmss(s.get('start'))}] {s.get('text','')}", bullet=True)

    _h(doc, "(b) Full 추출 — 자막 + 화면 OCR + 차트 VLM", 1)
    _p(doc, f"음성 [audio]: 위 (a)의 {len(segments)}개 세그먼트 전체. 아래는 시각 레이어(화면/차트):", italic=True)
    _p(doc, "화면 OCR 숫자 [screen]", bold=True)
    for fr in frames_data:
        ns = _frame_numbers(fr)
        if ns:
            ap = (fr.get("axis_price") or {}).get("text")
            _p(doc, f"[screen][{_mmss(fr.get('t'))}] " + ", ".join(ns[:18]) + (f"  (마지막가 라벨:{ap})" if ap else ""),
               bullet=True)
    pfs = delta.get("price_fields", []) if isinstance(delta, dict) else []
    if pfs:
        _p(doc, "화면 실시간 가격 [screen] (OCR 헤더/마지막가 — SOURCE OF TRUTH · 종목별)", bold=True)
        _p(doc, "   " + " · ".join(f"{p.get('symbol')} ~{p.get('price'):,.2f}"
                                   + ("✓음성일치" if p.get("audio_agrees") else "") for p in pfs), bullet=True)
    axis = _axis_levels_all(frames_data)
    if axis:
        _p(doc, "차트 축 눈금/스케일 [screen] (OCR — 균등 눈금 · 가격 레벨 아님)", bold=True)
        _p(doc, "   " + ", ".join(t for t, _ in axis[:12]), bullet=True)
    _p(doc, "차트 구조 [chart] (Qwen3-VL 시각 해석 · 미검증)", bold=True)
    for o in (vlm_obs or []):
        d = _vlm_desc(o.get("parsed", {}))
        if d:
            _p(doc, f"[chart][{_mmss(o.get('t'))}] {d}", bullet=True)

    _h(doc, "(c) 요약 비교 — Whisper-only vs Full (KO + EN)", 1)
    _na = "(요약 불가 — Gemini API 키 없음/응답 없음)"
    _na_en = "(summary unavailable — no Gemini key/response)"
    _p(doc, "한국어 (Korean)", bold=True)
    t1 = doc.add_table(rows=1, cols=2)
    try:
        t1.style = "Table Grid"
    except Exception:
        pass
    _cell_md(t1.rows[0].cells[0], wonly_ko or _na, "Whisper-only 요약 (음성만)")
    _cell_md(t1.rows[0].cells[1], full_ko or _na, "Full 요약 (OCR+Whisper+VLM)")
    _p(doc, "English — faithful rendering of the SAME points (same tags + [mm:ss]; Korean quotes kept)", bold=True)
    t2 = doc.add_table(rows=1, cols=2)
    try:
        t2.style = "Table Grid"
    except Exception:
        pass
    _cell_md(t2.rows[0].cells[0], wonly_en or _na_en, "Whisper-only summary (audio only)")
    _cell_md(t2.rows[0].cells[1], full_en or _na_en, "Full summary (OCR+Whisper+VLM)")

    _h(doc, "(d) 시각 판독이 추가한 것 — delta (데이터로 계산, LLM 아님)", 1)
    _p(doc, "한국어 (Korean)", bold=True)
    _render_delta_block(doc, delta, en=False)
    _p(doc, "English — same computed numbers / codes / K (no new facts)", bold=True)
    _render_delta_block(doc, delta, en=True)
    doc.save(str(jd / "comparison.docx"))


def finalize(jd: Path, meta, segments, frames_data, vlm_obs, call_data, fused, primary=None) -> dict:
    """Write raw_full(.docx/.json) + summary_full(.docx KO) + summary_full_en.docx + comparison.docx
    (KO+EN summaries, focus-ticker + watchlist delta). Returns the file map."""
    out = {"summary": None, "summary_en": None, "raw": None, "raw_json": None, "comparison": None}
    write_raw_full(jd, meta, segments, frames_data, vlm_obs)              # verbatim Korean source (not translated)
    out["raw"], out["raw_json"] = "raw_full.docx", "raw_full.json"
    delta = compute_delta(segments, frames_data, vlm_obs, fused, primary)
    wonly_ko, wonly_en, full_ko, full_en = build_summaries(meta, segments, frames_data, vlm_obs, call_data, fused)
    if full_ko and full_ko.strip():
        write_summary_full(jd, meta, full_ko, segments, frames_data, vlm_obs, en=False)
        out["summary"] = "summary_full.docx"
    if full_en and full_en.strip():
        write_summary_full(jd, meta, full_en, segments, frames_data, vlm_obs, en=True)
        out["summary_en"] = "summary_full_en.docx"
    write_comparison(jd, meta, segments, frames_data, vlm_obs, wonly_ko, full_ko, wonly_en, full_en, delta)
    out["comparison"] = "comparison.docx"
    return out
