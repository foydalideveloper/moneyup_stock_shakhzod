"""Assemble + render ONE grounded fused fact sheet per video (base env).

Writes ``<video_id>.json`` (machine) and ``<video_id>.md`` (human) under
``data/_moneyup_advisor/factsheets/``. Every number traces back to PaddleOCR; the VLM appears only
as flagged VIDEO-ONLY observations; ex-ante calls are separated from ex-post commentary.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional

from moneyup_advisor import config, ocr_parse, tickers

_METHODS = {
    "transcript": "faster-whisper large-v3 (cuda/float16)",
    "ocr": "PaddleOCR 3.7 / PP-OCRv5 (korean) — SOURCE OF TRUTH for numbers",
    "vlm": "Qwen3-VL-30B-A3B q4_K_M via Ollama — pattern/context only, never numbers",
    "geometry": "OpenCV candle geometry — descriptive trend, no prices",
}
_FIELD_KO = {"price": "현재가", "change_pct": "등락률", "change_abs": "전일대비",
             "volume": "거래량", "open": "시가", "high": "고가", "low": "저가"}


def _merge_watchlist(frames: List[dict]) -> List[dict]:
    by_code = defaultdict(lambda: {"prices": Counter(), "name": None, "score": 0.0})
    for fr in frames:
        for row in ocr_parse.watchlist_rows(fr):
            c = row["code"]
            if row.get("price"):
                by_code[c]["prices"][row["price"]] += 1
            by_code[c]["name"] = tickers.display_name(c)
            by_code[c]["score"] = max(by_code[c]["score"], row.get("name_match_score") or 0)
    out = []
    for c, d in by_code.items():
        price = d["prices"].most_common(1)[0][0] if d["prices"] else None
        out.append({"ticker": c, "name": d["name"], "ocr_price": price,
                    "name_match_score": round(d["score"], 3), "n_frames": sum(d["prices"].values())})
    return sorted(out, key=lambda r: -(r["ocr_price"] or 0))


def build(video: Dict, segments: List[dict], frames: List[dict], fused: Dict,
          calls: Dict, vlm_obs: List[dict], primary_code: Optional[str], degraded: bool = False) -> Dict:
    agg = fused["ocr_aggregate"]
    primary_numbers = [{**pn, "label_ko": _FIELD_KO.get(pn["field"], pn["field"])}
                       for pn in fused.get("primary_numbers", [])]
    sheet = {
        "schema": "moneyup_advisor.factsheet.v1",
        "video_id": video.get("video_id"), "title": video.get("title"),
        "channel": video.get("channel"), "channel_handle": video.get("channel_handle"),
        "url": video.get("url"),
        "publish_date": video.get("publish_date"), "publish_datetime": video.get("publish_datetime"),
        "duration_s": video.get("duration_s"),
        "extracted_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "methods": _METHODS,
        "rules": ["PaddleOCR is the single source of truth for every number; Qwen3-VL never overrides it.",
                  "Every fact is keyed on the 6-digit KRX ticker; the Korean name is a display label only."],
        "primary_ticker": primary_code,
        # Korean name when a Korean stock is the subject; else the visual reader's named subject
        # (e.g. a US/global index label) — never a forced Korean ticker.
        "primary_name": (tickers.display_name(primary_code) if primary_code else video.get("subject_label")),
        "n_transcript_segments": len(segments),
        "n_frames_ocr": len(frames),
        "degraded": bool(degraded),
        "degraded_reason": ("video stream present but 0 frames OCR'd after retry — re-extraction queued"
                            if degraded else None),
        "tickers_seen": {c: n for c, n in sorted(agg["codes_seen"].items(), key=lambda kv: -kv[1])},
        "primary_numbers": primary_numbers,
        "watchlist_ocr": _merge_watchlist(frames),
        "fusion_summary": fused["summary"],
        "price_fields": fused.get("price_fields", []),
        "timeline": fused["timeline"],
        "exante_calls": calls.get("exante", []),
        "expost_commentary": calls.get("expost", []),
        "vlm_observations": [{"t": o.get("t"), "frame": o.get("frame"),
                              "parsed": o.get("parsed"), "tag": "VIDEO-ONLY (unverified)"}
                             for o in (vlm_obs or [])],
        "transcript_excerpt": [{"mmss": _mmss(s["start"]), "text": s["text"]}
                               for s in segments[:8]],
    }
    return sheet


def _mmss(t):
    from tagent.news.youtube_source import mmss
    return mmss(t)


def _fmt(v):
    return f"{v:,}" if isinstance(v, int) else (f"{v}" if v is not None else "—")


def render_markdown(s: Dict) -> str:
    L = []
    L.append(f"# 머니업 Fact Sheet — {s.get('primary_name') or '?'} "
             f"({s.get('primary_ticker') or '?'})")
    L.append("")
    L.append(f"**Video:** [{s.get('title')}]({s.get('url')})  ")
    L.append(f"**Channel:** {s.get('channel')} ({s.get('channel_handle')})  ")
    L.append(f"**Published:** {s.get('publish_datetime') or s.get('publish_date') or '?'} · "
             f"**Duration:** {s.get('duration_s')}s · **Video id:** `{s.get('video_id')}`  ")
    L.append(f"**Extracted:** {s.get('extracted_at')} · "
             f"frames OCR'd: {s.get('n_frames_ocr')} · transcript segs: {s.get('n_transcript_segments')}")
    L.append("")
    L.append("> " + "  \n> ".join(s.get("rules", [])))
    L.append("")
    L.append(f"## {s.get('primary_name') or 'Primary'} numbers — OCR vs stated (audio)")
    L.append("PaddleOCR is the source of truth; the host also **states** the primary's numbers in the "
             "narration. These videos show a sector watchlist (peers) on the HTS, so the primary's own "
             "현재가/거래량 are usually AUDIO-grounded.")
    if s.get("primary_numbers"):
        L.append("| field | 한글 | OCR | stated (audio) | tag | when |")
        L.append("|---|---|--:|--:|---|---|")
        for d in s["primary_numbers"]:
            L.append(f"| {d['field']} | {d['label_ko']} | {_fmt(d.get('ocr_value'))} | "
                     f"{_fmt(d.get('stated_value'))} | **{d['tag']}** | {_mmss(d.get('t'))} |")
    else:
        L.append("_No primary numbers grounded (neither OCR-located nor clearly stated)._")
    L.append("")
    if s.get("watchlist_ocr"):
        L.append("## 관심종목 watchlist (OCR — best effort, keyed on 6-digit code)")
        L.append("| ticker | name | OCR 현재가 | name match |")
        L.append("|---|---|--:|--:|")
        for r in s["watchlist_ocr"][:20]:
            L.append(f"| {r['ticker']} | {r['name']} | {_fmt(r['ocr_price'])} | "
                     f"{r['name_match_score']} |")
        L.append("")
    L.append("## Ex-ante calls (testable — ticker / direction / price / when / published)")
    if s.get("exante_calls"):
        L.append("| ticker | name | direction | stated price | in-video | published | deeplink |")
        L.append("|---|---|---|--:|---|---|---|")
        for c in s["exante_calls"]:
            L.append(f"| {c['ticker']} | {c['name']} | **{c['direction']}** | "
                     f"{_fmt(c.get('stated_price'))} | {c['mmss']} | "
                     f"{c.get('publish_datetime') or c.get('publish_date')} | "
                     f"[jump]({c['deeplink']}) |")
        L.append("")
        for c in s["exante_calls"]:
            L.append(f"- **{c['name']} ({c['ticker']}) → {c['direction']}** @ {c['mmss']} "
                     f"(matched: {', '.join(c.get('matched', []))})  \n  > {c['quote'][:240]}")
    else:
        L.append("_No ex-ante calls detected._")
    L.append("")
    L.append("## Ex-post commentary (NOT a fresh call — past picks / realized returns)")
    if s.get("expost_commentary"):
        for c in s["expost_commentary"][:12]:
            L.append(f"- {c['name']} ({c['ticker']}) @ {c['mmss']} "
                     f"(markers: {', '.join(c.get('expost_markers', []))})  \n  > {c['quote'][:200]}")
    else:
        L.append("_None detected._")
    L.append("")
    L.append("## Fused timeline  (AGREE / AUDIO-ONLY / VIDEO-ONLY / CONFLICT)")
    L.append(f"**Counts:** " + " · ".join(f"{k}={v}" for k, v in s.get("fusion_summary", {}).items()))
    L.append("")
    L.append("| time | tag | ticker | kind | video (OCR/geom/VLM) | audio |")
    L.append("|---|---|---|---|---|---|")
    for e in s.get("timeline", []):
        L.append(f"| {e.get('mmss')} | **{e.get('tag')}** | {e.get('ticker') or '—'} | "
                 f"{e.get('kind')} | {(e.get('video') or '—')} | {(e.get('audio') or '—')} |")
    L.append("")
    L.append("## Qwen3-VL observations (VIDEO-ONLY — unverified, never promoted to facts)")
    for o in s.get("vlm_observations", []):
        p = o.get("parsed", {})
        L.append(f"- @{_mmss(o.get('t'))} `{o.get('frame')}` → "
                 f"{ {k: p.get(k) for k in ('chart_pattern','points_at','stance')} }")
    L.append("")
    L.append("## Transcript excerpt")
    for seg in s.get("transcript_excerpt", []):
        L.append(f"- [{seg['mmss']}] {seg['text']}")
    return "\n".join(L)


def save(sheet: Dict) -> Dict[str, str]:
    vid = sheet["video_id"]
    jpath = config.SHEET_DIR / f"{vid}.json"
    mpath = config.SHEET_DIR / f"{vid}.md"
    jpath.write_text(json.dumps(sheet, ensure_ascii=False, indent=2), encoding="utf-8")
    mpath.write_text(render_markdown(sheet), encoding="utf-8")
    return {"json": str(jpath), "md": str(mpath)}
