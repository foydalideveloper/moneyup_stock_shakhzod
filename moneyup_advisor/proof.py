"""Vision-only PROOF page — frame screenshots next to the exact data OCR'd from each (numbers +
confidence + timestamp) + the vision model's description of any on-screen drawing.

Used to prove the pipeline reads the SCREEN, not the audio: run on an audio-stripped clip, the
transcript is empty, and everything shown is read from pixels. Renders a self-contained HTML
(base64 images) for saving, or a live page (served images) for the :8077 dashboard.
"""
from __future__ import annotations

import base64
import html
import json
from typing import List, Tuple

from moneyup_advisor import config, ocr_parse
from tagent.news.youtube_source import mmss

HEADER = "Audio removed — transcript is empty. Everything below was read from the screen."


def _load(video_id: str):
    vis = json.loads((config.CACHE_DIR / f"{video_id}.vision.json").read_text(encoding="utf-8"))
    sheet = json.loads((config.SHEET_DIR / f"{video_id}.json").read_text(encoding="utf-8"))
    return vis, sheet


def select_frames(video_id: str, n: int = 8) -> Tuple[dict, dict, List[dict], dict]:
    vis, sheet = _load(video_id)
    frames = vis.get("frames", [])
    vlm = {o.get("frame"): o.get("parsed") for o in sheet.get("vlm_observations", [])}
    for f in frames:
        f["_nums"] = sum(1 for r in f.get("ocr", []) if ocr_parse.parse_num(r.get("text", "")))
        f["_vlm"] = vlm.get(f["frame"])
    chosen, seen = [], set()
    for f in [x for x in frames if x["_vlm"]] + sorted(frames, key=lambda x: -x["_nums"]):
        if f["frame"] in seen:
            continue
        seen.add(f["frame"])
        chosen.append(f)
        if len(chosen) >= n:
            break
    chosen.sort(key=lambda f: f.get("t", 0))
    return vis, sheet, chosen, vlm


def _numeric(f: dict, k: int = 14) -> List[Tuple[str, float]]:
    rows = [(r["text"], r["score"]) for r in f.get("ocr", []) if ocr_parse.parse_num(r.get("text", ""))]
    rows.sort(key=lambda x: -x[1])
    return rows[:k]


def _all_tokens(f: dict):
    """ALL OCR tokens on the frame split into (number_tokens, text_tokens) — nothing dropped."""
    nums, texts = [], []
    for r in f.get("ocr", []):
        t, s = r.get("text", ""), float(r.get("score", 0))
        (nums if ocr_parse.parse_num(t) else texts).append((t, s))
    nums.sort(key=lambda x: -x[1])
    texts.sort(key=lambda x: -x[1])
    return nums, texts


def _esc(x):
    return html.escape(str(x)) if x is not None else "—"


_CSS = """
body{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f6f8fa;color:#1f2328}
.bar{background:#0d1117;color:#fff;padding:16px 22px}.bar b{color:#58a6ff}
.proof{background:#fff3cd;border:1px solid #f0c36d;color:#7a5d00;padding:10px 16px;margin:14px 22px;border-radius:8px;font-weight:700}
.wrap{max-width:1180px;margin:0 auto;padding:8px 22px 40px}
.fr{display:flex;gap:16px;background:#fff;border:1px solid #d0d7de;border-radius:10px;padding:14px;margin:16px 0}
.fr img{width:560px;max-width:52%;border:1px solid #d0d7de;border-radius:6px}
.meta{flex:1;min-width:0}
table{border-collapse:collapse;width:100%;font-size:13px;margin:6px 0}
th,td{border:1px solid #d0d7de;padding:3px 7px;text-align:left}.num{text-align:right;font-variant-numeric:tabular-nums}
.k{color:#0969da;font-weight:700}.empty{color:#cf222e;font-weight:700}
small{color:#656d76}
details{margin:8px 0;border:1px solid #d0d7de;border-radius:6px;padding:6px 10px;background:#fafbfc}
summary{cursor:pointer;font-weight:700;color:#0969da}
.cols{display:flex;gap:14px;margin-top:6px}.cols>div{flex:1;min-width:0}
.cols table{font-size:12px}.cols td{padding:2px 6px}
.cnt{display:inline-block;background:#eaeef2;border-radius:10px;padding:0 8px;font-size:12px;margin-left:6px}
"""


def render_html(video_id: str, embed: bool = True, n: int = 8) -> str:
    vis, sheet, chosen, vlm = select_frames(video_id, n)
    fdir = config.video_frame_dir(video_id)
    stats = vis.get("stats", {})
    n_tx = sheet.get("n_transcript_segments", 0)

    def img_tag(fname):
        p = fdir / fname
        if embed and p.exists():
            b64 = base64.b64encode(p.read_bytes()).decode()
            return f"<img src='data:image/png;base64,{b64}'>"
        return f"<img src='/frame?id={_esc(video_id)}&f={_esc(fname)}'>"

    is_silent = not n_tx
    banner = ("🔇 " + HEADER) if is_silent else \
             "🔍 Full raw OCR — every token read from each frame is shown below (numbers + text); nothing dropped."
    tx_label = ("Transcript field (proof of no audio)" if is_silent else "Transcript field")
    B = [f"<!doctype html><html><head><meta charset='utf-8'><title>Vision proof — {_esc(video_id)}</title>",
         f"<style>{_CSS}</style></head><body>",
         f"<div class='bar'><b>머니업</b> AI Advisor — VISION RAW-OCR PROOF · {_esc(video_id)}</div>",
         f"<div class='proof'>{banner}</div>",
         "<div class='wrap'>",
         f"<p><small>source: {_esc(sheet.get('url'))} · device: <b>{_esc(vis.get('device'))}</b> · "
         f"sampled {_esc(stats.get('sampled'))} → OCR'd <b>{_esc(stats.get('ocr_frames'))}</b> distinct "
         f"@ {_esc(stats.get('sec_per_frame'))}s/frame · primary OCR ticker: "
         f"<b>{_esc(sheet.get('primary_ticker'))}</b></small></p>",
         f"<p><b>{tx_label}:</b> segments = "
         f"<span class='empty'>{n_tx}</span> {'<span class=empty>[EMPTY]</span>' if not n_tx else ''}"
         f"{' · excerpt = <span class=empty>[]</span>' if not n_tx else ''}</p>",
         f"<h3>{len(chosen)} frames — screenshot ↔ exact data read from the screen "
         f"(curated summary + full raw OCR)</h3>"]

    for f in chosen:
        nums = _numeric(f)
        all_nums, all_texts = _all_tokens(f)
        total = len(all_nums) + len(all_texts)
        ap = f.get("axis_price")
        _pn = ocr_parse.parse_num(ap.get("text", "")) if ap else None
        if not (_pn and not _pn["is_decimal"] and 100 <= _pn["value"] <= 9_999_999):
            ap = None                                  # suppress implausible axis reads (e.g. UI strings)
        p = f.get("_vlm") or {}
        crows = "".join(f"<tr><td class='num'>{_esc(t)}</td><td class='num'>{_esc(s)}</td></tr>"
                        for t, s in nums) or "<tr><td colspan=2><i>no numeric tokens</i></td></tr>"
        axis = (f"<p class='k'>chart price-axis current price: {_esc(ap['text'])} "
                f"({_esc(ap['color'])}, conf {_esc(ap['score'])})</p>") if ap else ""

        def col(title, items):
            body = "".join(f"<tr><td>{_esc(t)}</td><td class='num'>{_esc(round(s, 3))}</td></tr>"
                           for t, s in items) or "<tr><td colspan=2><i>none</i></td></tr>"
            return (f"<div><b>{title} <span class='cnt'>{len(items)}</span></b>"
                    f"<table class='tok'><tr><th>token</th><th class='num'>conf</th></tr>{body}</table></div>")
        raw = (f"<details><summary>full raw OCR (everything read on this frame) — {total} tokens</summary>"
               f"<div class='cols'>{col('number tokens', all_nums)}{col('text / label tokens', all_texts)}"
               f"</div></details>")
        vlmdesc = ("<p><b>Vision model (drawing/pattern, no numbers):</b><br>"
                   f"pattern: {_esc(p.get('chart_pattern'))}<br>"
                   f"points at: {_esc(p.get('points_at'))}<br>"
                   f"stance: {_esc(p.get('stance'))}</p>") if p else \
                  "<p><small>(no VLM description for this frame)</small></p>"
        B.append(
            f"<div class='fr'>{img_tag(f['frame'])}<div class='meta'>"
            f"<p><b>@ {mmss(f.get('t', 0))}</b> · frame <code>{_esc(f['frame'])}</code> · "
            f"<b>{total} tokens</b> (<b>{len(all_nums)}</b> number, <b>{len(all_texts)}</b> text)</p>{axis}"
            f"<p><small>curated key numbers (top by confidence):</small></p>"
            f"<table><tr><th>number</th><th class='num'>confidence</th></tr>{crows}</table>"
            f"{raw}{vlmdesc}</div></div>")
    B.append("</div></body></html>")
    return "".join(B)


def save(video_id: str) -> str:
    out = config.DATA_DIR / f"proof_{video_id}.html"
    out.write_text(render_html(video_id, embed=True), encoding="utf-8")
    return str(out)
