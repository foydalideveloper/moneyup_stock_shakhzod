"""Render the grounded daily YouTube report to downloadable .docx AND .pdf (bilingual).

Input is the dict from :func:`tagent.news.youtube_report.build_youtube_report`
(meta/summary/recommendations/sensitive_news/prices/per_stock/catalysts/sources, with ko + en
mirrors). ``render_report(report, lang, out_dir, basename)`` writes both files and returns their
paths. ``lang`` ("ko" default / "en") selects the language: prose comes from the ``en`` mirror when
rendering English; prices/sources/per-stock structure is shared.

Grounding is preserved end-to-end: every recommendation / news / per-stock item shows the channel +
speaker + the VERBATIM quote and a CLICKABLE timestamp deeplink (▶ mm:ss); price cells show "N/A"
for any missing value (never a blank-fabricated number); the catalysts section is omitted when empty.

  * .docx via python-docx — real external hyperlinks, a clean Table-Grid style.
  * .pdf  via reportlab using a BUILT-IN Korean CID font (HYSMyeongJo-Medium) so Korean renders with
    NO bundled font files; clickable links via <a href>.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from tagent.config import DATA_DIR

_KST = timezone(timedelta(hours=9))
PDF_KR_FONT = "HYSMyeongJo-Medium"          # reportlab built-in Korean CID font (no bundled files)
_PDF_FONT = [None]                          # cached resolved PDF font name


def _pdf_font_candidates():
    """(font_name, ttf_path) candidates, preferred first: a bundled TTF, then common system Korean
    fonts (Malgun on Windows, Nanum on Linux). Falls back to the CID font when none exist."""
    fonts = Path(__file__).resolve().parent.parent / "static" / "fonts"
    return [
        ("NanumGothic", str(fonts / "NanumGothic.ttf")),       # bundled (if present)
        ("MalgunGothic", r"C:\Windows\Fonts\malgun.ttf"),      # Windows
        ("NanumGothic", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),  # Linux
        ("AppleGothic", "/System/Library/Fonts/AppleSDGothicNeo.ttc"),       # macOS
    ]


def register_pdf_font() -> str:
    """Register and return a Korean-capable PDF font name. Prefers a full-coverage TTF (bundled /
    system) so no glyphs drop; otherwise the built-in HYSMyeongJo CID font; else Helvetica. Cached."""
    if _PDF_FONT[0]:
        return _PDF_FONT[0]
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfbase.ttfonts import TTFont
    for name, p in _pdf_font_candidates():
        try:
            if os.path.exists(p):
                pdfmetrics.registerFont(TTFont(name, p))
                _PDF_FONT[0] = name
                return name
        except Exception:
            continue
    try:
        pdfmetrics.registerFont(UnicodeCIDFont(PDF_KR_FONT))
        _PDF_FONT[0] = PDF_KR_FONT
        return PDF_KR_FONT
    except Exception:
        _PDF_FONT[0] = "Helvetica"
        return "Helvetica"

LABELS = {
    "ko": {
        "title": "유튜브 시장 리포트", "generated": "생성", "window": "구간",
        "s1": "1. 핵심 요약", "s2": "2. 추천", "s2cols": ["종목", "액션", "근거"],
        "s3": "3. 민감·중요 뉴스",
        "s5": "4. 종목별 분석", "s6": "5. 촉매·일정", "s7": "6. 출처 영상",
        "who": "출연", "watch": "영상 보기", "insights": "건", "none": "없음",
        "single": "단일 영상 리포트", "analyzed": "분석된 영상", "scanned": "기타 스캔됨",
        "footer": "모든 영상 근거는 실제 자막 인용 + 타임스탬프 링크. 근거 없으면 미포함.",
    },
    "en": {
        "title": "YouTube Market Report", "generated": "Generated", "window": "Window",
        "s1": "1. Key Summary", "s2": "2. Recommendations", "s2cols": ["Stock", "Action", "Rationale"],
        "s3": "3. Sensitive / High-impact News",
        "s5": "4. Per-stock Analysis", "s6": "5. Catalysts / Schedule", "s7": "6. Source Videos",
        "who": "Speaker", "watch": "Watch", "insights": "insights", "none": "none",
        "single": "Single-video report", "analyzed": "Analyzed videos", "scanned": "Other scanned",
        "footer": "Every claim is grounded in a real caption quote + timestamp link. No source -> not included.",
    },
}


def _header_meta(L, meta) -> str:
    """The header sub-line: a single-video report shows the video's publish time + a clear label;
    a window report shows the 'generated · window start → end' range."""
    gen = f'{L["generated"]}: {_fmt_kst(meta.get("generated_at_kst"))}'
    if meta.get("single_video"):
        return f'{gen} · {L["single"]} ({_fmt_kst(meta.get("video_published_kst"))})'
    return (f'{gen} · {L["window"]}: {_fmt_kst(meta.get("window_start"))} → '
            f'{_fmt_kst(meta.get("window_end"))}')


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _fmt_kst(iso) -> str:
    s = str(iso or "").strip()
    if not s:
        return ""
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        dt = dt if dt.tzinfo else dt.replace(tzinfo=_KST)
        return dt.astimezone(_KST).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return s


def _stamp(report) -> str:
    s = str((report.get("meta") or {}).get("generated_at_kst") or "").strip()
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s) if s else datetime.now(_KST)
        dt = dt if dt.tzinfo else dt.replace(tzinfo=_KST)
    except ValueError:
        dt = datetime.now(_KST)
    return dt.astimezone(_KST).strftime("%Y%m%d_%H%M")


def _fmt_price(v) -> str:
    if v is None:
        return "N/A"                                  # never blank-fabricate
    try:
        return f"{float(v):,.0f}"
    except (TypeError, ValueError):
        return "N/A"


def _fmt_pct(v) -> str:
    if v is None:
        return "N/A"
    try:
        return f"{float(v):+.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def _rationale_text(rec) -> str:
    """channel + speaker + verbatim quote (the clickable ▶ timestamp is added separately)."""
    who = rec.get("channel", "")
    sp = rec.get("speaker", "")
    if sp:
        who = f"{who} · {sp}" if who else sp
    quote = rec.get("quote") or rec.get("reason", "")
    return f"{who}: “{quote}”" if who else f"“{quote}”"


def _rec_summary(rec) -> str:
    """The clean, multi-sentence rationale (summary) that LEADS the §2 근거 cell."""
    return str(rec.get("reason") or rec.get("summary") or "").strip()


def _rec_citation(rec) -> str:
    """The smaller secondary citation under the summary: channel · speaker + the raw verbatim quote."""
    who = rec.get("channel", "")
    sp = rec.get("speaker", "")
    if sp:
        who = f"{who} · {sp}" if who else sp
    q = str(rec.get("quote", "")).strip()
    if not q:
        return who
    return f"{who}: “{q}”" if who else f"“{q}”"


def _tslabel(item) -> str:
    ts = item.get("timestamp") or item.get("timestamp_mmss") or ""
    return f"▶ {ts}" if ts else "▶"


def _split_sources(sources):
    """(analyzed videos with insights, count of 0-insight scanned videos) for the decluttered §7."""
    analyzed = [s for s in (sources or []) if (s.get("n_insights") or 0) > 0]
    scanned = sum(1 for s in (sources or []) if (s.get("n_insights") or 0) <= 0)
    return analyzed, scanned


def _localize(report, lang) -> dict:
    """Pick prose from the en mirror when rendering English; share structural data."""
    en = report.get("en") if lang == "en" else None

    def pick(key):
        if en and en.get(key) is not None:
            return en.get(key)
        return report.get(key) or []
    overview = (en.get("overview") if (en and en.get("overview") is not None)
                else report.get("overview") or "")
    per_stock = (en.get("per_stock") if (en and en.get("per_stock") is not None)
                 else report.get("per_stock") or {})
    return {"overview": overview, "summary": pick("summary"), "recommendations": pick("recommendations"),
            "sensitive_news": pick("sensitive_news"), "catalysts": pick("catalysts"),
            "per_stock": per_stock, "prices": report.get("prices") or [],
            "sources": report.get("sources") or [], "meta": report.get("meta") or {}}


# --------------------------------------------------------------------------- #
# DOCX
# --------------------------------------------------------------------------- #
def _docx_hyperlink(paragraph, url, text, color="0563C1"):
    """Add a real external hyperlink run to a python-docx paragraph."""
    from docx.opc.constants import RELATIONSHIP_TYPE as RT
    from docx.oxml.ns import qn
    from docx.oxml.shared import OxmlElement
    if not url:
        paragraph.add_run(text)
        return
    r_id = paragraph.part.relate_to(url, RT.HYPERLINK, is_external=True)
    link = OxmlElement("w:hyperlink")
    link.set(qn("r:id"), r_id)
    run = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    c = OxmlElement("w:color"); c.set(qn("w:val"), color); rpr.append(c)
    u = OxmlElement("w:u"); u.set(qn("w:val"), "single"); rpr.append(u)
    run.append(rpr)
    t = OxmlElement("w:t"); t.text = text; run.append(t)
    link.append(run)
    paragraph._p.append(link)


def _set_kr_font(doc, name="Malgun Gothic"):
    from docx.oxml.ns import qn
    from docx.oxml.shared import OxmlElement
    style = doc.styles["Normal"]
    style.font.name = name
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts"); rpr.append(rfonts)
    for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rfonts.set(qn(a), name)


def _render_docx(report, data, lang, path):
    from docx import Document
    from docx.shared import Pt
    L = LABELS[lang]
    meta = data["meta"]
    doc = Document()
    _set_kr_font(doc)

    doc.add_heading(L["title"], level=0)
    doc.add_paragraph().add_run(_header_meta(L, meta))

    # §1 핵심 요약 — overview + a grounded multi-sentence synthesis per major stock
    doc.add_heading(L["s1"], level=1)
    if data.get("overview"):
        doc.add_paragraph().add_run(data["overview"])
    if data["summary"]:
        for b in data["summary"]:
            para = doc.add_paragraph(style="List Bullet")
            para.add_run(f"[{b.get('stock','')}] {b.get('text','')}  ")
            _docx_hyperlink(para, b.get("deeplink", ""), f"{_tslabel(b)} {b.get('channel','')}")
    elif not data.get("overview"):
        doc.add_paragraph(L["none"])

    # §2 recommendations
    doc.add_heading(L["s2"], level=1)
    if data["recommendations"]:
        tbl = doc.add_table(rows=1, cols=3)
        tbl.style = "Table Grid"
        for i, h in enumerate(L["s2cols"]):
            tbl.rows[0].cells[i].paragraphs[0].add_run(h).bold = True
        for r in data["recommendations"]:
            cells = tbl.add_row().cells
            cells[0].paragraphs[0].add_run(f"{r.get('stock','')} ({r.get('ticker','')})")
            cells[1].paragraphs[0].add_run(str(r.get("action", "")))
            cell = cells[2]
            cell.paragraphs[0].add_run(_rec_summary(r))                 # clean multi-sentence summary leads
            cite = cell.add_paragraph()                                 # raw verbatim quote — smaller citation
            cr = cite.add_run(_rec_citation(r) + "  ")
            cr.italic = True
            cr.font.size = Pt(8)
            _docx_hyperlink(cite, r.get("deeplink", ""), _tslabel(r))
    else:
        doc.add_paragraph(L["none"])

    # §3 sensitive news
    doc.add_heading(L["s3"], level=1)
    if data["sensitive_news"]:
        for n in data["sensitive_news"]:
            para = doc.add_paragraph(style="List Bullet")
            para.add_run(f"[{n.get('stock','')}] {n.get('text','')}  ")
            _docx_hyperlink(para, n.get("deeplink", ""), f"{_tslabel(n)} {n.get('channel','')}")
    else:
        doc.add_paragraph(L["none"])

    # §5 per-stock analysis
    doc.add_heading(L["s5"], level=1)
    if data["per_stock"]:
        for stock, items in data["per_stock"].items():
            doc.add_heading(str(stock), level=2)
            for it in items:
                para = doc.add_paragraph(style="List Bullet")
                said = it.get("summary") or it.get("_reason") or ""
                who = it.get("speaker") or it.get("channel") or ""
                para.add_run(f"{said}  ({L['who']}: {who})  “{it.get('quote','')}”  ")
                _docx_hyperlink(para, it.get("deeplink", ""), _tslabel(it))
    else:
        doc.add_paragraph(L["none"])

    # §6 catalysts — OMITTED entirely when empty (no invented events)
    if data["catalysts"]:
        doc.add_heading(L["s6"], level=1)
        for c in data["catalysts"]:
            para = doc.add_paragraph(style="List Bullet")
            para.add_run(f"[{c.get('stock','')}] {c.get('event','')}  ")
            _docx_hyperlink(para, c.get("deeplink", ""), f"{_tslabel(c)} {c.get('channel','')}")

    # §7 source videos — analyzed (insights) listed; off-topic 0-insight videos collapsed to one line
    doc.add_heading(L["s7"], level=1)
    analyzed, scanned = _split_sources(data["sources"])
    if analyzed:
        doc.add_paragraph().add_run(L["analyzed"]).bold = True
        for s in analyzed:
            para = doc.add_paragraph(style="List Bullet")
            para.add_run(f"{s.get('channel','')} — {s.get('title','')}  "
                         f"({_fmt_kst(s.get('published_at_kst'))} · {s.get('n_insights',0)} {L['insights']})  ")
            _docx_hyperlink(para, s.get("url", ""), L["watch"])
    elif not scanned:
        doc.add_paragraph(L["none"])
    if scanned:
        doc.add_paragraph(f"{L['scanned']}: {scanned}개" if lang == "ko" else f"{L['scanned']}: {scanned}")

    doc.add_paragraph()
    foot = doc.add_paragraph()
    foot.add_run(L["footer"]).italic = True
    doc.save(str(path))


# --------------------------------------------------------------------------- #
# PDF (reportlab + built-in Korean CID font)
# --------------------------------------------------------------------------- #
def _esc(s) -> str:
    return str(s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _pdf_link(text, url) -> str:
    """An inline <a href> fragment (url & text escaped for reportlab's mini-XML)."""
    if not url:
        return _esc(text)
    return f'<a href="{_esc(url)}" color="#0563C1">{_esc(text)}</a>'


def _render_pdf(report, data, lang, path):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    FONT = register_pdf_font()                                  # TTF (full coverage) or CID fallback
    L = LABELS[lang]
    meta = data["meta"]
    body = ParagraphStyle("body", fontName=FONT, fontSize=9, leading=13, alignment=TA_LEFT)
    h1 = ParagraphStyle("h1", fontName=FONT, fontSize=13, leading=17, spaceBefore=10, spaceAfter=4)
    title = ParagraphStyle("title", fontName=FONT, fontSize=18, leading=22, spaceAfter=6)
    small = ParagraphStyle("small", fontName=FONT, fontSize=8, leading=11, textColor=colors.grey)
    cite = ParagraphStyle("cite", fontName=FONT, fontSize=7.5, leading=10, textColor=colors.grey)

    story = [Paragraph(_esc(L["title"]), title),
             Paragraph(_esc(_header_meta(L, meta)), small), Spacer(1, 6)]

    def _table(headers, rows, col_widths):
        head = [Paragraph(f"<b>{_esc(h)}</b>", body) for h in headers]
        body_rows = [[Paragraph(c, body) if isinstance(c, str) else c for c in row] for row in rows]
        t = Table([head] + body_rows, colWidths=col_widths, repeatRows=1)
        t.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
        return t

    # §1 — overview + a grounded multi-sentence synthesis per major stock
    story.append(Paragraph(_esc(L["s1"]), h1))
    if data.get("overview"):
        story.append(Paragraph(_esc(data["overview"]), body))
        story.append(Spacer(1, 4))
    for b in data["summary"] or []:
        story.append(Paragraph(f"• [{_esc(b.get('stock',''))}] {_esc(b.get('text',''))} "
                               + _pdf_link(f"{_tslabel(b)} {b.get('channel','')}", b.get("deeplink", "")), body))
    if not data["summary"] and not data.get("overview"):
        story.append(Paragraph(_esc(L["none"]), body))

    # §2
    story.append(Paragraph(_esc(L["s2"]), h1))
    if data["recommendations"]:
        rows = []
        for r in data["recommendations"]:
            lead = Paragraph(_esc(_rec_summary(r)), body)      # clean multi-sentence summary leads
            sub = Paragraph(_esc(_rec_citation(r)) + " "       # raw verbatim quote — smaller citation
                            + _pdf_link(_tslabel(r), r.get("deeplink", "")), cite)
            rows.append([f"{_esc(r.get('stock',''))} ({_esc(r.get('ticker',''))})",
                         _esc(r.get("action", "")), [lead, sub]])   # a flowable list in the cell (splittable)
        story.append(_table(L["s2cols"], rows, [38 * mm, 22 * mm, 110 * mm]))
    else:
        story.append(Paragraph(_esc(L["none"]), body))

    # §3
    story.append(Paragraph(_esc(L["s3"]), h1))
    for n in data["sensitive_news"] or []:
        story.append(Paragraph(f"• [{_esc(n.get('stock',''))}] {_esc(n.get('text',''))} "
                               + _pdf_link(f"{_tslabel(n)} {n.get('channel','')}", n.get("deeplink", "")), body))
    if not data["sensitive_news"]:
        story.append(Paragraph(_esc(L["none"]), body))

    # §5 per-stock
    story.append(Paragraph(_esc(L["s5"]), h1))
    if data["per_stock"]:
        for stock, items in data["per_stock"].items():
            story.append(Paragraph(f"<b>{_esc(stock)}</b>", body))
            for it in items:
                said = it.get("summary") or it.get("_reason") or ""
                who = it.get("speaker") or it.get("channel") or ""
                story.append(Paragraph(
                    f"&nbsp;&nbsp;• {_esc(said)} ({_esc(L['who'])}: {_esc(who)}) "
                    f"“{_esc(it.get('quote',''))}” "
                    + _pdf_link(_tslabel(it), it.get("deeplink", "")), body))
    else:
        story.append(Paragraph(_esc(L["none"]), body))

    # §6 catalysts — omit section if empty
    if data["catalysts"]:
        story.append(Paragraph(_esc(L["s6"]), h1))
        for c in data["catalysts"]:
            story.append(Paragraph(f"• [{_esc(c.get('stock',''))}] {_esc(c.get('event',''))} "
                                   + _pdf_link(f"{_tslabel(c)} {c.get('channel','')}", c.get("deeplink", "")), body))

    # §7 sources — analyzed listed; off-topic 0-insight videos collapsed to one line
    story.append(Paragraph(_esc(L["s7"]), h1))
    analyzed, scanned = _split_sources(data["sources"])
    if analyzed:
        story.append(Paragraph(f"<b>{_esc(L['analyzed'])}</b>", body))
        for s in analyzed:
            story.append(Paragraph(
                f"• {_esc(s.get('channel',''))} — {_esc(s.get('title',''))} "
                f"({_esc(_fmt_kst(s.get('published_at_kst')))} · {s.get('n_insights',0)} {_esc(L['insights'])}) "
                + _pdf_link(L["watch"], s.get("url", "")), body))
    elif not scanned:
        story.append(Paragraph(_esc(L["none"]), body))
    if scanned:
        tail = f"{L['scanned']}: {scanned}개" if lang == "ko" else f"{L['scanned']}: {scanned}"
        story.append(Paragraph(_esc(tail), small))

    story += [Spacer(1, 8), Paragraph(f"<i>{_esc(L['footer'])}</i>", small)]
    SimpleDocTemplate(str(path), pagesize=A4, topMargin=15 * mm, bottomMargin=15 * mm,
                      leftMargin=15 * mm, rightMargin=15 * mm, title=L["title"]).build(story)


# --------------------------------------------------------------------------- #
# HTML preview (same grounded dict; for the dashboard inline preview)
# --------------------------------------------------------------------------- #
def _h(text, url) -> str:
    if not url:
        return _esc(text)
    return f'<a href="{_esc(url)}" target="_blank" rel="noopener">{_esc(text)}</a>'


def report_to_html(report: dict, lang: str = "ko") -> str:
    """A compact grounded HTML snippet (no <html> wrapper) for the dashboard preview — built from the
    SAME report dict as the .docx/.pdf, so the preview can't drift from the files. Catalysts omitted
    when empty; price cells show N/A for None."""
    lang = "en" if str(lang).lower() == "en" else "ko"
    L = LABELS[lang]
    d = _localize(report, lang)
    m = d["meta"]
    out = [f'<div class="ytrep">',
           f'<h3>{_esc(L["title"])}</h3>',
           f'<div class="muted small">{_esc(_header_meta(L, m))}</div>']

    out.append(f'<h4>{_esc(L["s1"])}</h4>')
    if d.get("overview"):
        out.append(f'<p>{_esc(d["overview"])}</p>')          # overall grounded market overview
    if d["summary"]:
        lis = []
        for b in d["summary"]:
            link = _h(f'{_tslabel(b)} {b.get("channel", "")}', b.get("deeplink", ""))
            lis.append(f'<li>[{_esc(b.get("stock",""))}] {_esc(b.get("text",""))} {link}</li>')
        out.append("<ul>" + "".join(lis) + "</ul>")
    elif not d.get("overview"):
        out.append(f'<div class="muted small">{_esc(L["none"])}</div>')

    out.append(f'<h4>{_esc(L["s2"])}</h4>')
    if d["recommendations"]:
        cells = []
        for r in d["recommendations"]:
            ground = (f'<div class="muted small" style="margin-top:2px">{_esc(_rec_citation(r))} '
                      f'{_h(_tslabel(r), r.get("deeplink",""))}</div>')
            cells.append(f'<tr><td>{_esc(r.get("stock",""))} ({_esc(r.get("ticker",""))})</td>'
                         f'<td>{_esc(r.get("action",""))}</td>'
                         f'<td>{_esc(_rec_summary(r))}{ground}</td></tr>')
        out.append(f'<table class="ytrep-t"><thead><tr><th>{_esc(L["s2cols"][0])}</th>'
                   f'<th>{_esc(L["s2cols"][1])}</th><th>{_esc(L["s2cols"][2])}</th></tr></thead>'
                   f'<tbody>{"".join(cells)}</tbody></table>')
    else:
        out.append(f'<div class="muted small">{_esc(L["none"])}</div>')

    out.append(f'<h4>{_esc(L["s3"])}</h4>')
    if d["sensitive_news"]:
        out.append("<ul>" + "".join(
            f'<li>[{_esc(n.get("stock",""))}] {_esc(n.get("text",""))} '
            f'{_h(_tslabel(n), n.get("deeplink",""))}</li>' for n in d["sensitive_news"]) + "</ul>")
    else:
        out.append(f'<div class="muted small">{_esc(L["none"])}</div>')

    out.append(f'<h4>{_esc(L["s5"])}</h4>')
    if d["per_stock"]:
        for stock, items in d["per_stock"].items():
            out.append(f"<b>{_esc(stock)}</b><ul>" + "".join(
                f'<li>{_esc(it.get("summary") or it.get("_reason") or "")} '
                f'({_esc(L["who"])}: {_esc(it.get("speaker") or it.get("channel") or "")}) '
                f'“{_esc(it.get("quote",""))}” {_h(_tslabel(it), it.get("deeplink",""))}</li>'
                for it in items) + "</ul>")
    else:
        out.append(f'<div class="muted small">{_esc(L["none"])}</div>')

    if d["catalysts"]:                                       # omit section entirely when empty
        out.append(f'<h4>{_esc(L["s6"])}</h4><ul>' + "".join(
            f'<li>[{_esc(c.get("stock",""))}] {_esc(c.get("event",""))} '
            f'{_h(_tslabel(c), c.get("deeplink",""))}</li>' for c in d["catalysts"]) + "</ul>")

    out.append(f'<h4>{_esc(L["s7"])}</h4>')
    analyzed, scanned = _split_sources(d["sources"])
    if analyzed:
        out.append(f'<div class="small"><b>{_esc(L["analyzed"])}</b></div><ul>' + "".join(
            f'<li>{_esc(s.get("channel",""))} — {_esc(s.get("title",""))} '
            f'({_esc(_fmt_kst(s.get("published_at_kst")))} · {s.get("n_insights",0)} {_esc(L["insights"])}) '
            f'{_h(L["watch"], s.get("url",""))}</li>' for s in analyzed) + "</ul>")
    elif not scanned:
        out.append(f'<div class="muted small">{_esc(L["none"])}</div>')
    if scanned:
        tail = f"{L['scanned']}: {scanned}개" if lang == "ko" else f"{L['scanned']}: {scanned}"
        out.append(f'<div class="muted small">{_esc(tail)}</div>')

    out.append(f'<div class="muted small" style="margin-top:6px"><i>{_esc(L["footer"])}</i></div></div>')
    return "".join(out)


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def render_report(report: dict, lang: str = "ko", out_dir=None, basename: Optional[str] = None) -> dict:
    """Render ``report`` to .docx + .pdf in ``lang`` ("ko"/"en"). Returns {"docx": path, "pdf": path}.
    Filenames: ``<basename or 'youtube_report'>_<YYYYMMDD_HHMM>_<lang>.docx/.pdf``."""
    lang = "en" if str(lang).lower() == "en" else "ko"
    out = Path(out_dir) if out_dir else Path(DATA_DIR) / "reports"
    out.mkdir(parents=True, exist_ok=True)
    stamp = _stamp(report)
    base = basename or "youtube_report"
    docx_path = out / f"{base}_{stamp}_{lang}.docx"
    pdf_path = out / f"{base}_{stamp}_{lang}.pdf"
    data = _localize(report, lang)
    _render_docx(report, data, lang, docx_path)
    _render_pdf(report, data, lang, pdf_path)
    return {"docx": str(docx_path), "pdf": str(pdf_path)}
