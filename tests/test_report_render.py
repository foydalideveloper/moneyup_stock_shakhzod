"""Render the grounded YouTube report to .docx + .pdf — fully offline (libs local, no network).

Covers: both files exist & non-empty; the §4 시세 price section is never rendered (removed); ko vs
en pick the right language; an empty catalysts section is omitted; and a deeplink hyperlink is present
in the .docx (and a /URI link annotation in the .pdf)."""

import zipfile
from pathlib import Path

from tagent.news.report_render import register_pdf_font, render_report, report_to_html

_DEEPLINK = "https://www.youtube.com/watch?v=v1&t=120s"


def _sample_report(catalysts=None, en=None):
    return {
        "meta": {"generated_at_kst": "2026-06-15T15:50:00+09:00",
                 "window_start": "2026-06-14T00:00:00+09:00", "window_end": "2026-06-15T15:50:00+09:00",
                 "lang": "ko", "n_videos": 1, "n_insights": 1, "channels": ["한경TV"]},
        "summary": [{"stock": "SK하이닉스", "ticker": "000660", "text": "HBM 공급계약 소식",
                     "channel": "한경TV", "timestamp": "02:00", "deeplink": _DEEPLINK}],
        "recommendations": [{"stock": "SK하이닉스", "ticker": "000660", "action": "관심(WATCH)",
                             "conviction": "high", "reason": "HBM 공급계약", "quote": "엔비디아와 HBM 공급계약을 체결",
                             "channel": "한경TV", "speaker": "김 위원", "timestamp": "02:00", "deeplink": _DEEPLINK}],
        "sensitive_news": [{"stock": "SK하이닉스", "ticker": "000660", "category": "M&A/deal",
                            "text": "대형 공급계약", "quote": "엔비디아와 HBM 공급계약을 체결", "channel": "한경TV",
                            "timestamp": "02:00", "deeplink": _DEEPLINK}],
        "per_stock": {"SK하이닉스": [{"summary": "HBM 공급계약", "quote": "엔비디아와 HBM 공급계약을 체결",
                                    "channel": "한경TV", "speaker": "김 위원", "timestamp_mmss": "02:00",
                                    "deeplink": _DEEPLINK}]},
        "prices": [{"name": "삼성전자", "ticker": "005930", "current": 72000.0, "today_open": 71000.0,
                    "prev_close": 70000.0, "week_ago_close": None, "month_ago_close": None,
                    "change_pct": 2.86, "source": "키움/KRX"}],
        "catalysts": catalysts if catalysts is not None else [],
        "sources": [{"channel": "한경TV", "title": "증시 브리핑", "published_at_kst": "2026-06-14T11:00:00+09:00",
                     "url": "https://www.youtube.com/watch?v=v1", "n_insights": 1}],
        "en": en,
    }


def _docx_xml(path):
    with zipfile.ZipFile(path) as z:
        doc = z.read("word/document.xml").decode("utf-8")
        rels = z.read("word/_rels/document.xml.rels").decode("utf-8")
    return doc, rels


# --------------------------------------------------------------------------- #
# 1) both files exist & non-empty
# --------------------------------------------------------------------------- #
def test_renders_both_files_nonempty(tmp_path):
    out = render_report(_sample_report(), lang="ko", out_dir=tmp_path)
    for k in ("docx", "pdf"):
        p = Path(out[k])
        assert p.exists() and p.stat().st_size > 0
    assert out["docx"].endswith("_ko.docx") and out["pdf"].endswith("_ko.pdf")
    assert "20260615_1550" in out["docx"]                  # stamp from generated_at_kst


# --------------------------------------------------------------------------- #
# 2) the §4 시세 (price) section is REMOVED — never rendered, even if the report carries a prices list
# --------------------------------------------------------------------------- #
def test_price_section_removed_from_render(tmp_path):
    out = render_report(_sample_report(), lang="ko", out_dir=tmp_path)   # fixture still carries a prices list
    doc, _ = _docx_xml(out["docx"])
    assert "시세" not in doc                                 # no §4 시세 heading
    assert "현재가" not in doc                                # no price column headers
    assert "72,000" not in doc and "+2.86%" not in doc      # no price values rendered
    assert "종목별 분석" in doc                               # the other sections still render intact


# --------------------------------------------------------------------------- #
# 3) ko vs en pick the right language (labels + en-mirror prose)
# --------------------------------------------------------------------------- #
def test_language_selects_labels_and_en_mirror(tmp_path):
    en = {"summary": [{"stock": "SK하이닉스", "ticker": "000660", "text": "HBM supply deal",
                       "channel": "한경TV", "timestamp": "02:00", "deeplink": _DEEPLINK}],
          "recommendations": [{"stock": "SK하이닉스", "ticker": "000660", "action": "WATCH",
                               "reason": "HBM supply deal", "quote": "엔비디아와 HBM 공급계약을 체결",
                               "channel": "한경TV", "speaker": "Kim", "timestamp": "02:00", "deeplink": _DEEPLINK}],
          "sensitive_news": [], "catalysts": []}
    ko = render_report(_sample_report(), lang="ko", out_dir=tmp_path)
    en_out = render_report(_sample_report(en=en), lang="en", out_dir=tmp_path)
    ko_doc, _ = _docx_xml(ko["docx"])
    en_doc, _ = _docx_xml(en_out["docx"])
    assert "핵심 요약" in ko_doc and "유튜브 시장 리포트" in ko_doc       # Korean labels
    assert "Key Summary" in en_doc and "YouTube Market Report" in en_doc  # English labels
    assert "HBM supply deal" in en_doc                     # translated prose from the en mirror
    assert en_out["docx"].endswith("_en.docx") and en_out["pdf"].endswith("_en.pdf")


# --------------------------------------------------------------------------- #
# 4) empty catalysts section omitted; present when there are items
# --------------------------------------------------------------------------- #
def test_empty_catalysts_section_omitted(tmp_path):
    empty_doc, _ = _docx_xml(render_report(_sample_report(catalysts=[]), out_dir=tmp_path)["docx"])
    assert "촉매·일정" not in empty_doc                      # no invented events -> section omitted
    cat = [{"stock": "SK하이닉스", "ticker": "000660", "event": "다음 주 실적발표 예정",
            "quote": "다음 주 실적발표가 예정", "channel": "한경TV", "timestamp": "03:00", "deeplink": _DEEPLINK}]
    full_doc, _ = _docx_xml(render_report(_sample_report(catalysts=cat), out_dir=tmp_path)["docx"])
    assert "촉매·일정" in full_doc and "실적발표" in full_doc


# --------------------------------------------------------------------------- #
# 5) a deeplink hyperlink is present (docx external rel + pdf /URI annotation)
# --------------------------------------------------------------------------- #
def test_deeplink_hyperlink_present(tmp_path):
    out = render_report(_sample_report(), lang="ko", out_dir=tmp_path)
    doc, rels = _docx_xml(out["docx"])
    assert _DEEPLINK in rels.replace("&amp;", "&")          # real clickable deeplink (& is XML-escaped)
    assert 'TargetMode="External"' in rels and "hyperlink" in doc
    pdf_bytes = Path(out["pdf"]).read_bytes()
    assert b"/URI" in pdf_bytes                             # reportlab wrote a link annotation


# --------------------------------------------------------------------------- #
# 6) §2 readability: clean summary leads, raw quote is the secondary citation
# --------------------------------------------------------------------------- #
def test_recommendation_cell_summary_leads_then_quote_citation():
    rep = _sample_report()
    rep["recommendations"][0]["reason"] = "HBM 공급계약으로 메모리 업황이 개선됩니다. 목표가 상향이 기대됩니다."
    rep["recommendations"][0]["quote"] = "엔비디아와 HBM 공급계약을 체결"
    html = report_to_html(rep, "ko")
    summ_i = html.find("HBM 공급계약으로 메모리 업황이 개선됩니다")
    quote_i = html.find("엔비디아와 HBM 공급계약을 체결")
    assert summ_i != -1 and quote_i != -1                   # both the summary AND the raw quote shown
    assert summ_i < quote_i                                 # clean summary LEADS; quote is the secondary citation
    assert "김 위원" in html                                 # speaker cited in the secondary line


# --------------------------------------------------------------------------- #
# 7) §7 declutter: analyzed videos listed; off-topic 0-insight ones collapsed
# --------------------------------------------------------------------------- #
def test_sources_split_analyzed_and_collapse_scanned():
    rep = _sample_report()
    rep["sources"] = [
        {"channel": "한경TV", "title": "삼성전자 분석", "published_at_kst": "2026-06-14T11:00:00+09:00",
         "url": "https://youtu.be/a", "n_insights": 3},
        {"channel": "헬스채널", "title": "무릎 건강 운동법", "published_at_kst": "2026-06-14T09:00:00+09:00",
         "url": "https://youtu.be/b", "n_insights": 0},
        {"channel": "스포츠", "title": "어제 야구 하이라이트", "published_at_kst": "2026-06-14T08:00:00+09:00",
         "url": "https://youtu.be/c", "n_insights": 0},
    ]
    html = report_to_html(rep, "ko")
    assert "분석된 영상" in html and "삼성전자 분석" in html         # analyzed video listed
    assert "기타 스캔됨: 2개" in html                            # the two 0-insight videos collapsed to one line
    assert "무릎 건강 운동법" not in html and "야구 하이라이트" not in html  # off-topic NOT enumerated


def test_pdf_font_registers_korean_capable():
    assert register_pdf_font()                              # a usable Korean font name (TTF or CID)
