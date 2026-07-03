"""Dashboard YouTube-report endpoints — fully mocked (injected fake builder/renderer; no network).

Covers: /youtube_report/window and /video return the expected JSON shape; the lang toggle passes
through to the builder; /video requires a url; and /youtube_report/file/{name} serves only from
data/reports (refuses traversal / non-report names)."""

from pathlib import Path

from fastapi.testclient import TestClient

from tagent.dashboard import create_app, safe_report_name


class _FakeStock:
    def quote(self, s): return {"symbol": s}


class _FakeCrypto:
    def quote(self, s): return {"symbol": s}


def _fake_report(lang="ko", n_videos=1):
    return {"meta": {"lang": lang, "n_videos": n_videos, "n_insights": 1,
                     "generated_at_kst": "2026-06-15T15:50:00+09:00",
                     "window_start": "2026-06-14T00:00:00+09:00", "window_end": "2026-06-15T15:50:00+09:00"},
            "summary": [{"stock": "SK하이닉스", "ticker": "000660", "text": "HBM 공급계약",
                         "channel": "한경TV", "timestamp": "02:00",
                         "deeplink": "https://www.youtube.com/watch?v=v1&t=120s"}],
            "recommendations": [], "sensitive_news": [], "per_stock": {}, "catalysts": [],
            "prices": [], "sources": [], "en": None}


def _app(tmp_path):
    captured = {}
    reports = Path(tmp_path) / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    def builder(lang="ko", end=None, watchlist=None):
        captured["window"] = {"lang": lang, "end": end, "watchlist": watchlist}
        return _fake_report(lang)

    def video(url="", lang="ko", watchlist=None, refresh=False):
        captured["video"] = {"url": url, "lang": lang, "watchlist": watchlist, "refresh": refresh}
        return _fake_report(lang)

    def renderer(report, lang, out_dir, basename):
        return {"docx": str(Path(out_dir) / f"{basename}_20260615_1550_{lang}.docx"),
                "pdf": str(Path(out_dir) / f"{basename}_20260615_1550_{lang}.pdf")}

    app = create_app(_FakeStock(), _FakeCrypto(), report_builder=builder, report_video=video,
                     report_renderer=renderer, reports_dir=str(reports))
    return TestClient(app), captured, reports


# --------------------------------------------------------------------------- #
# /window
# --------------------------------------------------------------------------- #
def test_window_endpoint_shape_and_lang_passthrough(tmp_path):
    c, captured, _ = _app(tmp_path)
    d = c.post("/youtube_report/window", json={"lang": "en", "end": "2026-06-15T15:50:00+09:00",
                                               "watchlist": ["005930", "000660"]}).json()
    assert set(d) >= {"meta", "html_preview", "docx_url", "pdf_url"}
    assert isinstance(d["html_preview"], str) and d["html_preview"]
    assert d["docx_url"] == "/youtube_report/file/youtube_report_20260615_1550_en.docx"
    assert d["pdf_url"].endswith("_en.pdf") and d["pdf_url"].startswith("/youtube_report/file/")
    assert d["meta"]["lang"] == "en"
    # the lang toggle + window args reach the builder
    assert captured["window"]["lang"] == "en"
    assert captured["window"]["end"] == "2026-06-15T15:50:00+09:00"
    assert captured["window"]["watchlist"] == ["005930", "000660"]


def test_window_defaults_to_korean(tmp_path):
    c, captured, _ = _app(tmp_path)
    d = c.post("/youtube_report/window", json={}).json()
    assert captured["window"]["lang"] == "ko" and d["docx_url"].endswith("_ko.docx")


# --------------------------------------------------------------------------- #
# /video
# --------------------------------------------------------------------------- #
def test_video_endpoint_shape_and_passthrough(tmp_path):
    c, captured, _ = _app(tmp_path)
    d = c.post("/youtube_report/video",
               json={"url": "https://youtu.be/dMVpoiK36EU", "lang": "ko"}).json()
    assert set(d) >= {"meta", "html_preview", "docx_url", "pdf_url"}
    assert d["docx_url"] == "/youtube_report/file/youtube_video_20260615_1550_ko.docx"
    assert captured["video"]["url"] == "https://youtu.be/dMVpoiK36EU" and captured["video"]["lang"] == "ko"


def test_video_requires_url(tmp_path):
    c, _, _ = _app(tmp_path)
    d = c.post("/youtube_report/video", json={"lang": "ko"}).json()
    assert d.get("error") == "url required"


# --------------------------------------------------------------------------- #
# /file — serve only from data/reports; refuse traversal
# --------------------------------------------------------------------------- #
def test_file_endpoint_serves_generated_report(tmp_path):
    c, _, reports = _app(tmp_path)
    fname = "youtube_report_20260615_1550_ko.docx"
    (reports / fname).write_bytes(b"DOCXDATA")
    r = c.get(f"/youtube_report/file/{fname}")
    assert r.status_code == 200 and r.content == b"DOCXDATA"


def test_file_endpoint_refuses_traversal(tmp_path):
    c, _, reports = _app(tmp_path)
    # write a secret OUTSIDE reports/ that a traversal would try to reach
    (Path(tmp_path) / "secret.docx").write_bytes(b"SECRET")
    for bad in ("%2e%2e%2f%2e%2e%2fsecret.docx", "%2e%2e%2fsecret.docx", "sub%2fx.docx"):
        r = c.get(f"/youtube_report/file/{bad}")
        assert r.status_code in (400, 404) and b"SECRET" not in r.content


def test_file_endpoint_rejects_non_report_extensions(tmp_path):
    c, _, reports = _app(tmp_path)
    (reports / "x.txt").write_bytes(b"nope")
    assert c.get("/youtube_report/file/x.txt").status_code == 400


# --------------------------------------------------------------------------- #
# single-video: ONE cached extraction -> deterministic, identical across languages
# --------------------------------------------------------------------------- #
def test_single_video_ko_en_render_identical_stocks_and_actions():
    """KO and EN built from ONE insight set must have an identical stock list + identical
    BUY/SELL/WATCH actions — only the prose is translated. (The cross-language contradiction bug:
    EN said SK하이닉스 BUY while KO said WATCH because each language re-extracted separately.)"""
    from tagent.news.youtube_report import build_report_from_videos
    segs = [{"start": 10.0, "text": "SK하이닉스 목표가 25만원 매수 의견입니다"},
            {"start": 20.0, "text": "삼성전기는 비중축소 매도가 좋겠습니다"}]
    insights = [
        {"stock": "000660", "summary": "SK하이닉스 목표가 25만원 매수.",
         "quote": "SK하이닉스 목표가 25만원 매수 의견입니다", "action": "매수", "target_price": "25만원"},
        {"stock": "009150", "summary": "삼성전기 비중축소 매도.",
         "quote": "삼성전기는 비중축소 매도가 좋겠습니다", "action": "매도"},
    ]
    entry = {"video_id": "v1", "channel": "한경TV", "title": "t",
             "published_at": "2026-06-14T09:00:00Z", "segments": segs, "insights": insights}

    def fake_tr(texts):                                         # batch translator: prose only
        return [f"EN::{t}" for t in texts]

    ko = build_report_from_videos([entry], lang="ko", translate_fn=None, prices_fn=None)
    en = build_report_from_videos([entry], lang="en", translate_fn=fake_tr, prices_fn=None)
    ko_recs = [(r["stock"], r["action"]) for r in ko["recommendations"]]
    en_recs = [(r["stock"], r["action"]) for r in en["en"]["recommendations"]]   # EN mirror = rendered
    assert ko_recs and ko_recs == en_recs                      # same stocks + same calls across languages
    assert en["en"]["recommendations"][0]["reason"].startswith("EN::")   # EN prose really translated


def test_single_video_extraction_uses_stronger_pro_model(monkeypatch, tmp_path):
    """Single-video EXTRACTION must use the stronger batch model (gemini-2.5-pro), not flash —
    flash gave shallow, generic results. Translation stays on flash (tested elsewhere)."""
    from tagent import dashboard, gemini
    from tagent.config import SETTINGS
    captured = {}

    def fake_build_extractor(settings, model=None):
        captured["model"] = model
        return lambda segments, vmeta, channel: [
            {"stock": "000660", "summary": "SK하이닉스 300만원.", "quote": segments[0]["text"],
             "action": "매수", "target_price": "300만원"}]

    monkeypatch.setattr(gemini, "build_extractor", fake_build_extractor)
    segs = [{"start": 1.0, "text": "SK하이닉스 목표가 300만원 매수"}]
    dashboard._default_report_video(
        url="https://youtu.be/abcdefghijk", lang="ko",
        transcribe_fn=lambda vid: (segs, "whisper"),           # extract_fn NOT injected -> builds real
        metadata_fn=lambda vid: {"title": "t", "channel": "한경TV", "published_at": "2026-06-14T09:00:00Z"},
        prices_fn=None, data_dir=str(tmp_path))
    assert captured["model"] == (SETTINGS.gemini_extract_model or SETTINGS.gemini_interactive_model)
    assert captured["model"] == SETTINGS.gemini_extract_model   # = the stronger pro model


def test_single_video_refresh_busts_cache_and_reextracts(tmp_path):
    """``refresh=True`` ignores a cached (weak) extraction and re-extracts, overwriting the cache."""
    from tagent import dashboard
    n = {"extract": 0}

    def make(stock):
        def extract_fn(segments, vmeta, channel):
            n["extract"] += 1
            return [{"stock": stock, "summary": "s", "quote": segments[0]["text"], "action": "매수"}]
        return extract_fn

    base = dict(transcribe_fn=lambda vid: ([{"start": 1.0, "text": "SK하이닉스 목표가 300만원 매수"}], "w"),
                metadata_fn=lambda vid: {"title": "t", "channel": "한경TV", "published_at": "2026-06-14T09:00:00Z"},
                prices_fn=None, data_dir=str(tmp_path))
    url = "https://youtu.be/abcdefghijk"
    dashboard._default_report_video(url=url, lang="ko", extract_fn=make("000660"), **base)
    assert n["extract"] == 1
    dashboard._default_report_video(url=url, lang="ko", extract_fn=make("000660"), **base)
    assert n["extract"] == 1                                    # 2nd plain request -> cache hit
    r = dashboard._default_report_video(url=url, lang="ko", extract_fn=make("005930"), refresh=True, **base)
    assert n["extract"] == 2                                    # refresh -> re-extracted
    assert r["recommendations"][0]["ticker"] == "005930"       # new result, and cache overwritten
    r2 = dashboard._default_report_video(url=url, lang="ko", extract_fn=make("000660"), **base)
    assert n["extract"] == 2 and r2["recommendations"][0]["ticker"] == "005930"   # reuses refreshed cache


def test_single_video_caches_and_reuses_extraction(tmp_path):
    """A 2nd request for the same video (any language) reuses the cached transcript + insights —
    no re-transcribe, no re-extract, no metadata fetch — so it's deterministic AND fast."""
    from tagent import dashboard
    calls = {"transcribe": 0, "extract": 0, "metadata": 0}
    segs = [{"start": 10.0, "text": "SK하이닉스 목표가 25만원 매수 의견입니다"}]

    def transcribe_fn(vid):
        calls["transcribe"] += 1
        return segs, "whisper"

    def extract_fn(segments, vmeta, channel):
        calls["extract"] += 1
        return [{"stock": "000660", "summary": "SK하이닉스 목표가 25만원 매수.",
                 "quote": "SK하이닉스 목표가 25만원 매수 의견입니다", "action": "매수", "target_price": "25만원"}]

    def metadata_fn(vid):
        calls["metadata"] += 1
        return {"title": "t", "channel": "한경TV", "published_at": "2026-06-14T09:00:00Z"}

    kw = dict(transcribe_fn=transcribe_fn, extract_fn=extract_fn, metadata_fn=metadata_fn,
              prices_fn=None, data_dir=str(tmp_path))
    url = "https://youtu.be/abcdefghijk"
    r1 = dashboard._default_report_video(url=url, lang="ko", **kw)
    assert calls == {"transcribe": 1, "extract": 1, "metadata": 1}      # 1st request: extract once
    r2 = dashboard._default_report_video(url=url, lang="ko", **kw)
    assert calls == {"transcribe": 1, "extract": 1, "metadata": 1}      # 2nd: cache hit -> no new calls
    assert [x["stock"] for x in r1["recommendations"]] == [x["stock"] for x in r2["recommendations"]]
    assert r2["recommendations"][0]["action"] == "BUY"                  # deterministic same call


def test_safe_report_name_validator():
    assert safe_report_name("youtube_report_20260615_1550_ko.docx") == "youtube_report_20260615_1550_ko.docx"
    assert safe_report_name("a.pdf") == "a.pdf"
    for bad in ("../x.docx", "..\\x.pdf", "a/b.docx", "x.txt", "", "report.docx.exe", ".."):
        assert safe_report_name(bad) is None
