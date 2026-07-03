"""ADDITIVE Supabase push — payload assembly, missing-creds skip, key safety, and the run_daily_email
wiring. Fully mocked: NO network (a fake requests-like session captures every call)."""

import importlib.util
import json
import pathlib

from tagent.news.supabase_push import (
    SupabaseClient, build_content_json, build_recommendations, build_rows, build_sources,
    push_report, storage_path,
)

# import the runner script (scripts/ isn't a package) to test its maybe_supabase_push() wiring
_RUNNER = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "run_daily_email.py"
_spec = importlib.util.spec_from_file_location("run_daily_email", _RUNNER)
rde = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rde)


# --------------------------------------------------------------------------- #
# fixtures: a grounded sample report + a fake HTTP session (no network)
# --------------------------------------------------------------------------- #
def _sample_report():
    return {
        "meta": {"n_insights": 3, "n_videos": 2, "lang": "ko",
                 "generated_at_kst": "2026-06-15T06:50:00+09:00",
                 "window_start": "2026-06-14T00:00:00+09:00",
                 "window_end": "2026-06-15T06:50:00+09:00"},
        "overview": "이번 구간 핵심 요약입니다.",
        "summary": [
            {"stock": "SK하이닉스", "ticker": "000660", "text": "HBM 수요 강세.",
             "channel": "한경TV", "timestamp": "02:00",
             "deeplink": "https://www.youtube.com/watch?v=v1&t=120s"},
            {"stock": "삼성전자", "ticker": "005930", "text": "파운드리 회복.",
             "channel": "삼프로TV", "timestamp": "03:20",
             "deeplink": "https://www.youtube.com/watch?v=v1&t=200s"},
        ],
        "recommendations": [
            {"stock": "SK하이닉스", "ticker": "000660", "action": "BUY", "reason": "HBM 공급계약.",
             "quote": "SK하이닉스가 엔비디아와 HBM 공급계약을 체결했습니다", "channel": "한경TV",
             "speaker": "김연구원", "timestamp": "02:00",
             "deeplink": "https://www.youtube.com/watch?v=v1&t=120s"},
            {"stock": "NVIDIA", "ticker": "NVDA", "action": "관심(WATCH)", "reason": "AI 수요.",
             "quote": "엔비디아 수요가 강합니다", "channel": "삼프로TV", "speaker": "",
             "timestamp": "05:00", "deeplink": "https://www.youtube.com/watch?v=v2&t=300s"},
        ],
        "per_stock": {
            "SK하이닉스": [{"ticker": "000660", "stock": "000660", "summary": "HBM 공급계약",
                            "quote": "SK하이닉스가 엔비디아와 HBM 공급계약을 체결했습니다",
                            "deeplink": "https://www.youtube.com/watch?v=v1&t=120s"}],
        },
        "prices": [
            {"name": "SK하이닉스", "ticker": "000660", "current": 200000.0, "today_open": 198000.0,
             "prev_close": 195000.0, "week_ago_close": 190000.0, "month_ago_close": 180000.0,
             "change_pct": 2.56, "source": "키움/KRX"},
            {"name": "삼성전자", "ticker": "005930", "current": 80000.0, "today_open": 79500.0,
             "prev_close": 79000.0, "week_ago_close": 78000.0, "month_ago_close": 75000.0,
             "change_pct": 1.27, "source": "키움/KRX"},
        ],
        "sources": [
            {"channel": "한경TV", "title": "한경TV 브리핑", "url": "https://www.youtube.com/watch?v=v1",
             "published_at_kst": "2026-06-14T11:00:00+09:00", "n_insights": 2},
            {"channel": "삼프로TV", "title": "삼프로 라이브", "url": "https://www.youtube.com/watch?v=v2",
             "published_at_kst": "2026-06-14T20:00:00+09:00", "n_insights": 1},
        ],
        "en": {
            "overview": "This window's key summary.",
            "summary": [
                {"stock": "SK하이닉스", "ticker": "000660", "text": "Strong HBM demand."},
                {"stock": "삼성전자", "ticker": "005930", "text": "Foundry recovery."},
            ],
        },
    }


def _make_files(tmp_path):
    """Create the 4 already-rendered artifacts and return the {docx_ko/pdf_ko/docx_en/pdf_en} map."""
    names = {"docx_ko": "youtube_report_20260615_0650_ko.docx",
             "pdf_ko": "youtube_report_20260615_0650_ko.pdf",
             "docx_en": "youtube_report_20260615_0650_en.docx",
             "pdf_en": "youtube_report_20260615_0650_en.pdf"}
    files = {}
    for key, name in names.items():
        p = pathlib.Path(tmp_path) / name
        p.write_bytes(key.encode())
        files[key] = str(p)
    return files


class _FakeResp:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._payload = payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class _FakeSession:
    """Captures every .post (URL/headers/body) instead of hitting the network."""

    def __init__(self, insert_payload=None):
        self.calls = []
        self._insert_payload = insert_payload if insert_payload is not None else [{"id": 123}]

    def post(self, url, headers=None, data=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers or {}, "data": data, "json": json})
        if "/rest/v1/" in url:
            return _FakeResp(201, self._insert_payload)
        return _FakeResp(200, {"Key": url.split("/object/", 1)[-1]})

    def storage_calls(self):
        return [c for c in self.calls if "/storage/v1/object/" in c["url"]
                and "/object/public/" not in c["url"]]

    def rest_calls(self):
        return [c for c in self.calls if "/rest/v1/" in c["url"]]


# --------------------------------------------------------------------------- #
# 1) the full push assembles the upload + insert payloads
# --------------------------------------------------------------------------- #
def test_push_uploads_four_files_and_inserts_row(tmp_path):
    sess = _FakeSession()
    client = SupabaseClient("https://proj.supabase.co/", "service-key-secret", session=sess)
    status = push_report(_sample_report(), _make_files(tmp_path), client=client, date_str="2026-06-15")

    assert status["pushed"] is True
    storage, rest = sess.storage_calls(), sess.rest_calls()
    assert len(storage) == 4 and len(rest) == 1                      # 4 uploads + 1 row

    # every storage upload: <date>/<filename> path, service-role auth, upsert, raw bytes (not json)
    for c in storage:
        assert c["url"].startswith(
            "https://proj.supabase.co/storage/v1/object/youtube-reports/2026-06-15/")
        assert c["headers"]["apikey"] == "service-key-secret"
        assert c["headers"]["Authorization"] == "Bearer service-key-secret"
        assert c["headers"]["x-upsert"] == "true"
        assert isinstance(c["data"], (bytes, bytearray)) and c["json"] is None
    cts = {c["url"].rsplit(".", 1)[-1]: c["headers"]["Content-Type"] for c in storage}
    assert cts["pdf"] == "application/pdf"
    assert cts["docx"] == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

    # the 4 public URLs returned (and embedded in the row)
    fu = status["files"]
    assert fu["docx_ko_url"] == ("https://proj.supabase.co/storage/v1/object/public/"
                                 "youtube-reports/2026-06-15/youtube_report_20260615_0650_ko.docx")
    assert fu["pdf_en_url"].endswith("/youtube-reports/2026-06-15/youtube_report_20260615_0650_en.pdf")

    # the inserted orch_reports row
    r = rest[0]
    assert r["url"] == "https://proj.supabase.co/rest/v1/orch_reports"
    assert r["headers"]["Prefer"] == "return=representation"
    assert r["data"] is None                                          # JSON body, not raw bytes
    body = r["json"]
    assert body["report_type"] == "youtube_report" and body["delivery_channel"] == "gpu_youtube"
    cj = body["content_json"]
    assert cj["period"] == "daily"
    assert cj["generated_at_kst"] == "2026-06-15T06:50:00+09:00"
    assert cj["window"] == {"start": "2026-06-14T00:00:00+09:00", "end": "2026-06-15T06:50:00+09:00"}
    assert cj["email_subject"] == "유튜브 시장 리포트 — 2026-06-15"
    assert "오늘의 핵심" in cj["email_body_ko"]
    assert cj["files"] == fu                                          # same 4 URLs embedded in the row
    assert cj["report"]["name"] == "YouTube Market Analysis (grounded)"
    assert cj["report"]["summary_ko"] == "이번 구간 핵심 요약입니다."
    assert cj["report"]["summary_en"] == "This window's key summary."
    assert status["row"] == {"id": 123}                               # returned representation


# --------------------------------------------------------------------------- #
# 2) content_json builders (pure)
# --------------------------------------------------------------------------- #
def test_build_rows_joins_prices_and_insights():
    by = {r["t"]: r for r in build_rows(_sample_report())}
    assert set(by) == {"000660", "005930", "NVDA"}                    # summary + recs union
    s = by["005930"]
    assert (s["close"], s["open"], s["prev_close"], s["change_pct"]) == (80000.0, 79500.0, 79000.0, 1.27)
    assert s["ko"] == "삼성전자" and s["grounded_summary"] == "파운드리 회복."
    h = by["000660"]
    assert h["action"] == "BUY" and h["source_channel"] == "한경TV" and h["timestamp"] == "02:00"
    assert h["top_quote"].startswith("SK하이닉스가 엔비디아")
    # a recommended-but-unpriced linked global: present, price None (never invented), English name mapped
    n = by["NVDA"]
    assert n["close"] is None and n["change_pct"] is None
    assert n["action"] == "관심(WATCH)" and n["en"] == "NVIDIA"


def test_build_recommendations_and_sources():
    recs = build_recommendations(_sample_report())
    assert [r["ticker"] for r in recs] == ["000660", "NVDA"]
    assert recs[0]["action"] == "BUY" and recs[0]["channel"] == "한경TV"
    assert recs[0]["quote"].startswith("SK하이닉스가") and recs[0]["reason"] == "HBM 공급계약."
    srcs = build_sources(_sample_report())
    assert srcs[0] == {"channel": "한경TV", "title": "한경TV 브리핑",
                       "url": "https://www.youtube.com/watch?v=v1",
                       "published_at": "2026-06-14T11:00:00+09:00", "n_insights": 2}


def test_build_content_json_shape_with_explicit_urls():
    urls = {"docx_ko_url": "A", "pdf_ko_url": "B", "docx_en_url": "C", "pdf_en_url": "D"}
    cj = build_content_json(_sample_report(), urls, subject="SUBJ", body_ko="BODY")
    assert cj["email_subject"] == "SUBJ" and cj["email_body_ko"] == "BODY"
    assert cj["files"] == urls
    assert set(cj["report"]) == {"name", "rows", "recommendations", "summary_ko", "summary_en", "sources"}
    assert len(cj["report"]["rows"]) == 3 and len(cj["report"]["sources"]) == 2


def test_storage_path():
    assert storage_path("2026-06-15", "a_ko.docx") == "2026-06-15/a_ko.docx"


# --------------------------------------------------------------------------- #
# 3) subject + body match the email EXACTLY (same pure functions)
# --------------------------------------------------------------------------- #
def test_subject_and_body_match_the_email_exactly(tmp_path):
    from tagent.news.email_report import build_email_message, core_takeaways, report_subject
    rep = _sample_report()
    sess = _FakeSession()
    client = SupabaseClient("https://proj.supabase.co", "k", session=sess)
    cj = push_report(rep, _make_files(tmp_path), client=client, date_str="2026-06-15")["content_json"]

    assert cj["email_subject"] == report_subject(rep, date_str="2026-06-15")
    assert cj["email_body_ko"] == core_takeaways(rep)
    msg = build_email_message(rep, [], sender="me@x.com", recipient="b@y.com", date_str="2026-06-15")
    assert cj["email_subject"] == msg["Subject"]                      # identical subject
    body = msg.get_body(preferencelist=("plain",)).get_content()
    assert cj["email_body_ko"].strip() == body.strip()               # identical Korean body


# --------------------------------------------------------------------------- #
# 4) missing creds -> skip cleanly (no network), never crash
# --------------------------------------------------------------------------- #
class _NoCredSettings:
    supabase_url = ""
    supabase_key = ""


def test_missing_creds_skips_cleanly(monkeypatch, tmp_path):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    status = push_report(_sample_report(), _make_files(tmp_path), settings=_NoCredSettings(),
                         date_str="2026-06-15")
    assert status == {"pushed": False, "reason": "no-creds"}          # nothing uploaded/inserted


# --------------------------------------------------------------------------- #
# 5) the service-role key is NEVER leaked into a log/repr/return value
# --------------------------------------------------------------------------- #
def test_service_role_key_never_leaked(tmp_path):
    sess = _FakeSession()
    client = SupabaseClient("https://proj.supabase.co", "TOP-SECRET-KEY", session=sess)
    assert "TOP-SECRET-KEY" not in repr(client)                       # repr hides the key
    status = push_report(_sample_report(), _make_files(tmp_path), client=client, date_str="2026-06-15")
    assert "TOP-SECRET-KEY" not in json.dumps(status, ensure_ascii=False)   # not in the returned status


# --------------------------------------------------------------------------- #
# 6) run_daily_email wiring — flags + best-effort behaviour (email path unchanged)
# --------------------------------------------------------------------------- #
def test_maybe_push_respects_no_supabase():
    calls = []
    status = rde.maybe_supabase_push(_sample_report(), {}, date_str="2026-06-15", subject="S",
                                     no_supabase=True, push_fn=lambda *a, **k: calls.append(1))
    assert status == {"pushed": False, "reason": "disabled"} and not calls   # push never invoked


def test_maybe_push_forwards_files_and_subject():
    captured = {}

    def _fake_push(report, files, *, date_str, subject):
        captured.update(report=report, files=files, date_str=date_str, subject=subject)
        return {"pushed": True, "files": {"docx_ko_url": "u"}}

    files = {"docx_ko": "a.docx", "pdf_ko": "a.pdf", "docx_en": "b.docx", "pdf_en": "b.pdf"}
    status = rde.maybe_supabase_push(_sample_report(), files, date_str="2026-06-15", subject="SUBJ",
                                     push_fn=_fake_push)
    assert status["pushed"] is True
    assert captured["files"] == files and captured["subject"] == "SUBJ"
    assert captured["date_str"] == "2026-06-15"


def test_maybe_push_swallows_errors_so_the_daily_run_never_fails():
    def _boom(*a, **k):
        raise RuntimeError("network down")
    status = rde.maybe_supabase_push(_sample_report(), {}, date_str="d", subject="s", push_fn=_boom)
    assert status == {"pushed": False, "reason": "error"}


# --------------------------------------------------------------------------- #
# 7) the email path is UNCHANGED — deliver() still sends KO+EN, now also exposing the files
# --------------------------------------------------------------------------- #
class _Settings:
    gmail_user = "me@gmail.com"
    gmail_app_password = "app-secret-pw"
    boss_email = "boss@corp.com"

    def has_email_creds(self):
        return True


def _fake_render(report, lang="ko", out_dir="."):
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    docx = out / f"youtube_report_{lang}.docx"; docx.write_bytes(b"DOCX-" + lang.encode())
    pdf = out / f"youtube_report_{lang}.pdf"; pdf.write_bytes(b"PDF-" + lang.encode())
    return {"docx": str(docx), "pdf": str(pdf)}


def test_deliver_still_emails_and_now_exposes_rendered_files(tmp_path):
    sent = []
    status = rde.deliver(_sample_report(), _Settings(), out_dir=str(tmp_path), date_str="2026-06-15",
                         render_fn=_fake_render, send_fn=lambda m: sent.append(m))
    assert status["sent"] is True and len(sent) == 1                  # email still sent (unchanged)
    names = sorted(p.get_filename() for p in sent[0].iter_attachments())
    assert names == ["youtube_report_en.docx", "youtube_report_en.pdf",
                     "youtube_report_ko.docx", "youtube_report_ko.pdf"]
    # NEW: the same rendered files are exposed for the additive push
    assert set(status["files"]) == {"docx_ko", "pdf_ko", "docx_en", "pdf_en"}
    assert status["files"]["docx_ko"].endswith("youtube_report_ko.docx")
