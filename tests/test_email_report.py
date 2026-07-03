"""Daily report email — assembly, dry-run, and credential handling. Fully mocked; NO network/SMTP."""

import importlib.util
import pathlib

import pytest

import tagent.news.email_report as email_report
from tagent.news.email_report import (
    EmailError, build_email_message, core_takeaways, parse_recipients, report_subject,
    send_message, write_eml,
)

# import the runner script (scripts/ isn't a package) to test its deliver() logic
_RUNNER = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "run_daily_email.py"
_spec = importlib.util.spec_from_file_location("run_daily_email", _RUNNER)
rde = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rde)


def _sample_report(n_insights=4):
    return {
        "meta": {"n_insights": n_insights, "n_videos": 2, "lang": "ko",
                 "generated_at_kst": "2026-06-15T06:50:00+09:00",
                 "window_start": "2026-06-14T00:00:00+09:00", "window_end": "2026-06-15T06:50:00+09:00"},
        "overview": "이번 구간 핵심 요약입니다.",
        "summary": [
            {"stock": "SK하이닉스", "ticker": "000660", "text": "HBM 수요 강세로 목표가 상향 기대.",
             "deeplink": "https://www.youtube.com/watch?v=v1&t=120s"},
            {"stock": "삼성전자", "ticker": "005930", "text": "파운드리 회복 신호.",
             "deeplink": "https://www.youtube.com/watch?v=v1&t=200s"},
            {"stock": "원익QNC", "ticker": "", "text": "쿼츠 국산화 수혜.",
             "deeplink": "https://www.youtube.com/watch?v=v2&t=60s"},
            {"stock": "현대차", "ticker": "005380", "text": "환율 수혜.",
             "deeplink": "https://www.youtube.com/watch?v=v2&t=300s"},
        ],
        "recommendations": [
            {"stock": "SK하이닉스", "ticker": "000660", "action": "BUY", "target_price": "300만원",
             "reason": "HBM.", "deeplink": "https://www.youtube.com/watch?v=v1&t=120s"},
            {"stock": "삼성전기", "ticker": "009150", "action": "SELL", "target_price": "",
             "reason": "비중축소.", "deeplink": "https://www.youtube.com/watch?v=v2&t=420s"},
            {"stock": "현대차", "ticker": "005380", "action": "관심(WATCH)", "target_price": "70만원",
             "reason": "관망.", "deeplink": "https://www.youtube.com/watch?v=v2&t=300s"},
        ],
    }


class _Settings:
    def __init__(self, creds=True):
        self.gmail_user = "me@gmail.com" if creds else ""
        self.gmail_app_password = "app-secret-pw" if creds else ""
        self.boss_email = "boss@corp.com" if creds else ""

    def has_email_creds(self):
        return bool(self.gmail_user and self.gmail_app_password and self.boss_email)


def _fake_render(report, lang="ko", out_dir="."):
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    docx = out / f"youtube_report_{lang}.docx"
    pdf = out / f"youtube_report_{lang}.pdf"
    docx.write_bytes(b"DOCX-" + lang.encode())
    pdf.write_bytes(b"PDF-" + lang.encode())
    return {"docx": str(docx), "pdf": str(pdf)}


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #
def test_build_email_message_assembles_subject_body_and_attachments(tmp_path):
    a1 = tmp_path / "report_ko.docx"; a1.write_bytes(b"DOCX")
    a2 = tmp_path / "report_ko.pdf"; a2.write_bytes(b"%PDF-1.4")
    rep = _sample_report()
    msg = build_email_message(rep, [str(a1), str(a2)], sender="me@gmail.com", recipient="boss@corp.com")

    assert msg["Subject"] == "유튜브 시장 리포트 — 2026-06-15"
    assert msg["From"] == "me@gmail.com" and msg["To"] == "boss@corp.com"
    body = msg.get_body(preferencelist=("plain",)).get_content()
    assert "오늘의 핵심" in body
    assert "SK하이닉스" in body and "HBM 수요 강세" in body                 # top takeaway
    assert "https://www.youtube.com/watch?v=v1&t=120s" in body              # with its source link
    assert "BUY" in body and "SELL" in body and "300만원" in body           # directional rec summary
    assert "관심(WATCH)" not in body                                        # WATCH-only excluded
    names = sorted(p.get_filename() for p in msg.iter_attachments())
    assert names == ["report_ko.docx", "report_ko.pdf"]                     # both attached


def test_core_takeaways_caps_to_three_and_falls_back_to_overview():
    body = core_takeaways(_sample_report(), max_takeaways=3)
    assert body.count("\n1. ") == 1 and "3. " in body and "4. " not in body  # exactly top 3
    empty = {"meta": {}, "overview": "시장 개요 한 줄.", "summary": [], "recommendations": []}
    b2 = core_takeaways(empty)
    assert "시장 개요 한 줄." in b2 and "명확한 매수/매도 의견 없음" in b2


def test_report_subject_uses_window_end_when_no_generated_at():
    rep = {"meta": {"window_end": "2026-03-09T10:00:00+09:00"}}
    assert report_subject(rep) == "유튜브 시장 리포트 — 2026-03-09"


# --------------------------------------------------------------------------- #
# multiple recipients — comma-separated BOSS_EMAIL -> To header + SMTP to_addrs
# --------------------------------------------------------------------------- #
def test_parse_recipients_splits_trims_and_drops_empties():
    assert parse_recipients("a@x.com, b@y.com, c@z.com") == ["a@x.com", "b@y.com", "c@z.com"]
    assert parse_recipients(" solo@z.com ") == ["solo@z.com"]
    assert parse_recipients("a@x.com,, ,b@y.com,") == ["a@x.com", "b@y.com"]   # blanks dropped
    assert parse_recipients(["a@x.com", " b@y.com "]) == ["a@x.com", "b@y.com"]
    assert parse_recipients("") == [] and parse_recipients(None) == []


def test_build_email_message_sets_all_recipients_on_to_header():
    msg = build_email_message(_sample_report(), [], sender="me@gmail.com",
                              recipient="a@x.com, b@y.com, c@z.com")
    assert msg["To"] == "a@x.com, b@y.com, c@z.com"
    one = build_email_message(_sample_report(), [], sender="me@gmail.com", recipient="solo@z.com")
    assert one["To"] == "solo@z.com"                                          # single still works


def test_send_message_passes_full_recipient_list_to_sendmail(monkeypatch):
    captured = {}

    class _FakeServer:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def login(self, u, p): captured["login_user"] = u
        def send_message(self, msg, to_addrs=None): captured["to_addrs"] = to_addrs

    monkeypatch.setattr(email_report.smtplib, "SMTP_SSL",
                        lambda host, port, context=None: _FakeServer())

    msg = build_email_message(_sample_report(), [], sender="me@gmail.com",
                              recipient="a@x.com, b@y.com, c@z.com")
    send_message(msg, user="me@gmail.com", app_password="app-secret-pw")
    assert captured["to_addrs"] == ["a@x.com", "b@y.com", "c@z.com"]          # list, not raw string
    assert msg["To"] == "a@x.com, b@y.com, c@z.com"                           # N in the To header

    msg2 = build_email_message(_sample_report(), [], sender="me@gmail.com", recipient="solo@z.com")
    send_message(msg2, user="me@gmail.com", app_password="app-secret-pw")
    assert captured["to_addrs"] == ["solo@z.com"]                             # single address still works


# --------------------------------------------------------------------------- #
# send credentials — clear error, never a crash, never leak the password
# --------------------------------------------------------------------------- #
def test_send_message_missing_creds_raises_clear_error():
    msg = build_email_message(_sample_report(), [], sender="me@gmail.com", recipient="boss@corp.com")
    with pytest.raises(EmailError) as e:
        send_message(msg, user="", app_password="")
    assert "GMAIL_USER" in str(e.value) and "GMAIL_APP_PASSWORD" in str(e.value)
    with pytest.raises(EmailError):                                         # user set but no password
        send_message(msg, user="me@gmail.com", app_password="")


def test_assembled_message_never_contains_password(tmp_path):
    msg = build_email_message(_sample_report(), [], sender="me@gmail.com", recipient="boss@corp.com")
    assert b"app-secret-pw" not in bytes(msg)                                # password never in the email


# --------------------------------------------------------------------------- #
# runner deliver() — dry-run, empty-skip, missing creds, real send
# --------------------------------------------------------------------------- #
def test_deliver_dry_run_writes_eml_without_sending(tmp_path):
    sent = []
    status = rde.deliver(_sample_report(), _Settings(creds=True), out_dir=str(tmp_path), dry_run=True,
                         date_str="2026-06-15", render_fn=_fake_render, send_fn=lambda m: sent.append(m))
    assert status["sent"] is False and status["reason"] == "dry-run"
    eml = pathlib.Path(status["eml"])
    assert eml.exists() and eml.suffix == ".eml"
    assert not sent                                                          # nothing sent
    raw = eml.read_bytes()
    assert b"report_ko.docx" in raw and b"report_en.pdf" in raw             # KO+EN attached


def test_deliver_skips_empty_report(tmp_path):
    sent, rendered = [], []
    status = rde.deliver(_sample_report(n_insights=0), _Settings(True), out_dir=str(tmp_path),
                         render_fn=lambda *a, **k: rendered.append(1) or _fake_render(*a, **k),
                         send_fn=lambda m: sent.append(m))
    assert status == {"sent": False, "reason": "empty"}
    assert not sent and not rendered                                         # neither rendered nor sent


def test_deliver_missing_creds_is_clear_error_not_crash(tmp_path):
    sent = []
    status = rde.deliver(_sample_report(), _Settings(creds=False), out_dir=str(tmp_path),
                         render_fn=_fake_render, send_fn=lambda m: sent.append(m))
    assert status["sent"] is False and status["reason"] == "no-creds"
    assert not sent                                                          # no send attempted


def test_deliver_status_and_to_header_reflect_all_recipients(tmp_path):
    s = _Settings(creds=True)
    s.boss_email = "a@x.com, b@y.com, c@z.com"               # comma-separated BOSS_EMAIL
    sent = []
    status = rde.deliver(_sample_report(), s, out_dir=str(tmp_path), date_str="2026-06-15",
                         render_fn=_fake_render, send_fn=lambda m: sent.append(m))
    assert status["sent"] is True and status["to"] == "a@x.com, b@y.com, c@z.com"
    assert sent[0]["To"] == "a@x.com, b@y.com, c@z.com"


def test_deliver_sends_with_creds_and_attaches_ko_en(tmp_path):
    sent = []
    status = rde.deliver(_sample_report(), _Settings(creds=True), out_dir=str(tmp_path),
                         date_str="2026-06-15", render_fn=_fake_render, send_fn=lambda m: sent.append(m))
    assert status["sent"] is True and status["to"] == "boss@corp.com"
    assert len(sent) == 1
    names = sorted(p.get_filename() for p in sent[0].iter_attachments())
    assert names == ["youtube_report_en.docx", "youtube_report_en.pdf",
                     "youtube_report_ko.docx", "youtube_report_ko.pdf"]      # KO + EN docx/pdf
