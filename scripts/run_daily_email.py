"""Daily: refresh the window -> build the grounded WINDOW report -> render KO+EN -> email to the boss.

Intended for a ~6:50 AM scheduled run. The incremental batch transcribes ONLY new videos (cached ones
are skipped), the report is built ONCE (with the EN mirror) and rendered to KO + EN .docx/.pdf, then
emailed to BOSS_EMAIL via Gmail SMTP with a Korean-first '오늘의 핵심' body. An empty (0-insight) report
is logged and NOT sent. ``--dry-run`` builds + renders + writes a .eml WITHOUT sending. The Gmail app
password is never printed.

Usage:
    python scripts/run_daily_email.py
    python scripts/run_daily_email.py --dry-run
    python scripts/run_daily_email.py --no-refresh        # skip the batch, build from existing logs
"""

import argparse
import datetime
import math
import pathlib
import sys
from zoneinfo import ZoneInfo

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from tagent.config import DATA_DIR, SETTINGS              # noqa: E402
from tagent.news.email_report import (                    # noqa: E402
    EmailError, build_email_message, report_subject, send_message, write_eml,
)
from tagent.news.report_render import render_report        # noqa: E402

KST = ZoneInfo("Asia/Seoul")


def _translate_fn():
    """KO->EN BATCH translator on the fast model when a Gemini key is set, else None."""
    if not SETTINGS.has_gemini_key():
        return None
    try:
        from tagent.gemini import GeminiClient, translate_batch_fn
        model = getattr(SETTINGS, "gemini_interactive_model", None) or SETTINGS.gemini_model
        return translate_batch_fn(GeminiClient(SETTINGS.gemini_api_key, model=model))
    except Exception:
        return None


def refresh_window() -> None:
    """Incremental batch: transcribe only NEW videos published since yesterday 00:00 KST. Best-effort —
    on any problem it warns and the report is still built from whatever is already logged."""
    if not SETTINGS.has_youtube_key():
        print("[refresh] YOUTUBE_API_KEY not set — skipping; building from existing logs.")
        return
    try:
        from tagent.gemini import build_extractor
        from tagent.news.youtube_source import YouTubeSource, channels_from_env
        from tagent.youtube_audit import AuditWriter
        channels = [c for c in channels_from_env(SETTINGS.youtube_channels) if c.channel_id]
        if not channels:
            print("[refresh] no channels configured — skipping.")
            return
        now = datetime.datetime.now(KST)
        y = now - datetime.timedelta(days=1)
        since = datetime.datetime(y.year, y.month, y.day, tzinfo=KST)        # yesterday 00:00 KST
        lookback = max(1, math.ceil((now - since).total_seconds() / 3600))
        src = YouTubeSource(SETTINGS.youtube_api_key, extract_fn=build_extractor(SETTINGS),
                            enable_fallbacks=True, proxy_url=SETTINGS.youtube_proxy_url,
                            webshare=SETTINGS.webshare_proxy(),
                            pace_seconds=SETTINGS.youtube_transcript_pace_seconds)
        reports = src.batch_transcribe(channels, lookback_hours=lookback, max_per_channel=8,
                                       audit=AuditWriter())
        n_new = sum(1 for r in reports if not str(r.get("status", "")).startswith("skipped"))
        print(f"[refresh] {len(reports)} videos in window · {n_new} newly transcribed (cached skipped).")
    except Exception as e:
        print(f"[refresh] warning: refresh failed ({type(e).__name__}) — building from existing logs.")


def deliver(report: dict, settings, *, out_dir=None, dry_run=False, date_str=None,
            render_fn=None, send_fn=None) -> dict:
    """Render KO+EN and email the report (or write a .eml on dry-run). Returns a status dict.
    SKIPs sending a 0-insight report. ``render_fn``/``send_fn`` are injectable for tests."""
    render_fn = render_fn or render_report
    out = pathlib.Path(out_dir) if out_dir else (pathlib.Path(DATA_DIR) / "reports")
    date_str = date_str or datetime.datetime.now(KST).strftime("%Y-%m-%d")

    n_insights = int(report.get("meta", {}).get("n_insights", 0) or 0)
    if n_insights == 0:
        print("[skip] report has 0 grounded insights — NOT sending an empty report.")
        return {"sent": False, "reason": "empty"}

    if not dry_run and not settings.has_email_creds():
        print("[error] missing email credentials — set GMAIL_USER, GMAIL_APP_PASSWORD and BOSS_EMAIL "
              "in .env (nothing sent).")
        return {"sent": False, "reason": "no-creds"}

    ko = render_fn(report, lang="ko", out_dir=str(out))
    en = render_fn(report, lang="en", out_dir=str(out))
    attachments = [ko["docx"], ko["pdf"], en["docx"], en["pdf"]]
    # the rendered artifacts, keyed for the additive Supabase push (re-uses these exact files)
    files = {"docx_ko": ko["docx"], "pdf_ko": ko["pdf"], "docx_en": en["docx"], "pdf_en": en["pdf"]}

    sender = settings.gmail_user or "noreply@localhost"
    recipient = settings.boss_email or "boss@unknown"
    msg = build_email_message(report, attachments, sender=sender, recipient=recipient, date_str=date_str)

    if dry_run:
        eml = write_eml(msg, out / f"daily_email_{date_str}.eml")
        print(f"[dry-run] wrote {eml} · {len(attachments)} attachments · subject: {msg['Subject']} "
              f"— NOT sent.")
        return {"sent": False, "reason": "dry-run", "eml": str(eml), "subject": msg["Subject"],
                "files": files}

    send = send_fn or (lambda m: send_message(m, user=settings.gmail_user,
                                              app_password=settings.gmail_app_password))
    send(msg)
    print(f"[ok] emailed daily report to {msg['To']} · subject: {msg['Subject']} · "
          f"{len(attachments)} attachments (KO+EN docx/pdf).")
    return {"sent": True, "to": msg["To"], "subject": msg["Subject"], "n_insights": n_insights,
            "files": files}


def maybe_supabase_push(report: dict, files: dict, *, date_str: str, subject: str,
                        no_supabase: bool = False, push_fn=None) -> dict:
    """ADDITIVE final step: push the rendered report to Supabase for the VIP Agent. Best-effort —
    a missing-creds skip or any failure is logged and swallowed (the email is already delivered, so
    the daily run never fails because of the push). The service-role key is never logged. ``push_fn``
    is injectable for tests; the default lazily imports tagent.news.supabase_push.push_report."""
    if no_supabase:
        print("[supabase] skipped (--no-supabase).")
        return {"pushed": False, "reason": "disabled"}
    if push_fn is None:
        from tagent.news.supabase_push import push_report as push_fn
    try:
        status = push_fn(report, files, date_str=date_str, subject=subject)
    except Exception as e:                                  # never crash the daily run on a push error
        print(f"[supabase] warning: push failed ({type(e).__name__}) — report unaffected.")
        return {"pushed": False, "reason": "error"}
    if status.get("pushed"):
        urls = status.get("files", {})
        print(f"[supabase] uploaded {sum(1 for v in urls.values() if v)} files + inserted "
              f"orch_reports row for the VIP Agent.")
    elif status.get("reason") == "no-creds":
        print("[supabase] SUPABASE_URL/SUPABASE_KEY not set — skipping push (email unaffected).")
    return status


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="build+render+write .eml, do NOT send")
    ap.add_argument("--no-refresh", action="store_true", help="skip the incremental batch refresh")
    ap.add_argument("--out", default=None, help="output dir (default: data/reports)")
    ap.add_argument("--no-supabase", action="store_true", help="skip the additive Supabase push")
    ap.add_argument("--supabase-only", action="store_true",
                    help="render + push to Supabase WITHOUT emailing (a safe push test)")
    args = ap.parse_args()

    if not args.no_refresh:
        refresh_window()

    from tagent.news.youtube_report import build_youtube_report
    translate = _translate_fn()                            # build ONCE with the EN mirror, render both
    report = build_youtube_report(lang="ko", translate_fn=translate, prices_fn="auto")
    m = report["meta"]
    print(f"[report] {m['n_videos']} videos · {m['n_insights']} grounded insights · "
          f"window {m['window_start']} → {m['window_end']}")

    date_str = datetime.datetime.now(KST).strftime("%Y-%m-%d")   # shared by the email + the push
    out = pathlib.Path(args.out) if args.out else (pathlib.Path(DATA_DIR) / "reports")

    # --supabase-only: render the SAME KO+EN files and push, but do NOT email (a safe test path)
    if args.supabase_only:
        if int(m.get("n_insights", 0) or 0) == 0:
            print("[skip] report has 0 grounded insights — NOT pushing an empty report.")
            return 0
        ko = render_report(report, lang="ko", out_dir=str(out))
        en = render_report(report, lang="en", out_dir=str(out))
        files = {"docx_ko": ko["docx"], "pdf_ko": ko["pdf"], "docx_en": en["docx"], "pdf_en": en["pdf"]}
        subject = report_subject(report, date_str=date_str)
        maybe_supabase_push(report, files, date_str=date_str, subject=subject)
        return 0

    try:
        status = deliver(report, SETTINGS, out_dir=str(out), dry_run=args.dry_run, date_str=date_str)
    except EmailError as e:
        print(f"[FAIL] {e}")                               # never contains the password
        return 1

    # FINAL step: push to Supabase after a real send (additive; dry-run/empty/no-creds don't push)
    if status.get("sent") and status.get("files"):
        subject = status.get("subject") or report_subject(report, date_str=date_str)
        maybe_supabase_push(report, status["files"], date_str=date_str, subject=subject,
                            no_supabase=args.no_supabase)

    if status.get("sent") or status.get("reason") in ("dry-run", "empty"):
        return 0
    return 1                                               # no-creds / other non-send -> failure exit


if __name__ == "__main__":
    raise SystemExit(main())
