"""Email the grounded daily YouTube report to the boss via Gmail SMTP.

Pure assembly (``build_email_message``, ``core_takeaways``, ``report_subject``) is separated from the
network send (``send_message``) so the email can be built + written to a ``.eml`` with NO network, and
so tests never touch SMTP. The Gmail app password is NEVER logged or echoed.
"""

from __future__ import annotations

import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path
from typing import List, Optional

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465                                              # implicit SSL


class EmailError(Exception):
    """Raised on missing credentials or an SMTP failure (message never contains the password)."""


def parse_recipients(raw) -> List[str]:
    """A list of addresses from a comma-separated string (or an iterable): trimmed, empties dropped.
    BOSS_EMAIL may be 'a@x.com, b@y.com, c@z.com' or a single address."""
    if raw is None:
        return []
    parts = raw.split(",") if isinstance(raw, str) else list(raw)
    return [a for a in (str(p).strip() for p in parts) if a]


def _mime_for(suffix: str):
    s = suffix.lower()
    if s == ".pdf":
        return "application", "pdf"
    if s == ".docx":
        return "application", "vnd.openxmlformats-officedocument.wordprocessingml.document"
    return "application", "octet-stream"


def _report_date(report: dict) -> str:
    """YYYY-MM-DD for the subject — from the report's generated_at_kst, else its window end."""
    meta = report.get("meta", {}) if isinstance(report, dict) else {}
    for key in ("generated_at_kst", "window_end"):
        val = str(meta.get(key, "") or "")
        if len(val) >= 10:
            return val[:10]
    return ""


def report_subject(report: dict, *, date_str: Optional[str] = None) -> str:
    d = date_str or _report_date(report)
    return f"유튜브 시장 리포트 — {d}".rstrip(" —")


def _n_insights(report: dict) -> int:
    return int(report.get("meta", {}).get("n_insights", 0) or 0)


def core_takeaways(report: dict, *, max_takeaways: int = 3) -> str:
    """The Korean-first '오늘의 핵심' body: top grounded takeaways + the BUY/SELL recommendation
    summary, each with its source link. Built ONLY from the grounded report (no new facts)."""
    lines: List[str] = ["오늘의 핵심", ""]

    # top grounded takeaways (the §1 per-stock synthesis carries a source deep-link)
    takeaways = [b for b in (report.get("summary") or []) if str(b.get("text", "")).strip()]
    if not takeaways and str(report.get("overview", "")).strip():
        takeaways = [{"stock": "", "text": report["overview"], "deeplink": ""}]
    lines.append("[핵심 포인트]")
    if takeaways:
        for i, b in enumerate(takeaways[:max_takeaways], 1):
            stock = str(b.get("stock", "")).strip()
            head = f"{i}. {stock} — {b['text']}" if stock else f"{i}. {b['text']}"
            lines.append(head)
            if b.get("deeplink"):
                lines.append(f"   ▶ {b['deeplink']}")
    else:
        lines.append("- (자막 근거 인사이트 없음)")
    lines.append("")

    # BUY/SELL recommendation summary (directional calls first; WATCH-only excluded)
    recs = report.get("recommendations") or []
    directional = [r for r in recs if str(r.get("action", "")) and "WATCH" not in str(r.get("action", ""))]
    lines.append("[투자의견 요약]")
    if directional:
        for r in directional:
            tp = str(r.get("target_price", "") or "").strip()
            tail = f"  · 목표가 {tp}" if tp else ""
            lines.append(f"- {r.get('stock', '')} {r.get('action', '')}{tail}")
            if r.get("deeplink"):
                lines.append(f"   ▶ {r['deeplink']}")
    else:
        lines.append("- 명확한 매수/매도 의견 없음 (관심 종목은 첨부 리포트 참고)")
    lines.append("")

    gen = str(report.get("meta", {}).get("generated_at_kst", "") or "")
    lines.append("전체 내용은 첨부된 한글/영문 리포트(.docx·.pdf)를 참고하세요.")
    if gen:
        lines.append(f"(생성: {gen})")
    return "\n".join(lines)


def build_email_message(report: dict, attachments, *, sender: str, recipient,
                        subject: Optional[str] = None, date_str: Optional[str] = None) -> EmailMessage:
    """Assemble the EmailMessage (subject + Korean-first body + .docx/.pdf attachments). No network.
    ``recipient`` may be a single address, a comma-separated string, or a list — all are set on To:."""
    msg = EmailMessage()
    msg["Subject"] = subject or report_subject(report, date_str=date_str)
    msg["From"] = sender
    msg["To"] = ", ".join(parse_recipients(recipient)) or str(recipient or "")
    msg.set_content(core_takeaways(report))
    for path in (attachments or []):
        p = Path(path)
        if not p.exists():
            continue
        maintype, subtype = _mime_for(p.suffix)
        msg.add_attachment(p.read_bytes(), maintype=maintype, subtype=subtype, filename=p.name)
    return msg


def send_message(msg: EmailMessage, *, user: str, app_password: str,
                 host: str = SMTP_HOST, port: int = SMTP_PORT) -> None:
    """Send ``msg`` over Gmail SMTP (implicit SSL). Raises EmailError on missing creds — the password
    is NEVER included in any log line or exception message."""
    if not user or not app_password:
        raise EmailError("missing Gmail credentials — set GMAIL_USER and GMAIL_APP_PASSWORD in .env")
    recipients = parse_recipients(msg.get("To", ""))         # full list -> SMTP to_addrs (not raw string)
    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL(host, port, context=context) as server:
            server.login(user, app_password)
            server.send_message(msg, to_addrs=recipients or None)
    except smtplib.SMTPException as e:                       # message scrubbed of any secret
        raise EmailError(f"SMTP send failed: {type(e).__name__}") from None


def write_eml(msg: EmailMessage, path) -> Path:
    """Write the assembled message to a .eml file (for --dry-run; no network)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(bytes(msg))
    return p
