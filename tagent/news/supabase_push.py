"""ADDITIVE Supabase push so the "Oasis VIP Agent" can deliver the EXACT same daily report.

This sits ALONGSIDE the existing Gmail email — it never changes the report/email logic. After the
6:50 email has rendered + sent the KO+EN ``.docx``/``.pdf``, this uploads those already-rendered files
to Supabase Storage and inserts ONE ``public.orch_reports`` row whose ``content_json`` carries the
exact email subject + Korean body + the structured grounded report. The VIP Agent reads the row,
downloads the same files, and re-delivers them byte-for-byte to the boss.

Two-step push (both via the Supabase HTTP API, service-role key in the header only — NEVER logged):
  1. Storage: ``POST /storage/v1/object/<bucket>/<date>/<filename>`` (``x-upsert: true``) for each of
     the 4 files; the public object URL is ``/storage/v1/object/public/<bucket>/<date>/<filename>``.
  2. REST:    ``POST /rest/v1/orch_reports`` (``Prefer: return=representation``) inserting one row:
     ``{report_type:"youtube_report", delivery_channel:"gpu_youtube", content_json:{...}}``.

Missing ``SUPABASE_URL`` / ``SUPABASE_KEY`` -> skip cleanly (return a no-creds status, never crash).
``requests`` is imported lazily and the HTTP client is injectable, so the unit tests run fully mocked
(no network). The subject + Korean body are recomputed from the SAME pure functions the email uses
(:func:`report_subject` / :func:`core_takeaways`), so they match the delivered email exactly.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

from tagent.config import SETTINGS
from tagent.news.email_report import core_takeaways, report_subject
from tagent.news.youtube_report import LINKED_GLOBALS

_KST = timezone(timedelta(hours=9))

BUCKET = "youtube-reports"
TABLE = "orch_reports"
REPORT_TYPE = "youtube_report"
DELIVERY_CHANNEL = "gpu_youtube"
REPORT_NAME = "YouTube Market Analysis (grounded)"

# content-types for the rendered artifacts (mirrors the email's MIME mapping)
_CONTENT_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


def _content_type(suffix: str) -> str:
    return _CONTENT_TYPES.get(str(suffix).lower(), "application/octet-stream")


def _report_date(report: dict, *, now=None) -> str:
    """YYYY-MM-DD for the storage folder — the report's generated_at_kst, else its window end, else
    today (KST). Mirrors how the email derives its date so files land under the same folder."""
    meta = report.get("meta", {}) if isinstance(report, dict) else {}
    for key in ("generated_at_kst", "window_end"):
        val = str(meta.get(key, "") or "")
        if len(val) >= 10:
            return val[:10]
    return (now or datetime.now(_KST)).astimezone(_KST).strftime("%Y-%m-%d")


def storage_path(date_str: str, filename: str) -> str:
    """The in-bucket object path: ``<date>/<filename>`` (no leading slash)."""
    return f"{date_str}/{filename}"


# --------------------------------------------------------------------------- #
# content_json builders (pure — assembled straight from the grounded report)
# --------------------------------------------------------------------------- #
def build_rows(report: dict) -> List[dict]:
    """One combined per-stock row joining the grounded insights with the REAL price table, keyed by
    ticker. ``t``=ticker, ``ko``/``en``=stock name (English from the linked-global map, else the
    Korean name as a safe fallback — giant Korean names aren't translated), price fields from the
    §4 table (None when unfetched — never invented), and the strongest grounded call/quote/link.
    Ordered by importance: §1 summary stocks first, then any remaining recommended / priced stocks."""
    prices = {str(p.get("ticker") or ""): p for p in (report.get("prices") or []) if p.get("ticker")}
    recs = {str(r.get("ticker") or ""): r for r in (report.get("recommendations") or []) if r.get("ticker")}
    ko_sum = {str(b.get("ticker") or ""): b for b in (report.get("summary") or []) if b.get("ticker")}

    # strongest verbatim quote per ticker (per_stock is keyed by NAME, but each insight carries ticker)
    quote_by_ticker: Dict[str, str] = {}
    for items in (report.get("per_stock") or {}).values():
        for it in items:
            t = str(it.get("ticker") or it.get("stock") or "")
            if t and t not in quote_by_ticker and it.get("quote"):
                quote_by_ticker[t] = it.get("quote", "")

    order: List[str] = []
    seen = set()
    for group in (ko_sum, recs, prices):
        for t in group:
            if t and t not in seen:
                seen.add(t)
                order.append(t)

    rows: List[dict] = []
    for t in order:
        p = prices.get(t, {})
        r = recs.get(t, {})
        b = ko_sum.get(t, {})
        ko_name = b.get("stock") or p.get("name") or r.get("stock") or t
        rows.append({
            "t": t,
            "ko": ko_name,
            "en": LINKED_GLOBALS.get(t) or ko_name,
            "close": p.get("current"),
            "open": p.get("today_open"),
            "prev_close": p.get("prev_close"),
            "change_pct": p.get("change_pct"),
            "action": r.get("action", ""),
            "grounded_summary": b.get("text") or r.get("reason") or "",
            "top_quote": r.get("quote") or quote_by_ticker.get(t, ""),
            "timestamp": b.get("timestamp") or r.get("timestamp") or "",
            "deeplink": b.get("deeplink") or r.get("deeplink") or "",
            "source_channel": b.get("channel") or r.get("channel") or "",
        })
    return rows


def build_recommendations(report: dict) -> List[dict]:
    """The §2 recommendations flattened to the VIP-Agent shape."""
    out: List[dict] = []
    for r in report.get("recommendations") or []:
        out.append({
            "stock": r.get("stock", ""),
            "ticker": r.get("ticker", ""),
            "action": r.get("action", ""),
            "reason": r.get("reason", ""),
            "quote": r.get("quote", ""),
            "channel": r.get("channel", ""),
            "timestamp": r.get("timestamp", "") or r.get("timestamp_mmss", ""),
            "deeplink": r.get("deeplink", ""),
        })
    return out


def build_sources(report: dict) -> List[dict]:
    """The §7 source videos flattened to the VIP-Agent shape."""
    out: List[dict] = []
    for s in report.get("sources") or []:
        out.append({
            "channel": s.get("channel", ""),
            "title": s.get("title", ""),
            "url": s.get("url", ""),
            "published_at": s.get("published_at_kst", "") or s.get("published_at", ""),
            "n_insights": s.get("n_insights", 0),
        })
    return out


def build_content_json(report: dict, file_urls: Dict[str, str], *, subject: str,
                       body_ko: str) -> dict:
    """Assemble the ``content_json`` payload (the exact email subject + Korean body + the structured
    grounded report + the 4 Storage URLs). ``file_urls`` keys: docx_ko_url/pdf_ko_url/docx_en_url/
    pdf_en_url."""
    meta = report.get("meta", {}) if isinstance(report, dict) else {}
    en = report.get("en") or {}
    return {
        "period": "daily",
        "generated_at_kst": meta.get("generated_at_kst", ""),
        "window": {"start": meta.get("window_start", ""), "end": meta.get("window_end", "")},
        "email_subject": subject,
        "email_body_ko": body_ko,
        "files": {
            "docx_ko_url": file_urls.get("docx_ko_url", ""),
            "pdf_ko_url": file_urls.get("pdf_ko_url", ""),
            "docx_en_url": file_urls.get("docx_en_url", ""),
            "pdf_en_url": file_urls.get("pdf_en_url", ""),
        },
        "report": {
            "name": REPORT_NAME,
            "rows": build_rows(report),
            "recommendations": build_recommendations(report),
            "summary_ko": report.get("overview", "") or "",
            "summary_en": en.get("overview", "") or "",
            "sources": build_sources(report),
        },
    }


# --------------------------------------------------------------------------- #
# Supabase HTTP client (service-role key only in headers — NEVER logged)
# --------------------------------------------------------------------------- #
def _requests():
    import requests  # lazy: keeps import light + offline-safe; tests inject a fake session
    return requests


class SupabaseClient:
    """Thin Supabase Storage + REST client. The service-role ``key`` rides only in request headers
    (apikey / Authorization) and is never printed. ``session`` (a requests-like object exposing
    ``.post``) is injectable for tests; the default is a real ``requests`` session."""

    def __init__(self, url: str, key: str, *, bucket: str = BUCKET, session=None, timeout: int = 30):
        self.url = str(url or "").rstrip("/")
        self._key = key
        self.bucket = bucket
        self.timeout = timeout
        self.session = session or _requests().Session()

    def __repr__(self) -> str:                                # never leak the key in logs/tracebacks
        return f"<SupabaseClient url={self.url!r} bucket={self.bucket!r}>"

    def _headers(self, extra: Optional[dict] = None) -> dict:
        h = {"apikey": self._key, "Authorization": f"Bearer {self._key}"}
        if extra:
            h.update(extra)
        return h

    def public_url(self, path: str) -> str:
        """The public download URL for an uploaded object."""
        return f"{self.url}/storage/v1/object/public/{self.bucket}/{path}"

    def upload_file(self, path: str, data: bytes, content_type: str) -> str:
        """Upload (upsert) ``data`` to ``<bucket>/<path>`` and return its public URL."""
        endpoint = f"{self.url}/storage/v1/object/{self.bucket}/{path}"
        resp = self.session.post(
            endpoint,
            headers=self._headers({"Content-Type": content_type, "x-upsert": "true"}),
            data=data, timeout=self.timeout)
        resp.raise_for_status()
        return self.public_url(path)

    def insert_row(self, table: str, row: dict) -> dict:
        """Insert one row, returning the inserted representation (or {} if none returned)."""
        endpoint = f"{self.url}/rest/v1/{table}"
        resp = self.session.post(
            endpoint,
            headers=self._headers({"Content-Type": "application/json",
                                   "Prefer": "return=representation"}),
            json=row, timeout=self.timeout)
        resp.raise_for_status()
        try:
            data = resp.json()
        except Exception:
            data = None
        if isinstance(data, list):
            return data[0] if data else {}
        return data or {}


# --------------------------------------------------------------------------- #
# the push (orchestrates upload + insert; skips cleanly without creds)
# --------------------------------------------------------------------------- #
# local file path -> the content_json["files"] URL key
_FILE_URL_KEYS = (("docx_ko", "docx_ko_url"), ("pdf_ko", "pdf_ko_url"),
                  ("docx_en", "docx_en_url"), ("pdf_en", "pdf_en_url"))


def push_report(report: dict, files: Dict[str, str], *, settings=None, client: Optional[SupabaseClient] = None,
                date_str: Optional[str] = None, subject: Optional[str] = None,
                body_ko: Optional[str] = None, now=None) -> dict:
    """Upload the 4 already-rendered files + insert one ``orch_reports`` row. ADDITIVE: never raises
    on missing creds — returns ``{"pushed": False, "reason": "no-creds"}`` instead.

    ``files`` maps docx_ko/pdf_ko/docx_en/pdf_en -> local file paths (the email's rendered artifacts).
    ``subject``/``body_ko`` default to the SAME pure functions the email uses, so they match exactly.
    ``client`` is injectable for tests; otherwise a real :class:`SupabaseClient` is built from creds.
    """
    settings = settings if settings is not None else SETTINGS
    url = getattr(settings, "supabase_url", "") or os.getenv("SUPABASE_URL", "")
    key = getattr(settings, "supabase_key", "") or os.getenv("SUPABASE_KEY", "")
    if client is None and (not url or not key):
        return {"pushed": False, "reason": "no-creds"}

    date_str = date_str or _report_date(report, now=now)
    subject = subject if subject is not None else report_subject(report, date_str=date_str)
    body_ko = body_ko if body_ko is not None else core_takeaways(report)
    client = client or SupabaseClient(url, key)

    file_urls: Dict[str, str] = {}
    for local_key, url_key in _FILE_URL_KEYS:
        local = (files or {}).get(local_key)
        if not local:
            file_urls[url_key] = ""
            continue
        p = Path(local)
        path = storage_path(date_str, p.name)
        file_urls[url_key] = client.upload_file(path, p.read_bytes(), _content_type(p.suffix))

    content_json = build_content_json(report, file_urls, subject=subject, body_ko=body_ko)
    row = {"report_type": REPORT_TYPE, "delivery_channel": DELIVERY_CHANNEL, "content_json": content_json}
    inserted = client.insert_row(TABLE, row)
    return {"pushed": True, "files": file_urls, "row": inserted, "content_json": content_json}
