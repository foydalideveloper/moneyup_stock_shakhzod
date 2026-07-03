"""YouTube fetch AUDIT LOG — verifiable proof of exactly what was pulled.

On each fetch the raw listing (channel, video id + title + publishedAt) and the fetched transcript
(segments with start times) are saved to ``data/youtube_fetch_log/<date>/fetch.jsonl`` (UTF-8, one
line per video). Each persisted insight is tagged with a ``fetch_log_ref`` (video id + the transcript
span used + the log path), so every insight is traceable back to its raw source. ``fetch_log_view``
summarises which videos were fetched, when, how many transcript segments, and which insights came
from them. Pure / no network.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from tagent.config import DATA_DIR

AUDIT_ROOT = "youtube_fetch_log"


def _as_date(d):
    if d is None:
        return datetime.now(timezone.utc).date()
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    s = "".join(ch for ch in str(d) if ch.isdigit())[:8]
    try:
        return datetime.strptime(s, "%Y%m%d").date()
    except ValueError:
        return datetime.now(timezone.utc).date()


def audit_dir(d=None, data_dir=None) -> Path:
    base = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
    return base / AUDIT_ROOT / str(_as_date(d))


def fetch_log_path(d=None, data_dir=None) -> Path:
    return audit_dir(d, data_dir) / "fetch.jsonl"


def load_fetch_log(d=None, data_dir=None) -> List[dict]:
    p = fetch_log_path(d, data_dir)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def iter_log_dates(data_dir=None) -> List[str]:
    """All ``youtube_fetch_log/<date>`` folder names present (sorted ascending)."""
    base = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
    root = base / AUDIT_ROOT
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def latest_entries_by_video(data_dir=None) -> Dict[str, dict]:
    """Newest log entry per video_id across ALL date folders (by ``fetched_at``) — so callers (the
    batch transcriber) can SKIP re-transcribing a video already captured. Append-only friendly: the
    later/newest entry wins, matching how the report builder dedupes."""
    best: Dict[str, dict] = {}
    for d in iter_log_dates(data_dir):
        for e in load_fetch_log(d, data_dir):
            vid = e.get("video_id")
            if not vid:
                continue
            prev = best.get(vid)
            if prev is None or str(e.get("fetched_at", "")) >= str(prev.get("fetched_at", "")):
                best[vid] = e
    return best


def load_insights_from_log(d=None, data_dir=None) -> List[dict]:
    """All grounded insights already logged for a date, flattened newest/most-important first — so a
    consumer (e.g. the daily briefing) can populate from the batch runner's output WITHOUT re-fetching
    transcripts. Each is a render-ready claim dict (stock, summary, quote, deeplink, …)."""
    rows: List[dict] = []
    for e in load_fetch_log(d, data_dir):
        for ins in e.get("insights", []):
            rows.append(ins)
    rows.sort(key=lambda r: (str(r.get("published_at", "")), r.get("importance", 0.0)), reverse=True)
    return rows


def _ref(date_str: str, video_id: str, ins: dict) -> dict:
    """The fetch-log reference stamped onto an insight: video id + the transcript span used."""
    return {"video_id": video_id, "date": date_str,
            "log": f"{AUDIT_ROOT}/{date_str}/fetch.jsonl",
            "span_start": ins.get("start"), "span_quote": str(ins.get("quote", ""))[:240]}


class AuditWriter:
    """Persists each fetched video's raw listing + transcript + insights once per day, and tags
    every insight with its ``fetch_log_ref``. Reloads the day's seen video ids on construction so a
    video isn't re-logged on every poll (the raw transcript is the proof — logged once)."""

    def __init__(self, date=None, data_dir=None, now=None):
        self.date = _as_date(date)
        self.date_str = str(self.date)
        self.data_dir = data_dir
        self._now = now
        self._seen = {e.get("video_id") for e in load_fetch_log(self.date, data_dir)}

    def _fetched_at(self) -> str:
        if self._now is None:
            return datetime.now(timezone.utc).isoformat()
        return self._now() if callable(self._now) else str(self._now)

    def record(self, meta: dict, channel: str, segments, insights, method: str = "",
               force: bool = False) -> None:
        vid = meta.get("video_id", "")
        for ins in (insights or []):                       # tag EVERY insight with its fetch-log ref
            ins["fetch_log_ref"] = _ref(self.date_str, vid, ins)
        # log each raw source once/day — EXCEPT a forced re-extract, which appends a fresh entry so the
        # newer insights (read back by latest fetched_at) supersede the stale ones.
        if not vid or (vid in self._seen and not force):
            return
        self._seen.add(vid)

        def _seg(s):                                       # keep the whisper label (machine-transcribed)
            e = {"start": s.get("start"), "text": s.get("text", "")}
            if s.get("source"):
                e["source"] = s["source"]
            return e

        entry = {
            "video_id": vid, "channel": channel,
            "title": meta.get("video_title") or meta.get("title", ""),
            "published_at": meta.get("published_at", ""), "fetched_at": self._fetched_at(),
            "method": method,                              # which strategy got it: api/ytdlp/whisper/""
            "n_segments": len(segments or []),
            "segments": [_seg(s) for s in (segments or [])],
            # FULL insight dicts (render-ready) so the daily briefing can replay them from the log,
            # minus the self-referential fetch_log_ref (it just points back here)
            "insights": [{k: v for k, v in i.items() if k != "fetch_log_ref"} for i in (insights or [])],
        }
        p = fetch_log_path(self.date, self.data_dir)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:          # UTF-8 so Korean titles/transcripts read cleanly
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def fetch_log_view(date=None, data_dir=None) -> dict:
    """Summary for the dashboard: which videos were fetched, when, how many transcript segments, and
    which insights came from them."""
    entries = load_fetch_log(date, data_dir)
    videos = []
    for e in entries:
        segs = e.get("segments", [])
        videos.append({
            "video_id": e.get("video_id"), "channel": e.get("channel", ""), "title": e.get("title", ""),
            "published_at": e.get("published_at", ""), "fetched_at": e.get("fetched_at", ""),
            "method": e.get("method", ""),                  # api / ytdlp / whisper
            "n_segments": e.get("n_segments", 0), "n_insights": len(e.get("insights", [])),
            "n_whisper": sum(1 for s in segs if s.get("source") == "whisper"),
            "url": f"https://www.youtube.com/watch?v={e.get('video_id')}" if e.get("video_id") else "",
            "insights": [{"stock": i.get("stock"), "summary": i.get("summary", ""),
                          "deeplink": i.get("deeplink", "")} for i in e.get("insights", [])],
        })
    return {"enabled": bool(videos), "date": str(_as_date(date)), "n_videos": len(videos),
            "n_insights": sum(v["n_insights"] for v in videos), "videos": videos,
            "note": "" if videos else "no YouTube fetches logged yet today"}
