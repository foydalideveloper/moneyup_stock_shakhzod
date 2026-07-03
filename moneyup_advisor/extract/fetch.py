"""yt-dlp: list recent 머니업 videos (with publish datetime) + download video+audio.

Reuses ``tagent.news.youtube_source._ytdlp_argv`` (READ-ONLY import) to build the
``python -m yt_dlp`` argv, then applies the Step-0-proven non-DRM / un-throttled flags
(``player_client=android_vr`` + ``-N 8`` concurrent fragments). Downloads the h264 1080p
video (fmt 137 — best for OCR) and m4a audio (fmt 140). Nothing here touches the daily pipeline.
"""
from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone
from typing import Dict, List, Optional

from moneyup_advisor import config


def _yt(*args, timeout: int = 240) -> str:
    """Run ``python -m yt_dlp <args>`` with UTF-8 forced; return stdout (stderr captured/ignored)."""
    from tagent.news.youtube_source import _ytdlp_argv          # READ-ONLY reuse
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run(_ytdlp_argv(*args), capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env, timeout=timeout,
                       creationflags=config.CREATE_NO_WINDOW)
    return r.stdout or ""


def _client_args() -> List[str]:
    return ["--extractor-args", f"youtube:player_client={config.YTDLP_PLAYER_CLIENT}"]


def _parse_dt(upload_date: str, ts: str) -> tuple:
    """(publish_date 'YYYY-MM-DD', publish_datetime ISO-Z or None) from yt-dlp fields."""
    pub_date, pub_dt = "", None
    if ts and ts.strip().isdigit():
        dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
        pub_dt = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        pub_date = dt.strftime("%Y-%m-%d")
    elif upload_date and upload_date.strip().isdigit() and len(upload_date.strip()) == 8:
        u = upload_date.strip()
        pub_date = f"{u[0:4]}-{u[4:6]}-{u[6:8]}"
    return pub_date, pub_dt


def video_meta(video_id: str) -> Optional[Dict]:
    """Fetch one video's metadata (id, title, publish datetime, duration). None on failure."""
    meta = _yt(f"https://www.youtube.com/watch?v={video_id}", "--skip-download", "--no-warnings",
               "--print", "%(id)s\t%(upload_date)s\t%(timestamp)s\t%(duration)s\t%(title)s",
               timeout=120)
    line = next((l for l in meta.splitlines() if l.strip() and "\t" in l), "")
    if not line:
        return None
    parts = line.split("\t")
    vid2 = parts[0].strip()
    upload_date = parts[1] if len(parts) > 1 else ""
    ts = parts[2] if len(parts) > 2 else ""
    dur = parts[3] if len(parts) > 3 else ""
    title = parts[4] if len(parts) > 4 else ""
    pub_date, pub_dt = _parse_dt(upload_date, ts)
    return {
        "video_id": vid2, "title": title.strip(),
        "channel": config.MONEYUP_CHANNEL_NAME, "channel_handle": config.MONEYUP_HANDLE,
        "url": f"https://www.youtube.com/watch?v={vid2}",
        "publish_date": pub_date, "publish_datetime": pub_dt,
        "duration_s": int(float(dur)) if dur.replace(".", "").isdigit() else None,
    }


def list_recent(n: int = 5) -> List[Dict]:
    """The ``n`` most recent 머니업 uploads (newest first), each with publish datetime + duration."""
    url = f"https://www.youtube.com/{config.MONEYUP_HANDLE}/videos"
    out = _yt(url, "--flat-playlist", "--playlist-end", str(n),
              "--print", "%(id)s", "--no-warnings", timeout=120)
    ids = [ln.strip() for ln in out.splitlines() if ln.strip() and "ERROR" not in ln]
    videos: List[Dict] = []
    for vid in ids[:n]:
        m = video_meta(vid)
        if m:
            videos.append(m)
    return videos


def download(video_id: str) -> Dict[str, Optional[str]]:
    """Download video (fmt 137) + audio (fmt 140) for ``video_id`` (skips if already present)."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    vpath = config.VIDEO_DIR / f"{video_id}.mp4"
    apath = config.VIDEO_DIR / f"{video_id}.m4a"
    n = str(config.YTDLP_CONCURRENCY)
    if not vpath.exists():
        _yt("-f", config.VIDEO_FORMAT, *_client_args(), "-N", n,
            "-o", str(vpath), url, "--no-warnings", timeout=600)
    if not apath.exists():
        _yt("-f", config.AUDIO_FORMAT, *_client_args(), "-N", n,
            "-o", str(apath), url, "--no-warnings", timeout=600)
    return {"video_path": str(vpath) if vpath.exists() else None,
            "audio_path": str(apath) if apath.exists() else None}
