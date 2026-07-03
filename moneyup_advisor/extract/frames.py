"""Full-coverage frame sampling (base env, ffmpeg) — every DISTINCT screen, no cap.

Samples at FRAME_SAMPLE_FPS and drops near-duplicate frames with ffmpeg ``mpdecimate`` (frame-diff
dedupe), writing each surviving frame as ``frame_<ms>ms.png`` using its real pts_time. A second
perceptual-hash dedupe runs in the vision worker. There is NO frame cap — a whole ~30-min video is
fully covered. The worker then OCRs the WHOLE screen of every distinct frame.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import List, Sequence

from moneyup_advisor import config

_PTS_RE = re.compile(r"pts_time:([0-9.]+)")


def _run(cmd: Sequence[str], cwd=None, timeout: int = 1800) -> str:
    r = subprocess.run(list(cmd), cwd=cwd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=timeout,
                       creationflags=config.CREATE_NO_WINDOW)
    return (r.stdout or "") + (r.stderr or "")


def sample(video_path: str, video_id: str, call_timestamps: Sequence[float] = ()) -> List[str]:
    """Extract every distinct frame (fps sample + mpdecimate), named frame_<ms>ms.png. No cap."""
    out_dir = config.video_frame_dir(video_id)
    for p in out_dir.glob("*.png"):
        p.unlink(missing_ok=True)
    fps = config.FRAME_SAMPLE_FPS
    # Run with cwd=out_dir and RELATIVE filenames: a Windows absolute path inside the -vf filter
    # (metadata=print:file=C:\...) breaks ffmpeg's filter parser (':' and '\').
    # mpdecimate DEFAULTS drop frames that don't differ greatly from the previous (ticking digits /
    # cursor) and keep genuine screen changes. (Custom low-frac params kept near-dupes -> too many.)
    _run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(Path(video_path).resolve()),
          "-vf", f"fps={fps},mpdecimate,metadata=print:file=_times.txt",
          "-vsync", "vfr", "-an", "raw_%06d.png"], cwd=str(out_dir))
    meta = out_dir / "_times.txt"
    pts: List[float] = []
    if meta.exists():
        pts = [float(m) for m in _PTS_RE.findall(meta.read_text(encoding="utf-8", errors="replace"))]
    raw = sorted(out_dir.glob("raw_*.png"))
    out: List[str] = []
    for i, p in enumerate(raw):
        t = pts[i] if i < len(pts) else (i / fps)
        dest = out_dir / f"frame_{int(t * 1000)}ms.png"
        if dest.exists() and dest != p:
            p.unlink(missing_ok=True)
        else:
            p.rename(dest)
        out.append(str(dest))
    return out
