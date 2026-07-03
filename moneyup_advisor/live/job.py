# -*- coding: utf-8 -*-
"""LIVE demo job runner — runs the REAL extraction pipeline on ONLY the first N minutes (default 5)
of a YouTube link or an uploaded file, streaming fine-grained events to a per-job folder so the page
can animate it like a live agent reading the screen.

    python -m moneyup_advisor.live.job --job <id> --url <youtube-url>
    python -m moneyup_advisor.live.job --job <id> --file <path-to-video>

Reuses the production functions UNCHANGED (fetch / transcript / frames / vision_worker / vlm /
calls / fuse / factsheet). Additive + isolated:
  * crops to the first N minutes (bounded work),
  * uses the job_id as the per-video key (frames/cache live under namespaced paths, cleaned after),
  * writes the fact sheet + KEEPS the 5-min clip(.mp4/.m4a) + full-res frame JPEGs INSIDE the job
    folder (never factsheets/, which the collector/playbook scan),
  * emits per-frame OCR tokens (text+score+bbox), so the page draws boxes and "reads" the screen.
No mock data: every number/transcript line/box/tag is produced by the real models.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

from moneyup_advisor import calls, config, factsheet, fuse, pipeline, tickers
from moneyup_advisor.extract import fetch, frames, transcript, vlm

LIVE_DIR = config.DATA_DIR / "live_jobs"
CLIP_SECONDS = int(os.getenv("MONEYUP_LIVE_SECONDS", "300"))          # first 5 min (demo knob)
MAX_RAM_GB = float(os.getenv("MONEYUP_LIVE_MAX_RAM_GB", "16"))        # RAM watchdog cap
UPSCALE = float(os.getenv("MONEYUP_LIVE_UPSCALE", "1.0"))             # >1.0 = upscale frames before OCR
THUMB_W = int(os.getenv("MONEYUP_LIVE_THUMB_W", "560"))               # larger, sharper thumbnails
FRAME_CAP = int(os.getenv("MONEYUP_LIVE_FRAME_CAP", "0"))             # >0 = cap distinct frames (fast demo)
LIVE_VLM_MAX = int(os.getenv("MONEYUP_LIVE_VLM_MAX", str(config.VLM_MAX_FRAMES)))  # VLM anchor count
# ROBUST single-file selector: prefer a browser+OCR-friendly avc1 video + m4a audio, then ANY
# <=1080p video+audio merge, then a PRE-MERGED <=1080p stream, then absolute best. Always pairs a
# video stream with an audio stream (or a combined stream), with multiple fallbacks.
LIVE_FMT = os.getenv("MONEYUP_LIVE_FMT",
                     "bestvideo[height<=1080][vcodec^=avc1]+bestaudio[ext=m4a]/"
                     "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best")


# --------------------------------------------------------------------------- #
# per-job event I/O (the page polls these files)
# --------------------------------------------------------------------------- #
class JobIO:
    def __init__(self, job_dir: Path):
        self.dir = job_dir
        self.events = job_dir / "events.jsonl"
        self.status_p = job_dir / "status.json"
        self.thumbs = job_dir / "thumbs"
        self.full = job_dir / "full"
        self.thumbs.mkdir(parents=True, exist_ok=True)
        self.full.mkdir(parents=True, exist_ok=True)
        self._i = 0
        self._lock = threading.Lock()
        self._pct = 0
        self._stage = "start"
        self._state = "running"

    def emit(self, **ev):
        with self._lock:
            self._i += 1
            ev["i"] = self._i
            ev["ts"] = round(time.time(), 2)
            if ev.get("pct") is not None:
                self._pct = ev["pct"]
            if ev.get("type") == "stage":
                self._stage = ev.get("stage", self._stage)
            with open(self.events, "a", encoding="utf-8") as f:
                f.write(json.dumps(ev, ensure_ascii=False) + "\n")
            self._write_status()

    def status(self, state: str, pct: Optional[int] = None, stage: Optional[str] = None):
        with self._lock:
            self._state = state
            if pct is not None:
                self._pct = pct
            if stage is not None:
                self._stage = stage
            self._write_status()

    def _write_status(self):
        tmp = self.status_p.with_suffix(".tmp")
        tmp.write_text(json.dumps({"state": self._state, "pct": self._pct,
                                   "stage": self._stage, "i": self._i}), encoding="utf-8")
        tmp.replace(self.status_p)

    def stage(self, stage, state, pct=None, msg=None):
        self.emit(type="stage", stage=stage, state=state, pct=pct, msg=msg)

    def log(self, msg, pct=None):
        self.emit(type="log", msg=msg, pct=pct)

    def save_frame_images(self, src_png: Path) -> Tuple[str, str, int, int]:
        """Write a full-res JPEG (full/) for the lightbox + a sharp thumbnail (thumbs/) for the grid.
        Returns (name, name, w, h). Falls back to copying the PNG if PIL is unavailable."""
        try:
            from PIL import Image
            name = src_png.stem + ".jpg"
            im = Image.open(src_png).convert("RGB")
            w, h = im.size
            im.save(self.full / name, "JPEG", quality=88)
            t = im.copy()
            t.thumbnail((THUMB_W, THUMB_W))
            t.save(self.thumbs / name, "JPEG", quality=82)
            return name, name, w, h
        except Exception:
            try:
                shutil.copy(src_png, self.full / src_png.name)
                shutil.copy(src_png, self.thumbs / src_png.name)
                return src_png.name, src_png.name, 0, 0
            except Exception:
                return "", "", 0, 0


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def parse_youtube_id(url: str) -> Optional[str]:
    url = (url or "").strip()
    if re.fullmatch(r"[\w-]{11}", url):
        return url
    m = (re.search(r"[?&]v=([\w-]{11})", url) or re.search(r"youtu\.be/([\w-]{11})", url)
         or re.search(r"/shorts/([\w-]{11})", url) or re.search(r"/live/([\w-]{11})", url)
         or re.search(r"/embed/([\w-]{11})", url))
    return m.group(1) if m else None


def _sections_arg(seconds: int) -> str:
    return f"*0:00-{seconds // 60}:{seconds % 60:02d}"            # yt-dlp --download-sections range


def _is_moneyup(*fields) -> bool:
    """True iff the REAL channel/handle is 머니업 (so non-머니업 creators get a neutral label — Task 1.4)."""
    s = " ".join(str(f) for f in fields).lower()
    return ("머니업" in s) or (config.MONEYUP_HANDLE.lower().lstrip("@") in s) or ("money_go1330" in s)


def quick_meta(yid: str, timeout: int = 20) -> Optional[dict]:
    """Best-effort title/date/CHANNEL with a SHORT timeout (don't let a slow metadata probe stall the demo).
    The REAL channel/uploader is read (not hard-coded 머니업) so a non-머니업 video is labelled generically."""
    try:
        out = fetch._yt(f"https://www.youtube.com/watch?v={yid}", "--skip-download", "--no-warnings",
                        "--print", "%(id)s\t%(title)s\t%(timestamp)s\t%(upload_date)s\t%(duration)s\t%(channel)s\t%(uploader_id)s",
                        timeout=timeout)
    except Exception:
        return None
    line = next((l for l in out.splitlines() if "\t" in l), "")
    if not line:
        return None
    p = (line.split("\t") + [""] * 7)[:7]
    pub_date, pub_dt = fetch._parse_dt(p[3], p[2])
    real_channel = p[5].strip() or config.MONEYUP_CHANNEL_NAME
    handle = p[6].strip() or config.MONEYUP_HANDLE
    is_mu = _is_moneyup(real_channel, handle)
    return {"video_id": p[0].strip() or yid, "title": p[1].strip(),
            "channel": real_channel, "channel_handle": handle, "is_moneyup": is_mu,
            "url": f"https://www.youtube.com/watch?v={yid}", "publish_date": pub_date,
            "publish_datetime": pub_dt,
            "duration_s": int(float(p[4])) if p[4].replace('.', '').isdigit() else None}


def _ffmpeg(args: List[str], timeout=900) -> int:
    return subprocess.run(["ffmpeg", "-y", "-loglevel", "error"] + args, capture_output=True,
                          text=True, timeout=timeout, creationflags=config.CREATE_NO_WINDOW).returncode


def _exists(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 0


def has_streams(path) -> Tuple[bool, bool]:
    """(has_video, has_audio) via ffprobe — used to CONFIRM a usable download/clip."""
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
                              "-of", "default=nw=1:nk=1", str(path)], capture_output=True, text=True,
                             timeout=60, creationflags=config.CREATE_NO_WINDOW).stdout
        return ("video" in out, "audio" in out)
    except Exception:
        return (Path(path).exists(), False)


def _clean_src(jd: Path):
    for p in list(jd.glob("src.*")):
        try:
            p.unlink()
        except Exception:
            pass


def _find_src(jd: Path) -> Optional[Path]:
    cands = [c for c in jd.glob("src.*")
             if not c.name.endswith(".part") and _exists(c)]
    if not cands:
        return None
    mp4 = [c for c in cands if c.suffix.lower() == ".mp4"]
    return (mp4 or sorted(cands, key=lambda p: -p.stat().st_size))[0]


def download(yid: str, jd: Path, seconds: int, io: "JobIO") -> Optional[Path]:
    """ROBUST download of the first `seconds` of a YouTube video as ONE file with BOTH video+audio.

    Tries, in order: (1) --download-sections with the android_vr client, (2) sections with the default
    client, (3) FULL download + (later) ffmpeg crop with android_vr, (4) full with default. yt-dlp's
    own retries handle transient throttling; we additionally retry across clients/strategies and
    VERIFY a real video stream landed before returning. Returns the source file (or None)."""
    url = f"https://www.youtube.com/watch?v={yid}"
    out_tmpl = str(jd / "src.%(ext)s")
    n = str(config.YTDLP_CONCURRENCY)
    base = ["-f", LIVE_FMT, "--merge-output-format", "mp4", "-N", n, "--retries", "3",
            "--fragment-retries", "10", "--extractor-retries", "3", "--no-warnings",
            "-o", out_tmpl, url]
    sec = _sections_arg(seconds)
    attempts = [("sectioned · android_vr", True, fetch._client_args(), 600),
                ("sectioned · default client", True, [], 600),
                ("FULL download + crop · android_vr", False, fetch._client_args(), 1200),
                ("FULL download + crop · default client", False, [], 1200)]
    for label, sectioned, client, tmo in attempts:
        _clean_src(jd)
        argv = list(base) + list(client) + (["--download-sections", sec] if sectioned else [])
        io.log(f"download: {label} …")
        try:
            fetch._yt(*argv, timeout=tmo)
        except Exception as e:
            io.log(f"  attempt failed: {str(e)[:90]}")
        src = _find_src(jd)
        if src:
            hv, ha = has_streams(src)
            io.log(f"  got {src.name} ({src.stat().st_size // 1_000_000}MB) video={hv} audio={ha}")
            if hv:                                          # real video stream landed -> use it
                return src
        time.sleep(3)                                       # ease YouTube throttling between attempts
    return None


def build_clip(src: Path, jd: Path, seconds: int):
    """Crop the first `seconds` of `src` (ONE file with video+audio — a YouTube download or an upload)
    into clip.mp4 (video+audio, for viewing/download AND frame sampling) and clip.m4a (audio only, for
    Whisper AND the page's audio player). Stream-copy first, re-encode fallback. Returns
    (clip_v, clip_a, has_audio)."""
    clip_v = jd / "clip.mp4"
    clip_a = jd / "clip.m4a"
    ok = (_ffmpeg(["-i", str(src), "-t", str(seconds), "-c", "copy", "-movflags", "+faststart",
                   str(clip_v)]) == 0 and _exists(clip_v) and has_streams(clip_v)[0])
    if not ok:                                                  # codec not mp4-copyable -> re-encode h264/aac
        ok = (_ffmpeg(["-i", str(src), "-t", str(seconds), "-c:v", "libx264", "-preset", "veryfast",
                       "-crf", "20", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart",
                       str(clip_v)]) == 0 and _exists(clip_v) and has_streams(clip_v)[0])
    if not ok:
        return None, None, False
    has_audio = (_ffmpeg(["-i", str(src), "-t", str(seconds), "-vn", "-c:a", "copy",
                          str(clip_a)]) == 0 and _exists(clip_a))
    if not has_audio:                                           # opus/other or copy failed -> aac (browser-safe)
        has_audio = (_ffmpeg(["-i", str(src), "-t", str(seconds), "-vn", "-c:a", "aac", "-b:a", "160k",
                              str(clip_a)]) == 0 and _exists(clip_a))
    return clip_v, clip_a, has_audio


def _cap_frames(frame_paths: List[str], cap: int) -> List[str]:
    """Fast demo: keep ~`cap` EVENLY-SPACED frames, delete the rest so the worker OCRs only these."""
    n = len(frame_paths)
    keep = sorted(set(round(i * (n - 1) / max(1, cap - 1)) for i in range(cap)))
    kept = []
    for i, fp in enumerate(frame_paths):
        if i in keep:
            kept.append(fp)
        else:
            try:
                Path(fp).unlink()
            except Exception:
                pass
    return kept


def upscale_frames(vid: str, factor: float):
    """OPTIONAL accuracy boost (item 6): upscale each sampled frame before OCR so PaddleOCR reads the
    dense HTS watchlist better. Off by default (factor=1.0). Keeps OCR coords == displayed image."""
    if factor <= 1.001:
        return
    try:
        from PIL import Image
    except Exception:
        return
    fdir = config.video_frame_dir(vid)
    for p in sorted(fdir.glob("frame_*.png")):
        try:
            im = Image.open(p)
            w, h = im.size
            im.resize((int(w * factor), int(h * factor)), Image.LANCZOS).save(p)
        except Exception:
            pass


_NUM_RE = re.compile(r"[+\-▲▼]?\s*[\d][\d,]*(?:\.\d+)?%?")


_RAW_NOISE = re.compile(r"^\d{1,4}\.\d$")     # 999.9-style one-decimal OCR-grid artifact


def _raw_ok(text) -> bool:
    """Drop OCR-grid garbage (999.9 runs, bare 0/00/000) from the raw token dump (Task 1.3)."""
    t = str(text or "").strip()
    return bool(t) and not _RAW_NOISE.match(t) and not (re.fullmatch(r"\d{1,3}", t) and int(t) < 100)


def ocr_event(fr: dict) -> dict:
    """Per-frame OCR event carrying every PLAUSIBLE token (text+score+bbox) so the page can draw boxes,
    animate the read, and offer 'see all extracted'. OCR-grid noise (999.9) is filtered (Task 1.3)."""
    rows = fr.get("ocr", []) or []
    toks = [{"t": str(r.get("text", "")), "s": round(float(r.get("score") or 0), 2), "b": r.get("bbox")}
            for r in rows if _raw_ok(r.get("text"))]
    ap = fr.get("axis_price") or {}
    ap_tag = (ap.get("color") or ap.get("source") or "").strip()      # color box OR off-grid last-price
    return {"frame": fr.get("frame"), "t": fr.get("t"), "w": fr.get("w"), "h": fr.get("h"),
            "tokens": toks, "n": len(toks),
            "axis_price": ((ap.get("text") + (f" ({ap_tag})" if ap_tag else "")) if ap.get("text") else None)}


def vlm_live(frames_data: List[dict], vid: str, cap: int, io: JobIO) -> List[dict]:
    """Production VLM frame-pick (pipeline._vlm_frames), but emit each Qwen description as it lands."""
    chart = [f for f in frames_data if f.get("chart", {}).get("has_chart")]
    pool = sorted(chart or frames_data, key=lambda f: f.get("t", 0))
    if not pool:
        return []
    vlm.warmup()                                        # pre-load Qwen so the first read doesn't cold-race
    step = max(1, len(pool) // cap)
    picks = pool[::step][:cap]
    fdir = config.video_frame_dir(vid)
    _DESC = ("trend_structure", "moving_averages", "key_levels", "recent_event", "instrument", "stance",
             "chart_pattern", "points_at")
    obs = []
    for f in picks:
        parsed = vlm.describe_frame(str(fdir / f["frame"]))
        if not parsed:
            continue
        obs.append({"t": f.get("t"), "frame": f["frame"], "parsed": parsed})
        desc = "; ".join(f"{k}={parsed.get(k)}" for k in _DESC
                         if parsed.get(k) and str(parsed.get(k)).strip().lower() not in vlm.EMPTY_VALUES)
        io.emit(type="vlm", t=f.get("t"), frame=f["frame"], desc=desc or parsed.get("note", ""),
                stance=parsed.get("stance"), pattern=parsed.get("trend_structure") or parsed.get("chart_pattern"))
    return obs


def ram_watchdog(io: JobIO, stop: threading.Event):
    """Bound the demo's own process-tree RAM (keep the watchdog, like the collector)."""
    try:
        import psutil
    except Exception:
        return
    me = psutil.Process(os.getpid())
    while not stop.wait(3):
        try:
            tot = me.memory_info().rss
            for c in me.children(recursive=True):
                try:
                    tot += c.memory_info().rss
                except Exception:
                    pass
            if tot / 1e9 >= MAX_RAM_GB:
                io.emit(type="error", msg=f"RAM watchdog: demo tree exceeded {MAX_RAM_GB} GB — aborted")
                io.status("error")
                for c in me.children(recursive=True):
                    try:
                        c.kill()
                    except Exception:
                        pass
                os._exit(3)
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
def run(job_id: str, url: Optional[str], file: Optional[str]) -> int:
    jd = LIVE_DIR / job_id
    jd.mkdir(parents=True, exist_ok=True)
    io = JobIO(jd)
    stop = threading.Event()
    threading.Thread(target=ram_watchdog, args=(io, stop), daemon=True).start()
    vid = job_id                                                     # per-video key (namespaced)
    io.status("running", 0, "start")
    io.log(f"first {CLIP_SECONDS}s only · real pipeline", pct=1)
    try:
        tickers.ensure_name_map()                                   # ticker resolution (cached)

        # ---- 1) DOWNLOAD / accept upload ----
        io.stage("download", "running", 2, "fetching source…")
        if file:
            src = Path(file)
            if not src.exists():
                raise RuntimeError(f"uploaded file not found: {file}")
            meta = {"video_id": vid, "title": f"LOCAL: {src.name}", "channel": "local file",
                    "channel_handle": "", "url": src.name, "publish_date": None,
                    "publish_datetime": None, "duration_s": None, "source": "local"}
            io.emit(type="meta", title=meta["title"], url=None, video_id=vid,
                    clip_seconds=CLIP_SECONDS, source="file")
            io.log(f"uploaded file: {src.name}")
        else:
            yid = parse_youtube_id(url)
            if not yid:
                raise RuntimeError("could not parse a YouTube video id from the link")
            meta = quick_meta(yid)                          # best-effort title/date, short timeout
            if not meta:
                meta = {
                    "video_id": yid, "title": "", "channel": config.MONEYUP_CHANNEL_NAME,
                    "channel_handle": config.MONEYUP_HANDLE, "is_moneyup": True,   # probe failed → assume 머니업
                    "url": f"https://www.youtube.com/watch?v={yid}",
                    "publish_date": None, "publish_datetime": None, "duration_s": None}
            io.emit(type="meta", title=meta.get("title"), url=meta.get("url"), video_id=yid,
                    clip_seconds=CLIP_SECONDS, source="youtube")
            io.log(f"downloading first {CLIP_SECONDS}s (<=1080p, robust) · {meta.get('title','')[:48]}")
            src = download(yid, jd, CLIP_SECONDS, io)
            if not src:
                raise RuntimeError("could not fetch a usable video after retries — YouTube may be "
                                   "throttling or the formats are unavailable. Try again or a different link.")
        io.stage("download", "done", 15, "source downloaded")

        # ---- 2) CROP to first N minutes (clip.mp4 = video+audio, clip.m4a = audio) + CONFIRM streams ----
        io.stage("crop", "running", 17, f"ffmpeg -t {CLIP_SECONDS}")
        clip_v, clip_a, has_audio = build_clip(src, jd, CLIP_SECONDS)
        if not clip_v or not has_streams(clip_v)[0]:
            raise RuntimeError("no usable VIDEO stream after download+crop (cannot sample frames)")
        if not has_audio:
            io.log("warning: clip has no audio stream — transcript will be empty (frames still extracted)")
        io.emit(type="clip", video="clip.mp4", audio="clip.m4a" if has_audio else None)
        io.stage("crop", "done", 22,
                 f"clip = first {CLIP_SECONDS}s · video OK · audio {'OK' if has_audio else 'none'} (kept)")

        # ---- 3) AUDIO → Whisper (REAL) ----
        io.stage("whisper", "running", 24, "transcribing (Whisper large-v3)…")
        segments = transcript.transcribe(str(clip_a), vid, refresh=True) if has_audio else []
        for s in segments:
            io.emit(type="transcript", start=s["start"], end=s.get("end"),
                    mmss=_mmss(s["start"]), text=s["text"])
        io.stage("whisper", "done", 45, f"{len(segments)} transcript segments")

        # ---- 4) FRAME sampling (REAL, ffmpeg) + full-res JPEG + sharp thumbnail per frame ----
        io.stage("frames", "running", 47, "sampling distinct frames (ffmpeg)…")
        frame_paths = frames.sample(str(clip_v), vid, pipeline._call_timestamps(segments))
        if FRAME_CAP > 0 and len(frame_paths) > FRAME_CAP:    # fast demo: sparser frames (OCR only these)
            frame_paths = _cap_frames(frame_paths, FRAME_CAP)
            io.log(f"fast mode: capped to {len(frame_paths)} evenly-spaced frames")
        if UPSCALE > 1.001:
            io.log(f"upscaling frames x{UPSCALE} for denser OCR…")
            upscale_frames(vid, UPSCALE)
        for idx, fp in enumerate(frame_paths):
            p = Path(fp)
            full_name, thumb_name, w, h = io.save_frame_images(p)
            m = re.search(r"_(\d+)ms", p.name)
            t = round(int(m.group(1)) / 1000.0, 1) if m else 0.0
            if thumb_name:
                io.emit(type="frame", frame=p.name, name=thumb_name, full=full_name,
                        t=t, w=w, h=h, idx=idx)
        io.stage("frames", "done", 55, f"{len(frame_paths)} frames sampled")

        # ---- 5) OCR (REAL, PaddleOCR via the production worker) — emit EVERY token + bbox ----
        io.stage("ocr", "running", 57, "PaddleOCR reading every distinct frame…")
        frames_data = pipeline.run_vision_worker(vid, refresh=True)
        fd_by_name = {fr.get("frame"): fr for fr in frames_data}
        for fr in frames_data:
            io.emit(type="ocr", **ocr_event(fr))
        dev = ""
        try:
            dev = json.loads((config.CACHE_DIR / f"{vid}.vision.json").read_text(encoding="utf-8")).get("device", "")
        except Exception:
            pass
        io.stage("ocr", "done", 75, f"{len(frames_data)} frames OCR'd ({dev or 'ocr'})")

        # ---- 6) VISION / Qwen3-VL (REAL, pattern/context only) ----
        io.stage("vlm", "running", 77, "Qwen3-VL describing representative frames…")
        vlm_obs = vlm_live(frames_data, vid, LIVE_VLM_MAX, io)
        io.stage("vlm", "done", 90, f"{len(vlm_obs)} VLM observations")

        # ---- 7) FUSION (REAL) ----
        io.stage("fusion", "running", 92, "fusing audio ↔ video on one timeline…")
        primary, subject_label = pipeline.resolve_subject(meta.get("title", ""), segments, frames_data, vlm_obs)
        if subject_label:
            meta["subject_label"] = subject_label                # US/global index label when no Korean stock
        call_data = calls.extract(segments, meta, primary)
        fused = fuse.fuse(segments, frames_data, primary, vlm_obs, call_data)
        for e in fused.get("timeline", []):
            detail = e.get("video") or e.get("audio") or ""
            io.emit(type="fusion", tag=e.get("tag"), kind=e.get("kind"), ticker=e.get("ticker"),
                    mmss=e.get("mmss"), detail=detail)
        io.stage("fusion", "done", 90, " · ".join(f"{k}={v}" for k, v in fused.get("summary", {}).items()))

        # ---- 8) FACT SHEET (REAL build) — written to the JOB folder, NEVER factsheets/ ----
        io.stage("sheet", "running", 92, "assembling the grounded fact sheet…")
        sheet = factsheet.build(meta, segments, frames_data, fused, call_data, vlm_obs, primary, degraded=False)
        (jd / "factsheet.json").write_text(json.dumps(sheet, ensure_ascii=False, indent=2), encoding="utf-8")
        (jd / "factsheet.md").write_text(factsheet.render_markdown(sheet), encoding="utf-8")
        # index this run (keyed by video id) so the same link auto-replays + the picker/export can find it
        (jd / "info.json").write_text(json.dumps({
            "job_id": job_id, "video_id": meta.get("video_id"), "title": meta.get("title"),
            "url": meta.get("url"), "source": "file" if file else "youtube",
            "n_frames": sheet.get("n_frames_ocr"), "n_segments": sheet.get("n_transcript_segments"),
            "clip_seconds": CLIP_SECONDS, "has_audio": bool(has_audio), "fast": FRAME_CAP > 0,
            "done": True, "created": time.time(),
        }, ensure_ascii=False), encoding="utf-8")
        io.emit(type="sheet", primary_ticker=sheet.get("primary_ticker"),
                primary_name=sheet.get("primary_name"), title=meta.get("title"),
                n_frames=sheet.get("n_frames_ocr"), n_segments=sheet.get("n_transcript_segments"),
                n_exante=len(call_data.get("exante", [])), n_vlm=len(vlm_obs),
                fusion_summary=fused.get("summary", {}),
                exante=[{"ticker": c["ticker"], "name": c["name"], "direction": c["direction"],
                         "mmss": c.get("mmss"), "stated_price": c.get("stated_price"),
                         "quote": c.get("quote")} for c in call_data.get("exante", [])])
        io.stage("sheet", "done", 94, "fact sheet ready")

        # ---- 9) RAW DUMP + GROUNDED SUMMARY (text synthesis via Gemini 3.1 Pro — API, NO GPU) ----
        io.stage("summary", "running", 95, "raw dump (.docx) + grounded digest + comparison (Gemini 3.1 Pro)…")
        from moneyup_advisor.live import digest
        try:
            files = digest.finalize(jd, meta, segments, frames_data, vlm_obs, call_data, fused, primary)
        except Exception as e:
            io.log(f"digest error ({str(e)[:120]})")
            files = {k: (v if (jd / v).exists() else None) for k, v in
                     {"summary": "summary_full.docx", "summary_en": "summary_full_en.docx",
                      "raw": "raw_full.docx", "raw_json": "raw_full.json", "comparison": "comparison.docx"}.items()}
        io.stage("summary", "done", 100,
                 "digest (KO+EN) + comparison ready" if files.get("summary") else "raw + comparison ready (summary skipped)")
        io.emit(type="files", summary=files.get("summary"), summary_en=files.get("summary_en"),
                raw=files.get("raw"), raw_json=files.get("raw_json"), comparison=files.get("comparison"))
        io.emit(type="done", pct=100)
        io.status("done", 100, "done")
        return 0
    except Exception as e:
        io.emit(type="error", msg=f"{type(e).__name__}: {str(e)[:200]}")
        io.status("error")
        (jd / "job_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        return 1
    finally:
        stop.set()
        _cleanup(vid, jd)


def _mmss(t):
    t = int(float(t or 0))
    return f"{t // 60:02d}:{t % 60:02d}"


def _cleanup(vid: str, jd: Path):
    """Isolation hygiene: drop the demo's namespaced frames + cache + raw source downloads. KEEPS the
    clip (mp4+m4a), full-res frame JPEGs, thumbs, events and fact sheet in the job folder for replay.
    NEVER touches factsheets/ (we never wrote there)."""
    try:
        fdir = config.FRAME_DIR / vid
        if fdir.is_dir():
            shutil.rmtree(fdir, ignore_errors=True)
        for suf in (".transcript.json", ".vision.json", ".vlm.json"):
            p = config.CACHE_DIR / f"{vid}{suf}"
            if p.exists():
                p.unlink()
        for p in list(jd.glob("src.*")):                             # raw downloads only (clip kept)
            try:
                p.unlink()
            except Exception:
                pass
    except Exception:
        pass


def main():
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--url")
    ap.add_argument("--file")
    a = ap.parse_args()
    if not a.url and not a.file:
        print("need --url or --file", file=sys.stderr)
        return 2
    return run(a.job, a.url, a.file)


if __name__ == "__main__":
    sys.exit(main())
