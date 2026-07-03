"""Paths, model ids, ports and runtime knobs for the 머니업 AI Advisor.

Pure stdlib so it imports cleanly in BOTH the base env and the isolated ``moneyup`` env
(the vision worker imports it too). Everything regenerable lives under ``data/_moneyup_advisor/``,
which is git-ignored via the repo's ``/data/`` rule — so no weights/frames/outputs enter git.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Windows: spawn EVERY child CLI (yt-dlp / ffmpeg / ffprobe / vision worker / taskkill / nvidia-smi)
# WITHOUT a visible console window, so nothing flashes on screen while the batch runs. CREATE_NO_WINDOW
# gives the child a HIDDEN console which its OWN children inherit (e.g. the ffmpeg yt-dlp spawns to
# merge formats), so the whole process tree stays windowless. 0 on non-Windows (the flag is
# Windows-only; the conditional short-circuits so the attribute is never accessed off-Windows).
CREATE_NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

# --- repo layout ----------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "_moneyup_advisor"          # git-ignored (under /data/)
VIDEO_DIR = DATA_DIR / "videos"                              # downloaded mp4/m4a (git-ignored)
FRAME_DIR = DATA_DIR / "frames"                             # sampled frames per video
SHEET_DIR = DATA_DIR / "factsheets"                        # the fused fact sheets (json + md)
CACHE_DIR = DATA_DIR / "cache"                             # ticker maps, transcripts, etc.

for _d in (DATA_DIR, VIDEO_DIR, FRAME_DIR, SHEET_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- the channel ----------------------------------------------------------- #
# 머니업 — confirmed in Step 0 (the stock-call show: one ticker on an HTS screen per video).
MONEYUP_HANDLE = os.getenv("MONEYUP_HANDLE", "@money_go1330")
MONEYUP_CHANNEL_ID = os.getenv("MONEYUP_CHANNEL_ID", "UCMBvYoyXh6IOc7R68gVpsEg")
MONEYUP_CHANNEL_NAME = "머니업"

# --- models ---------------------------------------------------------------- #
def _ollama_host() -> str:
    # OLLAMA_HOST is often set to "0.0.0.0[:11434]" for the SERVER to bind all interfaces — that is
    # NOT a usable CLIENT url. Normalize: add scheme, and dial 127.0.0.1 instead of 0.0.0.0.
    h = (os.getenv("OLLAMA_HOST") or "http://127.0.0.1:11434").strip()
    if not h.startswith("http"):
        h = "http://" + h
    h = h.replace("0.0.0.0", "127.0.0.1")
    if h.count(":") < 2 and not h.rstrip("/").endswith(("11434",)):   # add default port if missing
        h = h.rstrip("/") + ":11434"
    return h


OLLAMA_HOST = _ollama_host()
QWEN_VL_MODEL = os.getenv("MONEYUP_QWEN_MODEL", "qwen3-vl:30b-a3b-instruct-q4_K_M")
WHISPER_MODEL = os.getenv("MONEYUP_WHISPER_MODEL", "large-v3")   # we force large-v3 (process-local)

# --- the isolated vision env (PaddleOCR + OpenCV) -------------------------- #
MONEYUP_ENV_PYTHON = os.getenv(
    "MONEYUP_ENV_PYTHON", r"C:\Users\A\miniconda3\envs\moneyup\python.exe")

# --- yt-dlp: the Step-0-proven non-DRM, un-throttled path ------------------ #
YTDLP_PLAYER_CLIENT = os.getenv("MONEYUP_YTDLP_CLIENT", "android_vr")
YTDLP_CONCURRENCY = int(os.getenv("MONEYUP_YTDLP_N", "8"))
# Format FALLBACK chains: prefer h264 1080p30 (137) for clean OCR frames, but many videos only have
# 60fps h264 (299) or AV1 (398/399) / opus audio (251) — fall back so the download never comes up empty.
VIDEO_FORMAT = os.getenv("MONEYUP_VIDEO_FMT", "137/299/398/399/bv*[height<=1080]/bv*/b")
AUDIO_FORMAT = os.getenv("MONEYUP_AUDIO_FMT", "140/251/139/ba/b")

# --- sampling / OCR caps (keep CPU OCR tractable across several videos) ----- #
# FULL-COVERAGE sampling: take a frame every 1/FRAME_SAMPLE_FPS s, then drop near-duplicate frames
# (ffmpeg mpdecimate + a worker perceptual-hash pass) and OCR EVERY distinct screen — NO frame cap.
# Fully processing a ~30-min video is expected (fast on GPU; ~20s/frame on CPU is acceptable overnight).
FRAME_SAMPLE_FPS = float(os.getenv("MONEYUP_FPS", "2.0"))            # base sampling rate
PHASH_HAMMING = int(os.getenv("MONEYUP_PHASH_HAM", "6"))            # worker dedupe distance threshold
# The WHOLE screen is OCR'd (watchlist + chart title bar/code + price axis + 호가 ladder + annotations).
OCR_WIDTH_FRAC = float(os.getenv("MONEYUP_OCR_W", "1.0"))           # 1.0 = whole frame
VLM_MAX_FRAMES = int(os.getenv("MONEYUP_VLM_MAX", "5"))
VLM_MAX_FRAMES = int(os.getenv("MONEYUP_VLM_MAX", "5"))              # Qwen is the slow part

# --- dashboard ------------------------------------------------------------- #
# DIFFERENT port from the existing dashboard (which is 8000) — brand-new, isolated app.
DASHBOARD_PORT = int(os.getenv("MONEYUP_DASH_PORT", "8077"))
DASHBOARD_HOST = os.getenv("MONEYUP_DASH_HOST", "127.0.0.1")

# --- watch-window for the Phase-1 falsification (publish -> N trading days) -- #
FALSIFY_HORIZON_DAYS = int(os.getenv("MONEYUP_FALSIFY_DAYS", "10"))


def video_frame_dir(video_id: str) -> Path:
    d = FRAME_DIR / video_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def deeplink(video_id: str, seconds) -> str:
    s = int(max(0.0, float(seconds or 0.0)))
    return f"https://www.youtube.com/watch?v={video_id}&t={s}s"
