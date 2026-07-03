"""End-to-end Phase-0 extraction orchestrator (base env).

Per video: download -> Whisper transcript -> ffmpeg frame sampling -> [moneyup-env vision worker:
PaddleOCR + OpenCV] -> Qwen3-VL pattern/context -> ex-ante/ex-post calls -> timeline fusion ->
ONE grounded fused fact sheet (json + md).

The ONLY cross-env hop is the vision worker, invoked as a subprocess in the isolated ``moneyup``
conda env. Everything else runs in the base env (reusing youtube_source's Whisper/yt-dlp helpers
READ-ONLY).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional

from moneyup_advisor import calls, config, factsheet, fuse, tickers
from moneyup_advisor.calls import _AVOID, _EXPOST, _LONG, _SHORT
from moneyup_advisor.extract import fetch, frames, transcript, vlm

_KEYWORDS = _LONG + _SHORT + _AVOID + _EXPOST


def _call_timestamps(segments: List[dict]) -> List[float]:
    """Timestamps of 'call segments' (action/ex-post keyword present) — dense-sample frames here."""
    ts = []
    for s in segments:
        if any(k in s.get("text", "") for k in _KEYWORDS):
            ts.append(round(float(s.get("start", 0.0) or 0.0), 1))
    return sorted(set(ts))


def run_vision_worker(video_id: str, refresh: bool = False) -> List[dict]:
    """Invoke the moneyup-env PaddleOCR+OpenCV worker on the sampled frames; return frame results.
    Reuses the cached vision JSON unless ``refresh`` (OCR is the slow CPU step)."""
    frames_dir = str(config.video_frame_dir(video_id))
    out_path = config.CACHE_DIR / f"{video_id}.vision.json"
    if out_path.exists() and not refresh:
        return json.loads(out_path.read_text(encoding="utf-8")).get("frames", [])
    out_json = str(out_path)
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
           "PYTHONPATH": str(config.REPO_ROOT)}
    cmd = [config.MONEYUP_ENV_PYTHON, "-m", "moneyup_advisor.vision_worker",
           frames_dir, "--out", out_json]
    r = subprocess.run(cmd, cwd=str(config.REPO_ROOT), env=env, capture_output=True,
                       text=True, encoding="utf-8", errors="replace", timeout=3600,
                       creationflags=config.CREATE_NO_WINDOW)
    if r.returncode != 0:
        print(f"  [vision worker stderr] {(r.stderr or '')[-600:]}")
        return []
    data = json.loads(Path(out_json).read_text(encoding="utf-8"))
    return data.get("frames", [])


def _primary_code(title: str, frames_data: List[dict]) -> Optional[str]:
    """A KOREAN 6-digit code ONLY if confidently resolved; else None. Never forces the Korean universe:
    name matching uses min_len=3 (so 'LS' can't false-match inside an English title word), and the OCR
    fallback requires a code on >=2 frames (one-off OCR noise isn't a subject)."""
    for c in tickers.codes_in_text(title):
        return c
    for c in tickers.names_in_text(title, min_len=3):
        return c
    from collections import Counter
    seen = Counter()
    for fr in frames_data:
        for c in tickers.codes_in_text(" ".join(r.get("text", "") for r in fr.get("ocr", []))):
            seen[c] += 1
    for c, n in seen.most_common():
        if n >= 2:
            return c
    return None


_US_INDEX = {
    "nasdaq composite": "NASDAQ Composite (^IXIC)", "nasdaq": "NASDAQ (^IXIC)",
    "s&p 500": "S&P 500 (^GSPC)", "s&p500": "S&P 500 (^GSPC)", "s&p": "S&P 500 (^GSPC)",
    "dow jones": "Dow Jones (^DJI)", "dow": "Dow Jones (^DJI)", "russell 2000": "Russell 2000 (^RUT)",
    "russell": "Russell 2000 (^RUT)", "nikkei": "Nikkei 225 (^N225)", "hang seng": "Hang Seng (^HSI)",
    "bitcoin": "Bitcoin (BTC)", "btc": "Bitcoin (BTC)", "ethereum": "Ethereum (ETH)",
}


def resolve_subject(title: str, segments: List[dict], frames_data: List[dict],
                    vlm_obs: List[dict]) -> tuple:
    """(primary_code_or_None, subject_label). Returns a Korean code + name when one is the confident
    subject; otherwise detects a named US/global index/symbol from the chart label (VLM instrument) and
    the audio — and labels THAT, never a forced Korean ticker. (None, None) if undetermined."""
    code = _primary_code(title, frames_data)
    if code:
        return code, tickers.display_name(code)
    for o in vlm_obs or []:                              # the VLM's read of the chart's instrument label
        instr = str((o.get("parsed", {}) or {}).get("instrument", "")).lower()
        for k in sorted(_US_INDEX, key=len, reverse=True):
            if k in instr:
                return None, _US_INDEX[k]
    text = ((title or "") + " " + " ".join(s.get("text", "") for s in segments)).lower()
    for k in sorted(_US_INDEX, key=len, reverse=True):
        if k in text:
            return None, _US_INDEX[k]
    return None, None


def _vlm_frames(frames_data: List[dict], video_id: str, cap: int) -> List[dict]:
    """Pick representative frames (prefer chart frames, spread over time) for Qwen3-VL."""
    chart = [f for f in frames_data if f.get("chart", {}).get("has_chart")]
    pool = chart or frames_data
    pool = sorted(pool, key=lambda f: f.get("t", 0))
    if not pool:
        return []
    vlm.warmup()                                        # pre-load Qwen so the first read doesn't cold-race
    step = max(1, len(pool) // cap)
    picks = pool[::step][:cap]
    fdir = config.video_frame_dir(video_id)
    obs = []
    for f in picks:
        parsed = vlm.describe_frame(str(fdir / f["frame"]))
        if parsed:
            obs.append({"t": f.get("t"), "frame": f["frame"], "parsed": parsed})
    return obs


def _cleanup_media(vid: str) -> None:
    """After a fact sheet is safely on disk, delete that video's regenerable HEAVY media —
    the downloaded mp4/m4a (``data/_moneyup_advisor/videos/``) and its sampled frames
    (``data/_moneyup_advisor/frames/<vid>/``) — to bound disk use. The fact sheet, the cache
    (transcript/vision/vlm json), the playbook and the Phase-1 outputs are NEVER touched, so the
    video still counts as 'done' (sheet exists) and is never re-downloaded. A user-supplied
    ``--local`` source is safe: it lives outside ``videos/`` so it is never matched.
    Set ``MONEYUP_KEEP_MEDIA=1`` to disable. Best-effort: any error is logged, never raised."""
    if os.getenv("MONEYUP_KEEP_MEDIA", "0") not in ("0", "", "false", "no"):
        return
    freed = 0
    try:
        for p in (config.VIDEO_DIR / f"{vid}.mp4", config.VIDEO_DIR / f"{vid}.m4a"):
            if p.exists():
                freed += p.stat().st_size
                p.unlink()
        fdir = config.FRAME_DIR / vid
        if fdir.is_dir():
            for f in fdir.rglob("*"):
                if f.is_file():
                    try:
                        freed += f.stat().st_size
                    except OSError:
                        pass
            shutil.rmtree(fdir, ignore_errors=True)
    except Exception as e:
        print(f"  [cleanup] skipped for {vid}: {str(e)[:120]}")
        return
    if freed:
        print(f"  cleanup: freed {freed/1e6:.0f} MB (mp4/m4a + frames for {vid}; fact sheet kept)")


def process_video(video: Dict, use_vlm: bool = True, refresh: bool = False,
                  local_path: Optional[str] = None) -> Optional[Dict]:
    vid = video["video_id"]
    print(f"\n=== {vid} | {video.get('title','')[:60]} ===")
    if local_path:                                     # LOCAL file: no download, no audio
        dl = {"video_path": local_path, "audio_path": None}
        if not os.path.exists(local_path):
            print("  local file missing; skipping")
            return None
    else:
        dl = fetch.download(vid)
        if not dl.get("audio_path") or not dl.get("video_path"):
            print("  download failed; skipping")
            return None
    if dl.get("audio_path"):
        print("  transcribing (Whisper large-v3)…")
        segments = transcript.transcribe(dl["audio_path"], vid, refresh=refresh)
    else:
        segments = []                                  # silent / audio-stripped -> empty transcript
        print("  no audio stream -> transcript EMPTY (vision-only)")
    print(f"    {len(segments)} segments")
    fdir = config.video_frame_dir(vid)
    existing = sorted(fdir.glob("*.png"))
    if existing and not refresh:
        frame_paths = [str(p) for p in existing]
        print(f"  reusing {len(frame_paths)} cached frames (use --refresh to re-sample)")
    else:
        print("  sampling frames (ffmpeg scene + call anchors)…")
        frame_paths = frames.sample(dl["video_path"], vid, _call_timestamps(segments))
    print(f"    {len(frame_paths)} frames -> OCR")
    print("  running vision worker (moneyup env: PaddleOCR + OpenCV)…")
    frames_data = run_vision_worker(vid, refresh=refresh)
    print(f"    {len(frames_data)} frames analyzed")
    # GUARD: a video WITH a video stream MUST yield frames. 0 frames = DEGRADED extraction (not audio-only).
    # Retry sample+OCR ONCE; if still 0, write a FLAGGED (not silent) sheet, queue for re-extraction, log loud.
    has_stream = bool(dl.get("video_path")) and os.path.exists(dl.get("video_path") or "")
    if not frames_data and has_stream and not local_path:
        print("  [guard] 0 frames OCR'd on a video that HAS a video stream — retrying sample+OCR once (fresh)…")
        try:
            frames.sample(dl["video_path"], vid, _call_timestamps(segments))
            frames_data = run_vision_worker(vid, refresh=True)
        except Exception as e:
            print(f"  [guard] retry error: {str(e)[:100]}")
        print(f"    [guard] after retry: {len(frames_data)} frames")
    degraded = (not frames_data) and has_stream and not local_path
    if degraded:
        print(f"  [guard] !!! DEGRADED {vid}: video stream present but 0 frames after retry — "
              f"flagging the sheet 'degraded' and queuing for re-extraction !!!")
        try:                                               # queue id for the collector to re-extract (--refresh)
            dq = config.DATA_DIR / "degraded.json"
            cur = json.loads(dq.read_text(encoding="utf-8")) if dq.exists() else []
            if vid not in cur:
                cur.append(vid)
                dq.write_text(json.dumps(cur), encoding="utf-8")
        except Exception as e:
            print(f"  [guard] queue write error: {str(e)[:80]}")
    primary = _primary_code(video.get("title", ""), frames_data)
    print(f"  primary ticker: {primary} ({tickers.display_name(primary) if primary else '?'})")
    vlm_cache = config.CACHE_DIR / f"{vid}.vlm.json"
    if not use_vlm:
        vlm_obs = []
    elif vlm_cache.exists() and not refresh:
        vlm_obs = json.loads(vlm_cache.read_text(encoding="utf-8"))      # reuse (Qwen is expensive)
    else:
        vlm_obs = _vlm_frames(frames_data, vid, config.VLM_MAX_FRAMES)
        vlm_cache.write_text(json.dumps(vlm_obs, ensure_ascii=False), encoding="utf-8")
    print(f"    {len(vlm_obs)} VLM observations")
    call_data = calls.extract(segments, video, primary)
    fused = fuse.fuse(segments, frames_data, primary, vlm_obs, call_data)
    sheet = factsheet.build(video, segments, frames_data, fused, call_data, vlm_obs, primary, degraded=degraded)
    paths = factsheet.save(sheet)
    print(f"  fact sheet -> {paths['md']}")
    if os.path.exists(paths["json"]):                  # ONLY after the sheet is safely written
        _cleanup_media(vid)                            # drop mp4/m4a + frames (keep sheet/cache)
    print(f"  fusion: {fused['summary']} | ex-ante calls: {len(call_data['exante'])} "
          f"| ex-post: {len(call_data['expost'])}")
    del segments, frames_data, vlm_obs, fused, call_data   # release per-video buffers (RAM hygiene)
    gc.collect()
    return sheet


def main():
    for _s in (sys.stdout, sys.stderr):                  # Korean/emoji-safe on the cp949 console
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description="머니업 Phase-0 extraction")
    ap.add_argument("--max", type=int, default=4, help="number of recent videos")
    ap.add_argument("--video", action="append", help="specific video id(s)")
    ap.add_argument("--local", help="path to a LOCAL video file (vision-only; no download)")
    ap.add_argument("--label", help="id/label for the local file (default: filename stem)")
    ap.add_argument("--no-vlm", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()

    print("building ticker name<->code map (pykrx, cached)…")
    m = tickers.ensure_name_map()
    print(f"  {len(m.get('code2name', {}))} names mapped; {len(tickers.code_set())} valid codes")

    if a.local:
        from pathlib import Path as _P
        label = a.label or _P(a.local).stem
        dur = None
        try:
            import subprocess as _sp
            dur = int(float(_sp.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                     "-of", "default=nk=1:nw=1", a.local],
                                    capture_output=True, text=True,
                                    creationflags=config.CREATE_NO_WINDOW).stdout.strip() or 0))
        except Exception:
            pass
        meta = {"video_id": label, "title": f"LOCAL: {_P(a.local).name}", "channel": "local file",
                "channel_handle": "", "url": a.local, "publish_date": None,
                "publish_datetime": None, "duration_s": dur, "source": "local"}
        try:
            s = process_video(meta, use_vlm=not a.no_vlm, refresh=a.refresh, local_path=a.local)
            print(f"\nDONE local: {label} -> sheet {'OK' if s else 'FAILED'}")
        except Exception:
            print(f"  ERROR on local {label}:\n{traceback.format_exc()}")
        return

    if a.video:
        videos = []
        for v in a.video:
            m = fetch.video_meta(v) or {
                "video_id": v, "title": "", "channel": config.MONEYUP_CHANNEL_NAME,
                "channel_handle": config.MONEYUP_HANDLE,
                "url": f"https://www.youtube.com/watch?v={v}",
                "publish_date": None, "publish_datetime": None, "duration_s": None}
            videos.append(m)
    else:
        print(f"listing {a.max} recent 머니업 videos…")
        videos = fetch.list_recent(a.max)
    print(f"  {len(videos)} videos")

    done = []
    for v in videos[: a.max if not a.video else len(videos)]:
        try:
            s = process_video(v, use_vlm=not a.no_vlm, refresh=a.refresh)
            if s:
                done.append(s["video_id"])
        except Exception:
            print(f"  ERROR on {v.get('video_id')}:\n{traceback.format_exc()}")
    if not a.no_vlm:
        vlm.unload()
    print(f"\nDONE: {len(done)} fact sheet(s): {done}")
    print(f"sheets in: {config.SHEET_DIR}")


if __name__ == "__main__":
    sys.exit(main())
