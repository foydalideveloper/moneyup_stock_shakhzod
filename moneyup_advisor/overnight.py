"""Overnight 머니업 batch — full-coverage GPU, idempotent, hard-stop 05:00 KST.

* Lists ALL @money_go1330 uploads (newest first), skips any that already have a fact sheet
  (idempotent + crash-resume: a video that crashed mid-way left no sheet -> retried; its cached
  transcript/vision/vlm make the retry fast).
* Per video logs: id, frames, sec/frame, total time, #ex-ante calls + a RUNNING TOTAL.
* HARD STOP 05:00 KST: never STARTS a new video at/after 05:00 (a video already running finishes),
  then UNLOADS all GPU models (Qwen3-VL via Ollama; Whisper + PaddleOCR are per-call/subprocess and
  free themselves; the process then exits) so the 5:30 transcribe + 6:20 report get a clean GPU.
* Writes only under data/_moneyup_advisor/ (shown on the :8077 dashboard). Touches no daily-pipeline code.

Run:  python -m moneyup_advisor.overnight        (base env)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone

from moneyup_advisor import config, factsheet, tickers          # noqa: F401 (factsheet via pipeline)
from moneyup_advisor.extract import fetch, vlm
from moneyup_advisor.pipeline import process_video

KST = timezone(timedelta(hours=9))                              # KST = UTC+9, no DST (exact)
LOG = config.DATA_DIR / "overnight_batch.log"
REPORT = config.DATA_DIR / "overnight_report.json"


def now_kst():
    return datetime.now(KST)


def next_5am():
    n = now_kst()
    five = n.replace(hour=5, minute=0, second=0, microsecond=0)
    return five if n < five else five + timedelta(days=1)


def log(msg: str):
    line = f"[{now_kst():%Y-%m-%d %H:%M:%S KST}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def all_video_ids():
    """Every @money_go1330 upload id, newest first."""
    out = fetch._yt(f"https://www.youtube.com/{config.MONEYUP_HANDLE}/videos",
                    "--flat-playlist", "--print", "%(id)s", "--no-warnings", timeout=300)
    return [l.strip() for l in out.splitlines() if l.strip() and "ERROR" not in l]


def recent_ids(n: int = 15):
    """The n NEWEST channel uploads (cheap head-only flat-playlist), newest-first.

    Used by the CONTINUOUS new-upload check during a run: the full all_video_ids() listing happens
    once at startup, but this cheap head poll lets a video published mid-run jump the queue ASAP."""
    out = fetch._yt(f"https://www.youtube.com/{config.MONEYUP_HANDLE}/videos",
                    "--flat-playlist", "--playlist-end", str(n),
                    "--print", "%(id)s", "--no-warnings", timeout=120)
    return [l.strip() for l in out.splitlines() if l.strip() and "ERROR" not in l]


def done(vid: str) -> bool:
    return (config.SHEET_DIR / f"{vid}.json").exists()


def vision_stats(vid: str) -> dict:
    p = config.CACHE_DIR / f"{vid}.vision.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("stats", {})
    except Exception:
        return {}


def exante_count(vid: str) -> int:
    p = config.SHEET_DIR / f"{vid}.json"
    try:
        return len(json.loads(p.read_text(encoding="utf-8")).get("exante_calls", []))
    except Exception:
        return 0


# WATCHDOG: process each video in its OWN subprocess with a hard timeout, so one stuck
# download/transcode/OCR can never stall the whole batch — on timeout we kill the entire process
# tree (yt-dlp / ffmpeg / vision worker) and move on.
WATCHDOG_MIN = int(os.getenv("MONEYUP_WATCHDOG_MIN", "25"))
# CONTINUOUS new-upload check: re-list the channel head at most this often (minutes) and jump any
# brand-new upload to the FRONT of the queue, so a video published mid-run is processed ASAP (not
# only at the next 07:00 restart).
NEW_CHECK_MIN = int(os.getenv("MONEYUP_NEWCHECK_MIN", "15"))
# RAM SAFETY VALVE: kill a video whose process tree (pipeline + yt-dlp/ffmpeg/vision worker) grows past
# MAX_RAM_GB, or if free system RAM falls below MIN_FREE_GB — so one pathological (very long) video can't
# balloon to ~22 GB and OOM-stall the 22h run. Both thresholds are well under the box's 32 GB.
MAX_RAM_GB = float(os.getenv("MONEYUP_MAX_RAM_GB", "16"))
MIN_FREE_GB = float(os.getenv("MONEYUP_MIN_FREE_GB", "3"))
# Only let the low-system-free signal kill a video if the COLLECTOR's own tree is a real contributor
# (>= this) — otherwise another app (e.g. ComfyUI using ~14 GB) would falsely trip the valve on the
# tiny collector. The collector's own leak is still caught by MAX_RAM_GB.
MIN_FREE_KILL_GB = float(os.getenv("MONEYUP_MIN_FREE_KILL_GB", "6"))
LOCK = config.DATA_DIR / "overnight.lock"                  # single-instance guard (PID lockfile)
_DEGRADED = config.DATA_DIR / "degraded.json"             # ids w/ a video stream but 0 frames -> re-extract


def _kill_tree(pid: int):
    subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True,
                   creationflags=config.CREATE_NO_WINDOW)
    try:
        vlm.unload()                                       # free GPU (Whisper/PaddleOCR die with tree)
    except Exception:
        pass


def _tree_rss_gb(pid: int) -> float:
    """Resident memory (GB) of the pipeline subprocess + ALL descendants (yt-dlp/ffmpeg/vision worker)."""
    try:
        import psutil
        proc = psutil.Process(pid)
        total = 0
        for pr in [proc] + proc.children(recursive=True):
            try:
                total += pr.memory_info().rss
            except Exception:
                pass
        return total / 1e9
    except Exception:
        return 0.0


def _avail_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except Exception:
        return 1e9                                          # unknown -> never trip on this signal


def _is_running_overnight(pid: int) -> bool:
    try:
        import psutil
        return "moneyup_advisor.overnight" in " ".join(psutil.Process(pid).cmdline())
    except Exception:
        return False                                        # gone / unknown -> treat lock as stale


def _acquire_lock() -> bool:
    """Single-instance guard: refuse to start if another LIVE overnight collector holds the lock.
    Self-healing: a stale lock (previous crash, pid not running) is overwritten."""
    try:
        if LOCK.exists():
            old = int((LOCK.read_text(encoding="utf-8").strip() or "0"))
            if old and old != os.getpid() and _is_running_overnight(old):
                log(f"single-instance guard: overnight collector already running (pid {old}); exiting.")
                return False
        LOCK.write_text(str(os.getpid()), encoding="utf-8")
        return True
    except Exception as e:
        log(f"lock warning (continuing): {str(e)[:80]}")
        return True                                         # never block startup on a lock error


def _release_lock():
    try:
        if LOCK.exists() and LOCK.read_text(encoding="utf-8").strip() == str(os.getpid()):
            LOCK.unlink()
    except Exception:
        pass


def _load_degraded():
    try:
        v = json.loads(_DEGRADED.read_text(encoding="utf-8"))
        return v if isinstance(v, list) else []
    except Exception:
        return []


def _remove_degraded(vid):
    try:
        cur = _load_degraded()
        if vid in cur:
            cur.remove(vid)
            _DEGRADED.write_text(json.dumps(cur), encoding="utf-8")
    except Exception:
        pass


def _sheet_frames(vid):
    try:
        return int(json.loads((config.SHEET_DIR / f"{vid}.json").read_text(encoding="utf-8")).get("n_frames_ocr") or 0)
    except Exception:
        return 0


def run_one(vid: str, timeout_s: int, deadline=None, refresh: bool = False) -> str:
    """Run the pipeline for one video in a subprocess. Polls so it can ALSO honor the 05:00 deadline
    MID-video (kills the tree at the deadline, not just between videos). ``refresh`` forces a full
    re-extract that ignores cached frames/vision (for degraded re-extraction). Returns
    'ok' / 'fail' / 'timeout' / 'deadline' / 'ram'."""
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    argv = [sys.executable, "-m", "moneyup_advisor.pipeline", "--video", vid]
    if refresh:
        argv.append("--refresh")
    p = subprocess.Popen(argv, cwd=str(config.REPO_ROOT), env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         creationflags=config.CREATE_NO_WINDOW)
    start = time.time()
    while True:
        try:
            p.wait(timeout=10)                             # poll every 10s
            break
        except subprocess.TimeoutExpired:
            if deadline is not None and now_kst() >= deadline:
                _kill_tree(p.pid)
                return "deadline"                          # 05:00 hard-stop even mid-video
            if time.time() - start >= timeout_s:
                _kill_tree(p.pid)
                return "timeout"
            rss, avail = _tree_rss_gb(p.pid), _avail_gb()   # RAM safety valve (caps the COLLECTOR's peak)
            if rss >= MAX_RAM_GB or (avail <= MIN_FREE_GB and rss >= MIN_FREE_KILL_GB):
                log(f"RAM valve: {vid} tree={rss:.1f}GB sys-free={avail:.1f}GB "
                    f"(cap {MAX_RAM_GB}GB; sys-floor {MIN_FREE_GB}GB only when tree>={MIN_FREE_KILL_GB}GB) "
                    f"— killing tree")
                _kill_tree(p.pid)
                return "ram"
    return "ok" if done(vid) else "fail"


def write_report(rep: dict):
    REPORT.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    import sys
    for _s in (sys.stdout, sys.stderr):                    # Korean-safe when run head-less via .bat
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if not _acquire_lock():                                 # single-instance guard
        return
    import atexit
    atexit.register(_release_lock)
    deadline = next_5am()
    log(f"=== OVERNIGHT BATCH START — hard-stop {deadline:%Y-%m-%d %H:%M KST} ===")
    tickers.ensure_name_map()
    ids = all_video_ids()                                  # newest-first (index 0 = newest)
    have = sum(done(v) for v in ids)
    # NEW uploads (newer than the newest already-done video) FIRST, then backlog NEWEST-first.
    done_pos = [i for i, v in enumerate(ids) if done(v)]
    frontier = min(done_pos) if done_pos else len(ids)     # position of the newest done video
    new_uploads = [ids[i] for i in range(frontier) if not done(ids[i])]          # newest-first
    backlog = [ids[i] for i in range(frontier, len(ids)) if not done(ids[i])]    # already newest-first
    queue = new_uploads + backlog                          # new first, then backlog NEWEST-first
    known = set(ids)                                        # every id seen in a listing (new-upload check)
    log(f"channel videos={len(ids)} · have sheets={have} · NEW uploads={len(new_uploads)} (first) · "
        f"backlog={len(backlog)} (newest-first) · to process={len(queue)}")
    if new_uploads:
        log(f"  new-upload priority queue (first 8): {new_uploads[:8]}")

    n_done = n_fail = total_calls = 0
    total_secs = 0.0
    failures = []

    def snapshot(stopped_reason=None):
        write_report({
            "updated": f"{now_kst():%Y-%m-%d %H:%M:%S KST}",
            "videos_processed": n_done,
            "total_exante_calls": total_calls,
            "avg_min_per_video": round(total_secs / n_done / 60, 2) if n_done else None,
            "failures": failures, "n_failures": n_fail,
            "channel_total": len(ids), "remaining": len(queue),
            "hard_stop": f"{deadline:%Y-%m-%d %H:%M KST}", "stopped_reason": stopped_reason,
        })

    snapshot("running")
    stop_reason = "all done"
    log(f"watchdog: per-video timeout = {WATCHDOG_MIN} min")
    # ---- RE-EXTRACT degraded sheets FIRST (0 frames on a video w/ a stream): force --refresh, 1 at a time ----
    degraded_q = _load_degraded()
    if degraded_q:
        log(f"re-extracting {len(degraded_q)} DEGRADED sheet(s) with --refresh (one at a time): {degraded_q[:8]}")
        for dv in list(degraded_q):
            if now_kst() >= deadline:
                log("05:00 KST reached during degraded re-extraction — stopping.")
                break
            t0 = time.time()
            st = run_one(dv, WATCHDOG_MIN * 60, deadline, refresh=True)
            fr = _sheet_frames(dv)
            if st == "ok" and fr > 0:
                _remove_degraded(dv)
                log(f"RE-EXTRACT OK {dv} | frames={fr} time={time.time()-t0:.0f}s — fixed; removed from queue")
            else:
                log(f"RE-EXTRACT {st} {dv} | frames={fr} — KEPT in degraded queue (will retry next run)")
            snapshot("running")
    last_new_check = now_kst()
    while queue:
        if now_kst() >= deadline:
            stop_reason = "05:00 KST hard-stop"
            log("05:00 KST reached — NOT starting another video. Stopping.")
            break
        # --- CONTINUOUS new-upload check: a video published mid-run jumps to the FRONT of the queue,
        #     ahead of the backlog, so it is processed ASAP (not only at the next 07:00 restart). ---
        if (now_kst() - last_new_check) >= timedelta(minutes=NEW_CHECK_MIN):
            last_new_check = now_kst()
            try:
                fresh = [v for v in recent_ids(15) if v not in known]
            except Exception as e:
                fresh = []
                log(f"new-upload check failed: {str(e)[:120]}")
            for v in fresh:
                known.add(v)                               # remember even if done, so we check once
            qset = set(queue)
            fresh = [v for v in fresh if not done(v) and v not in qset]          # newest-first
            if fresh:
                queue[:0] = fresh                          # jump to FRONT, ahead of the backlog
                log(f"NEW upload(s) detected -> jumped to FRONT of queue: {fresh}")
        vid = queue.pop(0)                                 # newest-first head
        if done(vid):
            continue
        t0 = time.time()
        status = run_one(vid, WATCHDOG_MIN * 60, deadline)     # isolated subprocess; honors 05:00 mid-video
        dt = time.time() - t0
        if status == "deadline":
            stop_reason = "05:00 KST hard-stop (killed mid-video, GPU freed)"
            log(f"05:00 KST reached DURING {vid} — killed it & freed GPU. Stopping.")
            break
        if status == "ok":
            st = vision_stats(vid)
            nc = exante_count(vid)
            n_done += 1
            total_calls += nc
            total_secs += dt
            log(f"OK {vid} | frames={st.get('ocr_frames')} sec/frame={st.get('sec_per_frame')} "
                f"time={dt:.0f}s exante={nc} || TOTAL videos={n_done} calls={total_calls} "
                f"avg={total_secs/n_done/60:.1f}min/video fails={n_fail}")
        elif status == "timeout":
            n_fail += 1
            failures.append({"video_id": vid, "error": f"watchdog timeout >{WATCHDOG_MIN}min"})
            log(f"FAIL {vid} | watchdog timeout (> {WATCHDOG_MIN} min) — killed tree, moving on "
                f"|| fails={n_fail}")
        elif status == "ram":
            n_fail += 1
            failures.append({"video_id": vid, "error": f"RAM safety valve (tree >{MAX_RAM_GB}GB / low free RAM)"})
            log(f"FAIL {vid} | RAM safety valve (tree >{MAX_RAM_GB}GB or sys-free <{MIN_FREE_GB}GB) — "
                f"killed tree, moving on || fails={n_fail}")
        else:
            n_fail += 1
            failures.append({"video_id": vid, "error": "no fact sheet (download/processing empty)"})
            log(f"FAIL {vid} | no fact sheet (download/processing empty) || fails={n_fail}")
        snapshot("running")

    # ---- shutdown: free the GPU for the 5:30 / 6:20 daily jobs ----
    log("unloading GPU models (Qwen3-VL via Ollama keep_alive=0; Whisper/PaddleOCR are per-call) …")
    try:
        vlm.unload()
    except Exception:
        pass
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30,
                             creationflags=config.CREATE_NO_WINDOW).stdout.strip()
        log(f"GPU memory.used after unload: {smi}")
    except Exception:
        pass
    # ---- re-score Phase 1 so the 5/20-day verdict matures automatically (CPU/pykrx, no GPU) ----
    log("re-scoring Phase 1 (matures 5/20-day windows) …")
    try:
        from moneyup_advisor import phase1_callscore as p1
        rep = p1.run()
        (config.DATA_DIR / "phase1_callscore.json").write_text(
            json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        h = rep["horizons"]
        log("phase1 updated: " + " · ".join(
            f"{d}d n={h[str(d)]['n_scorable']} t={h[str(d)]['abnormal']['tstat']}"
            for d in rep["horizons_days"]))
    except Exception as e:
        log(f"phase1 rescore failed: {str(e)[:140]}")

    # ---- Phase 2 playbook auto-update (incremental: cached per-video; Gemini REST, NOT the GPU) ----
    if os.getenv("MONEYUP_PLAYBOOK_AUTO", "1") not in ("0", "false", "no"):
        log("updating Phase 2 playbook (incremental) …")
        try:
            from moneyup_advisor import playbook as pb
            agg = pb.build()
            log(f"playbook updated: {len(agg.get('rules', []))} rules "
                f"over {agg.get('meta', {}).get('n_videos')} videos")
        except Exception as e:
            log(f"playbook update failed: {str(e)[:140]}")

    snapshot(stop_reason)
    log(f"=== MORNING REPORT === videos={n_done} exante_calls={total_calls} "
        f"avg={round(total_secs/n_done/60,2) if n_done else 0}min/video failures={n_fail} "
        f"reason='{stop_reason}' (full report: {REPORT})")


if __name__ == "__main__":
    main()
