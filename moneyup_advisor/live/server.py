# -*- coding: utf-8 -*-
"""LIVE 'watch it extract' demo server — stdlib http.server on its OWN port (default 8078).

    python -m moneyup_advisor.live.server          # http://127.0.0.1:8078/live

Brand-new, isolated app: a different module + different port from the dashboard (:8077). It only
(a) serves the demo page, (b) starts ONE on-demand job at a time (spawning live.job as a subprocess),
(c) serves the per-job folder the job writes to, and (d) PAUSES the collector for the demo run and
RESUMES it after (server-managed, so it always resumes even if the job crashes). It imports/edits
NOTHING in the collector, daily report, playbook, or existing dashboard pages.
"""
from __future__ import annotations

import ctypes
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from moneyup_advisor import config
from moneyup_advisor.live.page import PAGE

LIVE_DIR = config.DATA_DIR / "live_jobs"
LIVE_DIR.mkdir(parents=True, exist_ok=True)
HOST = os.getenv("MONEYUP_LIVE_HOST", "127.0.0.1")
PORT = int(os.getenv("MONEYUP_LIVE_PORT", "8078"))
PAUSE_COLLECTOR = os.getenv("MONEYUP_LIVE_PAUSE", "1") not in ("0", "false", "no")
COLLECTOR_TASK = os.getenv("MONEYUP_LIVE_COLLECTOR_TASK", "MoneyUp_Batch")
_PAUSED_MARKER = LIVE_DIR / ".collector_paused"

_CTYPE = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
          ".json": "application/json; charset=utf-8", ".md": "text/markdown; charset=utf-8",
          ".txt": "text/plain; charset=utf-8", ".html": "text/html; charset=utf-8",
          ".mp4": "video/mp4", ".m4v": "video/mp4", ".m4a": "audio/mp4",
          ".webm": "video/webm", ".mov": "video/quicktime", ".mp3": "audio/mpeg",
          ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}
_MEDIA = {".mp4", ".m4v", ".m4a", ".webm", ".mov", ".mp3"}     # served with HTTP Range (seekable)

# ---- single-job state (one demo job at a time) ----
_LOCK = threading.Lock()
_CURRENT = {"job": None, "proc": None}

# ---- startup self-check (item 8): is the GPU/ML stack healthy before we demo to the boss? ----
_SELFCHECK = {"whisper": "checking", "ollama": "checking", "model": None, "checked": False}


def _run_selfcheck():
    """Probe the two things a real run needs: faster_whisper (the wedge symptom) and Ollama+Qwen.
    Runs once in a background thread on startup; result is surfaced via /health so the page can warn
    BEFORE a job is started."""
    try:
        r = subprocess.run([sys.executable, "-c", "import faster_whisper"], capture_output=True,
                           timeout=45, creationflags=config.CREATE_NO_WINDOW)
        _SELFCHECK["whisper"] = "ok" if r.returncode == 0 else "error"
    except subprocess.TimeoutExpired:
        _SELFCHECK["whisper"] = "wedged"            # import hung -> GPU/ML stack wedged (reboot)
    except Exception:
        _SELFCHECK["whisper"] = "error"
    try:
        import urllib.request
        data = json.loads(urllib.request.urlopen(config.OLLAMA_HOST + "/api/tags", timeout=5).read())
        names = [m.get("name", "") for m in (data.get("models") or [])]
        base = config.QWEN_VL_MODEL.split(":")[0]
        _SELFCHECK["ollama"] = "ok"
        _SELFCHECK["model"] = any(base in n for n in names)
    except Exception:
        _SELFCHECK["ollama"] = "down"
        _SELFCHECK["model"] = False
    _SELFCHECK["checked"] = True
    print(f"[live] self-check: whisper={_SELFCHECK['whisper']} ollama={_SELFCHECK['ollama']} "
          f"qwen_model={_SELFCHECK['model']}")


# --------------------------------------------------------------------------- #
# collector pause / resume (external; never edits collector code)
# --------------------------------------------------------------------------- #
def _overnight_pids():
    try:
        import psutil
    except Exception:
        return []
    out = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name = (p.info.get("name") or "").lower()
            cl = p.info.get("cmdline") or []
            # PRECISE: a real collector is `python -m moneyup_advisor.overnight` — match the module as a
            # standalone argv TOKEN on a python process, so we never false-match a shell/grep/diagnostic
            # whose command line merely CONTAINS the string (which would wrongly suspend it).
            if "python" in name and "moneyup_advisor.overnight" in cl:
                out.append(p)
        except Exception:
            pass
    return out


_PROCESS_SUSPEND_RESUME = 0x0800


def _suspend_pid(pid: int, suspend: bool) -> bool:
    """Freeze/unfreeze ONE process via ntdll (no kill, no restart). Best-effort."""
    try:
        h = ctypes.windll.kernel32.OpenProcess(_PROCESS_SUSPEND_RESUME, False, int(pid))
        if not h:
            return False
        fn = ctypes.windll.ntdll.NtSuspendProcess if suspend else ctypes.windll.ntdll.NtResumeProcess
        fn(h)
        ctypes.windll.kernel32.CloseHandle(h)
        return True
    except Exception:
        return False


def pause_collector() -> bool:
    """SUSPEND the single running collector's process tree (freeze its CPU/GPU/network/disk) for the
    demo run — NO kill, NO restart, lock untouched, so a duplicate collector can never be spawned.
    Returns True if we suspended something (so the caller knows to resume afterwards)."""
    if not PAUSE_COLLECTOR:
        return False
    try:
        import psutil  # noqa: F401
    except Exception:
        return False
    masters = _overnight_pids()
    if not masters:
        return False
    pids = []
    for m in masters:
        try:
            kids = m.children(recursive=True)
        except Exception:
            kids = []
        for c in kids:                          # children first so the master can't spawn more
            if _suspend_pid(c.pid, True):
                pids.append(c.pid)
        if _suspend_pid(m.pid, True):
            pids.append(m.pid)
    if not pids:
        return False
    try:
        _PAUSED_MARKER.write_text(json.dumps(pids), encoding="utf-8")
    except Exception:
        pass
    print(f"[live] paused (suspended) collector tree: {pids}")
    return True


def _parse_paused_marker():
    """Read the paused marker into a list[int] of PIDs, tolerating ANY content:
    a JSON list ([123,456]), a bare int / "1" (the legacy kill-design marker), a comma/space list,
    or garbage. Never raises — a malformed marker yields [] and is logged."""
    if not _PAUSED_MARKER.exists():
        return []
    raw = ""
    try:
        raw = (_PAUSED_MARKER.read_text(encoding="utf-8") or "").strip()
    except Exception as e:
        print(f"[live] could not read paused marker (ignored): {str(e)[:120]}")
        return []
    if not raw:
        return []
    val = raw
    try:
        val = json.loads(raw)
    except Exception:
        val = raw                               # not JSON -> treat as a raw string below
    if isinstance(val, bool):                   # bool is an int subclass; ignore True/False
        return []
    if isinstance(val, int):
        return [val]
    if isinstance(val, (list, tuple)):
        return [int(p) for p in val if isinstance(p, int) and not isinstance(p, bool)]
    if isinstance(val, str):
        return [int(tok) for tok in val.replace(",", " ").split() if tok.lstrip("-").isdigit()]
    return []


def resume_collector():
    """RESUME (un-suspend) the collector tree we froze; also un-suspends any live overnight tree as a
    safety net. Robust to a stale/legacy/malformed marker (logs + continues, never crashes). Never
    spawns a new process, so it cannot create duplicate collectors."""
    pids = []
    try:
        pids = _parse_paused_marker()
    except Exception as e:                       # belt-and-suspenders: parsing must never crash startup
        print(f"[live] paused-marker parse failed (ignored): {str(e)[:120]}")
        pids = []
    for pid in pids:
        try:
            _suspend_pid(pid, False)
        except Exception:
            pass
    try:                                        # safety: thaw any current overnight tree too
        for m in _overnight_pids():
            _suspend_pid(m.pid, False)
            for c in m.children(recursive=True):
                _suspend_pid(c.pid, False)
    except Exception:
        pass
    try:                                        # always clear the marker, even if it was malformed
        if _PAUSED_MARKER.exists():
            _PAUSED_MARKER.unlink()
    except Exception as e:
        print(f"[live] could not clear paused marker (ignored): {str(e)[:120]}")
    if pids:
        print(f"[live] resumed (un-suspended) collector tree: {pids}")


# --------------------------------------------------------------------------- #
# job lifecycle
# --------------------------------------------------------------------------- #
def _new_job_id() -> str:
    return "live_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + os.urandom(2).hex()


def _job_active() -> bool:
    proc = _CURRENT.get("proc")
    return proc is not None and proc.poll() is None


def _spawn(job_id: str, url=None, file=None, fast=False):
    jd = LIVE_DIR / job_id
    jd.mkdir(parents=True, exist_ok=True)
    argv = [sys.executable, "-m", "moneyup_advisor.live.job", "--job", job_id]
    argv += (["--file", file] if file else ["--url", url])
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
           "PYTHONPATH": str(config.REPO_ROOT)}
    if fast:                                            # FAST DEMO: real pipeline over the FULL 5 min, sparser
        env["MONEYUP_LIVE_SECONDS"] = os.getenv("MONEYUP_LIVE_FAST_SECONDS", "300")   # full 5 minutes
        env["MONEYUP_FPS"] = os.getenv("MONEYUP_LIVE_FAST_FPS", "0.5")                # sample sparsely across it
        env["MONEYUP_LIVE_FRAME_CAP"] = os.getenv("MONEYUP_LIVE_FAST_CAP", "36")      # ~36 frames spread over 5 min
        env["MONEYUP_LIVE_FMT"] = os.getenv(                                          # 720p -> faster download
            "MONEYUP_LIVE_FAST_FMT",
            "bestvideo[height<=720][vcodec^=avc1]+bestaudio[ext=m4a]/bestvideo[height<=720]+bestaudio/"
            "best[height<=720]/best")
        env["MONEYUP_LIVE_VLM_MAX"] = os.getenv("MONEYUP_LIVE_FAST_VLM", "3")
        fw = os.getenv("MONEYUP_LIVE_FAST_WHISPER", "")
        if fw:                                          # optional faster Whisper (default keeps large-v3)
            env["MONEYUP_WHISPER_MODEL"] = fw
    log = open(jd / "job.log", "w", encoding="utf-8")
    paused = pause_collector()
    proc = subprocess.Popen(argv, cwd=str(config.REPO_ROOT), env=env, stdout=log, stderr=log,
                            creationflags=config.CREATE_NO_WINDOW)
    _CURRENT["job"], _CURRENT["proc"] = job_id, proc

    def _watch():
        try:
            proc.wait()
        finally:
            log.close()
            if paused:
                resume_collector()
    threading.Thread(target=_watch, daemon=True).start()
    return job_id


def start_job(url=None, file=None, fast=False):
    with _LOCK:
        if _job_active():
            return None, "a demo job is already running — one at a time"
        job_id = _new_job_id()
        _spawn(job_id, url=url, file=file, fast=fast)
        return job_id, None


# --------------------------------------------------------------------------- #
# run index (auto-replay by link) + static export
# --------------------------------------------------------------------------- #
def _parse_yt_id(url):
    url = (url or "").strip()
    if re.fullmatch(r"[\w-]{11}", url):
        return url
    m = (re.search(r"[?&]v=([\w-]{11})", url) or re.search(r"youtu\.be/([\w-]{11})", url)
         or re.search(r"/shorts/([\w-]{11})", url) or re.search(r"/live/([\w-]{11})", url)
         or re.search(r"/embed/([\w-]{11})", url))
    return m.group(1) if m else None


def _completed_runs():
    """Every completed run (info.json present + done), newest first — for the picker + lookup."""
    out = []
    for d in LIVE_DIR.glob("live_*"):
        ip = d / "info.json"
        if ip.is_file():
            try:
                info = json.loads(ip.read_text(encoding="utf-8"))
                if info.get("done") and (d / "events.jsonl").is_file():
                    out.append(info)
            except Exception:
                pass
    out.sort(key=lambda r: r.get("created", 0), reverse=True)
    return out


def _find_cached(video_id):
    if not video_id:
        return None
    for r in _completed_runs():
        if r.get("video_id") == video_id:
            return r
    return None


def build_export(job_id):
    """Bundle one completed run into a SELF-CONTAINED static folder under live_jobs/_exports/<job>/
    (index.html with the events EMBEDDED + media via relative ./ paths). It plays the full animated
    replay in a browser with NO server and NO GPU, and is ready to deploy to Vercel. No secrets."""
    from moneyup_advisor.live.page import static_index
    jd = LIVE_DIR / job_id
    if not (jd / "events.jsonl").is_file():
        raise ValueError("unknown or incomplete run")
    info = {}
    try:
        info = json.loads((jd / "info.json").read_text(encoding="utf-8"))
    except Exception:
        pass
    out = LIVE_DIR / "_exports" / job_id
    if out.exists():
        shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("thumbs", "full"):
        if (jd / sub).is_dir():
            shutil.copytree(jd / sub, out / sub)
    for f in ("clip.mp4", "clip.m4a", "factsheet.json", "factsheet.md", "summary_full.docx",
              "summary_full_en.docx", "raw_full.docx", "raw_full.json", "comparison.docx"):
        if (jd / f).is_file():
            shutil.copy(jd / f, out / f)
    evs = [json.loads(l) for l in (jd / "events.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    (out / "index.html").write_text(
        static_index(info.get("title") or job_id, json.dumps(evs, ensure_ascii=False)), encoding="utf-8")
    (out / "vercel.json").write_text(json.dumps({"cleanUrls": True}), encoding="utf-8")
    (out / "README.txt").write_text(
        "MoneyUp live-extraction replay (research / mock only).\n"
        "Self-contained static site: open index.html in a browser, or deploy this folder to Vercel:\n"
        "    npm i -g vercel\n    cd \"<this folder>\"\n    vercel deploy --prod\n", encoding="utf-8")
    return out


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
def _safe_job_file(job: str, name: str):
    if not job or "/" in job or "\\" in job or ".." in job:
        return None
    base = (LIVE_DIR / job).resolve()
    target = (base / name).resolve()
    if base not in target.parents and target != base:
        return None
    return target


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype="text/html; charset=utf-8", code=200):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except Exception:
            pass

    def _json(self, obj, code=200):
        self._send(json.dumps(obj, ensure_ascii=False), "application/json; charset=utf-8", code)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path in ("/", "/live"):
            self._send(PAGE)
        elif u.path == "/health":
            self._json({"ok": True, "active": _job_active(), "job": _CURRENT.get("job"),
                        "selfcheck": _SELFCHECK})
        elif u.path == "/status":
            self._status(q)
        elif u.path == "/file":
            self._file(q)
        elif u.path == "/lookup":                     # is this video already processed? (auto-replay)
            c = _find_cached((q.get("vid") or [""])[0])
            self._json({"cached": bool(c), "job": (c or {}).get("job_id"), "info": c})
        elif u.path == "/runs":                       # the processed-video picker
            self._json({"runs": _completed_runs()[:60]})
        elif u.path == "/export":                     # build the self-contained static replay bundle
            try:
                p = build_export((q.get("job") or [""])[0])
                self._json({"ok": True, "path": str(p)})
            except Exception as e:
                self._json({"error": str(e)[:200]}, 400)
        else:
            self._send("404 — open <a href='/live'>/live</a>", code=404)

    def _status(self, q):
        job = (q.get("job") or [""])[0]
        cursor = int((q.get("cursor") or ["0"])[0] or 0)
        jd = LIVE_DIR / job
        if not job or not jd.is_dir():
            self._json({"error": "unknown job"}, 404)
            return
        events = []
        evp = jd / "events.jsonl"
        if evp.exists():
            try:
                lines = evp.read_text(encoding="utf-8").splitlines()
                for ln in lines[cursor:]:
                    if ln.strip():
                        events.append(json.loads(ln))
                cursor = len(lines)
            except Exception:
                pass
        st = {}
        sp = jd / "status.json"
        if sp.exists():
            try:
                st = json.loads(sp.read_text(encoding="utf-8"))
            except Exception:
                pass
        self._json({"state": st.get("state", "running"), "pct": st.get("pct", 0),
                    "stage": st.get("stage"), "cursor": cursor, "events": events})

    def _file(self, q):
        job = (q.get("job") or [""])[0]
        name = (q.get("name") or [""])[0]
        target = _safe_job_file(job, name)
        if not target or not target.is_file():
            self._send("not found", code=404)
            return
        ctype = _CTYPE.get(target.suffix.lower(), "application/octet-stream")
        size = target.stat().st_size
        rng = self.headers.get("Range")
        if rng and target.suffix.lower() in _MEDIA:           # seekable media (clip + audio player)
            self._send_range(target, size, ctype, rng)
            return
        try:
            data = target.read_bytes()
        except Exception:
            self._send("error", code=500)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            self.wfile.write(data)
        except Exception:
            pass

    def _send_range(self, target, size, ctype, rng):
        import re as _re
        m = _re.match(r"bytes=(\d*)-(\d*)", (rng or "").strip())
        start, end = 0, size - 1
        if m:
            if m.group(1):
                start = int(m.group(1))
            if m.group(2):
                end = int(m.group(2))
        start = max(0, start)
        end = min(end, size - 1)
        if start > end:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return
        length = end - start + 1
        self.send_response(206)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            with open(target, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(65536, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except Exception:
            pass

    def do_POST(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        length = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(length) if length else b""
        if u.path == "/start":
            try:
                data = json.loads(body or b"{}")
            except Exception:
                data = {}
            url = (data.get("url") or "").strip()
            fresh = bool(data.get("fresh"))
            fast = bool(data.get("fast"))
            if not url:
                self._json({"error": "no url"}, 400)
                return
            vid = _parse_yt_id(url)
            if not fresh and vid:                          # AUTO-REPLAY: already processed -> instant, zero GPU
                cached = _find_cached(vid)
                if cached:
                    self._json({"job": cached["job_id"], "cached": True, "info": cached})
                    return
            job_id, err = start_job(url=url, fast=fast)
            self._json({"error": err}, 409) if err else self._json({"job": job_id, "cached": False})
        elif u.path == "/upload":
            name = (q.get("name") or ["upload.mp4"])[0]
            fast = (q.get("fast") or ["0"])[0] in ("1", "true", "yes")
            ext = os.path.splitext(name)[1].lower() or ".mp4"
            if ext not in (".mp4", ".mkv", ".mov", ".webm", ".m4v", ".avi"):
                self._json({"error": f"unsupported file type {ext}"}, 400)
                return
            if not body:
                self._json({"error": "empty upload"}, 400)
                return
            with _LOCK:
                if _job_active():
                    self._json({"error": "a demo job is already running — one at a time"}, 409)
                    return
                job_id = _new_job_id()
                jd = LIVE_DIR / job_id
                jd.mkdir(parents=True, exist_ok=True)
                up = jd / ("upload" + ext)
                up.write_bytes(body)
                _spawn(job_id, file=str(up), fast=fast)
            self._json({"job": job_id, "cached": False})
        else:
            self._json({"error": "not found"}, 404)


def serve(host=HOST, port=PORT):
    if _PAUSED_MARKER.exists():                # recover: a prior run died while the collector was paused
        print("[live] found a collector-paused marker on startup -- clearing it (log + continue)")
        try:
            resume_collector()                 # robust: handles stale/legacy/malformed markers
        except Exception as e:                 # must NEVER crash startup on a bad marker
            print(f"[live] startup resume failed (ignored): {str(e)[:160]}")
            try:
                _PAUSED_MARKER.unlink()
            except Exception:
                pass
    threading.Thread(target=_run_selfcheck, daemon=True).start()   # GPU/ML stack health (item 8)
    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"MoneyUp LIVE demo -> http://{host}:{port}/live   "
          f"(isolated | collector pause={'on' if PAUSE_COLLECTOR else 'off'} | "
          f"clip={os.getenv('MONEYUP_LIVE_SECONDS','300')}s | dashboard :8077 untouched)")
    try:
        srv.serve_forever()
    finally:
        try:
            resume_collector()                 # never leave the collector paused
        except Exception as e:
            print(f"[live] shutdown resume failed (ignored): {str(e)[:160]}")


if __name__ == "__main__":
    serve()
