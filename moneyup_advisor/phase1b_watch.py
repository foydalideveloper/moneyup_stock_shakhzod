"""Phase 1B auto-run watcher: wait until the degraded-sheet queue is DRAINED (or only terminal,
repeatedly-failing ids remain), then run the Phase 1B scorer ONCE on the clean corpus, log it, stop.

"Clean" = degraded.json has been UNCHANGED for STABLE_MIN minutes (> the 25-min per-video watchdog,
so a single slow re-extract can't look 'stable'). That covers both empty AND only-terminal-left (e.g.
1JXAsnyXiSA whose source video won't download). Idempotent: a done-flag makes it a no-op after it runs
(so an at-logon re-launch on reboot just exits). Uses zero GPU (pykrx prices only) — safe alongside the
collector. NEVER runs before the queue is drained.

    python -m moneyup_advisor.phase1b_watch
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime

from moneyup_advisor import config

DEG = config.DATA_DIR / "degraded.json"
DONE = config.DATA_DIR / "phase1b_watch.done"
LOG = config.DATA_DIR / "phase1b_watch.log"
RESULT = config.DATA_DIR / "phase1b_callscore.json"
STABLE_MIN = int(__import__("os").getenv("MONEYUP_P1B_STABLE_MIN", "60"))   # > 25-min watchdog
POLL_S = 300
MAX_WAIT_H = 48


def _log(msg: str):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _degraded():
    try:
        v = json.loads(DEG.read_text(encoding="utf-8"))
        return v if isinstance(v, list) else []
    except Exception:
        return []


def main():
    if DONE.exists():
        _log("done flag present — Phase 1B already run on a clean corpus; exiting.")
        return
    _log(f"watcher started — polling degraded.json until unchanged for {STABLE_MIN} min (drained/terminal).")
    t_start = time.time()
    last, t_change = None, time.time()
    while True:
        cur = _degraded()
        if cur != last:
            last, t_change = cur, time.time()
            _log(f"degraded queue = {len(cur)} remaining {cur[:6]}")
        stable_min = (time.time() - t_change) / 60.0
        if stable_min >= STABLE_MIN:
            _log(f"queue STABLE {stable_min:.0f} min at {len(cur)} id(s) {cur} — treating corpus as clean "
                 f"(empty or only terminal). Running Phase 1B ONCE.")
            break
        if (time.time() - t_start) / 3600.0 >= MAX_WAIT_H:
            _log(f"max wait {MAX_WAIT_H}h hit — running Phase 1B anyway on current corpus.")
            break
        time.sleep(POLL_S)

    _log("running: python -m moneyup_advisor.phase1b (full corpus, pykrx prices, zero GPU)…")
    try:
        r = subprocess.run([sys.executable, "-m", "moneyup_advisor.phase1b"],
                           cwd=str(config.REPO_ROOT), capture_output=True, text=True,
                           encoding="utf-8", errors="replace",
                           creationflags=config.CREATE_NO_WINDOW, timeout=7200)
    except Exception as e:
        _log(f"phase1b launch error: {str(e)[:200]}")
        return
    _log("phase1b output (tail):\n  " + "\n  ".join((r.stdout or "").splitlines()[-14:]))
    if r.returncode == 0 and RESULT.exists():
        DONE.write_text(datetime.now().isoformat(), encoding="utf-8")
        _log(f"Phase 1B COMPLETE -> {RESULT}  (done flag set; watcher stops).")
    else:
        _log(f"phase1b FAILED rc={r.returncode}; stderr tail:\n  "
             + "\n  ".join((r.stderr or "").splitlines()[-10:]))


if __name__ == "__main__":
    main()
