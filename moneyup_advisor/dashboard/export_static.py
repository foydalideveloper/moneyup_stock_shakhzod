"""Export the live :8077 dashboard to a self-contained STATIC site under ``dist/`` (for Vercel).

ADDITIVE + READ-ONLY: this never imports-and-runs the server, never touches the running :8077
process, never writes into ``data/_moneyup_advisor/`` (only into ``dist/``), and loads zero models /
uses zero GPU. It reuses the dashboard's OWN renderer functions (``server._index`` / ``_playbook`` /
``_video`` / ``_phase1`` and ``proof.render_html``) so every page is byte-identical to what the server
serves today — the stdlib analog of Flask's test_client (the dashboard is stdlib ``http.server``, not
Flask, so there is no app/url_map/test_client to use).

Routes mirrored (every route ``do_GET`` serves):
  /  ·  /?sort=published  ·  /playbook  ·  /phase1            -> HTML pages
  /playbook_raw  ·  /phase1_raw  ·  /raw?id=<id>              -> raw JSON
  /video?id=<id>  ·  /proof?id=<id>                           -> per-video HTML
  /frame?id=<id>&f=<png>  (referenced by proof pages)         -> copied if the frame still exists,
                                                                else replaced with a clean placeholder

Static layout (Vercel ``cleanUrls`` friendly, and works over file://):
  /                 -> index.html
  /?sort=published  -> published/index.html
  /playbook         -> playbook/index.html
  /phase1           -> phase1/index.html
  /video?id=X       -> video/X/index.html
  /proof?id=X       -> proof/X/index.html
  /raw?id=X         -> raw/X.json
  /playbook_raw     -> playbook_raw.json     /phase1_raw -> phase1_raw.json

    python -m moneyup_advisor.dashboard.export_static            # writes ./dist
    python -m moneyup_advisor.dashboard.export_static <dir>      # writes <dir>
"""
from __future__ import annotations

import json
import posixpath
import re
import shutil
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from moneyup_advisor import config
from moneyup_advisor.dashboard import server

DIST = config.REPO_ROOT / "dist"
SKIPPED: list = []


# --------------------------------------------------------------------------- #
# byte-faithful + safe: reuse the server's own renderers; snapshot the sheets once;
# neutralize the only two data/-writing side effects (mkdir) so nothing touches data/.
# --------------------------------------------------------------------------- #
_SHEETS_CACHE = None


def _cached_sheets(sort: str = "processed"):
    """Same data + ordering as server._sheets(), read ONCE (snapshot). A fact sheet that is
    mid-write / unparseable is skipped and logged (tolerated, never crashes)."""
    global _SHEETS_CACHE
    if _SHEETS_CACHE is None:
        _SHEETS_CACHE = []
        for p in sorted(config.SHEET_DIR.glob("*.json")):
            try:
                _SHEETS_CACHE.append(json.loads(p.read_text(encoding="utf-8")))
            except Exception as e:
                SKIPPED.append(f"factsheet {p.name} (mid-write?): {str(e)[:60]}")
    out = list(_SHEETS_CACHE)
    if sort == "published":
        out.sort(key=lambda s: (s.get("publish_datetime") or s.get("publish_date") or ""), reverse=True)
    else:
        out.sort(key=lambda s: (s.get("extracted_at") or ""), reverse=True)
    return out


server._sheets = _cached_sheets                              # additive monkeypatch (snapshot; same bytes)
config.video_frame_dir = lambda vid: config.FRAME_DIR / vid  # SAFETY: drop the mkdir side effect (read-only)


def renderable_ids():
    """video ids that actually render a page = those with a parseable fact sheet (excludes mid-write)."""
    return sorted({s.get("video_id") for s in server._sheets() if s.get("video_id")})


# --------------------------------------------------------------------------- #
# route -> static output path
# --------------------------------------------------------------------------- #
def to_static(url: str):
    u = urlparse(url)
    q = parse_qs(u.query)
    def g(k):
        return (q.get(k) or [""])[0]
    path = u.path
    if path == "/":
        return "published/index.html" if g("sort") == "published" else "index.html"
    if path == "/playbook":
        return "playbook/index.html"
    if path == "/playbook_raw":
        return "playbook_raw.json"
    if path == "/phase1":
        return "phase1/index.html"
    if path == "/phase1_raw":
        return "phase1_raw.json"
    if path == "/video":
        return f"video/{g('id')}/index.html"
    if path == "/raw":
        return f"raw/{g('id')}.json"
    if path == "/proof":
        return f"proof/{g('id')}/index.html"
    if path == "/frame":
        return f"frame/{g('id')}/{g('f')}"
    return None                                             # external / unknown -> leave untouched


# --------------------------------------------------------------------------- #
# render a route byte-faithfully via the server's OWN functions -> (text, kind)
# --------------------------------------------------------------------------- #
def render(url: str):
    u = urlparse(url)
    q = parse_qs(u.query)
    def g(k):
        return (q.get(k) or [""])[0]
    path = u.path
    if path == "/":
        return server._index(g("sort") or "processed"), "html"
    if path == "/playbook":
        return server._playbook(), "html"
    if path == "/phase1":
        return server._phase1(), "html"
    if path == "/playbook_raw":
        p = config.DATA_DIR / "playbook" / "playbook.json"
        return (p.read_text(encoding="utf-8") if p.exists() else "{}"), "json"
    if path == "/phase1_raw":
        p = config.DATA_DIR / "phase1_callscore.json"
        return (p.read_text(encoding="utf-8") if p.exists() else "{}"), "json"
    if path == "/video":
        return server._video(g("id")), "html"
    if path == "/raw":
        s = next((x for x in server._sheets() if x.get("video_id") == g("id")), {})
        return json.dumps(s, ensure_ascii=False, indent=2), "json"
    if path == "/proof":
        import moneyup_advisor.proof as proof                # lazy (like the server); no model / no GPU
        return proof.render_html(g("id"), embed=False), "html"
    raise ValueError(f"unknown route {url}")


# --------------------------------------------------------------------------- #
# rewrite links + handle proof frame images
# --------------------------------------------------------------------------- #
_FRAME_RX = re.compile(r"<img src='(/frame\?[^']*)'>")
_LINK_RX = re.compile(r"(href|src)='(/[^']*)'")             # internal absolute links only (start with /)


def handle_frames(htmlstr: str, src_path: str) -> str:
    """Proof pages reference frames via /frame?id=&f=. Copy each frame that still exists into
    dist/frame/<id>/<png> (relative link); if it was auto-deleted, drop a CLEAN placeholder so the
    page still renders with no broken-image icon and no layout break."""
    src_dir = posixpath.dirname(src_path)

    def repl(m):
        q = parse_qs(urlparse(m.group(1)).query)
        vid = (q.get("id") or [""])[0]
        fn = (q.get("f") or [""])[0]
        srcfile = config.FRAME_DIR / vid / fn
        if fn and srcfile.exists():
            dst = DIST / "frame" / vid / fn
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(srcfile, dst)
            rel = posixpath.relpath(f"frame/{vid}/{fn}", src_dir or ".")
            return f"<img src='{rel}'>"
        return ("<div class='frame-removed' style='width:560px;max-width:52%;border:1px dashed #d0d7de;"
                "border-radius:6px;padding:28px 14px;box-sizing:border-box;text-align:center;color:#8b949e;"
                "background:#fafbfc'>🗙 frame image removed by disk-cleanup · the exact OCR read from this "
                "frame is shown →</div>")

    return _FRAME_RX.sub(repl, htmlstr)


def rewrite_links(htmlstr: str, src_path: str) -> str:
    """Rewrite internal absolute links (href/src starting with /) to RELATIVE static paths, so
    navigation works over file:// AND on Vercel. External http(s) links are left untouched."""
    src_dir = posixpath.dirname(src_path)

    def repl(m):
        attr, url = m.group(1), m.group(2)
        tgt = to_static(url)
        if tgt is None:
            return m.group(0)
        return f"{attr}='{posixpath.relpath(tgt, src_dir or '.')}'"

    return _LINK_RX.sub(repl, htmlstr)


def write(path_rel: str, data) -> None:
    dst = DIST / path_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        dst.write_text(data, encoding="utf-8")
    else:
        dst.write_bytes(data)


# --------------------------------------------------------------------------- #
# security: never ship secrets. Scan dist/ for secret-shaped strings AND the actual
# values from .env; abort the whole export if anything matches.
# --------------------------------------------------------------------------- #
# secret-shaped patterns (NOT the bare words KEY/TOKEN — the proof pages legitimately say "OCR tokens";
# we match the dangerous forms: supabase/service_role/password + *_KEY/*_TOKEN env names + key value shapes)
_SECRET_RX = re.compile(
    r"supabase|service_role|password|passwd|[A-Za-z]*_?api[_-]?key|secret[_-]?key|access[_-]?key|"
    r"private[_-]?key|gemini_api_key|[A-Z][A-Z0-9_]{2,}_(?:KEY|TOKEN|SECRET)\b|"
    r"bearer\s+[A-Za-z0-9._-]{12,}|sk-[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_-]{30,}|ghp_[A-Za-z0-9]{30,}|"
    r"eyJ[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}",
    re.IGNORECASE)


# only the values of SECRET-named .env keys are treated as must-not-leak strings; benign config
# (WHISPER_MODEL=large-v3, channel handles, ports, …) legitimately appears in the data and is ignored.
_SECRET_KEY_RX = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|PASSWD|PW|SUPABASE|SERVICE_ROLE|CREDENTIAL|AUTH",
                            re.IGNORECASE)


def _env_secret_values():
    vals = []
    envp = config.REPO_ROOT / ".env"
    if envp.exists():
        for line in envp.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.strip().strip('"').strip("'")
            if len(v) >= 8 and _SECRET_KEY_RX.search(k):
                vals.append(v)
    return vals


def secret_scan():
    hits = []
    env_vals = _env_secret_values()
    for f in DIST.rglob("*"):
        if not f.is_file() or f.suffix.lower() not in (".html", ".json", ".txt", ".js", ".css", ".svg"):
            continue
        try:
            txt = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        m = _SECRET_RX.search(txt)
        if m:
            hits.append((f.relative_to(DIST), f"pattern:{m.group(0)[:32]}"))
        for v in env_vals:
            if v and v in txt:
                hits.append((f.relative_to(DIST), "ENV-VALUE-LEAK"))
    return hits


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def export(dist=None):
    global DIST
    if dist is not None:
        d = Path(dist)
        DIST = d if d.is_absolute() else (config.REPO_ROOT / d)
    # safe clean (re-runnable): only ever wipe a dir literally named 'dist'
    if DIST.exists():
        if DIST.name != "dist":
            print(f"refusing to clear non-'dist' directory: {DIST}")
            return 2
        shutil.rmtree(DIST)
    DIST.mkdir(parents=True, exist_ok=True)

    routes = ["/", "/?sort=published", "/playbook", "/phase1", "/playbook_raw", "/phase1_raw"]
    for vid in renderable_ids():
        routes += [f"/video?id={vid}", f"/raw?id={vid}", f"/proof?id={vid}"]

    n_html = n_json = 0
    for route in routes:
        sp = to_static(route)
        try:
            body, kind = render(route)
        except Exception as e:
            SKIPPED.append(f"{route} -> {sp}: {str(e)[:80]}")
            continue
        if kind == "html":
            body = rewrite_links(handle_frames(body, sp), sp)
            n_html += 1
        else:
            n_json += 1
        write(sp, body)

    write("vercel.json", json.dumps({"cleanUrls": True}, indent=2))

    # ---- security gate ----
    hits = secret_scan()
    if hits:
        print("\n!!! SECRET SCAN TRIPPED — aborting (NOT safe to deploy) !!!")
        for rel, what in hits[:50]:
            print(f"   {rel}  [{what}]")
        print(f"   {len(hits)} match(es). dist/ left in place for inspection; do NOT deploy.")
        return 1

    files = [f for f in DIST.rglob("*") if f.is_file()]
    total = sum(f.stat().st_size for f in files)
    print(f"\n[export] dist  : {DIST}")
    print(f"[export] pages : {n_html} HTML · {n_json} JSON · {len(files)} files total")
    print(f"[export] size  : {total/1e6:.2f} MB")
    if SKIPPED:
        print(f"[export] skipped {len(SKIPPED)} (tolerated):")
        for s in SKIPPED[:20]:
            print(f"   - {s}")
    print("[export] secret scan: CLEAN (no SUPABASE / *_KEY / *_TOKEN / PASSWORD / SERVICE_ROLE / key-shapes / .env values)")
    return 0


def main():
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    dist = sys.argv[1] if len(sys.argv) > 1 else None
    sys.exit(export(dist))


if __name__ == "__main__":
    main()
