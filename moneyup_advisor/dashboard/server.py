"""Standalone 머니업 Advisor dashboard — stdlib http.server, brand-new port (default 8077).

Reads the fused fact sheets from ``data/_moneyup_advisor/factsheets/*.json`` and renders them.
Completely separate from the existing dashboard (port 8000): different module, different port,
no shared state, no edits to the existing app.

    python -m moneyup_advisor.dashboard.server            # serves http://127.0.0.1:8077
"""
from __future__ import annotations

import html
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from moneyup_advisor import config

_TAG_COLOR = {"AGREE": "#1a7f37", "AUDIO-ONLY": "#0969da",
              "VIDEO-ONLY": "#bc4c00", "CONFLICT": "#cf222e"}
_CSS = """
body{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f6f8fa;color:#1f2328}
header{background:#0d1117;color:#fff;padding:14px 22px}
header b{color:#58a6ff}
.wrap{max-width:1100px;margin:0 auto;padding:18px 22px}
.card{background:#fff;border:1px solid #d0d7de;border-radius:8px;padding:14px 18px;margin:14px 0}
table{border-collapse:collapse;width:100%;margin:6px 0;font-size:13px}
th,td{border:1px solid #d0d7de;padding:5px 8px;text-align:left;vertical-align:top}
th{background:#f6f8fa}
.num{text-align:right;font-variant-numeric:tabular-nums}
.tag{color:#fff;padding:1px 7px;border-radius:10px;font-size:11px;font-weight:700}
a{color:#0969da;text-decoration:none}a:hover{text-decoration:underline}
.pill{display:inline-block;background:#eaeef2;border-radius:12px;padding:1px 9px;margin:2px;font-size:12px}
.dir-long{color:#1a7f37;font-weight:700}.dir-short{color:#cf222e;font-weight:700}.dir-avoid{color:#9a6700;font-weight:700}
small{color:#656d76}
"""


def _esc(x):
    return html.escape(str(x)) if x is not None else "—"


def _fmt(v):
    return f"{v:,}" if isinstance(v, int) else (_esc(v) if v is not None else "—")


def _sheets(sort: str = "processed"):
    out = []
    for p in sorted(config.SHEET_DIR.glob("*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    if sort == "published":
        out.sort(key=lambda s: (s.get("publish_datetime") or s.get("publish_date") or ""), reverse=True)
    else:                                                  # recently PROCESSED first (default)
        out.sort(key=lambda s: (s.get("extracted_at") or ""), reverse=True)
    return out


def _page(body, title="머니업 AI Advisor"):
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{_esc(title)}</title>"
            f"<style>{_CSS}</style></head><body>"
            f"<header><b>머니업</b> AI Advisor — Phase 0 fact sheets "
            f"<small style='color:#8b949e'>(grounded · OCR=source of truth · mock-only)</small></header>"
            f"<div class='wrap'>{body}</div></body></html>")


def _phase1():
    p = config.DATA_DIR / "phase1_callscore.json"
    if not p.exists():
        return _page("<div class='card'>No Phase-1 results yet. Run "
                     "<code>python -m moneyup_advisor.phase1_callscore</code>. <a href='/'>← back</a></div>")
    r = json.loads(p.read_text(encoding="utf-8"))

    def tcell(t):
        if t is None:
            return "—"
        col = "#1a7f37" if t >= r["tstat_bar"] else "#cf222e" if t <= -r["tstat_bar"] else "#57606a"
        return f"<b style='color:{col}'>{t}</b>"

    B = [f"<p><a href='/'>← fact sheets</a> · <a href='/phase1_raw'>raw JSON</a></p>",
         "<div class='card'><h2>Phase 1 — Test A: ex-ante call scoring vs real KRX prices</h2>",
         f"<p><small>{_esc(r['generated'])} · data through {_esc(r['data_through'])} · "
         f"calls={r['n_calls']} (long={r['n_long']} short={r['n_short']} avoid={r['n_avoid']}) · "
         f"tickers={r['n_tickers']} · publish {_esc(r['publish_range'][0])}…{_esc(r['publish_range'][1])}<br>"
         f"index proxy: {_esc(r['index_proxy'])}<br>"
         f"cost {r['cost_roundtrip_pct']}% round-trip · β-lookback {r['beta_lookback_days']}d · "
         f"bar t≥{r['tstat_bar']} (green=passes, red=significantly negative)</small></p>",
         "<table><tr><th>Horizon</th><th class='num'>scorable</th><th class='num'>pending</th>"
         "<th class='num'>β-adj mean%</th><th class='num'>β-adj hit</th><th class='num'>β-adj t</th>"
         "<th class='num'>RAW mean%</th><th class='num'>RAW hit</th><th class='num'>RAW t</th></tr>"]
    for h in r["horizons_days"]:
        x = r["horizons"][str(h)]
        a, rw = x["abnormal"], x["raw"]
        B.append(f"<tr><td>{h}d</td><td class='num'>{x['n_scorable']}</td><td class='num'>{x['n_pending']}</td>"
                 f"<td class='num'>{_esc(a['mean_pct'])}</td><td class='num'>{_esc(a['hit_rate'])}</td>"
                 f"<td class='num'>{tcell(a['tstat'])}</td>"
                 f"<td class='num'>{_esc(rw['mean_pct'])}</td><td class='num'>{_esc(rw['hit_rate'])}</td>"
                 f"<td class='num'>{tcell(rw['tstat'])}</td></tr>")
    B.append("</table>")
    av = r["avoid_bucket"]
    B.append(f"<p><small><b>avoid bucket</b>: {av['n_calls']} calls — {_esc(av['note'])}: " +
             " · ".join(f"{h}d n={av['by_horizon'][str(h)]['n_scorable']} "
                        f"mean%={_esc(av['by_horizon'][str(h)]['abnormal_as_short']['mean_pct'])} "
                        f"t={_esc(av['by_horizon'][str(h)]['abnormal_as_short']['tstat'])}"
                        for h in r["horizons_days"]) + "</small></p>")
    passed = any((r["horizons"][str(h)]["abnormal"]["tstat"] or 0) >= r["tstat_bar"] for h in r["horizons_days"])
    B.append(f"<p><b>Verdict:</b> {'a horizon clears t≥3.5' if passed else 'no horizon clears t≥3.5 — no demonstrated positive edge on this sample'}</p>")
    B.append("</div>")
    return _page("".join(B), "Phase 1 — 머니업")


def _phase1b():
    p = config.DATA_DIR / "phase1b_callscore.json"
    if not p.exists():
        return _page("<div class='card'><h2>Phase 1B — faithful re-test</h2>"
                     "<p>No Phase 1B result yet — the watcher runs it ONCE after the degraded re-extraction "
                     "queue drains. <a href='/phase1'>← Test A</a> · <a href='/'>home</a></p>"
                     "<p><small>research / mock only — not a trading signal</small></p></div>", "Phase 1B — 머니업")
    r = json.loads(p.read_text(encoding="utf-8"))

    def t(x):
        return "—" if x is None else _esc(x)
    B = [f"<p><a href='/'>← fact sheets</a> · <a href='/phase1'>Test A</a> · <a href='/phase1b_raw'>raw JSON</a></p>",
         "<div class='card'><h2>Phase 1B — faithful, play-type-aware re-test</h2>",
         f"<div class='proof' style='background:#fff3cd;border:1px solid #f0c36d;color:#7a5d00;padding:8px 12px;"
         f"border-radius:6px'>⚠️ {t(r.get('banner'))}</div>",
         f"<p><small>{t(r.get('generated'))} · data through {t(r.get('data_through'))} · calls={t(r.get('n_calls'))} "
         f"entered={t(r.get('n_entered'))} non-triggered={t(r.get('non_triggered'))} ({t(r.get('non_triggered_rate'))}) "
         f"· unclassified={t(r.get('n_unclassified'))} · inferred-level rows={t(r.get('n_inferred_levels'))}<br>"
         f"K = segments×horizons = <b>{t(r.get('K_tests'))}</b> · base bar t≥{t(r.get('base_tstat_bar'))} · "
         f"<b>deflated bar t≥{t(r.get('deflated_tstat_bar'))}</b> (Bonferroni)</small></p>",
         f"<p><b>Decision:</b> {t(r.get('decision'))}</p></div>",
         "<div class='card'><h3>Segments (entry→exit β-adj net, his timeframe)</h3><table>"
         "<tr><th>segment</th><th>hypothesis</th><th class='num'>n</th><th class='num'>mean%</th>"
         "<th class='num'>hit</th><th class='num'>t</th><th class='num'>1d</th><th class='num'>5d</th>"
         "<th class='num'>20d</th><th>verdict</th></tr>"]
    for seg, s in (r.get("segments") or {}).items():
        pr, fx = s.get("primary", {}), s.get("fixed_horizons", {})
        B.append(f"<tr><td><b>{_esc(seg)}</b></td><td>{t(s.get('hypothesis'))}</td>"
                 f"<td class='num'>{t(s.get('n_entered'))}</td><td class='num'>{t(pr.get('mean_pct'))}</td>"
                 f"<td class='num'>{t(pr.get('hit_rate'))}</td><td class='num'>{t(pr.get('t'))}</td>"
                 f"<td class='num'>{t((fx.get('1') or {}).get('mean_pct'))}</td>"
                 f"<td class='num'>{t((fx.get('5') or {}).get('mean_pct'))}</td>"
                 f"<td class='num'>{t((fx.get('20') or {}).get('mean_pct'))}</td>"
                 f"<td>{t(s.get('verdict'))}</td></tr>")
    B.append("</table></div>")
    tb = r.get("test_b", {})
    B.append(f"<div class='card'><h3>Test B — event study (CAAR t0..t+5)</h3>"
             f"<p>events={t(tb.get('n_events'))} · clusters={t(tb.get('n_clusters'))} · "
             f"CAAR={t(tb.get('caar_pct'))}% · t={t(tb.get('t'))} · drift={t(tb.get('drift'))} · "
             f"pass={t(tb.get('pass'))} <small>(event-clustered SE · bar t≥3.5 · ≥30 events)</small></p></div>")
    return _page("".join(B), "Phase 1B — 머니업")


def _short_dt(x):
    """'2026-06-24T07:00:00Z' -> '06-24 07:00'."""
    if not x:
        return "—"
    x = str(x).replace("T", " ").replace("Z", "")
    return x[5:16]


def _index(sort: str = "processed"):
    sheets = _sheets(sort)
    n = len(sheets)
    toggle = ("published" if sort == "processed" else "processed")
    head = (f"<div class='card'><p><b><a href='/scoreboard'>🟢 Live scoreboard</a></b>"
            f" &nbsp;·&nbsp; <b><a href='/phase1'>📊 Phase 1 — call scoring (Test A)</a></b>"
            f" &nbsp;·&nbsp; <b><a href='/phase1b'>🧪 Phase 1B — faithful re-test</a></b>"
            f" &nbsp;·&nbsp; <b><a href='/playbook'>📖 Phase 2 — Playbook</a></b>"
            f" &nbsp;·&nbsp; <b>{n}</b> fact sheets &nbsp;·&nbsp; "
            f"sort: <b>{'recently processed' if sort=='processed' else 'published date'}</b> "
            f"(<a href='/?sort={toggle}'>switch to {toggle}</a>) &nbsp;·&nbsp; "
            f"<small id='ref'>auto-refresh 45s</small></p></div>")
    rows = [head, "<div class='card'><h2>Fact sheets</h2><table>",
            "<tr><th>processed</th><th>published</th><th>ticker</th><th>name</th><th>title</th>"
            "<th>ex-ante</th><th>ex-post</th><th>fusion</th><th></th></tr>"]
    for s in sheets:
        fs = s.get("fusion_summary", {})
        fus = " ".join(f"<span class='tag' style='background:{_TAG_COLOR.get(k,'#777')}'>{k[:1]}{v}</span>"
                       for k, v in fs.items())
        deg = "<span class='tag' style='background:#cf222e'>⚠ DEGRADED</span> " if s.get("degraded") else ""
        rows.append(
            f"<tr><td><small>{_esc(_short_dt(s.get('extracted_at')))}</small></td>"
            f"<td>{_esc(s.get('publish_date'))}</td><td>{_esc(s.get('primary_ticker'))}</td>"
            f"<td>{_esc(s.get('primary_name'))}</td>"
            f"<td>{deg}<a href='/video?id={_esc(s.get('video_id'))}'>{_esc((s.get('title') or '')[:50])}</a></td>"
            f"<td class='num'>{len(s.get('exante_calls', []))}</td>"
            f"<td class='num'>{len(s.get('expost_commentary', []))}</td>"
            f"<td>{fus}</td><td><a href='/video?id={_esc(s.get('video_id'))}'>open</a></td></tr>")
    if n == 0:
        rows.append("<tr><td colspan=9><i>No fact sheets yet.</i></td></tr>")
    rows.append("</table></div>")
    # JS poll: reload the index every 45s so newly-processed sheets appear without manual refresh
    rows.append("<script>setTimeout(function(){location.reload();}, 45000);</script>")
    return _page("".join(rows))


def _tag(t):
    return f"<span class='tag' style='background:{_TAG_COLOR.get(t,'#777')}'>{_esc(t)}</span>"


def _mmss_sec(mmss):
    import re as _re
    p = [int(x) for x in _re.findall(r"\d+", str(mmss or ""))]
    return (p[0]*3600+p[1]*60+p[2]) if len(p) == 3 else (p[0]*60+p[1]) if len(p) == 2 else (p[0] if p else 0)


def _ev_links(evs):
    out = []
    for e in evs or []:
        vid = e.get("video_id", "")
        sec = _mmss_sec(e.get("mmss"))
        url = f"https://www.youtube.com/watch?v={vid}&t={sec}s"
        out.append(f"<div><a href='{_esc(url)}'>{_esc(vid)}@{_esc(e.get('mmss'))}</a> "
                   f"<small>“{_esc((e.get('quote') or '')[:120])}”</small></div>")
    return "".join(out)


def _playbook():
    p = config.DATA_DIR / "playbook" / "playbook.json"
    if not p.exists():
        return _page("<div class='card'>No playbook yet. Run "
                     "<code>python -m moneyup_advisor.playbook</code>. <a href='/'>← back</a></div>")
    a = json.loads(p.read_text(encoding="utf-8"))
    m = a.get("meta", {})
    B = [f"<p><a href='/'>← fact sheets</a> · <a href='/playbook_raw'>raw JSON</a></p>",
         f"<div class='card'><h2>머니업 Playbook — his method (descriptive)</h2>"
         f"<p><small>{_esc(m.get('generated'))} · model {_esc(m.get('model'))} · "
         f"{_esc(m.get('n_videos'))} videos</small></p>"
         f"<div class='proof' style='background:#fff3cd;border:1px solid #f0c36d;color:#7a5d00;"
         f"padding:8px 12px;border-radius:6px'>⚠️ {_esc(m.get('discipline'))}</div></div>"]
    # rules
    B.append("<div class='card'><h3>Rules / setups (ranked by # videos)</h3>")
    for i, r in enumerate(a.get("rules", []), 1):
        B.append(f"<div style='margin:10px 0;border-left:3px solid #0969da;padding-left:10px'>"
                 f"<b>{i}. {_esc(r.get('canonical'))}</b> "
                 f"<span class='pill'>{_esc(r.get('kind'))}</span>"
                 f"<span class='pill'>{_esc(r.get('n_videos'))} videos</span>"
                 f"{('<br><small>reasoning: '+_esc(r.get('reasoning'))+'</small>') if r.get('reasoning') else ''}"
                 f"<div style='margin-top:4px'>{_ev_links(r.get('evidence', [])[:4])}</div></div>")
    B.append("</div>")
    # indicators
    B.append("<div class='card'><h3>Indicators he watches</h3><p>" +
             " ".join(f"<span class='pill'>{_esc(i.get('name'))} · {_esc(i.get('n_videos',0))}</span>"
                      for i in a.get("indicators_ranked", [])) + "</p></div>")
    # triggers
    B.append("<div class='card'><h3>Triggers</h3>")
    for side, lbl, col in (("buy", "BUY", "#1a7f37"), ("sell", "SELL", "#cf222e"), ("avoid", "AVOID", "#9a6700")):
        ts = a.get("triggers", {}).get(side, [])
        B.append(f"<p><b style='color:{col}'>{lbl}</b></p>")
        for t in ts:
            B.append(f"<div style='margin:4px 0'>{_esc(t.get('condition'))} "
                     f"<span class='pill'>{_esc(t.get('n_videos',0))}</span>"
                     f"<div>{_ev_links(t.get('evidence', [])[:2])}</div></div>")
    B.append("</div>")
    # vocabulary
    B.append("<div class='card'><h3>Vocabulary / catchphrases</h3>")
    for v in a.get("vocabulary", []):
        B.append(f"<div style='margin:4px 0'><b>{_esc(v.get('term'))}</b> — {_esc(v.get('meaning'))}"
                 f"<div>{_ev_links(v.get('examples', [])[:2])}</div></div>")
    B.append("</div>")
    return _page("".join(B), "머니업 Playbook")


def _video(vid):
    s = next((x for x in _sheets() if x.get("video_id") == vid), None)
    if not s:
        return _page("<div class='card'>Not found. <a href='/'>← back</a></div>")
    B = [f"<p><a href='/'>← all sheets</a> · <a href='/raw?id={_esc(vid)}'>raw JSON</a> · "
         f"<a href='/proof?id={_esc(vid)}'>🔍 full raw OCR (every token per frame)</a></p>"]
    B.append(f"<div class='card'><h2>{_esc(s.get('primary_name'))} "
             f"({_esc(s.get('primary_ticker'))})</h2>"
             f"<p><a href='{_esc(s.get('url'))}'>{_esc(s.get('title'))}</a><br>"
             f"<small>{_esc(s.get('channel'))} · published {_esc(s.get('publish_datetime') or s.get('publish_date'))} "
             f"· {_esc(s.get('duration_s'))}s · {_esc(s.get('n_frames_ocr'))} frames OCR'd · "
             f"extracted {_esc(s.get('extracted_at'))}</small></p>"
             + "".join(f"<div><small>· {_esc(r)}</small></div>" for r in s.get("rules", []))
             + "</div>")

    # primary numbers — OCR vs stated (audio), tagged
    pn = s.get("primary_numbers", [])
    if pn:
        t = ["<div class='card'><h3>Primary numbers <small>(OCR = source of truth · vs stated audio)</small></h3>"
             "<table><tr><th>field</th><th>한글</th><th class='num'>OCR</th>"
             "<th class='num'>stated (audio)</th><th>tag</th></tr>"]
        for d in pn:
            t.append(f"<tr><td>{_esc(d.get('field'))}</td><td>{_esc(d.get('label_ko'))}</td>"
                     f"<td class='num'>{_fmt(d.get('ocr_value'))}</td>"
                     f"<td class='num'>{_fmt(d.get('stated_value'))}</td>"
                     f"<td>{_tag(d.get('tag'))}</td></tr>")
        t.append("</table></div>")
        B.append("".join(t))

    # ex-ante calls
    ec = s.get("exante_calls", [])
    t = ["<div class='card'><h3>Ex-ante calls <small>(testable)</small></h3>"]
    if ec:
        t.append("<table><tr><th>ticker</th><th>name</th><th>direction</th><th class='num'>stated price</th>"
                 "<th>in-video</th><th>published</th><th></th></tr>")
        for c in ec:
            t.append(f"<tr><td>{_esc(c['ticker'])}</td><td>{_esc(c['name'])}</td>"
                     f"<td class='dir-{_esc(c['direction'])}'>{_esc(c['direction'])}</td>"
                     f"<td class='num'>{_fmt(c.get('stated_price'))}</td><td>{_esc(c['mmss'])}</td>"
                     f"<td>{_esc(c.get('publish_datetime') or c.get('publish_date'))}</td>"
                     f"<td><a href='{_esc(c['deeplink'])}'>jump</a></td></tr>")
        t.append("</table>")
        for c in ec:
            t.append(f"<p><small>{_esc(c['name'])} → <b>{_esc(c['direction'])}</b> @ {_esc(c['mmss'])}: "
                     f"“{_esc(c['quote'][:240])}”</small></p>")
    else:
        t.append("<p><i>none detected</i></p>")
    t.append("</div>")
    B.append("".join(t))

    # ex-post
    xp = s.get("expost_commentary", [])
    t = ["<div class='card'><h3>Ex-post commentary <small>(NOT a fresh call)</small></h3>"]
    t.append("".join(f"<p><small>{_esc(c['name'])} ({_esc(c['ticker'])}) @ {_esc(c['mmss'])}: "
                     f"“{_esc(c['quote'][:200])}”</small></p>" for c in xp) or "<p><i>none</i></p>")
    t.append("</div>")
    B.append("".join(t))

    # timeline
    fs = s.get("fusion_summary", {})
    t = ["<div class='card'><h3>Fused timeline</h3><p>"
         + " ".join(f"{_tag(k)} {v}" for k, v in fs.items()) + "</p><table>"
         "<tr><th>time</th><th>tag</th><th>ticker</th><th>kind</th><th>video</th><th>audio</th></tr>"]
    for e in s.get("timeline", []):
        t.append(f"<tr><td>{_esc(e.get('mmss'))}</td><td>{_tag(e.get('tag'))}</td>"
                 f"<td>{_esc(e.get('ticker'))}</td><td>{_esc(e.get('kind'))}</td>"
                 f"<td>{_esc(e.get('video'))}</td><td>{_esc(e.get('audio'))}</td></tr>")
    t.append("</table></div>")
    B.append("".join(t))

    # watchlist + VLM
    wl = s.get("watchlist_ocr", [])
    if wl:
        t = ["<div class='card'><h3>관심종목 watchlist <small>(OCR, keyed on code)</small></h3><table>"
             "<tr><th>ticker</th><th>name</th><th class='num'>OCR 현재가</th></tr>"]
        for r in wl[:20]:
            t.append(f"<tr><td>{_esc(r['ticker'])}</td><td>{_esc(r['name'])}</td>"
                     f"<td class='num'>{_fmt(r.get('ocr_price'))}</td></tr>")
        t.append("</table></div>")
        B.append("".join(t))
    vo = s.get("vlm_observations", [])
    if vo:
        t = ["<div class='card'><h3>Qwen3-VL <small>(VIDEO-ONLY, unverified)</small></h3>"]
        for o in vo:
            p = o.get("parsed", {})
            t.append(f"<p><small>@{_esc(o.get('t'))}s — "
                     f"pattern: {_esc(p.get('chart_pattern'))} · points_at: {_esc(p.get('points_at'))} "
                     f"· stance: {_esc(p.get('stance'))}</small></p>")
        t.append("</div>")
        B.append("".join(t))
    return _page("".join(B), f"{s.get('primary_name')} — 머니업")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype="text/html; charset=utf-8", code=200):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/":
            self._send(_index((q.get("sort") or ["processed"])[0]))
        elif u.path == "/scoreboard":
            from moneyup_advisor.dashboard import scoreboard
            self._send(_page(scoreboard.render_body(), "Live strategy scoreboard — 머니업"))
        elif u.path == "/phase1":
            self._send(_phase1())
        elif u.path == "/phase1_raw":
            p = config.DATA_DIR / "phase1_callscore.json"
            self._send(p.read_text(encoding="utf-8") if p.exists() else "{}",
                       "application/json; charset=utf-8")
        elif u.path == "/phase1b":
            self._send(_phase1b())
        elif u.path == "/phase1b_raw":
            p = config.DATA_DIR / "phase1b_callscore.json"
            self._send(p.read_text(encoding="utf-8") if p.exists() else "{}",
                       "application/json; charset=utf-8")
        elif u.path == "/playbook":
            self._send(_playbook())
        elif u.path == "/playbook_raw":
            p = config.DATA_DIR / "playbook" / "playbook.json"
            self._send(p.read_text(encoding="utf-8") if p.exists() else "{}",
                       "application/json; charset=utf-8")
        elif u.path == "/proof":
            from moneyup_advisor import proof
            vid = (q.get("id") or [""])[0]
            try:
                self._send(proof.render_html(vid, embed=False))
            except Exception as e:
                self._send(f"<div class='card'>No proof for {html.escape(vid)} ({e}). <a href='/'>back</a></div>", code=404)
        elif u.path == "/frame":
            vid = (q.get("id") or [""])[0]
            fname = (q.get("f") or [""])[0]
            if "/" in fname or "\\" in fname or ".." in fname or not fname.endswith(".png"):
                self._send("bad", code=400)
            else:
                fp = config.video_frame_dir(vid) / fname
                if fp.exists():
                    self._send(fp.read_bytes(), "image/png")
                else:
                    self._send("not found", code=404)
        elif u.path == "/video":
            self._send(_video((q.get("id") or [""])[0]))
        elif u.path == "/raw":
            s = next((x for x in _sheets() if x.get("video_id") == (q.get("id") or [""])[0]), {})
            self._send(json.dumps(s, ensure_ascii=False, indent=2),
                       "application/json; charset=utf-8")
        else:
            self._send("<div class='card'>404 <a href='/'>home</a></div>", code=404)


def serve(host=None, port=None):
    host = host or config.DASHBOARD_HOST
    port = port or config.DASHBOARD_PORT
    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"머니업 Advisor dashboard → http://{host}:{port}  (existing dashboard on :8000 untouched)")
    srv.serve_forever()


if __name__ == "__main__":
    serve()
