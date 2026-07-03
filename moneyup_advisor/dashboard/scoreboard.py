# -*- coding: utf-8 -*-
"""Task 7 — LIVE STRATEGY SCOREBOARD section for the 8077 dashboard. ISOLATED to the dashboard.

Two panels:
  • LIVE STOCKS    — price + computed indicators for the watchlist, from kiwoom_feed.get_quotes().
                     Labelled "MOCK DATA" when KIWOOM_ENV=mock; flips to real automatically with prod keys.
  • STRATEGY SCOREBOARD — one row per strategy, driven by REGISTRY (new strategies/versions auto-appear).
                     Numbers are pulled by computed path from the qa_audit JSONs (NOT retyped):
                     phase1b_callscore / phase1b_holdout / phase1b_capslice / phase1b_offense / phase1b_h1_holdout.

Honest design: every % is net-of-market + 0.50% round-trip cost (β/size-adjusted, from phase1b); BACKTEST
(in-sample, OOS) and LIVE-FORWARD are distinct columns; only MECHANIZED strategies get a number (vague
discretionary ones = "not scorable"); "pending" where not yet measured; a paper/not-tradeable result is
amber and never green; nothing is fabricated; mostly-red is expected.

Isolation: reads qa_audit JSONs + kiwoom_feed only. Imports nothing from the collector/pipeline/report/
playbook/extraction. The dashboard server adds ONE /scoreboard route; no other file changes.
"""
from __future__ import annotations
import datetime as dt
import html
import json
import time

from moneyup_advisor import config
from moneyup_advisor.live import kiwoom_feed

QA = config.DATA_DIR / "qa_audit"
FORWARD_START = "2026-06-30"          # live-forward tracking starts now (Task 7 ships)

GREEN, RED, AMBER, GREY = "#1a7f37", "#cf222e", "#9a6700", "#57606a"


def _esc(x):
    return html.escape(str(x)) if x is not None else "—"


def _load(fn):
    p = QA / fn
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except Exception:
        return {}


def _dig(d, path):
    for k in path:
        d = d.get(k) if isinstance(d, dict) else None
        if d is None:
            return None
    return d


def _stat(src):
    """src = (filename, [json path]) -> normalized {n, mean, t} pulled straight from the qa_audit JSON."""
    if not src:
        return None
    fn, path = src
    d = _dig(_load(fn), path)
    if not isinstance(d, dict):
        return None
    n = d.get("n", d.get("n_entered"))
    mean = d.get("mean_pct", d.get("mean_net_pct"))
    if n is None and mean is None:
        return None
    return {"n": n, "mean": mean, "t": d.get("t")}


# ------------------------------------------------------------------ strategy registry
# Each strategy lists version(s); each version points at where its numbers LIVE in the qa_audit JSONs.
# Add a strategy (new dict) or a version (append to "versions") → the board grows a row / a "vs prev" delta.
REGISTRY = [
    {"key": "mu_long", "name": "머니업 buy-calls (long, pooled)", "mechanized": True, "tradeable": True,
     "status": "LOSES — confirmed OOS",
     "versions": [{"v": "v1",
                   "insample": ("phase1b_holdout.json", ["exploratory_in_sample", "long_pooled"]),
                   "oos": ("phase1b_holdout.json", ["holdout_OOS", "long_pooled"])}]},
    {"key": "dipbuy", "name": "DIP-BUY setup", "mechanized": True, "tradeable": True,
     "status": "significantly negative",
     "versions": [{"v": "v1", "insample": ("phase1b_offense.json", ["track_A", "play_buckets", "DIP-BUY"]),
                   "oos": None}]},
    {"key": "subneed", "name": "수급 (기관/외인) setup", "mechanized": True, "tradeable": True,
     "status": "significantly negative",
     "versions": [{"v": "v1", "insample": ("phase1b_offense.json", ["track_A", "setup_buckets", "수급_기관외인"]),
                   "oos": None}]},
    {"key": "shortaccum", "name": "공매도 매집선 setup", "mechanized": True, "tradeable": True,
     "status": "significantly negative",
     "versions": [{"v": "v1", "insample": ("phase1b_offense.json", ["track_A", "setup_buckets", "공매도_매집선"]),
                   "oos": None}]},
    {"key": "pullback", "name": "눌림목 / 지지 setup", "mechanized": True, "tradeable": True,
     "status": "significantly negative",
     "versions": [{"v": "v1", "insample": ("phase1b_offense.json", ["track_A", "setup_buckets", "눌림목_지지"]),
                   "oos": None}]},
    {"key": "megacap", "name": "mega-cap calls (005930 / 035420 / 000660)", "mechanized": True, "tradeable": True,
     "status": "negative OOS",
     "versions": [{"v": "v1", "insample": ("phase1b_offense.json", ["mega_cap_slice"]),
                   "oos": ("phase1b_h1_holdout.json", [])}]},
    {"key": "contrarian", "name": "contrarian — fade his calls (short)", "mechanized": True, "tradeable": False,
     "status": "PAPER ONLY · shortability unresolved · NOT tradeable",
     "versions": [{"v": "v1", "insample": ("phase1b_holdout.json", ["exploratory_in_sample", "contrarian_mirror"]),
                   "oos": ("phase1b_holdout.json", ["holdout_OOS", "contrarian_mirror"])}]},
    {"key": "discretionary", "name": "머니업 재량 단기매매 (감각 · 미기계화)", "mechanized": False, "tradeable": None,
     "status": "not scorable (discretionary)",
     "versions": [{"v": "v1", "insample": None, "oos": None}]},
]


def _cell(stat, tradeable=True, mechanized=True):
    """(html, mean) — colour: green only for a passing TRADEABLE edge; red = sig. negative; amber = paper/
    inconclusive; grey = pending / not-scorable. Never green for a non-tradeable (paper) signal."""
    if not mechanized:
        return f"<span style='color:{GREY}'>not scorable</span>", None
    if not stat or stat.get("mean") is None:
        return f"<span style='color:{GREY}'>pending</span>", None
    mean, t, n = stat["mean"], stat.get("t"), stat.get("n")
    if tradeable is False:
        col = AMBER
    elif mean > 0 and (t or 0) >= 3.92:
        col = GREEN
    elif mean < 0 and (t or 0) <= -3.5:
        col = RED
    else:
        col = AMBER
    return f"<b style='color:{col}'>{mean:+.2f}%</b> <small>n={_esc(n)}, t={_esc(t)}</small>", mean


def _delta(strat):
    vs = strat["versions"]
    if len(vs) < 2:
        return "—"
    cur, prev = _stat(vs[-1].get("insample")), _stat(vs[-2].get("insample"))
    if not cur or not prev or cur["mean"] is None or prev["mean"] is None:
        return "—"
    d = cur["mean"] - prev["mean"]
    col = GREEN if d > 0 else RED if d < 0 else GREY
    return f"<b style='color:{col}'>{d:+.2f}pp</b> <small>vs {_esc(vs[-2]['v'])}</small>"


def _scoreboard_panel():
    B = ["<div class='card'><h2>📋 Strategy scoreboard</h2>",
         f"<p><small>net-vs-market %, <b>after 0.50% round-trip cost</b> (β/size-adjusted, from phase1b) · "
         f"BACKTEST and LIVE-FORWARD are separate · only mechanized strategies are scored · "
         f"<span style='color:{GREEN}'>green</span>=passing tradeable edge · "
         f"<span style='color:{RED}'>red</span>=significantly negative · "
         f"<span style='color:{AMBER}'>amber</span>=paper/inconclusive (never green) · "
         f"<span style='color:{GREY}'>grey</span>=pending/not-scorable · numbers pulled live from the "
         f"qa_audit JSONs (not retyped) · mostly-red is expected</small></p>",
         "<table><tr><th>strategy</th><th>ver</th>"
         "<th class='num'>Backtest — in-sample</th><th class='num'>Backtest — OOS</th>"
         "<th class='num'>Live-forward</th><th>status</th><th class='num'>vs prev (Δ)</th></tr>"]
    for s in REGISTRY:
        v = s["versions"][-1]
        ins, _ = _cell(_stat(v.get("insample")), s["tradeable"], s["mechanized"])
        oos, _ = _cell(_stat(v.get("oos")), s["tradeable"], s["mechanized"])
        live = (f"<span style='color:{GREY}'>pending (from {FORWARD_START})</span>" if s["mechanized"]
                else f"<span style='color:{GREY}'>not scorable</span>")
        st_col = AMBER if s["tradeable"] is False else GREY
        B.append(f"<tr><td><b>{_esc(s['name'])}</b></td><td>{_esc(v['v'])}</td>"
                 f"<td class='num'>{ins}</td><td class='num'>{oos}</td><td class='num'>{live}</td>"
                 f"<td style='color:{st_col}'>{_esc(s['status'])}</td><td class='num'>{_delta(s)}</td></tr>")
    B.append("</table><p><small>Registry-driven: add a strategy dict (new row) or append a version "
             "(populates the “vs prev” delta) and the board updates on next load.</small></p></div>")
    return "".join(B)


# ------------------------------------------------------------------ live stocks panel
_DAILY = {}   # ticker -> (ts, [closes])


def _daily_closes(ticker):
    now = time.time()
    if ticker in _DAILY and now - _DAILY[ticker][0] < 600:
        return _DAILY[ticker][1]
    closes = []
    try:                                            # best-effort daily history for TA indicators
        import FinanceDataReader as fdr
        end = dt.date.today()
        df = fdr.DataReader(ticker, (end - dt.timedelta(days=120)).isoformat(), end.isoformat())
        closes = [float(x) for x in df["Close"].tolist() if x == x][-60:]
    except Exception:
        closes = []
    _DAILY[ticker] = (now, closes)
    return closes


def _rsi(closes, n=14):
    if len(closes) < n + 1:
        return None
    import numpy as np
    d = np.diff(closes[-(n + 1):])
    up, dn = d[d > 0].sum() / n, -d[d < 0].sum() / n
    return round(100.0 if dn == 0 else 100 - 100 / (1 + up / dn), 1)


def _pct(a, b):
    return round((a / b - 1) * 100, 2) if (a and b) else None


def _live_panel():
    try:
        f = kiwoom_feed.feed()
        quotes = f.get_quotes(kiwoom_feed.WATCHLIST)
        is_mock = f._src().endswith("mock")
        env = f.env or "?"
    except Exception as e:
        return (f"<div class='card'><h2>📈 Live stocks</h2>"
                f"<p style='color:{RED}'>Kiwoom feed unavailable: {_esc(e)}</p></div>")
    badge = (f"<span class='tag' style='background:{RED}'>⚠ MOCK DATA (KIWOOM_ENV={_esc(env)})</span>"
             if is_mock else f"<span class='tag' style='background:{GREEN}'>LIVE</span>")
    B = [f"<div class='card'><h2>📈 Live stocks &nbsp;{badge}</h2>",
         f"<p><small>source: {_esc(f._src())} · majors + theme names · computed indicators "
         f"(intraday from quote · RSI14/SMA20 from daily, best-effort) · "
         f"{'<b style=color:%s>test quotes — NOT real prices</b>' % RED if is_mock else 'real-time'}</small></p>",
         "<table><tr><th>ticker</th><th>name</th><th class='num'>price</th><th class='num'>chg%</th>"
         "<th class='num'>vs open</th><th class='num'>day range</th><th class='num'>volume</th>"
         "<th class='num'>RSI14</th><th class='num'>vs SMA20</th></tr>"]
    for tk, q in quotes.items():
        if not q.get("ok"):
            B.append(f"<tr><td>{_esc(tk)}</td><td colspan=8 style='color:{GREY}'>"
                     f"unavailable — {_esc(q.get('error'))}</td></tr>")
            continue
        px, op, hi, lo = q.get("price"), q.get("open"), q.get("high"), q.get("low")
        vs_open = _pct(px, op)
        rng = round((px - lo) / (hi - lo) * 100, 0) if (px and hi and lo and hi != lo) else None
        closes = _daily_closes(tk)
        rsi = _rsi(closes)
        sma20 = _pct(px, sum(closes[-20:]) / 20) if len(closes) >= 20 else None
        cp = q.get("change_pct")
        ccol = GREEN if (cp or 0) > 0 else RED if (cp or 0) < 0 else GREY
        B.append(
            f"<tr><td>{_esc(tk)}</td><td>{_esc(q.get('name'))}</td>"
            f"<td class='num'>{q['price']:,}</td>"
            f"<td class='num' style='color:{ccol}'>{cp:+}%</td>"
            f"<td class='num'>{('%+.2f%%' % vs_open) if vs_open is not None else '—'}</td>"
            f"<td class='num'>{('%d%%' % rng) if rng is not None else '—'}</td>"
            f"<td class='num'>{q.get('volume'):,}</td>"
            f"<td class='num'>{_esc(rsi)}</td>"
            f"<td class='num'>{('%+.2f%%' % sma20) if sma20 is not None else '—'}</td></tr>")
    B.append(f"</table><p><small>updated {_esc(next(iter(quotes.values()), {}).get('ts'))} KST · "
             f"auto-refresh 30s</small></p></div>")
    return "".join(B)


def render_body():
    return (_live_panel() + _scoreboard_panel()
            + "<script>setTimeout(function(){location.reload();}, 30000);</script>")
