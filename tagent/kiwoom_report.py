"""Kiwoom 수급 report — per watchlist stock: 공매도 trend, 외국인·기관 net (수급), 프로그램 net,
each distilled to ONE honest line (e.g. "외국인 3일 연속 순매도 → 매도 압력"). Generic filler is
dropped: a line is only surfaced when there's a concrete streak / direction.

INFORMATIONAL only — not a validated edge and not a trading signal. Pure interpretation funcs
(operate on the parsed DataFrames from :mod:`tagent.data.kiwoom_flows`) so they unit-test with
synthetic data, no network. If a TR is empty on mock, the report FLAGS it ("ready for a live
account") rather than inventing numbers.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import List, Optional, Sequence

import pandas as pd

from tagent.data.kiwoom_flows import (
    KiwoomFlowsError, fetch_foreign_trend, fetch_investor_net, fetch_program_trading,
    fetch_short_selling,
)

REPORT_LABEL = "Kiwoom 수급·공매도·프로그램 — 정보용 (검증된 엣지 아님, 매매 신호 아님)"
_NO_DATA = "데이터 없음 (mock 비어있음 — 실계좌에서 채워짐)"


def tail_streak(values: Sequence[float]) -> tuple:
    """(sign, length) of the current consecutive same-sign run at the TAIL (most recent).
    sign ∈ {-1, 0, 1}; length 0 when the latest value is zero/NaN."""
    vals = [float(v) for v in values if v == v]          # drop NaN
    if not vals:
        return (0, 0)
    last = vals[-1]
    sign = 1 if last > 0 else (-1 if last < 0 else 0)
    if sign == 0:
        return (0, 0)
    n = 0
    for v in reversed(vals):
        s = 1 if v > 0 else (-1 if v < 0 else 0)
        if s != sign:
            break
        n += 1
    return (sign, n)


def _empty(kind: str, info: Optional[dict]) -> dict:
    return {"kind": kind, "text": None, "status": "no-data", "informative": False,
            "note": _NO_DATA, "empty": True, "info": info or {}}


# --------------------------------------------------------------------------- #
# one-line interpretations (drop filler: text is None unless something concrete)
# --------------------------------------------------------------------------- #
def interpret_short(df: pd.DataFrame, info: Optional[dict] = None) -> dict:
    """공매도 추세 한 줄. Rising short ratio/qty -> 매도 압력; falling -> 완화; else neutral (no line)."""
    if df is None or df.empty:
        return _empty("short", info)
    metric = "short_ratio" if df.get("short_ratio") is not None and df["short_ratio"].abs().sum() > 0 \
        else "short_qty"
    s = df[metric].dropna()
    latest = float(s.iloc[-1]) if len(s) else 0.0
    recent = s.tail(3)
    text, status, informative = None, "neutral", False
    if len(recent) >= 2:
        up = recent.is_monotonic_increasing and recent.iloc[-1] > recent.iloc[0]
        down = recent.is_monotonic_decreasing and recent.iloc[-1] < recent.iloc[0]
        unit = "비중" if metric == "short_ratio" else "잔량"
        if up:
            text = f"공매도 {unit} {len(recent)}일 연속 증가 → 매도 압력 강화"
            status, informative = "bearish", True
        elif down:
            text = f"공매도 {unit} 감소 추세 → 압력 완화"
            status, informative = "bullish", True
    return {"kind": "short", "text": text, "status": status, "informative": informative,
            "metric": metric, "latest": latest, "empty": False, "info": info or {}}


def interpret_supply(df: pd.DataFrame, info: Optional[dict] = None) -> dict:
    """외국인·기관 수급 한 줄. Streaks (≥3일) or aligned direction headline; else low-priority."""
    if df is None or df.empty:
        return _empty("supply", info)
    f_sign, f_len = tail_streak(df.get("foreign_net", pd.Series(dtype=float)))
    i_sign, i_len = tail_streak(df.get("inst_net", pd.Series(dtype=float)))

    def _phrase(name, sign, ln):
        if sign == 0:
            return None
        side = "순매수" if sign > 0 else "순매도"
        return f"{name} {ln}일 연속 {side}" if ln >= 2 else f"{name} {side}"

    parts = [p for p in (_phrase("외국인", f_sign, f_len), _phrase("기관", i_sign, i_len)) if p]
    if not parts:
        return {"kind": "supply", "text": None, "status": "neutral", "informative": False,
                "foreign_streak": [f_sign, f_len], "inst_streak": [i_sign, i_len],
                "empty": False, "info": info or {}}
    if f_sign and f_sign == i_sign:                      # both same direction -> clear pressure
        press = "매수세 유입" if f_sign > 0 else "매도 압력"
    elif f_sign and i_sign and f_sign != i_sign:
        press = "수급 엇갈림"
    else:
        press = "매수세 유입" if (f_sign or i_sign) > 0 else "매도 압력"
    informative = bool(max(f_len, i_len) >= 3 or (f_sign and f_sign == i_sign))
    return {"kind": "supply", "text": " · ".join(parts) + f" → {press}", "status":
            ("bullish" if "매수" in press else ("bearish" if "매도" in press else "mixed")),
            "informative": informative, "foreign_streak": [f_sign, f_len],
            "inst_streak": [i_sign, i_len], "empty": False, "info": info or {}}


def interpret_program(df: pd.DataFrame, info: Optional[dict] = None) -> dict:
    """프로그램매매 net 한 줄. Net buy/sell (with a streak note); zero -> no line."""
    if df is None or df.empty:
        return _empty("program", info)
    net = df.get("program_net", pd.Series(dtype=float)).dropna()
    if not len(net):
        return _empty("program", info)
    sign, length = tail_streak(net)
    latest = float(net.iloc[-1])
    if sign == 0:
        return {"kind": "program", "text": None, "status": "neutral", "informative": False,
                "latest": latest, "empty": False, "info": info or {}}
    side = "순매수 유입" if sign > 0 else "순매도"
    streak = f" ({length}일 연속)" if length >= 2 else ""
    return {"kind": "program", "text": f"프로그램 {side}{streak}",
            "status": "bullish" if sign > 0 else "bearish", "informative": True,
            "latest": latest, "streak": [sign, length], "empty": False, "info": info or {}}


# --------------------------------------------------------------------------- #
# assemble the per-stock report
# --------------------------------------------------------------------------- #
def _safe_fetch(fn, api_id: str):
    """Run a fetch, turning a TR rejection / network failure into an HONEST flagged-empty info
    (never a silent neutral) so the report says "needs live account" instead of faking."""
    try:
        return fn()
    except KiwoomFlowsError as e:
        return None, {"api_id": api_id, "empty": True, "error": str(e)[:140],
                      "note": "TR 거부 — 정확한 파라미터/엔드포인트 + 실계좌 필요"}
    except Exception as e:
        return None, {"api_id": api_id, "empty": True,
                      "error": f"{type(e).__name__}: {str(e)[:100]}",
                      "note": "요청 실패 — 실계좌/네트워크 필요"}


def stock_supply_card(symbol: str, *, auth, session=None, env: str = "mock", base_url=None,
                      start_date: str = "", end_date: str = "", use_foreign_trend: bool = False) -> dict:
    """One stock: pull the three TRs and reduce each to a one-line read. The 공매도/수급/프로그램
    TRs require a date window (strt_dt / dt) — default a recent window so a LIVE call is
    well-formed; a rejection is FLAGGED honestly, never shown as neutral."""
    end_date = end_date or date.today().strftime("%Y%m%d")
    start_date = start_date or (date.today() - timedelta(days=40)).strftime("%Y%m%d")
    short_df, si = _safe_fetch(lambda: fetch_short_selling(
        symbol, auth=auth, session=session, env=env, base_url=base_url,
        start_date=start_date, end_date=end_date), "ka10014")
    if use_foreign_trend:                                # ka10008 (foreign only) as the 수급 source
        sup_df, ui = _safe_fetch(lambda: fetch_foreign_trend(
            symbol, auth=auth, session=session, env=env, base_url=base_url), "ka10008")
    else:
        sup_df, ui = _safe_fetch(lambda: fetch_investor_net(
            symbol, auth=auth, session=session, env=env, base_url=base_url, base_date=end_date), "ka10059")
    prog_df, pi = _safe_fetch(lambda: fetch_program_trading(
        symbol, auth=auth, session=session, env=env, base_url=base_url, base_date=end_date), "ka90013")

    short, supply, program = (interpret_short(short_df, si), interpret_supply(sup_df, ui),
                              interpret_program(prog_df, pi))
    for blk, info in ((short, si), (supply, ui), (program, pi)):
        if info.get("error"):                            # surface the real rejection, not silence
            blk.update({"empty": True, "status": "no-data", "informative": False,
                        "note": f"{info.get('note', 'TR 거부')} [{info['error']}]"})
    return {"symbol": symbol, "short": short, "supply": supply, "program": program}


def build_kiwoom_report(symbols: Sequence[str], *, auth, session=None, env: str = "mock",
                        base_url=None, start_date: str = "", end_date: str = "",
                        generated: Optional[str] = None) -> dict:
    """Per-watchlist 수급/공매도/프로그램 report. Honest: empty TRs are flagged, not faked."""
    cards: List[dict] = []
    empty_tr = set()
    for sym in symbols:
        try:
            card = stock_supply_card(sym, auth=auth, session=session, env=env, base_url=base_url,
                                     start_date=start_date, end_date=end_date)
        except Exception as e:                           # one stock's failure is not fatal
            card = {"symbol": sym, "error": str(e)[:120],
                    "short": {}, "supply": {}, "program": {}}
        for blk in ("short", "supply", "program"):
            b = card.get(blk) or {}
            if b.get("empty"):
                empty_tr.add((b.get("info") or {}).get("api_id", blk))
        cards.append(card)
    return {
        "enabled": True, "env": env, "label": REPORT_LABEL, "generated": generated,
        "stocks": cards, "empty_endpoints": sorted(x for x in empty_tr if x),
        "note": ("빈 응답 TR: " + ", ".join(sorted(x for x in empty_tr if x)) +
                 " — mock에서 비어있음, 실계좌에서 채워짐" if empty_tr else ""),
    }


def render_report_text(payload: dict) -> str:
    """Plain-text 수급 briefing — one stock per block, only informative lines (filler dropped)."""
    if not payload or not payload.get("enabled"):
        return "Kiwoom 수급 리포트 비활성 (KIWOOM 키 필요)."
    out = [f"=== {payload.get('label', 'Kiwoom 수급')} · env {payload.get('env', '')} ==="]
    for c in payload.get("stocks", []):
        out.append(f"\n[{c.get('symbol')}]")
        if c.get("error"):
            out.append(f"  · error: {c['error']}")
            continue
        any_line = False
        for blk, tag in (("short", "공매도"), ("supply", "수급"), ("program", "프로그램")):
            b = c.get(blk) or {}
            if b.get("text"):
                out.append(f"  · {tag}: {b['text']}")
                any_line = True
            elif b.get("empty"):
                out.append(f"  · {tag}: {b.get('note', '데이터 없음')}")
        if not any_line:
            out.append("  · 특이 수급 신호 없음")
    if payload.get("note"):
        out.append(f"\n{payload['note']}")
    return "\n".join(out)
