"""Kiwoom 공매도 / 수급(투자자) / 프로그램매매 REST fetchers.

TR codes (openapi.kiwoom.com → 국내주식):
  * ka10014  공매도추이              -> short-selling trend (qty / amount / ratio)
  * ka10059  종목별투자자기관별      -> per-day net buy by investor type (외국인/기관/개인)
  * ka10060  종목별투자자기관별차트  -> same, cumulative chart variant
  * ka10008  주식외국인종목별매매동향 -> per-day foreign net + holding qty / ratio
  * ka90005  프로그램매매추이 시간대별 -> intraday program-trading net
  * ka90013  프로그램매매추이 일자별   -> daily program-trading net

Same auth/feed pattern as ka10080 (:mod:`tagent.data.kiwoom_minute`): ``POST {base}{endpoint}``
with headers ``api-id`` + ``authorization: Bearer <token>`` and optional ``cont-yn`` / ``next-key``
continuation. Parsing is DEFENSIVE (documented field names + aliases + first-list fallback) and
net flows keep their SIGN (+순매수 / −순매도) — unlike OHLCV, the sign IS the information.

Mock-first: ``requests`` is injected in tests so nothing hits the network; the token lives only
in the Authorization header and is never logged.

HONEST: Kiwoom MOCK frequently returns these series EMPTY (they need a funded/live account). The
fetchers surface that as ``info["empty"] = True`` ("ready for a live account") rather than faking.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from tagent.config import KIWOOM_REST_URLS

# api-id -> REST endpoint + the response array key(s) that hold the per-row records.
TR_SPECS: Dict[str, dict] = {
    "ka10014": {"endpoint": "/api/dostk/shsa", "kind": "short",
                "array": ("shrts_trnsn", "shtcrt_trnsn", "short_trnsn")},
    "ka10059": {"endpoint": "/api/dostk/stkinfo", "kind": "investor",
                "array": ("stk_invsr_orgn", "invsr_orgn", "stk_invsr_orgn_chart")},
    "ka10060": {"endpoint": "/api/dostk/stkinfo", "kind": "investor",
                "array": ("stk_invsr_orgn_chart", "stk_invsr_orgn", "invsr_orgn")},
    "ka10008": {"endpoint": "/api/dostk/frgnistt", "kind": "foreign",
                "array": ("stk_frgnr", "frgnr_trnsn", "frgn")},
    "ka90005": {"endpoint": "/api/dostk/stkinfo", "kind": "program",
                "array": ("prm_trde_trnsn", "prm_trde_dt", "prm_trde")},
    "ka90013": {"endpoint": "/api/dostk/stkinfo", "kind": "program",
                "array": ("prm_trde_trnsn", "prm_trde_dt", "prm_trde")},
}

_DATE_KEYS = ("dt", "stck_bsop_date", "bsop_dt", "trde_dt", "cntr_dt", "date", "tm", "cntr_tm")


class KiwoomFlowsError(RuntimeError):
    """Raised when a flows TR request is rejected by the server."""


def _requests():
    try:
        import requests
    except ImportError as e:  # pragma: no cover
        raise KiwoomFlowsError("requests not installed. Run: pip install -r requirements.txt") from e
    return requests


def _to_signed(raw) -> float:
    """Parse a Kiwoom numeric string KEEPING its sign (+순매수 / −순매도). "" / None -> 0.0.

    Unlike :func:`tagent.feeds.kiwoom_feed._to_number` (which returns the magnitude for prices),
    flow values carry direction in the sign, so we must preserve it."""
    s = str(raw if raw is not None else "").strip().replace(",", "")
    if not s:
        return 0.0
    neg = s.startswith("-")
    s2 = s.lstrip("+-")
    if not s2:
        return 0.0
    try:
        v = float(s2)
    except ValueError:
        return 0.0
    return -v if neg else v


def _pick(row: dict, keys: Sequence[str]):
    for k in keys:
        v = row.get(k)
        if v not in (None, ""):
            return v
    return None


def _parse_label(raw):
    """A row's date/time label -> pd.Timestamp when parseable, else the cleaned string."""
    s = "".join(ch for ch in str(raw or "").strip() if ch.isdigit())
    for ln, fmt in ((14, "%Y%m%d%H%M%S"), (8, "%Y%m%d"), (6, "%Y%m"), (4, "%H%M")):
        if len(s) == ln or (ln == 8 and len(s) >= 8):
            try:
                return pd.Timestamp(datetime.strptime(s[:ln], fmt))
            except ValueError:
                continue
    return str(raw).strip() or None


def build_tr_request(api_id: str, base_url: str, token: str, body: dict,
                     cont_yn: str = "N", next_key: str = "") -> Tuple[str, dict, dict]:
    """(url, headers, body) for one TR page — same header shape as ka10080."""
    if api_id not in TR_SPECS:
        raise ValueError(f"unknown TR api-id {api_id}; known: {sorted(TR_SPECS)}")
    url = f"{base_url}{TR_SPECS[api_id]['endpoint']}"
    headers = {
        "Content-Type": "application/json;charset=UTF-8",
        "authorization": f"Bearer {token}",
        "api-id": api_id,
        "cont-yn": cont_yn,
        "next-key": next_key,
    }
    return url, headers, dict(body)


def _extract_list(data: dict, array_keys: Sequence[str]) -> list:
    if not isinstance(data, dict):
        return []
    for k in array_keys:
        v = data.get(k)
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict)]
    for v in data.values():                              # fallback: any list-of-dicts in the payload
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return v
    return []


def _cont_headers(resp) -> Tuple[str, str]:
    h = getattr(resp, "headers", {}) or {}
    cont = str(h.get("cont-yn") or h.get("cont_yn") or "N").strip().upper()
    nxt = (h.get("next-key") or h.get("next_key") or "").strip()
    return cont, nxt


def _fetch_tr_rows(api_id: str, body: dict, *, auth, session=None, env: str = "mock",
                   base_url: Optional[str] = None, max_pages: int = 5,
                   pause: float = 0.0) -> Tuple[list, dict]:
    """POST a TR, following cont-yn/next-key up to ``max_pages``; return (raw_rows, info).
    The token (from ``auth``) only ever rides in the Authorization header."""
    base = base_url or KIWOOM_REST_URLS.get(env.lower(), KIWOOM_REST_URLS["mock"])
    token = auth.get_token()
    sess = session or _requests()
    spec = TR_SPECS[api_id]
    rows: list = []
    cont_yn, next_key, pages = "N", "", 0
    while pages < max_pages:
        url, headers, b = build_tr_request(api_id, base, token, body, cont_yn, next_key)
        resp = sess.post(url, headers=headers, json=b, timeout=10)
        try:
            data = resp.json()
        except Exception as e:
            raise KiwoomFlowsError(f"{api_id} returned non-JSON: {e}")
        code = data.get("return_code")
        if str(code) not in ("0", "None"):
            raise KiwoomFlowsError(f"{api_id} rejected (return_code={code}): {data.get('return_msg')}")
        rows.extend(_extract_list(data, spec["array"]))
        pages += 1
        cont, nxt = _cont_headers(resp)
        if cont != "Y" or not nxt:
            break
        cont_yn, next_key = "Y", nxt
        if pause:
            time.sleep(pause)
    return rows, {"api_id": api_id, "endpoint": spec["endpoint"], "pages": pages,
                  "source": f"kiwoom {env}"}


def _to_df(records: List[dict], index: str = "date") -> pd.DataFrame:
    """Records -> DataFrame indexed by ``index`` (date/label), sorted, de-duplicated."""
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df = df[df[index].notna()].drop_duplicates(subset=[index], keep="last")
    df = df.set_index(index).sort_index()
    return df


# --------------------------------------------------------------------------- #
# parsers — one clean per-stock series each (signed net flows)
# --------------------------------------------------------------------------- #
def parse_short(rows: Sequence[dict]) -> pd.DataFrame:
    """ka10014 -> daily short-selling: short_qty, short_amt, short_ratio, close, avg_price."""
    recs = []
    for r in rows:
        d = _parse_label(_pick(r, _DATE_KEYS))
        if d is None:
            continue
        recs.append({
            "date": d,
            "short_qty": _to_signed(_pick(r, ("shrts_qty", "short_qty", "shtcrt_qty", "dpst_qty"))),
            "short_amt": _to_signed(_pick(r, ("shrts_amt", "short_amt", "shtcrt_amt"))),
            "short_ratio": _to_signed(_pick(r, ("shrts_wght", "shrts_rt", "short_ratio", "wght"))),
            "close": abs(_to_signed(_pick(r, ("cur_prc", "close_pric", "stck_prpr")))),  # price magnitude
            "avg_price": _to_signed(_pick(r, ("shrts_avg_pric", "avg_pric", "shtcrt_avg_pric"))),
        })
    return _to_df(recs)


def parse_investor_net(rows: Sequence[dict]) -> pd.DataFrame:
    """ka10059/ka10060 -> daily net buy by investor type (signed): foreign / institution /
    individual / pension."""
    recs = []
    for r in rows:
        d = _parse_label(_pick(r, _DATE_KEYS))
        if d is None:
            continue
        recs.append({
            "date": d,
            "foreign_net": _to_signed(_pick(r, ("frgnr_invsr", "frgn_invsr", "frgnr", "frgn"))),
            "inst_net": _to_signed(_pick(r, ("orgn", "orgn_invsr", "inst", "orgn_netprps"))),
            "individual_net": _to_signed(_pick(r, ("ind_invsr", "indi", "individual", "prsm"))),
            "pension_net": _to_signed(_pick(r, ("penf", "pnsn", "yg_invsr", "pension"))),
        })
    return _to_df(recs)


def parse_foreign(rows: Sequence[dict]) -> pd.DataFrame:
    """ka10008 -> daily foreign net + holding qty / ratio + close."""
    recs = []
    for r in rows:
        d = _parse_label(_pick(r, _DATE_KEYS))
        if d is None:
            continue
        recs.append({
            "date": d,
            "close": abs(_to_signed(_pick(r, ("cur_prc", "close_pric", "stck_prpr")))),  # price magnitude
            "foreign_net": _to_signed(_pick(r, ("frgnr_ntby_qty", "ntby_qty", "for_netprps_qty",
                                                "frgnr_netprps", "net_buy"))),
            "holding_qty": _to_signed(_pick(r, ("poss_stkcnt", "frgnr_hold_qty", "hold_qty",
                                                "frgnr_poss_stkcnt"))),
            "holding_ratio": _to_signed(_pick(r, ("frgnr_wght", "wght", "frgnr_hold_rt",
                                                  "hold_rt", "limit_exh_rt"))),
        })
    return _to_df(recs)


def parse_program(rows: Sequence[dict]) -> pd.DataFrame:
    """ka90005/ka90013 -> program-trading net (signed) by date (or intraday time label)."""
    recs = []
    for r in rows:
        d = _parse_label(_pick(r, _DATE_KEYS))
        if d is None:
            continue
        recs.append({
            "date": d,
            "program_net": _to_signed(_pick(r, ("prm_netprps_amt", "netprps_amt", "prm_ntby_qty",
                                                "ntby_qty", "prm_net", "prm_netprps"))),
            "program_buy": _to_signed(_pick(r, ("prm_buy_amt", "buy_amt", "prm_buy_qty"))),
            "program_sell": _to_signed(_pick(r, ("prm_sell_amt", "sel_amt", "prm_sel_qty"))),
        })
    return _to_df(recs)


# --------------------------------------------------------------------------- #
# fetchers — (df, info) where info["empty"] honestly flags a mock-empty TR
# --------------------------------------------------------------------------- #
def _finish(df: pd.DataFrame, info: dict) -> Tuple[pd.DataFrame, dict]:
    info = {**info, "rows": int(len(df)), "empty": bool(df.empty)}
    if df.empty:
        info["note"] = "empty on this endpoint — likely needs a funded/live account (mock often blank)"
    return df, info


def fetch_short_selling(stk_cd: str, *, auth, session=None, env: str = "mock",
                        base_url: Optional[str] = None, start_date: str = "", end_date: str = "",
                        max_pages: int = 5) -> Tuple[pd.DataFrame, dict]:
    body = {"stk_cd": str(stk_cd), "tm_tp": "1"}
    if start_date:
        body["strt_dt"] = str(start_date)
    if end_date:
        body["end_dt"] = str(end_date)
    rows, info = _fetch_tr_rows("ka10014", body, auth=auth, session=session, env=env,
                                base_url=base_url, max_pages=max_pages)
    return _finish(parse_short(rows), info)


def fetch_investor_net(stk_cd: str, *, auth, session=None, env: str = "mock",
                       base_url: Optional[str] = None, api_id: str = "ka10059",
                       base_date: str = "", amt_qty_tp: str = "2",
                       max_pages: int = 5) -> Tuple[pd.DataFrame, dict]:
    """ka10059 (default) or ka10060 — per-day net buy by investor type. ``amt_qty_tp`` "2"=수량,
    "1"=금액."""
    if api_id not in ("ka10059", "ka10060"):
        raise ValueError("investor-net TR must be ka10059 or ka10060")
    body = {"stk_cd": str(stk_cd), "amt_qty_tp": str(amt_qty_tp), "trde_tp": "0", "unit_tp": "1000"}
    if base_date:
        body["dt"] = str(base_date)
    rows, info = _fetch_tr_rows(api_id, body, auth=auth, session=session, env=env,
                                base_url=base_url, max_pages=max_pages)
    return _finish(parse_investor_net(rows), info)


def fetch_foreign_trend(stk_cd: str, *, auth, session=None, env: str = "mock",
                        base_url: Optional[str] = None,
                        max_pages: int = 5) -> Tuple[pd.DataFrame, dict]:
    body = {"stk_cd": str(stk_cd)}
    rows, info = _fetch_tr_rows("ka10008", body, auth=auth, session=session, env=env,
                                base_url=base_url, max_pages=max_pages)
    return _finish(parse_foreign(rows), info)


def fetch_program_trading(stk_cd: str, *, auth, session=None, env: str = "mock",
                          base_url: Optional[str] = None, api_id: str = "ka90013",
                          base_date: str = "", mrkt_tp: str = "P00101",
                          max_pages: int = 5) -> Tuple[pd.DataFrame, dict]:
    """ka90013 (일자별, default) or ka90005 (시간대별) — program-trading net. ``mrkt_tp`` selects
    the market (KOSPI/KOSDAQ); ``stk_cd`` narrows to one name where the endpoint supports it."""
    if api_id not in ("ka90005", "ka90013"):
        raise ValueError("program TR must be ka90005 or ka90013")
    body = {"stk_cd": str(stk_cd), "amt_qty_tp": "1", "mrkt_tp": str(mrkt_tp), "stex_tp": "1"}
    if base_date:
        body["date"] = str(base_date)
    if api_id == "ka90005":
        body["min_tic_tp"] = "1"
    rows, info = _fetch_tr_rows(api_id, body, auth=auth, session=session, env=env,
                                base_url=base_url, max_pages=max_pages)
    return _finish(parse_program(rows), info)
