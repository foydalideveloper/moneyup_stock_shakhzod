"""Small KRX derivatives-investor source — KOSPI200 FUTURES investor net flow.

pykrx exposes futures OHLCV but NOT derivatives investor trading, so this hits the KRX MDC
JSON API (data.krx.co.kr getJsonData) REUSING pykrx's authenticated session (the same
KRX_ID/KRX_PW login that unlocks 공매도/수급). Endpoint MDCSTAT13101 returns, per query day,
the net buy (순매수 = NETBID) by investor type for a derivatives product; we extract 외국인
(foreign) and 기관합계 (institutional) and stitch a daily series. Cached to
data/kospi200_fut_investor.csv (columns: foreign_net, inst_net = net-buy VALUE in KRW).

NOTE (this environment): the derivatives/ETF investor aggregates return all-zero here
(only per-issue stock flows are populated), so the live series is empty/degenerate — see
scripts/run_futures_flow.py, which reports the data wall honestly. The fetch is correct and
ready for a populated KRX.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from tagent.config import DATA_DIR

MDC_URL = "https://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
INVESTOR_BLD = "dbms/MDC/STAT/standard/MDCSTAT13101"      # 파생상품 투자자별 거래실적 (net by type)
KOSPI200_FUT_PROD = "KRDRVFUK2I"                          # KOSPI200 futures product
CSV = "kospi200_fut_investor.csv"
_HEADERS = {"User-Agent": "Mozilla/5.0", "X-Requested-With": "XMLHttpRequest",
            "Referer": "https://data.krx.co.kr/contents/MDC/MDI/outerLoader/index.cmd"}


def _authed_session():
    """pykrx's KRX-authenticated requests session (warms the login, then returns it)."""
    from tagent.data.krx_source import ensure_krx_login
    ensure_krx_login()
    from pykrx import stock
    try:                                                  # a gated call establishes the authed session
        stock.get_market_trading_value_by_investor("20240102", "20240105", "069500")
    except Exception:
        pass
    from pykrx.website.comm import webio
    import requests
    s = webio.get_session()
    if s is None:
        s = requests.Session()
    return s


def _net_for_day(session, day: str, prod: str) -> Optional[dict]:
    """{foreign_net, inst_net} (net-buy VALUE, KRW) for one trading day, or None."""
    data = {"bld": INVESTOR_BLD, "locale": "ko_KR", "csvxls_isNo": "false",
            "prodId": prod, "strtDd": day, "endDd": day}
    try:
        out = session.post(MDC_URL, data=data, headers=_HEADERS, timeout=12).json().get("output") or []
    except Exception:
        return None
    by = {r.get("INVST_TP_NM"): r for r in out if isinstance(r, dict)}

    def val(name):
        r = by.get(name)
        if not r:
            return float("nan")
        return float(str(r.get("NETBID_TRDVAL", "0")).replace(",", "") or 0)

    return {"foreign_net": val("외국인"), "inst_net": val("기관합계")}


def fetch_kospi200_futures_flow(start, end, prod: str = KOSPI200_FUT_PROD,
                                session=None, calendar=None) -> pd.DataFrame:
    """Daily KOSPI200-futures investor net flow over [start, end] (foreign + institutional
    net-buy VALUE). Iterates trading days (one MDC call each). ``calendar`` (a DatetimeIndex
    of trading days) avoids weekend calls; defaults to business days. ``session`` injectable
    for tests. Returns a [date] frame (may be all-NaN/zero where KRX has no data)."""
    session = session or _authed_session()
    days = (pd.DatetimeIndex(calendar) if calendar is not None
            else pd.bdate_range(start, end))
    rows = {}
    for d in days:
        rec = _net_for_day(session, pd.Timestamp(d).strftime("%Y%m%d"), prod)
        if rec is not None:
            rows[pd.Timestamp(d)] = rec
    df = pd.DataFrame(rows).T.sort_index()
    df.index.name = "date"
    return df


def flow_csv_path(data_dir=None) -> Path:
    return (Path(data_dir) if data_dir is not None else Path(DATA_DIR)) / CSV


def load_futures_flow(data_dir=None) -> pd.DataFrame:
    """Cached KOSPI200-futures investor flow, or empty frame if not fetched."""
    p = flow_csv_path(data_dir)
    if not p.exists():
        return pd.DataFrame(columns=["foreign_net", "inst_net"])
    return pd.read_csv(p, index_col="date", parse_dates=True).sort_index()
