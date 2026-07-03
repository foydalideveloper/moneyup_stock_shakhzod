"""VKOSPI (KOSPI200 implied-volatility index) source for the VRP trial.

VKOSPI is KRX's 30-day implied-vol index from KOSPI200 options. It is NOT in pykrx's index
universe (163 indices, none volatility) and the KRX MDC index store does not expose it in
this environment, so ``fetch_vkospi`` attempts the MDC index-daily endpoint (reusing pykrx's
authenticated session) and returns whatever it gets — empty here. ``load_vkospi`` reads a
cached CSV (date, vkospi) if one is dropped in. The runner reports the data wall honestly
when VKOSPI is unavailable, rather than faking implied vol from realized.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tagent.config import DATA_DIR

CSV = "vkospi_1d.csv"
MDC_URL = "https://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
# VKOSPI daily index series (KRX MDC). Code/bld kept here for when a populated KRX is reachable.
VKOSPI_BLD = "dbms/MDC/STAT/standard/MDCSTAT00701"       # 개별지수 시세추이
_HEADERS = {"User-Agent": "Mozilla/5.0", "X-Requested-With": "XMLHttpRequest",
            "Referer": "https://data.krx.co.kr/contents/MDC/MDI/outerLoader/index.cmd"}


def vkospi_csv_path(data_dir=None) -> Path:
    return (Path(data_dir) if data_dir is not None else Path(DATA_DIR)) / CSV


def load_vkospi(data_dir=None) -> pd.Series:
    """Cached VKOSPI daily level (Series indexed by date), or empty if unavailable."""
    p = vkospi_csv_path(data_dir)
    if not p.exists():
        return pd.Series(dtype=float)
    df = pd.read_csv(p, index_col=0, parse_dates=True).sort_index()
    col = "vkospi" if "vkospi" in df.columns else df.columns[0]
    return df[col].astype(float)


def fetch_vkospi(start, end, session=None) -> pd.Series:
    """Best-effort VKOSPI daily series via the KRX MDC (authenticated session). Returns an
    empty Series if the endpoint yields no data (the case in this environment)."""
    if session is None:
        from tagent.data.krx_deriv_investor import _authed_session
        session = _authed_session()
    data = {"bld": VKOSPI_BLD, "locale": "ko_KR", "csvxls_isNo": "false",
            "indIdx": "5", "indIdx2": "300", "strtDd": pd.Timestamp(start).strftime("%Y%m%d"),
            "endDd": pd.Timestamp(end).strftime("%Y%m%d")}
    try:
        out = session.post(MDC_URL, data=data, headers=_HEADERS, timeout=15).json().get("output") or []
    except Exception:
        return pd.Series(dtype=float)
    rows = {}
    for r in out:
        if not isinstance(r, dict):
            continue
        d = r.get("TRD_DD")
        v = str(r.get("CLSPRC_IDX", r.get("IDX", "0"))).replace(",", "")
        try:
            rows[pd.Timestamp(d)] = float(v)
        except Exception:
            pass
    return pd.Series(rows).sort_index()


def is_degenerate(vkospi) -> bool:
    """True if VKOSPI is empty or all-zero/NaN (no usable implied vol)."""
    s = pd.Series(vkospi, dtype=float).dropna()
    return len(s) == 0 or bool((s == 0).all())
