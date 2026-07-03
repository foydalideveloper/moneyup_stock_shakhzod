"""Korean 수급 (investor-flow) data layer — DAILY foreign/institutional net buying.

KRX publishes, per ticker per day, the **net buying** of each investor group
(외국인 foreign, 기관 institutional, 개인 retail, ...). This layer fetches the
foreign + institutional net series via pykrx and normalizes them into a tidy
canonical frame:

    foreign_net   외국인합계 순매수  — foreign net buying that day (value or volume)
    inst_net      기관합계  순매수  — institutional net buying that day

pykrx function (verified from its own docstring):

    get_market_trading_value_by_date(from, to, ticker, on="순매수")
      -> columns: 기관합계, 기타법인, 개인, 외국인합계, 전체   (net VALUE, KRW)
    get_market_trading_volume_by_date(...) -> same columns, net VOLUME (shares)

Like the 공매도 endpoints, this per-ticker investor endpoint is **gated behind a
free KRX account** (pykrx reads ``KRX_ID`` / ``KRX_PW`` from the environment).
Without them it returns an empty frame; the fetch detects that and raises a clear
error instead of silently producing nothing. You can also drop a manual CSV at
``data/<code>_flows.csv``.

This is DAILY / end-of-day data, so the feature layer
(:mod:`tagent.features_flows`) shifts by 1 day to stay leak-free — exactly like
the short-selling features.

pykrx is imported lazily (via :mod:`tagent.data.krx_source`), so the package and
test suite import cleanly without it; tests inject a fake module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from tagent.config import DATA_DIR
from tagent.data import krx_source  # call _import_pykrx via module so tests can patch it
from tagent.data.krx_source import _yyyymmdd
from tagent.data.short_selling import _read_csv_any_encoding

# Canonical output columns.
FLOW_COLS = ["foreign_net", "inst_net"]

# Map KRX/Korean headers (and English) to canonical names.
_ALIASES = {
    "foreign_net": ["foreign_net", "외국인합계", "외국인", "외국인순매수", "foreign"],
    "inst_net": ["inst_net", "기관합계", "기관", "기관순매수", "institution",
                 "institutional"],
}
_DATE_ALIASES = ["date", "dt", "timestamp", "날짜", "일자", "기준일", "거래일"]

# KRX column names used when pulling from pykrx.
_KRX_FOREIGN_COL = "외국인합계"
_KRX_INST_COL = "기관합계"

# A fetch function takes a symbol and returns a *raw* DataFrame (any headers).
FlowFetchFn = Callable[[str], pd.DataFrame]


def flows_csv_path(symbol: str, data_dir: Optional[Path] = None) -> Path:
    base = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
    return base / f"{symbol.upper()}_flows.csv"


def _norm_header(s) -> str:
    """Whitespace-insensitive, lowercased key. KRX exports headers like
    '기관 합계' / '외국인 합계' (with a space) while our aliases are space-free."""
    return "".join(str(s).lower().split())


def _build_rename_map(columns) -> dict:
    lookup = {}
    for canonical, aliases in _ALIASES.items():
        for a in aliases:
            lookup[_norm_header(a)] = canonical
    rename = {}
    for col in columns:
        key = _norm_header(col)
        if key in lookup:
            rename[col] = lookup[key]
    return rename


def _normalize_flows(raw: pd.DataFrame) -> pd.DataFrame:
    """Coerce an arbitrary investor-flow export into the canonical schema."""
    df = raw.copy()

    # 1) Date index.
    if not isinstance(df.index, pd.DatetimeIndex):
        date_col = next(
            (c for c in df.columns if str(c).lower().strip() in _DATE_ALIASES), None)
        if date_col is not None:
            df = df.set_index(date_col)
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]
    df.index.name = "date"

    # 2) Rename known columns to canonical names.
    df = df.rename(columns=_build_rename_map(df.columns))

    # 3) Coerce numerics; keep only columns we understand.
    for col in FLOW_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[[c for c in FLOW_COLS if c in df.columns]].copy()

    if "foreign_net" not in df.columns and "inst_net" not in df.columns:
        raise ValueError(
            "Investor-flow data needs at least foreign_net or inst_net (after "
            f"alias resolution). Got: {list(raw.columns)}")

    # 4) Ensure both canonical columns exist (as NaN) so the schema is stable.
    for col in FLOW_COLS:
        if col not in df.columns:
            df[col] = np.nan

    df = df[FLOW_COLS]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _csv_fetch_factory(data_dir: Optional[Path]) -> FlowFetchFn:
    def _fetch(symbol: str) -> pd.DataFrame:
        path = flows_csv_path(symbol, data_dir)
        if not path.exists():
            raise FileNotFoundError(
                f"No investor-flow CSV at {path}. Fetch one with krx_flows_fetch "
                "(needs KRX_ID/KRX_PW), or drop a manual KRX export there.")
        return _read_csv_any_encoding(path)
    return _fetch


def load_flows(symbol: str, fetch: Optional[FlowFetchFn] = None,
               data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load DAILY foreign/institutional net-buying flows into the canonical schema.

    By default reads ``data/<SYMBOL>_flows.csv``. Pass ``fetch`` (e.g.
    :func:`krx_flows_fetch`) to pull live from KRX; the same normalization is
    applied either way.
    """
    fetch = fetch or _csv_fetch_factory(data_dir)
    raw = fetch(symbol)
    return _normalize_flows(raw)


def krx_flows_fetch(start, end, on: str = "순매수", use_volume: bool = False,
                    cache: bool = True, data_dir: Optional[Path] = None) -> FlowFetchFn:
    """Build a ``fetch(symbol)`` that pulls KR investor flows via pykrx.

    Returns foreign + institutional **net** series (value by default, or volume
    if ``use_volume``) for ``[start, end]``, and caches ``data/<code>_flows.csv``.
    Raises a clear error if KRX returns nothing (the endpoint is gated behind a
    free KRX account exposed via KRX_ID / KRX_PW).
    """
    def _fetch(symbol: str) -> pd.DataFrame:
        path = flows_csv_path(symbol, data_dir)
        if cache and path.exists():
            return _read_csv_any_encoding(path)

        krx_source.ensure_krx_login()   # 수급 endpoint is gated -> needs KRX_ID/KRX_PW
        stock = krx_source._import_pykrx()
        fn = (stock.get_market_trading_volume_by_date if use_volume
              else stock.get_market_trading_value_by_date)
        df = fn(_yyyymmdd(start), _yyyymmdd(end), str(symbol), on=on)
        if df is None or df.empty:
            raise ValueError(
                f"No KRX investor-flow data for {symbol}. The per-ticker 수급 "
                "endpoint is gated — set a free KRX account in KRX_ID / KRX_PW "
                "env vars, or drop a manual CSV at "
                f"{flows_csv_path(symbol, data_dir)}.")

        out = pd.DataFrame(index=pd.to_datetime(df.index))
        out["foreign_net"] = pd.to_numeric(df[_KRX_FOREIGN_COL].values, errors="coerce")
        out["inst_net"] = pd.to_numeric(df[_KRX_INST_COL].values, errors="coerce")
        out.index.name = "date"
        out = out.sort_index()
        if cache and not out.empty:
            out.to_csv(path, index_label="date")
        return out

    return _fetch
