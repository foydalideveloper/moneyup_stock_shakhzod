"""Additional gated Korean signals (대차/신용/프로그램/투자자상세/공매도잔고).

Same authenticated pykrx path as 공매도/수급 (needs KRX_ID/KRX_PW). Some signals
have a pykrx function; the rest are not exposed by pykrx and must be exported
manually from data.krx.co.kr — for those, the live fetch raises
:class:`ManualCsvRequired` with the exact menu path + columns rather than failing
silently, and :func:`load_signal` reads a manual ``data/<code>_<signal>.csv`` if
present.

Signals (key → source):
  * ``investor_detail`` 투자자 상세 순매수 — pykrx ``get_market_trading_value_by_date(detail=True)``
    (연기금/금융투자/보험/투신/사모 …). REAL.
  * ``short_balance``   공매도 순보유잔고 — pykrx ``get_shorting_balance_by_date``
    (공매도잔고/공매도금액/비중). REAL (chunked, ~2y/request limit).
  * ``lending``         대차잔고 (securities-lending) — NOT in pykrx → manual CSV.
  * ``margin``          신용거래융자 잔고 — NOT in pykrx → manual CSV.
  * ``program``         프로그램 매매 순매수 — NOT in pykrx → manual CSV.

Each loads into a tidy frame indexed by **date** with the signal's canonical
columns; caches to ``data/<code>_<signal>.csv``. Build leak-free features with
:mod:`tagent.features_signals`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from tagent.config import DATA_DIR
from tagent.data import krx_source  # _import_pykrx via module so tests can patch it
from tagent.data.krx_source import _fetch_chunked, _yyyymmdd
from tagent.data.short_selling import _read_csv_any_encoding

_DATE_ALIASES = ["date", "dt", "timestamp", "날짜", "일자", "기준일", "거래일"]


class ManualCsvRequired(RuntimeError):
    """Raised when a signal isn't in pykrx and needs a manual KRX CSV export."""

    def __init__(self, spec: "SignalSpec", path: Path):
        self.spec = spec
        self.path = path
        super().__init__(
            f"'{spec.key}' ({spec.name}) is not exposed by pykrx — export it "
            f"manually from KRX and save to {path}.\n"
            f"  Menu:    {spec.manual_menu}\n"
            f"  Columns: 일자 + {spec.manual_columns}")


@dataclass
class SignalSpec:
    key: str
    name: str
    value_cols: List[str]                       # canonical output columns
    aliases: Dict[str, List[str]]               # canonical -> [raw headers]
    pykrx_supported: bool
    manual_menu: str = ""
    manual_columns: str = ""
    fetcher: Optional[Callable] = field(default=None, repr=False)


# --------------------------------------------------------------------------- #
# pykrx-backed fetchers (build canonical-named raw frames)
# --------------------------------------------------------------------------- #
def _fetch_investor_detail(stock, ticker: str, start, end) -> pd.DataFrame:
    df = stock.get_market_trading_value_by_date(
        _yyyymmdd(start), _yyyymmdd(end), str(ticker), on="순매수", detail=True)
    if df is None or df.empty:
        raise ValueError(
            f"No KRX investor-detail data for {ticker} — needs KRX_ID/KRX_PW.")
    out = pd.DataFrame(index=pd.to_datetime(df.index))

    def col(*names):
        for n in names:
            if n in df.columns:
                return pd.to_numeric(df[n].values, errors="coerce")
        return np.nan

    out["pension_net"] = col("연기금", "연기금 등")
    out["fininv_net"] = col("금융투자")
    out["insurance_net"] = col("보험")
    out["trust_net"] = col("투신")
    out["privequity_net"] = col("사모")
    out.index.name = "date"
    return out.sort_index()


def _fetch_short_balance(stock, ticker: str, start, end) -> pd.DataFrame:
    df = _fetch_chunked(stock.get_shorting_balance_by_date, ticker, start, end)
    if df is None or df.empty:
        raise ValueError(
            f"No KRX short-balance data for {ticker} — needs KRX_ID/KRX_PW.")
    out = pd.DataFrame(index=pd.to_datetime(df.index))
    out["short_balance"] = pd.to_numeric(df["공매도잔고"].values, errors="coerce")
    out["short_balance_value"] = pd.to_numeric(df["공매도금액"].values, errors="coerce")
    out["short_balance_ratio"] = pd.to_numeric(df["비중"].values, errors="coerce")
    out.index.name = "date"
    return out.sort_index()


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
SIGNALS: Dict[str, SignalSpec] = {
    "investor_detail": SignalSpec(
        key="investor_detail", name="투자자 상세 순매수 (pension/fininv/insurance/trust/PE)",
        value_cols=["pension_net", "fininv_net", "insurance_net", "trust_net",
                    "privequity_net"],
        aliases={
            "pension_net": ["pension_net", "연기금", "연기금 등"],
            "fininv_net": ["fininv_net", "금융투자"],
            "insurance_net": ["insurance_net", "보험"],
            "trust_net": ["trust_net", "투신"],
            "privequity_net": ["privequity_net", "사모"],
        },
        pykrx_supported=True, fetcher=_fetch_investor_detail),
    "short_balance": SignalSpec(
        key="short_balance", name="공매도 순보유잔고 (short balance)",
        value_cols=["short_balance", "short_balance_value", "short_balance_ratio"],
        aliases={
            "short_balance": ["short_balance", "공매도잔고", "공매도잔고수량"],
            "short_balance_value": ["short_balance_value", "공매도금액", "공매도잔고금액"],
            "short_balance_ratio": ["short_balance_ratio", "비중", "공매도잔고비중"],
        },
        pykrx_supported=True, fetcher=_fetch_short_balance),
    "lending": SignalSpec(
        key="lending", name="대차잔고 (securities-lending balance)",
        value_cols=["lending_balance", "lending_value"],
        aliases={
            "lending_balance": ["lending_balance", "대차잔고수량", "대차잔고", "잔고수량",
                                "대차거래잔고수량"],
            "lending_value": ["lending_value", "대차잔고금액", "잔고금액", "대차거래잔고금액"],
        },
        pykrx_supported=False,
        manual_menu="data.krx.co.kr → 통계 → 증권상품 → 대차거래 → [대차거래 잔고추이/내역 (개별종목)]",
        manual_columns="대차잔고수량, 대차잔고금액"),
    "margin": SignalSpec(
        key="margin", name="신용거래융자 잔고 (margin/credit balance)",
        value_cols=["margin_balance", "margin_value"],
        aliases={
            "margin_balance": ["margin_balance", "신용융자잔고수량", "융자잔고수량",
                               "신용잔고수량", "신용융자잔고"],
            "margin_value": ["margin_value", "신용융자잔고금액", "융자잔고금액", "신용잔고금액"],
        },
        pykrx_supported=False,
        manual_menu="data.krx.co.kr → 통계 → 주식 → 기타 → [신용거래융자 잔고 (개별종목)]",
        manual_columns="신용융자잔고수량(주), 신용융자잔고금액(원)"),
    "program": SignalSpec(
        key="program", name="프로그램 매매 순매수 (program trading net)",
        value_cols=["program_net", "program_net_vol"],
        aliases={
            "program_net": ["program_net", "프로그램순매수거래대금", "순매수거래대금",
                            "프로그램매매순매수금액", "순매수금액"],
            "program_net_vol": ["program_net_vol", "프로그램순매수수량", "순매수수량",
                                "프로그램매매순매수수량"],
        },
        pykrx_supported=False,
        manual_menu="data.krx.co.kr → 통계 → 주식 → 세부안내 → [프로그램매매 추이 (개별종목)] (순매수)",
        manual_columns="프로그램 순매수 거래대금(원), 순매수 수량(주)"),
}


def signal_csv_path(signal: str, symbol: str, data_dir: Optional[Path] = None) -> Path:
    base = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
    return base / f"{symbol.upper()}_{signal}.csv"


def _norm_header(s) -> str:
    return "".join(str(s).lower().split())


def _normalize_signal(spec: SignalSpec, raw: pd.DataFrame) -> pd.DataFrame:
    """Coerce an arbitrary export into the signal's canonical schema."""
    df = raw.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        date_col = next((c for c in df.columns
                         if _norm_header(c) in {_norm_header(a) for a in _DATE_ALIASES}), None)
        if date_col is not None:
            df = df.set_index(date_col)
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]
    df.index.name = "date"

    lookup = {}
    for canon, names in spec.aliases.items():
        for n in names:
            lookup[_norm_header(n)] = canon
    df = df.rename(columns={c: lookup[_norm_header(c)] for c in df.columns
                            if _norm_header(c) in lookup})

    for c in spec.value_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    present = [c for c in spec.value_cols if c in df.columns]
    if not present:
        raise ValueError(
            f"'{spec.key}' data needs at least one of {spec.value_cols} "
            f"(after alias resolution). Got: {list(raw.columns)}")
    for c in spec.value_cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[spec.value_cols]
    return df[~df.index.duplicated(keep="last")].sort_index()


def _csv_fetch_factory(signal: str, data_dir: Optional[Path]):
    def _fetch(symbol: str) -> pd.DataFrame:
        path = signal_csv_path(signal, symbol, data_dir)
        if not path.exists():
            raise FileNotFoundError(
                f"No {signal} CSV at {path}. Fetch via krx_signal_fetch (if pykrx "
                "supports it) or drop a manual KRX export there.")
        return _read_csv_any_encoding(path)
    return _fetch


def load_signal(signal: str, symbol: str, fetch: Optional[Callable] = None,
                data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load a signal into its canonical schema (CSV by default, or via `fetch`)."""
    spec = SIGNALS[signal]
    fetch = fetch or _csv_fetch_factory(signal, data_dir)
    return _normalize_signal(spec, fetch(symbol))


def krx_signal_fetch(signal: str, start, end, cache: bool = True,
                     data_dir: Optional[Path] = None) -> Callable:
    """Build a ``fetch(symbol)`` that pulls `signal` live via pykrx (authenticated).

    For signals pykrx does NOT support, the returned callable raises
    :class:`ManualCsvRequired` (menu path + columns) instead of failing silently.
    """
    spec = SIGNALS[signal]

    def _fetch(symbol: str) -> pd.DataFrame:
        path = signal_csv_path(signal, symbol, data_dir)
        if cache and path.exists():
            return _read_csv_any_encoding(path)
        if not spec.pykrx_supported:
            raise ManualCsvRequired(spec, path)
        krx_source.ensure_krx_login()
        stock = krx_source._import_pykrx()
        raw = spec.fetcher(stock, symbol, start, end)
        if cache and not raw.empty:
            raw.to_csv(path, index_label="date")
        return raw

    return _fetch
