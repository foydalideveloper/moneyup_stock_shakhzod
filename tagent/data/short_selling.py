"""Korean short-selling (공매도) data layer — DAILY / end-of-day, delayed.

KRX publishes short-selling statistics **once per day, after the close**, and the
public release is **delayed** (figures for trading day D become available the
following business day, and short-balance / 공매도잔고 lags further). Treat every
value here as known only *after* its date — never intraday. The feature layer
(:mod:`tagent.features_short`) enforces this with a 1-day shift so nothing leaks.

This loader returns a tidy DataFrame indexed by date with the canonical columns:

    short_volume   공매도 거래량  — shares sold short that day (required)
    volume         거래량        — total shares traded that day (optional*)
    short_ratio    공매도 비중     — short_volume / volume (computed when volume is
                                  present; else completed via complete_short_ratio)
    short_value    공매도 거래대금  — KRW value sold short (optional)
    short_balance  공매도 잔고     — outstanding short balance, when available (optional)

*Some KRX exports (the 공매도거래 dataset) carry short volume but NOT total market
volume. Those load fine with volume/short_ratio left NaN; supply the denominator
afterwards with :func:`complete_short_ratio` (e.g. from the price OHLCV volume).

The data *source* is pluggable. A CSV export works today; a live Kiwoom/KRX API
adapter can be dropped in later by passing a different ``fetch`` callable that
returns a raw DataFrame — nothing downstream changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from tagent.config import DATA_DIR

# Canonical output columns. short_value / short_balance are optional (may be NaN).
SHORT_COLS = ["short_volume", "volume", "short_ratio", "short_value", "short_balance"]

# Map messy real-world headers (English + common KRX/Korean labels) to canonical
# names. Matching is case-insensitive and whitespace-stripped.
_ALIASES = {
    "short_volume": ["short_volume", "short_qty", "shortvol", "공매도", "공매도수량",
                     "공매도거래량", "공매도수량(주)",
                     # KRX data.krx.co.kr 공매도거래 export (multi-line header flattened):
                     "공매도 수량_거래량_전체", "공매도수량_거래량_전체"],
    "volume": ["volume", "total_volume", "거래량", "총거래량", "거래량(주)"],
    "short_ratio": ["short_ratio", "ratio", "공매도비중", "비중", "공매도 비중"],
    "short_value": ["short_value", "value", "공매도금액", "공매도거래대금", "공매도대금",
                    # KRX export:
                    "공매도 금액_거래대금_전체", "공매도금액_거래대금_전체"],
    "short_balance": ["short_balance", "balance", "공매도잔고", "잔고",
                      "공매도잔고수량", "대차잔고",
                      # KRX export net-holding balance (quantity):
                      "공매도 수량_순보유잔고수량", "공매도수량_순보유잔고수량"],
}
_DATE_ALIASES = ["date", "dt", "timestamp", "일자", "날짜", "기준일", "거래일"]

# A fetch function takes a symbol and returns a *raw* DataFrame (any headers).
FetchFn = Callable[[str], pd.DataFrame]


def _build_rename_map(columns) -> dict:
    """Resolve each incoming column to a canonical name (or leave it alone)."""
    lookup = {}
    for canonical, aliases in _ALIASES.items():
        for a in aliases:
            lookup[a.lower().strip()] = canonical
    rename = {}
    for col in columns:
        key = str(col).lower().strip()
        if key in lookup:
            rename[col] = lookup[key]
    return rename


def _normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """Coerce an arbitrary short-selling export into the canonical schema."""
    df = raw.copy()

    # 1) Find / set the date index.
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
    for col in SHORT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[[c for c in SHORT_COLS if c in df.columns]].copy()

    if "short_volume" not in df.columns:
        raise ValueError(
            "Short-selling data needs at least a short_volume column (after alias "
            f"resolution). Got: {list(raw.columns)}")

    # 4) Make sure every canonical column exists (as NaN) so the schema is stable.
    for col in SHORT_COLS:
        if col not in df.columns:
            df[col] = np.nan

    # 5) Compute short_ratio from total volume when possible. Some KRX exports
    #    (the 공매도거래 dataset) carry short volume but NOT total market volume —
    #    in that case short_ratio stays NaN here and is completed later from the
    #    price file via complete_short_ratio().
    if df["short_ratio"].isna().all() and df["volume"].notna().any():
        vol = df["volume"].replace(0.0, np.nan)
        df["short_ratio"] = df["short_volume"] / vol

    df = df[SHORT_COLS]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def complete_short_ratio(short_df: pd.DataFrame, total_volume: pd.Series) -> pd.DataFrame:
    """Fill ``volume`` / ``short_ratio`` from an external total-volume series.

    KRX's 공매도거래 export has short-sell volume but no total market volume, so
    ``short_ratio = short_volume / volume`` can't be computed from it alone. Pass
    the total daily volume (e.g. ``load_history(code)["volume"]``) and this fills
    the missing denominator (aligning by date) and recomputes the ratio where it
    is still NaN. Returns a new DataFrame; the input is not mutated.
    """
    out = short_df.copy()
    vol = pd.to_numeric(total_volume, errors="coerce").reindex(out.index)
    out["volume"] = out["volume"].fillna(vol)
    need = out["short_ratio"].isna()
    safe_vol = out["volume"].replace(0.0, np.nan)
    out.loc[need, "short_ratio"] = out.loc[need, "short_volume"] / safe_vol[need]
    return out


def short_csv_path(symbol: str, data_dir: Optional[Path] = None) -> Path:
    base = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
    return base / f"{symbol.upper()}_short.csv"


# KRX exports from data.krx.co.kr are usually cp949/EUC-KR, not UTF-8.
_CSV_ENCODINGS = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]


def _read_csv_any_encoding(path: Path) -> pd.DataFrame:
    """Read a CSV trying UTF-8 first, then Korean cp949/EUC-KR."""
    last_err: Optional[Exception] = None
    for enc in _CSV_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, UnicodeError) as e:
            last_err = e
    raise ValueError(
        f"Could not decode {path} as any of {_CSV_ENCODINGS}: {last_err}")


def _csv_fetch_factory(data_dir: Optional[Path]) -> FetchFn:
    def _fetch(symbol: str) -> pd.DataFrame:
        path = short_csv_path(symbol, data_dir)
        if not path.exists():
            raise FileNotFoundError(
                f"No short-selling CSV at {path}. Export one there, or pass a "
                "custom fetch= (e.g. a Kiwoom/KRX adapter).")
        return _read_csv_any_encoding(path)
    return _fetch


def load_short_selling(symbol: str, fetch: Optional[FetchFn] = None,
                       data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load DAILY/EOD short-selling data for a symbol into the canonical schema.

    By default reads ``data/<SYMBOL>_short.csv``. Pass ``fetch`` to swap in a
    live source later (Kiwoom/KRX) — it just needs to return a raw DataFrame; the
    same normalization (alias resolution, ratio math) is applied either way.
    """
    fetch = fetch or _csv_fetch_factory(data_dir)
    raw = fetch(symbol)
    return _normalize(raw)
