"""Korean short-selling data layer + feature tests (synthetic, no network)."""

import numpy as np
import pandas as pd

from tagent.data.short_selling import (
    SHORT_COLS,
    complete_short_ratio,
    load_short_selling,
    _normalize,
)
from tagent.features import make_features
from tagent.features_short import (
    SHORT_BAN_WINDOWS,
    SHORT_FEATURE_COLS,
    in_short_ban,
    make_short_features,
    merge_short_features,
)
from tagent.ml.train import build_dataset


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _synthetic_short(n=40, seed=1):
    """Daily short-selling frame with a clear spike to test alignment."""
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    rng = np.random.default_rng(seed)
    volume = rng.integers(1_000_000, 2_000_000, n).astype(float)
    short_volume = (volume * rng.uniform(0.05, 0.15, n)).round()
    df = pd.DataFrame({
        "short_volume": short_volume,
        "volume": volume,
        "short_ratio": short_volume / volume,
        "short_value": short_volume * 100.0,
        "short_balance": np.cumsum(rng.integers(-5_000, 5_000, n)).astype(float) + 1e6,
    }, index=idx)
    df.index.name = "date"
    return df


def _sample_ohlcv(idx, seed=0):
    rng = np.random.default_rng(seed)
    n = len(idx)
    price = np.maximum(100 + np.cumsum(rng.normal(0, 1, n)), 1.0)
    close = pd.Series(price, index=idx)
    return pd.DataFrame({
        "open": close.shift(1).fillna(close),
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": pd.Series(rng.integers(1e6, 2e6, n), index=idx).astype(float),
    })


# --------------------------------------------------------------------------- #
# loader / ratio math
# --------------------------------------------------------------------------- #
def test_ratio_math_computed_when_absent():
    raw = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02"],
        "short_volume": [100.0, 200.0],
        "volume": [1000.0, 1000.0],
    })
    df = _normalize(raw)
    assert list(df["short_ratio"]) == [0.1, 0.2]
    assert isinstance(df.index, pd.DatetimeIndex)


def test_korean_aliases_and_csv_load(tmp_path):
    """Korean KRX-style headers resolve to the canonical schema via a CSV."""
    csv = tmp_path / "005930_short.csv"
    csv.write_text(
        "일자,공매도,거래량,공매도잔고\n"
        "2024-01-01,150,1500,9000\n"
        "2024-01-02,300,1000,9500\n",
        encoding="utf-8",
    )
    df = load_short_selling("005930", data_dir=tmp_path)
    assert list(df.columns) == ["short_volume", "volume", "short_ratio",
                                "short_value", "short_balance"]
    assert df["short_ratio"].iloc[0] == 0.1   # 150 / 1500
    assert df["short_ratio"].iloc[1] == 0.3   # 300 / 1000
    assert df["short_balance"].iloc[1] == 9500.0
    assert df["short_value"].isna().all()      # not present in this export


def test_pluggable_fetch():
    """A custom fetch (stand-in for a Kiwoom/KRX API) feeds the same pipeline."""
    def fake_api(symbol):
        return pd.DataFrame({
            "date": pd.date_range("2024-03-01", periods=3, freq="D"),
            "short_volume": [10.0, 20.0, 30.0],
            "volume": [100.0, 100.0, 100.0],
        })

    df = load_short_selling("ANY", fetch=fake_api)
    assert list(df["short_ratio"]) == [0.1, 0.2, 0.3]


def test_zero_volume_ratio_is_nan():
    raw = pd.DataFrame({
        "date": ["2024-01-01"], "short_volume": [50.0], "volume": [0.0],
    })
    df = _normalize(raw)
    assert np.isnan(df["short_ratio"].iloc[0])


# --------------------------------------------------------------------------- #
# no-lookahead alignment
# --------------------------------------------------------------------------- #
def test_no_lookahead_alignment():
    """A short-ratio spike on day D must NOT appear on the price bar for day D.

    Because the data is EOD/delayed, the bar at D may only see data dated < D, so
    the spike should surface one day later (D+1).
    """
    idx = pd.date_range("2024-01-01", periods=8, freq="D")
    short = pd.DataFrame({
        "short_volume": [100, 100, 100, 100, 900, 100, 100, 100],  # spike at idx[4]
        "volume": [1000.0] * 8,
        "short_value": np.nan,
        "short_balance": np.nan,
    }, index=idx)
    short["short_ratio"] = short["short_volume"] / short["volume"]

    feats = make_short_features(short, idx, z_window=3)
    spike_day = idx[4]
    next_day = idx[5]

    # On the spike's own date the aligned ratio reflects the PRIOR day (0.1),
    # never the spike (0.9): no future leakage.
    assert feats.loc[spike_day, "short_ratio"] == 0.1
    assert feats.loc[next_day, "short_ratio"] == 0.9   # surfaces exactly one day later


def test_alignment_uses_only_past_dates():
    """Every aligned value at bar T must come from a short date strictly < T."""
    idx = pd.date_range("2024-02-01", periods=10, freq="D")
    short = _synthetic_short(n=10)
    short.index = idx  # same calendar as the price bars

    feats = make_short_features(short, idx, z_window=3)
    # The very first bar has no prior short day available -> NaN.
    assert np.isnan(feats["short_ratio"].iloc[0])
    # Each later bar equals the *previous* day's raw short_ratio (1-day delay).
    expected_prev = short["short_ratio"].shift(1)
    for t in idx[1:]:
        assert np.isclose(feats.loc[t, "short_ratio"], expected_prev.loc[t])


def test_feature_columns_present():
    idx = pd.date_range("2024-01-01", periods=30, freq="D")
    feats = make_short_features(_synthetic_short(n=30), idx)
    assert list(feats.columns) == SHORT_FEATURE_COLS
    assert feats.index.equals(idx)


def test_balance_change_zero_when_no_balance():
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    short = _synthetic_short(n=10)
    short["short_balance"] = np.nan
    feats = make_short_features(short, idx, z_window=3)
    # Degrades to a constant 0 (not NaN) so rows survive the study's dropna().
    valid = feats["short_balance_change"].dropna()
    assert (valid == 0.0).all() and not valid.empty


# --------------------------------------------------------------------------- #
# merge / row counts
# --------------------------------------------------------------------------- #
def test_merge_adds_columns_and_preserves_rows():
    idx = pd.date_range("2024-01-01", periods=60, freq="D")
    ohlcv = _sample_ohlcv(idx)
    base = make_features(ohlcv, dropna=False)
    short = _synthetic_short(n=60)
    short.index = idx

    merged = merge_short_features(base, short)
    # Same rows, just extra columns (merge does not drop price bars).
    assert len(merged) == len(base)
    assert set(SHORT_FEATURE_COLS).issubset(merged.columns)
    # No accidental column collisions / duplications.
    assert len(merged.columns) == len(base.columns) + len(SHORT_FEATURE_COLS)


def test_build_dataset_short_is_optional_and_nonbreaking():
    idx = pd.date_range("2022-01-01", periods=120, freq="D")
    history = {"AAA": _sample_ohlcv(idx, seed=2)}

    X_base, y_base = build_dataset(history, horizon=5)
    assert not set(SHORT_FEATURE_COLS).issubset(X_base.columns)  # unchanged default

    short = _synthetic_short(n=120)
    short.index = idx
    X_short, y_short = build_dataset(history, horizon=5, short_data={"AAA": short})

    assert set(SHORT_FEATURE_COLS).issubset(X_short.columns)     # columns merged in
    # Row count stays sane: merging never *adds* rows, and labels/index match.
    assert 0 < len(X_short) <= len(X_base)
    assert X_short.index.equals(y_short.index)


# --------------------------------------------------------------------------- #
# real KRX export shape: cp949 + 공매도거래 headers (no total volume)
# --------------------------------------------------------------------------- #
# Header exactly as data.krx.co.kr's 공매도거래 export (flattened multi-line).
_KRX_HEADER = ("일자,공매도 수량_거래량_전체,공매도 수량_거래량_업틱룰적용,"
               "공매도 수량_거래량_업틱룰예외,공매도 수량_순보유잔고수량,"
               "공매도 금액_거래대금_전체,공매도 금액_거래대금_업틱룰적용,"
               "공매도 금액_거래대금_업틱룰예외,공매도 금액_순보유잔고금액")


def _write_krx_cp949(path, dates_desc, short_vol_desc, balance_desc):
    """Write a KRX-style cp949 CSV (rows newest-first, like the real export)."""
    lines = [_KRX_HEADER]
    for d, sv, bal in zip(dates_desc, short_vol_desc, balance_desc):
        # cols: 일자, svol, uptick, exc, balance, value, val_uptick, val_exc, bal_val
        lines.append(f'"{d}","{sv}","0","0","{bal}","{sv * 70000}","0","0","0"')
    path.write_bytes(("\n".join(lines) + "\n").encode("cp949"))


def test_krx_cp949_export_maps_and_completes_ratio(tmp_path):
    # Newest-first, like KRX. Ascending balance over time: 5000..5400.
    _write_krx_cp949(
        tmp_path / "005930_short.csv",
        dates_desc=["2024/01/05", "2024/01/04", "2024/01/03", "2024/01/02", "2024/01/01"],
        short_vol_desc=[100, 100, 900, 100, 100],
        balance_desc=[5400, 5300, 5200, 5100, 5000],
    )

    df = load_short_selling("005930", data_dir=tmp_path)
    # cp949 decoded + Korean headers mapped to the canonical schema, sorted ascending.
    assert list(df.columns) == SHORT_COLS
    assert isinstance(df.index, pd.DatetimeIndex)
    assert list(df["short_volume"]) == [100, 100, 900, 100, 100]
    assert list(df["short_balance"]) == [5000, 5100, 5200, 5300, 5400]
    assert (df["short_value"] > 0).all()
    # This export has NO total volume -> volume + short_ratio stay NaN for now.
    assert df["volume"].isna().all()
    assert df["short_ratio"].isna().all()

    # Complete the ratio from a total-volume series (e.g. the price OHLCV volume).
    total_vol = pd.Series(1000.0, index=df.index)
    done = complete_short_ratio(df, total_vol)
    assert list(done["short_ratio"]) == [0.1, 0.1, 0.9, 0.1, 0.1]   # short_vol / 1000
    assert list(done["volume"]) == [1000.0] * 5


def test_krx_export_no_lookahead(tmp_path):
    """A short spike on day D must not surface on the price bar for day D."""
    dates_asc = ["2024/01/01", "2024/01/02", "2024/01/03", "2024/01/04", "2024/01/05"]
    _write_krx_cp949(
        tmp_path / "005930_short.csv",
        dates_desc=list(reversed(dates_asc)),
        short_vol_desc=[100, 100, 900, 100, 100][::-1],   # spike at 2024/01/03
        balance_desc=[5400, 5300, 5200, 5100, 5000],
    )
    short = load_short_selling("005930", data_dir=tmp_path)
    idx = pd.to_datetime(dates_asc)
    short = complete_short_ratio(short, pd.Series(1000.0, index=short.index))

    feats = make_short_features(short, idx, z_window=2)
    spike_day = pd.Timestamp("2024-01-03")
    next_day = pd.Timestamp("2024-01-04")
    # EOD/delayed: the spike (0.9) is NOT visible on its own bar, only the day after.
    assert feats.loc[spike_day, "short_ratio"] == 0.1
    assert feats.loc[next_day, "short_ratio"] == 0.9


# --------------------------------------------------------------------------- #
# short-selling-ban window flagging
# --------------------------------------------------------------------------- #
def test_in_short_ban_flags_ban_windows():
    idx = pd.to_datetime([
        "2019-06-01",   # before any ban
        "2020-03-16",   # COVID ban start (inclusive)
        "2020-08-01",   # inside COVID ban
        "2021-05-02",   # COVID ban end (inclusive)
        "2021-05-03",   # day after lift -> not banned
        "2022-01-01",   # between bans
        "2023-11-06",   # 2023 ban start (inclusive)
        "2024-06-15",   # inside 2023 ban
        "2025-03-31",   # 2023 ban end (inclusive)
        "2025-04-01",   # after lift -> not banned
    ])
    mask = in_short_ban(idx)
    assert list(mask) == [False, True, True, True, False,
                          False, True, True, True, False]
    assert len(SHORT_BAN_WINDOWS) == 2
