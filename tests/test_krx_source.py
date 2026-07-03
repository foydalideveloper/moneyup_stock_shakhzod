"""KRX (pykrx) loader tests — fully mocked, no network.

A fake ``stock`` module returns the same Korean-headed frames pykrx produces, so
we verify the schema mapping (OHLCV + 공매도) and CSV caching offline.
"""

import os

import numpy as np
import pandas as pd
import pytest

from tagent.config import SETTINGS
from tagent.data import krx_source
from tagent.data.short_selling import SHORT_COLS, load_short_selling


# --------------------------------------------------------------------------- #
# fake pykrx
# --------------------------------------------------------------------------- #
class FakeStock:
    """Mimics the pykrx.stock functions we use, with KRX's Korean columns."""

    def __init__(self):
        self.calls = []

    def _idx(self, n):
        idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"][:n])
        idx.name = "날짜"
        return idx

    def get_market_ohlcv_by_date(self, fromdate, todate, ticker):
        self.calls.append(("ohlcv", fromdate, todate, ticker))
        idx = self._idx(3)
        return pd.DataFrame({
            "시가": [78200, 78500, 77000],
            "고가": [79800, 78800, 77300],
            "저가": [78200, 77000, 76100],
            "종가": [79600, 77000, 76200],
            "거래량": [17142847, 21753644, 15823002],
            "등락률": [1.40, -3.27, -1.04],
        }, index=idx)

    def get_shorting_volume_by_date(self, fromdate, todate, ticker):
        self.calls.append(("svol", fromdate, todate, ticker))
        idx = self._idx(3)
        return pd.DataFrame({
            "공매도": [100.0, 200.0, 300.0],
            "매수": [1000.0, 1000.0, 1500.0],   # total volume (pykrx's odd label)
            "비중": [10.0, 20.0, 20.0],
        }, index=idx)

    def get_shorting_balance_by_date(self, fromdate, todate, ticker):
        self.calls.append(("sbal", fromdate, todate, ticker))
        idx = self._idx(3)
        return pd.DataFrame({
            "공매도잔고": [5000.0, 6000.0, 5500.0],
            "상장주식수": [1e9, 1e9, 1e9],
            "공매도금액": [5e5, 6e5, 5.5e5],
            "시가총액": [1e12, 1e12, 1e12],
            "비중": [0.5, 0.6, 0.55],
        }, index=idx)


@pytest.fixture
def fake_krx(monkeypatch):
    fake = FakeStock()
    monkeypatch.setattr(krx_source, "_import_pykrx", lambda: fake)
    return fake


# --------------------------------------------------------------------------- #
# OHLCV
# --------------------------------------------------------------------------- #
def test_get_krx_history_maps_to_canonical_schema(fake_krx, tmp_path):
    df = krx_source.get_krx_history("005930", "2024-01-01", "2024-01-31",
                                    data_dir=tmp_path)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.name == "timestamp"
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df["close"].iloc[0] == 79600
    assert df["volume"].iloc[1] == 21753644
    # Dates are passed to pykrx in YYYYMMDD form.
    assert fake_krx.calls[0] == ("ohlcv", "20240101", "20240131", "005930")


def test_get_krx_history_caches_csv(fake_krx, tmp_path):
    krx_source.get_krx_history("005930", "2024-01-01", "2024-01-31", data_dir=tmp_path)
    path = tmp_path / "005930_1d.csv"
    assert path.exists()
    reread = pd.read_csv(path, index_col="timestamp", parse_dates=True)
    assert list(reread.columns) == ["open", "high", "low", "close", "volume"]
    assert len(reread) == 3


def test_get_krx_history_drops_zero_close_rows(fake_krx, monkeypatch, tmp_path):
    def with_zero(fromdate, todate, ticker):
        idx = pd.to_datetime(["2024-01-02", "2024-01-03"]); idx.name = "날짜"
        return pd.DataFrame({
            "시가": [100, 0], "고가": [110, 0], "저가": [95, 0],
            "종가": [105, 0], "거래량": [1000, 0], "등락률": [1.0, 0.0],
        }, index=idx)
    monkeypatch.setattr(fake_krx, "get_market_ohlcv_by_date", with_zero)
    df = krx_source.get_krx_history("005930", "2024-01-01", "2024-01-05",
                                    save=False, data_dir=tmp_path)
    assert len(df) == 1  # the all-zero non-trading row is dropped


# --------------------------------------------------------------------------- #
# short-selling fetch (through the existing normalizer)
# --------------------------------------------------------------------------- #
def test_krx_short_fetch_maps_to_canonical_schema(fake_krx, tmp_path):
    fetch = krx_source.krx_short_fetch("2024-01-01", "2024-01-31",
                                       cache=False, data_dir=tmp_path)
    df = load_short_selling("005930", fetch=fetch)

    assert list(df.columns) == SHORT_COLS
    assert list(df["short_volume"]) == [100.0, 200.0, 300.0]
    assert list(df["volume"]) == [1000.0, 1000.0, 1500.0]
    # short_ratio is recomputed as a fraction (not pykrx's percent 비중).
    assert df["short_ratio"].iloc[0] == 0.1
    assert df["short_ratio"].iloc[1] == 0.2
    # balance + value come from get_shorting_balance_by_date.
    assert list(df["short_balance"]) == [5000.0, 6000.0, 5500.0]
    assert df["short_value"].iloc[2] == 5.5e5


def test_krx_short_fetch_caches_and_round_trips(fake_krx, tmp_path):
    fetch = krx_source.krx_short_fetch("2024-01-01", "2024-01-31",
                                       cache=True, data_dir=tmp_path)
    df1 = load_short_selling("005930", fetch=fetch)
    cache = tmp_path / "005930_short.csv"
    assert cache.exists()

    # Second call: a cached CSV exists, so pykrx must NOT be hit again.
    fake_krx.calls.clear()
    df2 = load_short_selling("005930", fetch=fetch)
    assert fake_krx.calls == []
    pd.testing.assert_frame_equal(df1, df2)


def test_krx_short_fetch_balance_optional(fake_krx, monkeypatch, tmp_path):
    monkeypatch.setattr(fake_krx, "get_shorting_balance_by_date",
                        lambda *a, **k: pd.DataFrame())   # no balance available
    fetch = krx_source.krx_short_fetch("2024-01-01", "2024-01-31",
                                       cache=False, data_dir=tmp_path)
    df = load_short_selling("005930", fetch=fetch)
    assert df["short_balance"].isna().all()
    assert df["short_value"].isna().all()
    # core volume-based fields still present + ratio computed
    assert list(df["short_volume"]) == [100.0, 200.0, 300.0]
    assert df["short_ratio"].iloc[0] == 0.1


def test_krx_short_fetch_empty_volume_raises(fake_krx, monkeypatch, tmp_path):
    monkeypatch.setattr(fake_krx, "get_shorting_volume_by_date",
                        lambda *a, **k: pd.DataFrame())   # login-gated -> empty
    fetch = krx_source.krx_short_fetch("2024-01-01", "2024-01-31",
                                       cache=False, data_dir=tmp_path)
    with pytest.raises(ValueError, match="KRX short-selling volume"):
        load_short_selling("005930", fetch=fetch)


# --------------------------------------------------------------------------- #
# KRX login support (reads KRX_ID / KRX_PW from settings/.env, no network)
# --------------------------------------------------------------------------- #
def test_settings_has_krx_login(monkeypatch):
    monkeypatch.setattr(SETTINGS, "krx_id", "", raising=False)
    monkeypatch.setattr(SETTINGS, "krx_pw", "", raising=False)
    assert SETTINGS.has_krx_login() is False
    monkeypatch.setattr(SETTINGS, "krx_id", "someid", raising=False)
    monkeypatch.setattr(SETTINGS, "krx_pw", "somepw", raising=False)
    assert SETTINGS.has_krx_login() is True


def test_ensure_krx_login_exports_env(monkeypatch):
    monkeypatch.setattr(SETTINGS, "krx_id", "someid", raising=False)
    monkeypatch.setattr(SETTINGS, "krx_pw", "somepw", raising=False)
    monkeypatch.delenv("KRX_ID", raising=False)
    monkeypatch.delenv("KRX_PW", raising=False)
    assert krx_source.ensure_krx_login() is True
    # pykrx reads these from the environment to auto-login.
    assert os.environ["KRX_ID"] == "someid"
    assert os.environ["KRX_PW"] == "somepw"


def test_ensure_krx_login_false_without_creds(monkeypatch):
    monkeypatch.setattr(SETTINGS, "krx_id", "", raising=False)
    monkeypatch.setattr(SETTINGS, "krx_pw", "", raising=False)
    monkeypatch.delenv("KRX_ID", raising=False)
    monkeypatch.delenv("KRX_PW", raising=False)
    assert krx_source.ensure_krx_login() is False


# --------------------------------------------------------------------------- #
# yearly chunking for the long-range 공매도 fetch
# --------------------------------------------------------------------------- #
def test_fetch_chunked_concats_yearly():
    calls = []

    def fake_fn(a, b, ticker):
        calls.append((a, b))
        yr = a[:4]
        idx = pd.to_datetime([f"{yr}-06-01"]); idx.name = "날짜"
        return pd.DataFrame({"x": [int(yr)]}, index=idx)

    out = krx_source._fetch_chunked(fake_fn, "005930", "2016-01-01", "2018-12-31")
    assert len(out) == 3                                   # 2016, 2017, 2018
    assert [a[:4] for a, _b in calls] == ["2016", "2017", "2018"]
    assert list(out["x"]) == [2016, 2017, 2018]           # sorted ascending


def test_fetch_chunked_skips_failing_chunk():
    def fake_fn(a, b, ticker):
        if a.startswith("2017"):
            raise KeyError("output")                      # a bad year shouldn't abort
        idx = pd.to_datetime([f"{a[:4]}-06-01"]); idx.name = "날짜"
        return pd.DataFrame({"x": [1]}, index=idx)

    out = krx_source._fetch_chunked(fake_fn, "005930", "2016-01-01", "2018-12-31")
    assert len(out) == 2                                   # 2016 + 2018, 2017 skipped


def test_fetch_chunked_empty_returns_empty():
    out = krx_source._fetch_chunked(lambda *a: pd.DataFrame(), "X", "2016-01-01", "2017-12-31")
    assert out.empty
