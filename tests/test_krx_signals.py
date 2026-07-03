"""Gated KR-signal loader tests — fully mocked pykrx, no network."""

import numpy as np
import pandas as pd
import pytest

from tagent.data import krx_source
from tagent.data.krx_signals import (
    SIGNALS,
    ManualCsvRequired,
    krx_signal_fetch,
    load_signal,
)

SUPPORTED = ["investor_detail", "short_balance"]
MANUAL = ["lending", "margin", "program"]


# --------------------------------------------------------------------------- #
# fake pykrx (Korean columns, like the authenticated endpoints return)
# --------------------------------------------------------------------------- #
class FakeStock:
    def __init__(self, empty=False):
        self.empty = empty
        self.calls = []

    def _idx(self, n=3):
        idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"][:n])
        idx.name = "날짜"
        return idx

    def get_market_trading_value_by_date(self, f, t, ticker, on="순매수", detail=False):
        self.calls.append(("investor", f, t, ticker, on, detail))
        if self.empty:
            return pd.DataFrame()
        idx = self._idx()
        return pd.DataFrame({
            "금융투자": [1e9, -2e9, 3e9], "보험": [1e7, 2e7, -1e7],
            "투신": [5e6, -5e6, 1e6], "사모": [2e6, 2e6, -2e6],
            "은행": [0, 0, 0], "기타금융": [0, 0, 0],
            "연기금": [4e8, -3e8, 5e8], "기타법인": [0, 0, 0],
            "개인": [-1e9, 1e9, -2e9], "외국인": [3e8, 1e8, -4e8],
            "기타외국인": [0, 0, 0], "전체": [0, 0, 0],
        }, index=idx)

    def get_shorting_balance_by_date(self, f, t, ticker):
        self.calls.append(("balance", f, t, ticker))
        if self.empty:
            return pd.DataFrame()
        idx = self._idx()
        return pd.DataFrame({
            "공매도잔고": [3196343, 3230073, 3100000], "상장주식수": [5.97e9] * 3,
            "공매도금액": [2.54e11, 2.49e11, 2.40e11], "시가총액": [4.75e14] * 3,
            "비중": [0.05, 0.05, 0.052],
        }, index=idx)


@pytest.fixture
def fake_krx(monkeypatch):
    fake = FakeStock()
    monkeypatch.setattr(krx_source, "_import_pykrx", lambda: fake)
    return fake


# --------------------------------------------------------------------------- #
# supported signals: real pykrx -> canonical schema
# --------------------------------------------------------------------------- #
def test_investor_detail_maps_to_canonical(fake_krx, tmp_path):
    fetch = krx_signal_fetch("investor_detail", "2024-01-01", "2024-01-31",
                             cache=False, data_dir=tmp_path)
    df = load_signal("investor_detail", "005930", fetch=fetch)
    assert list(df.columns) == SIGNALS["investor_detail"].value_cols
    assert list(df["pension_net"]) == [4e8, -3e8, 5e8]      # 연기금
    assert list(df["fininv_net"]) == [1e9, -2e9, 3e9]       # 금융투자
    # detail=True was requested.
    assert fake_krx.calls[0][5] is True


def test_short_balance_maps_to_canonical(fake_krx, tmp_path):
    fetch = krx_signal_fetch("short_balance", "2024-01-01", "2024-01-31",
                             cache=False, data_dir=tmp_path)
    df = load_signal("short_balance", "005930", fetch=fetch)
    assert list(df.columns) == SIGNALS["short_balance"].value_cols
    assert list(df["short_balance"]) == [3196343, 3230073, 3100000]   # 공매도잔고
    assert df["short_balance_ratio"].iloc[0] == 0.05                  # 비중


@pytest.mark.parametrize("sig", SUPPORTED)
def test_supported_caches_and_round_trips(fake_krx, tmp_path, sig):
    fetch = krx_signal_fetch(sig, "2024-01-01", "2024-01-31", cache=True, data_dir=tmp_path)
    df1 = load_signal(sig, "005930", fetch=fetch)
    assert (tmp_path / f"005930_{sig}.csv").exists()
    fake_krx.calls.clear()
    df2 = load_signal(sig, "005930", fetch=fetch)     # served from cache
    assert fake_krx.calls == []
    pd.testing.assert_frame_equal(df1, df2)


@pytest.mark.parametrize("sig", SUPPORTED)
def test_supported_gated_empty_raises(monkeypatch, tmp_path, sig):
    monkeypatch.setattr(krx_source, "_import_pykrx", lambda: FakeStock(empty=True))
    fetch = krx_signal_fetch(sig, "2024-01-01", "2024-01-31", cache=False, data_dir=tmp_path)
    with pytest.raises(ValueError):
        load_signal(sig, "005930", fetch=fetch)


# --------------------------------------------------------------------------- #
# manual signals: clear ManualCsvRequired, not a silent failure
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("sig", MANUAL)
def test_manual_signal_raises_with_guidance(tmp_path, sig):
    fetch = krx_signal_fetch(sig, "2024-01-01", "2024-01-31", cache=False, data_dir=tmp_path)
    with pytest.raises(ManualCsvRequired) as exc:
        fetch("005930")
    msg = str(exc.value)
    assert "data.krx.co.kr" in msg and "Columns" in msg
    assert SIGNALS[sig].name.split()[0] in msg or sig in msg


def test_manual_lending_csv_cp949_maps(tmp_path):
    # KRX-style manual export (cp949, Korean headers) for 대차잔고.
    path = tmp_path / "005930_lending.csv"
    path.write_bytes(
        ("일자,대차잔고수량,대차잔고금액\n"
         "2024/01/03,1000,50000000\n"
         "2024/01/02,1200,60000000\n").encode("cp949"))
    df = load_signal("lending", "005930", data_dir=tmp_path)
    assert list(df.columns) == ["lending_balance", "lending_value"]
    assert list(df["lending_balance"]) == [1200, 1000]      # sorted ascending by date
    assert isinstance(df.index, pd.DatetimeIndex)


def test_manual_program_csv_maps(tmp_path):
    path = tmp_path / "005930_program.csv"
    path.write_text("date,program_net,program_net_vol\n"
                    "2024-01-01,1000000,500\n2024-01-02,-2000000,-800\n", encoding="utf-8")
    df = load_signal("program", "005930", data_dir=tmp_path)
    assert list(df["program_net"]) == [1000000, -2000000]


def test_missing_manual_csv_is_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_signal("margin", "005930", data_dir=tmp_path)


def test_registry_has_five_signals():
    assert set(SIGNALS) == {"investor_detail", "short_balance", "lending",
                            "margin", "program"}
    assert sum(s.pykrx_supported for s in SIGNALS.values()) == 2
