"""Kiwoom ka10080 minute-bar fetcher — mocked HTTP, no network, no secrets."""

import pandas as pd
import pytest

from tagent.data.intraday_history import load_intraday
from tagent.data.kiwoom_minute import (
    KiwoomMinuteError, bars_to_df, build_minute_request, fetch_minute_bars,
    parse_minute_bars, save_minutes,
)

BASE = "https://mockapi.kiwoom.com"


def _bar(tm, o, h, l, c, v):
    return {"cntr_tm": tm, "open_pric": o, "high_pric": h, "low_pric": l,
            "cur_prc": c, "trde_qty": v}


class FakeResp:
    def __init__(self, data, headers=None):
        self._data = data
        self.headers = headers or {}
        self.status_code = 200

    def json(self):
        return self._data


class FakeSession:
    """Returns queued responses for successive /chart POSTs, recording each call."""
    def __init__(self, pages):
        self.pages = list(pages)
        self.calls = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json})
        return self.pages[len(self.calls) - 1]


class FakeAuth:
    def get_token(self):
        return "TOK"                                   # never a real secret


# --------------------------------------------------------------------------- #
# request building
# --------------------------------------------------------------------------- #
def test_build_minute_request_shape():
    url, h, b = build_minute_request(BASE, "TOK", "005930", interval=5,
                                     cont_yn="Y", next_key="K1")
    assert url == f"{BASE}/api/dostk/chart"
    assert h["api-id"] == "ka10080" and h["authorization"] == "Bearer TOK"
    assert h["cont-yn"] == "Y" and h["next-key"] == "K1"
    assert b == {"stk_cd": "005930", "tic_scope": "5", "upd_stkpc_tp": "1"}


def test_build_minute_request_rejects_bad_interval():
    with pytest.raises(ValueError):
        build_minute_request(BASE, "TOK", "005930", interval=2)


def test_build_minute_request_optional_base_date():
    _, _, b = build_minute_request(BASE, "TOK", "005930", interval=1, base_date="20240624")
    assert b["date"] == "20240624"


# --------------------------------------------------------------------------- #
# response parsing
# --------------------------------------------------------------------------- #
def test_parse_minute_bars_ohlcv_and_sign_stripping():
    data = {"return_code": 0, "stk_min_pole_chart_qry": [
        _bar("20240624093000", "+74000", "+74200", "-73900", "+74100", "120"),
        _bar("20240624093100", "74100", "74500", "74050", "74400", "80")]}
    rows = parse_minute_bars(data)
    assert len(rows) == 2
    r0 = rows[0]
    assert r0["open"] == 74000 and r0["high"] == 74200 and r0["low"] == 73900   # sign stripped
    assert r0["close"] == 74100 and r0["volume"] == 120
    assert r0["timestamp"] == pd.Timestamp("2024-06-24 09:30:00")


def test_parse_minute_bars_alias_array_key_and_bad_rows():
    data = {"chart": [                                   # wrapper alias for the list
        _bar("20240624093000", "100", "101", "99", "100", "5"),
        {"cntr_tm": "", "cur_prc": "100"}]}              # no valid time -> skipped
    rows = parse_minute_bars(data)
    assert len(rows) == 1 and rows[0]["close"] == 100


def test_bars_to_df_schema_sorted_dedup_zero_dropped():
    rows = [
        {"timestamp": pd.Timestamp("2024-06-24 09:31"), "open": 2, "high": 2, "low": 2, "close": 2, "volume": 1},
        {"timestamp": pd.Timestamp("2024-06-24 09:30"), "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
        {"timestamp": pd.Timestamp("2024-06-24 09:32"), "open": 0, "high": 0, "low": 0, "close": 0, "volume": 0},
    ]
    df = bars_to_df(rows)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.name == "timestamp" and df.index.is_monotonic_increasing
    assert len(df) == 2                                  # zero-close (non-trading) bar dropped
    assert df["close"].tolist() == [1.0, 2.0]


# --------------------------------------------------------------------------- #
# pagination / continuation
# --------------------------------------------------------------------------- #
def test_fetch_minute_bars_follows_continuation():
    p1 = FakeResp({"return_code": 0, "stk_min_pole_chart_qry": [
        _bar("20240624093000", "100", "101", "99", "100", "10")]},
        headers={"cont-yn": "Y", "next-key": "K1"})
    p2 = FakeResp({"return_code": 0, "stk_min_pole_chart_qry": [
        _bar("20240624093100", "100", "102", "100", "101", "20")]},
        headers={"cont-yn": "N", "next-key": ""})
    sess = FakeSession([p1, p2])
    df, info = fetch_minute_bars("005930", 1, auth=FakeAuth(), session=sess)
    assert info["pages"] == 2 and info["rows"] == 2
    # second call carried the continuation cursor from page 1's headers
    assert sess.calls[1]["headers"]["cont-yn"] == "Y"
    assert sess.calls[1]["headers"]["next-key"] == "K1"
    assert df.index.min() == pd.Timestamp("2024-06-24 09:30:00")
    assert info["days"] == 1


def test_fetch_minute_bars_stops_when_cont_no():
    p1 = FakeResp({"return_code": 0, "stk_min_pole_chart_qry": [
        _bar("20240624093000", "100", "101", "99", "100", "10")]},
        headers={"cont-yn": "N"})
    sess = FakeSession([p1])
    _, info = fetch_minute_bars("005930", 1, auth=FakeAuth(), session=sess)
    assert info["pages"] == 1 and len(sess.calls) == 1


def test_fetch_minute_bars_raises_on_error_code():
    bad = FakeResp({"return_code": 8030, "return_msg": "모의/실투 mismatch"})
    with pytest.raises(KiwoomMinuteError) as e:
        fetch_minute_bars("005930", 1, auth=FakeAuth(), session=FakeSession([bad]))
    assert "8030" in str(e.value)


# --------------------------------------------------------------------------- #
# CSV schema feeds the backtester loader
# --------------------------------------------------------------------------- #
def test_saved_csv_loads_back_through_intraday_history(tmp_path):
    p1 = FakeResp({"return_code": 0, "stk_min_pole_chart_qry": [
        _bar("20240624093000", "100", "101", "99", "100", "10"),
        _bar("20240624093100", "100", "102", "100", "101", "20")]},
        headers={"cont-yn": "N"})
    df, _ = fetch_minute_bars("005930", 1, auth=FakeAuth(), session=FakeSession([p1]))
    path = save_minutes(df, "005930", interval=1, data_dir=tmp_path)
    assert path.name == "005930_1m.csv"
    loaded = load_intraday("005930", data_dir=tmp_path)      # the backtester's loader
    assert list(loaded.columns) == ["open", "high", "low", "close", "volume"]
    assert len(loaded) == 2 and loaded.index.name == "timestamp"
    assert loaded["close"].tolist() == [100.0, 101.0]
