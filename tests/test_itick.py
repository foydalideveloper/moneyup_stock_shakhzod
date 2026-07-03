"""iTick minute-bar client — mocked HTTP, no network, no secrets."""

import datetime as dt

import pandas as pd
import pytest

from tagent.data.intraday_history import load_intraday
from tagent.data.itick_source import (
    ItickError, bars_to_df, build_request, fetch_klines, parse_klines, save_minutes,
)


class FakeResp:
    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data


class FakeGet:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "params": params})
        return FakeResp(self.responses[len(self.calls) - 1])


def _ms(y, mo, d, h, mi):
    return int(dt.datetime(y, mo, d, h, mi, tzinfo=dt.timezone.utc).timestamp() * 1000)


def _bar(t, o, h, l, c, v):
    return {"t": t, "o": o, "h": h, "l": l, "c": c, "v": v, "tu": o * v}


# --------------------------------------------------------------------------- #
# request building
# --------------------------------------------------------------------------- #
def test_build_request_shape():
    url, headers, params = build_request("005930", region="KR", interval=1, limit=500, token="TOK")
    assert url.endswith("/stock/kline")
    assert headers["token"] == "TOK" and headers["accept"] == "application/json"
    assert params == {"region": "KR", "code": "005930", "kType": 1, "limit": 500}


def test_build_request_kType_mapping_and_et():
    _, _, p5 = build_request("X", interval=5, token="T", et=123)
    assert p5["kType"] == 2 and p5["et"] == 123
    assert build_request("X", interval=60, token="T")[2]["kType"] == 5


def test_build_request_rejects_bad_interval():
    with pytest.raises(ValueError):
        build_request("X", interval=2, token="T")


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def test_parse_klines_fields_and_kst():
    data = {"code": 0, "msg": None, "data": [
        _bar(_ms(2024, 6, 24, 0, 30), 100, 102, 99, 101, 1000)]}
    rows = parse_klines(data)
    assert len(rows) == 1
    r = rows[0]
    assert (r["open"], r["high"], r["low"], r["close"], r["volume"]) == (100, 102, 99, 101, 1000)
    assert r["timestamp"] == pd.Timestamp("2024-06-24 09:30:00")     # UTC -> KST naive


def test_parse_klines_raises_on_error_code():
    with pytest.raises(ItickError) as e:
        parse_klines({"code": 40001, "msg": "region not supported", "data": None})
    assert "40001" in str(e.value)


def test_parse_klines_raises_on_auth_error_envelope():
    # a 401-style body has no "code"/"data", just a message -> must surface, not look empty
    with pytest.raises(ItickError) as e:
        parse_klines({"message": "Invalid API key in request"})
    assert "Invalid API key" in str(e.value)


def test_bars_to_df_schema_sorted_dedup_zero_dropped():
    rows = [
        {"timestamp": pd.Timestamp("2024-06-24 09:31"), "open": 2, "high": 2, "low": 2, "close": 2, "volume": 1},
        {"timestamp": pd.Timestamp("2024-06-24 09:30"), "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
        {"timestamp": pd.Timestamp("2024-06-24 09:32"), "open": 0, "high": 0, "low": 0, "close": 0, "volume": 0},
    ]
    df = bars_to_df(rows)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.name == "timestamp" and df.index.is_monotonic_increasing
    assert len(df) == 2 and df["close"].tolist() == [1.0, 2.0]      # zero-close dropped


# --------------------------------------------------------------------------- #
# pagination + CSV round-trip into the backtester loader
# --------------------------------------------------------------------------- #
def test_fetch_klines_paginates_older_and_passes_token():
    p1 = {"code": 0, "data": [_bar(_ms(2024, 6, 24, 0, 31), 100, 101, 99, 100, 10),
                              _bar(_ms(2024, 6, 24, 0, 30), 100, 101, 99, 100, 10)]}
    p2 = {"code": 0, "data": [_bar(_ms(2024, 6, 24, 0, 29), 100, 101, 99, 100, 10)]}
    sess = FakeGet([p1, p2])
    df, info = fetch_klines("005930", token="TOK", region="KR", interval=1, limit=2,
                            max_pages=5, session=sess)
    assert info["pages"] == 2 and info["rows"] == 3
    assert sess.calls[0]["headers"]["token"] == "TOK"               # key header-only
    assert "et" in sess.calls[1]["params"]                          # 2nd page pages older


def test_saved_csv_round_trips_through_loader(tmp_path):
    data = {"code": 0, "data": [_bar(_ms(2024, 6, 24, 0, 30), 100, 102, 99, 101, 10),
                                _bar(_ms(2024, 6, 24, 0, 31), 101, 103, 100, 102, 20)]}
    df, _ = fetch_klines("005930", token="T", region="KR", limit=100,
                         session=FakeGet([data]))
    path = save_minutes(df, "005930", data_dir=tmp_path)
    assert path.name == "005930_1m.csv"
    loaded = load_intraday("005930", data_dir=tmp_path)
    assert list(loaded.columns) == ["open", "high", "low", "close", "volume"]
    assert len(loaded) == 2 and loaded.index.name == "timestamp"
    assert loaded["close"].tolist() == [101.0, 102.0]
