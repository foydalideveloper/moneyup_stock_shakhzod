"""EODHD intraday minute client — mocked HTTP, no network, no secrets."""

import datetime as dt

import pandas as pd
import pytest

from tagent.data.intraday_history import load_intraday
from tagent.data.eodhd_source import (
    EodhdError, bars_to_df, build_request, fetch_minutes, parse_intraday, save_minutes,
)


class FakeResp:
    def __init__(self, data, status=200):
        self._data = data
        self.status_code = status

    def json(self):
        return self._data


class FakeGet:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": params})
        return self.responses[len(self.calls) - 1]


def _unix(y, mo, d, h, mi):
    return int(dt.datetime(y, mo, d, h, mi, tzinfo=dt.timezone.utc).timestamp())


def _bar(ts, o, h, l, c, v):
    return {"timestamp": ts, "gmtoffset": 0,
            "datetime": dt.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S"),
            "open": o, "high": h, "low": l, "close": c, "volume": v}


# --------------------------------------------------------------------------- #
# request building
# --------------------------------------------------------------------------- #
def test_build_request_adds_exchange_and_params():
    url, params = build_request("005930", exchange="KO", interval="1m",
                                from_ts=100, to_ts=200, api_token="TOK")
    assert url.endswith("/api/intraday/005930.KO")
    assert params["api_token"] == "TOK" and params["interval"] == "1m" and params["fmt"] == "json"
    assert params["from"] == 100 and params["to"] == 200


def test_build_request_keeps_symbol_with_exchange():
    url, _ = build_request("035420.KO", api_token="T")
    assert url.endswith("/api/intraday/035420.KO")        # not doubled


def test_build_request_rejects_bad_interval():
    with pytest.raises(ValueError):
        build_request("005930", interval="2m", api_token="T")


# --------------------------------------------------------------------------- #
# parsing (UTC -> KST naive)
# --------------------------------------------------------------------------- #
def test_parse_intraday_fields_and_kst():
    rows = parse_intraday([_bar(_unix(2024, 6, 24, 0, 30), 100, 102, 99, 101, 1000)])
    assert len(rows) == 1
    r = rows[0]
    assert (r["open"], r["high"], r["low"], r["close"], r["volume"]) == (100, 102, 99, 101, 1000)
    assert r["timestamp"] == pd.Timestamp("2024-06-24 09:30:00")     # +9h KST naive


def test_parse_intraday_skips_null_close_and_raises_on_error_envelope():
    rows = parse_intraday([_bar(_unix(2024, 6, 24, 0, 30), 1, 1, 1, 1, 1),
                           {"timestamp": _unix(2024, 6, 24, 0, 31), "close": None}])
    assert len(rows) == 1                                  # null-close empty minute skipped
    with pytest.raises(EodhdError) as e:
        parse_intraday({"message": "Invalid API key"})    # error envelope is a dict
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
    assert len(df) == 2 and df["close"].tolist() == [1.0, 2.0]


# --------------------------------------------------------------------------- #
# pagination + HTTP errors + CSV round-trip
# --------------------------------------------------------------------------- #
def test_fetch_minutes_pages_older_windows():
    w1 = FakeResp([_bar(_unix(2024, 6, 24, 0, 31), 100, 101, 99, 100, 10),
                   _bar(_unix(2024, 6, 24, 0, 30), 100, 101, 99, 100, 10)])
    w2 = FakeResp([])                                      # empty -> history exhausted
    sess = FakeGet([w1, w2])
    df, info = fetch_minutes("005930", api_token="TOK", session=sess,
                             end_ts=_unix(2024, 6, 24, 1, 0), max_windows=5)
    assert info["windows"] == 2 and info["rows"] == 2
    assert sess.calls[0]["params"]["api_token"] == "TOK"  # token query-only
    assert sess.calls[1]["params"]["to"] < sess.calls[0]["params"]["to"]   # paged older


def test_fetch_minutes_raises_on_http_error():
    sess = FakeGet([FakeResp({"message": "Forbidden: upgrade plan"}, status=403)])
    with pytest.raises(EodhdError) as e:
        fetch_minutes("005930", api_token="T", session=sess, max_windows=1)
    assert "403" in str(e.value)


def test_saved_csv_round_trips_through_loader(tmp_path):
    data = [_bar(_unix(2024, 6, 24, 0, 30), 100, 102, 99, 101, 10),
            _bar(_unix(2024, 6, 24, 0, 31), 101, 103, 100, 102, 20)]
    df, _ = fetch_minutes("005930", api_token="T", session=FakeGet([FakeResp(data), FakeResp([])]),
                          end_ts=_unix(2024, 6, 24, 1, 0), max_windows=2)
    path = save_minutes(df, "005930", data_dir=tmp_path)
    assert path.name == "005930_1m.csv"
    loaded = load_intraday("005930", data_dir=tmp_path)
    assert list(loaded.columns) == ["open", "high", "low", "close", "volume"]
    assert len(loaded) == 2 and loaded["close"].tolist() == [101.0, 102.0]
