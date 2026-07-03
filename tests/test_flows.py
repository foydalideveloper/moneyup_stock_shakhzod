"""수급 (investor-flow) data layer + feature tests — fully offline (mock pykrx)."""

import numpy as np
import pandas as pd
import pytest

from tagent.data import krx_source
from tagent.data.flows_source import (
    FLOW_COLS,
    _normalize_flows,
    krx_flows_fetch,
    load_flows,
)
from tagent.features import make_features
from tagent.features_flows import (
    FLOW_FEATURE_COLS,
    flow_feature_columns,
    make_flow_features,
    merge_flow_features,
)


# --------------------------------------------------------------------------- #
# fake pykrx (Korean investor columns, as get_market_trading_value_by_date)
# --------------------------------------------------------------------------- #
class FakeStock:
    def __init__(self, empty=False):
        self.empty = empty
        self.calls = []

    def _frame(self, foreign, inst):
        idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        idx.name = "날짜"
        return pd.DataFrame({
            "기관합계": inst,
            "기타법인": [0, 0, 0],
            "개인": [-(f + i) for f, i in zip(foreign, inst)],
            "외국인합계": foreign,
            "전체": [0, 0, 0],
        }, index=idx)

    def get_market_trading_value_by_date(self, fromdate, todate, ticker, on="순매수"):
        self.calls.append(("value", fromdate, todate, ticker, on))
        if self.empty:
            return pd.DataFrame()           # gated -> empty, like KRX without login
        return self._frame([1e9, -2e9, 3e9], [-5e8, 4e8, -6e8])

    def get_market_trading_volume_by_date(self, fromdate, todate, ticker, on="순매수"):
        self.calls.append(("volume", fromdate, todate, ticker, on))
        if self.empty:
            return pd.DataFrame()
        return self._frame([100, -200, 300], [-50, 40, -60])


@pytest.fixture
def fake_krx(monkeypatch):
    fake = FakeStock()
    monkeypatch.setattr(krx_source, "_import_pykrx", lambda: fake)
    return fake


# --------------------------------------------------------------------------- #
# loader: schema mapping
# --------------------------------------------------------------------------- #
def test_krx_flows_fetch_maps_to_canonical_schema(fake_krx, tmp_path):
    fetch = krx_flows_fetch("2024-01-01", "2024-01-31", cache=False, data_dir=tmp_path)
    df = load_flows("005930", fetch=fetch)

    assert list(df.columns) == FLOW_COLS                 # [foreign_net, inst_net]
    assert list(df["foreign_net"]) == [1e9, -2e9, 3e9]   # 외국인합계
    assert list(df["inst_net"]) == [-5e8, 4e8, -6e8]     # 기관합계
    assert isinstance(df.index, pd.DatetimeIndex)
    # Dates passed to pykrx as YYYYMMDD, with on="순매수".
    assert fake_krx.calls[0][:4] == ("value", "20240101", "20240131", "005930")
    assert fake_krx.calls[0][4] == "순매수"


def test_krx_flows_fetch_volume_variant(fake_krx, tmp_path):
    fetch = krx_flows_fetch("2024-01-01", "2024-01-31", use_volume=True,
                            cache=False, data_dir=tmp_path)
    df = load_flows("005930", fetch=fetch)
    assert list(df["foreign_net"]) == [100, -200, 300]
    assert fake_krx.calls[0][0] == "volume"


def test_krx_flows_fetch_caches_and_round_trips(fake_krx, tmp_path):
    fetch = krx_flows_fetch("2024-01-01", "2024-01-31", cache=True, data_dir=tmp_path)
    df1 = load_flows("005930", fetch=fetch)
    assert (tmp_path / "005930_flows.csv").exists()

    fake_krx.calls.clear()
    df2 = load_flows("005930", fetch=fetch)   # served from cache, no pykrx call
    assert fake_krx.calls == []
    pd.testing.assert_frame_equal(df1, df2)


def test_gated_empty_response_raises_clearly(monkeypatch, tmp_path):
    gated = FakeStock(empty=True)
    monkeypatch.setattr(krx_source, "_import_pykrx", lambda: gated)
    fetch = krx_flows_fetch("2024-01-01", "2024-01-31", cache=False, data_dir=tmp_path)
    with pytest.raises(ValueError, match="gated|KRX_ID"):
        load_flows("005930", fetch=fetch)


def test_normalize_accepts_english_headers():
    raw = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02"],
        "foreign_net": [10.0, -20.0],
        "inst_net": [-5.0, 6.0],
    })
    df = _normalize_flows(raw)
    assert list(df.columns) == FLOW_COLS
    assert list(df["foreign_net"]) == [10.0, -20.0]


# --------------------------------------------------------------------------- #
# features: columns + NO lookahead
# --------------------------------------------------------------------------- #
def _flow_frame(n=30, seed=0):
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "foreign_net": rng.normal(0, 1e9, n),
        "inst_net": rng.normal(0, 5e8, n),
    }, index=idx)


def test_flow_feature_columns_present():
    idx = pd.date_range("2024-01-01", periods=30, freq="D")
    feats = make_flow_features(_flow_frame(30), idx, sum_window=5, z_window=10)
    assert list(feats.columns) == FLOW_FEATURE_COLS
    assert FLOW_FEATURE_COLS == flow_feature_columns(5)
    assert feats.index.equals(idx)


def test_flows_no_lookahead():
    """A flow spike on day D must not surface on the price bar for day D."""
    idx = pd.date_range("2024-01-01", periods=8, freq="D")
    flow = pd.DataFrame({
        "foreign_net": [1, 1, 1, 1, 999, 1, 1, 1],   # spike at idx[4]
        "inst_net": [0.0] * 8,
    }, index=idx)
    feats = make_flow_features(flow, idx, sum_window=2, z_window=3)
    spike_day, next_day = idx[4], idx[5]
    # EOD/delayed: spike value is visible only the day AFTER, never on its own bar.
    assert feats.loc[spike_day, "foreign_net"] == 1
    assert feats.loc[next_day, "foreign_net"] == 999


def test_first_bar_has_no_flow_value():
    idx = pd.date_range("2024-02-01", periods=10, freq="D")
    flow = _flow_frame(10); flow.index = idx
    feats = make_flow_features(flow, idx, sum_window=2, z_window=3)
    # No prior day available for the very first bar -> NaN (no leakage from day 0).
    assert np.isnan(feats["foreign_net"].iloc[0])
    expected_prev = flow["foreign_net"].shift(1)
    for t in idx[1:]:
        assert np.isclose(feats.loc[t, "foreign_net"], expected_prev.loc[t])


# --------------------------------------------------------------------------- #
# merge helper: additive + non-mutating
# --------------------------------------------------------------------------- #
def test_merge_flow_features_nonmutating_and_additive():
    idx = pd.date_range("2024-01-01", periods=60, freq="D")
    rng = np.random.default_rng(1)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, 60)), index=idx).clip(lower=1)
    ohlcv = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                          "close": close, "volume": rng.integers(1e6, 2e6, 60).astype(float)})
    base = make_features(ohlcv, dropna=False)
    base_cols_before = list(base.columns)

    flow = _flow_frame(60); flow.index = idx
    merged = merge_flow_features(base, flow)

    assert list(base.columns) == base_cols_before          # input not mutated
    assert len(merged) == len(base)                        # no rows added/dropped
    assert set(FLOW_FEATURE_COLS).issubset(merged.columns)
    assert len(merged.columns) == len(base.columns) + len(FLOW_FEATURE_COLS)


# --------------------------------------------------------------------------- #
# real KRX [12009] export shape: cp949 + spaced headers '기관 합계'/'외국인 합계'
# --------------------------------------------------------------------------- #
def test_krx_12009_cp949_spaced_headers(tmp_path):
    # Header exactly as data.krx.co.kr [12009] 투자자별 거래실적 daily/net export.
    header = "일자,기관 합계,기타법인,개인,외국인 합계,전체"
    rows = [
        '"2024/01/03","100","-5","-130","30","0"',   # newest-first, like KRX
        '"2024/01/02","200","10","-260","50","0"',
        '"2024/01/01","-50","0","20","30","0"',
    ]
    path = tmp_path / "005930_flows.csv"
    path.write_bytes(("\n".join([header] + rows) + "\n").encode("cp949"))

    df = load_flows("005930", data_dir=tmp_path)
    assert list(df.columns) == FLOW_COLS                       # [foreign_net, inst_net]
    assert isinstance(df.index, pd.DatetimeIndex)
    # Sorted ascending; spaced headers '기관 합계'/'외국인 합계' map despite the space.
    assert list(df["inst_net"]) == [-50.0, 200.0, 100.0]      # 기관 합계
    assert list(df["foreign_net"]) == [30.0, 50.0, 30.0]      # 외국인 합계
