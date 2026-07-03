"""Foreign-futures-flow trial — flow z-score/signal, no-lookahead, costs, decay, source parse.
Synthetic data + injected session, no network."""

import numpy as np
import pandas as pd

from tagent.data.krx_deriv_investor import _net_for_day, fetch_kospi200_futures_flow
from tagent.futures_flow import (
    by_year, calendar_time_t, decay_split, flow_backtest, flow_signal, flow_zscore,
    is_degenerate,
)


# --------------------------------------------------------------------------- #
# 1) flow z-score + long/flat/short signal + next-day no-lookahead
# --------------------------------------------------------------------------- #
def _flow(n=160, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    return pd.Series(rng.normal(0.0, 1.0, n), index=idx)


def test_flow_signal_long_flat_short():
    f = _flow()
    f.iloc[-1] = 50.0                                      # big recent foreign buy -> high z
    sig = flow_signal(f, z_enter=1.0, lookback=3, z_window=20)
    assert sig.iloc[-1] == 1.0                             # strong inflow -> LONG
    g = _flow(seed=1)
    g.iloc[-1] = -50.0
    assert flow_signal(g, z_enter=1.0, lookback=3, z_window=20).iloc[-1] == -1.0   # outflow -> SHORT
    assert set(sig.dropna().unique()) <= {-1.0, 0.0, 1.0}


def test_backtest_no_lookahead_and_position_lagged():
    f = _flow()
    px = pd.Series(100 * np.cumprod(1 + np.r_[0.0, np.random.default_rng(2).normal(0, 0.01, len(f) - 1)]),
                   index=f.index)
    net = flow_backtest(f, px, lookback=3, z_window=20)
    # append future flow + price; the past net must not change (signal lagged, z trailing)
    ext_idx = pd.date_range(f.index[-1] + pd.offsets.BDay(1), periods=30, freq="B")
    rng = np.random.default_rng(5)
    f2 = pd.concat([f, pd.Series(rng.normal(0, 1, 30), index=ext_idx)])
    px2 = pd.concat([px, pd.Series(float(px.iloc[-1]) * np.cumprod(1 + rng.normal(0, 0.01, 30)), index=ext_idx)])
    net2 = flow_backtest(f2, px2, lookback=3, z_window=20)
    common = net.index[:-2]
    assert np.allclose(net.reindex(common).to_numpy(), net2.reindex(common).to_numpy())


# --------------------------------------------------------------------------- #
# 2) futures cost / roll: higher cost -> lower net; flips pay turnover
# --------------------------------------------------------------------------- #
def test_costs_drag_net():
    f = _flow(seed=3)
    px = pd.Series(100 * np.cumprod(1 + np.r_[0.0, np.random.default_rng(4).normal(0, 0.012, len(f) - 1)]),
                   index=f.index)
    cheap = flow_backtest(f, px, cost_round_trip=0.0005, lookback=3, z_window=20)
    pricey = flow_backtest(f, px, cost_round_trip=0.0040, lookback=3, z_window=20)
    assert pricey.sum() < cheap.sum()
    noroll = flow_backtest(f, px, roll_annual=0.0, lookback=3, z_window=20)
    assert noroll.sum() >= cheap.sum() - 1e-9


# --------------------------------------------------------------------------- #
# 3) calendar-time t + decay split + degenerate guard
# --------------------------------------------------------------------------- #
def test_calendar_t_decay_and_degenerate():
    f = _flow(seed=6)
    px = pd.Series(100 * np.cumprod(1 + np.r_[0.0, np.random.default_rng(7).normal(0, 0.01, len(f) - 1)]),
                   index=f.index)
    net = flow_backtest(f, px, lookback=3, z_window=20)
    t, n = calendar_time_t(net)
    assert n == len(net.dropna()) and np.isfinite(t)
    dec = decay_split(net)
    assert "early" in dec and "late" in dec and dec["mid"] is not None
    by = by_year(net)
    assert all("sharpe" in v for v in by.values())
    # degenerate guard: empty / all-zero flow has no tradable signal
    assert is_degenerate(pd.Series([0.0, 0.0, 0.0]))
    assert is_degenerate(pd.Series(dtype=float))
    assert not is_degenerate(pd.Series([1.0, -2.0, 0.0]))


# --------------------------------------------------------------------------- #
# 4) KRX derivatives-investor source parsing (injected session, no network)
# --------------------------------------------------------------------------- #
class _Resp:
    def __init__(self, j):
        self._j = j

    def json(self):
        return self._j


class _Session:
    """Returns canned MDC output keyed by endDd; records that posts happened."""
    def __init__(self, by_day):
        self.by_day = by_day

    def post(self, url, data=None, headers=None, timeout=None):
        return _Resp({"output": self.by_day.get(data["endDd"], [])})


def _mdc_row(name, netval):
    return {"INVST_TP_NM": name, "NETBID_TRDVAL": netval, "NETBID_TRDVOL": "0"}


def test_deriv_investor_parse_and_fetch():
    day = "20240105"
    out = [_mdc_row("외국인", "1,234,000"), _mdc_row("기관합계", "-500"), _mdc_row("개인", "999")]
    rec = _net_for_day(_Session({day: out}), day, "KRDRVFUK2I")
    assert rec["foreign_net"] == 1_234_000.0 and rec["inst_net"] == -500.0   # commas stripped
    # fetch over an injected calendar -> daily frame with foreign_net/inst_net
    cal = pd.to_datetime(["2024-01-05", "2024-01-08"])
    sess = _Session({"20240105": out, "20240108": [_mdc_row("외국인", "10"), _mdc_row("기관합계", "20")]})
    df = fetch_kospi200_futures_flow("2024-01-05", "2024-01-08", session=sess, calendar=cal)
    assert list(df.columns) == ["foreign_net", "inst_net"] and len(df) == 2
    assert df.loc["2024-01-08", "foreign_net"] == 10.0
