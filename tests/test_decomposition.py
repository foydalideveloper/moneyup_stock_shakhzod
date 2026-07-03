"""Decomposition studies — filter-beta vs ranking-alpha, PEAD beta-adjustment, SSF.

Pure-function unit tests on synthetic panels (no network). Verify the regime-timed
wrapper, the momentum decomposition plumbing + splits, the beta math + no-lookahead,
and the single-stock-futures availability map.
"""

import numpy as np
import pandas as pd

from tagent.decomposition import (
    CASH_ROUND_TRIP, FUT_ROUND_TRIP, beta_of, momentum_decomposition,
    pead_beta_adjust, regime_timed,
)
from tagent.kr_universe import (
    load_ssf_available, mark_ssf_available, membership_panel, save_ssf_available,
)


# --------------------------------------------------------------------------- #
# 1) regime_timed — gate, cost, no-lookahead
# --------------------------------------------------------------------------- #
def _updown_nav(n_up=40, n_down=25, start=100.0):
    up = [start * 1.01 ** i for i in range(n_up)]
    down = [up[-1] * 0.97 ** j for j in range(1, n_down + 1)]
    idx = pd.date_range("2021-01-01", periods=n_up + n_down, freq="B")
    return pd.Series(up + down, index=idx)


def test_regime_timed_gates_downtrend_to_cash_and_is_costfree_when_zero():
    nav = _updown_nav()
    ret = pd.Series(0.005, index=nav.index)
    net = regime_timed(ret, nav, regime_ma=10, one_way_cost=0.0)
    # deep in the crash the lagged NAV is below its MA -> exposure 0 -> net 0
    assert (net.iloc[-5:].abs() < 1e-12).all()
    # in the strong uptrend (well past warmup) exposure is full -> net == ret (cost-free)
    assert abs(net.iloc[30] - 0.005) < 1e-12


def test_regime_timed_charges_switch_cost():
    nav = _updown_nav()
    ret = pd.Series(0.005, index=nav.index)
    free = regime_timed(ret, nav, regime_ma=10, one_way_cost=0.0)
    paid = regime_timed(ret, nav, regime_ma=10, one_way_cost=0.001)
    # cost only differs on exposure-switch bars; total drag = (#switches incl. entry) * one_way
    drag = (free - paid).sum()
    exp = regime_timed(ret, nav, 10, 0.0) / ret             # recover exposure (ret constant !=0)
    n_switch = exp.diff().abs().fillna(exp.iloc[0]).gt(1e-9).sum()
    assert abs(drag - n_switch * 0.001) < 1e-9 and drag > 0


def test_regime_timed_no_lookahead():
    nav = _updown_nav(40, 25)
    ret = pd.Series(0.004, index=nav.index)
    net = regime_timed(ret, nav, regime_ma=10, one_way_cost=0.0005)
    # append more (future) bars; the past prefix must not change
    nav2 = pd.concat([nav, _updown_nav(20, 0, start=float(nav.iloc[-1]))[:20].set_axis(
        pd.date_range(nav.index[-1] + pd.offsets.BDay(1), periods=20, freq="B"))])
    ret2 = pd.Series(0.004, index=nav2.index)
    net2 = regime_timed(ret2, nav2, regime_ma=10, one_way_cost=0.0005)
    assert np.allclose(net.iloc[:50].to_numpy(), net2.reindex(net.index).iloc[:50].to_numpy())


# --------------------------------------------------------------------------- #
# 2) momentum_decomposition — plumbing + delta identity + splits
# --------------------------------------------------------------------------- #
def _panel(n=520, drifts=(-0.001, 0.0, 0.0008, 0.0015, 0.0022, 0.003), seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-04", periods=n, freq="B")     # spans 2021-2022
    out = {}
    for i, dr in enumerate(drifts):
        steps = rng.normal(dr, 0.012, n)
        close = 100 * np.cumprod(1 + np.r_[0.0, steps[:-1]])
        out[f"{i:06d}"] = pd.DataFrame({"open": close, "close": close}, index=idx)  # 6-digit codes
    return out, idx


def test_momentum_decomposition_structure_and_deltas():
    from tagent.xs_momentum import XSMomConfig
    panel, idx = _panel()
    memb = membership_panel({"2021-01-04": list(panel)}, idx, symbols=list(panel))
    index_close = pd.Series(100 * np.cumprod(1 + np.full(len(idx), 0.0006)), index=idx)
    cfg = XSMomConfig(lookback=15, skip_recent=1, rebalance=3, top_q=0.34,
                      hysteresis=0.1, allow_short=False)
    d = momentum_decomposition(panel, memb, index_close, cfg=cfg, regime_ma=5,
                               ppy=252, exclude_year=2022)
    assert set(d["nets"]) == {"filtered_basket", "filtered_index", "momentum_book"}
    for k in d["nets"]:
        assert isinstance(d["nets"][k], pd.Series) and d["stats"][k]["n"] > 0
        for stat in ("cagr", "sharpe", "max_drawdown"):
            assert stat in d["stats"][k]
    # delta identity: ranking-alpha vs basket == book - basket
    rab = d["ranking_alpha_vs_basket"]
    assert abs(rab["cagr"] - (d["stats"]["momentum_book"]["cagr"]
                              - d["stats"]["filtered_basket"]["cagr"])) < 1e-12
    # by-year has both calendar years; ex-2022 split drops 2022 observations
    assert {"2021", "2022"} <= set(d["by_year"]["momentum_book"])
    assert d["ex_year"]["momentum_book"]["n"] < d["stats"]["momentum_book"]["n"]
    assert d["costs"]["cash_round_trip"] == CASH_ROUND_TRIP


# --------------------------------------------------------------------------- #
# 3) beta math + PEAD beta-adjustment + no-lookahead
# --------------------------------------------------------------------------- #
def test_beta_of_recovers_known_slope_and_guards():
    rng = np.random.default_rng(0)
    mkt = pd.Series(rng.normal(0, 0.01, 400))
    assert abs(beta_of(1.3 * mkt, mkt) - 1.3) < 1e-9          # exact linear -> slope 1.3
    assert abs(beta_of(-0.7 * mkt, mkt) + 0.7) < 1e-9
    assert beta_of(mkt.iloc[:10], mkt.iloc[:10], min_obs=30) == 0.0   # too few obs
    assert beta_of(pd.Series([0.01] * 50), pd.Series([0.0] * 50)) == 0.0  # zero mkt variance


def _beta_panel(n=200, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    steps = rng.normal(0.0004, 0.011, n)
    index = pd.Series(100 * np.cumprod(1 + np.r_[0.0, steps[:-1]]), index=idx)
    stock = 5.0 * index                                       # identical returns -> beta 1
    panel = {"AAA": pd.DataFrame({"open": stock, "close": stock}, index=idx)}
    return panel, index, idx


def test_pead_beta_adjust_removes_market_and_no_lookahead():
    panel, index, idx = _beta_panel()
    hold, ei = 10, 60
    entry = idx[ei]
    ret = float(panel["AAA"]["close"].iloc[ei + hold] / panel["AAA"]["close"].iloc[ei] - 1.0)
    trades = pd.DataFrame([{"entry": entry, "symbol": "AAA", "ret": ret, "r0": 0.01}])
    adj = pead_beta_adjust(trades, panel, index, hold=hold, beta_window=30)
    row = adj.iloc[0]
    exp_mkt = float(index.iloc[ei + hold] / index.iloc[ei] - 1.0)
    assert abs(row["beta"] - 1.0) < 1e-9                      # stock == scaled index
    assert abs(row["mkt"] - exp_mkt) < 1e-12
    assert abs(row["abnormal"]) < 1e-9                        # ret was pure market -> ~0 alpha
    # no-lookahead: appending bars AFTER the exit doesn't change beta/mkt/abnormal
    ext_idx = pd.date_range(idx[-1] + pd.offsets.BDay(1), periods=40, freq="B")
    ext = pd.Series(float(index.iloc[-1]) * np.cumprod(1 + np.full(40, 0.003)), index=ext_idx)
    index2 = pd.concat([index, ext])
    stock2 = 5.0 * index2
    panel2 = {"AAA": pd.DataFrame({"open": stock2, "close": stock2}, index=index2.index)}
    adj2 = pead_beta_adjust(trades, panel2, index2, hold=hold, beta_window=30).iloc[0]
    assert abs(adj2["beta"] - row["beta"]) < 1e-12 and abs(adj2["abnormal"] - row["abnormal"]) < 1e-12


# --------------------------------------------------------------------------- #
# 4) SSF availability
# --------------------------------------------------------------------------- #
def test_ssf_available_mark_and_roundtrip(tmp_path):
    df = mark_ssf_available(["005930", "000660", "111111"], {"005930", "000660"})
    got = dict(zip(df["ticker"], df["ssf_available"]))
    assert got == {"005930": True, "000660": True, "111111": False}
    save_ssf_available(["005930", "111111"], {"005930"}, data_dir=tmp_path)
    loaded = load_ssf_available(data_dir=tmp_path)
    assert loaded == {"005930": True, "111111": False}
