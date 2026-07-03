"""Long-history index calibration + ranking-alpha satellite tests (synthetic, no network)."""

import numpy as np
import pandas as pd

from tagent.index_calibration import (
    by_decade, cap_weighted_basket, filtered_index_net, load_index_close,
    momentum_book_weights, sector_exposure, spanning_regression, sub_periods,
    turnover_cost_itemization,
)
from tagent.kr_universe import membership_panel


# --------------------------------------------------------------------------- #
# 1) long-history loader + by-decade / sub-period splits
# --------------------------------------------------------------------------- #
def test_load_index_close_and_by_decade(tmp_path):
    idx = pd.date_range("1995-01-02", "2012-12-31", freq="B")
    close = pd.Series(100 * np.cumprod(1 + np.full(len(idx), 0.0003)), index=idx)
    pd.DataFrame({"timestamp": idx, "open": close.values, "high": close.values,
                  "low": close.values, "close": close.values, "volume": 1}).to_csv(
        tmp_path / "xx_1d.csv", index=False)
    got = load_index_close("xx", data_dir=tmp_path)
    assert len(got) == len(idx) and abs(got.iloc[-1] - close.iloc[-1]) < 1e-6
    ret = got.pct_change(fill_method=None).dropna()
    dec = by_decade(ret)
    assert {"1990s", "2000s", "2010s"} <= set(dec) and all(dec[k]["n"] > 0 for k in dec)
    sp = sub_periods(ret, {"win": ("2000-01-01", "2002-12-31")})
    assert sp["win"]["n"] > 0


# --------------------------------------------------------------------------- #
# 2) filtered index — gate, roll friction
# --------------------------------------------------------------------------- #
def test_filtered_index_gate_and_roll_friction():
    up = [100 * 1.004 ** i for i in range(60)]
    down = [up[-1] * 0.97 ** j for j in range(1, 31)]
    idx = pd.date_range("2001-01-01", periods=90, freq="B")
    close = pd.Series(up + down, index=idx)
    no_roll = filtered_index_net(close, regime_ma=10, roll_annual=0.0, switch_cost=0.0, ppy=252)
    roll = filtered_index_net(close, regime_ma=10, roll_annual=0.0252, switch_cost=0.0, ppy=252)
    # strong-uptrend bar: exposure 1 -> net == fwd (no roll) and fwd - daily_roll (with roll)
    fwd = close.pct_change(fill_method=None).shift(-1)
    assert abs(no_roll.iloc[40] - fwd.iloc[40]) < 1e-12
    assert abs((no_roll.iloc[40] - roll.iloc[40]) - 0.0252 / 252) < 1e-12   # one day's roll drag
    # deep downtrend: exposure 0 -> net 0 (no roll charged while in cash)
    assert (roll.iloc[-5:].abs() < 1e-12).all()


# --------------------------------------------------------------------------- #
# 3) cap-weighted basket — weights proportional to market cap
# --------------------------------------------------------------------------- #
def test_cap_weighted_basket_weights_by_cap():
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    panel = {
        "000001": pd.DataFrame({"close": [100, 110, 110, 110, 110]}, index=idx),  # big mover up
        "000002": pd.DataFrame({"close": [100, 100, 100, 100, 100]}, index=idx),
    }
    shares = {"000001": 1_000.0, "000002": 100.0}            # name1 ~10x the cap of name2
    memb = membership_panel({"2020-01-01": ["000001", "000002"]}, idx,
                            symbols=["000001", "000002"])
    cw = cap_weighted_basket(panel, memb, shares)
    # bar 0: caps 100*1000=100000 vs 100*100=10000 -> w1=0.909; fwd1=+10%, fwd2=0 -> ~+9.09%
    assert abs(cw.iloc[0] - (100_000 / 110_000) * 0.10) < 1e-6


# --------------------------------------------------------------------------- #
# 4) sector-capped momentum — no sector exceeds the cap
# --------------------------------------------------------------------------- #
def _sector_panel(seed=0):
    rng = np.random.default_rng(seed)
    n = 80
    idx = pd.date_range("2021-01-04", periods=n, freq="B")
    names, sectors, drifts = [], {}, {}
    # sector A heavily over-represented (10 names) + B..E (5 each); A has the best momentum
    plan = [("A", 10, 0.004), ("B", 5, 0.001), ("C", 5, 0.0005), ("D", 5, 0.0), ("E", 5, -0.001)]
    k = 0
    for sec, cnt, dr in plan:
        for _ in range(cnt):
            code = f"{k:06d}"
            steps = rng.normal(dr, 0.008, n)
            names.append(code)
            sectors[code] = sec
            drifts[code] = 100 * np.cumprod(1 + np.r_[0.0, steps[:-1]])
            k += 1
    close = pd.DataFrame(drifts, index=idx)
    memb = membership_panel({"2021-01-04": names}, idx, symbols=names)
    return close, memb, sectors


def test_sector_cap_limits_concentration():
    close, memb, sectors = _sector_panel()
    W_plain = momentum_book_weights(close, memb, sectors, top_q=0.5, rebalance=5,
                                    skip=2, lookback=10, sector_cap=None)
    W_cap = momentum_book_weights(close, memb, sectors, top_q=0.5, rebalance=5,
                                  skip=2, lookback=10, sector_cap=0.30)
    max_plain = sector_exposure(W_plain, sectors).replace(0, np.nan).max()
    max_cap = sector_exposure(W_cap, sectors).replace(0, np.nan).max()
    assert max_plain > 0.45                                  # plain piles into sector A
    assert max_cap <= 0.30 + 1e-9                            # cap holds
    assert abs(W_cap.iloc[-1].sum() - 1.0) < 1e-9           # still fully invested


# --------------------------------------------------------------------------- #
# 5) spanning regression — recover alpha + beta with a NW t and CI
# --------------------------------------------------------------------------- #
def test_spanning_regression_recovers_alpha_beta():
    rng = np.random.default_rng(1)
    months = pd.date_range("2018-01-31", periods=72, freq="ME")
    basket = pd.Series(rng.normal(0.0, 0.04, len(months)), index=months)
    book = 0.003 + 0.8 * basket + pd.Series(rng.normal(0.0, 0.001, len(months)), index=months)
    sr = spanning_regression(book, basket, nw_lag=6)
    assert abs(sr["alpha_m"] - 0.003) < 0.002 and abs(sr["beta"] - 0.8) < 0.05
    assert sr["n_months"] == len(months) and sr["t_alpha"] > 2.0
    lo, hi = sr["ci_m"]
    assert lo < sr["alpha_m"] < hi and abs(sr["alpha_ann"] - sr["alpha_m"] * 12) < 1e-12


# --------------------------------------------------------------------------- #
# 6) turnover / instrument-cost itemization
# --------------------------------------------------------------------------- #
def test_turnover_cost_itemization():
    turn = pd.Series([2.0] * 5)                              # sum 10 over 2 years -> 5x/yr two-way
    item = turnover_cost_itemization(turn, n_years=2.0, cash_rt=0.0020, fut_rt=0.0005)
    assert abs(item["turnover_twoway_annual"] - 5.0) < 1e-9
    assert abs(item["turnover_oneway_annual"] - 2.5) < 1e-9
    assert abs(item["cash_cost_annual"] - 5.0 * 0.0010) < 1e-12     # one-way 0.10%
    assert abs(item["fut_cost_annual"] - 5.0 * 0.00025) < 1e-12
    assert abs(item["instrument_cost_diff_annual"] - (0.005 - 0.00125)) < 1e-12
