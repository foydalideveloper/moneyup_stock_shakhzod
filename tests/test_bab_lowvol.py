"""Beta-neutral low-vol (BAB) — SSF filter, sector balance, beta rank/neutrality, costs.
Synthetic panels, no network."""

import numpy as np
import pandas as pd

from tagent.bab_lowvol import (
    bab_book, bab_net, calendar_time_t, realized_beta, rolling_beta, ssf_liquid_universe,
)
from tagent.kr_universe import membership_panel


# --------------------------------------------------------------------------- #
# 1) SSF-liquid filter
# --------------------------------------------------------------------------- #
def test_ssf_liquid_universe_filters_to_ssf_and_data():
    syms = ["000001", "000002", "000003", "000004"]
    ssf = {"000001": True, "000002": True, "000003": False, "000004": True}
    sectors = {"000001": "bank", "000002": "semi", "000004": "auto"}
    have = ["000001", "000002", "000003"]                # 000004 SSF but no price data
    u = ssf_liquid_universe(syms, ssf, sectors, have)
    assert set(u) == {"000001", "000002"}                # SSF AND has-data only
    assert u["000001"] == "bank"


# --------------------------------------------------------------------------- #
# fixture: names with KNOWN betas to the market, in 2 sectors
# --------------------------------------------------------------------------- #
def _panel(n=160, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    mkt_r = rng.normal(0.0003, 0.01, n)
    market = pd.Series(100 * np.cumprod(1 + np.r_[0.0, mkt_r[:-1]]), index=idx)
    # 2 sectors x 6 names; betas span low..high within each sector
    betas = {"A": [0.4, 0.6, 0.9, 1.1, 1.4, 1.6], "B": [0.5, 0.7, 1.0, 1.2, 1.5, 1.7]}
    panel, universe, k = {}, {}, 0
    for sec, bs in betas.items():
        for b in bs:
            code = f"{k:06d}"
            r = b * mkt_r + rng.normal(0.0, 0.002, n)
            panel[code] = pd.DataFrame({"open": 100 * np.cumprod(1 + np.r_[0.0, r[:-1]]),
                                        "close": 100 * np.cumprod(1 + np.r_[0.0, r[:-1]])}, index=idx)
            universe[code] = sec
            k += 1
    memb = membership_panel({"2018-01-01": list(panel)}, idx, symbols=list(panel))
    return panel, market, universe, memb, idx


# --------------------------------------------------------------------------- #
# 2) beta ranking recovers known betas
# --------------------------------------------------------------------------- #
def test_rolling_beta_recovers_known_beta():
    panel, market, universe, memb, idx = _panel()
    beta = rolling_beta(panel, market, window=40)
    last = beta.iloc[-1].dropna()
    # name 000000 has beta ~0.4, name 000005 ~1.6 -> ordered and near the true values
    assert last["000000"] < last["000005"]
    assert abs(last["000000"] - 0.4) < 0.2 and abs(last["000005"] - 1.6) < 0.25


# --------------------------------------------------------------------------- #
# 3) sector-balanced + beta-neutral leverage (each leg beta -> 1, net ~0)
# --------------------------------------------------------------------------- #
def test_bab_book_sector_neutral_and_beta_neutral():
    panel, market, universe, memb, idx = _panel()
    beta = rolling_beta(panel, market, window=40)
    W, meta = bab_book(panel, beta, universe, membership=memb, beta_window=40, q=0.33)
    assert len(meta) > 0
    # each leg vol-levered to beta 1: lev_L*beta_L ~ 1 and lev_H*beta_H ~ 1 (uncapped here)
    assert abs((meta["lev_L"] * meta["beta_L"]).mean() - 1.0) < 1e-6
    assert abs((meta["lev_H"] * meta["beta_H"]).mean() - 1.0) < 1e-6
    # sector-neutral: long-leg and short-leg sector weights are identical at a rebalance
    d = meta.index[-1]
    w = W.loc[d]
    lw = {s: 0.0 for s in set(universe.values())}
    sw = {s: 0.0 for s in set(universe.values())}
    for n in W.columns:
        if w[n] > 0:
            lw[universe[n]] += w[n]
        elif w[n] < 0:
            sw[universe[n]] += -w[n]
    lt, st = sum(lw.values()), sum(sw.values())
    for s in lw:
        assert abs(lw[s] / lt - sw[s] / st) < 1e-9         # identical sector split both legs
    # realized beta of the net book ~ 0 (the whole point)
    net = bab_net(panel, W)
    assert abs(realized_beta(net, market)) < 0.35


# --------------------------------------------------------------------------- #
# 4) monthly no-lookahead
# --------------------------------------------------------------------------- #
def test_bab_monthly_no_lookahead():
    panel, market, universe, memb, idx = _panel()
    beta = rolling_beta(panel, market, window=40)
    W, _ = bab_book(panel, beta, universe, membership=memb, beta_window=40)
    # weights are piecewise-constant within a month (only change at rebalance)
    changed = (W.diff().abs().sum(axis=1) > 1e-12)
    assert changed.sum() < len(W) / 10                     # few change-days (monthly)
    # appending future bars doesn't change past weights
    ext_idx = pd.date_range(idx[-1] + pd.offsets.BDay(1), periods=40, freq="B")
    rng = np.random.default_rng(3)
    ext = {c: pd.DataFrame({"open": 100 * np.cumprod(1 + rng.normal(0, 0.01, 40)),
                            "close": 100 * np.cumprod(1 + rng.normal(0, 0.01, 40))}, index=ext_idx)
           for c in panel}
    panel2 = {c: pd.concat([panel[c], ext[c]]) for c in panel}
    market2 = pd.concat([market, pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, 40)), index=ext_idx)])
    memb2 = membership_panel({"2018-01-01": list(panel2)}, panel2["000000"].index, symbols=list(panel2))
    beta2 = rolling_beta(panel2, market2, window=40)
    W2, _ = bab_book(panel2, beta2, universe, membership=memb2, beta_window=40)
    common = W.index[60:]                                   # past prefix (post-warmup)
    assert np.allclose(W.loc[common].to_numpy(), W2.loc[common].to_numpy(), atol=1e-9)


# --------------------------------------------------------------------------- #
# 5) all-in costs: higher cost -> lower net; borrow hits the short notional; calendar t
# --------------------------------------------------------------------------- #
def test_costs_and_calendar_t():
    panel, market, universe, memb, idx = _panel()
    beta = rolling_beta(panel, market, window=40)
    W, _ = bab_book(panel, beta, universe, membership=memb, beta_window=40)
    base = bab_net(panel, W)
    pricey = bab_net(panel, W, tx_per_side=0.005, roll_annual=0.05, borrow_annual=0.10)
    assert pricey.mean() < base.mean()                     # all-in costs drag the spread
    # borrow applies only to the short notional (positive shorts -> positive borrow drag)
    no_borrow = bab_net(panel, W, borrow_annual=0.0)
    assert no_borrow.mean() >= base.mean() - 1e-12
    t, n = calendar_time_t(base, lag=21)
    assert n == len(base.dropna()) and np.isfinite(t)
