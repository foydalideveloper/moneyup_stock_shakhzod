"""Risk-managed momentum overlay — regime filter + vol targeting. Synthetic, no net.

Invariants: exposure is cut in a confirmed market downtrend, scaled inversely to
recent realized vol (capped at 1x, long-only), and strictly no-lookahead (future
bars never change a past exposure — the engine indexes forward returns, so signals
are lagged one bar).
"""

import numpy as np
import pandas as pd

from tagent.risk_managed import (
    RiskOverlayConfig, apply_overlay, combined_exposure, regime_exposure,
    vol_target_exposure,
)


def _idx(n, start="2016-01-04"):
    return pd.date_range(start, periods=n, freq="B")


# --------------------------------------------------------------------------- #
# 1) market-regime filter cuts exposure in a downtrend
# --------------------------------------------------------------------------- #
def test_regime_filter_cuts_exposure_in_downtrend():
    # NAV ramps up for 60 bars then falls steadily for 60 bars
    up = np.linspace(100, 200, 60)
    down = np.linspace(200, 80, 60)
    nav = pd.Series(np.r_[up, down], index=_idx(120))
    cfg = RiskOverlayConfig(regime_ma=20, regime_off=0.0)
    exp = regime_exposure(nav, cfg)
    # deep in the uptrend -> full exposure; deep in the downtrend -> to cash
    assert exp.iloc[55] == 1.0
    assert exp.iloc[-1] == 0.0
    # partial de-risk allowed (regime_off > 0) keeps that level in the downtrend
    exp2 = regime_exposure(nav, RiskOverlayConfig(regime_ma=20, regime_off=0.3))
    assert abs(exp2.iloc[-1] - 0.3) < 1e-12


def test_regime_warmup_is_full_exposure():
    nav = pd.Series(np.linspace(100, 90, 30), index=_idx(30))   # falling but no MA yet
    exp = regime_exposure(nav, RiskOverlayConfig(regime_ma=50))
    assert (exp == 1.0).all()                                   # no confirmed trend during warmup


# --------------------------------------------------------------------------- #
# 2) volatility targeting scales inversely to realized vol, capped at 1x
# --------------------------------------------------------------------------- #
def test_vol_target_scales_inversely_and_caps_at_one():
    rng = np.random.default_rng(0)
    calm = rng.normal(0, 0.005, 120)         # low vol
    wild = rng.normal(0, 0.05, 120)          # 10x vol
    net = pd.Series(np.r_[calm, wild], index=_idx(240))
    cfg = RiskOverlayConfig(vol_target=0.15, vol_window=40, vol_cap=1.0)
    exp = vol_target_exposure(net, cfg)
    # capped at 1x everywhere (long-only, no leverage)
    assert exp.max() <= 1.0 + 1e-12
    # turbulent regime gets much less exposure than the calm regime
    assert exp.iloc[-1] < 0.5
    assert exp.iloc[110] > exp.iloc[-1]


def test_vol_target_warmup_defaults_to_cap():
    net = pd.Series(np.full(10, 0.001), index=_idx(10))
    exp = vol_target_exposure(net, RiskOverlayConfig(vol_window=63))
    assert (exp == 1.0).all()                                   # not enough history -> full


# --------------------------------------------------------------------------- #
# 3) no-lookahead: future bars never change a past exposure
# --------------------------------------------------------------------------- #
def test_overlay_is_no_lookahead():
    rng = np.random.default_rng(1)
    n = 300
    net = pd.Series(rng.normal(0.001, 0.02, n), index=_idx(n))
    nav = pd.Series(100 * np.cumprod(1 + rng.normal(0.0005, 0.015, n)), index=_idx(n))
    cfg = RiskOverlayConfig(regime_ma=50, vol_window=40)
    base = combined_exposure(net, nav, cfg)
    # append arbitrary FUTURE bars
    extra = 40
    net2 = pd.concat([net, pd.Series(rng.normal(0, 0.08, extra), index=_idx(extra, "2018-01-01"))])
    nav2 = pd.concat([nav, pd.Series(nav.iloc[-1] * np.cumprod(1 + rng.normal(-0.01, 0.05, extra)),
                                     index=_idx(extra, "2018-01-01"))])
    after = combined_exposure(net2, nav2, cfg)
    assert np.allclose(base.to_numpy(), after.reindex(base.index).to_numpy(), equal_nan=True)


def test_overlay_never_adds_leverage_and_reduces_vol():
    rng = np.random.default_rng(2)
    n = 600
    net = pd.Series(rng.normal(0.0008, 0.025, n), index=_idx(n))
    nav = pd.Series(100 * np.cumprod(1 + rng.normal(0.0003, 0.02, n)), index=_idx(n))
    out = apply_overlay(net, nav, RiskOverlayConfig(vol_target=0.15, vol_window=63), mode="combined")
    assert out["exposure"].max() <= 1.0 + 1e-12                 # long-only cap respected
    # de-risking should not increase realized vol of the book
    assert out["stats"]["ann_vol"] <= out["raw_stats"]["ann_vol"] + 1e-9


def test_apply_overlay_modes_run():
    rng = np.random.default_rng(3)
    n = 300
    net = pd.Series(rng.normal(0.001, 0.02, n), index=_idx(n))
    nav = pd.Series(100 * np.cumprod(1 + rng.normal(0.0005, 0.02, n)), index=_idx(n))
    for mode in ("regime", "vol", "combined"):
        out = apply_overlay(net, nav, mode=mode)
        assert len(out["net"]) == n and "sharpe" in out["stats"]
        assert 0.0 <= out["avg_exposure"] <= 1.0 + 1e-12
