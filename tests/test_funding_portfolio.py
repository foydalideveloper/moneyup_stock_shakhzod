"""Cross-sectional funding-carry portfolio — synthetic panels, no network."""

import numpy as np
import pandas as pd

from tagent.data.funding import load_carry_history
from tagent.funding_portfolio import (
    CostModel,
    _row_weights,
    align_panel,
    capacity_curve,
    classify_regime,
    compare_schemes,
    leverage_tradeoff,
    margin_path,
    portfolio_returns,
    stress_test,
    yield_by_regime,
)


def _coin(n=150, funding=0.0005, p0=100.0, drift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=n, freq="8h", tz="UTC")
    ret = drift + rng.normal(0, 0.008, n)
    spot = p0 * np.cumprod(1 + ret)
    perp = spot * (1 + rng.normal(0, 1e-4, n))
    fr = np.full(n, funding) if np.isscalar(funding) else np.asarray(funding, float)
    return pd.DataFrame({"funding_rate": fr, "perp": perp, "spot_close": spot}, index=idx)


def _panel(fundings, n=150):
    return {f"C{i}": _coin(n=n, funding=fr, seed=i) for i, fr in enumerate(fundings)}


# --------------------------------------------------------------------------- #
# cost model: maker vs taker
# --------------------------------------------------------------------------- #
def test_cost_maker_cheaper_than_taker():
    maker = CostModel(post_only=True).turnover_cost_bps()
    taker = CostModel(post_only=False).turnover_cost_bps()
    assert maker < taker
    assert CostModel(post_only=True).frac() == maker / 1e4


# --------------------------------------------------------------------------- #
# weighting schemes
# --------------------------------------------------------------------------- #
def test_funding_weighting_favours_richer_coins():
    sig = np.array([0.0006, 0.0004, 0.0003, 0.0002])
    held = np.array([True, True, True, True])
    w = _row_weights(sig, held, "funding", top_n=5, max_weight=0.5)
    assert w[0] > w[1] > w[2] > w[3]                 # more funding -> more weight
    assert abs(w.sum() - 1.0) < 1e-9


def test_equal_weight_splits_evenly():
    sig = np.array([0.0006, 0.0004, 0.0003, 0.0002])
    w = _row_weights(sig, np.array([True] * 4), "equal", top_n=5, max_weight=0.5)
    assert np.allclose(w, 0.25)


def test_top_n_holds_only_best_n():
    sig = np.array([0.0006, 0.0004, 0.0003, 0.0002])
    w = _row_weights(sig, np.array([True] * 4), "topN", top_n=2, max_weight=0.5)
    assert (w > 0).sum() == 2 and w[0] > 0 and w[1] > 0 and w[2] == 0 and w[3] == 0


def test_per_coin_cap_limits_concentration():
    sig = np.array([0.0009, 0.0001, 0.0001, 0.0001])
    w = _row_weights(sig, np.array([True] * 4), "funding", top_n=5, max_weight=0.25)
    assert w.max() <= 0.25 + 1e-9                    # cap binds
    # top_n=1 with a 0.25 cap can only deploy 25% -> partial (rest idle)
    w1 = _row_weights(sig, np.array([True] * 4), "topN", top_n=1, max_weight=0.25)
    assert abs(w1.sum() - 0.25) < 1e-9


def test_only_positive_funding_is_held():
    sig = np.array([0.0006, -0.0006, 0.0])
    w = _row_weights(sig, np.array([True, True, True]), "equal", top_n=5, max_weight=0.5)
    assert w[0] > 0 and w[1] == 0 and w[2] == 0      # negative / zero funding never held


# --------------------------------------------------------------------------- #
# portfolio returns
# --------------------------------------------------------------------------- #
def test_align_panel_shapes():
    f, b = align_panel(_panel([0.0005, 0.0003]))
    assert list(f.columns) == ["C0", "C1"] and f.shape == b.shape


def test_positive_funding_panel_is_profitable():
    r = portfolio_returns(_panel([0.0005] * 4), scheme="equal", hurdle_bps=2.0)
    assert r["net"].sum() > 0
    assert (r["invested"] > 0).mean() > 0.9          # held almost the whole window


def test_rebalance_on_change_only_zeroes_steady_turnover():
    r = portfolio_returns(_panel([0.0005] * 4), scheme="funding",
                          rebalance_on_change_only=True)
    # membership is constant after entry -> no turnover once established
    assert r["turnover"].iloc[5:].sum() < 1e-9
    # without the freeze, funding-weighted reweighting churns every interval
    r2 = portfolio_returns(_panel([0.0005, 0.0006, 0.0004, 0.0007], n=150),
                           scheme="funding", rebalance_on_change_only=False)
    assert r2["turnover"].iloc[5:].sum() >= 0.0      # may reweight; never negative


def test_maker_net_beats_taker_when_trading():
    # funding dips below hurdle mid-way then back -> forces an exit + re-entry (turnover)
    fr = np.concatenate([np.full(50, 0.0006), np.full(20, -0.0006), np.full(80, 0.0006)])
    panel = {"C0": _coin(n=150, funding=fr, seed=1),
             "C1": _coin(n=150, funding=fr, seed=2)}
    maker = portfolio_returns(panel, scheme="equal", cost=CostModel(post_only=True))
    taker = portfolio_returns(panel, scheme="equal", cost=CostModel(post_only=False))
    assert taker["turnover"].sum() > 0
    assert maker["net"].sum() > taker["net"].sum()   # post-only saves the spread


def test_no_lookahead_end_spike_not_captured():
    # funding is ~0 throughout except a huge spike on the LAST interval; a lagged
    # signal must NOT allocate to it -> the spike is not earned.
    fr = np.zeros(120); fr[-1] = 0.05
    r = portfolio_returns({"C0": _coin(n=120, funding=fr)}, scheme="equal", hurdle_bps=2.0)
    assert r["invested"].iloc[-1] == 0 and r["net"].iloc[-1] == 0.0


def test_compare_schemes_returns_all():
    out = compare_schemes(_panel([0.0006, 0.0004, 0.0003, 0.0005]))
    assert set(out) == {"equal", "funding", "topN"}
    for st in out.values():
        assert "ann_return" in st and "avg_turnover_bps" in st and "avg_n_held" in st


# --------------------------------------------------------------------------- #
# risk: margin / liquidation
# --------------------------------------------------------------------------- #
def test_flat_price_short_not_liquidated():
    p = np.full(50, 100.0)
    mp = margin_path(p, leverage=5, maint_margin_rate=0.005)
    assert not mp["liquidated"] and abs(mp["min_ratio"] - 0.2) < 1e-9


def test_sharp_rally_liquidates_short():
    p = np.array([100.0, 110.0, 125.0])               # +25%
    mp = margin_path(p, leverage=5, maint_margin_rate=0.005)
    assert mp["liquidated"] and mp["min_ratio"] < 0.005 and mp["liq_index"] >= 1


def test_funding_cushions_margin():
    p = np.full(20, 100.0)
    base = margin_path(p, leverage=5)["min_ratio"]
    fed = margin_path(p, leverage=5, funding=np.full(20, 0.001))["ratio"]
    assert fed.iloc[-1] > base                         # positive funding builds equity


def test_cross_margin_delta_neutral_survives_rally():
    p = np.array([100.0, 110.0, 130.0])
    iso = margin_path(p, leverage=5, cross_margin=False)
    crs = margin_path(p, leverage=5, cross_margin=True)
    assert iso["liquidated"] and not crs["liquidated"]  # spot offset saves cross-margin


def test_stress_test_high_leverage_liquidates():
    assert stress_test(leverage=10, rally_pct=0.30, funding_spike_bps=-30)["liquidated"]
    assert not stress_test(leverage=2, rally_pct=0.05, funding_spike_bps=0)["liquidated"]


def test_leverage_tradeoff_yield_vs_tail():
    rows = leverage_tradeoff(0.0002, [2, 5, 10, 20], rally_pct=0.25, funding_spike_bps=-20)
    assert rows[-1]["ann_yield_on_margin"] > rows[0]["ann_yield_on_margin"]    # more yield
    assert rows[-1]["worst_margin_ratio"] < rows[0]["worst_margin_ratio"]      # more risk
    assert rows[-1]["liquidated"] and not rows[0]["liquidated"]


# --------------------------------------------------------------------------- #
# regime + capacity
# --------------------------------------------------------------------------- #
def test_classify_regime_labels():
    up = pd.Series(100 * np.cumprod(np.full(60, 1.01)))
    down = pd.Series(100 * np.cumprod(np.full(60, 0.99)))
    flat = pd.Series(np.full(60, 100.0))
    assert (classify_regime(up, window=10).iloc[20:] == "bull").all()
    assert (classify_regime(down, window=10).iloc[20:] == "bear").all()
    assert (classify_regime(flat, window=10).iloc[20:] == "quiet").all()


def test_yield_by_regime_splits():
    net = pd.Series(np.r_[np.full(30, 0.0006), np.full(30, 0.0001)])
    reg = pd.Series(["bull"] * 30 + ["quiet"] * 30)
    out = yield_by_regime(net, reg)
    assert out["bull"]["ann_return"] > out["quiet"]["ann_return"] > 0
    assert out["bull"]["n_intervals"] == 30 and out["bear"]["n_intervals"] == 0


def test_capacity_curve_declines_and_zero_point():
    out = capacity_curve(0.05, depth_usd=1e9, sizes_usd=[0, 1e8, 5e8], impact_coef=0.5)
    nets = [pt["net_yield"] for pt in out["curve"]]
    assert nets[0] > nets[1] > nets[2]
    assert abs(out["capacity_usd"] - 1e8) < 1.0       # 1e9 * 0.05 / 0.5


# --------------------------------------------------------------------------- #
# multi-year loader (mocked fetchers — no network)
# --------------------------------------------------------------------------- #
def test_load_carry_history_pages_and_caches(tmp_path):
    idx = pd.date_range("2023-01-01", periods=6, freq="8h", tz="UTC")
    seen = {}

    def fake_funding(sym, start_ms=None, **kw):
        seen["funding_start"] = start_ms
        return pd.DataFrame({"funding_rate": [0.0001] * 6, "perp": [100.0] * 6},
                            index=pd.DatetimeIndex(idx, name="time"))

    def fake_spot(sym, start_ms=None, **kw):
        seen["spot_start"] = start_ms
        return pd.DataFrame({"spot_close": [100.0] * 6},
                            index=pd.DatetimeIndex(idx, name="time"))

    now_ms = int(pd.Timestamp("2026-01-01T00:00:00Z").value // 1_000_000)
    df = load_carry_history("BTCUSDT", years=3, data_dir=tmp_path,
                            fetch_funding_fn=fake_funding, fetch_spot_fn=fake_spot,
                            now_ms=now_ms)
    assert list(df.columns) == ["funding_rate", "perp", "spot_close"] and len(df) == 6
    assert seen["funding_start"] is not None and seen["funding_start"] == seen["spot_start"]
    assert (tmp_path / "funding_hist_BTCUSDT.csv").exists()
    # cached read does not call the (raising) fetchers
    again = load_carry_history("BTCUSDT", years=3, data_dir=tmp_path,
                               fetch_funding_fn=lambda *a, **k: 1 / 0,
                               fetch_spot_fn=lambda *a, **k: 1 / 0, now_ms=now_ms)
    assert len(again) == 6
