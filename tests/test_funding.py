"""Cash-and-carry funding study — synthetic series, no network."""

import numpy as np
import pandas as pd

from tagent.data.funding import fetch_funding, fetch_spot_8h, load_carry, merge_carry
from tagent.funding_study import (
    basket,
    carry_returns,
    carry_stats,
    funding_flips,
)


def _carry_df(n=120, funding=0.0001, spot0=100.0, drift=0.0, seed=0):
    """Synthetic 8h carry frame: constant funding, spot & perp track (+small basis)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=n, freq="8h", tz="UTC")
    ret = drift + rng.normal(0, 0.01, n)
    spot = spot0 * np.cumprod(1 + ret)
    perp = spot * (1 + rng.normal(0, 1e-4, n))          # perp ~ spot, tiny basis noise
    fr = np.full(n, funding) if np.isscalar(funding) else np.asarray(funding)
    return pd.DataFrame({"funding_rate": fr, "perp": perp, "spot_close": spot}, index=idx)


# --------------------------------------------------------------------------- #
# carry calc
# --------------------------------------------------------------------------- #
def test_positive_funding_yields_positive_carry():
    df = _carry_df(n=200, funding=0.0001)               # +1 bp / 8h
    r = carry_returns(df, fee_bps=7, spread_bps=1, min_funding_bps=0)
    assert r["in_carry"].iloc[-1] == 1
    # Held the whole time -> net ~ funding minus a one-off entry cost. Positive.
    assert r["net"].sum() > 0
    st = carry_stats(r["net"], r["in_carry"])
    assert st["ann_return"] > 0 and st["in_carry_pct"] > 90


def test_funding_received_equals_rate_when_flat_prices():
    # Flat prices, no basis noise -> gross per interval == funding rate exactly.
    n = 50
    idx = pd.date_range("2026-01-01", periods=n, freq="8h", tz="UTC")
    df = pd.DataFrame({"funding_rate": 0.0002, "perp": 100.0, "spot_close": 100.0}, index=idx)
    r = carry_returns(df, fee_bps=0, spread_bps=0, min_funding_bps=0)
    # after the first (entry) interval, gross each interval == funding == 0.0002
    assert np.allclose(r["gross"].iloc[1:], 0.0002)


# --------------------------------------------------------------------------- #
# fee handling
# --------------------------------------------------------------------------- #
def test_higher_fees_reduce_net():
    df = _carry_df(n=200, funding=0.0001)
    low = carry_returns(df, fee_bps=2, spread_bps=0).net.sum()
    high = carry_returns(df, fee_bps=20, spread_bps=5).net.sum()
    assert high < low


def test_churn_costs_more_than_holding():
    # Alternating funding around the hurdle with NO hysteresis -> churn -> more cost.
    n = 120
    alt = np.where(np.arange(n) % 2 == 0, 0.0003, -0.0003)
    churn = carry_returns(_carry_df(n=n, funding=alt), fee_bps=10, spread_bps=2,
                          min_funding_bps=0, lookback=1, hysteresis_bps=0)
    steady = carry_returns(_carry_df(n=n, funding=0.0002), fee_bps=10, spread_bps=2,
                           min_funding_bps=0, lookback=1, hysteresis_bps=0)
    assert churn["cost"].sum() > steady["cost"].sum()


def test_hysteresis_prevents_churn():
    # Same alternating funding, but a hysteresis band holds the position -> far
    # fewer transitions (and lower cost) than the no-hysteresis case.
    n = 120
    alt = np.where(np.arange(n) % 2 == 0, 0.0003, -0.0001)   # dips but not below -3bps
    no_h = carry_returns(_carry_df(n=n, funding=alt), min_funding_bps=0, lookback=1,
                         hysteresis_bps=0)
    with_h = carry_returns(_carry_df(n=n, funding=alt), min_funding_bps=0, lookback=1,
                           hysteresis_bps=3)
    assert with_h["in_carry"].diff().abs().sum() < no_h["in_carry"].diff().abs().sum()


# --------------------------------------------------------------------------- #
# sign logic + no lookahead
# --------------------------------------------------------------------------- #
def test_sustained_negative_funding_stays_flat():
    df = _carry_df(n=100, funding=-0.0002)              # perp longs get paid -> short pays
    r = carry_returns(df, min_funding_bps=0, lookback=3)
    # hurdle 0 and funding negative -> not in carry (avoid paying funding)
    assert r["in_carry"].sum() == 0
    assert np.allclose(r["net"], 0.0)


def test_decision_uses_past_funding_only():
    # Funding jumps positive only at the last interval; the lagged signal means
    # we are NOT yet in carry on that interval (no lookahead onto the new rate).
    n = 30
    fr = np.zeros(n); fr[-1] = 0.01
    df = _carry_df(n=n, funding=fr)
    r = carry_returns(df, min_funding_bps=5, lookback=1)   # hurdle 5bps
    assert r["in_carry"].iloc[-1] == 0      # signal is the *previous* (zero) funding


def test_negative_funding_while_forced_in_loses():
    # If we ARE in the carry (hurdle below the negative rate) and funding is
    # negative, the short PAYS -> negative gross.
    df = _carry_df(n=60, funding=-0.0002)
    r = carry_returns(df, min_funding_bps=-100, lookback=1)   # always in
    assert r["in_carry"].iloc[-1] == 1
    assert r["gross"].iloc[5] < 0


# --------------------------------------------------------------------------- #
# stats + flips + basket
# --------------------------------------------------------------------------- #
def test_stats_fields_and_drawdown_sign():
    df = _carry_df(n=300, funding=0.0001)
    r = carry_returns(df)
    st = carry_stats(r["net"], r["in_carry"])
    assert set(st) >= {"ann_return", "ann_vol", "sharpe", "max_drawdown", "n_intervals"}
    assert st["max_drawdown"] <= 0.0 and st["n_intervals"] == 300


def test_funding_flips_detects_negatives():
    fr = np.full(50, 0.0001); fr[10:15] = -0.0005       # a 5-interval negative streak
    df = _carry_df(n=50, funding=fr)
    fl = funding_flips(df, n=3)
    assert fl["longest_neg_streak"] == 5
    assert fl["min_funding_bps"] < 0
    assert fl["neg_pct"] > 0 and len(fl["worst_events"]) == 3


def test_basket_aggregates_coins():
    a = carry_returns(_carry_df(n=100, funding=0.0001, seed=1))
    b = carry_returns(_carry_df(n=100, funding=0.0002, seed=2))
    out = basket({"A": a, "B": b})
    assert len(out["net"]) == 100
    assert out["stats"]["n_intervals"] > 0


# --------------------------------------------------------------------------- #
# data loader (mocked HTTP — no network)
# --------------------------------------------------------------------------- #
def test_merge_carry_aligns_on_8h_boundary():
    t0 = pd.Timestamp("2026-01-01T00:00:00Z")
    fund = pd.DataFrame({"funding_rate": [0.0001, 0.0002], "perp": [100.0, 101.0]},
                        index=pd.DatetimeIndex([t0, t0 + pd.Timedelta("8h")], name="time"))
    spot = pd.DataFrame({"spot_close": [99.9, 100.8]},
                        index=pd.DatetimeIndex([t0, t0 + pd.Timedelta("8h")], name="time"))
    m = merge_carry(fund, spot)
    assert list(m.columns) == ["funding_rate", "perp", "spot_close"]
    assert len(m) == 2 and m["spot_close"].iloc[0] == 99.9


def test_load_carry_with_injected_fetchers_and_cache(tmp_path):
    idx = pd.date_range("2026-01-01", periods=4, freq="8h", tz="UTC")

    def fake_funding(sym, limit=1000):
        return pd.DataFrame({"funding_rate": [0.0001] * 4, "perp": [100.0] * 4},
                            index=pd.DatetimeIndex(idx, name="time"))

    def fake_spot(sym, limit=1000):
        return pd.DataFrame({"spot_close": [100.0] * 4},
                            index=pd.DatetimeIndex(idx, name="time"))

    df = load_carry("BTCUSDT", data_dir=tmp_path,
                    fetch_funding_fn=fake_funding, fetch_spot_fn=fake_spot)
    assert list(df.columns) == ["funding_rate", "perp", "spot_close"] and len(df) == 4
    assert (tmp_path / "funding_BTCUSDT.csv").exists()
    # second call reads cache (fetchers that would raise are not called)
    df2 = load_carry("BTCUSDT", data_dir=tmp_path,
                     fetch_funding_fn=lambda *a, **k: (_ for _ in ()).throw(AssertionError),
                     fetch_spot_fn=lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    assert len(df2) == 4
