"""Point-in-time KR universe — survivorship correction. Synthetic, no network.

A fake pykrx ``stock`` is injected so reconstruction is exercised without KRX. The
core invariants: no future listings leak in, delisted names are kept while they
were members, membership is strictly as-of (no lookahead), and the engine's
optional ``membership`` mask actually gates selection + the basket benchmark.
"""

import numpy as np
import pandas as pd

from tagent.kr_universe import (
    load_members, membership_panel, pit_members, save_members, top_caps_asof,
    top_caps_on, universe_symbols,
)
from tagent.xs_momentum import XSMomConfig, _weights, backtest


class FakeStock:
    """Stand-in for pykrx.stock: as-of market-cap snapshots keyed by YYYYMMDD."""

    def __init__(self, caps):
        self.caps = caps                       # {yyyymmdd: {ticker: market_cap}}

    def get_market_cap_by_ticker(self, date, market):
        snap = self.caps.get(date)
        if snap is None:
            return pd.DataFrame()
        return pd.DataFrame({"종가": 1, "시가총액": list(snap.values()), "거래량": 1},
                            index=list(snap.keys()))


def _coin(prices, start="2016-01-04"):
    idx = pd.date_range(start, periods=len(prices), freq="B")
    return pd.DataFrame({"close": np.asarray(prices, float)}, index=idx)


# --------------------------------------------------------------------------- #
# as-of-date reconstruction
# --------------------------------------------------------------------------- #
def test_top_caps_on_ranks_by_market_cap_and_caps_at_n():
    stock = FakeStock({"20200102": {"000001": 30, "000002": 50, "000003": 10, "000004": 40}})
    top = top_caps_on("2020-01-02", top_n=2, market="KOSPI", stock=stock)
    assert top == ["000002", "000004"]                 # largest two, in cap order


def test_top_caps_on_drops_zero_caps_on_holiday():
    # KRX returns all-zero market caps on a non-trading day -> must come back EMPTY,
    # not a ticker-ordered garbage "top-N".
    stock = FakeStock({"20200101": {"000001": 0, "000002": 0, "000003": 0}})
    assert top_caps_on("2020-01-01", top_n=5, market="KOSPI", stock=stock) == []


def test_top_caps_asof_walks_forward_to_first_trading_day():
    stock = FakeStock({"20200101": {"000001": 0, "000002": 0},        # holiday (zeros)
                       "20200102": {"000002": 50, "000001": 30}})     # next day real
    got = top_caps_asof("2020-01-01", top_n=5, markets=("KOSPI",), stock=stock)
    assert got == ["000001", "000002"]                                # taken from 01-02


def test_pit_members_skips_empty_snapshots():
    stock = FakeStock({"20200101": {"000002": 50, "000001": 30}})   # only Jan has data
    members = pit_members(["2020-01-01", "2020-02-01"], top_n=5, stock=stock)
    assert set(members) == {"2020-01-01"}              # empty Feb snapshot dropped
    assert members["2020-01-01"] == ["000001", "000002"]


# --------------------------------------------------------------------------- #
# point-in-time membership: no future listings, delisted kept while member
# --------------------------------------------------------------------------- #
def _two_snapshots():
    # A is large in Jan then GONE (delisted/left); C only LISTED from Feb.
    return {"2020-01-01": ["A", "B"], "2020-02-01": ["B", "C"]}

def test_membership_no_future_listing_and_delisted_handling():
    idx = pd.to_datetime(["2019-12-31", "2020-01-15", "2020-02-15"])
    panel = membership_panel(_two_snapshots(), idx, symbols=["A", "B", "C"])
    # before any snapshot -> nobody is a member
    assert not panel.loc["2019-12-31"].any()
    # January: A and B members; C is NOT (not yet "listed") -> no future leak
    assert bool(panel.loc["2020-01-15", "A"]) and bool(panel.loc["2020-01-15", "B"])
    assert not bool(panel.loc["2020-01-15", "C"])
    # February: A has dropped out (delisted/left), C is now in
    assert not bool(panel.loc["2020-02-15", "A"])
    assert bool(panel.loc["2020-02-15", "B"]) and bool(panel.loc["2020-02-15", "C"])
    # the delisted name is still in the downloadable union (so its losses are seen)
    assert "A" in universe_symbols(_two_snapshots())


def test_membership_is_asof_no_lookahead():
    idx = pd.date_range("2020-01-01", "2020-03-31", freq="D")
    base = membership_panel(_two_snapshots(), idx, symbols=["A", "B", "C"])
    # appending a FUTURE snapshot must not change any past day's membership
    ext = dict(_two_snapshots(), **{"2020-03-01": ["C", "D"]})
    after = membership_panel(ext, idx, symbols=["A", "B", "C", "D"])
    common = ["A", "B", "C"]
    pre_march = idx[idx < "2020-03-01"]
    assert base.loc[pre_march, common].equals(after.loc[pre_march, common])
    # and D (only known from March) is never a member before March
    assert not after.loc[pre_march, "D"].any()


# --------------------------------------------------------------------------- #
# the engine honours the membership mask (selection + basket)
# --------------------------------------------------------------------------- #
def _panel4():
    rng = np.random.default_rng(0)
    out = {}
    for i, drift in enumerate([0.003, 0.002, 0.001, 0.0]):
        steps = rng.normal(drift, 0.02, 400)
        out[f"S{i}"] = _coin(100 * np.cumprod(1 + np.r_[0.0, steps[:-1]]))
    return out

def test_membership_mask_excludes_non_members_from_selection():
    panel = _panel4()
    idx = next(iter(panel.values())).index
    # only S2,S3 are ever eligible — even though S0,S1 have the strongest momentum
    memb = pd.DataFrame(False, index=idx, columns=["S0", "S1", "S2", "S3"])
    memb[["S2", "S3"]] = True
    cfg = XSMomConfig(lookback=60, skip_recent=0, rebalance=21, top_q=0.5,
                      allow_short=False, cost_bps=0, slippage_bps=0)
    w = _weights(pd.DataFrame({c: d["close"] for c, d in panel.items()}), cfg, membership=memb)
    assert np.allclose(w[["S0", "S1"]].to_numpy(), 0.0)     # non-members never held
    assert w[["S2", "S3"]].to_numpy().sum() > 0             # members are


def test_membership_basket_is_over_members_only():
    panel = _panel4()
    idx = next(iter(panel.values())).index
    memb = pd.DataFrame(True, index=idx, columns=["S0", "S1", "S2", "S3"])
    memb[["S2", "S3"]] = False                              # basket = mean of S0,S1 only
    cfg = XSMomConfig(lookback=60, rebalance=21, allow_short=False,
                      cost_bps=0, slippage_bps=0)
    r = backtest(panel, cfg, periods_per_year=252, membership=memb)
    wide = pd.DataFrame({c: d["close"] for c, d in panel.items()})
    fwd = wide.pct_change().shift(-1)
    expect = fwd[["S0", "S1"]].mean(axis=1).reindex(r["basket"].index)
    assert np.allclose(r["basket"].to_numpy(), expect.to_numpy(), equal_nan=True)


def test_backtest_membership_default_none_is_unchanged():
    panel = _panel4()
    cfg = XSMomConfig(lookback=60, rebalance=21, allow_short=False)
    a = backtest(panel, cfg, periods_per_year=252)
    b = backtest(panel, cfg, periods_per_year=252, membership=None)
    assert np.allclose(a["net"].to_numpy(), b["net"].to_numpy())   # additive: no behaviour change


# --------------------------------------------------------------------------- #
# cache round-trip
# --------------------------------------------------------------------------- #
def test_members_cache_roundtrip(tmp_path):
    m = {"2020-01-01": ["000660", "005930"], "2020-02-01": ["005930", "035420"]}
    save_members(m, data_dir=tmp_path)
    back = load_members(data_dir=tmp_path)
    assert back == m
    assert load_members(data_dir=tmp_path / "missing") == {}
