"""Opening-range (B) + VWAP-reversion (C) strategies — mock data, no network."""

import numpy as np
import pandas as pd

from tagent.strategies import opening_range as orb
from tagent.strategies import vwap_reversion as vwr


def make_day(bars, date="2023-01-03", start="09:00"):
    """bars: list of (open, high, low, close, volume)."""
    idx = pd.date_range(f"{date} {start}", periods=len(bars), freq="min")
    cols = ["open", "high", "low", "close", "volume"]
    return pd.DataFrame([dict(zip(cols, b)) for b in bars], index=idx)


# --------------------------------------------------------------------------- #
# B: opening-range breakout
# --------------------------------------------------------------------------- #
def test_opening_range_high_low():
    day = make_day([(100, 101, 99, 100, 10), (100, 100, 98, 99, 10), (100, 102, 100, 101, 10),
                    (100, 100, 100, 100, 10)])
    hi, lo = orb.opening_range(day, 3)
    assert hi == 102 and lo == 98


def test_opening_range_breakout_entry():
    # OR (first 3 bars) high = 101; bar 3 stays inside, bar 4 breaks above
    day = make_day([(100, 101, 99, 100, 10), (100, 100, 99, 100, 10), (100, 101, 100, 100, 10),
                    (100, 100, 100, 100, 10), (101, 102, 101, 101, 50)])
    plan = orb.make_strategy(orb.OpeningRangeParams(open_minutes=3, take_pct=0.01, stop_pct=0.01))(day, {})
    assert plan is not None and plan.entry_index == 4
    assert plan.entry_price == 101.0 and plan.take_pct == 0.01 and plan.stop_pct == 0.01


def test_opening_range_no_breakout_returns_none():
    day = make_day([(100, 101, 99, 100, 10), (100, 100, 99, 100, 10), (100, 101, 100, 100, 10),
                    (100, 100, 99, 100, 10), (100, 100, 99, 100, 10)])     # never exceeds 101
    assert orb.make_strategy(orb.OpeningRangeParams(open_minutes=3))(day, {}) is None


def test_opening_range_volume_confirmation_gates_then_passes():
    # OR avg volume = 10; require breakout volume >= 2x (>=20). bar3 vol 12 -> skip; bar4 vol 50 -> take
    day = make_day([(100, 101, 99, 100, 10), (100, 100, 99, 100, 10), (100, 101, 100, 100, 10),
                    (101, 102, 101, 101, 12), (101, 103, 101, 102, 50)])
    p = orb.OpeningRangeParams(open_minutes=3, take_pct=0.01, stop_pct=0.01, vol_mult=2.0)
    plan = orb.make_strategy(p)(day, {})
    assert plan is not None and plan.entry_index == 4               # low-volume breakout skipped


# --------------------------------------------------------------------------- #
# C: VWAP reversion
# --------------------------------------------------------------------------- #
def test_intraday_vwap_cumulative():
    day = make_day([(10, 10, 10, 10, 1), (8, 12, 8, 10, 1), (20, 20, 20, 20, 2)])
    vwap = vwr.intraday_vwap(day)
    # tp = [10,10,20]; cum_pv=[10,20,60]; cum_v=[1,2,4]; vwap=[10,10,15]
    assert np.allclose(vwap, [10.0, 10.0, 15.0])


def _oversold_day():
    # 4 anchor bars at 100 (high volume), then a bar that drops ~1.5% below VWAP
    bars = [(100, 100, 100, 100, 100)] * 4
    bars += [(99, 99, 98, 98.5, 1), (98.5, 99, 98, 98.5, 1), (99, 100, 99, 100, 1)]
    return make_day(bars)


def test_vwap_reversion_oversold_trigger_and_target():
    day = _oversold_day()
    p = vwr.VwapReversionParams(dev_pct=0.01, take_pct=0.015, stop_pct=0.01, warmup=3)
    plan = vwr.make_strategy(p)(day, {})
    assert plan is not None and plan.entry_index == 4              # first oversold bar
    assert plan.entry_price == 98.5
    vwap = vwr.intraday_vwap(day)
    expected_take = min(0.015, vwap[4] / 98.5 - 1.0)              # exit at VWAP touch, capped at Y
    assert abs(plan.take_pct - expected_take) < 1e-9 and plan.stop_pct == 0.01


def test_vwap_reversion_no_lookahead():
    day = _oversold_day()
    p = vwr.VwapReversionParams(dev_pct=0.01, take_pct=0.015, stop_pct=0.01, warmup=3)
    full = vwr.make_strategy(p)(day, {})
    trunc = vwr.make_strategy(p)(day.iloc[:5], {})                 # only bars 0..4 (the trigger bar)
    assert full.entry_index == trunc.entry_index == 4
    assert abs(full.take_pct - trunc.take_pct) < 1e-9             # VWAP at t uses only bars <= t


def test_vwap_reversion_no_trigger_returns_none():
    day = make_day([(100, 100, 100, 100, 10)] * 8)                # flat at VWAP -> never oversold
    assert vwr.make_strategy(vwr.VwapReversionParams(dev_pct=0.01, warmup=3))(day, {}) is None


def test_default_grids_nonempty():
    assert len(orb.default_grid()) == 18 and len(vwr.default_grid()) == 18
