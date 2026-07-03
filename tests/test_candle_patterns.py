"""Candlestick pattern detectors — synthetic OHLC, no network."""

import pandas as pd

from tagent.candle_patterns import (
    detect_at, detect_patterns, engulfing, inside_bar, pin_bar, star,
    three_methods, three_soldiers,
)


def _df(bars):
    return pd.DataFrame(bars, columns=["open", "high", "low", "close"]).assign(volume=1.0)


def _arrs(bars):
    d = _df(bars)
    return (d["open"].to_numpy(float), d["high"].to_numpy(float),
            d["low"].to_numpy(float), d["close"].to_numpy(float))


# --------------------------------------------------------------------------- #
# engulfing
# --------------------------------------------------------------------------- #
def test_bullish_engulfing_fires():
    o, h, l, c = _arrs([(10.0, 10.2, 9.4, 9.5),    # bearish
                        (9.4, 10.7, 9.3, 10.6)])   # bullish, body engulfs prior
    assert engulfing(o, h, l, c, 1) == "bullish"


def test_bearish_engulfing_fires():
    o, h, l, c = _arrs([(9.5, 10.2, 9.4, 10.0),    # bullish
                        (10.1, 10.2, 9.2, 9.3)])   # bearish, body engulfs prior
    assert engulfing(o, h, l, c, 1) == "bearish"


def test_engulfing_not_on_two_same_colour():
    o, h, l, c = _arrs([(10.0, 10.5, 9.9, 10.4), (10.4, 11.0, 10.3, 10.9)])  # both bullish
    assert engulfing(o, h, l, c, 1) is None


# --------------------------------------------------------------------------- #
# pin bar
# --------------------------------------------------------------------------- #
def test_hammer_is_bullish():
    o, h, l, c = _arrs([(10.0, 10.1, 9.0, 10.05)])   # tiny body up top, long lower wick
    assert pin_bar(o, h, l, c, 0) == "bullish"


def test_shooting_star_is_bearish():
    o, h, l, c = _arrs([(10.0, 11.0, 9.98, 10.05)])  # long upper wick, tiny body at bottom
    assert pin_bar(o, h, l, c, 0) == "bearish"


def test_pin_bar_not_on_balanced_candle():
    o, h, l, c = _arrs([(10.0, 10.6, 9.4, 10.5)])    # big body, no dominant wick
    assert pin_bar(o, h, l, c, 0) is None


# --------------------------------------------------------------------------- #
# inside bar
# --------------------------------------------------------------------------- #
def test_inside_bar_takes_mother_direction():
    o, h, l, c = _arrs([(9.0, 11.0, 9.0, 10.8),      # big bullish mother
                        (10.0, 10.5, 9.5, 10.2)])    # inside
    assert inside_bar(o, h, l, c, 1) == "bullish"
    o2, h2, l2, c2 = _arrs([(11.0, 11.0, 9.0, 9.2),  # bearish mother
                            (10.0, 10.5, 9.5, 10.0)])
    assert inside_bar(o2, h2, l2, c2, 1) == "bearish"


def test_inside_bar_not_when_breaks_range():
    o, h, l, c = _arrs([(10.0, 10.5, 9.5, 10.2), (10.0, 11.0, 9.0, 10.1)])  # wider
    assert inside_bar(o, h, l, c, 1) is None


# --------------------------------------------------------------------------- #
# morning / evening star
# --------------------------------------------------------------------------- #
def test_morning_star_is_bullish():
    o, h, l, c = _arrs([(11.0, 11.1, 9.9, 10.0),     # big bearish
                        (9.85, 9.95, 9.7, 9.8),      # small star below
                        (9.9, 10.8, 9.85, 10.7)])    # big bullish, closes above bar0 mid (10.5)
    assert star(o, h, l, c, 2) == "bullish"


def test_evening_star_is_bearish():
    o, h, l, c = _arrs([(10.0, 11.1, 9.9, 11.0),     # big bullish
                        (11.15, 11.3, 11.1, 11.2),   # small star above
                        (11.1, 11.15, 10.2, 10.3)])  # big bearish, closes below bar0 mid (10.5)
    assert star(o, h, l, c, 2) == "bearish"


def test_star_not_when_middle_body_large():
    o, h, l, c = _arrs([(11.0, 11.1, 9.9, 10.0),
                        (9.9, 10.9, 9.8, 10.8),      # middle is a big body, not a star
                        (9.9, 10.8, 9.85, 10.7)])
    assert star(o, h, l, c, 2) is None


# --------------------------------------------------------------------------- #
# three soldiers / crows
# --------------------------------------------------------------------------- #
def test_three_white_soldiers_bullish():
    o, h, l, c = _arrs([(10.0, 10.95, 9.98, 10.9),
                        (10.5, 11.95, 10.48, 11.9),
                        (11.5, 12.95, 11.48, 12.9)])
    assert three_soldiers(o, h, l, c, 2) == "bullish"


def test_three_black_crows_bearish():
    o, h, l, c = _arrs([(13.0, 13.02, 12.05, 12.1),
                        (12.5, 12.52, 11.05, 11.1),
                        (11.5, 11.52, 10.05, 10.1)])
    assert three_soldiers(o, h, l, c, 2) == "bearish"


def test_three_soldiers_not_on_mixed():
    o, h, l, c = _arrs([(10.0, 10.9, 9.9, 10.8),
                        (10.8, 11.0, 10.2, 10.3),    # bearish in the middle
                        (10.3, 11.2, 10.2, 11.1)])
    assert three_soldiers(o, h, l, c, 2) is None


# --------------------------------------------------------------------------- #
# three methods
# --------------------------------------------------------------------------- #
def test_rising_three_methods_bullish():
    o, h, l, c = _arrs([(10.0, 12.1, 9.9, 12.0),     # long bullish
                        (11.8, 11.9, 11.5, 11.6),    # small bearish, inside
                        (11.6, 11.7, 11.3, 11.4),    # small bearish, inside
                        (11.4, 11.5, 11.1, 11.2),    # small bearish, inside
                        (11.3, 12.6, 11.2, 12.5)])   # long bullish, closes above bar0 close
    assert three_methods(o, h, l, c, 4) == "bullish"


def test_falling_three_methods_bearish():
    o, h, l, c = _arrs([(12.0, 12.1, 9.9, 10.0),     # long bearish
                        (10.2, 10.5, 10.1, 10.4),    # small bullish, inside
                        (10.4, 10.7, 10.3, 10.6),
                        (10.6, 10.9, 10.5, 10.8),
                        (10.7, 10.8, 9.4, 9.5)])      # long bearish, closes below bar0 close
    assert three_methods(o, h, l, c, 4) == "bearish"


def test_three_methods_not_when_middle_breaks_range():
    o, h, l, c = _arrs([(10.0, 12.1, 9.9, 12.0),
                        (11.8, 13.0, 11.5, 11.6),    # breaks above bar0 high -> not inside
                        (11.6, 11.7, 11.3, 11.4),
                        (11.4, 11.5, 11.1, 11.2),
                        (11.3, 12.6, 11.2, 12.5)])
    assert three_methods(o, h, l, c, 4) is None


# --------------------------------------------------------------------------- #
# aggregator + no-lookahead
# --------------------------------------------------------------------------- #
def test_detect_patterns_returns_named_signals_at_last_bar():
    df = _df([(10.0, 10.2, 9.4, 9.5), (9.4, 10.7, 9.3, 10.6)])   # bullish engulfing at bar 1
    out = detect_patterns(df)
    names = {(p["name"], p["direction"]) for p in out}
    assert ("engulfing", "bullish") in names
    assert all(p["index"] == 1 for p in out) and out[0]["label"]


def test_detect_at_no_lookahead():
    # a pattern detected at index i must NOT change when LATER bars are altered
    base = [(10.0, 10.2, 9.4, 9.5), (9.4, 10.7, 9.3, 10.6), (10.6, 10.7, 10.5, 10.55)]
    a = detect_at(_df(base), 1)
    base2 = base[:2] + [(10.6, 99.0, 0.1, 0.2)]   # wildly different FUTURE bar
    b = detect_at(_df(base2), 1)
    assert a == b and a                            # identical (only bars <= 1 used)
