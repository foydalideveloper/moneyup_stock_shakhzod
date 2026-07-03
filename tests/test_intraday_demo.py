"""Intraday show-don't-tell analysis — pure functions over mock ticks, no network."""

from datetime import datetime, timezone

from tagent.intraday_demo import (
    analyze_ticks, build_candles, candle_parts, cross_signal, moving_averages,
    parse_interval, regime_safety,
)


def T(h, m, s):
    return datetime(2026, 6, 9, h, m, s, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# candle aggregation — the candle is COMPUTED from raw ticks
# --------------------------------------------------------------------------- #
def test_parse_interval_units():
    assert parse_interval("1min") == 60
    assert parse_interval("5min") == 300
    assert parse_interval("30s") == 30
    assert parse_interval("1h") == 3600


def test_build_candles_aggregates_ohlcv_per_bar():
    ticks = [
        (T(10, 0, 5), 100, 1), (T(10, 0, 20), 102, 2),     # high of bar 0
        (T(10, 0, 40), 99, 1), (T(10, 0, 55), 101, 1),     # low then close of bar 0
        (T(10, 1, 5), 101.5, 3), (T(10, 1, 30), 103, 1),   # bar 1
    ]
    bars = build_candles(ticks, "1min")
    assert len(bars) == 2
    b0 = bars[0]
    assert (b0["open"], b0["high"], b0["low"], b0["close"]) == (100, 102, 99, 101)
    assert b0["volume"] == 5 and b0["n"] == 4
    b1 = bars[1]
    assert (b1["open"], b1["high"], b1["low"], b1["close"]) == (101.5, 103, 101.5, 103)
    assert b1["volume"] == 4 and b1["n"] == 2


def test_build_candles_sorts_ticks_no_lookahead():
    # ticks shuffled + a later bar appended -> earlier bar identical (no-lookahead)
    base = [(T(10, 0, 55), 101, 1), (T(10, 0, 5), 100, 1), (T(10, 0, 20), 102, 1)]
    b_first = build_candles(base, "1min")[0]
    extended = base + [(T(10, 1, 10), 999, 5)]              # arbitrary future bar
    b_first2 = build_candles(extended, "1min")[0]
    assert b_first == b_first2
    assert (b_first["open"], b_first["high"], b_first["close"]) == (100, 102, 101)


# --------------------------------------------------------------------------- #
# body color + wick math
# --------------------------------------------------------------------------- #
def test_candle_parts_red_up_with_wicks():
    p = candle_parts({"open": 100, "high": 102, "low": 99, "close": 101})
    assert p["color"] == "red"                              # close>open -> red (양봉)
    assert p["body"] == 1 and p["upper_wick"] == 1 and p["lower_wick"] == 1
    assert p["range"] == 3 and p["up"] is True


def test_candle_parts_blue_down():
    p = candle_parts({"open": 101, "high": 101.5, "low": 98, "close": 99})
    assert p["color"] == "blue"                             # close<open -> blue (음봉)
    assert p["body"] == 2 and p["upper_wick"] == 0.5 and p["lower_wick"] == 1


def test_candle_parts_doji():
    p = candle_parts({"open": 100, "high": 101, "low": 99, "close": 100})
    assert p["color"] == "doji" and p["body"] == 0


# --------------------------------------------------------------------------- #
# moving averages + golden/dead cross
# --------------------------------------------------------------------------- #
def test_moving_averages_warmup_and_value():
    ma = moving_averages([1, 2, 3, 4, 5, 6], windows=[3])
    assert ma[3] == [None, None, 2.0, 3.0, 4.0, 5.0]


def test_cross_signal_golden_dead_none():
    assert cross_signal([1, 1, 3], [2, 2, 2]) == "golden"   # short crosses up through long
    assert cross_signal([3, 3, 1], [2, 2, 2]) == "dead"     # short crosses down through long
    assert cross_signal([3, 4], [1, 2]) is None             # stays above -> no cross
    assert cross_signal([None, 1], [None, 2]) is None       # <2 valid pairs -> None


# --------------------------------------------------------------------------- #
# safety halt on linked-market crash
# --------------------------------------------------------------------------- #
def test_regime_safety_halts_when_linked_breaches_threshold():
    assert regime_safety({"AAPL": -0.04, "MSFT": 0.01}) == "HALT"   # -4% <= -3%
    assert regime_safety({"AAPL": -0.01, "MSFT": 0.02}) == "OK"
    assert regime_safety({"NDX": -0.03}) == "HALT"                  # exactly at threshold
    assert regime_safety([-0.05, 0.0]) == "HALT"                   # list input
    assert regime_safety([]) == "OK" and regime_safety({}) == "OK"
    assert regime_safety({"X": -0.10}, threshold=-0.15) == "OK"     # custom threshold


# --------------------------------------------------------------------------- #
# full pipeline
# --------------------------------------------------------------------------- #
def test_analyze_ticks_pipeline():
    ticks = []
    price = 100.0
    # 8 one-minute bars, rising -> 5MA should sit below price, golden-ish structure
    for minute in range(8):
        for sec in (5, 25, 45):
            price += 0.5
            ticks.append((T(10, minute, sec), price, 1))
    out = analyze_ticks(ticks, linked_returns={"SPY": -0.005}, interval="1min",
                        windows=(3, 5, 8))
    assert out["n_candles"] == 8 and out["n_ticks"] == 24
    assert set(out["mas"]) == {3, 5, 8}
    assert out["mas"][3] is not None                        # enough bars for the 3-MA
    assert out["safety"] == "OK"
    assert out["latest_candle"]["color"] == "red"           # rising -> up bar
    assert len(out["ticks"]) <= 12 and out["cross_pair"] == [3, 5]


def test_analyze_ticks_safety_halts():
    ticks = [(T(10, 0, s), 100 + s, 1) for s in (5, 20, 40)]
    out = analyze_ticks(ticks, linked_returns={"SPY": -0.06}, interval="1min")
    assert out["safety"] == "HALT"
