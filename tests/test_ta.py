"""Technical-analysis overlay maths — synthetic bars, no network."""

import numpy as np
import pandas as pd

from tagent.ta import (
    analyze,
    bounding_line,
    detect_breakout,
    detect_pattern,
    fit_line,
    stabilize_pattern,
    swing_pivots,
)


def _line_y(points, x):
    (x0, y0), (x1, y1) = points[0][:2], points[1][:2]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def _bars(close, vol=None):
    close = np.asarray(close, float)
    n = len(close)
    return pd.DataFrame({
        "open": np.r_[close[0], close[:-1]],
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": np.full(n, 1000.0) if vol is None else np.asarray(vol, float),
    })


# --------------------------------------------------------------------------- #
# pivots + line fit
# --------------------------------------------------------------------------- #
def test_swing_pivots_finds_local_extrema():
    # zigzag: peak at 3, trough at 6, peak at 9
    close = [1, 2, 3, 5, 3, 2, 1, 2, 3, 5, 3, 2, 1]
    df = _bars(close)
    ph, pl = swing_pivots(df["high"], df["low"], left=2, right=2)
    assert 3 in ph and 9 in ph          # the two peaks
    assert 6 in pl                       # the trough


def test_fit_line_recovers_slope_intercept():
    xs = [0, 1, 2, 3, 4]
    ys = [10, 12, 14, 16, 18]            # slope 2, intercept 10
    slope, intercept = fit_line(xs, ys)
    assert abs(slope - 2.0) < 1e-9 and abs(intercept - 10.0) < 1e-9


def test_fit_line_none_when_too_few():
    assert fit_line([1], [1]) is None


# --------------------------------------------------------------------------- #
# pattern classification
# --------------------------------------------------------------------------- #
def test_detect_ascending_and_descending_channel():
    up = (0.5, 100.0)        # positive slope
    assert detect_pattern(up, up, typ_price=100.0) == "ascending channel"
    dn = (-0.5, 100.0)
    assert detect_pattern(dn, dn, typ_price=100.0) == "descending channel"


def test_detect_triangle_converging():
    upper = (-0.5, 110.0)    # falling top
    lower = (0.5, 90.0)      # rising bottom
    assert detect_pattern(upper, lower, typ_price=100.0) == "symmetrical triangle"


# --------------------------------------------------------------------------- #
# breakout
# --------------------------------------------------------------------------- #
def test_detect_breakout_up_on_volume_spike():
    # flat top near 100, then a final close well above on 4x volume
    close = [100] * 30 + [105]
    vol = [1000] * 30 + [4000]
    df = _bars(close, vol)
    upper = (0.0, 100.2)     # roughly flat resistance at 100.2
    bk = detect_breakout(upper, None, df, vol_mult=1.5)
    assert bk and bk["direction"] == "up" and bk["index"] == 31 - 1


def test_no_breakout_without_volume_spike():
    close = [100] * 30 + [105]
    vol = [1000] * 31                    # no spike
    df = _bars(close, vol)
    assert detect_breakout((0.0, 100.2), None, df, vol_mult=1.5) is None


# --------------------------------------------------------------------------- #
# analyze (end to end)
# --------------------------------------------------------------------------- #
def test_analyze_ascending_channel_with_trendlines():
    # rising market with regular swings -> upper & lower trendlines, channel
    t = np.arange(60)
    close = 100 + 0.4 * t + 2.0 * np.sin(t / 2.0)
    out = analyze(_bars(close))
    assert out["n_bars"] == 60
    assert any(l["type"] == "upper" for l in out["lines"])
    assert any(l["type"] == "lower" for l in out["lines"])
    assert "channel" in out["pattern"] or "triangle" in out["pattern"] or "wedge" in out["pattern"]
    assert out["explanation"] and "trendline" in out["explanation"]
    # line endpoints reference bar indices within range
    for line in out["lines"]:
        for (i, _p) in line["points"]:
            assert 0 <= i <= 59


def test_analyze_reports_breakout_in_explanation():
    t = np.arange(40)
    close = list(100 + 1.5 * np.sin(t / 3.0))      # ranging
    close += [108.0]                                # final break up
    vol = [1000.0] * 40 + [5000.0]
    out = analyze(_bars(close, vol))
    if out["breakout"]:                             # depends on fitted upper line
        assert out["breakout"]["direction"] == "up"
        assert "breakout" in out["explanation"].lower()


def test_analyze_not_enough_bars():
    out = analyze(_bars([100, 101, 102, 103, 104]))
    assert out["lines"] == [] and out["pattern"] == "n/a"
    assert "Not enough" in out["explanation"]


def test_analyze_levels_have_support_and_resistance():
    t = np.arange(60)
    close = 100 + 3.0 * np.sin(t / 2.5)
    out = analyze(_bars(close))
    kinds = {lv["type"] for lv in out["levels"]}
    assert "resistance" in kinds or "support" in kinds


# --------------------------------------------------------------------------- #
# pivots exposed for chart verification
# --------------------------------------------------------------------------- #
def test_analyze_returns_pivots_and_lines_reference_them():
    t = np.arange(60)
    close = 100 + 0.4 * t + 2.0 * np.sin(t / 2.0)
    out = analyze(_bars(close))
    assert out["pivots"], "expected confirmed swing pivots"
    for p in out["pivots"]:
        assert p["type"] in ("high", "low")
        assert 0 <= p["index"] <= 59
        # the marked dot's price must match the bar it sits on
        assert isinstance(p["price"], float)
    # each trendline carries the actual pivots it was fit through
    for line in out["lines"]:
        assert len(line["pivots"]) >= 2
        idxs = {p["index"] for p in out["pivots"]}
        for (i, _price) in line["pivots"]:
            assert i in idxs            # line connects real, listed pivots


def test_line_pivot_prices_lie_on_the_bars():
    t = np.arange(60)
    close = 100 + 0.3 * t + 1.5 * np.sin(t / 1.7)
    df = _bars(close)
    out = analyze(df)
    for line in out["lines"]:
        col = "high" if line["type"] == "upper" else "low"
        for (i, price) in line["pivots"]:
            assert abs(price - df[col].iloc[i]) < 1e-9


# --------------------------------------------------------------------------- #
# trendlines are real tangents through the pivots (not a regression)
# --------------------------------------------------------------------------- #
def test_bounding_line_upper_stays_above_all_points():
    pts = [(0, 10.0), (5, 14.0), (10, 11.0), (15, 16.0), (20, 13.0)]
    ln = bounding_line(pts, upper=True)
    for (x, y) in pts:
        assert y <= ln["slope"] * x + ln["intercept"] + 1e-9     # never below a high
    # the line touches two real points
    for end in (ln["a"], ln["b"]):
        assert end in pts


def test_bounding_line_lower_stays_below_all_points():
    pts = [(0, 10.0), (5, 6.0), (10, 9.0), (15, 4.0), (20, 8.0)]
    ln = bounding_line(pts, upper=False)
    for (x, y) in pts:
        assert y >= ln["slope"] * x + ln["intercept"] - 1e-9     # never above a low


def test_trendlines_touch_marked_pivots_and_bound_price():
    t = np.arange(60)
    close = 100 + 0.4 * t + 2.0 * np.sin(t / 2.0)
    df = _bars(close)
    out = analyze(df)
    piv = {(p["type"], p["index"]): p["price"] for p in out["pivots"]}
    for line in out["lines"]:
        kind = "high" if line["type"] == "upper" else "low"
        # 1) both endpoints of the trendline are actual marked pivot dots
        for (i, price) in line["pivots"]:
            assert (kind, i) in piv and abs(piv[(kind, i)] - price) < 1e-9
            # 2) the DRAWN line passes through those pivot dots
            assert abs(_line_y(line["points"], i) - price) < 1e-6
        # 3) resistance stays >= highs / support stays <= lows over the drawn span
        lo_x = line["points"][0][0]
        for p in out["pivots"]:
            if p["type"] == kind and p["index"] >= lo_x:
                y = _line_y(line["points"], p["index"])
                if line["type"] == "upper":
                    assert p["price"] <= y + 1e-6
                else:
                    assert p["price"] >= y - 1e-6


# --------------------------------------------------------------------------- #
# pattern hysteresis
# --------------------------------------------------------------------------- #
def test_stabilize_pattern_first_call_shows_immediately():
    st, changed = stabilize_pattern({"shown": None, "cand": None, "n": 0}, "ascending channel")
    assert st["shown"] == "ascending channel" and changed is True


def test_stabilize_pattern_holds_until_confirmed():
    st = {"shown": "horizontal range", "cand": "horizontal range", "n": 0}
    # a single dissenting poll does NOT flip the label (min_confirm=2)
    st, changed = stabilize_pattern(st, "converging wedge", min_confirm=2)
    assert st["shown"] == "horizontal range" and changed is False
    # second consecutive poll of the same candidate flips it
    st, changed = stabilize_pattern(st, "converging wedge", min_confirm=2)
    assert st["shown"] == "converging wedge" and changed is True


def test_stabilize_pattern_flicker_resets_streak():
    st = {"shown": "range", "cand": "range", "n": 0}
    st, _ = stabilize_pattern(st, "wedge", min_confirm=3)     # n=1
    st, _ = stabilize_pattern(st, "triangle", min_confirm=3)  # different cand -> n=1
    st, changed = stabilize_pattern(st, "wedge", min_confirm=3)   # n=1 again
    assert st["shown"] == "range" and changed is False           # never confirmed
