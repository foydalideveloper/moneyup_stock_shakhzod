"""Intraday minute backtester + gap-reversal — mock minute data, no network."""

import pandas as pd

from tagent.data.intraday_history import overnight_for_dates
from tagent.intraday_backtest import (
    CostModel, DayPlan, by_year, metrics, run, simulate_day, walk_forward,
)
from tagent.strategies.gap_reversal import GapReversalParams, default_grid, make_strategy


def make_day(date_str, bars, start_hhmm="09:00"):
    """bars: list of (open, high, low, close)."""
    idx = pd.date_range(f"{date_str} {start_hhmm}", periods=len(bars), freq="min")
    return pd.DataFrame({"open": [b[0] for b in bars], "high": [b[1] for b in bars],
                         "low": [b[2] for b in bars], "close": [b[3] for b in bars],
                         "volume": [100] * len(bars)}, index=idx)


def panel(*days):
    df = pd.concat(days).sort_index()
    df.index.name = "timestamp"
    return df


COST = CostModel(fee_bps=1.5, slippage_bps=10.0, sell_tax_bps=18.0)   # round trip = 0.41%


# --------------------------------------------------------------------------- #
# cost accounting per round trip
# --------------------------------------------------------------------------- #
def test_round_trip_cost_fraction():
    assert abs(COST.round_trip_frac() - 0.0041) < 1e-12       # 2*1.5 + 2*10 + 18 bps


def test_simulate_day_take_profit_net_of_costs():
    # entry 100, take +1% -> a later bar trades up to 101; net = gross - round-trip cost
    day = make_day("2023-01-03", [(100, 100, 100, 100), (100, 100.2, 99.9, 100.1),
                                  (100.1, 101.5, 100.0, 101.0)])
    plan = DayPlan(entry_index=0, entry_price=100.0, take_pct=0.01, stop_pct=0.02)
    t = simulate_day(day, plan, COST.round_trip_frac())
    assert t.reason == "take"
    assert abs(t.gross - 0.01) < 1e-9
    assert abs(t.net - (0.01 - 0.0041)) < 1e-9                # costs deducted exactly once


def test_simulate_day_stop_and_close():
    # stop at 98 hit
    d_stop = make_day("2023-01-03", [(100, 100, 100, 100), (100, 100.1, 97.5, 98.0)])
    ts = simulate_day(d_stop, DayPlan(0, 100.0, 0.05, 0.02), COST.round_trip_frac())
    assert ts.reason == "stop" and abs(ts.gross - (-0.02)) < 1e-9
    # neither hit -> exit at last close
    d_close = make_day("2023-01-03", [(100, 100, 100, 100), (100, 100.3, 99.8, 100.2)])
    tc = simulate_day(d_close, DayPlan(0, 100.0, 0.05, 0.05), COST.round_trip_frac())
    assert tc.reason == "close" and abs(tc.gross - (100.2 / 100.0 - 1)) < 1e-9


def test_simulate_day_stop_checked_before_take_within_bar():
    # a bar that touches BOTH the take and the stop -> conservative: stop wins
    day = make_day("2023-01-03", [(100, 100, 100, 100), (100, 102, 97, 100)])
    t = simulate_day(day, DayPlan(0, 100.0, 0.01, 0.02), COST.round_trip_frac())
    assert t.reason == "stop"


# --------------------------------------------------------------------------- #
# gap / entry / overnight logic
# --------------------------------------------------------------------------- #
def _two_day_gap_panel(open2=97.0):
    d1 = make_day("2023-01-03", [(100, 100, 100, 100)] * 6)             # prev_close = 100
    # day 2 gaps to open2; flat bars so no take/stop -> exit at close
    d2 = make_day("2023-01-04", [(open2, open2, open2, open2)] * 6)
    return panel(d1, d2)


def test_gap_reversal_fires_on_systemic_gap_down():
    p = _two_day_gap_panel(open2=97.0)                                   # -3% gap
    overnight = {pd.Timestamp("2023-01-04").date(): -0.01}              # systemic down
    params = GapReversalParams(gap_pct=0.02, take_pct=0.05, stop_pct=0.05,
                               entry_minute=2, overnight_thresh=-0.005)
    trades = run(p, make_strategy(params), cost=COST, overnight=overnight)
    assert len(trades) == 1 and trades[0].date == "2023-01-04"
    assert trades[0].entry_price == 97.0                                # entry at the gapped bar


def test_gap_reversal_skips_small_gap_and_idiosyncratic():
    params = GapReversalParams(gap_pct=0.02, entry_minute=2, overnight_thresh=-0.005)
    # small gap (-1%) -> no trade even with systemic overnight
    p_small = _two_day_gap_panel(open2=99.0)
    assert run(p_small, make_strategy(params), cost=COST,
               overnight={pd.Timestamp("2023-01-04").date(): -0.01}) == []
    # big gap but overnight POSITIVE (idiosyncratic, not systemic) -> no trade
    p_big = _two_day_gap_panel(open2=97.0)
    assert run(p_big, make_strategy(params), cost=COST,
               overnight={pd.Timestamp("2023-01-04").date(): +0.004}) == []


def test_first_day_has_no_prev_close_so_no_trade():
    d1 = make_day("2023-01-03", [(97, 97, 97, 97)] * 6)
    params = GapReversalParams(gap_pct=0.02, entry_minute=2, overnight_thresh=-0.005)
    assert run(panel(d1), make_strategy(params), cost=COST,
               overnight={pd.Timestamp("2023-01-03").date(): -0.01}) == []


# --------------------------------------------------------------------------- #
# no-lookahead
# --------------------------------------------------------------------------- #
def test_no_lookahead_future_bars_do_not_change_trade():
    overnight = {pd.Timestamp("2023-01-04").date(): -0.01}
    params = GapReversalParams(gap_pct=0.02, take_pct=0.01, stop_pct=0.05,
                               entry_minute=1, overnight_thresh=-0.005)
    # day2: take hits early at bar 2; later bars are then irrelevant
    d1 = make_day("2023-01-03", [(100, 100, 100, 100)] * 4)
    base_bars = [(97, 97, 97, 97), (97, 97, 97, 97), (97, 98.5, 97, 98.0)]   # 97*1.01=97.97 take
    d2 = make_day("2023-01-04", base_bars)
    t1 = run(panel(d1, d2), make_strategy(params), cost=COST, overnight=overnight)[0]
    # append arbitrary future bars AFTER the exit -> identical trade
    d2_ext = make_day("2023-01-04", base_bars + [(98, 200, 1, 150), (150, 150, 150, 150)])
    t2 = run(panel(d1, d2_ext), make_strategy(params), cost=COST, overnight=overnight)[0]
    assert (t1.reason, t1.exit_price, t1.net) == (t2.reason, t2.exit_price, t2.net)
    assert t1.reason == "take"


def test_overnight_for_dates_is_no_lookahead():
    spy = pd.Series([0.01, -0.02, 0.005],
                    index=pd.to_datetime(["2023-01-03", "2023-01-04", "2023-01-05"]))
    # KR 2023-01-05 should see the last US session STRICTLY before it = 2023-01-04 (-0.02)
    o = overnight_for_dates(spy, [pd.Timestamp("2023-01-05")])
    assert abs(o[pd.Timestamp("2023-01-05").date()] - (-0.02)) < 1e-12
    # appending a future US session does not change the earlier mapping
    spy2 = pd.concat([spy, pd.Series([0.5], index=pd.to_datetime(["2023-01-06"]))])
    o2 = overnight_for_dates(spy2, [pd.Timestamp("2023-01-05")])
    assert o2 == o


# --------------------------------------------------------------------------- #
# metrics + walk-forward split
# --------------------------------------------------------------------------- #
def test_metrics_basic():
    from tagent.intraday_backtest import Trade
    trades = [Trade("2023-01-03", "", "", 100, 102, 0.02, 0.016, "take"),
              Trade("2023-01-04", "", "", 100, 99, -0.01, -0.014, "stop")]
    m = metrics(trades, n_days=2)
    assert m["n_trades"] == 2 and m["win_rate"] == 0.5
    assert abs(m["expectancy"] - (0.016 - 0.014) / 2) < 1e-12
    assert abs(m["trades_per_day"] - 1.0) < 1e-12
    assert m["profit_factor"] > 0


def test_walk_forward_split_is_chronological():
    from tagent.data.intraday_history import make_synthetic_minutes
    import numpy as np
    days = [d.date() for d in pd.bdate_range("2023-01-02", periods=160)]   # ~8 months
    rng = np.random.default_rng(1)
    overnight = {d: float(rng.normal(0, 0.012)) for d in days}
    df = make_synthetic_minutes(n_days=160, seed=5, overnight=overnight, bars_per_day=20)
    grid = [GapReversalParams(gap_pct=0.005, take_pct=0.01, stop_pct=0.01, entry_minute=2,
                              overnight_thresh=-0.003),
            GapReversalParams(gap_pct=0.01, take_pct=0.015, stop_pct=0.01, entry_minute=2,
                              overnight_thresh=-0.003)]
    wf = walk_forward(df, make_strategy, grid, overnight=overnight,
                      train_months=3, test_months=1)
    assert wf["n_folds"] >= 2
    prev_test = None
    for f in wf["folds"]:
        assert max(f["train_months"]) < f["test_months"][0]        # train precedes test
        if prev_test is not None:
            assert f["test_months"][0] > prev_test                 # folds roll forward
        prev_test = f["test_months"][0]
    # every OOS trade falls inside some fold's test month (no leakage from train)
    test_months = {tm for f in wf["folds"] for tm in f["test_months"]}
    for t in wf["oos"]:
        assert t.date[:7] in test_months


def test_default_grid_nonempty_and_by_year():
    assert len(default_grid()) == 18
    from tagent.intraday_backtest import Trade
    yb = by_year([Trade("2023-01-03", "", "", 100, 101, 0.01, 0.006, "take"),
                  Trade("2024-02-03", "", "", 100, 101, 0.01, 0.006, "take")])
    assert set(yb) == {"2023", "2024"}
