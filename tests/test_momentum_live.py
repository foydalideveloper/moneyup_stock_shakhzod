"""Live KR momentum paper trader — mock data, no network.

Invariants: monthly rebalance cadence, the regime filter goes to cash in a
downtrend, long-only top-quantile weights sum to 1, state persists across a
restart, and the whole step is strictly no-lookahead (future bars never change a
past rebalance).
"""

import numpy as np
import pandas as pd

from tagent.momentum_live import (
    MomentumLiveConfig, MomentumLiveTrader, MonthlyRebalanceScheduler, diff_holdings,
    holdings_turnover, momentum_scores, normalize_ticker, regime_is_on,
    should_rebalance, target_weights,
)

CFG = MomentumLiveConfig(lookback=60, skip_recent=5, regime_ma=30, top_q=0.3,
                         market="kr", capital=10_000.0)


def _panel(n=340, k=10, seed=0, lo=-0.0005, hi=0.003, start="2018-01-01"):
    """k trending names (persistent cross-sectional spread), overall rising."""
    rng = np.random.default_rng(seed)
    drifts = np.linspace(lo, hi, k)
    idx = pd.date_range(start, periods=n, freq="B")
    out = {}
    for i in range(k):
        steps = rng.normal(drifts[i], 0.008, n)
        out[f"S{i}"] = pd.DataFrame({"close": 100 * np.cumprod(1 + np.r_[0.0, steps[:-1]])}, index=idx)
    return out


def _panel_down(n=220, k=8, start="2018-01-01"):
    """Basket ramps up then falls hard -> in a confirmed downtrend at the last bar."""
    idx = pd.date_range(start, periods=n, freq="B")
    half = n // 2
    shape = np.r_[np.linspace(100, 220, half), np.linspace(220, 120, n - half)]
    return {f"S{i}": pd.DataFrame({"close": shape * (1 + 0.02 * i)}, index=idx) for i in range(k)}


# --------------------------------------------------------------------------- #
# monthly rebalance cadence
# --------------------------------------------------------------------------- #
def test_should_rebalance_is_monthly():
    assert should_rebalance("2024-03-01", None)
    assert should_rebalance("2024-03-15", "2024-02")
    assert not should_rebalance("2024-03-20", "2024-03")


def test_step_rebalances_once_per_month(tmp_path):
    panel = _panel()
    uni = list(panel)
    tr = MomentumLiveTrader(CFG, data_dir=tmp_path)
    idx = panel["S0"].index
    d1 = idx[300]
    same = idx[idx.strftime("%Y-%m") == d1.strftime("%Y-%m")]
    d1b = same[-1]                          # last business day of d1's month (same month)
    later = idx[idx.strftime("%Y-%m") > d1.strftime("%Y-%m")][5]   # a later month
    assert d1.strftime("%Y-%m") == d1b.strftime("%Y-%m") != later.strftime("%Y-%m")
    tr.step(panel, uni, asof=d1)
    assert tr.cycle == 1
    tr.step(panel, uni, asof=d1b)          # same calendar month -> no-op
    assert tr.cycle == 1
    tr.step(panel, uni, asof=later)
    assert tr.cycle == 2
    # ...but --rebalance-now (force) overrides the monthly guard
    tr.step(panel, uni, asof=later, force=True)
    assert tr.cycle == 3


# --------------------------------------------------------------------------- #
# regime filter -> cash in a downtrend
# --------------------------------------------------------------------------- #
def test_regime_filter_goes_to_cash_in_downtrend(tmp_path):
    panel = _panel_down()
    uni = list(panel)
    asof = panel["S0"].index[-1]
    assert regime_is_on(panel, CFG, asof=asof, universe=uni) is False
    tr = MomentumLiveTrader(CFG, data_dir=tmp_path)
    st = tr.step(panel, uni, asof=asof)
    assert st["regime"] == "cash" and st["exposure_pct"] == 0.0
    assert st["n_held"] == 0 and tr.weights == {}


def test_regime_filter_in_market_in_uptrend(tmp_path):
    panel = _panel()
    uni = list(panel)
    asof = panel["S0"].index[-1]
    assert regime_is_on(panel, CFG, asof=asof, universe=uni) is True
    st = MomentumLiveTrader(CFG, data_dir=tmp_path).step(panel, uni, asof=asof)
    assert st["regime"] == "in-market" and st["exposure_pct"] == 100.0 and st["n_held"] > 0


# --------------------------------------------------------------------------- #
# long-only top-quantile weights
# --------------------------------------------------------------------------- #
def test_long_only_top_quantile_weights_sum_to_one():
    panel = _panel()
    uni = list(panel)
    asof = panel["S0"].index[-1]
    scores = momentum_scores(panel, CFG, asof=asof)
    w = target_weights(scores, uni, True, CFG)
    assert len(w) == max(1, round(len(scores) * CFG.top_q))   # top quantile only
    assert all(v > 0 for v in w.values())                     # long-only, no shorts
    assert abs(sum(w.values()) - 1.0) < 1e-9                  # fully invested, capped 1x
    # the selected names are the highest-momentum ones
    chosen = set(w)
    top_by_score = set(scores.sort_values(ascending=False).head(len(w)).index)
    assert chosen == top_by_score


def test_target_weights_empty_when_regime_off():
    panel = _panel()
    scores = momentum_scores(panel, CFG, asof=panel["S0"].index[-1])
    assert target_weights(scores, list(panel), False, CFG) == {}


# --------------------------------------------------------------------------- #
# persistence across restart
# --------------------------------------------------------------------------- #
def test_state_persists_across_restart(tmp_path):
    panel = _panel()
    uni = list(panel)
    a = MomentumLiveTrader(CFG, data_dir=tmp_path)
    a.step(panel, uni, asof=panel["S0"].index[300])
    a.step(panel, uni, asof=panel["S0"].index[325])
    snap = a.status()
    # a fresh trader reads the same persisted state
    b = MomentumLiveTrader(CFG, data_dir=tmp_path)
    assert b.cycle == a.cycle == 2
    assert b.weights == a.weights
    assert abs(b.equity - a.equity) < 1e-9
    assert b.status()["equity"] == snap["equity"]
    assert (tmp_path / "momentum_live_state.json").exists()
    assert (tmp_path / "momentum_live.csv").exists()


# --------------------------------------------------------------------------- #
# BUY / SELL / HOLD diff
# --------------------------------------------------------------------------- #
def test_diff_holdings_first_run_is_all_buy():
    d = diff_holdings([], ["A", "B", "C"])
    assert d["to_buy"] == ["A", "B", "C"] and d["to_sell"] == [] and d["to_hold"] == []


def test_diff_holdings_buy_sell_hold():
    d = diff_holdings(["A", "B", "C"], ["B", "C", "D"])
    assert d["to_buy"] == ["D"] and d["to_sell"] == ["A"] and d["to_hold"] == ["B", "C"]


def test_diff_holdings_go_to_cash_is_all_sell():
    d = diff_holdings(["A", "B"], [])
    assert d["to_sell"] == ["A", "B"] and d["to_buy"] == [] and d["to_hold"] == []


def test_normalize_ticker_across_universe_source_formats():
    # zero-pad, Kiwoom 'A' prefix, and exchange suffix all canonicalize to the same code
    assert normalize_ticker("5930") == "005930"
    assert normalize_ticker("A005930") == "005930"
    assert normalize_ticker("005930.KS") == "005930"
    assert normalize_ticker("000660.KQ") == "000660"


def test_diff_holdings_normalizes_so_same_holdings_are_hold_not_rotation():
    # identical economic book in two different source formats -> all HOLD (no fake churn)
    d = diff_holdings(["005930", "000660"], ["005930.KS", "A000660"])
    assert d["to_hold"] == ["000660", "005930"]
    assert d["to_buy"] == [] and d["to_sell"] == []


def test_holdings_turnover_zero_when_same_partial_when_rotated():
    assert holdings_turnover(["005930", "000660"], ["A005930", "000660.KS"]) == 0.0
    # 4 distinct names leave/enter out of a union of 6 -> 4/6
    assert abs(holdings_turnover(["A", "B", "C", "D"], ["C", "D", "E", "F"]) - 4 / 6) < 1e-9
    assert holdings_turnover([], ["A", "B"]) == 1.0       # first buy from cash = full


def test_step_records_first_run_all_buy_and_next_rebalance(tmp_path):
    panel = _panel()
    uni = list(panel)
    asof = panel["S0"].index[-1]
    st = MomentumLiveTrader(CFG, data_dir=tmp_path).step(panel, uni, asof=asof)
    held = sorted(h["symbol"] for h in st["held"])
    assert st["actions"]["to_buy"] == held                 # first run -> everything is a BUY
    assert st["actions"]["to_sell"] == [] and st["actions"]["to_hold"] == []
    assert st["last_rebalanced"][:7] == asof.strftime("%Y-%m")
    assert st["next_rebalance"] == str(pd.Period(asof.strftime("%Y-%m"), "M") + 1)


def test_step_actions_diff_and_persist(tmp_path):
    panel = _panel()
    uni = list(panel)
    tr = MomentumLiveTrader(CFG, data_dir=tmp_path)
    tr.step(panel, uni, asof=panel["S0"].index[300])
    prev = set(tr.weights)
    st2 = tr.step(panel, uni, asof=panel["S0"].index[325], force=True)
    new = set(tr.weights)
    a = st2["actions"]
    assert set(a["to_buy"]) == new - prev                  # actions match the holdings change
    assert set(a["to_sell"]) == prev - new
    assert set(a["to_hold"]) == new & prev
    # actions + next-rebalance survive a restart
    b = MomentumLiveTrader(CFG, data_dir=tmp_path)
    assert b.last_actions == tr.last_actions
    assert b.status()["next_rebalance"] == st2["next_rebalance"]
    assert b.last_turnover == tr.last_turnover                  # turnover persists too


def test_step_reports_turnover(tmp_path):
    panel = _panel()
    uni = list(panel)
    tr = MomentumLiveTrader(CFG, data_dir=tmp_path)
    st = tr.step(panel, uni, asof=panel["S0"].index[-1])
    assert st["turnover_pct"] == 100.0 and st["high_turnover"] is True   # first buy from cash
    assert 0.0 <= st["turnover_pct"] <= 100.0


# --------------------------------------------------------------------------- #
# monthly auto-rebalance scheduler
# --------------------------------------------------------------------------- #
def test_scheduler_triggers_once_per_new_month():
    calls = {"n": 0}
    bump = lambda: calls.__setitem__("n", calls["n"] + 1)
    sched = MonthlyRebalanceScheduler(rebalance_fn=bump, month_done_fn=lambda: None, min_interval_s=0)
    assert sched.tick(now=pd.Timestamp("2026-03-05")) is True and calls["n"] == 1
    assert sched.tick(now=pd.Timestamp("2026-03-20")) is False and calls["n"] == 1   # idempotent
    assert sched.tick(now=pd.Timestamp("2026-04-01")) is True and calls["n"] == 2    # new month
    assert sched.tick(now=pd.Timestamp("2026-04-28")) is False and calls["n"] == 2


def test_scheduler_respects_persisted_month_guard():
    calls = {"n": 0}
    bump = lambda: calls.__setitem__("n", calls["n"] + 1)
    # state already shows this month rebalanced (manual script ran, or a restart)
    sched = MonthlyRebalanceScheduler(rebalance_fn=bump, month_done_fn=lambda: "2026-03", min_interval_s=0)
    assert sched.tick(now=pd.Timestamp("2026-03-10")) is False and calls["n"] == 0   # no double-rebalance
    assert sched.tick(now=pd.Timestamp("2026-04-02")) is True and calls["n"] == 1


def test_scheduler_throttles_within_interval():
    calls = {"n": 0}
    bump = lambda: calls.__setitem__("n", calls["n"] + 1)
    sched = MonthlyRebalanceScheduler(rebalance_fn=bump, month_done_fn=lambda: None, min_interval_s=86400)
    assert sched.tick(now=pd.Timestamp("2026-03-01T00:00:00")) is True
    assert sched.tick(now=pd.Timestamp("2026-03-01T01:00:00")) is False              # throttled <1 day
    assert calls["n"] == 1


# --------------------------------------------------------------------------- #
# strictly no-lookahead
# --------------------------------------------------------------------------- #
def test_step_is_no_lookahead(tmp_path):
    panel = _panel(seed=3)
    uni = list(panel)
    asof = panel["S0"].index[300]
    # A: trader sees data only up to asof
    trunc = {s: df.loc[df.index <= asof] for s, df in panel.items()}
    a = MomentumLiveTrader(CFG, data_dir=tmp_path / "a")
    sa = a.step(trunc, uni, asof=asof)
    # B: trader sees the FULL panel (with future bars after asof) but same asof
    spiked = {s: df.copy() for s, df in panel.items()}
    for df in spiked.values():
        df.loc[df.index > asof, "close"] *= 3.0          # arbitrary future shock
    b = MomentumLiveTrader(CFG, data_dir=tmp_path / "b")
    sb = b.step(spiked, uni, asof=asof)
    assert a.weights == b.weights                          # future bars don't change selection
    assert abs(sa["equity"] - sb["equity"]) < 1e-9
    assert sa["regime"] == sb["regime"]
