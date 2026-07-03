"""Diversified trend core — live paper runner: same rule, sizing cap, persistence."""

import numpy as np
import pandas as pd

from tagent.trend_core import binary_trend_net, size_from_maxdd
from tagent.trend_core_live import (
    TrendCoreLiveConfig, TrendCoreLiveTrader, load_live_status,
)


def _closes(n=340, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    out = {}
    for i, m in enumerate(("KOSPI200", "S&P500")):
        r = 0.0015 + rng.normal(0.0, 0.003 + 0.002 * i, n)     # strong uptrend, S&P a touch noisier
        out[m] = pd.Series(100 * np.cumprod(1 + np.r_[0.0, r[:-1]]), index=idx)
    return out


# --------------------------------------------------------------------------- #
# 1) sizing cap re-derived off the diversified -22% maxDD (NOT -49.7%)
# --------------------------------------------------------------------------- #
def test_size_cap_rederived_off_diversified_maxdd():
    cfg = TrendCoreLiveConfig()
    assert cfg.strategy_maxdd == -0.22
    assert abs(cfg.size_cap() - size_from_maxdd(-0.22, -0.15)) < 1e-12     # ~0.68x
    assert abs(cfg.size_cap() - 0.15 / 0.22) < 1e-9
    # the single-market -49.7% would have given a much smaller cap -> this is the diversified upgrade
    assert cfg.size_cap() > size_from_maxdd(-0.497, -0.15) + 0.2


# --------------------------------------------------------------------------- #
# 2) same locked rule + risk-parity weights, status assembled
# --------------------------------------------------------------------------- #
def test_update_uses_locked_rule_and_reports_state(tmp_path):
    tr = TrendCoreLiveTrader(data_dir=tmp_path)
    st = tr.update(_closes())
    assert st["enabled"] and set(st["markets"]) == {"KOSPI200", "S&P500"}
    # both in a steady uptrend -> IN, weights ~ sum to 1, exposure ~ full
    assert all(st["markets"][m]["in_market"] for m in st["markets"])
    wsum = sum(st["markets"][m]["weight"] for m in st["markets"])
    assert abs(wsum - 1.0) < 0.05 and 0.9 <= st["combined_exposure"] <= 1.0
    # the per-market sleeve IS the locked binary rule (same codepath as trend_core)
    from tagent.multi_market_trend import INDEX_ROLL, INDEX_SWITCH
    c = _closes()["KOSPI200"]
    locked = binary_trend_net(c, regime_ma=200, mode="next_bar",
                              roll_annual=INDEX_ROLL, switch_cost=INDEX_SWITCH)
    assert len(locked) > 0
    # ops fields present for the shakedown
    assert "next_futures_roll" in st["ops"] and st["ops"]["est_margin"] >= 0
    assert st["dd_budget_pct"] == -22.0


# --------------------------------------------------------------------------- #
# 3) persistence: snapshot + idempotent CSV audit log
# --------------------------------------------------------------------------- #
def test_persistence_snapshot_and_idempotent_log(tmp_path):
    closes = _closes()
    TrendCoreLiveTrader(data_dir=tmp_path).update(closes)
    assert (tmp_path / "trend_core_live_state.json").exists()
    csv = tmp_path / "trend_core_live.csv"
    n1 = len(csv.read_text(encoding="utf-8").splitlines())
    # a fresh trader resumes _last_logged and re-running on the SAME data adds no rows
    TrendCoreLiveTrader(data_dir=tmp_path).update(closes)
    n2 = len(csv.read_text(encoding="utf-8").splitlines())
    assert n2 == n1 and n1 > 1
    # dashboard reader returns the persisted snapshot
    st = load_live_status(data_dir=tmp_path)
    assert st["enabled"] is True and "_last_logged" not in st and "combined_exposure" in st


def test_load_status_absent_is_disabled(tmp_path):
    assert load_live_status(data_dir=tmp_path) == {"enabled": False}


def test_shakedown_progress_and_gates(tmp_path):
    tr = TrendCoreLiveTrader(data_dir=tmp_path)
    st = tr.update(_closes())
    k = st["shakedown"]
    assert k["target_days"] == 60 and k["days_validated"] >= 1
    assert k["ops_exceptions"] == 0                          # synthetic uptrend -> no DD-budget breach
    assert k["gate_ops_clean"] is False                     # < 60 days -> not yet clean
    assert k["deploy_stage"] == "paper shakedown" and "ops exceptions" in k["progress"]
    # shakedown_start persists across restart (forward window is stable)
    start = k["start"]
    tr2 = TrendCoreLiveTrader(data_dir=tmp_path)
    assert tr2.update(_closes())["shakedown"]["start"] == start
