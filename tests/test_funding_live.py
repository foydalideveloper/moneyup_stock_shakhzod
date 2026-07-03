"""Live funding-carry paper runner — mock funding/prices, no network."""

import numpy as np

from tagent.funding_live import (
    FundingCarryTrader, LiveConfig, carry_weights, load_live_status,
    target_set, update_hold,
)

UNI = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")


def _cfg(**kw):
    base = dict(universe=UNI, capital=10_000.0, hurdle_bps=1.0, band_bps=4.0,
                top_n=2, max_weight=0.6, leverage=3.0)
    base.update(kw)
    return LiveConfig(**base)


def _prices(p=100.0):
    return {c: p for c in UNI}


# --------------------------------------------------------------------------- #
# selection: hurdle, hysteresis, ranking, cap, top-N
# --------------------------------------------------------------------------- #
def test_hurdle_gates_entry_only_above_threshold():
    cfg = _cfg()
    held = update_hold({"BTCUSDT": 0.0005, "ETHUSDT": 0.00005, "XRPUSDT": -0.0002}, {}, cfg)
    assert held["BTCUSDT"] is True              # 5bp >= 1bp hurdle
    assert held["ETHUSDT"] is False            # 0.5bp < hurdle
    assert held["XRPUSDT"] is False            # negative


def test_hysteresis_holds_through_mild_negative_exits_on_sustained():
    cfg = _cfg()
    held = update_hold({"BTCUSDT": 0.0005}, {}, cfg)            # enter
    held = update_hold({"BTCUSDT": -0.00005}, held, cfg)        # -0.5bp > exit(-3bp): hold
    assert held["BTCUSDT"] is True
    held = update_hold({"BTCUSDT": -0.0005}, held, cfg)         # -5bp < -3bp: exit
    assert held["BTCUSDT"] is False


def test_target_set_takes_top_n_by_funding():
    cfg = _cfg(top_n=2)
    held = {c: True for c in UNI}
    s = target_set({"BTCUSDT": 0.0006, "ETHUSDT": 0.0004, "SOLUSDT": 0.0002, "XRPUSDT": 0.0001},
                   held, cfg)
    assert s == {"BTCUSDT", "ETHUSDT"}


def test_carry_weights_funding_weighted_capped():
    cfg = _cfg(top_n=4, max_weight=0.4)
    w = carry_weights({"BTCUSDT": 0.0008, "ETHUSDT": 0.0004, "SOLUSDT": 0.0002, "XRPUSDT": 0.0001},
                      {"BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"}, cfg)
    assert w["BTCUSDT"] >= w["ETHUSDT"] >= w["SOLUSDT"]         # ranked by funding
    assert max(w.values()) <= 0.4 + 1e-9                        # cap binds
    assert abs(sum(w.values()) - 1.0) < 1e-9                    # fully deployed


# --------------------------------------------------------------------------- #
# accrual + cost + hold-don't-churn
# --------------------------------------------------------------------------- #
def test_accrues_funding_on_held_positions(tmp_path):
    t = FundingCarryTrader(_cfg(), data_dir=tmp_path)
    fund = {"BTCUSDT": 0.0005, "ETHUSDT": 0.0003}
    t.step(fund, _prices(), funding_time=1)                     # enter (no accrual yet)
    s1 = t.status()
    assert s1["n_held"] == 2 and s1["accrued"] == 0.0 and s1["costs"] > 0
    t.step(fund, _prices(), funding_time=2)                     # held -> accrue funding
    s2 = t.status()
    expected = sum(fund[h["coin"]] * h["weight"] * 10_000 for h in s1["held"])
    assert abs(s2["accrued"] - expected) < 1e-6
    assert s2["equity"] > 10_000.0 - s2["costs"] + s2["accrued"] - 1e-6


def test_holding_same_set_costs_nothing_more(tmp_path):
    t = FundingCarryTrader(_cfg(), data_dir=tmp_path)
    fund = {"BTCUSDT": 0.0005, "ETHUSDT": 0.0003}
    t.step(fund, _prices(), funding_time=1)
    c1 = t.status()["costs"]
    for ft in (2, 3, 4):
        t.step(fund, _prices(), funding_time=ft)               # same held set -> frozen
    s = t.status()
    assert abs(s["costs"] - c1) < 1e-9                          # no extra turnover cost
    assert s["accrued"] > 0                                     # but funding keeps accruing


def test_rebalance_when_set_changes_costs_more(tmp_path):
    t = FundingCarryTrader(_cfg(top_n=2), data_dir=tmp_path)
    t.step({"BTCUSDT": 0.0005, "ETHUSDT": 0.0003}, _prices(), funding_time=1)
    c1 = t.status()["costs"]
    # SOL overtakes ETH and ETH funding collapses -> held set changes -> turnover
    t.step({"BTCUSDT": 0.0005, "ETHUSDT": -0.0005, "SOLUSDT": 0.0006}, _prices(), funding_time=2)
    assert t.status()["costs"] > c1
    assert {h["coin"] for h in t.status()["held"]} == {"BTCUSDT", "SOLUSDT"}


# --------------------------------------------------------------------------- #
# margin guard at low leverage
# --------------------------------------------------------------------------- #
def test_margin_guard_derisks_on_sharp_rally(tmp_path):
    t = FundingCarryTrader(_cfg(leverage=3.0, margin_buffer=0.03), data_dir=tmp_path)
    t.step({"BTCUSDT": 0.0005, "ETHUSDT": 0.0003}, {"BTCUSDT": 100.0, "ETHUSDT": 100.0},
           funding_time=1)
    assert "BTCUSDT" in {h["coin"] for h in t.status()["held"]}
    # BTC perp rallies +35% -> short margin ratio 1/3 - 0.35 < maint+buffer -> guard out
    t.step({"BTCUSDT": 0.0005, "ETHUSDT": 0.0003}, {"BTCUSDT": 135.0, "ETHUSDT": 100.0},
           funding_time=2)
    s = t.status()
    assert "BTCUSDT" not in {h["coin"] for h in s["held"]}      # de-risked
    assert any(g["coin"] == "BTCUSDT" for g in s["guard_events"])


def test_low_leverage_has_large_headroom(tmp_path):
    t = FundingCarryTrader(_cfg(leverage=3.0), data_dir=tmp_path)
    t.step({"BTCUSDT": 0.0005}, {"BTCUSDT": 100.0}, funding_time=1)
    m = t.status()["margin"]
    assert m["min_margin_ratio"] > 0.3 and m["headroom_pct"] > 30.0   # ~33% tolerance at 3x


# --------------------------------------------------------------------------- #
# dedup + persistence
# --------------------------------------------------------------------------- #
def test_dedups_same_funding_interval(tmp_path):
    t = FundingCarryTrader(_cfg(), data_dir=tmp_path)
    t.step({"BTCUSDT": 0.0005}, _prices(), funding_time=5)
    t.step({"BTCUSDT": 0.0005}, _prices(), funding_time=5)      # same interval -> ignored
    t.step({"BTCUSDT": 0.0005}, _prices(), funding_time=4)      # older -> ignored
    assert t.cycle == 1


def test_persists_across_restart(tmp_path):
    t = FundingCarryTrader(_cfg(), data_dir=tmp_path)
    fund = {"BTCUSDT": 0.0005, "ETHUSDT": 0.0003}
    t.step(fund, _prices(), funding_time=1)
    t.step(fund, _prices(), funding_time=2)
    eq, acc, wt = t.status()["equity"], t.accrued, dict(t.weights)

    again = FundingCarryTrader(_cfg(), data_dir=tmp_path)       # reload
    assert abs(again.status()["equity"] - eq) < 1e-6
    assert abs(again.accrued - acc) < 1e-9 and again.weights == wt
    assert again.in_carry.get("BTCUSDT") is True
    again.step(fund, _prices(), funding_time=2)                 # stale -> ignored
    assert again.cycle == 2
    assert (tmp_path / "funding_live.csv").exists()


def test_load_live_status_for_dashboard(tmp_path):
    assert load_live_status(tmp_path) == {"enabled": False, "held": []}   # nothing yet
    t = FundingCarryTrader(_cfg(), data_dir=tmp_path)
    t.step({"BTCUSDT": 0.0005, "ETHUSDT": 0.0003}, _prices(), funding_time=1)
    st = load_live_status(tmp_path)
    assert st["enabled"] is True and st["n_held"] == 2 and "ann_yield_pct" in st
    assert st["margin"]["leverage"] == 3.0
