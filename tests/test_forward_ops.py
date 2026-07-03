"""Forward-operations layer — candidate promote/kill rules, no-lookahead, trigger monitor,
persistence. Synthetic, no network."""

import numpy as np
import pandas as pd

from tagent.forward_ops import (
    CANDIDATES, CandidateSpec, DailyAdvanceScheduler, KILL, PROMOTE, TRACKING,
    evaluate_candidate, load_advance_state, load_forward_ops, load_registry, save_registry,
    spanning_status, trigger_monitor, write_advance_state, write_forward_ops,
)

_SPEC = CandidateSpec("x", "X", min_months=12, kill_months=24)


def _daily(drift, noise, n=640, seed=0, start="2018-01-01"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    return pd.Series(drift + rng.normal(0.0, noise, n), index=idx)


# --------------------------------------------------------------------------- #
# 1) generic spanning rule: promote / kill / tracking
# --------------------------------------------------------------------------- #
def test_spanning_promote_when_ci_excludes_zero():
    fwd = _daily(0.001, 0.002, seed=1)                 # strong, consistent positive alpha
    bench = _daily(0.0, 0.01, seed=2)                  # uncorrelated market
    s = spanning_status(fwd, bench, _SPEC)
    assert s["status"] == PROMOTE and s["spanning"]["ci_excludes_zero"] is True
    assert s["spanning"]["alpha_ann_pct"] > 0


def test_spanning_kill_when_negative_over_kill_window():
    fwd = _daily(-0.0006, 0.003, n=640, seed=3)        # ~30 months, negative
    bench = _daily(0.0, 0.01, seed=4)
    s = spanning_status(fwd, bench, _SPEC)
    assert s["forward_equity_pct"] < 0 and s["forward_months"] >= 24
    assert s["status"] == KILL


def test_spanning_tracking_when_ci_includes_zero():
    # 18 months: enough to run the regression (>=12mo) but < kill window (24mo), so the
    # outcome is decided purely by the alpha CI -> noisy/insignificant alpha = keep tracking
    fwd = _daily(0.0001, 0.01, n=380, seed=5)
    bench = _daily(0.0, 0.01, n=380, seed=6)
    s = spanning_status(fwd, bench, _SPEC)
    assert 12 <= s["forward_months"] < 24
    assert s["status"] == TRACKING and s["spanning"]["ci_excludes_zero"] is False


def test_spanning_tracking_when_insufficient_history():
    fwd = _daily(0.001, 0.002, n=120, seed=7)          # ~6 months < 12
    s = spanning_status(fwd, _daily(0.0, 0.01, n=120, seed=8), _SPEC)
    assert s["status"] == TRACKING and "insufficient forward history" in s["note"]


# --------------------------------------------------------------------------- #
# 2) evaluate_candidate: forward window (no-lookahead) + satellite mapping
# --------------------------------------------------------------------------- #
def test_evaluate_candidate_forward_window_no_lookahead():
    rets = _daily(0.001, 0.002, n=640, seed=9)
    bench = _daily(0.0, 0.01, n=640, seed=10)
    mid = rets.index[400]
    c = evaluate_candidate(_SPEC, rets, bench, registered_on=mid)
    assert c["registered_on"] == str(mid.date())
    assert c["forward_days"] == int((rets.index >= mid).sum())   # only dates on/after registration
    assert c["in_sample_context"] is not None                    # full-sample context shown separately


def test_evaluate_candidate_satellite_status_mapping():
    spec = CandidateSpec("sat", "Sat", rule="satellite")
    for sat_status, want in (("PROMOTE", PROMOTE), ("KILL_ARCHIVE", KILL), ("FROZEN", TRACKING)):
        c = evaluate_candidate(spec, pd.Series(dtype=float), None, registered_on="2026-06-10",
                               satellite_eval={"status": sat_status, "reasons": ["r"]})
        assert c["status"] == want and c["rule"] == "satellite"


# --------------------------------------------------------------------------- #
# 3) trigger monitor
# --------------------------------------------------------------------------- #
def test_trigger_monitor_fires_on_promote_kill_and_reversal():
    statuses = [{"name": "vrp", "status": PROMOTE, "note": "CI excludes 0"},
                {"name": "bab", "status": KILL, "note": "neg 24mo"},
                {"name": "pead", "status": TRACKING}]
    sat = {"status": "FROZEN", "reasons": ["(b) concentration reversed (EW>CW ...) AND fwd ..."]}
    t = trigger_monitor(statuses, sat)
    assert t["any_triggered"] and t["n_triggers"] == 3        # promote + kill + satellite reversal
    kinds = {(x["who"], x["kind"]) for x in t["triggers"]}
    assert ("vrp", "promote") in kinds and ("bab", "kill") in kinds
    assert ("momentum-satellite", "promote") in kinds
    # nothing fires when all tracking and no reversal
    t2 = trigger_monitor([{"name": "vrp", "status": TRACKING}], {"status": "FROZEN", "reasons": ["no trigger"]})
    assert t2["any_triggered"] is False


# --------------------------------------------------------------------------- #
# 4) registry of pre-registered candidates + persistence
# --------------------------------------------------------------------------- #
def test_candidate_registry_covers_the_five_and_satellite_special():
    names = {c.name for c in CANDIDATES}
    assert {"vrp", "pead", "leadlag", "bab", "momentum-satellite"} <= names
    sat = next(c for c in CANDIDATES if c.name == "momentum-satellite")
    assert sat.rule == "satellite"


def test_forward_ops_and_registry_roundtrip(tmp_path):
    write_forward_ops({"enabled": True, "candidates": [{"name": "vrp"}]}, data_dir=tmp_path)
    assert load_forward_ops(data_dir=tmp_path)["enabled"] is True
    assert load_forward_ops(data_dir=tmp_path / "missing")["enabled"] is False   # absent -> disabled
    save_registry({"vrp": "2026-06-10"}, data_dir=tmp_path)
    assert load_registry(data_dir=tmp_path)["vrp"] == "2026-06-10"


# --------------------------------------------------------------------------- #
# 5) daily auto-advance scheduler: once per new day, throttled, idempotent, no-lookahead
# --------------------------------------------------------------------------- #
def test_daily_scheduler_fires_once_per_new_day():
    """Fires the FIRST tick of a new calendar day and never again that day — the once/day
    guard reads the persisted day-done so a fresh process can't double-advance either."""
    clk = {"t": pd.Timestamp("2026-06-11 00:00:00")}     # 2026-06-11 is a Thursday
    done = {"day": None}                                  # stands in for persisted last_advanced
    calls = []

    def advance():
        d = clk["t"].strftime("%Y-%m-%d")
        calls.append(d)
        done["day"] = d                                  # the advance persists today's stamp

    sch = DailyAdvanceScheduler(advance, day_done_fn=lambda: done["day"],
                                clock=lambda: clk["t"], min_interval_s=0.0)
    assert sch.tick() is True                             # day 1 -> fires once
    assert sch.tick() is False                            # same day -> no double-count
    clk["t"] = pd.Timestamp("2026-06-11 18:00:00")
    assert sch.tick() is False                            # still the same calendar day
    clk["t"] = pd.Timestamp("2026-06-12 09:00:00")
    assert sch.tick() is True                             # new day -> fires
    assert calls == ["2026-06-11", "2026-06-12"]

    # a brand-new scheduler (fresh process) on the same day must NOT re-advance: the
    # persisted day-done already names today.
    fresh = DailyAdvanceScheduler(advance, day_done_fn=lambda: done["day"],
                                  clock=lambda: clk["t"], min_interval_s=0.0)
    assert fresh.tick() is False
    assert calls == ["2026-06-11", "2026-06-12"]


def test_daily_scheduler_throttles_within_interval():
    clk = {"t": pd.Timestamp("2026-06-11 00:00:00")}
    calls = []
    sch = DailyAdvanceScheduler(lambda: calls.append(1), clock=lambda: clk["t"],
                                min_interval_s=3600.0)
    assert sch.tick() is True and len(calls) == 1
    clk["t"] = pd.Timestamp("2026-06-11 00:30:00")        # 30 min later, < 1h throttle
    assert sch.tick() is False and len(calls) == 1        # throttled, not re-run


def test_advance_state_roundtrip_and_next_business_day(tmp_path):
    # absent -> nulls
    assert load_advance_state(data_dir=tmp_path)["last_advanced"] is None
    # Thursday 2026-06-11 advanced -> next business day is Friday 2026-06-12
    write_advance_state("2026-06-11", as_of="2026-06-10", ran=["trend_core", "forward_ops"],
                        data_dir=tmp_path)
    a = load_advance_state(data_dir=tmp_path)
    assert a["last_advanced"] == "2026-06-11" and a["next_advance"] == "2026-06-12"
    assert a["as_of"] == "2026-06-10" and a["ran"] == ["trend_core", "forward_ops"]
    # Friday -> next business day skips the weekend to Monday 2026-06-15
    write_advance_state("2026-06-12", data_dir=tmp_path)
    assert load_advance_state(data_dir=tmp_path)["next_advance"] == "2026-06-15"


def test_scheduler_drives_trend_core_idempotently_and_no_lookahead(tmp_path):
    """End-to-end: the scheduler drives the REAL trend-core advance unit. Two ticks on the
    same day add no audit rows (idempotent), and the shakedown anchor never moves backward
    while new bars only extend it forward (no-lookahead)."""
    from tagent.trend_core_live import TrendCoreLiveTrader, load_live_status

    def closes(n):
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        out = {}
        for i, m in enumerate(("KOSPI200", "S&P500")):
            rng = np.random.default_rng(i)
            r = 0.0012 + rng.normal(0.0, 0.003, n)
            out[m] = pd.Series(100 * np.cumprod(1 + np.r_[0.0, r[:-1]]), index=idx)
        return out

    state = {"n": 300, "day": "2026-06-11"}

    def advance():
        st = TrendCoreLiveTrader(data_dir=tmp_path).update(closes(state["n"]))
        write_advance_state(state["day"], as_of=st["as_of"], ran=["trend_core"], data_dir=tmp_path)

    sch = DailyAdvanceScheduler(
        advance, day_done_fn=lambda: load_advance_state(data_dir=tmp_path).get("last_advanced"),
        clock=lambda: pd.Timestamp(state["day"] + " 09:00"), min_interval_s=0.0)

    csv = tmp_path / "trend_core_live.csv"
    assert sch.tick() is True                              # day 1 advance
    n_rows1 = len(csv.read_text(encoding="utf-8").splitlines())
    start1 = load_live_status(data_dir=tmp_path)["shakedown"]["start"]
    assert sch.tick() is False                             # same day -> idempotent
    assert len(csv.read_text(encoding="utf-8").splitlines()) == n_rows1   # no new rows

    # next day, 5 genuinely-new bars arrive -> advance extends the log forward only
    state["n"], state["day"] = 305, "2026-06-12"
    assert sch.tick() is True
    lines = csv.read_text(encoding="utf-8").splitlines()
    assert len(lines) == n_rows1 + 5                       # exactly the 5 new trading days
    dates = [ln.split(",")[0] for ln in lines[1:]]
    assert len(dates) == len(set(dates))                   # no backfilled / duplicate dates
    sk = load_live_status(data_dir=tmp_path)["shakedown"]
    assert sk["start"] == start1                           # anchor unchanged (no-lookahead)
    assert sk["days_validated"] == 6                       # original day + 5 new, all >= anchor
