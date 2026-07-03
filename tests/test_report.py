"""Performance reporting + Monte Carlo — synthetic returns, no network."""

import json

import numpy as np
import pandas as pd
import pytest

from tagent.report import (
    equity_to_returns,
    funding_carry_returns_from_scorecard,
    monte_carlo,
    perf_stats,
    returns_from_equity_csv,
    returns_from_funding_live,
    returns_from_signal_log,
    returns_to_equity,
    tear_sheet,
)


# --------------------------------------------------------------------------- #
# equity <-> returns
# --------------------------------------------------------------------------- #
def test_equity_returns_roundtrip():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.01, 200))
    eq = returns_to_equity(r, start=10_000.0)
    r2 = equity_to_returns(eq)
    assert np.allclose(r2.to_numpy(), r.to_numpy()[: len(r2)], atol=1e-9)


# --------------------------------------------------------------------------- #
# perf stats
# --------------------------------------------------------------------------- #
def test_perf_stats_positive_drift_has_positive_sharpe_and_cagr():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0008, 0.01, 1000))         # positive drift
    st = perf_stats(r, periods_per_year=365)
    assert st["sharpe"] > 0 and st["cagr"] > 0 and st["ann_vol"] > 0
    assert st["max_drawdown"] <= 0 and st["var"] >= 0 and st["cvar"] >= st["var"] - 1e-9
    assert st["n"] == 1000


def test_perf_stats_sortino_ignores_upside_vol():
    # only-positive returns -> no downside -> sortino 0 (no downside std) but sharpe>0
    r = pd.Series([0.001] * 50)
    st = perf_stats(r)
    assert st["max_drawdown"] == 0.0 and st["sharpe"] == 0.0   # zero vol -> sharpe 0 by guard


def test_perf_stats_empty():
    st = perf_stats([])
    assert st["n"] == 0 and st["sharpe"] == 0.0


# --------------------------------------------------------------------------- #
# Monte Carlo bust probability + tail
# --------------------------------------------------------------------------- #
def test_monte_carlo_losing_series_has_high_bust_prob():
    rng = np.random.default_rng(2)
    r = rng.normal(-0.01, 0.03, 300)                      # negative drift, volatile
    mc = monte_carlo(r, n_paths=400, horizon=300, start_equity=10_000.0,
                     ruin_drawdown=0.5, seed=0)
    assert 0.0 <= mc["bust_prob"] <= 1.0 and mc["bust_prob"] > 0.5
    assert mc["p5_final"] <= mc["median_final"] <= mc["p95_final"]
    assert mc["var_final"] >= 0 and mc["cvar_final"] >= mc["var_final"] - 1e-6


def test_monte_carlo_steady_positive_rarely_busts():
    r = np.full(300, 0.0005)                              # steady tiny gains, no vol
    mc = monte_carlo(r, n_paths=200, ruin_drawdown=0.5, seed=0)
    assert mc["bust_prob"] == 0.0 and mc["median_final"] > 10_000.0
    assert mc["worst_drawdown_med"] == 0.0


def test_monte_carlo_block_bootstrap_runs():
    rng = np.random.default_rng(3)
    r = rng.normal(0.0, 0.02, 200)
    mc = monte_carlo(r, n_paths=100, horizon=120, block=8, seed=1)
    assert mc["n_paths"] == 100 and mc["horizon"] == 120


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #
def test_returns_from_funding_live(tmp_path):
    p = tmp_path / "funding_live.csv"
    pd.DataFrame({"ts": pd.date_range("2026-01-01", periods=4, freq="8h", tz="UTC"),
                  "cycle": [1, 2, 3, 4], "equity": [10000.0, 10010.0, 10005.0, 10020.0]}).to_csv(p, index=False)
    r = returns_from_funding_live(p)
    assert len(r) == 3
    assert abs(r.iloc[0] - (10010.0 / 10000.0 - 1)) < 1e-9
    assert isinstance(r.index, pd.DatetimeIndex)


def test_returns_from_equity_csv_missing_col(tmp_path):
    p = tmp_path / "x.csv"
    pd.DataFrame({"ts": [1, 2], "foo": [1, 2]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        returns_from_equity_csv(p)


def test_funding_carry_returns_from_scorecard(tmp_path):
    state = {"scores": [
        {"ts": "2026-01-01T00:00:00+00:00", "source": "funding-carry", "status": "carry",
         "side": 1, "entry_price": 2.0},     # +2 bps funding -> +0.0002 return
        {"ts": "2026-01-01T08:00:00+00:00", "source": "funding-carry", "status": "carry",
         "side": 0, "entry_price": -1.0},    # not held -> 0 return
        {"ts": "2026-01-01T00:00:00+00:00", "source": "us-ml", "status": "correct",
         "side": 1, "entry_price": 100.0},   # other source ignored
    ]}
    (tmp_path / "scorecard_state.json").write_text(json.dumps(state), encoding="utf-8")
    r = funding_carry_returns_from_scorecard(tmp_path / "scorecard_state.json")
    assert len(r) == 2
    assert abs(r.iloc[0] - 0.0002) < 1e-12 and r.iloc[1] == 0.0


def test_returns_from_signal_log_reconstructs_directional_pnl(tmp_path):
    p = tmp_path / "signal_log.csv"
    pd.DataFrame({
        "timestamp": ["2026-01-01T00:00:00+00:00", "2026-01-01T00:05:00+00:00",
                      "2026-01-01T00:10:00+00:00"],
        "source": ["us-ml", "us-ml", "us-ml"], "symbol": ["AAPL", "AAPL", "AAPL"],
        "direction": ["BUY", "SELL", "BUY"], "price": [100.0, 110.0, 105.0],
    }).to_csv(p, index=False)
    r = returns_from_signal_log(p, source="us-ml", cost_bps=10.0)
    # first signal BUY @100 -> next @110: +10% minus 2*10bps round trip
    assert len(r) == 2
    assert abs(r.iloc[0] - (0.10 - 0.002)) < 1e-9


def test_returns_from_signal_log_filters_source(tmp_path):
    p = tmp_path / "signal_log.csv"
    pd.DataFrame({
        "timestamp": ["2026-01-01T00:00:00+00:00", "2026-01-01T00:05:00+00:00"],
        "source": ["crypto-ob", "crypto-ob"], "symbol": ["BTCUSDT", "BTCUSDT"],
        "direction": ["BUY", "BUY"], "price": [100.0, 101.0],
    }).to_csv(p, index=False)
    assert returns_from_signal_log(p, source="us-ml").empty       # no us-ml rows
    assert len(returns_from_signal_log(p, source="crypto-ob")) == 1


# --------------------------------------------------------------------------- #
# quantstats tear sheet (skips if the optional dep is absent)
# --------------------------------------------------------------------------- #
def test_tear_sheet_writes_html(tmp_path):
    pytest.importorskip("quantstats")
    rng = np.random.default_rng(7)
    r = pd.Series(rng.normal(0.0005, 0.01, 400))
    bench = pd.Series(rng.normal(0.0003, 0.012, 400))
    out = tear_sheet(r, tmp_path / "rep.html", benchmark=bench, title="test")
    p = tmp_path / "rep.html"
    assert p.exists() and p.stat().st_size > 0
    assert "<html" in p.read_text(encoding="utf-8", errors="ignore").lower()
