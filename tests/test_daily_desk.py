"""Daily Desk — recommendations, briefing, session monitor, paper track. Mock, no net."""

import numpy as np
import pandas as pd

import pytest

from tagent.daily_desk import (
    DeskConfig, DeskTracker, assert_sane, briefing_row, buy_candidates, momentum_ranks,
    sanity_check, sell_list, session_signals,
)


def _panel(n=90, drifts=(-0.002, 0.0, 0.001, 0.002, 0.003, 0.004), seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2019-01-01", periods=n, freq="B")
    out = {}
    for i, dr in enumerate(drifts):
        steps = rng.normal(dr, 0.004, n)
        close = 100 * np.cumprod(1 + np.r_[0.0, steps[:-1]])
        out[f"00{i}0"] = pd.DataFrame({"open": close, "high": close, "low": close,
                                       "close": close, "volume": [1000 + 10 * i] * n}, index=idx)
    return out


def _cfg(panel, **kw):
    base = dict(watchlist=tuple(panel), lookback=20, skip=5, recent_window=5,
                regime_ma=20, top_n=3, top_group=3)
    base.update(kw)
    return DeskConfig(**base)


# --------------------------------------------------------------------------- #
# 1) recommendations
# --------------------------------------------------------------------------- #
def test_momentum_ranks_orders_by_12_1_and_no_lookahead():
    panel = _panel()
    cfg = _cfg(panel)
    r = momentum_ranks(panel, cfg)
    assert list(r["symbol"])[0] == "0050"                  # highest-drift name ranks #1
    assert r["rank"].tolist() == list(range(1, len(r) + 1))
    # no-lookahead: ranks as of a past date don't change when future bars are appended
    asof = align_idx = pd.Timestamp(next(iter(panel.values())).index[60])
    r60 = momentum_ranks(panel, cfg, asof=asof)
    ext = {s: pd.concat([d, d.tail(20) * 9]) for s, d in panel.items()}    # garbage future
    r60b = momentum_ranks(ext, cfg, asof=asof)
    assert r60["symbol"].tolist() == r60b["symbol"].tolist()


def test_buy_candidates_have_reason_stop_and_label():
    panel = _panel()
    cfg = _cfg(panel)
    bc = buy_candidates(panel, cfg)
    assert bc["regime"] == "in-market" and len(bc["buys"]) == cfg.top_n
    b = bc["buys"][0]
    assert b["stop"] < b["price"] and "momentum rank #1" in b["reason"]
    assert "momentum" in b["basis"]


def test_buy_candidates_downtrend_withholds_buys():
    panel = _panel(drifts=(-0.01, -0.008, -0.006, -0.009, -0.007, -0.011))   # all falling
    cfg = _cfg(panel)
    bc = buy_candidates(panel, cfg)
    assert bc["regime"] == "cash" and bc["buys"] == [] and "CASH" in bc["note"]


def test_sell_list_reasons():
    panel = _panel()
    cfg = _cfg(panel)
    close = pd.DataFrame({c: d["close"] for c, d in panel.items()})
    # hold the WORST momentum name (0000) -> dropped out; + a stop broken on 0050
    holdings = ["0000", "0050"]
    stops = {"0050": float(close["0050"].iloc[-1]) * 1.10}   # stop above price -> broke stop
    sells = sell_list(holdings, panel, cfg, entry_stops=stops)
    by = {s["symbol"]: s["reasons"] for s in sells}
    assert "0000" in by and any("dropped out" in r for r in by["0000"])
    assert "0050" in by and any("broke stop" in r for r in by["0050"])


# --------------------------------------------------------------------------- #
# 2) in-session monitor
# --------------------------------------------------------------------------- #
def test_session_signals_assembles_price_rsi_vol_and_labels():
    idx = pd.date_range("2026-06-10 09:00", periods=40, freq="min")
    close = 100 + np.arange(40) * 0.1 + np.where(np.arange(40) % 2 == 0, 0.3, -0.3)  # zigzag up
    df = pd.DataFrame({"open": close, "high": close + 0.1, "low": close - 0.1,
                       "close": close, "volume": [100] * 39 + [500]}, index=idx)
    sig = session_signals("000660", df, vol_window=20, us_armed=True, safety="OK")
    assert sig["price"] == round(float(close[-1]), 2)
    assert sig["rsi"] is not None and sig["vol_vs_avg"] > 1     # last bar volume spike
    assert sig["us_shock"] == "ARMED" and sig["safety"] == "OK"
    assert "decision-support" in sig["label"] and sig["candle"]["color"] in ("red", "blue", "doji")


def test_session_signals_empty_bars():
    sig = session_signals("005930", pd.DataFrame(), us_armed=False)
    assert sig["price"] is None and sig["n_bars"] == 0 and sig["us_shock"] == "OK"


# --------------------------------------------------------------------------- #
# 3) briefing
# --------------------------------------------------------------------------- #
def test_briefing_row_change_and_vol_vs_avg():
    idx = pd.date_range("2026-01-01", periods=30, freq="B")
    df = pd.DataFrame({"open": 100.0, "high": 1, "low": 1,
                       "close": [100.0] * 29 + [105.0], "volume": [1000] * 29 + [3000]}, index=idx)
    row = briefing_row("000660", df, news_sentiment="bullish", short_ratio=0.04)
    assert row["price"] == 105.0 and abs(row["change"] - 0.05) < 1e-9
    assert row["vol_vs_avg"] > 2 and row["news"] == "bullish" and row["short_ratio"] == 0.04


# --------------------------------------------------------------------------- #
# 4) paper track
# --------------------------------------------------------------------------- #
def test_desk_tracker_book_net_of_cost_dedup_persist(tmp_path):
    cfg = DeskConfig(slippage_bps=15.0, per_trade_frac=0.05)
    tr = DeskTracker(cfg, data_dir=tmp_path)
    cf = cfg.cost().round_trip_frac()
    st = tr.book("2024-02-01", "000660", 0.04)
    assert st["n_trades"] == 1 and abs(tr.equity - 10_000.0 * (1 + 0.05 * (0.04 - cf))) < 1e-6
    eq = tr.equity
    tr.book("2024-02-01", "000660", 0.04)                  # dedup
    assert tr.equity == eq and tr.trades == 1
    b = DeskTracker(cfg, data_dir=tmp_path)                 # persistence
    assert b.trades == 1 and abs(b.equity - eq) < 1e-9
    assert (tmp_path / "daily_desk_track.json").exists()


# --------------------------------------------------------------------------- #
# 5) data-quality guard — corrupt (split-artifact) prices must fail loudly
# --------------------------------------------------------------------------- #
def test_sanity_check_passes_clean_and_flags_split_artifact():
    panel = _panel()                                        # smooth random walks -> clean
    assert sanity_check(panel) == {}
    assert_sane(panel)                                      # does not raise
    # inject a ~9x split-artifact jump (mixed adjusted/unadjusted) into one name
    bad = {s: d.copy() for s, d in panel.items()}
    victim = "0010"
    bad[victim].iloc[-1, bad[victim].columns.get_loc("close")] *= 9.0
    flagged = sanity_check(bad)
    assert victim in flagged and abs(flagged[victim][0][1]) > 0.35   # the >35% move surfaces
    with pytest.raises(ValueError, match="sanity check failed"):
        assert_sane(bad)


def test_real_watchlist_cache_is_clean():
    """The cached KR watchlist daily files must contain no implausible (split-artifact)
    moves — a regression guard so corrupted prices can never silently feed the desk."""
    from tagent.stock_momentum import load_stock_panel
    panel = load_stock_panel("kr", symbols=list(DeskConfig().watchlist),
                             fields=["close"], min_bars=60)
    if not panel:                                           # cache not present in this env
        pytest.skip("no cached KR watchlist data")
    assert sanity_check(panel) == {}
