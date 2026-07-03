"""Classic 12-1 stock momentum — synthetic panels + CSV loaders, no network."""

import numpy as np
import pandas as pd

from tagent.stock_momentum import (
    KR_COST, classic_config, load_benchmark, load_stock_panel,
)
from tagent.xs_momentum import XSMomConfig, backtest, momentum_signal, _weights
from tagent.xs_momentum_validate import walk_forward


def _coin(prices, start="2016-01-04"):
    idx = pd.date_range(start, periods=len(prices), freq="B", tz=None)
    return pd.DataFrame({"close": np.asarray(prices, float)}, index=idx)


def _panel(n=900, k=12, seed=0, persist=True):
    rng = np.random.default_rng(seed)
    drifts = np.linspace(-0.001, 0.001, k)
    out = {}
    for i in range(k):
        steps = rng.normal(drifts[i], 0.02, n)
        if not persist:
            steps *= np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
        out[f"S{i}"] = _coin(100 * np.cumprod(1 + np.r_[0.0, steps[:-1]]))
    return out


def _wide(panel):
    return pd.DataFrame({c: df["close"] for c, df in panel.items()})


# --------------------------------------------------------------------------- #
# classic 12-1 config + signal
# --------------------------------------------------------------------------- #
def test_classic_config_is_12_1_monthly_long_only_for_kr():
    cfg = classic_config("kr")
    assert cfg.lookback == 252 and cfg.skip_recent == 21 and cfg.rebalance == 21
    assert cfg.allow_short is False                        # KR shorting restricted -> long-only
    assert cfg.cost_bps == KR_COST["cost_bps"]            # KR costs (commission + sell tax + spread)


def test_us_config_long_short_with_us_costs():
    cfg = classic_config("us", allow_short=True)
    assert cfg.allow_short is True and cfg.cost_bps == 2.0


def test_12_1_signal_skips_recent_month():
    # +1%/day for 300 days; 12-1 = return over [t-21-252, t-21], excluding last 21d
    px = 100 * np.cumprod(np.full(300, 1.01))
    close = _wide({"A": _coin(px)})
    sig = momentum_signal(close, lookback=252, skip_recent=21)
    t = 290
    assert abs(sig["A"].iloc[t] - (px[t - 21] / px[t - 21 - 252] - 1.0)) < 1e-9
    assert pd.isna(sig["A"].iloc[200])                    # not enough history (need 273 bars)


# --------------------------------------------------------------------------- #
# monthly rebalance + long-only variant
# --------------------------------------------------------------------------- #
def test_monthly_rebalance_holds_weights_21_days():
    w = _weights(_wide(_panel(n=600)), classic_config("kr"))
    for t in range(1, len(w)):
        if t % 21 != 0:
            assert np.allclose(w.iloc[t].to_numpy(), w.iloc[t - 1].to_numpy())


def test_long_only_has_no_shorts_and_sums_to_one_when_invested():
    w = _weights(_wide(_panel(n=600)), classic_config("kr", allow_short=False))
    invested = w[(w != 0).any(axis=1)]
    assert (invested.to_numpy() >= -1e-12).all()          # never short
    assert np.allclose(invested.sum(axis=1).to_numpy(), 1.0)  # fully long, sums to 1


def test_long_only_vs_long_short_both_run_and_report_basket():
    panel = _panel(n=700, persist=True)
    lo = backtest(panel, classic_config("kr", allow_short=False), periods_per_year=252)
    ls = backtest(panel, classic_config("kr", allow_short=True), periods_per_year=252)
    assert "basket_stats" in lo and lo["n_bars"] > 0      # long-only vs index comparison available
    assert lo["weights"].min().min() >= -1e-12            # long-only has no short weights
    assert ls["weights"].min().min() < 0                  # long-short does


# --------------------------------------------------------------------------- #
# walk-forward no-lookahead on a stock panel
# --------------------------------------------------------------------------- #
def test_walk_forward_no_lookahead_on_stocks():
    panel = _panel(n=900, persist=True, seed=2)
    cfg = classic_config("kr", allow_short=False)
    wf1 = walk_forward(panel, lookbacks=(126, 252), train_bars=400, test_bars=120,
                       cfg=cfg, periods_per_year=252)
    ext = {c: pd.concat([df, _coin([df["close"].iloc[-1] * 2] * 60, start="2024-01-01")])
           for c, df in panel.items()}
    wf2 = walk_forward(ext, lookbacks=(126, 252), train_bars=400, test_bars=120,
                       cfg=cfg, periods_per_year=252)
    assert wf1["n_folds"] >= 1
    assert wf1["folds"][0]["chosen_lookback"] == wf2["folds"][0]["chosen_lookback"]
    assert abs(wf1["folds"][0]["test_return"] - wf2["folds"][0]["test_return"]) < 1e-9
    for f in wf1["folds"]:
        assert f["train_end"] < f["test_start"]           # train precedes test


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #
def _write_csv(tmp, name, prices, start="2016-01-04"):
    idx = pd.date_range(start, periods=len(prices), freq="B")
    pd.DataFrame({"timestamp": idx, "open": prices, "high": prices, "low": prices,
                  "close": prices, "volume": 1000}).to_csv(tmp / f"{name}_1d.csv", index=False)


def test_load_stock_panel_kr_vs_us(tmp_path):
    _write_csv(tmp_path, "005930", 100 + np.arange(400))      # KR 6-digit
    _write_csv(tmp_path, "000660", 50 + np.arange(400))       # KR
    _write_csv(tmp_path, "AAPL", 200 + np.arange(400))        # US
    _write_csv(tmp_path, "SPY", 300 + np.arange(400))         # benchmark, excluded
    _write_csv(tmp_path, "klines_BTCUSDT", 100 + np.arange(400))  # crypto, excluded
    kr = load_stock_panel("kr", data_dir=tmp_path, min_bars=100)
    us = load_stock_panel("us", data_dir=tmp_path, min_bars=100)
    assert set(kr) == {"005930", "000660"} and "close" in kr["005930"].columns
    assert set(us) == {"AAPL"}                                # SPY + crypto excluded


def test_load_stock_panel_respects_min_bars(tmp_path):
    _write_csv(tmp_path, "005930", 100 + np.arange(50))       # too short
    assert load_stock_panel("kr", data_dir=tmp_path, min_bars=300) == {}


def test_load_benchmark_returns_series(tmp_path):
    _write_csv(tmp_path, "SPY", 100 * np.cumprod(np.full(120, 1.001)))
    r = load_benchmark("SPY", data_dir=tmp_path)
    assert r is not None and len(r) == 119
    assert load_benchmark("MISSING", data_dir=tmp_path) is None
