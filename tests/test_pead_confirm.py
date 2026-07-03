"""PEAD confirmation — non-overlapping logic, NW t-stat, live tracker. Mock, no net."""

import numpy as np
import pandas as pd

from tagent.pead_live import PeadLiveConfig, PeadLiveTrader
from tagent.strategies.earnings_drift import market_regime, newey_west_tstat, pead_trades


def _panel(sym_rows):
    out = {}
    for sym, recs in sym_rows.items():
        idx = pd.to_datetime([r[0] for r in recs])
        out[sym] = pd.DataFrame({"open": [r[1] for r in recs], "high": 1, "low": 1,
                                 "close": [r[2] for r in recs], "volume": 1}, index=idx)
    return out


# --------------------------------------------------------------------------- #
# non-overlapping logic
# --------------------------------------------------------------------------- #
def test_non_overlapping_skips_while_holding():
    idx = pd.bdate_range("2020-01-01", periods=20)
    o = [100.0 + 2 * i for i in range(20)]             # open gaps UP above prior close each day
    c = [100.0 + 2 * i + 1 for i in range(20)]         # rising -> positive reaction + drift
    panel = {"A": pd.DataFrame({"open": o, "high": 1, "low": 1, "close": c, "volume": 1}, index=idx)}
    # three earnings filings close together; hold=5 -> the 2nd/3rd fall inside the 1st hold
    ev = {"A": [idx[1], idx[3], idx[8]]}
    overlap = pead_trades(panel, ev, hold=5, conditional=True, non_overlapping=False)
    nonov = pead_trades(panel, ev, hold=5, conditional=True, non_overlapping=True)
    assert len(overlap) == 3
    # non-overlapping: entry1 @ idx[2], exit idx[7]; the idx[3] filing (entry idx[4]) is skipped;
    # the idx[8] filing (entry idx[9]) is after exit -> kept. So 2 trades.
    assert len(nonov) == 2
    entries = [pd.Timestamp(e).date() for e in nonov["entry"]]
    assert entries == [idx[2].date(), idx[9].date()]


def test_pead_trades_columns_and_no_lookahead_entry_after_filing():
    idx = pd.bdate_range("2020-01-02", periods=10)
    panel = _panel({"A": [(d, 100.0, 100.0) for d in idx]})
    panel["A"].loc[idx[2], "open"] = 102.0             # entry-day gap up (entry is the day AFTER filing idx[1])
    panel["A"].loc[idx[5], "close"] = 110.0
    tr = pead_trades(panel, {"A": [idx[1]]}, hold=3, conditional=True)
    assert list(tr.columns) == ["entry", "symbol", "ret", "r0"]
    assert tr["entry"].iloc[0] == idx[2] and tr["r0"].iloc[0] > 0


def test_market_regime_lagged_and_no_lookahead():
    idx = pd.bdate_range("2020-01-01", periods=40)
    close = list(range(100, 125)) + list(range(124, 109, -1))   # 25 up then 15 down
    panel = {"A": pd.DataFrame({"open": close, "high": 1, "low": 1, "close": close,
                               "volume": 1}, index=idx)}
    reg = market_regime(panel, ma_window=10)
    assert bool(reg.iloc[20]) is True and bool(reg.iloc[-1]) is False   # up-regime vs downtrend
    # appending a FUTURE bar must not change any past regime value (lagged, no-lookahead)
    ext_idx = pd.bdate_range("2020-01-01", periods=41)
    ext = {"A": pd.DataFrame({"open": close + [200], "high": 1, "low": 1,
                              "close": close + [200], "volume": 1}, index=ext_idx)}
    reg2 = market_regime(ext, ma_window=10)
    assert reg.equals(reg2.reindex(reg.index))


def test_pead_regime_gate_skips_downtrend_entries():
    idx = pd.bdate_range("2020-01-02", periods=10)
    panel = _panel({"A": [(d, 100.0, 100.0) for d in idx]})
    panel["A"].loc[idx[2], "open"] = 102.0                       # entry-day gap up -> r0>0
    panel["A"].loc[idx[5], "close"] = 110.0
    ev = {"A": [idx[1]]}
    on = pd.Series(True, index=idx)
    off = pd.Series(False, index=idx)
    assert len(pead_trades(panel, ev, hold=3, conditional=True, regime=off)) == 0   # downtrend -> skip
    assert len(pead_trades(panel, ev, hold=3, conditional=True, regime=on)) == 1    # in-market -> kept


def test_pead_live_regime_flag_in_status(tmp_path):
    tr = PeadLiveTrader(PeadLiveConfig(use_regime=True), data_dir=tmp_path)
    assert tr.status()["regime_gated"] is True
    assert PeadLiveTrader(PeadLiveConfig(), data_dir=tmp_path / "x").status()["regime_gated"] is False


def test_newey_west_tstat_shrinks_with_positive_autocorrelation():
    rng = np.random.default_rng(0)
    base = rng.normal(0.01, 0.02, 400)
    overlap = base + np.r_[0.0, base[:-1]]             # induce positive serial correlation
    plain = overlap.mean() / (overlap.std() / np.sqrt(len(overlap)))
    nw = newey_west_tstat(overlap, lag=5)
    assert abs(nw) < abs(plain)                        # HAC correction reduces the t-stat


# --------------------------------------------------------------------------- #
# live tracker accounting
# --------------------------------------------------------------------------- #
CFG = PeadLiveConfig(hold=20, capital=10_000.0, slippage_bps=15.0)   # round trip 0.51%


def test_pead_book_trade_net_of_cost_and_dedup(tmp_path):
    tr = PeadLiveTrader(CFG, data_dir=tmp_path)
    cf = CFG.cost().round_trip_frac()
    net = 0.03 - cf
    st = tr.book_trade("2024-02-01", "005930", 0.03)
    assert st["n_trades"] == 1
    assert abs(tr.equity - 10_000.0 * (1 + CFG.per_trade_frac * net)) < 1e-6   # fractional sizing
    assert abs(st["expectancy_pct"] - net * 100) < 1e-9                        # expectancy is per-trade net
    assert st["win_rate_pct"] == 100.0
    eq = tr.equity
    tr.book_trade("2024-02-01", "005930", 0.03)        # same (entry,symbol) -> deduped
    assert tr.equity == eq and tr.trades == 1


def test_pead_persistence_and_open_positions(tmp_path):
    a = PeadLiveTrader(CFG, data_dir=tmp_path)
    a.book_trade("2024-02-01", "005930", 0.02)
    a.book_trade("2024-02-05", "000660", -0.01)
    a.set_open_positions([{"symbol": "035420", "entry": "2024-06-01"}])
    b = PeadLiveTrader(CFG, data_dir=tmp_path)          # reload
    assert b.trades == 2 and abs(b.equity - a.equity) < 1e-9
    assert b.status()["n_trades"] == 2
    assert a.status()["n_open"] == 1 and a.status()["state"] == "ARMED"
    assert (tmp_path / "pead_live_state.json").exists() and (tmp_path / "pead_live.csv").exists()
