"""Post-earnings/disclosure drift (PEAD) — hold after a positive earnings disclosure.

For each earnings-type OpenDART disclosure on date D, enter at the FIRST tradable open
strictly after D (filings are treated as known only by end of day D, so we never trade
the filing day itself — no-lookahead). The "positive surprise" proxy is the
announcement reaction r0 = entry_open / prev_close - 1 (both known at the entry open);
optionally take only r0 > 0. Then measure the DRIFT = exit_close / entry_open - 1 over
``hold`` trading days, net of one KR round trip.

Pure event study; disclosures + prices are injected, so it is unit-tested with mock
data and no network.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from tagent.strategies.us_leadlag_daily import _wide, summarize  # noqa: F401  (re-export)


def earnings_events(disclosures: Sequence[dict],
                    types: Sequence[str] = ("earnings",)) -> Dict[str, List[pd.Timestamp]]:
    """{symbol: sorted unique filing dates} for the earnings-type disclosures."""
    out: Dict[str, set] = {}
    for d in disclosures:
        if d.get("type") in types and d.get("symbol"):
            out.setdefault(str(d["symbol"]).zfill(6), set()).add(pd.Timestamp(d.get("time")))
    return {s: sorted(ds) for s, ds in out.items()}


def market_regime(panel, ma_window: int = 200, membership: Optional[pd.DataFrame] = None) -> pd.Series:
    """Boolean per-date market-regime gate: is the broad KR market (equal-weight basket
    NAV) AT/ABOVE its ``ma_window`` moving average? LAGGED one bar so the regime known
    at the entry open uses only data through the prior close (strictly no-lookahead).
    During the MA warmup -> True (no confirmed downtrend yet)."""
    closes = _wide(panel, "close")
    closes = closes.where(closes > 0)
    if membership is not None:
        m = membership.reindex(index=closes.index, columns=closes.columns).fillna(False)
        closes = closes.where(m.to_numpy(dtype=bool))
    basket = closes.pct_change().mean(axis=1)              # EW market return
    nav = (1.0 + basket.fillna(0.0)).cumprod()
    nav_lag = nav.shift(1)                                 # known at the bar's open
    sma = nav_lag.rolling(ma_window, min_periods=ma_window).mean()
    on = nav_lag >= sma
    on[sma.isna()] = True                                 # warmup -> in-market
    return on


def pead_trades(panel, events_by_sym: Dict[str, List], hold: int = 10,
                conditional: bool = True, membership: Optional[pd.DataFrame] = None,
                non_overlapping: bool = False, regime: Optional[pd.Series] = None) -> pd.DataFrame:
    """Per-event PEAD trades as a DataFrame [entry, symbol, ret, r0].

    Entry = first tradable open strictly AFTER the filing date (no-lookahead). r0 =
    entry_open/prev_close-1 (announcement reaction, known at entry). ret =
    exit_close/entry_open-1 over ``hold`` days. ``conditional`` keeps only r0>0.
    ``non_overlapping`` holds at most ONE position per name at a time (skips a new
    signal while that name is still held) so the per-name returns are independent.
    ``regime`` (optional boolean Series by date) gates trades to in-market days only
    (skip a trade whose entry is in a confirmed downtrend) — strictly no-lookahead when
    the series is lagged (see :func:`market_regime`).
    """
    opens, closes = _wide(panel, "open"), _wide(panel, "close")
    opens, closes = opens.where(opens > 0), closes.where(closes > 0)
    idx = closes.index
    memb = (membership.reindex(index=idx, columns=closes.columns).fillna(False)
            if membership is not None else None)
    reg = regime.reindex(idx) if regime is not None else None
    recs: List[dict] = []
    for sym, dates in events_by_sym.items():
        if sym not in closes.columns:
            continue
        o, c = opens[sym], closes[sym]
        last_exit_i = -1
        for D in sorted(pd.Timestamp(x) for x in dates):
            after = idx[idx > D]
            before = idx[idx <= D]
            if len(after) <= hold or len(before) == 0:
                continue
            entry = after[0]
            ei = idx.get_loc(entry)
            if ei + hold >= len(idx):
                continue
            if non_overlapping and ei <= last_exit_i:
                continue                                  # still holding this name -> skip
            entry_open, prev_close, exit_close = o.loc[entry], c.loc[before[-1]], c.iloc[ei + hold]
            if pd.isna(entry_open) or pd.isna(prev_close) or prev_close <= 0 or pd.isna(exit_close):
                continue
            if memb is not None and not bool(memb.at[entry, sym]):
                continue                                  # not a member at entry -> skip (PIT)
            if reg is not None and not bool(reg.get(entry, True)):
                continue                                  # market in a downtrend -> skip (regime gate)
            r0 = entry_open / prev_close - 1.0
            if conditional and r0 <= 0:
                continue
            recs.append({"entry": entry, "symbol": sym,
                         "ret": exit_close / entry_open - 1.0, "r0": r0})
            last_exit_i = ei + hold                       # block overlapping entries for this name
    df = pd.DataFrame(recs, columns=["entry", "symbol", "ret", "r0"])
    return df.sort_values("entry").reset_index(drop=True)


def pead_event_returns(panel, events_by_sym: Dict[str, List], hold: int = 10,
                       conditional: bool = True, membership: Optional[pd.DataFrame] = None,
                       non_overlapping: bool = False, regime: Optional[pd.Series] = None) -> pd.Series:
    """Per-event drift returns as a Series indexed by entry date (see pead_trades)."""
    df = pead_trades(panel, events_by_sym, hold, conditional, membership, non_overlapping, regime)
    return pd.Series(df["ret"].to_numpy(), index=pd.DatetimeIndex(df["entry"])) if len(df) \
        else pd.Series(dtype=float)


def newey_west_tstat(returns, lag: int) -> float:
    """t-stat of the mean with Newey-West (Bartlett) HAC correction for serial
    correlation up to ``lag`` (use lag~hold to account for overlapping holds)."""
    x = np.asarray(pd.Series(returns).dropna(), float)
    n = len(x)
    if n < 3:
        return 0.0
    e = x - x.mean()
    s = float(e @ e) / n
    for k in range(1, min(lag, n - 1) + 1):
        w = 1.0 - k / (lag + 1.0)
        s += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = (s / n) ** 0.5
    return float(x.mean() / se) if se > 0 else 0.0
