"""Event-driven minute backtester with realistic KR costs — honest by construction.

A strategy looks at one trading day's minute bars (+ context: previous close, US
overnight return) and returns at most one :class:`DayPlan` (entry bar + take/stop).
The backtester then walks the minute bars FORWARD from the entry, exiting on the
stop, the take-profit, or the close — whichever comes first — and books the trade
NET of realistic Korean round-trip costs (0.18% sell tax + commission + slippage).

No-lookahead is structural: the plan is formed from bar 0 / pre-open context, the
exit scan only reads bars at or after entry, and ``prev_close`` comes from the prior
day only. Walk-forward selects parameters on TRAIN months and applies them on the
next unseen TEST month.

Pure numpy/pandas; unit-tested on mock minute data with no network.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Callable, Dict, List, Optional, Sequence

import pandas as pd


@dataclass
class CostModel:
    """Per-side costs in basis points. KR charges a sell-side securities tax."""
    fee_bps: float = 1.5         # commission per side
    slippage_bps: float = 10.0   # ~0.1% slippage per side
    sell_tax_bps: float = 18.0   # ~0.18% securities transaction tax (sell only)

    def round_trip_frac(self) -> float:
        """Total round-trip cost as a fraction of notional: buy (fee+slip) + sell
        (fee+slip+tax)."""
        return (2.0 * self.fee_bps + 2.0 * self.slippage_bps + self.sell_tax_bps) / 1e4


@dataclass
class DayPlan:
    entry_index: int             # bar offset within the day to enter at
    entry_price: float           # executable entry price (that bar's open)
    take_pct: float              # +Y take-profit
    stop_pct: float              # -Z stop-loss


@dataclass
class Trade:
    date: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    gross: float
    net: float
    reason: str                  # take | stop | close
    symbol: str = ""


# strategy: (day_df, ctx) -> Optional[DayPlan]; ctx has prev_close, overnight_return, date
Strategy = Callable[[pd.DataFrame, dict], Optional[DayPlan]]


def group_days(minute_df: pd.DataFrame) -> List[dict]:
    """Split a minute panel into ordered per-day records, each carrying the PRIOR
    day's close (None for the first day) — no-lookahead context."""
    df = minute_df.sort_index()
    recs: List[dict] = []
    prev_close: Optional[float] = None
    for day, g in df.groupby(df.index.normalize()):
        recs.append({"date": pd.Timestamp(day).date(), "df": g, "prev_close": prev_close})
        prev_close = float(g["close"].iloc[-1])
    return recs


def _overnight_lookup(overnight, date) -> Optional[float]:
    if overnight is None:
        return None
    key = pd.Timestamp(date).date()
    if hasattr(overnight, "get"):
        v = overnight.get(key)
        if v is None and hasattr(overnight, "index"):       # a Series indexed by date/ts
            try:
                v = float(overnight.loc[pd.Timestamp(date)])
            except Exception:
                v = None
        return None if v is None or (isinstance(v, float) and pd.isna(v)) else float(v)
    return None


def simulate_day(day_df: pd.DataFrame, plan: DayPlan, cost_frac: float,
                 symbol: str = "") -> Trade:
    """Walk the day's bars from entry, exit on stop / take / close (stop checked
    first within a bar = pessimistic), and book the trade net of ``cost_frac``."""
    high = day_df["high"].to_numpy(float)
    low = day_df["low"].to_numpy(float)
    close = day_df["close"].to_numpy(float)
    times = day_df.index
    ei, ep = plan.entry_index, float(plan.entry_price)
    take, stop = ep * (1.0 + plan.take_pct), ep * (1.0 - plan.stop_pct)
    xi, xp, reason = len(close) - 1, float(close[-1]), "close"
    for j in range(ei + 1, len(close)):
        if low[j] <= stop:                                   # adverse first (conservative)
            xi, xp, reason = j, stop, "stop"
            break
        if high[j] >= take:
            xi, xp, reason = j, take, "take"
            break
    gross = xp / ep - 1.0
    return Trade(symbol=symbol, date=str(pd.Timestamp(times[0]).date()),
                 entry_time=str(times[ei]), exit_time=str(times[xi]),
                 entry_price=ep, exit_price=xp, gross=gross, net=gross - cost_frac,
                 reason=reason)


def _run_records(recs: Sequence[dict], strategy: Strategy, cost_frac: float,
                 overnight, symbol: str) -> List[Trade]:
    trades: List[Trade] = []
    for r in recs:
        ctx = {"prev_close": r["prev_close"], "date": r["date"],
               "overnight_return": _overnight_lookup(overnight, r["date"])}
        plan = strategy(r["df"], ctx)
        if plan is not None:
            trades.append(simulate_day(r["df"], plan, cost_frac, symbol))
    return trades


def run(minute_df: pd.DataFrame, strategy: Strategy, cost: Optional[CostModel] = None,
        overnight=None, symbol: str = "") -> List[Trade]:
    """Backtest a strategy over every day in ``minute_df``; returns the trade list."""
    cost = cost or CostModel()
    return _run_records(group_days(minute_df), strategy, cost.round_trip_frac(), overnight, symbol)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def _empty_metrics() -> dict:
    return {"n_trades": 0, "win_rate": 0.0, "avg_net": 0.0, "expectancy": 0.0,
            "avg_gross": 0.0, "gross_expectancy": 0.0, "profit_factor": 0.0,
            "max_drawdown": 0.0, "trades_per_day": 0.0, "total_net": 0.0}


def metrics(trades: Sequence[Trade], n_days: Optional[int] = None) -> dict:
    """Net-of-cost performance: win rate, expectancy (avg net), profit factor, max
    drawdown (compounded per-trade equity), trades/day."""
    n = len(trades)
    if n == 0:
        return _empty_metrics()
    nets = [t.net for t in trades]
    grosses = [t.gross for t in trades]
    wins = [x for x in nets if x > 0]
    losses = [x for x in nets if x < 0]
    eq, peak, mdd = 1.0, 1.0, 0.0
    for x in nets:
        eq *= (1.0 + x)
        peak = max(peak, eq)
        mdd = min(mdd, eq / peak - 1.0)
    pf = (sum(wins) / abs(sum(losses))) if losses else float("inf")
    n_days = n_days if n_days else len({t.date for t in trades})
    return {"n_trades": n, "win_rate": len(wins) / n, "avg_net": mean(nets),
            "expectancy": mean(nets), "avg_gross": mean(grosses),
            "gross_expectancy": mean(grosses), "profit_factor": pf,
            "max_drawdown": mdd, "trades_per_day": n / n_days if n_days else 0.0,
            "total_net": sum(nets)}


def by_year(trades: Sequence[Trade]) -> Dict[str, dict]:
    buckets: Dict[str, List[Trade]] = {}
    for t in trades:
        buckets.setdefault(t.date[:4], []).append(t)
    return {y: metrics(ts) for y, ts in sorted(buckets.items())}


# --------------------------------------------------------------------------- #
# walk-forward by month
# --------------------------------------------------------------------------- #
def walk_forward(minute_df: pd.DataFrame, make_strategy: Callable[[object], Strategy],
                 param_grid: Sequence, cost: Optional[CostModel] = None, overnight=None,
                 train_months: int = 6, test_months: int = 1, select: str = "expectancy",
                 symbol: str = "") -> dict:
    """Rolling monthly walk-forward: pick params with the best in-sample ``select``
    metric on the TRAIN months, apply on the next TEST month(s), roll, concatenate
    OOS trades. Parameters never see the window they are scored on."""
    cost = cost or CostModel()
    cf = cost.round_trip_frac()
    recs = group_days(minute_df)
    months = sorted({(r["date"].year, r["date"].month) for r in recs})
    by_m: Dict[tuple, List[dict]] = {m: [] for m in months}
    for r in recs:
        by_m[(r["date"].year, r["date"].month)].append(r)

    folds: List[dict] = []
    oos: List[Trade] = []
    i = train_months
    while i < len(months):
        train_ms = months[max(0, i - train_months):i]
        test_ms = months[i:i + test_months]
        if not test_ms:
            break
        train_recs = [r for m in train_ms for r in by_m[m]]
        test_recs = [r for m in test_ms for r in by_m[m]]
        best_params, best_score = param_grid[0], float("-inf")
        for params in param_grid:
            tr = _run_records(train_recs, make_strategy(params), cf, overnight, symbol)
            if tr:
                sc = metrics(tr).get(select, float("-inf"))
                if sc > best_score:
                    best_score, best_params = sc, params
        te = _run_records(test_recs, make_strategy(best_params), cf, overnight, symbol)
        oos.extend(te)
        folds.append({"train_months": [f"{y}-{m:02d}" for y, m in train_ms],
                      "test_months": [f"{y}-{m:02d}" for y, m in test_ms],
                      "params": best_params, "train_score": best_score,
                      "n_test_trades": len(te)})
        i += test_months
    return {"oos": oos, "folds": folds, "n_folds": len(folds),
            "stats": metrics(oos)}
