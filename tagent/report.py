"""Performance reporting: quantstats HTML tear sheets + a Monte-Carlo bust test.

Takes ANY agent's equity curve / return series — the live funding runner
(data/funding_live.csv), the scorecard state (data/scorecard_state.json), or the
offline studies' net-return Series — and produces:

* a **quantstats** HTML tear sheet (Sharpe / Sortino / CAGR / max drawdown /
  VaR / CVaR) vs an optional buy-and-hold benchmark (:func:`tear_sheet`);
* a bootstrap **Monte Carlo** giving the honest tail: bust (ruin) probability and
  the final-equity distribution (:func:`monte_carlo`).

The stats + Monte Carlo are pure numpy/pandas (unit-tested on synthetic returns,
no network); quantstats is imported lazily only inside :func:`tear_sheet`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

INTERVALS_8H_PER_YEAR = 3 * 365          # funding cadence
DAILY_PER_YEAR = 365


# --------------------------------------------------------------------------- #
# equity <-> returns
# --------------------------------------------------------------------------- #
def equity_to_returns(equity) -> pd.Series:
    """Equity curve -> simple per-step returns (drops the first NaN)."""
    e = pd.Series(equity, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    return e.pct_change().dropna()


def returns_to_equity(returns, start: float = 1.0) -> pd.Series:
    """Returns -> equity curve, with the starting value prepended so that
    ``equity_to_returns`` recovers the original series exactly."""
    r = pd.Series(returns, dtype=float).fillna(0.0)
    eq = start * (1.0 + r).cumprod()
    return pd.concat([pd.Series([float(start)]), eq]).reset_index(drop=True)


def _max_drawdown(equity) -> float:
    e = np.asarray(equity, float)
    if len(e) == 0:
        return 0.0
    peak = np.maximum.accumulate(e)
    return float((e / peak - 1.0).min())


# --------------------------------------------------------------------------- #
# performance stats
# --------------------------------------------------------------------------- #
def perf_stats(returns, periods_per_year: int = DAILY_PER_YEAR,
               var_level: float = 0.95) -> dict:
    """Sharpe / Sortino / CAGR / vol / max drawdown / VaR / CVaR from a return
    series (VaR/CVaR are reported as POSITIVE loss fractions at ``var_level``)."""
    r = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    n = len(r)
    if n == 0:
        return {"sharpe": 0.0, "sortino": 0.0, "cagr": 0.0, "ann_vol": 0.0,
                "total_return": 0.0, "max_drawdown": 0.0, "var": 0.0, "cvar": 0.0, "n": 0}
    mean = float(r.mean())
    std = float(r.std(ddof=1)) if n > 1 else 0.0
    downside = r[r < 0]
    dstd = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    eq = (1.0 + r).cumprod()
    final = float(eq.iloc[-1])
    total = final - 1.0
    cagr = (final ** (periods_per_year / n) - 1.0) if final > 0 else float("nan")
    rt = math.sqrt(periods_per_year)
    q = (1.0 - var_level) * 100.0
    var_cut = float(np.percentile(r, q))
    tail = r[r <= var_cut]
    return {
        "sharpe": mean / std * rt if std > 0 else 0.0,
        "sortino": mean / dstd * rt if dstd > 0 else 0.0,
        "cagr": cagr,
        "ann_vol": std * rt,
        "total_return": total,
        "max_drawdown": _max_drawdown(eq.to_numpy()),
        "var": -var_cut,                                   # positive loss at var_level
        "cvar": -float(tail.mean()) if len(tail) else -var_cut,
        "n": int(n),
    }


# --------------------------------------------------------------------------- #
# Monte Carlo bust / ruin probability + tail (the key honest number)
# --------------------------------------------------------------------------- #
def monte_carlo(returns, n_paths: int = 2000, horizon: Optional[int] = None,
                start_equity: float = 10_000.0, ruin_drawdown: float = 0.5,
                block: int = 1, seed: int = 0) -> dict:
    """Bootstrap the return series into ``n_paths`` equity paths and measure the
    tail: BUST (ruin) probability = fraction of paths whose drawdown breaches
    ``ruin_drawdown`` (or that go to zero), plus the final-equity distribution.

    ``block`` > 1 uses a block bootstrap (preserves short-run autocorrelation —
    funding spikes / rallies cluster), which is the honest choice for carry.
    """
    r = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if len(r) == 0:
        return {"bust_prob": 0.0, "ruin_drawdown": ruin_drawdown, "n_paths": 0,
                "horizon": 0, "start_equity": start_equity, "median_final": start_equity,
                "p5_final": start_equity, "p95_final": start_equity, "median_return": 0.0,
                "worst_drawdown_med": 0.0, "var_final": 0.0, "cvar_final": 0.0}
    horizon = int(horizon or len(r))
    rng = np.random.default_rng(seed)
    block = max(1, int(block))
    finals = np.empty(n_paths)
    worst_dd = np.empty(n_paths)
    busts = 0
    for i in range(n_paths):
        if block <= 1:
            samp = rng.choice(r, size=horizon, replace=True)
        else:
            nb = math.ceil(horizon / block)
            hi = max(1, len(r) - block + 1)
            starts = rng.integers(0, hi, size=nb)
            samp = np.concatenate([r[s:s + block] for s in starts])[:horizon]
        eq = start_equity * np.cumprod(1.0 + samp)
        peak = np.maximum.accumulate(eq)
        dd = float((eq / peak - 1.0).min())
        worst_dd[i] = dd
        finals[i] = eq[-1]
        if dd <= -ruin_drawdown or eq[-1] <= 0:
            busts += 1
    p5 = float(np.percentile(finals, 5))
    tail = finals[finals <= p5]
    return {
        "bust_prob": busts / n_paths,
        "ruin_drawdown": ruin_drawdown,
        "n_paths": n_paths, "horizon": horizon, "start_equity": start_equity,
        "median_final": float(np.median(finals)),
        "p5_final": p5, "p95_final": float(np.percentile(finals, 95)),
        "median_return": float(np.median(finals) / start_equity - 1.0),
        "worst_drawdown_med": float(np.median(worst_dd)),
        "var_final": float(start_equity - p5),                       # 95% VaR on final equity
        "cvar_final": float(start_equity - tail.mean()) if len(tail) else float(start_equity - p5),
    }


# --------------------------------------------------------------------------- #
# loaders for the various agent equity sources
# --------------------------------------------------------------------------- #
def returns_from_equity_csv(path, equity_col: str = "equity",
                            time_col: Optional[str] = "ts") -> pd.Series:
    df = pd.read_csv(path)
    if equity_col not in df.columns:
        raise ValueError(f"{path}: no '{equity_col}' column")
    eq = pd.to_numeric(df[equity_col], errors="coerce")
    r = eq.pct_change()
    if time_col and time_col in df.columns:
        idx = pd.to_datetime(df[time_col], utc=True, errors="coerce")
        r.index = idx
    return r.dropna()


def returns_from_funding_live(path) -> pd.Series:
    """Per-cycle returns of the live funding-carry runner (data/funding_live.csv)."""
    return returns_from_equity_csv(path, equity_col="equity", time_col="ts")


def funding_carry_returns_from_scorecard(state_path) -> pd.Series:
    """Reconstruct the funding-carry per-cycle returns from the scorecard state's
    carry records (entry_price = the 8h funding in bps; return = bps/1e4 when held)."""
    s = json.loads(Path(state_path).read_text(encoding="utf-8"))
    recs = [r for r in s.get("scores", [])
            if r.get("source") == "funding-carry" and r.get("status") == "carry"]
    vals = [(float(r.get("entry_price", 0.0)) / 1e4 if int(r.get("side", 0)) != 0 else 0.0)
            for r in recs]
    idx = pd.to_datetime([r.get("ts") for r in recs], utc=True, errors="coerce")
    return pd.Series(vals, index=idx, dtype=float)


def returns_from_signal_log(path, source: Optional[str] = None,
                            cost_bps: float = 10.0) -> pd.Series:
    """Crude per-signal realized returns from data/signal_log.csv: for each distinct
    signal, the price move to that symbol's NEXT signal, in the signal's direction,
    net of a round-trip cost. (A rough per-agent curve for reporting — the scorecard
    holds the authoritative marked book.)"""
    from tagent.scorecard import side_of
    df = pd.read_csv(path)
    if source is not None:
        df = df[df["source"] == source]
    if df.empty:
        return pd.Series(dtype=float)
    df = df.sort_values("timestamp")
    out_ts, out_r = [], []
    c = cost_bps / 1e4
    for _, g in df.groupby("symbol"):
        g = g.sort_values("timestamp")
        px = pd.to_numeric(g["price"], errors="coerce").to_numpy(float)
        sides = [side_of(x) for x in g["direction"]]
        tss = list(g["timestamp"])
        for i in range(len(g) - 1):
            if sides[i] == 0 or not (px[i] > 0) or not (px[i + 1] > 0):
                continue
            out_ts.append(tss[i])
            out_r.append(sides[i] * (px[i + 1] / px[i] - 1.0) - 2.0 * c)
    idx = pd.to_datetime(out_ts, utc=True, errors="coerce")
    return pd.Series(out_r, index=idx, dtype=float).sort_index()


# --------------------------------------------------------------------------- #
# quantstats HTML tear sheet (lazy import)
# --------------------------------------------------------------------------- #
def tear_sheet(returns, output, benchmark=None, title: str = "strategy") -> str:
    """Write a quantstats HTML tear sheet for ``returns`` to ``output``.

    A synthetic daily DatetimeIndex is used when the series isn't datetime-indexed
    so quantstats can render; the precise, cadence-correct figures live in
    :func:`perf_stats`. ``benchmark`` is an optional aligned return Series
    (e.g. buy-and-hold). Requires the optional ``quantstats`` dependency.
    """
    import quantstats as qs
    r = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        raise ValueError("no returns to report")
    if not isinstance(r.index, pd.DatetimeIndex):
        r.index = pd.date_range("2020-01-01", periods=len(r), freq="D")
    else:
        r.index = r.index.tz_localize(None)
    bench = None
    if benchmark is not None:
        b = pd.Series(benchmark, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
        if not b.empty:
            b.index = r.index[:len(b)] if len(b) <= len(r) else \
                pd.date_range("2020-01-01", periods=len(b), freq="D")
            bench = b.reindex(r.index).fillna(0.0)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    qs.reports.html(r, benchmark=bench, output=str(out), title=title,
                    download_filename=str(out))
    return str(out)
