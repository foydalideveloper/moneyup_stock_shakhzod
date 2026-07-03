"""TRUE out-of-sample backtest (no leakage) + buy-and-hold benchmark.

Unlike run_backtest.py (which scores a model trained on ALL history over that
same history -> in-sample, over-optimistic), this trains per fold and trades
ONLY on each fold's held-out bars. Every position therefore comes from a model
that never saw that bar.

Usage:
    python scripts/run_backtest_oos.py --symbols AAPL MSFT NVDA --threshold 0.55
"""

import argparse
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from tagent.backtest import position_from_proba, run_backtest  # noqa: E402
from tagent.config import SETTINGS  # noqa: E402
from tagent.data.historical import load_history  # noqa: E402
from tagent.features import make_features  # noqa: E402
from tagent.labels import triple_barrier_labels  # noqa: E402
from tagent.ml.train import walk_forward_predict  # noqa: E402

_STAT_KEYS = ["total_return", "annual_return", "sharpe", "max_drawdown",
              "n_trades", "hit_rate", "final_equity"]


def _fmt(v) -> str:
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def backtest_symbol_oos(symbol: str, interval: str, tp: float, sl: float,
                        horizon: int, threshold: float, n_splits: int, gap: int):
    """Returns (strategy_stats, buy_hold_stats, n_oos_bars) or None if too little data."""
    df = load_history(symbol, interval)

    # Align features + triple-barrier labels, drop rows that lack either.
    feats = make_features(df, dropna=False)
    labels = triple_barrier_labels(df, tp_pct=tp, sl_pct=sl, horizon=horizon)
    data = feats.copy()
    data["label"] = labels
    data = data.dropna()
    if len(data) < (n_splits + 1) * 2:
        return None

    X = data.drop(columns=["label"])
    y = data["label"].astype(int)

    # Out-of-sample probabilities; keep only the held-out (test-fold) bars.
    oos = walk_forward_predict(X, y, n_splits=n_splits, gap=gap).dropna()
    if oos.empty:
        return None

    position = position_from_proba(oos, threshold)
    close = df["close"].reindex(oos.index)

    strat = run_backtest(close, position,
                         cost_bps=SETTINGS.transaction_cost_bps,
                         slippage_bps=SETTINGS.slippage_bps)
    # Buy-and-hold over the SAME out-of-sample window, same costs (one entry).
    hold_pos = pd.Series(1.0, index=close.index)
    hold = run_backtest(close, hold_pos,
                        cost_bps=SETTINGS.transaction_cost_bps,
                        slippage_bps=SETTINGS.slippage_bps)
    return strat.stats, hold.stats, len(oos)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["AAPL", "MSFT", "NVDA"])
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--tp", type=float, default=0.04, help="take-profit fraction")
    ap.add_argument("--sl", type=float, default=0.02, help="stop-loss fraction")
    ap.add_argument("--horizon", type=int, default=10, help="bars to hold")
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--gap", type=int, default=5)
    args = ap.parse_args()

    for symbol in args.symbols:
        try:
            out = backtest_symbol_oos(symbol, args.interval, args.tp, args.sl,
                                      args.horizon, args.threshold,
                                      args.n_splits, args.gap)
        except FileNotFoundError:
            print(f"\n{symbol}: no saved data (run download_data.py first).")
            continue
        if out is None:
            print(f"\n{symbol}: not enough history for {args.n_splits} folds.")
            continue

        strat, hold, n_oos = out
        print(f"\n=== OUT-OF-SAMPLE backtest: {symbol} "
              f"(threshold {args.threshold}, {n_oos} held-out bars) ===")
        print(f"  {'metric':16s} {'strategy':>12s} {'buy & hold':>12s}")
        for k in _STAT_KEYS:
            print(f"  {k:16s} {_fmt(strat[k]):>12s} {_fmt(hold[k]):>12s}")

    print("\nThese are leak-free: each bar was traded by a model that never saw it.")
    print("Compare against buy & hold over the same window. An edge must beat it "
          "AFTER costs, out of sample.")
