"""HONEST out-of-sample test: do 공매도 (short-selling) features add edge?

For each KR stock we build ONE leak-free out-of-sample experiment and evaluate
two models on the *same* held-out bars and folds:

    (a) WITHOUT short features — price/technical features only (features.py)
    (b) WITH short features merged (features_short.py), everything else identical

then compare both against BUY & HOLD over the same window. Because both arms
share the exact same rows and TimeSeriesSplit folds, any difference is
attributable to the short-selling features alone — not to a different test window.

This reuses the real machinery: walk_forward_predict (no leakage) + run_backtest
(with costs). Run scripts/download_krx.py first.

Usage:
    python scripts/run_short_study.py --tickers 005930 000660 035420 --threshold 0.55
"""

import argparse
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from tagent.backtest import position_from_proba, run_backtest  # noqa: E402
from tagent.config import SETTINGS  # noqa: E402
from tagent.data.historical import load_history  # noqa: E402
from tagent.data.short_selling import (  # noqa: E402
    complete_short_ratio,
    load_short_selling,
)
from tagent.features import make_features  # noqa: E402
from tagent.features_short import SHORT_FEATURE_COLS, merge_short_features  # noqa: E402
from tagent.labels import triple_barrier_labels  # noqa: E402
from tagent.ml.train import walk_forward_predict  # noqa: E402

_STAT_KEYS = ["total_return", "annual_return", "sharpe", "max_drawdown",
              "n_trades", "hit_rate", "final_equity"]


def _fmt(v) -> str:
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def _bt(close: pd.Series, position: pd.Series):
    return run_backtest(close, position,
                        cost_bps=SETTINGS.transaction_cost_bps,
                        slippage_bps=SETTINGS.slippage_bps).stats


def study_symbol(symbol: str, interval: str, tp: float, sl: float, horizon: int,
                 threshold: float, n_splits: int, gap: int):
    """Returns (stats_without, stats_with_or_None, stats_hold, n_oos) or None."""
    df = load_history(symbol, interval)

    feats = make_features(df, dropna=False)
    try:
        short = load_short_selling(symbol)
        # KRX 공매도거래 exports lack total volume -> complete short_ratio from the
        # price file's (real KRX) volume so the ratio is properly defined.
        short = complete_short_ratio(short, df["volume"])
        feats_all = merge_short_features(feats, short)
        has_short = True
    except FileNotFoundError:
        feats_all = feats
        has_short = False

    labels = triple_barrier_labels(df, tp_pct=tp, sl_pct=sl, horizon=horizon)
    data = feats_all.copy()
    data["label"] = labels
    data = data.dropna()                       # common rows: base (+short) + label
    if len(data) < (n_splits + 1) * 2:
        return None

    y = data["label"].astype(int)
    X_all = data.drop(columns=["label"])
    base_cols = [c for c in X_all.columns if c not in SHORT_FEATURE_COLS]

    # Arm (a): price-only model on these exact rows/folds.
    oos_a = walk_forward_predict(X_all[base_cols], y, n_splits=n_splits, gap=gap).dropna()
    if oos_a.empty:
        return None
    close = df["close"].reindex(oos_a.index)
    stats_without = _bt(close, position_from_proba(oos_a, threshold))
    stats_hold = _bt(close, pd.Series(1.0, index=close.index))

    stats_with = None
    if has_short:
        # Arm (b): same rows/folds, base + short features.
        oos_b = walk_forward_predict(X_all, y, n_splits=n_splits, gap=gap).reindex(oos_a.index)
        stats_with = _bt(close, position_from_proba(oos_b, threshold))

    return stats_without, stats_with, stats_hold, len(oos_a)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=["005930", "000660", "035420"])
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--tp", type=float, default=0.04)
    ap.add_argument("--sl", type=float, default=0.02)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--gap", type=int, default=5)
    args = ap.parse_args()

    any_short = False
    for symbol in args.tickers:
        try:
            out = study_symbol(symbol, args.interval, args.tp, args.sl,
                               args.horizon, args.threshold, args.n_splits, args.gap)
        except FileNotFoundError:
            print(f"\n{symbol}: no saved OHLCV (run download_krx.py first).")
            continue
        if out is None:
            print(f"\n{symbol}: not enough history for {args.n_splits} folds.")
            continue

        without, with_short, hold, n_oos = out
        any_short = any_short or (with_short is not None)
        wlabel = "with short" if with_short is not None else "(no short data)"
        print(f"\n=== OOS short-feature study: {symbol} "
              f"(threshold {args.threshold}, {n_oos} held-out bars) ===")
        print(f"  {'metric':16s} {'no short':>12s} {wlabel:>14s} {'buy & hold':>12s}")
        for k in _STAT_KEYS:
            w = _fmt(with_short[k]) if with_short is not None else "-"
            print(f"  {k:16s} {_fmt(without[k]):>12s} {w:>14s} {_fmt(hold[k]):>12s}")

    print("\nLeak-free: each bar was traded by a model that never saw it; both arms "
          "share identical rows/folds, so any gap is the short features' doing.")
    if not any_short:
        print("NOTE: no short-selling data was found for any ticker, so the WITH-short "
              "arm could not run. Provide data/<TICKER>_short.csv (KRX_ID/KRX_PW + "
              "download_krx.py, or a manual KRX export) to complete the comparison.")
