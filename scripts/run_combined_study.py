"""HONEST 4-arm OOS study: do 공매도 and/or 수급 features add edge?

For each KR stock, build ONE leak-free out-of-sample experiment and evaluate four
models on the *same* held-out bars and folds:

    (a) technicals only            (features.py)
    (b) technicals + 공매도         (+ short-selling features)
    (c) technicals + 수급           (+ foreign/institutional flow features)
    (d) technicals + both

then compare all four against BUY & HOLD over the same window. Because every arm
shares the exact same rows and TimeSeriesSplit folds, any difference is
attributable to the added features alone — not a different test window.

Reuses walk_forward_predict (no leakage) + run_backtest (with costs) + the
existing short/flow merges. Arms whose data is missing are skipped (printed "-").

Run scripts/download_krx.py first (OHLCV). 공매도 needs data/<code>_short.csv;
수급 needs data/<code>_flows.csv (KRX_ID/KRX_PW or a manual export).

Usage:
    python scripts/run_combined_study.py --tickers 005930 000660 035420
"""

import argparse
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

# KR output has Korean + em-dashes; force UTF-8 so it prints on Windows cp949.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from tagent.backtest import position_from_proba, run_backtest  # noqa: E402
from tagent.config import SETTINGS  # noqa: E402
from tagent.data.flows_source import load_flows  # noqa: E402
from tagent.data.historical import load_history  # noqa: E402
from tagent.data.short_selling import complete_short_ratio, load_short_selling  # noqa: E402
from tagent.features import make_features  # noqa: E402
from tagent.features_flows import flow_feature_columns, merge_flow_features  # noqa: E402
from tagent.features_short import SHORT_FEATURE_COLS, merge_short_features  # noqa: E402
from tagent.labels import triple_barrier_labels  # noqa: E402
from tagent.ml.train import walk_forward_predict  # noqa: E402

# Arm keys in display order; "hold" is the benchmark.
ARMS = [("tech", "technicals"), ("short", "+공매도"), ("flow", "+수급"),
        ("both", "+both"), ("hold", "buy&hold")]


def _bt(close: pd.Series, position: pd.Series):
    return run_backtest(close, position,
                        cost_bps=SETTINGS.transaction_cost_bps,
                        slippage_bps=SETTINGS.slippage_bps).stats


def study_symbol(symbol: str, interval: str, tp: float, sl: float, horizon: int,
                 threshold: float, n_splits: int, gap: int):
    """Returns (stats_by_arm, n_oos, has_short, has_flow) or None."""
    df = load_history(symbol, interval)
    feats = make_features(df, dropna=False)

    has_short = has_flow = False
    try:
        short = complete_short_ratio(load_short_selling(symbol), df["volume"])
        feats = merge_short_features(feats, short)
        has_short = True
    except FileNotFoundError:
        pass
    try:
        feats = merge_flow_features(feats, load_flows(symbol))
        has_flow = True
    except FileNotFoundError:
        pass

    labels = triple_barrier_labels(df, tp_pct=tp, sl_pct=sl, horizon=horizon)
    data = feats.copy()
    data["label"] = labels
    data = data.dropna()                  # common rows across ALL present features
    if len(data) < (n_splits + 1) * 2:
        return None

    y = data["label"].astype(int)
    X = data.drop(columns=["label"])
    short_cols = [c for c in SHORT_FEATURE_COLS if c in X.columns]
    flow_cols = [c for c in flow_feature_columns() if c in X.columns]
    base_cols = [c for c in X.columns if c not in short_cols and c not in flow_cols]

    def oos(cols):
        return walk_forward_predict(X[cols], y, n_splits=n_splits, gap=gap)

    oos_a = oos(base_cols).dropna()
    if oos_a.empty:
        return None
    idx = oos_a.index
    close = df["close"].reindex(idx)

    stats = {"tech": _bt(close, position_from_proba(oos_a, threshold)),
             "hold": _bt(close, pd.Series(1.0, index=close.index))}
    if has_short:
        stats["short"] = _bt(close, position_from_proba(oos(base_cols + short_cols).reindex(idx), threshold))
    if has_flow:
        stats["flow"] = _bt(close, position_from_proba(oos(base_cols + flow_cols).reindex(idx), threshold))
    if has_short and has_flow:
        stats["both"] = _bt(close, position_from_proba(oos(base_cols + short_cols + flow_cols).reindex(idx), threshold))
    return stats, len(idx), has_short, has_flow


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

    saw_short = saw_flow = False
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
        stats, n_oos, has_short, has_flow = out
        saw_short = saw_short or has_short
        saw_flow = saw_flow or has_flow

        print(f"\n=== combined OOS study: {symbol} "
              f"(threshold {args.threshold}, {n_oos} held-out bars) ===")
        header = "  " + " ".join(f"{label:>12s}" for _k, label in ARMS)
        print(header)
        for metric in ("total_return", "sharpe"):
            row = [f"{metric:14s}"]
            for k, _label in ARMS:
                v = stats.get(k, {}).get(metric)
                row.append(f"{v:>12.4f}" if v is not None else f"{'-':>12s}")
            print("  " + " ".join(row))

    print("\nLeak-free: every bar was traded by a model that never saw it; all arms "
          "share identical rows/folds, so any gap is the added features' doing.")
    if not saw_short:
        print("NOTE: no 공매도 data found — provide data/<code>_short.csv.")
    if not saw_flow:
        print("NOTE: no 수급 data found — provide data/<code>_flows.csv "
              "(KRX_ID/KRX_PW + the flows fetch, or a manual KRX export).")
