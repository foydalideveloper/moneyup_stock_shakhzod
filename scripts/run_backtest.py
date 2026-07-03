"""Backtest the trained model on one symbol's history (with realistic costs).

Usage:  python scripts/run_backtest.py --symbol AAPL --threshold 0.55
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
from tagent.ml.predict import Predictor  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default=SETTINGS.watchlist[0])
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--threshold", type=float, default=0.55)
    args = ap.parse_args()

    df = load_history(args.symbol, args.interval)
    pred = Predictor.load()

    feats = make_features(df, dropna=True).reindex(columns=pred.features)
    proba = pd.Series(pred.model.predict_proba(feats)[:, 1], index=feats.index)
    position = position_from_proba(proba, args.threshold)
    close = df["close"].reindex(feats.index)

    result = run_backtest(
        close, position,
        cost_bps=SETTINGS.transaction_cost_bps,
        slippage_bps=SETTINGS.slippage_bps,
    )

    print(f"\n=== Backtest: {args.symbol} (threshold {args.threshold}) ===")
    for k, v in result.stats.items():
        print(f"  {k:16s}: {v:.4f}" if isinstance(v, float) else f"  {k:16s}: {v}")
    print("\nNote: costs are included. A great-looking backtest still needs "
          "out-of-sample + paper-trading validation before real money.")
