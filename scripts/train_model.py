"""Train the LightGBM model with walk-forward validation.

Run download_data.py first. Usage:
    python scripts/train_model.py --interval 1d --tp 0.04 --sl 0.02 --horizon 10
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from tagent.config import SETTINGS  # noqa: E402
from tagent.data.historical import load_history  # noqa: E402
from tagent.data.short_selling import load_short_selling  # noqa: E402
from tagent.ml.train import build_dataset, save_model, walk_forward_train  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--tp", type=float, default=0.04, help="take-profit fraction")
    ap.add_argument("--sl", type=float, default=0.02, help="stop-loss fraction")
    ap.add_argument("--horizon", type=int, default=10, help="bars to hold")
    ap.add_argument("--with-short", action="store_true",
                    help="merge KR short-selling features from data/<SYM>_short.csv when present")
    args = ap.parse_args()

    history = {}
    for sym in SETTINGS.watchlist:
        try:
            history[sym] = load_history(sym, args.interval)
        except FileNotFoundError:
            print(f"  (skipping {sym}: no saved data)")
    if not history:
        sys.exit("No data found. Run: python scripts/download_data.py first.")

    short_data = {}
    if args.with_short:
        for sym in history:
            try:
                short_data[sym] = load_short_selling(sym)
            except FileNotFoundError:
                print(f"  (no short-selling CSV for {sym}: skipping its short features)")

    X, y = build_dataset(history, tp_pct=args.tp, sl_pct=args.sl,
                         horizon=args.horizon, short_data=short_data or None)
    model, metrics = walk_forward_train(X, y)
    path = save_model(model, X.columns, metrics)

    print("\n=== Walk-forward results ===")
    for k, v in metrics.items():
        print(f"  {k:18s}: {v}")
    print(f"\nModel saved to: {path}")
    print("Reminder: out-of-sample accuracy near 0.5 means little/no edge. "
          "Paper-trade before trusting it.")
