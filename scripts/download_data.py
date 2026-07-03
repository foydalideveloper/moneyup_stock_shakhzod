"""Download historical OHLCV for the watchlist (training/backtest data).

Usage:  python scripts/download_data.py --period 3y --interval 1d
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from tagent.config import SETTINGS  # noqa: E402
from tagent.data.historical import download_history  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="3y", help="e.g. 1y, 3y, 5y, max")
    ap.add_argument("--interval", default="1d", help="e.g. 1d, 1h, 30m")
    args = ap.parse_args()

    data = download_history(SETTINGS.watchlist, period=args.period, interval=args.interval)
    print(f"Downloaded {len(data)} symbols ({args.interval}):")
    for sym, df in data.items():
        print(f"  {sym:6s} {len(df):6d} rows  {df.index.min()} -> {df.index.max()}")
    print("Saved as CSV in the data/ folder.")
