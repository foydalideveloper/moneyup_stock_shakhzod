"""Download ~3y of daily KR OHLCV (+ short-selling when available) via pykrx.

OHLCV needs no credentials. The 공매도 (short-selling) endpoints require a free
KRX account in the KRX_ID / KRX_PW environment variables; if they're missing the
short fetch is skipped per symbol (with a note) and OHLCV is still saved.

Usage:
    python scripts/download_krx.py                       # default KR watchlist, 3y
    python scripts/download_krx.py --tickers 005930 000660 --years 5
    KRX_ID=you KRX_PW=secret python scripts/download_krx.py   # include short data
"""

import argparse
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from tagent.data.krx_source import get_krx_history, krx_short_fetch  # noqa: E402
from tagent.data.short_selling import load_short_selling  # noqa: E402

# Default KR watchlist: Samsung Electronics, SK hynix, NAVER.
DEFAULT_TICKERS = ["005930", "000660", "035420"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    ap.add_argument("--years", type=int, default=3, help="years of history")
    ap.add_argument("--end", default=None, help="end date YYYY-MM-DD (default: today)")
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--no-short", action="store_true", help="skip 공매도 data")
    args = ap.parse_args()

    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=args.years)
    s_str, e_str = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    print(f"KRX download {s_str} -> {e_str} ({args.interval})")

    fetch = krx_short_fetch(start, end)
    for ticker in args.tickers:
        try:
            ohlcv = get_krx_history(ticker, start, end, interval=args.interval)
            print(f"  {ticker}: OHLCV {len(ohlcv):5d} rows  "
                  f"{ohlcv.index.min().date()} -> {ohlcv.index.max().date()}")
        except Exception as e:  # network / pykrx issues shouldn't abort the batch
            print(f"  {ticker}: OHLCV FAILED ({e})")
            continue

        if args.no_short:
            continue
        try:
            short = load_short_selling(ticker, fetch=fetch)
            cov = int(short["short_balance"].notna().sum())
            print(f"  {ticker}: short  {len(short):5d} rows  "
                  f"(balance on {cov} days) -> data/{ticker}_short.csv")
        except Exception as e:
            print(f"  {ticker}: short SKIPPED ({e})")
            print("           (set KRX_ID / KRX_PW for 공매도 access, "
                  "or drop a CSV at data/<TICKER>_short.csv)")

    print("\nDone. Next: python scripts/run_short_study.py "
          f"--tickers {' '.join(args.tickers)}")
