"""Evaluate iTick as a minute-data vendor for the intraday backtest.

Fetches 1-min OHLCV for a few liquid KR names via iTick, saves to data/<SYM>_1m.csv
(the backtester schema), and reports rows / date range / first+last bars and a sanity
check vs the Kiwoom bars we already have. Prints the EXACT server error if a symbol
fails (e.g. region unsupported). Token is read from .env and never printed.

Usage:
    python scripts/download_itick_minutes.py
    python scripts/download_itick_minutes.py --region KR --symbols 005930 000660 035420
    python scripts/download_itick_minutes.py --max-pages 10
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd  # noqa: E402

from tagent.config import SETTINGS  # noqa: E402
from tagent.data.intraday_history import load_intraday  # noqa: E402
from tagent.data.itick_source import ItickError, fetch_klines, save_minutes  # noqa: E402


def _kiwoom_last_close(sym):
    """Last close from an existing Kiwoom-fetched CSV, for a sanity comparison."""
    try:
        df = load_intraday(sym)
        if len(df):
            return float(df["close"].iloc[-1]), str(df.index.max())
    except Exception:
        pass
    return None, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["005930", "000660", "035420"])
    ap.add_argument("--region", default="KR")
    ap.add_argument("--interval", type=int, default=1)
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--max-pages", type=int, default=20)
    ap.add_argument("--pause", type=float, default=0.3)
    ap.add_argument("--save", action="store_true", help="write data/<SYM>_1m.csv (overwrites)")
    args = ap.parse_args()

    if not SETTINGS.has_itick_key():
        print("Missing ITICK_API_KEY in .env.")
        return 2
    print(f"iTick minute-bar evaluation — region={args.region}, interval={args.interval}min, "
          f"limit={args.limit}, max_pages={args.max_pages}")

    ok = 0
    for sym in args.symbols:
        kw_close, kw_when = _kiwoom_last_close(sym)
        try:
            df, info = fetch_klines(sym, token=SETTINGS.itick_api_key, region=args.region,
                                    interval=args.interval, limit=args.limit,
                                    max_pages=args.max_pages, pause=args.pause)
        except ItickError as e:
            print(f"\n[{sym}] iTick FAILED: {str(e)[:160]}")     # exact code/msg, never the key
            continue
        except Exception as e:
            print(f"\n[{sym}] error: {str(e)[:160]}")
            continue
        if df.empty:
            print(f"\n[{sym}] no bars returned ({info['pages']} pages).")
            continue
        ok += 1
        span = (df.index.max() - df.index.min()).days + 1
        print(f"\n=== {sym}  ({args.interval}min) ===")
        print(f"  rows {info['rows']:,}  ·  pages {info['pages']}  ·  trading days {info['days']}  "
              f"·  calendar span {span}d")
        print(f"  range {info['start']}  ->  {info['end']}")
        rows = df.reset_index()
        for tag, part in (("first", rows.head(2)), ("last", rows.tail(2))):
            for _, r in part.iterrows():
                print(f"  {tag:5s} {r['timestamp']}  O {r['open']:,.0f} H {r['high']:,.0f} "
                      f"L {r['low']:,.0f} C {r['close']:,.0f} V {r['volume']:,.0f}")
        if kw_close is not None:
            it_close = float(df["close"].iloc[-1])
            diff = (it_close / kw_close - 1.0) * 100 if kw_close else float("nan")
            print(f"  sanity vs Kiwoom last close {kw_close:,.0f} @ {kw_when[:16]}: "
                  f"iTick {it_close:,.0f}  ({diff:+.2f}%)")
        if args.save:
            print(f"  saved -> {save_minutes(df, sym)}")

    print(f"\n{ok}/{len(args.symbols)} symbols returned data from iTick region={args.region}.")
    if ok == 0:
        print("iTick returned nothing for KR — likely Korea/KOSPI is not covered (it is not in")
        print("iTick's documented region list). Try a supported region to confirm the key works,")
        print("then conclude a different KR minute-data vendor is needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
