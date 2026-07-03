"""Evaluate EODHD as the KR minute-data vendor for the intraday backtest.

Fetches 1-min OHLCV for liquid KR names (.KO) via EODHD, paging back in 120-day
windows as far as the plan allows, saves to data/<SYM>_1m.csv (backtester schema),
and reports rows / date range / trading days / first+last bars and a sanity check vs
the Kiwoom bars we already have. Surfaces the EXACT HTTP/auth error on failure. The
api_token is read from .env and never printed.

Usage:
    python scripts/download_eodhd_minutes.py
    python scripts/download_eodhd_minutes.py --symbols 005930 000660 035420 --max-windows 16
    python scripts/download_eodhd_minutes.py --save
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from tagent.config import SETTINGS  # noqa: E402
from tagent.data.eodhd_source import EodhdError, fetch_minutes, save_minutes  # noqa: E402
from tagent.data.intraday_history import load_intraday  # noqa: E402


def _kiwoom_last_close(sym):
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
    ap.add_argument("--exchange", default="KO")
    ap.add_argument("--interval", default="1m")
    ap.add_argument("--max-windows", type=int, default=16, help="120-day windows per symbol")
    ap.add_argument("--pause", type=float, default=0.3)
    ap.add_argument("--save", action="store_true", help="write data/<SYM>_1m.csv (overwrites)")
    args = ap.parse_args()

    if not SETTINGS.has_eodhd_key():
        print("Missing EODHD_API_KEY in .env.")
        return 2
    print(f"EODHD minute-bar evaluation — exchange=.{args.exchange}, interval={args.interval}, "
          f"max_windows={args.max_windows} (120d each)")

    ok = 0
    for sym in args.symbols:
        kw_close, kw_when = _kiwoom_last_close(sym)
        try:
            df, info = fetch_minutes(sym, api_token=SETTINGS.eodhd_api_key, exchange=args.exchange,
                                     interval=args.interval, max_windows=args.max_windows,
                                     pause=args.pause)
        except EodhdError as e:
            print(f"\n[{sym}.{args.exchange}] EODHD FAILED: {str(e)[:160]}")   # exact error, never the key
            continue
        except Exception as e:
            print(f"\n[{sym}.{args.exchange}] error: {str(e)[:160]}")
            continue
        if df.empty:
            print(f"\n[{sym}.{args.exchange}] no bars returned ({info['windows']} windows).")
            continue
        ok += 1
        span_days = (df.index.max() - df.index.min()).days + 1
        years = span_days / 365.25
        # rough completeness: KR session ~= 391 1-min bars/day
        completeness = info["rows"] / (info["days"] * 391) if info["days"] else 0.0
        print(f"\n=== {sym}.{args.exchange}  ({args.interval}) ===")
        print(f"  rows {info['rows']:,}  ·  windows {info['windows']}  ·  trading days {info['days']}  "
              f"·  span {span_days}d (~{years:.2f}y)")
        print(f"  range {info['start']}  ->  {info['end']}")
        print(f"  ~bar completeness vs 391/day: {completeness*100:.0f}%")
        rows = df.reset_index()
        for tag, part in (("first", rows.head(2)), ("last", rows.tail(2))):
            for _, r in part.iterrows():
                print(f"  {tag:5s} {r['timestamp']}  O {r['open']:,.0f} H {r['high']:,.0f} "
                      f"L {r['low']:,.0f} C {r['close']:,.0f} V {r['volume']:,.0f}")
        if kw_close is not None:
            it_close = float(df["close"].iloc[-1])
            diff = (it_close / kw_close - 1.0) * 100 if kw_close else float("nan")
            print(f"  sanity vs Kiwoom last close {kw_close:,.0f} @ {kw_when[:16]}: "
                  f"EODHD {it_close:,.0f}  ({diff:+.2f}%)")
        if args.save:
            print(f"  saved -> {save_minutes(df, sym)}")

    print(f"\n{ok}/{len(args.symbols)} symbols returned data from EODHD (.{args.exchange}).")
    if ok == 0:
        print("EODHD returned nothing for KR — check the error above (auth/plan/exchange).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
