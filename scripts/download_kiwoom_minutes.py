"""Fetch Kiwoom minute bars (ka10080) for liquid KR names — MOCK-first.

Authenticates (KIWOOM_ENV, default mock), pulls minute bars per symbol following
continuation (연속조회) to get as much history as the server returns, saves each to
data/<SYM>_<interval>m.csv in the intraday backtester's schema, and reports rows,
date range, and the first/last few bars — so we can see plainly how much history
Kiwoom actually gives us.

Secrets are never printed (only "token OK"); no orders are placed.

Usage:
    python scripts/download_kiwoom_minutes.py
    python scripts/download_kiwoom_minutes.py --symbols 005930 000660 005380 --interval 1
    python scripts/download_kiwoom_minutes.py --max-pages 50
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
from tagent.data.kiwoom_minute import fetch_minute_bars, save_minutes  # noqa: E402
from tagent.feeds.kiwoom_auth import KiwoomAuth, KiwoomAuthError  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["005930", "000660", "005380"])
    ap.add_argument("--interval", type=int, default=1, help="1/3/5/10/15/30/60 min")
    ap.add_argument("--max-pages", type=int, default=50, help="continuation pages per symbol")
    ap.add_argument("--pause", type=float, default=0.25, help="seconds between pages")
    args = ap.parse_args()

    if not SETTINGS.has_kiwoom_keys():
        print("Missing KIWOOM_APP_KEY / KIWOOM_SECRET_KEY in .env.")
        return 2

    env = SETTINGS.kiwoom_env
    base = SETTINGS.kiwoom_rest_url()
    print(f"Kiwoom env: {env}  ({base})   interval: {args.interval}min   max pages: {args.max_pages}")

    try:
        auth = KiwoomAuth(SETTINGS.kiwoom_app_key, SETTINGS.kiwoom_secret_key,
                          env=env, base_url=base)
        auth.get_token()                                 # validate before fetching
    except KiwoomAuthError as e:
        print(f"AUTH FAILED: {e}")                       # message never contains the secret
        return 1
    print("token OK")                                    # never print the token

    total_rows = 0
    for sym in args.symbols:
        try:
            df, info = fetch_minute_bars(sym, args.interval, auth=auth, env=env, base_url=base,
                                         max_pages=args.max_pages, pause=args.pause)
        except Exception as e:
            print(f"\n[{sym}] FETCH FAILED: {str(e)[:120]}")
            continue
        if df.empty:
            print(f"\n[{sym}] no bars returned ({info['pages']} pages).")
            continue
        path = save_minutes(df, sym, interval=args.interval)
        total_rows += info["rows"]
        span_days = (df.index.max() - df.index.min()).days + 1
        print(f"\n=== {sym}  ({args.interval}min) ===")
        print(f"  rows {info['rows']:,}  ·  pages {info['pages']}  ·  trading days {info['days']}  "
              f"·  calendar span {span_days}d")
        print(f"  range {info['start']}  ->  {info['end']}")
        print(f"  saved -> {path}")
        with_cols = df.reset_index()[["timestamp", "open", "high", "low", "close", "volume"]]
        print("  first 3 bars:")
        for _, r in with_cols.head(3).iterrows():
            print(f"    {r['timestamp']}  O {r['open']:,.0f} H {r['high']:,.0f} "
                  f"L {r['low']:,.0f} C {r['close']:,.0f} V {r['volume']:,.0f}")
        print("  last 3 bars:")
        for _, r in with_cols.tail(3).iterrows():
            print(f"    {r['timestamp']}  O {r['open']:,.0f} H {r['high']:,.0f} "
                  f"L {r['low']:,.0f} C {r['close']:,.0f} V {r['volume']:,.0f}")

    print(f"\nDone. {total_rows:,} total bars saved across {len(args.symbols)} symbols.")
    print("How much history Kiwoom gave us = the 'trading days' above; if it's far short of")
    print("our backtest window, we still need a vendor for deeper intraday history.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
