"""Reconstruct a POINT-IN-TIME KR large-cap universe and fetch its OHLCV.

Survivorship fix for the momentum study: instead of today's 50 liquid names, take
the top-N by market cap **as of each month** (pykrx ``get_market_cap_by_ticker``),
union them across 2016->today, and download daily OHLCV for the whole union —
including names that have since dropped out of the top-N or DELISTED. A name that
was large in 2017 but gone by 2024 is therefore present in the backtest while it
was a member, which is exactly what corrects the bias.

Outputs (all gitignored under data/):
  * data/kr_pit_members.csv   — long-format {date,ticker} monthly membership
  * data/<code>_1d.csv        — daily OHLCV per union member (reuses the standard cache)

Needs a working KRX login (KRX_ID/KRX_PW) for the as-of market-cap snapshots; OHLCV
is open. Resumes from cache: re-running skips members + OHLCV already on disk.

Usage:
    python scripts/download_kr_pit_universe.py                       # top-100 KOSPI, 2016+
    python scripts/download_kr_pit_universe.py --top-n 150 --markets KOSPI KOSDAQ
    python scripts/download_kr_pit_universe.py --members-only        # just rebuild membership
"""

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd  # noqa: E402

from tagent.data.krx_source import ensure_krx_login, get_krx_history  # noqa: E402
from tagent.kr_universe import (  # noqa: E402
    load_members, monthly_asof_dates, save_members, top_caps_asof, universe_symbols,
)


def build_members(start, end, top_n, markets, refresh):
    cached = {} if refresh else load_members()
    dates = monthly_asof_dates(start, end)
    have = set(cached)
    todo = [d for d in dates if d.strftime("%Y-%m-%d") not in have]
    if not todo:
        print(f"Membership: {len(cached)} monthly snapshots cached, nothing to fetch.")
        return cached
    if not ensure_krx_login():
        print("WARNING: no KRX_ID/KRX_PW — as-of market-cap snapshots will be empty.")
    from pykrx import stock
    print(f"Reconstructing membership: {len(todo)} months x top-{top_n} {list(markets)}...")
    fresh = dict(cached)
    for i, d in enumerate(todo, 1):
        try:                                              # snap on the first TRADING day
            members = top_caps_asof(d, top_n=top_n, markets=markets, stock=stock)
        except Exception as e:
            members = []
            print(f"  {d.date()}: ERR {str(e)[:50]}")
        if members:
            fresh[d.strftime("%Y-%m-%d")] = members
        if i % 12 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)} months  ({d.date()}: {len(members)} names)", flush=True)
        time.sleep(0.15)
    save_members(fresh)
    print(f"Saved {len(fresh)} snapshots -> data/kr_pit_members.csv")
    return fresh


def fetch_ohlcv(symbols, start, end, pause):
    have, fetched, failed = 0, 0, 0
    print(f"\nOHLCV for {len(symbols)} union members, {start.date()} -> {end.date()} "
          "(incl. since-delisted)...")
    for i, code in enumerate(symbols, 1):
        path = pathlib.Path("data") / f"{code}_1d.csv"
        if path.exists():
            have += 1
            continue
        try:
            df = get_krx_history(code, start, end, interval="1d")
            if len(df):
                fetched += 1
            else:
                failed += 1            # delisted before window / no data
        except Exception as e:
            failed += 1
            if failed <= 10:
                print(f"  {code}: FAIL {str(e)[:50]}")
        if i % 25 == 0:
            print(f"  {i}/{len(symbols)}  cached:{have} new:{fetched} miss:{failed}", flush=True)
        time.sleep(pause)
    print(f"OHLCV done. cached:{have} new:{fetched} missing:{failed}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=100)
    ap.add_argument("--markets", nargs="+", default=["KOSPI"])
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--pause", type=float, default=0.2)
    ap.add_argument("--refresh", action="store_true", help="rebuild membership ignoring cache")
    ap.add_argument("--members-only", action="store_true", help="skip the OHLCV download")
    args = ap.parse_args()

    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.today().normalize()
    start = pd.Timestamp(args.start)

    members = build_members(start, end, args.top_n, args.markets, args.refresh)
    symbols = universe_symbols(members)
    print(f"\nPoint-in-time universe: {len(symbols)} distinct names ever in top-{args.top_n} "
          f"{args.markets} ({len(members)} monthly snapshots).")
    if not args.members_only:
        fetch_ohlcv(symbols, start, end, args.pause)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
