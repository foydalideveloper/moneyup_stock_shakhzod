"""Fetch KR fundamentals (PER/PBR/EPS/BPS/DIV) over the point-in-time universe.

For each cached monthly membership snapshot date, pull pykrx get_market_fundamental
for the universe symbols and cache the long-format result to data/kr_fundamentals.csv
(resumes — dates already cached are skipped). Free, no KRX login needed for
fundamentals; survivorship-correct because it uses the same as-of monthly dates as
the momentum study's point-in-time membership.

Usage:
    python scripts/download_kr_fundamentals.py
    python scripts/download_kr_fundamentals.py --markets KOSPI KOSDAQ
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

from tagent.data.kr_fundamental import (  # noqa: E402
    build_fundamentals, load_fundamentals, save_fundamentals,
)
from tagent.kr_universe import load_members, universe_symbols  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", nargs="+", default=["KOSPI"])
    ap.add_argument("--refresh", action="store_true", help="ignore cache, refetch all dates")
    ap.add_argument("--pause", type=float, default=0.15)
    args = ap.parse_args()

    members = load_members()
    if not members:
        print("No data/kr_pit_members.csv — run scripts/download_kr_pit_universe.py first.")
        return 1
    dates = sorted(members)
    symbols = universe_symbols(members)
    cached = pd.DataFrame() if args.refresh else load_fundamentals()
    have_dates = set(cached["date"]) if not cached.empty else set()
    todo = [d for d in dates if d not in have_dates]
    print(f"Fundamentals: {len(dates)} snapshot dates, {len(symbols)} symbols; "
          f"{len(todo)} dates to fetch ({len(have_dates)} cached).")
    if not todo:
        print("Nothing to fetch.")
        return 0

    from tagent.data.krx_source import ensure_krx_login
    if not ensure_krx_login():                              # fundamentals are KRX-login-gated
        print("WARNING: no KRX_ID/KRX_PW — get_market_fundamental will return empty.")
    from pykrx import stock
    parts = [cached] if not cached.empty else []
    for i, d in enumerate(todo, 1):
        try:
            df = build_fundamentals([pd.Timestamp(d)], markets=args.markets,
                                    symbols=symbols, stock=stock)
        except Exception as e:
            print(f"  {d}: ERR {str(e)[:50]}")
            continue
        if not df.empty:
            parts.append(df)
        if i % 12 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)} dates ({d}: {0 if df is None else len(df)} rows)", flush=True)
        time.sleep(args.pause)

    full = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not full.empty:
        full = full.drop_duplicates(subset=["date", "ticker"], keep="last")
        path = save_fundamentals(full)
        print(f"Saved {len(full):,} rows ({full['date'].nunique()} dates) -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
