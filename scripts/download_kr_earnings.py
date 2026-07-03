"""Fetch OpenDART earnings-type disclosures for the point-in-time KR universe.

For each PIT universe symbol, pulls its disclosures over the study window via the
corp_code path (paginated) and keeps the EARNINGS-tagged ones, caching to
data/kr_earnings_disclosures.csv (long format: time, symbol, type, title). Free
OpenDART key (OPENDART_API_KEY); key never logged.

Usage:
    python scripts/download_kr_earnings.py
    python scripts/download_kr_earnings.py --start 2016-01-01
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
from tagent.kr_universe import load_members, universe_symbols  # noqa: E402
from tagent.news.opendart_source import OpenDartSource, ensure_corp_map  # noqa: E402

OUT = pathlib.Path("data") / "kr_earnings_disclosures.csv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default=None)
    args = ap.parse_args()
    if not SETTINGS.has_opendart_key():
        print("Missing OPENDART_API_KEY in .env.")
        return 2
    members = load_members()
    if not members:
        print("No PIT membership — run scripts/download_kr_pit_universe.py first.")
        return 1
    syms = universe_symbols(members)
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.today().normalize()
    bgn, end = pd.Timestamp(args.start).strftime("%Y%m%d"), end.strftime("%Y%m%d")

    src = OpenDartSource(SETTINGS.opendart_api_key)
    corp_map = ensure_corp_map(SETTINGS.opendart_api_key)
    print(f"Fetching earnings disclosures for {len(syms)} names, {bgn}->{end} ...")
    rows = []
    for i, code in enumerate(syms, 1):
        try:
            ds = src.recent_disclosures([code], bgn, end, corp_map=corp_map, max_pages=10)
        except Exception as e:
            if i <= 5:
                print(f"  {code}: ERR {str(e)[:50]}")
            ds = []
        for d in ds:
            if d.get("type") == "earnings":
                rows.append({"time": d["time"], "symbol": d["symbol"],
                             "type": d["type"], "title": d["title"]})
        if i % 25 == 0 or i == len(syms):
            print(f"  {i}/{len(syms)} names  ({len(rows)} earnings rows so far)", flush=True)

    df = pd.DataFrame(rows, columns=["time", "symbol", "type", "title"])
    df = df.drop_duplicates()
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df):,} earnings disclosures ({df['symbol'].nunique()} names) -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
