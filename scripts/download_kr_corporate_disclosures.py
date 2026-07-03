"""Fetch OpenDART treasury-cancellation / acquisition / supply-contract disclosures.

For each PIT-universe symbol, pulls disclosures YEAR-BY-YEAR (so the newest-first page
cap never truncates old events) and keeps three pre-registered event types, caching to
data/kr_corporate_disclosures.csv (long: time, symbol, type, title). Incremental writes
survive a kill. Free OpenDART key; key never logged.

Classification (LOCKED, see dart_event_studies_spec.md):
  cancellation = '소각' & not '정정' & not '자회사'
  acquisition  = '자기주식취득' & '결정' & not '신탁' & not '처분' & not '정정'
  contract     = ('단일판매' | '공급계약') & not '정정'

Usage: python scripts/download_kr_corporate_disclosures.py [--start 2016-01-01]
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass

import pandas as pd  # noqa: E402

from tagent.config import SETTINGS  # noqa: E402
from tagent.kr_universe import load_members, universe_symbols  # noqa: E402
from tagent.news.opendart_source import OpenDartSource, ensure_corp_map  # noqa: E402

OUT = pathlib.Path("data") / "kr_corporate_disclosures.csv"


def classify_event(title: str):
    """Map a disclosure title to a pre-registered event type, or None to drop it."""
    t = str(title or "")
    if "정정" in t:                                   # corrections excluded from all types
        return None
    if "소각" in t and "자회사" not in t:               # treasury cancellation (own shares)
        return "cancellation"
    if "자기주식취득" in t and "결정" in t and "신탁" not in t and "처분" not in t:
        return "acquisition"                          # direct acquisition decision (announcement)
    if "단일판매" in t or "공급계약" in t:
        return "contract"
    return None


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
    years = list(range(pd.Timestamp(args.start).year, end.year + 1))

    src = OpenDartSource(SETTINGS.opendart_api_key)
    corp_map = ensure_corp_map(SETTINGS.opendart_api_key)
    print(f"Fetching cancellation/acquisition/contract for {len(syms)} names, "
          f"{years[0]}-{years[-1]} (year-chunked) ...")
    rows, seen = [], set()
    for i, code in enumerate(syms, 1):
        for yr in years:
            bgn = f"{yr}0101"
            ey = min(pd.Timestamp(f"{yr}-12-31"), end).strftime("%Y%m%d")
            try:
                ds = src.recent_disclosures([code], bgn, ey, corp_map=corp_map, max_pages=4)
            except Exception:
                ds = []
            for d in ds:
                ev = classify_event(d["title"])
                if ev is None:
                    continue
                key = (d["time"], d["symbol"], ev)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"time": d["time"], "symbol": d["symbol"], "type": ev,
                             "title": d["title"].strip()})
        if i % 25 == 0 or i == len(syms):
            pd.DataFrame(rows, columns=["time", "symbol", "type", "title"]).to_csv(OUT, index=False)
            from collections import Counter
            cc = Counter(r["type"] for r in rows)
            print(f"  {i}/{len(syms)} names  events: {dict(cc)}", flush=True)

    df = pd.DataFrame(rows, columns=["time", "symbol", "type", "title"]).drop_duplicates()
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df):,} events ({df['symbol'].nunique()} names) -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
