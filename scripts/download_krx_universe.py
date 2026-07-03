"""Fetch a ~50-name KOSPI/KOSDAQ universe: OHLCV + 공매도 + 수급, max history.

Requires a WORKING KRX login (KRX_ID/KRX_PW in .env) for the gated 공매도/수급
endpoints; OHLCV is open. Each dataset caches to data/<code>_{1d,short,flows}.csv,
so the run RESUMES from cache — a KRX session timeout (~1h) just means re-running
picks up where it left off. Per-stock/per-dataset failures are caught and logged,
never aborting the batch.

KR short-selling data starts ~2016, so we fetch from 2016-01-01. Short-selling-ban
windows are NOT filtered here (we keep the raw data); the study
(run_broad_study.py) flags/excludes them via tagent.features_short.SHORT_BAN_WINDOWS.

Usage:
    python scripts/download_krx_universe.py                 # default ~50 names
    python scripts/download_krx_universe.py --tickers 005930 000660 --start 2018-01-01
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

from tagent.data.flows_source import flows_csv_path, krx_flows_fetch, load_flows  # noqa: E402
from tagent.data.krx_source import ensure_krx_login, get_krx_history, krx_short_fetch  # noqa: E402
from tagent.data.short_selling import load_short_selling, short_csv_path  # noqa: E402

# ~50 liquid KOSPI (large/mid cap) + KOSDAQ names.
KOSPI = [
    "005930", "000660", "035420", "005380", "000270", "005490", "035720",
    "051910", "006400", "028260", "105560", "055550", "012330", "066570",
    "015760", "017670", "034730", "068270", "003670", "096770", "011200",
    "009150", "010130", "032830", "086790", "024110", "011170", "018260",
    "010950", "030200", "090430", "051900", "000810", "097950", "036570",
]
KOSDAQ = [
    "247540", "086520", "196170", "066970", "028300", "263750", "293490",
    "357780", "058470", "240810", "112040", "095340", "005290", "067160",
    "078600",
]
DEFAULT_TICKERS = KOSPI + KOSDAQ


def _has(path) -> bool:
    return pathlib.Path(path).exists()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    ap.add_argument("--start", default="2016-01-01", help="OHLCV/수급 start (max history)")
    ap.add_argument("--short-start", default=None,
                    help="공매도 start (default = --start). KRX caps 공매도 at ~2y/request "
                         "and ~12s/call, so a later start keeps a big universe tractable.")
    ap.add_argument("--end", default=None, help="default: today")
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--with-balance", action="store_true",
                    help="also fetch 공매도잔고 (doubles 공매도 calls; off by default)")
    ap.add_argument("--pause", type=float, default=0.3, help="seconds between stocks")
    args = ap.parse_args()

    import pandas as pd
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.today().normalize()
    start = pd.Timestamp(args.start)
    short_start = pd.Timestamp(args.short_start) if args.short_start else start

    if not ensure_krx_login():
        print("WARNING: no KRX_ID/KRX_PW — 공매도/수급 will be skipped (OHLCV only).")
    short_fetch = krx_short_fetch(short_start, end, with_balance=args.with_balance)
    flow_fetch = krx_flows_fetch(start, end)

    n_ok = {"ohlcv": 0, "short": 0, "flows": 0}
    print(f"Universe: {len(args.tickers)} names, {start.date()} -> {end.date()}")
    for i, code in enumerate(args.tickers, 1):
        tag = f"[{i:2d}/{len(args.tickers)}] {code}"

        # 1) OHLCV (open access).
        try:
            if _has(pathlib.Path("data") / f"{code}_{args.interval}.csv"):
                n_ok["ohlcv"] += 1; o = "cached"
            else:
                df = get_krx_history(code, start, end, interval=args.interval)
                n_ok["ohlcv"] += 1; o = f"{len(df)} rows"
        except Exception as e:
            o = f"FAIL ({str(e)[:60]})"

        # 2) 공매도 (gated).
        try:
            if _has(short_csv_path(code)):
                n_ok["short"] += 1; s = "cached"
            else:
                sdf = load_short_selling(code, fetch=short_fetch)
                n_ok["short"] += 1; s = f"{len(sdf)} rows"
        except Exception as e:
            s = f"skip ({str(e)[:50]})"

        # 3) 수급 (gated).
        try:
            if _has(flows_csv_path(code)):
                n_ok["flows"] += 1; f = "cached"
            else:
                fdf = load_flows(code, fetch=flow_fetch)
                n_ok["flows"] += 1; f = f"{len(fdf)} rows"
        except Exception as e:
            f = f"skip ({str(e)[:50]})"

        print(f"{tag}  ohlcv:{o:>10s}  공매도:{s:>12s}  수급:{f:>12s}", flush=True)
        if args.pause:
            time.sleep(args.pause)

    print(f"\nDone. OHLCV {n_ok['ohlcv']}, 공매도 {n_ok['short']}, 수급 {n_ok['flows']} "
          f"of {len(args.tickers)} names cached in data/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
