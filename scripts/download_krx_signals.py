"""Fetch the additional gated KR signals for the universe (authenticated pykrx).

Signals pykrx supports (investor_detail, short_balance) are fetched and cached to
data/<code>_<signal>.csv. Signals pykrx does NOT support (lending/대차,
margin/신용, program/프로그램) are reported with the exact KRX menu path + columns
to export manually — they don't abort the run.

Needs a WORKING KRX login (KRX_ID/KRX_PW in .env). short_balance has KRX's ~2y
per-request cap, so it's fetched in yearly chunks (slow ~1min/name); a later
--short-start keeps a big universe tractable.

Usage:
    python scripts/download_krx_signals.py --tickers 005930 000660 035420
    python scripts/download_krx_signals.py --signals investor_detail short_balance
"""

import argparse
import pathlib
import sys
import time

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from tagent.data.krx_signals import (  # noqa: E402
    SIGNALS, ManualCsvRequired, krx_signal_fetch, load_signal,
)
from tagent.data.krx_source import ensure_krx_login  # noqa: E402

try:
    from scripts.download_krx_universe import DEFAULT_TICKERS
except Exception:
    DEFAULT_TICKERS = ["005930", "000660", "035420"]

SUPPORTED = [k for k, s in SIGNALS.items() if s.pykrx_supported]
MANUAL = [k for k, s in SIGNALS.items() if not s.pykrx_supported]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    ap.add_argument("--signals", nargs="+", default=SUPPORTED, choices=list(SIGNALS))
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--short-start", default="2022-01-01", help="short_balance start (~2y/request cap)")
    ap.add_argument("--end", default=None)
    ap.add_argument("--pause", type=float, default=0.3)
    args = ap.parse_args()

    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.today().normalize()
    if not ensure_krx_login():
        print("WARNING: no KRX_ID/KRX_PW — gated signals will be empty.")

    fetched, manual_needed = {s: 0 for s in args.signals}, set()
    print(f"Signals {args.signals} for {len(args.tickers)} names")
    for sig in args.signals:
        if not SIGNALS[sig].pykrx_supported:
            manual_needed.add(sig)
            continue
        start = pd.Timestamp(args.short_start) if sig == "short_balance" else pd.Timestamp(args.start)
        fetch = krx_signal_fetch(sig, start, end)
        for i, code in enumerate(args.tickers, 1):
            try:
                df = load_signal(sig, code, fetch=fetch)
                fetched[sig] += 1
                print(f"  [{sig}] {code}: {len(df)} rows", flush=True)
            except ManualCsvRequired:
                manual_needed.add(sig); break
            except Exception as e:
                print(f"  [{sig}] {code}: FAILED ({str(e)[:60]})", flush=True)
            time.sleep(args.pause)

    print("\n=== summary ===")
    for sig in args.signals:
        if SIGNALS[sig].pykrx_supported:
            print(f"  REAL   {sig}: {fetched[sig]}/{len(args.tickers)} cached")
    for sig in sorted(manual_needed | set(MANUAL)):
        spec = SIGNALS[sig]
        print(f"  MANUAL {sig} ({spec.name}): needs a KRX CSV")
        print(f"           {spec.manual_menu}")
        print(f"           columns: 일자 + {spec.manual_columns}  ->  data/<code>_{sig}.csv")
