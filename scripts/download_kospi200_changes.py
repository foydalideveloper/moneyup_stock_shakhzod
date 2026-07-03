"""Reconstruct KOSPI200 add/delete events + fetch event-name prices (pykrx, cached).

Snapshots KOSPI200 constituents before/after each June/December review (diff = adds/deletes),
writes data/kospi200_changes.csv (effective, ticker, action), then fetches daily prices for
every event name not already cached (so the fade study has both legs' prices). MSCI Korea is
a separate provider (not in KRX/pykrx) -> KOSPI200 only.

Usage: python scripts/download_kospi200_changes.py [--start-year 2016]
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

from tagent.config import DATA_DIR  # noqa: E402
from tagent.data.krx_source import ensure_krx_login, get_krx_history  # noqa: E402
from tagent.index_rebalance import reconstruct_changes, review_effective_dates  # noqa: E402
from tagent.index_calibration import load_index_close  # noqa: E402

OUT = pathlib.Path(DATA_DIR) / "kospi200_changes.csv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=2016)
    ap.add_argument("--end-year", type=int, default=2026)
    args = ap.parse_args()
    ensure_krx_login()
    from pykrx import stock

    ti = load_index_close("kospi200_long").index
    effs = review_effective_dates(args.start_year, args.end_year, ti)
    print(f"Reconstructing KOSPI200 changes over {len(effs)} reviews ({args.start_year}-{args.end_year}) ...")
    snapshots = {}
    for eff in effs:
        ei = ti.searchsorted(eff)
        if ei - 10 < 0 or ei + 10 >= len(ti):
            continue
        before_d, after_d = ti[ei - 10], ti[ei + 10]       # ~10 trading days each side
        try:
            before = set(stock.get_index_portfolio_deposit_file("1028", before_d.strftime("%Y%m%d")))
            after = set(stock.get_index_portfolio_deposit_file("1028", after_d.strftime("%Y%m%d")))
        except Exception as e:
            print(f"  {eff.date()}: snapshot ERR {str(e)[:40]}")
            continue
        if before and after:
            snapshots[eff] = {"before": before, "after": after}
            print(f"  {eff.date()}: +{len(after-before)} adds / -{len(before-after)} deletes", flush=True)

    changes = reconstruct_changes(snapshots)
    changes.to_csv(OUT, index=False)
    print(f"\nSaved {len(changes)} changes ({changes['action'].value_counts().to_dict()}) -> {OUT}")

    # fetch prices for event names not already cached
    names = sorted(set(changes["ticker"]))
    missing = [n for n in names if not (pathlib.Path(DATA_DIR) / f"{n}_1d.csv").exists()]
    print(f"\nFetching prices for {len(missing)}/{len(names)} uncached event names ...")
    for i, code in enumerate(missing, 1):
        try:
            get_krx_history(code, "2015-01-01", "2026-06-10", save=True)
        except Exception:
            pass
        if i % 25 == 0 or i == len(missing):
            print(f"  {i}/{len(missing)} fetched", flush=True)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
