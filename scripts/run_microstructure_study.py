"""Does crypto order-book microstructure predict short-term price moves?

Reads data/microstructure_<symbol>.csv (from scripts/record_orderbook.py) and,
per feature and horizon (5s/30s/60s), reports the Information Coefficient and a
STRICT out-of-sample test (train on the earlier part, test on the later) — OOS IC,
directional hit rate, and the typical move vs trading costs. Prints a clear
edge / no-edge verdict.

Usage:
    python scripts/run_microstructure_study.py                 # BTCUSDT, ETHUSDT
    python scripts/run_microstructure_study.py --symbols BTCUSDT --cost-bps 10
"""

import argparse
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from tagent.config import DATA_DIR  # noqa: E402
from tagent.microstructure import (  # noqa: E402
    DEFAULT_FEATURES, DEFAULT_HORIZONS, study, verdict,
)


def _load(symbol, data_dir):
    path = pathlib.Path(data_dir) / f"microstructure_{symbol}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    ap.add_argument("--horizons", nargs="+", type=int, default=DEFAULT_HORIZONS)
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--cost-bps", type=float, default=10.0,
                    help="round-trip cost hurdle (spread + fees), basis points")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    args = ap.parse_args()

    any_edge = False
    for sym in args.symbols:
        df = _load(sym, args.data_dir)
        if df is None or len(df) < 60:
            print(f"\n{sym}: no/too-little data "
                  f"({0 if df is None else len(df)} rows). Run record_orderbook.py longer.")
            continue
        span_s = (pd.to_datetime(df['ts']).max() - pd.to_datetime(df['ts']).min()).total_seconds()
        rows = study(df, horizons=args.horizons, train_frac=args.train_frac)
        v = verdict(rows, cost_bps=args.cost_bps)
        any_edge = any_edge or v["has_edge"]

        print(f"\n=== {sym}: {len(df)} rows, {span_s/60:.1f} min, "
              f"OOS test = last {(1-args.train_frac)*100:.0f}% (no lookahead) ===")
        print(f"  {'feature':18s} {'H(s)':>4s} {'IC':>7s} {'OOS-IC':>7s} "
              f"{'OOS-hit':>8s} {'|move|bps':>9s} {'n_test':>7s}")
        for r in rows:
            print(f"  {r['feature']:18s} {r['horizon_s']:>4d} {r['ic']:>7.3f} "
                  f"{r['oos_ic']:>7.3f} "
                  f"{(r['oos_hit'] if r['oos_hit']==r['oos_hit'] else float('nan')):>8.3f} "
                  f"{r['mean_abs_bps']:>9.2f} {r['n_test']:>7d}")
        if v["has_edge"]:
            print(f"  -> EDGE (OOS-IC≥{v['ic_thresh']}, hit≥{v['hit_thresh']}, "
                  f"move>{v['cost_bps']}bps):")
            for e in v["edges"]:
                print(f"       {e['feature']} @ {e['horizon_s']}s: "
                      f"OOS-IC {e['oos_ic']:+.3f}, hit {e['oos_hit']:.3f}, move {e['mean_abs_bps']:.1f}bps")
        elif v["statistical_only"]:
            print(f"  -> statistical signal but NOT tradable (clears OOS-IC + hit, "
                  f"but the move is below the {v['cost_bps']}bps cost):")
            for e in v["statistical_only"]:
                print(f"       {e['feature']} @ {e['horizon_s']}s: "
                      f"OOS-IC {e['oos_ic']:+.3f}, hit {e['oos_hit']:.3f}, move only {e['mean_abs_bps']:.1f}bps")
        else:
            print(f"  -> no signal (nothing clears OOS-IC≥{v['ic_thresh']} + hit≥{v['hit_thresh']})")

    print("\n================ VERDICT ================")
    if any_edge:
        print("  POSSIBLE EDGE found on at least one feature/horizon (see above).")
        print("  Treat with caution: confirm on MUCH more data + walk-forward, net of costs.")
    else:
        print("  NO TRADABLE EDGE. Order-book imbalance may predict the very-short-horizon")
        print("  DIRECTION (a real statistical signal), but the move is too small to clear")
        print("  the spread + fees — so there is no net-of-cost edge in this recording.")
    print("  (Forward returns use future mid; OOS split is strictly chronological.)")
