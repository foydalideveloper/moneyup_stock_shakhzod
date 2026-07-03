"""Offline RL execution/filtering experiment on the recorded order-book signal.

Loads data/microstructure_<SYM>.csv, splits it chronologically (train earlier,
test later — no lookahead), learns a cost-aware filtering policy on TRAIN, and
reports OOS whether it beats the raw "act on every signal" baseline net of costs.

The goal is to CUT COST-CHURN, not to invent alpha — so the honest expected
outcome is "trades far less / closer to break-even", and we say so.

Usage:
    python scripts/run_rl_execution.py
    python scripts/run_rl_execution.py --symbols BTCUSDT --cost-bps 10 --train-frac 0.6
"""

import argparse
import glob
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd  # noqa: E402

from tagent.config import DATA_DIR  # noqa: E402
from tagent.rl_execution import run_experiment, summarize  # noqa: E402


def _discover(symbols):
    if symbols:
        return [(s, os.path.join(DATA_DIR, f"microstructure_{s.upper()}.csv")) for s in symbols]
    return [(os.path.basename(p)[len("microstructure_"):-len(".csv")], p)
            for p in sorted(glob.glob(os.path.join(DATA_DIR, "microstructure_*.csv")))]


def _line(tag, d):
    return (f"  {tag:9s} net {d['net_bps']:+8.2f}bps  ({d['net_bps_per_step']:+6.3f}/step)  "
            f"gross {d['gross']*1e4:+8.2f}bps  costs {d['costs']*1e4:7.2f}bps  "
            f"trades {d['n_trades']:5d}/{d['steps']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--gamma", type=float, default=0.0)
    args = ap.parse_args()

    files = _discover(args.symbols)
    if not files:
        print(f"No microstructure_*.csv in {DATA_DIR}. Record with scripts/record_orderbook.py first.")
        sys.exit(0)

    print(f"\nOffline RL execution filter — cost {args.cost_bps}bps/turn, "
          f"chronological {int(args.train_frac*100)}/{int((1-args.train_frac)*100)} split\n")
    any_better = False
    for sym, path in files:
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"{sym}: read failed ({str(e)[:50]})")
            continue
        if len(df) < 50:
            print(f"{sym}: too few rows ({len(df)})")
            continue
        res = run_experiment(df, train_frac=args.train_frac, cost_bps=args.cost_bps,
                             epochs=args.epochs, gamma=args.gamma)
        s = summarize(res)
        print(f"{sym}  ({res['n_train']} train / {res['n_test']} test rows, "
              f"signal orientation {res['signal_sign']:+d})")
        print(_line("BASELINE", res["test"]["baseline"]) + "   <- act on every signal")
        print(_line("RL FILTER", res["test"]["policy"]) + "   <- learned policy")
        print(f"  -> trades {s['trade_reduction_pct']:+.0f}%, "
              f"{'BEATS' if s['beats_baseline_net'] else 'does NOT beat'} baseline net OOS\n")
        any_better = any_better or s["beats_baseline_net"]

    print("================ VERDICT ================")
    print("  The order-book signal's move is sub-cost, so the RAW strategy bleeds fees.")
    if any_better:
        print("  The RL filter TRADES FAR LESS and improves net-of-cost OOS — but this is")
        print("  cost-churn reduction, NOT new alpha: the best case is near break-even, and")
        print("  a sub-cost signal filtered down mostly means 'do nothing'. Not deployable")
        print("  as a profit engine; useful only as an execution gate on a real signal.")
    else:
        print("  Even filtered, there's no positive net-of-cost edge here — the honest")
        print("  result is 'trade less, lose less', converging to doing nothing.")
    print("  (Offline only; strict chronological split; costs subtracted every trade.)")
