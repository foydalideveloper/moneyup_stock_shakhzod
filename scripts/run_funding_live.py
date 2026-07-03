"""LIVE paper runner for the cross-sectional funding-carry strategy.

Polls live Binance funding (public API, no key), and on each NEW 8h funding cycle
runs the selection + delta-neutral carry book (tagent/funding_live.py). Logs
positions + equity to data/funding_live.csv and prints live status. Paper only —
places NO real orders.

Usage:
    python scripts/run_funding_live.py
    python scripts/run_funding_live.py --capital 10000 --top-n 8 --leverage 3 --poll-seconds 300
    python scripts/run_funding_live.py --reset
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

from tagent.config import DATA_DIR  # noqa: E402
from tagent.data.funding import LIQUID_COINS, fetch_live_funding  # noqa: E402
from tagent.funding_live import FundingCarryTrader, LiveConfig  # noqa: E402


def _print_status(st):
    m = st["margin"]
    print(f"\n[cycle {st['cycle']}]  equity ${st['equity']:,.2f}  "
          f"({st['pct_change']:+.3f}%)  accrued +${st['accrued']:,.2f}  "
          f"costs ${st['costs']:,.2f}  annualized {st['ann_yield_pct']:+.2f}%")
    print(f"  margin: min ratio {m['min_margin_ratio']:.3f} "
          f"(~{m['headroom_pct']:.1f}% headroom @ {m['leverage']:.0f}x, maint {m['maint_margin_rate']:.3f})")
    if st["held"]:
        for h in st["held"]:
            print(f"    {h['coin']:10s} {h['weight']*100:5.1f}%  ${h['notional']:>8,.0f}  "
                  f"funding {h['funding_bps']:+.2f} bps/8h")
    else:
        print("    no coins held — funding below hurdle; sitting flat (no money-losing carry).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=LIQUID_COINS)
    ap.add_argument("--capital", type=float, default=10_000.0)
    ap.add_argument("--hurdle-bps", type=float, default=1.0)
    ap.add_argument("--band-bps", type=float, default=4.0)
    ap.add_argument("--top-n", type=int, default=8)
    ap.add_argument("--max-weight", type=float, default=0.25)
    ap.add_argument("--leverage", type=float, default=3.0)
    ap.add_argument("--poll-seconds", type=float, default=300.0)
    ap.add_argument("--reset", action="store_true", help="wipe state + log and start fresh")
    args = ap.parse_args()

    if args.reset:
        for f in ("funding_live_state.json", "funding_live.csv"):
            p = pathlib.Path(DATA_DIR) / f
            if p.exists():
                p.unlink()
        print("Reset: cleared funding_live state + log.")

    cfg = LiveConfig(universe=tuple(args.symbols), capital=args.capital,
                     hurdle_bps=args.hurdle_bps, band_bps=args.band_bps,
                     top_n=args.top_n, max_weight=args.max_weight, leverage=args.leverage)
    trader = FundingCarryTrader(cfg)
    print(f"\nLIVE funding-carry paper runner — {len(args.symbols)} coins, "
          f"${args.capital:,.0f} capital, top-{args.top_n}, {args.leverage:.0f}x, "
          f"hurdle {args.hurdle_bps}bps. Paper only; resumes from data/funding_live_state.json.")
    print("Polling live Binance funding (public, no key). Ctrl-C to stop.\n")

    while True:
        try:
            snap = fetch_live_funding(args.symbols)
            funding = {c: v["funding_rate"] for c, v in snap.items()}
            prices = {c: v["mark"] for c, v in snap.items()}
            # next funding boundary is shared across perps; use it as the cycle id
            ft = max((v["next_funding_time"] for v in snap.values()), default=0)
            new = trader.last_ft is None or ft > trader.last_ft
            st = trader.step(funding, prices, funding_time=ft)
            if new:
                _print_status(st)
            else:
                hd = ", ".join(h["coin"] for h in st["held"]) or "flat"
                print(f"  (no new funding cycle yet; holding {hd}; "
                      f"equity ${st['equity']:,.2f})", flush=True)
        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as e:
            print(f"  poll error: {str(e)[:80]}")
        time.sleep(args.poll_seconds)
