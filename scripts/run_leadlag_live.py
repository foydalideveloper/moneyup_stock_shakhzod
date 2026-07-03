"""Live PAPER runner for the US-shock bounce edge.

Each day: read the prior US overnight return (SPY); if <= threshold, ARM ("buy KR
large-caps at open, sell at close today") and paper-track the realised EW KR intraday
return net of cost. Prints the ARMED/OK state + the paper track record. --backfill
replays the cached history to seed the track. PAPER only — no orders, no secrets.

Data: data/leadlag/ (download_leadlag_history.py) — SPY + KR large-caps.

Usage:
    python scripts/run_leadlag_live.py --backfill
    python scripts/run_leadlag_live.py --status-only
    python scripts/run_leadlag_live.py --threshold -0.02 --slippage-bps 25
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from tagent.data.intraday_history import load_spy_daily_returns, overnight_for_dates  # noqa: E402
from tagent.leadlag_live import LeadlagLiveConfig, LeadlagLiveTrader  # noqa: E402
from tagent.stock_momentum import load_stock_panel  # noqa: E402
from tagent.strategies.us_leadlag_daily import _wide  # noqa: E402
from scripts.download_leadlag_history import KR_BASKET, LEADLAG_DIR  # noqa: E402


def _ew_intraday(panel):
    opens, closes = _wide(panel, "open"), _wide(panel, "close")
    opens, closes = opens.where(opens > 0), closes.where(closes > 0)
    return (closes / opens - 1.0).mean(axis=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=-0.02)
    ap.add_argument("--slippage-bps", type=float, default=15.0)
    ap.add_argument("--us-symbol", default="SPY")
    ap.add_argument("--backfill", action="store_true")
    ap.add_argument("--status-only", action="store_true")
    args = ap.parse_args()

    cfg = LeadlagLiveConfig(threshold=args.threshold, slippage_bps=args.slippage_bps,
                            us_symbol=args.us_symbol)
    trader = LeadlagLiveTrader(cfg)

    if not args.status_only:
        d = str(LEADLAG_DIR)
        spy = load_spy_daily_returns(data_dir=d, symbol=args.us_symbol)
        panel = load_stock_panel("kr", data_dir=d, symbols=KR_BASKET, fields=["open", "close"], min_bars=200)
        if spy.empty or not panel:
            print("Missing data/leadlag/ — run scripts/download_leadlag_history.py first.")
            return 1
        ew = _ew_intraday(panel)
        overnight = overnight_for_dates(spy, ew.index)
        dates = list(ew.index)
        if not args.backfill:                          # daily mode: settle only the latest day
            dates = dates[-1:]
        for dt in dates:
            on = overnight.get(dt.date())
            trader.settle(dt, on, float(ew.loc[dt]) if dt in ew.index else None)
        st = trader.status()
        print(f"Universe: {len(panel)} KR large-caps; US signal {args.us_symbol}; "
              f"threshold {cfg.threshold*100:+.1f}%; round-trip cost {st['round_trip_cost_pct']:.2f}%.")
    else:
        st = trader.status()

    if not st.get("enabled") or st.get("n_trades", 0) == 0 and not st.get("last_date"):
        print("\nNo track yet — run with --backfill (or daily) to start.")
        return 0

    armed = (st["state"] == "ARMED")
    print("\n=== US-shock bounce — LIVE paper track ===")
    print(f"  TODAY ({st.get('last_date')}): US overnight {st.get('last_us_overnight_pct')}%  ->  "
          f"{'ARMED — buy KR open, sell close' if armed else 'OK (flat, no trade)'}")
    print(f"  equity   : ${st['equity']:,.2f}  ({'+' if st['pct_change']>=0 else ''}{st['pct_change']:.2f}%) "
          f"from ${st['capital']:,.0f}")
    print(f"  trades   : {st['n_trades']}  ·  win rate {st['win_rate_pct']:.0f}%  ·  "
          f"expectancy/trade {st['expectancy_pct']:+.4f}%  (net of {st['round_trip_cost_pct']:.2f}% cost)")
    print(f"  since    : {st.get('first_ts')}")
    print(f"\n  state -> data/leadlag_live_state.json   log -> data/leadlag_live.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
