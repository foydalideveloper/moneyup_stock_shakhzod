"""Live PAPER tracker for PEAD (post-earnings drift, positive reaction).

Arms on a positive-reaction earnings disclosure and paper-holds N days net of cost.
--backfill replays the cached non-overlapping events to seed the track; daily mode
books any trades that have just completed and shows currently-open positions.

HONEST: PEAD survives overlap + cost and is momentum-independent, but it is
REGIME-DEPENDENT (positive ~5/11 years) — paper-only forward evidence, not deployable.

Usage:
    python scripts/run_pead_live.py --backfill
    python scripts/run_pead_live.py --status-only
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

from tagent.kr_universe import load_members, membership_panel, universe_symbols  # noqa: E402
from tagent.pead_live import PeadLiveConfig, PeadLiveTrader  # noqa: E402
from tagent.stock_momentum import load_stock_panel  # noqa: E402
from tagent.strategies.earnings_drift import earnings_events, market_regime, pead_trades  # noqa: E402
from tagent.xs_momentum import align_close  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold", type=int, default=20)
    ap.add_argument("--slippage-bps", type=float, default=15.0)
    ap.add_argument("--regime", action="store_true", help="arm only when KR market > 200d MA")
    ap.add_argument("--backfill", action="store_true")
    ap.add_argument("--status-only", action="store_true")
    args = ap.parse_args()
    cfg = PeadLiveConfig(hold=args.hold, slippage_bps=args.slippage_bps, use_regime=args.regime)
    trader = PeadLiveTrader(cfg)

    if not args.status_only:
        members = load_members()
        epath = pathlib.Path("data") / "kr_earnings_disclosures.csv"
        if not members or not epath.exists():
            print("Need PIT membership + data/kr_earnings_disclosures.csv (download_kr_earnings.py).")
            return 1
        syms = universe_symbols(members)
        panel = load_stock_panel("kr", symbols=syms, fields=["open", "close"], min_bars=60)
        idx = align_close(panel).index
        memb = membership_panel(members, idx, symbols=list(panel))
        ed = pd.read_csv(epath, dtype={"symbol": str})
        ev = earnings_events([{"type": "earnings", "symbol": str(r.symbol).zfill(6), "time": r.time}
                              for r in ed.itertuples()])
        regime = market_regime(panel, ma_window=cfg.regime_ma, membership=memb) if cfg.use_regime else None
        trades = pead_trades(panel, ev, hold=cfg.hold, conditional=True, membership=memb,
                             non_overlapping=True, regime=regime)
        for t in trades.itertuples():                  # book chronologically (deduped)
            trader.book_trade(t.entry, t.symbol, t.ret)
        # currently-"open" positions = entries within the last `hold` trading days
        if len(idx) > cfg.hold:
            cutoff = idx[-cfg.hold]
            opens = [{"symbol": t.symbol, "entry": str(pd.Timestamp(t.entry).date())}
                     for t in trades.itertuples() if pd.Timestamp(t.entry) >= cutoff]
            trader.set_open_positions(opens)
        st = trader.status()
        print(f"Universe {len(panel)} PIT names; {len(trades)} non-overlapping {cfg.hold}d "
              f"positive-reaction trades; cost {st['round_trip_cost_pct']:.2f}%.")
    else:
        st = trader.status()

    if not st.get("enabled") or st["n_trades"] == 0:
        print("\nNo track yet — run with --backfill.")
        return 0
    print("\n=== PEAD — LIVE paper track (REGIME-DEPENDENT; forward evidence only) ===")
    print(f"  equity   : ${st['equity']:,.2f}  ({'+' if st['pct_change']>=0 else ''}{st['pct_change']:.2f}%) "
          f"from ${st['capital']:,.0f}")
    print(f"  trades   : {st['n_trades']}  ·  win {st['win_rate_pct']:.0f}%  ·  "
          f"expectancy/trade {st['expectancy_pct']:+.4f}%  (net of {st['round_trip_cost_pct']:.2f}% cost)")
    print(f"  open now : {st['n_open']} position(s)  [{st['state']}]   since {st.get('first_ts')}")
    if st["open_positions"]:
        print("  held:", ", ".join(f"{p['symbol']}({p['entry']})" for p in st["open_positions"][:12]))
    print("\n  state -> data/pead_live_state.json   log -> data/pead_live.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
