"""Run the live PAPER track for the deployable KR momentum strategy.

Point-in-time top-N KR universe + 12-1 momentum + market-regime filter, LONG-ONLY,
monthly rebalanced. Prints the current target holdings, exposure %, regime state
(in-market / cash), and the paper track record vs an equal-weight basket. PAPER
only — no orders.

Universe: the CURRENT point-in-time top-N by market cap (pykrx, needs KRX login);
falls back to the latest cached snapshot in data/kr_pit_members.csv if offline.
Prices: cached data/<code>_1d.csv (download with scripts/download_kr_pit_universe.py).

Usage:
    python scripts/run_momentum_live.py                 # show status; rebalance if a new month
    python scripts/run_momentum_live.py --rebalance-now # force a rebalance now
    python scripts/run_momentum_live.py --top-n 100 --status-only
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

from tagent.kr_universe import load_members, top_caps_on, universe_symbols  # noqa: E402
from tagent.momentum_live import MomentumLiveConfig, MomentumLiveTrader  # noqa: E402
from tagent.stock_momentum import load_stock_panel  # noqa: E402
from tagent.xs_momentum import align_close  # noqa: E402


def current_universe(top_n, market="KOSPI"):
    """Today's point-in-time top-N by market cap (live via pykrx), or the latest
    cached snapshot if the network/login isn't available."""
    try:
        from tagent.data.krx_source import ensure_krx_login
        ensure_krx_login()
        from pykrx import stock
        today = pd.Timestamp.today().normalize()
        for back in range(0, 7):                       # walk back to the last trading day
            uni = top_caps_on(today - pd.Timedelta(days=back), top_n=top_n, market=market, stock=stock)
            if uni:
                return uni, f"live pykrx top-{top_n} {market} ({(today - pd.Timedelta(days=back)).date()})"
    except Exception as e:
        print(f"  (live universe unavailable: {str(e)[:60]} — using cached snapshot)")
    members = load_members()
    if members:
        latest = sorted(members)[-1]
        return members[latest], f"cached snapshot {latest}"
    return [], "no universe available"


def rebalance_once(trader, top_n=100, force=False):
    """Do ONE live rebalance: build the current point-in-time universe, load the
    cached price panel, and step the trader at the latest available bar. Returns
    ``(status_or_None, message)``. Reused by both this CLI and the dashboard's
    auto-scheduler so there's a single rebalance path."""
    cfg = trader.cfg
    uni, src = current_universe(top_n)
    if not uni:
        return None, "No universe available. Run scripts/download_kr_pit_universe.py first."
    syms = sorted(set(uni) | set(universe_symbols(load_members() or {})))
    panel = load_stock_panel("kr", symbols=syms, min_bars=cfg.lookback + cfg.skip_recent + 5)
    uni = [s for s in uni if s in panel]
    asof = max((df.index.max() for df in panel.values()), default=None)
    if asof is None or not uni:
        return None, "No cached OHLCV. Run scripts/download_kr_pit_universe.py first."
    st = trader.step(panel, uni, asof=asof, force=force)
    return st, f"Universe: {len(uni)} names ({src}); priced panel {len(panel)} names; asof {asof.date()}."


def _members_asof(members, asof):
    """The point-in-time member list from the latest snapshot dated <= asof."""
    snaps = [d for d in sorted(members) if pd.Timestamp(d) <= pd.Timestamp(asof)]
    return members[snaps[-1]] if snaps else []


def backfill(trader, cfg):
    """Seed a multi-year monthly paper track from cached history, using each month's
    point-in-time snapshot as the universe. No-lookahead: every step slices to asof."""
    members = load_members()
    if not members:
        print("No data/kr_pit_members.csv — run scripts/download_kr_pit_universe.py first.")
        return 1
    syms = universe_symbols(members)
    panel = load_stock_panel("kr", symbols=syms, min_bars=cfg.lookback + cfg.skip_recent + 5)
    idx = align_close(panel).index
    if len(idx) == 0:
        print("No cached OHLCV. Run scripts/download_kr_pit_universe.py first.")
        return 1
    start, end = idx.min(), idx.max()
    print(f"Backfill: {len(panel)} names, {start.date()} -> {end.date()}, monthly...")
    # exactly one asof per calendar month: that month's LAST trading day
    month_ends = idx.to_series().groupby(idx.to_period("M")).max()
    warmup = cfg.lookback + cfg.skip_recent + 2
    n = 0
    for asof in month_ends:
        if len(idx[idx <= asof]) < warmup:
            continue
        uni = [s for s in _members_asof(members, asof) if s in panel]
        if len(uni) < 5:
            continue
        trader.step(panel, uni, asof=asof)
        n += 1
    print(f"Backfilled {n} monthly rebalances.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=100)
    ap.add_argument("--rebalance-now", action="store_true")
    ap.add_argument("--status-only", action="store_true")
    ap.add_argument("--backfill", action="store_true",
                    help="replay monthly over cached history to seed the paper track")
    ap.add_argument("--capital", type=float, default=10_000.0)
    args = ap.parse_args()

    cfg = MomentumLiveConfig(top_n=args.top_n, capital=args.capital)
    trader = MomentumLiveTrader(cfg)

    if args.backfill:
        rc = backfill(trader, cfg)
        if rc:
            return rc
        st = trader.status()
    elif not args.status_only:
        st, info = rebalance_once(trader, args.top_n, force=args.rebalance_now)
        print(info)
        if st is None:
            return 1
    else:
        st = trader.status()

    if not st.get("enabled") or st.get("cycle", 0) == 0:
        print("\nNo rebalance yet. Run without --status-only (or with --rebalance-now) to start the track.")
        return 0

    eqsign = "+" if st["pct_change"] >= 0 else ""
    bksign = "+" if st["basket_pct"] >= 0 else ""
    print("\n=== KR momentum (long-only) — LIVE paper track ===")
    print(f"  strategy : {st['strategy']}")
    print(f"  regime   : {st['regime'].upper()}  (exposure {st['exposure_pct']:.0f}%)   "
          f"in-market {st['frac_in_market_pct']:.0f}% of {st['cycle']} months")
    print(f"  equity   : ${st['equity']:,.2f}  ({eqsign}{st['pct_change']:.2f}%)   "
          f"from ${st['capital']:,.0f}   net of costs ${st['costs']:,.2f}")
    print(f"  basket   : ${st['basket_equity']:,.2f}  ({bksign}{st['basket_pct']:.2f}%)   "
          f"(equal-weight universe, for comparison)")
    print(f"  annualized: {st['ann_return_pct']:+.2f}%   since {st.get('first_ts','—')[:10]}   "
          f"universe {st['universe_size']}")
    print(f"  rebalanced: {str(st.get('last_rebalanced') or '—')[:10]}   "
          f"next: {st.get('next_rebalance') or '—'}")
    act = st.get("actions") or {}
    fmt_a = lambda xs: ", ".join(xs) if xs else "—"
    tnote = "  ⚠ HIGH" if st.get("high_turnover") else ""
    print(f"  turnover  : {st.get('turnover_pct', 0.0):.0f}% of names changed{tnote}")
    print(f"\n  BUY : {fmt_a(act.get('to_buy'))}")
    print(f"  SELL: {fmt_a(act.get('to_sell'))}")
    print(f"  HOLD: {fmt_a(act.get('to_hold'))}")
    if st.get("high_turnover"):
        print("  WARNING: turnover >60% in one rebalance — check the universe source/format "
              "(cached snapshot vs live pykrx) before trusting the rotation.")
    if st["held"]:
        print(f"\n  target holdings ({st['n_held']}, equal-weight):")
        for h in st["held"]:
            print(f"    {h['symbol']}   {h['weight']*100:5.2f}%   ${h['notional']:,.0f}")
    else:
        print("\n  target holdings: NONE — in cash (market below its regime moving average).")
    print(f"\n  log -> data/momentum_live.csv   state -> data/momentum_live_state.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
