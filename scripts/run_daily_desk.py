"""Daily Desk runner — pre-open semiconductor recommendations + briefing + paper track.

Builds the morning recommendations (momentum BUYs + SELL/REDUCE for paper-holdings) and
the consolidated briefing for the semi watchlist, paper-tracks the BUYs forward (net of
cost), and writes data/daily_desk.json for the dashboard. Prints the briefing to screen.

Honest labels: recommendations = validated momentum edge; in-session timing (dashboard)
= decision-support only. PAPER only — no orders. Cached daily data; news optional.

Usage:
    python scripts/run_daily_desk.py
    python scripts/run_daily_desk.py --backfill         # forward-test the BUYs monthly
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

from tagent.daily_desk import (  # noqa: E402
    DeskConfig, DeskTracker, assert_sane, briefing_table, buy_candidates, momentum_ranks,
    sell_list, write_desk_snapshot,
)
from tagent.stock_momentum import load_stock_panel  # noqa: E402
from tagent.xs_momentum import align_close  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true")
    ap.add_argument("--hold", type=int, default=21)
    args = ap.parse_args()
    cfg = DeskConfig()

    # Watchlist desk: load the authoritative split-adjusted KR daily bars from pykrx
    # (data/<code>_1d.csv). No PIT membership mask here — this is a fixed semi watchlist,
    # not the survivorship universe. Refresh with scripts/download_kr_pit_universe.py or
    # tagent.data.krx_source.get_krx_history.
    memb = None
    panel = {}
    for s in cfg.watchlist:
        panel.update(load_stock_panel("kr", symbols=[s], fields=["open", "close"], min_bars=60))
    if not panel:
        print("No cached daily data for the watchlist — run get_krx_history for the codes in data/.")
        return 1
    # data-quality guard: corrupt (split-artifact) prices fail loudly, not as fake P&L
    assert_sane(panel)
    idx = align_close(panel).index
    asof = idx[-1]

    tracker = DeskTracker(cfg)
    if args.backfill:
        close = align_close(panel)
        opens = pd.DataFrame({c: d["open"] for c, d in panel.items()}).reindex(index=close.index)
        i = max(cfg.lookback + cfg.skip + 2, 260)
        while i + args.hold < len(close):
            d = close.index[i]
            bc = buy_candidates(panel, cfg, asof=d, membership=memb)
            for b in bc.get("buys", []):
                s = b["symbol"]
                eo = opens[s].iloc[i + 1] if i + 1 < len(close) else None     # buy NEXT open
                ex = close[s].iloc[i + 1 + args.hold] if i + 1 + args.hold < len(close) else None
                if eo and ex and eo > 0:
                    tracker.book(close.index[i + 1], s, ex / eo - 1.0)         # net of cost inside
            i += args.hold
        print(f"Backfilled desk BUY paper-track: {tracker.trades} trades.")

    # latest pre-open recommendations + briefing
    bc = buy_candidates(panel, cfg, asof=asof, membership=memb)
    ranks = momentum_ranks(panel, cfg, asof=asof, membership=memb)
    prev_buys = list(ranks.head(cfg.top_n)["symbol"]) if len(ranks) else []
    stops = {b["symbol"]: b["stop"] for b in bc.get("buys", [])}
    sells = sell_list(prev_buys, panel, cfg, asof=asof, membership=memb, entry_stops=stops)
    brief = briefing_table(cfg.watchlist, {s: align_close(panel)[[s]].rename(columns={s: "close"}).assign(
        volume=float("nan")) if s in align_close(panel).columns else None for s in cfg.watchlist})
    # better briefing: use the full OHLCV-ish panel (open+close; volume not loaded here)
    brief = briefing_table(cfg.watchlist, {s: panel.get(s) for s in cfg.watchlist})

    payload = {
        "enabled": True, "generated": str(asof.date()),
        "regime": bc.get("regime"), "note": bc.get("note"),
        "labels": {"recommendations": "validated momentum edge (12-1, point-in-time)",
                   "in_session": "decision-support only — no proven intraday edge"},
        "buys": bc.get("buys", []), "sells": sells, "briefing": brief,
        "track": tracker.status(), "us_context": ["NVDA", "AMD", "MU", "AVGO", "SOXX"],
    }
    write_desk_snapshot(payload)

    print(f"\n=== DAILY DESK — pre-open ({asof.date()}) · regime {payload['regime']} ===")
    print("BUY candidates (momentum edge):" if payload["buys"] else f"  {payload.get('note','no BUYs')}")
    for b in payload["buys"]:
        print(f"  BUY {b['symbol']}  ~{b['price']:,.0f}  stop {b['stop']:,.0f}  — {b['reason']}")
    if sells:
        print("SELL/REDUCE (paper-holdings):")
        for s in sells:
            print(f"  SELL {s['symbol']}  ~{s['price']:,.0f}  — {'; '.join(s['reasons'])}")
    print("\nBriefing:")
    for r in brief:
        ch = f"{r['change']*100:+.2f}%" if r["change"] is not None else "—"
        print(f"  {r['symbol']}  px {r['price']}  {ch}")
    t = payload["track"]
    print(f"\nPaper track of BUYs: {t['n_trades']} trades · win {t['win_rate_pct']:.0f}% · "
          f"expectancy {t['expectancy_pct']:+.3f}%/trade (net of {t['round_trip_cost_pct']:.2f}%) · "
          f"equity ${t['equity']:,.0f}")
    print("  snapshot -> data/daily_desk.json   (recommendations = momentum edge; in-session = support)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
