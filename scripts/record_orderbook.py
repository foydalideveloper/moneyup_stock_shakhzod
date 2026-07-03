"""Record live crypto order-book microstructure + TRADE PRINTS (Binance public WS).

Writes two files per coin (no API key needed):

  data/microstructure_<symbol>.csv  — ~hz rows/sec of depth state:
    ts, mid, best_bid, best_ask, bid_size, ask_size, depth_imbalance,
    absorption_ratio, spoof_ratio, filled_qty, cancelled_qty

  data/microstructure_trades_<symbol>.csv — EVERY trade print (for the realistic
  market-making sim's queue-position fills):
    ts, price, qty, side   (side +1 = buy-aggressor/lifts ask, -1 = sell-aggressor/hits bid)

Side is classified by trade price vs the latest mid (the public @trade stream
doesn't expose the maker flag here). Leave it running for HOURS on a few liquid
coins, then backtest with scripts/run_market_making.py (realistic mode).

Usage:
    python scripts/record_orderbook.py                       # BTC, ETH, SOL, BNB
    python scripts/record_orderbook.py --symbols btc eth sol --hz 2
"""

import argparse
import csv
import pathlib
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from tagent.config import DATA_DIR  # noqa: E402
from tagent.feeds.crypto_feed import CryptoFeed  # noqa: E402

COLS = ["ts", "mid", "best_bid", "best_ask", "bid_size", "ask_size",
        "depth_imbalance", "absorption_ratio", "spoof_ratio",
        "filled_qty", "cancelled_qty"]
TRADE_COLS = ["ts", "price", "qty", "side"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"])
    ap.add_argument("--hz", type=float, default=1.0, help="max DEPTH rows/sec per symbol")
    ap.add_argument("--no-trades", action="store_true", help="skip recording trade prints")
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    try:
        feed = CryptoFeed(symbols=args.symbols)
    except ValueError as e:
        print(f"Bad symbol: {e}")
        return 2

    base = pathlib.Path(args.data_dir) if args.data_dir else pathlib.Path(DATA_DIR)
    min_dt = 1.0 / max(args.hz, 1e-6)
    writers, files, last_t = {}, {}, {}
    twriters, tcounts, last_mid = {}, {}, {}
    counts = {s: 0 for s in feed.subscriptions}

    def _csv_writer(path, header):
        new = (not path.exists()) or path.stat().st_size == 0
        fh = open(path, "a", newline="", encoding="utf-8")
        w = csv.writer(fh)
        if new:
            w.writerow(header)
            fh.flush()
        return w, fh

    def _writer(symbol):
        if symbol not in writers:
            writers[symbol], files[symbol] = _csv_writer(
                base / f"microstructure_{symbol}.csv", COLS)
        return writers[symbol]

    def _twriter(symbol):
        if symbol not in twriters:
            w, fh = _csv_writer(base / f"microstructure_trades_{symbol}.csv", TRADE_COLS)
            twriters[symbol] = (w, fh)
        return twriters[symbol]

    def on_trade(t):
        if args.no_trades or t.price <= 0:
            return
        mid = last_mid.get(t.symbol)
        side = 0 if mid is None else (1 if t.price >= mid else -1)   # +1 lifts ask, -1 hits bid
        w, fh = _twriter(t.symbol)
        w.writerow([datetime.now(timezone.utc).isoformat(), t.price, t.size, side])
        fh.flush()
        tcounts[t.symbol] = tcounts.get(t.symbol, 0) + 1

    def on_ob(ob):
        if not ob.bids or not ob.asks:
            return
        best_bid, bid_size = ob.bids[0]
        best_ask, ask_size = ob.asks[0]
        mid = (best_bid + best_ask) / 2.0
        last_mid[ob.symbol] = mid                        # fresh each book update (trade side)
        now = time.time()
        if now - last_t.get(ob.symbol, 0.0) < min_dt:   # throttle DEPTH rows to ~hz/sec
            return
        last_t[ob.symbol] = now
        f = feed.book_features(ob.symbol) or {}
        _writer(ob.symbol).writerow([
            datetime.now(timezone.utc).isoformat(), f"{mid:.8f}", best_bid, best_ask,
            bid_size, ask_size, f.get("depth_imbalance", ""),
            f.get("absorption_ratio", ""), f.get("spoof_ratio", ""),
            f.get("filled_qty", ""), f.get("cancelled_qty", ""),
        ])
        files[ob.symbol].flush()
        counts[ob.symbol] = counts.get(ob.symbol, 0) + 1
        total = sum(counts.values())
        if total % 60 == 0:
            print("  " + "  ".join(f"{s}:{n}" for s, n in counts.items()), flush=True)

    feed.on_orderbook(on_ob)
    if not args.no_trades:
        feed.on_trade(on_trade)
    tnote = "" if args.no_trades else " + trade prints"
    print(f"Recording {', '.join(feed.subscriptions)} -> data/microstructure_<symbol>.csv{tnote} "
          f"(~{args.hz:.0f}/sec depth). Public Binance WS, no key. Ctrl+C to stop.\n")
    try:
        feed.run()
    except KeyboardInterrupt:
        pass
    finally:
        feed.stop()
        for fh in list(files.values()) + [fh for _, fh in twriters.values()]:
            try:
                fh.flush(); fh.close()
            except Exception:
                pass
        print(f"\nStopped. depth rows: " + ", ".join(f"{s}={n}" for s, n in counts.items()))
        if tcounts:
            print("trade prints: " + ", ".join(f"{s}={n}" for s, n in tcounts.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
