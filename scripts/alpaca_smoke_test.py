"""Smallest Alpaca check: authenticate, show PAPER balance, fetch AAPL quote+trade.

Confirms your keys work before any trading. REST, so it works regardless of
market hours. NEVER prints the secret key.

Usage:
    python scripts/alpaca_smoke_test.py
    python scripts/alpaca_smoke_test.py --symbol MSFT
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from tagent.config import SETTINGS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="AAPL")
    args = ap.parse_args()

    if not SETTINGS.has_alpaca_keys():
        print("Missing ALPACA_API_KEY / ALPACA_SECRET_KEY in .env.")
        return 2

    from alpaca.trading.client import TradingClient
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestQuoteRequest, StockLatestTradeRequest

    # paper=True -> the paper-trading endpoint; never the live account.
    try:
        trading = TradingClient(SETTINGS.alpaca_api_key, SETTINGS.alpaca_secret_key,
                                paper=True)
        acct = trading.get_account()
    except Exception as e:
        print(f"AUTH FAILED: {e}")          # never echoes the secret
        return 1

    print("connection OK")                  # never print the token/secret
    print(f"  account   : {acct.account_number}  ({acct.status})")
    print(f"  cash      : {acct.cash}")
    print(f"  equity    : {acct.equity}")
    print(f"  buying_pwr: {acct.buying_power}")

    try:
        clock = trading.get_clock()
        print(f"  market    : {'OPEN' if clock.is_open else 'CLOSED'} "
              f"(next open {clock.next_open}, next close {clock.next_close})")
    except Exception as e:
        print(f"  clock unavailable: {e}")

    sym = args.symbol.upper()
    try:
        data = StockHistoricalDataClient(SETTINGS.alpaca_api_key, SETTINGS.alpaca_secret_key)
        q = data.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=sym))[sym]
        t = data.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=sym))[sym]
        print(f"\n{sym} quote: bid {q.bid_price} x{q.bid_size}  "
              f"ask {q.ask_price} x{q.ask_size}")
        print(f"{sym} trade: {t.price} x{t.size}  @ {t.timestamp}")
    except Exception as e:
        print(f"\n{sym} market data error: {e}")
        return 1

    print("\nREST OK. Paper keys work. Train a model, then run scripts/run_paper_trader.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
