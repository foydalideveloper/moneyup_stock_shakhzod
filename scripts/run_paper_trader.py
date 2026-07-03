"""Start the LIVE PAPER trader on Alpaca (US stocks).

PAPER ONLY: the TradingClient is built with paper=True and there is no live
option. During US market hours it runs the trained model on the watchlist each
new daily bar, applies the risk layer, and places paper orders — logging every
decision to data/paper_trades.csv. It stays idle when the market is closed and
reconnects on transient errors.

Prereqs: ALPACA_API_KEY / ALPACA_SECRET_KEY in .env, and a trained model
(scripts/train_model.py). Confirm keys first with scripts/alpaca_smoke_test.py.

Usage:
    python scripts/run_paper_trader.py
    python scripts/run_paper_trader.py --threshold 0.55 --poll 60
    python scripts/run_paper_trader.py --once   # single pass now (e.g. to test wiring)
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from tagent.config import SETTINGS, _DEFAULT_WATCHLIST  # noqa: E402
from tagent.ml.predict import Predictor  # noqa: E402
from tagent.paper import PaperTrader  # noqa: E402
from tagent.risk import RiskManager, RiskParams  # noqa: E402

# Alpaca is a US broker -> always use the US watchlist (ignore any KR override).
US_WATCHLIST = list(_DEFAULT_WATCHLIST)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.55,
                    help="min model probability to buy")
    ap.add_argument("--poll", type=int, default=60, help="seconds between passes")
    ap.add_argument("--symbols", nargs="+", default=US_WATCHLIST)
    ap.add_argument("--once", action="store_true",
                    help="run a single pass now and exit (forced, ignores market hours)")
    args = ap.parse_args()

    if not SETTINGS.has_alpaca_keys():
        print("Missing ALPACA_API_KEY / ALPACA_SECRET_KEY in .env.")
        return 2

    try:
        predictor = Predictor.load()
    except FileNotFoundError:
        print("No trained model. Run: python scripts/train_model.py")
        return 2

    from alpaca.trading.client import TradingClient
    from alpaca.data.historical import StockHistoricalDataClient

    # paper=True is hard-wired — this script can never place live orders.
    trading = TradingClient(SETTINGS.alpaca_api_key, SETTINGS.alpaca_secret_key,
                            paper=True)
    data = StockHistoricalDataClient(SETTINGS.alpaca_api_key, SETTINGS.alpaca_secret_key)

    # Size risk off the actual paper-account equity.
    equity = SETTINGS.account_equity
    try:
        equity = float(trading.get_account().equity)
    except Exception as e:
        print(f"(could not read account equity, using {equity}: {e})")
    risk = RiskManager(RiskParams(
        account_equity=equity,
        risk_per_trade_pct=SETTINGS.risk_per_trade_pct,
        max_position_pct=SETTINGS.max_position_pct,
        stop_loss_pct=SETTINGS.stop_loss_pct,
        take_profit_pct=SETTINGS.take_profit_pct,
        max_daily_loss_pct=SETTINGS.max_daily_loss_pct,
    ))

    trader = PaperTrader(predictor, trading, data, settings=SETTINGS, risk=risk,
                         watchlist=args.symbols, threshold=args.threshold)
    print(f"Paper account equity: {equity:,.0f}  | model AUC "
          f"{predictor.metrics.get('cv_auc_mean', float('nan')):.3f}")

    if args.once:
        decisions = trader.run_once(force=True)
        for d in decisions:
            print(f"  {d.symbol}: {d.action}  proba={d.proba:.3f}  qty={d.qty}  {d.reason}")
        print(f"\n{len(decisions)} decisions logged to data/paper_trades.csv")
        trader.close()
        return 0

    trader.run(poll_seconds=args.poll)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
