"""Live PAPER-trading loop on Alpaca (US). Paper only — never live.

Each new daily bar during US market hours: compute features for the watchlist,
run the trained model, apply the existing risk layer (sizing, stops, daily-loss
kill switch), and place **paper** market orders — logging every decision to CSV.

The decision logic (:func:`decide_long`) is a pure function, and the Alpaca
trading/data clients are injected into :class:`PaperTrader`, so the whole
decision/risk/order wiring is unit-testable with fakes and no network.

Safety: this module never constructs a live client. The runner script builds the
TradingClient with ``paper=True``; nothing here can route to a live account.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Set

import pandas as pd

from tagent.config import DATA_DIR, SETTINGS, Settings
from tagent.risk import RiskManager, RiskParams

_LOG_COLS = ["ts", "symbol", "action", "proba", "price", "qty", "stop", "take", "reason"]


@dataclass
class Decision:
    symbol: str
    action: str          # buy | hold_low_proba | skip_held | halted | no_size | error
    proba: float
    price: float
    qty: int
    stop: float
    take: float
    reason: str


def decide_long(symbol: str, proba: float, price: float, threshold: float,
                held: Set[str], risk: RiskManager) -> Decision:
    """Pure long-only decision: gate on risk halt, probability, existing position,
    and risk-based sizing. Returns a Decision describing the action + why."""
    if not risk.can_trade():
        return Decision(symbol, "halted", proba, price, 0, 0.0, 0.0,
                        f"risk halted: {risk.halt_reason}")
    if price <= 0:
        return Decision(symbol, "error", proba, price, 0, 0.0, 0.0, "no valid price")
    if proba < threshold:
        return Decision(symbol, "hold_low_proba", proba, price, 0, 0.0, 0.0,
                        f"proba {proba:.3f} < threshold {threshold:.2f}")
    if symbol in held:
        return Decision(symbol, "skip_held", proba, price, 0, 0.0, 0.0,
                        "already holding — no add")
    stop, take = risk.stop_take_prices(price, "buy")
    qty = risk.position_size(price, stop)
    if qty <= 0:
        return Decision(symbol, "no_size", proba, price, 0, stop, take,
                        "risk sizing returned 0 shares")
    return Decision(symbol, "buy", proba, price, qty, stop, take,
                    f"proba {proba:.3f} >= {threshold:.2f} | {qty} sh, "
                    f"stop {stop:.2f}, target {take:.2f}")


class PaperTrader:
    def __init__(self, predictor, trading_client, data_client,
                 settings: Settings = SETTINGS, risk: Optional[RiskManager] = None,
                 watchlist: Optional[List[str]] = None, threshold: float = 0.55,
                 lookback_days: int = 200, log_path: Optional[str] = None):
        self.predictor = predictor
        self.trading = trading_client
        self.data = data_client
        self.settings = settings
        self.threshold = threshold
        self.lookback_days = lookback_days
        self.watchlist = [s.upper() for s in (watchlist or settings.watchlist)]
        self.risk = risk or RiskManager(RiskParams(
            account_equity=settings.account_equity,
            risk_per_trade_pct=settings.risk_per_trade_pct,
            max_position_pct=settings.max_position_pct,
            stop_loss_pct=settings.stop_loss_pct,
            take_profit_pct=settings.take_profit_pct,
            max_daily_loss_pct=settings.max_daily_loss_pct,
        ))
        self._last_bar: dict = {}      # symbol -> last acted bar date (one act/bar)

        self._log_path = Path(log_path) if log_path else (Path(DATA_DIR) / "paper_trades.csv")
        new = (not self._log_path.exists()) or self._log_path.stat().st_size == 0
        self._fh = open(self._log_path, "a", newline="", encoding="utf-8")
        self._log = csv.writer(self._fh)
        if new:
            self._log.writerow(_LOG_COLS)
            self._fh.flush()

    # ---------------- broker/data access (thin, injectable) ---------------- #
    def market_open(self) -> bool:
        try:
            return bool(self.trading.get_clock().is_open)
        except Exception:
            return False

    def held_symbols(self) -> Set[str]:
        try:
            return {p.symbol for p in self.trading.get_all_positions()}
        except Exception:
            return set()

    def recent_bars(self, symbol: str) -> pd.DataFrame:
        """The MOST RECENT `lookback_days` daily OHLCV bars for `symbol`.

        Alpaca returns bars forward from `start`, so we request a wide window
        (no small `limit`, which would truncate to the *oldest* bars) and keep the
        tail — i.e. the latest available bars, which feature/prediction need.
        """
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        start = datetime.now(timezone.utc) - timedelta(days=self.lookback_days * 2 + 15)
        req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day,
                               start=start)
        df = self.data.get_stock_bars(req).df
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level=0)
        df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].copy()
        df.index.name = "timestamp"
        return df.tail(self.lookback_days)

    def latest_price(self, symbol: str) -> float:
        from alpaca.data.requests import StockLatestTradeRequest
        t = self.data.get_stock_latest_trade(
            StockLatestTradeRequest(symbol_or_symbols=symbol))[symbol]
        return float(getattr(t, "price", 0) or 0)

    def place_order(self, d: Decision):
        """Submit a PAPER market BUY for the decision's quantity."""
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        req = MarketOrderRequest(symbol=d.symbol, qty=d.qty, side=OrderSide.BUY,
                                 time_in_force=TimeInForce.DAY)
        return self.trading.submit_order(req)

    # ---------------- core loop ---------------- #
    def evaluate(self, symbol: str, held: Set[str]) -> Optional[Decision]:
        """Compute proba + price for one symbol and return its Decision."""
        bars = self.recent_bars(symbol)
        if bars is None or bars.empty:
            return None
        proba = self.predictor.predict_proba_latest(bars)
        try:
            price = self.latest_price(symbol) or float(bars["close"].iloc[-1])
        except Exception:
            price = float(bars["close"].iloc[-1])
        d = decide_long(symbol, proba, price, self.threshold, held, self.risk)
        d._bar_date = bars.index[-1]  # type: ignore[attr-defined]
        return d

    def run_once(self, force: bool = False) -> List[Decision]:
        """One pass over the watchlist. Skips symbols already acted on this bar
        (unless force). Places paper buys and logs every decision."""
        if not force and not self.market_open():
            return []
        held = self.held_symbols()
        decisions: List[Decision] = []
        for sym in self.watchlist:
            try:
                d = self.evaluate(sym, held)
            except Exception as e:  # one bad symbol shouldn't stop the loop
                self._record(Decision(sym, "error", 0.0, 0.0, 0, 0.0, 0.0, str(e)[:120]))
                continue
            if d is None:
                continue
            bar_date = getattr(d, "_bar_date", None)
            if not force and self._last_bar.get(sym) == bar_date:
                continue  # already acted on this bar
            if d.action == "buy":
                try:
                    self.place_order(d)
                except Exception as e:
                    d = Decision(sym, "order_error", d.proba, d.price, d.qty,
                                 d.stop, d.take, str(e)[:120])
            self._last_bar[sym] = bar_date
            self._record(d)
            decisions.append(d)
        return decisions

    def run(self, poll_seconds: int = 60, max_backoff: int = 300) -> None:
        print("PAPER trading (Alpaca). Live orders are NEVER placed. Ctrl+C to stop.")
        print(f"Watchlist: {', '.join(self.watchlist)}  | threshold {self.threshold}")
        last_day = None
        backoff = poll_seconds
        try:
            while True:
                try:
                    today = datetime.now(timezone.utc).date()
                    if today != last_day:
                        self.risk.reset_day()
                        last_day = today
                    if self.market_open():
                        acted = self.run_once()
                        for d in acted:
                            if d.action in ("buy", "order_error", "halted"):
                                print(f"  {d.symbol}: {d.action} — {d.reason}")
                    backoff = poll_seconds
                    time.sleep(poll_seconds)
                except KeyboardInterrupt:
                    raise
                except Exception as e:  # network hiccup -> reconnect with backoff
                    print(f"[paper] loop error: {e}; retrying in {backoff}s")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, max_backoff)
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            self.close()

    # ---------------- logging ---------------- #
    def _record(self, d: Decision) -> None:
        self._log.writerow([datetime.now(timezone.utc).isoformat(), d.symbol, d.action,
                            f"{d.proba:.4f}", f"{d.price:.4f}", d.qty,
                            f"{d.stop:.4f}", f"{d.take:.4f}", d.reason])
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.flush(); self._fh.close()
        except Exception:
            pass
