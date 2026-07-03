"""Live orchestrator: feed -> state -> strategy -> risk -> alerts (+ tick log).

The feed is injectable, so tests can drive the Monitor with fake Quote/Trade
events and no network. In production the default AlpacaFeed is used.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from tagent.alerts import Alerter
from tagent.config import DATA_DIR, SETTINGS, Settings
from tagent.feeds.base import MarketDataFeed, OrderBookSnapshot, Quote, Trade
from tagent.orderbook import OrderBookMemory
from tagent.risk import RiskManager, RiskParams
from tagent.state import SymbolState
from tagent.strategy import StrategyParams, evaluate


class Monitor:
    def __init__(self, settings: Settings = SETTINGS,
                 feed: Optional[MarketDataFeed] = None,
                 alerter: Optional[Alerter] = None,
                 risk: Optional[RiskManager] = None,
                 strategy_params: Optional[StrategyParams] = None,
                 log_path: Optional[str] = None):
        self.settings = settings
        self.states = {
            sym: SymbolState(sym, window_seconds=settings.rolling_window_seconds)
            for sym in settings.watchlist
        }
        # Level-2 depth memory + a per-symbol buffer of trades that printed
        # since the last order-book snapshot (used to tell fills from cancels).
        self.orderbooks = {sym: OrderBookMemory(sym) for sym in settings.watchlist}
        self._trades_since_ob: dict[str, List[float]] = {
            sym: [] for sym in settings.watchlist
        }
        self.params = strategy_params or StrategyParams()
        self.alerter = alerter or Alerter(
            cooldown_seconds=settings.alert_cooldown_seconds,
            telegram_token=settings.telegram_bot_token,
            telegram_chat_id=settings.telegram_chat_id,
        )
        self.risk = risk or RiskManager(RiskParams(
            account_equity=settings.account_equity,
            risk_per_trade_pct=settings.risk_per_trade_pct,
            max_position_pct=settings.max_position_pct,
            stop_loss_pct=settings.stop_loss_pct,
            take_profit_pct=settings.take_profit_pct,
            max_daily_loss_pct=settings.max_daily_loss_pct,
        ))
        self.feed = feed

        self._log_path = Path(log_path) if log_path else (Path(DATA_DIR) / "ticks.csv")
        self._log_fh = open(self._log_path, "a", newline="")
        self._log = csv.writer(self._log_fh)
        if self._log_path.stat().st_size == 0:
            self._log.writerow(["ts", "symbol", "type", "price", "bid", "ask"])

    # ---------------- event handlers ---------------- #
    def on_quote(self, q: Quote) -> None:
        st = self.states.get(q.symbol)
        if st is None:
            return
        st.update_quote(q.bid_price, q.ask_price, q.bid_size, q.ask_size, q.timestamp)
        self._log.writerow([q.timestamp.isoformat(), q.symbol, "quote", "", q.bid_price, q.ask_price])
        self._run_rules(q.symbol)

    def on_trade(self, t: Trade) -> None:
        st = self.states.get(t.symbol)
        if st is None:
            return
        st.update_trade(t.price, t.size, t.timestamp)
        # Remember the print so the next order-book diff can attribute vanished
        # levels to fills vs. cancels.
        buf = self._trades_since_ob.get(t.symbol)
        if buf is not None:
            buf.append(t.price)
        self._log.writerow([t.timestamp.isoformat(), t.symbol, "trade", t.price, "", ""])
        self._run_rules(t.symbol)

    def on_orderbook(self, ob: OrderBookSnapshot) -> None:
        mem = self.orderbooks.get(ob.symbol)
        if mem is None:
            return
        trades = self._trades_since_ob.get(ob.symbol, [])
        mem.update(ob, recent_trades=trades)
        trades.clear()  # consumed; reset the window for the next snapshot
        best_bid = ob.bids[0][0] if ob.bids else ""
        best_ask = ob.asks[0][0] if ob.asks else ""
        self._log.writerow(
            [ob.timestamp.isoformat(), ob.symbol, "orderbook", "", best_bid, best_ask])
        self._run_rules(ob.symbol)

    def orderbook_snapshot(self, symbol: str) -> Optional[dict]:
        """Expose the Level-2 flow features for a symbol (features/agents)."""
        mem = self.orderbooks.get(symbol)
        return mem.snapshot() if mem is not None else None

    def _run_rules(self, symbol: str) -> None:
        if not self.risk.can_trade():
            return  # kill switch / daily loss limit active
        st = self.states[symbol]
        for sig in evaluate(st, self.params, stale_after=self.settings.stale_after_seconds):
            entry = sig.price
            stop, take = self.risk.stop_take_prices(entry, sig.side)
            shares = self.risk.position_size(entry, stop)
            if shares <= 0:
                continue  # risk layer says no size -> skip the alert
            sig.reason += f" | size {shares} sh, stop {stop:.2f}, target {take:.2f}"
            self.alerter.dispatch(sig)

    # ---------------- lifecycle ---------------- #
    def run(self) -> None:
        if self.feed is None:
            if not self.settings.has_alpaca_keys():
                raise SystemExit(
                    "Missing ALPACA_API_KEY / ALPACA_SECRET_KEY. "
                    "Copy .env.example to .env and fill them in.")
            from tagent.feeds.alpaca_feed import AlpacaFeed
            self.feed = AlpacaFeed(
                self.settings.watchlist,
                self.settings.alpaca_api_key,
                self.settings.alpaca_secret_key,
                feed=self.settings.data_feed,
            )
        self.feed.on_quote(self.on_quote)
        self.feed.on_trade(self.on_trade)
        self.feed.on_orderbook(self.on_orderbook)
        print(f"Watching {len(self.settings.watchlist)} symbols: "
              f"{', '.join(self.settings.watchlist)}")
        print("Real-time data flows during US market hours. Ctrl+C to stop.\n")
        try:
            self.feed.run()
        finally:
            self.close()

    def close(self) -> None:
        try:
            self._log_fh.flush()
            self._log_fh.close()
        except Exception:
            pass
