"""Risk management — the layer that actually prevents catastrophic mistakes.

Prediction accuracy does not keep you safe; bounded losses do. This module
sizes positions by risk, computes stop/take-profit prices, tracks the day's
realized loss, and trips a kill switch when the daily loss limit is breached.

All pure logic — no network, fully unit-testable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class RiskParams:
    account_equity: float = 100_000.0
    risk_per_trade_pct: float = 0.5   # % of equity risked per trade
    max_position_pct: float = 10.0    # max % of equity in one position
    stop_loss_pct: float = 2.0
    take_profit_pct: float = 4.0
    max_daily_loss_pct: float = 3.0


class RiskManager:
    def __init__(self, params: RiskParams):
        self.params = params
        self.realized_pnl_today: float = 0.0
        self.halted: bool = False
        self.halt_reason: str = ""

    # ---------------- sizing ---------------- #
    def position_size(self, entry_price: float, stop_price: float | None = None) -> int:
        """Number of shares to buy so that hitting the stop loses ~risk_per_trade.

        Capped by max_position_pct. Returns 0 if trading is halted or inputs are
        invalid.
        """
        if self.halted or entry_price <= 0:
            return 0

        # Per-share risk: explicit stop if given, else the default stop %.
        if stop_price is not None and stop_price > 0:
            per_share_risk = abs(entry_price - stop_price)
        else:
            per_share_risk = entry_price * self.params.stop_loss_pct / 100.0
        if per_share_risk <= 0:
            return 0

        risk_amount = self.params.account_equity * self.params.risk_per_trade_pct / 100.0
        shares_by_risk = risk_amount / per_share_risk

        max_value = self.params.account_equity * self.params.max_position_pct / 100.0
        shares_by_value = max_value / entry_price

        shares = math.floor(min(shares_by_risk, shares_by_value))
        return max(0, int(shares))

    # ---------------- stop / take-profit ---------------- #
    def stop_take_prices(self, entry_price: float, side: str) -> tuple[float, float]:
        sl = self.params.stop_loss_pct / 100.0
        tp = self.params.take_profit_pct / 100.0
        if side == "buy":
            return entry_price * (1 - sl), entry_price * (1 + tp)
        elif side == "sell":  # short
            return entry_price * (1 + sl), entry_price * (1 - tp)
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")

    # ---------------- daily loss / kill switch ---------------- #
    def register_trade_result(self, pnl: float) -> None:
        self.realized_pnl_today += pnl
        max_loss = self.params.account_equity * self.params.max_daily_loss_pct / 100.0
        if self.realized_pnl_today <= -max_loss:
            self.halted = True
            self.halt_reason = (f"daily loss limit hit: {self.realized_pnl_today:.2f} "
                                f"<= -{max_loss:.2f}")

    def trip_kill_switch(self, reason: str) -> None:
        self.halted = True
        self.halt_reason = reason

    def can_trade(self) -> bool:
        return not self.halted

    def reset_day(self) -> None:
        self.realized_pnl_today = 0.0
        self.halted = False
        self.halt_reason = ""
