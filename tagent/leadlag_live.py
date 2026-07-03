"""Live PAPER runner for the US-shock bounce edge (the #2 daily candidate).

Each day: look at the prior US overnight return (SPY). If it is <= ``threshold``
(a sharp US down-night), ARM — "BUY KR large-caps at the open, sell at the close
today" — and paper-track the realised equal-weight KR intraday return NET of one KR
round trip. On a quiet US night it stays OK (flat, no trade). Persists a track
record so it resumes across restarts (data/leadlag_live_state.json + leadlag_live.csv).

Honest status: on the survivorship-corrected universe this edge only marginally
clears the base cost and turns negative under conservative slippage — so this runner
is to gather FORWARD paper evidence, not a proven money-maker. Configure a
conservative cost to keep the track honest.

STRICTLY NO-LOOKAHEAD: the arm decision uses the US session that closed BEFORE the KR
open; the trade is opened at that open and closed the same day. Pure + injectable;
unit-tested with mock data and no network. PAPER only — no orders.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from tagent.config import DATA_DIR
from tagent.intraday_backtest import CostModel


@dataclass
class LeadlagLiveConfig:
    threshold: float = -0.02          # ARM when the US overnight return <= this
    capital: float = 10_000.0
    us_symbol: str = "SPY"
    slippage_bps: float = 15.0        # conservative-ish (round trip ~0.51%)

    def cost(self) -> CostModel:
        return CostModel(slippage_bps=self.slippage_bps)


def is_armed(us_overnight: Optional[float], threshold: float) -> bool:
    """True when the prior US overnight return is a sharp enough drop to trade."""
    return us_overnight is not None and not pd.isna(us_overnight) and float(us_overnight) <= threshold


class LeadlagLiveTrader:
    CSV_FIELDS = ["date", "us_overnight", "armed", "kr_intraday", "net", "equity"]

    def __init__(self, cfg: Optional[LeadlagLiveConfig] = None, data_dir=None, clock=None):
        self.cfg = cfg or LeadlagLiveConfig()
        self.dir = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
        self.state_path = self.dir / "leadlag_live_state.json"
        self.csv_path = self.dir / "leadlag_live.csv"
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.equity = self.cfg.capital
        self.trades = 0
        self.wins = 0
        self.sum_net = 0.0
        self.armed = False
        self.last_date: Optional[str] = None
        self.last_us_overnight: Optional[float] = None
        self.last_settled: Optional[str] = None       # last fully-processed trading day
        self.first_ts: Optional[str] = None
        self._load()

    # ------------------------------------------------------------------ #
    def settle(self, date, us_overnight: Optional[float],
               kr_intraday_ret: Optional[float] = None) -> dict:
        """Process one day. Updates the ARM state from ``us_overnight``; books a paper
        trade (EW KR intraday return net of cost) only on a NEW armed day for which the
        realised ``kr_intraday_ret`` is known. Deduped by date (no double counting)."""
        ds = str(pd.Timestamp(date).date())
        self.last_date = ds
        self.last_us_overnight = None if us_overnight is None else float(us_overnight)
        self.armed = is_armed(us_overnight, self.cfg.threshold)
        new_day = self.last_settled is None or ds > self.last_settled

        if new_day and self.armed and kr_intraday_ret is not None:
            net = float(kr_intraday_ret) - self.cfg.cost().round_trip_frac()
            self.equity *= (1.0 + net)
            self.trades += 1
            self.wins += int(net > 0)
            self.sum_net += net
            self.last_settled = ds
            self._append_csv(ds, us_overnight, True, kr_intraday_ret, net)
        elif new_day and not self.armed:
            self.last_settled = ds                    # quiet US night: seen, flat
        if self.first_ts is None:
            self.first_ts = ds
        self._save()
        return self.status()

    # ------------------------------------------------------------------ #
    def status(self) -> dict:
        cap = self.cfg.capital
        wr = (self.wins / self.trades) if self.trades else 0.0
        exp = (self.sum_net / self.trades) if self.trades else 0.0
        return {
            "enabled": True, "strategy": "US-shock bounce (buy KR open / sell close)",
            "state": "ARMED" if self.armed else "OK", "armed": self.armed,
            "threshold": self.cfg.threshold, "threshold_pct": round(self.cfg.threshold * 100, 2),
            "round_trip_cost_pct": round(self.cfg.cost().round_trip_frac() * 100, 3),
            "last_date": self.last_date,
            "last_us_overnight_pct": (round(self.last_us_overnight * 100, 3)
                                      if self.last_us_overnight is not None else None),
            "capital": cap, "equity": round(self.equity, 2),
            "pct_change": round((self.equity / cap - 1.0) * 100, 3),
            "n_trades": self.trades, "win_rate_pct": round(wr * 100, 1),
            "expectancy_pct": round(exp * 100, 4),
            "first_ts": self.first_ts,
        }

    # ------------------------------------------------------------------ #
    def _save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        state = {"cfg": asdict(self.cfg), "equity": self.equity, "trades": self.trades,
                 "wins": self.wins, "sum_net": self.sum_net, "armed": self.armed,
                 "last_date": self.last_date, "last_us_overnight": self.last_us_overnight,
                 "last_settled": self.last_settled, "first_ts": self.first_ts}
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        tmp.replace(self.state_path)

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            s = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return
        self.equity = float(s.get("equity", self.cfg.capital))
        self.trades = int(s.get("trades", 0))
        self.wins = int(s.get("wins", 0))
        self.sum_net = float(s.get("sum_net", 0.0))
        self.armed = bool(s.get("armed", False))
        self.last_date = s.get("last_date")
        self.last_us_overnight = s.get("last_us_overnight")
        self.last_settled = s.get("last_settled")
        self.first_ts = s.get("first_ts")

    def _append_csv(self, ds, us_overnight, armed, kr_ret, net) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        new = not self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(self.CSV_FIELDS)
            w.writerow([ds, f"{(us_overnight or 0)*100:.3f}", armed,
                        f"{(kr_ret or 0)*100:.3f}", f"{net*100:.3f}", round(self.equity, 2)])


def load_live_status(data_dir=None) -> dict:
    """Persisted status for the dashboard (no stepping). {'enabled': False} if unstarted."""
    path = (Path(data_dir) if data_dir is not None else Path(DATA_DIR)) / "leadlag_live_state.json"
    if not path.exists():
        return {"enabled": False}
    try:
        return LeadlagLiveTrader(data_dir=data_dir).status()
    except Exception:
        return {"enabled": False}
