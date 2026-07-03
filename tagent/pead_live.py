"""Live PAPER tracker for the PEAD edge (post-earnings drift, positive reaction).

Arms when a name gaps UP on an earnings disclosure (positive announcement reaction)
and paper-holds it ``hold`` trading days, booking the drift NET of one KR round trip.
Persists a track record (pead_live_state.json + pead_live.csv) so it resumes.

HONEST status: PEAD survives overlap + conservative cost and is momentum-INDEPENDENT,
but it is REGIME-DEPENDENT (positive in only ~5/11 years, concentrated in bull years),
so this is to gather FORWARD paper evidence — not a proven money-maker. Configure a
conservative cost. Positions can overlap across names (a per-signal paper book, not a
single-capital constraint).

No-lookahead: a trade is only booked once its entry (the day AFTER the filing) AND its
exit (``hold`` days later) are both in the past. Pure + injectable; PAPER only.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd

from tagent.config import DATA_DIR
from tagent.intraday_backtest import CostModel


@dataclass
class PeadLiveConfig:
    hold: int = 20                    # trading days held after entry
    capital: float = 10_000.0
    slippage_bps: float = 15.0        # conservative-ish (round trip ~0.51%)
    per_trade_frac: float = 0.05      # capital deployed per trade (~20 concurrent positions);
    #                                   PEAD holds many overlapping names, so each trade is sized
    #                                   to a fraction, NOT the whole book (avoids fake compounding).
    use_regime: bool = False          # arm only when the KR market is above its MA (the gate is
    regime_ma: int = 200              #   applied upstream in pead_trades; recorded here for display).

    def cost(self) -> CostModel:
        return CostModel(slippage_bps=self.slippage_bps)


class PeadLiveTrader:
    CSV_FIELDS = ["entry", "symbol", "gross", "net", "equity"]

    def __init__(self, cfg: Optional[PeadLiveConfig] = None, data_dir=None, clock=None):
        self.cfg = cfg or PeadLiveConfig()
        self.dir = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
        self.state_path = self.dir / "pead_live_state.json"
        self.csv_path = self.dir / "pead_live.csv"
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.equity = self.cfg.capital
        self.trades = 0
        self.wins = 0
        self.sum_net = 0.0
        self.booked: set = set()                       # "entry|symbol" keys (dedup)
        self.last_date: Optional[str] = None
        self.first_ts: Optional[str] = None
        self.open_positions: List[dict] = []           # transient (for display, not persisted)
        self._load()

    def book_trade(self, entry, symbol: str, gross_ret: float) -> dict:
        """Book one COMPLETED PEAD trade (entry + exit both realised), net of cost.
        Deduped by (entry, symbol) so replays/backfills never double-count."""
        ds = str(pd.Timestamp(entry).date())
        key = f"{ds}|{symbol}"
        if key in self.booked:
            return self.status()
        net = float(gross_ret) - self.cfg.cost().round_trip_frac()
        self.equity *= (1.0 + self.cfg.per_trade_frac * net)   # fractional sizing (concurrent book)
        self.trades += 1
        self.wins += int(net > 0)
        self.sum_net += net
        self.booked.add(key)
        if self.first_ts is None:
            self.first_ts = ds
        if self.last_date is None or ds > self.last_date:
            self.last_date = ds
        self._append_csv(ds, symbol, gross_ret, net)
        self._save()
        return self.status()

    def set_open_positions(self, positions: List[dict]) -> None:
        """Record currently-held (not-yet-exited) PEAD positions for the dashboard."""
        self.open_positions = list(positions)

    def status(self) -> dict:
        cap = self.cfg.capital
        wr = (self.wins / self.trades) if self.trades else 0.0
        exp = (self.sum_net / self.trades) if self.trades else 0.0
        return {
            "enabled": True, "strategy": "PEAD (buy positive-reaction earnings, hold N days)",
            "hold": self.cfg.hold, "regime_gated": self.cfg.use_regime,
            "round_trip_cost_pct": round(self.cfg.cost().round_trip_frac() * 100, 3),
            "capital": cap, "equity": round(self.equity, 2),
            "pct_change": round((self.equity / cap - 1.0) * 100, 3),
            "n_trades": self.trades, "win_rate_pct": round(wr * 100, 1),
            "expectancy_pct": round(exp * 100, 4),
            "open_positions": list(self.open_positions),
            "n_open": len(self.open_positions),
            "state": "ARMED" if self.open_positions else "OK",
            "last_date": self.last_date, "first_ts": self.first_ts,
        }

    # ------------------------------------------------------------------ #
    def _save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        state = {"cfg": asdict(self.cfg), "equity": self.equity, "trades": self.trades,
                 "wins": self.wins, "sum_net": self.sum_net, "booked": sorted(self.booked),
                 "last_date": self.last_date, "first_ts": self.first_ts}
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
        self.booked = set(s.get("booked", []))
        self.last_date = s.get("last_date")
        self.first_ts = s.get("first_ts")

    def _append_csv(self, ds, symbol, gross, net) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        new = not self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(self.CSV_FIELDS)
            w.writerow([ds, symbol, f"{gross*100:.3f}", f"{net*100:.3f}", round(self.equity, 2)])


def load_live_status(data_dir=None) -> dict:
    """Persisted PEAD status for the dashboard (no stepping)."""
    path = (Path(data_dir) if data_dir is not None else Path(DATA_DIR)) / "pead_live_state.json"
    if not path.exists():
        return {"enabled": False}
    try:
        return PeadLiveTrader(data_dir=data_dir).status()
    except Exception:
        return {"enabled": False}
