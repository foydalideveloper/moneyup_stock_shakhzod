"""Live PAPER runner for the diversified KOSPI+S&P trend core — the ops shakedown.

Runs the SAME locked rule as Track A/C (binary 200d-MA, next-bar lagged execution,
`trend_core.binary_trend_net`) on KOSPI-200 + S&P 500, combined via inverse-trailing-vol
risk parity (`multi_market_trend.risk_parity_weights`, monthly vol-rebalance) — i.e. the
exact same code path that produced the Sharpe +0.71 backtest, no re-research.

Sizing is re-derived off the DIVERSIFIED book's −22% maxDD (NOT single-market −49.7%):
tolerable −15% account DD / 22% ≈ 0.68x cap (deploy_spec.md A). The paper book tracks the
strategy itself (unsized) so its drawdown is read directly against the −22% budget; the
deploy size cap is reported for the account translation.

Persistent (data/trend_core_live_state.json snapshot + trend_core_live.csv audit log).
Logs the ops-validation fields for the 1–3 month shakedown: per-market signal on live data,
sleeve weights, combined exposure, modeled fills/costs, futures roll schedule, est. margin.
PAPER only — no orders. Pure/injectable; unit-tested with mock data, no network.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from tagent.config import DATA_DIR
from tagent.multi_market_trend import (
    INDEX_ROLL, INDEX_SWITCH, REGIME_MA, VOL_WINDOW, WEIGHT_CAP, market_sleeve,
    risk_parity_weights,
)
from tagent.trend_core import size_from_maxdd

DEFAULT_FILES = {"KOSPI200": "kospi200_long", "S&P500": "gspc"}
DEFAULT_KINDS = {"KOSPI200": "index", "S&P500": "index"}
FUTURES_MARGIN_PCT = 0.15            # ~ initial margin as a fraction of notional (display)
ROLL_MONTHS = (3, 6, 9, 12)          # index futures roll quarterly (IMM months)
SHAKEDOWN_TARGET_DAYS = 60           # ~1-3 months of clean ops -> ready for 1 mini contract


@dataclass
class TrendCoreLiveConfig:
    capital: float = 10_000.0
    markets: tuple = ("KOSPI200", "S&P500")
    files: dict = field(default_factory=lambda: dict(DEFAULT_FILES))
    kinds: dict = field(default_factory=lambda: dict(DEFAULT_KINDS))
    vol_window: int = VOL_WINDOW
    rebalance: str = "ME"            # monthly vol-rebalance
    weight_cap: float = WEIGHT_CAP
    regime_ma: int = REGIME_MA
    strategy_maxdd: float = -0.22    # diversified KOSPI+S&P maxDD (sizing basis, NOT -49.7%)
    tolerable_account_dd: float = -0.15

    def size_cap(self) -> float:
        return size_from_maxdd(self.strategy_maxdd, self.tolerable_account_dd)


def _next_roll(asof: pd.Timestamp) -> str:
    """Next quarterly index-futures roll month-end (display only)."""
    y, m = asof.year, asof.month
    for rm in ROLL_MONTHS:
        if rm >= m:
            return str((pd.Timestamp(year=y, month=rm, day=1) + pd.offsets.MonthEnd(0)).date())
    return str((pd.Timestamp(year=y + 1, month=ROLL_MONTHS[0], day=1) + pd.offsets.MonthEnd(0)).date())


class TrendCoreLiveTrader:
    CSV_FIELDS = ["date", "KOSPI200_in", "S&P500_in", "w_KOSPI200", "w_S&P500",
                  "exposure", "net", "equity", "buyhold_equity", "drawdown"]

    def __init__(self, cfg: Optional[TrendCoreLiveConfig] = None, data_dir=None, clock=None):
        self.cfg = cfg or TrendCoreLiveConfig()
        self.dir = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
        self.state_path = self.dir / "trend_core_live_state.json"
        self.csv_path = self.dir / "trend_core_live.csv"
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._status: dict = {"enabled": False}
        self._last_logged: Optional[str] = None
        self._shakedown_start: Optional[str] = None      # date forward ops-validation began
        self._load_meta()

    # ------------------------------------------------------------------ #
    def _sleeves(self, closes: Dict[str, pd.Series]) -> pd.DataFrame:
        cols = {}
        for m in self.cfg.markets:
            c = pd.Series(closes[m], dtype=float).dropna()
            cols[m] = market_sleeve(c, self.cfg.kinds.get(m, "index"))
        return pd.DataFrame(cols).sort_index()

    def _signals(self, closes: Dict[str, pd.Series]) -> Dict[str, bool]:
        """Latest in/out per market (close >= 200d MA) — the live signal, no-lookahead."""
        out = {}
        for m in self.cfg.markets:
            c = pd.Series(closes[m], dtype=float).dropna()
            sma = c.rolling(self.cfg.regime_ma, min_periods=self.cfg.regime_ma).mean()
            out[m] = bool(c.iloc[-1] >= sma.iloc[-1]) if pd.notna(sma.iloc[-1]) else True
        return out

    def update(self, closes: Dict[str, pd.Series]) -> dict:
        """Recompute the combined trend book from the cached closes (deterministic),
        settle every realised day, persist the status snapshot + append new audit rows."""
        sleeves = self._sleeves(closes)[list(self.cfg.markets)].dropna(how="any")
        if sleeves.empty:
            return self.status()
        weights = risk_parity_weights(sleeves, self.cfg.vol_window, self.cfg.rebalance,
                                      self.cfg.weight_cap)
        net = (weights * sleeves).sum(axis=1).reindex(sleeves.index).dropna()
        # buy & hold benchmark: same weights, always invested (raw forward returns)
        raw = pd.DataFrame({m: pd.Series(closes[m], dtype=float).dropna().pct_change(
            fill_method=None).shift(-1) for m in self.cfg.markets}).reindex(sleeves.index)
        bh = (weights * raw).sum(axis=1).reindex(net.index).dropna()

        eq = self.cfg.capital * (1.0 + net).cumprod()
        bh_eq = self.cfg.capital * (1.0 + bh.reindex(net.index).fillna(0.0)).cumprod()
        peak = eq.cummax()
        dd = float(eq.iloc[-1] / peak.iloc[-1] - 1.0)

        sig = self._signals(closes)
        w_last = weights.dropna().iloc[-1] if len(weights.dropna()) else pd.Series(0.0, index=self.cfg.markets)
        exposure = float(sum(float(w_last.get(m, 0.0)) * (1.0 if sig[m] else 0.0) for m in self.cfg.markets))
        asof = net.index[-1]
        size_cap = self.cfg.size_cap()
        deployed_notional = size_cap * exposure * float(eq.iloc[-1])
        shakedown = self._shakedown(net, eq, peak, asof)

        self._status = {
            "enabled": True, "strategy": "diversified trend core (KOSPI200 + S&P500, binary 200d, risk-parity)",
            "as_of": str(asof.date()), "first_ts": str(net.index[0].date()), "n_days": int(len(net)),
            "markets": {m: {"in_market": sig[m], "weight": round(float(w_last.get(m, 0.0)), 3)}
                        for m in self.cfg.markets},
            "combined_exposure": round(exposure, 3),
            "capital": self.cfg.capital, "equity": round(float(eq.iloc[-1]), 2),
            "buyhold_equity": round(float(bh_eq.iloc[-1]), 2),
            "pct_change": round((float(eq.iloc[-1]) / self.cfg.capital - 1.0) * 100, 3),
            "vs_buyhold_pct": round((float(eq.iloc[-1]) / float(bh_eq.iloc[-1]) - 1.0) * 100, 3),
            "drawdown_pct": round(dd * 100, 3),
            "dd_budget_pct": round(self.cfg.strategy_maxdd * 100, 1),
            "dd_budget_used_pct": round(dd / self.cfg.strategy_maxdd * 100, 1) if self.cfg.strategy_maxdd else None,
            "size_cap_x": round(size_cap, 3),
            "account_dd_budget_pct": round(self.cfg.tolerable_account_dd * 100, 1),
            "ops": {
                "modeled_fill": "next-bar at close (signal known at prior close)",
                "modeled_cost": f"index {INDEX_SWITCH*2*100:.2f}% round trip + {INDEX_ROLL*1e4:.0f}bps/yr roll",
                "actual_fill": None,            # filled once live (paper has no broker fills)
                "rebalance": f"monthly vol-rebalance ({self.cfg.rebalance})",
                "next_futures_roll": _next_roll(asof),
                "est_margin": round(FUTURES_MARGIN_PCT * deployed_notional, 2),
                "est_margin_note": f"~{FUTURES_MARGIN_PCT:.0%} of {size_cap:.2f}x-sized deployed notional",
            },
            "shakedown": shakedown,
        }
        self._append_new_rows(net, weights, sig, eq, bh_eq, peak)
        self._save()
        return self._status

    def _shakedown(self, net, eq, peak, asof) -> dict:
        """deploy_spec.md ops-shakedown progress + gates. Counts FORWARD trading days since
        ops-validation began and ops EXCEPTIONS (here: days the drawdown breached the −22%
        budget). Gate: ~60 clean ops-days -> 1 mini contract -> scale on operational confidence."""
        if self._shakedown_start is None:                # first run -> ops validation begins now
            self._shakedown_start = str(asof.date())
        start = pd.Timestamp(self._shakedown_start)
        fwd = net.index[net.index >= start]
        days = int(len(fwd))
        budget = self.cfg.strategy_maxdd
        dd_fwd = (eq.reindex(fwd) / peak.reindex(fwd) - 1.0) if days else pd.Series(dtype=float)
        exceptions = int((dd_fwd < budget).sum())        # drawdown breached the budget
        ops_clean = days >= SHAKEDOWN_TARGET_DAYS and exceptions == 0
        stage = ("scale on operational confidence (not P&L)" if days >= SHAKEDOWN_TARGET_DAYS * 2
                 else ("READY: deploy 1 mini contract" if ops_clean else "paper shakedown"))
        return {
            "start": self._shakedown_start, "days_validated": days,
            "target_days": SHAKEDOWN_TARGET_DAYS, "ops_exceptions": exceptions,
            "progress": f"{days}/{SHAKEDOWN_TARGET_DAYS} trading days validated, {exceptions} ops exceptions",
            "gate_ops_clean": ops_clean, "deploy_stage": stage,
            "next_gate": (f"{max(0, SHAKEDOWN_TARGET_DAYS - days)} more clean ops-days -> 1 mini "
                          "contract; then scale on operational confidence, not P&L"),
        }

    # ------------------------------------------------------------------ #
    def status(self) -> dict:
        return self._status

    def _append_new_rows(self, net, weights, sig, eq, bh_eq, peak) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        new = not self.csv_path.exists()
        rows = []
        for d in net.index:
            ds = str(pd.Timestamp(d).date())
            if self._last_logged is not None and ds <= self._last_logged:
                continue
            w = weights.reindex([d]).iloc[0] if d in weights.index else pd.Series(dtype=float)
            ddv = float(eq.loc[d] / peak.loc[d] - 1.0)
            rows.append([ds, sig.get("KOSPI200"), sig.get("S&P500"),
                         round(float(w.get("KOSPI200", float("nan"))), 4),
                         round(float(w.get("S&P500", float("nan"))), 4),
                         "", f"{float(net.loc[d])*100:.3f}", round(float(eq.loc[d]), 2),
                         round(float(bh_eq.loc[d]), 2), f"{ddv*100:.2f}"])
        if not rows:
            return
        with self.csv_path.open("a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(self.CSV_FIELDS)
            w.writerows(rows)
        self._last_logged = rows[-1][0]

    def _save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        payload = {**self._status, "_last_logged": self._last_logged,
                   "_shakedown_start": self._shakedown_start, "cfg": asdict(self.cfg)}
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.state_path)

    def _load_meta(self) -> None:
        if not self.state_path.exists():
            return
        try:
            s = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return
        self._last_logged = s.get("_last_logged")
        self._shakedown_start = s.get("_shakedown_start")
        self._status = {k: v for k, v in s.items()
                        if k not in ("_last_logged", "_shakedown_start", "cfg")} or {"enabled": False}


def load_live_status(data_dir=None) -> dict:
    """Persisted snapshot for the dashboard (no stepping). {'enabled': False} if unstarted."""
    path = (Path(data_dir) if data_dir is not None else Path(DATA_DIR)) / "trend_core_live_state.json"
    if not path.exists():
        return {"enabled": False}
    try:
        s = json.loads(path.read_text(encoding="utf-8"))
        return {k: v for k, v in s.items() if k not in ("_last_logged", "_shakedown_start", "cfg")}
    except Exception:
        return {"enabled": False}
