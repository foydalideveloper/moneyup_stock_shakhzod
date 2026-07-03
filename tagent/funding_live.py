"""Live PAPER runner for the cross-sectional funding-carry strategy.

Each 8h funding cycle: take the live funding snapshot for the LIQUID_COINS
universe, run the same selection as the deepened backtest (rank by funding,
hurdle + hysteresis + per-coin cap from :mod:`tagent.funding_portfolio`), and
hold a **delta-neutral** paper position (long spot + short perp) per selected
coin — accruing the funding, subtracting maker costs, and guarding the short-perp
leg's margin at LOW leverage (via :func:`tagent.funding_portfolio.margin_path`).

It's PAPER only (no orders), reproducible offline (funding/price snapshots are
injected in tests), and persists so it resumes across restarts. Weights are
FROZEN while the held set is unchanged (hold-don't-churn, the study's key lesson);
turnover/cost is only paid when a coin enters or leaves.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from tagent.config import DATA_DIR
from tagent.data.funding import LIQUID_COINS
from tagent.funding_portfolio import CostModel, _row_weights, margin_path
from tagent.funding_study import INTERVALS_PER_YEAR


@dataclass
class LiveConfig:
    universe: tuple = tuple(LIQUID_COINS)
    capital: float = 10_000.0          # paper capital deployed across the basket
    hurdle_bps: float = 1.0            # enter when 8h funding >= this
    band_bps: float = 4.0             # exit only when funding < hurdle - band (sustained neg)
    top_n: int = 8                    # hold at most N coins
    max_weight: float = 0.25          # per-coin cap (fraction of capital)
    scheme: str = "funding"           # funding-weighted
    leverage: float = 3.0             # LOW leverage on the short-perp leg
    maint_margin_rate: float = 0.005  # exchange maintenance margin
    margin_buffer: float = 0.03       # guard: de-risk a coin if ratio < maint + buffer
    cost: CostModel = field(default_factory=CostModel)   # maker post-only by default


def update_hold(funding: Dict[str, float], in_carry: Dict[str, bool], cfg: LiveConfig,
                exclude=()) -> Dict[str, bool]:
    """Per-coin hysteresis: enter at the hurdle, exit only on sustained-negative
    funding (below hurdle - band). ``exclude`` forces a coin out (margin guard)."""
    enter, exit_ = cfg.hurdle_bps / 1e4, (cfg.hurdle_bps - cfg.band_bps) / 1e4
    new = {}
    for c in cfg.universe:
        held = in_carry.get(c, False)
        f = funding.get(c)
        if c in exclude:
            held = False
        elif f is None or not np.isfinite(f):
            pass                                  # no data -> keep state
        elif not held and f >= enter:
            held = True
        elif held and f < exit_:
            held = False
        new[c] = held
    return new


def target_set(funding: Dict[str, float], held: Dict[str, bool], cfg: LiveConfig) -> set:
    """The coins we actually hold this cycle: the top-N held coins by funding."""
    coins = [c for c in cfg.universe if held.get(c)]
    coins.sort(key=lambda c: -(funding.get(c) if funding.get(c) is not None else -9.0))
    return set(coins[:max(1, cfg.top_n)])


def carry_weights(funding: Dict[str, float], pos_set: set, cfg: LiveConfig) -> Dict[str, float]:
    """Funding-weighted, per-coin-capped weights over the held set (reuses the
    backtest's cap-with-redistribution)."""
    coins = list(cfg.universe)
    sig = np.array([max(funding.get(c, 0.0) or 0.0, 1e-9) if c in pos_set else 0.0
                    for c in coins], float)
    mask = np.array([c in pos_set for c in coins], bool)
    w = _row_weights(sig, mask, cfg.scheme, cfg.top_n, cfg.max_weight)
    return {c: float(w[i]) for i, c in enumerate(coins) if w[i] > 1e-12}


class FundingCarryTrader:
    """Delta-neutral funding-carry paper book over the universe (persistent)."""

    CSV_FIELDS = ["ts", "cycle", "equity", "accrued", "costs", "n_held",
                  "ann_yield_pct", "min_margin_ratio", "held"]

    def __init__(self, cfg: Optional[LiveConfig] = None, data_dir=None, clock=None):
        self.cfg = cfg or LiveConfig()
        self.dir = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
        self.state_path = self.dir / "funding_live_state.json"
        self.csv_path = self.dir / "funding_live.csv"
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.weights: Dict[str, float] = {}      # coin -> fraction of capital
        self.in_carry: Dict[str, bool] = {}
        self.entry_perp: Dict[str, float] = {}
        self.last_perp: Dict[str, float] = {}
        self.last_funding: Dict[str, float] = {}
        self.accrued = 0.0
        self.costs = 0.0
        self.last_ft: Optional[int] = None
        self.cycle = 0
        self.first_ts: Optional[str] = None
        self.guard_events: list = []
        self._load()

    # ------------------------------------------------------------------ #
    def step(self, funding: Dict[str, float], prices: Dict[str, float],
             funding_time: int, ts=None) -> dict:
        """Run one 8h cycle. ``funding`` = {coin: 8h rate}, ``prices`` = {coin:
        perp/mark price}. Deduped by ``funding_time`` (no double accrual)."""
        ft = int(funding_time)
        if self.last_ft is not None and ft <= self.last_ft:
            return self.status()                  # stale/same interval -> nothing
        ts = ts or self._clock().replace(microsecond=0)
        if self.first_ts is None:
            self.first_ts = ts.isoformat()
        cap = self.cfg.capital
        self.last_funding = {c: funding[c] for c in self.cfg.universe
                             if funding.get(c) is not None}
        for c, p in prices.items():
            if p is not None and np.isfinite(p):
                self.last_perp[c] = float(p)

        # 1) accrue funding on positions held INTO this cycle (short receives +funding)
        for c, w in self.weights.items():
            fr = funding.get(c)
            if fr is not None and np.isfinite(fr):
                self.accrued += fr * w * cap

        # 2) margin guard: de-risk any held short whose perp ran against it
        exclude = set()
        for c, w in self.weights.items():
            if w <= 0:
                continue
            ep, pp = self.entry_perp.get(c), self.last_perp.get(c)
            if ep and pp:
                ratio = 1.0 / self.cfg.leverage - (pp / ep - 1.0)
                if ratio < self.cfg.maint_margin_rate + self.cfg.margin_buffer:
                    exclude.add(c)
                    self.guard_events.append(
                        {"ts": ts.isoformat(), "coin": c, "ratio": round(ratio, 4)})

        # 3) selection (hysteresis) -> target position set; FREEZE weights if unchanged
        self.in_carry = update_hold(funding, self.in_carry, self.cfg, exclude=exclude)
        new_set = target_set(funding, self.in_carry, self.cfg) - exclude
        prev_set = {c for c, w in self.weights.items() if w > 0}
        if new_set == prev_set and new_set:
            new_weights = dict(self.weights)      # held set unchanged -> no rebalance
        else:
            new_weights = carry_weights(funding, new_set, self.cfg)

        # 4) cost on turnover (only on real entries/exits)
        coins = set(self.weights) | set(new_weights)
        turnover = sum(abs(new_weights.get(c, 0.0) - self.weights.get(c, 0.0)) for c in coins)
        self.costs += turnover * cap * self.cfg.cost.frac()

        # 5) entry prices: open new legs at current perp, drop closed ones
        for c in list(self.entry_perp):
            if new_weights.get(c, 0.0) <= 0:
                self.entry_perp.pop(c, None)
        for c, w in new_weights.items():
            if w > 0 and c not in self.entry_perp and self.last_perp.get(c):
                self.entry_perp[c] = self.last_perp[c]

        self.weights = new_weights
        self.last_ft = ft
        self.cycle += 1
        self._save()
        self._append_csv(ts)
        return self.status()

    # ------------------------------------------------------------------ #
    def _margin(self) -> dict:
        """Worst-case margin ratio + headroom across held shorts (low leverage)."""
        base = 1.0 / self.cfg.leverage
        ratios = []
        for c, w in self.weights.items():
            if w <= 0:
                continue
            ep, pp = self.entry_perp.get(c), self.last_perp.get(c)
            r = base - (pp / ep - 1.0) if (ep and pp) else base
            ratios.append(r)
        min_ratio = min(ratios) if ratios else base
        headroom = min_ratio - self.cfg.maint_margin_rate
        return {"min_margin_ratio": round(min_ratio, 4),
                "headroom_pct": round(headroom * 100.0, 2),       # further adverse move tolerated
                "maint_margin_rate": self.cfg.maint_margin_rate,
                "leverage": self.cfg.leverage}

    def status(self) -> dict:
        cap = self.cfg.capital
        net = self.accrued - self.costs
        equity = cap + net
        held = [{"coin": c, "weight": round(w, 4), "notional": round(w * cap, 2),
                 "funding_bps": round((self.last_funding.get(c) or 0.0) * 1e4, 3)}
                for c, w in sorted(self.weights.items(), key=lambda kv: -kv[1]) if w > 0]
        ann = (net / cap * (INTERVALS_PER_YEAR / self.cycle) * 100.0) if self.cycle else 0.0
        return {
            "enabled": True, "cycle": self.cycle, "capital": cap,
            "equity": round(equity, 2), "accrued": round(self.accrued, 2),
            "costs": round(self.costs, 2), "net": round(net, 2),
            "pct_change": round(net / cap * 100.0, 3),
            "ann_yield_pct": round(ann, 2), "n_held": len(held), "held": held,
            "margin": self._margin(), "guard_events": self.guard_events[-10:],
            "first_ts": self.first_ts,
        }

    # ------------------------------------------------------------------ #
    def _save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        state = {"weights": self.weights, "in_carry": self.in_carry,
                 "entry_perp": self.entry_perp, "last_perp": self.last_perp,
                 "last_funding": self.last_funding, "accrued": self.accrued,
                 "costs": self.costs, "last_ft": self.last_ft, "cycle": self.cycle,
                 "first_ts": self.first_ts, "guard_events": self.guard_events,
                 "capital": self.cfg.capital}
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
        self.weights = {k: float(v) for k, v in (s.get("weights") or {}).items()}
        self.in_carry = {k: bool(v) for k, v in (s.get("in_carry") or {}).items()}
        self.entry_perp = {k: float(v) for k, v in (s.get("entry_perp") or {}).items()}
        self.last_perp = {k: float(v) for k, v in (s.get("last_perp") or {}).items()}
        self.last_funding = {k: float(v) for k, v in (s.get("last_funding") or {}).items()}
        self.accrued = float(s.get("accrued", 0.0))
        self.costs = float(s.get("costs", 0.0))
        self.last_ft = s.get("last_ft")
        self.cycle = int(s.get("cycle", 0))
        self.first_ts = s.get("first_ts")
        self.guard_events = s.get("guard_events", []) or []

    def _append_csv(self, ts) -> None:
        st = self.status()
        self.dir.mkdir(parents=True, exist_ok=True)
        new = not self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(self.CSV_FIELDS)
            w.writerow([ts.isoformat(), st["cycle"], st["equity"], st["accrued"],
                        st["costs"], st["n_held"], st["ann_yield_pct"],
                        st["margin"]["min_margin_ratio"],
                        "|".join(f"{h['coin']}:{h['weight']}" for h in st["held"])])


def load_live_status(data_dir=None) -> dict:
    """Read the persisted live state for the dashboard (no stepping). Returns
    ``{"enabled": False}`` if the runner hasn't produced state yet."""
    path = (Path(data_dir) if data_dir is not None else Path(DATA_DIR)) / "funding_live_state.json"
    if not path.exists():
        return {"enabled": False, "held": []}
    try:
        return FundingCarryTrader(data_dir=data_dir).status()
    except Exception:
        return {"enabled": False, "held": []}
