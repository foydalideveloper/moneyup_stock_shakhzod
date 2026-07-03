"""Unified FORWARD-OPERATIONS layer — out-of-sample auditions with pre-registered rules.

Every non-deployable candidate auto-tracks FORWARD (from its registration date) and is
judged ONLY by a pre-registered promotion / kill rule — never discretionarily. Nothing here
deploys; it's the audition that decides what graduates.

Generic rule (VRP, PEAD, lead-lag, low-vol BAB): spanning the candidate's FORWARD returns on
buy-&-hold of its market — PROMOTE when the spanning-alpha 95% CI EXCLUDES zero (a real edge
beyond beta); KILL/ARCHIVE when the forward return is NEGATIVE over ``kill_months``; else keep
TRACKING. The momentum satellite uses its own special rule (tagent.satellite_freeze).

No-lookahead: the forward window starts at ``registered_on`` and only uses returns dated on/
after it; the in-sample backtest alpha is shown as CONTEXT only, never as the promotion metric.

Pure functions + JSON persistence; unit-tested with mock data (no network).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from tagent.config import DATA_DIR
from tagent.index_calibration import spanning_regression

MIN_MONTHS = 12                  # minimum forward history before the rule can fire
KILL_MONTHS = 24                 # forward window over which a negative book is archived
TRACKING, PROMOTE, KILL = "tracking", "promote-eligible", "kill-eligible"


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    label: str
    rule: str = "spanning"       # "spanning" (generic) | "satellite" (special)
    min_months: int = MIN_MONTHS
    kill_months: int = KILL_MONTHS


CANDIDATES: List[CandidateSpec] = [
    CandidateSpec("vrp", "Options VRP (VKOSPI timing)"),
    CandidateSpec("pead", "PEAD (post-earnings drift)"),
    CandidateSpec("leadlag", "US-shock lead-lag bounce"),
    CandidateSpec("bab", "Low-vol BAB (beta-neutral)"),
    CandidateSpec("momentum-satellite", "Momentum satellite (12-1, sector-capped)", rule="satellite"),
]


def _months(n_days: int) -> float:
    return n_days / 21.0


def _spanning_block(returns: pd.Series, benchmark: pd.Series) -> Optional[dict]:
    """Annualized spanning alpha + NW t + 95% CI of ``returns`` on ``benchmark`` (monthly),
    or None if too short for the regression."""
    r = pd.Series(returns).dropna()
    if len(r) < 60 or benchmark is None:
        return None
    sr = spanning_regression(r, pd.Series(benchmark, dtype=float).reindex(r.index))
    lo, hi = sr["ci_ann"]
    return {"alpha_ann_pct": round(sr["alpha_ann"] * 100, 2), "alpha_t": round(sr["t_alpha"], 2),
            "ci_ann_pct": [round(lo * 100, 2), round(hi * 100, 2)], "n_months": sr["n_months"],
            "ci_excludes_zero": bool(lo > 0)}


def spanning_status(fwd_returns: pd.Series, benchmark: Optional[pd.Series],
                    spec: CandidateSpec) -> dict:
    """Forward status under the generic spanning rule (promote if CI excludes 0; kill if
    negative over kill_months; else tracking)."""
    r = pd.Series(fwd_returns).dropna()
    n = len(r)
    months = _months(n)
    eq = float((1.0 + r).prod()) if n else 1.0
    out = {"forward_days": n, "forward_months": round(months, 1),
           "forward_equity_pct": round((eq - 1.0) * 100, 2), "spanning": None}
    if months < spec.min_months:
        out["status"] = TRACKING
        out["note"] = f"insufficient forward history ({months:.1f}/{spec.min_months}mo)"
        return out
    blk = _spanning_block(r, benchmark)
    out["spanning"] = blk
    promote = bool(blk and blk["ci_excludes_zero"])
    kill = (months >= spec.kill_months) and (eq - 1.0 < 0.0)
    out["status"] = PROMOTE if promote else (KILL if kill else TRACKING)
    out["note"] = ("forward spanning-alpha CI excludes 0" if promote else
                   (f"forward return negative over {spec.kill_months}mo -> archive" if kill else
                    "forward alpha CI still includes 0 -> keep tracking"))
    return out


def evaluate_candidate(spec: CandidateSpec, returns: pd.Series, benchmark: Optional[pd.Series],
                       registered_on, satellite_eval: Optional[dict] = None) -> dict:
    """One candidate's forward status. ``returns`` is the candidate's full daily return
    series; the forward window is the part dated on/after ``registered_on`` (no-lookahead).
    The full-sample spanning alpha is reported as in-sample CONTEXT only."""
    r = pd.Series(returns).dropna()
    if len(r) and not isinstance(r.index, pd.DatetimeIndex):
        r.index = pd.to_datetime(r.index, errors="coerce")
        r = r[r.index.notna()]
    reg = pd.Timestamp(registered_on) if registered_on is not None else (r.index[0] if len(r) else None)
    fwd = r[r.index >= reg] if (reg is not None and len(r)) else r.iloc[:0]
    base = {"name": spec.name, "label": spec.label, "rule": spec.rule,
            "registered_on": str(reg.date()) if reg is not None else None,
            "in_sample_context": _spanning_block(r, benchmark)}
    if spec.rule == "satellite" and satellite_eval is not None:
        st = {"PROMOTE": PROMOTE, "KILL_ARCHIVE": KILL, "FROZEN": TRACKING}.get(
            satellite_eval.get("status", "FROZEN"), TRACKING)
        base.update({"status": st, "forward_days": len(fwd),
                     "forward_months": round(_months(len(fwd)), 1),
                     "reasons": satellite_eval.get("reasons", []), "spanning": None,
                     "note": "satellite special rule (CI-excludes-0 OR concentration-reversal)"})
        return base
    base.update(spanning_status(fwd, benchmark, spec))
    return base


def trigger_monitor(statuses: List[dict], satellite_eval: Optional[dict] = None) -> dict:
    """OK / TRIGGERED per regime trigger: each candidate's promote/kill condition + the
    satellite's mega-cap-reversal promotion check. Returns the fired triggers."""
    fired = []
    for s in statuses:
        if s.get("status") == PROMOTE:
            fired.append({"who": s["name"], "kind": "promote", "what": s.get("note", "")})
        elif s.get("status") == KILL:
            fired.append({"who": s["name"], "kind": "kill", "what": s.get("note", "")})
    if satellite_eval is not None:
        reasons = satellite_eval.get("reasons", [])
        reversal = any("concentration reversed" in str(x) for x in reasons)
        if reversal:
            fired.append({"who": "momentum-satellite", "kind": "promote",
                          "what": "mega-cap concentration reversal with positive forward contribution"})
    return {"any_triggered": bool(fired), "n_triggers": len(fired), "triggers": fired,
            "checked": len(statuses) + (1 if satellite_eval is not None else 0)}


# --------------------------------------------------------------------------- #
# snapshot for the dashboard
# --------------------------------------------------------------------------- #
def forward_ops_path(data_dir=None) -> Path:
    return (Path(data_dir) if data_dir is not None else Path(DATA_DIR)) / "forward_ops.json"


def write_forward_ops(payload: dict, data_dir=None) -> Path:
    path = forward_ops_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_forward_ops(data_dir=None) -> dict:
    path = forward_ops_path(data_dir)
    if not path.exists():
        return {"enabled": False, "candidates": [], "triggers": {"any_triggered": False, "triggers": []}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"enabled": False, "candidates": []}


def load_registry(data_dir=None) -> Dict[str, str]:
    """Persisted {candidate: registered_on} so the forward window is stable across runs."""
    p = (Path(data_dir) if data_dir is not None else Path(DATA_DIR)) / "forward_ops_registry.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_registry(reg: Dict[str, str], data_dir=None) -> None:
    p = (Path(data_dir) if data_dir is not None else Path(DATA_DIR)) / "forward_ops_registry.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(reg, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# daily-advance: stamp + once-per-day scheduler (hands-free dashboard operation)
# --------------------------------------------------------------------------- #
def advance_state_path(data_dir=None) -> Path:
    return (Path(data_dir) if data_dir is not None else Path(DATA_DIR)) / "forward_advance.json"


def write_advance_state(day, as_of=None, ran=None, data_dir=None) -> Path:
    """Persist the daily-advance stamp the dashboard panels show ("last advanced / next
    advance"). ``day`` is the calendar day the advance ran (the once/day guard key);
    ``next_advance`` is the following business day."""
    nxt = str((pd.Timestamp(day) + pd.offsets.BDay(1)).date())
    payload = {"last_advanced": str(day), "next_advance": nxt,
               "as_of": as_of, "ran": list(ran or [])}
    p = advance_state_path(data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


def load_advance_state(data_dir=None) -> dict:
    """Read the daily-advance stamp (no advancing). Absent -> nulls."""
    p = advance_state_path(data_dir)
    if not p.exists():
        return {"last_advanced": None, "next_advance": None, "as_of": None, "ran": []}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"last_advanced": None, "next_advance": None, "as_of": None, "ran": []}


class DailyAdvanceScheduler:
    """Advance the forward operations once per new calendar day — the daily analog of
    ``momentum_live.MonthlyRebalanceScheduler``.

    Tick it frequently (e.g. hourly) from a background thread. It throttles real attempts
    to ``min_interval_s`` apart and only fires when today hasn't been advanced yet —
    determined from BOTH an in-process guard (``_last_run_day``) and ``day_done_fn`` (which
    reads the persisted ``last_advanced`` from disk, so a fresh process / the manual scripts
    can't cause a double-advance). ``advance_fn`` is a zero-arg callable that performs one
    day's advance; everything is injected so this is unit-tested with no network.
    """

    def __init__(self, advance_fn, day_done_fn=None, clock=None, min_interval_s: float = 3600.0):
        self.advance_fn = advance_fn
        self.day_done_fn = day_done_fn
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.min_interval_s = float(min_interval_s)
        self._last_run_day: Optional[str] = None
        self._last_tick = None

    def due(self, now=None) -> bool:
        """True if today still needs an advance (in-process guard + persisted day-done)."""
        day = pd.Timestamp(now or self._clock()).strftime("%Y-%m-%d")
        if day == self._last_run_day:
            return False
        done = self.day_done_fn() if self.day_done_fn else None
        return day != done

    def tick(self, now=None) -> bool:
        """One scheduler tick. Advances iff a new day is due (and the throttle interval has
        elapsed). Returns True iff it triggered an advance."""
        now = pd.Timestamp(now or self._clock())
        if (self._last_tick is not None and self.min_interval_s > 0
                and (now - pd.Timestamp(self._last_tick)).total_seconds() < self.min_interval_s):
            return False
        self._last_tick = now
        if not self.due(now):
            return False
        self.advance_fn()
        self._last_run_day = now.strftime("%Y-%m-%d")
        return True
