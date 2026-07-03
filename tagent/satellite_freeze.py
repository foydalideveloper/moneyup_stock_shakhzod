"""Freeze the ranking satellite as an automatic paper track (PRE-REGISTERED).

The cross-sectional 12-1 ranking sleeve is a low-conviction satellite (survives a 30%
sector cap but its spanning-alpha CI includes zero over ~10yr — see
`index_calibration` / `alpha-vs-beta-decompositions`). We FREEZE the book (no more
variants) and run it as a paper track whose promotion/kill is decided ONLY by the
pre-registered rule below — never discretionarily, and it never auto-deploys.

PROMOTE if EITHER
  (a) the combined backtest+forward spanning-alpha 95% CI EXCLUDES zero, OR
  (b) concentration REVERSES (EW basket beats the CW index trailing 12mo, OR the index
      top-2 weight falls >= ``top2_drop_material``) WHILE the forward ranking
      contribution stays positive.
KILL -> archive if the forward ranking contribution is NEGATIVE over ``kill_months``.
Otherwise FROZEN — keep paper-tracking.

Pure logic + JSON persistence; unit-tested with mock inputs (no network).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Tuple

from tagent.config import DATA_DIR

PROMOTE, KILL, FROZEN = "PROMOTE", "KILL_ARCHIVE", "FROZEN"


@dataclass(frozen=True)
class SatelliteSpec:
    """The FROZEN ranking book + LOCKED promotion/kill thresholds. No more variants."""
    # frozen book
    lookback: int = 252
    skip: int = 21
    rebalance: int = 21
    top_q: float = 0.2
    sector_cap: float = 0.30
    cost_round_trip: float = 0.0020
    # locked promotion / kill thresholds
    top2_drop_material: float = 0.05      # >=5pp fall in index top-2 weight = "material"
    kill_months: int = 24
    deploy: bool = False                  # never auto-deploys


@dataclass
class SatelliteEvaluation:
    status: str
    reasons: list = field(default_factory=list)
    inputs: dict = field(default_factory=dict)


def evaluate_satellite(combined_alpha_ci: Tuple[float, float],
                       ew_beats_cw_12mo: bool,
                       top2_weight_drop: float,
                       fwd_ranking_contrib_12mo: float,
                       fwd_ranking_contrib_24mo: Optional[float],
                       spec: Optional[SatelliteSpec] = None) -> SatelliteEvaluation:
    """Apply the pre-registered promotion/kill rule to the current evidence.

    ``combined_alpha_ci`` = (lo, hi) 95% CI of the combined backtest+forward spanning
    alpha; ``ew_beats_cw_12mo`` = EW basket out-returned the CW index over trailing 12mo;
    ``top2_weight_drop`` = fall (fraction, e.g. 0.06) in the index top-2 weight;
    ``fwd_ranking_contrib_12mo/24mo`` = forward (live paper) ranking-over-basket return
    contribution (24mo None until enough history). Promotion takes precedence over kill.
    """
    spec = spec or SatelliteSpec()
    lo, hi = combined_alpha_ci
    reasons = []

    promote_a = lo > 0
    if promote_a:
        reasons.append(f"(a) combined spanning-alpha CI [{lo:+.3%},{hi:+.3%}] excludes 0")

    concentration_reversed = bool(ew_beats_cw_12mo) or (top2_weight_drop >= spec.top2_drop_material)
    promote_b = concentration_reversed and (fwd_ranking_contrib_12mo > 0)
    if promote_b:
        reasons.append(
            f"(b) concentration reversed (EW>CW 12mo={bool(ew_beats_cw_12mo)}, "
            f"top2 drop {top2_weight_drop:+.1%}>={spec.top2_drop_material:.0%}) "
            f"AND fwd ranking contrib {fwd_ranking_contrib_12mo:+.3%}>0")

    kill = (fwd_ranking_contrib_24mo is not None) and (fwd_ranking_contrib_24mo < 0)

    if promote_a or promote_b:
        status = PROMOTE
    elif kill:
        status = KILL
        reasons.append(f"forward ranking contrib {fwd_ranking_contrib_24mo:+.3%} < 0 over "
                       f"{spec.kill_months}mo -> archive")
    else:
        status = FROZEN
        reasons.append("no promotion trigger met; forward contrib not negative over 24mo "
                       "(or 24mo history incomplete) -> stay frozen, keep paper-tracking")

    return SatelliteEvaluation(status=status, reasons=reasons, inputs={
        "combined_alpha_ci": [lo, hi], "ew_beats_cw_12mo": bool(ew_beats_cw_12mo),
        "top2_weight_drop": top2_weight_drop,
        "fwd_ranking_contrib_12mo": fwd_ranking_contrib_12mo,
        "fwd_ranking_contrib_24mo": fwd_ranking_contrib_24mo})


def satellite_state_path(data_dir=None) -> Path:
    base = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
    return base / "satellite_paper.json"


def write_satellite_state(evaluation: SatelliteEvaluation, spec: Optional[SatelliteSpec] = None,
                          generated: Optional[str] = None, data_dir=None) -> Path:
    """Persist the frozen spec + latest evaluation (decision-support; never auto-deploys)."""
    spec = spec or SatelliteSpec()
    payload = {"frozen": True, "deploy": False, "generated": generated,
               "spec": asdict(spec), "evaluation": asdict(evaluation)}
    path = satellite_state_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_satellite_state(data_dir=None) -> dict:
    path = satellite_state_path(data_dir)
    if not path.exists():
        return {"frozen": True, "deploy": False, "evaluation": None}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"frozen": True, "deploy": False, "evaluation": None}
