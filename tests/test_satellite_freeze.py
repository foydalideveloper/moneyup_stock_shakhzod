"""Satellite freeze — pre-registered promotion/kill logic + frozen no-deploy state."""

from tagent.satellite_freeze import (
    FROZEN, KILL, PROMOTE, SatelliteSpec, evaluate_satellite, load_satellite_state,
    write_satellite_state,
)


def _ev(**kw):
    base = dict(combined_alpha_ci=(-0.01, 0.10), ew_beats_cw_12mo=False, top2_weight_drop=0.0,
                fwd_ranking_contrib_12mo=0.0, fwd_ranking_contrib_24mo=None)
    base.update(kw)
    return evaluate_satellite(**base)


def test_frozen_when_no_trigger():
    assert _ev().status == FROZEN                              # CI spans 0, no reversal, no 24mo kill


def test_promote_a_ci_excludes_zero():
    assert _ev(combined_alpha_ci=(0.01, 0.09)).status == PROMOTE


def test_promote_b_concentration_reversal_with_positive_forward():
    # EW beats CW AND forward ranking contribution positive
    assert _ev(ew_beats_cw_12mo=True, fwd_ranking_contrib_12mo=0.02).status == PROMOTE
    # top-2 weight falls materially (>=5pp) AND forward positive
    assert _ev(top2_weight_drop=0.06, fwd_ranking_contrib_12mo=0.01).status == PROMOTE
    # reversal but forward NOT positive -> not promoted (stays frozen)
    assert _ev(ew_beats_cw_12mo=True, fwd_ranking_contrib_12mo=-0.01).status == FROZEN
    # immaterial top-2 drop -> no reversal trigger
    assert _ev(top2_weight_drop=0.02, fwd_ranking_contrib_12mo=0.01).status == FROZEN


def test_kill_when_forward_negative_over_24mo():
    assert _ev(fwd_ranking_contrib_24mo=-0.005).status == KILL


def test_promotion_takes_precedence_over_kill():
    # CI excludes zero (promote) even though 24mo forward is negative
    assert _ev(combined_alpha_ci=(0.005, 0.08), fwd_ranking_contrib_24mo=-0.01).status == PROMOTE


def test_state_roundtrip_is_frozen_and_no_deploy(tmp_path):
    ev = _ev()
    write_satellite_state(ev, SatelliteSpec(), generated="2026-06-10", data_dir=tmp_path)
    st = load_satellite_state(data_dir=tmp_path)
    assert st["frozen"] is True and st["deploy"] is False
    assert st["evaluation"]["status"] == FROZEN
    assert st["spec"]["sector_cap"] == 0.30 and st["spec"]["deploy"] is False
