"""Unified forward-operations runner — register/track every candidate + triggers (cached).

Refreshes the trend-core ops shakedown, then auditions each non-deployable candidate
forward (pre-registered promotion/kill), evaluates the momentum-satellite special rule, runs
the trigger monitor, and writes data/forward_ops.json for the dashboard. PAPER/mock only.

Usage: python scripts/run_forward_ops.py
"""

import pathlib
import sys
from typing import Optional

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd  # noqa: E402

from tagent.forward_ops import (  # noqa: E402
    CANDIDATES, evaluate_candidate, load_registry, save_registry, trigger_monitor, write_forward_ops,
)
from tagent.index_calibration import load_index_close, load_sectors, load_shares  # noqa: E402
from tagent.satellite_freeze import SatelliteSpec, evaluate_satellite  # noqa: E402


def _kospi_buyhold():
    idx = load_index_close("kospi200_long")
    return idx, idx.pct_change(fill_method=None)


def _vrp_series(idx):
    from tagent.data.vkospi_source import is_degenerate, load_vkospi
    from tagent.vrp import vrp_backtest
    vk = load_vkospi()
    if is_degenerate(vk):
        return None, None
    active = idx[idx.index >= vk.index.min()]
    net = vrp_backtest(vk, active)
    bench = active.pct_change(fill_method=None).reindex(net.index)
    return net, bench


def _satellite_eval():
    """Compute the satellite's promotion inputs from the decomposition pipeline and apply the
    pre-registered rule (backtest-only at registration; forward contribution accrues live)."""
    try:
        from tagent.decomposition import CASH_ROUND_TRIP, _one_way
        from tagent.index_calibration import cap_weighted_basket, momentum_book_weights, spanning_regression, _book_net
        from tagent.kr_universe import load_members, membership_panel, universe_symbols
        from tagent.risk_managed import RiskOverlayConfig, regime_exposure
        from tagent.stock_momentum import load_stock_panel
        from tagent.xs_momentum import _apply_membership, align_close
        members = load_members()
        shares, sectors = load_shares(), load_sectors()
        if not members or not shares or not sectors:
            return None
        syms = universe_symbols(members)
        panel = load_stock_panel("kr", symbols=syms, min_bars=60)
        close = align_close(panel)
        idx = close.index
        memb = membership_panel(members, idx, symbols=list(panel))
        cash_ow = _one_way(CASH_ROUND_TRIP)
        fwd = close.pct_change(fill_method=None).shift(-1)
        valid = fwd.notna().any(axis=1)
        ew = _apply_membership(fwd, memb).mean(axis=1)[valid]
        nav = (1.0 + ew.fillna(0.0)).cumprod()
        capwt = cap_weighted_basket(panel, memb, shares).reindex(ew.index)

        def rf(ret, n, ow):
            ex = regime_exposure(n, RiskOverlayConfig(regime_ma=200, regime_off=0.0)).reindex(ret.index).fillna(1.0)
            sw = ex.diff()
            if len(sw):
                sw.iloc[0] = ex.iloc[0]
            return ex * ret - sw.abs() * ow
        net_basket = rf(ew, nav, cash_ow)
        W = momentum_book_weights(close, memb, sectors, sector_cap=0.30)
        mom, _ = _book_net(close, W, cash_ow)
        net_cap = rf(mom.reindex(ew.index), nav, 0.0)
        sr = spanning_regression(net_cap, net_basket)
        last = ew.index[-252:]
        ew12 = float((1 + ew.reindex(last).fillna(0)).prod() - 1)
        cw12 = float((1 + capwt.reindex(last).fillna(0)).prod() - 1)
        ev = evaluate_satellite(combined_alpha_ci=sr["ci_ann"], ew_beats_cw_12mo=ew12 > cw12,
                                top2_weight_drop=0.0, fwd_ranking_contrib_12mo=0.0,
                                fwd_ranking_contrib_24mo=None, spec=SatelliteSpec())
        from dataclasses import asdict
        return asdict(ev)
    except Exception as e:
        print(f"  (satellite eval skipped: {str(e)[:60]})")
        return None


def advance_once() -> Optional[dict]:
    """Register/refresh every candidate's forward audition, re-evaluate the pre-registered
    promote/kill rules + trigger monitor, and write data/forward_ops.json. Returns the
    payload (None if KOSPI200 data is missing). Registration dates are stable (no-lookahead)
    and re-running on unchanged data is a no-op. Reusable by the dashboard's daily advance."""
    idx, _ = _kospi_buyhold()
    if idx.empty:
        return None
    asof = str(idx.index[-1].date())
    reg = load_registry()
    sat_eval = _satellite_eval()

    # candidate return series + benchmark (None where the series accrues via its own live tracker)
    vrp_net, vrp_bench = _vrp_series(idx)
    series = {"vrp": (vrp_net, vrp_bench)}        # others register-only; forward accrues live

    cards = []
    for spec in CANDIDATES:
        reg.setdefault(spec.name, asof)            # register the forward audition (stable date)
        rets, bench = series.get(spec.name, (None, None))
        if rets is None and spec.rule != "satellite":
            # no precomputed series here -> register, await forward data from its own tracker
            cards.append({"name": spec.name, "label": spec.label, "rule": spec.rule,
                          "registered_on": reg[spec.name], "status": "tracking",
                          "forward_days": 0, "in_sample_context": None,
                          "note": "registered; forward returns accrue via the candidate's own daily tracker"})
            continue
        cards.append(evaluate_candidate(spec, rets if rets is not None else pd.Series(dtype=float),
                                        bench, reg[spec.name], satellite_eval=sat_eval))
    save_registry(reg)

    trig = trigger_monitor(cards, sat_eval)
    payload = {"enabled": True, "as_of": asof, "label": "out-of-sample audition — not deployed",
               "candidates": cards, "triggers": trig}
    write_forward_ops(payload)
    return payload


def main() -> int:
    payload = advance_once()
    if payload is None:
        print("Need data/kospi200_long_1d.csv.")
        return 1
    asof, cards, trig = payload["as_of"], payload["candidates"], payload["triggers"]

    print(f"=== FORWARD CANDIDATES (out-of-sample audition — not deployed) · as of {asof} ===")
    for c in cards:
        ctx = c.get("in_sample_context")
        ctx_s = (f"in-sample α {ctx['alpha_ann_pct']:+.1f}%/yr t{ctx['alpha_t']:+.1f}" if ctx else "—")
        print(f"  {c['label']:34s} [{c['status']:16s}] fwd {c.get('forward_days',0)}d · {ctx_s} · {c.get('note','')[:46]}")
    print(f"\n=== TRIGGER MONITOR === {'TRIGGERED' if trig['any_triggered'] else 'OK'} "
          f"({trig['n_triggers']} of {trig['checked']} checks fired)")
    for t in trig["triggers"]:
        print(f"  TRIGGERED [{t['kind']}] {t['who']}: {t['what']}")
    print("\n  -> data/forward_ops.json written. PAPER/mock only — nothing deploys.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
