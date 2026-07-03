"""Lock down the trend-futures core + freeze the satellite (cached CSVs, no network).

PART 1  Drawdown diagnostic: which episode made the long-history filtered core's worst
        DD, and how much the realistic trade-next-bar execution lag worsens it.
PART 2  Continuous-exposure v2 (pre-registered, ContinuousV2Spec) vs the binary filter,
        full 1990-2026: Sharpe / maxDD / by decade.
PART 3  Freeze the ranking satellite as an automatic paper track; apply the
        pre-registered promotion/kill rule to current evidence; write satellite_paper.json.

Usage: python scripts/run_trend_core.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tagent.decomposition import PPY, _one_way, _series_stats, CASH_ROUND_TRIP  # noqa: E402
from tagent.index_calibration import (  # noqa: E402
    cap_weighted_basket, filtered_index_net, load_index_close, load_sectors, load_shares,
    momentum_book_weights, spanning_regression, _book_net,
)
from tagent.kr_universe import load_members, membership_panel, universe_symbols  # noqa: E402
from tagent.risk_managed import RiskOverlayConfig, regime_exposure  # noqa: E402
from tagent.satellite_freeze import SatelliteSpec, evaluate_satellite, write_satellite_state  # noqa: E402
from tagent.stock_momentum import load_stock_panel  # noqa: E402
from tagent.trend_core import (  # noqa: E402
    ContinuousV2Spec, binary_trend_net, by_decade, continuous_trend_net, drawdown_episode,
    execution_lag_compare, size_from_maxdd,
)
from tagent.xs_momentum import _apply_membership, align_close  # noqa: E402


def _row(tag, st, extra=""):
    return (f"  {tag:22s} CAGR {st['cagr']:+8.2%}  Sharpe {st['sharpe']:+5.2f}  "
            f"maxDD {st['max_drawdown']:+7.2%}{extra}")


def part1_drawdown(close):
    print("\n################# PART 1 — DRAWDOWN DIAGNOSTIC #################")
    net = filtered_index_net(close, regime_ma=200)               # the committed product
    ep = drawdown_episode(net)
    st = _series_stats(net, PPY)
    print(f"Deployable filtered core (1990->): full maxDD {st['max_drawdown']:+.2%}, Sharpe {st['sharpe']:+.2f}.")
    print(f"  WORST DD episode: {ep['depth']:+.2%}  peak {ep['peak_date'].date()} -> trough "
          f"{ep['trough_date'].date()} ({ep['peak_to_trough_days']} bars); "
          f"recovery {ep['recovery_date'].date() if ep['recovery_date'] is not None else 'NOT within sample'}.")
    yr = ep["trough_date"].year
    tag = "1997 IMF / KRW crisis" if yr in (1997, 1998) else f"{yr} episode"
    print(f"  -> the {tag} is the binding drawdown.")

    print("\n  Execution lag — same-bar (idealised) vs next-bar (realistic):")
    cmp = execution_lag_compare(close, regime_ma=200)
    for mode in ("same_bar", "next_bar"):
        s, e = cmp[mode]["stats"], cmp[mode]["episode"]
        print(_row(f"{mode}", s, f"   worstDD {e['depth']:+.2%} ({e['trough_date'].date()})"))
    worse = cmp["next_bar"]["stats"]["max_drawdown"] - cmp["same_bar"]["stats"]["max_drawdown"]
    print(f"  -> realistic next-bar execution {'WORSENS' if worse < 0 else 'does not worsen'} the maxDD by "
          f"{worse:+.2%}. SIZING MUST SURVIVE {st['max_drawdown']:+.2%} (the realistic figure), not ~-23%.")
    print(f"  -> size rule: tolerable -15% account DD / |{st['max_drawdown']:.1%}| = "
          f"{size_from_maxdd(st['max_drawdown'], -0.15):.2f}x cap (see deploy_spec.md).")
    return net


def part2_continuous(close):
    print("\n################# PART 2 — CONTINUOUS-EXPOSURE v2 (pre-registered) #################")
    spec = ContinuousV2Spec()
    print(f"v2: clip(logistic({spec.logistic_k}*z) * ({spec.target_vol}/realized_vol), 0,1); "
          f"ma={spec.regime_ma}, std_window={spec.std_window}, vol_window={spec.vol_window}. LOCKED.\n")
    binv = binary_trend_net(close, regime_ma=200, mode="next_bar")
    v2 = continuous_trend_net(close, spec)
    sb, s2 = _series_stats(binv, PPY), _series_stats(v2, PPY)
    print(_row("binary 200d (v1)", sb))
    print(_row("continuous (v2)", s2))
    print("\n  === by decade (Sharpe | maxDD): binary -> v2 ===")
    db, d2 = by_decade(binv, PPY), by_decade(v2, PPY)
    for k in sorted(set(db) | set(d2)):
        b, c = db.get(k, {}), d2.get(k, {})
        print(f"  {k}: Sharpe {b.get('sharpe', float('nan')):+.2f} -> {c.get('sharpe', float('nan')):+.2f}   "
              f"maxDD {b.get('max_drawdown', float('nan')):+.2%} -> {c.get('max_drawdown', float('nan')):+.2%}")
    print("\n  === VERDICT (Part 2) ===")
    better_dd = s2["max_drawdown"] > sb["max_drawdown"]
    better_sh = s2["sharpe"] >= sb["sharpe"] - 0.02
    if better_dd and better_sh:
        print(f"  v2 improves/keeps Sharpe ({sb['sharpe']:+.2f}->{s2['sharpe']:+.2f}) AND cuts maxDD "
              f"({sb['max_drawdown']:+.2%}->{s2['max_drawdown']:+.2%}) — adopt v2 as the core exposure rule.")
    elif better_dd:
        print(f"  v2 cuts maxDD ({sb['max_drawdown']:+.2%}->{s2['max_drawdown']:+.2%}) at a Sharpe cost "
              f"({sb['sharpe']:+.2f}->{s2['sharpe']:+.2f}) — a risk trade-off (pre-registered, reported as-is).")
    else:
        print(f"  v2 does NOT dominate (Sharpe {sb['sharpe']:+.2f}->{s2['sharpe']:+.2f}, maxDD "
              f"{sb['max_drawdown']:+.2%}->{s2['max_drawdown']:+.2%}). Keep binary; v2 logged, not adopted.")
    return v2


def part3_freeze_satellite(idx_long):
    print("\n################# PART 3 — FREEZE THE SATELLITE (paper track only) #################")
    members, shares, sectors = load_members(), load_shares(), load_sectors()
    if not members or not shares or not sectors:
        print("  skipped (need PIT membership + kr_shares.csv + kr_sectors.csv); spec is frozen regardless.")
        return
    syms = universe_symbols(members)
    panel = load_stock_panel("kr", symbols=syms, min_bars=60)
    close = align_close(panel)
    idx = close.index
    memb = membership_panel(members, idx, symbols=list(panel))
    cash_ow = _one_way(CASH_ROUND_TRIP)

    fwd = close.pct_change(fill_method=None).shift(-1)
    valid = fwd.notna().any(axis=1)
    ew_basket = _apply_membership(fwd, memb).mean(axis=1)[valid]
    basket_nav = (1.0 + ew_basket.fillna(0.0)).cumprod()
    capwt = cap_weighted_basket(panel, memb, shares).reindex(ew_basket.index)

    def _rf(ret, nav, ow):
        exp = regime_exposure(nav, RiskOverlayConfig(regime_ma=200, regime_off=0.0)).reindex(ret.index).fillna(1.0)
        sw = exp.diff()
        if len(sw):
            sw.iloc[0] = exp.iloc[0]
        return exp * ret - sw.abs() * ow

    net_basket = _rf(ew_basket, basket_nav, cash_ow)
    W_cap = momentum_book_weights(close, memb, sectors, sector_cap=0.30)
    mom_cap, _ = _book_net(close, W_cap, cash_ow)
    net_cap = _rf(mom_cap.reindex(ew_basket.index), basket_nav, 0.0)

    sr = spanning_regression(net_cap, net_basket)                # backtest-only at freeze (no forward yet)
    ci = sr["ci_ann"]
    # concentration metric: did EW beat CW (cap-wt) basket over the trailing ~12mo?
    last = ew_basket.index[-252:]
    ew_12 = float((1 + ew_basket.reindex(last).fillna(0)).prod() - 1)
    cw_12 = float((1 + capwt.reindex(last).fillna(0)).prod() - 1)
    ew_beats_cw = ew_12 > cw_12

    ev = evaluate_satellite(combined_alpha_ci=ci, ew_beats_cw_12mo=ew_beats_cw,
                            top2_weight_drop=0.0,            # baseline at freeze (no prior to diff)
                            fwd_ranking_contrib_12mo=0.0,    # no forward paper track yet
                            fwd_ranking_contrib_24mo=None)
    path = write_satellite_state(ev, SatelliteSpec(), generated=str(idx[-1].date()))
    print(f"Frozen book: 12-1, monthly, top-quintile, 30% sector cap. Backtest spanning-alpha "
          f"{sr['alpha_ann']:+.2%}/yr, 95%CI [{ci[0]:+.2%},{ci[1]:+.2%}].")
    print(f"  trailing-12mo: EW {ew_12:+.1%} vs CW {cw_12:+.1%} -> EW beats CW: {ew_beats_cw} "
          f"(concentration {'reversed' if ew_beats_cw else 'intact'}).")
    print(f"  STATUS: {ev.status}.  " + " | ".join(ev.reasons))
    print(f"  -> {path} written (frozen, deploy=false). Promotion/kill is rule-driven only; "
          "no more variants, never auto-deploys.")


def main() -> int:
    close = load_index_close("kospi200_long")
    if close.empty:
        print("Need data/kospi200_long_1d.csv — run scripts/download_kr_aux_data.py.")
        return 1
    part1_drawdown(close)
    part2_continuous(close)
    part3_freeze_satellite(close)
    print("\n(Pre-registered: v2 constants + satellite rules LOCKED in deploy_spec.md before running; "
          "results reported as-is. No-lookahead: signals at close t earn t->t+1.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
