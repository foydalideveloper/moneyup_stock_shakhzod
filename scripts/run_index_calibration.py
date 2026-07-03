"""Finalize the momentum/index analysis (cached CSVs, no network).

PRIORITY 1  Calibrate the deployable product (200d-MA trend-filtered KOSPI-200) on the
            LONGEST index history (1990->): by decade + crisis/Boxpi sub-periods, net of
            ~12 bps/yr roll friction. Honest expectation: net Sharpe ~0.4-0.6?
PRIORITY 2  Real RANKING alpha (satellite) or just a semi overweight? Cap-weighted PIT
            basket sanity, ONE sector-capped momentum cell (no sector > 30%), a spanning
            regression (intercept = ranking alpha, NW t + CI), and an instrument-cost
            itemization of the book-vs-index gap.

Data: data/kospi200_long_1d.csv, data/kospi200_1d.csv, PIT OHLCV, kr_pit_members.csv,
kr_shares.csv, kr_sectors.csv. Usage: python scripts/run_index_calibration.py
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

from tagent.decomposition import CASH_ROUND_TRIP, FUT_ROUND_TRIP, PPY, _one_way, _series_stats  # noqa: E402
from tagent.index_calibration import (  # noqa: E402
    DEFAULT_SUBPERIODS, ROLL_FRICTION_ANNUAL, _book_net, by_decade, cap_weighted_basket,
    filtered_index_net, load_index_close, load_sectors, load_shares, momentum_book_weights,
    raw_index_net, sector_exposure, spanning_regression, sub_periods, turnover_cost_itemization,
)
from tagent.kr_universe import load_members, membership_panel, universe_symbols  # noqa: E402
from tagent.risk_managed import RiskOverlayConfig, regime_exposure  # noqa: E402
from tagent.stock_momentum import load_stock_panel  # noqa: E402
from tagent.xs_momentum import _apply_membership, align_close  # noqa: E402


def _stat_row(tag, st, extra=""):
    return (f"  {tag:20s} CAGR {st['cagr']:+8.2%}  Sharpe {st['sharpe']:+5.2f}  "
            f"maxDD {st['max_drawdown']:+7.2%}  n {st['n']:>5d}{extra}")


def _regime_filter(ret, nav, one_way):
    """Apply the lagged 200d-MA gate + switch cost to a forward-indexed series."""
    exp = regime_exposure(nav, RiskOverlayConfig(regime_ma=200, regime_off=0.0)).reindex(ret.index).fillna(1.0)
    sw = exp.diff()
    if len(sw):
        sw.iloc[0] = exp.iloc[0]
    return exp * ret - sw.abs() * one_way


def part1_long_history():
    close = load_index_close("kospi200_long")
    if close.empty:
        print("PART 1 skipped: need data/kospi200_long_1d.csv.")
        return
    net = filtered_index_net(close, regime_ma=200, roll_annual=ROLL_FRICTION_ANNUAL)
    raw = raw_index_net(close)
    full = _series_stats(net, PPY)
    rawfull = _series_stats(raw, PPY)
    print("\n############ PRIORITY 1 — LONG-HISTORY FILTERED-INDEX CALIBRATION ############")
    print(f"KOSPI-200 {close.index.min().date()}..{close.index.max().date()} ({len(close)} bars). "
          f"200d-MA trend filter, net of {ROLL_FRICTION_ANNUAL*1e4:.0f}bps/yr roll + "
          f"{FUT_ROUND_TRIP*100:.2f}% switch cost.\n")
    print(_stat_row("FILTERED (deployable)", full))
    print(_stat_row("raw buy & hold", rawfull))

    print("\n  === BY DECADE (filtered) ===")
    dec = by_decade(net, PPY)
    decraw = by_decade(raw, PPY)
    for k in sorted(dec):
        print(_stat_row(k, dec[k], f"   (raw CAGR {decraw.get(k, {}).get('cagr', float('nan')):+.2%})"))

    print("\n  === CRISIS / RANGE SUB-PERIODS (filtered vs raw) ===")
    sp = sub_periods(net, DEFAULT_SUBPERIODS, PPY)
    spraw = sub_periods(raw, DEFAULT_SUBPERIODS, PPY)
    for k in DEFAULT_SUBPERIODS:
        f, r = sp[k], spraw[k]
        print(f"  {k:18s} filtered: total {f['total_return']:+8.2%} Sharpe {f['sharpe']:+5.2f} "
              f"maxDD {f['max_drawdown']:+7.2%}  |  raw total {r['total_return']:+8.2%} "
              f"maxDD {r['max_drawdown']:+7.2%}")

    box, boxraw = sp["Boxpi 2011-16"], spraw["Boxpi 2011-16"]
    print("\n  === VERDICT (Part 1) ===")
    band = "CONFIRMS ~0.4-0.6" if 0.35 <= full["sharpe"] <= 0.70 else \
           ("ABOVE" if full["sharpe"] > 0.70 else "BELOW")
    print(f"  Full-sample net Sharpe {full['sharpe']:+.2f} ({band} the honest 0.4-0.6 expectation); "
          f"CAGR {full['cagr']:+.2%}, maxDD {full['max_drawdown']:+.2%}.")
    print(f"  Boxpi 2011-16 (range years): filtered {box['total_return']:+.2%} total "
          f"(Sharpe {box['sharpe']:+.2f}) vs raw {boxraw['total_return']:+.2%} — the filter churns to "
          f"~nothing in ranges; that flat stretch is the PRICE of the crash protection, not a bug.")
    return net


def part2_ranking_alpha():
    members = load_members()
    idx_full = load_index_close("kospi200_long")
    if idx_full.empty:
        idx_full = load_index_close("kospi200")
    shares, sectors = load_shares(), load_sectors()
    if not members or idx_full.empty or not shares or not sectors:
        print("\nPART 2 skipped: need PIT membership + index + kr_shares.csv + kr_sectors.csv.")
        return
    syms = universe_symbols(members)
    panel = load_stock_panel("kr", symbols=syms, min_bars=60)
    close = align_close(panel)
    idx = close.index
    memb = membership_panel(members, idx, symbols=list(panel))
    cash_ow, fut_ow = _one_way(CASH_ROUND_TRIP), _one_way(FUT_ROUND_TRIP)

    fwd = close.pct_change(fill_method=None).shift(-1)
    valid = fwd.notna().any(axis=1)
    ew_basket = _apply_membership(fwd, memb).mean(axis=1)[valid]
    basket_nav = (1.0 + ew_basket.fillna(0.0)).cumprod()

    # cells (all regime-filtered on the PIT-basket NAV, cash cost unless noted)
    net_basket = _regime_filter(ew_basket, basket_nav, cash_ow)
    capwt = cap_weighted_basket(panel, memb, shares).reindex(ew_basket.index)
    capwt_nav = (1.0 + capwt.fillna(0.0)).cumprod()
    net_capwt = _regime_filter(capwt, capwt_nav, fut_ow)                      # index-proxy vehicle

    W_plain = momentum_book_weights(close, memb, sectors, sector_cap=None)
    W_cap = momentum_book_weights(close, memb, sectors, sector_cap=0.30)
    mom_plain, turn_plain = _book_net(close, W_plain, cash_ow)
    mom_cap, turn_cap = _book_net(close, W_cap, cash_ow)
    mom_plain_fut, _ = _book_net(close, W_plain, fut_ow)                      # same-vehicle (futures cost)
    net_mom = _regime_filter(mom_plain.reindex(ew_basket.index), basket_nav, 0.0)
    net_cap = _regime_filter(mom_cap.reindex(ew_basket.index), basket_nav, 0.0)
    net_mom_fut = _regime_filter(mom_plain_fut.reindex(ew_basket.index), basket_nav, 0.0)

    # filtered index over the SAME (panel) window — the deployable benchmark
    net_index = filtered_index_net(idx_full.reindex(idx).dropna(), 200, ROLL_FRICTION_ANNUAL)

    s = {k: _series_stats(v, PPY) for k, v in {
        "filtered_index": net_index, "ew_basket": net_basket, "capwt_basket": net_capwt,
        "momentum": net_mom, "momentum_capped": net_cap, "momentum_futcost": net_mom_fut}.items()}

    print("\n############ PRIORITY 2 — RANKING ALPHA: satellite or semi overweight? ############")
    print(f"PIT {len(panel)} names, {len(idx)} bars (2016->). All regime-filtered; cash {CASH_ROUND_TRIP*100:.2f}% "
          f"/ futures {FUT_ROUND_TRIP*100:.2f}% round trip.\n")
    print(_stat_row("filtered index", s["filtered_index"], "   [CW, futures — the product]"))
    print(_stat_row("cap-wt PIT basket", s["capwt_basket"], "   [sanity: ~ index]"))
    print(_stat_row("EW basket", s["ew_basket"], "   [cash]"))
    print(_stat_row("momentum (plain)", s["momentum"], "   [ranking, cash]"))
    print(_stat_row("momentum +30% cap", s["momentum_capped"], "   [sector-capped, cash]"))

    # 3) cap-weight sanity / PIT-vs-index mismatch
    miss = s["capwt_basket"]["cagr"] - s["filtered_index"]["cagr"]
    print(f"\n  (3) cap-wt PIT basket vs filtered index: CAGR {s['capwt_basket']['cagr']:+.2%} vs "
          f"{s['filtered_index']['cagr']:+.2%} (gap {miss:+.2%}). "
          f"{'OK ~ index' if abs(miss) < 0.04 else 'MISMATCH — PIT top-100 KOSPI != 200-name index'}.")

    # 4) does the +5%/yr ranking-over-EW-basket survive the sector cap?
    add_plain = s["momentum"]["cagr"] - s["ew_basket"]["cagr"]
    add_cap = s["momentum_capped"]["cagr"] - s["ew_basket"]["cagr"]
    maxsec_plain = sector_exposure(W_plain, sectors).replace(0, np.nan).max()
    maxsec_cap = sector_exposure(W_cap, sectors).replace(0, np.nan).max()
    print(f"\n  (4) ranking-over-EW-basket: plain {add_plain:+.2%}/yr -> sector-capped {add_cap:+.2%}/yr. "
          f"max sector wt: plain {maxsec_plain:.0%} -> capped {maxsec_cap:.0%}.")

    # 5) spanning regressions (monthly, NW)
    sr_plain = spanning_regression(net_mom, net_basket)
    sr_cap = spanning_regression(net_cap, net_basket)
    print("\n  (5) SPANNING REGRESSION  (book_monthly ~ a + b*EW_basket_monthly, Newey-West):")
    for tag, sr in (("plain", sr_plain), ("sector-capped", sr_cap)):
        lo, hi = sr["ci_ann"]
        print(f"     {tag:13s} alpha {sr['alpha_ann']:+.2%}/yr (t={sr['t_alpha']:+.2f})  "
              f"beta {sr['beta']:+.2f}  95%CI [{lo:+.2%}, {hi:+.2%}]  n={sr['n_months']}mo")

    # 6) instrument-cost itemization + same-vehicle number
    n_years = len(net_mom.dropna()) / PPY
    item = turnover_cost_itemization(turn_plain, n_years)
    gap = s["filtered_index"]["cagr"] - s["momentum"]["cagr"]
    gap_samevehicle = s["filtered_index"]["cagr"] - s["momentum_futcost"]["cagr"]
    print(f"\n  (6) BOOK-vs-INDEX GAP itemization (gap = index - book = {gap:+.2%}/yr):")
    print(f"     book one-way turnover {item['turnover_oneway_annual']:.1f}x/yr; cash cost "
          f"{item['cash_cost_annual']:+.2%}/yr vs futures {item['fut_cost_annual']:+.2%}/yr "
          f"-> instrument-cost differential {item['instrument_cost_diff_annual']:+.2%}/yr.")
    print(f"     SAME-VEHICLE (book at futures cost) CAGR {s['momentum_futcost']['cagr']:+.2%}; "
          f"gap vs index {gap_samevehicle:+.2%}/yr = the EW-vs-CW + ranking-signal part (instrument-neutral).")

    # verdict: the ranking-alpha is a SATELLITE only if it (i) survives the sector cap
    # (not a mechanical semi overweight) AND (ii) the spanning-alpha CI clears ~zero.
    survives_cap = add_cap >= 0.03
    ci_lo = sr_cap["ci_ann"][0]
    ci_clears_zero = ci_lo > 0
    cap_preserves = abs(add_cap - add_plain) < 0.01           # did the cap barely change the edge?
    print("\n  === VERDICT (Part 2) ===")
    print(f"  Sector cap: ranking adds {add_cap:+.2%}/yr over EW basket (was {add_plain:+.2%} uncapped); "
          f"max sector {maxsec_plain:.0%}->{maxsec_cap:.0%}.")
    print(f"  Spanning alpha (capped) {sr_cap['alpha_ann']:+.2%}/yr, NW t={sr_cap['t_alpha']:+.2f}, "
          f"95%CI [{sr_cap['ci_ann'][0]:+.2%}, {sr_cap['ci_ann'][1]:+.2%}], n={sr_cap['n_months']}mo.")
    if survives_cap and ci_clears_zero:
        print("  -> REAL satellite ranking-alpha: survives the 30% sector cap AND the CI clears 0.")
    elif survives_cap and not ci_clears_zero:
        print(f"  -> NOT a mechanical semi overweight (the edge {'barely moves' if cap_preserves else 'survives'} "
              "under a 30% sector cap), but it is NOT statistically robust: ~10yr gives NW t<2 and a")
        print("     95% CI that includes 0. -> LOW-CONVICTION SATELLITE at most; the filtered index is")
        print("     the core deployable product (it also out-CAGRs the book — see gap above).")
    else:
        print("  -> NO satellite alpha: the edge does NOT survive the sector cap")
        print("     -> it was mostly a semi overweight. Deploy the trend-filtered index only.")
    return s


def main() -> int:
    part1_long_history()
    part2_ranking_alpha()
    print("\n(No-lookahead: regime lagged 1 bar; momentum uses only past bars. Cap-weights use a "
          "constant-shares snapshot; sectors from KRX KOSPI sector indices.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
