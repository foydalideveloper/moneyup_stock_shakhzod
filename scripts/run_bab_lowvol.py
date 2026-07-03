"""Beta-neutral low-vol (BAB) — one pre-registered trial (cached CSVs, no network).

Universe = SSF-liquid PIT names only (both legs). Sector-neutral, beta-neutral leverage,
monthly rebalance, all-in SSF costs. Judged at calendar-time NW t >= ~2.5-3 AND realized
beta ~0 AND survives costs AND not carried by one sector.

Usage: python scripts/run_bab_lowvol.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from collections import Counter  # noqa: E402

import pandas as pd  # noqa: E402

from tagent.bab_lowvol import (  # noqa: E402
    BORROW_ANNUAL, ROLL_ANNUAL, TX_PER_SIDE, bab_book, bab_net, calendar_time_t,
    realized_beta, rolling_beta, sector_contributions, ssf_liquid_universe,
)
from tagent.decomposition import PPY, _series_stats  # noqa: E402
from tagent.index_calibration import load_index_close, load_sectors  # noqa: E402
from tagent.kr_universe import load_members, load_ssf_available, membership_panel, universe_symbols  # noqa: E402
from tagent.stock_momentum import load_stock_panel  # noqa: E402
from tagent.xs_momentum import align_close  # noqa: E402


def main() -> int:
    members = load_members()
    ssf, sectors = load_ssf_available(), load_sectors()
    mkt = load_index_close("kospi200_long")
    if mkt.empty:
        mkt = load_index_close("kospi200")
    if not members or not ssf or not sectors or mkt.empty:
        print("Need PIT membership + kr_ssf_available.csv + kr_sectors.csv + KOSPI200.")
        return 1
    syms = universe_symbols(members)
    panel = load_stock_panel("kr", symbols=syms, min_bars=300)
    idx = align_close(panel).index
    universe = ssf_liquid_universe(syms, ssf, sectors, list(panel))

    # HEADLINE: real-size universe + sector concentration
    cc = Counter(universe.values())
    print("############ BETA-NEUTRAL LOW-VOL (BAB) — one pre-registered trial ############")
    print(f"HEADLINE — SSF-liquid universe (real tradable size, BOTH legs): {len(universe)} names.")
    print("  sector concentration: " + "  ".join(f"{s}:{n}" for s, n in cc.most_common(8)))
    print(f"  -> banks/financials + semis dominate; sector-neutral construction is MANDATORY "
          f"(within-sector beta sort, identical sector weights both legs).\n")

    memb = membership_panel(members, idx, symbols=list(panel))
    beta = rolling_beta(panel, mkt)
    W, meta = bab_book(panel, beta, universe, membership=memb)
    base = bab_net(panel, W)
    # conservative-cost variant (double the all-in costs)
    cons = bab_net(panel, W, tx_per_side=TX_PER_SIDE * 2, roll_annual=ROLL_ANNUAL * 2,
                   borrow_annual=BORROW_ANNUAL * 2)
    rb = realized_beta(base, mkt)
    ct_t, ct_n = calendar_time_t(base, lag=21)

    sb, sc = _series_stats(base, PPY), _series_stats(cons, PPY)
    print("=== beta-neutral construction check ===")
    if len(meta):
        print(f"  avg beta_L {meta['beta_L'].mean():.2f} (levered x{meta['lev_L'].mean():.2f}) | "
              f"beta_H {meta['beta_H'].mean():.2f} (de-levered x{meta['lev_H'].mean():.2f}) | "
              f"sectors/reb {meta['n_sectors'].mean():.0f}")
    print(f"  REALIZED beta of the BAB book vs KOSPI-200: {rb:+.2f}  (target ~0)")

    print("\n=== long-short spread (net of all-in SSF costs) ===")
    print(f"  base  costs (tx {TX_PER_SIDE*1e4:.0f}bp/side, roll {ROLL_ANNUAL*100:.1f}%/yr, "
          f"borrow {BORROW_ANNUAL*100:.0f}%/yr):  CAGR {sb['cagr']:+.2%}  Sharpe {sb['sharpe']:+.2f}  "
          f"maxDD {sb['max_drawdown']:+.2%}")
    print(f"  conservative (2x costs):                          CAGR {sc['cagr']:+.2%}  "
          f"Sharpe {sc['sharpe']:+.2f}  maxDD {sc['max_drawdown']:+.2%}")
    print(f"  CALENDAR-TIME Newey-West t (lag21): {ct_t:+.2f}  ({ct_n} days)")

    print("\n=== by year (net BAB return, base costs) ===")
    ser = base.dropna()
    pos = ny = 0
    for y, seg in ser.groupby(ser.index.year):
        ny += 1
        pos += int(seg.sum() > 0)
        tag = " <-2025" if y == 2025 else ""
        print(f"  {y}: total {(((1+seg).prod()-1))*100:+7.2f}%{tag}")
    ex25 = ser[ser.index.year != 2025]
    print(f"  ex-2025: Sharpe {_series_stats(ex25, PPY)['sharpe']:+.2f}, "
          f"total {((1+ex25).prod()-1)*100:+.2f}% ; positive years {pos}/{ny}")

    # sector contribution — is it carried by one sector?
    contrib = sector_contributions(panel, W, universe)
    tot = sum(v for v in contrib.values())
    print("\n=== sector contribution to cumulative spread (not carried by one?) ===")
    top = sorted(contrib.items(), key=lambda kv: -abs(kv[1]))[:6]
    for s, v in top:
        print(f"  {s:10s}: {v*100:+7.2f}%  ({(v/tot*100) if abs(tot)>1e-9 else float('nan'):+5.0f}% of total)")
    top_share = (abs(top[0][1]) / abs(tot)) if abs(tot) > 1e-9 else float("nan")

    # VERDICT
    print("\n=== VERDICT (pre-registered: NW t>=2.5-3 AND beta~0 AND survives costs AND not 1 sector) ===")
    beta_ok = abs(rb) <= 0.30
    t_ok = ct_t >= 2.5
    cost_ok = sc["cagr"] > 0
    sector_ok = (top_share < 0.5) if top_share == top_share else False
    clears = beta_ok and t_ok and cost_ok and sector_ok
    print(f"  realized beta ~0: {'YES' if beta_ok else 'NO'} ({rb:+.2f}) | NW t>=2.5: {'YES' if t_ok else 'no'} "
          f"({ct_t:+.2f}) | survives 2x costs: {'YES' if cost_ok else 'no'} ({sc['cagr']:+.2%}) | "
          f"not 1 sector: {'YES' if sector_ok else 'no'} (top {top_share*100:.0f}%)")
    print(f"  -> {'CLEARS the bar — real beta-neutral low-vol edge.' if clears else 'does NOT clear the bar (flat/negative is fine, expected).'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
