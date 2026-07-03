"""Confirm whether PEAD is a REAL, independent edge — not overlap or momentum.

On the point-in-time survivorship-corrected universe, net of conservative cost:
  1) NON-OVERLAPPING events (one position per name at a time) with a properly-adjusted
     t-stat (the non-overlapping plain t is valid; Newey-West on the overlapping set is
     a cross-check),
  2) INDEPENDENCE from momentum: regress trade returns on the momentum rank at entry
     (alpha + t), and show net expectancy WITHIN each momentum tercile,
  3) by-year + conservative-slippage robustness.

Cached CSVs only (PIT OHLCV + data/kr_earnings_disclosures.csv). No network.
Usage: python scripts/validate_kr_pead.py
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tagent.kr_universe import load_members, membership_panel, universe_symbols  # noqa: E402
from tagent.stock_momentum import load_stock_panel  # noqa: E402
from tagent.report import perf_stats  # noqa: E402
from tagent.strategies.earnings_drift import (  # noqa: E402
    earnings_events, market_regime, newey_west_tstat, pead_trades, summarize,
)
from tagent.strategies.us_leadlag_daily import _wide  # noqa: E402
from tagent.xs_momentum import align_close, momentum_signal  # noqa: E402

COSTS = {"0.41%": 0.0041, "0.51%": 0.0051, "0.71%": 0.0071}


def _net_line(label, rets, cost_for_t, t):
    g = float(rets.mean())
    cells = "  ".join(f"{(g - c)*100:+7.3f}%" for c in COSTS.values())
    net_cons = g - COSTS["0.71%"]
    clears = "YES" if (net_cons > 0 and len(rets) >= 50 and t >= 1.5) else ("thin" if net_cons > 0 else "no")
    print(f"  {label:>16s} {len(rets):6d} {(rets>0).mean():5.0%}  {cells}   {clears} (t={t:+.2f})")


def _ols_alpha(y, x):
    """OLS y ~ a + b*x; return (alpha, t_alpha, beta, t_beta)."""
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    s2 = float(resid @ resid) / (n - k)
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    return beta[0], beta[0] / se[0], beta[1], beta[1] / se[1]


def main() -> int:
    argparse.ArgumentParser().parse_args()
    members = load_members()
    epath = pathlib.Path("data") / "kr_earnings_disclosures.csv"
    if not members or not epath.exists():
        print("Need PIT membership + data/kr_earnings_disclosures.csv (download_kr_earnings.py).")
        return 1
    syms = universe_symbols(members)
    panel = load_stock_panel("kr", symbols=syms, fields=["open", "close"], min_bars=60)
    idx = align_close(panel).index
    memb = membership_panel(members, idx, symbols=list(panel))
    ed = pd.read_csv(epath, dtype={"symbol": str})
    ev = earnings_events([{"type": "earnings", "symbol": str(r.symbol).zfill(6), "time": r.time}
                          for r in ed.itertuples()])
    mom_pct = momentum_signal(align_close(panel), 252, 21).rank(axis=1, pct=True)

    print(f"PIT universe {len(panel)} names; {ed['symbol'].nunique()} names w/ earnings, "
          f"{len(ed)} disclosures. Positive-reaction PEAD, NET of cost; survivorship-corrected.")

    # 1) OVERLAPPING vs NON-OVERLAPPING with honest significance
    print(f"\n=== 1) OVERLAP-ADJUSTED SIGNIFICANCE (per-event net; t-stat honest) ===")
    print(f"  {'variant':>16s} {'obs':>6s} {'win':>5s}  net@0.41%  net@0.51%  net@0.71%   clears@0.71%")
    best = None
    for hold in (10, 20):
        ov = pead_trades(panel, ev, hold=hold, conditional=True, membership=memb, non_overlapping=False)
        nov = pead_trades(panel, ev, hold=hold, conditional=True, membership=memb, non_overlapping=True)
        # overlapping: Newey-West (lag=hold) on net@0.71
        nw_t = newey_west_tstat(ov["ret"] - COSTS["0.71%"], lag=hold)
        _net_line(f"{hold}d overlap(NW)", ov["ret"], COSTS["0.71%"], nw_t)
        # non-overlapping: plain t on net@0.71 (independent obs)
        s = summarize(nov["ret"], COSTS["0.71%"])
        _net_line(f"{hold}d NON-overlap", nov["ret"], COSTS["0.71%"], s["tstat"])
        if best is None or hold == 20:
            best = (hold, nov)

    hold, nov = best
    print(f"\n  (non-overlapping = one position per name at a time; {len(nov)} independent "
          f"{hold}-day trades.)")

    # 2) INDEPENDENCE FROM MOMENTUM
    print("\n=== 2) INDEPENDENCE FROM MOMENTUM (non-overlapping 20d, net@0.71%) ===")
    m = np.array([mom_pct.at[r.entry, r.symbol] if (r.entry in mom_pct.index and r.symbol in mom_pct.columns)
                  else np.nan for r in nov.itertuples()])
    net = nov["ret"].to_numpy() - COSTS["0.71%"]
    ok = ~np.isnan(m)
    a, ta, b, tb = _ols_alpha(net[ok], m[ok])
    print(f"  regress net ~ a + b*momentum_rank:  alpha {a*100:+.3f}% (t={ta:+.2f})  "
          f"beta {b*100:+.3f}%/rank (t={tb:+.2f})  n={int(ok.sum())}")
    print(f"  -> {'ALPHA survives momentum control' if (a>0 and ta>=2.0) else 'alpha NOT significant after momentum'}.")
    mm = m[ok]
    netm = net[ok]
    terc = pd.qcut(pd.Series(mm), 3, labels=["low-mom", "mid-mom", "high-mom"], duplicates="drop")
    print("  net expectancy WITHIN each momentum tercile (independent if positive across all):")
    terc_pos = []
    for lab in terc.cat.categories:
        seg = netm[(terc == lab).to_numpy()]
        if len(seg):
            terc_pos.append(seg.mean() > 0)
            print(f"    {lab:>9s}: n={len(seg):5d}  net {seg.mean()*100:+.3f}%  win {(seg>0).mean():.0%}")

    # 3) BY YEAR (non-overlapping 20d net@0.71)
    print("\n=== 3) BY YEAR (non-overlapping 20d, net@0.71%) ===")
    ser = pd.Series(net, index=pd.DatetimeIndex(nov["entry"]))
    pos_years = n_years = 0
    for y, seg in ser.groupby(ser.index.year):
        n_years += 1
        pos_years += int(seg.mean() > 0)
        tag = " <-shock" if y in (2018, 2020, 2022) else ""
        print(f"    {y}: n={len(seg):4d}  net/trade {seg.mean()*100:+.3f}%  win {(seg>0).mean():.0%}{tag}")

    # 4) MARKET-REGIME FILTER (does it rescue PEAD as it did momentum?)
    print("\n=== 4) MARKET-REGIME FILTER (only ARM when KR market > 200d MA) ===")
    regime = market_regime(panel, ma_window=200, membership=memb)
    nov_reg = pead_trades(panel, ev, hold=hold, conditional=True, membership=memb,
                          non_overlapping=True, regime=regime)
    print(f"  trades: {len(nov)} (no filter) -> {len(nov_reg)} (regime-filtered, "
          f"{100*len(nov_reg)/max(1,len(nov)):.0f}% kept).")
    s_no = summarize(nov["ret"], COSTS["0.71%"])
    s_rg = summarize(nov_reg["ret"], COSTS["0.71%"])
    print(f"  {'':>14s} {'n':>5s} {'win':>5s} {'net@0.51%':>10s} {'net@0.71%':>10s} {'t@0.71':>7s}")
    print(f"  {'WITHOUT filter':>14s} {s_no['n']:5d} {s_no['win']:5.0%} "
          f"{(nov['ret'].mean()-COSTS['0.51%'])*100:+9.3f}% {s_no['net_exp']*100:+9.3f}% {s_no['tstat']:+7.2f}")
    print(f"  {'WITH regime':>14s} {s_rg['n']:5d} {s_rg['win']:5.0%} "
          f"{(nov_reg['ret'].mean()-COSTS['0.51%'])*100:+9.3f}% {s_rg['net_exp']*100:+9.3f}% {s_rg['tstat']:+7.2f}")

    def _by_year_pos(df):
        ser = pd.Series(df["ret"].to_numpy() - COSTS["0.71%"], index=pd.DatetimeIndex(df["entry"]))
        yrs = {int(y): float(seg.mean()) for y, seg in ser.groupby(ser.index.year)}
        return yrs, sum(1 for v in yrs.values() if v > 0), len(yrs)

    yr_no, pos_no, ny_no = _by_year_pos(nov)
    yr_rg, pos_rg, ny_rg = _by_year_pos(nov_reg)
    print(f"\n  by-year net@0.71% (WITHOUT -> WITH regime); positive years: {pos_no}/{ny_no} -> {pos_rg}/{ny_rg}")
    for y in sorted(set(yr_no) | set(yr_rg)):
        a0 = yr_no.get(y, float("nan")) * 100
        a1 = yr_rg.get(y, float("nan")) * 100
        flip = " FLIP+" if (yr_no.get(y, 0) <= 0 < yr_rg.get(y, -1)) else ""
        print(f"    {y}: {a0:+7.3f}%  ->  {a1:+7.3f}%{flip}")

    # daily paper-book equity (equal-weight across concurrently-open positions) -> CAGR/Sharpe/maxDD
    def _daily_book(df, cf):
        closes, opens = _wide(panel, "close").where(lambda x: x > 0), _wide(panel, "open").where(lambda x: x > 0)
        ix = closes.index
        from collections import defaultdict
        bucket = defaultdict(list)
        for t in df.itertuples():
            ei = ix.get_loc(t.entry)
            for j in range(ei, min(ei + hold, len(ix))):
                d = ix[j]
                if j == ei:
                    r = closes.at[d, t.symbol] / opens.at[d, t.symbol] - 1.0 - cf
                else:
                    r = closes.at[d, t.symbol] / closes.at[ix[j - 1], t.symbol] - 1.0
                if pd.notna(r):
                    bucket[d].append(r)
        return pd.Series({d: float(np.mean(v)) for d, v in bucket.items()}).sort_index()

    for label, df in (("WITHOUT filter", nov), ("WITH regime", nov_reg)):
        book = _daily_book(df, COSTS["0.71%"])
        st = perf_stats(book, 252)
        print(f"  paper book ({label:>14s}, EW concurrent, net@0.71%): CAGR {st['cagr']:+7.2%}  "
              f"Sharpe {st['sharpe']:+5.2f}  maxDD {st['max_drawdown']:+6.1%}  days {st['n']}")

    # verdict across the axes the skeptic cares about
    s20 = s_no
    overlap_ok = s20["net_exp"] > 0 and s20["n"] >= 50 and s20["tstat"] >= 1.5
    mom_indep = abs(tb) < 2.0 and all(terc_pos)            # momentum doesn't explain it
    broad = pos_years >= n_years - 1
    regime_broad = pos_rg >= ny_rg - 1
    regime_helps = pos_rg / max(1, ny_rg) > pos_no / max(1, ny_no) and s_rg["net_exp"] >= s_no["net_exp"]
    print("\n=== VERDICT ===")
    print(f"  (1) survives overlap + conservative cost? {'YES' if overlap_ok else 'no'} "
          f"(non-overlap 20d net@0.71% {s20['net_exp']*100:+.3f}%/trade, t={s20['tstat']:+.2f}, n={s20['n']}; NW t cross-checked)")
    print(f"  (2) independent of momentum?              {'YES' if mom_indep else 'no'} "
          f"(momentum beta t={tb:+.2f}~0, net positive in all momentum terciles; alpha t={ta:+.2f})")
    print(f"  (3) broad across years?                   {'YES' if broad else 'NO'} "
          f"(positive {pos_years}/{n_years} yrs unfiltered)")
    print(f"  (4) does the REGIME FILTER rescue it?     {'YES' if (regime_helps and regime_broad) else ('helps' if regime_helps else 'NO')} "
          f"(positive {pos_rg}/{ny_rg} yrs WITH regime; net@0.71% {s_rg['net_exp']*100:+.3f}%, t={s_rg['tstat']:+.2f})")
    if regime_helps and regime_broad:
        print("  -> REGIME-FILTERED PEAD becomes broad/deployable-grade (positive across most years).")
    elif regime_helps:
        print("  -> The regime filter HELPS (fewer bad years, higher net) but PEAD is STILL not")
        print("     year-broad enough to call deployable — keep it as paper-tracked forward evidence.")
    else:
        print("  -> The regime filter does NOT rescue PEAD; it stays regime-dependent / not deployable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
