"""Decisive decompositions — is the edge ALPHA or BETA? (cached CSVs, no network)

PART 1  Momentum: how much does cross-sectional RANKING add over a regime-filtered
        equal-weight basket (cash) and a regime-filtered KOSPI-200 index (futures)?
PART 2  PEAD: does the surviving positive-reaction 20d drift survive BETA-ADJUSTMENT
        (event return minus pre-event beta x KOSPI-200 over the held window)?

Free data we already have: PIT OHLCV (data/<code>_1d.csv), point-in-time membership
(data/kr_pit_members.csv), earnings (data/kr_earnings_disclosures.csv), the KOSPI-200
index (data/kospi200_1d.csv), and SSF availability (data/kr_ssf_available.csv).
Corrected 2026 costs: 0.20% round-trip cash, 0.05% futures.

Usage: python scripts/run_decomposition.py
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

from tagent.config import DATA_DIR  # noqa: E402
from tagent.decomposition import (  # noqa: E402
    CASH_ROUND_TRIP, FUT_ROUND_TRIP, momentum_decomposition, pead_beta_adjust,
)
from tagent.kr_universe import load_members, load_ssf_available, membership_panel, universe_symbols  # noqa: E402
from tagent.stock_momentum import load_stock_panel  # noqa: E402
from tagent.strategies.earnings_drift import (  # noqa: E402
    earnings_events, market_regime, newey_west_tstat, pead_trades,
)
from tagent.strategies.us_leadlag_daily import summarize  # noqa: E402
from tagent.xs_momentum import align_close  # noqa: E402

PPY = 252


def _load_index(name="kospi200"):
    p = pathlib.Path(DATA_DIR) / f"{name}_1d.csv"
    if not p.exists():
        return None
    return pd.read_csv(p, index_col="timestamp", parse_dates=True).sort_index()["close"].astype(float)


def _row(tag, st, extra=""):
    return (f"  {tag:18s} CAGR {st['cagr']:+8.2%}  Sharpe {st['sharpe']:+5.2f}  "
            f"maxDD {st['max_drawdown']:+7.2%}  total {st['total_return']:+9.2%}{extra}")


def part1_momentum():
    members = load_members()
    idx_close = _load_index()
    if not members or idx_close is None:
        print("PART 1 skipped: need data/kr_pit_members.csv + data/kospi200_1d.csv.")
        return
    syms = universe_symbols(members)
    panel = load_stock_panel("kr", symbols=syms, min_bars=60)
    idx = align_close(panel).index
    memb = membership_panel(members, idx, symbols=list(panel))

    d = momentum_decomposition(panel, memb, idx_close, regime_ma=200,
                               cash_round_trip=CASH_ROUND_TRIP, fut_round_trip=FUT_ROUND_TRIP,
                               ppy=PPY, exclude_year=2025)
    s = d["stats"]
    print("\n############### PART 1 — MOMENTUM: filter-beta vs ranking-alpha ###############")
    print(f"PIT universe {len(panel)} names, {len(idx)} bars. 200d-MA regime, net of corrected cost "
          f"(cash {CASH_ROUND_TRIP*100:.2f}% / futures {FUT_ROUND_TRIP*100:.2f}% round trip).\n")
    print(_row("(a) filt. basket", s["filtered_basket"], "   [no ranking, cash]"))
    print(_row("(b) filt. index", s["filtered_index"], "   [200d-MA on K200, futures]"))
    print(_row("(c) momentum book", s["momentum_book"], "   [ranking + filter, cash]"))

    rab, rai = d["ranking_alpha_vs_basket"], d["ranking_alpha_vs_index"]
    print(f"\n  RANKING adds over (a) basket:  CAGR {rab['cagr']:+.2%}/yr  Sharpe {rab['sharpe']:+.2f}  "
          f"maxDD {rab['max_drawdown']:+.2%}")
    print(f"  RANKING adds over (b) index:   CAGR {rai['cagr']:+.2%}/yr  Sharpe {rai['sharpe']:+.2f}  "
          f"maxDD {rai['max_drawdown']:+.2%}")

    print("\n  === by year (total return) ===")
    yb = d["by_year"]
    years = sorted(set().union(*[set(yb[k]) for k in yb]))
    print(f"  {'year':>6s} {'(a)basket':>11s} {'(b)index':>11s} {'(c)momentum':>12s} {'rank-(a)':>10s}")
    for y in years:
        a = yb["filtered_basket"].get(y, {}).get("total_return", float("nan"))
        b = yb["filtered_index"].get(y, {}).get("total_return", float("nan"))
        c = yb["momentum_book"].get(y, {}).get("total_return", float("nan"))
        print(f"  {y:>6s} {a:+10.2%} {b:+10.2%} {c:+11.2%} {c-a:+9.2%}")

    ex, exy = d["ex_year"], d["exclude_year"]
    if ex:
        rabx = d["ranking_alpha_vs_basket_ex"]
        print(f"\n  === EXCLUDING {exy} (stress case) ===")
        print(_row("(a) filt. basket", ex["filtered_basket"]))
        print(_row("(c) momentum book", ex["momentum_book"]))
        print(f"  RANKING adds over (a) ex-{exy}:  CAGR {rabx['cagr']:+.2%}/yr  Sharpe {rabx['sharpe']:+.2f}")

    # VERDICT: the ranking must beat BOTH controls to justify a stock-picking book —
    # the EW basket (does ranking add cross-sectional value?) AND the trend-filtered
    # index (is the cheap one-contract product actually worse?).
    add_full = rab["cagr"]
    add_ex = d["ranking_alpha_vs_basket_ex"]["cagr"] if ex else add_full
    beats_basket = add_full >= 0.03 and add_ex >= 0.03
    beats_index = s["momentum_book"]["cagr"] >= s["filtered_index"]["cagr"] and \
        s["momentum_book"]["sharpe"] >= s["filtered_index"]["sharpe"]
    print("\n  === VERDICT (Part 1) ===")
    print(f"  Ranking adds {add_full:+.2%}/yr full-sample, {add_ex:+.2%}/yr ex-{exy}, over the EW basket.")
    print(f"  vs the trend-filtered KOSPI-200 index, the book is {rai['cagr']:+.2%}/yr CAGR, "
          f"{rai['sharpe']:+.2f} Sharpe.")
    if beats_basket and beats_index:
        print("  -> RANKING is a real stock-picking edge: it beats BOTH the EW basket and the")
        print("     trend-filtered index. Keep the book.")
    elif beats_basket and not beats_index:
        print("  -> SPLIT: ranking DOES add over the EW basket (>~3%/yr), but the trend-filtered")
        print("     cap-weighted KOSPI-200 index (one mini-futures contract, 0.05% cost) BEATS the")
        print("     whole stock-picking book on CAGR AND Sharpe. The deployable product is")
        print("     TREND-FILTERED INDEX EXPOSURE; the ranking's tilt doesn't beat cheap index beta.")
    else:
        print("  -> Ranking adds < ~3%/yr over the filtered basket: the deployable product is")
        print("     TREND-FILTERED INDEX EXPOSURE (one KOSPI-200 mini-futures contract, 0.05% cost),")
        print("     NOT a stock-picking edge. The filter (beta-timing) does the work, not the ranking.")
    return d


def part2_pead():
    members = load_members()
    epath = pathlib.Path(DATA_DIR) / "kr_earnings_disclosures.csv"
    idx_close = _load_index()
    if not members or not epath.exists() or idx_close is None:
        print("\nPART 2 skipped: need PIT membership + earnings + data/kospi200_1d.csv.")
        return
    syms = universe_symbols(members)
    panel = load_stock_panel("kr", symbols=syms, fields=["open", "close"], min_bars=60)
    idx = align_close(panel).index
    memb = membership_panel(members, idx, symbols=list(panel))
    ed = pd.read_csv(epath, dtype={"symbol": str})
    ev = earnings_events([{"type": "earnings", "symbol": str(r.symbol).zfill(6), "time": r.time}
                          for r in ed.itertuples()])
    hold = 20
    nov = pead_trades(panel, ev, hold=hold, conditional=True, membership=memb, non_overlapping=True)
    adj = pead_beta_adjust(nov, panel, idx_close, hold=hold, beta_window=120)

    cash, hedged = CASH_ROUND_TRIP, CASH_ROUND_TRIP + FUT_ROUND_TRIP
    raw = nov["ret"]
    abn = adj["abnormal"]
    s_raw = summarize(raw, cash)
    s_adj = summarize(abn, hedged)
    nw_raw = newey_west_tstat(raw - cash, lag=hold)
    nw_adj = newey_west_tstat(abn - hedged, lag=hold)

    print("\n################# PART 2 — PEAD: alpha vs beta (beta-adjusted) #################")
    print(f"Positive-reaction, {hold}d, non-overlapping, PIT survivorship-corrected. {len(nov)} events; "
          f"beta vs KOSPI-200 over 120 pre-event days.")
    print(f"Net of corrected cost: raw {cash*100:.2f}% (cash), beta-adjusted {hedged*100:.2f}% "
          f"(cash + futures hedge).\n")
    print(f"  {'variant':>16s} {'n':>5s} {'win':>5s} {'net_exp':>10s} {'t(plain)':>9s} {'t(NW)':>7s}")
    print(f"  {'RAW event ret':>16s} {s_raw['n']:5d} {s_raw['win']:5.0%} {s_raw['net_exp']*100:+9.3f}% "
          f"{s_raw['tstat']:+8.2f} {nw_raw:+7.2f}")
    print(f"  {'BETA-ADJUSTED':>16s} {s_adj['n']:5d} {s_adj['win']:5.0%} {s_adj['net_exp']*100:+9.3f}% "
          f"{s_adj['tstat']:+8.2f} {nw_adj:+7.2f}")
    print(f"\n  mean beta {adj['beta'].mean():.2f}; mean held-window K200 return "
          f"{adj['mkt'].mean()*100:+.2f}% (the beta exposure being removed).")

    # by year (beta-adjusted net), so we see if alpha is broad or one rally
    ser = pd.Series(abn.to_numpy() - hedged, index=pd.DatetimeIndex(adj["entry"]))
    pos = ny = 0
    print("\n  === by year (beta-adjusted net) ===")
    for y, seg in ser.groupby(ser.index.year):
        ny += 1
        pos += int(seg.mean() > 0)
        print(f"    {y}: n={len(seg):4d}  net/trade {seg.mean()*100:+.3f}%  win {(seg>0).mean():.0%}")

    print("\n  === VERDICT (Part 2) ===")
    t_raw, t_adj = s_raw["tstat"], s_adj["tstat"]
    print(f"  raw t={t_raw:+.2f} (net {s_raw['net_exp']*100:+.3f}%) -> beta-adjusted t={t_adj:+.2f} "
          f"(net {s_adj['net_exp']*100:+.3f}%); positive {pos}/{ny} yrs.")
    if s_adj["net_exp"] <= 0 or t_adj < 1.0:
        print("  -> COLLAPSES under beta-adjustment: the 'drift' was market beta. KILL it.")
    elif t_adj >= 3.0:
        print("  -> SURVIVES INTACT (t>=3): genuine, hedgeable ALPHA. Deployable (futures-hedged).")
    else:
        print(f"  -> SURVIVES BUT WEAKENED (t~{t_adj:.1f}, below the t>3 bar): real but marginal alpha.")
        print("     PAPER-ONLY — keep forward-tracking; not size-up deployable.")
    return adj


def ssf_report():
    members = load_members()
    ssf = load_ssf_available()
    if not members or not ssf:
        return
    syms = universe_symbols(members)
    n = sum(1 for s in syms if ssf.get(s, False))
    watch = ["000660", "005930", "042700", "009150"]
    wmark = ", ".join(f"{s}:{'Y' if ssf.get(s) else 'N'}" for s in watch)
    print(f"\n  SSF availability: {n}/{len(syms)} universe names tradable via a single-stock future "
          f"({100*n/len(syms):.0f}%).  watchlist [{wmark}]")
    print("  (KOSPI-200 membership proxy; the filtered-index leg is a listed futures contract.)")


def main() -> int:
    part1_momentum()
    part2_pead()
    ssf_report()
    print("\n(No-lookahead: regime lagged 1 bar; event betas use only pre-event windows. "
          "pykrx omits the final delisting-day crash — absolute returns mildly optimistic.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
