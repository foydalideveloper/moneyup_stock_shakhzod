"""Three more KR daily edges — same rigor as momentum (PIT, net of conservative cost).

VOL-SPIKE bounce / FLOWS-following (수급) / EARNINGS drift (PEAD), all on the
point-in-time survivorship-corrected universe, net of conservative round-trip cost
(0.41% / 0.51% / 0.71%), with t-stats, by-year, walk-forward (flows), and an
independence check vs momentum / the US-shock signal. Honest: flat or negative is fine.

Needs cached PIT universe + OHLCV, data/leadlag/{SPY,VIX}_1d.csv (download_leadlag_history.py),
50+ *_flows.csv, and data/kr_earnings_disclosures.csv (download_kr_earnings.py).

Usage: python scripts/validate_kr_more_edges.py
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd  # noqa: E402

from tagent.data.intraday_history import load_spy_daily_returns, overnight_for_dates  # noqa: E402
from tagent.kr_universe import load_members, membership_panel, universe_symbols  # noqa: E402
from tagent.report import perf_stats  # noqa: E402
from tagent.stock_momentum import classic_config, load_stock_panel  # noqa: E402
from tagent.strategies.earnings_drift import earnings_events, pead_event_returns  # noqa: E402
from tagent.strategies.flows_following import flows_backtest, flows_signal, net_buy_panel  # noqa: E402
from tagent.strategies import vol_spike_bounce as vsb  # noqa: E402
from tagent.strategies.us_leadlag_daily import summarize  # noqa: E402
from tagent.xs_momentum import align_close, backtest  # noqa: E402
from tagent.xs_momentum_validate import by_year, walk_forward_signal  # noqa: E402

PPY = 252
COSTS = {"0.41%": 0.0041, "0.51%": 0.0051, "0.71%": 0.0071}
TARGET_N = 50
LEADLAG = pathlib.Path("data") / "leadlag"


def _sweep_line(label, ps, n_events=None):
    """Per-stock event line: n / win / net@each cost / t-stat@0.71%."""
    if ps.empty:
        print(f"  {label:>8s}  (no events)")
        return
    g = float(ps.mean())
    cells = "  ".join(f"{(g - c)*100:+7.3f}%" for c in COSTS.values())
    cons = summarize(ps, COSTS["0.71%"])
    flag = "YES" if (cons["net_exp"] > 0 and cons["n"] >= TARGET_N and cons["tstat"] >= 1.5) else \
           ("thin" if cons["net_exp"] > 0 else "no")
    ev = f"{n_events:4d}ev " if n_events is not None else ""
    print(f"  {label:>8s} {ev}{len(ps):6d}obs {(ps>0).mean():5.0%}  {cells}   {flag}(t={cons['tstat']:+.2f})")


def main() -> int:
    argparse.ArgumentParser().parse_args()
    members = load_members()
    if not members:
        print("No PIT membership — run scripts/download_kr_pit_universe.py first.")
        return 1
    syms = universe_symbols(members)
    panel = load_stock_panel("kr", symbols=syms, fields=["open", "close"], min_bars=60)
    idx = align_close(panel).index
    memb = membership_panel(members, idx, symbols=list(panel))
    print(f"PIT universe: {len(panel)} names, {idx.min().date()}->{idx.max().date()}. "
          f"NET of conservative round-trip cost {list(COSTS)}; survivorship-corrected.")

    # momentum net (for the independence correlation)
    mom_net = backtest(panel, classic_config("kr", allow_short=False), PPY, membership=memb)["net"]

    # ---------------- 1) VOL-SPIKE BOUNCE ----------------
    vix = load_spy_daily_returns(data_dir=str(LEADLAG), symbol="VIX")
    spy = load_spy_daily_returns(data_dir=str(LEADLAG), symbol="SPY")
    print("\n=== 1) VOL-SPIKE BOUNCE (buy KR open / sell close after a VIX jump) — per-stock ===")
    if vix.empty:
        print("  no data/leadlag/VIX_1d.csv — run download_leadlag_history.py")
    else:
        print(f"  {'VIXjump':>8s} {'':>5s} {'obs':>6s} {'win':>5s}  "
              + "  ".join(f"net@{k}" for k in COSTS) + "   clears@0.71%")
        for thr in (0.10, 0.15, 0.20, 0.30):
            ps = vsb.event_stock_returns(panel, vix, threshold=thr, hold=1, membership=memb)
            _sweep_line(f"+{thr*100:.0f}%", ps, n_events=len(vsb.event_dates(panel, vix, thr)))
        # independence vs the US-shock signal (overlap of event days)
        vdays = set(vsb.event_dates(panel, vix, 0.20))
        sdays = set(idx[(pd.Series([overnight_for_dates(spy, idx).get(d.date(), 0.0)
                                    for d in idx], index=idx) <= -0.02).values])
        if vdays:
            ov = len(vdays & sdays) / len(vdays)
            print(f"  independence: {ov:.0%} of VIX+20% days are also US<=-2% days "
                  f"({len(vdays)} VIX days, {len(sdays)} US-shock days).")

    # ---------------- 2) FLOWS-FOLLOWING (수급) ----------------
    print("\n=== 2) FLOWS-FOLLOWING (long strongest foreign+inst net buyers) ===")
    flow_syms = [s for s in syms if (pathlib.Path("data") / f"{s}_flows.csv").exists()]
    print(f"  flows available for {len(flow_syms)}/{len(syms)} PIT names (cached subset).")
    base = backtest(panel, classic_config("kr", allow_short=False), PPY, membership=memb)
    bs = base["basket_stats"]
    for slip in (5.0, 25.0):
        cfg = classic_config("kr", allow_short=False)
        from dataclasses import replace
        cfg = replace(cfg, slippage_bps=slip)
        r = flows_backtest(panel, symbols=flow_syms, lookback=5, hold=5, top_q=0.2,
                           membership=memb, cfg=cfg, periods_per_year=PPY)
        s = r["stats"]
        rt = (cfg.cost_bps + slip) / 1e4
        flag = "beats basket" if (s["sharpe"] > bs["sharpe"] and s["total_return"] > bs["total_return"]) else ""
        print(f"  slippage {slip:>4.0f}bp (turnover cost): Sharpe {s['sharpe']:+5.2f}  "
              f"CAGR {s['cagr']:+7.2%}  maxDD {s['max_drawdown']:+6.1%}  {flag}")
        if slip == 5.0:
            corr = r["net"].reindex(mom_net.index).corr(mom_net)
            print(f"     basket Sharpe {bs['sharpe']:+.2f}; corr(flows,momentum nets) = {corr:+.2f}")
    flows_sig = flows_signal(net_buy_panel(flow_syms)).reindex(columns=list(align_close(panel).columns))
    wf = walk_forward_signal(panel, flows_sig, top_qs=(0.1, 0.2, 0.3), train_bars=756, test_bars=252,
                             cfg=classic_config("kr", allow_short=False), periods_per_year=PPY, membership=memb)
    print(f"  walk-forward OOS: Sharpe {wf['stats']['sharpe']:+.2f}  CAGR {wf['stats']['cagr']:+.2%}  n={wf['stats']['n']}")

    # ---------------- 3) EARNINGS DRIFT (PEAD) ----------------
    print("\n=== 3) EARNINGS DRIFT / PEAD (hold after a positive earnings disclosure) — per-event ===")
    epath = pathlib.Path("data") / "kr_earnings_disclosures.csv"
    if not epath.exists():
        print("  no data/kr_earnings_disclosures.csv — run download_kr_earnings.py")
    else:
        ed = pd.read_csv(epath, dtype={"symbol": str})
        ev = earnings_events([{"type": "earnings", "symbol": str(r.symbol).zfill(6), "time": r.time}
                              for r in ed.itertuples()])
        print(f"  {ed['symbol'].nunique()} names, {len(ed)} earnings disclosures "
              f"({ed['time'].min()[:10]}->{ed['time'].max()[:10]}).")
        print(f"  {'hold':>8s} {'':>5s} {'obs':>6s} {'win':>5s}  "
              + "  ".join(f"net@{k}" for k in COSTS) + "   clears@0.71%")
        for hold in (5, 10, 20):
            ps = pead_event_returns(panel, ev, hold=hold, conditional=True, membership=memb)
            _sweep_line(f"{hold}d+", ps)
        ps_all = pead_event_returns(panel, ev, hold=10, conditional=False, membership=memb)
        _sweep_line("10d~all", ps_all)

    print("\n=== VERDICT ===")
    print("  Headline = per-stock NET on the PIT universe at CONSERVATIVE cost (0.51%/0.71%), "
          "t>=1.5, n>=50. 'thin'/'no' = does not survive realistic execution (an honest result).")
    print("(No-lookahead: VIX/US as-of pre-open; 수급 lagged; earnings entered the day AFTER filing.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
