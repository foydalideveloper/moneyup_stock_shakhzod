"""Track C — multi-market trend extension (cached CSVs, no network).

Same locked rule (binary 200d-MA, next-bar) on KOSPI-200 / S&P 500 / BTC / ETH; combine
via inverse-vol risk parity (monthly rebalance on trailing vol). Reports per-market +
combined (with AND without crypto), by-decade, diversification benefit, and the 2011-16
Boxpi smoothing test. Variants logged: monthly vs quarterly rebalance.

Usage: python scripts/run_multi_market_trend.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from tagent.decomposition import PPY, _series_stats  # noqa: E402
from tagent.multi_market_trend import (  # noqa: E402
    MARKETS, WEIGHT_CAP, load_sleeves, portfolio_report, subperiod_stats,
)


def _row(tag, st, extra=""):
    return (f"  {tag:22s} CAGR {st['cagr']:+8.2%}  Sharpe {st['sharpe']:+5.2f}  "
            f"maxDD {st['max_drawdown']:+7.2%}  n {st['n']:>5d}{extra}")


def main() -> int:
    sleeves = load_sleeves()
    have = [m for m in MARKETS if m in sleeves.columns]
    if "KOSPI200" not in have or "S&P500" not in have:
        print("Need data/kospi200_long_1d.csv + data/gspc_1d.csv (run yfinance fetch).")
        return 1
    print("############ TRACK C — MULTI-MARKET TREND (same locked rule, no per-market tuning) ############")
    print(f"Markets available: {have}. Costs: index 0.05%RT+12bps roll; crypto 0.1%RT+10%/yr funding. "
          f"Risk parity: inverse trailing-vol, cap {WEIGHT_CAP:.0%}, monthly rebalance on VOL.\n")

    # 1) per-market trend, each over its OWN full history
    print("=== per-market trend sleeve (own full history) ===")
    for m in have:
        s = sleeves[m].dropna()
        print(_row(m, _series_stats(s, PPY), f"   {s.index.min().date()}..{s.index.max().date()}"))

    # 2) combined WITHOUT crypto (KOSPI+S&P), long sample
    nocrypto = ["KOSPI200", "S&P500"]
    rep_long = portfolio_report(sleeves, nocrypto, rebalance="ME")
    print(f"\n=== COMBINED (no crypto: {nocrypto}) — long sample {rep_long['start']}..{rep_long['end']} ===")
    for m in nocrypto:
        print(_row(f"  {m}", rep_long["per_market"][m]))
    print(_row("COMBINED", rep_long["combined"],
               f"   best-single Sharpe {rep_long['best_single_sharpe']:+.2f}"))
    print("  by decade (combined): " + "  ".join(
        f"{k}:{v['sharpe']:+.2f}" for k, v in sorted(rep_long["combined_by_decade"].items())))

    # 3) combined WITH crypto (all 4) over the common (crypto-limited) sample + same-window no-crypto
    if "BTC" in have and "ETH" in have:
        allm = ["KOSPI200", "S&P500", "BTC", "ETH"]
        rep_all = portfolio_report(sleeves, allm, rebalance="ME")
        print(f"\n=== COMBINED (with crypto: {allm}) — common sample {rep_all['start']}..{rep_all['end']} ===")
        for m in allm:
            print(_row(f"  {m}", rep_all["per_market"][m]))
        print(_row("COMBINED +crypto", rep_all["combined"],
                   f"   best-single Sharpe {rep_all['best_single_sharpe']:+.2f}"))
        # same short window WITHOUT crypto, to isolate crypto's marginal contribution
        start, end = rep_all["start"], rep_all["end"]
        nc_short = subperiod_stats(rep_long["combined_net"], start, end)
        print(_row("COMBINED no-crypto*", nc_short, f"   *same {start}..{end} window"))
        print(f"  -> crypto's marginal effect on Sharpe over {start}+: "
              f"{rep_all['combined']['sharpe'] - nc_short['sharpe']:+.2f}")

    # 4) diversification verdict
    div = rep_long["combined"]["sharpe"] - rep_long["best_single_sharpe"]
    print(f"\n=== DIVERSIFICATION (no-crypto, long sample) ===")
    print(f"  combined Sharpe {rep_long['combined']['sharpe']:+.2f} vs best single "
          f"{rep_long['best_single_sharpe']:+.2f} (lift {div:+.2f}); "
          f"toward 0.7-1.0? {'YES' if rep_long['combined']['sharpe'] >= 0.70 else 'partial' if rep_long['combined']['sharpe'] >= 0.60 else 'no'}.")

    # 5) Boxpi 2011-16 smoothing: KOSPI alone vs KOSPI+S&P combined
    print("\n=== BOXPI 2011-16 (KOSPI alone vs KOSPI+S&P combined) ===")
    k_box = subperiod_stats(sleeves["KOSPI200"], "2011-01-01", "2016-12-31")
    c_box = subperiod_stats(rep_long["combined_net"], "2011-01-01", "2016-12-31")
    print(_row("KOSPI alone", k_box))
    print(_row("KOSPI+S&P combined", c_box))
    print(f"  -> Boxpi {'SMOOTHED' if c_box['sharpe'] > k_box['sharpe'] and c_box['total_return'] > k_box['total_return'] else 'not smoothed'}: "
          f"Sharpe {k_box['sharpe']:+.2f}->{c_box['sharpe']:+.2f}, total {k_box['total_return']:+.1%}->{c_box['total_return']:+.1%}.")

    # 6) rebalance variant (quarterly) — logged
    rep_q = portfolio_report(sleeves, nocrypto, rebalance="QE")
    print(f"\n=== variant: QUARTERLY rebalance (no crypto) ===")
    print(_row("COMBINED (quarterly)", rep_q["combined"]))

    print("\n=== VERDICT (Track C) ===")
    lift_ok = rep_long["combined"]["sharpe"] > max(0.60, rep_long["best_single_sharpe"])
    boxpi_ok = c_box["sharpe"] > k_box["sharpe"]
    if lift_ok and boxpi_ok:
        print(f"  Multi-market trend LIFTS Sharpe to {rep_long['combined']['sharpe']:+.2f} (> best single "
              f"{rep_long['best_single_sharpe']:+.2f}) and SMOOTHS the Boxpi range. The diversified")
        print("  trend book is the deployable insurance — same rule, lower single-market risk.")
    elif lift_ok:
        print(f"  Sharpe lifts to {rep_long['combined']['sharpe']:+.2f} but the Boxpi range is not fully")
        print("  smoothed — partial diversification benefit.")
    else:
        print(f"  Combined Sharpe {rep_long['combined']['sharpe']:+.2f} does NOT clearly beat the best single")
        print("  market — diversification benefit is weak on this sample.")
    print("\n(No-lookahead: trend signal & risk-parity weights both lagged; weights from trailing VOL only. "
          "Crypto sleeve is short-history/high-vol — see with/without-crypto split.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
