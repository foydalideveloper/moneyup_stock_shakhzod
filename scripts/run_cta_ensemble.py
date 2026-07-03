"""Standard CTA trend ensemble vs the single-lookback binary core — ONE pre-registered
trial (cached CSVs, no network).

Multi-lookback (1/3/12mo) TSMOM, vol-targeted risk parity, 4 futures sleeves
(KOSPI200 / S&P500 / crypto / USD-KRW) vs the Track-C binary 200d core. Reports
with/without crypto and with/without USD/KRW, by-decade, and the 2011-16 Boxpi test.

Usage: python scripts/run_cta_ensemble.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from tagent import multi_market_trend as mm  # noqa: E402
from tagent.cta_ensemble import LOOKBACKS, cta_combined, ensemble_report, load_sleeves  # noqa: E402
from tagent.decomposition import PPY, _series_stats  # noqa: E402


def _row(tag, st, extra=""):
    return (f"  {tag:26s} CAGR {st['cagr']:+8.2%}  Sharpe {st['sharpe']:+5.2f}  "
            f"maxDD {st['max_drawdown']:+7.2%}  n {st['n']:>5d}{extra}")


def _sub(net, a, b):
    import pandas as pd
    net = net.dropna()
    return _series_stats(net.loc[(net.index >= pd.Timestamp(a)) & (net.index <= pd.Timestamp(b))], PPY)


def main() -> int:
    cta = load_sleeves()
    have = [m for m in ("KOSPI200", "S&P500", "crypto", "USDKRW") if m in cta.columns
            and cta[m].notna().any()]
    binsl = mm.load_sleeves()
    print("############ CTA TREND ENSEMBLE — one pre-registered trial (no fitting) ############")
    print(f"Signal: blended TSMOM lookbacks {LOOKBACKS} (1/3/12mo), SAME everywhere; vol-target "
          f"sleeves -> inverse-vol risk parity -> 10% portfolio vol target; monthly. Sleeves: {have}.\n")

    print("=== per-sleeve CTA (own full history) ===")
    for m in have:
        print(_row(m, _series_stats(cta[m].dropna(), PPY)))

    # head-to-head on the SAME 2 markets (KOSPI+S&P), long sample: ensemble vs binary core
    ks = ["KOSPI200", "S&P500"]
    ens_ks = ensemble_report(cta, ks)
    bin_ks = mm.combined_return(binsl, ks)
    print(f"\n=== KOSPI+S&P, long sample {ens_ks['start']}..{ens_ks['end']} ===")
    print(_row("binary core (200d)", _series_stats(bin_ks, PPY)))
    print(_row("CTA ensemble (1/3/12)", ens_ks["combined"]))
    print("  ensemble by decade: " + "  ".join(
        f"{k}:{v['sharpe']:+.2f}" for k, v in sorted(ens_ks["by_decade"].items())))

    # add USD/KRW (2003+) then crypto (2017+) — does each extra futures sleeve help?
    variants = {}
    for label, mk in (("KOSPI+S&P+FX", ["KOSPI200", "S&P500", "USDKRW"]),
                      ("KOSPI+S&P+crypto", ["KOSPI200", "S&P500", "crypto"]),
                      ("ALL 4 sleeves", ["KOSPI200", "S&P500", "crypto", "USDKRW"])):
        mk = [m for m in mk if m in have]
        if len(mk) >= 2:
            variants[label] = ensemble_report(cta, mk)
    print("\n=== ENSEMBLE market-set robustness (no single sleeve should carry it) ===")
    for label, rep in variants.items():
        print(_row(label, rep["combined"], f"   {rep['start']}..{rep['end']}"))

    # same-window honesty: full 4-sleeve window, with vs without crypto, with vs without FX
    if "ALL 4 sleeves" in variants:
        s, e = variants["ALL 4 sleeves"]["start"], variants["ALL 4 sleeves"]["end"]
        print(f"\n=== same {s}..{e} window — drop one sleeve type ===")
        for label, mk in (("all 4", ["KOSPI200", "S&P500", "crypto", "USDKRW"]),
                          ("no crypto", ["KOSPI200", "S&P500", "USDKRW"]),
                          ("no FX", ["KOSPI200", "S&P500", "crypto"]),
                          ("KOSPI+S&P only", ["KOSPI200", "S&P500"])):
            net = cta_combined(cta, [m for m in mk if m in have])
            print(_row(label, _sub(net, s, e)))

    # 2011-16 Boxpi: ensemble (with FX, pre-crypto) vs binary core
    print("\n=== BOXPI 2011-16 (binary core vs CTA ensemble) ===")
    box_bin = _sub(bin_ks, "2011-01-01", "2016-12-31")
    ens_fx = cta_combined(cta, [m for m in ["KOSPI200", "S&P500", "USDKRW"] if m in have])
    box_ens = _sub(ens_fx, "2011-01-01", "2016-12-31")
    print(_row("binary core", box_bin))
    print(_row("CTA ensemble (+FX)", box_ens))

    # VERDICT
    base_sharpe = _series_stats(bin_ks, PPY)["sharpe"]
    ens_sharpe = ens_ks["combined"]["sharpe"]
    print("\n=== VERDICT (one trial) ===")
    sharpe_better = ens_sharpe > base_sharpe + 0.02
    boxpi_better = box_ens["sharpe"] > box_bin["sharpe"]
    print(f"  KOSPI+S&P Sharpe: binary {base_sharpe:+.2f} -> CTA ensemble {ens_sharpe:+.2f} "
          f"({'better' if sharpe_better else 'not better'}).")
    print(f"  Boxpi 2011-16 Sharpe: binary {box_bin['sharpe']:+.2f} -> ensemble {box_ens['sharpe']:+.2f} "
          f"({'smoother' if boxpi_better else 'not smoother'}).")
    if sharpe_better and boxpi_better:
        print("  -> The standard CTA ensemble BEATS the binary core on Sharpe AND smooths Boxpi further")
        print("     — the multi-lookback/multi-sleeve managed-futures book is the next deployable step.")
    elif sharpe_better or boxpi_better:
        print("  -> MIXED: the ensemble improves one axis but not both — a marginal, not decisive, upgrade.")
    else:
        print("  -> The ensemble does NOT beat the binary core here — added complexity without payoff;")
        print("     keep the simpler binary 2-market core. (Honest: more knobs != more edge.)")
    print("\n(One pre-registered trial: same TSMOM lookbacks/params on every market, nothing fitted. "
          "No-lookahead: signals/vol/scalar all lagged. Crypto/FX shown with- and without- to check carry.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
