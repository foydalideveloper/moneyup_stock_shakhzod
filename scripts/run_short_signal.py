"""Run the ONE pre-registered YouTuber-공매도-signal trial on free pykrx cache. Honest verdict.

Loads the cached daily 공매도 (data/<CODE>_short.csv) + close (data/<CODE>_1d.csv) for the KR
universe, builds the LOCKED short-ratio-change cross-sectional L/S book, EXCLUDES short-ban
windows, costs at 1x and a conservative 2x, and reports calendar-time Newey-West t + by-year for
HOLD in {1,3,5}, both the OVERHANG and SQUEEZE directions. 대차잔고 (short_balance) is flagged if
absent (free cache has none) — not faked.

Usage: python scripts/run_short_signal.py
"""

import glob
import os
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
from tagent.data.short_selling import load_short_selling  # noqa: E402
from tagent.features_short import in_short_ban  # noqa: E402
from tagent.short_signal import (  # noqa: E402
    HOLD, SLIPPAGE_STRESS, T_BAR, backtest, evaluate, is_degenerate, short_ratio_wide,
    signal_change,
)
from tagent.stock_momentum import load_stock_panel  # noqa: E402
from tagent.xs_momentum import align_close  # noqa: E402


def _load():
    codes = sorted({os.path.basename(p).split("_short.csv")[0]
                    for p in glob.glob(os.path.join(str(DATA_DIR), "*_short.csv"))})
    short_by_sym, panel = {}, {}
    balance_have = 0
    for c in codes:
        try:
            sdf = load_short_selling(c, data_dir=DATA_DIR)
        except Exception:
            continue
        if "short_balance" in sdf and sdf["short_balance"].notna().any():
            balance_have += 1
        p = load_stock_panel("kr", symbols=[c], fields=["close"], min_bars=60)
        if c in p:
            short_by_sym[c] = sdf
            panel[c] = p[c]
    return short_by_sym, panel, len(codes), balance_have


def _verdict_line(tag, ev):
    flag = "CLEARS ✓" if ev["clears"] else "does NOT clear"
    return (f"  {tag}: t(2x)={ev['t_2x']:+.2f} (1x {ev['t_1x']:+.2f}) · ret2x {ev['ret_2x_pct']:+.2f}% · "
            f"Sharpe2x {ev['sharpe_2x']:+.2f} · n={ev['n']} · stable={ev['stable_by_year']} -> {flag}")


def main() -> int:
    short_by_sym, panel, n_codes, balance_have = _load()
    if len(short_by_sym) < 3:
        print(f"DATA WALL: only {len(short_by_sym)} stocks with both 공매도 + price cache "
              f"(of {n_codes} short files). Need the pykrx 공매도 pull — verdict withheld, not faked.")
        return 1

    close = align_close(panel).sort_index()
    sr = short_ratio_wide(short_by_sym, close).reindex(index=close.index, columns=close.columns)
    sig = signal_change(sr)                                   # primary: release-delayed 3-day change
    ban = in_short_ban(close.index)

    sig_first = sig.dropna(how="all").index.min()
    print(f"=== 공매도 SIGNAL TRIAL (pre-registered) · {len(short_by_sym)} stocks · "
          f"signal {('%s' % sig_first.date()) if pd.notna(sig_first) else 'n/a'}"
          f"..{close.index.max().date()} ===")
    print(f"  short_balance(대차잔고) present: {balance_have}/{n_codes} stocks -> SECONDARY "
          f"{'NOT tested (free cache ~empty; needs the gated 잔고 pull)' if balance_have < 5 else 'available'}; "
          f"this trial tests the 공매도 비중 CHANGE only.")
    print(f"  short-ban days excluded: {int(ban.sum())}/{len(ban)} "
          f"({ban.mean()*100:.0f}% of the window; signal is a regulatory artifact there)")
    nonban = ~ban.reindex(sig.index).fillna(False).to_numpy(dtype=bool)
    if is_degenerate(sig.loc[nonban]):
        print("  DATA WALL: no cross-sectional short-signal variation outside ban windows — verdict withheld.")
        return 1

    any_clear = False
    for hold in (1, 3, HOLD):
        print(f"\n[HOLD={hold}d]")
        for overhang, name in ((True, "OVERHANG (long low-short / short high-short)"),
                               (False, "SQUEEZE  (long high-short / short low-short)")):
            net1 = backtest(sig, close, hold=hold, overhang=overhang, ban_mask=ban,
                            cost_round_trip=0.0020)
            net2 = backtest(sig, close, hold=hold, overhang=overhang, ban_mask=ban,
                            cost_round_trip=0.0020 * SLIPPAGE_STRESS)
            ev = evaluate(net1, net2, t_bar=T_BAR)
            any_clear = any_clear or ev["clears"]
            print(_verdict_line(name, ev))

    print(f"\n=== VERDICT: {'an edge CLEARS the bar' if any_clear else 'NO edge clears the bar'} "
          f"(|calendar t|>= {T_BAR}, survives 2x cost, stable by-year). 대차잔고 not tested (free cache). ===")
    print("  Honest: flat/negative is the expected null; nothing faked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
