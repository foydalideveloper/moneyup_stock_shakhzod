"""BROAD OOS study for statistical power: technicals vs +공매도 vs +수급 vs +both.

3 names can't separate signal from noise. This runs the leak-free 4-arm A/B/C/D
across a large KOSPI/KOSDAQ universe over max history, every arm trained on the
*same* held-out bars/folds per name, then reports AGGREGATE results: mean/median
OOS return & Sharpe per arm, how many names each arm beats buy & hold, and how
many it beats technicals-only.

Short-selling-BAN windows (tagent.features_short.SHORT_BAN_WINDOWS) are EXCLUDED
by default — during bans 공매도 is a regulatory artifact (~0), not signal. Use
--keep-ban to include them.

OHLCV auto-fetches via pykrx if missing. 공매도/수급 come from data/<code>_short.csv
/ _flows.csv (run download_krx_universe.py first; needs a working KRX login).

Usage:
    python scripts/run_broad_study.py --years 0       # 0 = max history (from 2016)
"""

import argparse
import pathlib
import statistics
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from tagent.backtest import position_from_proba, run_backtest  # noqa: E402
from tagent.config import SETTINGS  # noqa: E402
from tagent.data.flows_source import load_flows  # noqa: E402
from tagent.data.historical import load_history  # noqa: E402
from tagent.data.short_selling import complete_short_ratio, load_short_selling  # noqa: E402
from tagent.features import make_features  # noqa: E402
from tagent.features_flows import flow_feature_columns, merge_flow_features  # noqa: E402
from tagent.features_short import (  # noqa: E402
    SHORT_FEATURE_COLS, in_short_ban, merge_short_features,
)
from tagent.labels import triple_barrier_labels  # noqa: E402
from tagent.ml.train import walk_forward_predict  # noqa: E402

# Try to import the universe list (so default matches the downloader).
try:
    from scripts.download_krx_universe import DEFAULT_TICKERS
except Exception:
    DEFAULT_TICKERS = ["005930", "000660", "035420"]

ARMS = ["tech", "short", "flow", "both"]


def _bt(close, position):
    return run_backtest(close, position,
                        cost_bps=SETTINGS.transaction_cost_bps,
                        slippage_bps=SETTINGS.slippage_bps).stats


def _ensure_ohlcv(symbol, interval, start, end):
    try:
        return load_history(symbol, interval)
    except FileNotFoundError:
        from tagent.data.krx_source import get_krx_history
        get_krx_history(symbol, start, end, interval=interval)
        return load_history(symbol, interval)


def study_symbol(symbol, interval, tp, sl, horizon, threshold, n_splits, gap,
                 start, end, keep_ban):
    """Returns dict(arm->stats) + 'hold' + meta, or None. Only computes arms whose
    data exists; all computed arms share identical rows/folds."""
    df = _ensure_ohlcv(symbol, interval, start, end)
    feats = make_features(df, dropna=False)

    has_short = has_flow = False
    try:
        short = complete_short_ratio(load_short_selling(symbol), df["volume"])
        feats = merge_short_features(feats, short)
        has_short = True
    except FileNotFoundError:
        pass
    try:
        feats = merge_flow_features(feats, load_flows(symbol))
        has_flow = True
    except FileNotFoundError:
        pass

    labels = triple_barrier_labels(df, tp_pct=tp, sl_pct=sl, horizon=horizon)
    data = feats.copy()
    data["label"] = labels
    data = data.dropna()
    if not keep_ban:
        data = data[~in_short_ban(data.index).values]
    if len(data) < (n_splits + 1) * 3:
        return None

    y = data["label"].astype(int)
    X = data.drop(columns=["label"])
    short_cols = [c for c in SHORT_FEATURE_COLS if c in X.columns]
    flow_cols = [c for c in flow_feature_columns() if c in X.columns]
    base_cols = [c for c in X.columns if c not in short_cols and c not in flow_cols]

    def oos(cols):
        return walk_forward_predict(X[cols], y, n_splits=n_splits, gap=gap)

    oos_a = oos(base_cols).dropna()
    if oos_a.empty:
        return None
    idx = oos_a.index
    close = df["close"].reindex(idx)

    out = {"n_oos": len(idx), "has_short": has_short, "has_flow": has_flow,
           "hold": _bt(close, pd.Series(1.0, index=close.index)),
           "tech": _bt(close, position_from_proba(oos_a, threshold))}
    if has_short:
        out["short"] = _bt(close, position_from_proba(oos(base_cols + short_cols).reindex(idx), threshold))
    if has_flow:
        out["flow"] = _bt(close, position_from_proba(oos(base_cols + flow_cols).reindex(idx), threshold))
    if has_short and has_flow:
        out["both"] = _bt(close, position_from_proba(oos(base_cols + short_cols + flow_cols).reindex(idx), threshold))
    return out


def _summary(label, vals):
    if not vals:
        return f"  {label:14s}  (none)"
    return (f"  {label:14s}  ret mean {statistics.mean([v[0] for v in vals]):+7.2%} "
            f"median {statistics.median([v[0] for v in vals]):+7.2%}  | "
            f"Sharpe mean {statistics.mean([v[1] for v in vals]):+5.2f}  n={len(vals)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    ap.add_argument("--years", type=int, default=0, help="0 = max history from --start")
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--tp", type=float, default=0.04)
    ap.add_argument("--sl", type=float, default=0.02)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--gap", type=int, default=5)
    ap.add_argument("--keep-ban", action="store_true", help="include short-ban windows")
    args = ap.parse_args()

    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.today().normalize()
    start = (end - pd.DateOffset(years=args.years)) if args.years else pd.Timestamp(args.start)

    # Per-arm collectors of (total_return, sharpe); only over names where the arm ran.
    by_arm = {a: [] for a in ARMS}
    hold = []
    beat_hold = {a: 0 for a in ARMS}
    beat_tech = {a: 0 for a in ARMS}
    n_full = 0      # names where ALL 4 arms ran (apples-to-apples for +both)
    full_arm = {a: [] for a in ARMS}

    for sym in args.tickers:
        try:
            r = study_symbol(sym, args.interval, args.tp, args.sl, args.horizon,
                             args.threshold, args.n_splits, args.gap, start, end,
                             args.keep_ban)
        except Exception as e:
            print(f"  {sym}: skipped ({str(e)[:70]})")
            continue
        if r is None:
            print(f"  {sym}: not enough data")
            continue

        h = r["hold"]["total_return"]
        hold.append((h, r["hold"]["sharpe"]))
        t = r["tech"]["total_return"]
        line = f"  {sym}: tech {t:+7.2%}(Sh{r['tech']['sharpe']:+.2f}) hold {h:+7.2%} [{r['n_oos']}b]"
        for a in ARMS:
            if a in r:
                by_arm[a].append((r[a]["total_return"], r[a]["sharpe"]))
                beat_hold[a] += int(r[a]["total_return"] > h)
                beat_tech[a] += int(r[a]["total_return"] > t)
                if a not in ("tech",):
                    line += f" {a}{r[a]['total_return']:+7.2%}"
        if all(a in r for a in ARMS):
            n_full += 1
            for a in ARMS:
                full_arm[a].append((r[a]["total_return"], r[a]["sharpe"]))
        print(line, flush=True)

    print(f"\n=== AGGREGATE ({'max history' if not args.years else str(args.years)+'y'}, "
          f"ban {'INCLUDED' if args.keep_ban else 'EXCLUDED'}, threshold {args.threshold}) ===")
    for a in ARMS:
        print(_summary(a, by_arm[a]))
    print(_summary("buy & hold", hold))
    print("\n  beat buy & hold:  " +
          "  ".join(f"{a} {beat_hold[a]}/{len(by_arm[a])}" for a in ARMS))
    print("  beat technicals:  " +
          "  ".join(f"{a} {beat_tech[a]}/{len(by_arm[a])}" for a in ARMS if a != "tech"))
    if n_full:
        print(f"\n  Apples-to-apples on the {n_full} names with BOTH 공매도+수급:")
        for a in ARMS:
            print("   " + _summary(a, full_arm[a]))
