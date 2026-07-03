"""Robust broad OOS study: ALL features + ensemble + purged CV + feature selection.

Technicals + 공매도 + 수급 + investor_detail + short_balance, merged leak-free.
Per name: PURGED + EMBARGOED walk-forward, a CALIBRATED ENSEMBLE (LightGBM +
XGBoost + CatBoost), on globally-SELECTED features (low-importance dropped). Two
arms on the same rows/folds — technicals-only vs all-selected — aggregated across
the 50-name universe, vs buy & hold. Short-selling-ban windows excluded; the
window is 2022+ (short + short_balance start there).

Run download_krx_universe.py and download_krx_signals.py first.

Usage:
    python scripts/run_robust_study.py
    python scripts/run_robust_study.py --calib-cv 3 --threshold 0.55
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

from tagent.backtest import run_backtest  # noqa: E402
from tagent.config import SETTINGS  # noqa: E402
from tagent.data.flows_source import load_flows  # noqa: E402
from tagent.data.historical import load_history  # noqa: E402
from tagent.data.krx_signals import SIGNALS, load_signal  # noqa: E402
from tagent.data.short_selling import complete_short_ratio, load_short_selling  # noqa: E402
from tagent.features import make_features  # noqa: E402
from tagent.features_flows import flow_feature_columns, merge_flow_features  # noqa: E402
from tagent.features_short import (  # noqa: E402
    SHORT_FEATURE_COLS, in_short_ban, merge_short_features,
)
from tagent.features_signals import merge_signal_features, signal_feature_columns  # noqa: E402
from tagent.labels import triple_barrier_labels  # noqa: E402
from tagent.ml.models import make_model  # noqa: E402
from tagent.ml.train import select_features, walk_forward_predict  # noqa: E402

try:
    from scripts.download_krx_universe import DEFAULT_TICKERS
except Exception:
    DEFAULT_TICKERS = ["005930", "000660", "035420"]

# Feature groups (for the survival report).
INVESTOR_COLS = signal_feature_columns(SIGNALS["investor_detail"].value_cols)
SHORTBAL_COLS = signal_feature_columns(SIGNALS["short_balance"].value_cols)
GROUPS = {
    "공매도": SHORT_FEATURE_COLS,
    "수급": flow_feature_columns(),
    "investor_detail": INVESTOR_COLS,
    "short_balance": SHORTBAL_COLS,
}


def _bt(close, position):
    return run_backtest(close, position, cost_bps=SETTINGS.transaction_cost_bps,
                        slippage_bps=SETTINGS.slippage_bps).stats


def _pos_quantile(oos, q):
    """Long the top (1-q) highest-conviction bars per name (relative threshold —
    the meaningful way to act on CALIBRATED probabilities)."""
    if oos.empty:
        return oos
    return (oos >= oos.quantile(q)).astype(float)


def _group_of(col):
    for g, cols in GROUPS.items():
        if col in cols:
            return g
    return "technicals"


def build_name(symbol, tp, sl, horizon, keep_ban=False):
    """Return (X_all, y, close, base_cols) for a name, or None."""
    df = load_history(symbol)
    feats = make_features(df, dropna=False)
    try:
        feats = merge_short_features(feats, complete_short_ratio(load_short_selling(symbol), df["volume"]))
    except FileNotFoundError:
        pass
    try:
        feats = merge_flow_features(feats, load_flows(symbol))
    except FileNotFoundError:
        pass
    for sig in ("investor_detail", "short_balance"):
        try:
            feats = merge_signal_features(feats, load_signal(sig, symbol),
                                          value_cols=SIGNALS[sig].value_cols)
        except FileNotFoundError:
            pass

    labels = triple_barrier_labels(df, tp_pct=tp, sl_pct=sl, horizon=horizon)
    data = feats.copy()
    data["label"] = labels
    data = data.dropna()
    if not keep_ban:
        data = data[~in_short_ban(data.index).values]
    if len(data) < 60:
        return None
    y = data["label"].astype(int)
    X = data.drop(columns=["label"])
    base_cols = [c for c in X.columns
                 if _group_of(c) == "technicals"]
    return X, y, df["close"], base_cols


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    ap.add_argument("--tp", type=float, default=0.04)
    ap.add_argument("--sl", type=float, default=0.02)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--quantile", type=float, default=0.70,
                    help="trade bars with calibrated proba in the top (1-q) per name")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--embargo", type=int, default=2)
    ap.add_argument("--calib-cv", type=int, default=3)
    ap.add_argument("--min-frac", type=float, default=0.003, help="feature-selection cutoff")
    args = ap.parse_args()

    # 1) Build per-name panels.
    panels = {}
    for sym in args.tickers:
        try:
            r = build_name(sym, args.tp, args.sl, args.horizon)
        except FileNotFoundError:
            print(f"  {sym}: missing data — skipped")
            continue
        if r is not None:
            panels[sym] = r
    if not panels:
        sys.exit("No usable names. Run download_krx_universe.py / download_krx_signals.py.")

    # 2) GLOBAL feature selection on the pooled all-feature dataset.
    Xp = pd.concat([p[0] for p in panels.values()], ignore_index=True)
    yp = pd.concat([p[1] for p in panels.values()], ignore_index=True)
    selected, imp = select_features(Xp, yp, min_frac=args.min_frac)
    print(f"\n=== feature selection (pooled {len(Xp)} rows, {len(Xp.columns)} feats "
          f"-> {len(selected)} kept) ===")
    survived = {}
    for col in Xp.columns:
        g = _group_of(col)
        survived.setdefault(g, [0, 0])
        survived[g][1] += 1
        if col in selected:
            survived[g][0] += 1
    for g, (kept, total) in survived.items():
        keep_names = [c for c in selected if _group_of(c) == g]
        print(f"  {g:16s}: {kept}/{total} survive" +
              (f"  ({', '.join(keep_names)})" if g != "technicals" and keep_names else ""))
    print("  top 12 by gain: " + ", ".join(f"{c}={int(v)}" for c, v in imp.head(12).items()))

    def factory():
        return make_model("ensemble", calibrate="isotonic", calib_cv=args.calib_cv)

    # 3) Per-name OOS: technicals-only vs all-selected, same rows/folds.
    tech_ret, all_ret, hold_ret, tech_sh, all_sh = [], [], [], [], []
    beat_hold = {"tech": 0, "all": 0}
    all_beat_tech = 0
    print(f"\n=== per-name OOS (ensemble {make_model('ensemble').kinds}, purged "
          f"horizon={args.horizon}+embargo={args.embargo}, top-{(1-args.quantile)*100:.0f}% conviction) ===")
    for sym, (X, y, close, base_cols) in panels.items():
        sel = [c for c in selected if c in X.columns]
        try:
            oos_t = walk_forward_predict(X[base_cols], y, n_splits=args.n_splits,
                                         horizon=args.horizon, embargo=args.embargo,
                                         make_model=factory).dropna()
            if oos_t.empty:
                continue
            idx = oos_t.index
            oos_a = walk_forward_predict(X[sel], y, n_splits=args.n_splits,
                                         horizon=args.horizon, embargo=args.embargo,
                                         make_model=factory).reindex(idx)
        except Exception as e:
            print(f"  {sym}: failed ({str(e)[:60]})")
            continue
        c = close.reindex(idx)
        st = _bt(c, _pos_quantile(oos_t, args.quantile))
        sa = _bt(c, _pos_quantile(oos_a, args.quantile))
        sh = _bt(c, pd.Series(1.0, index=c.index))
        tech_ret.append(st["total_return"]); all_ret.append(sa["total_return"]); hold_ret.append(sh["total_return"])
        tech_sh.append(st["sharpe"]); all_sh.append(sa["sharpe"])
        beat_hold["tech"] += int(st["total_return"] > sh["total_return"])
        beat_hold["all"] += int(sa["total_return"] > sh["total_return"])
        all_beat_tech += int(sa["total_return"] > st["total_return"])
        print(f"  {sym}: tech {st['total_return']:+7.2%}(Sh{st['sharpe']:+.2f})  "
              f"all {sa['total_return']:+7.2%}(Sh{sa['sharpe']:+.2f})  hold {sh['total_return']:+7.2%} [{len(idx)}b]",
              flush=True)

    n = len(tech_ret)
    print(f"\n=== AGGREGATE over {n} names (2022+ window, ban-excluded) ===")
    def agg(label, rets, shs=None):
        s = f"  {label:16s}  ret mean {statistics.mean(rets):+7.2%}  median {statistics.median(rets):+7.2%}"
        if shs:
            s += f"  | Sharpe mean {statistics.mean(shs):+5.2f}"
        return s
    if n:
        print(agg("technicals", tech_ret, tech_sh))
        print(agg("ALL features", all_ret, all_sh))
        print(agg("buy & hold", hold_ret))
        print(f"\n  beats buy & hold:  technicals {beat_hold['tech']}/{n}   ALL {beat_hold['all']}/{n}")
        print(f"  ALL beats technicals: {all_beat_tech}/{n}")
