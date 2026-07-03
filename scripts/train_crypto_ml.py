"""Train the crypto ML agent — a per-coin LightGBM on historical Binance klines.

Features = technical (returns/RSI/MACD/momentum/volatility/volume) PLUS order-book
microstructure where a recording overlaps; triple-barrier labels on the kline
horizon; purged/embargoed walk-forward CV (same as the US model). Saves
models/crypto_<SYM>.pkl, which the dashboard's crypto-ml agent loads live.

Usage:
    python scripts/train_crypto_ml.py                       # default coins, 1h bars, 2y
    python scripts/train_crypto_ml.py --symbols BTCUSDT ETHUSDT --interval 1h --years 2
    python scripts/train_crypto_ml.py --tp 0.04 --sl 0.02 --horizon 10
"""

import argparse
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd  # noqa: E402

from tagent.config import DATA_DIR  # noqa: E402
from tagent.crypto_ml import fetch_klines, save_crypto_model, train_crypto_model  # noqa: E402


def _micro(symbol):
    """Optional microstructure recording for the coin (if one was captured)."""
    path = os.path.join(DATA_DIR, f"microstructure_{symbol.upper()}.csv")
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--years", type=float, default=2.0)
    ap.add_argument("--tp", type=float, default=0.04)
    ap.add_argument("--sl", type=float, default=0.02)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--n-splits", type=int, default=4)
    args = ap.parse_args()

    print(f"\nTraining crypto ML — {args.interval} bars, ~{args.years}y, "
          f"triple-barrier tp{args.tp}/sl{args.sl}/h{args.horizon}\n")
    for sym in args.symbols:
        try:
            klines = fetch_klines(sym, interval=args.interval, years=args.years)
        except Exception as e:
            print(f"  {sym}: kline fetch failed ({str(e)[:50]})")
            continue
        micro = _micro(sym)
        try:
            model, feats, metrics = train_crypto_model(
                klines, micro_df=micro, tp_pct=args.tp, sl_pct=args.sl,
                horizon=args.horizon, n_splits=args.n_splits)
        except Exception as e:
            print(f"  {sym}: train failed ({str(e)[:60]})")
            continue
        path = save_crypto_model(model, feats, metrics, sym)
        micro_n = sum(c in feats for c in ("depth_imbalance", "absorption_ratio", "spoof_ratio"))
        print(f"  {sym:10s} {len(klines):5d} bars  CV acc {metrics['cv_accuracy_mean']:.3f} "
              f"AUC {metrics['cv_auc_mean']:.3f}  pos {metrics['positive_rate']:.2f}  "
              f"{len(feats)} feats ({micro_n} micro)  -> {os.path.basename(path)}")

    print("\nDone. The dashboard's crypto-ml agent will load these per-coin models live.")
    print("(Honest note: crypto ML on klines is weak/regime-dependent like the US model —")
    print(" the scorecard forward-tests it net of costs alongside order-book + TA.)")
