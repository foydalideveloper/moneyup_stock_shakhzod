"""Live PAPER ops shakedown for the diversified KOSPI+S&P trend core (deploy_spec.md).

Recomputes the locked binary-200d / next-bar / risk-parity book from the cached daily
closes and persists the paper track + ops log. Re-run daily (or let the dashboard read
the snapshot). PAPER only — no orders.

Usage: python scripts/run_trend_core_live.py
"""

import pathlib
import sys
from typing import Optional

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from tagent.index_calibration import load_index_close  # noqa: E402
from tagent.trend_core_live import TrendCoreLiveConfig, TrendCoreLiveTrader  # noqa: E402


def advance_once(cfg: Optional[TrendCoreLiveConfig] = None) -> dict:
    """Advance the trend-core ops shakedown one step from the cached closes and persist.
    Reusable by the dashboard's daily auto-advance. Raises FileNotFoundError if a market's
    daily CSV is missing. Idempotent: re-running on unchanged data adds no audit rows."""
    cfg = cfg or TrendCoreLiveConfig()
    closes = {}
    for m in cfg.markets:
        c = load_index_close(cfg.files[m])
        if c.empty:
            raise FileNotFoundError(f"Missing data for {m} (data/{cfg.files[m]}_1d.csv)")
        closes[m] = c
    return TrendCoreLiveTrader(cfg).update(closes)


def main() -> int:
    try:
        st = advance_once()
    except FileNotFoundError as e:
        print(f"{e} — run download_multi_market.py / download_kr_aux_data.py.")
        return 1

    print("=== DIVERSIFIED TREND CORE — live paper ops shakedown ===")
    print(f"as of {st['as_of']}  ({st['n_days']} days since {st['first_ts']})")
    for m, d in st["markets"].items():
        print(f"  {m:9s}: {'IN ' if d['in_market'] else 'OUT'}  weight {d['weight']:.2f}")
    print(f"  combined exposure {st['combined_exposure']:.0%}  (deploy size cap {st['size_cap_x']:.2f}x "
          f"for a {st['account_dd_budget_pct']:.0f}% account)")
    print(f"  equity ${st['equity']:,.0f} ({st['pct_change']:+.1f}%)  vs buy&hold ${st['buyhold_equity']:,.0f} "
          f"({st['vs_buyhold_pct']:+.1f}%)")
    print(f"  drawdown {st['drawdown_pct']:+.1f}%  vs budget {st['dd_budget_pct']:.0f}% "
          f"({st['dd_budget_used_pct']:.0f}% of budget used)")
    ops = st["ops"]
    print(f"  ops: fill[{ops['modeled_fill']}] cost[{ops['modeled_cost']}] {ops['rebalance']}")
    print(f"       next futures roll ~{ops['next_futures_roll']}; est. margin ${ops['est_margin']:,.0f} "
          f"({ops['est_margin_note']})")
    print("  -> data/trend_core_live_state.json + trend_core_live.csv written. PAPER only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
