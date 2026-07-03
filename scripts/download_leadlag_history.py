"""Download LONG daily history (2001+) for the US-lead-lag study via yfinance (free).

KR large-caps with deep history (Yahoo ``.KS`` tickers) + the US overnight signals
(SPY broad, QQQ Nasdaq-100, SOXX semis). Saved into data/leadlag/<code>_1d.csv in the
standard timestamp+OHLCV schema, ISOLATED from the 2016+ point-in-time studies so the
validated momentum/factor numbers are untouched.

Intraday open->close returns are split-neutral, so unadjusted prices are fine.
Survivorship caveat: this is a FIXED basket of today's long-lived large caps (no
point-in-time membership back to 2001) — flagged in the runner's report.

Usage:
    python scripts/download_leadlag_history.py
    python scripts/download_leadlag_history.py --start 2001-01-01
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

LEADLAG_DIR = pathlib.Path("data") / "leadlag"

# Long-lived liquid KR large-caps (6-digit code -> Yahoo .KS ticker).
KR_BASKET = ["005930", "000660", "005380", "005490", "015760", "006400", "051910",
             "000270", "012330", "009150", "010950", "000810", "017670", "030200",
             "035420", "066570", "010130", "034730", "003550", "042700"]
# Semis / chip-equipment subset (linkage to US chips should be strongest).
KR_CHIP = ["005930", "000660", "009150", "042700"]
US_SIGNALS = {"SPY": "SPY", "QQQ": "QQQ", "SOXX": "SOXX", "VIX": "^VIX"}


def _fetch(ticker: str, start: str) -> pd.DataFrame:
    import yfinance as yf
    d = yf.download(ticker, start=start, auto_adjust=False, progress=False)
    if d is None or d.empty:
        return pd.DataFrame()
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    d.index = pd.to_datetime(d.index)
    d.index.name = "timestamp"
    return d.dropna()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2001-01-01")
    args = ap.parse_args()
    LEADLAG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading long history (>= {args.start}) into {LEADLAG_DIR}/ ...")
    n_ok = 0
    for code in KR_BASKET:
        df = _fetch(f"{code}.KS", args.start)
        if df.empty:
            print(f"  {code}.KS: no data")
            continue
        df.to_csv(LEADLAG_DIR / f"{code}_1d.csv", index_label="timestamp")
        n_ok += 1
        print(f"  {code}.KS  {len(df):5d} rows  {df.index.min().date()} -> {df.index.max().date()}")
    for name, tk in US_SIGNALS.items():
        df = _fetch(tk, args.start)
        if df.empty:
            print(f"  {name}: no data")
            continue
        df.to_csv(LEADLAG_DIR / f"{name}_1d.csv", index_label="timestamp")
        print(f"  {name:5s} {len(df):5d} rows  {df.index.min().date()} -> {df.index.max().date()}")
    print(f"\nDone. {n_ok}/{len(KR_BASKET)} KR names + {len(US_SIGNALS)} US signals "
          f"saved to {LEADLAG_DIR}/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
