"""Download long daily histories for the Track C multi-market trend book (yfinance).

Writes data/{gspc,btc,eth}_1d.csv (KOSPI-200 long history comes from
download_kr_aux_data.py). Free data; auto-adjusted. Usage:
    python scripts/download_multi_market.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd  # noqa: E402

from tagent.config import DATA_DIR  # noqa: E402

SPECS = [("^GSPC", "gspc", "1990-01-01"), ("BTC-USD", "btc", "2014-01-01"),
         ("ETH-USD", "eth", "2015-01-01"), ("KRW=X", "usdkrw", "2003-01-01")]


def main() -> int:
    import yfinance as yf
    end = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    for tk, name, start in SPECS:
        try:
            df = yf.download(tk, start=start, end=end.strftime("%Y-%m-%d"),
                             progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            out = df[["Open", "High", "Low", "Close", "Volume"]].rename(columns=str.lower).dropna()
            out.index.name = "timestamp"
            out.to_csv(pathlib.Path(DATA_DIR) / f"{name}_1d.csv")
            print(f"  {name}: {len(out)} rows {out.index.min().date()}..{out.index.max().date()}")
        except Exception as e:
            print(f"  {name}: ERR {str(e)[:70]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
