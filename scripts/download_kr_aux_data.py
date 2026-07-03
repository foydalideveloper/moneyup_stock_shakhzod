"""Download the auxiliary KR data the index-calibration study needs (pykrx, cached).

Writes (idempotent; safe to re-run):
  * data/kospi200_long_1d.csv  — KOSPI-200 index (code 1028) from 1990 (longest history)
  * data/kospi200_1d.csv       — same index, 2016-> (matches the PIT panel window)
  * data/kr_shares.csv         — shares-outstanding snapshot (constant-shares cap weights)
  * data/kr_sectors.csv        — name -> KRX KOSPI sector (semi/electronics = 전기전자)
  * data/kr_ssf_available.csv  — single-stock-futures availability (KOSPI-200 proxy)

Sector fetches are ~14s each (20 calls) under the sandbox network throttle, so the
sector step writes incrementally and the whole run takes a few minutes. Data files are
gitignored; this script is the reproducible source. Usage: python scripts/download_kr_aux_data.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass

import pandas as pd  # noqa: E402

from tagent.config import DATA_DIR  # noqa: E402
from tagent.data.krx_source import ensure_krx_login  # noqa: E402
from tagent.kr_universe import load_members, save_ssf_available, universe_symbols  # noqa: E402

DATA = pathlib.Path(DATA_DIR)
# KOSPI sector index codes -> short names (전기전자 = electronics/semis, the overweight concern)
SECTOR_NAMES = {"1005": "음식료담배", "1006": "섬유의류", "1007": "종이목재", "1008": "화학",
                "1009": "제약", "1010": "비금속", "1011": "금속", "1012": "기계장비",
                "1013": "전기전자", "1014": "의료정밀", "1015": "운송장비", "1016": "유통",
                "1017": "전기가스", "1018": "건설", "1019": "운송창고", "1020": "통신",
                "1021": "금융", "1024": "증권", "1025": "보험", "1026": "일반서비스"}


def _save_index(stock, start, name):
    df = stock.get_index_ohlcv_by_date(start, "20260610", "1028").rename(
        columns={"시가": "open", "고가": "high", "저가": "low", "종가": "close", "거래량": "volume"})
    out = df[["open", "high", "low", "close", "volume"]].copy()
    out.index.name = "timestamp"
    out.to_csv(DATA / f"{name}_1d.csv")
    print(f"  {name}_1d.csv: {len(out)} rows {out.index.min().date()}..{out.index.max().date()}")


def main() -> int:
    ensure_krx_login()
    from pykrx import stock

    syms = set(universe_symbols(load_members()))
    _save_index(stock, "19900101", "kospi200_long")
    _save_index(stock, "20160101", "kospi200")

    # shares-outstanding snapshot (KOSPI + KOSDAQ)
    sh = {}
    for mkt in ("KOSPI", "KOSDAQ"):
        cap = stock.get_market_cap_by_ticker("20240102", market=mkt)
        col = "상장주식수" if "상장주식수" in cap.columns else cap.columns[-1]
        for t in cap.index:
            if t in syms:
                sh[t] = int(cap.loc[t, col])
    pd.DataFrame(sorted(sh.items()), columns=["ticker", "shares"]).to_csv(DATA / "kr_shares.csv", index=False)
    print(f"  kr_shares.csv: {len(sh)}/{len(syms)} names")

    # SSF availability (KOSPI-200 membership proxy)
    k200 = set(stock.get_index_portfolio_deposit_file("1028"))
    save_ssf_available(sorted(syms), k200)
    print(f"  kr_ssf_available.csv: {sum(s in k200 for s in syms)}/{len(syms)} SSF-available")

    # name -> sector (incremental write; ~14s/call)
    sectors, ok = {}, 0
    for code, nm in SECTOR_NAMES.items():
        try:
            for t in stock.get_index_portfolio_deposit_file(code):
                if t in syms and t not in sectors:
                    sectors[t] = nm
            ok += 1
            tmp = {**{t: "기타KOSDAQ" for t in syms}, **sectors}
            pd.DataFrame(sorted(tmp.items()), columns=["ticker", "sector"]).to_csv(
                DATA / "kr_sectors.csv", index=False)
            print(f"  sector ok {code} {nm}", flush=True)
        except Exception as e:
            print(f"  sector ERR {code} {repr(e)[:50]}", flush=True)
    print(f"  kr_sectors.csv: {len(syms)} names, {ok}/{len(SECTOR_NAMES)} sector indices")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
