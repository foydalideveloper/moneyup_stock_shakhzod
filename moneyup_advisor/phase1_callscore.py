"""Phase 1 — Test A: ex-ante call scoring vs REAL KRX prices (read-only, no live trading).

Reads the fact sheets' ex-ante calls, pulls real daily OHLCV via pykrx (stock + a KOSPI200 proxy),
enters at the OPEN of the first trading session AFTER the publish datetime (no look-ahead), and scores
1/5/20-trading-day BETA-ADJUSTED abnormal returns net of 0.50% round-trip cost. Reports, per horizon,
#scorable / #pending, mean net abnormal return, hit-rate and a Newey-West t-stat (bar t >= 3.5), with a
beta-naive (raw) column side-by-side. 'avoid' calls go in a separate bucket.

Market proxy: KODEX 200 ETF (069500) — tracks KOSPI200 ~1:1; pykrx's index endpoint is unavailable in
this environment, the OHLCV endpoint works. Nothing here is a trading signal; mock/analysis only.
"""
from __future__ import annotations

import glob
import json
import math
from datetime import datetime, time as dtime, timedelta, timezone
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np

from moneyup_advisor import config

KST = timezone(timedelta(hours=9))
INDEX_PROXY = "069500"                       # KODEX 200 ETF ~ KOSPI200
INDEX_LABEL = "KODEX 200 ETF (069500) — KOSPI200 proxy (KRX index endpoint unavailable here)"
HORIZONS = (1, 5, 20)
COST_ROUNDTRIP = 0.005
TSTAT_BAR = 3.5
BETA_LOOKBACK = 252
MARKET_OPEN = dtime(9, 0)                     # KRX session open, KST


# --------------------------------------------------------------------------- #
# load ex-ante calls (read-only) + real prices
# --------------------------------------------------------------------------- #
def load_calls() -> List[dict]:
    calls = []
    for p in sorted(glob.glob(str(config.SHEET_DIR / "*.json"))):
        try:
            s = json.loads(open(p, encoding="utf-8").read())
        except Exception:
            continue
        for c in s.get("exante_calls", []):
            pub = c.get("publish_datetime") or (
                (c.get("publish_date") + "T23:59:59Z") if c.get("publish_date") else None)
            if not c.get("ticker") or not pub:
                continue
            calls.append({"video_id": s.get("video_id"), "ticker": str(c["ticker"]).zfill(6),
                          "name": c.get("name"), "direction": c.get("direction"),
                          "publish": pub, "mmss": c.get("mmss")})
    return calls


def _to_kst(iso: str) -> datetime:
    dt = datetime.strptime(iso.replace("Z", "+0000"), "%Y-%m-%dT%H:%M:%S%z")
    return dt.astimezone(KST)


@lru_cache(maxsize=128)
def _ohlcv(ticker: str, start: str, end: str):
    """{date: (open, close)} via pykrx (positional cols: open=0, close=3). {} on failure."""
    from pykrx import stock
    try:
        df = stock.get_market_ohlcv_by_date(start, end, ticker)
    except Exception:
        return {}
    if df is None or len(df) == 0:
        return {}
    out = {}
    for i in range(len(df)):
        d = df.index[i].date()
        o, cl = float(df.iloc[i, 0]), float(df.iloc[i, 3])
        if o > 0 and cl > 0:
            out[d] = (o, cl)
    return out


def _closes(series: Dict) -> tuple:
    dates = sorted(series)
    return dates, np.array([series[d][1] for d in dates], float)


# --------------------------------------------------------------------------- #
# scoring one call
# --------------------------------------------------------------------------- #
def _beta(stock_s: Dict, idx_s: Dict, entry_date) -> float:
    """β vs the index from trailing ~1y daily returns ending the session BEFORE entry. 1.0 fallback."""
    sd = sorted(d for d in stock_s if d < entry_date)
    common = [d for d in sd if d in idx_s][-(BETA_LOOKBACK + 1):]
    if len(common) < 30:
        return 1.0
    sc = np.array([stock_s[d][1] for d in common], float)
    ic = np.array([idx_s[d][1] for d in common], float)
    rs, ri = np.diff(sc) / sc[:-1], np.diff(ic) / ic[:-1]
    var = ri.var()
    return float(np.cov(rs, ri)[0, 1] / var) if var > 0 else 1.0


def score_call(call: dict, stock_s: Dict, idx_s: Dict, last_date) -> dict:
    """Per-horizon scoring for one call. Returns entry/beta + {h: {...}} with status scorable/pending."""
    pub_kst = _to_kst(call["publish"])
    idx_dates = sorted(idx_s)                          # the canonical trading calendar
    # entry = first session whose 09:00 KST open is strictly after the publish moment (no look-ahead)
    entry = next((d for d in idx_dates
                  if datetime.combine(d, MARKET_OPEN, KST) > pub_kst), None)
    res = {"entry_date": None, "beta": None, "horizons": {}}
    if entry is None or entry not in stock_s:
        for h in HORIZONS:
            res["horizons"][h] = {"status": "no_price"}
        return res
    res["entry_date"] = entry.isoformat()
    i0 = idx_dates.index(entry)
    beta = _beta(stock_s, idx_s, entry)
    res["beta"] = round(beta, 3)
    sign = {"long": 1, "short": -1, "avoid": -1}.get(call["direction"], 0)
    s_open = stock_s[entry][0]
    i_open = idx_s[entry][0]
    for h in HORIZONS:
        j = i0 + h - 1                                 # exit session (h-day hold: open[i0] -> close[j])
        if j >= len(idx_dates):
            res["horizons"][h] = {"status": "pending"}      # not enough sessions elapsed yet
            continue
        exit_d = idx_dates[j]
        if exit_d > last_date or exit_d not in stock_s or exit_d not in idx_s:
            res["horizons"][h] = {"status": "pending"}
            continue
        g_stock = stock_s[exit_d][1] / s_open - 1.0
        g_idx = idx_s[exit_d][1] / i_open - 1.0
        abn = g_stock - beta * g_idx                   # beta-adjusted abnormal (stock vs market)
        net_abn = sign * abn - COST_ROUNDTRIP
        net_raw = sign * g_stock - COST_ROUNDTRIP      # beta-naive
        res["horizons"][h] = {"status": "scorable", "exit_date": exit_d.isoformat(),
                              "gross_stock": round(g_stock, 4), "gross_index": round(g_idx, 4),
                              "net_abnormal": round(net_abn, 4), "net_raw": round(net_raw, 4)}
    return res


# --------------------------------------------------------------------------- #
# aggregation + Newey-West t-stat
# --------------------------------------------------------------------------- #
def _nw_tstat(x: List[float], lag: int) -> Optional[float]:
    """t-stat of the mean with Newey-West HAC long-run variance (lag=0 -> naive). None if n<2."""
    a = np.asarray(x, float)
    n = len(a)
    if n < 2:
        return None
    d = a - a.mean()
    lrv = (d * d).sum() / n
    for l in range(1, min(lag, n - 1) + 1):
        lrv += 2 * (1 - l / (lag + 1)) * (d[l:] * d[:-l]).sum() / n
    se = math.sqrt(lrv / n) if lrv > 0 else None
    return round(float(a.mean() / se), 2) if se else None


def _agg(rows: List[dict], key: str, lag: int) -> dict:
    vals = [r[key] for r in rows]
    if not vals:
        return {"n": 0, "mean_pct": None, "hit_rate": None, "tstat": None}
    a = np.array(vals, float)
    return {"n": len(a), "mean_pct": round(float(a.mean()) * 100, 3),
            "hit_rate": round(float((a > 0).mean()), 3),
            "tstat": _nw_tstat(vals, lag)}


def run() -> dict:
    calls = load_calls()
    pubs = [_to_kst(c["publish"]).date() for c in calls]
    start = (min(pubs) - timedelta(days=BETA_LOOKBACK + 180)).strftime("%Y%m%d")
    end = datetime.now(KST).strftime("%Y%m%d")
    idx_s = _ohlcv(INDEX_PROXY, start, end)
    last_date = max(idx_s) if idx_s else datetime.now(KST).date()

    scored = []
    for c in calls:
        ss = _ohlcv(c["ticker"], start, end)
        sc = score_call(c, ss, idx_s, last_date)
        scored.append({**c, **sc})

    n_long = sum(c["direction"] == "long" for c in calls)
    n_short = sum(c["direction"] == "short" for c in calls)
    n_avoid = sum(c["direction"] == "avoid" for c in calls)

    horizons = {}
    avoid_cut = {}
    for h in HORIZONS:
        lag = h - 1
        main = [s["horizons"][h] for s in scored
                if s["direction"] in ("long", "short") and s["horizons"].get(h, {}).get("status") == "scorable"]
        pending = sum(1 for s in scored if s["direction"] in ("long", "short")
                      and s["horizons"].get(h, {}).get("status") == "pending")
        horizons[str(h)] = {
            "n_scorable": len(main), "n_pending": pending,
            "abnormal": _agg(main, "net_abnormal", lag),     # beta-adjusted, net of cost
            "raw": _agg(main, "net_raw", lag),               # beta-naive, net of cost
        }
        av = [s["horizons"][h] for s in scored
              if s["direction"] == "avoid" and s["horizons"].get(h, {}).get("status") == "scorable"]
        avoid_cut[str(h)] = {"n_scorable": len(av), "abnormal_as_short": _agg(av, "net_abnormal", lag)}

    return {
        "generated": datetime.now(KST).strftime("%Y-%m-%d %H:%M KST"),
        "test": "Phase 1 — Test A (ex-ante call scoring vs real KRX prices)",
        "index_proxy": INDEX_LABEL, "data_through": last_date.isoformat(),
        "cost_roundtrip_pct": COST_ROUNDTRIP * 100, "tstat_bar": TSTAT_BAR,
        "beta_lookback_days": BETA_LOOKBACK, "horizons_days": list(HORIZONS),
        "n_calls": len(calls), "n_tickers": len({c["ticker"] for c in calls}),
        "n_long": n_long, "n_short": n_short, "n_avoid": n_avoid,
        "publish_range": [min(pubs).isoformat(), max(pubs).isoformat()],
        "horizons": horizons,
        "avoid_bucket": {"n_calls": n_avoid, "note": "treated as short (sign -1) in this secondary cut",
                         "by_horizon": avoid_cut},
        "per_call": [{"video_id": s["video_id"], "ticker": s["ticker"], "name": s["name"],
                      "direction": s["direction"], "publish": s["publish"],
                      "entry_date": s["entry_date"], "beta": s["beta"],
                      "h": {str(h): s["horizons"].get(h) for h in HORIZONS}} for s in scored],
    }


def render_table(rep: dict) -> str:
    L = [f"Phase 1 — Test A · {rep['generated']} · data through {rep['data_through']}",
         f"calls={rep['n_calls']} (long={rep['n_long']} short={rep['n_short']} avoid={rep['n_avoid']}) "
         f"· tickers={rep['n_tickers']} · publish {rep['publish_range'][0]}..{rep['publish_range'][1]}",
         f"index proxy: {rep['index_proxy']}",
         f"cost {rep['cost_roundtrip_pct']}% round-trip · β-lookback {rep['beta_lookback_days']}d · "
         f"bar t>={rep['tstat_bar']}", "",
         "Horizon | scor | pend | β-adj mean% | β-adj hit | β-adj t | RAW mean% | RAW hit | RAW t",
         "--------|------|------|------------|-----------|---------|-----------|---------|------"]
    for h in rep["horizons_days"]:
        x = rep["horizons"][str(h)]
        a, r = x["abnormal"], x["raw"]
        L.append(f"{h:>3}d    | {x['n_scorable']:>4} | {x['n_pending']:>4} | "
                 f"{a['mean_pct']!s:>10} | {a['hit_rate']!s:>9} | {a['tstat']!s:>7} | "
                 f"{r['mean_pct']!s:>9} | {r['hit_rate']!s:>7} | {r['tstat']!s:>5}")
    L.append("")
    L.append(f"avoid bucket: {rep['avoid_bucket']['n_calls']} calls (secondary cut = treat as short)")
    for h in rep["horizons_days"]:
        ab = rep["avoid_bucket"]["by_horizon"][str(h)]
        L.append(f"  {h}d: n={ab['n_scorable']} β-adj mean%={ab['abnormal_as_short']['mean_pct']} "
                 f"t={ab['abnormal_as_short']['tstat']}")
    verdict = any((rep["horizons"][str(h)]["abnormal"]["tstat"] or 0) >= rep["tstat_bar"]
                  for h in rep["horizons_days"])
    L.append("")
    L.append(f"VERDICT: {'a horizon clears t>=3.5' if verdict else 'no horizon clears t>=3.5 (no demonstrated edge)'}")
    return "\n".join(L)


def main():
    import sys
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    rep = run()
    out = config.DATA_DIR / "phase1_callscore.json"
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(render_table(rep))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
