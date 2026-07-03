"""Differentiated recommendation blend for the daily briefing #4.

  * CORE — our validated 12-1 MOMENTUM view: rank the KR watchlist by point-in-time 12-1
    momentum on cached daily closes; cited "our momentum model". This is our own signal, so it
    carries a provenance citation (not a URL).
  * PROVIDERS — external, each pick MUST carry a real source URL or it is dropped:
      - KR broker research via Naver (목표주가 / 투자의견 articles; per-pick broker attribution),
      - 13F whale holdings (Berkshire / BlackRock / Vanguard / Fidelity) from a cached holdings
        file, each citing its whalewisdom URL.

Pure / injectable: ``momentum_picks`` takes a panel (or loads cached closes), broker picks take a
Naver source, 13F takes a holdings dict — so the unit tests run fully mocked, no network. Honest:
a provider with nothing cited contributes nothing (→ "no coverage"); we never invent a pick.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from tagent.config import DATA_DIR

MOM_LOOKBACK, MOM_SKIP = 252, 21
MOM_CITE = "our-momentum-model: 12-1 모멘텀 (PIT, 검증된 엣지)"
# Korean broker names we attribute a research headline to when present in the title.
_BROKERS = ["키움", "삼성", "미래에셋", "유안타", "한화", "한국투자", "NH", "KB", "대신", "하나",
            "신한", "교보", "메리츠", "DB", "현대차", "SK", "하이투자", "IBK", "케이프"]
WHALE_FILE = "whale_13f.json"


# --------------------------------------------------------------------------- #
# CORE — our 12-1 momentum view (cited as our own model)
# --------------------------------------------------------------------------- #
def momentum_picks(symbols: Sequence[str], *, panel: Optional[dict] = None, asof=None,
                   top_n: int = 5, lookback: int = MOM_LOOKBACK, skip: int = MOM_SKIP,
                   data_dir=None) -> List[dict]:
    """Top-``top_n`` watchlist names by 12-1 momentum (no-lookahead), each a cited pick.
    Loads cached KR closes if ``panel`` is None. Names without enough history are skipped."""
    if panel is None:
        from tagent.stock_momentum import load_stock_panel
        panel = {}
        for s in symbols:
            try:
                panel.update(load_stock_panel("kr", symbols=[str(s)], fields=["close"],
                                              min_bars=lookback + skip + 1, data_dir=data_dir))
            except Exception:
                continue
    if not panel:
        return []
    from tagent.xs_momentum import align_close
    close = align_close(panel)
    if asof is not None:
        close = close.loc[close.index <= pd.Timestamp(asof)]
    need = lookback + skip + 1
    scored = []
    for s in symbols:
        if str(s) not in close.columns:
            continue
        cs = close[str(s)].dropna()
        if len(cs) < need:
            continue
        mom = float(cs.iloc[-1 - skip] / cs.iloc[-1 - skip - lookback] - 1.0)       # 12-1
        scored.append({"symbol": str(s), "mom": mom, "price": float(cs.iloc[-1])})
    scored.sort(key=lambda r: r["mom"], reverse=True)
    picks = []
    for i, r in enumerate(scored[:top_n]):
        picks.append({"symbol": r["symbol"], "side": "buy" if r["mom"] > 0 else "avoid",
                      "reason": f"12-1 모멘텀 {r['mom'] * 100:+.1f}% (랭크 {i + 1}/{len(scored)})",
                      "cite": MOM_CITE, "score": round(r["mom"], 4), "rank": i + 1})
    return picks


# --------------------------------------------------------------------------- #
# PROVIDER — KR broker research via Naver (each cited; per-broker attribution)
# --------------------------------------------------------------------------- #
def _broker_of(title: str) -> str:
    """The broker that authored a research headline. A bare broker name is only accepted when it
    is NOT immediately followed by another hangul syllable — so '삼성' inside '삼성전자' is not
    mistaken for 삼성증권, while '미래에셋 ...' and '키움증권' both resolve correctly."""
    t = str(title or "")
    for b in _BROKERS:
        if re.search(re.escape(b) + r"증권", t) or re.search(re.escape(b) + r"(?![가-힣])", t):
            return f"{b}증권"
    return "증권가"


def naver_broker_picks(naver, symbols: Sequence[str], *, names: Optional[Dict[str, str]] = None,
                       per_symbol: int = 4) -> dict:
    """KR broker target-price / opinion picks from Naver — one provider, each pick carrying its
    own broker house + the real article URL. Off-topic / uncited articles are dropped."""
    from tagent.news.naver_source import WATCHLIST_QUERY, is_relevant
    names = names or WATCHLIST_QUERY
    picks: List[dict] = []
    for code in symbols:
        nm = names.get(str(code), str(code))
        try:
            arts = naver.search(f"{nm} 목표주가", display=per_symbol)
        except Exception:
            arts = []
        for a in arts:
            if not a.get("url") or not is_relevant(a, nm):
                continue
            title = a.get("title", "")
            side = "sell" if any(k in title for k in ("매도", "하향", "목표가 하향")) else "buy"
            picks.append({"symbol": str(code), "house": _broker_of(title), "source": "naver-research",
                          "reason": title, "url": a["url"], "side": side})
    return {"house": "KR브로커(Naver)", "source": "naver-research", "picks": picks}


# --------------------------------------------------------------------------- #
# PROVIDER — 13F whale holdings (cached; each cites its whalewisdom URL)
# --------------------------------------------------------------------------- #
def _whale_path(data_dir=None) -> Path:
    return (Path(data_dir) if data_dir is not None else Path(DATA_DIR)) / WHALE_FILE


def load_whale_holdings(data_dir=None) -> dict:
    """{whale: {code: {url, shares?, reason?}}} from data/whale_13f.json, or {} if absent."""
    p = _whale_path(data_dir)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def whale_13f_picks(symbols: Sequence[str], *, holdings: Optional[dict] = None, data_dir=None) -> dict:
    """13F whale (Berkshire/BlackRock/Vanguard/Fidelity) holdings of watchlist names, each citing
    its whalewisdom URL. Empty holdings -> no picks (honest, never invented)."""
    holdings = load_whale_holdings(data_dir) if holdings is None else holdings
    want = {str(s) for s in symbols}
    picks: List[dict] = []
    for whale, hd in (holdings or {}).items():
        for code, info in (hd or {}).items():
            if str(code) not in want:
                continue
            url = (info or {}).get("url", "")
            if not url:                                  # no source -> drop
                continue
            shares = (info or {}).get("shares")
            reason = (info or {}).get("reason") or (f"{whale} 13F 보유" +
                                                    (f" ({shares:,}주)" if isinstance(shares, (int, float)) else ""))
            picks.append({"symbol": str(code), "house": whale, "source": "13F/whalewisdom",
                          "reason": reason, "url": url, "side": "buy"})
    return {"house": "13F whales", "source": "13F/whalewisdom", "picks": picks}


def build_recommendation_live(symbols: Sequence[str], *, naver=None, panel: Optional[dict] = None,
                              whale_holdings: Optional[dict] = None, kiwoom_picks: Sequence[str] = (),
                              today=None, top_n: int = 5, lookback: int = MOM_LOOKBACK,
                              skip: int = MOM_SKIP, data_dir=None) -> dict:
    """Convenience: assemble momentum view + providers and reduce to the #4 report."""
    from tagent.daily_briefing import build_recommendation_report
    our = momentum_picks(symbols, panel=panel, top_n=top_n, lookback=lookback, skip=skip,
                         data_dir=data_dir)
    providers = []
    if naver is not None:
        providers.append(naver_broker_picks(naver, symbols))
    providers.append(whale_13f_picks(symbols, holdings=whale_holdings, data_dir=data_dir))
    return build_recommendation_report(symbols, providers=providers, our_view=our,
                                       kiwoom_picks=kiwoom_picks, today=today)
