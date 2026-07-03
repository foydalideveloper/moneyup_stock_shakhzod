"""Top-source whitelist — keep only tier-1 outlets in the feed + briefing newspaper.

  * KR — by article DOMAIN: 한국경제(hankyung), 매일경제(mk), 머니투데이(mt) + optional 서울경제(sedaily)
    / 연합뉴스(yna). Low-tier outlets (pinpointnews, hansbiz, EBN, aitimes, dizzotv, …) are dropped.
  * US — by Finnhub's per-article publisher (``source``/``publisher``): Bloomberg, WSJ, Reuters, CNBC.

Non-news items (YouTube insights, Kiwoom signals) are always kept — they aren't newspaper outlets.
Whitelists are configurable via SOURCE_WHITELIST_KR / SOURCE_WHITELIST_US in .env. Pure, no network.
"""

from __future__ import annotations

from typing import List, Optional, Sequence
from urllib.parse import urlparse

DEFAULT_KR = ["hankyung.com", "mk.co.kr", "mt.co.kr", "sedaily.com", "yna.co.kr"]
DEFAULT_US = ["bloomberg", "wsj", "wall street journal", "reuters", "cnbc"]

# domain substring -> readable outlet (for display).
_KR_OUTLET = {
    "hankyung.com": "한국경제", "hankyung": "한국경제", "mk.co.kr": "매일경제",
    "mt.co.kr": "머니투데이", "moneytoday": "머니투데이", "mtn.co.kr": "머니투데이방송",
    "sedaily.com": "서울경제", "yna.co.kr": "연합뉴스", "yonhapnews": "연합뉴스",
}


def parse_whitelist(value: str, default: Sequence[str]) -> List[str]:
    items = [x.strip().lower() for x in str(value or "").split(",") if x.strip()]
    return items or list(default)


def domain_of(url: str) -> str:
    try:
        net = urlparse(str(url or "")).netloc.lower()
    except Exception:
        return ""
    return net[4:] if net.startswith("www.") else net


def outlet_name(url: str) -> str:
    """A readable KR outlet name for a URL, else its bare domain."""
    d = domain_of(url)
    for frag, name in _KR_OUTLET.items():
        if frag in d:
            return name
    return d


def is_top_source(item: dict, *, kr_whitelist: Optional[Sequence[str]] = None,
                  us_whitelist: Optional[Sequence[str]] = None) -> bool:
    """True if a news item is from a whitelisted top outlet. KR matched on the URL domain; US on the
    Finnhub publisher; non-news (youtube/kiwoom) always kept."""
    kr = [w.lower() for w in (kr_whitelist or DEFAULT_KR)]
    us = [w.lower() for w in (us_whitelist or DEFAULT_US)]
    src = str(item.get("source") or "").lower()
    market = str(item.get("market") or "").lower()
    if src in ("youtube", "kiwoom"):
        return True
    url = str(item.get("url") or item.get("link") or "").lower()
    if src == "finnhub" or market == "us":
        pub = str(item.get("publisher") or (item.get("extra") or {}).get("publisher") or "").lower()
        hay = f"{pub} {url}"
        return any(w in hay for w in us)
    d = domain_of(url)
    return bool(d) and any(w in d for w in kr)


def filter_top_sources(items: Sequence[dict], *, kr_whitelist=None, us_whitelist=None) -> List[dict]:
    return [it for it in (items or []) if is_top_source(it, kr_whitelist=kr_whitelist,
                                                        us_whitelist=us_whitelist)]
