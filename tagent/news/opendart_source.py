"""OpenDART (Korean disclosures) client — recent filings for our KR watchlist.

API (opendart.fss.or.kr): ``GET /api/list.json?crtfc_key=<key>&bgn_de&end_de&page_no&
page_count`` returns recent disclosures; each item carries ``stock_code`` so we filter
to the watchlist. Every filing is tagged with a RULE-BASED importance type (earnings /
dilution / contract / correction / buyback / ...) and a bullish/bearish/neutral guess
from keyword rules — transparent alerts, not price prediction.

``requests`` is imported lazily and the session is injectable, so tests parse canned
payloads with no network. The API key travels only in the query string and is never
logged.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from tagent.config import DATA_DIR

LIST_URL = "https://opendart.fss.or.kr/api/list.json"
VIEWER_URL = "https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcept_no}"
CORPCODE_URL = "https://opendart.fss.or.kr/api/corpCode.xml"
CORP_MAP_FILE = "dart_corp_map.json"

# --- rule-based importance: (tag, [keywords]) checked in order ---
_IMPORTANCE_RULES = [
    ("correction", ["정정"]),
    ("dilution", ["유상증자", "전환사채", "신주인수권", "교환사채", "감자"]),
    ("buyback", ["자기주식", "자사주"]),
    ("contract", ["단일판매", "공급계약"]),
    ("earnings", ["실적", "잠정", "분기보고서", "반기보고서", "사업보고서", "결산", "영업실적"]),
    ("governance", ["합병", "분할", "주식교환", "최대주주"]),
    ("distress", ["관리종목", "상장폐지", "거래정지", "횡령", "배임", "불성실공시", "부도"]),
]
_IMPORTANT_TAGS = {"dilution", "contract", "earnings", "buyback", "distress", "governance"}

_BULLISH_KW = ["자기주식", "자사주", "무상증자", "단일판매", "공급계약", "흑자전환", "배당", "수주"]
_BEARISH_KW = ["유상증자", "전환사채", "신주인수권", "교환사채", "감자", "적자", "관리종목",
               "상장폐지", "거래정지", "횡령", "배임", "불성실공시", "부도"]


def classify_importance(report_nm: str) -> str:
    """Map a disclosure title to a rule-based type tag (else 'other')."""
    t = str(report_nm or "")
    for tag, kws in _IMPORTANCE_RULES:
        if any(k in t for k in kws):
            return tag
    return "other"


def guess_sentiment(report_nm: str) -> str:
    """Rule-based bullish / bearish / neutral guess from the title keywords.
    Corrections (정정) are treated as neutral (direction unknown)."""
    t = str(report_nm or "")
    if "정정" in t:
        return "neutral"
    bull = any(k in t for k in _BULLISH_KW)
    bear = any(k in t for k in _BEARISH_KW)
    if bull and not bear:
        return "bullish"
    if bear and not bull:
        return "bearish"
    return "neutral"


def _iso(rcept_dt: str) -> str:
    s = "".join(ch for ch in str(rcept_dt or "") if ch.isdigit())[:8]
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}" if len(s) == 8 else str(rcept_dt)


def _epoch(rcept_dt: str) -> float:
    from datetime import datetime, timezone
    s = "".join(ch for ch in str(rcept_dt or "") if ch.isdigit())[:8]
    try:
        return datetime.strptime(s, "%Y%m%d").replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return 0.0


def normalize_disclosure(item: dict) -> dict:
    """Raw OpenDART list item -> normalized alert dict."""
    report = item.get("report_nm", "")
    rcept = item.get("rcept_no", "")
    tag = classify_importance(report)
    return {
        "source": "opendart", "market": "kr",
        "time": _iso(item.get("rcept_dt")), "ts": _epoch(item.get("rcept_dt")),
        "symbol": str(item.get("stock_code") or "").strip(),
        "title": report, "type": tag,
        "important": tag in _IMPORTANT_TAGS,
        "sentiment": guess_sentiment(report),
        "filer": item.get("flr_nm") or item.get("corp_name", ""),
        "url": VIEWER_URL.format(rcept_no=rcept) if rcept else "",
    }


def parse_list_response(data: dict) -> List[dict]:
    """Extract the disclosure list from a list.json payload (empty if no data)."""
    if not isinstance(data, dict):
        return []
    if str(data.get("status")) not in ("000", "None"):
        return []                                         # 013 = no data, 0xx = key/limit errors
    return [x for x in (data.get("list") or []) if isinstance(x, dict)]


class OpenDartSource:
    """Fetch + tag recent KR disclosures for a watchlist (key never logged)."""

    def __init__(self, api_key: str, session=None):
        self.api_key = api_key
        self._session = session

    def _get(self, params: dict) -> dict:
        session = self._session
        if session is None:
            import requests  # lazy
            session = requests
        resp = session.get(LIST_URL, params={**params, "crtfc_key": self.api_key}, timeout=10)
        try:
            return resp.json()
        except Exception:
            return {}

    def recent_disclosures_for(self, stock_codes: Sequence[str], bgn_de: str, end_de: str,
                               data_dir=None, session=None, **kw) -> List[dict]:
        """Convenience: load (or fetch+cache) the stock->corp map and query by
        corp_code, so each watchlist name reliably returns ITS disclosures rather
        than being filtered out of the all-market recent list."""
        corp_map = ensure_corp_map(self.api_key, data_dir=data_dir,
                                   session=session or self._session)
        return self.recent_disclosures(stock_codes, bgn_de, end_de, corp_map=corp_map, **kw)

    def recent_disclosures(self, stock_codes: Sequence[str], bgn_de: str, end_de: str,
                           max_pages: int = 3, page_count: int = 100,
                           corp_map: Optional[Dict[str, str]] = None) -> List[dict]:
        """Recent disclosures for ``stock_codes`` over [bgn_de, end_de] (YYYYMMDD).

        Without ``corp_map`` we page the all-market recent list and filter by
        ``stock_code``; with it we query each mapped corp_code directly. Returns
        normalized alerts, newest first."""
        wanted = {str(s).zfill(6) for s in stock_codes}
        out: List[dict] = []
        if corp_map:
            for code in wanted:
                cc = corp_map.get(code)
                if not cc:
                    continue
                page = 1
                while page <= max_pages:                  # paginate this corp's history
                    data = self._get({"corp_code": cc, "bgn_de": bgn_de, "end_de": end_de,
                                      "page_no": page, "page_count": page_count})
                    items = parse_list_response(data)
                    out += [normalize_disclosure(x) for x in items]
                    total_page = int(data.get("total_page") or 1) if isinstance(data, dict) else 1
                    if page >= total_page or not items:
                        break
                    page += 1
        else:
            for page in range(1, max_pages + 1):
                data = self._get({"bgn_de": bgn_de, "end_de": end_de,
                                  "page_no": page, "page_count": page_count})
                items = parse_list_response(data)
                out += [normalize_disclosure(x) for x in items
                        if str(x.get("stock_code") or "").zfill(6) in wanted]
                total_page = int(data.get("total_page") or 1) if isinstance(data, dict) else 1
                if page >= total_page or not items:
                    break
        out.sort(key=lambda a: a["ts"], reverse=True)
        return out


# --------------------------------------------------------------------------- #
# stock_code -> corp_code map (DART corpCode.xml, cached once)
# --------------------------------------------------------------------------- #
def parse_corpcode_zip(content: bytes) -> Dict[str, str]:
    """Parse DART's corpCode.xml ZIP bytes into {stock_code: corp_code} for LISTED
    companies (those with a 6-digit stock_code)."""
    import io
    import zipfile
    import xml.etree.ElementTree as ET
    zf = zipfile.ZipFile(io.BytesIO(content))
    xml_name = next((n for n in zf.namelist() if n.lower().endswith(".xml")), None)
    if xml_name is None:
        return {}
    root = ET.fromstring(zf.read(xml_name))
    out: Dict[str, str] = {}
    for li in root.iter("list"):
        sc = (li.findtext("stock_code") or "").strip()
        cc = (li.findtext("corp_code") or "").strip()
        if cc and sc.isdigit() and len(sc) == 6:
            out[sc] = cc
    return out


def corp_map_path(data_dir=None) -> Path:
    base = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
    return base / CORP_MAP_FILE


def save_corp_map(corp_map: Dict[str, str], data_dir=None) -> Path:
    path = corp_map_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(corp_map, ensure_ascii=False), encoding="utf-8")
    return path


def load_corp_map(data_dir=None) -> Dict[str, str]:
    """Cached {stock_code: corp_code}, or {} if not downloaded yet."""
    path = corp_map_path(data_dir)
    if not os.path.exists(path):
        return {}
    try:
        return {str(k): str(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}
    except Exception:
        return {}


def fetch_corp_map(api_key: str, session=None) -> Dict[str, str]:
    """Download corpCode.xml (a ZIP) once and parse it. Key only in the query
    string; never logged."""
    if session is None:
        import requests  # lazy
        session = requests
    resp = session.get(CORPCODE_URL, params={"crtfc_key": api_key}, timeout=30)
    content = getattr(resp, "content", b"") or b""
    return parse_corpcode_zip(content)


def ensure_corp_map(api_key: str, data_dir=None, session=None,
                    refresh: bool = False) -> Dict[str, str]:
    """Return the stock->corp map, downloading + caching it on first use. On a
    download failure, fall back to whatever is cached (possibly empty)."""
    if not refresh:
        cached = load_corp_map(data_dir)
        if cached:
            return cached
    try:
        fresh = fetch_corp_map(api_key, session=session)
        if fresh:
            save_corp_map(fresh, data_dir)
            return fresh
    except Exception:
        pass
    return load_corp_map(data_dir)
