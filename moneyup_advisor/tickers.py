"""Ticker resolution — RULE 2: every fact is keyed on the 6-digit KRX code, never the name.

Resolution priority (most reliable first):
  1. An on-screen 6-digit code OCR'd from the HTS (digits are read reliably) that is a REAL
     listed code (validated against ``data/dart_corp_map.json``'s 3,920 codes).
  2. An exact company-name match (transcript / title / OCR name) -> code.
  3. A FUZZY name match (difflib) -> code, to recover a glyph-dropped OCR name
     (e.g. 하나머티리얼즈 -> "해나머티리"). The Korean name is a DISPLAY LABEL only.

The name<->code map is built once from pykrx (cached, git-ignored) and seeded with the curated
map already in ``tagent.news.youtube_source`` (imported READ-ONLY).
"""
from __future__ import annotations

import json
import os
import re
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from moneyup_advisor import config

_CODE_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)")
_NAME_MAP_CACHE = config.CACHE_DIR / "krx_name_map.json"
_DART_MAP = config.REPO_ROOT / "data" / "dart_corp_map.json"


def _norm(s: str) -> str:
    """Lowercase + strip ALL whitespace/punctuation so HTS spacing/glyph noise doesn't matter."""
    return re.sub(r"[^0-9a-z가-힣]", "", str(s or "").lower())


# --------------------------------------------------------------------------- #
# valid-code set — the cheap, offline ground truth for "is this a real ticker?"
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def code_set() -> frozenset:
    """The ~3,920 listed 6-digit KRX codes (keys of dart_corp_map.json). Offline."""
    try:
        d = json.loads(_DART_MAP.read_text(encoding="utf-8"))
        return frozenset(str(k).zfill(6) for k in d.keys() if str(k).isdigit())
    except Exception:
        return frozenset()


def valid_code(code: str) -> bool:
    c = str(code or "").zfill(6)
    return c in code_set()


# --------------------------------------------------------------------------- #
# name <-> code map — built once (pykrx), cached, seeded by youtube_source
# --------------------------------------------------------------------------- #
def _seed_from_youtube_source() -> Tuple[Dict[str, str], Dict[str, str]]:
    """(name->code, code->display) from the curated map in youtube_source (READ-ONLY import)."""
    name2code, code2disp = {}, {}
    try:
        from tagent.news.youtube_source import WATCHLIST_TICKERS, TICKER_NAMES
        for alias, code in WATCHLIST_TICKERS.items():
            if str(code).isdigit():                       # KR codes only (skip NVDA etc.)
                name2code[_norm(alias)] = str(code).zfill(6)
        for code, disp in TICKER_NAMES.items():
            if str(code).isdigit():
                code2disp[str(code).zfill(6)] = disp
    except Exception:
        pass
    return name2code, code2disp


def _dart_key() -> str:
    """OPENDART_API_KEY from env, else parsed straight out of the repo .env (read-only)."""
    k = os.getenv("OPENDART_API_KEY")
    if k:
        return k.strip()
    try:
        for line in (config.REPO_ROOT / ".env").read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("OPENDART_API_KEY="):
                return line.split("=", 1)[1].strip()
    except Exception:
        pass
    return ""


def _build_from_dart() -> Dict[str, str]:
    """{6-digit stock_code: corp_name} for ALL listed KR firms via DART corpCode.xml (one call,
    cached by caller). The robust source — the daily pipeline uses the same endpoint. {} on failure."""
    import io
    import urllib.request
    import zipfile
    from xml.etree import ElementTree as ET
    key = _dart_key()
    if not key:
        return {}
    try:
        url = f"https://opendart.fss.or.kr/api/corpCode.xml?crtfc_key={key}"
        raw = urllib.request.urlopen(url, timeout=40).read()
        zf = zipfile.ZipFile(io.BytesIO(raw))
        root = ET.fromstring(zf.read(zf.namelist()[0]))
        out: Dict[str, str] = {}
        for li in root.iter("list"):
            sc = (li.findtext("stock_code") or "").strip()
            nm = (li.findtext("corp_name") or "").strip()
            if sc.isdigit() and len(sc) == 6 and nm:
                out[sc.zfill(6)] = nm
        return out
    except Exception:
        return {}


def _build_from_pykrx() -> Dict[str, str]:
    """{code: official_name} for KOSPI+KOSDAQ via pykrx. Network; cached by caller. {} on failure."""
    out: Dict[str, str] = {}
    try:
        from pykrx import stock
        import datetime as _dt
        today = _dt.date.today().strftime("%Y%m%d")
        for market in ("KOSPI", "KOSDAQ"):
            try:
                codes = stock.get_market_ticker_list(today, market=market)
            except TypeError:
                codes = stock.get_market_ticker_list(market=market)
            for c in codes or []:
                try:
                    out[str(c).zfill(6)] = stock.get_market_ticker_name(c)
                except Exception:
                    continue
    except Exception:
        pass
    return out


def ensure_name_map(rebuild: bool = False) -> Dict[str, Dict[str, str]]:
    """Load (or build+cache) the name<->code map. Returns {"name2code":{}, "code2name":{}}.

    Cached to a git-ignored JSON so reruns skip the pykrx network call. Always usable: even with
    no network it returns the youtube_source seed."""
    seed_n2c, seed_c2d = _seed_from_youtube_source()
    if not rebuild and _NAME_MAP_CACHE.exists():
        cached = json.loads(_NAME_MAP_CACHE.read_text(encoding="utf-8"))
        cached["name2code"].update(seed_n2c)
        return cached
    code2name = _build_from_dart() or _build_from_pykrx()    # DART first (robust), pykrx fallback
    code2name_disp = dict(code2name)
    code2name_disp.update(seed_c2d)
    name2code = {_norm(n): c for c, n in code2name.items()}
    name2code.update(seed_n2c)                              # curated aliases win
    payload = {"code2name": code2name_disp, "name2code": name2code}
    try:
        _NAME_MAP_CACHE.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    return payload


@lru_cache(maxsize=1)
def _maps() -> Tuple[Dict[str, str], Dict[str, str], List[Tuple[str, str]]]:
    m = ensure_name_map()
    name2code = m.get("name2code", {})
    code2name = m.get("code2name", {})
    # list of (normalized_name, code) sorted longest-first for greedy substring matching
    pairs = sorted(((n, c) for n, c in name2code.items() if n), key=lambda x: -len(x[0]))
    return name2code, code2name, pairs


def display_name(code: str) -> str:
    code = str(code or "").zfill(6)
    return _maps()[1].get(code, code)


# --------------------------------------------------------------------------- #
# the resolvers
# --------------------------------------------------------------------------- #
def codes_in_text(text: str) -> List[str]:
    """Bare valid 6-digit codes literally present in ``text`` (e.g. spoken/typed '005930')."""
    return sorted({m.zfill(6) for m in _CODE_RE.findall(str(text or "")) if valid_code(m)})


def names_in_text(text: str, min_len: int = 2) -> List[str]:
    """All known company names mentioned in ``text`` -> codes (greedy longest-match, deduped).

    Used on transcript segments and titles. Returns codes in order of first appearance."""
    low = _norm(text)
    if not low:
        return []
    _, _, pairs = _maps()
    found, seen = [], set()
    for name, code in pairs:
        if len(name) < min_len:
            continue
        if name in low and code not in seen:
            seen.add(code)
            found.append((low.index(name), code))
    return [c for _, c in sorted(found)]


def fuzzy_name_to_code(name: str, cutoff: float = 0.72) -> Optional[Tuple[str, float]]:
    """Best fuzzy code match for a (possibly glyph-dropped) OCR'd name. (code, score) or None."""
    q = _norm(name)
    if len(q) < 2:
        return None
    name2code, _, _ = _maps()
    if q in name2code:
        return name2code[q], 1.0
    best, best_score = None, 0.0
    for n, c in name2code.items():
        if not n or abs(len(n) - len(q)) > 4:
            continue
        sm = SequenceMatcher(None, q, n)
        sc = sm.ratio()
        # OCR drops/garbles a leading or trailing glyph -> a long shared run still pins the name.
        # Reward a long common substring covering most of the shorter (OCR'd) string.
        blk = sm.find_longest_match(0, len(q), 0, len(n)).size
        if blk >= 3 and blk / min(len(q), len(n)) >= 0.6:
            sc += 0.18
        if n.startswith(q[: max(2, len(q) - 1)]) or q.startswith(n[: max(2, len(n) - 1)]):
            sc += 0.08
        if sc > best_score:
            best, best_score = c, sc
    return (best, round(min(best_score, 1.0), 3)) if best and best_score >= cutoff else None


def resolve(token: str) -> Optional[str]:
    """Resolve a single token (code or name) to a 6-digit code, or None. Code beats name."""
    codes = codes_in_text(token)
    if codes:
        return codes[0]
    names = names_in_text(token)
    if names:
        return names[0]
    fz = fuzzy_name_to_code(token)
    return fz[0] if fz else None
