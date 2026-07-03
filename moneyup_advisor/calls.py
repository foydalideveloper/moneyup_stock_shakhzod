"""Ex-ante ticker calls vs ex-post commentary — from the timestamped transcript (base env).

EX-ANTE call (for the later Phase-1 falsification) carries EXACTLY:
  ticker (6-digit) · direction (long/short/avoid) · stated_price (if any) · in-video timestamp ·
  video publish datetime.
EX-POST commentary (past recommendation / track-record / realized return) is kept SEPARATE so it
can never be mistaken for a fresh, testable call.

Detection is transparent rule-based (Korean keyword + price regex). Every item keeps its verbatim
quote + deep-link. The ticker is resolved to a 6-digit code (Rule 2); the name is a label only.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from moneyup_advisor import config, tickers
from tagent.news.youtube_source import mmss            # READ-ONLY reuse

_LONG = ["매수", "사라", "사세요", "사야", "담아", "담으", "비중확대", "비중 확대", "분할매수",
         "분할 매수", "들어가", "매집", "저점매수", "저점 매수", "롱", "불타기", "추가매수"]
_SHORT = ["매도", "팔아", "파세요", "비중축소", "비중 축소", "손절", "차익실현", "정리", "숏",
          "청산", "처분"]
_AVOID = ["관망", "보유", "지켜", "홀딩", "대기", "쉬어", "회피", "보수적", "현금 비중", "현금비중"]
_EXPOST = ["추천했던", "추천드렸던", "추천 드렸던", "지난번", "저번", "전에 말씀", "말씀드렸",
           "말씀 드렸", "수익률", "수익 실현", "익절", "챙겼", "적중", "맞췄", "이미 매도",
           "보유중이신", "추천종목", "이전에", "지난주", "콜이", "지난 영상"]

# Descriptive compounds that CONTAIN an action word but are NOT a call (market colour, order book,
# net flows, profit-taking). Stripped before direction detection so they can't trigger a false call.
_NOISE = ["매도세", "매수세", "순매수", "순매도", "차익매도", "차익 매도", "단계차익매도",
          "매도물량", "매수물량", "매도호가", "매수호가", "매도잔량", "매수잔량",
          "매도세력", "매수세력", "공매도", "매도벽", "매수벽", "매물", "매도가", "매수가"]

_PRICE_WON = re.compile(r"([0-9][0-9,]{2,})\s*원")
_TARGET = re.compile(r"(?:목표가|목표주가|적정가|적정주가)\D{0,4}([0-9][0-9,]{2,})")
# Korean myriad amount: an expression starting at N억 or N만, incl. following 천/백/십/plain digits.
_KOR_AMT = re.compile(r"\d[\d,]*\s*(?:억|만)[\s\d,억만천백십]*")
_KOR_TOK = re.compile(r"(\d[\d,]*)\s*(억|만|천|백|십)?")
_MULT = {"천": 1000, "백": 100, "십": 10}


def _parse_kor_amount(s: str) -> Optional[int]:
    """Parse a Korean myriad amount -> int. Myriad grouping with the finance shorthand that 천/백
    after 억 (without an intervening 만) scale by 만: '1억 2천' -> 120,000,000; '30만 9,800' -> 309,800."""
    parts = [(int(n.replace(",", "")), u) for n, u in _KOR_TOK.findall(s) if n]
    if not parts:
        return None
    total = section = 0
    pending_man = False                                   # set after 억; makes 천->천만, 백->백만
    for num, u in parts:
        if u == "억":
            section = (section + num) * 100_000_000
            total += section
            section = 0
            pending_man = True
        elif u == "만":
            section = (section + num) * 10_000
            total += section
            section = 0
            pending_man = False
        elif u in _MULT:
            section += num * _MULT[u] * (10_000 if pending_man else 1)
        else:
            section += num
    v = total + section
    return v if v > 0 else None


def parse_korean_price(text: str) -> Optional[int]:
    """First plausible KRW price in ``text`` -> int. Korean 억/만 FIRST (so '30만 9,800원' -> 309,800,
    not 9,800), skipping share counts ('…만주'); then '목표가 N' / 'N원'. Handles '1억 2천' -> 120,000,000."""
    pos = 0
    while True:
        m = _KOR_AMT.search(text, pos)
        if not m:
            break
        if not text[m.end():m.end() + 1] == "주":          # '…만주' = share count, not a price
            v = _parse_kor_amount(m.group(0))
            if v and v >= 1000:
                return v
        pos = m.end()
    m = _TARGET.search(text)
    if m:
        return int(m.group(1).replace(",", ""))
    m = _PRICE_WON.search(text)
    if m:
        return int(m.group(1).replace(",", ""))
    return None


def _strip_noise(text: str) -> str:
    for n in _NOISE:                          # remove descriptive compounds first
        text = text.replace(n, " ")
    return text


def _direction(text: str) -> Optional[str]:
    text = _strip_noise(text)
    li = next((text.find(k) for k in _LONG if k in text), -1)
    si = next((text.find(k) for k in _SHORT if k in text), -1)
    if li >= 0 and (si < 0 or li <= si):
        return "long"
    if si >= 0:
        return "short"
    if any(k in text for k in _AVOID):
        return "avoid"
    return None


def _ticker_for(text: str, primary_code: Optional[str]) -> Optional[str]:
    """Resolve the call's ticker. A bare code wins; else a NAMED stock (>=3 chars, so 2-char names
    don't false-match inside words) — but if the primary is among them, prefer the primary (these
    are single-stock videos). Else the primary."""
    codes = tickers.codes_in_text(text)
    if codes:
        return codes[0]
    names = tickers.names_in_text(text, min_len=3)
    if names:
        return primary_code if (primary_code in names) else names[0]
    return primary_code                       # the video is about the primary stock


def extract(segments: List[Dict], video: Dict, primary_code: Optional[str]) -> Dict[str, List[dict]]:
    """{'exante':[...], 'expost':[...]} from transcript segments + video meta."""
    vid = video.get("video_id", "")
    pub_dt = video.get("publish_datetime") or video.get("publish_date")
    exante, expost = [], []
    seen = set()
    for i, seg in enumerate(segments):
        text = seg.get("text", "")
        direction = _direction(text)
        is_expost = any(k in text for k in _EXPOST)
        if direction is None and not is_expost:
            continue
        code = _ticker_for(text, primary_code)
        if not code:
            continue
        t = float(seg.get("start", 0.0) or 0.0)
        # a little lead-in context for the quote (prev + this + next)
        lo, hi = max(0, i - 1), min(len(segments), i + 2)
        quote = " ".join(s.get("text", "") for s in segments[lo:hi]).strip()
        price = parse_korean_price(quote)
        item = {
            "ticker": code, "name": tickers.display_name(code),
            "direction": direction, "stated_price": price,
            "in_video_t": round(t, 1), "mmss": mmss(t),
            "publish_datetime": pub_dt, "publish_date": video.get("publish_date"),
            "deeplink": config.deeplink(vid, t), "quote": quote,
            "matched": [k for k in (_LONG + _SHORT + _AVOID) if k in text][:3],
        }
        if is_expost:
            item["expost_markers"] = [k for k in _EXPOST if k in text][:3]
            expost.append(item)
        else:
            key = (code, direction)
            if key in seen:                   # one ex-ante call per (ticker, direction): keep first
                continue
            seen.add(key)
            exante.append(item)
    return {"exante": exante, "expost": expost}
