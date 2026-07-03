"""Turn the vision worker's raw OCR rows into GROUNDED numbers (base env).

PaddleOCR is the single source of truth for every number (Rule 1). This module:
  * parses numeric tokens (commas stripped, %/decimal/sign detected),
  * uses on-screen LABELS (현재가/등락률/대비/거래량/시가/고가/저가) + bbox proximity to assign
    each number to a field for the frame's ACTIVE ticker (the 6-digit code in the header band),
  * aggregates each field across frames by most-frequent high-confidence value (robust to OCR
    jitter), keeping the supporting evidence (frame, timestamp, confidence),
  * best-effort reads the 관심종목 watchlist rows (name+price+change), name fuzzily -> 6-digit code.

Every number carries provenance; nothing is invented.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional

from moneyup_advisor import tickers

_LABELS = {"현재가": "price", "등락률": "change_pct", "대비": "change_abs",
           "거래량": "volume", "시가": "open", "고가": "high", "저가": "low"}
_NUM_RE = re.compile(r"^[+\-▲▼]?\s*[\d,]+(?:\.\d+)?%?$")


def parse_num(text: str) -> Optional[dict]:
    """Parse one token -> {value, raw, is_pct, is_decimal, sign}. None if not numeric."""
    t = str(text or "").strip()
    if not _NUM_RE.match(t):
        return None
    is_pct = "%" in t
    sign = None
    if t[:1] in "+▲":
        sign = "+"
    elif t[:1] in "-▼":
        sign = "-"
    core = re.sub(r"[+\-▲▼%\s]", "", t).replace(",", "")
    if not core or core == ".":
        return None
    try:
        val = float(core) if "." in core else int(core)
    except ValueError:
        return None
    return {"value": val, "raw": t, "is_pct": is_pct, "is_decimal": "." in core, "sign": sign}


def _center(bbox):
    if not bbox:
        return None
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def frame_active_code(frame: dict) -> Optional[str]:
    """The ticker the frame is 'on': a valid 6-digit code in the top header band, else the lone
    code anywhere in the frame, else None."""
    h = frame.get("h", 1080)
    top, anywhere = [], []
    for row in frame.get("ocr", []):
        for code in tickers.codes_in_text(row.get("text", "")):
            anywhere.append(code)
            c = _center(row.get("bbox"))
            if c and c[1] < h * 0.30:
                top.append(code)
    if top:
        return Counter(top).most_common(1)[0][0]
    uniq = set(anywhere)
    return next(iter(uniq)) if len(uniq) == 1 else None


def _field_ok(field: str, pn: dict) -> bool:
    """Format + plausibility sanity per field, so a label can't bind to a wrongly-typed/implausible
    number (e.g. a row-rank digit '3' grabbed as a price)."""
    if field == "change_pct":
        return pn["is_pct"] or pn["is_decimal"]
    if field == "volume":
        return (not pn["is_decimal"]) and pn["value"] >= 1000
    if field in ("price", "open", "high", "low"):           # KRW prices: integer, >=50
        return (not pn["is_pct"]) and (not pn["is_decimal"]) and 50 <= pn["value"] <= 90_000_000
    if field == "change_abs":
        return (not pn["is_pct"]) and pn["value"] >= 1
    return True


def _primary_row_y(frame: dict, primary_code: str, w: float):
    """The y of the PRIMARY ticker's DATA row in this frame. Prefer the name in the left column (a
    watchlist data row, where the values sit), else the on-screen 6-digit code. None if the primary
    isn't on this frame (then no fields are taken from it)."""
    if not primary_code:
        return None
    for row in frame.get("ocr", []):                        # name in the left column = the data row
        c = _center(row.get("bbox"))
        if c and c[0] < w * 0.34:
            fz = tickers.fuzzy_name_to_code(row.get("text", ""))
            if fz and fz[0] == primary_code:
                return c[1]
    for row in frame.get("ocr", []):                        # else the exact code (e.g. a 현재가 panel)
        c = _center(row.get("bbox"))
        if c and primary_code in tickers.codes_in_text(row.get("text", "")):
            return c[1]
    return None


def labeled_fields(frame: dict, primary_code: Optional[str] = None) -> Dict[str, dict]:
    """{field: {value, score, raw, t}} for the PRIMARY ticker in this frame.

    Handles BOTH HTS layouts grounded-ly:
      * COLUMN-HEADER table (관심종목: 현재가/대비/등락률/거래량 share a header row, values stacked
        below): each value must be COLUMN-ALIGNED with its header AND in the PRIMARY ticker's ROW
        (found by its code / fuzzy name). If the primary isn't on the table, NOTHING is taken — never
        the top row of someone else's data.
      * INLINE panel (현재가 window: 'label  value' on one line): same-row, to the right.
    If nothing qualifies, the field is OMITTED (no fabricated/mis-columned number — grounding rule).
    """
    h, w = frame.get("h", 1080), frame.get("w", 1920)
    nums, labels = [], []
    for row in frame.get("ocr", []):
        c = _center(row.get("bbox"))
        if c is None:
            continue
        pn = parse_num(row.get("text", ""))
        if pn:
            nums.append((c, pn, float(row.get("score", 0))))
        txt = (row.get("text", "") or "").strip()
        for lbl, field in _LABELS.items():
            # exact-ish label only: reject embedded strays like '거래량단순 52060120' / '보통 현재가 100%'
            if lbl in txt and len(txt) <= len(lbl) + 3:
                labels.append((c, field))
    if not labels or not nums:
        return {}

    # column-header table iff >=2 field labels sit on (nearly) the same row
    lys = sorted(lc[1] for lc, _ in labels)
    header_mode = len(labels) >= 2 and (lys[-1] - lys[0]) < h * 0.03
    active_y = _primary_row_y(frame, primary_code, w)
    if header_mode and active_y is None:
        return {}                                           # multi-stock table, primary not on it

    def match(field, lc):
        best, best_d = None, 1e18
        for (nc, pn, sc) in nums:
            if not _field_ok(field, pn):
                continue
            dx, dy = abs(nc[0] - lc[0]), abs(nc[1] - lc[1])
            if header_mode:                                 # value below header, primary's row
                if dx > w * 0.035 or nc[1] <= lc[1] or abs(nc[1] - active_y) > h * 0.018:
                    continue
                d = dy + dx * 4
            else:                                           # inline: same row, to the right
                if dy >= h * 0.02 or nc[0] < lc[0] - w * 0.01:
                    continue
                if active_y is not None and abs(nc[1] - active_y) > h * 0.05:
                    continue                                # if we know the primary's row, stay near it
                d = dx + dy * 3
            if d < best_d:
                best, best_d = (pn, sc, nc), d
        return best

    out: Dict[str, dict] = {}
    for lc, field in labels:
        if field in out:
            continue
        m = match(field, lc)
        if m:
            pn, sc, _ = m
            out[field] = {"value": pn["value"], "score": round(sc, 3),
                          "raw": pn["raw"], "t": frame.get("t")}
    return out


def watchlist_rows(frame: dict) -> List[dict]:
    """Best-effort 관심종목 read: [{name, code, price, change}] from the left panel. Name->code is
    FUZZY (display label only); the code is what we key on."""
    h, w = frame.get("h", 1080), frame.get("w", 1920)
    rows = []
    for row in frame.get("ocr", []):
        c = _center(row.get("bbox"))
        if c and c[0] < w * 0.30:
            rows.append((c[1], c[0], row))
    rows.sort(key=lambda r: (r[0], r[1]))            # by (y, x); never compare the row dict
    clusters: List[List] = []
    for cy, cx, row in rows:
        if clusters and abs(cy - clusters[-1][0][0]) < h * 0.015:
            clusters[-1].append((cy, cx, row))
        else:
            clusters.append([(cy, cx, row)])
    out = []
    for cl in clusters:
        cl.sort(key=lambda x: x[1])                       # left-to-right (HTS columns L->R)
        name_tok, nums = None, []                         # name, then numbers in COLUMN ORDER
        for _, _, row in cl:
            txt = row["text"]
            pn = parse_num(txt)
            if pn and not pn["is_pct"]:
                nums.append(pn["value"])
            elif name_tok is None and re.search(r"[가-힣A-Za-z]", txt) and not pn:
                name_tok = txt
        if not name_tok or not nums:
            continue
        fz = tickers.fuzzy_name_to_code(name_tok)
        # columns are 종목명 | 현재가 | 대비 | (거래량) -> leftmost number is 현재가 (NOT the max,
        # which would grab the volume column).
        price = nums[0]
        change = nums[1] if len(nums) > 1 else None
        out.append({"name": name_tok, "code": fz[0] if fz else None,
                    "name_match_score": fz[1] if fz else None,
                    "price": price, "change_abs": change})
    return [r for r in out if r["code"]]


def aggregate(frames: List[dict], primary_code: Optional[str]) -> dict:
    """Aggregate labeled fields for the primary ticker across frames + list all codes seen.

    Returns {primary_code, fields:{field:{value,score,n_frames,evidence[]}}, codes_seen:{code:n}}."""
    codes_seen = Counter()
    per_field_vals = defaultdict(list)                    # field -> [(value, score, t, frame)]
    axis_vals = []                                        # chart price-axis current-price box
    for fr in frames:
        for c in tickers.codes_in_text(" ".join(r.get("text", "") for r in fr.get("ocr", []))):
            codes_seen[c] += 1
        ap = fr.get("axis_price")                         # the chart shows the PRIMARY stock
        if ap:
            pn = parse_num(ap.get("text", ""))
            if pn and not pn["is_decimal"] and 50 <= pn["value"] <= 90_000_000:
                axis_vals.append((pn["value"], float(ap.get("score", 0)), fr.get("t"), fr.get("frame")))
        # labeled fields are constrained to the primary's row inside labeled_fields(primary_code)
        for field, info in labeled_fields(fr, primary_code).items():
            per_field_vals[field].append((info["value"], info["score"], fr.get("t"), fr.get("frame")))

    def _mode(vals):
        counts = Counter(v[0] for v in vals)
        best_val, _ = max(counts.items(), key=lambda kv: (kv[1], max(
            s for (v, s, _, _) in vals if v == kv[0])))
        ev = [{"t": t, "frame": fn, "score": s} for (v, s, t, fn) in vals if v == best_val]
        ev.sort(key=lambda e: -(e["score"] or 0))
        return {"value": best_val, "score": max(e["score"] for e in ev),
                "n_frames": counts[best_val], "candidates": dict(counts), "evidence": ev[:4]}

    fields = {field: _mode(vals) for field, vals in per_field_vals.items()}
    # The chart price-axis current-price box OCR-grounds the PRIMARY's 현재가 (it isn't in the peer
    # watchlist). It's the authoritative live price for the primary -> use it as the price field.
    if axis_vals:
        fields["price"] = {**_mode(axis_vals), "source": "chart_axis"}
    return {"primary_code": primary_code, "fields": fields, "codes_seen": dict(codes_seen)}
