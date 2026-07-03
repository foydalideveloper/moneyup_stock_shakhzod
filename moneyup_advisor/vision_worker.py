"""Vision worker — RUNS IN THE ISOLATED ``moneyup`` CONDA ENV (paddleocr + opencv).

Invoked by the base-env orchestrator:
    <moneyup-python> -m moneyup_advisor.vision_worker <frames_dir> --out <result.json>

FULL COVERAGE: OCRs the WHOLE screen of every DISTINCT frame (perceptual-hash dedupe of the
ffmpeg-sampled frames) — watchlist + chart title bar (primary code) + chart price axis + 호가 ladder
+ annotations. Also:
  * axis_price: the current-price box on the chart price axis (a red/blue filled cell) -> OCR-grounds
    the primary's 현재가 even though the primary isn't in the peer watchlist.
  * chart: OpenCV candle geometry (descriptive trend only, never a price).

GPU-aware: uses paddlepaddle-gpu when present (RTX 5090 / cu129), else CPU. Reports device + sec/frame.
PaddleOCR remains the sole number source.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from moneyup_advisor import config
    _PHASH_HAM = config.PHASH_HAMMING
    _OCR_W = config.OCR_WIDTH_FRAC
except Exception:
    _PHASH_HAM, _OCR_W = 6, 1.0
# RAM safety: bound the number of distinct frames OCR'd per video so one pathological (multi-hour) upload
# can't accumulate PaddleOCR memory unbounded. Normal 머니업 videos are ~400-500 frames, far below this.
_MAX_OCR_FRAMES = int(os.getenv("MONEYUP_MAX_OCR_FRAMES", "4000"))

_MS_RE = re.compile(r"_(\d+)ms")
_NUMTOK_RE = re.compile(r"^[+\-▲▼]?\s*[\d,]+(?:\.\d+)?%?$")


def _gpu_available():
    try:
        import paddle
        return bool(paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0)
    except Exception:
        return False


def _build_ocr():
    """(PaddleOCR, device_str). GPU when available (server detector, more accurate + fast on GPU),
    else CPU mobile detector with mkldnn off (Paddle 3.x CPU/PIR+oneDNN crash workaround)."""
    from paddleocr import PaddleOCR
    gpu = _gpu_available()
    rec = "korean_PP-OCRv5_mobile_rec"
    if gpu:
        base = dict(use_doc_orientation_classify=False, use_doc_unwarping=False,
                    use_textline_orientation=False, text_recognition_model_name=rec,
                    text_detection_model_name="PP-OCRv5_server_det")
        for kw in (dict(device="gpu", **base), base):
            try:
                return PaddleOCR(**kw), "gpu"
            except TypeError:
                continue
    base = dict(use_doc_orientation_classify=False, use_doc_unwarping=False,
                use_textline_orientation=False, text_recognition_model_name=rec,
                text_detection_model_name="PP-OCRv5_mobile_det", enable_mkldnn=False)
    try:
        return PaddleOCR(**base), "cpu"
    except TypeError:
        return PaddleOCR(lang="korean", enable_mkldnn=False), "cpu"


def _ocr_rows(ocr, img):
    rows = []
    res = ocr.predict(img) if hasattr(ocr, "predict") else None
    if res:
        for page in res:
            d = page if isinstance(page, dict) else getattr(page, "json", {}).get("res", {})
            texts = d.get("rec_texts") or []
            scores = d.get("rec_scores") or []
            polys = d.get("rec_polys") or d.get("dt_polys") or []
            for i, (t, s) in enumerate(zip(texts, scores)):
                bbox = None
                if i < len(polys):
                    pts = np.array(polys[i]).reshape(-1, 2)
                    bbox = [int(pts[:, 0].min()), int(pts[:, 1].min()),
                            int(pts[:, 0].max()), int(pts[:, 1].max())]
                rows.append({"text": str(t), "score": round(float(s), 3), "bbox": bbox})
    return rows


def _ahash(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = cv2.resize(g, (16, 16), interpolation=cv2.INTER_AREA).astype(np.int16)
    return (s > s.mean()).flatten()


_OHLC_RE = re.compile(r"([OHLC])\s*([0-9][0-9,\s]*\.?[0-9]*)")
_PCT_RE = re.compile(r"([+\-–]?[0-9][0-9,.\s]*)\s*%")
_SYM_RE = re.compile(r"^[A-Z]{2,8}(USD|USDT|KRW)?$")
_SYM_STOP = {"INVES", "INVESTING", "INDICATORS", "ALERT", "ALERTS", "REPLAY", "TRADE", "PUBLISH",
             "UNNAMED", "USD", "VOL", "USDT", "OHLC"}


def _num_ws(s):
    """Parse a numeric string that PaddleOCR may have split with spaces/commas ('60, 162.73' -> 60162.73)."""
    s = str(s).replace(" ", "").replace(",", "").replace("–", "-").rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def _axis_numeric(img, rows):
    """Far-right price-axis numeric candidates within the dominant-magnitude cluster (drops 999.9 garbage).
    Returns [(value, y_center, text, score)] — shared by the SCALE-tick and last-price-label readers."""
    h, w = img.shape[:2]
    cands = []
    for r in rows:
        b = r.get("bbox")
        if not b:
            continue
        if (b[0] + b[2]) / 2.0 < w * 0.80:               # right-side axis region only
            continue
        v = _axis_value(r.get("text", ""))
        if v is None:
            continue
        cands.append((v, (b[1] + b[3]) / 2.0, r.get("text", "").strip(), float(r.get("score") or 0)))
    if len(cands) < 3:
        return []
    vals = sorted(c[0] for c in cands)
    med = vals[len(vals) // 2]
    if med <= 0:
        return []
    return [c for c in cands if 0.4 * med <= c[0] <= 2.5 * med]   # dominant-magnitude cluster


def _axis_price(img, rows):
    """The on-screen LAST-PRICE label on the right axis. First choice: the numeric token whose cell has a
    saturated red/blue highlight box. Fallback: the precise OFF-GRID axis value (e.g. 58,486.35 sitting off
    the round 60,000/62,000 scale ladder) — that off-grid number IS the last-price, not a scale tick.
    Returns {text, value, color|source, score, bbox?} or None."""
    h, w = img.shape[:2]
    best, best_cx = None, -1
    for r in rows:
        b = r.get("bbox")
        if not b or not _NUMTOK_RE.match(r.get("text", "").strip()):
            continue
        cx = (b[0] + b[2]) / 2.0
        if cx < w * 0.84:
            continue
        x0, x1 = max(0, b[0] - 5), min(w, b[2] + 5)
        y0, y1 = max(0, b[1] - 3), min(h, b[3] + 3)
        patch = img[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        B, G, R = patch[:, :, 0].astype(int), patch[:, :, 1].astype(int), patch[:, :, 2].astype(int)
        red = float(((R > 120) & (G < 110) & (B < 110)).mean())
        blue = float(((B > 120) & (R < 110) & (G < 160)).mean())
        color = "red" if red > 0.25 else "blue" if blue > 0.25 else None
        if color and cx > best_cx:
            best, best_cx = {"text": r["text"], "value": _axis_value(r["text"]),
                             "color": color, "source": "color-box", "score": r["score"], "bbox": b}, cx
    if best is not None:
        return best
    # no highlighted box -> the OFF-GRID axis value is the last-price label, but ONLY when the right axis is
    # a legitimate round price SCALE (>=4 validated ticks). A Korean HTS right-panel is not a price ladder,
    # so this returns None there (no bogus last-price from stray HTS numbers).
    scale = _axis_levels(img, rows)
    if len(scale) < 4:
        return None
    svals = sorted(l["value"] for l in scale)
    diffs = sorted(svals[i + 1] - svals[i] for i in range(len(svals) - 1))
    step = diffs[len(diffs) // 2] if diffs else 0
    if step <= 0:
        return None
    off = [c for c in _axis_numeric(img, rows)
           if svals[0] - step <= c[0] <= svals[-1] + step        # within the scale's price range
           and min(abs(c[0] - sv) for sv in svals) > 0.10 * step]  # and off every scale tick
    if not off:
        return None
    off.sort(key=lambda c: -c[3])                        # most-confident off-grid value = last-price
    v, y, t, s = off[0]
    return {"text": t, "value": v, "color": None, "source": "off-grid-axis", "score": round(s, 3)}


def _price_header(img, rows):
    """TradingView-style OHLC header (top-left) + on-screen symbol/timeframe. The header's C (close) is the
    EXACT current price. Returns {symbol, timeframe, o, h, l, c, change_pct} or None. Header may be OCR'd as
    one token: 'O60,162.73 H60,191.99 L58,056.00 C58,523.93 -1,638.80 (-2.72%) Vol9.56K'."""
    h, w = img.shape[:2]
    close = o = hi = lo = chg = tf = None
    for r in rows:
        b = r.get("bbox")
        if not b or (b[1] + b[3]) / 2.0 > 0.16 * h:      # top band only
            continue
        txt = r.get("text", "")
        if "C" in txt and "O" in txt and re.search(r"[OC]\s*[0-9]", txt):
            d = {k: _num_ws(v) for k, v in _OHLC_RE.findall(txt)}
            if d.get("C"):
                close, o, hi, lo = d.get("C"), d.get("O"), d.get("H"), d.get("L")
                m = _PCT_RE.search(txt)
                if m:
                    chg = _num_ws(m.group(1))
        if tf is None:
            mt = re.search(r"\b(1D|1W|1M|4H|1H|15m|5m|1m|30m|일봉|주봉|월봉)\b", txt)
            if mt:
                tf = mt.group(1)
    # on-screen symbol: leftmost plausible ticker token in the top band (USD-pairs preferred)
    sym_c = []
    for r in rows:
        b = r.get("bbox")
        if not b or (b[1] + b[3]) / 2.0 > 0.14 * h or (b[0] + b[2]) / 2.0 > 0.40 * w:
            continue
        s = r.get("text", "").strip().upper()
        if s in _SYM_STOP or not _SYM_RE.fullmatch(s):
            continue
        score = (2 if s.endswith(("USD", "USDT", "KRW")) else 0) + (1 if 3 <= len(s) <= 6 else 0)
        sym_c.append((-score, b[0], s))
    sym_c.sort()
    symbol = sym_c[0][2] if sym_c else None
    if close is None:                                    # require a real OHLC header C (the defining TradingView
        return None                                      # signal); a Korean HTS has none -> no false price_field
    return {"symbol": symbol, "timeframe": tf, "o": o, "h": hi, "l": lo, "c": close, "change_pct": chg}


def _axis_value(t: str):
    """Parse an axis label to a positive float (commas stripped). None if not a plain number."""
    s = t.replace(",", "").replace("%", "").strip()
    if not re.fullmatch(r"[+\-]?\d+(?:\.\d+)?", s):
        return None
    try:
        v = abs(float(s))
    except Exception:
        return None
    return v if v > 0 else None


def _axis_levels(img, rows):
    """Right price-axis SCALE ticks: the round, evenly-spaced gridline labels (60,000 / 62,000 / 64,000…).
    These are the axis SCALE — NOT price levels (do not promote to key support/resistance). A precise
    OFF-GRID axis value is the last-price label, handled by _axis_price and dropped here. Plausibility
    filter (dominant magnitude cluster, monotonic ladder, round-grid) drops 999.9 garbage. Returns round
    scale ticks [{value, text, score}] high->low, or [] when the axis isn't a legible even scale."""
    cands = _axis_numeric(img, rows)
    if len(cands) < 3:
        return []
    cands.sort(key=lambda c: c[1])                       # top -> bottom
    seen, ladder = set(), []
    for v, y, t, s in cands:
        rk = round(v)
        if rk in seen:
            continue
        seen.add(rk)
        ladder.append({"value": v, "text": t, "y": round(y), "score": round(s, 3)})
    if len(ladder) < 3:
        return []
    vseq = [l["value"] for l in ladder]
    desc = sum(vseq[i] >= vseq[i + 1] for i in range(len(vseq) - 1))
    if desc < 0.6 * (len(vseq) - 1):                     # not a clean ladder -> not a legible axis
        return []
    # keep ONLY round grid ticks (median step); drop the off-grid last-price label rung
    svals = sorted(l["value"] for l in ladder)
    diffs = sorted(svals[i + 1] - svals[i] for i in range(len(svals) - 1))
    step = diffs[len(diffs) // 2] if diffs else 0
    if step <= 0:
        return []
    ticks = [l for l in ladder if abs(l["value"] - round(l["value"] / step) * step) < 0.10 * step]
    if len(ticks) < 3:
        return []
    return [{"value": l["value"], "text": l["text"], "score": l["score"]} for l in ticks]


def _chart_geometry(img):
    h, w = img.shape[:2]
    x0, x1 = int(w * 0.40), w
    y0, y1 = int(h * 0.10), int(h * 0.85)
    roi = img[y0:y1, x0:x1]
    if roi.size == 0:
        return {}
    b, g, r = roi[:, :, 0].astype(int), roi[:, :, 1].astype(int), roi[:, :, 2].astype(int)
    red = ((r > 130) & (g < 110) & (b < 110)).astype(np.uint8)
    blue = ((b > 130) & (r < 110) & (g < 160)).astype(np.uint8)
    candle = (red | blue)
    rh, rw = candle.shape
    cols_red = int((red.sum(axis=0) > 3).sum())
    cols_blue = int((blue.sum(axis=0) > 3).sum())
    if candle.sum() < 50:
        return {"has_chart": False, "red_cols": cols_red, "blue_cols": cols_blue}
    ys = np.arange(rh).reshape(-1, 1)
    col_mass = candle.sum(axis=0)
    valid = col_mass > 3
    cx = np.where(valid)[0]
    cy = (candle * ys).sum(axis=0)[valid] / np.maximum(col_mass[valid], 1)
    trend_dir, slope_norm = "flat", 0.0
    if len(cx) >= 8:
        slope = float(np.polyfit(cx, cy, 1)[0])
        slope_norm = round(-slope / rh * len(cx), 4)
        trend_dir = "up" if slope < -0.05 else "down" if slope > 0.05 else "flat"
    occupied = np.where(candle.any(axis=1))[0]
    swing = {"high_frac": round(float(occupied.min()) / rh, 3),
             "low_frac": round(float(occupied.max()) / rh, 3)} if len(occupied) else {}
    return {"has_chart": True, "trend_dir": trend_dir, "trend_strength": slope_norm,
            "red_cols": cols_red, "blue_cols": cols_blue, "swing": swing}


def process_dir(frames_dir: str) -> dict:
    ocr, device = _build_ocr()
    out = {"device": device, "frames": []}
    last_hash = None
    n_seen = n_dup = 0
    t_ocr = 0.0
    crop = _OCR_W < 0.999
    capped = False
    for p in sorted(Path(frames_dir).glob("frame_*.png"), key=lambda x: int(_MS_RE.search(x.name).group(1))
                    if _MS_RE.search(x.name) else 0):
        if len(out["frames"]) >= _MAX_OCR_FRAMES:        # RAM safety valve: bound a pathological long video
            capped = True
            break
        img = cv2.imread(str(p))
        if img is None:
            continue
        n_seen += 1
        hsh = _ahash(img)
        if last_hash is not None and int(np.count_nonzero(hsh != last_hash)) <= _PHASH_HAM:
            n_dup += 1                                   # perceptual-hash dedupe: skip near-identical
            del img
            continue
        last_hash = hsh
        m = _MS_RE.search(p.name)
        t = int(m.group(1)) / 1000.0 if m else 0.0
        h, w = img.shape[:2]
        ocr_img = img[:, :int(w * _OCR_W)] if crop else img      # whole screen by default
        t0 = time.time()
        rows = _ocr_rows(ocr, ocr_img)
        t_ocr += time.time() - t0
        out["frames"].append({
            "frame": p.name, "t": round(t, 2), "source": "frame", "w": w, "h": h,
            "ocr": rows, "chart": _chart_geometry(img), "axis_price": _axis_price(img, rows),
            "axis_levels": _axis_levels(img, rows), "price_field": _price_header(img, rows),
        })
        del img, ocr_img                                 # free the decoded frame(s) immediately
        if n_seen % 200 == 0:                            # periodic GC mitigates per-predict accumulation
            gc.collect()
    n = len(out["frames"])
    out["stats"] = {"sampled": n_seen, "deduped_skipped": n_dup, "ocr_frames": n, "capped": capped,
                    "ocr_seconds": round(t_ocr, 1),
                    "sec_per_frame": round(t_ocr / n, 2) if n else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames_dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    result = process_dir(a.frames_dir)
    Path(a.out).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    st = result["stats"]
    print(f"vision_worker[{result['device']}]: sampled={st['sampled']} "
          f"deduped={st['deduped_skipped']} ocr_frames={st['ocr_frames']} "
          f"{st['sec_per_frame']}s/frame total={st['ocr_seconds']}s -> {a.out}")


if __name__ == "__main__":
    main()
