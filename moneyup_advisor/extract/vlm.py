"""Qwen3-VL (Ollama) — PATTERN / CONTEXT only. NEVER a source of numbers.

Per Rule 1, the VLM describes what it SEES (chart pattern, trend, what the host points at /
circles, overall stance). It is explicitly told NOT to invent numbers — PaddleOCR owns every
number. Whatever the VLM says is treated as a VIDEO-ONLY observation downstream and is dropped /
flagged if it isn't corroborated by OCR or audio (e.g. the 호가 ladder it hallucinated in Step 0).
"""
from __future__ import annotations

import base64
import json
import time
import urllib.request
from typing import Dict, Optional

from moneyup_advisor import config

# Structured English schema the model MUST fill — works for an English chart OR a Korean HTS. It reads
# technical STRUCTURE (even on a clean line chart) and is forbidden from inventing any number (PaddleOCR
# owns numbers). "not legible" is allowed PER FIELD, but a clear chart must never be blanket-blanked.
_PROMPT = (
    "You are reading ONE frame from a stock/market video — a trading screen or a price chart. Report ONLY "
    "what is actually visible, as a structured JSON object. Describe the technical STRUCTURE even on a clean "
    "line chart. Read on-screen text/labels in whatever language they appear.\n"
    "CRITICAL — you are a VISUAL INTERPRETER, not a data source: do NOT invent any number. Give a numeric "
    "price level ONLY if it is clearly legible on the axis or a label; otherwise describe it qualitatively "
    "(e.g. 'below the 50-day MA'). PaddleOCR owns every number. Use \"not legible\" for any single field that "
    "isn't readable — but NEVER blank an entire frame that clearly shows a chart.\n"
    "Answer with JSON only:\n"
    '{"screen_type":"chart|trading_screen|table|talking_head|other",'
    '"instrument":"what is charted, exactly as labelled (e.g. NASDAQ Composite / ^IXIC / 삼성전자), or not legible",'
    '"timeframe":"1D|1W|1M|intraday|not legible",'
    '"trend_structure":"higher-highs/higher-lows | lower-highs/lower-lows | range/sideways | not legible",'
    '"moving_averages":"which MAs are visible, price above/below them, any recent cross/break, slope — or none visible",'
    '"key_levels":"support/resistance/round levels read OFF the axis; numeric ONLY if legible, else qualitative",'
    '"recent_event":"gap up/down | breakout/breakdown | test/bounce of a level or MA | none",'
    '"visible_price":"current/last price ONLY if shown on screen, else not legible",'
    '"stance":"bullish|bearish|neutral|unclear","confidence":"high|medium|low"}'
)
# values that mean "nothing here" — used downstream to know a field is empty (any language)
EMPTY_VALUES = {"", "안 보임", "not legible", "none", "none visible", "n/a", "unclear", "unknown"}


def describe_frame(image_path: str, keep_alive: str = "8m", timeout: int = 600,
                   retries: int = 2) -> Optional[Dict]:
    """Qwen3-VL structured chart read of one frame (visual interpretation only — never numbers). Retries on
    transient Ollama/GPU-load failures (the model can lose a cold-load race while Whisper holds the GPU)."""
    try:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
    except Exception:
        return None
    payload = {
        "model": config.QWEN_VL_MODEL,
        "messages": [{"role": "user", "content": _PROMPT, "images": [b64]}],
        "stream": False, "keep_alive": keep_alive,
        "format": "json",                              # ask Ollama to constrain to JSON
        "options": {"temperature": 0.1, "num_ctx": 8192},
    }
    body = json.dumps(payload).encode()
    for attempt in range(retries + 1):
        req = urllib.request.Request(
            f"{config.OLLAMA_HOST}/api/chat", data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read().decode())
            content = (data.get("message", {}) or {}).get("content", "")
            if not content:
                raise ValueError("empty content")
            try:
                parsed = json.loads(content)
            except Exception:
                parsed = {"note": str(content)[:300]}
            parsed["_raw"] = content
            parsed["_model"] = config.QWEN_VL_MODEL
            return parsed
        except Exception:
            if attempt < retries:
                time.sleep(8)                          # a cold 30B load takes ~24s — wait it out, then retry
                continue
            return None


def warmup(timeout: int = 120) -> bool:
    """Pre-load Qwen3-VL with a tiny text call so the FIRST real describe_frame doesn't lose a cold-load
    race (the model takes ~20-30s to load; a concurrent image+JSON generation during load 500s)."""
    payload = {"model": config.QWEN_VL_MODEL, "keep_alive": "8m", "stream": False,
               "messages": [{"role": "user", "content": "ok"}]}
    req = urllib.request.Request(f"{config.OLLAMA_HOST}/api/chat", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=timeout).read()
        return True
    except Exception:
        return False


def unload() -> None:
    """Free the GPU by asking Ollama to unload the model (keep_alive=0)."""
    payload = {"model": config.QWEN_VL_MODEL, "messages": [], "keep_alive": 0, "stream": False}
    req = urllib.request.Request(
        f"{config.OLLAMA_HOST}/api/chat", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=30).read()
    except Exception:
        pass
