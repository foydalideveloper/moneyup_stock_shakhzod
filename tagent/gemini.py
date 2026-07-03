"""Google Gemini (generativelanguage REST) — minimal client + grounded LLM hooks.

The LLM backend for (A) YouTube insight extraction and (B) Naver article-relevance judging. The
API key travels only in the ``x-goog-api-key`` header and is never logged; ``requests`` is lazy
and the session is injectable, so tests run fully mocked with no network.

Everything stays GROUNDED downstream: the YouTube hook is wrapped by
``youtube_source.llm_extract_fn`` which DROPS any insight whose quote isn't in the transcript, and
the Naver relevance hook only ever DROPS articles (never adds / rewrites), keeping the cited link.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Callable, List, Optional

_TRANSLATE_CACHE = {}            # (target, sha1(source)) -> translated; persists for instant re-runs

GENAI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_MODEL = "gemini-flash-latest"


class GeminiError(RuntimeError):
    """Raised on a transport / JSON failure talking to Gemini."""


class GeminiClient:
    """Thin generateContent client. ``session`` injectable for tests; key only in the header."""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, session=None, timeout: float = 120.0):
        self.api_key = api_key
        self.model = model or DEFAULT_MODEL
        self._session = session
        # The 2.5 "thinking" models can spend >30s on a full 30k-char transcript; a tight read
        # timeout used to raise mid-generation and the error was swallowed → 0 insights. Give them room.
        self.timeout = timeout

    def _post(self, body: dict) -> dict:
        sess = self._session
        if sess is None:
            import requests  # lazy
            sess = requests
        url = GENAI_URL.format(model=self.model)
        resp = sess.post(url, headers={"x-goog-api-key": self.api_key,
                                       "Content-Type": "application/json"}, json=body, timeout=self.timeout)
        try:
            return resp.json()
        except Exception as e:
            raise GeminiError(f"gemini non-JSON response: {e}")

    def generate(self, prompt: str, *, system: Optional[str] = None, temperature: float = 0.2,
                 json_out: bool = True, thinking_budget: Optional[int] = None) -> str:
        """Return the model's text for ``prompt`` ("" on a malformed response). ``thinking_budget``
        (e.g. 0) caps the 2.5 models' reasoning tokens — set 0 for mechanical tasks like translation
        so each call returns fast instead of paying fixed 'thinking' latency."""
        gen = {"temperature": temperature}
        if json_out:
            gen["responseMimeType"] = "application/json"
        if thinking_budget is not None:
            gen["thinkingConfig"] = {"thinkingBudget": thinking_budget}
        body = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": gen}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        data = self._post(body)
        try:
            parts = data["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts)
        except Exception:
            return ""


def parse_json_list(text: str) -> list:
    """Parse a JSON array from a model response, tolerating ```json fences / a wrapping object."""
    s = str(text or "").strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE).strip()
    try:
        v = json.loads(s)
    except Exception:
        m = re.search(r"\[.*\]", s, flags=re.DOTALL)
        if not m:
            return []
        try:
            v = json.loads(m.group(0))
        except Exception:
            return []
    if isinstance(v, list):
        return v
    if isinstance(v, dict):
        return v.get("items") or v.get("results") or []
    return []


# --------------------------------------------------------------------------- #
# (A) YouTube — full-transcript insight extraction (grounded downstream)
# --------------------------------------------------------------------------- #
_YT_SYSTEM = (
    "You are a Korean equity research analyst. Read the FULL timestamped transcript and extract, "
    "for EACH watchlist stock SUBSTANTIVELY discussed, EVERY actionable point the speaker actually "
    "makes about it — especially: 목표가/적정가 (target/fair price), 투자의견 (매수/매도/보유/비중확대/"
    "비중축소), 실적·수주·계약 (earnings / orders / deals), and concrete catalysts or the stated reason "
    "the stock moves. Do NOT stop at one stock and do NOT collapse multiple stocks into one, and NEVER "
    "emit a generic sector/theme bucket (조선, 정유·화학·항공 …) — always name the specific companies the "
    "speaker actually said. Use ONLY the provided transcript: every field must be supported by a VERBATIM "
    "quote copied from it. If you are unsure or it isn't stated, OMIT it — never invent prices, "
    "partnerships, ratings, or events.")


def youtube_prompt(transcript, watchlist) -> tuple:
    """The shared (system, user) YouTube-extraction prompt — used by BOTH the Gemini and OpenAI
    extractors so they return the SAME grounded schema and obey the SAME rules."""
    wl = ", ".join(f"{name}({code})" for code, name in (watchlist or {}).items())
    prompt = (
        f"WATCHLIST: {wl}\n\nTRANSCRIPT (timestamped):\n{str(transcript)[:30000]}\n\n"
        "Capture EVERY distinct point the speaker actually makes about the watchlist stocks. A "
        "10-15 min video should yield SEVERAL distinct insights (not one short line). Output a JSON "
        "array of objects (MULTIPLE objects per stock are expected when several points are made):\n"
        '{"stock": "<ticker code or name; if ONE point applies to several stocks named together, set '
        'this to all of them, e.g. \\"삼성전자와 SK하이닉스\\", and write ONE combined object — do NOT '
        'repeat the same sentence for each stock>", '
        '"summary": "<2-3 sentence Korean insight: (1) the claim, (2) the reasoning/context the '
        'speaker gave, (3) the implication — strictly from the transcript, add no facts beyond what '
        'is said>", '
        '"quote": "<a short quote copied VERBATIM from the transcript that supports it>", '
        '"action": "<매수|매도|보유|비중확대|비중축소|관심 — ONLY if the speaker gave a directional call, '
        'else omit>", '
        '"target_price": "<목표가/적정가 EXACTLY as stated, e.g. 9만원, else omit>", '
        '"category": "<M&A/deal|earnings|target-price|opinion|price-move reason — best fit>"}. '
        "Also include ANY other Korean stock the speaker substantively discusses (e.g. HPSP, 솔브레인, "
        "신세계) under its spoken name — not only the watchlist. "
        "Emit SPECIFIC NAMED companies — NEVER a generic sector/theme bucket. If the speaker discusses a "
        "sector by naming individual stocks in it (e.g. 조선: 삼성중공업·한화오션·현대중공업; 정유·화학·항공: "
        "the names actually said), emit ONE object PER NAMED stock — do NOT output a '조선' or "
        "'정유·화학·항공주' bucket. Only describe a theme in prose inside a named stock's summary; the "
        "\"stock\" field must always be real company name(s)/ticker(s), never a sector word. "
        "EVERY explicit 목표가/적정가 or 매수/매도 call is a MUST-CAPTURE insight in its own right (e.g. an "
        "'SK하이닉스 300만원' target must appear as an SK하이닉스 object with target_price set) — never drop "
        "or merge these into a sector line. An ASPIRATIONAL price level the speaker says a stock can "
        "reach / aim for (도전/돌파/노린다/갈 수 있다/간다 + 금액, e.g. '하이닉스 300만 원 도전') IS a "
        "target_price for THAT stock — capture it. When ONE sentence assigns DIFFERENT price levels to "
        "DIFFERENT named stocks (e.g. '삼성전자 … 하이닉스 … 40만 원 … 300만 원'), emit a SEPARATE object per "
        "stock and pair them IN THE ORDER STATED — the FIRST-named stock takes the FIRST price, the second "
        "the second (삼성전자→40만 원, 하이닉스→300만 원); do NOT merge them or reverse the pairing. "
        "Auto-transcribed names may be phonetically GARBLED — map them to the real listed company in the "
        "\"stock\" field (e.g. 삼선전자→삼성전자, 하이스/3익닉스→SK하이닉스, 하나오션→한화오션) while keeping the "
        "\"quote\" copied VERBATIM (garbled spelling and all) so it stays grounded. "
        "Set action whenever a buy/sell/hold or a target/fair price is "
        "stated — those matter most; do NOT infer a target from the video title (only from what is said). "
        "Each summary must be DISTINCT (never duplicate text across objects). "
        "The quote MUST be copied verbatim from the transcript; if unsure, omit. Output JSON only.")
    return _YT_SYSTEM, prompt


def youtube_analyze_fn(client: GeminiClient, fallback_model: str = DEFAULT_MODEL) -> Callable[[str, dict], list]:
    """An ``analyze_fn(full_transcript, watchlist)`` for ``youtube_source.llm_extract_fn``. Returns
    one or more grounded objects PER substantively-discussed stock:
    ``{stock, summary, quote, action?, target_price?, category?}`` — capturing target prices, buy/sell
    calls, results and catalysts (each backed by a verbatim quote). If the primary (stronger) model
    returns nothing — e.g. it 404s — it retries once with ``fallback_model`` (gemini-flash-latest)."""
    def _gen(transcript: str, watchlist: dict) -> str:
        import os
        import sys
        dbg = os.getenv("YT_EXTRACT_DEBUG")
        system, prompt = youtube_prompt(transcript, watchlist)
        try:
            out = client.generate(prompt, system=system, temperature=0.1)
        except Exception as e:                                                   # timeout/network/404
            if dbg:
                print(f"[yt-extract] {client.model} RAISED {type(e).__name__}: {str(e)[:120]}",
                      file=sys.stderr)
            out = ""
        if dbg:
            print(f"[yt-extract] {client.model} raw len={len(out or '')} parsed="
                  f"{len(parse_json_list(out))} :: {str(out)[:300]!r}", file=sys.stderr)
        # Fall through to the fallback model on an empty/unparseable result OR a primary failure
        # (404, but also a timeout that raised above) — never let one slow call yield 0 insights.
        if not parse_json_list(out) and fallback_model and client.model != fallback_model:
            fb = GeminiClient(client.api_key, model=fallback_model, session=client._session,
                              timeout=client.timeout)
            try:
                out = fb.generate(prompt, system=system, temperature=0.1)        # retry on 404/empty/timeout
            except Exception as e:
                if dbg:
                    print(f"[yt-extract] fallback {fallback_model} RAISED {type(e).__name__}: "
                          f"{str(e)[:120]}", file=sys.stderr)
                out = out or ""
            if dbg:
                print(f"[yt-extract] fallback {fallback_model} raw len={len(out or '')} "
                      f"parsed={len(parse_json_list(out))}", file=sys.stderr)
        return out

    def _analyze(transcript: str, watchlist: dict) -> list:
        try:
            return parse_json_list(_gen(transcript, watchlist))
        except Exception:
            return []
    return _analyze


def youtube_extract_fn(client: GeminiClient, *, watchlist: Optional[dict] = None,
                       fallback_model: str = DEFAULT_MODEL):
    """Grounded YouTube ``extract_fn`` backed by Gemini (= llm_extract_fn ∘ youtube_analyze_fn)."""
    from tagent.news.youtube_source import llm_extract_fn
    return llm_extract_fn(youtube_analyze_fn(client, fallback_model=fallback_model), watchlist=watchlist)


def extract_client(settings, session=None, model: Optional[str] = None) -> Optional["GeminiClient"]:
    """A GeminiClient for extraction, or None if no key. ``model`` overrides the default (the
    OVERNIGHT GEMINI_EXTRACT_MODEL); interactive callers pass GEMINI_INTERACTIVE_MODEL. The analyze
    layer falls back to gemini-flash-latest if the chosen model 404s."""
    if not settings.has_gemini_key():
        return None
    m = model or getattr(settings, "gemini_extract_model", None) or settings.gemini_model
    return GeminiClient(settings.gemini_api_key, model=m, session=session)


def extractor_label(settings) -> str:
    """A short, secret-free label of the active extractor for logs/prints."""
    provider = (getattr(settings, "llm_provider", "") or "gemini").lower()
    if provider == "openai" and settings.has_openai_key():
        return f"OpenAI ({getattr(settings, 'openai_model', 'gpt-4o')})"
    if settings.has_gemini_key():
        model = getattr(settings, "gemini_extract_model", None) or settings.gemini_model
        return f"Gemini ({model} -> flash on 404)"
    return "grounded default (no LLM key)"


def build_extractor(settings, *, watchlist=None, model=None, session=None, openai_session=None):
    """The grounded YouTube extract_fn for the configured provider (LLM_PROVIDER=gemini|openai),
    or None when no key. ``model`` overrides the Gemini model (interactive paths pass
    GEMINI_INTERACTIVE_MODEL; batch/--reextract use the default GEMINI_EXTRACT_MODEL). OpenAI is
    opt-in; on a missing OPENAI_API_KEY or any construction error it GRACEFULLY falls back to Gemini.
    Both providers share the same prompt/schema and grounding."""
    provider = (getattr(settings, "llm_provider", "") or "gemini").lower()
    if provider == "openai" and settings.has_openai_key():
        try:
            from tagent.openai_llm import OpenAIClient, openai_extract_fn
            client = OpenAIClient(settings.openai_api_key,
                                  model=getattr(settings, "openai_model", "gpt-4o"),
                                  session=openai_session)
            return openai_extract_fn(client, watchlist=watchlist)
        except Exception:
            pass                                              # graceful fallback to Gemini
    ec = extract_client(settings, session=session, model=model)
    return youtube_extract_fn(ec, watchlist=watchlist) if ec is not None else None


# --------------------------------------------------------------------------- #
# (C) Translation — KO -> English for the bilingual report (prose only; no new facts)
# --------------------------------------------------------------------------- #
def translate_fn(client: GeminiClient, target: str = "English") -> Callable[[str], str]:
    """A text translator (``text -> translated``) for the report's bilingual output. Low temperature;
    translates PROSE ONLY and keeps numbers, percentages, URLs and ticker codes (e.g. 000660, NVDA)
    intact; never adds, removes, or invents facts. On any failure it returns the original text."""
    def _translate(text: str) -> str:
        s = str(text or "").strip()
        if not s:
            return ""
        prompt = (
            f"Translate the following Korean equity-research text to {target}. Translate the prose "
            "ONLY. Keep ALL numbers, percentages, URLs, ticker codes (e.g. 000660, NVDA) and proper "
            "nouns exactly as-is. Do NOT add, remove, or invent any facts, partnerships, prices, or "
            "events. Output only the translation.\n\n" + s)
        try:
            out = client.generate(prompt, temperature=0.0, json_out=False, thinking_budget=0)
            return str(out).strip() or s
        except Exception:
            return s
    return _translate


def _parse_marked(text: str, n: int):
    """Parse an LLM response delimited by ``@@i@@`` markers into n segments (robust to multi-line
    translations). Returns the list of n segments, or None if any index is missing (caller then
    falls back to PER-FIELD translation — never to the untranslated original)."""
    parts = re.split(r"@@\s*(\d+)\s*@@", str(text or ""))     # [pre, "0", seg0, "1", seg1, ...]
    out = {}
    for i in range(1, len(parts) - 1, 2):
        try:
            out[int(parts[i])] = parts[i + 1].strip()
        except ValueError:
            continue
    if all(k in out for k in range(n)):
        return [out[k] for k in range(n)]
    return None


def _sha(text: str) -> str:
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()


def translate_batch_fn(client: GeminiClient, target: str = "English",
                       chunk_size: int = 8) -> Callable[[list], list]:
    """A CHUNKED + CONCURRENT + CACHED batch translator ``[str] -> [str]`` for the whole report.
    Translates in small ``@@i@@``-delimited chunks (~``chunk_size`` fields each) run CONCURRENTLY in a
    thread pool — so N chunks finish in ~the time of one, not N× — and caches by source-text hash so
    re-runs are instant. A chunk whose markers don't parse falls back to per-field translation FOR THAT
    CHUNK ONLY (still English, never the Korean original). Numbers / % / URLs / tickers / timestamps
    stay intact."""
    one = translate_fn(client, target)                        # single-string translator for fallback

    def _chunk(items: list) -> list:
        marked = "\n".join(f"@@{i}@@ {t}" for i, t in enumerate(items))
        prompt = (
            f"Translate EACH segment below to {target}. Segments are delimited by line-start markers "
            "@@0@@, @@1@@, … . Translate prose ONLY; keep ALL numbers, percentages, URLs, ticker codes "
            "(e.g. 000660, NVDA), timestamps and proper nouns exactly as-is; do NOT add, remove, or "
            "invent any facts, prices, or events. Output the SAME @@i@@ markers, each followed by ONLY "
            "that segment's translation.\n\n" + marked)
        try:
            parsed = _parse_marked(client.generate(prompt, temperature=0.0, json_out=False,
                                                   thinking_budget=0), len(items))
        except Exception:
            parsed = None
        return parsed if parsed is not None else [one(t) for t in items]   # per-chunk per-field fallback

    def _translate(texts: list) -> list:
        items = [str(x or "") for x in (texts or [])]
        out = [None] * len(items)
        todo_i, todo_t = [], []
        for i, t in enumerate(items):
            if not t.strip():
                out[i] = ""
            else:
                hit = _TRANSLATE_CACHE.get((target, _sha(t)))
                if hit is not None:
                    out[i] = hit                              # instant on re-runs
                else:
                    todo_i.append(i)
                    todo_t.append(t)
        cs = max(1, chunk_size)
        chunks = [(s, todo_t[s:s + cs]) for s in range(0, len(todo_t), cs)]
        if len(chunks) <= 1:                                   # 0/1 chunk -> no pool overhead
            results = [_chunk(c) for _s, c in chunks]
        else:
            # The chunk calls are independent HTTP requests; run them CONCURRENTLY so ~N chunks finish
            # in roughly the time of one (EN was ~6x slower running them sequentially). ex.map preserves
            # order; the cache is written below in the main thread (no concurrent dict writes).
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(12, len(chunks))) as pool:
                results = list(pool.map(lambda sc: _chunk(sc[1]), chunks))
        for (s, _chunk_items), vals in zip(chunks, results):
            for j, val in enumerate(vals):
                gi = todo_i[s + j]
                out[gi] = val
                _TRANSLATE_CACHE[(target, _sha(items[gi]))] = val
        return out
    return _translate


# --------------------------------------------------------------------------- #
# (B) Naver — per-article relevance judging (only ever DROPS; keeps the link)
# --------------------------------------------------------------------------- #
def naver_relevance_fn(client: GeminiClient) -> Callable[[list, str], list]:
    """A batch relevance filter ``(items, name) -> kept_items`` for ``NaverNewsSource``. Gemini
    judges which articles are SUBSTANTIVELY about the tagged stock's business/price/products/deals;
    incidental mentions, promos and same-name articles are dropped. On any failure it falls back to
    the strict 'name in the title (subject)' rule — never inventing or rewriting an item."""
    def _filter(items: list, name: str) -> list:
        items = list(items or [])
        if not items:
            return []
        listing = "\n".join(f"{i}. {a.get('title', '')} — {str(a.get('summary', ''))[:80]}"
                            for i, a in enumerate(items))
        prompt = (
            f"종목: {name}\n\n기사 목록:\n{listing}\n\n"
            f"각 기사가 '{name}'의 사업/실적/주가/제품/계약/규제 등 본질에 대해 실질적으로 다루는지 판단하라. "
            "단순 언급, 광고/프로모션, 동명이인·유사명, 게임/스포츠/행사 등 곁가지 기사는 제외한다. "
            "실질적으로 관련된 기사의 인덱스만 JSON 배열로 출력하라 (예: [0, 2]). JSON만 출력.")
        try:
            idxs = parse_json_list(client.generate(prompt))
            keep = {int(x) for x in idxs if str(x).lstrip("-").isdigit()}
            return [a for i, a in enumerate(items) if i in keep]
        except Exception:
            nm = str(name or "")
            return [a for a in items if nm and nm in str(a.get("title", ""))]   # strict fallback
    return _filter
