"""Gemini LLM backend — client, YouTube extract (grounded), Naver relevance. Mocked, no network."""

import json
import re

import tagent.gemini as _gem
from tagent.gemini import (
    GeminiClient, extract_client, naver_relevance_fn, parse_json_list, translate_batch_fn,
    youtube_analyze_fn, youtube_extract_fn,
)


class _EchoSess:
    """Translate by echoing each @@i@@ segment as 'EN:<seg>'; a single (no-marker) prompt -> 'EN1'."""
    def __init__(self): self.calls = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append(json)
        prompt = json["contents"][0]["parts"][0]["text"]
        segs = re.findall(r"(?m)^@@(\d+)@@ (.*)$", prompt)        # only the marked block lines
        body = "\n".join(f"@@{i}@@ EN:{t}" for i, t in segs) if segs else "EN1"
        return _Resp(_gen(body))


def test_translate_batch_fn_one_call_marked():
    _gem._TRANSLATE_CACHE.clear()
    sess = _Sess(_gen("@@0@@ A-en\n@@1@@ B-en\n@@2@@ C-en"))
    tr = translate_batch_fn(GeminiClient("K", session=sess))
    assert tr(["x1", "x2", "x3"]) == ["A-en", "B-en", "C-en"]      # marker-delimited segments parsed in order
    assert len(sess.calls) == 1                                    # ALL prose in ONE call (EN not ~2x KO)
    assert tr([]) == []


def test_translate_batch_fn_chunks_not_per_field():
    _gem._TRANSLATE_CACHE.clear()
    sess = _EchoSess()
    tr = translate_batch_fn(GeminiClient("K", session=sess), chunk_size=10)
    out = tr([f"문장{i}" for i in range(25)])                       # 25 fields -> 3 chunk calls (10/10/5)
    assert out == [f"EN:문장{i}" for i in range(25)] and all("EN:" in o for o in out)
    assert len(sess.calls) == 3                                    # chunked, NOT 25 per-field calls


def test_translate_batch_fn_caches_repeats():
    _gem._TRANSLATE_CACHE.clear()
    sess = _EchoSess()
    tr = translate_batch_fn(GeminiClient("K", session=sess), chunk_size=10)
    tr(["가", "나"])
    n = len(sess.calls)
    assert tr(["가", "나"]) == ["EN:가", "EN:나"] and len(sess.calls) == n   # cached -> NO new calls


def test_translate_batch_fn_runs_chunks_concurrently():
    import threading
    import time
    _gem._TRANSLATE_CACHE.clear()
    state = {"inflight": 0, "max": 0}
    lock = threading.Lock()

    class _SlowSess:
        def post(self, url, headers=None, json=None, timeout=None):
            with lock:
                state["inflight"] += 1
                state["max"] = max(state["max"], state["inflight"])
            time.sleep(0.2)                                   # simulate a slow API call
            with lock:
                state["inflight"] -= 1
            prompt = json["contents"][0]["parts"][0]["text"]
            segs = re.findall(r"(?m)^@@(\d+)@@ (.*)$", prompt)
            return _Resp(_gen("\n".join(f"@@{i}@@ EN:{t}" for i, t in segs)))

    tr = translate_batch_fn(GeminiClient("K", session=_SlowSess()), chunk_size=2)
    t0 = time.time()
    out = tr([f"문장{i}" for i in range(8)])                   # 8 fields / 2 = 4 chunks
    dt = time.time() - t0
    assert out == [f"EN:문장{i}" for i in range(8)]            # order preserved, all translated
    assert state["max"] >= 2                                  # chunks overlapped -> concurrent, not serial
    assert dt < 0.6                                           # serial would be 4 x 0.2s = 0.8s


def test_translate_disables_thinking_for_speed():
    _gem._TRANSLATE_CACHE.clear()
    sess = _Sess(_gen("@@0@@ A-en"))
    translate_batch_fn(GeminiClient("K", session=sess))(["가"])
    gc = sess.calls[0]["json"]["generationConfig"]
    assert gc.get("thinkingConfig") == {"thinkingBudget": 0}   # translation = mechanical -> no thinking


def test_translate_batch_fn_per_field_fallback_is_english_not_korean():
    _gem._TRANSLATE_CACHE.clear()
    # batch chunk unparseable -> PER-FIELD translation (English), never the Korean original
    seq = _SeqSess([_gen("garbled, no markers"),                  # 1) batch chunk: unparseable
                    _gen("english-A"), _gen("english-B")])        # 2-3) per-field translations
    out = translate_batch_fn(GeminiClient("K", session=seq))(["가", "나"])
    assert out == ["english-A", "english-B"]                      # English (not 가/나)
    assert len(seq.calls) == 3                                    # 1 failed chunk + 2 per-field calls


def test_extract_client_model_override():
    class _S:
        gemini_api_key = "K"; gemini_model = "gemini-flash-latest"; gemini_extract_model = "gemini-2.5-pro"
        def has_gemini_key(self): return True
    assert extract_client(_S()).model == "gemini-2.5-pro"            # default = batch extract model
    assert extract_client(_S(), model="gemini-2.5-flash").model == "gemini-2.5-flash"   # interactive override


class _Resp:
    def __init__(self, data): self._data = data
    def json(self): return self._data


class _Sess:
    def __init__(self, data): self.data, self.calls = data, []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json})
        return _Resp(self.data)


def _gen(text):
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


# --------------------------------------------------------------------------- #
# client + JSON parsing (key in header, never in URL)
# --------------------------------------------------------------------------- #
def test_gemini_generate_keeps_key_in_header_not_url():
    sess = _Sess(_gen("hello"))
    out = GeminiClient("SECRET", "gemini-2.0-flash", session=sess).generate("hi", json_out=False)
    assert out == "hello"
    h = sess.calls[0]["headers"]
    assert h["x-goog-api-key"] == "SECRET" and "SECRET" not in sess.calls[0]["url"]


def test_parse_json_list_tolerates_fences_and_garbage():
    assert parse_json_list('```json\n[{"a":1}]\n```') == [{"a": 1}]
    assert parse_json_list("[1, 2, 3]") == [1, 2, 3]
    assert parse_json_list('{"items":[9]}') == [9]
    assert parse_json_list("no json here") == []


# --------------------------------------------------------------------------- #
# (A) YouTube — full-transcript analyze + grounded extract (drops hallucinations)
# --------------------------------------------------------------------------- #
class _SeqSess:
    """Returns a different payload per successive POST (to simulate a 404/empty then a fallback)."""
    def __init__(self, payloads): self.payloads, self.calls = list(payloads), []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json})
        p = self.payloads[min(len(self.calls) - 1, len(self.payloads) - 1)]
        return _Resp(p)


def test_youtube_prompt_forbids_sector_buckets_and_requires_named_stocks():
    from tagent.gemini import youtube_prompt
    system, prompt = youtube_prompt("거래소 전사본", {"000660": "SK하이닉스"})
    text = system + prompt
    assert "SPECIFIC NAMED companies" in prompt                # name companies, not sectors
    assert "bucket" in prompt and "한화오션" in prompt          # explicit named-stock guidance
    assert "조선" in text and ("never" in text or "NEVER" in text)
    assert "MUST-CAPTURE" in prompt and "300만원" in prompt     # explicit target price must be captured


def test_youtube_analyze_falls_back_to_flash_when_primary_404s():
    insights = [{"stock": "000660", "summary": "HBM", "quote": "SK하이닉스 공급계약"}]
    # 1st call (pro) -> 404-style error JSON (no candidates -> empty); 2nd (flash) -> insights
    sess = _SeqSess([{"error": {"code": 404, "message": "model not found"}}, _gen(json.dumps(insights, ensure_ascii=False))])
    out = youtube_analyze_fn(GeminiClient("K", "gemini-2.5-pro", session=sess))("T", {"000660": "SK하이닉스"})
    assert out and out[0]["stock"] == "000660"                 # recovered via the flash fallback
    assert len(sess.calls) == 2                                # primary then fallback
    assert "gemini-2.5-pro" in sess.calls[0]["url"] and "gemini-flash-latest" in sess.calls[1]["url"]


def test_youtube_analyze_falls_back_when_primary_times_out():
    # Regression: a slow "thinking" model on a full 30k-char transcript used to exceed the read
    # timeout, raise mid-generation, and get swallowed -> 0 insights (empty report). Now a primary
    # FAILURE (not just a 404/empty) must fall through to the fallback model instead of dropping all.
    insights = [{"stock": "000660", "summary": "HBM", "quote": "SK하이닉스 공급계약"}]
    payload = _gen(json.dumps(insights, ensure_ascii=False))

    class _TimeoutThenOk:
        def __init__(self): self.calls = []
        def post(self, url, headers=None, json=None, timeout=None):
            self.calls.append(url)
            if "gemini-flash-latest" in url:                       # fallback model -> succeeds
                return _Resp(payload)
            raise TimeoutError("read timed out")                   # primary model -> times out

    sess = _TimeoutThenOk()
    out = youtube_analyze_fn(GeminiClient("K", "gemini-2.5-flash", session=sess))("T", {"000660": "SK하이닉스"})
    assert out and out[0]["stock"] == "000660"                     # recovered despite the primary timeout
    assert len(sess.calls) == 2 and "gemini-flash-latest" in sess.calls[1]


def test_gemini_client_default_timeout_is_generous():
    # The tight 30s read timeout was the silent-failure trigger; the client now allows the thinking
    # models room to finish on a long transcript.
    assert GeminiClient("K").timeout >= 90


def test_youtube_analyze_no_fallback_when_primary_succeeds():
    insights = [{"stock": "000660", "summary": "HBM", "quote": "SK하이닉스 공급계약"}]
    sess = _SeqSess([_gen(json.dumps(insights, ensure_ascii=False))])
    youtube_analyze_fn(GeminiClient("K", "gemini-2.5-pro", session=sess))("T", {"000660": "SK하이닉스"})
    assert len(sess.calls) == 1                                # primary succeeded -> no retry


def test_extract_client_uses_extract_model_and_handles_no_key():
    class _S:
        gemini_api_key = "K"; gemini_model = "gemini-flash-latest"; gemini_extract_model = "gemini-2.5-pro"
        def has_gemini_key(self): return True
    assert extract_client(_S()).model == "gemini-2.5-pro"

    class _S0:
        gemini_api_key = ""; gemini_model = "gemini-flash-latest"; gemini_extract_model = "gemini-2.5-pro"
        def has_gemini_key(self): return False
    assert extract_client(_S0()) is None


def test_youtube_analyze_fn_sends_transcript_and_watchlist():
    insights = [{"stock": "000660", "summary": "HBM 공급계약", "quote": "SK하이닉스 공급계약"}]
    sess = _Sess(_gen(json.dumps(insights, ensure_ascii=False)))
    out = youtube_analyze_fn(GeminiClient("K", session=sess))("FULL TRANSCRIPT HERE", {"000660": "SK하이닉스"})
    assert out[0]["stock"] == "000660"
    sent = sess.calls[0]["json"]["contents"][0]["parts"][0]["text"]
    assert "SK하이닉스" in sent and "FULL TRANSCRIPT HERE" in sent       # full context + watchlist in prompt


def test_gemini_youtube_extract_grounds_and_drops_hallucination():
    segs = [{"text": "오늘 시황입니다", "start": 0.0},
            {"text": "SK하이닉스가 엔비디아와 HBM 공급계약을 체결했습니다", "start": 120.0}]
    insights = [
        {"stock": "000660", "summary": "엔비디아와 HBM 공급계약 — 메모리 업황 개선 신호.",
         "quote": "SK하이닉스가 엔비디아와 HBM 공급계약을 체결"},
        {"stock": "005930", "summary": "삼성전자가 화성에 우주기지를 짓는다.",
         "quote": "삼성전자가 화성에 우주기지를 건설한다"},                # NOT in transcript
    ]
    sess = _Sess(_gen(json.dumps(insights, ensure_ascii=False)))
    ex = youtube_extract_fn(GeminiClient("K", session=sess))
    out = ex(segs, {"video_id": "v", "title": "t", "published_at": "2026-06-12T09:00:00Z"}, "한경TV")
    assert [o["stock"] for o in out] == ["000660"]                     # hallucinated 005930 dropped
    assert out[0]["start"] == 120 and out[0]["summary"] and out[0]["deeplink"].endswith("&t=120s")


# --------------------------------------------------------------------------- #
# (B) Naver relevance — Gemini judge + strict title-subject fallback
# --------------------------------------------------------------------------- #
def test_naver_relevance_fn_keeps_substantive_and_falls_back_on_failure():
    arts = [{"title": "삼성전자 HBM 신규 라인 가동", "summary": ""},
            {"title": "노타, AI 어워드 수상", "summary": "삼성 출신 창업가"},     # incidental -> drop
            {"title": "FC온라인 새 시즌 개막", "summary": "네이버 검색 1위"}]      # passing name -> drop
    rel = naver_relevance_fn(GeminiClient("K", session=_Sess(_gen("[0]"))))
    assert [a["title"] for a in rel(arts, "삼성전자")] == ["삼성전자 HBM 신규 라인 가동"]

    class _Boom:
        def post(self, *a, **k): raise RuntimeError("network down")
    rel2 = naver_relevance_fn(GeminiClient("K", session=_Boom()))
    # on failure -> strict 'name in TITLE' fallback (only the title that names 삼성전자 survives)
    assert [a["title"] for a in rel2(arts, "삼성전자")] == ["삼성전자 HBM 신규 라인 가동"]
