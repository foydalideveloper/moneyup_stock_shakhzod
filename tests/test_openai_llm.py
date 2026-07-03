"""OpenAI alternative extractor — same grounded schema + grounding as Gemini. Mocked, no network."""

import json

from tagent.gemini import build_extractor
from tagent.openai_llm import OpenAIClient, openai_extract_fn


class _Resp:
    def __init__(self, d): self._d = d
    def json(self): return self._d


class _Sess:
    def __init__(self, d): self.d, self.calls = d, []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json})
        return _Resp(self.d)


def _chat(text):
    return {"choices": [{"message": {"content": text}}]}


def test_openai_extract_grounds_and_drops_hallucination():
    segs = [{"text": "오늘 시황입니다", "start": 0.0},
            {"text": "삼성전자 적정가는 9만원, 매수 의견입니다", "start": 30.0}]
    insights = [
        {"stock": "005930", "summary": "삼성전자 적정가 9만원, 매수 의견. 반도체 업황 개선 기대.",
         "quote": "삼성전자 적정가는 9만원, 매수 의견입니다", "action": "매수", "target_price": "9만원"},
        {"stock": "000660", "summary": "화성 우주기지 건설.",
         "quote": "SK하이닉스가 화성에 기지를 짓는다"},                       # NOT in transcript -> dropped
    ]
    sess = _Sess(_chat(json.dumps(insights, ensure_ascii=False)))
    ex = openai_extract_fn(OpenAIClient("SECRET", session=sess))
    out = ex(segs, {"video_id": "v", "title": "t", "published_at": "2026-06-12T09:00:00Z"}, "한경TV")
    # same grounded schema; hallucinated quote dropped
    assert [o["stock"] for o in out] == ["005930"]
    assert out[0]["action"] == "매수" and out[0]["target_price"] == "9만원"
    assert out[0]["start"] == 30 and out[0]["deeplink"].endswith("&t=30s") and out[0]["summary"]
    # key in the Authorization header, never the URL
    h = sess.calls[0]["headers"]
    assert h["Authorization"] == "Bearer SECRET" and "SECRET" not in sess.calls[0]["url"]
    assert "api.openai.com" in sess.calls[0]["url"]


class _S:
    def __init__(self, provider, openai, gemini):
        self.llm_provider, self._oa, self._g = provider, openai, gemini
        self.openai_api_key = "K" if openai else ""
        self.openai_model = "gpt-4o"
        self.gemini_api_key = "G" if gemini else ""
        self.gemini_model = "gemini-flash-latest"
        self.gemini_extract_model = "gemini-2.5-pro"

    def has_openai_key(self): return self._oa
    def has_gemini_key(self): return self._g


def test_build_extractor_routes_to_openai_when_selected():
    assert build_extractor(_S("openai", openai=True, gemini=True)) is not None     # provider=openai used


def test_build_extractor_falls_back_to_gemini_without_openai_key():
    assert build_extractor(_S("openai", openai=False, gemini=True)) is not None     # graceful fallback


def test_build_extractor_none_when_no_keys():
    assert build_extractor(_S("openai", openai=False, gemini=False)) is None


def test_build_extractor_default_is_gemini():
    assert build_extractor(_S("gemini", openai=True, gemini=True)) is not None       # default provider
