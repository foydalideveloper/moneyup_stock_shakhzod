"""OpenAI (chat completions REST) — an ALTERNATIVE grounded YouTube extractor (Gemini stays default).

A thin client + a grounded extract_fn that returns the SAME schema as the Gemini path
(stock/ticker/action/target_price/2-3 sentence summary/verbatim quote) and obeys the SAME grounding
rule — any insight whose quote isn't verbatim in the transcript is DROPPED downstream by
``youtube_source.llm_extract_fn``. This lets us A/B the same cached transcripts via ``--reextract``
under either provider. The API key travels only in the Authorization header and is never logged;
``requests`` is lazy and the session is injectable, so tests run fully mocked (no network).
"""

from __future__ import annotations

from typing import Callable, Optional

CHAT_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4o"


class OpenAIError(RuntimeError):
    """Raised on a transport / JSON failure talking to OpenAI."""


class OpenAIClient:
    """Thin chat-completions client. ``session`` injectable for tests; key only in the header."""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, session=None):
        self.api_key = api_key
        self.model = model or DEFAULT_MODEL
        self._session = session

    def _post(self, body: dict) -> dict:
        sess = self._session
        if sess is None:
            import requests  # lazy
            sess = requests
        resp = sess.post(CHAT_URL, headers={"Authorization": f"Bearer {self.api_key}",
                                            "Content-Type": "application/json"}, json=body, timeout=60)
        try:
            return resp.json()
        except Exception as e:
            raise OpenAIError(f"openai non-JSON response: {e}")

    def generate(self, prompt: str, *, system: Optional[str] = None, temperature: float = 0.1) -> str:
        """Return the assistant message text for ``prompt`` ("" on a malformed response)."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        data = self._post({"model": self.model, "messages": messages, "temperature": temperature})
        try:
            return data["choices"][0]["message"]["content"] or ""
        except Exception:
            return ""


def openai_analyze_fn(client: OpenAIClient) -> Callable[[str, dict], list]:
    """An ``analyze_fn(full_transcript, watchlist)`` (same shape as the Gemini one) backed by OpenAI,
    using the SHARED YouTube prompt/schema. Returns [] on any failure."""
    from tagent.gemini import parse_json_list, youtube_prompt

    def _analyze(transcript: str, watchlist: dict) -> list:
        try:
            system, prompt = youtube_prompt(transcript, watchlist)
            return parse_json_list(client.generate(prompt, system=system, temperature=0.1))
        except Exception:
            return []
    return _analyze


def openai_extract_fn(client: OpenAIClient, *, watchlist: Optional[dict] = None):
    """Grounded YouTube ``extract_fn`` backed by OpenAI (= llm_extract_fn ∘ openai_analyze_fn): the
    verbatim-quote grounding (drop hallucinations) is applied by ``llm_extract_fn``, identical to
    the Gemini path."""
    from tagent.news.youtube_source import llm_extract_fn
    return llm_extract_fn(openai_analyze_fn(client), watchlist=watchlist)
