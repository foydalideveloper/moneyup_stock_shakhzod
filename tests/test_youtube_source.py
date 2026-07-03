"""Media monitoring (YouTube/TV) — GROUNDED, timestamped, deep-linked catalyst extraction.
Listing, timestamped transcripts, deep-link building, filler suppression, never-fabricate, and
the honest display-only framing. Fully mocked: no network, no model."""

from datetime import datetime, timezone

from tagent.news.youtube_source import (
    Channel, YouTubeSource, build_media_payload, channels_from_env, deeplink,
    default_analyze, default_extract_catalysts, detect_catalyst, extract_tickers,
    fetch_segments, fetch_transcript, llm_extract_fn, media_sentiment, mmss,
    parse_timed_text, render_catalyst_line, render_catalysts_report, webshare_proxy_url,
)
from tagent.news.youtube_source import _whisper_prompt  # noqa: E402  (glossary helper)


def test_grounded_summary_is_two_to_three_sentences():
    # a grounded multi-segment span -> a 2-3 sentence summary (claim + context + implication)
    segs = [{"text": "SK하이닉스 관련 소식입니다", "start": 0.0},
            {"text": "SK하이닉스가 엔비디아와 HBM 공급계약을 체결했습니다", "start": 10.0},
            {"text": "메모리 업황 개선이 기대됩니다", "start": 20.0}]
    out = default_extract_catalysts(segs, {"video_id": "v", "title": "t"}, "한경TV")
    summ = next(o for o in out if o["stock"] == "000660")["summary"]
    assert summ.count(".") >= 2 and "공급계약" in summ        # >= 2 sentences, grounded in the transcript


def test_whisper_prompt_glossary_and_env_override(monkeypatch):
    monkeypatch.delenv("WHISPER_PROMPT", raising=False)
    base = _whisper_prompt()
    assert "SK하이닉스" in base and "휴머노이드" in base and "HBM" in base   # default finance glossary
    monkeypatch.setenv("WHISPER_PROMPT", "삼성전자 커스텀")
    assert _whisper_prompt() == "삼성전자 커스텀"             # env override


class _Resp:
    def __init__(self, data): self._data = data
    def json(self): return self._data


# QUOTA-CHEAP polling fixtures: channels.list(contentDetails) -> uploads playlist id (UU…);
# playlistItems.list -> recent uploads, newest-first (video id in contentDetails.videoId).
_UPLOADS = {"items": [{"id": "UC1", "contentDetails": {"relatedPlaylists": {"uploads": "UU1"}}}]}
_PLAYLIST = {"items": [
    {"contentDetails": {"videoId": "v1", "videoPublishedAt": "2026-06-11T09:00:00Z"},
     "snippet": {"title": "증시 브리핑", "channelTitle": "한국경제TV",
                 "publishedAt": "2026-06-11T09:00:00Z", "resourceId": {"videoId": "v1"}}},
    {"contentDetails": {"videoId": "v2", "videoPublishedAt": "2026-06-11T07:00:00Z"},
     "snippet": {"title": "마감 시황", "channelTitle": "한국경제TV",
                 "publishedAt": "2026-06-11T07:00:00Z", "resourceId": {"videoId": "v2"}}},
]}

# A realistic transcript: a filler line, a deal line, an earnings line, a price-move line.
_SEGS = [
    {"text": "오늘 시장 전반을 짚어보겠습니다", "start": 0.0, "duration": 6.0},          # no stock/catalyst
    {"text": "삼성전자는 AI 잠재력이 큰 장기 유망주입니다", "start": 12.0, "duration": 5.0},   # stock + FILLER
    {"text": "SK하이닉스가 엔비디아와 HBM 공급계약을 체결했습니다", "start": 83.0, "duration": 7.0},
    {"text": "현대차 영업이익이 컨센서스를 크게 상회했습니다", "start": 140.0, "duration": 6.0},
    {"text": "포스코는 오늘 급락했습니다", "start": 200.0, "duration": 4.0},
]
_VIDEO = {"video_id": "v1", "title": "증시 브리핑", "published_at": "2026-06-11T09:00:00Z"}


# --------------------------------------------------------------------------- #
# 1) video listing
# --------------------------------------------------------------------------- #
class _ByUrl:
    """Dispatch GETs by URL fragment (so channels.list and search.list differ)."""
    def __init__(self, by_url): self.by_url, self.calls = by_url, []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        for frag, data in self.by_url.items():
            if frag in url:
                return _Resp(data)
        return _Resp({})


def test_resolve_channel_id_handle_and_passthrough():
    by = {"/channels": {"items": [{"id": "UCabc123"}]}}
    src = YouTubeSource("KEY", session=_ByUrl(by))
    assert src.resolve_channel_id("@hkwowtv") == "UCabc123"   # @handle -> UC id
    assert src.resolve_channel_id("UCxyz") == "UCxyz"         # UC id passes through (no API call)


def test_list_recent_videos_uses_uploads_playlist_not_search():
    # QUOTA-CHEAP path: channels.list (uploads playlist, cached) + playlistItems.list — NEVER
    # search.list (the 100-unit endpoint that exhausted the daily quota).
    sess = _ByUrl({"/channels": _UPLOADS, "/playlistItems": _PLAYLIST})
    src = YouTubeSource("SECRET_KEY", session=sess)
    vids = src.list_recent_videos("UC1")
    assert [v["video_id"] for v in vids] == ["v1", "v2"]
    urls = [u for u, _ in sess.calls]
    assert any("/channels" in u for u in urls) and any("/playlistItems" in u for u in urls)
    assert all("/search" not in u for u in urls)             # the expensive endpoint is gone
    pl = next(p for u, p in sess.calls if "/playlistItems" in u)
    assert pl["playlistId"] == "UU1" and pl["key"] == "SECRET_KEY"   # queried by uploads id
    assert src.list_recent_videos("") == []


def test_uploads_playlist_id_is_cached_no_repeat_channel_lookup():
    sess = _ByUrl({"/channels": _UPLOADS, "/playlistItems": _PLAYLIST})
    src = YouTubeSource("KEY", session=sess)
    src.list_recent_videos("UC1")
    n_channels = sum("/channels" in u for u, _ in sess.calls)
    src.list_recent_videos("UC1")                            # poll again
    assert sum("/channels" in u for u, _ in sess.calls) == n_channels   # cached -> 0 extra channel lookups


def test_handle_resolves_uploads_published_after_filters_and_unresolvable():
    by = {"/channels": {"items": [{"id": "UCabc",
            "contentDetails": {"relatedPlaylists": {"uploads": "UUabc"}}}]}, "/playlistItems": _PLAYLIST}
    src = YouTubeSource("KEY", session=_ByUrl(by))
    assert src.resolve_uploads_playlist("@hkwowtv") == "UUabc"          # @handle -> uploads playlist
    vids = src.list_recent_videos("@hkwowtv", published_after="2026-06-11T08:00:00Z")
    assert [v["video_id"] for v in vids] == ["v1"]                      # v2 (07:00) older than window -> dropped
    # an unresolvable channel yields NO videos (never invented)
    src2 = YouTubeSource("KEY", session=_ByUrl({"/channels": {"items": []}, "/playlistItems": _PLAYLIST}))
    assert src2.resolve_uploads_playlist("@nope") == "" and src2.list_recent_videos("@nope") == []


def test_playlist_items_deduped_by_video_id():
    dup = {"items": _PLAYLIST["items"] + [_PLAYLIST["items"][0]]}       # v1 repeated in the listing
    src = YouTubeSource("KEY", session=_ByUrl({"/channels": _UPLOADS, "/playlistItems": dup}))
    assert [v["video_id"] for v in src.list_recent_videos("UC1")] == ["v1", "v2"]


# --------------------------------------------------------------------------- #
# 2) transcripts WITH timestamps (+ optional Whisper fallback)
# --------------------------------------------------------------------------- #
def test_fetch_segments_keeps_start_seconds_with_whisper_fallback():
    segs = fetch_segments("v1", transcript_fn=lambda vid, langs: _SEGS)
    assert [s["start"] for s in segs][:3] == [0.0, 12.0, 83.0]   # start-second preserved per segment
    assert segs[2]["text"] == "SK하이닉스가 엔비디아와 HBM 공급계약을 체결했습니다"

    def _boom(vid, langs):
        raise RuntimeError("no captions")
    # captions missing -> Whisper fallback (no timing) becomes a single t=0 segment, labeled whisper
    fb = fetch_segments("v1", transcript_fn=_boom, whisper_fn=lambda vid: "위스퍼 텍스트")
    assert fb == [{"text": "위스퍼 텍스트", "start": 0.0, "duration": 0.0, "source": "whisper"}]
    assert fetch_segments("v1", transcript_fn=_boom) == []       # no hook -> empty, never raises
    assert fetch_transcript("v1", transcript_fn=lambda vid, langs: _SEGS).startswith("오늘 시장")


# --------------------------------------------------------------------------- #
# 3) deep-link + mm:ss building
# --------------------------------------------------------------------------- #
def test_deeplink_and_mmss_build_exact_moment():
    assert deeplink("abc123", 83) == "https://www.youtube.com/watch?v=abc123&t=83s"
    assert deeplink("abc123", 83.9) == "https://www.youtube.com/watch?v=abc123&t=83s"   # int seconds
    assert deeplink("", 10) == ""
    assert mmss(83) == "01:23" and mmss(5) == "00:05" and mmss(3725) == "1:02:05"


# --------------------------------------------------------------------------- #
# 4) catalyst detection + generic-filler suppression
# --------------------------------------------------------------------------- #
def test_detect_catalyst_categories_and_suppresses_filler():
    assert detect_catalyst("엔비디아와 공급계약을 체결") == "M&A/deal"
    assert detect_catalyst("미국의 수출규제와 관세 우려") == "regulatory/geopolitical"
    assert detect_catalyst("영업이익이 컨센서스를 상회") == "earnings"
    assert detect_catalyst("오늘 급락했습니다") == "price-move reason"
    # generic hype with no concrete catalyst -> None (suppressed)
    assert detect_catalyst("삼성전자는 AI 잠재력이 큰 장기 유망주입니다") is None
    assert detect_catalyst("시장 전반을 살펴보겠습니다") is None


def test_extract_tickers_maps_korean_english_and_codes():
    assert extract_tickers("삼성전자와 SK하이닉스 강세, NVIDIA 동반 상승") == ["000660", "005930", "NVDA"]
    assert extract_tickers("Apple earnings were strong") == ["AAPL"]
    assert extract_tickers("pineapple smoothie") == [] and extract_tickers("") == []


# --------------------------------------------------------------------------- #
# 5) GROUNDED extraction — verbatim quote + real start; filler / no-stock omitted
# --------------------------------------------------------------------------- #
def test_extract_catalysts_is_grounded_and_omits_filler():
    claims = default_extract_catalysts(_SEGS, _VIDEO, channel_name="한경TV")
    stocks = sorted({c["stock"] for c in claims})
    assert stocks == ["000660", "005380", "005490", "NVDA"]    # deal(2: 000660+NVDA) + earnings + move
    assert len(claims) == 4
    assert "005930" not in {c["stock"] for c in claims}        # 삼성전자 line was FILLER -> suppressed

    deal = next(c for c in claims if c["stock"] == "000660")
    assert "SK하이닉스가 엔비디아와 HBM 공급계약을 체결했습니다" in deal["quote"]   # span includes the catalyst
    assert deal["summary"] and "공급계약" in deal["summary"]      # a summary of the actual point
    assert deal["start"] == 83 and deal["timestamp_mmss"] == "01:23"   # links to where 000660 is discussed
    assert deal["deeplink"] == "https://www.youtube.com/watch?v=v1&t=83s"
    assert deal["category"] == "M&A/deal" and deal["channel"] == "한경TV"
    assert deal["video_title"] == "증시 브리핑"
    # required output keys per the boss's spec
    assert {"stock", "channel", "video_title", "summary", "quote", "timestamp_mmss", "deeplink"} <= set(deal)


def test_every_claim_carries_a_quote_and_source_link():
    claims = default_extract_catalysts(_SEGS, _VIDEO, "한경TV")
    assert claims
    for c in claims:
        assert c["quote"]                                       # a real quote
        assert c["deeplink"].startswith("https://www.youtube.com/watch?v=v1&t=")
        assert c["source_link"] == c["deeplink"]                # every claim cites its source


def test_never_fabricate_quote_span_is_verbatim_transcript():
    claims = default_extract_catalysts(_SEGS, _VIDEO, "한경TV")
    full = " ".join(s["text"] for s in _SEGS)
    seg_starts = {int(s["start"]) for s in _SEGS}
    for c in claims:
        assert c["quote"] in full                               # the span is a verbatim transcript slice
        assert c["start"] in seg_starts                         # deep-links to a real segment second


def test_llm_extract_uses_full_transcript_grounds_and_drops_hallucination():
    segs = [{"text": "오늘 시황을 살펴봅니다", "start": 0.0},
            {"text": "SK하이닉스가 엔비디아와 HBM 공급계약을 체결했고 추가 증설도 검토 중입니다", "start": 120.0},
            {"text": "이로써 메모리 업황 개선이 기대됩니다", "start": 126.0}]
    captured = {}

    def fake_llm(transcript, watchlist):
        captured["transcript"] = transcript
        return [
            {"stock": "000660", "summary": "엔비디아와 HBM 공급계약 + 증설 검토로 메모리 업황 개선 기대.",
             "quote": "SK하이닉스가 엔비디아와 HBM 공급계약을 체결했고 추가 증설도 검토"},
            {"stock": "005930", "summary": "삼성전자가 화성에 우주기지를 짓는다.",        # NOT in transcript
             "quote": "삼성전자가 화성에 우주기지를 건설한다고 발표했습니다"},
        ]

    out = llm_extract_fn(fake_llm)(segs, {"video_id": "vid", "title": "브리핑",
                                          "published_at": "2026-06-11T09:00:00Z"}, "한경TV")
    # the LLM saw the FULL timestamped transcript (every line + its mm:ss)
    assert "[02:00]" in captured["transcript"] and "시황" in captured["transcript"] and "업황 개선" in captured["transcript"]
    # grounded insight kept, linked to the real span start (120s = 02:00); abstractive summary
    assert [o["stock"] for o in out] == ["000660"]
    assert out[0]["start"] == 120 and out[0]["timestamp_mmss"] == "02:00"
    assert out[0]["deeplink"].endswith("&t=120s") and "업황" in out[0]["summary"]
    # the hallucinated 삼성전자 claim (quote not in the transcript) was DROPPED, never surfaced
    assert "005930" not in {o["stock"] for o in out}


def test_llm_extract_no_duplicate_summary_across_stocks():
    # one point applied to two stocks must NOT become the same sentence copied onto each
    segs = [{"text": "삼성전자와 SK하이닉스 모두 HBM 수혜가 기대됩니다", "start": 10.0}]
    q = "삼성전자와 SK하이닉스 모두 HBM 수혜가 기대됩니다"

    def fake(t, wl):
        return [{"stock": "005930", "summary": "두 종목 모두 HBM 수혜가 기대된다.", "quote": q},
                {"stock": "000660", "summary": "두 종목 모두 HBM 수혜가 기대된다.", "quote": q}]
    out = llm_extract_fn(fake)(segs, {"video_id": "v", "title": "t"}, "한경TV")
    assert len(out) == 1                                     # identical summary -> ONE combined insight


def test_llm_extract_combined_multi_stock_summary_names_both():
    segs = [{"text": "삼성전자와 SK하이닉스 모두 HBM 수혜가 기대됩니다", "start": 10.0}]

    def fake(t, wl):
        return [{"stock": "삼성전자와 SK하이닉스", "summary": "삼성전자와 SK하이닉스 모두 HBM 수혜가 기대된다.",
                 "quote": "삼성전자와 SK하이닉스 모두 HBM 수혜가 기대됩니다"}]
    out = llm_extract_fn(fake)(segs, {"video_id": "v", "title": "t"}, "한경TV")
    assert len(out) == 1
    assert "삼성전자" in out[0]["summary"] and "SK하이닉스" in out[0]["summary"]   # one object names both


def test_ground_quote_matches_quote_spanning_short_segments():
    # REGRESSION (single-video 0-insights): a quote that spans several short whisper segments — or
    # whose spacing differs from the captions — must still ground (was rejected -> 0 insights).
    from tagent.news.youtube_source import _ground_quote
    segs = [{"text": "삼성전자와 SK하이닉스 모두", "start": 10.0},
            {"text": "HBM 수요 강세로", "start": 12.0},
            {"text": "목표가 상향이 기대됩니다", "start": 14.0}]
    q = "삼성전자와 SK하이닉스 모두 HBM 수요 강세로 목표가 상향이 기대됩니다"   # spans all three segments
    assert _ground_quote(segs, q) == 10.0
    assert _ground_quote(segs, "삼성전자와  SK하이닉스  모두 HBM") == 10.0      # extra/odd spacing still matches
    assert _ground_quote(segs, "여기에 없는 완전히 다른 문장입니다") is None     # genuine hallucination still dropped


def test_llm_extract_keeps_multi_segment_spanning_insights():
    # the single-video regression end-to-end: spanning quotes from short segments survive grounding
    segs = [{"text": "삼성전자와 SK하이닉스 모두", "start": 10.0}, {"text": "HBM 수혜가 큽니다", "start": 12.0},
            {"text": "솔브레인은 소재 국산화로", "start": 30.0}, {"text": "적정가 35만원 매수입니다", "start": 32.0}]

    def fake(t, wl):
        return [{"stock": "삼성전자와 SK하이닉스", "summary": "두 종목 HBM 수혜.",
                 "quote": "삼성전자와 SK하이닉스 모두 HBM 수혜가 큽니다"},
                {"stock": "솔브레인", "summary": "솔브레인 적정가 35만원 매수.",
                 "quote": "솔브레인은 소재 국산화로 적정가 35만원 매수입니다", "action": "매수", "target_price": "35만원"}]
    out = llm_extract_fn(fake)(segs, {"video_id": "v", "title": "t"}, "한경TV")
    assert len(out) == 2                                       # both spanning-quote insights kept (not 0)
    sb = next(o for o in out if o["stock"] == "솔브레인")
    assert sb["action"] == "매수" and sb["target_price"] == "35만원"


def test_llm_extract_keeps_non_watchlist_stock_correctly_named():
    # a stock NOT in the ticker map (HPSP) must be kept with its spoken name + empty ticker
    segs = [{"text": "HPSP는 목표가 5만원 매수 의견입니다", "start": 10.0}]

    def fake(t, wl):
        return [{"stock": "HPSP", "summary": "HPSP 매수 의견.", "quote": "HPSP는 목표가 5만원 매수 의견입니다",
                 "action": "매수", "target_price": "5만원"}]
    out = llm_extract_fn(fake)(segs, {"video_id": "v", "title": "t"}, "한경TV")
    assert len(out) == 1
    assert out[0]["stock"] == "HPSP" and out[0]["stock_name"] == "HPSP" and out[0]["ticker"] == ""
    assert out[0]["action"] == "매수" and out[0]["target_price"] == "5만원"


def test_llm_extract_keeps_several_distinct_insights():
    # a content-rich video -> several distinct grounded insights (cap raised above the old 4)
    segs = [{"text": f"삼성전자 포인트{i} 관련 내용입니다", "start": float(i * 10)} for i in range(5)]

    def fake(t, wl):
        return [{"stock": "005930", "summary": f"삼성전자 포인트{i} 분석.",
                 "quote": f"삼성전자 포인트{i} 관련 내용입니다"} for i in range(5)]
    out = llm_extract_fn(fake)(segs, {"video_id": "v", "title": "t"}, "한경TV")
    assert len(out) == 5                                     # all 5 distinct points kept


def test_llm_extract_emits_one_insight_per_discussed_stock_with_action_and_target():
    # a multi-stock transcript with target prices + buy calls -> an insight PER stock (not just one)
    segs = [
        {"text": "삼성전자 적정가는 9만원으로 봅니다", "start": 10.0},
        {"text": "SK하이닉스는 목표가 25만원, 적극 매수 의견입니다", "start": 30.0},
        {"text": "현대차는 비중축소, 매도로 보겠습니다", "start": 50.0},
        {"text": "네이버는 실적 개선이 뚜렷합니다", "start": 70.0},
    ]

    def fake_llm(transcript, watchlist):
        return [
            {"stock": "005930", "summary": "삼성전자 적정가 9만원", "quote": "삼성전자 적정가는 9만원으로 봅니다",
             "target_price": "9만원", "category": "target-price"},
            {"stock": "000660", "summary": "SK하이닉스 목표가 25만원, 매수", "quote": "SK하이닉스는 목표가 25만원, 적극 매수 의견입니다",
             "action": "매수", "target_price": "25만원"},
            {"stock": "005380", "summary": "현대차 매도 의견", "quote": "현대차는 비중축소, 매도로 보겠습니다",
             "action": "매도"},
            {"stock": "035420", "summary": "네이버 실적 개선", "quote": "네이버는 실적 개선이 뚜렷합니다"},
        ]
    out = llm_extract_fn(fake_llm)(segs, {"video_id": "v", "title": "주력상품",
                                          "published_at": "2026-06-11T09:00:00Z"}, "박병주")
    by = {o["stock"]: o for o in out}
    assert set(by) == {"005930", "000660", "005380", "035420"}      # ALL 4 discussed stocks captured
    assert by["005930"]["target_price"] == "9만원"                   # target price carried (grounded)
    assert by["000660"]["action"] == "매수" and by["000660"]["target_price"] == "25만원"
    assert by["005380"]["action"] == "매도"


def test_default_drops_passing_mentions_and_uses_context_span():
    segs = [{"text": "먼저 삼성전자 잠깐 언급하고 넘어가겠습니다", "start": 0.0},      # passing mention, NO catalyst
            {"text": "환율과 금리 이야기입니다", "start": 5.0},
            {"text": "SK하이닉스 관련해서 보면", "start": 40.0},
            {"text": "엔비디아와 대형 공급계약을 체결했다는 소식입니다", "start": 45.0},   # catalyst names NVDA
            {"text": "하이닉스 목표가 상향이 잇따릅니다", "start": 50.0}]
    out = default_extract_catalysts(segs, {"video_id": "v", "title": "t"}, "한경TV")
    stocks = {o["stock"] for o in out}
    assert "NVDA" in stocks and "005930" not in stocks          # passing 삼성전자 mention dropped
    nv = next(o for o in out if o["stock"] == "NVDA")
    assert "공급계약" in nv["quote"] and "하이닉스" in nv["quote"]   # multi-segment CONTEXT span, not 1 line
    assert nv["start"] == 45 and nv["summary"]


def test_no_coverage_when_transcript_lacks_the_stock_or_catalyst():
    # distinct video ids so the per-video transcript cache doesn't mask each case
    src = YouTubeSource("K")
    # transcript names a stock but states NO catalyst -> omit (never invent)
    src._transcript_fn = lambda vid, langs: [{"text": "삼성전자 장기 전망은 밝습니다", "start": 0.0}]
    assert src.video_catalysts({"video_id": "v1", "title": "t"}) == []
    # transcript states a catalyst but about NO watchlist stock -> omit
    src._transcript_fn = lambda vid, langs: [{"text": "환율 급등이 두드러졌습니다", "start": 0.0}]
    assert src.video_catalysts({"video_id": "v2", "title": "t"}) == []
    # no transcript at all -> omit (no fallbacks enabled -> never hits the network)
    src._transcript_fn = lambda vid, langs: []
    assert src.video_catalysts({"video_id": "v3", "title": "t"}) == []


# --------------------------------------------------------------------------- #
# 6) end-to-end briefing — newest / most-important first across channels
# --------------------------------------------------------------------------- #
def test_media_briefing_grounded_newest_and_most_important_first():
    transcripts = {
        "v1": [{"text": "SK하이닉스가 엔비디아와 HBM 공급계약을 체결", "start": 30.0}],    # deal (imp ~4)
        "v2": [{"text": "삼성전자 영업이익이 어닝서프라이즈", "start": 10.0}],            # earnings (imp ~3)
    }
    src = YouTubeSource("KEY", session=_ByUrl({"/channels": _UPLOADS, "/playlistItems": _PLAYLIST}),
                        transcript_fn=lambda vid, langs: transcripts[vid])
    rows = src.media_briefing([Channel("한경TV", "UC1")],
                              now=datetime(2026, 6, 11, 10, tzinfo=timezone.utc))
    assert rows and rows[0]["published_at"] == "2026-06-11T09:00:00Z"   # v1 newest first
    assert rows[0]["stock"] == "000660" and rows[0]["category"] == "M&A/deal"
    assert {"stock", "channel", "video_title", "quote", "timestamp_mmss", "deeplink"} <= set(rows[0])
    assert all(r["deeplink"] and r["quote"] for r in rows)
    # the v2 (older) earnings claim comes after all v1 claims
    assert any(r["stock"] == "005930" for r in rows)
    assert src.media_briefing([Channel("noid", "")]) == []             # unconfigured channel skipped


# --------------------------------------------------------------------------- #
# 7) rendering for dashboard + daily report
# --------------------------------------------------------------------------- #
def test_render_catalyst_line_and_report():
    claims = default_extract_catalysts(_SEGS, _VIDEO, "한경TV")
    deal = next(c for c in claims if c["stock"] == "000660")
    line = render_catalyst_line(deal)
    assert "SK하이닉스" in line and "한경TV" in line and "01:23" in line
    assert deal["quote"] in line and deal["deeplink"] in line
    report = render_catalysts_report(claims)
    assert report.startswith("Media catalysts (grounded") and "→" in report
    assert "none in the lookback window" in render_catalysts_report([])


# --------------------------------------------------------------------------- #
# 8) honest framing: display-only, grounded, never a trading/halt signal
# --------------------------------------------------------------------------- #
def test_build_media_payload_is_grounded_display_only():
    claims = default_extract_catalysts(_SEGS, _VIDEO, "한경TV")
    p = build_media_payload(claims)
    assert p["enabled"] is True and p["grounded"] is True and p["signal"] is False
    assert "awareness only" in p["label"] and p["n"] == len(claims)
    assert "safety" not in p and "halt" not in p and "state" not in p
    # a ready-to-paste grounded block for the daily report/email
    assert p["report"].startswith("Media catalysts (grounded")


def test_channels_from_env_and_backcompat_analyze():
    chs = channels_from_env("한국경제TV=UC1, Bloomberg=UC2 ,bare")
    assert [(c.name, c.channel_id) for c in chs] == [("한국경제TV", "UC1"), ("Bloomberg", "UC2"), ("bare", "")]
    assert media_sentiment("급등 강세 호재") > 0 and media_sentiment("급락 약세 폭락") < 0
    a = default_analyze("삼성전자 급등 강세", "삼성전자")
    assert a["sentiment"] == "bullish" and a["summary"]


# --------------------------------------------------------------------------- #
# 9) transcript fallback chain — api -> yt-dlp -> whisper (bypass the IP-block)
# --------------------------------------------------------------------------- #
def _boom_fn(vid, langs):
    raise RuntimeError("IP-blocked")


def test_fallback_chain_stops_at_first_success_api():
    calls = []
    src = YouTubeSource(
        "K", mode="auto",
        transcript_fn=lambda vid, langs: (calls.append("api") or [{"text": "삼성전자 공급계약 체결", "start": 5.0}]),
        ytdlp_fn=lambda vid, langs: (calls.append("ytdlp") or []),
        whisper_fn=lambda vid, langs: (calls.append("whisper") or []))
    segs, method = src.fetch_video_transcript("vA")
    assert method == "api" and segs and calls == ["api"]   # stop at api; yt-dlp/whisper never run


def test_fallback_falls_through_to_ytdlp_when_api_blocked():
    calls = []
    src = YouTubeSource(
        "K", mode="auto", transcript_fn=_boom_fn,
        ytdlp_fn=lambda vid, langs: (calls.append("ytdlp") or [{"text": "SK하이닉스 HBM 공급계약", "start": 12.0}]),
        whisper_fn=lambda vid, langs: (calls.append("whisper") or [{"text": "x", "start": 0.0}]))
    segs, method = src.fetch_video_transcript("vB")
    assert method == "ytdlp" and segs[0]["start"] == 12.0
    assert "whisper" not in calls          # stopped at yt-dlp; whisper not reached


def test_fallback_uses_whisper_last_and_labels_machine_transcribed():
    src = YouTubeSource(
        "K", mode="auto", transcript_fn=_boom_fn,
        ytdlp_fn=lambda vid, langs: [],                    # no auto-subs available
        whisper_fn=lambda vid, langs: [{"text": "현대차 영업이익 컨센서스 상회", "start": 30.0}])
    segs, method = src.fetch_video_transcript("vC")
    assert method == "whisper" and segs
    assert all(s["source"] == "whisper" for s in segs)     # labeled machine-transcribed


def test_disabled_fallbacks_return_empty_without_network():
    # no injected fns + enable_fallbacks False -> NO strategy runs -> ([], "") (never touches network)
    src = YouTubeSource("K")
    src._transcript_fn = lambda vid, langs: []
    assert src.fetch_video_transcript("vD") == ([], "")


def test_transcript_cached_once_and_paced_between_videos():
    fetched, slept = [], []
    src = YouTubeSource(
        "K", pace_seconds=3.0, sleep_fn=lambda s: slept.append(s),
        transcript_fn=lambda vid, langs: (fetched.append(vid) or [{"text": "삼성전자 급등", "start": 1.0}]))
    src.fetch_video_transcript("v1")        # first -> no pacing delay
    src.fetch_video_transcript("v2")        # second -> one inter-video delay
    src.fetch_video_transcript("v1")        # cached -> no fetch, no delay
    assert fetched == ["v1", "v2"]          # each distinct video fetched exactly once (cached)
    assert slept == [3.0]                   # exactly one inter-video pacing delay


def test_parse_timed_text_vtt_strips_tags_and_dedupes():
    vtt = ("WEBVTT\n\n00:00:01.000 --> 00:00:03.000\n<c>삼성전자</c> 급등\n\n"
           "00:00:03.000 --> 00:00:05.000\n삼성전자 급등\n\n"               # rolling duplicate -> dropped
           "00:00:05.500 --> 00:00:08.000\nSK하이닉스 <00:00:06.000>공급계약\n")
    segs = parse_timed_text(vtt)
    assert [s["start"] for s in segs] == [1.0, 5.5]        # dup dropped; both real cues kept
    assert segs[0]["text"] == "삼성전자 급등" and "<" not in segs[1]["text"]
    assert "공급계약" in segs[1]["text"]
    srt = "1\n00:00:02,500 --> 00:00:04,000\n네이버 신고가\n"   # SRT (comma ms) parses too
    assert parse_timed_text(srt)[0]["start"] == 2.5


def test_video_report_returns_method_and_audit_records_full_transcript(tmp_path):
    from tagent.youtube_audit import AuditWriter, fetch_log_view
    src = YouTubeSource(
        "K", transcript_fn=_boom_fn,
        ytdlp_fn=lambda vid, langs: [{"text": "SK하이닉스 엔비디아 HBM 공급계약 체결", "start": 40.0},
                                     {"text": "추가 증설도 검토", "start": 46.0}])
    audit = AuditWriter(date="2026-06-11", data_dir=tmp_path)
    insights, segments, method = src.video_report(
        {"video_id": "vR", "title": "브리핑", "published_at": "2026-06-11T09:00:00Z"}, "한경TV", audit=audit)
    assert method == "ytdlp" and len(segments) == 2 and insights   # api blocked -> yt-dlp captured it
    v = fetch_log_view("2026-06-11", data_dir=tmp_path)["videos"][0]
    assert v["method"] == "ytdlp" and v["n_segments"] == 2         # FULL transcript written + method
    assert insights[0]["fetch_log_ref"]["video_id"] == "vR"        # insight traceable to its source


# --------------------------------------------------------------------------- #
# 10) proxy passthrough — the cheap caption paths (api + yt-dlp) use the .env proxy
# --------------------------------------------------------------------------- #
def test_webshare_proxy_url_built_with_rotate_suffix():
    url = webshare_proxy_url({"proxy_username": "user", "proxy_password": "pw"})
    assert url == "http://user-rotate:pw@p.webshare.io:80"
    assert webshare_proxy_url(None) == "" and webshare_proxy_url({}) == ""


def test_caption_proxy_prefers_explicit_url_then_webshare():
    s1 = YouTubeSource("K", proxy_url="http://p:1@host:8080")
    assert s1.caption_proxy_url() == "http://p:1@host:8080"          # explicit PROXY_URL wins
    s2 = YouTubeSource("K", webshare={"proxy_username": "u", "proxy_password": "x"})
    assert s2.caption_proxy_url() == "http://u-rotate:x@p.webshare.io:80"   # built from Webshare creds
    assert YouTubeSource("K").caption_proxy_url() == ""              # none configured


def test_ytdlp_whisper_skips_live_and_overlong_videos():
    from tagent.news.youtube_source import _ytdlp_whisper
    captured = {}
    _ytdlp_whisper("v1", run_fn=lambda cmd: captured.update(cmd=list(cmd)),
                   transcribe_fn=lambda path, lang: [])
    cmd = captured["cmd"]
    # a livestream / multi-hour video would hang CPU Whisper -> yt-dlp match-filter skips it
    assert "--match-filter" in cmd
    f = cmd[cmd.index("--match-filter") + 1]
    assert "!is_live" in f and "duration <" in f


def test_ytdlp_subs_passes_proxy_flag_to_command():
    from tagent.news.youtube_source import _ytdlp_subs
    captured = {}

    def fake_run(cmd):
        captured["cmd"] = list(cmd)
    # with a proxy, --proxy <url> must be in the yt-dlp argv
    _ytdlp_subs("v1", run_fn=fake_run, proxy_url="http://u-rotate:x@p.webshare.io:80")
    cmd = captured["cmd"]
    assert "--proxy" in cmd and cmd[cmd.index("--proxy") + 1] == "http://u-rotate:x@p.webshare.io:80"
    # without a proxy, no --proxy flag
    _ytdlp_subs("v1", run_fn=fake_run)
    assert "--proxy" not in captured["cmd"]


def test_enabled_fallbacks_wire_proxy_into_ytdlp_strategy():
    # the source threads its caption proxy into the (b) yt-dlp strategy when fallbacks are enabled
    captured = {}
    src = YouTubeSource("K", enable_fallbacks=True, proxy_url="http://prox:9@h:1",
                        mode="auto", transcript_fn=_boom_fn)   # api fails -> chain reaches yt-dlp
    import tagent.news.youtube_source as ys
    orig = ys._ytdlp_subs
    try:
        ys._ytdlp_subs = lambda vid, langs, proxy_url="": (captured.update(proxy=proxy_url) or
                                                           [{"text": "삼성전자 급등", "start": 1.0}])
        segs, method = src.fetch_video_transcript("vP")
    finally:
        ys._ytdlp_subs = orig
    assert method == "ytdlp" and captured["proxy"] == "http://prox:9@h:1"


# --------------------------------------------------------------------------- #
# 11) GPU-aware Whisper device/compute selection (no real model load)
# --------------------------------------------------------------------------- #
def test_whisper_model_size_from_env_default_small(monkeypatch):
    from tagent.news.youtube_source import _whisper_model
    monkeypatch.delenv("WHISPER_MODEL", raising=False)
    assert _whisper_model() == "small"                          # default
    for m in ("tiny", "base", "small", "medium", "large-v3"):
        assert _whisper_model(m) == m                           # all valid sizes pass through
    assert _whisper_model("LARGE-V3") == "large-v3"             # case-insensitive
    assert _whisper_model("huge") == "small" and _whisper_model("") == "small"   # unknown -> small
    monkeypatch.setenv("WHISPER_MODEL", "medium")
    assert _whisper_model() == "medium"                         # reads .env


def test_whisper_device_auto_picks_cuda_when_gpu_available(monkeypatch):
    import tagent.news.youtube_source as ys
    monkeypatch.delenv("WHISPER_DEVICE", raising=False)
    monkeypatch.delenv("WHISPER_COMPUTE_TYPE", raising=False)
    monkeypatch.delenv("WHISPER_COMPUTE", raising=False)
    monkeypatch.setattr(ys, "_cuda_available", lambda: True)
    assert ys._whisper_device_compute("auto") == ("cuda", "float16")     # ~50x faster on a CUDA box
    assert ys._whisper_device_compute() == ("cuda", "float16")           # default device is "auto"


def test_whisper_device_auto_falls_back_to_cpu_without_gpu(monkeypatch):
    import tagent.news.youtube_source as ys
    monkeypatch.delenv("WHISPER_COMPUTE_TYPE", raising=False)
    monkeypatch.delenv("WHISPER_COMPUTE", raising=False)
    monkeypatch.setattr(ys, "_cuda_available", lambda: False)
    assert ys._whisper_device_compute("auto") == ("cpu", "int8")


def test_whisper_device_explicit_and_compute_override():
    import tagent.news.youtube_source as ys
    assert ys._whisper_device_compute("cuda") == ("cuda", "float16")     # compute defaults per device
    assert ys._whisper_device_compute("cpu") == ("cpu", "int8")
    assert ys._whisper_device_compute("cuda", "int8_float16") == ("cuda", "int8_float16")   # explicit


def test_whisper_device_reads_env(monkeypatch):
    import tagent.news.youtube_source as ys
    monkeypatch.setattr(ys, "_cuda_available", lambda: True)
    monkeypatch.setenv("WHISPER_DEVICE", "cpu")              # env pins cpu even though a GPU is present
    monkeypatch.setenv("WHISPER_COMPUTE_TYPE", "int8")
    assert ys._whisper_device_compute() == ("cpu", "int8")
    monkeypatch.setenv("WHISPER_DEVICE", "auto")
    monkeypatch.delenv("WHISPER_COMPUTE_TYPE", raising=False)
    monkeypatch.delenv("WHISPER_COMPUTE", raising=False)
    assert ys._whisper_device_compute() == ("cuda", "float16")   # auto -> cuda (mocked available)


def test_whisper_vad_env_toggle():
    from tagent.news.youtube_source import _whisper_vad
    assert _whisper_vad() is True and _whisper_vad("1") is True          # default ON
    for off in ("0", "false", "no", "off", "OFF", ""):
        assert _whisper_vad(off) is False
    assert _whisper_vad("true") is True and _whisper_vad("yes") is True


def test_whisper_transcribe_cuda_init_failure_falls_back_to_cpu(monkeypatch):
    import sys
    import types

    import tagent.news.youtube_source as ys
    monkeypatch.delenv("WHISPER_DEVICE", raising=False)
    monkeypatch.delenv("WHISPER_COMPUTE_TYPE", raising=False)
    monkeypatch.delenv("WHISPER_COMPUTE", raising=False)
    monkeypatch.delenv("WHISPER_VAD", raising=False)
    monkeypatch.setattr(ys, "_cuda_available", lambda: True)     # auto -> cuda first
    calls = []
    kw = {}

    class _FakeModel:                                            # no real model load
        def __init__(self, size, device="cpu", compute_type="int8"):
            calls.append((device, compute_type))
            if device == "cuda":
                raise RuntimeError("no CUDA driver")             # GPU init fails
        def transcribe(self, path, language="ko", **kwargs):
            kw.update(kwargs)
            return [types.SimpleNamespace(text="삼성전자 급등", start=1.0, end=2.0)], None

    fake = types.ModuleType("faster_whisper")
    fake.WhisperModel = _FakeModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)
    monkeypatch.setenv("WHISPER_PROMPT", "삼성전자, SK하이닉스, HBM")
    out = ys._whisper_transcribe("/tmp/a.mp3", "ko")
    assert calls == [("cuda", "float16"), ("cpu", "int8")]       # tried GPU, fell back safely to CPU
    assert out and out[0]["text"] == "삼성전자 급등" and out[0]["start"] == 1.0
    # anti-garble glossary + anti-hallucination settings passed to the transcriber
    assert kw.get("initial_prompt") == "삼성전자, SK하이닉스, HBM"
    assert kw.get("vad_filter") is True and kw.get("condition_on_previous_text") is False
    # anti-hallucination settings passed to faster-whisper .transcribe()
    assert kw["vad_filter"] is True and kw["condition_on_previous_text"] is False


# --------------------------------------------------------------------------- #
# 12) YOUTUBE_TRANSCRIPT_MODE — auto / whisper / api strategy ordering
# --------------------------------------------------------------------------- #
def test_transcript_mode_normalizes_and_defaults_from_env(monkeypatch):
    from tagent.news.youtube_source import _transcript_mode
    assert _transcript_mode("whisper") == "whisper" and _transcript_mode("API") == "api"
    assert _transcript_mode("bogus") == "auto"                   # unknown -> auto
    monkeypatch.setenv("YOUTUBE_TRANSCRIPT_MODE", "whisper")
    assert _transcript_mode() == "whisper"                       # default reads .env
    monkeypatch.delenv("YOUTUBE_TRANSCRIPT_MODE", raising=False)
    assert _transcript_mode() == "auto"


def _ordering_src(calls, mode):
    """A source whose 3 strategies each record their name, so we can see the order tried."""
    return YouTubeSource(
        "K", mode=mode,
        transcript_fn=lambda vid, langs: (calls.append("api") or [{"text": "삼성전자 공급계약", "start": 5.0}]),
        ytdlp_fn=lambda vid, langs: (calls.append("ytdlp") or [{"text": "SK 공급계약", "start": 6.0}]),
        whisper_fn=lambda vid, langs: (calls.append("whisper") or [{"text": "현대차 실적", "start": 7.0}]))


def test_mode_whisper_goes_straight_to_whisper():
    calls = []
    segs, method = _ordering_src(calls, "whisper").fetch_video_transcript("v1")
    assert method == "whisper" and calls == ["whisper"]          # Whisper first; api/subs skipped
    assert all(s["source"] == "whisper" for s in segs)


def test_mode_whisper_falls_back_to_api_when_whisper_fails():
    calls = []
    src = YouTubeSource(
        "K", mode="whisper",
        whisper_fn=lambda vid, langs: (calls.append("whisper") or []),     # Whisper yields nothing
        transcript_fn=lambda vid, langs: (calls.append("api") or [{"text": "삼성전자 공급계약", "start": 5.0}]),
        ytdlp_fn=lambda vid, langs: (calls.append("ytdlp") or []))
    segs, method = src.fetch_video_transcript("v1")
    assert method == "api" and calls == ["whisper", "api"]       # api/subs are the fallback after Whisper


def test_mode_api_uses_api_only():
    calls = []
    src = YouTubeSource(
        "K", mode="api", transcript_fn=_boom_fn,                 # api fails -> nothing else tried
        ytdlp_fn=lambda vid, langs: (calls.append("ytdlp") or [{"text": "x", "start": 1.0}]),
        whisper_fn=lambda vid, langs: (calls.append("whisper") or [{"text": "y", "start": 1.0}]))
    segs, method = src.fetch_video_transcript("v1")
    assert segs == [] and method == "" and calls == []          # api-only: subs/whisper never run


def test_mode_auto_is_default_order(monkeypatch):
    monkeypatch.delenv("YOUTUBE_TRANSCRIPT_MODE", raising=False)
    calls = []
    segs, method = _ordering_src(calls, None).fetch_video_transcript("v1")   # None -> env -> auto
    assert method == "api" and calls == ["api"]                 # auto = api first
