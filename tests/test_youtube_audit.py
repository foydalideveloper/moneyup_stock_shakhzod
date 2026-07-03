"""YouTube fetch AUDIT LOG — verifiable proof of exactly what was pulled.

Fully mocked (no network, no model). Covers: the raw fetch is persisted (channel, video id/title/
publishedAt + transcript segments WITH start times); each insight references its fetch-log entry
(video id + the transcript span used); the fetch_log_view summary; per-day dedup; an end-to-end
media_briefing(audit=...) pass; batch transcribe -> log -> replay; the /youtube_audit endpoint is
REMOVED. Korean text round-trips through the UTF-8 JSONL log."""

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from tagent.news.youtube_source import Channel, YouTubeSource
from tagent.youtube_audit import (
    AuditWriter, fetch_log_path, fetch_log_view, load_fetch_log, load_insights_from_log,
)

_DT = datetime(2026, 6, 11, 10, tzinfo=timezone.utc)   # fixed anchor for batch_transcribe(now=...)

# A realistic timestamped transcript: a deal line + a price-move line (Korean, with start seconds).
_SEGS = [
    {"text": "오늘 시장 전반을 짚어보겠습니다", "start": 0.0, "duration": 6.0},
    {"text": "SK하이닉스가 엔비디아와 HBM 공급계약을 체결했습니다", "start": 83.0, "duration": 7.0},
    {"text": "포스코는 오늘 급락했습니다", "start": 200.0, "duration": 4.0},
]
_VIDEO = {"video_id": "v1", "title": "증시 브리핑", "published_at": "2026-06-11T09:00:00Z"}
# quota-cheap listing mocks: channels.list -> uploads playlist; playlistItems.list -> uploads
_UPLOADS = {"items": [{"id": "UC1", "contentDetails": {"relatedPlaylists": {"uploads": "UU1"}}}]}
_PLAYLIST = {"items": [
    {"contentDetails": {"videoId": "v1", "videoPublishedAt": "2026-06-11T09:00:00Z"},
     "snippet": {"title": "증시 브리핑", "channelTitle": "한국경제TV",
                 "publishedAt": "2026-06-11T09:00:00Z", "resourceId": {"videoId": "v1"}}},
]}


class _Resp:
    def __init__(self, data): self._data = data
    def json(self): return self._data


class _ByUrl:
    """Dispatch GETs by URL fragment (channels.list vs playlistItems.list)."""
    def __init__(self, by_url): self.by_url, self.calls = by_url, []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        for frag, data in self.by_url.items():
            if frag in url:
                return _Resp(data)
        return _Resp({})


def _src(**kw):
    return YouTubeSource("SECRET_KEY", transcript_fn=lambda vid, langs: _SEGS, **kw)


# --------------------------------------------------------------------------- #
# 1) the raw fetch is PERSISTED — channel, listing meta, and transcript spans w/ start times
# --------------------------------------------------------------------------- #
def test_fetch_log_persisted_with_segments_and_start_times(tmp_path):
    audit = AuditWriter(date="2026-06-11", data_dir=tmp_path, now=lambda: "2026-06-11T00:00:00+00:00")
    insights = _src().video_catalysts(_VIDEO, "한국경제TV", audit=audit)
    assert insights, "the grounded extractor should find catalysts in this transcript"

    p = fetch_log_path("2026-06-11", data_dir=tmp_path)
    assert p.exists(), "the fetch log JSONL must be written under youtube_fetch_log/<date>/"
    entries = load_fetch_log("2026-06-11", data_dir=tmp_path)
    assert len(entries) == 1
    e = entries[0]
    # raw listing meta
    assert e["video_id"] == "v1" and e["channel"] == "한국경제TV"
    assert e["title"] == "증시 브리핑" and e["published_at"] == "2026-06-11T09:00:00Z"
    assert e["fetched_at"] == "2026-06-11T00:00:00+00:00"
    # the fetched transcript: every segment saved WITH its start second
    assert e["n_segments"] == 3
    assert [s["start"] for s in e["segments"]] == [0.0, 83.0, 200.0]
    assert e["segments"][1]["text"] == "SK하이닉스가 엔비디아와 HBM 공급계약을 체결했습니다"  # Korean round-trips
    # the resulting insights are logged alongside their raw source
    assert any(i["stock"] == "000660" for i in e["insights"])  # SK하이닉스 deal


# --------------------------------------------------------------------------- #
# 2) each insight references its fetch-log entry — video id + the transcript SPAN used
# --------------------------------------------------------------------------- #
def test_insight_links_to_fetch_log_entry_with_span(tmp_path):
    audit = AuditWriter(date="2026-06-11", data_dir=tmp_path)
    insights = _src().video_catalysts(_VIDEO, "한국경제TV", audit=audit)
    hynix = next(i for i in insights if i["stock"] == "000660")
    ref = hynix["fetch_log_ref"]
    assert ref["video_id"] == "v1"                       # links back to the fetched video
    assert ref["date"] == "2026-06-11"
    assert ref["log"] == "youtube_fetch_log/2026-06-11/fetch.jsonl"   # path to the raw source
    # the transcript SPAN used: the start second + the verbatim quote that grounds the insight
    assert ref["span_start"] == hynix["start"]
    assert "공급계약" in ref["span_quote"]
    assert ref["span_quote"] == hynix["quote"][:240]


def test_no_audit_means_no_fetch_log_ref(tmp_path):
    # without an audit writer, insights are unchanged (back-compat with existing youtube tests)
    insights = _src().video_catalysts(_VIDEO, "한국경제TV")
    assert insights and all("fetch_log_ref" not in i for i in insights)


# --------------------------------------------------------------------------- #
# 3) the fetch-log VIEW — which videos fetched, when, how many segments, which insights
# --------------------------------------------------------------------------- #
def test_fetch_log_view_summarises(tmp_path):
    audit = AuditWriter(date="2026-06-11", data_dir=tmp_path)
    _src().video_catalysts(_VIDEO, "한국경제TV", audit=audit)

    view = fetch_log_view("2026-06-11", data_dir=tmp_path)
    assert view["enabled"] is True and view["n_videos"] == 1
    assert view["date"] == "2026-06-11"
    v = view["videos"][0]
    assert v["video_id"] == "v1" and v["channel"] == "한국경제TV"
    assert v["n_segments"] == 3 and v["n_insights"] >= 1
    assert v["url"] == "https://www.youtube.com/watch?v=v1"   # a clickable link to the raw video
    assert view["n_insights"] == sum(x["n_insights"] for x in view["videos"])


def test_fetch_log_view_empty_is_honest(tmp_path):
    view = fetch_log_view("2026-06-11", data_dir=tmp_path)
    assert view["enabled"] is False and view["videos"] == [] and view["n_videos"] == 0
    assert "no YouTube fetches logged" in view["note"]


# --------------------------------------------------------------------------- #
# 4) per-day dedup — the raw transcript is the proof, logged ONCE even across repeated polls
# --------------------------------------------------------------------------- #
def test_video_logged_once_per_day_across_polls(tmp_path):
    src = _src()
    src.video_catalysts(_VIDEO, "한국경제TV", audit=AuditWriter(date="2026-06-11", data_dir=tmp_path))
    src.video_catalysts(_VIDEO, "한국경제TV", audit=AuditWriter(date="2026-06-11", data_dir=tmp_path))
    entries = load_fetch_log("2026-06-11", data_dir=tmp_path)
    assert len(entries) == 1, "a re-fetched video must not be re-logged (reloads seen ids on construction)"


# --------------------------------------------------------------------------- #
# 5) end-to-end: media_briefing(audit=...) logs every fetched video
# --------------------------------------------------------------------------- #
def test_media_briefing_with_audit_persists_log(tmp_path):
    src = YouTubeSource("SECRET_KEY", session=_ByUrl({"/channels": _UPLOADS, "/playlistItems": _PLAYLIST}),
                        transcript_fn=lambda vid, langs: _SEGS)
    audit = AuditWriter(date="2026-06-11", data_dir=tmp_path)
    rows = src.media_briefing([Channel("한국경제TV", "UC1")], now=_DT, audit=audit)  # pin window (mock pub 06-11)
    assert rows and all("fetch_log_ref" in r for r in rows)
    entries = load_fetch_log("2026-06-11", data_dir=tmp_path)
    assert [e["video_id"] for e in entries] == ["v1"]


# --------------------------------------------------------------------------- #
# 6) the /youtube_audit dashboard endpoint is REMOVED (the log file is the source of truth)
# --------------------------------------------------------------------------- #
class _FakeStock:
    def quote(self, s): return {"symbol": s}


class _FakeCrypto:
    def quote(self, s): return {"symbol": s}


def test_youtube_audit_endpoint_removed():
    from tagent.dashboard import create_app
    # the UI panel + endpoint were removed; the fetch.jsonl log is the single source of truth
    r = TestClient(create_app(_FakeStock(), _FakeCrypto())).get("/youtube_audit")
    assert r.status_code == 404
    import inspect
    assert "youtube_audit" not in inspect.signature(create_app).parameters


# --------------------------------------------------------------------------- #
# 7) BATCH transcribe ALL listed videos into the log + replay insights from it
# --------------------------------------------------------------------------- #
_PLAYLIST2 = {"items": [
    {"contentDetails": {"videoId": "v1", "videoPublishedAt": "2026-06-11T09:00:00Z"},
     "snippet": {"title": "오전 브리핑", "channelTitle": "한국경제TV", "resourceId": {"videoId": "v1"}}},
    {"contentDetails": {"videoId": "v2", "videoPublishedAt": "2026-06-11T08:00:00Z"},
     "snippet": {"title": "마감 시황", "channelTitle": "한국경제TV", "resourceId": {"videoId": "v2"}}},
    {"contentDetails": {"videoId": "v3", "videoPublishedAt": "2026-06-11T07:00:00Z"},
     "snippet": {"title": "특징주", "channelTitle": "한국경제TV", "resourceId": {"videoId": "v3"}}},
]}


def test_batch_transcribe_writes_every_video_to_log(tmp_path):
    # each listed video -> its FULL transcript + insights on its own fetch.jsonl line
    src = YouTubeSource("K", session=_ByUrl({"/channels": _UPLOADS, "/playlistItems": _PLAYLIST2}),
                        transcript_fn=lambda vid, langs: _SEGS)
    audit = AuditWriter(date="2026-06-11", data_dir=tmp_path)
    reports = src.batch_transcribe([Channel("한국경제TV", "UC1")], now=_DT, audit=audit)
    assert [r["video_id"] for r in reports] == ["v1", "v2", "v3"]
    assert all(r["method"] == "api" and r["n_segments"] == len(_SEGS) for r in reports)
    entries = load_fetch_log("2026-06-11", data_dir=tmp_path)
    assert [e["video_id"] for e in entries] == ["v1", "v2", "v3"]   # one line per video
    assert all(e["n_segments"] == len(_SEGS) for e in entries)      # FULL transcript per video
    assert all(any(i["stock"] == "000660" for i in e["insights"]) for e in entries)


def test_batch_transcribe_max_videos_caps_total(tmp_path):
    src = YouTubeSource("K", session=_ByUrl({"/channels": _UPLOADS, "/playlistItems": _PLAYLIST2}),
                        transcript_fn=lambda vid, langs: _SEGS)
    audit = AuditWriter(date="2026-06-11", data_dir=tmp_path)
    reports = src.batch_transcribe([Channel("한국경제TV", "UC1")], max_videos=2, now=_DT, audit=audit)
    assert [r["video_id"] for r in reports] == ["v1", "v2"]         # capped at 2
    assert [e["video_id"] for e in load_fetch_log("2026-06-11", data_dir=tmp_path)] == ["v1", "v2"]


def test_load_insights_from_log_replays_full_claims(tmp_path):
    src = YouTubeSource("K", session=_ByUrl({"/channels": _UPLOADS, "/playlistItems": _PLAYLIST}),
                        transcript_fn=lambda vid, langs: _SEGS)
    src.batch_transcribe([Channel("한국경제TV", "UC1")], now=_DT,
                         audit=AuditWriter(date="2026-06-11", data_dir=tmp_path))
    claims = load_insights_from_log("2026-06-11", data_dir=tmp_path)
    assert claims, "the daily briefing replays these without re-transcribing"
    c = next(c for c in claims if c["stock"] == "000660")
    # render-ready full claim (not just the subset) so #3 can populate from the log
    assert c["deeplink"] and c["timestamp_mmss"] and c["channel"] == "한국경제TV"
    assert "공급계약" in c["quote"]


# --------------------------------------------------------------------------- #
# 8) INCREMENTAL / idempotent batch — skip cached, transcribe delta, retry failures
# --------------------------------------------------------------------------- #
def _seed_entry(tmp_path, date_str, vid, segs, insights, method="ytdlp", pub="2026-06-11T09:00:00Z",
                fetched_at=None):
    AuditWriter(date=date_str, data_dir=tmp_path, now=fetched_at).record(
        {"video_id": vid, "video_title": f"{vid} 제목", "published_at": pub}, "한국경제TV",
        segs, insights, method=method)


def test_latest_entries_by_video_keeps_newest_by_fetched_at(tmp_path):
    from tagent.youtube_audit import latest_entries_by_video
    # two entries for one id across date folders -> the NEWER fetched_at wins (the reextracted one)
    _seed_entry(tmp_path, "2026-06-14", "vX", _SEGS, [{"stock": "000660", "summary": "OLD short"}],
                fetched_at="2026-06-14T23:00:00+00:00")
    _seed_entry(tmp_path, "2026-06-15", "vX", _SEGS, [{"stock": "000660", "summary": "NEW longer summary"}],
                fetched_at="2026-06-15T00:30:00+00:00")
    e = latest_entries_by_video(tmp_path)["vX"]
    assert e["insights"][0]["summary"] == "NEW longer summary"


def test_reextract_supersedes_same_day_stale_entry(tmp_path):
    from tagent.youtube_audit import latest_entries_by_video
    # the primary bug: a STALE entry already in TODAY's folder must be superseded by a forced reextract
    _seed_entry(tmp_path, "2026-06-12", "v1", _SEGS, [{"stock": "005930", "summary": "STALE"}],
                fetched_at="2026-06-12T00:00:00+00:00")

    def extract(segments, meta, channel):                   # action/target-price-aware reextract
        return [{"stock": "005930", "stock_name": "삼성전자", "ticker": "005930",
                 "quote": segments[0]["text"], "summary": "FRESH 3-sentence summary.",
                 "action": "매수", "target_price": "29만원", "timestamp_mmss": "00:00", "deeplink": ""}]
    src = YouTubeSource("K", transcript_fn=lambda v, l: _SEGS, extract_fn=extract)
    src.batch_transcribe([Channel("한국경제TV", "UC1")], lookback_hours=48, now=_DT, reextract=True,
                         audit=AuditWriter(date="2026-06-12", data_dir=tmp_path,
                                           now="2026-06-12T05:00:00+00:00"))
    e = latest_entries_by_video(tmp_path)["v1"]
    assert e["insights"][0]["summary"] == "FRESH 3-sentence summary."   # forced append supersedes stale same-day
    assert e["insights"][0].get("action") == "매수" and e["insights"][0].get("target_price") == "29만원"


def test_reextract_covers_all_window_videos_ignoring_channel_cap(tmp_path):
    # reextract must refresh EVERY cached video in the window (not just max_per_channel)
    for i in range(8):
        _seed_entry(tmp_path, "2026-06-11", f"w{i}", _SEGS, [{"stock": "000660", "summary": "old"}],
                    fetched_at="2026-06-11T10:00:00+00:00")
    seen = []

    def extract(segments, meta, channel):
        seen.append(meta.get("video_id"))
        return [{"stock": "000660", "stock_name": "SK하이닉스", "ticker": "000660",
                 "quote": segments[0]["text"], "summary": "refreshed", "timestamp_mmss": "00:00", "deeplink": ""}]
    src = YouTubeSource("K", extract_fn=extract)
    rows = src.batch_transcribe([Channel("한국경제TV", "UC1")], lookback_hours=48, now=_DT,
                                max_per_channel=3, reextract=True,
                                audit=AuditWriter(date="2026-06-12", data_dir=tmp_path))
    assert len(seen) == 8 and len(rows) == 8                  # all 8 refreshed despite max_per_channel=3


def test_batch_incremental_skips_cached_transcribes_new_retries_failure(tmp_path):
    # v1 already has a COMPLETE transcript; v2 is a prior 0-segment FAILURE; v3 is brand new
    _seed_entry(tmp_path, "2026-06-10", "v1", _SEGS, [{"stock": "000660", "quote": _SEGS[1]["text"]}])
    _seed_entry(tmp_path, "2026-06-10", "v2", [], [], method="")
    calls = []
    src = YouTubeSource("K", session=_ByUrl({"/channels": _UPLOADS, "/playlistItems": _PLAYLIST2}),
                        transcript_fn=lambda vid, langs: (calls.append(vid) or _SEGS))
    rows = src.batch_transcribe([Channel("한국경제TV", "UC1")], lookback_hours=48, now=_DT,
                                audit=AuditWriter(date="2026-06-12", data_dir=tmp_path))
    by = {r["video_id"]: r for r in rows}
    assert by["v1"]["status"].startswith("skipped (cached 3 seg") and by["v1"]["n_segments"] == 3
    assert by["v2"]["status"].startswith("re-transcribed")          # prior 0-seg failure -> retry
    assert by["v3"]["status"] == "transcribed (new)"
    assert "v1" not in calls                                        # cached -> transcriber NOT called
    assert set(calls) == {"v2", "v3"}                              # only the delta (failure + new) transcribed


def test_batch_reextract_refreshes_insights_without_transcribing(tmp_path):
    _seed_entry(tmp_path, "2026-06-10", "v1", _SEGS, [{"stock": "000660", "summary": "old"}])
    tcalls, ecalls = [], []

    def extract(segments, meta, channel):
        ecalls.append(meta.get("video_id"))
        return [{"stock": "005930", "stock_name": "삼성전자", "ticker": "005930",
                 "quote": segments[0]["text"], "summary": "fresh", "timestamp_mmss": "00:00", "deeplink": ""}]
    src = YouTubeSource("K", session=_ByUrl({"/channels": _UPLOADS, "/playlistItems": _PLAYLIST}),
                        transcript_fn=lambda vid, langs: (tcalls.append(vid) or _SEGS), extract_fn=extract)
    rows = src.batch_transcribe([Channel("한국경제TV", "UC1")], lookback_hours=48, now=_DT, reextract=True,
                                audit=AuditWriter(date="2026-06-12", data_dir=tmp_path))
    r = rows[0]
    assert r["video_id"] == "v1" and r["status"].startswith("re-extracted (cached 3 seg")
    assert tcalls == []                                            # NO re-transcription
    assert ecalls == ["v1"] and r["insights"][0]["summary"] == "fresh"   # insights re-run on cached transcript
    latest = load_insights_from_log("2026-06-12", data_dir=tmp_path)     # a fresh entry appended today
    assert latest and latest[0]["summary"] == "fresh"


def test_cached_complete_rules():
    from tagent.news.youtube_source import _cached_complete
    full = {"n_segments": 3, "segments": [{"start": 0.0}, {"start": 300.0}, {"start": 590.0}]}
    assert _cached_complete(full, {}) is True
    assert _cached_complete(None, {}) is False
    assert _cached_complete({"n_segments": 0, "segments": []}, {}) is False       # 0-seg -> retry
    trunc = {"n_segments": 1, "segments": [{"start": 10.0}]}
    assert _cached_complete(trunc, {"duration": 600}) is False                    # truncated -> retry
    assert _cached_complete(full, {"duration": 600}) is True                      # last 590 >= 50% of 600
