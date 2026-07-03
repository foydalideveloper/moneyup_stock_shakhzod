"""GROUNDED daily YouTube report builder — fully mocked (no network, no LLM).

Covers the anti-hallucination guarantees that fix the rival's fake report: window filtering by
published_at, the giant-Korean universe (drop ETFs / US-noise; linked globals only when tied to a
giant in the same video), dropping a fabricated insight whose quote isn't in the transcript,
rejecting templated/duplicate reasons, Action defaulting to WATCH without a directional quote,
catalysts only when a schedule is actually mentioned, number-grounding, and the Japanese-text guard."""

import json
from pathlib import Path

from tagent.news.youtube_report import _action, build_report_from_videos, build_youtube_report

KST = "+09:00"


# --------------------------------------------------------------------------- #
# action mapping: explicit call / target price -> BUY/SELL (not WATCH)
# --------------------------------------------------------------------------- #
def test_action_uses_explicit_call_and_target_price():
    assert _action("", {"action": "매수"}) == "BUY"
    assert _action("", {"action": "매도"}) == "SELL"
    assert _action("", {"action": "비중확대"}) == "BUY"
    assert _action("", {"target_price": "9만원"}) == "BUY"          # a stated target price -> BUY call
    assert _action("삼성전자 적정가는 9만원으로 봅니다") == "BUY"        # target-price wording in the quote -> BUY
    assert _action("그냥 지켜보겠습니다") == "관심(WATCH)"             # no directional view -> WATCH


def test_recommendation_prefers_directional_over_higher_importance_watch():
    q1 = "삼성전자 그냥 지켜보겠습니다"                         # WATCH, but higher importance
    q2 = "삼성전자는 30만원 이하면 매수입니다"                  # grounded directional call (lower importance)
    entry = {"video_id": "v", "channel": "한경TV", "title": "t", "published_at": "2026-06-13T01:00:00Z",
             "segments": [{"start": 10, "text": q1}, {"start": 20, "text": q2}],
             "insights": [
                 {"ticker": "005930", "stock": "005930", "stock_name": "삼성전자", "quote": q1,
                  "summary": "관망", "importance": 9.0, "timestamp_mmss": "00:10", "deeplink": "x"},
                 {"ticker": "005930", "stock": "005930", "stock_name": "삼성전자", "quote": q2,
                  "summary": "30만원 이하 매수", "action": "매수", "importance": 1.0,
                  "timestamp_mmss": "00:20", "deeplink": "y"}]}
    rep = build_report_from_videos([entry], now="2026-06-13T15:50:00+09:00")
    rec = next(r for r in rep["recommendations"] if r["ticker"] == "005930")
    assert rec["action"] == "BUY"                            # directional call preferred over the WATCH one
    assert len(rep["per_stock"]["삼성전자"]) == 2             # both grounded insights kept in per-stock detail


def test_section1_overview_plus_grounded_per_stock_synthesis():
    s1 = "삼성전자가 엔비디아와 HBM 공급계약을 체결했습니다."
    s2 = "이로 인해 메모리 업황 개선이 기대됩니다."
    s3 = "목표가 상향 가능성도 거론됩니다."
    segs = [{"start": 10, "text": "삼성전자 HBM 공급계약 소식"},
            {"start": 20, "text": "메모리 업황 개선"}, {"start": 30, "text": "목표가 상향"}]
    ins = [{"ticker": "005930", "stock": "005930", "stock_name": "삼성전자", "quote": "삼성전자 HBM 공급계약 소식",
            "summary": s1, "importance": 5.0, "timestamp_mmss": "00:10", "deeplink": "a"},
           {"ticker": "005930", "stock": "005930", "stock_name": "삼성전자", "quote": "메모리 업황 개선",
            "summary": s2, "importance": 3.0, "timestamp_mmss": "00:20", "deeplink": "b"},
           {"ticker": "005930", "stock": "005930", "stock_name": "삼성전자", "quote": "목표가 상향",
            "summary": s3, "importance": 1.0, "timestamp_mmss": "00:30", "deeplink": "c"}]
    entry = {"video_id": "v", "channel": "한경TV", "title": "t", "published_at": "2026-06-13T01:00:00Z",
             "segments": segs, "insights": ins}
    rep = build_report_from_videos([entry], now="2026-06-13T15:50:00+09:00")
    # overall overview: multi-sentence, grounded (restates the top insight content)
    assert rep["overview"] and rep["overview"].count(".") >= 2 and s1 in rep["overview"]
    # §1 per-stock synthesis: combines MULTIPLE grounded insight summaries (not one short line)
    syn = next(b for b in rep["summary"] if b["ticker"] == "005930")["text"]
    assert s1 in syn and s2 in syn
    # grounded: the synthesis only restates this stock's own insight summaries (no new facts)
    assert syn.replace(s1, "").replace(s2, "").replace(s3, "").strip() == ""
    assert len(rep["per_stock"]["삼성전자"]) == 3              # §5 detail keeps every insight


def test_en_report_body_is_english_on_batch_parse_mismatch():
    from tagent.gemini import translate_batch_fn
    # a client whose BATCH call is unparseable but PER-FIELD call returns English
    class _Cli:
        api_key, model, _session = "K", "m", None

        def generate(self, prompt, system=None, temperature=0.0, json_out=True, thinking_budget=None):
            return "garbled no markers" if "@@0@@" in prompt else "ENGLISH"
    tr = translate_batch_fn(_Cli())
    q = "삼성전자가 HBM 공급계약을 체결했습니다"
    entry = {"video_id": "v", "channel": "한경TV", "title": "t", "published_at": "2026-06-13T01:00:00Z",
             "segments": [{"start": 10, "text": q}],
             "insights": [{"ticker": "005930", "stock": "005930", "stock_name": "삼성전자", "quote": q,
                           "summary": "HBM 공급계약 체결", "importance": 5.0, "timestamp_mmss": "00:10",
                           "deeplink": "a"}]}
    rep = build_report_from_videos([entry], lang="en", translate_fn=tr, now="2026-06-13T15:50:00+09:00")
    # EN body is actually English (per-field fallback), NOT the Korean original
    assert rep["en"]["overview"] == "ENGLISH"
    assert rep["en"]["summary"][0]["text"] == "ENGLISH"
    assert rep["en"]["per_stock"]["삼성전자"][0]["summary"] == "ENGLISH"   # §5 per-stock text English too
    # the EN render body carries English text, not the Korean original
    from tagent.news.report_render import report_to_html
    html = report_to_html(rep, "en")
    assert "ENGLISH" in html and "HBM 공급계약 체결" not in html


def test_window_price_table_removed(tmp_path):
    # §4 시세 (price table) was REMOVED: the WINDOW report builds with NO prices and the prices_fn is
    # never called, even when supplied. The discussed stocks still flow into the report body.
    _write(tmp_path, "2026-06-14", [
        _video("v1", "한경TV", "2026-06-14T02:00:00Z", [
            _ins("000660", "SK하이닉스", "SK하이닉스가 엔비디아와 HBM 공급계약을 체결했습니다"),
            _ins("005930", "삼성전자", "삼성전자가 어닝서프라이즈를 기록했습니다", summary="어닝서프라이즈"),
        ]),
    ])
    captured = {}

    def prices_fn(codes):
        captured["codes"] = list(codes)
        return [{"name": "x", "ticker": c, "current": None, "today_open": None, "prev_close": None,
                 "week_ago_close": None, "month_ago_close": None, "change_pct": None, "source": "키움/KRX"}
                for c in codes]
    rep = build_youtube_report(data_dir=tmp_path, prices_fn=prices_fn, **_WIN)
    assert rep["prices"] == []                                # price section gone
    assert "codes" not in captured                            # prices_fn never invoked
    insight_tickers = {i["ticker"] for v in rep["per_stock"].values() for i in v}
    assert insight_tickers == {"000660", "005930"}            # discussed stocks still present
    # the rendered report carries no 시세 heading/table
    from tagent.news.report_render import report_to_html
    assert "시세" not in report_to_html(rep, "ko")


def test_section1_and_section3_do_not_repeat(tmp_path):
    # §1 synthesis caps at ~4 sentences; a high-impact insight left OUT of §1 surfaces in §3,
    # while a high-impact insight already headlined in §1 is dropped from §3 (no duplicate text).
    segs = [{"start": i * 10, "text": f"포인트{i} 내용"} for i in range(1, 6)]
    sens = {1: "M&A/deal", 5: "earnings"}                      # 1 (lowest imp) + 5 (highest imp) high-impact
    ins = [{"ticker": "005930", "stock": "005930", "stock_name": "삼성전자", "quote": f"포인트{i} 내용",
            "summary": f"포인트{i} 요약.", "importance": float(i),
            "category": sens.get(i, "discussed"), "timestamp_mmss": f"00:{i:02d}", "deeplink": "x"}
           for i in range(1, 6)]
    entry = {"video_id": "v", "channel": "한경TV", "title": "t", "published_at": "2026-06-13T01:00:00Z",
             "segments": segs, "insights": ins}
    rep = build_report_from_videos([entry], now="2026-06-13T15:50:00+09:00")
    s1 = (rep["overview"] + " " + " ".join(b["text"] for b in rep["summary"]))
    s3_texts = [n["text"] for n in rep["sensitive_news"]]
    # no §3 text is repeated from §1
    assert all(t not in s1 for t in s3_texts)
    # the §1-headlined high-impact item (포인트5, top importance) was dropped from §3
    assert "포인트5 요약." in s1 and "포인트5 요약." not in s3_texts
    # the distinct high-impact item NOT in §1 (포인트1, capped out) surfaces in §3
    assert "포인트1 요약." in s3_texts


def test_single_video_keeps_non_watchlist_stocks_correctly_named():
    qh = "HPSP는 목표가 5만원, 매수 의견입니다"
    qs = "솔브레인은 실적이 개선되고 있습니다"
    qsi = "신세계는 비중 축소가 필요합니다"
    qg = "삼성전자는 HBM 수혜가 기대됩니다"
    segs = [{"start": 10, "text": qh}, {"start": 20, "text": qs}, {"start": 30, "text": qsi}, {"start": 40, "text": qg}]
    ins = [{"ticker": "", "stock": "HPSP", "stock_name": "HPSP", "quote": qh, "summary": "HPSP 매수.",
            "action": "매수", "importance": 4.0, "timestamp_mmss": "00:10", "deeplink": "a"},
           {"ticker": "", "stock": "솔브레인", "stock_name": "솔브레인", "quote": qs, "summary": "솔브레인 실적 개선.",
            "importance": 3.0, "timestamp_mmss": "00:20", "deeplink": "b"},
           {"ticker": "", "stock": "신세계", "stock_name": "신세계", "quote": qsi, "summary": "신세계 비중 축소.",
            "action": "매도", "importance": 2.0, "timestamp_mmss": "00:30", "deeplink": "c"},
           {"ticker": "005930", "stock": "005930", "stock_name": "삼성전자", "quote": qg, "summary": "삼성전자 HBM 수혜.",
            "importance": 5.0, "timestamp_mmss": "00:40", "deeplink": "d"}]
    entry = {"video_id": "v", "channel": "한경TV", "title": "t", "published_at": "2026-06-13T01:00:00Z",
             "segments": segs, "insights": ins}
    rep = build_report_from_videos([entry], now="2026-06-13T15:50:00+09:00")
    # EVERY discussed stock kept under its OWN correct name (not forced onto a giant, not dropped)
    assert set(rep["per_stock"]) == {"HPSP", "솔브레인", "신세계", "삼성전자"}
    acts = {r["stock"]: r["action"] for r in rep["recommendations"]}
    assert acts["HPSP"] == "BUY" and acts["신세계"] == "SELL"   # spoken calls -> grounded actions


def test_single_video_price_table_removed():
    qg = "삼성전자는 HBM 수혜가 기대됩니다"
    qh = "HPSP는 매수 의견입니다"
    entry = {"video_id": "v", "channel": "한경TV", "title": "t", "published_at": "2026-06-13T01:00:00Z",
             "segments": [{"start": 10, "text": qg}, {"start": 20, "text": qh}],
             "insights": [{"ticker": "005930", "stock": "005930", "stock_name": "삼성전자", "quote": qg,
                           "summary": "삼성전자 HBM 수혜.", "importance": 5.0, "timestamp_mmss": "00:10", "deeplink": "d"},
                          {"ticker": "403870", "stock": "403870", "stock_name": "HPSP", "quote": qh,
                           "summary": "HPSP 매수.", "action": "매수", "importance": 3.0, "timestamp_mmss": "00:20",
                           "deeplink": "h"}]}
    captured = {}

    def prices_fn(codes):
        captured["codes"] = list(codes)
        return [{"name": "x", "ticker": c, "current": None, "today_open": None, "prev_close": None,
                 "week_ago_close": None, "month_ago_close": None, "change_pct": None, "source": "키움/KRX"}
                for c in codes]
    rep = build_report_from_videos([entry], now="2026-06-13T15:50:00+09:00", prices_fn=prices_fn)
    assert rep["prices"] == []                                  # price section removed
    assert "codes" not in captured                              # prices_fn never invoked
    # the mentioned stocks still flow into the report body
    assert {it["ticker"] for items in rep["per_stock"].values() for it in items} == {"005930", "403870"}


def test_single_video_report_meta_and_no_prices():
    quote = "SK하이닉스는 목표가 25만원, 적극 매수 의견입니다"
    entry = {"video_id": "v1", "channel": "박병주의 주력상품", "title": "주력상품",
             "published_at": "2026-06-13T01:00:00Z",   # KST 10:00 06-13
             "segments": [{"start": 30, "text": quote}],
             "insights": [{"ticker": "000660", "stock": "000660", "stock_name": "SK하이닉스",
                           "quote": quote, "summary": "목표가 25만원 매수", "action": "매수",
                           "target_price": "25만원", "timestamp_mmss": "00:30",
                           "deeplink": "https://www.youtube.com/watch?v=v1&t=30s"}]}
    fake_prices = [{"name": "SK하이닉스", "ticker": "000660", "current": 250000.0, "today_open": 248000.0,
                    "prev_close": 245000.0, "week_ago_close": None, "month_ago_close": None,
                    "change_pct": 2.04, "source": "키움/KRX"}]
    rep = build_report_from_videos([entry], now="2026-06-13T15:50:00+09:00", prices_fn=lambda c: fake_prices)
    m = rep["meta"]
    assert m["single_video"] is True and m["video_published_kst"]          # labeled single-video
    assert m["window_start"] != m["window_end"]                           # NOT a zero-length "09:00 -> 09:00"
    assert rep["prices"] == []                                            # §4 시세 removed; the prices_fn result is ignored
    rec = next(r for r in rep["recommendations"] if r["ticker"] == "000660")
    assert rec["action"] == "BUY" and rec.get("target_price") == "25만원"   # explicit call honored


def _write(data_dir, date_str, entries):
    p = Path(data_dir) / "youtube_fetch_log" / date_str
    p.mkdir(parents=True, exist_ok=True)
    with (p / "fetch.jsonl").open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def _ins(ticker, name, quote, summary=None, **kw):
    base = {"ticker": ticker, "stock": ticker, "stock_name": name, "quote": quote,
            "summary": summary if summary is not None else quote, "timestamp_mmss": "02:00",
            "deeplink": f"https://www.youtube.com/watch?v=vid&t=120s", "start": 120,
            "importance": 4.0, "sentiment_score": 0.5, "category": "M&A/deal"}
    base.update(kw)
    return base


def _video(vid, channel, published_at, insights, segments=None):
    # transcript = the insight quotes joined (so grounded quotes are 'present verbatim')
    segs = segments if segments is not None else [{"start": 120, "text": i["quote"]} for i in insights]
    return {"video_id": vid, "channel": channel, "title": f"{channel} 브리핑",
            "published_at": published_at, "fetched_at": published_at, "method": "ytdlp",
            "n_segments": len(segs), "segments": segs, "insights": insights}


# window: start/end in KST; the dashboard may pass an arbitrary end (e.g. today 15:50)
_WIN = dict(start="2026-06-14T00:00:00+09:00", end="2026-06-14T23:59:00+09:00",
            now="2026-06-14T15:50:00+09:00")


# --------------------------------------------------------------------------- #
# 1) window filtering by published_at
# --------------------------------------------------------------------------- #
def test_window_filters_by_published_at(tmp_path):
    _write(tmp_path, "2026-06-14", [
        _video("inA", "삼프로TV", "2026-06-14T01:00:00Z",       # KST 10:00 06-14 -> IN
               [_ins("000660", "SK하이닉스", "SK하이닉스가 엔비디아와 HBM 공급계약을 체결했습니다")]),
        _video("outB", "삼프로TV", "2026-06-10T01:00:00Z",      # 06-10 -> OUT of window
               [_ins("005930", "삼성전자", "삼성전자가 어닝서프라이즈를 기록했습니다")]),
    ])
    rep = build_youtube_report(data_dir=tmp_path, **_WIN)
    assert rep["meta"]["n_videos"] == 1
    assert [s["title"] for s in rep["sources"]] == ["삼프로TV 브리핑"]
    assert "SK하이닉스" in rep["per_stock"] and "삼성전자" not in rep["per_stock"]


# --------------------------------------------------------------------------- #
# 2) universe: keep giants, drop ETFs / US-noise; linked global only if tied to a giant
# --------------------------------------------------------------------------- #
def test_universe_keeps_giants_drops_etfs_and_us_noise(tmp_path):
    _write(tmp_path, "2026-06-14", [
        # giant + linked NVIDIA in the SAME video, plus an ETF + a US-only small cap
        _video("v1", "한경TV", "2026-06-14T02:00:00Z", [
            _ins("000660", "SK하이닉스", "SK하이닉스가 엔비디아와 HBM 공급계약을 체결했습니다"),
            _ins("NVDA", "엔비디아", "엔비디아가 차세대 GPU를 공개했습니다"),          # linked -> kept (giant present)
            _ins("", "KODEX 200", "KODEX 200 ETF 자금이 유입되고 있습니다"),          # ETF -> dropped
            _ins("SNDK", "SanDisk", "SanDisk 낸드 가격이 올랐습니다"),                # US-only noise -> dropped
        ]),
        # NVIDIA alone (no Korean giant in this video) -> linked global dropped
        _video("v2", "MTN", "2026-06-14T03:00:00Z", [
            _ins("NVDA", "엔비디아", "엔비디아 단독 이야기입니다 GPU 수요가 강합니다"),
        ]),
    ])
    rep = build_youtube_report(data_dir=tmp_path, **_WIN)
    stocks = set(rep["per_stock"])
    assert "SK하이닉스" in stocks and "NVIDIA" in stocks         # giant + tied linked global kept
    assert "KODEX 200" not in stocks and "SanDisk" not in stocks # ETF + US noise dropped
    # the NVIDIA-only video contributed nothing (no giant to tie to)
    assert rep["sources"][[s["url"] for s in rep["sources"]].index("https://www.youtube.com/watch?v=v2")]["n_insights"] == 0
    tickers = {r["ticker"] for r in rep["recommendations"]}
    assert "" not in tickers and "SNDK" not in tickers


# --------------------------------------------------------------------------- #
# 3) a fabricated insight (quote NOT in the transcript) is dropped
# --------------------------------------------------------------------------- #
def test_fabricated_quote_not_in_transcript_is_dropped(tmp_path):
    # transcript only talks about SK하이닉스; the KODEX claim's quote is NOT in it (the rival's bug)
    real = _ins("000660", "SK하이닉스", "SK하이닉스가 엔비디아와 HBM 공급계약을 체결했습니다")
    fake = _ins("005930", "삼성전자",
                "KODEX 200이 엔비디아와 AI 인프라 계약을 체결했습니다")            # fabricated, not in segs
    vid = _video("v1", "한경TV", "2026-06-14T02:00:00Z", [real, fake],
                 segments=[{"start": 120, "text": real["quote"]}])               # transcript has ONLY the real line
    _write(tmp_path, "2026-06-14", [vid])
    rep = build_youtube_report(data_dir=tmp_path, **_WIN)
    assert "SK하이닉스" in rep["per_stock"] and "삼성전자" not in rep["per_stock"]
    assert rep["meta"]["dropped_ungrounded"] == 1
    # the fabricated NVIDIA/KODEX partnership never surfaces anywhere
    blob = json.dumps(rep, ensure_ascii=False)
    assert "KODEX 200이 엔비디아와 AI 인프라" not in blob


# --------------------------------------------------------------------------- #
# 4) duplicate / templated reasons rejected (rival reused one sentence across stocks)
# --------------------------------------------------------------------------- #
def test_templated_duplicate_reasons_are_rejected(tmp_path):
    same = "AI 수혜가 기대되는 종목입니다"                                          # identical 'reason' for two stocks
    _write(tmp_path, "2026-06-14", [
        _video("v1", "한경TV", "2026-06-14T02:00:00Z", [
            _ins("000660", "SK하이닉스", "SK하이닉스 관련 코멘트입니다 " + same, summary=same),
            _ins("005930", "삼성전자", "삼성전자 관련 코멘트입니다 " + same, summary=same),
        ]),
    ])
    rep = build_youtube_report(data_dir=tmp_path, **_WIN)
    reasons = [r["reason"] for r in rep["recommendations"]]
    assert same not in reasons                                   # the templated reason is dropped, not reused
    assert len(reasons) == len(set(reasons))                     # no two recommendations share a reason


# --------------------------------------------------------------------------- #
# 5) Action is WATCH without a directional quote; BUY/SELL only when expressed
# --------------------------------------------------------------------------- #
def test_action_defaults_to_watch_and_buy_only_when_stated(tmp_path):
    _write(tmp_path, "2026-06-14", [
        _video("v1", "한경TV", "2026-06-14T02:00:00Z", [
            _ins("000660", "SK하이닉스", "SK하이닉스 실적이 좋았습니다", summary="실적 호조"),  # no view -> WATCH
        ]),
        _video("v2", "삼프로TV", "2026-06-14T03:00:00Z", [
            _ins("005930", "삼성전자", "삼성전자는 지금 매수 의견입니다", summary="매수 의견"),   # directional -> BUY
        ]),
    ])
    rep = build_youtube_report(data_dir=tmp_path, **_WIN)
    by = {r["ticker"]: r["action"] for r in rep["recommendations"]}
    assert by["000660"] == "관심(WATCH)" and by["005930"] == "BUY"


# --------------------------------------------------------------------------- #
# 6) catalysts: empty unless a schedule/event is explicitly mentioned
# --------------------------------------------------------------------------- #
def test_catalysts_empty_when_none_mentioned(tmp_path):
    _write(tmp_path, "2026-06-14", [
        _video("v1", "한경TV", "2026-06-14T02:00:00Z", [
            _ins("000660", "SK하이닉스", "SK하이닉스가 엔비디아와 HBM 공급계약을 체결했습니다"),
        ]),
    ])
    rep = build_youtube_report(data_dir=tmp_path, **_WIN)
    assert rep["catalysts"] == []                                # no invented "NVIDIA GTC 2026" etc.


def test_catalysts_present_only_when_schedule_quoted(tmp_path):
    _write(tmp_path, "2026-06-14", [
        _video("v1", "한경TV", "2026-06-14T02:00:00Z", [
            _ins("000660", "SK하이닉스", "SK하이닉스 다음 주 실적발표가 예정되어 있습니다"),
        ]),
    ])
    rep = build_youtube_report(data_dir=tmp_path, **_WIN)
    assert len(rep["catalysts"]) == 1 and rep["catalysts"][0]["ticker"] == "000660"
    assert rep["catalysts"][0]["deeplink"]                       # grounded with a link


# --------------------------------------------------------------------------- #
# 7) number grounding + Japanese guard (the rival invented numbers and shipped Japanese)
# --------------------------------------------------------------------------- #
def test_ungrounded_number_falls_back_to_verbatim_quote(tmp_path):
    quote = "SK하이닉스가 엔비디아와 HBM 공급계약을 체결했습니다"                    # no digits in transcript
    _write(tmp_path, "2026-06-14", [
        _video("v1", "한경TV", "2026-06-14T02:00:00Z", [
            _ins("000660", "SK하이닉스", quote, summary="목표가 20만원 제시"),       # invented number
        ]),
    ])
    rep = build_youtube_report(data_dir=tmp_path, **_WIN)
    rec = next(r for r in rep["recommendations"] if r["ticker"] == "000660")
    assert "20만원" not in rec["reason"] and rec["reason"] == quote   # invented number not surfaced


def test_japanese_text_under_korean_heading_is_rejected(tmp_path):
    quote = "SK하이닉스가 엔비디아와 HBM 공급계약을 체결했습니다"
    _write(tmp_path, "2026-06-14", [
        _video("v1", "한경TV", "2026-06-14T02:00:00Z", [
            _ins("000660", "SK하이닉스", quote, summary="これは日本語の説明です"),    # Japanese -> reject
        ]),
    ])
    rep = build_youtube_report(data_dir=tmp_path, **_WIN)
    assert rep["meta"]["rejected_japanese"] == 1 and "SK하이닉스" not in rep["per_stock"]


# --------------------------------------------------------------------------- #
# 8) bilingual: English mirror only when a translate_fn is injected
# --------------------------------------------------------------------------- #
def test_english_mirror_only_with_translate_fn(tmp_path):
    _write(tmp_path, "2026-06-14", [
        _video("v1", "한경TV", "2026-06-14T02:00:00Z", [
            _ins("000660", "SK하이닉스", "SK하이닉스가 엔비디아와 HBM 공급계약을 체결했습니다", summary="HBM 공급계약"),
        ]),
    ])
    assert build_youtube_report(data_dir=tmp_path, **_WIN)["en"] is None     # no translator -> None
    calls = []

    def tr(texts):                                          # BATCH translator: ONE call for the report
        calls.append(list(texts))
        return [f"EN[{t}]" for t in texts]
    rep = build_youtube_report(data_dir=tmp_path, translate_fn=tr, **_WIN)
    assert rep["en"]["summary"][0]["text"] == "EN[HBM 공급계약]"
    assert rep["en"]["recommendations"][0]["reason"].startswith("EN[")
    assert len(calls) == 1                                   # all prose translated in a single batch call
