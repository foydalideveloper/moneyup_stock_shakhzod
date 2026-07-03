"""Daily 4-report briefing — grounded, cited, deduped, persisted. Synthetic, no network."""

from datetime import datetime, timezone

from tagent.daily_briefing import (
    SeenSet, breaking_category, build_daily_briefing, build_kiwoom_briefing, build_newspaper_report,
    build_recommendation_report, build_youtube_briefing, capture_breaking, citation, is_market_wide,
    load_briefing, persist_briefing, reconcile_direction, valid_date,
)

TODAY = datetime(2026, 6, 11, tzinfo=timezone.utc)


def _ts(d):
    return datetime(2026, 6, d, tzinfo=timezone.utc).timestamp()


def _news(symbol, title, url, sent="neutral", d=11, source="naver", summary=""):
    return {"symbol": symbol, "title": title, "url": url, "sentiment": sent, "ts": _ts(d),
            "time": f"2026-06-{d:02d}", "source": source, "summary": summary}


def _claim(stock, channel, quote, vid="v1", start=83, pub="2026-06-11T09:00:00Z", cat="M&A/deal", imp=4.0):
    return {"stock": stock, "stock_name": stock, "channel": channel, "video_title": "브리핑",
            "quote": quote, "timestamp_mmss": "01:23", "start": start,
            "deeplink": f"https://www.youtube.com/watch?v={vid}&t={start}s",
            "source_link": f"https://www.youtube.com/watch?v={vid}&t={start}s",
            "category": cat, "importance": imp, "published_at": pub, "source": "youtube"}


_KIWOOM_PAYLOAD = {
    "enabled": True, "env": "mock", "label": "Kiwoom 수급", "generated": "2026-06-11",
    "empty_endpoints": ["ka90013"], "note": "빈 응답 TR: ka90013",
    "stocks": [{"symbol": "005930",
        "short": {"text": "공매도 비중 3일 연속 증가 → 매도 압력 강화", "status": "bearish",
                  "informative": True, "empty": False, "info": {"api_id": "ka10014"}},
        "supply": {"text": "외국인 3일 연속 순매도 → 매도 압력", "status": "bearish",
                   "informative": True, "empty": False, "info": {"api_id": "ka10059"}},
        "program": {"text": None, "empty": True, "note": "데이터 없음 (mock 비어있음 — 실계좌)",
                    "info": {"api_id": "ka90013"}}}]}


# --------------------------------------------------------------------------- #
# date / dedup / consistency primitives
# --------------------------------------------------------------------------- #
def test_valid_date_drops_future_and_stale():
    assert valid_date(_ts(11), TODAY) is True
    assert valid_date(_ts(12), TODAY) is False                # future -> dropped
    assert valid_date(datetime(2026, 5, 1, tzinfo=timezone.utc).timestamp(), TODAY) is False  # >14d stale
    assert valid_date(None, TODAY) is True                    # undated kept


def test_breaking_classifier_whole_word_and_exclusions():
    # whole-word: 'war' inside 'software'/'forward'/'award'/'warning' must NOT match
    assert breaking_category("Software engine the Broadcom bears overlook") is None
    assert breaking_category("Forward guidance award warning issued") is None
    # metaphorical war is excluded
    assert breaking_category("Price war among retailers intensifies") is None
    assert breaking_category("AI war heats up between chipmakers") is None
    assert breaking_category("Buffett's war chest grows") is None
    # real kinetic war / geopolitics / macro DO match
    assert breaking_category("Russia launches a full invasion") == "war"
    assert breaking_category("전면전 발발, 미사일 공습") == "war"
    assert breaking_category("US imposes new tariffs and sanctions") == "geopolitical"
    assert breaking_category("CPI comes in hot, inflation surges") == "macro"
    assert breaking_category("연준 FOMC 금리 인상 충격") == "macro"
    # 'fed' is whole-word: 'feed' / 'federal' do not trigger macro
    assert breaking_category("New animal feed factory opens") is None


def test_newspaper_source_filter_drops_nonwhitelisted():
    from tagent.news.source_whitelist import is_top_source
    naver = [_news("005930", "삼성전자 HBM 증설", "https://www.hankyung.com/1", "neutral"),
             _news("005930", "삼성 노타 수상", "https://www.dizzotv.com/2", "neutral")]   # low-tier -> drop
    rep = build_newspaper_report(["005930"], naver_items=naver, today=TODAY,
                                 source_filter=lambda it: is_top_source(it))
    assert rep["items"] and rep["items"][0]["url"] == "https://www.hankyung.com/1"
    assert all("dizzotv" not in it["url"] for it in rep["items"])    # non-whitelisted dropped


def test_breaking_excludes_korean_war_metaphors():
    # 반도체전쟁/칩전쟁/가격전쟁/무역전쟁 are metaphors -> NOT war (and not breaking on their own)
    for meta in ["반도체전쟁 격화", "칩 전쟁 심화", "가격전쟁 점입가경", "무역전쟁 우려", "AI 전쟁 본격화"]:
        assert breaking_category(meta) is None, meta
    # but a REAL kinetic war / real tariff policy still classifies
    assert breaking_category("이스라엘 전면전 발발, 미사일 공습") == "war"
    assert breaking_category("미중 관세 전면 인상") == "geopolitical"


def test_breaking_drops_single_company_defense_ai():
    # a 국방 AI single-company headline (names a watchlist co.) is NOT a market-wide shock
    item = {"title": "네이버클라우드의 국방 AI 전쟁 대응 전략 공개", "summary": "", "url": "https://x/def",
            "ts": _ts(11)}
    assert is_market_wide(item["title"]) is False
    assert capture_breaking([item], TODAY) == []               # dropped, not surfaced
    # a market-wide macro shock with no single company IS surfaced
    macro = {"title": "CPI 쇼크에 코스피 전반 급락", "summary": "", "url": "https://x/cpi", "ts": _ts(11)}
    assert any(b["url"] == "https://x/cpi" for b in capture_breaking([macro], TODAY))


def test_breaking_market_wide_gate_excludes_single_company():
    # strong macro indicators are market-wide even if a company is named
    assert is_market_wide("CPI 쇼크에 삼성전자 동반 급락") is True
    # a single-company headline is NOT a market-wide shock
    assert is_market_wide("Broadcom stock bears overlook the software engine") is False
    assert is_market_wide("네이버클라우드의 국방 AI 전략 공개") is False
    # a genuine market-wide shock names no single company
    assert is_market_wide("미중 관세 전면 확대로 코스피 급락") is True


def test_capture_breaking_kills_the_two_real_false_positives():
    items = [
        {"title": "The Software Engine The Broadcom Stock Bears Overlook", "summary": "",
         "url": "https://f/sw", "ts": _ts(11)},                                       # 'software' != war
        {"title": "네이버클라우드의 국방 AI 전략은 '옴니모달·현장 엔지니어'",
         "summary": "AI war 시대, 사이버 공격 대응 강화", "url": "https://m/def", "ts": _ts(11)},  # single-company
        {"title": "삼성전자 급락, 외국인 매도세", "summary": "", "url": "https://x/one", "ts": _ts(11)},  # single-company macro
        {"title": "Consumer Price Index: Inflation At 4.2% In May", "summary": "",
         "url": "https://f/cpi", "ts": _ts(11)},                                       # REAL macro
        {"title": "미중 관세 전면 확대로 코스피 급락, 지정학 리스크 고조", "summary": "",
         "url": "https://m/tariff", "ts": _ts(11)},                                    # REAL geopolitical
        {"title": "북한 미사일 발사로 한반도 긴장 고조", "summary": "", "url": "https://m/missile",
         "ts": _ts(11)},                                                               # REAL war
    ]
    cats = {b["url"]: b["breaking_category"] for b in capture_breaking(items, TODAY)}
    # the two real false positives + the single-company macro are NOT breaking
    assert "https://f/sw" not in cats and "https://m/def" not in cats and "https://x/one" not in cats
    # the genuine market-wide shocks ARE, correctly categorized
    assert cats.get("https://f/cpi") == "macro"
    assert cats.get("https://m/tariff") == "geopolitical"
    assert cats.get("https://m/missile") == "war"


def test_reconcile_direction_no_up_on_down_day():
    r = reconcile_direction("bullish", -2.0)                  # bullish news, price DOWN
    assert r["consistent"] is False and "하락" in r["note"] and r["price_dir"] == "down"
    assert reconcile_direction("bullish", 1.5)["consistent"] is True
    assert reconcile_direction("neutral", None)["consistent"] is True


# --------------------------------------------------------------------------- #
# #1 NEWSPAPER — breaking, cited reason, consistency guard, no-coverage
# --------------------------------------------------------------------------- #
def test_newspaper_surfaces_breaking_at_top_and_cites_every_item():
    naver = [
        _news("005930", "삼성전자, HBM 공급계약 체결", "https://hankyung.com/a1", "bullish"),
        _news("", "미중 관세 전면 확대 — 지정학 충격", "https://mk.co.kr/war1", "bearish", summary="관세 전쟁"),
    ]
    finnhub = [_news("000660", "SK hynix supply deal", "https://finnhub.io/x", "bullish", source="finnhub")]
    rep = build_newspaper_report(["005930", "000660", "035420"], naver_items=naver,
                                 finnhub_items=finnhub, prices={"005930": 1.2, "000660": 0.5}, today=TODAY)
    # breaking macro/geopolitical surfaced + cited
    assert rep["breaking"] and rep["breaking"][0]["breaking_category"] in ("war", "geopolitical", "macro")
    assert all(b["url"] for b in rep["breaking"])
    # every per-stock item cites a real URL
    assert rep["items"] and all(it["url"].startswith("http") for it in rep["items"])
    # 035420 has no article -> no coverage (never invented)
    assert "035420" in rep["no_coverage"]
    # the breaking item is NOT duplicated as a per-stock line
    assert "https://mk.co.kr/war1" not in [it["url"] for it in rep["items"]]


def test_newspaper_consistency_guard_flags_up_on_down_day():
    naver = [_news("005930", "삼성전자 호재 — 목표가 상향", "https://hankyung.com/up", "bullish")]
    rep = build_newspaper_report(["005930"], naver_items=naver, prices={"005930": -3.1}, today=TODAY)
    it = rep["items"][0]
    assert it["consistent"] is False and "불일치" in it["note"] and it["price_dir"] == "down"


def test_newspaper_drops_future_and_uncited():
    naver = [
        _news("005930", "미래 날짜 기사", "https://hankyung.com/fut", d=12),     # future -> dropped
        {"symbol": "005930", "title": "링크 없음", "url": "", "ts": _ts(11)},    # uncited -> dropped
    ]
    rep = build_newspaper_report(["005930"], naver_items=naver, today=TODAY)
    assert rep["items"] == [] and "005930" in rep["no_coverage"]


# --------------------------------------------------------------------------- #
# #2 KIWOOM — one-line interpretations + empty-TR honesty (no price tables)
# --------------------------------------------------------------------------- #
def test_kiwoom_briefing_uses_one_line_interpretations_and_flags_empty():
    rep = build_kiwoom_briefing(_KIWOOM_PAYLOAD, today=TODAY)
    card = rep["items"][0]
    texts = [ln["text"] for ln in card["lines"] if ln.get("text")]
    assert "외국인 3일 연속 순매도 → 매도 압력" in texts
    assert "공매도 비중 3일 연속 증가 → 매도 압력 강화" in texts
    assert any(ln.get("empty") for ln in card["lines"])       # program TR empty flagged, not faked
    assert "ka90013" in rep["empty_endpoints"]
    # every kiwoom line carries a TR provenance citation
    assert all(ln.get("cite", "").startswith("kiwoom:") for ln in card["lines"])


# --------------------------------------------------------------------------- #
# #3 YOUTUBE — grounded claims; "no coverage" when nothing
# --------------------------------------------------------------------------- #
def test_youtube_briefing_grounded_and_no_coverage():
    claims = [_claim("000660", "한경TV", "SK하이닉스 엔비디아 HBM 공급계약 체결")]
    rep = build_youtube_briefing(claims, today=TODAY)
    it = rep["items"][0]
    assert it["deeplink"].endswith("&t=83s") and it["cite"] == it["deeplink"] and it["quote"]
    # a claim with no deeplink is never surfaced
    rep2 = build_youtube_briefing([{**_claim("005930", "X", "q"), "deeplink": ""}], today=TODAY)
    assert rep2["items"] == [] and "no coverage" in rep2["note"]


# --------------------------------------------------------------------------- #
# #4 RECOMMENDATION — cited blend, distinct-from-Kiwoom, agreement/disagreement
# --------------------------------------------------------------------------- #
def test_recommendation_blend_is_cited_distinct_and_shows_agreement():
    providers = [
        {"house": "미래에셋", "source": "broker", "picks": [
            {"symbol": "005930", "reason": "HBM 수요", "url": "https://miraeasset.com/r1", "side": "buy"}]},
        {"house": "Berkshire", "source": "13F/whalewisdom", "picks": [
            {"symbol": "000660", "reason": "stake added", "url": "https://whalewisdom.com/x", "side": "buy"},
            {"symbol": "005930", "reason": "no link here", "url": "", "side": "sell"}]},  # uncited -> dropped
    ]
    our = [{"symbol": "005930", "reason": "12-1 모멘텀 상위", "side": "buy", "cite": "momentum-engine"}]
    rep = build_recommendation_report(["005930", "000660"], providers=providers, our_view=our,
                                      kiwoom_picks=["005930"], today=TODAY)
    by = {it["symbol"]: it for it in rep["items"]}
    # 005930: 미래에셋 + our view, all buy -> 동의; it IS on Kiwoom's list -> not distinct
    assert "미래에셋" in by["005930"]["houses"] and by["005930"]["our_view"]
    assert by["005930"]["agreement"] == "동의" and by["005930"]["distinct_from_kiwoom"] is False
    # 000660: only Berkshire (cited) -> distinct from Kiwoom's list
    assert by["000660"]["distinct_from_kiwoom"] is True
    # the uncited Berkshire 005930 sell did NOT inject a 'sell' side (it was dropped)
    assert by["005930"]["sides"] == ["buy"]
    # every surfaced rec carries a citation
    assert all(citation(it) for it in rep["items"])


def test_recommendation_no_coverage_when_all_uncited():
    providers = [{"house": "X", "picks": [{"symbol": "005930", "reason": "r", "url": ""}]}]
    rep = build_recommendation_report(["005930"], providers=providers, today=TODAY)
    assert rep["items"] == [] and "no coverage" in rep["note"]


# --------------------------------------------------------------------------- #
# CROSS-REPORT — shared breaking, dedup across all four, every claim cited, walls
# --------------------------------------------------------------------------- #
def test_full_briefing_dedup_breaking_and_every_claim_cited():
    naver = [
        _news("005930", "삼성전자 공급계약", "https://hankyung.com/s1", "bullish"),
        _news("005930", "삼성전자 공급계약", "https://hankyung.com/s1", "bullish"),   # exact dup URL
        _news("", "연준 금리 충격 — 증시 급락", "https://mk.co.kr/fed", "bearish", summary="fomc 금리"),
    ]
    payload = build_daily_briefing(
        symbols=["005930", "000660"], today=TODAY, naver_items=naver,
        youtube_claims=[_claim("000660", "한경TV", "SK하이닉스 공급계약 체결")],
        kiwoom_payload=_KIWOOM_PAYLOAD,
        providers=[{"house": "미래에셋", "picks": [
            {"symbol": "000660", "reason": "HBM", "url": "https://miraeasset.com/r", "side": "buy"}]}],
        prices={"005930": 0.8})
    reps = payload["reports"]
    # shared breaking surfaced at the top
    assert payload["breaking"] and payload["breaking"][0]["breaking_category"] == "macro"
    # duplicate article URL collapsed to a single newspaper item
    assert sum(1 for it in reps["newspaper"]["items"] if it["url"] == "https://hankyung.com/s1") == 1
    # every surfaced claim across ALL four reports carries a citation
    for name in ("newspaper", "youtube", "recommendation"):
        for it in reps[name]["items"]:
            assert citation(it).startswith("http")           # external -> real link
    for c in reps["kiwoom"]["items"]:
        assert all(citation(ln) for ln in c["lines"])        # kiwoom -> TR provenance


def test_full_briefing_reports_a_wall_when_source_unreachable():
    payload = build_daily_briefing(symbols=["005930"], today=TODAY,
                                   walls={"newspaper": "Naver/Finnhub 429 rate-limited"})
    news = payload["reports"]["newspaper"]
    assert news["wall"] and news["items"] == [] and "not filled" in news["note"]


# --------------------------------------------------------------------------- #
# PERSISTENCE — accumulate within a day; separate folder per date; never overwrite
# --------------------------------------------------------------------------- #
def test_persist_accumulates_within_day_and_separates_per_date(tmp_path):
    day1a = build_daily_briefing(symbols=["005930"], today=TODAY,
                                 naver_items=[_news("005930", "기사1", "https://h.com/1", "neutral")])
    r1 = persist_briefing(day1a, data_dir=tmp_path)
    assert r1["counts"]["newspaper"] == 1
    # same day, a NEW article -> accumulates (old one preserved, runs increments)
    day1b = build_daily_briefing(symbols=["005930"], today=TODAY,
                                 naver_items=[_news("005930", "기사2", "https://h.com/2", "neutral")])
    persist_briefing(day1b, data_dir=tmp_path)
    back = load_briefing(TODAY, data_dir=tmp_path)
    urls = {it["url"] for it in back["reports"]["newspaper"]["items"]}
    assert urls == {"https://h.com/1", "https://h.com/2"}      # accumulated, history not overwritten
    assert back["reports"]["newspaper"]["runs"] == 2
    # a different date writes a separate folder, leaving day-1 intact
    other = datetime(2026, 6, 12, tzinfo=timezone.utc)
    persist_briefing(build_daily_briefing(symbols=["005930"], today=other,
                     naver_items=[_news("005930", "다른날", "https://h.com/3", "neutral", d=12)]),
                     data_dir=tmp_path)
    assert (tmp_path / "briefings" / "2026-06-11").exists()
    assert (tmp_path / "briefings" / "2026-06-12").exists()
    assert {it["url"] for it in load_briefing(TODAY, data_dir=tmp_path)["reports"]["newspaper"]["items"]} == \
        {"https://h.com/1", "https://h.com/2"}                 # day-1 untouched by day-2 write


def test_daily_refresh_persists_recommendation_for_the_panel(tmp_path):
    """The recommendation #4 (momentum + cited provider) flows through assembly, persists, and
    reads back — the data path the dashboard's daily-refresh panel renders."""
    from tagent.recommendation import momentum_picks
    import pandas as pd
    idx = pd.date_range("2026-01-01", periods=8, freq="B")
    panel = {"005930": pd.DataFrame({"close": [100, 101, 102, 103, 104, 140, 150, 160]}, index=idx)}
    our = momentum_picks(["005930"], panel=panel, lookback=5, skip=1, top_n=5)
    providers = [{"house": "미래에셋증권", "picks": [
        {"symbol": "005930", "reason": "목표가 상향", "url": "https://mk.co.kr/s", "side": "buy"}]}]
    payload = build_daily_briefing(symbols=["005930"], today=TODAY, providers=providers, our_view=our)
    persist_briefing(payload, data_dir=tmp_path)
    rec = load_briefing(TODAY, data_dir=tmp_path)["reports"]["recommendation"]
    assert rec["items"] and rec["items"][0]["symbol"] == "005930"
    assert rec["items"][0]["our_view"] and "미래에셋증권" in rec["items"][0]["houses"]
    assert rec["items"][0]["cite"]                             # cited (real link or provenance)
