"""Continuous intraday feed — dual timestamps, dedupe/append, paced throttle, NEW marking,
append-only persistence, no fabrication. Synthetic, no network."""

from tagent.intraday_feed import (
    SourceThrottle, append_feed, feed_view, from_kiwoom_cards, from_news_items, from_youtube_claims,
    kst_hhmm, kst_stamp, load_feed, make_item, poll_and_persist, poll_once, refilter_feed,
)


# --------------------------------------------------------------------------- #
# 1) item construction — grounded gate + dual timestamps
# --------------------------------------------------------------------------- #
def test_make_item_grounded_and_dual_timestamp():
    it = make_item("naver", link="https://hk.com/a", title="t", symbol="005930",
                   published_at="2026-06-11T09:00:00+00:00", published_ts=1000.0)
    assert it["id"] == "u:https://hk.com/a" and it["link_type"] == "url"
    assert it["published_at"] == "2026-06-11T09:00:00+00:00" and it["published_ts"] == 1000.0
    # GROUNDED: external news with no link / a non-http link -> None (never invented)
    assert make_item("naver", link="") is None
    assert make_item("finnhub", link="not-a-url") is None
    # youtube deep-link + mm:ss; kiwoom provenance allowed without http
    yt = make_item("youtube", link="https://www.youtube.com/watch?v=v1&t=83s",
                   link_type="deeplink", mmss="01:23")
    assert yt["link_type"] == "deeplink" and yt["mmss"] == "01:23"
    kw = make_item("kiwoom", link="kiwoom:ka10014", external=False, title="005930 공매도 증가")
    assert kw and kw["link_type"] == "provenance" and kw["id"].startswith("s:kiwoom:")


def test_adapters_capture_dual_timestamps_and_drop_uncited():
    news = [{"url": "https://hk/1", "title": "a", "symbol": "005930", "ts": 1000.0, "time": "2026-06-11"},
            {"url": "", "title": "no link", "ts": 1.0}]                # dropped (no link)
    fi = from_news_items(news, "naver")
    assert len(fi) == 1 and fi[0]["published_at"] == "2026-06-11" and fi[0]["published_ts"] == 1000.0
    claims = [{"deeplink": "https://youtu/v?t=83s", "quote": "q", "stock": "000660",
               "published_at": "2026-06-11T09:00:00Z", "timestamp_mmss": "01:23"}]
    yfi = from_youtube_claims(claims)
    assert yfi[0]["mmss"] == "01:23" and yfi[0]["published_ts"] > 0 and yfi[0]["link_type"] == "deeplink"
    cards = [{"symbol": "005930", "lines": [{"text": "공매도 증가", "cite": "kiwoom:ka10014", "kind": "short"},
                                            {"text": None, "empty": True}]}]   # empty line skipped
    kfi = from_kiwoom_cards(cards)
    assert len(kfi) == 1 and kfi[0]["source"] == "kiwoom" and "공매도" in kfi[0]["title"]


def test_no_fabrication_uncited_items_dropped():
    assert from_news_items([{"title": "no url", "ts": 1.0}], "finnhub") == []
    assert from_youtube_claims([{"quote": "q", "published_at": "2026-06-11T09:00:00Z"}]) == []   # no deeplink


# --------------------------------------------------------------------------- #
# 2) paced throttle
# --------------------------------------------------------------------------- #
def test_source_throttle_per_source_interval():
    th = SourceThrottle({"naver": 600, "youtube": 1800})
    assert th.due("naver", now=1000) is True
    th.mark("naver", now=1000)
    assert th.due("naver", now=1300) is False                # 300s < 600s -> throttled
    assert th.due("naver", now=1600) is True                 # 600s elapsed
    assert th.due("youtube", now=1000) is True               # never fetched yet
    th.mark("youtube", now=1000)
    assert th.due("youtube", now=2000) is False              # 1000s < 1800s (IP-block guard)


# --------------------------------------------------------------------------- #
# 3) poll — dedupe, detected_at stamping, throttle-skip
# --------------------------------------------------------------------------- #
def test_poll_once_dedupes_stamps_detected_at_and_skips_throttled():
    th = SourceThrottle({"naver": 600, "finnhub": 600})

    def naver_fetch():
        return from_news_items([{"url": "https://hk/1", "title": "a", "ts": 2.0, "time": "t2"},
                                {"url": "https://hk/2", "title": "b", "ts": 1.0, "time": "t1"}], "naver")

    def finnhub_fetch():
        return from_news_items([{"url": "https://f/1", "title": "c", "ts": 3.0, "time": "t3"}], "finnhub")

    seen = set()
    fetchers = {"naver": naver_fetch, "finnhub": finnhub_fetch}
    r1 = poll_once(fetchers, seen, detected_at="2026-06-11T10:00:00", throttle=th, now_ts=1000)
    assert r1["n_new"] == 3 and set(r1["fetched"]) == {"naver", "finnhub"}
    assert all(i["detected_at"] == "2026-06-11T10:00:00" for i in r1["new"])
    assert r1["new"][0]["published_ts"] == 3.0               # newest-first by published time
    # soon after: throttled -> not fetched, nothing new
    r2 = poll_once(fetchers, seen, detected_at="2026-06-11T10:05:00", throttle=th, now_ts=1100)
    assert r2["fetched"] == [] and r2["n_new"] == 0 and set(r2["skipped"]) == {"naver", "finnhub"}
    # throttle elapsed but SAME items -> deduped (no new), originals keep their detected_at
    r3 = poll_once({"naver": naver_fetch}, seen, detected_at="2026-06-11T10:20:00", throttle=th, now_ts=1700)
    assert "naver" in r3["fetched"] and r3["n_new"] == 0


# --------------------------------------------------------------------------- #
# 4) append-only persistence — accumulate, dedupe, preserve first-seen detected_at
# --------------------------------------------------------------------------- #
def test_feed_append_only_accumulates_and_preserves_detected_at(tmp_path):
    day = "2026-06-11"

    def f1():
        return from_news_items([{"url": "https://h/1", "title": "a", "ts": 1.0, "time": "t1"}], "naver")
    r1 = poll_and_persist({"naver": f1}, day, detected_at="2026-06-11T10:00:00",
                          throttle=SourceThrottle({"naver": 600}), now_ts=1000, data_dir=tmp_path)
    assert r1["appended"] == 1 and len(load_feed(day, tmp_path)) == 1

    def f2():
        return from_news_items([{"url": "https://h/1", "title": "a", "ts": 1.0, "time": "t1"},
                                {"url": "https://h/2", "title": "b", "ts": 2.0, "time": "t2"}], "naver")
    r2 = poll_and_persist({"naver": f2}, day, detected_at="2026-06-11T10:20:00",
                          throttle=SourceThrottle({"naver": 600}), now_ts=2000, data_dir=tmp_path)
    assert r2["appended"] == 1                               # only the genuinely-new url
    feed = load_feed(day, tmp_path)
    assert {i["link"] for i in feed} == {"https://h/1", "https://h/2"}
    first = next(i for i in feed if i["link"] == "https://h/1")
    assert first["detected_at"] == "2026-06-11T10:00:00"     # first-seen time preserved (append-only)


# --------------------------------------------------------------------------- #
# 5) feed view — newest-first + NEW marking; honest empty
# --------------------------------------------------------------------------- #
def test_feed_view_newest_first_and_new_marking(tmp_path):
    day = "2026-06-11"
    append_feed([{"id": "u:1", "url": "https://1", "link": "https://1", "published_ts": 1.0,
                  "detected_at": "2026-06-11T10:00:00"},
                 {"id": "u:2", "url": "https://2", "link": "https://2", "published_ts": 2.0,
                  "detected_at": "2026-06-11T10:20:00"}], day, tmp_path)
    v = feed_view(day, since="2026-06-11T10:10:00", data_dir=tmp_path)
    assert [i["id"] for i in v["items"]] == ["u:2", "u:1"]   # newest DETECTED first
    assert v["items"][0]["is_new"] is True and v["items"][1]["is_new"] is False
    assert v["latest_detected_at"] == "2026-06-11T10:20:00" and v["n_new"] == 1
    assert feed_view("2026-06-12", data_dir=tmp_path)["note"].startswith("no items")


# --------------------------------------------------------------------------- #
# 6) KST display — boss is in Korea; render times in Asia/Seoul (storage stays UTC)
# --------------------------------------------------------------------------- #
def test_kst_helpers_convert_utc_to_seoul():
    assert kst_hhmm("2026-06-11T08:29:35+00:00") == "17:29"   # 08:29 UTC + 9h = 17:29 KST
    assert kst_hhmm("2026-06-11T08:29:35Z") == "17:29"        # 'Z' handled
    assert kst_stamp("2026-06-11T08:28:00+00:00") == "06-11 17:28"
    assert kst_stamp("2026-06-11") == "2026-06-11"            # date-only -> no fabricated time
    assert kst_hhmm("") == "" and kst_stamp(None) == ""


def test_feed_view_renders_times_in_kst(tmp_path):
    day = "2026-06-11"
    append_feed([{"id": "u:1", "link": "https://hk/1", "published_ts": 5.0,
                  "detected_at": "2026-06-11T08:29:35+00:00",        # a Naver item detected ~08:29 UTC
                  "published_at": "2026-06-11T08:28:00+00:00"}], day, tmp_path)
    it = feed_view(day, data_dir=tmp_path)["items"][0]
    # DISPLAYED in KST (Asia/Seoul) — ~17:xx, NOT 08:xx
    assert it["detected_kst"] == "17:29" and it["detected_kst"][:2] != "08"
    assert it["published_kst"] == "06-11 17:28"
    # storage stays UTC internally
    assert it["detected_at"] == "2026-06-11T08:29:35+00:00"
    assert feed_view(day, data_dir=tmp_path)["tz"] == "Asia/Seoul (KST)"


# --------------------------------------------------------------------------- #
# 7) re-filter the persisted feed — top-source whitelist + relevance, link kept
# --------------------------------------------------------------------------- #
def test_refilter_feed_drops_nontop_and_tangential_keeps_link(tmp_path):
    day = "2026-06-12"
    append_feed([
        {"id": "u:1", "source": "naver", "symbol": "005930", "title": "삼성전자, HBM 라인 증설",
         "link": "https://www.hankyung.com/1", "detected_at": "2026-06-12T01:00:00+00:00"},
        {"id": "u:2", "source": "naver", "symbol": "035420", "title": "티빙·CJ 개인정보 유출",     # low-tier + tangential
         "link": "https://www.dizzotv.com/2", "detected_at": "2026-06-12T01:00:00+00:00"},
        {"id": "u:3", "source": "naver", "symbol": "005930", "title": "이안, 영화제 수상",          # top-source but off-subject
         "link": "https://www.hankyung.com/3", "detected_at": "2026-06-12T01:00:00+00:00"},
        {"id": "yt:1", "source": "youtube", "title": "insight", "link": "https://youtu.be/x&t=10s",
         "deeplink": "https://youtu.be/x&t=10s", "detected_at": "2026-06-12T01:00:00+00:00"},
    ], day, tmp_path)
    info = refilter_feed(day, data_dir=tmp_path)
    links = {i["link"] for i in load_feed(day, tmp_path)}
    assert links == {"https://www.hankyung.com/1", "https://youtu.be/x&t=10s"}   # top+relevant + youtube
    assert "https://www.dizzotv.com/2" not in links     # low-tier dropped
    assert "https://www.hankyung.com/3" not in links    # tangential '이안 수상' (no 삼성전자 in title) dropped
    assert info["dropped"] == 2 and info["after"] == 2
    assert all(i.get("link") for i in load_feed(day, tmp_path))      # cited link preserved on kept items
