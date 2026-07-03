"""Top-source whitelist — KR by domain, US by Finnhub publisher; non-news kept. Pure."""

from tagent.news.source_whitelist import (
    domain_of, filter_top_sources, is_top_source, outlet_name, parse_whitelist,
)


def test_kr_domain_whitelist_keeps_tier1_drops_lowtier():
    assert is_top_source({"source": "naver", "market": "kr", "url": "https://www.hankyung.com/a/1"}) is True
    assert is_top_source({"source": "naver", "url": "https://www.mk.co.kr/article/1"}) is True
    assert is_top_source({"source": "naver", "url": "https://news.mt.co.kr/x"}) is True
    for low in ("pinpointnews.co.kr", "hansbiz.co.kr", "ebn.co.kr", "aitimes.com", "dizzotv.com"):
        assert is_top_source({"source": "naver", "url": f"https://{low}/x"}) is False, low


def test_us_publisher_whitelist():
    assert is_top_source({"source": "finnhub", "market": "us", "publisher": "Bloomberg",
                          "url": "https://finnhub.io/x"}) is True
    assert is_top_source({"source": "finnhub", "publisher": "Reuters"}) is True
    assert is_top_source({"source": "finnhub", "publisher": "CNBC"}) is True
    assert is_top_source({"source": "finnhub", "publisher": "Random Blog"}) is False
    assert is_top_source({"source": "finnhub", "publisher": ""}) is False        # unknown -> dropped


def test_non_news_kept_and_custom_whitelist():
    assert is_top_source({"source": "youtube"}) is True and is_top_source({"source": "kiwoom"}) is True
    assert is_top_source({"source": "naver", "url": "https://x.co.kr/a"}, kr_whitelist=["x.co.kr"]) is True
    items = [{"source": "naver", "url": "https://hankyung.com/1"},
             {"source": "naver", "url": "https://dizzotv.com/2"},
             {"source": "youtube", "url": "https://youtu.be/x"}]
    kept = filter_top_sources(items)
    assert [i["url"] for i in kept] == ["https://hankyung.com/1", "https://youtu.be/x"]


def test_domain_outlet_and_parse_whitelist():
    assert domain_of("https://www.hankyung.com/article") == "hankyung.com"
    assert outlet_name("https://www.mk.co.kr/a") == "매일경제"
    assert parse_whitelist("a.com, b.com ,", ["z"]) == ["a.com", "b.com"]
    assert parse_whitelist("", ["z.com"]) == ["z.com"]                           # empty -> default
