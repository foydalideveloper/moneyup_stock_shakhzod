"""Naver News (KR newspapers) — REAL article links on every item. Mocked, no network."""

from tagent.news.naver_source import NaverNewsSource, normalize_news


class _Resp:
    def __init__(self, data): self._data = data
    def json(self): return self._data


class _Session:
    def __init__(self, data): self.data, self.calls = data, []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": params, "headers": headers})
        return _Resp(self.data)


_NEWS = {"items": [
    {"title": "<b>삼성전자</b>, HBM 공급계약 체결", "originallink": "https://www.hankyung.com/article/1",
     "link": "https://n.news.naver.com/1", "description": "삼성전자가 대형 <b>공급계약</b>을 맺었다",
     "pubDate": "Wed, 10 Jun 2026 09:00:00 +0900"},
    {"title": "코스피 약세 마감", "originallink": "https://www.mk.co.kr/article/2",
     "link": "https://n.news.naver.com/2", "description": "지수가 하락 우려 속에 약세를 보였다",
     "pubDate": "Wed, 10 Jun 2026 07:00:00 +0900"},
    {"title": "링크 없는 기사", "originallink": "", "link": "",
     "description": "본문", "pubDate": "Wed, 10 Jun 2026 06:00:00 +0900"},
]}


def test_normalize_news_strips_html_and_uses_real_article_link():
    item = _NEWS["items"][0]
    n = normalize_news(item, symbol="005930")
    assert n["title"] == "삼성전자, HBM 공급계약 체결"          # <b> tags stripped
    assert n["url"] == "https://www.hankyung.com/article/1"     # the REAL article link (originallink)
    assert n["source"] == "naver" and n["market"] == "kr" and n["symbol"] == "005930"
    assert n["ts"] > 0 and n["time"].startswith("2026-06-10")
    assert n["sentiment"] == "bullish"                         # 공급계약/체결 -> bullish


def test_search_keeps_only_items_with_a_real_link_newest_first():
    sess = _Session(_NEWS)
    src = NaverNewsSource("CID", "SECRET", session=sess)
    rows = src.search("삼성전자", display=10)
    # the link-less item is dropped (no source -> not surfaced); newest first
    assert [r["url"] for r in rows] == ["https://www.hankyung.com/article/1",
                                        "https://www.mk.co.kr/article/2"]
    assert all(r["url"] for r in rows)
    # credentials travel in headers (never in the query string / logs)
    h = sess.calls[0]["headers"]
    assert h["X-Naver-Client-Id"] == "CID" and h["X-Naver-Client-Secret"] == "SECRET"
    assert "CID" not in str(sess.calls[0]["params"])


def test_watchlist_news_tags_symbol_and_every_item_has_url():
    src = NaverNewsSource("CID", "SECRET", session=_Session(_NEWS))
    rows = src.watchlist_news(["005930", "000660"], per_symbol=5,
                              names={"005930": "삼성", "000660": "코스피"})
    assert rows and all(r["url"] for r in rows)                 # real link on every item
    assert {r["symbol"] for r in rows} == {"005930", "000660"}  # each query tagged to its code


def test_watchlist_news_title_subject_fallback_drops_body_only_mentions():
    data = {"items": [
        {"title": "삼성전자, HBM 신규 라인 증설", "originallink": "https://h/on", "link": "",
         "description": "투자 확대", "pubDate": "Wed, 10 Jun 2026 09:00:00 +0900"},
        {"title": "노타, AI 어워드 수상", "originallink": "https://x/off", "link": "",          # 삼성전자 only in body
         "description": "삼성 출신 창업가가 만든 노타", "pubDate": "Wed, 10 Jun 2026 08:00:00 +0900"},
    ]}
    src = NaverNewsSource("CID", "SECRET", session=_Session(data))
    rows = src.watchlist_news(["005930"], per_symbol=5, names={"005930": "삼성전자"})
    # strict default: the name must be the TITLE subject -> the incidental '노타' piece is dropped
    assert [r["url"] for r in rows] == ["https://h/on"]


def test_watchlist_news_uses_injected_relevance_fn():
    data = {"items": [
        {"title": "삼성전자 A", "originallink": "https://1", "link": "", "description": "",
         "pubDate": "Wed, 10 Jun 2026 09:00:00 +0900"},
        {"title": "삼성전자 B", "originallink": "https://2", "link": "", "description": "",
         "pubDate": "Wed, 10 Jun 2026 08:00:00 +0900"}]}
    src = NaverNewsSource("CID", "SECRET", session=_Session(data))
    rows = src.watchlist_news(["005930"], per_symbol=5, names={"005930": "삼성전자"},
                              relevance_fn=lambda items, name: items[:1])   # e.g. a Gemini judge
    assert [r["url"] for r in rows] == ["https://1"]


def test_watchlist_news_relevance_filter_drops_offtopic():
    data = {"items": [
        {"title": "<b>삼성전자</b>, HBM 신규 라인 가동", "originallink": "https://hankyung.com/on",
         "link": "", "description": "삼성전자 투자", "pubDate": "Wed, 10 Jun 2026 09:00:00 +0900"},
        {"title": "삼성메디슨 초음파 수출 확대", "originallink": "https://x.com/off",
         "link": "", "description": "의료기기", "pubDate": "Wed, 10 Jun 2026 08:00:00 +0900"},
    ]}
    src = NaverNewsSource("CID", "SECRET", session=_Session(data))
    rows = src.watchlist_news(["005930"], per_symbol=5, names={"005930": "삼성전자"})
    # only the article actually mentioning 삼성전자 survives (삼성메디슨 is off-topic, dropped)
    assert [r["url"] for r in rows] == ["https://hankyung.com/on"]
    assert all("삼성전자" in (r["title"] + r["summary"]) for r in rows)
