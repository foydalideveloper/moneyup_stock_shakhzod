"""Differentiated recommendation blend — momentum core + Naver broker + 13F whales. Mocked."""

import pandas as pd

from tagent.daily_briefing import citation
from tagent.recommendation import (
    build_recommendation_live, momentum_picks, naver_broker_picks, whale_13f_picks,
)


def _panel(closes):
    idx = pd.date_range("2026-01-01", periods=len(next(iter(closes.values()))), freq="B")
    return {s: pd.DataFrame({"close": v}, index=idx) for s, v in closes.items()}


class _Naver:
    """Mock Naver returning canned broker-research articles per query."""
    def __init__(self, by_query): self.by_query, self.calls = by_query, []

    def search(self, query, display=10, sort="date"):
        self.calls.append(query)
        return list(self.by_query.get(query, []))


def _art(title, url, summary=""):
    return {"title": title, "url": url, "summary": summary, "ts": 1.0, "sentiment": "neutral"}


# --------------------------------------------------------------------------- #
# CORE — 12-1 momentum picks (cited "our momentum model")
# --------------------------------------------------------------------------- #
def test_momentum_picks_rank_by_12_1_and_cite():
    # 8 bars; with lookback=5, skip=1 the score = close[-2]/close[-7]-1
    closes = {
        "AAA": [100, 102, 104, 106, 108, 150, 160, 170],   # strong up
        "BBB": [100, 100, 100, 100, 100, 101, 101, 101],   # flat
        "CCC": [100, 98, 96, 94, 92, 70, 68, 66],          # down
    }
    picks = momentum_picks(["AAA", "BBB", "CCC"], panel=_panel(closes), lookback=5, skip=1, top_n=2)
    assert [p["symbol"] for p in picks] == ["AAA", "BBB"]   # ranked by momentum, top-2
    assert picks[0]["rank"] == 1 and picks[0]["score"] > picks[1]["score"]
    assert picks[0]["side"] == "buy" and "our-momentum-model" in picks[0]["cite"]
    assert "모멘텀" in picks[0]["reason"]
    # a name without enough history is skipped (not invented)
    short = momentum_picks(["DDD"], panel=_panel({"DDD": [1, 2, 3]}), lookback=5, skip=1)
    assert short == []


# --------------------------------------------------------------------------- #
# PROVIDER — Naver broker picks (per-broker attribution, cited, off-topic dropped)
# --------------------------------------------------------------------------- #
def test_naver_broker_picks_attribute_house_and_drop_offtopic():
    naver = _Naver({
        "삼성전자 목표주가": [
            _art("미래에셋 '삼성전자' 목표주가 10만원 상향", "https://mk.co.kr/r1"),
            _art("삼성메디슨 초음파 수출", "https://x.com/off"),        # off-topic (no 삼성전자) -> dropped
            _art("'삼성전자' 매도 의견 하향", "https://hk.com/r2"),
            _art("링크 없는 삼성전자 리포트", ""),                       # uncited -> dropped
        ]})
    prov = naver_broker_picks(naver, ["005930"], names={"005930": "삼성전자"})
    picks = prov["picks"]
    assert {p["url"] for p in picks} == {"https://mk.co.kr/r1", "https://hk.com/r2"}
    assert any(p["house"] == "미래에셋증권" and p["side"] == "buy" for p in picks)
    assert any(p["side"] == "sell" for p in picks)        # 매도/하향 -> sell


# --------------------------------------------------------------------------- #
# PROVIDER — 13F whales (cited whalewisdom URL or dropped)
# --------------------------------------------------------------------------- #
def test_whale_13f_picks_cited_or_dropped():
    holdings = {
        "Berkshire": {"005930": {"url": "https://whalewisdom.com/b1", "shares": 1000000}},
        "BlackRock": {"000660": {"url": ""}},              # no url -> dropped
        "Vanguard": {"999999": {"url": "https://whalewisdom.com/v1"}},  # not in watchlist -> skipped
    }
    prov = whale_13f_picks(["005930", "000660"], holdings=holdings)
    assert [p["symbol"] for p in prov["picks"]] == ["005930"]
    p = prov["picks"][0]
    assert p["house"] == "Berkshire" and p["url"] == "https://whalewisdom.com/b1"
    assert whale_13f_picks(["005930"], holdings={})["picks"] == []   # empty -> no coverage


# --------------------------------------------------------------------------- #
# BLEND — differentiated, cited, distinct-from-Kiwoom, with agreement
# --------------------------------------------------------------------------- #
def test_build_recommendation_live_blends_and_is_cited():
    closes = {"005930": [100, 101, 102, 103, 104, 140, 150, 160],
              "000660": [100, 100, 100, 100, 100, 120, 121, 122],
              "035420": [100, 99, 98, 97, 96, 80, 79, 78]}
    naver = _Naver({"삼성전자 목표주가": [_art("미래에셋 '삼성전자' 목표가 상향", "https://mk.co.kr/s")],
                    "SK하이닉스 목표주가": [], "네이버 목표주가": []})
    rep = build_recommendation_live(
        ["005930", "000660", "035420"], naver=naver, panel=_panel(closes),
        whale_holdings={"BlackRock": {"005930": {"url": "https://whalewisdom.com/x"}}},
        kiwoom_picks=[], today="2026-06-11", lookback=5, skip=1)
    by = {it["symbol"]: it for it in rep["items"]}
    assert rep["n"] >= 1 and "005930" in by
    # 005930 has momentum + 미래에셋 + BlackRock -> multiple sources, all buy -> 동의
    assert by["005930"]["our_view"] and "미래에셋증권" in by["005930"]["houses"]
    assert by["005930"]["agreement"] == "동의" and by["005930"]["distinct_from_kiwoom"] is True
    # every surfaced pick carries a citation (url or provenance)
    assert all(citation(it) for it in rep["items"])
    # momentum alone still produces cited picks (not "no coverage")
    assert rep["note"] == ""
