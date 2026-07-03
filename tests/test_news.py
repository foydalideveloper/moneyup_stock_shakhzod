"""News/alerts layer (OpenDART + Finnhub) — mocked HTTP, no network, no secrets."""

from tagent.news.alerts import (
    NewsAlertConfig, build_news_payload, combine_alerts, safety_signal,
)
from tagent.news.finnhub_source import (
    FinnhubSource, aggregate_sentiment, headline_sentiment, normalize_news,
    sentiment_label,
)
import io
import zipfile

from tagent.news.opendart_source import (
    OpenDartSource, classify_importance, ensure_corp_map, guess_sentiment,
    load_corp_map, parse_corpcode_zip, parse_list_response, save_corp_map,
)


class FakeResp:
    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data


class FakeGet:
    """Records each GET and returns queued responses (for OpenDART/Finnhub)."""
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": params})
        return FakeResp(self.responses[len(self.calls) - 1])


def _od_item(report, code="005930", dt="20260609", rcept="20260609000123"):
    return {"report_nm": report, "stock_code": code, "rcept_dt": dt, "rcept_no": rcept,
            "corp_name": "회사", "flr_nm": "회사"}


# --------------------------------------------------------------------------- #
# OpenDART rule tagging
# --------------------------------------------------------------------------- #
def test_classify_importance_rules():
    assert classify_importance("주요사항보고서(유상증자결정)") == "dilution"
    assert classify_importance("단일판매ㆍ공급계약체결") == "contract"
    assert classify_importance("[기재정정]사업보고서") == "correction"     # 정정 wins
    assert classify_importance("분기보고서 (2026.03)") == "earnings"
    assert classify_importance("자기주식취득결정") == "buyback"
    assert classify_importance("상장폐지 관련 안내") == "distress"
    assert classify_importance("그냥 일반 공시") == "other"


def test_guess_sentiment_rules():
    assert guess_sentiment("유상증자 결정") == "bearish"
    assert guess_sentiment("자기주식 취득 결정") == "bullish"
    assert guess_sentiment("단일판매ㆍ공급계약체결") == "bullish"
    assert guess_sentiment("[정정] 무엇이든") == "neutral"
    assert guess_sentiment("분기보고서") == "neutral"


def test_parse_list_response_status():
    assert parse_list_response({"status": "013", "message": "no data"}) == []
    assert parse_list_response({"status": "000", "list": [_od_item("x")]}) != []


def test_opendart_recent_disclosures_filters_and_sorts():
    data = {"status": "000", "total_page": 1, "list": [
        _od_item("주요사항보고서(유상증자결정)", "005930", "20260609"),
        _od_item("단일판매ㆍ공급계약체결", "000660", "20260608"),
        _od_item("아무공시", "999999", "20260609"),            # not in watchlist
    ]}
    src = OpenDartSource("SECRETKEY", session=FakeGet([data]))
    res = src.recent_disclosures(["005930", "000660"], "20260601", "20260609", max_pages=1)
    assert len(res) == 2                                       # 999999 filtered out
    assert res[0]["symbol"] == "005930" and res[0]["time"] == "2026-06-09"   # newest first
    assert res[0]["type"] == "dilution" and res[0]["sentiment"] == "bearish" and res[0]["important"]
    assert res[1]["type"] == "contract" and res[1]["sentiment"] == "bullish"
    assert res[0]["url"].startswith("https://dart.fss.or.kr/")
    # the key travels only in the request params (never logged elsewhere)
    assert src._session.calls[0]["params"]["crtfc_key"] == "SECRETKEY"


def _corpcode_zip(entries):
    """entries: list of (corp_code, corp_name, stock_code) -> corpCode.xml ZIP bytes."""
    xml = "<?xml version='1.0' encoding='utf-8'?><result>" + "".join(
        f"<list><corp_code>{c}</corp_code><corp_name>{n}</corp_name>"
        f"<stock_code>{s}</stock_code><modify_date>20260101</modify_date></list>"
        for c, n, s in entries) + "</result>"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("CORPCODE.xml", xml.encode("utf-8"))
    return buf.getvalue()


class FakeZipGet:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": params})
        return type("R", (), {"content": self.content})()


def test_parse_corpcode_zip_keeps_listed_only():
    content = _corpcode_zip([("00126380", "삼성전자", "005930"),
                             ("00164779", "비상장회사", "       "),     # blank stock_code -> skip
                             ("00111111", "다른상장", "000660")])
    m = parse_corpcode_zip(content)
    assert m == {"005930": "00126380", "000660": "00111111"}


def test_recent_disclosures_uses_corp_code_path():
    data = {"status": "000", "list": [_od_item("주요사항보고서(유상증자결정)", "005930", "20260609")]}
    src = OpenDartSource("SECRETKEY", session=FakeGet([data]))
    res = src.recent_disclosures(["005930"], "20260601", "20260609",
                                 corp_map={"005930": "00126380"})
    assert len(res) == 1 and res[0]["symbol"] == "005930" and res[0]["type"] == "dilution"
    # queried BY corp_code (not the all-market page-filter path)
    assert src._session.calls[0]["params"]["corp_code"] == "00126380"
    assert "page_no" in src._session.calls[0]["params"]


def test_ensure_corp_map_caches_and_fetches(tmp_path):
    # cached map is returned without any network/session
    save_corp_map({"005930": "00126380"}, data_dir=tmp_path)
    assert load_corp_map(data_dir=tmp_path) == {"005930": "00126380"}
    assert ensure_corp_map("KEY", data_dir=tmp_path) == {"005930": "00126380"}
    # empty cache -> downloads + parses + caches via the injected session
    fresh_dir = tmp_path / "fresh"
    fake = FakeZipGet(_corpcode_zip([("00126380", "삼성전자", "005930")]))
    got = ensure_corp_map("KEY", data_dir=fresh_dir, session=fake, refresh=True)
    assert got == {"005930": "00126380"}
    assert fake.calls[0]["params"]["crtfc_key"] == "KEY"          # key only in params
    assert load_corp_map(data_dir=fresh_dir) == {"005930": "00126380"}   # cached for next time


# --------------------------------------------------------------------------- #
# Finnhub rule sentiment
# --------------------------------------------------------------------------- #
def test_headline_sentiment_and_label():
    assert headline_sentiment("NVDA beats estimates, shares surge") > 0
    assert headline_sentiment("Company misses, shares plunge on downgrade") < 0
    assert headline_sentiment("Company holds annual shareholder meeting") == 0.0
    assert sentiment_label(0.5) == "bullish" and sentiment_label(-0.5) == "bearish"
    assert sentiment_label(0.0) == "neutral"


def test_finnhub_company_news_parses_sorts_and_passes_token():
    rows = [
        {"datetime": 1718000000, "headline": "NVDA beats and shares surge", "url": "u1", "source": "Reuters"},
        {"datetime": 1718100000, "headline": "NVDA faces lawsuit, shares plunge", "url": "u2", "source": "WSJ"},
    ]
    fh = FinnhubSource("TOKEN", session=FakeGet([rows]))
    res = fh.company_news("nvda", "2026-06-01", "2026-06-09")
    assert res[0]["ts"] == 1718100000                          # newest first
    assert res[0]["sentiment"] == "bearish" and res[1]["sentiment"] == "bullish"
    assert res[0]["symbol"] == "NVDA"
    assert fh._session.calls[0]["params"]["token"] == "TOKEN"  # key only in params


def test_finnhub_news_sentiment_mapping():
    assert FinnhubSource("T", session=FakeGet([{"companyNewsScore": 0.8}])).news_sentiment("NVDA") == 0.6
    assert FinnhubSource("T", session=FakeGet([{}])).news_sentiment("NVDA") is None
    fh = FinnhubSource("T", session=FakeGet([{"sentiment": {"bullishPercent": 0.3, "bearishPercent": 0.7}}]))
    assert fh.news_sentiment("NVDA") == -0.4


def test_aggregate_sentiment():
    assert aggregate_sentiment([{"sentiment_score": 0.5}, {"sentiment_score": -0.1}]) == 0.2
    assert aggregate_sentiment([]) == 0.0


# --------------------------------------------------------------------------- #
# combined feed + safety kill-switch
# --------------------------------------------------------------------------- #
def test_combine_alerts_newest_first():
    kr = [{"ts": 100, "market": "kr"}, {"ts": 300, "market": "kr"}]
    us = [{"ts": 200, "market": "us"}]
    assert [a["ts"] for a in combine_alerts(kr, us)] == [300, 200, 100]


def test_safety_signal_thresholds():
    cfg = NewsAlertConfig(overnight_halt=-0.03, sentiment_halt=-0.40)
    assert safety_signal(0.1, -0.05, cfg)["state"] == "HALT"   # overnight breach
    assert safety_signal(-0.5, 0.0, cfg)["state"] == "HALT"    # sentiment breach
    assert safety_signal(0.1, -0.01, cfg)["state"] == "OK"
    assert safety_signal(None, None, cfg)["state"] == "OK"     # missing data -> OK (no false halt)
    both = safety_signal(-0.5, -0.05, cfg)
    assert both["halt"] is True and len(both["reasons"]) == 2


def test_build_news_payload_combines_and_halts():
    kr = [{"ts": 100, "market": "kr", "title": "t"}]
    us = [{"ts": 200, "market": "us", "sentiment_score": -0.6}]
    p = build_news_payload(kr, us, overnight_return=0.0, cfg=NewsAlertConfig(sentiment_halt=-0.4))
    assert p["enabled"] and p["n_kr"] == 1 and p["n_us"] == 1
    assert p["safety"]["state"] == "HALT"                      # mean linked sentiment -0.6 <= -0.4
    assert p["alerts"][0]["ts"] == 200                         # newest first


def test_settings_has_news_key_helpers():
    from tagent.config import Settings
    s = Settings()
    assert hasattr(s, "opendart_api_key") and hasattr(s, "finnhub_api_key")
    assert isinstance(s.has_opendart_key(), bool) and isinstance(s.has_finnhub_key(), bool)
