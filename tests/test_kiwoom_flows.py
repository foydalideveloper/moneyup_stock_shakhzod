"""Kiwoom 공매도/수급/프로그램 fetchers + report — mocked HTTP, no network, no secrets."""

import pandas as pd
import pytest

from tagent.data.kiwoom_flows import (
    KiwoomFlowsError, TR_SPECS, build_tr_request, fetch_foreign_trend, fetch_investor_net,
    fetch_program_trading, fetch_short_selling, parse_investor_net, parse_program, parse_short,
)
from tagent.kiwoom_report import (
    build_kiwoom_report, interpret_program, interpret_short, interpret_supply, render_report_text,
    tail_streak,
)

BASE = "https://mockapi.kiwoom.com"


class FakeResp:
    def __init__(self, data, headers=None):
        self._data, self.headers, self.status_code = data, headers or {}, 200

    def json(self):
        return self._data


class FakeAuth:
    def get_token(self):
        return "TOK"                                     # never a real secret


class ByApiId:
    """Dispatch POSTs by the api-id header so one session serves all TRs in a report."""
    def __init__(self, by_id):
        self.by_id, self.calls = by_id, []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json})
        return self.by_id.get(headers.get("api-id"), FakeResp({"return_code": 0}))


class Queue:
    """Returns queued responses in order (for continuation tests)."""
    def __init__(self, pages):
        self.pages, self.calls = list(pages), []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json})
        return self.pages[len(self.calls) - 1]


# --------------------------------------------------------------------------- #
# 1) request building per TR code (endpoint + api-id header + body)
# --------------------------------------------------------------------------- #
def test_build_tr_request_per_code_endpoint_and_header():
    for api_id, ep in [("ka10014", "/api/dostk/shsa"), ("ka10059", "/api/dostk/stkinfo"),
                       ("ka10008", "/api/dostk/frgnistt"), ("ka90013", "/api/dostk/stkinfo")]:
        url, h, b = build_tr_request(api_id, BASE, "TOK", {"stk_cd": "005930"},
                                     cont_yn="Y", next_key="K1")
        assert url == f"{BASE}{ep}" and TR_SPECS[api_id]["endpoint"] == ep
        assert h["api-id"] == api_id and h["authorization"] == "Bearer TOK"
        assert h["cont-yn"] == "Y" and h["next-key"] == "K1" and b["stk_cd"] == "005930"


def test_build_tr_request_rejects_unknown_code():
    with pytest.raises(ValueError):
        build_tr_request("ka99999", BASE, "TOK", {})


def test_fetchers_send_the_right_api_id_and_body():
    # ka10014 公매도: tm_tp + optional date window
    s = Queue([FakeResp({"return_code": 0, "shrts_trnsn": []})])
    fetch_short_selling("005930", auth=FakeAuth(), session=s, start_date="20260601", end_date="20260611")
    assert s.calls[0]["headers"]["api-id"] == "ka10014"
    assert s.calls[0]["url"] == f"{BASE}/api/dostk/shsa"
    assert s.calls[0]["json"]["strt_dt"] == "20260601" and s.calls[0]["json"]["end_dt"] == "20260611"
    # ka10060 variant honored
    s2 = Queue([FakeResp({"return_code": 0, "stk_invsr_orgn": []})])
    fetch_investor_net("005930", auth=FakeAuth(), session=s2, api_id="ka10060")
    assert s2.calls[0]["headers"]["api-id"] == "ka10060"
    # ka90005 시간대별 adds min_tic_tp
    s3 = Queue([FakeResp({"return_code": 0, "prm_trde_trnsn": []})])
    fetch_program_trading("005930", auth=FakeAuth(), session=s3, api_id="ka90005")
    assert s3.calls[0]["headers"]["api-id"] == "ka90005" and s3.calls[0]["json"]["min_tic_tp"] == "1"


def test_fetcher_raises_on_error_return_code():
    bad = Queue([FakeResp({"return_code": 8030, "return_msg": "mock/live mismatch"})])
    with pytest.raises(KiwoomFlowsError) as e:
        fetch_short_selling("005930", auth=FakeAuth(), session=bad)
    assert "8030" in str(e.value)


def test_continuation_follows_cont_yn_next_key():
    p1 = FakeResp({"return_code": 0, "shrts_trnsn": [
        {"dt": "20260610", "shrts_qty": "100", "shrts_wght": "1.0"}]},
        headers={"cont-yn": "Y", "next-key": "K1"})
    p2 = FakeResp({"return_code": 0, "shrts_trnsn": [
        {"dt": "20260611", "shrts_qty": "120", "shrts_wght": "1.2"}]},
        headers={"cont-yn": "N"})
    s = Queue([p1, p2])
    df, info = fetch_short_selling("005930", auth=FakeAuth(), session=s)
    assert info["pages"] == 2 and len(df) == 2
    assert s.calls[1]["headers"]["cont-yn"] == "Y" and s.calls[1]["headers"]["next-key"] == "K1"


# --------------------------------------------------------------------------- #
# 2) parsing — signed net flows, dates, defensive aliases, empty flag
# --------------------------------------------------------------------------- #
def test_parse_short_dates_and_fields():
    df = parse_short([{"dt": "20260610", "shrts_qty": "1000", "shrts_wght": "2.5", "cur_prc": "-74000"},
                      {"dt": "20260611", "shrts_qty": "1500", "shrts_wght": "3.1", "cur_prc": "73000"}])
    assert list(df.index) == [pd.Timestamp("2026-06-10"), pd.Timestamp("2026-06-11")]
    assert df["short_ratio"].tolist() == [2.5, 3.1] and df["short_qty"].tolist() == [1000.0, 1500.0]
    assert df["close"].tolist() == [74000.0, 73000.0]    # price magnitude (sign stripped)


def test_parse_investor_net_keeps_sign():
    df = parse_investor_net([
        {"dt": "20260610", "frgnr_invsr": "-5000", "orgn": "+2000", "ind_invsr": "+3000"},
        {"dt": "20260611", "frgnr_invsr": "-8000", "orgn": "-1000", "ind_invsr": "+9000"}])
    assert df["foreign_net"].tolist() == [-5000.0, -8000.0]   # SIGN preserved (순매도)
    assert df["inst_net"].tolist() == [2000.0, -1000.0]


def test_parse_program_alias_and_fallback_list():
    # uses an alias array key + signed net
    df = parse_program([{"dt": "20260611", "prm_netprps_amt": "-1234"}])
    assert df["program_net"].tolist() == [-1234.0]
    # first-list fallback when the named arrays are absent
    from tagent.data.kiwoom_flows import _extract_list
    assert _extract_list({"weird_key": [{"a": 1}]}, ("nope",)) == [{"a": 1}]


def test_fetch_flags_empty_on_mock_blank():
    s = Queue([FakeResp({"return_code": 0, "shrts_trnsn": []})])
    df, info = fetch_short_selling("005930", auth=FakeAuth(), session=s)
    assert df.empty and info["empty"] is True and "live account" in info["note"]


def test_foreign_trend_parses_holding():
    s = Queue([FakeResp({"return_code": 0, "stk_frgnr": [
        {"dt": "20260611", "cur_prc": "74000", "ntby_qty": "-3000", "poss_stkcnt": "1000000",
         "wght": "51.2"}]})])
    df, info = fetch_foreign_trend("005930", auth=FakeAuth(), session=s)
    assert df["foreign_net"].iloc[0] == -3000.0 and df["holding_ratio"].iloc[0] == 51.2
    assert info["empty"] is False


# --------------------------------------------------------------------------- #
# 3) interpretation logic — the one-line reads
# --------------------------------------------------------------------------- #
def test_tail_streak():
    assert tail_streak([1, -1, -2, -3]) == (-1, 3)        # 3-day net-sell run
    assert tail_streak([1, 2, 3]) == (1, 3)
    assert tail_streak([5, -1, 0]) == (0, 0)              # latest zero -> no streak
    assert tail_streak([]) == (0, 0)


def test_interpret_supply_foreign_sell_streak():
    df = pd.DataFrame({"foreign_net": [-1, -2, -3], "inst_net": [-1, -1, -2]},
                      index=pd.to_datetime(["2026-06-09", "2026-06-10", "2026-06-11"]))
    r = interpret_supply(df)
    assert "외국인 3일 연속 순매도" in r["text"] and "매도 압력" in r["text"]
    assert r["status"] == "bearish" and r["informative"] is True


def test_interpret_supply_mixed_is_low_priority():
    df = pd.DataFrame({"foreign_net": [5, 6], "inst_net": [-5, -6]},
                      index=pd.to_datetime(["2026-06-10", "2026-06-11"]))
    r = interpret_supply(df)
    assert "수급 엇갈림" in r["text"] and r["informative"] is False   # not a headline


def test_interpret_short_rising_and_empty():
    df = pd.DataFrame({"short_ratio": [1.0, 2.0, 3.0], "short_qty": [1, 2, 3]},
                      index=pd.to_datetime(["2026-06-09", "2026-06-10", "2026-06-11"]))
    r = interpret_short(df)
    assert "공매도 비중 3일 연속 증가" in r["text"] and "매도 압력" in r["text"] and r["status"] == "bearish"
    e = interpret_short(pd.DataFrame())
    assert e["text"] is None and e["empty"] is True and e["status"] == "no-data"


def test_interpret_program_net_buy_with_streak():
    df = pd.DataFrame({"program_net": [100, 200, 300]},
                      index=pd.to_datetime(["2026-06-09", "2026-06-10", "2026-06-11"]))
    r = interpret_program(df)
    assert "프로그램 순매수 유입" in r["text"] and "3일 연속" in r["text"] and r["informative"] is True
    assert interpret_program(pd.DataFrame())["text"] is None


# --------------------------------------------------------------------------- #
# 4) end-to-end report — three TRs per stock + honest empty flagging
# --------------------------------------------------------------------------- #
def test_build_kiwoom_report_combines_three_trs_and_flags_empty():
    by_id = {
        "ka10014": FakeResp({"return_code": 0, "shrts_trnsn": [
            {"dt": "20260609", "shrts_wght": "1.0"}, {"dt": "20260610", "shrts_wght": "2.0"},
            {"dt": "20260611", "shrts_wght": "3.0"}]}),
        "ka10059": FakeResp({"return_code": 0, "stk_invsr_orgn": [
            {"dt": "20260609", "frgnr_invsr": "-1", "orgn": "-1"},
            {"dt": "20260610", "frgnr_invsr": "-2", "orgn": "-1"},
            {"dt": "20260611", "frgnr_invsr": "-3", "orgn": "-2"}]}),
        "ka90013": FakeResp({"return_code": 0, "prm_trde_trnsn": []}),   # empty on mock -> flagged
    }
    sess = ByApiId(by_id)
    payload = build_kiwoom_report(["005930"], auth=FakeAuth(), session=sess, env="mock")
    card = payload["stocks"][0]
    assert "공매도 비중 3일 연속 증가" in card["short"]["text"]
    assert "외국인 3일 연속 순매도" in card["supply"]["text"]
    assert card["program"]["empty"] is True
    assert "ka90013" in payload["empty_endpoints"]        # honest: not faked
    text = render_report_text(payload)
    assert "공매도" in text and "수급" in text and "실계좌" in text


def test_report_flags_rejected_tr_honestly_not_neutral():
    """A REJECTED TR must be flagged (empty + the real reason), never shown as a silent neutral —
    the honesty guard that mirrors the live mock behaviour (ka10059 needs 'dt', etc.)."""
    class Dispatch:
        def post(self, url, headers=None, json=None, timeout=None):
            if headers.get("api-id") == "ka10014":            # returns rows (no streak) -> neutral
                return FakeResp({"return_code": 0, "shrts_trnsn": [
                    {"dt": "20260610", "shrts_wght": "1.0"}, {"dt": "20260611", "shrts_wght": "1.0"}]})
            return FakeResp({"return_code": 2, "return_msg": "입력 값 오류[필수입력 파라미터=dt]"})
    rep = build_kiwoom_report(["005930"], auth=FakeAuth(), session=Dispatch(), env="mock")
    card = rep["stocks"][0]
    # ka10059 수급 + ka90013 프로그램 were REJECTED -> flagged empty WITH the reason, not neutral
    assert card["supply"]["empty"] is True and "거부" in card["supply"]["note"]
    assert card["program"]["empty"] is True and "거부" in card["program"]["note"]
    assert "ka10059" in rep["empty_endpoints"] and "ka90013" in rep["empty_endpoints"]
    # ka10014 returned data with no notable trend -> a genuine (honest) neutral, text None
    assert card["short"]["empty"] is False and card["short"]["text"] is None
