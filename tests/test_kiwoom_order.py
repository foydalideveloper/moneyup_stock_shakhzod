"""Kiwoom MOCK order placement — request building, parse, cancel, mock-only guard. Mocked HTTP."""

import pytest

from tagent.kiwoom_order import (
    BUY_TR, CANCEL_TR, KiwoomOrderError, build_order_request, buy_body, cancel, cancel_body,
    kr_tick, parse_order_result, place_buy, snap_to_tick,
)

BASE = "https://mockapi.kiwoom.com"


class FakeResp:
    def __init__(self, data):
        self._data, self.status_code, self.headers = data, 200, {}

    def json(self):
        return self._data


class FakeSession:
    """Dispatches order POSTs by api-id header and records every call."""
    def __init__(self, by_id):
        self.by_id, self.calls = by_id, []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json})
        return FakeResp(self.by_id.get(headers.get("api-id"), {"return_code": 0, "ord_no": "X"}))


class FakeAuth:
    def get_token(self):
        return "TOK"                                     # never a real secret


# --------------------------------------------------------------------------- #
# 1) request / body building
# --------------------------------------------------------------------------- #
def test_build_order_request_endpoint_and_headers():
    url, h, b = build_order_request(BASE, "TOK", BUY_TR, {"stk_cd": "005930"})
    assert url == f"{BASE}/api/dostk/ordr"
    assert h["api-id"] == "kt10000" and h["authorization"] == "Bearer TOK"
    assert h["cont-yn"] == "N" and b["stk_cd"] == "005930"


def test_buy_body_market_vs_limit():
    m = buy_body("005930", 1)                            # no price -> 시장가
    assert m["trde_tp"] == "3" and m["ord_uv"] == "" and m["ord_qty"] == "1"
    assert m["dmst_stex_tp"] == "KRX" and m["stk_cd"] == "005930"
    lim = buy_body("005930", 1, price=70000)             # price -> 지정가
    assert lim["trde_tp"] == "0" and lim["ord_uv"] == "70000"
    from tagent.kiwoom_order import TRDE_AFTER_HOURS     # 시간외단일가 override
    ah = buy_body("005930", 1, price=284000, trde_tp=TRDE_AFTER_HOURS)
    assert ah["trde_tp"] == "62" and ah["ord_uv"] == "284000"


def test_cancel_body_all_quantity():
    c = cancel_body("0001234", "005930", 0)
    assert c["orig_ord_no"] == "0001234" and c["cncl_qty"] == "0" and c["stk_cd"] == "005930"
    assert cancel_body("0001234", "005930", 1)["cncl_qty"] == "1"


# --------------------------------------------------------------------------- #
# 2) response parsing (accepted vs rejected, order-id aliases)
# --------------------------------------------------------------------------- #
def test_parse_order_result_accepted_and_rejected():
    ok = parse_order_result({"return_code": 0, "return_msg": "정상", "ord_no": "0001234"})
    assert ok["accepted"] is True and ok["ord_no"] == "0001234" and ok["return_code"] == 0
    rej = parse_order_result({"return_code": 3, "return_msg": "주문가능금액 부족", "ord_no": ""})
    assert rej["accepted"] is False and rej["return_msg"] == "주문가능금액 부족"
    assert parse_order_result({"return_code": 0, "odno": "999"})["ord_no"] == "999"   # alias


# --------------------------------------------------------------------------- #
# 3) place + cancel round trip (mocked HTTP)
# --------------------------------------------------------------------------- #
def test_place_buy_then_cancel_roundtrip():
    sess = FakeSession({
        "kt10000": {"return_code": 0, "return_msg": "매수주문 접수", "ord_no": "0001234"},
        "kt10003": {"return_code": 0, "return_msg": "취소주문 접수", "ord_no": "0001235"},
    })
    bought = place_buy("005930", 1, price=70000, auth=FakeAuth(), session=sess, env="mock")
    assert bought["accepted"] is True and bought["ord_no"] == "0001234"
    # the buy hit kt10000 with our body
    assert sess.calls[0]["headers"]["api-id"] == "kt10000"
    assert sess.calls[0]["json"]["ord_qty"] == "1" and sess.calls[0]["json"]["ord_uv"] == "70000"
    cancelled = cancel(bought["ord_no"], "005930", auth=FakeAuth(), session=sess, env="mock")
    assert cancelled["accepted"] is True
    assert sess.calls[1]["headers"]["api-id"] == "kt10003"
    assert sess.calls[1]["json"]["orig_ord_no"] == "0001234"     # cancels the order we placed


def test_place_buy_reports_reject_without_raising():
    sess = FakeSession({"kt10000": {"return_code": 8030, "return_msg": "모의투자 미신청 계좌"}})
    res = place_buy("005930", 1, price=70000, auth=FakeAuth(), session=sess, env="mock")
    assert res["accepted"] is False and res["return_code"] == 8030 and "모의투자" in res["return_msg"]


# --------------------------------------------------------------------------- #
# 4) MOCK-ONLY safety guard + tick snap
# --------------------------------------------------------------------------- #
def test_orders_refused_outside_mock():
    for fn in (lambda: place_buy("005930", 1, auth=FakeAuth(), session=FakeSession({}), env="live"),
               lambda: cancel("1", "005930", auth=FakeAuth(), session=FakeSession({}), env="real")):
        with pytest.raises(KiwoomOrderError):
            fn()


def test_kr_tick_and_snap_down_to_valid_tick():
    assert kr_tick(70000) == 100 and kr_tick(3000) == 5 and kr_tick(300000) == 500
    assert snap_to_tick(70123) == 70100                  # 100-won band, rounded DOWN
    assert snap_to_tick(2999) == 2995                    # 5-won band
