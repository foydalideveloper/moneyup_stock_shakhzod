"""Order-execution log + panel — UTF-8 Korean roundtrip, parse/render, status, MOCK/LIVE badge."""

from tagent.order_log import load_order_events, log_order_event, order_panel, order_status


def test_order_log_utf8_roundtrip_and_panel(tmp_path):
    log_order_event(action="buy", stock="005930", side="BUY", qty=1, price=284000, ord_no="0001234",
                    return_code=0, return_msg="매수주문 정상 접수되었습니다", env="mock", accepted=True,
                    ts="2026-06-12T08:05:00+00:00", data_dir=tmp_path)
    log_order_event(action="cancel", stock="005930", side="BUY", qty=1, price=284000, ord_no="0001234",
                    return_code=0, return_msg="취소주문 접수", env="mock", accepted=True,
                    ts="2026-06-12T08:06:00+00:00", data_dir=tmp_path)
    evs = load_order_events(tmp_path)
    assert len(evs) == 2 and evs[0]["return_msg"] == "매수주문 정상 접수되었습니다"   # Korean clean (UTF-8)
    # the raw log file is UTF-8 (ensure_ascii=False) — not \uXXXX escapes
    raw = (tmp_path / "order_log.jsonl").read_text(encoding="utf-8")
    assert "매수주문 정상 접수되었습니다" in raw and "\\u" not in raw

    panel = order_panel(data_dir=tmp_path)
    assert panel["enabled"] is True and panel["n"] == 2
    newest = panel["orders"][0]                                       # newest-first -> the cancel
    assert newest["status"] == "CANCELLED" and newest["env"] == "mock" and newest["live"] is False
    assert newest["stock"] == "005930" and newest["side"] == "BUY" and newest["ord_no"] == "0001234"
    assert newest["time_kst"].endswith("17:06")                       # 08:06 UTC -> 17:06 KST


def test_order_status_and_live_badge():
    assert order_status({"action": "buy", "accepted": True}) == "ACCEPTED"
    assert order_status({"action": "cancel", "accepted": True}) == "CANCELLED"
    assert order_status({"action": "buy", "accepted": False}) == "REJECTED"
    assert order_status({"action": "fill", "accepted": True}) == "FILLED"
    live = order_panel([{"action": "buy", "stock": "X", "side": "BUY", "qty": 1, "price": 1,
                         "ord_no": "1", "return_code": 0, "return_msg": "", "env": "live",
                         "accepted": True, "ts": "2026-06-12T00:00:00+00:00"}])
    assert live["orders"][0]["live"] is True                          # LIVE badge
    assert order_panel([])["enabled"] is True and order_panel([])["n"] == 0   # honest empty
