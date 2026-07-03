"""Kiwoom REST order placement — 매수 / 취소 (MOCK 모의투자 only).

Order TRs (openapi.kiwoom.com → 국내주식 → 주문), same auth/header pattern as ka10080/the flow
TRs: ``POST {base}/api/dostk/ordr`` with headers ``api-id`` + ``authorization: Bearer <token>``.

  * kt10000  주식 매수주문   body {dmst_stex_tp, stk_cd, ord_qty, ord_uv, trde_tp, cond_uv}
  * kt10001  주식 매도주문   (same body shape)
  * kt10003  주식 취소주문   body {dmst_stex_tp, orig_ord_no, stk_cd, cncl_qty}

``trde_tp``: "0" 보통(지정가/limit), "3" 시장가(market). ``ord_uv`` is the limit price (blank for
market). The server returns the 주문번호 (``ord_no``) + return_code/return_msg.

SAFETY: this module is **MOCK-ONLY**. Every place/cancel asserts ``env == 'mock'`` and refuses
otherwise, so it can never fire a live order. The token rides only in the Authorization header and
is never logged. ``requests`` is injected in tests so nothing hits the network.
"""

from __future__ import annotations

from typing import Optional, Tuple

from tagent.config import KIWOOM_REST_URLS

ORDER_ENDPOINT = "/api/dostk/ordr"
BUY_TR = "kt10000"
SELL_TR = "kt10001"
CANCEL_TR = "kt10003"

# 매매구분 (trde_tp) codes used here.
TRDE_LIMIT = "0"            # 보통 (지정가)
TRDE_MARKET = "3"          # 시장가
TRDE_AFTER_HOURS = "62"    # 시간외단일가 (after-hours single-price auction, 16:00–18:00 KST)

# response aliases (defensive): the 주문번호 may come back under a few names.
_ORDNO_KEYS = ("ord_no", "odno", "order_no", "ordNo", "ord_no_str")


class KiwoomOrderError(RuntimeError):
    """Raised on a transport/JSON failure, or any attempt to order outside the mock env."""


def _assert_mock(env: str) -> None:
    if str(env).lower() != "mock":
        raise KiwoomOrderError(
            f"Refusing to place an order in env={env!r}: this module is MOCK-ONLY "
            "(모의투자). Live order placement is intentionally not supported here.")


def _requests():
    try:
        import requests
    except ImportError as e:  # pragma: no cover
        raise KiwoomOrderError("requests not installed. Run: pip install -r requirements.txt") from e
    return requests


def build_order_request(base_url: str, token: str, api_id: str, body: dict,
                        cont_yn: str = "N", next_key: str = "") -> Tuple[str, dict, dict]:
    """(url, headers, body) for one order TR — same header shape as the data TRs."""
    url = f"{base_url}{ORDER_ENDPOINT}"
    headers = {
        "Content-Type": "application/json;charset=UTF-8",
        "authorization": f"Bearer {token}",
        "api-id": api_id,
        "cont-yn": cont_yn,
        "next-key": next_key,
    }
    return url, headers, dict(body)


def buy_body(stk_cd: str, qty: int, price: Optional[float] = None, *, exchange: str = "KRX",
             trde_tp: Optional[str] = None) -> dict:
    """Body for kt10000 매수. ``price`` None/blank -> 시장가(market, trde_tp 3); else 지정가(limit, 0).
    ``trde_tp`` overrides the auto choice (e.g. "62" 시간외단일가 for the after-hours session)."""
    is_market = price is None or str(price) == ""
    return {
        "dmst_stex_tp": exchange,
        "stk_cd": str(stk_cd),
        "ord_qty": str(int(qty)),
        "ord_uv": "" if is_market else str(int(round(float(price)))),
        "trde_tp": str(trde_tp) if trde_tp is not None else (TRDE_MARKET if is_market else TRDE_LIMIT),
        "cond_uv": "",
    }


def cancel_body(orig_ord_no: str, stk_cd: str, cancel_qty: int = 0, *, exchange: str = "KRX") -> dict:
    """Body for kt10003 취소. ``cancel_qty`` 0 -> 전량(all)."""
    return {
        "dmst_stex_tp": exchange,
        "orig_ord_no": str(orig_ord_no),
        "stk_cd": str(stk_cd),
        "cncl_qty": "0" if int(cancel_qty) == 0 else str(int(cancel_qty)),
    }


def parse_order_result(data: dict) -> dict:
    """Order TR JSON -> {ord_no, return_code, return_msg, accepted}. ``accepted`` is True only when
    return_code == 0 (the server took the order). Never raises — a business reject is returned so
    the caller can report it honestly."""
    if not isinstance(data, dict):
        return {"ord_no": "", "return_code": None, "return_msg": "non-dict response", "accepted": False}
    ord_no = ""
    for k in _ORDNO_KEYS:
        v = data.get(k)
        if v not in (None, ""):
            ord_no = str(v).strip()
            break
    code = data.get("return_code")
    return {"ord_no": ord_no, "return_code": code,
            "return_msg": data.get("return_msg", ""), "accepted": str(code) == "0"}


def _post(api_id: str, body: dict, *, auth, session=None, env: str = "mock",
          base_url: Optional[str] = None) -> dict:
    _assert_mock(env)                                    # hard safety: mock only
    base = base_url or KIWOOM_REST_URLS.get(env.lower(), KIWOOM_REST_URLS["mock"])
    token = auth.get_token()
    sess = session or _requests()
    url, headers, b = build_order_request(base, token, api_id, body)
    resp = sess.post(url, headers=headers, json=b, timeout=10)
    try:
        data = resp.json()
    except Exception as e:
        raise KiwoomOrderError(f"{api_id} returned non-JSON: {e}")
    return parse_order_result(data)


def place_buy(stk_cd: str, qty: int = 1, price: Optional[float] = None, *, auth, session=None,
              env: str = "mock", base_url: Optional[str] = None, exchange: str = "KRX",
              trde_tp: Optional[str] = None) -> dict:
    """Submit ONE 매수 order (kt10000) on the MOCK server. Returns the parsed result (incl. the
    주문번호 and return_code) — a reject is reported, not raised."""
    return _post(BUY_TR, buy_body(stk_cd, qty, price, exchange=exchange, trde_tp=trde_tp),
                 auth=auth, session=session, env=env, base_url=base_url)


def cancel(orig_ord_no: str, stk_cd: str, cancel_qty: int = 0, *, auth, session=None,
           env: str = "mock", base_url: Optional[str] = None, exchange: str = "KRX") -> dict:
    """Submit a 취소주문 (kt10003) for a prior order on the MOCK server."""
    return _post(CANCEL_TR, cancel_body(orig_ord_no, stk_cd, cancel_qty, exchange=exchange),
                 auth=auth, session=session, env=env, base_url=base_url)


# --------------------------------------------------------------------------- #
# KR price-tick snap (so a resting limit order doesn't reject on 호가단위)
# --------------------------------------------------------------------------- #
def kr_tick(price: float) -> int:
    """KRX tick size for a price (post-2023 bands)."""
    p = float(price)
    for hi, tick in ((2_000, 1), (5_000, 5), (20_000, 10), (50_000, 50),
                     (200_000, 100), (500_000, 500)):
        if p < hi:
            return tick
    return 1_000


def snap_to_tick(price: float, *, down: bool = True) -> int:
    """Round ``price`` to a valid KRX tick (down by default, for a safe resting buy)."""
    t = kr_tick(price)
    n = int(float(price) // t) if down else int(round(float(price) / t))
    return max(t, n * t)
