"""Smallest possible Kiwoom check: auth + ONE 호가 REST snapshot (mock-first).

Confirms your key works BEFORE any streaming. REST works regardless of market
hours (WS streaming may be quiet outside KR market hours).

It does TWO things and nothing else:
  1) authenticate and print "token OK"  (NEVER prints the token or secret);
  2) request a 호가 (order book) snapshot for 005930 via REST api-id ka10004
     (POST {base}/api/dostk/mrkcond) and print the top bid/ask.

Usage:
    python scripts/kiwoom_smoke_test.py            # uses KIWOOM_ENV (default mock)
    python scripts/kiwoom_smoke_test.py --stk 000660
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from tagent.config import SETTINGS  # noqa: E402
from tagent.feeds.kiwoom_auth import KiwoomAuth, KiwoomAuthError  # noqa: E402
from tagent.feeds.kiwoom_feed import _to_number  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stk", default="005930", help="6-digit KRX code")
    args = ap.parse_args()

    if not SETTINGS.has_kiwoom_keys():
        print("Missing KIWOOM_APP_KEY / KIWOOM_SECRET_KEY in .env.")
        return 2

    env = SETTINGS.kiwoom_env
    base = SETTINGS.kiwoom_rest_url()
    print(f"Kiwoom env: {env}  ({base})")

    # 1) Authenticate.
    try:
        auth = KiwoomAuth(SETTINGS.kiwoom_app_key, SETTINGS.kiwoom_secret_key,
                          env=env, base_url=base)
        token = auth.get_token()
    except KiwoomAuthError as e:
        print(f"AUTH FAILED: {e}")
        return 1
    if not token:
        print("AUTH FAILED: empty token.")
        return 1
    print("token OK")  # never print the token itself

    # 2) ONE 호가 snapshot via REST (ka10004).
    import requests
    url = f"{base}/api/dostk/mrkcond"
    headers = {
        "Content-Type": "application/json;charset=UTF-8",
        "authorization": f"Bearer {token}",
        "api-id": "ka10004",  # 주식호가요청 (stock order book request)
    }
    try:
        resp = requests.post(url, headers=headers, json={"stk_cd": args.stk}, timeout=10)
        data = resp.json()
    except Exception as e:
        print(f"호가 request error: {e}")
        return 1

    if str(data.get("return_code")) not in ("0", "None"):
        print(f"호가 request rejected (return_code={data.get('return_code')}): "
              f"{data.get('return_msg')}")
        return 1

    # Best ask = 매도최우선호가 (sel_fpr_bid), best bid = 매수최우선호가 (buy_fpr_bid).
    best_ask = _to_number(data.get("sel_fpr_bid"))
    best_ask_qty = _to_number(data.get("sel_fpr_req"))
    best_bid = _to_number(data.get("buy_fpr_bid"))
    best_bid_qty = _to_number(data.get("buy_fpr_req"))

    print(f"\n=== {args.stk} order-book snapshot (REST ka10004) ===")
    print(f"  best bid: {best_bid:>12,.0f}  x {best_bid_qty:,.0f}")
    print(f"  best ask: {best_ask:>12,.0f}  x {best_ask_qty:,.0f}")
    if best_bid <= 0 and best_ask <= 0:
        print("  (no prices parsed — keys present in response: "
              f"{sorted(k for k in data if 'fpr' in k)})")
    print("\nREST OK. If this worked on mock, your key is good for streaming.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
