"""ONE-OFF proof of programmatic order execution on the Kiwoom MOCK (모의투자) server.

Places a single small resting BUY (1 share, limit a few % below the bid so it won't fill),
prints the order RESULT (주문번호 + return_code + message), then CANCELS it. MOCK ENV ONLY — the
order module refuses any non-mock env. Secrets are never printed. NOT live trading.

Usage: python scripts/run_kiwoom_order_proof.py            # 005930 Samsung, 1 share
       python scripts/run_kiwoom_order_proof.py --stk 000660
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from tagent.config import SETTINGS  # noqa: E402
from tagent.feeds.kiwoom_auth import KiwoomAuth, KiwoomAuthError  # noqa: E402
from tagent.feeds.kiwoom_feed import _to_number  # noqa: E402
from tagent.kiwoom_order import (  # noqa: E402
    TRDE_AFTER_HOURS, KiwoomOrderError, cancel, place_buy, snap_to_tick)


def _session_trde_tp():
    """Pick the order type for the current KR session: regular limit, or 시간외단일가 after close
    (16:00–18:00 KST). Returns (trde_tp_override_or_None, label)."""
    from datetime import datetime, timedelta, timezone
    h = datetime.now(timezone(timedelta(hours=9)))
    mins = h.hour * 60 + h.minute
    if 9 * 60 <= mins < 15 * 60 + 30:
        return None, "regular-hours limit"
    if 16 * 60 <= mins < 18 * 60:
        return TRDE_AFTER_HOURS, "시간외단일가 (after-hours single-price)"
    return None, "market closed (regular limit; mock will likely reject 장종료)"


def _best_bid(base, token, stk):
    """Best bid via REST ka10004 (호가) — to set a safe resting limit. None on failure."""
    import requests
    headers = {"Content-Type": "application/json;charset=UTF-8",
               "authorization": f"Bearer {token}", "api-id": "ka10004"}
    try:
        data = requests.post(f"{base}/api/dostk/mrkcond", headers=headers,
                             json={"stk_cd": stk}, timeout=10).json()
    except Exception:
        return None
    bid = _to_number(data.get("buy_fpr_bid"))
    return bid if bid > 0 else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stk", default="005930", help="6-digit KRX code (default Samsung)")
    ap.add_argument("--qty", type=int, default=1)
    args = ap.parse_args()

    if not SETTINGS.has_kiwoom_keys():
        print("Missing KIWOOM_APP_KEY / KIWOOM_SECRET_KEY in .env.")
        return 2
    env, base = SETTINGS.kiwoom_env, SETTINGS.kiwoom_rest_url()
    if env.lower() != "mock":
        print(f"REFUSING: KIWOOM_ENV={env!r} is not 'mock'. This proof is MOCK-ONLY.")
        return 2
    print(f"Kiwoom env: {env}  ({base})   [MOCK 모의투자 — not live]")

    try:
        auth = KiwoomAuth(SETTINGS.kiwoom_app_key, SETTINGS.kiwoom_secret_key, env=env, base_url=base)
        token = auth.get_token()
    except KiwoomAuthError as e:
        print(f"AUTH FAILED: {e}")
        return 1
    if not token:
        print("AUTH FAILED: empty token.")
        return 1
    print("token OK")                                    # never print the token itself

    # resting limit a few % below the bid so it won't fill (safe to place + cancel)
    bid = _best_bid(base, token, args.stk)
    price = snap_to_tick(bid * 0.95) if bid else None
    trde_tp, sess_label = _session_trde_tp()
    kind = f"limit {price:,}원 (~5% below bid {bid:,.0f})" if price else "market"
    print(f"\nSession: {sess_label}. Submitting BUY {args.qty} share(s) of {args.stk} as {kind} ...")

    from tagent.order_log import log_order_event
    try:
        res = place_buy(args.stk, args.qty, price=price, auth=auth, env=env, base_url=base,
                        trde_tp=trde_tp)
    except KiwoomOrderError as e:
        print(f"ORDER ERROR: {e}")
        return 1
    print(f"  BUY result: return_code={res['return_code']}  주문번호(ord_no)={res['ord_no'] or '—'}  "
          f"msg={res['return_msg']!r}")
    log_order_event(action="buy", stock=args.stk, side="BUY", qty=args.qty, price=price,
                    ord_no=res["ord_no"], return_code=res["return_code"], return_msg=res["return_msg"],
                    env=env, accepted=res["accepted"])

    if not res["accepted"] or not res["ord_no"]:
        print("\n=== PROOF: order NOT accepted by the mock server. Honest reject above "
              "(likely the mock account isn't provisioned for orders). Nothing cancelled. ===")
        return 1

    print(f"\n>>> API ACCEPTED the order. order id = {res['ord_no']}. Now cancelling it ...")
    try:
        cxl = cancel(res["ord_no"], args.stk, auth=auth, env=env, base_url=base)
    except KiwoomOrderError as e:
        print(f"CANCEL ERROR: {e}")
        return 1
    print(f"  CANCEL result: return_code={cxl['return_code']}  주문번호(ord_no)={cxl['ord_no'] or '—'}  "
          f"msg={cxl['return_msg']!r}")
    log_order_event(action="cancel", stock=args.stk, side="BUY", qty=args.qty, price=price,
                    ord_no=res["ord_no"], return_code=cxl["return_code"], return_msg=cxl["return_msg"],
                    env=env, accepted=cxl["accepted"])

    ok = cxl["accepted"]
    print(f"\n=== PROOF {'COMPLETE' if ok else 'PARTIAL'}: placed order {res['ord_no']} on MOCK and "
          f"{'cancelled it' if ok else 'cancel returned ' + str(cxl['return_code'])}. "
          "MOCK only — no live trade, no secrets printed. ===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
