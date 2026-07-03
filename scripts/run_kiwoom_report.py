"""Kiwoom 수급 report — per watchlist stock: 공매도 / 외국인·기관 net / 프로그램 net, each one line.

Pulls ka10014 (공매도추이), ka10059 (종목별투자자기관별), ka90013 (프로그램매매 일자별) per stock and
prints an honest one-line read of each. INFORMATIONAL only — not a validated edge, not a trading
signal. Mock-first; the token is never printed. Empty TRs (common on mock) are flagged, not faked.

Usage:
    python scripts/run_kiwoom_report.py
    python scripts/run_kiwoom_report.py --symbols 005930 000660 035420
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
from tagent.kiwoom_report import build_kiwoom_report, render_report_text  # noqa: E402

DEFAULT_SYMBOLS = ["005930", "000660", "035420"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    ap.add_argument("--start", default="", help="공매도 시작일 YYYYMMDD (optional)")
    ap.add_argument("--end", default="", help="공매도 종료일 YYYYMMDD (optional)")
    args = ap.parse_args()

    if not SETTINGS.has_kiwoom_keys():
        print("Missing KIWOOM_APP_KEY / KIWOOM_SECRET_KEY in .env.")
        return 2

    env, base = SETTINGS.kiwoom_env, SETTINGS.kiwoom_rest_url()
    print(f"Kiwoom env: {env}  ({base})")
    try:
        auth = KiwoomAuth(SETTINGS.kiwoom_app_key, SETTINGS.kiwoom_secret_key, env=env, base_url=base)
        if not auth.get_token():
            print("AUTH FAILED: empty token.")
            return 1
    except KiwoomAuthError as e:
        print(f"AUTH FAILED: {e}")
        return 1
    print("token OK")                                    # never print the token itself

    payload = build_kiwoom_report(args.symbols, auth=auth, env=env, base_url=base,
                                  start_date=args.start, end_date=args.end)
    print("\n" + render_report_text(payload))
    if payload.get("empty_endpoints"):
        print("\n(빈 TR은 mock 한계일 가능성이 큽니다 — 실계좌/실서버에서 다시 시도하세요.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
