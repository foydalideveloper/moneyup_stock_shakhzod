"""Forward-record KR's NEW microstructure sessions from the LIVE Kiwoom feed (mock-first).

Streams the live order book and appends the DEPTH-WEIGHTED quote (not just last trade) for every
in-session snapshot to an append-only, deduped store (data/microstructure_sessions.csv). It records
ONLY inside the two new windows — NXT pre-market 08:00–08:50 KST and the KRX night futures session
(~18:00–익일 05:00 KST) — so just leaving it running accrues the out-of-sample dataset. NO strategy,
NO orders. Secrets are read from .env and never printed.

Usage:
    python scripts/run_microstructure_recorder.py
    python scripts/run_microstructure_recorder.py --symbols 005930 000660 101S3000
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
from tagent.feeds.kiwoom_feed import KiwoomFeed  # noqa: E402
from tagent.microstructure_recorder import SESSIONS, MicrostructureRecorder  # noqa: E402

# Liquid defaults: a couple of large-caps for the NXT pre-market call + a KOSPI200 futures code
# for the night session. The recorder gates by session window, so subscribing is harmless off-hours.
DEFAULT_SYMBOLS = ["005930", "000660", "101S3000"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS,
                    help="Kiwoom codes to subscribe (NXT equities + night-futures codes)")
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    if not SETTINGS.has_kiwoom_keys():
        print("Set KIWOOM_APP_KEY / KIWOOM_SECRET_KEY in .env (mock keys are fine). "
              "Recorder needs the live feed.")
        return 1

    recorder = MicrostructureRecorder(data_dir=args.data_dir)
    feed = KiwoomFeed(args.symbols, app_key=SETTINGS.kiwoom_app_key,
                      secret_key=SETTINGS.kiwoom_secret_key, env=SETTINGS.kiwoom_env)
    feed.on_orderbook(recorder.record)               # record() is session-gated + deduped + no-lookahead

    print(f"=== KR new-microstructure forward-recorder ({SETTINGS.kiwoom_env}) ===")
    for s in SESSIONS:
        print(f"  · {s.label}")
    print(f"  subscribing {', '.join(args.symbols)} -> {recorder.path}")
    print("  records ONLY in-session depth-weighted quotes; append-only, deduped. "
          "PAPER/record-only — no orders. Ctrl+C to stop.\n")
    try:
        feed.run()
    except KeyboardInterrupt:
        pass
    finally:
        feed.stop()
        recorder.close()
        s = recorder.summary()
        print(f"\nStopped. +{s['rows_added_this_run']} rows this run; {s['total_rows']} total "
              f"across sessions {s['sessions'] or '—'}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
