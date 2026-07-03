"""Start the live monitor.

Auto-selects the feed: if Kiwoom keys are present it streams KR data via Kiwoom
(default watchlist 005930, 000660, 035420 unless WATCHLIST overrides it);
otherwise it falls back to the existing Alpaca (US) feed.

Usage:  python scripts/run_monitor.py
"""

import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from tagent.config import KIWOOM_DEFAULT_WATCHLIST, SETTINGS  # noqa: E402
from tagent.monitor import Monitor  # noqa: E402

if __name__ == "__main__":
    if SETTINGS.has_kiwoom_keys():
        # Default to the KR watchlist for Kiwoom unless the user set WATCHLIST.
        if not os.getenv("WATCHLIST"):
            SETTINGS.watchlist = list(KIWOOM_DEFAULT_WATCHLIST)
        from tagent.feeds.kiwoom_feed import KiwoomFeed
        feed = KiwoomFeed(SETTINGS.watchlist, SETTINGS.kiwoom_app_key,
                          SETTINGS.kiwoom_secret_key, env=SETTINGS.kiwoom_env)
        print(f"Using Kiwoom feed ({SETTINGS.kiwoom_env}) for "
              f"{', '.join(SETTINGS.watchlist)}")
        Monitor(settings=SETTINGS, feed=feed).run()
    else:
        Monitor().run()
