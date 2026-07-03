"""Alert delivery with debouncing.

In a fast market the same condition is true on many ticks in a row. The Alerter
suppresses repeats of the same (symbol, kind) within a cooldown window, so you
get one useful alert instead of hundreds. Prints to screen, and to Telegram if
configured. The clock is injectable so the debounce logic is unit-testable.
"""

from __future__ import annotations

import time
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Callable, Dict, Optional, Tuple

from tagent.strategy import Signal


class Alerter:
    def __init__(self, cooldown_seconds: float = 60.0,
                 telegram_token: str = "", telegram_chat_id: str = "",
                 clock: Callable[[], float] = time.time,
                 printer: Callable[[str], None] = print):
        self.cooldown = float(cooldown_seconds)
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        self._clock = clock
        self._print = printer
        self._last_sent: Dict[Tuple[str, str], float] = {}

    def _allowed(self, sig: Signal) -> bool:
        key = (sig.symbol, sig.kind)
        now = self._clock()
        if now - self._last_sent.get(key, float("-inf")) >= self.cooldown:
            self._last_sent[key] = now
            return True
        return False

    def dispatch(self, sig: Signal) -> bool:
        """Returns True if an alert was actually sent (False if debounced)."""
        if not self._allowed(sig):
            return False
        stamp = datetime.now().strftime("%H:%M:%S")
        line = (f"[{stamp}] {sig.side.upper()} {sig.symbol} @ {sig.price:.2f} "
                f"- {sig.reason}")
        self._print(line)
        self._send_telegram(line)
        return True

    def _send_telegram(self, text: str) -> None:
        if not self.telegram_token or not self.telegram_chat_id:
            return
        try:
            data = urllib.parse.urlencode(
                {"chat_id": self.telegram_chat_id, "text": text}).encode()
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=5)
        except Exception as e:  # never let an alert failure crash the monitor
            self._print(f"[alert] telegram send failed: {e}")
