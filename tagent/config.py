"""Central configuration.

Every tunable lives here so nothing is hard-coded elsewhere. Values default to
sensible numbers and can be overridden via environment variables (a .env file).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# Load .env if python-dotenv is installed (optional at import time).
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass

# Project root = parent of the tagent package directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _get_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def _get_list(name: str, default: List[str]) -> List[str]:
    raw = os.getenv(name)
    if not raw:
        return list(default)
    return [x.strip().upper() for x in raw.split(",") if x.strip()]


_DEFAULT_WATCHLIST = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN",
                      "GOOGL", "META", "AMD", "NFLX", "SPY"]

# Kiwoom REST/WebSocket base URLs (from openapi.kiwoom.com docs). "mock" is the
# simulation server (KRX only); "live" is production. Streaming runs on port
# 10000 at the documented /api/dostk/websocket path.
KIWOOM_REST_URLS = {
    "mock": "https://mockapi.kiwoom.com",
    "live": "https://api.kiwoom.com",
}
KIWOOM_WS_URLS = {
    "mock": "wss://mockapi.kiwoom.com:10000/api/dostk/websocket",
    "live": "wss://api.kiwoom.com:10000/api/dostk/websocket",
}
# Sensible KR default watchlist for Kiwoom (Samsung Elec, SK hynix, NAVER).
KIWOOM_DEFAULT_WATCHLIST = ["005930", "000660", "035420"]


@dataclass
class Settings:
    # --- live feed (Alpaca, US) ---
    alpaca_api_key: str = os.getenv("ALPACA_API_KEY", "")
    alpaca_secret_key: str = os.getenv("ALPACA_SECRET_KEY", "")
    data_feed: str = os.getenv("DATA_FEED", "iex")
    watchlist: List[str] = field(default_factory=lambda: _get_list("WATCHLIST", _DEFAULT_WATCHLIST))

    # --- live feed (Kiwoom, KR) ---
    # Secrets come from .env only; never hardcoded. KIWOOM_ENV picks mock|live.
    kiwoom_env: str = os.getenv("KIWOOM_ENV", "mock")
    kiwoom_app_key: str = os.getenv("KIWOOM_APP_KEY", "")
    kiwoom_secret_key: str = os.getenv("KIWOOM_SECRET_KEY", "")

    # --- KRX data login (unlocks pykrx's gated 공매도/수급 endpoints) ---
    # A free data.krx.co.kr account; pykrx reads these from the environment.
    krx_id: str = os.getenv("KRX_ID", "")
    krx_pw: str = os.getenv("KRX_PW", "")

    # --- news / disclosures (rule-based alerts, not price prediction) ---
    opendart_api_key: str = os.getenv("OPENDART_API_KEY", "")   # KR disclosures (opendart.fss.or.kr)
    finnhub_api_key: str = os.getenv("FINNHUB_API_KEY", "")     # US news + sentiment (finnhub.io)
    # --- media monitoring (YouTube/TV briefing — AWARENESS ONLY, not a trading signal) ---
    youtube_api_key: str = os.getenv("YOUTUBE_API_KEY", "")     # YouTube Data API (developers.google.com)
    youtube_channels: str = os.getenv("YOUTUBE_CHANNELS", "")   # "한국경제TV=UCxxxx,Bloomberg=UCyyyy"
    naver_client_id: str = os.getenv("NAVER_CLIENT_ID", "")     # Naver News search (한경/매경 article links)
    naver_client_secret: str = os.getenv("NAVER_CLIENT_SECRET", "")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")       # Google Gemini (generativelanguage API) LLM backend
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
    # stronger model for the OVERNIGHT batch / --reextract (falls back to gemini-flash-latest on 404)
    gemini_extract_model: str = os.getenv("GEMINI_EXTRACT_MODEL", "gemini-2.5-pro")
    # faster model for INTERACTIVE dashboard paths (single-video + window) and translation
    gemini_interactive_model: str = os.getenv("GEMINI_INTERACTIVE_MODEL", "gemini-2.5-flash")
    # extraction provider switch: gemini (default) | openai — lets us A/B the same cached transcripts
    llm_provider: str = os.getenv("LLM_PROVIDER", "gemini")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")      # OpenAI extractor (alt provider)
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")    # gpt-4o / gpt-4.1 / gpt-4o-mini
    # transcript fallback: optional proxy to bypass the youtube-transcript-api IP-block + pacing
    youtube_proxy_url: str = os.getenv("YOUTUBE_PROXY_URL") or os.getenv("PROXY_URL", "")  # http://user:pass@host:port
    webshare_proxy_username: str = (os.getenv("WEBSHARE_USER")            # Webshare rotating-residential proxy
                                    or os.getenv("WEBSHARE_PROXY_USERNAME", ""))
    webshare_proxy_password: str = (os.getenv("WEBSHARE_PASS")
                                    or os.getenv("WEBSHARE_PROXY_PASSWORD", ""))
    youtube_transcript_pace_seconds: float = _get_float("YOUTUBE_TRANSCRIPT_PACE_SECONDS", 2.0)
    # --- top-source whitelist (KR by domain, US by Finnhub publisher) for the feed + newspaper ---
    source_whitelist_kr: str = os.getenv("SOURCE_WHITELIST_KR",
                                         "hankyung.com,mk.co.kr,mt.co.kr,sedaily.com,yna.co.kr")
    source_whitelist_us: str = os.getenv("SOURCE_WHITELIST_US",
                                         "bloomberg,wsj,wall street journal,reuters,cnbc")

    # --- minute-data vendors (intraday backtest history) ---
    itick_api_key: str = os.getenv("ITICK_API_KEY", "")        # iTick OHLCV (itick.org)
    eodhd_api_key: str = os.getenv("EODHD_API_KEY", "")        # EODHD intraday (eodhd.com)

    # --- alerts ---
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")
    alert_cooldown_seconds: int = _get_int("ALERT_COOLDOWN_SECONDS", 60)

    # --- email delivery (Gmail SMTP; daily report -> boss) ---
    gmail_user: str = os.getenv("GMAIL_USER", "")               # sender Gmail address
    gmail_app_password: str = os.getenv("GMAIL_APP_PASSWORD", "")   # 16-char app password — NEVER logged
    boss_email: str = os.getenv("BOSS_EMAIL", "")              # recipient

    # --- Supabase push (VIP Agent delivery of the SAME .docx/.pdf; blank = push disabled) ---
    supabase_url: str = os.getenv("SUPABASE_URL", "")          # https://<project>.supabase.co
    supabase_key: str = os.getenv("SUPABASE_KEY", "")          # service-role key — NEVER logged

    # --- live state ---
    rolling_window_seconds: int = _get_int("ROLLING_WINDOW_SECONDS", 60)
    stale_after_seconds: float = _get_float("STALE_AFTER_SECONDS", 30.0)

    # --- risk ---
    account_equity: float = _get_float("ACCOUNT_EQUITY", 100_000.0)
    risk_per_trade_pct: float = _get_float("RISK_PER_TRADE_PCT", 0.5)
    max_position_pct: float = _get_float("MAX_POSITION_PCT", 10.0)
    stop_loss_pct: float = _get_float("STOP_LOSS_PCT", 2.0)
    take_profit_pct: float = _get_float("TAKE_PROFIT_PCT", 4.0)
    max_daily_loss_pct: float = _get_float("MAX_DAILY_LOSS_PCT", 3.0)

    # --- costs (for realistic backtests) ---
    transaction_cost_bps: float = _get_float("TRANSACTION_COST_BPS", 5.0)
    slippage_bps: float = _get_float("SLIPPAGE_BPS", 2.0)

    def has_alpaca_keys(self) -> bool:
        return bool(self.alpaca_api_key and self.alpaca_secret_key)

    def has_kiwoom_keys(self) -> bool:
        return bool(self.kiwoom_app_key and self.kiwoom_secret_key)

    def has_krx_login(self) -> bool:
        return bool(self.krx_id and self.krx_pw)

    def has_opendart_key(self) -> bool:
        return bool(self.opendart_api_key)

    def has_finnhub_key(self) -> bool:
        return bool(self.finnhub_api_key)

    def has_youtube_key(self) -> bool:
        return bool(self.youtube_api_key)

    def has_naver_keys(self) -> bool:
        return bool(self.naver_client_id and self.naver_client_secret)

    def has_gemini_key(self) -> bool:
        return bool(self.gemini_api_key)

    def has_openai_key(self) -> bool:
        return bool(self.openai_api_key)

    def has_youtube_proxy(self) -> bool:
        return bool(self.youtube_proxy_url or self.webshare_proxy())

    def webshare_proxy(self):
        """Webshare creds for youtube-transcript-api's WebshareProxyConfig, or None. Never logged."""
        if self.webshare_proxy_username and self.webshare_proxy_password:
            return {"proxy_username": self.webshare_proxy_username,
                    "proxy_password": self.webshare_proxy_password}
        return None

    def has_email_creds(self) -> bool:
        """All three pieces present to send the daily report (sender, app password, recipient)."""
        return bool(self.gmail_user and self.gmail_app_password and self.boss_email)

    def has_supabase_creds(self) -> bool:
        """Both pieces present to push the report to Supabase (URL + service-role key)."""
        return bool(self.supabase_url and self.supabase_key)

    def has_itick_key(self) -> bool:
        return bool(self.itick_api_key)

    def has_eodhd_key(self) -> bool:
        return bool(self.eodhd_api_key)

    def kiwoom_rest_url(self) -> str:
        return KIWOOM_REST_URLS.get(self.kiwoom_env.lower(), KIWOOM_REST_URLS["mock"])

    def kiwoom_ws_url(self) -> str:
        return KIWOOM_WS_URLS.get(self.kiwoom_env.lower(), KIWOOM_WS_URLS["mock"])


# Singleton used across the app.
SETTINGS = Settings()
