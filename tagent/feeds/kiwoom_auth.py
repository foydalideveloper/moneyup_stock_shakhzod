"""Kiwoom REST OAuth2 client (access-token issuance + caching).

Per the official docs (openapi.kiwoom.com → API 가이드):

    POST {base}/oauth2/token
    body: {"grant_type": "client_credentials", "appkey": ..., "secretkey": ...}
    resp: {"token": ..., "token_type": "bearer",
           "expires_dt": "YYYYMMDDHHMMSS", "return_code": 0, "return_msg": ""}

The token is cached and proactively refreshed shortly before ``expires_dt``.
Credentials are read from config/.env by the caller and are NEVER logged; auth
failures surface ``return_msg`` (and hints for the usual causes) but never the
app key or secret.

``requests`` is imported lazily so the package imports without it; tests inject a
fake ``session`` and ``clock`` and never hit the network.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

from tagent.config import KIWOOM_REST_URLS

# Kiwoom's expires_dt is Korea Standard Time (UTC+9).
_KST = timezone(timedelta(hours=9))
# Refresh this many seconds before the server-stated expiry.
_REFRESH_MARGIN_SECONDS = 60


class KiwoomAuthError(RuntimeError):
    """Raised when token issuance fails (bad key, IP not allowlisted, ...)."""


class KiwoomAuth:
    def __init__(self, app_key: str, secret_key: str, env: str = "mock",
                 base_url: Optional[str] = None, session=None,
                 clock: Optional[Callable[[], datetime]] = None):
        if not app_key or not secret_key:
            raise KiwoomAuthError(
                "Missing Kiwoom credentials. Set KIWOOM_APP_KEY / "
                "KIWOOM_SECRET_KEY in your .env.")
        self._app_key = app_key
        self._secret_key = secret_key
        self.env = env
        self.base_url = base_url or KIWOOM_REST_URLS.get(
            env.lower(), KIWOOM_REST_URLS["mock"])
        self._session = session                      # injectable for tests
        self._now = clock or (lambda: datetime.now(timezone.utc))
        self._token: Optional[str] = None
        self._expires_at: Optional[datetime] = None

    # ------------------------------------------------------------------ #
    def get_token(self, force: bool = False) -> str:
        """Return a valid access token, issuing/refreshing as needed."""
        if not force and self._token and not self._is_expiring():
            return self._token
        self._refresh()
        return self._token  # type: ignore[return-value]

    def _is_expiring(self) -> bool:
        if self._expires_at is None:
            return True
        margin = timedelta(seconds=_REFRESH_MARGIN_SECONDS)
        return self._now() >= (self._expires_at - margin)

    def _post_token(self) -> dict:
        session = self._session
        if session is None:
            try:
                import requests  # lazy
            except ImportError as e:  # pragma: no cover
                raise KiwoomAuthError(
                    "requests not installed. Run: pip install -r requirements.txt"
                ) from e
            session = requests
        url = f"{self.base_url}/oauth2/token"
        try:
            resp = session.post(
                url,
                json={
                    "grant_type": "client_credentials",
                    "appkey": self._app_key,
                    "secretkey": self._secret_key,
                },
                headers={"Content-Type": "application/json;charset=UTF-8"},
                timeout=10,
            )
        except Exception as e:  # network/DNS/timeout — never echoes secrets
            raise KiwoomAuthError(
                f"Could not reach Kiwoom token endpoint at {url}: {e}") from e

        status = getattr(resp, "status_code", 200)
        try:
            data = resp.json()
        except Exception:
            raise KiwoomAuthError(
                f"Kiwoom token endpoint returned non-JSON (HTTP {status}).")
        return data

    def _refresh(self) -> None:
        data = self._post_token()

        # Kiwoom signals success with return_code == 0.
        code = data.get("return_code")
        token = data.get("token")
        if (code not in (0, "0", None)) or not token:
            msg = data.get("return_msg") or "unknown error"
            raise KiwoomAuthError(
                f"Kiwoom auth failed (return_code={code}): {msg}. "
                "Check KIWOOM_APP_KEY/KIWOOM_SECRET_KEY, that the key is enabled "
                "for this environment (mock vs live), and that your IP is "
                "allowlisted in the Kiwoom developer console.")

        self._token = token
        self._expires_at = self._parse_expires(data.get("expires_dt"))

    @staticmethod
    def _parse_expires(expires_dt: Optional[str]) -> datetime:
        # Documented format "YYYYMMDDHHMMSS" in KST; fall back to ~23h if absent.
        if expires_dt:
            try:
                return datetime.strptime(str(expires_dt), "%Y%m%d%H%M%S").replace(
                    tzinfo=_KST)
            except ValueError:
                pass
        return datetime.now(timezone.utc) + timedelta(hours=23)
