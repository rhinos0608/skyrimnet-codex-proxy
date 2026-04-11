"""Gemini OAuth token cache with refresh lock."""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import aiohttp

logger = logging.getLogger("proxy")


class GeminiAuthCache:
    """Stores OAuth tokens loaded from ~/.gemini/oauth_creds.json."""

    def __init__(self):
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.client_id: Optional[str] = None
        self.client_secret: Optional[str] = None
        self.expires_at: Optional[datetime] = None
        self.session: Optional[aiohttp.ClientSession] = None
        self.project_id: Optional[str] = None
        # Lazy-initialised lock — see _get_lock(). Prevents concurrent refresh races.
        self._refresh_lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        """Lazy-init the refresh lock so it binds to the running event loop."""
        if self._refresh_lock is None:
            self._refresh_lock = asyncio.Lock()
        return self._refresh_lock

    @property
    def is_ready(self) -> bool:
        return self.access_token is not None

    def is_expired(self) -> bool:
        if not self.expires_at:
            return True
        return datetime.now() >= self.expires_at - timedelta(minutes=5)

    async def refresh_if_needed(self) -> bool:
        """Refresh access token if expired. Returns True if successful."""
        if not self.is_expired():
            return True
        if not self.refresh_token:
            logger.warning("Gemini token expired but missing refresh_token")
            return False

        async with self._get_lock():
            # Double-check: another coroutine may have refreshed while we waited
            if not self.is_expired():
                return True
            return await self._do_refresh()

    async def _do_refresh(self) -> bool:
        """Perform the actual token refresh. Caller must hold self._refresh_lock."""
        if not self.refresh_token:
            return False

        # Use credentials from file if present, otherwise fall back to Gemini CLI built-in creds
        import proxy
        client_id = self.client_id or proxy.GEMINI_OAUTH_CLIENT_ID
        client_secret = self.client_secret or proxy.GEMINI_OAUTH_CLIENT_SECRET

        logger.info("Refreshing Gemini OAuth token...")
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "grant_type": "refresh_token",
                    "refresh_token": self.refresh_token,
                    "client_id": client_id,
                    "client_secret": client_secret,
                }
                async with session.post(
                    "https://oauth2.googleapis.com/token",
                    data=payload,
                ) as resp:
                    if resp.status != 200:
                        error = await resp.text()
                        logger.error(f"Gemini token refresh failed: {resp.status} - {error[:200]}")
                        self.access_token = None
                        return False

                    data = await resp.json()
                    self.access_token = data.get("access_token")
                    expires_in = data.get("expires_in", 3600)
                    self.expires_at = datetime.now() + timedelta(seconds=expires_in)
                    logger.info(f"Gemini token refreshed, expires in {expires_in}s")
                    return True
        except Exception as e:
            logger.error(f"Gemini token refresh error: {e}")
            self.access_token = None
            return False

    def get_auth_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }


async def _fetch_gemini_project_id() -> None:
    """Call loadCodeAssist to get the managed project ID for this user/account.

    For personal Gemini Advanced subscribers the server returns a managed project.
    For GCP users the project comes from GOOGLE_CLOUD_PROJECT or the response.
    Mirrors the Gemini CLI: omit cloudaicompanionProject entirely when unknown.
    """
    import proxy

    # Check env var first (same priority as CLI)
    env_project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT_ID")
    if env_project:
        proxy.gemini_auth.project_id = env_project
        logger.info(f"Gemini project_id from env: {env_project}")
        return

    url = f"{proxy.GEMINI_CODE_ASSIST_ENDPOINT}/{proxy.GEMINI_CODE_ASSIST_API_VERSION}:loadCodeAssist"
    headers = proxy.gemini_auth.get_auth_headers()
    # Omit cloudaicompanionProject entirely (undefined in JS → field absent) so the
    # server resolves the project from the OAuth token / subscription.
    payload = {
        "metadata": {
            "ideType": "IDE_UNSPECIFIED",
            "platform": "PLATFORM_UNSPECIFIED",
            "pluginType": "GEMINI",
        },
    }
    try:
        async with proxy.gemini_auth.session.post(url, json=payload, headers=headers,
                                            timeout=aiohttp.ClientTimeout(total=10, connect=5)) as resp:
            if resp.status == 200:
                data = await resp.json()
                project_id = data.get("cloudaicompanionProject")
                proxy.gemini_auth.project_id = project_id or None
                if project_id:
                    logger.info(f"Gemini project_id from loadCodeAssist: {project_id}")
                else:
                    logger.info("Gemini loadCodeAssist returned no project — requests will omit project field")
            else:
                body = await resp.text()
                logger.warning(f"Gemini loadCodeAssist failed ({resp.status}): {body[:300]}")
    except Exception as e:
        logger.warning(f"Gemini loadCodeAssist error: {e}")


async def load_gemini_auth():
    """Load Gemini auth from ~/.gemini/oauth_creds.json (written by 'gemini auth login')."""
    import proxy

    if not os.path.exists(proxy.GEMINI_CREDS_FILE):
        logger.info(f"No Gemini credentials file at {proxy.GEMINI_CREDS_FILE} -- run 'gemini auth login'")
        return

    try:
        with open(proxy.GEMINI_CREDS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        proxy.gemini_auth.refresh_token = data.get("refresh_token")
        proxy.gemini_auth.client_id = data.get("client_id")
        proxy.gemini_auth.client_secret = data.get("client_secret")

        # Some formats include a cached access_token with an expiry_date (ms timestamp)
        if data.get("access_token"):
            proxy.gemini_auth.access_token = data["access_token"]
            expiry = data.get("expiry_date")
            if expiry:
                proxy.gemini_auth.expires_at = datetime.fromtimestamp(expiry / 1000)
            else:
                proxy.gemini_auth.expires_at = datetime.now() + timedelta(hours=1)

        if proxy.gemini_auth.refresh_token:
            if await proxy.gemini_auth.refresh_if_needed():
                proxy.gemini_auth.session = proxy.create_session()
                logger.info("Gemini auth loaded and token refreshed")
            elif proxy.gemini_auth.access_token:
                proxy.gemini_auth.session = proxy.create_session()
                logger.warning("Gemini token refresh failed, using cached token")
            else:
                logger.error("Gemini token refresh failed and no cached token available")
        elif proxy.gemini_auth.access_token:
            proxy.gemini_auth.session = proxy.create_session()
            logger.info("Gemini auth loaded (no refresh credentials -- token may expire)")
        else:
            logger.warning(f"Gemini credentials file found but missing required fields (need refresh_token or access_token)")

        # Fetch the real managed project ID from the Code Assist API
        if proxy.gemini_auth.is_ready and proxy.gemini_auth.session:
            await _fetch_gemini_project_id()

    except Exception as e:
        logger.error(f"Failed to load Gemini auth: {e}")
