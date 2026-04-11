"""Codex OAuth token cache with refresh lock."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

import aiohttp

logger = logging.getLogger("proxy")


class CodexAuthCache:
    """Stores OAuth tokens captured from Codex CLI auth flow."""

    def __init__(self):
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.expires_at: Optional[datetime] = None
        self.account_id: Optional[str] = None
        self.session: Optional[aiohttp.ClientSession] = None
        self._token_endpoint = "https://auth.openai.com/oauth/token"
        self._client_id = "app_EMoamEEZ73f0CkXaXp7hrann"  # Codex CLI OAuth client ID
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
        return datetime.now() >= self.expires_at - timedelta(minutes=5)  # 5min buffer

    async def refresh_if_needed(self) -> bool:
        """Refresh access token if expired. Returns True if successful."""
        if not self.is_expired():
            return True
        if not self.refresh_token:
            logger.warning("Codex token expired but no refresh token available")
            return False

        async with self._get_lock():
            # Double-check: another coroutine may have refreshed while we waited
            if not self.is_expired():
                return True
            return await self._do_refresh()

    async def _do_refresh(self) -> bool:
        """Perform the actual token refresh. Caller must hold self._refresh_lock."""
        logger.info("Refreshing Codex OAuth token...")
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "grant_type": "refresh_token",
                    "refresh_token": self.refresh_token,
                    "client_id": self._client_id,
                }
                async with session.post(self._token_endpoint, json=payload) as resp:
                    if resp.status != 200:
                        error = await resp.text()
                        logger.error(f"Token refresh failed: {resp.status} - {error[:200]}")
                        self.access_token = None
                        return False

                    data = await resp.json()
                    self.access_token = data.get("access_token")
                    if data.get("refresh_token"):
                        self.refresh_token = data["refresh_token"]
                    expires_in = data.get("expires_in", 3600)
                    self.expires_at = datetime.now() + timedelta(seconds=expires_in)
                    logger.info(f"Token refreshed, expires in {expires_in}s")
                    return True
        except Exception as e:
            logger.error(f"Token refresh error: {e}")
            self.access_token = None
            return False

    def get_auth_headers(self) -> dict:
        """Get headers for API requests."""
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }
