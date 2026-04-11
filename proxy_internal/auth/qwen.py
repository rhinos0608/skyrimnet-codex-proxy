"""Qwen CLI OAuth cache with file reload."""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import aiohttp

logger = logging.getLogger("proxy")


class QwenAuthCache:
    """Stores OAuth tokens loaded from ~/.qwen/oauth_creds.json (Qwen Code CLI).

    The Qwen CLI manages its own token lifecycle (device flow auth + refresh).
    This cache re-reads the credentials file on each request to pick up tokens
    refreshed by the CLI, rather than trying to refresh them ourselves (the
    chat.qwen.ai token endpoint has WAF protections that block non-browser
    HTTP clients).
    """

    def __init__(self):
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.expires_at: Optional[datetime] = None
        self.session: Optional[aiohttp.ClientSession] = None
        self._file_mtime: float = 0  # Track credential file changes
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

    def reload_from_file(self) -> bool:
        """Re-read credentials from disk if the file has changed. Returns True if token available."""
        import proxy
        qwen_creds_file = proxy.QWEN_CREDS_FILE
        if not os.path.exists(qwen_creds_file):
            return False
        try:
            mtime = os.path.getmtime(qwen_creds_file)
            if mtime <= self._file_mtime and self.access_token:
                return True  # File unchanged and we have a token

            with open(qwen_creds_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            new_token = data.get("access_token")
            if not new_token:
                return False

            # Only update if the token actually changed
            if new_token != self.access_token:
                self.access_token = new_token
                self.refresh_token = data.get("refresh_token")
                expiry = data.get("expiry_date")
                if expiry:
                    self.expires_at = datetime.fromtimestamp(expiry / 1000)
                else:
                    self.expires_at = datetime.now() + timedelta(hours=1)
                logger.info("Qwen credentials reloaded from file")

            self._file_mtime = mtime
            return True
        except Exception as e:
            logger.error(f"Failed to reload Qwen credentials: {e}")
            return False

    async def refresh_if_needed(self) -> bool:
        """Try to reload credentials from file (CLI manages token refresh).

        ``reload_from_file`` does synchronous disk I/O (``open``, ``json.load``,
        ``os.path.getmtime``) — when the creds file lives on slow storage this
        would otherwise block the event loop.  Run it in the default thread
        pool so the loop stays responsive while we hold the refresh lock.
        """
        if not self.is_expired():
            return True
        async with self._get_lock():
            # Double-check: another coroutine may have reloaded while we waited
            if not self.is_expired():
                return True
            # Re-read from file — the Qwen CLI may have refreshed the token.
            # ``asyncio.to_thread`` keeps the blocking I/O off the event loop.
            return await asyncio.to_thread(self.reload_from_file)

    def get_auth_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
            "X-DashScope-AuthType": "qwen-oauth",
            "X-DashScope-CacheControl": "enable",
        }


async def load_qwen_auth():
    """Load Qwen auth from ~/.qwen/oauth_creds.json (written by 'qwen code' login).

    The Qwen CLI manages token lifecycle via device flow OAuth.  We just read
    whatever token is on disk — the CLI writes it after each successful auth.
    On each request we also re-read the file to pick up refreshed tokens.
    """
    import proxy

    if not os.path.exists(proxy.QWEN_CREDS_FILE):
        logger.info(f"No Qwen credentials file at {proxy.QWEN_CREDS_FILE} -- run Qwen Code CLI login")
        return

    if proxy.qwen_auth.reload_from_file():
        proxy.qwen_auth.session = proxy.create_session()
        logger.info("Qwen auth loaded from credentials file")
    else:
        logger.warning("Qwen credentials file found but no valid access_token")
