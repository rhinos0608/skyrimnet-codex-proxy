"""Antigravity multi-account OAuth auth, encryption, and file persistence."""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Optional

import aiohttp

from proxy_internal.encryption import _get_fernet, _encrypt_value, _decrypt_value, _ENC_PREFIX

logger = logging.getLogger("proxy")

# Antigravity auth-file encryption field set.  Previously lived in proxy.py
# as `_ENCRYPTED_AUTH_FIELDS`; moved here because it is only used by the
# encrypt/decrypt helpers below.
_ENCRYPTED_AUTH_FIELDS = {"refresh_token", "access_token"}


def get_antigravity_headers() -> dict:
    """Get headers for Antigravity API requests."""
    import proxy
    platform = "WINDOWS" if sys.platform == "win32" else "MACOS"
    return {
        "User-Agent": f"antigravity/{proxy.ANTIGRAVITY_VERSION} {'windows/amd64' if sys.platform == 'win32' else 'darwin/arm64'}",
        "X-Goog-Api-Client": "google-cloud-sdk vscode_cloudshelleditor/0.1",
        "Client-Metadata": f'{{"ideType":"ANTIGRAVITY","platform":"{platform}","pluginType":"GEMINI"}}',
    }


class AntigravityAccount:
    """Stores OAuth tokens for a single Google Antigravity account."""

    def __init__(self):
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.expires_at: Optional[datetime] = None
        self.email: Optional[str] = None
        self.project_id: Optional[str] = None
        self.session: Optional[aiohttp.ClientSession] = None
        self._error_count: int = 0  # Track consecutive errors for this account
        # Lazy-initialised lock — see _get_lock(). Prevents concurrent refresh races
        # across coroutines sharing this account.
        self._refresh_lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        """Lazy-init the refresh lock so it binds to the running event loop."""
        if self._refresh_lock is None:
            self._refresh_lock = asyncio.Lock()
        return self._refresh_lock

    @property
    def is_ready(self) -> bool:
        return self.access_token is not None and not self.is_expired()

    def is_expired(self) -> bool:
        if not self.expires_at:
            return True
        return datetime.now() >= self.expires_at - timedelta(minutes=5)  # 5min buffer

    def to_dict(self) -> dict:
        """Serialize account to dict for saving."""
        return {
            "refresh_token": self.refresh_token,
            "access_token": self.access_token,
            "email": self.email,
            "project_id": self.project_id,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AntigravityAccount":
        """Create account from dict."""
        account = cls()
        account.refresh_token = data.get("refresh_token")
        account.email = data.get("email")
        account.project_id = data.get("project_id")
        if data.get("access_token"):
            account.access_token = data["access_token"]
            if data.get("expires_at"):
                try:
                    account.expires_at = datetime.fromisoformat(data["expires_at"])
                except ValueError:
                    account.expires_at = datetime.now() - timedelta(seconds=1)
            else:
                account.expires_at = datetime.now() - timedelta(seconds=1)
        return account

    async def refresh_if_needed(self) -> bool:
        """Refresh access token if expired. Returns True if successful."""
        if not self.is_expired():
            return True
        if not self.refresh_token:
            logger.warning(f"Antigravity token expired for {self.email} but no refresh token available")
            return False

        async with self._get_lock():
            # Double-check: another coroutine may have refreshed while we waited
            if not self.is_expired():
                return True
            return await self._do_refresh()

    async def _do_refresh(self) -> bool:
        """Perform the actual token refresh. Caller must hold self._refresh_lock."""
        import proxy
        logger.info(f"Refreshing Antigravity OAuth token for {self.email}...")
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "grant_type": "refresh_token",
                    "refresh_token": self.refresh_token,
                    "client_id": proxy.ANTIGRAVITY_CLIENT_ID,
                    "client_secret": proxy.ANTIGRAVITY_CLIENT_SECRET,
                }
                async with session.post("https://oauth2.googleapis.com/token", data=payload) as resp:
                    if resp.status != 200:
                        error = await resp.text()
                        logger.error(f"Antigravity token refresh failed for {self.email}: {resp.status} - {error[:200]}")
                        self.access_token = None
                        return False

                    data = await resp.json()
                    self.access_token = data.get("access_token")
                    if data.get("refresh_token"):
                        self.refresh_token = data["refresh_token"]
                    expires_in = data.get("expires_in", 3600)
                    self.expires_at = datetime.now() + timedelta(seconds=expires_in)
                    logger.info(f"Antigravity token refreshed for {self.email}, expires in {expires_in}s")
                    return True
        except Exception as e:
            logger.error(f"Antigravity token refresh error for {self.email}: {e}")
            self.access_token = None
            return False

    def get_auth_headers(self) -> dict:
        """Get headers for API requests."""
        headers = get_antigravity_headers()
        headers["Authorization"] = f"Bearer {self.access_token}"
        headers["Content-Type"] = "application/json"
        return headers


class AntigravityAuthCache:
    """Manages multiple Google Antigravity accounts with automatic fallback."""

    def __init__(self):
        self.accounts: list[AntigravityAccount] = []
        self._current_index: int = 0
        # Legacy single-account properties for backward compatibility
        self._legacy_account = AntigravityAccount()

    def _get_active_accounts(self) -> list[AntigravityAccount]:
        """Get list of accounts sorted by error count (prefer healthy accounts)."""
        if not self.accounts:
            return [self._legacy_account] if self._legacy_account.refresh_token else []
        # Sort by error count, but preserve order for accounts with same error count
        return sorted(self.accounts, key=lambda a: a._error_count)

    def get_next_account(self) -> Optional[AntigravityAccount]:
        """Get next available account using round-robin with error-aware prioritization."""
        active = self._get_active_accounts()
        if not active:
            return None
        # Pick the account with lowest error count, then round-robin among equals
        min_errors = min(a._error_count for a in active)
        good_accounts = [a for a in active if a._error_count == min_errors]
        if len(good_accounts) == 1:
            return good_accounts[0]
        # Round-robin among good accounts
        account = good_accounts[self._current_index % len(good_accounts)]
        self._current_index += 1
        return account

    def mark_account_error(self, account: AntigravityAccount, status_code: int):
        """Mark an account as having an error (increments error count)."""
        account._error_count += 1
        logger.warning(f"Antigravity account {account.email} got {status_code} error (error count: {account._error_count})")

    def mark_account_success(self, account: AntigravityAccount):
        """Mark an account as successful (resets error count)."""
        if account._error_count > 0:
            logger.info(f"Antigravity account {account.email} recovered, resetting error count")
        account._error_count = 0

    @property
    def is_ready(self) -> bool:
        """Check if any account is ready."""
        if self.accounts:
            return any(a.is_ready for a in self.accounts)
        return self._legacy_account.is_ready

    def is_expired(self) -> bool:
        """Check if current account is expired (for backward compatibility)."""
        account = self.get_next_account()
        return account.is_expired() if account else True

    @property
    def email(self) -> Optional[str]:
        """Get email of current/first ready account."""
        for account in self._get_active_accounts():
            if account.is_ready:
                return account.email
        if self._legacy_account.email:
            return self._legacy_account.email
        return self.accounts[0].email if self.accounts else None

    @property
    def project_id(self) -> Optional[str]:
        """Get project_id of current/first ready account."""
        for account in self._get_active_accounts():
            if account.is_ready:
                return account.project_id
        if self._legacy_account.project_id:
            return self._legacy_account.project_id
        return self.accounts[0].project_id if self.accounts else None

    @property
    def session(self) -> Optional[aiohttp.ClientSession]:
        """Get session of current/first ready account."""
        for account in self._get_active_accounts():
            if account.is_ready and account.session:
                return account.session
        return self._legacy_account.session

    async def refresh_if_needed(self) -> bool:
        """Refresh tokens for all accounts. Returns True if at least one is ready.

        Snapshots ``self.accounts`` at entry so a concurrent
        ``add_account``/``remove_account`` cannot mutate the list mid-loop
        (the per-account ``await`` yields the event loop between iterations).
        The final readiness check also uses the snapshot so the concurrency
        window is fully closed.
        """
        import proxy
        if self.accounts:
            accounts_snapshot = list(self.accounts)
            for account in accounts_snapshot:
                if account.refresh_token:
                    await account.refresh_if_needed()
                    if account.is_ready and not account.session:
                        account.session = proxy.create_session()
            return any(a.is_ready for a in accounts_snapshot)
        # Legacy single-account mode
        return await self._legacy_account.refresh_if_needed()

    def get_auth_headers(self) -> dict:
        """Get auth headers for current/first ready account."""
        for account in self._get_active_accounts():
            if account.is_ready:
                return account.get_auth_headers()
        return self._legacy_account.get_auth_headers()

    def add_account(self, account: AntigravityAccount):
        """Add a new account to the pool."""
        # Check if account with this email already exists
        for i, existing in enumerate(self.accounts):
            if existing.email == account.email:
                # Update existing account
                self.accounts[i] = account
                logger.info(f"Updated Antigravity account: {account.email}")
                return
        self.accounts.append(account)
        logger.info(f"Added Antigravity account: {account.email}")

    def remove_account(self, email: str) -> bool:
        """Remove an account by email. Returns True if removed."""
        for i, account in enumerate(self.accounts):
            if account.email == email:
                if account.session:
                    # Schedule session cleanup
                    asyncio.create_task(account.session.close())
                del self.accounts[i]
                logger.info(f"Removed Antigravity account: {email}")
                return True
        return False

    def get_all_accounts_info(self) -> list[dict]:
        """Get info about all accounts for display."""
        result = []
        for account in self.accounts:
            result.append({
                "email": account.email,
                "project_id": account.project_id,
                "is_ready": account.is_ready,
                "error_count": account._error_count,
            })
        # Include legacy account if it exists and not already in list
        if self._legacy_account.refresh_token:
            legacy_email = self._legacy_account.email
            if not any(a["email"] == legacy_email for a in result):
                result.insert(0, {
                    "email": legacy_email,
                    "project_id": self._legacy_account.project_id,
                    "is_ready": self._legacy_account.is_ready,
                    "error_count": self._legacy_account._error_count,
                })
        return result


def _decrypt_auth_account(account_data: dict) -> dict:
    """Decrypt encrypted fields in an antigravity account dict. Auto-migrates plaintext."""
    fernet = _get_fernet()
    out = dict(account_data)
    for field in _ENCRYPTED_AUTH_FIELDS:
        val = out.get(field)
        if isinstance(val, str) and val.startswith(_ENC_PREFIX):
            out[field] = _decrypt_value(fernet, val)
    return out


def _encrypt_auth_account(account_data: dict) -> dict:
    """Encrypt sensitive fields in an antigravity account dict before saving."""
    fernet = _get_fernet()
    out = dict(account_data)
    for field in _ENCRYPTED_AUTH_FIELDS:
        val = out.get(field)
        if isinstance(val, str) and val and not val.startswith(_ENC_PREFIX):
            out[field] = _encrypt_value(fernet, val)
    return out


async def load_antigravity_auth():
    """Load Antigravity auth from cached file. Supports both legacy single-account and new multi-account formats."""
    import proxy

    if not os.path.exists(proxy.ANTIGRAVITY_AUTH_FILE):
        logger.info("No Antigravity auth file found -- visit /config/antigravity-login")
        return

    try:
        with open(proxy.ANTIGRAVITY_AUTH_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        migrated = False
        # Check if it's the new multi-account format (has "accounts" array)
        if "accounts" in data and isinstance(data["accounts"], list):
            # New multi-account format
            for account_data in data["accounts"]:
                decrypted = _decrypt_auth_account(account_data)
                # Check if any field was plaintext (needs migration)
                for field in _ENCRYPTED_AUTH_FIELDS:
                    val = account_data.get(field, "")
                    if isinstance(val, str) and val and not val.startswith(_ENC_PREFIX):
                        migrated = True
                account = AntigravityAccount.from_dict(decrypted)
                if account.refresh_token:
                    if await account.refresh_if_needed():
                        account.session = proxy.create_session()
                        proxy.antigravity_auth.add_account(account)
                        logger.info(f"Antigravity auth loaded for {account.email}")
                    else:
                        # Still add the account even if refresh failed - it might work later
                        proxy.antigravity_auth.add_account(account)
                        logger.warning(f"Antigravity token refresh failed for {account.email} -- may need re-login")
            logger.info(f"Loaded {len(proxy.antigravity_auth.accounts)} Antigravity account(s)")
        else:
            # Legacy single-account format
            decrypted = _decrypt_auth_account(data)
            for field in _ENCRYPTED_AUTH_FIELDS:
                val = data.get(field, "")
                if isinstance(val, str) and val and not val.startswith(_ENC_PREFIX):
                    migrated = True
            account = AntigravityAccount.from_dict(decrypted)
            if account.refresh_token:
                if await account.refresh_if_needed():
                    account.session = proxy.create_session()
                    proxy.antigravity_auth._legacy_account = account
                    logger.info(f"Antigravity auth loaded for {account.email} (legacy mode)")
                else:
                    proxy.antigravity_auth._legacy_account = account
                    logger.warning("Antigravity token refresh failed -- re-login required")
        if migrated:
            _save_antigravity_auth()
            logger.info("Migrated antigravity-auth.json tokens to encrypted storage")
    except Exception as e:
        logger.error(f"Failed to load Antigravity auth: {e}")


def _save_antigravity_auth():
    """Save Antigravity auth to cached file in multi-account format, encrypting tokens."""
    import proxy
    accounts_data = []

    # Include legacy account if it has a refresh token
    if proxy.antigravity_auth._legacy_account.refresh_token:
        accounts_data.append(_encrypt_auth_account(proxy.antigravity_auth._legacy_account.to_dict()))

    # Include all multi-account accounts
    for account in proxy.antigravity_auth.accounts:
        # Avoid duplicates with legacy account
        if account.email != proxy.antigravity_auth._legacy_account.email:
            accounts_data.append(_encrypt_auth_account(account.to_dict()))

    # If only one account, save in legacy format for backward compatibility
    if len(accounts_data) == 1:
        with open(proxy.ANTIGRAVITY_AUTH_FILE, "w", encoding="utf-8") as f:
            json.dump(accounts_data[0], f, indent=2)
    else:
        data = {"accounts": accounts_data}
        with open(proxy.ANTIGRAVITY_AUTH_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
