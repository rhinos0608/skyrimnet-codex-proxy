"""
OpenAI-compatible proxy using Claude Max subscription and/or OpenAI Codex CLI subscription.

Architecture:
  Startup: spawns ONE claude --print OR codex from clean temp dir → captures auth
  Per-request: direct aiohttp call to api.anthropic.com or api.openai.com via persistent session

Optimizations:
  - Clean temp dir capture: ~350 chars system-reminder vs ~16K (no CLAUDE.md bloat)
  - Persistent aiohttp session: reuses TCP+TLS connection (saves ~200-500ms/request)
  - Direct API calls: no subprocess per request

⚠️  TOS WARNING: Codex auth capture intercepts OAuth tokens from the Codex CLI.
    This may violate OpenAI's Terms of Service. Use at your own risk.

⚠️  TOS WARNING: Antigravity auth uses Google OAuth to access Google's Antigravity IDE API.
    This may violate Google's Terms of Service. Use at your own risk.
"""

import asyncio
import json
import time
import uuid
import logging
import shutil
import os
import sys
import copy
import tempfile
from contextlib import asynccontextmanager
from typing import Optional
from datetime import datetime, timedelta

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from aiohttp import web
import aiohttp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("proxy")

DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_CODEX_MODEL = "gpt-5.3-codex"
INTERCEPTOR_PORT = 9999

CLAUDE_PATH = shutil.which("claude")
if CLAUDE_PATH:
    logger.info(f"Using Claude CLI: {CLAUDE_PATH}")
else:
    logger.warning("Claude CLI not found on PATH - Claude provider unavailable")

CODEX_PATH = shutil.which("codex")
if CODEX_PATH:
    logger.info(f"Using Codex CLI: {CODEX_PATH}")
else:
    logger.warning("Codex CLI not found on PATH - Codex provider unavailable")


def create_session() -> aiohttp.ClientSession:
    """Create an aiohttp session with proper timeout/connector config for Windows."""
    connector = aiohttp.TCPConnector(force_close=True, enable_cleanup_closed=True)
    timeout = aiohttp.ClientTimeout(total=300, connect=30)
    return aiohttp.ClientSession(connector=connector, timeout=timeout)


def _get_codex_command() -> tuple[str, list[str]]:
    """
    Get the correct command to run Codex CLI on the current platform.

    On Windows, .cmd files need to be executed through cmd.exe.
    Returns (executable, prefix_args) where prefix_args are additional args
    that go before the codex arguments.
    """
    if not CODEX_PATH:
        return ("", [])

    # On Windows, .cmd/.bat files must be run through cmd.exe
    if sys.platform == "win32" and CODEX_PATH.lower().endswith((".cmd", ".bat")):
        # Use full path to cmd.exe to ensure it's found regardless of cwd
        cmd_exe = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "cmd.exe")
        return (cmd_exe, ["/c", CODEX_PATH])

    return (CODEX_PATH, [])

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


def _load_config() -> dict:
    """Load persisted config from disk, return empty dict on missing/corrupt file."""
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_config(data: dict) -> None:
    """Persist config dict to disk."""
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# --- Auth cache + persistent session ---

class AuthCache:
    def __init__(self):
        self.headers: Optional[dict] = None
        self.body_template: Optional[dict] = None
        self.session: Optional[aiohttp.ClientSession] = None

    @property
    def is_ready(self):
        return self.headers is not None and self.body_template is not None

auth = AuthCache()

# --- Codex OAuth Auth cache ---

class CodexAuthCache:
    """Stores OAuth tokens captured from Codex CLI auth flow."""

    def __init__(self):
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.expires_at: Optional[datetime] = None
        self.account_id: Optional[str] = None
        self.session: Optional[aiohttp.ClientSession] = None
        self._token_endpoint = "https://auth.openai.com/oauth/token"
        self._client_id = "codex_cli"  # Hardcoded client ID for Codex CLI

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

codex_auth = CodexAuthCache()

# --- Antigravity (Google IDE) OAuth Auth cache ---

# Antigravity OAuth constants (from opencode-antigravity-auth plugin)
ANTIGRAVITY_CLIENT_ID = "1071006060591-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com"
ANTIGRAVITY_CLIENT_SECRET = "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"
ANTIGRAVITY_REDIRECT_URI = "http://localhost:51121/oauth-callback"
ANTIGRAVITY_SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/cclog",
    "https://www.googleapis.com/auth/experimentsandconfigs",
]
ANTIGRAVITY_ENDPOINT_DAILY = "https://daily-cloudcode-pa.sandbox.googleapis.com"
ANTIGRAVITY_ENDPOINT_PROD = "https://cloudcode-pa.googleapis.com"
ANTIGRAVITY_ENDPOINT = ANTIGRAVITY_ENDPOINT_DAILY  # Use daily sandbox by default
ANTIGRAVITY_DEFAULT_PROJECT_ID = "rising-fact-p41fc"
ANTIGRAVITY_VERSION = "1.18.3"
ANTIGRAVITY_OAUTH_PORT = 51121


def get_antigravity_headers() -> dict:
    """Get headers for Antigravity API requests."""
    platform = "WINDOWS" if sys.platform == "win32" else "MACOS"
    return {
        "User-Agent": f"antigravity/{ANTIGRAVITY_VERSION} {'windows/amd64' if sys.platform == 'win32' else 'darwin/arm64'}",
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

        logger.info(f"Refreshing Antigravity OAuth token for {self.email}...")
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "grant_type": "refresh_token",
                    "refresh_token": self.refresh_token,
                    "client_id": ANTIGRAVITY_CLIENT_ID,
                    "client_secret": ANTIGRAVITY_CLIENT_SECRET,
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
        """Refresh tokens for all accounts. Returns True if at least one is ready."""
        if self.accounts:
            for account in self.accounts:
                if account.refresh_token:
                    await account.refresh_if_needed()
                    if account.is_ready and not account.session:
                        account.session = create_session()
            return any(a.is_ready for a in self.accounts)
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


antigravity_auth = AntigravityAuthCache()

# --- OpenRouter + Round-Robin state ---
_cfg = _load_config()
openrouter_api_key: Optional[str] = _cfg.get("openrouter_api_key") or None
if openrouter_api_key:
    logger.info("OpenRouter API key loaded from config.json")
ollama_api_key: Optional[str] = _cfg.get("ollama_api_key") or None
if ollama_api_key:
    logger.info("Ollama Cloud API key loaded from config.json")
_round_robin_counter: int = 0


def parse_model_list(model_field: str) -> list[str]:
    """Parse comma-separated model list from request, trimming whitespace."""
    return [m.strip() for m in model_field.split(",") if m.strip()]


def pick_model_round_robin(models: list[str]) -> str:
    """Pick next model from list using round-robin."""
    global _round_robin_counter
    model = models[_round_robin_counter % len(models)]
    _round_robin_counter += 1
    return model


def is_ollama_model(model: str) -> bool:
    """Ollama models use 'ollama:model' or 'ollama:model:tag' prefix."""
    return model.lower().startswith("ollama:")


def is_openrouter_model(model: str) -> bool:
    """OpenRouter models use 'provider/model' format (contain '/')."""
    return "/" in model


def is_codex_model(model: str) -> bool:
    """Codex/OpenAI models use gpt-5.*-codex* naming or codex-* naming."""
    model_lower = model.lower()
    return (
        model_lower.startswith("gpt-5.") or
        model_lower.startswith("gpt-5-") or
        model_lower.startswith("codex-")
    )


def is_antigravity_model(model: str) -> bool:
    """Antigravity models use antigravity-* naming."""
    model_lower = model.lower()
    return (
        model_lower.startswith("antigravity-") or
        model_lower.startswith("gemini-3") or
        model_lower.startswith("gemini-2.5") or
        model_lower.startswith("gpt-oss-")
    )


# --- MITM Interceptor (startup only) ---

async def interceptor_handler(request):
    body = await request.read()
    headers = dict(request.headers)
    headers.pop("Host", None)
    headers.pop("host", None)

    try:
        parsed = json.loads(body)
    except Exception:
        parsed = {}

    model = parsed.get("model", "")
    real_url = f"https://api.anthropic.com{request.path_qs}"

    # Skip haiku warmup and token counting
    if "haiku" in model or "count_tokens" in request.path:
        # Retry on transient network failures
        max_retries = 3
        last_error = None
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(real_url, data=body, headers=headers) as resp:
                        resp_body = await resp.read()
                        return web.Response(body=resp_body, status=resp.status,
                            headers={"Content-Type": resp.headers.get("Content-Type", "application/json")})
            except (aiohttp.ClientError, OSError) as e:
                last_error = e
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
        # All retries failed - return error response
        logger.error(f"Interceptor request failed after {max_retries} attempts: {last_error}")
        return web.Response(status=503, text=f"Network error: {last_error}")

    # Capture auth headers and body template
    if not auth.is_ready:
        auth.headers = dict(headers)
        # Strip tool definitions (60KB dead weight) and extended thinking
        parsed.pop("tools", None)
        parsed.pop("thinking", None)
        parsed.pop("context_management", None)
        auth.body_template = parsed
        template_size = len(json.dumps(parsed))
        logger.info(f"Captured {len(auth.headers)} headers + template ({template_size:,} bytes, tools stripped)")

    # Forward to real API
    async with aiohttp.ClientSession() as session:
        async with session.post(real_url, data=body, headers=headers) as resp:
            resp_body = await resp.read()
            return web.Response(body=resp_body, status=resp.status,
                headers={"Content-Type": resp.headers.get("Content-Type", "text/event-stream")})


async def start_interceptor():
    """Start MITM interceptor and capture auth from a clean temp dir."""
    iapp = web.Application()
    iapp.router.add_route("*", "/{path_info:.*}", interceptor_handler)

    runner = web.AppRunner(iapp)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", INTERCEPTOR_PORT)
    await site.start()
    logger.info(f"Interceptor on port {INTERCEPTOR_PORT}")

    max_retries = 3
    for attempt in range(max_retries):
        # Use clean temp dir to minimize system-reminder bloat (no CLAUDE.md, no skills)
        # ignore_cleanup_errors=True prevents Windows errors when subprocess still holds dir handle
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            env = os.environ.copy()
            env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{INTERCEPTOR_PORT}"

            logger.info(f"Warming up: capturing auth from claude --print... (attempt {attempt + 1}/{max_retries})")
            try:
                proc = await asyncio.create_subprocess_exec(
                    CLAUDE_PATH, "--print",
                    "--output-format", "text",
                    "--model", DEFAULT_MODEL,
                    "--no-session-persistence",
                    "--system-prompt", "Say ok",
                    "ok",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                    cwd=tmpdir,
                )
                await asyncio.wait_for(proc.communicate(), timeout=90)
                if auth.is_ready:
                    break  # Success, exit retry loop
            except (asyncio.TimeoutError, Exception) as e:
                logger.warning(f"Warmup attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s

    if auth.is_ready:
        # Create persistent session for all future API calls
        auth.session = create_session()
        logger.info("Auth captured — direct API mode active (persistent session)")
    else:
        logger.error("Failed to capture auth headers after %d attempts!", max_retries)

    return runner


# --- Codex MITM Interceptor (startup only) ---

CODEX_INTERCEPTOR_PORT = 9998


async def codex_interceptor_handler(request):
    """Handle Codex CLI traffic - captures OAuth tokens from auth flow."""
    body = await request.read()
    headers = dict(request.headers)
    headers.pop("Host", None)
    headers.pop("host", None)

    try:
        parsed = json.loads(body)
    except Exception:
        parsed = {}

    real_url = f"https://api.openai.com{request.path_qs}"

    # Check if this is an auth token request (to our interceptor)
    # Codex CLI makes a request to localhost:1455 to receive the OAuth callback
    if request.path == "/auth/callback" or request.path == "/oauth/callback":
        # This is the OAuth callback - capture the tokens
        auth_code = request.query.get("code")
        if auth_code and not codex_auth.access_token:
            # Exchange code for tokens
            try:
                async with aiohttp.ClientSession() as session:
                    token_payload = {
                        "grant_type": "authorization_code",
                        "code": auth_code,
                        "client_id": "codex_cli",
                        "redirect_uri": f"http://127.0.0.1:{CODEX_INTERCEPTOR_PORT}/auth/callback",
                    }
                    async with session.post(
                        "https://auth.openai.com/oauth/token",
                        json=token_payload,
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            codex_auth.access_token = data.get("access_token")
                            codex_auth.refresh_token = data.get("refresh_token")
                            expires_in = data.get("expires_in", 3600)
                            codex_auth.expires_at = datetime.now() + timedelta(seconds=expires_in)
                            codex_auth.account_id = data.get("account_id")
                            logger.info(f"Captured Codex OAuth tokens (expires in {expires_in}s)")
                            return web.Response(
                                status=200,
                                text="<html><body><h1>Authentication successful!</h1><p>You can close this window.</p></body></html>",
                                content_type="text/html",
                            )
            except Exception as e:
                logger.error(f"Failed to exchange OAuth code: {e}")

    # Capture authorization header if present
    auth_header = headers.get("Authorization", "")
    if auth_header.startswith("Bearer ") and not codex_auth.access_token:
        token = auth_header[7:]
        codex_auth.access_token = token
        # Assume 1 hour expiry if we don't know
        codex_auth.expires_at = datetime.now() + timedelta(hours=1)
        logger.info("Captured Codex Bearer token from request header")

    # Forward to real API
    async with aiohttp.ClientSession() as session:
        async with session.request(
            method=request.method,
            url=real_url,
            data=body,
            headers=headers,
        ) as resp:
            resp_body = await resp.read()
            return web.Response(
                body=resp_body,
                status=resp.status,
                headers={
                    "Content-Type": resp.headers.get("Content-Type", "application/json"),
                },
            )


async def start_codex_interceptor():
    """Load Codex auth from cached auth.json file (from 'codex login')."""
    if not CODEX_PATH:
        logger.warning("Codex CLI not found, skipping Codex auth capture")
        return None

    # Try to read cached auth from ~/.codex/auth.json
    auth_file = os.path.expanduser("~/.codex/auth.json")
    if os.path.exists(auth_file):
        try:
            with open(auth_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            tokens = data.get("tokens", {})
            if tokens.get("access_token"):
                codex_auth.access_token = tokens["access_token"]
                codex_auth.refresh_token = tokens.get("refresh_token")
                codex_auth.account_id = tokens.get("account_id")

                # Parse expiry from last_refresh + assume 24hr validity
                last_refresh = data.get("last_refresh")
                if last_refresh:
                    try:
                        # Parse ISO format timestamp
                        last_refresh_dt = datetime.fromisoformat(last_refresh.replace("Z", "+00:00"))
                        # Make it naive for comparison
                        last_refresh_dt = last_refresh_dt.replace(tzinfo=None)
                        # Assume 24 hour token validity
                        codex_auth.expires_at = last_refresh_dt + timedelta(hours=24)
                    except Exception:
                        codex_auth.expires_at = datetime.now() + timedelta(hours=23)

                logger.info(f"Loaded Codex auth from {auth_file}")
                if codex_auth.is_expired():
                    logger.warning("Codex token may be expired, will attempt refresh")
        except Exception as e:
            logger.error(f"Failed to read Codex auth file: {e}")

    if codex_auth.is_ready:
        codex_auth.session = create_session()
        logger.info("Codex auth loaded - direct API mode active")
    else:
        logger.warning("Failed to load Codex auth")
        logger.info("Note: Run 'codex login' first, then restart proxy")

    return None  # No interceptor needed - we read from file


# --- Direct API call ---

def _build_api_body(system_prompt: Optional[str], messages: list, model: str) -> dict:
    """Build Anthropic API request body from template."""
    body = copy.deepcopy(auth.body_template)

    # 1. Replace system prompt (keep billing block 0)
    billing = body["system"][0]
    body["system"] = [billing]
    if system_prompt:
        body["system"].append({"type": "text", "text": system_prompt})

    # 2. Build full conversation, preserving template auth blocks in first user msg
    # The template's first user message contains system-reminder content blocks
    # that authenticate this as a Claude Code request — we must keep them.
    auth_blocks = []
    template_first = body["messages"][0] if body["messages"] else {}
    if isinstance(template_first.get("content"), list):
        for block in template_first["content"]:
            if isinstance(block, dict) and block.get("type") == "text":
                if "<system-reminder>" in block.get("text", ""):
                    auth_blocks.append(block)

    new_messages = []
    for i, m in enumerate(messages):
        if i == 0 and m["role"] == "user":
            # First user message: prepend auth blocks from template
            content = auth_blocks + [{"type": "text", "text": m["content"]}]
            new_messages.append({"role": "user", "content": content})
        else:
            new_messages.append({"role": m["role"], "content": [{"type": "text", "text": m["content"]}]})
    body["messages"] = new_messages

    # 3. Model, streaming, and disable extended thinking
    body["model"] = model
    body["stream"] = True
    body.pop("thinking", None)
    return body


async def call_api_direct(system_prompt: Optional[str], messages: list, model: str, max_tokens: int) -> str:
    """Direct API call, collects full response (non-streaming to caller)."""

    body = _build_api_body(system_prompt, messages, model)
    headers = dict(auth.headers)
    body_bytes = json.dumps(body).encode("utf-8")
    headers["Content-Length"] = str(len(body_bytes))

    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] -> {model} ({len(messages)} msgs)")
    start = time.time()

    session = auth.session or create_session()
    try:
        async with session.post(
            "https://api.anthropic.com/v1/messages?beta=true",
            data=body_bytes,
            headers=headers,
        ) as resp:
            elapsed = time.time() - start

            if resp.status != 200:
                error_body = await resp.read()
                error_text = error_body.decode("utf-8", errors="replace")
                logger.error(f"[{request_id}] API {resp.status}: {error_text[:300]}")
                if resp.status in (401, 403) or "credential" in error_text.lower():
                    logger.warning("Auth expired -- restart proxy to re-auth")
                    auth.headers = None
                    auth.body_template = None
                raise HTTPException(status_code=resp.status, detail=error_text[:200])

            resp_body = await resp.read()
            text_parts = []
            content_type = resp.headers.get("Content-Type", "")
            if "text/event-stream" in content_type:
                for line in resp_body.decode("utf-8", errors="replace").split("\n"):
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            continue
                        try:
                            event = json.loads(data_str)
                            if event.get("type") == "content_block_delta":
                                delta = event.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    text_parts.append(delta.get("text", ""))
                        except json.JSONDecodeError:
                            pass
            else:
                try:
                    data = json.loads(resp_body)
                    for block in data.get("content", []):
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                except json.JSONDecodeError:
                    pass

            response_text = "".join(text_parts)
            logger.info(f"[{request_id}] <- {len(response_text)} chars ({elapsed:.1f}s)")
            return response_text
    finally:
        if not auth.session:
            await session.close()


async def call_api_streaming(system_prompt: Optional[str], messages: list, model: str, max_tokens: int):
    """Direct API call, yields OpenAI-format SSE chunks as they arrive."""

    body = _build_api_body(system_prompt, messages, model)
    headers = dict(auth.headers)
    body_bytes = json.dumps(body).encode("utf-8")
    headers["Content-Length"] = str(len(body_bytes))

    request_id = uuid.uuid4().hex[:8]
    cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    created = int(time.time())
    logger.info(f"[{request_id}] -> {model} ({len(messages)} msgs, stream)")
    start = time.time()
    total_chars = 0

    session = auth.session or create_session()
    owns_session = not auth.session
    try:
        async with session.post(
            "https://api.anthropic.com/v1/messages?beta=true",
            data=body_bytes,
            headers=headers,
        ) as resp:
            if resp.status != 200:
                error_body = await resp.read()
                error_text = error_body.decode("utf-8", errors="replace")
                logger.error(f"[{request_id}] API {resp.status}: {error_text[:300]}")
                if resp.status in (401, 403) or "credential" in error_text.lower():
                    logger.warning("Auth expired -- restart proxy to re-auth")
                    auth.headers = None
                    auth.body_template = None
                # Yield an error chunk so client sees the failure
                err_chunk = {"id": cmpl_id, "object": "chat.completion.chunk", "created": created, "model": model,
                             "choices": [{"index": 0, "delta": {"content": f"[API Error {resp.status}]"}, "finish_reason": None}]}
                yield f"data: {json.dumps(err_chunk)}\n\n"
                yield "data: [DONE]\n\n"
                return

            # Stream Claude SSE -> OpenAI SSE
            # Send initial chunk with role
            role_chunk = {"id": cmpl_id, "object": "chat.completion.chunk", "created": created, "model": model,
                          "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]}
            yield f"data: {json.dumps(role_chunk)}\n\n"

            buffer = ""
            async for raw_chunk in resp.content.iter_any():
                buffer += raw_chunk.decode("utf-8", errors="replace")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if not data_str or data_str == "[DONE]":
                        continue
                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    if event.get("type") == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text = delta.get("text", "")
                            if text:
                                total_chars += len(text)
                                oai_chunk = {"id": cmpl_id, "object": "chat.completion.chunk", "created": created, "model": model,
                                             "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]}
                                yield f"data: {json.dumps(oai_chunk)}\n\n"

            # Final chunk with finish_reason
            stop_chunk = {"id": cmpl_id, "object": "chat.completion.chunk", "created": created, "model": model,
                          "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            yield f"data: {json.dumps(stop_chunk)}\n\n"
            yield "data: [DONE]\n\n"

            elapsed = time.time() - start
            logger.info(f"[{request_id}] <- {total_chars} chars ({elapsed:.1f}s, streamed)")
    finally:
        if owns_session:
            await session.close()


# --- Codex (OpenAI) API calls ---

def _convert_messages_to_codex_input(messages: list) -> list:
    """Convert OpenAI-format messages to Codex Responses API input format."""
    input_items = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        if role == "system":
            # System messages go in instructions field, not input
            continue
        elif role == "user":
            input_items.append({"role": "user", "content": content})
        elif role == "assistant":
            input_items.append({"role": "assistant", "content": content})
    return input_items


def _create_isolated_codex_home() -> tuple[str, dict]:
    """
    Create an isolated HOME directory with minimal Codex config.
    Returns (isolated_home_path, env_dict).
    """
    # Create isolated home directory
    isolated_home = tempfile.mkdtemp(prefix="codex_isolated_")
    codex_dir = os.path.join(isolated_home, ".codex")
    os.makedirs(codex_dir, exist_ok=True)

    # Write minimal config.toml (no instructions, no skills)
    config_content = f'''# Minimal isolated Codex config
model = "{DEFAULT_CODEX_MODEL}"

[approval]
mode = "suggest"

[features]
# Disable all optional features for clean isolation
'''
    with open(os.path.join(codex_dir, "config.toml"), "w", encoding="utf-8") as f:
        f.write(config_content)

    # Copy auth.json from real home if it exists
    real_auth_path = os.path.expanduser("~/.codex/auth.json")
    if os.path.exists(real_auth_path):
        import shutil
        shutil.copy2(real_auth_path, os.path.join(codex_dir, "auth.json"))

    # Build isolated environment
    env = os.environ.copy()
    env["HOME"] = isolated_home
    env["USERPROFILE"] = isolated_home  # Windows
    env["CODEX_HOME"] = codex_dir

    return isolated_home, env


def _cleanup_isolated_home(isolated_home: str) -> None:
    """Clean up the isolated HOME directory after use."""
    import shutil
    try:
        shutil.rmtree(isolated_home, ignore_errors=True)
    except Exception:
        pass


async def call_codex_direct(
    system_prompt: Optional[str],
    messages: list,
    model: str,
    max_tokens: int,
) -> str:
    """Spawn Codex CLI subprocess with isolated HOME (clean config, no global instructions)."""
    if not CODEX_PATH:
        raise HTTPException(status_code=503, detail="Codex CLI not installed")

    # Build prompt from messages
    prompt_parts = []
    if system_prompt:
        prompt_parts.append(f"System: {system_prompt}")
    for m in messages:
        role = m["role"].capitalize()
        prompt_parts.append(f"{role}: {m['content']}")
    prompt_parts.append("Assistant:")
    full_prompt = "\n\n".join(prompt_parts)

    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] -> Codex CLI {model} ({len(messages)} msgs, isolated)")
    start = time.time()

    isolated_home = None
    try:
        # Create isolated HOME with clean config
        isolated_home, env = _create_isolated_codex_home()

        executable, prefix_args = _get_codex_command()
        logger.info(f"[{request_id}] Executing: {executable} {' '.join(prefix_args)} exec --model {model}")
        proc = await asyncio.create_subprocess_exec(
            executable,
            *prefix_args,
            "exec",
            "--model", model,
            "--json",
            "--dangerously-bypass-approvals-and-sandbox",
            "--ephemeral",
            "-",  # Read prompt from stdin
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=isolated_home,  # Run from isolated directory
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=full_prompt.encode("utf-8")), timeout=180
        )
        elapsed = time.time() - start

        if proc.returncode != 0:
            error_text = stderr.decode("utf-8", errors="replace")
            logger.error(f"[{request_id}] Codex CLI failed: {error_text[:300]}")
            raise HTTPException(status_code=500, detail=f"Codex CLI error: {error_text[:200]}")

        # Parse JSONL output and extract agent_message text
        response_text = ""
        raw_output = stdout.decode("utf-8", errors="replace").strip()
        for line in raw_output.split("\n"):
            if line.strip():
                try:
                    event = json.loads(line)
                    event_type = event.get("type", "")

                    # Look for agent_message items
                    if event_type == "item.completed":
                        item = event.get("item", {})
                        if item.get("type") == "agent_message":
                            response_text = item.get("text", "")
                    elif event_type == "message.completed":
                        for content in event.get("message", {}).get("content", []):
                            if content.get("type") == "output_text":
                                response_text = content.get("text", "")
                except json.JSONDecodeError:
                    pass

        if not response_text:
            response_text = raw_output

        logger.info(f"[{request_id}] <- {len(response_text)} chars ({elapsed:.1f}s)")
        return response_text
    except asyncio.TimeoutError:
        logger.error(f"[{request_id}] Codex CLI timeout")
        raise HTTPException(status_code=504, detail="Codex CLI timeout")
    finally:
        if isolated_home:
            _cleanup_isolated_home(isolated_home)


async def call_codex_streaming(
    system_prompt: Optional[str],
    messages: list,
    model: str,
    max_tokens: int,
):
    """Spawn Codex CLI subprocess with isolated HOME and yield output as SSE stream."""
    if not CODEX_PATH:
        yield 'data: {"error": "Codex CLI not installed"}\n\n'
        yield "data: [DONE]\n\n"
        return

    # Build prompt from messages
    prompt_parts = []
    if system_prompt:
        prompt_parts.append(f"System: {system_prompt}")
    for m in messages:
        role = m["role"].capitalize()
        prompt_parts.append(f"{role}: {m['content']}")
    prompt_parts.append("Assistant:")
    full_prompt = "\n\n".join(prompt_parts)

    request_id = uuid.uuid4().hex[:8]
    cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    created = int(time.time())
    logger.info(f"[{request_id}] -> Codex CLI {model} ({len(messages)} msgs, isolated stream)")
    start = time.time()

    # Send initial chunk with role
    role_chunk = {
        "id": cmpl_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]
    }
    yield f"data: {json.dumps(role_chunk)}\n\n"

    isolated_home = None
    try:
        # Create isolated HOME with clean config
        isolated_home, env = _create_isolated_codex_home()

        executable, prefix_args = _get_codex_command()
        proc = await asyncio.create_subprocess_exec(
            executable,
            *prefix_args,
            "exec",
            "--model", model,
            "--json",
            "--dangerously-bypass-approvals-and-sandbox",
            "--ephemeral",
            "-",  # Read prompt from stdin
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=isolated_home,
        )

        # Collect all output first (Codex CLI buffers output)
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=full_prompt.encode("utf-8")), timeout=180
        )
        elapsed = time.time() - start

        if proc.returncode != 0:
            error_text = stderr.decode("utf-8", errors="replace")
            logger.error(f"[{request_id}] Codex CLI failed: {error_text[:300]}")
            err_chunk = {
                "id": cmpl_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"content": f"[Codex error: {error_text[:100]}]"}, "finish_reason": None}]
            }
            yield f"data: {json.dumps(err_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Parse JSONL and extract agent_message
        response_text = ""
        raw_output = stdout.decode("utf-8", errors="replace").strip()
        for line in raw_output.split("\n"):
            if line.strip():
                try:
                    event = json.loads(line)
                    if event.get("type") == "item.completed":
                        item = event.get("item", {})
                        if item.get("type") == "agent_message":
                            response_text = item.get("text", "")
                except json.JSONDecodeError:
                    pass

        # Stream the response in chunks for realistic streaming feel
        if response_text:
            chunk_size = 20  # characters per chunk
            for i in range(0, len(response_text), chunk_size):
                chunk_text = response_text[i:i+chunk_size]
                oai_chunk = {
                    "id": cmpl_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": chunk_text}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(oai_chunk)}\n\n"
                await asyncio.sleep(0.02)  # Small delay for streaming feel

        # Final chunk
        stop_chunk = {
            "id": cmpl_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        }
        yield f"data: {json.dumps(stop_chunk)}\n\n"
        yield "data: [DONE]\n\n"

        logger.info(f"[{request_id}] <- {len(response_text)} chars ({elapsed:.1f}s, simulated stream)")
    except asyncio.TimeoutError:
        logger.error(f"[{request_id}] Codex CLI timeout")
        err_chunk = {
            "id": cmpl_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": "[Codex CLI timeout]"}, "finish_reason": None}]
        }
        yield f"data: {json.dumps(err_chunk)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"[{request_id}] Codex CLI error: {e}")
        err_chunk = {
            "id": cmpl_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": f"[Codex CLI error: {e}]"}, "finish_reason": None}]
        }
        yield f"data: {json.dumps(err_chunk)}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        if isolated_home:
            _cleanup_isolated_home(isolated_home)


# --- OpenRouter API calls ---

async def call_openrouter_direct(system_prompt: Optional[str], messages: list, model: str, max_tokens: int, **extra_params) -> str:
    """Forward request to OpenRouter (OpenAI-compatible), collect full response."""
    if not openrouter_api_key:
        raise HTTPException(status_code=500, detail="OpenRouter API key not configured")

    oai_messages = []
    if system_prompt:
        oai_messages.append({"role": "system", "content": system_prompt})
    for m in messages:
        oai_messages.append({"role": m["role"], "content": m["content"]})

    payload = {"model": model, "messages": oai_messages, "max_tokens": max_tokens}
    payload.update({k: v for k, v in extra_params.items() if v is not None})
    headers = {"Authorization": f"Bearer {openrouter_api_key}", "Content-Type": "application/json"}

    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] -> OpenRouter {model} ({len(messages)} msgs)")
    start = time.time()

    session = auth.session or create_session()
    try:
        async with session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json=payload, headers=headers,
        ) as resp:
            elapsed = time.time() - start
            if resp.status != 200:
                error_body = await resp.text()
                logger.error(f"[{request_id}] OpenRouter {resp.status}: {error_body[:300]}")
                raise HTTPException(status_code=resp.status, detail=error_body[:200])
            data = await resp.json()
            text = data["choices"][0]["message"]["content"]
            logger.info(f"[{request_id}] <- {len(text)} chars ({elapsed:.1f}s, OpenRouter)")
            return text
    finally:
        if not auth.session:
            await session.close()


async def call_openrouter_streaming(system_prompt: Optional[str], messages: list, model: str, max_tokens: int, **extra_params):
    """Forward request to OpenRouter with streaming, passthrough SSE directly."""
    if not openrouter_api_key:
        yield 'data: {"error": "OpenRouter API key not configured"}\n\n'
        yield "data: [DONE]\n\n"
        return

    oai_messages = []
    if system_prompt:
        oai_messages.append({"role": "system", "content": system_prompt})
    for m in messages:
        oai_messages.append({"role": m["role"], "content": m["content"]})

    payload = {"model": model, "messages": oai_messages, "max_tokens": max_tokens, "stream": True}
    payload.update({k: v for k, v in extra_params.items() if v is not None})
    headers = {"Authorization": f"Bearer {openrouter_api_key}", "Content-Type": "application/json"}

    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] -> OpenRouter {model} ({len(messages)} msgs, stream)")
    start = time.time()

    session = auth.session or create_session()
    owns_session = not auth.session
    try:
        async with session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json=payload, headers=headers,
        ) as resp:
            if resp.status != 200:
                error_body = await resp.text()
                logger.error(f"[{request_id}] OpenRouter {resp.status}: {error_body[:300]}")
                cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
                err_chunk = {"id": cmpl_id, "object": "chat.completion.chunk",
                             "created": int(time.time()), "model": model,
                             "choices": [{"index": 0, "delta": {"content": f"[OpenRouter Error {resp.status}]"}, "finish_reason": None}]}
                yield f"data: {json.dumps(err_chunk)}\n\n"
                yield "data: [DONE]\n\n"
                return

            # OpenRouter returns OpenAI-format SSE — passthrough directly
            buffer = ""
            async for raw_chunk in resp.content.iter_any():
                buffer += raw_chunk.decode("utf-8", errors="replace")
                while "\n\n" in buffer:
                    event, buffer = buffer.split("\n\n", 1)
                    event = event.strip()
                    if event:
                        yield event + "\n\n"

            if buffer.strip():
                yield buffer.strip() + "\n\n"

            elapsed = time.time() - start
            logger.info(f"[{request_id}] <- stream done ({elapsed:.1f}s, OpenRouter)")
    finally:
        if owns_session:
            await session.close()


# --- Ollama API calls ---

async def call_ollama_direct(system_prompt: Optional[str], messages: list, model: str, max_tokens: int, **extra_params) -> str:
    """Forward request to Ollama (OpenAI-compatible), collect full response."""
    api_model = model[len("ollama:"):]
    if ollama_api_key:
        endpoint = "https://ollama.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {ollama_api_key}", "Content-Type": "application/json"}
    else:
        endpoint = "http://localhost:11434/v1/chat/completions"
        headers = {"Content-Type": "application/json"}

    oai_messages = []
    if system_prompt:
        oai_messages.append({"role": "system", "content": system_prompt})
    for m in messages:
        oai_messages.append({"role": m["role"], "content": m["content"]})

    payload = {"model": api_model, "messages": oai_messages, "max_tokens": max_tokens}
    payload.update({k: v for k, v in extra_params.items() if v is not None})

    request_id = uuid.uuid4().hex[:8]
    target = "Ollama Cloud" if ollama_api_key else f"Ollama local ({api_model})"
    logger.info(f"[{request_id}] -> {target} ({len(messages)} msgs)")
    start = time.time()

    session = auth.session or create_session()
    try:
        async with session.post(endpoint, json=payload, headers=headers) as resp:
            elapsed = time.time() - start
            if resp.status in (401, 403):
                raise HTTPException(status_code=401, detail="Ollama Cloud auth failed — check API key")
            if resp.status == 429:
                raise HTTPException(status_code=429, detail="Ollama rate limit exceeded")
            if resp.status != 200:
                error_body = await resp.text()
                logger.error(f"[{request_id}] Ollama {resp.status}: {error_body[:300]}")
                raise HTTPException(status_code=resp.status, detail=error_body[:200])
            data = await resp.json()
            text = data["choices"][0]["message"]["content"]
            logger.info(f"[{request_id}] <- {len(text)} chars ({elapsed:.1f}s, Ollama)")
            return text
    except aiohttp.ClientConnectorError:
        raise HTTPException(status_code=503, detail="Ollama not running at localhost:11434")
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Ollama request timed out")
    finally:
        if not auth.session:
            await session.close()


async def call_ollama_streaming(system_prompt: Optional[str], messages: list, model: str, max_tokens: int, **extra_params):
    """Forward request to Ollama with streaming, passthrough SSE directly."""
    api_model = model[len("ollama:"):]
    if ollama_api_key:
        endpoint = "https://ollama.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {ollama_api_key}", "Content-Type": "application/json"}
    else:
        endpoint = "http://localhost:11434/v1/chat/completions"
        headers = {"Content-Type": "application/json"}

    oai_messages = []
    if system_prompt:
        oai_messages.append({"role": "system", "content": system_prompt})
    for m in messages:
        oai_messages.append({"role": m["role"], "content": m["content"]})

    payload = {"model": api_model, "messages": oai_messages, "max_tokens": max_tokens, "stream": True}
    payload.update({k: v for k, v in extra_params.items() if v is not None})

    request_id = uuid.uuid4().hex[:8]
    target = "Ollama Cloud" if ollama_api_key else f"Ollama local ({api_model})"
    logger.info(f"[{request_id}] -> {target} ({len(messages)} msgs, stream)")
    start = time.time()

    session = auth.session or create_session()
    owns_session = not auth.session
    try:
        async with session.post(endpoint, json=payload, headers=headers) as resp:
            if resp.status != 200:
                error_body = await resp.text()
                logger.error(f"[{request_id}] Ollama {resp.status}: {error_body[:300]}")
                cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
                err_chunk = {"id": cmpl_id, "object": "chat.completion.chunk",
                             "created": int(time.time()), "model": model,
                             "choices": [{"index": 0, "delta": {"content": f"[Ollama Error {resp.status}]"}, "finish_reason": None}]}
                yield f"data: {json.dumps(err_chunk)}\n\n"
                yield "data: [DONE]\n\n"
                return

            # Ollama /v1 returns OpenAI-format SSE — passthrough directly
            buffer = ""
            async for raw_chunk in resp.content.iter_any():
                buffer += raw_chunk.decode("utf-8", errors="replace")
                while "\n\n" in buffer:
                    event, buffer = buffer.split("\n\n", 1)
                    event = event.strip()
                    if event:
                        yield event + "\n\n"

            if buffer.strip():
                yield buffer.strip() + "\n\n"

            elapsed = time.time() - start
            logger.info(f"[{request_id}] <- stream done ({elapsed:.1f}s, Ollama)")
    except aiohttp.ClientConnectorError:
        cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
        err_chunk = {"id": cmpl_id, "object": "chat.completion.chunk",
                     "created": int(time.time()), "model": model,
                     "choices": [{"index": 0, "delta": {"content": "[Ollama Error: not running at localhost:11434]"}, "finish_reason": None}]}
        yield f"data: {json.dumps(err_chunk)}\n\n"
        yield "data: [DONE]\n\n"
    except asyncio.TimeoutError:
        cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
        err_chunk = {"id": cmpl_id, "object": "chat.completion.chunk",
                     "created": int(time.time()), "model": model,
                     "choices": [{"index": 0, "delta": {"content": "[Ollama Error: request timed out]"}, "finish_reason": None}]}
        yield f"data: {json.dumps(err_chunk)}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        if owns_session:
            await session.close()


# --- Antigravity Auth Loading ---

ANTIGRAVITY_AUTH_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "antigravity-auth.json")


async def load_antigravity_auth():
    """Load Antigravity auth from cached file. Supports both legacy single-account and new multi-account formats."""
    global antigravity_auth

    if not os.path.exists(ANTIGRAVITY_AUTH_FILE):
        logger.info("No Antigravity auth file found -- visit /config/antigravity-login")
        return

    try:
        with open(ANTIGRAVITY_AUTH_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Check if it's the new multi-account format (has "accounts" array)
        if "accounts" in data and isinstance(data["accounts"], list):
            # New multi-account format
            for account_data in data["accounts"]:
                account = AntigravityAccount.from_dict(account_data)
                if account.refresh_token:
                    if await account.refresh_if_needed():
                        account.session = create_session()
                        antigravity_auth.add_account(account)
                        logger.info(f"Antigravity auth loaded for {account.email}")
                    else:
                        # Still add the account even if refresh failed - it might work later
                        antigravity_auth.add_account(account)
                        logger.warning(f"Antigravity token refresh failed for {account.email} -- may need re-login")
            logger.info(f"Loaded {len(antigravity_auth.accounts)} Antigravity account(s)")
        else:
            # Legacy single-account format
            account = AntigravityAccount.from_dict(data)
            if account.refresh_token:
                if await account.refresh_if_needed():
                    account.session = create_session()
                    antigravity_auth._legacy_account = account
                    logger.info(f"Antigravity auth loaded for {account.email} (legacy mode)")
                else:
                    antigravity_auth._legacy_account = account
                    logger.warning("Antigravity token refresh failed -- re-login required")
    except Exception as e:
        logger.error(f"Failed to load Antigravity auth: {e}")


def _save_antigravity_auth():
    """Save Antigravity auth to cached file in multi-account format."""
    accounts_data = []

    # Include legacy account if it has a refresh token
    if antigravity_auth._legacy_account.refresh_token:
        accounts_data.append(antigravity_auth._legacy_account.to_dict())

    # Include all multi-account accounts
    for account in antigravity_auth.accounts:
        # Avoid duplicates with legacy account
        if account.email != antigravity_auth._legacy_account.email:
            accounts_data.append(account.to_dict())

    # If only one account, save in legacy format for backward compatibility
    if len(accounts_data) == 1:
        with open(ANTIGRAVITY_AUTH_FILE, "w", encoding="utf-8") as f:
            json.dump(accounts_data[0], f, indent=2)
    else:
        data = {"accounts": accounts_data}
        with open(ANTIGRAVITY_AUTH_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


# --- Antigravity (Google IDE) API calls ---

def _convert_messages_to_antigravity(messages: list, system_prompt: Optional[str] = None) -> dict:
    """Convert OpenAI-format messages to Antigravity/Gemini-style request format."""
    # Build contents array (Gemini-style)
    contents = []
    for m in messages:
        role = m["role"]
        # Map assistant -> model for Gemini format
        gemini_role = "model" if role == "assistant" else role
        contents.append({
            "role": gemini_role,
            "parts": [{"text": m["content"]}]
        })

    request_body = {
        "contents": contents,
    }

    # System instruction must be an object with parts
    if system_prompt:
        request_body["systemInstruction"] = {
            "parts": [{"text": system_prompt}]
        }

    return request_body


def _get_antigravity_model_id(model: str) -> str:
    """Convert proxy model name to Antigravity API model ID."""
    model_lower = model.lower()

    # Strip antigravity- prefix if present
    if model_lower.startswith("antigravity-"):
        model_lower = model_lower[12:]  # Remove "antigravity-"

    # Map model names to Antigravity model IDs
    model_mappings = {
        "gemini-3-pro": "gemini-3-pro-preview",
        "gemini-3-pro-preview": "gemini-3-pro-preview",
        "gemini-3.1-pro": "gemini-3.1-pro-preview",
        "gemini-3.1-pro-preview": "gemini-3.1-pro-preview",
        "gemini-3-flash": "gemini-3-flash-preview",
        "gemini-3-flash-preview": "gemini-3-flash-preview",
        "gemini-2.5-flash": "gemini-2.5-flash",
        "gemini-2.5-pro": "gemini-2.5-pro",
        "claude-sonnet-4-6": "claude-sonnet-4-6",
        "claude-opus-4-6": "claude-opus-4-6-thinking",
        "claude-opus-4-6-thinking": "claude-opus-4-6-thinking",
        "gpt-oss-120b-medium": "gpt-oss-120b-medium",
    }

    return model_mappings.get(model_lower, model_lower)


async def call_antigravity_direct(
    system_prompt: Optional[str],
    messages: list,
    model: str,
    max_tokens: int,
) -> str:
    """Call Antigravity API directly with multi-account fallback on 503 errors."""
    if not antigravity_auth.is_ready:
        await antigravity_auth.refresh_if_needed()
        if not antigravity_auth.is_ready:
            raise HTTPException(status_code=503, detail="Antigravity auth not ready -- run /config/antigravity-login first")

    ag_model = _get_antigravity_model_id(model)
    request_contents = _convert_messages_to_antigravity(messages, system_prompt)

    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] -> Antigravity {ag_model} ({len(messages)} msgs)")
    start = time.time()

    # Get all available accounts for fallback
    all_accounts = antigravity_auth._get_active_accounts()
    if not all_accounts:
        raise HTTPException(status_code=503, detail="No Antigravity accounts available")

    last_error = None
    for account in all_accounts:
        if not account.is_ready:
            if not await account.refresh_if_needed():
                continue
            if account.is_ready and not account.session:
                account.session = create_session()

        if not account.is_ready:
            continue

        # Build payload with this account's project_id
        payload = {
            "project": account.project_id or ANTIGRAVITY_DEFAULT_PROJECT_ID,
            "model": ag_model,
            "request": {
                **request_contents,
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                    "temperature": 0.7,
                }
            },
            "userAgent": "antigravity",
            "requestId": uuid.uuid4().hex,
        }

        headers = account.get_auth_headers()
        session = account.session or create_session()
        owns_session = not account.session

        try:
            async with session.post(
                f"{ANTIGRAVITY_ENDPOINT}/v1internal:generateContent",
                json=payload,
                headers=headers,
            ) as resp:
                elapsed = time.time() - start

                if resp.status != 200:
                    error_body = await resp.text()
                    logger.error(f"[{request_id}] Antigravity {resp.status} (account: {account.email}): {error_body[:300]}")

                    # Handle specific error codes - fallback for transient/recoverable errors
                    if resp.status in (503, 429, 403):  # Service unavailable, rate limited, or permission denied
                        antigravity_auth.mark_account_error(account, resp.status)
                        last_error = HTTPException(status_code=resp.status, detail=error_body[:200])
                        logger.info(f"[{request_id}] Falling back to next account due to {resp.status}")
                        continue  # Try next account

                    if resp.status == 401:
                        # Token expired, try refresh and retry once
                        if await account.refresh_if_needed():
                            headers = account.get_auth_headers()
                            async with session.post(
                                f"{ANTIGRAVITY_ENDPOINT}/v1internal:generateContent",
                                json=payload,
                                headers=headers,
                            ) as retry_resp:
                                if retry_resp.status == 200:
                                    data = await retry_resp.json()
                                    antigravity_auth.mark_account_success(account)
                                    return _extract_antigravity_text(data)
                                elif retry_resp.status in (503, 429, 403):
                                    antigravity_auth.mark_account_error(account, retry_resp.status)
                                    last_error = HTTPException(status_code=retry_resp.status, detail=error_body[:200])
                                    continue  # Try next account
                        antigravity_auth.mark_account_error(account, resp.status)
                        last_error = HTTPException(status_code=resp.status, detail=error_body[:200])
                        continue  # Try next account

                    # For other errors, raise immediately
                    raise HTTPException(status_code=resp.status, detail=error_body[:200])

                data = await resp.json()
                response_text = _extract_antigravity_text(data)
                antigravity_auth.mark_account_success(account)
                logger.info(f"[{request_id}] <- {len(response_text)} chars ({elapsed:.1f}s, Antigravity, account: {account.email})")
                return response_text
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[{request_id}] Antigravity request error for {account.email}: {e}")
            antigravity_auth.mark_account_error(account, 500)
            last_error = HTTPException(status_code=500, detail=str(e))
        finally:
            if owns_session:
                await session.close()

    # All accounts failed
    if last_error:
        raise last_error
    raise HTTPException(status_code=503, detail="All Antigravity accounts failed")


def _extract_antigravity_text(data: dict) -> str:
    """Extract text from Antigravity response."""
    text_parts = []
    response = data.get("response", data)
    candidates = response.get("candidates", [])
    for candidate in candidates:
        content = candidate.get("content", {})
        parts = content.get("parts", [])
        for part in parts:
            # Skip thought blocks (they have thought: true or thoughtSignature)
            if part.get("thought") or part.get("thoughtSignature"):
                continue
            text = part.get("text", "")
            if text:
                text_parts.append(text)
    return "".join(text_parts)


async def call_antigravity_streaming(
    system_prompt: Optional[str],
    messages: list,
    model: str,
    max_tokens: int,
):
    """Call Antigravity API with streaming and multi-account fallback on initial errors."""
    if not antigravity_auth.is_ready:
        await antigravity_auth.refresh_if_needed()
        if not antigravity_auth.is_ready:
            yield 'data: {"error": "Antigravity auth not ready"}\n\n'
            yield "data: [DONE]\n\n"
            return

    ag_model = _get_antigravity_model_id(model)
    request_contents = _convert_messages_to_antigravity(messages, system_prompt)

    request_id = uuid.uuid4().hex[:8]
    cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    created = int(time.time())
    logger.info(f"[{request_id}] -> Antigravity {ag_model} ({len(messages)} msgs, stream)")
    start = time.time()
    total_chars = 0

    # Send initial chunk with role
    role_chunk = {
        "id": cmpl_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]
    }
    yield f"data: {json.dumps(role_chunk)}\n\n"

    # Get all available accounts for fallback
    all_accounts = antigravity_auth._get_active_accounts()
    if not all_accounts:
        err_chunk = {
            "id": cmpl_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": "[Antigravity Error: No accounts available]"}, "finish_reason": None}]
        }
        yield f"data: {json.dumps(err_chunk)}\n\n"
        yield "data: [DONE]\n\n"
        return

    last_error_status = None
    last_error_msg = None

    for account in all_accounts:
        if not account.is_ready:
            if not await account.refresh_if_needed():
                continue
            if account.is_ready and not account.session:
                account.session = create_session()

        if not account.is_ready:
            continue

        # Build payload with this account's project_id
        payload = {
            "project": account.project_id or ANTIGRAVITY_DEFAULT_PROJECT_ID,
            "model": ag_model,
            "request": {
                **request_contents,
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                    "temperature": 0.7,
                }
            },
            "userAgent": "antigravity",
            "requestId": uuid.uuid4().hex,
        }

        headers = account.get_auth_headers()
        headers["Accept"] = "text/event-stream"

        session = account.session or create_session()
        owns_session = not account.session

        async def make_stream_request(headers_to_use):
            """Make the streaming request, return (resp, should_retry) tuple."""
            resp = await session.post(
                f"{ANTIGRAVITY_ENDPOINT}/v1internal:streamGenerateContent?alt=sse",
                json=payload,
                headers=headers_to_use,
            )
            if resp.status == 401:
                # Token might be expired, try refresh
                if await account.refresh_if_needed():
                    return resp, True  # Signal retry needed
            return resp, False

        try:
            resp, should_retry = await make_stream_request(headers)

            if should_retry:
                # Close the old response and retry with new headers
                resp.close()
                headers = account.get_auth_headers()
                headers["Accept"] = "text/event-stream"
                resp, should_retry = await make_stream_request(headers)

            async with resp:
                if resp.status != 200:
                    error_body = await resp.text()
                    logger.error(f"[{request_id}] Antigravity {resp.status} (account: {account.email}): {error_body[:300]}")

                    # Handle specific error codes - try fallback for transient/recoverable errors
                    if resp.status in (503, 429, 403):
                        antigravity_auth.mark_account_error(account, resp.status)
                        last_error_status = resp.status
                        last_error_msg = error_body[:200]
                        logger.info(f"[{request_id}] Falling back to next account due to {resp.status}")
                        continue  # Try next account

                    if resp.status == 401:
                        antigravity_auth.mark_account_error(account, resp.status)
                        last_error_status = resp.status
                        last_error_msg = error_body[:200]
                        continue  # Try next account

                    # For other errors, send error and stop
                    err_chunk = {
                        "id": cmpl_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": {"content": f"[Antigravity Error {resp.status}]"}, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(err_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                # Parse Antigravity SSE -> OpenAI SSE
                buffer = ""
                async for raw_chunk in resp.content.iter_any():
                    buffer += raw_chunk.decode("utf-8", errors="replace")
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if not data_str:
                            continue
                        try:
                            event = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        # Extract text from Antigravity response
                        response = event.get("response", event)
                        candidates = response.get("candidates", [])
                        for candidate in candidates:
                            content = candidate.get("content", {})
                            parts = content.get("parts", [])
                            for part in parts:
                                # Skip thought blocks
                                if part.get("thought") or part.get("thoughtSignature"):
                                    continue
                                text = part.get("text", "")
                                if text:
                                    total_chars += len(text)
                                    oai_chunk = {
                                        "id": cmpl_id,
                                        "object": "chat.completion.chunk",
                                        "created": created,
                                        "model": model,
                                        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]
                                    }
                                    yield f"data: {json.dumps(oai_chunk)}\n\n"

                            # Check for finish reason
                            finish_reason = candidate.get("finishReason")
                            if finish_reason:
                                stop_chunk = {
                                    "id": cmpl_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": model,
                                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop" if finish_reason == "STOP" else finish_reason.lower()}]
                                }
                                yield f"data: {json.dumps(stop_chunk)}\n\n"

                # Final chunk if not already sent
                final_chunk = {
                    "id": cmpl_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"

                antigravity_auth.mark_account_success(account)
                elapsed = time.time() - start
                logger.info(f"[{request_id}] <- {total_chars} chars ({elapsed:.1f}s, Antigravity stream, account: {account.email})")
                return  # Success, exit the generator
        except Exception as e:
            logger.error(f"[{request_id}] Antigravity stream error for {account.email}: {e}")
            antigravity_auth.mark_account_error(account, 500)
            last_error_status = 500
            last_error_msg = str(e)
        finally:
            if owns_session:
                await session.close()

    # All accounts failed
    error_msg = f"[Antigravity Error: All accounts failed - {last_error_status}]"
    if last_error_msg:
        error_msg = f"[Antigravity Error {last_error_status}: {last_error_msg[:100]}]"
    err_chunk = {
        "id": cmpl_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"content": error_msg}, "finish_reason": None}]
    }
    yield f"data: {json.dumps(err_chunk)}\n\n"
    yield "data: [DONE]\n\n"


# --- OpenAI-compatible API ---

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: list[ChatMessage]
    max_tokens: Optional[int] = 4096
    temperature: Optional[float] = None
    stream: Optional[bool] = False

    class Config:
        extra = "allow"


@asynccontextmanager
async def lifespan(app):
    # Start interceptors - failures should not block other providers
    claude_runner = None
    codex_runner = None

    try:
        claude_runner = await start_interceptor()
    except Exception as e:
        logger.warning(f"Claude auth capture failed: {e}. Claude provider unavailable.")

    try:
        codex_runner = await start_codex_interceptor()
    except Exception as e:
        logger.warning(f"Codex auth capture failed: {e}. Codex provider unavailable.")

    # Load Antigravity auth from cached file
    await load_antigravity_auth()
    yield
    if auth.session:
        await auth.session.close()
    if codex_auth.session:
        await codex_auth.session.close()
    # Close all Antigravity account sessions
    for account in antigravity_auth.accounts:
        if account.session:
            await account.session.close()
    if antigravity_auth._legacy_account.session:
        await antigravity_auth._legacy_account.session.close()
    if claude_runner:
        await claude_runner.cleanup()
    if codex_runner:
        await codex_runner.cleanup()

app = FastAPI(title="Claude SkyrimNet Proxy", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    # Parse model list and pick via round-robin
    model_field = req.model or DEFAULT_MODEL
    models = parse_model_list(model_field)
    model = pick_model_round_robin(models) if len(models) > 1 else models[0]
    use_ollama = is_ollama_model(model)
    use_openrouter = not use_ollama and is_openrouter_model(model)
    use_codex = not use_ollama and is_codex_model(model)
    use_antigravity = not use_ollama and is_antigravity_model(model)

    if use_ollama:
        pass  # No auth check — local needs none, cloud key checked inside call function
    elif use_openrouter:
        pass  # No auth check needed for OpenRouter
    elif use_codex:
        if not codex_auth.is_ready:
            raise HTTPException(status_code=503, detail="Codex auth not ready -- ensure you have run 'codex login'")
    elif use_antigravity:
        if not antigravity_auth.is_ready:
            raise HTTPException(status_code=503, detail="Antigravity auth not ready -- visit /config/antigravity-login")
    else:
        if not auth.is_ready:
            raise HTTPException(status_code=503, detail="Claude auth not ready -- warming up")

    system_prompt = None
    anthropic_messages = []
    for msg in req.messages:
        if msg.role == "system":
            system_prompt = msg.content
        elif msg.role in ("user", "assistant"):
            anthropic_messages.append({"role": msg.role, "content": msg.content})

    if not anthropic_messages:
        raise HTTPException(status_code=400, detail="No user message provided")

    if anthropic_messages[0]["role"] != "user":
        anthropic_messages.insert(0, {"role": "user", "content": "Continue."})

    # Merge consecutive same-role messages
    merged = []
    for msg in anthropic_messages:
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1]["content"] += "\n\n" + msg["content"]
        else:
            merged.append(msg)

    max_tokens = req.max_tokens or 4096

    extra_params = {k: v for k, v in (req.model_extra or {}).items() if v is not None}

    # Route to correct provider
    if use_ollama:
        if req.stream:
            return StreamingResponse(
                call_ollama_streaming(system_prompt, merged, model, max_tokens, **extra_params),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        response = await call_ollama_direct(system_prompt, merged, model, max_tokens, **extra_params)
    elif use_openrouter:
        if req.stream:
            return StreamingResponse(
                call_openrouter_streaming(system_prompt, merged, model, max_tokens, **extra_params),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        response = await call_openrouter_direct(system_prompt, merged, model, max_tokens, **extra_params)
    elif use_codex:
        if req.stream:
            return StreamingResponse(
                call_codex_streaming(system_prompt, merged, model, max_tokens),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        response = await call_codex_direct(system_prompt, merged, model, max_tokens)
    elif use_antigravity:
        if req.stream:
            return StreamingResponse(
                call_antigravity_streaming(system_prompt, merged, model, max_tokens),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        response = await call_antigravity_direct(system_prompt, merged, model, max_tokens)
    else:
        if req.stream:
            return StreamingResponse(
                call_api_streaming(system_prompt, merged, model, max_tokens),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        response = await call_api_direct(system_prompt, merged, model, max_tokens)

    if not response:
        raise HTTPException(status_code=500, detail="Empty response")

    prompt_text = (system_prompt or "") + " ".join(m["content"] for m in merged)
    prompt_tokens = len(prompt_text) // 4
    completion_tokens = len(response) // 4

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response, "name": None},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "system_fingerprint": None,
    }


@app.post("/config/openrouter-key")
async def set_openrouter_key(request: Request):
    global openrouter_api_key
    data = await request.json()
    key = data.get("key", "").strip()
    cfg = _load_config()
    if not key:
        openrouter_api_key = None
        cfg.pop("openrouter_api_key", None)
        _save_config(cfg)
        logger.info("OpenRouter API key cleared")
        return {"status": "cleared"}
    openrouter_api_key = key
    cfg["openrouter_api_key"] = key
    _save_config(cfg)
    logger.info("OpenRouter API key configured and saved to config.json")
    return {"status": "saved"}


@app.post("/config/ollama-key")
async def set_ollama_key(request: Request):
    global ollama_api_key
    data = await request.json()
    key = data.get("key", "").strip()
    cfg = _load_config()
    if not key:
        ollama_api_key = None
        cfg.pop("ollama_api_key", None)
        _save_config(cfg)
        logger.info("Ollama API key cleared — using local endpoint (localhost:11434)")
        return {"status": "cleared"}
    ollama_api_key = key
    cfg["ollama_api_key"] = key
    _save_config(cfg)
    logger.info("Ollama Cloud API key configured and saved to config.json")
    return {"status": "saved"}


# --- Antigravity OAuth Login ---

# Store PKCE verifiers temporarily during OAuth flow
_oauth_states: dict[str, dict] = {}
_oauth_callback_server = None


async def start_oauth_callback_server():
    """Start a temporary server on port 51121 to handle OAuth callbacks."""
    global _oauth_callback_server

    async def callback_handler(request):
        global antigravity_auth

        code = request.query.get("code", "")
        state = request.query.get("state", "")
        error = request.query.get("error", "")

        if error:
            html = f"""<!DOCTYPE html>
<html><head><title>OAuth Error</title><style>
body {{ background:#0f172a; color:#e2e8f0; font-family:system-ui,sans-serif; max-width:600px; margin:60px auto; padding:20px; text-align:center }}
h1 {{ color:#f87171 }}</style></head>
<body><h1>❌ Authentication Failed</h1><p>Error: {error}</p>
<a href="http://localhost:8539/config/antigravity-login">Try again</a></body></html>"""
            return web.Response(text=html, content_type="text/html")

        if not code or not state:
            return web.Response(text="<h1>Missing code or state</h1>", status_code=400, content_type="text/html")

        stored = _oauth_states.pop(state, None)
        if not stored:
            return web.Response(text="<h1>Invalid state</h1>", status_code=400, content_type="text/html")

        verifier = stored["verifier"]
        project_id = stored.get("project_id", "")

        try:
            # Exchange code for tokens
            async with aiohttp.ClientSession() as session:
                token_payload = {
                    "client_id": ANTIGRAVITY_CLIENT_ID,
                    "client_secret": ANTIGRAVITY_CLIENT_SECRET,
                    "code": code,
                    "grant_type": "authorization_code",
                    "redirect_uri": ANTIGRAVITY_REDIRECT_URI,
                    "code_verifier": verifier,
                }
                async with session.post(
                    "https://oauth2.googleapis.com/token",
                    data=token_payload,
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        html = f"<h1>Token exchange failed: {error_text}</h1>"
                        return web.Response(text=html, status_code=400, content_type="text/html")

                    token_data = await resp.json()

                # Get user info
                async with session.get(
                    "https://www.googleapis.com/oauth2/v1/userinfo?alt=json",
                    headers={"Authorization": f"Bearer {token_data['access_token']}"},
                ) as resp:
                    user_info = await resp.json() if resp.status == 200 else {}

                # Get project ID if not provided
                if not project_id:
                    headers = get_antigravity_headers()
                    headers["Authorization"] = f"Bearer {token_data['access_token']}"
                    async with session.post(
                        f"{ANTIGRAVITY_ENDPOINT_PROD}/v1internal:loadCodeAssist",
                        headers=headers,
                        json={"metadata": {"ideType": "ANTIGRAVITY", "platform": "WINDOWS" if sys.platform == "win32" else "MACOS", "pluginType": "GEMINI"}},
                    ) as resp:
                        if resp.status == 200:
                            load_data = await resp.json()
                            if load_data.get("cloudaicompanionProject"):
                                proj = load_data["cloudaicompanionProject"]
                                project_id = proj.get("id", proj) if isinstance(proj, dict) else proj

            # Create new account and add to pool
            new_account = AntigravityAccount()
            new_account.access_token = token_data.get("access_token")
            new_account.refresh_token = token_data.get("refresh_token")
            new_account.email = user_info.get("email")
            new_account.project_id = project_id or ANTIGRAVITY_DEFAULT_PROJECT_ID
            new_account.expires_at = datetime.now() + timedelta(seconds=token_data.get("expires_in", 3600))
            new_account.session = create_session()

            # Add to the account pool
            antigravity_auth.add_account(new_account)

            # Save to file
            _save_antigravity_auth()

            logger.info(f"Antigravity auth successful for {new_account.email}")

            # Get account count for display
            account_count = len(antigravity_auth.get_all_accounts_info())
            accounts_text = f"({account_count} account{'s' if account_count != 1 else ''} configured)"

            # Return success page with redirect
            html = f"""<!DOCTYPE html>
<html><head><title>Login Success</title>
<style>
body {{ background:#0f172a; color:#e2e8f0; font-family:system-ui,sans-serif; max-width:600px; margin:60px auto; padding:20px; text-align:center }}
h1 {{ color:#4ade80 }}
.info {{ background:#1e293b; padding:16px; border-radius:8px; margin:20px 0; text-align:left }}
.label {{ color:#94a3b8; font-size:0.85rem }}
.value {{ color:#f1f5f9; font-family:monospace }}
.accounts {{ color:#67e8f9; font-size:0.9rem; margin-top:8px }}
</style></head>
<body>
<h1>✅ Authentication Successful</h1>
<div class="info">
<div style="margin-bottom:8px"><span class="label">Email:</span> <span class="value">{new_account.email}</span></div>
<div style="margin-bottom:8px"><span class="label">Project ID:</span> <span class="value">{new_account.project_id}</span></div>
</div>
<div class="accounts">{accounts_text}</div>
<p>Redirecting to dashboard in 3 seconds...</p>
<script>setTimeout(function() {{ window.location.href = 'http://localhost:8539/'; }}, 3000);</script>
</body></html>"""
            return web.Response(text=html, content_type="text/html")

        except Exception as e:
            logger.error(f"Antigravity OAuth error: {e}")
            return web.Response(text=f"<h1>Error: {e}</h1>", status_code=500, content_type="text/html")

    app = web.Application()
    app.router.add_get("/oauth-callback", callback_handler)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", ANTIGRAVITY_OAUTH_PORT)
    await site.start()
    _oauth_callback_server = runner
    logger.info(f"OAuth callback server started on port {ANTIGRAVITY_OAUTH_PORT}")
    return runner


@app.get("/config/antigravity-login")
async def antigravity_login(project_id: str = ""):
    """Initiate Antigravity OAuth login flow."""
    import secrets
    import base64
    import hashlib

    # Start callback server if not running
    global _oauth_callback_server
    if _oauth_callback_server is None:
        try:
            await start_oauth_callback_server()
        except Exception as e:
            logger.error(f"Failed to start OAuth callback server: {e}")
            return HTMLResponse(content=f"<h1>Failed to start OAuth callback server: {e}</h1>", status_code=500)

    # Generate PKCE verifier and challenge
    verifier = secrets.token_urlsafe(96)
    verifier_bytes = verifier.encode('utf-8')
    challenge_bytes = hashlib.sha256(verifier_bytes).digest()
    challenge = base64.urlsafe_b64encode(challenge_bytes).decode('utf-8').rstrip('=')

    # Store state
    state = secrets.token_urlsafe(16)
    _oauth_states[state] = {
        "verifier": verifier,
        "project_id": project_id,
    }

    # Build authorization URL
    auth_url = "https://accounts.google.com/o/oauth2/v2/auth"
    params = {
        "client_id": ANTIGRAVITY_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": ANTIGRAVITY_REDIRECT_URI,
        "scope": " ".join(ANTIGRAVITY_SCOPES),
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
        "access_type": "offline",
        "prompt": "consent",
    }
    url_with_params = auth_url + "?" + "&".join(f"{k}={v}" for k, v in params.items())

    return HTMLResponse(content=f"""<!DOCTYPE html>
<html><head><title>Antigravity Login</title>
<style>
  body {{ background:#0f172a; color:#e2e8f0; font-family:system-ui,sans-serif; max-width:600px; margin:60px auto; padding:20px; text-align:center }}
  h1 {{ color:#f8fafc; font-size:1.5rem }}
  .status {{ color:#4ade80; font-size:0.9rem; margin:10px 0 }}
  .btn {{ display:inline-block; background:#4285f4; color:white; padding:14px 28px; border-radius:6px; text-decoration:none; font-weight:600; margin:20px 0 }}
  .btn:hover {{ background:#3367d6 }}
  .note {{ color:#94a3b8; font-size:0.85rem; margin-top:20px }}
</style></head>
<body>
  <h1>🔐 Antigravity Login</h1>
  <div class="status">✓ OAuth callback server ready on port {ANTIGRAVITY_OAUTH_PORT}</div>
  <p>Click below to sign in with your Google account</p>
  <a href="{url_with_params}" class="btn">Sign in with Google</a>
  <p class="note">After signing in, you'll be redirected back to complete setup.<br>
  Requires a US-associated Google account.</p>
</body></html>
""")


@app.delete("/config/antigravity-account/{email}")
async def remove_antigravity_account(email: str):
    """Remove an Antigravity account by email."""
    # URL decode the email
    from urllib.parse import unquote
    decoded_email = unquote(email)

    if antigravity_auth.remove_account(decoded_email):
        _save_antigravity_auth()
        logger.info(f"Removed Antigravity account: {decoded_email}")
        return {"status": "removed", "email": decoded_email}
    return {"status": "not_found", "email": decoded_email}


@app.get("/v1/models")
async def list_models():
    data = [
        {"id": "claude-opus-4-6", "object": "model", "owned_by": "anthropic"},
        {"id": "claude-sonnet-4-6", "object": "model", "owned_by": "anthropic"},
        {"id": "claude-sonnet-4-5-20250929", "object": "model", "owned_by": "anthropic"},
        {"id": "claude-haiku-4-5-20251001", "object": "model", "owned_by": "anthropic"},
    ]
    if codex_auth.is_ready or CODEX_PATH:
        data.extend([
            {"id": "gpt-5.4", "object": "model", "owned_by": "openai"},
            {"id": "gpt-5.2", "object": "model", "owned_by": "openai"},
            {"id": "gpt-5.2-codex", "object": "model", "owned_by": "openai"},
            {"id": "gpt-5.1-codex-max", "object": "model", "owned_by": "openai"},
            {"id": "gpt-5.1-codex-mini", "object": "model", "owned_by": "openai"},
            {"id": "gpt-5.1", "object": "model", "owned_by": "openai"},
            {"id": "gpt-5.1-codex", "object": "model", "owned_by": "openai"},
            {"id": "gpt-5-codex", "object": "model", "owned_by": "openai"},
            {"id": "gpt-5-codex-mini", "object": "model", "owned_by": "openai"},
            {"id": "gpt-5", "object": "model", "owned_by": "openai"},
        ])
    if antigravity_auth.is_ready:
        data.extend([
            {"id": "antigravity-gemini-3.1-pro", "object": "model", "owned_by": "google"},
            {"id": "antigravity-gemini-3-pro", "object": "model", "owned_by": "google"},
            {"id": "antigravity-gemini-3-flash", "object": "model", "owned_by": "google"},
            {"id": "antigravity-gemini-2.5-pro", "object": "model", "owned_by": "google"},
            {"id": "antigravity-gemini-2.5-flash", "object": "model", "owned_by": "google"},
            {"id": "antigravity-claude-sonnet-4-6", "object": "model", "owned_by": "anthropic"},
            {"id": "antigravity-claude-opus-4-6-thinking", "object": "model", "owned_by": "anthropic"},
            {"id": "antigravity-gpt-oss-120b-medium", "object": "model", "owned_by": "other"},
        ])
    # Probe local Ollama with a short timeout — skip silently if unreachable
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=2, connect=1)
        ) as s:
            async with s.get("http://localhost:11434/v1/models") as ollama_resp:
                if ollama_resp.status == 200:
                    ollama_data = await ollama_resp.json()
                    for m in ollama_data.get("data", []):
                        model_id = m.get("id", "")
                        if model_id:
                            data.append({"id": f"ollama:{model_id}", "object": "model", "owned_by": "ollama"})
    except Exception:
        pass  # Ollama not running or unreachable — skip silently

    if openrouter_api_key:
        data.append({"id": "openrouter/*", "object": "model", "owned_by": "openrouter"})
    return {"object": "list", "data": data}


@app.get("/health")
async def health():
    accounts_info = antigravity_auth.get_all_accounts_info()
    return {
        "status": "healthy" if (auth.is_ready or codex_auth.is_ready or antigravity_auth.is_ready) else "warming_up",
        "claude": {
            "path": CLAUDE_PATH,
            "auth_cached": auth.is_ready,
        },
        "codex": {
            "path": CODEX_PATH,
            "auth_cached": codex_auth.is_ready,
            "token_expired": codex_auth.is_expired() if codex_auth.is_ready else None,
        },
        "antigravity": {
            "auth_cached": antigravity_auth.is_ready,
            "email": antigravity_auth.email,
            "project_id": antigravity_auth.project_id,
            "token_expired": antigravity_auth.is_expired() if antigravity_auth.is_ready else None,
            "accounts": accounts_info,
            "account_count": len(accounts_info),
        },
        "openrouter_configured": openrouter_api_key is not None,
        "ollama_configured": ollama_api_key is not None,
    }


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    status = "Ready" if (auth.is_ready or codex_auth.is_ready or antigravity_auth.is_ready) else "Warming up..."
    status_color = "#4ade80" if (auth.is_ready or codex_auth.is_ready or antigravity_auth.is_ready) else "#facc15"
    template_size = len(json.dumps(auth.body_template)) if auth.body_template else 0

    or_status = "Configured (saved)" if openrouter_api_key else "Not set"
    or_color = "#4ade80" if openrouter_api_key else "#64748b"

    codex_status = "Ready" if codex_auth.is_ready else ("Not authenticated" if CODEX_PATH else "Not installed")
    codex_color = "#4ade80" if codex_auth.is_ready else ("#facc15" if CODEX_PATH else "#64748b")

    claude_status = "Ready" if auth.is_ready else ("Not authenticated" if CLAUDE_PATH else "Not installed")
    claude_color = "#4ade80" if auth.is_ready else ("#facc15" if CLAUDE_PATH else "#64748b")

    ag_status = "Ready" if antigravity_auth.is_ready else "Not authenticated"
    ag_color = "#4ade80" if antigravity_auth.is_ready else "#facc15"
    ag_accounts = antigravity_auth.get_all_accounts_info()
    ag_account_count = len(ag_accounts)
    # Build account list HTML
    if ag_accounts:
        ag_accounts_html = "<div style='margin-top:8px;font-size:0.85rem'>"
        for acc in ag_accounts:
            status_icon = "✓" if acc["is_ready"] else "⚠"
            status_color_acc = "#4ade80" if acc["is_ready"] else "#facc15"
            error_badge = f" <span style='color:#f87171;font-size:0.75rem'>({acc['error_count']} errors)</span>" if acc.get("error_count", 0) > 0 else ""
            remove_link = f"<a href='#' onclick='removeAccount(\"{acc['email']}\");return false' style='color:#f87171;margin-left:8px;font-size:0.75rem'>[remove]</a>"
            ag_accounts_html += f"<div style='margin:4px 0'><span style='color:{status_color_acc}'>{status_icon}</span> {acc['email']}{error_badge}{remove_link}</div>"
        ag_accounts_html += "</div>"
    else:
        ag_accounts_html = ""

    # Claude models
    claude_models = [
        ("claude-opus-4-6", "Opus 4.6", "Most capable"),
        ("claude-sonnet-4-6", "Sonnet 4.6", "Latest"),
        ("claude-sonnet-4-5-20250929", "Sonnet 4.5", "Default"),
        ("claude-haiku-4-5-20251001", "Haiku 4.5", "Fastest"),
    ]
    claude_rows = "".join(
        f'<tr><td style="font-family:monospace;color:#93c5fd">{mid}</td><td>{name}</td><td style="color:#9ca3af">{desc}</td></tr>'
        for mid, name, desc in claude_models
    )

    # Codex models
    codex_models = [
        ("gpt-5.4", "GPT-5.4", "Flagship frontier"),
        ("gpt-5.2", "GPT-5.2", "General purpose"),
        ("gpt-5.2-codex", "GPT-5.2 Codex", "Code-optimized"),
        ("gpt-5.1-codex-max", "GPT-5.1 Max", "Long context"),
        ("gpt-5.1-codex-mini", "GPT-5.1 Mini", "Fast"),
        ("gpt-5.1", "GPT-5.1", "General purpose"),
        ("gpt-5.1-codex", "GPT-5.1 Codex", "Code-optimized"),
        ("gpt-5-codex", "GPT-5 Codex", "Code-optimized"),
        ("gpt-5-codex-mini", "GPT-5 Codex Mini", "Fast"),
        ("gpt-5", "GPT-5", "Reasoning"),
    ]
    codex_rows = "".join(
        f'<tr><td style="font-family:monospace;color:#93c5fd">{mid}</td><td>{name}</td><td style="color:#9ca3af">{desc}</td></tr>'
        for mid, name, desc in codex_models
    )

    # Antigravity models
    antigravity_models = [
        ("antigravity-gemini-3.1-pro", "Gemini 3.1 Pro", "Latest"),
        ("antigravity-gemini-3-pro", "Gemini 3 Pro", "Flagship"),
        ("antigravity-gemini-3-flash", "Gemini 3 Flash", "Fast"),
        ("antigravity-gemini-2.5-pro", "Gemini 2.5 Pro", "Stable"),
        ("antigravity-gemini-2.5-flash", "Gemini 2.5 Flash", "Fast"),
        ("antigravity-claude-sonnet-4-6", "Claude Sonnet 4.6", "Code"),
        ("antigravity-claude-opus-4-6-thinking", "Claude Opus 4.6", "Thinking"),
        ("antigravity-gpt-oss-120b-medium", "GPT-OSS 120B", "Open"),
    ]
    antigravity_rows = "".join(
        f'<tr><td style="font-family:monospace;color:#93c5fd">{mid}</td><td>{name}</td><td style="color:#9ca3af">{desc}</td></tr>'
        for mid, name, desc in antigravity_models
    )

    return f"""<!DOCTYPE html>
<html><head><title>Claude SkyrimNet Proxy</title>
<style>
  body {{ background:#0f172a; color:#e2e8f0; font-family:system-ui,sans-serif; max-width:800px; margin:40px auto; padding:0 20px }}
  h1 {{ color:#f8fafc; font-size:1.5rem; margin-bottom:4px }}
  .subtitle {{ color:#64748b; font-size:0.9rem; margin-bottom:30px }}
  .status {{ display:inline-block; padding:4px 12px; border-radius:12px; font-size:0.85rem; font-weight:600;
             background:{status_color}20; color:{status_color}; border:1px solid {status_color}40 }}
  .card {{ background:#1e293b; border-radius:8px; padding:20px; margin:16px 0; border:1px solid #334155 }}
  .provider-card {{ background:#1e293b; border-radius:8px; padding:16px; border:1px solid #334155 }}
  .provider-ready {{ border-color:#4ade8040 }}
  .provider-notready {{ border-color:#facc1540 }}
  table {{ width:100%; border-collapse:collapse }}
  th {{ text-align:left; color:#94a3b8; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.05em; padding:8px 12px; border-bottom:1px solid #334155 }}
  td {{ padding:6px 12px; border-bottom:1px solid #1e293b }}
  .label {{ color:#94a3b8; font-size:0.85rem }}
  .value {{ color:#f1f5f9; font-family:monospace; font-size:0.85rem }}
  .endpoint {{ background:#0f172a; padding:10px 14px; border-radius:6px; font-family:monospace; font-size:0.85rem; color:#67e8f9; margin:8px 0; border:1px solid #334155 }}
  textarea {{ width:100%; background:#0f172a; color:#e2e8f0; border:1px solid #334155; border-radius:6px; padding:10px; font-family:monospace; font-size:0.85rem; resize:vertical; box-sizing:border-box }}
  button {{ background:#3b82f6; color:white; border:none; padding:8px 20px; border-radius:6px; cursor:pointer; font-size:0.85rem; margin-top:8px }}
  button:hover {{ background:#2563eb }}
  button:disabled {{ background:#475569; cursor:wait }}
  #response {{ margin-top:12px; padding:12px; background:#0f172a; border-radius:6px; border:1px solid #334155; font-size:0.9rem; min-height:40px; white-space:pre-wrap }}
  .timing {{ color:#4ade80; font-size:0.8rem; margin-top:6px }}
  select {{ background:#0f172a; color:#e2e8f0; border:1px solid #334155; border-radius:6px; padding:8px 12px; font-family:monospace; font-size:0.85rem }}
  .provider-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; margin:16px 0 }}
  .provider-header {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:12px }}
  .provider-status {{ padding:4px 10px; border-radius:12px; font-size:0.75rem; font-weight:600 }}
  .status-ready {{ background:#4ade8020; color:#4ade80 }}
  .status-notready {{ background:#facc1520; color:#facc15 }}
  .status-offline {{ background:#64748b20; color:#64748b }}
</style></head>
<body>
  <h1>Claude SkyrimNet Proxy</h1>
  <div class="subtitle">OpenAI-compatible proxy using Claude Max or Codex CLI subscription</div>
  <span class="status">{status}</span>

  <div class="card" style="border-color:#f87171;background:#450a0a20">
    <h3 style="margin:0 0 8px;font-size:1rem;color:#f87171">⚠️ Terms of Service Warning</h3>
    <p style="margin:0;color:#fca5a5;font-size:0.85rem">
      The Codex provider intercepts OAuth tokens from the official Codex CLI. The Antigravity provider uses
      Google OAuth to access the Antigravity IDE API. These approaches exist in a
      <strong>gray area of Terms of Service</strong>. Use at your own risk.
    </p>
  </div>

  <div class="card">
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px">
      <div><span class="label">Endpoint</span><div class="endpoint">http://127.0.0.1:8539/v1/chat/completions</div></div>
      <div><span class="label">API Key</span><div class="endpoint">not required</div></div>
    </div>
  </div>

  <!-- Provider Slots -->
  <div class="provider-grid">
    <!-- Claude Provider -->
    <div class="provider-card {'provider-ready' if auth.is_ready else 'provider-notready'}">
      <div class="provider-header">
        <h3 style="margin:0;font-size:1rem;color:#f1f5f9">🟠 Claude (Anthropic)</h3>
        <span class="provider-status {'status-ready' if auth.is_ready else ('status-notready' if CLAUDE_PATH else 'status-offline')}">{claude_status}</span>
      </div>
      <div style="margin-bottom:8px"><span class="label">Auth Method:</span> <span class="value">MITM Interceptor</span></div>
      <table style="font-size:0.8rem">
        <thead><tr><th>Model ID</th><th>Name</th><th>Notes</th></tr></thead>
        <tbody>{claude_rows}</tbody>
      </table>
    </div>

    <!-- Codex Provider -->
    <div class="provider-card {'provider-ready' if codex_auth.is_ready else 'provider-notready'}">
      <div class="provider-header">
        <h3 style="margin:0;font-size:1rem;color:#f1f5f9">🟢 Codex (OpenAI)</h3>
        <span class="provider-status {'status-ready' if codex_auth.is_ready else ('status-notready' if CODEX_PATH else 'status-offline')}">{codex_status}</span>
      </div>
      <div style="margin-bottom:8px"><span class="label">Auth Method:</span> <span class="value">Isolated HOME</span></div>
      <table style="font-size:0.8rem">
        <thead><tr><th>Model ID</th><th>Name</th><th>Notes</th></tr></thead>
        <tbody>{codex_rows}</tbody>
      </table>
    </div>

    <!-- Antigravity Provider -->
    <div class="provider-card {'provider-ready' if antigravity_auth.is_ready else 'provider-notready'}">
      <div class="provider-header">
        <h3 style="margin:0;font-size:1rem;color:#f1f5f9">🔵 Antigravity (Google)</h3>
        <span class="provider-status {'status-ready' if antigravity_auth.is_ready else 'status-notready'}">{ag_status}</span>
      </div>
      <div style="margin-bottom:8px">
        <span class="label">Auth Method:</span> <span class="value">Google OAuth</span>
        <span style="margin-left:12px" class="label">Accounts:</span> <span class="value">{ag_account_count}</span>
      </div>
      {ag_accounts_html}
      <div style="margin:8px 0">
        <a href="/config/antigravity-login" style="color:#67e8f9;font-size:0.85rem" target="_blank">+ Add Google Account →</a>
      </div>
      <table style="font-size:0.8rem">
        <thead><tr><th>Model ID</th><th>Name</th><th>Notes</th></tr></thead>
        <tbody>{antigravity_rows}</tbody>
      </table>
    </div>
  </div>

  <!-- OpenRouter -->
  <div class="card">
    <h3 style="margin:0 0 8px; font-size:1rem; color:#f1f5f9">🔵 OpenRouter</h3>
    <div style="display:flex; gap:8px; align-items:center">
      <input type="password" id="orKey" placeholder="API key (sk-or-...)"
             style="flex:1; background:#0f172a; color:#e2e8f0; border:1px solid #334155;
                    border-radius:6px; padding:8px 12px; font-family:monospace; font-size:0.85rem">
      <button onclick="saveOrKey()" style="margin-top:0">Save</button>
      <span id="orStatus" style="color:{or_color}; font-size:0.85rem; font-weight:600">{or_status}</span>
    </div>
    <div style="color:#64748b; font-size:0.8rem; margin-top:8px">
      Use <code style="color:#67e8f9">provider/model</code> IDs (e.g. <code style="color:#67e8f9">openai/gpt-4o</code>)
    </div>
  </div>

  <!-- Quick Test -->
  <div class="card">
    <h3 style="margin:0 0 8px; font-size:1rem; color:#f1f5f9">Quick Test</h3>
    <div style="display:flex; gap:8px; align-items:center; margin-bottom:8px">
      <span class="label">Model:</span>
      <select id="modelSelect">
        <optgroup label="Claude Models">
          <option value="claude-sonnet-4-5-20250929">claude-sonnet-4-5-20250929 (default)</option>
          <option value="claude-opus-4-6">claude-opus-4-6</option>
          <option value="claude-sonnet-4-6">claude-sonnet-4-6</option>
          <option value="claude-haiku-4-5-20251001">claude-haiku-4-5-20251001</option>
        </optgroup>
        <optgroup label="Codex Models">
          <option value="gpt-5.4">gpt-5.4</option>
          <option value="gpt-5.2">gpt-5.2</option>
          <option value="gpt-5.2-codex">gpt-5.2-codex</option>
          <option value="gpt-5.1-codex-max">gpt-5.1-codex-max</option>
          <option value="gpt-5.1-codex-mini">gpt-5.1-codex-mini</option>
          <option value="gpt-5.1">gpt-5.1</option>
          <option value="gpt-5.1-codex">gpt-5.1-codex</option>
          <option value="gpt-5-codex">gpt-5-codex</option>
          <option value="gpt-5-codex-mini">gpt-5-codex-mini</option>
          <option value="gpt-5">gpt-5</option>
        </optgroup>
        <optgroup label="Antigravity Models">
          <option value="antigravity-gemini-3.1-pro">antigravity-gemini-3.1-pro</option>
          <option value="antigravity-gemini-3-pro">antigravity-gemini-3-pro</option>
          <option value="antigravity-gemini-3-flash">antigravity-gemini-3-flash</option>
          <option value="antigravity-gemini-2.5-pro">antigravity-gemini-2.5-pro</option>
          <option value="antigravity-gemini-2.5-flash">antigravity-gemini-2.5-flash</option>
          <option value="antigravity-claude-sonnet-4-6">antigravity-claude-sonnet-4-6</option>
          <option value="antigravity-claude-opus-4-6-thinking">antigravity-claude-opus-4-6-thinking</option>
        </optgroup>
      </select>
    </div>
    <textarea id="sysPrompt" rows="2" placeholder="System prompt">You are Lydia, a Nord housecarl sworn to protect the Dragonborn. Stay in character. One sentence only.</textarea>
    <textarea id="userMsg" rows="1" placeholder="User message" style="margin-top:6px">What do you think of dragons?</textarea>
    <button onclick="testChat()" id="testBtn">Send</button>
    <div id="response" style="display:none"></div>
    <div id="timing" class="timing"></div>
  </div>

<script>
async function saveOrKey() {{
  const key = document.getElementById('orKey').value.trim();
  const status = document.getElementById('orStatus');
  try {{
    await fetch('/config/openrouter-key', {{
      method: 'POST', headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{key: key}})
    }});
    status.textContent = key ? 'Saved' : 'Not set';
    status.style.color = key ? '#4ade80' : '#64748b';
    document.getElementById('orKey').value = '';
  }} catch(e) {{
    status.textContent = 'Error';
    status.style.color = '#f87171';
  }}
}}

async function testChat() {{
  const btn = document.getElementById('testBtn');
  const resp = document.getElementById('response');
  const timing = document.getElementById('timing');
  const model = document.getElementById('modelSelect').value;
  btn.disabled = true; btn.textContent = 'Waiting...';
  resp.style.display = 'block'; resp.textContent = '...';
  timing.textContent = '';
  const start = Date.now();
  try {{
    const r = await fetch('/v1/chat/completions', {{
      method: 'POST', headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{
        model: model,
        messages: [
          {{role: 'system', content: document.getElementById('sysPrompt').value}},
          {{role: 'user', content: document.getElementById('userMsg').value}}
        ]
      }})
    }});
    const data = await r.json();
    const elapsed = ((Date.now() - start) / 1000).toFixed(1);
    if (data.choices) {{
      resp.textContent = data.choices[0].message.content;
      timing.textContent = model + ' · ' + elapsed + 's';
    }} else {{
      resp.textContent = JSON.stringify(data, null, 2);
    }}
  }} catch(e) {{
    resp.textContent = 'Error: ' + e.message;
  }}
  btn.disabled = false; btn.textContent = 'Send';
}}

async function removeAccount(email) {{
  if (!confirm('Remove account ' + email + '?')) return;
  try {{
    const resp = await fetch('/config/antigravity-account/' + encodeURIComponent(email), {{
      method: 'DELETE'
    }});
    const data = await resp.json();
    if (data.status === 'removed') {{
      alert('Account removed: ' + email);
      location.reload();
    }} else {{
      alert('Account not found: ' + email);
    }}
  }} catch(e) {{
    alert('Error: ' + e.message);
  }}
}}
</script>
</body></html>"""


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8539)
