"""MITM interceptors for Claude and Codex CLI auth capture.

Each helper uses lazy ``import proxy`` at call time to read/mutate the
module-level auth caches, cached auth blocks, and port state.  Direct
``global X`` declarations are deliberately NOT used because Python's
``global`` keyword only references the *current module's* namespace —
after moving to proxy_internal.interceptors, a bare ``global
_cached_billing_block`` would mutate this file's namespace, not
proxy.py's.  Every write MUST go through ``proxy.<name> = value``.
"""

import asyncio
import json
import logging
import os
import socket
import tempfile
from datetime import datetime, timedelta
from typing import Optional

import aiohttp
from aiohttp import web

logger = logging.getLogger("proxy")


async def interceptor_handler(request):
    import proxy
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

    # Skip non-messages requests (e.g. HEAD / health checks from Claude CLI)
    if "/v1/messages" not in request.path:
        real_url = f"https://api.anthropic.com{request.path_qs}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(request.method, real_url, data=body, headers=headers) as resp:
                    resp_body = await resp.read()
                    return web.Response(body=resp_body, status=resp.status,
                        headers={"Content-Type": resp.headers.get("Content-Type", "application/json")})
        except Exception:
            return web.Response(status=200)

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
    if not proxy.auth.is_ready:
        proxy.auth.headers = dict(headers)
        # Strip tool definitions, system prompts, and other control-plane baggage
        # from the cached template. The warm-up request itself is still forwarded
        # unchanged so auth capture behaves exactly like the CLI invocation.
        sanitized = proxy._sanitize_claude_template(parsed)
        proxy.auth.body_template = sanitized
        # Cache the two fields reused on every inference call so _build_api_body
        # can skip copy.deepcopy(auth.body_template) entirely.
        proxy._cached_billing_block = parsed["system"][0] if parsed.get("system") else None
        first_msg = parsed["messages"][0] if parsed.get("messages") else {}
        proxy._cached_auth_blocks = [
            b for b in (first_msg.get("content") or [])
            if isinstance(b, dict) and b.get("type") == "text"
            and "<system-reminder>" in b.get("text", "")
        ]
        template_size = len(json.dumps(sanitized))
        logger.info(f"Captured {len(proxy.auth.headers)} headers + template ({template_size:,} bytes, tools/system stripped)")

    # Forward to real API
    forward_body = body
    if not proxy.auth.is_ready:
        forward_body = json.dumps(sanitized).encode("utf-8")
    async with aiohttp.ClientSession() as session:
        async with session.post(real_url, data=forward_body, headers=headers) as resp:
            resp_body = await resp.read()
            return web.Response(body=resp_body, status=resp.status,
                headers={"Content-Type": resp.headers.get("Content-Type", "text/event-stream")})


def _kill_stale_port(port: int) -> bool:
    """Try to kill a stale process occupying a port. Returns True if a process was killed."""
    import subprocess as _sp
    try:
        result = _sp.run(
            ["lsof", "-ti", f"tcp:{port}"], capture_output=True, text=True, timeout=5
        )
        pids = result.stdout.strip().split()
        if not pids:
            return False
        my_pid = str(os.getpid())
        for pid in pids:
            if pid and pid != my_pid:
                logger.warning(f"Killing stale process {pid} on port {port}")
                os.kill(int(pid), 9)
        # Give OS a moment to release the socket
        import time
        time.sleep(0.3)
        return True
    except Exception as e:
        logger.debug(f"Could not kill stale process on port {port}: {e}")
        return False


async def start_interceptor(port: int = None):
    """Start MITM interceptor and capture auth from a clean temp dir.

    Args:
        port: Port to bind the interceptor to. Defaults to INTERCEPTOR_PORT (9999).
              MCP mode uses MCP_INTERCEPTOR_PORT (9997) to avoid conflicts.
    """
    import proxy
    if port is None:
        port = proxy.INTERCEPTOR_PORT
    proxy._interceptor_port = port

    iapp = web.Application()
    iapp.router.add_route("*", "/{path_info:.*}", interceptor_handler)

    runner = web.AppRunner(iapp)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    try:
        await site.start()
    except OSError as e:
        if e.errno == 48:  # Address already in use
            logger.warning(f"Port {port} in use, killing stale process...")
            _kill_stale_port(port)
            # Rebuild runner — the old one's socket is in a bad state
            await runner.cleanup()
            runner = web.AppRunner(iapp)
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", port)
            await site.start()
        else:
            raise
    logger.info(f"Interceptor on port {port}")

    max_retries = 3
    for attempt in range(max_retries):
        # Use clean temp dir to minimize system-reminder bloat (no CLAUDE.md, no skills)
        # ignore_cleanup_errors=True prevents Windows errors when subprocess still holds dir handle
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            env = os.environ.copy()
            env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{port}"

            logger.info(f"Warming up: capturing auth from claude --print... (attempt {attempt + 1}/{max_retries})")
            try:
                proc = await asyncio.create_subprocess_exec(
                    proxy.CLAUDE_PATH, "--print",
                    "--output-format", "text",
                    "--model", proxy.DEFAULT_MODEL,
                    "--no-session-persistence",
                    "--system-prompt", "Say ok",
                    "ok",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                    cwd=tmpdir,
                )
                await asyncio.wait_for(proc.communicate(), timeout=90)
                if proxy.auth.is_ready:
                    break  # Success, exit retry loop
            except (asyncio.TimeoutError, Exception) as e:
                logger.warning(f"Warmup attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s

    if proxy.auth.is_ready:
        # Create persistent session for all future API calls
        proxy.auth.session = proxy.create_session()
        logger.info("Auth captured — direct API mode active (persistent session)")
    else:
        logger.error("Failed to capture auth headers after %d attempts!", max_retries)

    return runner


async def recapture_claude_auth() -> bool:
    """Re-run claude --print through the existing interceptor to refresh auth.
    Returns True if auth was successfully recaptured."""
    import proxy
    if not proxy.CLAUDE_PATH:
        return False

    async with proxy._claude_refresh_lock:
        # Another coroutine may have refreshed while we waited for the lock
        if proxy.auth.is_ready:
            return True

        logger.info("Recapturing Claude auth...")
        for attempt in range(3):
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
                env = os.environ.copy()
                env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{proxy._interceptor_port}"
                try:
                    proc = await asyncio.create_subprocess_exec(
                        proxy.CLAUDE_PATH, "--print",
                        "--output-format", "text",
                        "--model", proxy.DEFAULT_MODEL,
                        "--no-session-persistence",
                        "--system-prompt", "Say ok",
                        "ok",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        env=env,
                        cwd=tmpdir,
                    )
                    await asyncio.wait_for(proc.communicate(), timeout=90)
                    if proxy.auth.is_ready:
                        # Recreate the persistent session with fresh auth
                        if proxy.auth.session:
                            await proxy.auth.session.close()
                        proxy.auth.session = proxy.create_session()
                        logger.info("Claude auth recaptured successfully")
                        return True
                except (asyncio.TimeoutError, Exception) as e:
                    logger.warning(f"Auth recapture attempt {attempt + 1} failed: {e}")
                    if attempt < 2:
                        await asyncio.sleep(2 ** attempt)

        logger.error("Claude auth recapture failed after 3 attempts")
        return False


async def _claude_auth_refresh_loop():
    """Background task that proactively recaptures Claude auth before it expires.
    Claude CLI tokens typically last ~1 hour; we recapture every 45 minutes."""
    import proxy
    REFRESH_INTERVAL = 45 * 60  # 45 minutes
    while True:
        await asyncio.sleep(REFRESH_INTERVAL)
        if not proxy.CLAUDE_PATH:
            break
        logger.info("Proactive Claude auth refresh triggered")
        # Clear auth to force recapture through the interceptor
        proxy.auth.headers = None
        proxy.auth.body_template = None
        await recapture_claude_auth()


async def codex_interceptor_handler(request):
    """Handle Codex CLI traffic - captures OAuth tokens from auth flow."""
    import proxy
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
        if auth_code and not proxy.codex_auth.access_token:
            # Exchange code for tokens
            try:
                async with aiohttp.ClientSession() as session:
                    token_payload = {
                        "grant_type": "authorization_code",
                        "code": auth_code,
                        "client_id": "app_EMoamEEZ73f0CkXaXp7hrann",
                        "redirect_uri": f"http://127.0.0.1:{proxy.CODEX_INTERCEPTOR_PORT}/auth/callback",
                    }
                    async with session.post(
                        "https://auth.openai.com/oauth/token",
                        json=token_payload,
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            proxy.codex_auth.access_token = data.get("access_token")
                            proxy.codex_auth.refresh_token = data.get("refresh_token")
                            expires_in = data.get("expires_in", 3600)
                            proxy.codex_auth.expires_at = datetime.now() + timedelta(seconds=expires_in)
                            proxy.codex_auth.account_id = data.get("account_id")
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
    if auth_header.startswith("Bearer ") and not proxy.codex_auth.access_token:
        token = auth_header[7:]
        proxy.codex_auth.access_token = token
        # Assume 1 hour expiry if we don't know
        proxy.codex_auth.expires_at = datetime.now() + timedelta(hours=1)
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
    import proxy
    if not proxy.CODEX_PATH:
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
                proxy.codex_auth.access_token = tokens["access_token"]
                proxy.codex_auth.refresh_token = tokens.get("refresh_token")
                proxy.codex_auth.account_id = tokens.get("account_id")

                # Parse expiry from last_refresh + assume 24hr validity
                last_refresh = data.get("last_refresh")
                if last_refresh:
                    try:
                        # Parse ISO format timestamp
                        last_refresh_dt = datetime.fromisoformat(last_refresh.replace("Z", "+00:00"))
                        # Make it naive for comparison
                        last_refresh_dt = last_refresh_dt.replace(tzinfo=None)
                        # Assume 24 hour token validity
                        proxy.codex_auth.expires_at = last_refresh_dt + timedelta(hours=24)
                    except Exception:
                        proxy.codex_auth.expires_at = datetime.now() + timedelta(hours=23)

                logger.info(f"Loaded Codex auth from {auth_file}")
                if proxy.codex_auth.is_expired():
                    logger.warning("Codex token may be expired, will attempt refresh")
        except Exception as e:
            logger.error(f"Failed to read Codex auth file: {e}")

    if proxy.codex_auth.is_ready:
        proxy.codex_auth.session = proxy.create_session()
        logger.info("Codex auth loaded - direct API mode active")
    else:
        logger.warning("Failed to load Codex auth")
        logger.info("Note: Run 'codex login' first, then restart proxy")

    return None  # No interceptor needed - we read from file


def load_opencode_key():
    """Load OpenCode API keys from ~/.local/share/opencode/auth.json if not already configured.

    Zen and Go plans share the same API key, so a single key is sufficient for both.
    If only one key is available (from config.json or auth.json), it is used for both plans.
    """
    import proxy

    if not os.path.exists(proxy.OPENCODE_AUTH_FILE):
        if not proxy.opencode_api_key and not proxy.opencode_go_api_key:
            logger.info("No OpenCode auth file found -- set up OpenCode first or add key via /config/opencode-key")
        # Even without auth file, share whichever key we already have from config.json
        if proxy.opencode_api_key and not proxy.opencode_go_api_key:
            proxy.opencode_go_api_key = proxy.opencode_api_key
            logger.info("OpenCode Go using shared Zen API key from config")
        elif proxy.opencode_go_api_key and not proxy.opencode_api_key:
            proxy.opencode_api_key = proxy.opencode_go_api_key
            logger.info("OpenCode Zen using shared Go API key from config")
        return

    try:
        with open(proxy.OPENCODE_AUTH_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Load Zen key
        if not proxy.opencode_api_key:
            oc_entry = data.get("opencode", {})
            if oc_entry.get("type") == "api" and oc_entry.get("key"):
                proxy.opencode_api_key = oc_entry["key"]
                logger.info("OpenCode Zen API key loaded from auth file")

        # Load Go key
        if not proxy.opencode_go_api_key:
            go_entry = data.get("opencode-go", {})
            if go_entry.get("type") == "api" and go_entry.get("key"):
                proxy.opencode_go_api_key = go_entry["key"]
                logger.info("OpenCode Go API key loaded from auth file")

        # Zen and Go share the same key — fill in whichever is missing
        if proxy.opencode_api_key and not proxy.opencode_go_api_key:
            proxy.opencode_go_api_key = proxy.opencode_api_key
            logger.info("OpenCode Go using shared Zen API key")
        elif proxy.opencode_go_api_key and not proxy.opencode_api_key:
            proxy.opencode_api_key = proxy.opencode_go_api_key
            logger.info("OpenCode Zen using shared Go API key")
    except Exception as e:
        logger.error(f"Failed to load OpenCode auth: {e}")
