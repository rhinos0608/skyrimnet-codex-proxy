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
"""

import asyncio
import json
import time
import uuid
import logging
import shutil
import os
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

# --- OpenRouter + Round-Robin state ---
_cfg = _load_config()
openrouter_api_key: Optional[str] = _cfg.get("openrouter_api_key") or None
if openrouter_api_key:
    logger.info("OpenRouter API key loaded from config.json")
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
        async with aiohttp.ClientSession() as session:
            async with session.post(real_url, data=body, headers=headers) as resp:
                resp_body = await resp.read()
                return web.Response(body=resp_body, status=resp.status,
                    headers={"Content-Type": resp.headers.get("Content-Type", "application/json")})

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

    # Use clean temp dir to minimize system-reminder bloat (no CLAUDE.md, no skills)
    with tempfile.TemporaryDirectory() as tmpdir:
        env = os.environ.copy()
        env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{INTERCEPTOR_PORT}"

        logger.info("Warming up: capturing auth from claude --print...")
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
        await asyncio.wait_for(proc.communicate(), timeout=60)

    if auth.is_ready:
        # Create persistent session for all future API calls
        auth.session = aiohttp.ClientSession()
        logger.info("Auth captured — direct API mode active (persistent session)")
    else:
        logger.error("Failed to capture auth headers!")

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
        codex_auth.session = aiohttp.ClientSession()
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

    session = auth.session or aiohttp.ClientSession()
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

    session = auth.session or aiohttp.ClientSession()
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

        proc = await asyncio.create_subprocess_exec(
            CODEX_PATH,
            "exec",
            "--model", model,
            "--json",
            "--dangerously-bypass-approvals-and-sandbox",
            "--ephemeral",
            full_prompt,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=isolated_home,  # Run from isolated directory
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=180)
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

        proc = await asyncio.create_subprocess_exec(
            CODEX_PATH,
            "exec",
            "--model", model,
            "--json",
            "--dangerously-bypass-approvals-and-sandbox",
            "--ephemeral",
            full_prompt,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=isolated_home,
        )

        # Collect all output first (Codex CLI buffers output)
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=180)
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

    session = auth.session or aiohttp.ClientSession()
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

    session = auth.session or aiohttp.ClientSession()
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
    # Start both interceptors concurrently
    claude_runner = await start_interceptor()
    codex_runner = await start_codex_interceptor()
    yield
    if auth.session:
        await auth.session.close()
    if codex_auth.session:
        await codex_auth.session.close()
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
    use_openrouter = is_openrouter_model(model)
    use_codex = is_codex_model(model)

    if use_openrouter:
        pass  # No auth check needed for OpenRouter
    elif use_codex:
        if not codex_auth.is_ready:
            raise HTTPException(status_code=503, detail="Codex auth not ready -- ensure you have run 'codex login'")
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
    if use_openrouter:
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


@app.get("/v1/models")
async def list_models():
    data = [
        {"id": "claude-opus-4-6", "object": "model", "owned_by": "anthropic"},
        {"id": "claude-sonnet-4-5-20250929", "object": "model", "owned_by": "anthropic"},
        {"id": "claude-haiku-4-5-20251001", "object": "model", "owned_by": "anthropic"},
    ]
    if codex_auth.is_ready or CODEX_PATH:
        data.extend([
            {"id": "gpt-5.2", "object": "model", "owned_by": "openai"},
            {"id": "gpt-5.2-codex", "object": "model", "owned_by": "openai"},
            {"id": "gpt-5.1-codex-max", "object": "model", "owned_by": "openai"},
            {"id": "gpt-5.1-codex-mini", "object": "model", "owned_by": "openai"},
        ])
    if openrouter_api_key:
        data.append({"id": "openrouter/*", "object": "model", "owned_by": "openrouter"})
    return {"object": "list", "data": data}


@app.get("/health")
async def health():
    return {
        "status": "healthy" if (auth.is_ready or codex_auth.is_ready) else "warming_up",
        "claude": {
            "path": CLAUDE_PATH,
            "auth_cached": auth.is_ready,
        },
        "codex": {
            "path": CODEX_PATH,
            "auth_cached": codex_auth.is_ready,
            "token_expired": codex_auth.is_expired() if codex_auth.is_ready else None,
        },
        "openrouter_configured": openrouter_api_key is not None,
    }


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    status = "Ready" if auth.is_ready else "Warming up..."
    status_color = "#4ade80" if auth.is_ready else "#facc15"
    template_size = len(json.dumps(auth.body_template)) if auth.body_template else 0

    or_status = "Configured (saved)" if openrouter_api_key else "Not set"
    or_color = "#4ade80" if openrouter_api_key else "#64748b"

    codex_status = "Ready" if codex_auth.is_ready else ("Not authenticated" if CODEX_PATH else "Not installed")
    codex_color = "#4ade80" if codex_auth.is_ready else ("#facc15" if CODEX_PATH else "#64748b")

    claude_status = "Ready" if auth.is_ready else ("Not authenticated" if CLAUDE_PATH else "Not installed")
    claude_color = "#4ade80" if auth.is_ready else ("#facc15" if CLAUDE_PATH else "#64748b")

    # Claude models
    claude_models = [
        ("claude-opus-4-6", "Opus 4.6", "Most capable"),
        ("claude-sonnet-4-5-20250929", "Sonnet 4.5", "Default"),
        ("claude-haiku-4-5-20251001", "Haiku 4.5", "Fastest"),
    ]
    claude_rows = "".join(
        f'<tr><td style="font-family:monospace;color:#93c5fd">{mid}</td><td>{name}</td><td style="color:#9ca3af">{desc}</td></tr>'
        for mid, name, desc in claude_models
    )

    # Codex models
    codex_models = [
        ("gpt-5.2", "GPT-5.2", "General purpose"),
        ("gpt-5.2-codex", "GPT-5.2 Codex", "Code-optimized"),
        ("gpt-5.1-codex-max", "GPT-5.1 Max", "Long context"),
        ("gpt-5.1-codex-mini", "GPT-5.1 Mini", "Fast"),
    ]
    codex_rows = "".join(
        f'<tr><td style="font-family:monospace;color:#93c5fd">{mid}</td><td>{name}</td><td style="color:#9ca3af">{desc}</td></tr>'
        for mid, name, desc in codex_models
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
      The Codex provider intercepts OAuth tokens from the official Codex CLI. This approach exists in a
      <strong>gray area of OpenAI's Terms of Service</strong>. Use at your own risk.
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
          <option value="claude-haiku-4-5-20251001">claude-haiku-4-5-20251001</option>
        </optgroup>
        <optgroup label="Codex Models">
          <option value="gpt-5.2">gpt-5.2</option>
          <option value="gpt-5.2-codex">gpt-5.2-codex</option>
          <option value="gpt-5.1-codex-max">gpt-5.1-codex-max</option>
          <option value="gpt-5.1-codex-mini">gpt-5.1-codex-mini</option>
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
</script>
</body></html>"""


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8539)
