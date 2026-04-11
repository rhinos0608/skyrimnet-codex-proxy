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
import base64
import gc
import json
import random
import time
import uuid
import logging
import shutil
import os
import sys
import tempfile
import statistics
from collections import deque
from contextlib import asynccontextmanager
from typing import Optional, Union
from datetime import datetime, timedelta

from cryptography.fernet import Fernet

# Raise gen0 threshold to reduce stop-the-world pause frequency.
# Short-lived request objects die before gen0 reaches 10k, so this
# reduces pauses without accumulating long-lived garbage.
gc.set_threshold(10000, 20, 20)

# orjson (optional) — 3-5x faster serialisation for large request bodies.
# Falls back to stdlib json transparently if not installed.
try:
    import orjson as _json_lib
    def json_dumps(obj: object) -> bytes:
        return _json_lib.dumps(obj)
except ImportError:
    def json_dumps(obj: object) -> bytes:  # type: ignore[misc]
        return json.dumps(obj).encode("utf-8")

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from aiohttp import web
import aiohttp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("proxy")

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_CODEX_MODEL = "gpt-5.3-codex"
CODEX_FAST_REASONING_EFFORT = "low"
INTERCEPTOR_PORT = 9999
MCP_INTERCEPTOR_PORT = 9997  # Different port for MCP mode to avoid conflicts
_interceptor_port: int = INTERCEPTOR_PORT  # Actual port used (set by start_interceptor)

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

GEMINI_PATH = shutil.which("gemini")
if GEMINI_PATH:
    logger.info(f"Using Gemini CLI: {GEMINI_PATH}")
else:
    logger.info("Gemini CLI not found on PATH - Gemini CLI provider uses file-based auth only")

GEMINI_CREDS_FILE = os.path.expanduser("~/.gemini/oauth_creds.json")
GEMINI_PROJECTS_FILE = os.path.expanduser("~/.gemini/projects.json")
GEMINI_CODE_ASSIST_ENDPOINT = "https://cloudcode-pa.googleapis.com"
GEMINI_CODE_ASSIST_API_VERSION = "v1internal"
# OAuth client credentials from Gemini CLI source (used for token refresh)
GEMINI_OAUTH_CLIENT_ID = "681255809395-oo8ft2oprdrnp9e3aqf6av3hmdib135j.apps.googleusercontent.com"
GEMINI_OAUTH_CLIENT_SECRET = "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"

# --- OpenCode (Zen API at opencode.ai/zen/v1, Go API at opencode.ai/zen/go/v1) ---
OPENCODE_AUTH_FILE = os.path.expanduser("~/.local/share/opencode/auth.json")
OPENCODE_ZEN_URL = "https://opencode.ai/zen/v1"
OPENCODE_GO_URL = "https://opencode.ai/zen/go/v1"

# --- NVIDIA NIM (free tier at integrate.api.nvidia.com) ---
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Short name → full NVIDIA NIM API model path for free-tier models.
_NVIDIA_MODEL_ALIASES: dict[str, str] = {
    # Meta Llama
    "llama-3.1-8b": "meta/llama-3.1-8b-instruct",
    "llama-3.1-70b": "meta/llama-3.1-70b-instruct",
    "llama-3.1-405b": "meta/llama-3.1-405b-instruct",
    "llama-3.2-1b": "meta/llama-3.2-1b-instruct",
    "llama-3.2-3b": "meta/llama-3.2-3b-instruct",
    "llama-3.3-70b": "meta/llama-3.3-70b-instruct",
    "llama-4-maverick": "meta/llama-4-maverick-17b-128e-instruct",
    # NVIDIA
    "nemotron-4-340b": "nvidia/nemotron-4-340b-instruct",
    "nv-embed-v1": "nvidia/nv-embed-v1",
    # Moonshot AI (Kimi)
    "kimi-k2": "moonshotai/kimi-k2-instruct",
    "kimi-k2-0905": "moonshotai/kimi-k2-instruct-0905",
    "kimi-k2-thinking": "moonshotai/kimi-k2-thinking",
    # DeepSeek
    "deepseek-v3.1": "deepseek-ai/deepseek-v3.1",
    "deepseek-v3.1-terminus": "deepseek-ai/deepseek-v3.1-terminus",
    "deepseek-v3.2": "deepseek-ai/deepseek-v3.2",
    # Mistral AI
    "mixtral-8x7b": "mistralai/mixtral-8x7b-instruct-v0.1",
    "mistral-small-3.1": "mistralai/mistral-small-3.1-24b-instruct-2503",
    "mistral-medium-3": "mistralai/mistral-medium-3-instruct",
    "mistral-large-3": "mistralai/mistral-large-3-675b-instruct-2512",
    "mistral-nemotron": "mistralai/mistral-nemotron",
    "devstral-2": "mistralai/devstral-2-123b-instruct-2512",
    "magistral-small": "mistralai/magistral-small-2506",
    # Google
    "gemma-2-9b": "google/gemma-2-9b-it",
    "gemma-2-27b": "google/gemma-2-27b-it",
    "gemma-3-27b": "google/gemma-3-27b-it",
    # Microsoft
    "phi-3-mini": "microsoft/phi-3-mini-128k-instruct",
    "phi-3-medium": "microsoft/phi-3-medium-128k-instruct",
    "phi-3.5-mini": "microsoft/phi-3.5-mini-instruct",
    # Qwen
    "qwen3.5-122b": "qwen/qwen3.5-122b-a10b",
    "qwen3-coder-480b": "qwen/qwen3-coder-480b-a35b-instruct",
    # Z.AI
    "glm-4.7": "z-ai/glm4.7",
    # StepFun
    "step-3.5-flash": "stepfun-ai/step-3.5-flash",
    # ByteDance
    "seed-oss-36b": "bytedance/seed-oss-36b-instruct",
}


def _resolve_nvidia_model(model: str) -> str:
    """Strip the 'nvidia:' prefix and expand short aliases to full API paths."""
    raw = model[len("nvidia:"):]
    return _NVIDIA_MODEL_ALIASES.get(raw, raw)


_NVIDIA_UNSUPPORTED_PARAMS = {"reasoning", "top_k", "provider"}


def _nvidia_payload_fixup(payload: dict, extra_params: dict) -> None:
    """Apply NVIDIA NIM-specific payload adjustments.

    - Strips unsupported params (reasoning, top_k, provider).
    """
    payload.update({k: v for k, v in extra_params.items()
                    if v is not None and k not in _NVIDIA_UNSUPPORTED_PARAMS})


async def call_nvidia_direct(system_prompt: Optional[str], messages: list, model: str, max_tokens: int, **extra_params) -> str:
    """Forward request to NVIDIA NIM (OpenAI-compatible), collect full response."""
    if not nvidia_api_key:
        raise HTTPException(status_code=503, detail="NVIDIA NIM API key not configured — add it via /config/nvidia-key")

    api_model = _resolve_nvidia_model(model)
    endpoint = f"{NVIDIA_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {nvidia_api_key}", "Content-Type": "application/json"}

    payload = {"model": api_model, "messages": build_oai_messages(system_prompt, messages), "max_tokens": max_tokens}
    _nvidia_payload_fixup(payload, extra_params)

    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] -> NVIDIA NIM ({api_model}, {len(messages)} msgs)")
    start = time.time()

    session = third_party_session or auth.session or create_session()
    owns_session = session is not third_party_session and session is not auth.session
    try:
        async with session.post(endpoint, json=payload, headers=headers) as resp:
            elapsed = time.time() - start
            if resp.status in (401, 403):
                raise HTTPException(status_code=401, detail="NVIDIA NIM auth failed — check API key")
            if resp.status == 429:
                raise HTTPException(status_code=429, detail="NVIDIA NIM rate limit exceeded")
            if resp.status != 200:
                error_body = await resp.text()
                logger.error(f"[{request_id}] NVIDIA NIM {resp.status}: {error_body[:300]}")
                raise HTTPException(status_code=resp.status, detail=error_body[:200])
            data = await resp.json()
            choices = data.get("choices", [])
            if not choices:
                logger.warning(f"[{request_id}] NVIDIA NIM returned empty choices array")
                raise HTTPException(status_code=500, detail="NVIDIA NIM returned empty response")
            content = _extract_oai_content(choices[0].get("message", {}))
            if content is None:
                logger.warning(f"[{request_id}] NVIDIA NIM returned null content, full response: {str(data)[:500]}")
                raise HTTPException(status_code=500, detail="NVIDIA NIM returned empty content")
            logger.info(f"[{request_id}] <- {len(content)} chars ({elapsed:.1f}s, NVIDIA NIM)")
            return content
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="NVIDIA NIM request timed out")
    finally:
        if owns_session:
            await session.close()


async def call_nvidia_streaming(system_prompt: Optional[str], messages: list, model: str, max_tokens: int, **extra_params):
    """Forward request to NVIDIA NIM with streaming, passthrough SSE directly."""
    if not nvidia_api_key:
        err, done = yield_sse_error(model, "[NVIDIA NIM Error: API key not configured]")
        yield err; yield done
        return

    api_model = _resolve_nvidia_model(model)
    endpoint = f"{NVIDIA_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {nvidia_api_key}", "Content-Type": "application/json"}

    payload = {"model": api_model, "messages": build_oai_messages(system_prompt, messages), "stream": True}
    _nvidia_payload_fixup(payload, extra_params)

    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] -> NVIDIA NIM ({api_model}, {len(messages)} msgs, stream)")
    start = time.time()

    session = third_party_session or auth.session or create_session()
    owns_session = session is not third_party_session and session is not auth.session
    try:
        async with session.post(endpoint, json=payload, headers=headers) as resp:
            if resp.status != 200:
                error_body = await resp.text()
                logger.error(f"[{request_id}] NVIDIA NIM {resp.status}: {error_body[:300]}")
                err, done = yield_sse_error(model, f"[NVIDIA NIM Error {resp.status}]")
                yield err; yield done
                return

            saw_done = False
            async for event in passthrough_sse(resp, request_id, "NVIDIA NIM", start):
                if event.strip().startswith("data: [DONE]"):
                    saw_done = True
                yield event

            if not saw_done:
                yield "data: [DONE]\n\n"
            return
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
        logger.error(f"[{request_id}] NVIDIA NIM streaming error: {e}")
        err, done = yield_sse_error(model, f"[NVIDIA NIM Error: {e}]")
        yield err; yield done
    finally:
        if owns_session:
            await session.close()


# --- Qwen Code CLI (OAuth via portal.qwen.ai) ---
QWEN_CREDS_FILE = os.path.expanduser("~/.qwen/oauth_creds.json")
QWEN_BASE_URL = "https://portal.qwen.ai/v1"
QWEN_OAUTH_TOKEN_URL = "https://chat.qwen.ai/api/v1/oauth2/token"
QWEN_OAUTH_CLIENT_ID = "f0304373b74a44d2b584a3fb70ca9e56"


def create_session() -> aiohttp.ClientSession:
    """Create an aiohttp session with connection pooling for low-latency reuse."""
    connector = aiohttp.TCPConnector(limit=20, enable_cleanup_closed=True, ttl_dns_cache=300, keepalive_timeout=30)
    timeout = aiohttp.ClientTimeout(total=300, connect=30)
    return aiohttp.ClientSession(
        connector=connector,
        timeout=timeout,
        json_serialize=_json_serialize,
    )


def _json_serialize(obj: object) -> str:
    """Serialize JSON payloads using the fastest available backend."""
    return json_dumps(obj).decode("utf-8")


# --- Shared streaming / message utilities ---

# Fields added by reasoning models that should be stripped from SSE deltas
# so downstream consumers (SkyrimNet) only see plain content.
_REASONING_FIELDS = {"reasoning", "reasoning_content", "reasoning_details"}

# Fields that should never be retained from Claude CLI auth-capture templates.
_CLAUDE_TEMPLATE_STRIPPED_FIELDS = {"tools", "thinking", "context_management", "tool_choice", "system"}


_CHAT_ALLOWED_EXTRA = {
    "temperature",
    "top_p",
    "top_k",
    "frequency_penalty",
    "presence_penalty",
    "stop",
    "seed",
    "logit_bias",
    "n",
    "reasoning",
}


def _sanitize_claude_template(parsed: dict) -> dict:
    """Return a minimal Claude request template without prompt/tool baggage."""
    sanitized = dict(parsed)
    for field in _CLAUDE_TEMPLATE_STRIPPED_FIELDS:
        sanitized.pop(field, None)
    return sanitized


def _extract_oai_content(message: dict) -> Optional[str]:
    """Extract usable text from an OpenAI-format message, handling reasoning models.

    Reasoning models (GLM, Kimi, MiMo) may return content=null/empty while the actual
    output sits in 'reasoning' or 'reasoning_content'.  When content is empty and
    reasoning is present, we return None so the caller can handle it (e.g. retry
    without max_tokens).  We do NOT return reasoning as content because it's the
    model's internal chain-of-thought, not the answer.
    """
    return message.get("content") or None


def _is_reasoning_truncated(data: dict) -> bool:
    """Check if an OpenAI-format response was truncated mid-reasoning.

    Returns True when finish_reason is 'length' and content is empty but
    reasoning_content is present — meaning max_tokens was exhausted by the
    model's chain-of-thought before it could produce actual output.
    """
    choices = data.get("choices", [])
    if not choices:
        return False
    choice = choices[0]
    if choice.get("finish_reason") != "length":
        return False
    msg = choice.get("message", {})
    content = msg.get("content")
    reasoning = msg.get("reasoning_content") or msg.get("reasoning")
    return (not content) and bool(reasoning)


def _has_reasoning_without_content(data: dict) -> bool:
    """Return True when a choice carries reasoning fields but no user-visible content."""
    choices = data.get("choices", [])
    if not choices:
        return False
    msg = choices[0].get("message", {})
    content = msg.get("content")
    reasoning = msg.get("reasoning_content") or msg.get("reasoning")
    return (not content) and bool(reasoning)


def _strip_vision_content(content):
    """Collapse multimodal content blocks to text-only.

    If *content* is a list of dicts (OpenAI vision format), extract only the
    text parts and return a plain string.  Plain strings pass through unchanged.
    """
    if isinstance(content, list):
        parts = [c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"]
        return " ".join(p for p in parts if p) or ""
    return content


def build_oai_messages(system_prompt: Optional[str], messages: list, *, strip_vision: bool = False) -> list:
    """Build an OpenAI-format message list, prepending system prompt if present."""
    oai = []
    if system_prompt:
        oai.append({"role": "system", "content": system_prompt})
    for m in messages:
        content = m["content"]
        if strip_vision:
            content = _strip_vision_content(content)
        oai.append({"role": m["role"], "content": content})
    return oai


def _append_merged_message(messages: list[dict], role: str, content) -> None:
    """Append a message, coalescing adjacent messages from the same role.

    Content can be a string or a list (for vision/multimodal messages).
    """
    # Convert Pydantic models to native Python types
    if hasattr(content, "__iter__") and not isinstance(content, str):
        # List of Pydantic models - convert to dicts
        content = [
            c.model_dump() if hasattr(c, "model_dump") else
            c.dict() if hasattr(c, "dict") else c
            for c in content
        ]

    if messages and messages[-1]["role"] == role:
        prev_content = messages[-1]["content"]
        if isinstance(prev_content, str) and isinstance(content, str):
            messages[-1]["content"] += "\n\n" + content
        elif isinstance(prev_content, list) and isinstance(content, list):
            messages[-1]["content"].extend(content)
        else:
            # Mixed types, can't coalesce - append new message
            messages.append({"role": role, "content": content})
    else:
        messages.append({"role": role, "content": content})


def _normalize_chat_messages(messages: list) -> tuple[Optional[str], list[dict], list[dict]]:
    """Build provider-native and OpenAI-compatible views of the incoming chat history."""
    system_parts: list[str] = []
    merged_messages: list[dict] = []
    oai_messages: list[dict] = []

    for msg in messages:
        role = msg.role if hasattr(msg, "role") else msg["role"]
        content = msg.content if hasattr(msg, "content") else msg["content"]

        # Convert content to native Python types
        if hasattr(content, "__iter__") and not isinstance(content, str):
            content = [
                c.model_dump() if hasattr(c, "model_dump") else
                c.dict() if hasattr(c, "dict") else c
                for c in content
            ]

        if role == "system":
            # System messages are always strings
            if isinstance(content, list):
                content = " ".join(c.get("text", "") for c in content if c.get("text"))
            system_parts.append(content)
            _append_merged_message(oai_messages, role, content)
        elif role in ("user", "assistant"):
            _append_merged_message(merged_messages, role, content)
            _append_merged_message(oai_messages, role, content)

    if merged_messages and merged_messages[0]["role"] != "user":
        merged_messages.insert(0, {"role": "user", "content": "Continue."})

    return "\n\n".join(system_parts) or None, merged_messages, oai_messages


def _model_uses_oai_messages(model: str) -> bool:
    """Return True for providers that accept OpenAI-format chat messages directly."""
    return (
        is_ollama_model(model)
        or is_openrouter_model(model)
        or is_zai_model(model)
        or is_xiaomi_model(model)
        or is_opencode_model(model)
        or is_qwen_model(model)
        or is_fireworks_model(model)
        or is_nvidia_model(model)
    )


def make_sse_error_chunk(model: str, error_msg: str) -> str:
    """Build a single SSE error chunk in OpenAI chat.completion.chunk format."""
    cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    chunk = {
        "id": cmpl_id, "object": "chat.completion.chunk",
        "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "delta": {"content": error_msg}, "finish_reason": None}],
    }
    return f"data: {json.dumps(chunk)}\n\n"


def make_sse_content_chunk(model: str, content: str) -> str:
    """Build a single SSE content chunk in OpenAI chat.completion.chunk format."""
    cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    chunk = {
        "id": cmpl_id, "object": "chat.completion.chunk",
        "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }
    return f"data: {json.dumps(chunk)}\n\n"


def yield_sse_error(model: str, error_msg: str):
    """Return (error_chunk, done_marker) strings for SSE error responses."""
    return make_sse_error_chunk(model, error_msg), "data: [DONE]\n\n"


async def passthrough_sse(resp, request_id: str, provider_name: str, start: float):
    """Passthrough OpenAI-format SSE from an upstream response, yielding events.

    Used by OpenRouter, Ollama, Z.AI, OpenCode, Qwen — all OpenAI-compatible SSE.
    Strips reasoning fields (reasoning, reasoning_content, reasoning_details) from
    SSE deltas so downstream consumers only see plain content.  Reasoning-only
    chunks (content empty/null while reasoning present) are silently dropped.
    """
    buffer = bytearray()
    total_content = 0

    def _emit_event(event_bytes: bytes) -> tuple[Optional[str], int]:
        if not event_bytes:
            return None, 0

        event = event_bytes.decode("utf-8", errors="replace").strip()
        if not event:
            return None, 0
        if not event.startswith("data: ") or event.startswith("data: [DONE]"):
            return event + "\n\n", 0

        # Fast path: most chunks do not carry reasoning fields, so skip JSON
        # parsing entirely unless the event looks like a reasoning chunk.
        if b'"reasoning' not in event_bytes:
            return event + "\n\n", 0

        try:
            data = json.loads(event[6:])
            delta = data.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content")
            modified = False
            for rf in _REASONING_FIELDS:
                if delta.pop(rf, None) is not None:
                    modified = True
            if content:
                return (f"data: {json.dumps(data)}\n\n" if modified else event + "\n\n"), len(content)
            if modified:
                return None, 0
            return event + "\n\n", 0
        except (json.JSONDecodeError, KeyError, IndexError):
            return event + "\n\n", 0

    async for raw_chunk in resp.content.iter_any():
        buffer.extend(raw_chunk)
        while True:
            idx = buffer.find(b"\n\n")
            if idx < 0:
                break
            event_bytes = bytes(buffer[:idx]).strip()
            del buffer[:idx + 2]
            emitted, content_len = _emit_event(event_bytes)
            total_content += content_len
            if emitted is not None:
                yield emitted
    if buffer.strip():
        emitted, content_len = _emit_event(bytes(buffer).strip())
        total_content += content_len
        if emitted is not None:
            yield emitted
    elapsed = time.time() - start
    logger.info(f"[{request_id}] <- stream done ({elapsed:.1f}s, {provider_name}, {total_content} chars)")


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
KEY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".proxy.key")

# --- Token encryption ---

# Fields in config.json that contain secrets and should be encrypted at rest.
_ENCRYPTED_CONFIG_FIELDS = {"openrouter_api_key", "ollama_api_key", "zai_api_key", "xiaomi_api_key", "opencode_api_key", "opencode_go_api_key", "fireworks_api_key", "nvidia_api_key"}
# Fields in antigravity-auth account dicts that should be encrypted.
_ENCRYPTED_AUTH_FIELDS = {"refresh_token", "access_token"}
_ENC_PREFIX = "enc:"


def _get_fernet() -> Fernet:
    """Return a Fernet instance, generating a key file on first use."""
    if not os.path.exists(KEY_FILE):
        key = Fernet.generate_key()
        with open(KEY_FILE, "wb") as f:
            f.write(key)
        # Restrict permissions (best-effort on Windows)
        try:
            os.chmod(KEY_FILE, 0o600)
        except OSError:
            pass
        logger.info("Generated new encryption key (.proxy.key)")
    with open(KEY_FILE, "rb") as f:
        return Fernet(f.read().strip())


def _encrypt_value(fernet: Fernet, value: str) -> str:
    """Encrypt a plaintext string, returning an 'enc:...' token."""
    return _ENC_PREFIX + fernet.encrypt(value.encode("utf-8")).decode("ascii")


def _decrypt_value(fernet: Fernet, value: str) -> str:
    """Decrypt an 'enc:...' token back to plaintext. Returns as-is if not encrypted."""
    if not isinstance(value, str) or not value.startswith(_ENC_PREFIX):
        return value
    return fernet.decrypt(value[len(_ENC_PREFIX):].encode("ascii")).decode("utf-8")


def _load_config() -> dict:
    """Load persisted config from disk, decrypting secret fields."""
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    fernet = _get_fernet()
    migrated = False
    for field in _ENCRYPTED_CONFIG_FIELDS:
        val = data.get(field)
        if isinstance(val, str) and val:
            if val.startswith(_ENC_PREFIX):
                data[field] = _decrypt_value(fernet, val)
            else:
                # Plaintext on disk — encrypt it in place for next load.
                migrated = True
    if migrated:
        _save_config(data)
    return data


def _save_config(data: dict) -> None:
    """Persist config dict to disk, encrypting secret fields."""
    fernet = _get_fernet()
    out = dict(data)
    for field in _ENCRYPTED_CONFIG_FIELDS:
        val = out.get(field)
        if isinstance(val, str) and val and not val.startswith(_ENC_PREFIX):
            out[field] = _encrypt_value(fernet, val)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


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

# Persistent session dedicated to Ollama (Cloud + local) — avoids per-request TLS handshakes.
ollama_session: Optional[aiohttp.ClientSession] = None

# Shared session for OpenAI-compatible third-party providers.
third_party_session: Optional[aiohttp.ClientSession] = None

# Cached template parts — extracted once when auth is captured, reused on every request.
# Avoids copy.deepcopy(auth.body_template) overhead per inference call.
_cached_billing_block: Optional[dict] = None
_cached_auth_blocks: list = []

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
        self._client_id = "app_EMoamEEZ73f0CkXaXp7hrann"  # Codex CLI OAuth client ID

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

# --- Gemini CLI Auth cache ---

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

        # Use credentials from file if present, otherwise fall back to Gemini CLI built-in creds
        client_id = self.client_id or GEMINI_OAUTH_CLIENT_ID
        client_secret = self.client_secret or GEMINI_OAUTH_CLIENT_SECRET

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

gemini_auth = GeminiAuthCache()


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

    @property
    def is_ready(self) -> bool:
        return self.access_token is not None

    def is_expired(self) -> bool:
        if not self.expires_at:
            return True
        return datetime.now() >= self.expires_at - timedelta(minutes=5)

    def reload_from_file(self) -> bool:
        """Re-read credentials from disk if the file has changed. Returns True if token available."""
        if not os.path.exists(QWEN_CREDS_FILE):
            return False
        try:
            mtime = os.path.getmtime(QWEN_CREDS_FILE)
            if mtime <= self._file_mtime and self.access_token:
                return True  # File unchanged and we have a token

            with open(QWEN_CREDS_FILE, "r", encoding="utf-8") as f:
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
        """Try to reload credentials from file (CLI manages token refresh)."""
        if not self.is_expired():
            return True
        # Re-read from file — the Qwen CLI may have refreshed the token
        return self.reload_from_file()

    def get_auth_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
            "X-DashScope-AuthType": "qwen-oauth",
            "X-DashScope-CacheControl": "enable",
        }

qwen_auth = QwenAuthCache()

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
zai_api_key: Optional[str] = _cfg.get("zai_api_key") or None
if zai_api_key:
    logger.info("Z.AI API key loaded from config.json")
xiaomi_api_key: Optional[str] = _cfg.get("xiaomi_api_key") or None
if xiaomi_api_key:
    logger.info("Xiaomi API key loaded from config.json")
opencode_api_key: Optional[str] = _cfg.get("opencode_api_key") or None
if opencode_api_key:
    logger.info("OpenCode Zen API key loaded from config.json")
opencode_go_api_key: Optional[str] = _cfg.get("opencode_go_api_key") or None
if opencode_go_api_key:
    logger.info("OpenCode Go API key loaded from config.json")
fireworks_api_key: Optional[str] = _cfg.get("fireworks_api_key") or None
if fireworks_api_key:
    logger.info("Fireworks API key loaded from config.json")
nvidia_api_key: Optional[str] = _cfg.get("nvidia_api_key") or None
if nvidia_api_key:
    logger.info("NVIDIA NIM API key loaded from config.json")
timeout_routing_enabled: bool = _cfg.get("timeout_routing_enabled", True)
timeout_cutoff_seconds: float = _cfg.get("timeout_cutoff_seconds", 6.0)
max_total_seconds: float = _cfg.get("max_total_seconds", 9.0)
logger.info(f"Timeout routing: {'enabled' if timeout_routing_enabled else 'disabled'} (TTFT cutoff {timeout_cutoff_seconds}s, max total {max_total_seconds}s)")
_round_robin_counter: int = 0


# --- Per-model latency / reliability stats (in-memory, rolling window) ---

class ModelStatsTracker:
    """Rolling window of time-to-first-token measurements per model."""

    def __init__(self, window: int = 50):
        self._records: dict = {}  # model -> deque of (ttft_s, success)
        self._window = window

    def record(self, model: str, ttft: float, success: bool):
        if model not in self._records:
            self._records[model] = deque(maxlen=self._window)
        self._records[model].append((ttft, success))

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict for on-disk persistence."""
        return {
            "window": self._window,
            "records": {
                model: [[t, s] for t, s in records]
                for model, records in self._records.items()
            },
        }

    def load_from_dict(self, data: dict) -> None:
        """Restore state from a previously-saved dict. Preserves rolling-window
        maxlen — records exceeding the window are silently truncated to the tail."""
        window = int(data.get("window", self._window))
        self._window = window
        self._records = {}
        for model, entries in data.get("records", {}).items():
            dq = deque(maxlen=window)
            for entry in entries:
                if isinstance(entry, (list, tuple)) and len(entry) == 2:
                    dq.append((float(entry[0]), bool(entry[1])))
            if dq:
                self._records[model] = dq

    def get_stats(self) -> dict:
        out = {}
        for model, records in self._records.items():
            if not records:
                continue
            ttfts = [t for t, s in records if s]
            success_count = sum(1 for _, s in records if s)
            out[model] = {
                "samples": len(records),
                "success_rate": round(success_count / len(records), 3),
                "median_ttft_s": round(statistics.median(ttfts), 3) if ttfts else None,
                "p90_ttft_s": round(sorted(ttfts)[int(len(ttfts) * 0.9)], 3) if len(ttfts) >= 10 else None,
            }
        return out

    def get_reliable_model(self, exclude: str) -> Optional[str]:
        """Return a reliable fallback model, excluding the given one.

        Selection strategy (tiered):
          1. **Fast tier** — models with median TTFT < 3.0s AND success_rate >= 0.8.
             Pick one at random, weighted by score. Load-balances across good models
             so we don't hammer the single fastest one and trip its rate limits.
          2. **Best-effort tier** — if no model meets the fast-tier threshold, fall
             back to the single highest-scoring candidate (original behavior).

        Score = success_rate * 0.6 + speed_score * 0.3 + sample_confidence * 0.1
        Uses both TTFT stats and full request_stats for richer signal.
        """
        candidates = []       # all viable (success_rate >= 0.7)
        fast_candidates = []  # subset: median_ttft < 3.0 AND success_rate >= 0.8
        for model, records in self._records.items():
            if model == exclude or len(records) < 5:
                continue
            ttfts = [t for t, s in records if s]
            if not ttfts:
                continue
            # TTFT success rate from this tracker
            ttft_success = sum(1 for _, s in records if s) / len(records)

            # Blend in overall request success rate from request_stats
            req_success, req_median_lat = request_stats.get_model_reliability(model)
            req_samples = request_stats.get_model_sample_count(model)

            # Combined success rate: weight request_stats more if it has data
            if req_samples >= 5:
                success_rate = ttft_success * 0.4 + req_success * 0.6
            else:
                success_rate = ttft_success

            if success_rate < 0.7:
                continue

            median_ttft = statistics.median(ttfts)
            # Speed score: normalize to 0-1 range (0s = 1.0, 10s+ = 0.0)
            speed_score = max(0.0, 1.0 - median_ttft / 10.0)
            # Sample confidence: more samples = more trust (caps at 1.0 at 30 samples)
            total_samples = len(records) + req_samples
            sample_confidence = min(1.0, total_samples / 30.0)

            score = success_rate * 0.6 + speed_score * 0.3 + sample_confidence * 0.1
            candidates.append((model, score))
            if median_ttft < 3.0 and success_rate >= 0.8:
                fast_candidates.append((model, score))

        # Fast tier: weighted random pick among models that consistently beat 3s TTFT.
        if fast_candidates:
            models = [m for m, _ in fast_candidates]
            weights = [s for _, s in fast_candidates]
            return random.choices(models, weights=weights, k=1)[0]

        # Best-effort tier: no fast candidates, pick the single best.
        if not candidates:
            return None
        candidates.sort(key=lambda x: -x[1])
        return candidates[0][0]

model_stats = ModelStatsTracker()


# --- Full request stats tracker (all providers, streaming + direct) ---

class RequestStatsTracker:
    """Track request counts, latencies, and error rates per model per mode."""

    def __init__(self, window: int = 100):
        self._window = window
        self._data: dict = {}  # model -> mode -> {count, errors, latencies: deque}
        self._global_count = 0
        self._global_errors = 0
        self._start_time = time.time()

    def _ensure(self, model: str, mode: str):
        if model not in self._data:
            self._data[model] = {}
        if mode not in self._data[model]:
            self._data[model][mode] = {"count": 0, "errors": 0, "latencies": deque(maxlen=self._window)}

    def record(self, model: str, mode: str, latency: float, success: bool):
        self._ensure(model, mode)
        bucket = self._data[model][mode]
        bucket["count"] += 1
        bucket["latencies"].append(latency)
        self._global_count += 1
        if not success:
            bucket["errors"] += 1
            self._global_errors += 1

    def get_stats(self) -> dict:
        by_model = {}
        for model, modes in self._data.items():
            by_model[model] = {}
            for mode, bucket in modes.items():
                lats = list(bucket["latencies"])
                count = bucket["count"]
                errors = bucket["errors"]
                by_model[model][mode] = {
                    "requests": count,
                    "errors": errors,
                    "error_rate": round(errors / count, 3) if count else 0,
                    "median_latency_s": round(statistics.median(lats), 3) if lats else None,
                    "p90_latency_s": round(sorted(lats)[int(len(lats) * 0.9)], 3) if len(lats) >= 10 else None,
                    "avg_latency_s": round(sum(lats) / len(lats), 3) if lats else None,
                }
        uptime = time.time() - self._start_time
        return {
            "global": {
                "total_requests": self._global_count,
                "total_errors": self._global_errors,
                "error_rate": round(self._global_errors / self._global_count, 3) if self._global_count else 0,
                "uptime_s": round(uptime, 1),
                "requests_per_minute": round(self._global_count / (uptime / 60), 2) if uptime > 0 else 0,
            },
            "by_model": by_model,
        }

    def get_model_reliability(self, model: str) -> tuple[float, float]:
        """Return (success_rate, median_latency) across all modes for a model."""
        total, errors, lats = 0, 0, []
        for mode, bucket in self._data.get(model, {}).items():
            total += bucket["count"]
            errors += bucket["errors"]
            lats.extend(bucket["latencies"])
        success_rate = (total - errors) / total if total else 0
        median_lat = statistics.median(lats) if lats else 999
        return success_rate, median_lat

    def get_model_sample_count(self, model: str) -> int:
        return sum(b["count"] for b in self._data.get(model, {}).values())

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict for on-disk persistence."""
        data_out = {}
        for model, modes in self._data.items():
            data_out[model] = {}
            for mode, bucket in modes.items():
                data_out[model][mode] = {
                    "count": bucket["count"],
                    "errors": bucket["errors"],
                    "latencies": list(bucket["latencies"]),
                }
        return {
            "window": self._window,
            "data": data_out,
            "global_count": self._global_count,
            "global_errors": self._global_errors,
            "start_time": self._start_time,
        }

    def load_from_dict(self, data: dict) -> None:
        """Restore state from a previously-saved dict. Preserves rolling-window
        maxlen on latencies; counts and start_time carry over so
        requests_per_minute stays meaningful across restarts."""
        window = int(data.get("window", self._window))
        self._window = window
        self._data = {}
        for model, modes in data.get("data", {}).items():
            self._data[model] = {}
            for mode, bucket in modes.items():
                lats = deque(maxlen=window)
                for lat in bucket.get("latencies", []):
                    try:
                        lats.append(float(lat))
                    except (TypeError, ValueError):
                        pass
                self._data[model][mode] = {
                    "count": int(bucket.get("count", 0)),
                    "errors": int(bucket.get("errors", 0)),
                    "latencies": lats,
                }
        self._global_count = int(data.get("global_count", 0))
        self._global_errors = int(data.get("global_errors", 0))
        # Preserve the original start_time so uptime/rpm stay cumulative across
        # restarts. Fall back to now() if the saved file has no timestamp.
        self._start_time = float(data.get("start_time", time.time()))


request_stats = RequestStatsTracker()


# --- Stats persistence to disk -----------------------------------------------
# Historical stats survive proxy restarts so smart routing doesn't start cold
# every time. File is rewritten atomically via tmp+rename to avoid corruption.

STATS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stats.json")
_STATS_FILE_VERSION = 1
_stats_persist_task: Optional[asyncio.Task] = None
_stats_persist_interval_s = 60.0


def _load_stats_from_disk() -> None:
    """Restore ModelStatsTracker and RequestStatsTracker state from STATS_FILE.
    Missing or corrupted files are ignored — we just start with empty stats."""
    if not os.path.exists(STATS_FILE):
        logger.info(f"No stats file at {STATS_FILE} — starting with empty history")
        return
    try:
        with open(STATS_FILE, "r") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"Stats file {STATS_FILE} unreadable ({e}) — starting with empty history")
        return
    try:
        if payload.get("version") != _STATS_FILE_VERSION:
            logger.warning(f"Stats file version mismatch (got {payload.get('version')}, expected {_STATS_FILE_VERSION}) — ignoring")
            return
        model_stats.load_from_dict(payload.get("model_stats", {}))
        request_stats.load_from_dict(payload.get("request_stats", {}))
        saved_at = payload.get("saved_at")
        age_str = ""
        if isinstance(saved_at, (int, float)):
            age_min = (time.time() - saved_at) / 60.0
            age_str = f", saved {age_min:.1f} min ago"
        logger.info(
            f"Restored stats: {len(model_stats._records)} models in TTFT tracker, "
            f"{request_stats._global_count} total requests{age_str}"
        )
    except Exception as e:
        logger.warning(f"Failed to restore stats from {STATS_FILE}: {e} — starting with empty history")


def _save_stats_to_disk() -> None:
    """Atomically write current stats to STATS_FILE (tmp + rename)."""
    payload = {
        "version": _STATS_FILE_VERSION,
        "saved_at": time.time(),
        "model_stats": model_stats.to_dict(),
        "request_stats": request_stats.to_dict(),
    }
    tmp_path = STATS_FILE + ".tmp"
    try:
        with open(tmp_path, "w") as f:
            json.dump(payload, f)
        os.replace(tmp_path, STATS_FILE)
    except OSError as e:
        logger.warning(f"Failed to save stats to {STATS_FILE}: {e}")
        # Best-effort cleanup of the tmp file
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass


async def _stats_persist_loop() -> None:
    """Background task: flush stats to disk every _stats_persist_interval_s."""
    while True:
        try:
            await asyncio.sleep(_stats_persist_interval_s)
        except asyncio.CancelledError:
            return
        try:
            # File I/O off the event loop so we don't stall request handling.
            await asyncio.to_thread(_save_stats_to_disk)
        except Exception as e:
            logger.warning(f"Stats persist loop hiccup: {e}")


# Load historical stats at module import so model_stats and request_stats are
# warm before the first request arrives.
_load_stats_from_disk()


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
    """OpenRouter models use 'provider/model' format (contain '/').
    Excludes models with known prefixes (e.g. fireworks:, nvidia:) that may contain slashes."""
    low = model.lower()
    if low.startswith("fireworks:") or low.startswith("nvidia:"):
        return False
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


def is_gemini_cli_model(model: str) -> bool:
    """Gemini CLI models use gcli-* prefix."""
    return model.lower().startswith("gcli-")


def is_zai_model(model: str) -> bool:
    """Z.AI models use 'zai:model' prefix."""
    return model.lower().startswith("zai:")


def is_xiaomi_model(model: str) -> bool:
    """Xiaomi models use 'xiaomi:model' prefix."""
    return model.lower().startswith("xiaomi:")


def is_opencode_model(model: str) -> bool:
    """OpenCode models use 'opencode:model' or 'opencode-go:model' prefix."""
    m = model.lower()
    return m.startswith("opencode:") or m.startswith("opencode-go:")


def is_qwen_model(model: str) -> bool:
    """Qwen Code models use 'qwen:model' prefix."""
    return model.lower().startswith("qwen:")


def is_fireworks_model(model: str) -> bool:
    """Fireworks models use 'fireworks:model' prefix."""
    return model.lower().startswith("fireworks:")


def is_nvidia_model(model: str) -> bool:
    """NVIDIA NIM models use 'nvidia:model' prefix."""
    return model.lower().startswith("nvidia:")


# Canonical model name aliases — dot-version notation → hyphen notation used by the API
_MODEL_ALIASES: dict[str, str] = {
    "claude-sonnet-4.6": "claude-sonnet-4-6",
    "claude-opus-4.6": "claude-opus-4-6",
    "claude-haiku-4.5": "claude-haiku-4-5-20251001",
    "claude-sonnet-4.5": "claude-sonnet-4-5-20250929",
}


def normalize_model_name(model: str) -> str:
    """Resolve known model aliases to their canonical API identifiers."""
    return _MODEL_ALIASES.get(model, model)


def _pick_fallback_model(exclude_model: str) -> str:
    """Select the most reliable model from observed stats, or a safe provider default."""
    best = model_stats.get_reliable_model(exclude=exclude_model)
    if best:
        return best
    # Provider defaults in reliability priority order
    if auth.is_ready:
        return DEFAULT_MODEL
    if antigravity_auth.is_ready:
        return "antigravity-gemini-2.5-flash"
    if codex_auth.is_ready:
        return DEFAULT_CODEX_MODEL
    if gemini_auth.is_ready:
        return "gcli-gemini-2.5-flash"
    return DEFAULT_MODEL


def _make_streaming_gen(system_prompt, messages, model, max_tokens, oai_messages=None):
    """Route a model name to its streaming generator (no extra_params — used for fallback)."""
    model = normalize_model_name(model)
    routed_system_prompt = system_prompt
    routed_messages = messages
    if _model_uses_oai_messages(model):
        routed_system_prompt = None
        routed_messages = oai_messages if oai_messages is not None else messages
    if is_ollama_model(model):
        return call_ollama_streaming(routed_system_prompt, routed_messages, model, max_tokens)
    if is_openrouter_model(model):
        return call_openrouter_streaming(routed_system_prompt, routed_messages, model, max_tokens)
    if is_codex_model(model):
        return call_codex_streaming(routed_system_prompt, routed_messages, model, max_tokens)
    if is_antigravity_model(model):
        return call_antigravity_streaming(routed_system_prompt, routed_messages, model, max_tokens)
    if is_gemini_cli_model(model):
        return call_gemini_streaming(routed_system_prompt, routed_messages, model, max_tokens)
    if is_zai_model(model):
        return call_zai_streaming(routed_system_prompt, routed_messages, model, max_tokens)
    if is_xiaomi_model(model):
        return call_xiaomi_streaming(routed_system_prompt, routed_messages, model, max_tokens)
    if is_opencode_model(model):
        return call_opencode_streaming(routed_system_prompt, routed_messages, model, max_tokens)
    if is_qwen_model(model):
        return call_qwen_streaming(routed_system_prompt, routed_messages, model, max_tokens)
    if is_fireworks_model(model):
        return call_fireworks_streaming(routed_system_prompt, routed_messages, model, max_tokens)
    if is_nvidia_model(model):
        return call_nvidia_streaming(routed_system_prompt, routed_messages, model, max_tokens)
    return call_api_streaming(routed_system_prompt, routed_messages, model, max_tokens)


async def _stream_under_deadline(aiter, model, total_deadline, start_time, total_timeout, closer=None):
    """Iterate an async iterator under a wall-clock deadline, yielding chunks.

    On timeout: calls `closer` (if provided) to release upstream resources, records
    a failure in model_stats so `get_reliable_model()` penalizes this model in future
    fallback selection, and emits a synthetic finish_reason="stop" chunk + [DONE] so
    clients (SkyrimNet) treat this as a natural completion and don't retry. Without
    the synthetic [DONE], clients see a truncated stream and fire a second request —
    causing the "two generations per call" behavior we want to avoid.
    """
    while True:
        try:
            remaining = total_deadline - time.time()
            if remaining <= 0:
                raise asyncio.TimeoutError()
            chunk = await asyncio.wait_for(aiter.__anext__(), timeout=remaining)
        except StopAsyncIteration:
            return
        except asyncio.TimeoutError:
            if closer is not None:
                try:
                    await closer()
                except Exception:
                    pass
            elapsed = time.time() - start_time
            model_stats.record(model, elapsed, success=False)
            logger.warning(
                f"[timeout-routing] {model} stream exceeded total cap {total_timeout}s "
                f"({elapsed:.1f}s) — cutting mid-generation, recorded as failure for smart routing"
            )
            stop_chunk = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(stop_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            return
        yield chunk


async def _with_timeout_routing(gen, system_prompt, messages, oai_messages, model, max_tokens, ttft_timeout=None, total_timeout=None):
    """
    Wrap a streaming generator with two deadlines:

    Phase 1 (before first content token): enforce `ttft_timeout`. If no content arrives
    in time, cancel the stream and re-route to the most reliable fallback model. The total
    deadline also applies here so a hung stream gets the tighter of the two cutoffs.

    Phase 2 (after first content token): enforce `total_timeout` as a hard wall-clock ceiling.
    This cap applies to both the original stream AND any fallback stream kicked off from
    Phase 1, so a slow fallback can't blow past the cap either. If the upstream hasn't
    finished by then, close the stream mid-generation with [DONE]. We can't re-route
    mid-stream because partial content has already been sent to the client.

    Records TTFT to ModelStatsTracker on both success and timeout.
    """
    if ttft_timeout is None:
        ttft_timeout = timeout_cutoff_seconds
    if total_timeout is None:
        total_timeout = max_total_seconds

    aiter = gen.__aiter__()
    start = time.time()
    pre_content_chunks = []

    ttft_deadline = start + ttft_timeout
    total_deadline = start + total_timeout

    # --- Phase 1: wait for first real content chunk (TTFT enforcement + fallback) ---
    while True:
        try:
            phase1_deadline = min(ttft_deadline, total_deadline)
            remaining = phase1_deadline - time.time()
            if remaining <= 0:
                raise asyncio.TimeoutError()
            chunk = await asyncio.wait_for(aiter.__anext__(), timeout=remaining)
        except StopAsyncIteration:
            for c in pre_content_chunks:
                yield c
            return
        except asyncio.TimeoutError:
            try:
                await gen.aclose()
            except Exception:
                pass
            elapsed = time.time() - start
            model_stats.record(model, elapsed, success=False)
            fallback = _pick_fallback_model(exclude_model=model)
            logger.warning(
                f"[timeout-routing] {model} exceeded pre-content deadline "
                f"({elapsed:.1f}s, TTFT cutoff {ttft_timeout}s / total cap {total_timeout}s) — switching to {fallback}"
            )
            # Don't replay pre_content_chunks — the fallback generator will emit
            # its own role chunk. Replaying would send duplicate role/setup chunks.
            # The fallback stream shares the original `total_deadline` so a slow
            # fallback still gets hard-cut at `max_total_seconds` from the original
            # request start — otherwise a stuck fallback could run indefinitely.
            fallback_gen = _make_streaming_gen(system_prompt, messages, fallback, max_tokens, oai_messages)
            try:
                async for c in _stream_under_deadline(
                    fallback_gen.__aiter__(),
                    fallback,
                    total_deadline,
                    start,
                    total_timeout,
                    closer=fallback_gen.aclose,
                ):
                    yield c
            except Exception as e:
                logger.error(f"[timeout-routing] Fallback {fallback} failed: {e}")
                try:
                    await fallback_gen.aclose()
                except Exception:
                    pass
            return

        pre_content_chunks.append(chunk)

        # Detect first non-empty content chunk to mark TTFT
        # Cheap string check guards the expensive json.loads — role/done chunks don't contain '"content"'
        has_content = False
        try:
            if (isinstance(chunk, str) and chunk.startswith("data: ")
                    and "[DONE]" not in chunk and '"content"' in chunk):
                data = json.loads(chunk[6:])
                delta = data.get("choices", [{}])[0].get("delta", {})
                if delta.get("content"):
                    has_content = True
        except Exception:
            pass

        if has_content:
            model_stats.record(model, time.time() - start, success=True)
            break

    for c in pre_content_chunks:
        yield c

    # --- Phase 2: stream remainder under total wall-clock ceiling (hard cut) ---
    # If we blow past the ceiling, the call is marked as a failure in model_stats
    # so get_reliable_model() penalizes this model in future fallback selection —
    # the hard cap becomes a signal for the smart router, not just a hang-guard.
    async for c in _stream_under_deadline(aiter, model, total_deadline, start, total_timeout, closer=gen.aclose):
        yield c


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
    if not auth.is_ready:
        global _cached_billing_block, _cached_auth_blocks
        auth.headers = dict(headers)
        # Strip tool definitions, system prompts, and other control-plane baggage
        # from the cached template. The warm-up request itself is still forwarded
        # unchanged so auth capture behaves exactly like the CLI invocation.
        sanitized = _sanitize_claude_template(parsed)
        auth.body_template = sanitized
        # Cache the two fields reused on every inference call so _build_api_body
        # can skip copy.deepcopy(auth.body_template) entirely.
        _cached_billing_block = parsed["system"][0] if parsed.get("system") else None
        first_msg = parsed["messages"][0] if parsed.get("messages") else {}
        _cached_auth_blocks = [
            b for b in (first_msg.get("content") or [])
            if isinstance(b, dict) and b.get("type") == "text"
            and "<system-reminder>" in b.get("text", "")
        ]
        template_size = len(json.dumps(sanitized))
        logger.info(f"Captured {len(auth.headers)} headers + template ({template_size:,} bytes, tools/system stripped)")

    # Forward to real API
    forward_body = body
    if not auth.is_ready:
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


async def start_interceptor(port: int = INTERCEPTOR_PORT):
    """Start MITM interceptor and capture auth from a clean temp dir.

    Args:
        port: Port to bind the interceptor to. Defaults to INTERCEPTOR_PORT (9999).
              MCP mode uses MCP_INTERCEPTOR_PORT (9997) to avoid conflicts.
    """
    global _interceptor_port
    _interceptor_port = port

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


# --- Claude Auth Auto-Refresh ---

_claude_refresh_lock = asyncio.Lock()
_claude_refresh_task: Optional[asyncio.Task] = None


async def recapture_claude_auth() -> bool:
    """Re-run claude --print through the existing interceptor to refresh auth.
    Returns True if auth was successfully recaptured."""
    if not CLAUDE_PATH:
        return False

    async with _claude_refresh_lock:
        # Another coroutine may have refreshed while we waited for the lock
        if auth.is_ready:
            return True

        logger.info("Recapturing Claude auth...")
        for attempt in range(3):
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
                env = os.environ.copy()
                env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{_interceptor_port}"
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
                        # Recreate the persistent session with fresh auth
                        if auth.session:
                            await auth.session.close()
                        auth.session = create_session()
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
    REFRESH_INTERVAL = 45 * 60  # 45 minutes
    while True:
        await asyncio.sleep(REFRESH_INTERVAL)
        if not CLAUDE_PATH:
            break
        logger.info("Proactive Claude auth refresh triggered")
        # Clear auth to force recapture through the interceptor
        auth.headers = None
        auth.body_template = None
        await recapture_claude_auth()


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
                        "client_id": "app_EMoamEEZ73f0CkXaXp7hrann",
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


async def _fetch_gemini_project_id() -> None:
    """Call loadCodeAssist to get the managed project ID for this user/account.

    For personal Gemini Advanced subscribers the server returns a managed project.
    For GCP users the project comes from GOOGLE_CLOUD_PROJECT or the response.
    Mirrors the Gemini CLI: omit cloudaicompanionProject entirely when unknown.
    """
    # Check env var first (same priority as CLI)
    env_project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT_ID")
    if env_project:
        gemini_auth.project_id = env_project
        logger.info(f"Gemini project_id from env: {env_project}")
        return

    url = f"{GEMINI_CODE_ASSIST_ENDPOINT}/{GEMINI_CODE_ASSIST_API_VERSION}:loadCodeAssist"
    headers = gemini_auth.get_auth_headers()
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
        async with gemini_auth.session.post(url, json=payload, headers=headers,
                                            timeout=aiohttp.ClientTimeout(total=10, connect=5)) as resp:
            if resp.status == 200:
                data = await resp.json()
                project_id = data.get("cloudaicompanionProject")
                gemini_auth.project_id = project_id or None
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
    if not os.path.exists(GEMINI_CREDS_FILE):
        logger.info(f"No Gemini credentials file at {GEMINI_CREDS_FILE} -- run 'gemini auth login'")
        return

    try:
        with open(GEMINI_CREDS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        gemini_auth.refresh_token = data.get("refresh_token")
        gemini_auth.client_id = data.get("client_id")
        gemini_auth.client_secret = data.get("client_secret")

        # Some formats include a cached access_token with an expiry_date (ms timestamp)
        if data.get("access_token"):
            gemini_auth.access_token = data["access_token"]
            expiry = data.get("expiry_date")
            if expiry:
                gemini_auth.expires_at = datetime.fromtimestamp(expiry / 1000)
            else:
                gemini_auth.expires_at = datetime.now() + timedelta(hours=1)

        if gemini_auth.refresh_token:
            if await gemini_auth.refresh_if_needed():
                gemini_auth.session = create_session()
                logger.info("Gemini auth loaded and token refreshed")
            elif gemini_auth.access_token:
                gemini_auth.session = create_session()
                logger.warning("Gemini token refresh failed, using cached token")
            else:
                logger.error("Gemini token refresh failed and no cached token available")
        elif gemini_auth.access_token:
            gemini_auth.session = create_session()
            logger.info("Gemini auth loaded (no refresh credentials -- token may expire)")
        else:
            logger.warning(f"Gemini credentials file found but missing required fields (need refresh_token or access_token)")

        # Fetch the real managed project ID from the Code Assist API
        if gemini_auth.is_ready and gemini_auth.session:
            await _fetch_gemini_project_id()

    except Exception as e:
        logger.error(f"Failed to load Gemini auth: {e}")


# --- Direct API call ---

def _build_api_body(system_prompt: Optional[str], messages: list, model: str) -> dict:
    """Build Anthropic API request body from template."""
    # Shallow copy — we replace system/messages/model in-place so the template is never mutated.
    body = _sanitize_claude_template(auth.body_template)

    # 1. Build system array: billing block (cached) + optional user system prompt
    system: list = []
    if _cached_billing_block is not None:
        system.append(_cached_billing_block)
    if system_prompt:
        system.append({"type": "text", "text": system_prompt})
    body["system"] = system

    # 2. Build messages: auth blocks (cached) prepended to first user message
    new_messages = []
    for i, m in enumerate(messages):
        if i == 0 and m["role"] == "user":
            content = _cached_auth_blocks + [{"type": "text", "text": m["content"]}]
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
    body_bytes = json_dumps(body)
    headers["Content-Length"] = str(len(body_bytes))

    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] -> {model} ({len(messages)} msgs)")
    start = time.time()

    session = auth.session or create_session()
    last_error = None
    try:
        for attempt in range(3):
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
                        logger.warning("Auth expired -- triggering auto-refresh")
                        auth.headers = None
                        auth.body_template = None
                        refreshed = await recapture_claude_auth()
                        if refreshed and attempt < 2:
                            headers = dict(auth.headers)
                            headers["Content-Length"] = str(len(body_bytes))
                            session = auth.session or session
                            start = time.time()
                            continue
                    if resp.status >= 500 and attempt < 2:
                        logger.info(f"[{request_id}] Retrying after {resp.status} (attempt {attempt + 1}/3)")
                        last_error = HTTPException(status_code=resp.status, detail=error_text[:200])
                        await asyncio.sleep(2 ** attempt)
                        start = time.time()
                        continue
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
        raise last_error
    finally:
        if not auth.session:
            await session.close()


async def call_api_streaming(system_prompt: Optional[str], messages: list, model: str, max_tokens: int):
    """Direct API call, yields OpenAI-format SSE chunks as they arrive."""

    body = _build_api_body(system_prompt, messages, model)
    headers = dict(auth.headers)
    body_bytes = json_dumps(body)
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
        for attempt in range(3):
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
                        logger.warning("Auth expired -- triggering auto-refresh")
                        auth.headers = None
                        auth.body_template = None
                        refreshed = await recapture_claude_auth()
                        if refreshed and attempt < 2:
                            headers = dict(auth.headers)
                            headers["Content-Length"] = str(len(body_bytes))
                            session = auth.session or session
                            start = time.time()
                            continue
                    if resp.status >= 500 and attempt < 2:
                        logger.info(f"[{request_id}] Retrying after {resp.status} (attempt {attempt + 1}/3, stream)")
                        await asyncio.sleep(2 ** attempt)
                        start = time.time()
                        continue
                    # Yield an error chunk so client sees the failure
                    err, done = yield_sse_error(model, f"[API Error {resp.status}]")
                    yield err; yield done
                    return

                # Stream Claude SSE -> OpenAI SSE
                # Send initial chunk with role
                role_chunk = {"id": cmpl_id, "object": "chat.completion.chunk", "created": created, "model": model,
                              "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]}
                yield f"data: {json.dumps(role_chunk)}\n\n"

                # Pre-build the constant parts of every content chunk once per stream.
                # Per token: only json.dumps(text) varies — avoids 3 dict allocs per token.
                _chunk_pre = f'{{"id":"{cmpl_id}","object":"chat.completion.chunk","created":{created},"model":"{model}","choices":[{{"index":0,"delta":{{"content":'
                _chunk_suf = '},"finish_reason":null}]}'

                buf = bytearray()
                async for raw_chunk in resp.content.iter_any():
                    buf.extend(raw_chunk)
                    while b"\n" in buf:
                        idx = buf.index(b"\n")
                        line = buf[:idx].decode("utf-8", errors="replace").strip()
                        del buf[:idx + 1]
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if not data_str or data_str == "[DONE]":
                            continue
                        try:
                            event = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        etype = event.get("type", "")

                        # Surface stream-level errors instead of silently swallowing them
                        if etype == "error":
                            err_msg = event.get("error", {}).get("message", "Unknown stream error")
                            logger.error(f"[{request_id}] Stream error: {err_msg}")
                            yield f"data: {_chunk_pre}{json.dumps(f'[Stream Error: {err_msg}]')}{_chunk_suf}\n\n"
                            continue

                        if etype == "content_block_delta":
                            delta = event.get("delta", {})
                            if delta.get("type") == "text_delta":
                                text = delta.get("text", "")
                                if text:
                                    total_chars += len(text)
                                    yield f"data: {_chunk_pre}{json.dumps(text)}{_chunk_suf}\n\n"

                # Flush any remaining data in the buffer (last segment without trailing \n)
                if buf:
                    line = buf.decode("utf-8", errors="replace").strip()
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str and data_str != "[DONE]":
                            try:
                                event = json.loads(data_str)
                                if event.get("type") == "content_block_delta":
                                    delta = event.get("delta", {})
                                    if delta.get("type") == "text_delta":
                                        text = delta.get("text", "")
                                        if text:
                                            total_chars += len(text)
                                            yield f"data: {_chunk_pre}{json.dumps(text)}{_chunk_suf}\n\n"
                            except json.JSONDecodeError:
                                pass

                # Final chunk with finish_reason
                stop_chunk = {"id": cmpl_id, "object": "chat.completion.chunk", "created": created, "model": model,
                              "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                yield f"data: {json.dumps(stop_chunk)}\n\n"
                yield "data: [DONE]\n\n"

                elapsed = time.time() - start
                logger.info(f"[{request_id}] <- {total_chars} chars ({elapsed:.1f}s, streamed)")
                return
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
        logger.error(f"[{request_id}] Streaming error: {e}")
        err, done = yield_sse_error(model, f"[Streaming Error: {e}]")
        yield err; yield done
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


def _build_codex_exec_args(model: str) -> tuple[str, ...]:
    """Return the Codex CLI argv with fast-mode overrides."""
    return (
        "exec",
        "--model", model,
        "-c", f'model_reasoning_effort="{CODEX_FAST_REASONING_EFFORT}"',
        "--json",
        "--dangerously-bypass-approvals-and-sandbox",
        "--skip-git-repo-check",
        "-",
    )


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
            *_build_codex_exec_args(model),
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

                    # Current Codex CLI format: {"id": ..., "msg": {"type": "agent_message", "message": "..."}}
                    msg = event.get("msg", {})
                    if isinstance(msg, dict) and msg.get("type") == "agent_message":
                        text = msg.get("message", "")
                        if text:
                            response_text = text

                    # Legacy format: {"type": "item.completed", "item": {"type": "agent_message", "text": "..."}}
                    elif event_type == "item.completed":
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
            logger.warning(f"[{request_id}] Codex CLI: no agent_message found in output, raw length={len(raw_output)}")
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
            *_build_codex_exec_args(model),
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
                    event_type = event.get("type", "")

                    # Current format: {"id": ..., "msg": {"type": "agent_message", "message": "..."}}
                    msg = event.get("msg", {})
                    if isinstance(msg, dict) and msg.get("type") == "agent_message":
                        text = msg.get("message", "")
                        if text:
                            response_text = text

                    # Legacy format
                    elif event_type == "item.completed":
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
            logger.warning(f"[{request_id}] Codex CLI: no agent_message found in stream output, raw length={len(raw_output)}")
            response_text = raw_output

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
        err, done = yield_sse_error(model, "[Codex CLI timeout]")
        yield err; yield done
    except Exception as e:
        logger.error(f"[{request_id}] Codex CLI error: {e}")
        err, done = yield_sse_error(model, f"[Codex CLI error: {e}]")
        yield err; yield done
    finally:
        if isolated_home:
            _cleanup_isolated_home(isolated_home)


# --- OpenRouter API calls ---

async def call_openrouter_direct(system_prompt: Optional[str], messages: list, model: str, max_tokens: int, **extra_params) -> str:
    """Forward request to OpenRouter (OpenAI-compatible), collect full response."""
    if not openrouter_api_key:
        raise HTTPException(status_code=500, detail="OpenRouter API key not configured")

    payload = {"model": model, "messages": build_oai_messages(system_prompt, messages), "max_tokens": max_tokens}
    payload.update({k: v for k, v in extra_params.items() if v is not None})
    headers = {"Authorization": f"Bearer {openrouter_api_key}", "Content-Type": "application/json"}

    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] -> OpenRouter {model} ({len(messages)} msgs)")
    start = time.time()

    session = third_party_session or auth.session or create_session()
    owns_session = session is not third_party_session and session is not auth.session
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
        if owns_session:
            await session.close()


async def call_openrouter_streaming(system_prompt: Optional[str], messages: list, model: str, max_tokens: int, **extra_params):
    """Forward request to OpenRouter with streaming, passthrough SSE directly."""
    if not openrouter_api_key:
        yield 'data: {"error": "OpenRouter API key not configured"}\n\n'
        yield "data: [DONE]\n\n"
        return

    payload = {"model": model, "messages": build_oai_messages(system_prompt, messages), "max_tokens": max_tokens, "stream": True}
    payload.update({k: v for k, v in extra_params.items() if v is not None})
    headers = {"Authorization": f"Bearer {openrouter_api_key}", "Content-Type": "application/json"}

    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] -> OpenRouter {model} ({len(messages)} msgs, stream)")
    start = time.time()

    session = third_party_session or auth.session or create_session()
    owns_session = session is not third_party_session and session is not auth.session
    try:
        async with session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json=payload, headers=headers,
        ) as resp:
            if resp.status != 200:
                error_body = await resp.text()
                logger.error(f"[{request_id}] OpenRouter {resp.status}: {error_body[:300]}")
                err, done = yield_sse_error(model, f"[OpenRouter Error {resp.status}]")
                yield err; yield done
                return

            # OpenRouter returns OpenAI-format SSE — passthrough directly
            async for event in passthrough_sse(resp, request_id, "OpenRouter", start):
                yield event
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
        logger.error(f"[{request_id}] OpenRouter streaming error: {e}")
        err, done = yield_sse_error(model, f"[OpenRouter Error: {e}]")
        yield err; yield done
    finally:
        if owns_session:
            await session.close()


# --- Ollama API calls ---

_OLLAMA_UNSUPPORTED_PARAMS = {"top_k"}


def _ollama_payload_fixup(payload: dict, extra_params: dict) -> None:
    """Apply Ollama-specific payload adjustments for OpenAI-compatible calls."""
    payload.update({
        k: v for k, v in extra_params.items()
        if v is not None and k not in _OLLAMA_UNSUPPORTED_PARAMS
    })


async def call_ollama_direct(system_prompt: Optional[str], messages: list, model: str, max_tokens: int, **extra_params) -> str:
    """Forward request to Ollama (OpenAI-compatible), collect full response."""
    api_model = model[len("ollama:"):]
    if ollama_api_key:
        endpoint = "https://ollama.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {ollama_api_key}", "Content-Type": "application/json"}
    else:
        endpoint = "http://localhost:11434/v1/chat/completions"
        headers = {"Content-Type": "application/json"}

    payload = {"model": api_model, "messages": build_oai_messages(system_prompt, messages), "max_tokens": max_tokens}
    _ollama_payload_fixup(payload, extra_params)

    request_id = uuid.uuid4().hex[:8]
    target = "Ollama Cloud" if ollama_api_key else f"Ollama local ({api_model})"
    logger.info(f"[{request_id}] -> {target} ({len(messages)} msgs)")
    start = time.time()

    session = ollama_session or auth.session or create_session()
    owns_session = not ollama_session and not auth.session
    last_error = None
    try:
        for attempt in range(3):
            async with session.post(endpoint, json=payload, headers=headers) as resp:
                elapsed = time.time() - start
                if resp.status in (401, 403):
                    raise HTTPException(status_code=401, detail="Ollama Cloud auth failed — check API key")
                if resp.status == 429:
                    raise HTTPException(status_code=429, detail="Ollama rate limit exceeded")
                if resp.status != 200:
                    error_body = await resp.text()
                    logger.error(f"[{request_id}] Ollama {resp.status}: {error_body[:300]}")
                    if resp.status >= 500 and attempt < 2:
                        logger.info(f"[{request_id}] Retrying Ollama after {resp.status} (attempt {attempt + 1}/3)")
                        last_error = HTTPException(status_code=resp.status, detail=error_body[:200])
                        await asyncio.sleep(2 ** attempt)
                        start = time.time()
                        continue
                    raise HTTPException(status_code=resp.status, detail=error_body[:200])
                data = await resp.json()
                text = _extract_oai_content(data["choices"][0]["message"])
                if text is None and _has_reasoning_without_content(data):
                    retry_payload = dict(payload)
                    retry_payload.pop("max_tokens", None)
                    if _is_reasoning_truncated(data):
                        logger.info(f"[{request_id}] Ollama response exhausted max_tokens in reasoning; retrying without cap")
                    else:
                        logger.info(f"[{request_id}] Ollama returned reasoning without content; retrying once without max_tokens")
                    async with session.post(endpoint, json=retry_payload, headers=headers) as retry_resp:
                        if retry_resp.status != 200:
                            retry_error = await retry_resp.text()
                            raise HTTPException(status_code=retry_resp.status, detail=retry_error[:200])
                        retry_data = await retry_resp.json()
                        text = _extract_oai_content(retry_data["choices"][0]["message"])
                if text is None:
                    raise HTTPException(status_code=502, detail="Ollama returned no content")
                logger.info(f"[{request_id}] <- {len(text)} chars ({elapsed:.1f}s, Ollama)")
                return text
        raise last_error
    except aiohttp.ClientConnectorError:
        raise HTTPException(status_code=503, detail="Ollama not running at localhost:11434")
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Ollama request timed out")
    finally:
        if owns_session:
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

    payload = {"model": api_model, "messages": build_oai_messages(system_prompt, messages), "max_tokens": max_tokens, "stream": True}
    _ollama_payload_fixup(payload, extra_params)

    request_id = uuid.uuid4().hex[:8]
    target = "Ollama Cloud" if ollama_api_key else f"Ollama local ({api_model})"
    logger.info(f"[{request_id}] -> {target} ({len(messages)} msgs, stream)")
    start = time.time()

    session = ollama_session or auth.session or create_session()
    owns_session = not ollama_session and not auth.session
    try:
        for attempt in range(3):
            async with session.post(endpoint, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    error_body = await resp.text()
                    logger.error(f"[{request_id}] Ollama {resp.status}: {error_body[:300]}")
                    if resp.status >= 500 and attempt < 2:
                        logger.info(f"[{request_id}] Retrying Ollama after {resp.status} (attempt {attempt + 1}/3, stream)")
                        await asyncio.sleep(2 ** attempt)
                        start = time.time()
                        continue
                    err, done = yield_sse_error(model, f"[Ollama Error {resp.status}]")
                    yield err; yield done
                    return

                # Ollama /v1 returns OpenAI-format SSE — passthrough directly
                async for event in passthrough_sse(resp, request_id, "Ollama", start):
                    yield event
                return
    except aiohttp.ClientConnectorError:
        err, done = yield_sse_error(model, "[Ollama Error: not running at localhost:11434]")
        yield err; yield done
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
        logger.error(f"[{request_id}] Ollama streaming error: {e}")
        err, done = yield_sse_error(model, f"[Ollama Error: {e}]")
        yield err; yield done
    finally:
        if owns_session:
            await session.close()


# --- Z.AI ---

ZAI_BASE_URL = "https://api.z.ai/api/coding/paas/v4"


def _zai_supports_vision(api_model: str) -> bool:
    """Z.AI vision models contain 'v' as a suffix/segment (e.g. glm-5v-turbo, glm-4.6v)."""
    low = api_model.lower()
    # Match patterns like "glm-4.6v", "glm-5v-turbo", "glm-4.1v-thinking-flash"
    for part in low.replace(".", "-").split("-"):
        if part.endswith("v") and len(part) > 1:
            return True
    # Also match dedicated vision model names
    if "ocr" in low or "autoglm" in low:
        return True
    return False


async def call_zai_direct(system_prompt: Optional[str], messages: list, model: str, max_tokens: int, **extra_params) -> str:
    """Forward request to Z.AI (OpenAI-compatible), collect full response."""
    if not zai_api_key:
        raise HTTPException(status_code=503, detail="Z.AI API key not configured — add it via /config/zai-key")
    api_model = model[len("zai:"):]
    endpoint = f"{ZAI_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {zai_api_key}", "Content-Type": "application/json"}
    strip = not _zai_supports_vision(api_model)

    payload = {"model": api_model, "messages": build_oai_messages(system_prompt, messages, strip_vision=strip), "max_tokens": max_tokens}
    payload.update({k: v for k, v in extra_params.items() if v is not None})

    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] -> Z.AI ({api_model}, {len(messages)} msgs)")
    start = time.time()

    session = third_party_session or auth.session or create_session()
    owns_session = session is not third_party_session and session is not auth.session
    try:
        async with session.post(endpoint, json=payload, headers=headers) as resp:
            elapsed = time.time() - start
            if resp.status in (401, 403):
                raise HTTPException(status_code=401, detail="Z.AI auth failed — check API key")
            if resp.status == 429:
                raise HTTPException(status_code=429, detail="Z.AI rate limit exceeded")
            if resp.status != 200:
                error_body = await resp.text()
                logger.error(f"[{request_id}] Z.AI {resp.status}: {error_body[:300]}")
                raise HTTPException(status_code=resp.status, detail=error_body[:200])
            data = await resp.json()
            choices = data.get("choices", [])
            if not choices:
                logger.warning(f"[{request_id}] Z.AI returned empty choices array")
                raise HTTPException(status_code=500, detail="Z.AI returned empty response")
            content = _extract_oai_content(choices[0].get("message", {}))
            if content is None:
                # Reasoning model may have exhausted max_tokens on chain-of-thought — retry without cap
                if _is_reasoning_truncated(data):
                    logger.info(f"[{request_id}] Z.AI reasoning truncated, retrying without max_tokens")
                    retry_payload = dict(payload)
                    retry_payload.pop("max_tokens", None)
                    async with session.post(endpoint, json=retry_payload, headers=headers) as retry_resp:
                        if retry_resp.status == 200:
                            retry_data = await retry_resp.json()
                            content = _extract_oai_content(retry_data.get("choices", [{}])[0].get("message", {}))
                            if content:
                                logger.info(f"[{request_id}] <- {len(content)} chars ({time.time() - start:.1f}s, Z.AI, retry)")
                                return content
                logger.warning(f"[{request_id}] Z.AI returned null content, full response: {str(data)[:500]}")
                raise HTTPException(status_code=500, detail="Z.AI returned empty content")
            logger.info(f"[{request_id}] <- {len(content)} chars ({elapsed:.1f}s, Z.AI)")
            return content
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Z.AI request timed out")
    finally:
        if owns_session:
            await session.close()


async def call_zai_streaming(system_prompt: Optional[str], messages: list, model: str, max_tokens: int, **extra_params):
    """Forward request to Z.AI with streaming, passthrough SSE directly."""
    if not zai_api_key:
        err, done = yield_sse_error(model, "[Z.AI Error: API key not configured]")
        yield err; yield done
        return

    api_model = model[len("zai:"):]
    endpoint = f"{ZAI_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {zai_api_key}", "Content-Type": "application/json"}
    strip = not _zai_supports_vision(api_model)

    # Omit max_tokens for streaming — reasoning models (GLM) exhaust it on chain-of-thought
    payload = {"model": api_model, "messages": build_oai_messages(system_prompt, messages, strip_vision=strip), "stream": True}
    payload.update({k: v for k, v in extra_params.items() if v is not None})

    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] -> Z.AI ({api_model}, {len(messages)} msgs, stream)")
    start = time.time()

    session = third_party_session or auth.session or create_session()
    owns_session = session is not third_party_session and session is not auth.session
    try:
        # Z.AI sometimes returns a 200 with an empty streaming body; retry a couple times.
        for attempt in range(3):
            async with session.post(endpoint, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    error_body = await resp.text()
                    logger.error(f"[{request_id}] Z.AI {resp.status}: {error_body[:300]}")
                    err, done = yield_sse_error(model, f"[Z.AI Error {resp.status}]")
                    yield err; yield done
                    return

                headers_obj = getattr(resp, "headers", None)
                content_type_val = ""
                if headers_obj is not None:
                    try:
                        content_type_val = headers_obj.get("Content-Type") or ""
                        if asyncio.iscoroutine(content_type_val):
                            content_type_val = await content_type_val
                    except Exception:
                        content_type_val = ""
                content_type = str(content_type_val).lower()

                # Fallback: if upstream responds with non-streaming JSON despite stream=True.
                if "application/json" in content_type:
                    data = await resp.json()
                    content = _extract_oai_content(
                        data.get("choices", [{}])[0]
                        .get("message", {})
                    )
                    if not content:
                        err, done = yield_sse_error(model, "[Z.AI Error: empty JSON response]")
                        yield err; yield done
                        return
                    yield make_sse_content_chunk(model, content)
                    yield "data: [DONE]\n\n"
                    return

                yielded_any = False
                saw_done = False
                total_content = 0
                async for event in passthrough_sse(resp, request_id, "Z.AI", start):
                    yielded_any = True
                    if event.strip().startswith("data: [DONE]"):
                        saw_done = True
                    elif event.startswith("data: ") and '"content"' in event:
                        try:
                            d = json.loads(event[6:])
                            c = d.get("choices", [{}])[0].get("delta", {}).get("content") or ""
                            total_content += len(c)
                        except Exception:
                            pass
                    yield event

                if not yielded_any or total_content == 0:
                    logger.warning(
                        f"[{request_id}] Z.AI returned empty stream body (attempt {attempt + 1}/3)"
                        if not yielded_any else
                        f"[{request_id}] Z.AI stream had no content (attempt {attempt + 1}/3)"
                    )
                    if attempt < 2:
                        await asyncio.sleep(2 ** attempt)
                        start = time.time()
                        continue

                    # Last resort: try a non-streaming request and convert to SSE.
                    try:
                        direct_payload = dict(payload)
                        direct_payload.pop("stream", None)
                        async with session.post(endpoint, json=direct_payload, headers=headers) as direct_resp:
                            if direct_resp.status != 200:
                                err, done = yield_sse_error(
                                    model,
                                    f"[Z.AI Error {direct_resp.status}: empty stream + direct fallback failed]",
                                )
                                yield err; yield done
                                return
                            data = await direct_resp.json()
                            content = _extract_oai_content(
                                data.get("choices", [{}])[0]
                                .get("message", {})
                            )
                            if content:
                                yield make_sse_content_chunk(model, content)
                                yield "data: [DONE]\n\n"
                                return
                            else:
                                logger.warning(f"[{request_id}] Z.AI direct fallback also returned empty content")
                    except Exception as e:
                        logger.error(f"[{request_id}] Z.AI direct fallback failed after empty stream: {e}")

                    err, done = yield_sse_error(model, "[Z.AI Error: empty stream response]")
                    yield err; yield done
                    return

                if not saw_done:
                    yield "data: [DONE]\n\n"
                return
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
        logger.error(f"[{request_id}] Z.AI streaming error: {e}")
        err, done = yield_sse_error(model, f"[Z.AI Error: {e}]")
        yield err; yield done
    finally:
        if owns_session:
            await session.close()


# --- Xiaomi ---

XIAOMI_BASE_URL = "https://token-plan-sgp.xiaomimimo.com/v1"
XIAOMI_PLATFORM_URL = "https://api.xiaomimimo.com/v1"
# Models only available on the platform endpoint, not the SGP token-plan
_XIAOMI_PLATFORM_MODELS = {"mimo-v2-flash"}


async def call_xiaomi_direct(system_prompt: Optional[str], messages: list, model: str, max_tokens: int, **extra_params) -> str:
    """Forward request to Xiaomi (OpenAI-compatible), collect full response."""
    if not xiaomi_api_key:
        raise HTTPException(status_code=503, detail="Xiaomi API key not configured — add it via /config/xiaomi-key")
    api_model = model[len("xiaomi:"):]
    base = XIAOMI_PLATFORM_URL if api_model in _XIAOMI_PLATFORM_MODELS else XIAOMI_BASE_URL
    endpoint = f"{base}/chat/completions"
    headers = {"Authorization": f"Bearer {xiaomi_api_key}", "Content-Type": "application/json"}

    payload = {"model": api_model, "messages": build_oai_messages(system_prompt, messages), "max_tokens": max_tokens}
    payload.update({k: v for k, v in extra_params.items() if v is not None})

    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] -> Xiaomi ({api_model}, {len(messages)} msgs)")
    start = time.time()

    session = third_party_session or auth.session or create_session()
    owns_session = session is not third_party_session and session is not auth.session
    try:
        async with session.post(endpoint, json=payload, headers=headers) as resp:
            elapsed = time.time() - start
            if resp.status in (401, 403):
                raise HTTPException(status_code=401, detail="Xiaomi auth failed — check API key")
            if resp.status == 429:
                raise HTTPException(status_code=429, detail="Xiaomi rate limit exceeded")
            if resp.status != 200:
                error_body = await resp.text()
                logger.error(f"[{request_id}] Xiaomi {resp.status}: {error_body[:300]}")
                raise HTTPException(status_code=resp.status, detail=error_body[:200])
            data = await resp.json()
            choices = data.get("choices", [])
            if not choices:
                logger.warning(f"[{request_id}] Xiaomi returned empty choices array")
                raise HTTPException(status_code=500, detail="Xiaomi returned empty response")
            content = _extract_oai_content(choices[0].get("message", {}))
            if content is None:
                # Reasoning model may have exhausted max_tokens on chain-of-thought — retry without cap
                if _is_reasoning_truncated(data):
                    logger.info(f"[{request_id}] Xiaomi reasoning truncated, retrying without max_tokens")
                    retry_payload = dict(payload)
                    retry_payload.pop("max_tokens", None)
                    async with session.post(endpoint, json=retry_payload, headers=headers) as retry_resp:
                        if retry_resp.status == 200:
                            retry_data = await retry_resp.json()
                            content = _extract_oai_content(retry_data.get("choices", [{}])[0].get("message", {}))
                            if content:
                                logger.info(f"[{request_id}] <- {len(content)} chars ({time.time() - start:.1f}s, Xiaomi, retry)")
                                return content
                logger.warning(f"[{request_id}] Xiaomi returned null content, full response: {str(data)[:500]}")
                raise HTTPException(status_code=500, detail="Xiaomi returned empty content")
            logger.info(f"[{request_id}] <- {len(content)} chars ({elapsed:.1f}s, Xiaomi)")
            return content
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Xiaomi request timed out")
    finally:
        if owns_session:
            await session.close()


async def call_xiaomi_streaming(system_prompt: Optional[str], messages: list, model: str, max_tokens: int, **extra_params):
    """Forward request to Xiaomi with streaming, passthrough SSE directly."""
    if not xiaomi_api_key:
        err, done = yield_sse_error(model, "[Xiaomi Error: API key not configured]")
        yield err; yield done
        return

    api_model = model[len("xiaomi:"):]
    base = XIAOMI_PLATFORM_URL if api_model in _XIAOMI_PLATFORM_MODELS else XIAOMI_BASE_URL
    endpoint = f"{base}/chat/completions"
    headers = {"Authorization": f"Bearer {xiaomi_api_key}", "Content-Type": "application/json"}

    # Omit max_tokens for streaming — reasoning models exhaust it on chain-of-thought,
    # producing empty content.  NPC dialogue is naturally short.
    payload = {"model": api_model, "messages": build_oai_messages(system_prompt, messages), "stream": True}
    payload.update({k: v for k, v in extra_params.items() if v is not None})

    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] -> Xiaomi ({api_model}, {len(messages)} msgs, stream)")
    start = time.time()

    session = third_party_session or auth.session or create_session()
    owns_session = session is not third_party_session and session is not auth.session
    try:
        async with session.post(endpoint, json=payload, headers=headers) as resp:
            if resp.status != 200:
                error_body = await resp.text()
                logger.error(f"[{request_id}] Xiaomi {resp.status}: {error_body[:300]}")
                err, done = yield_sse_error(model, f"[Xiaomi Error {resp.status}]")
                yield err; yield done
                return

            async for event in passthrough_sse(resp, request_id, "Xiaomi", start):
                yield event
            yield "data: [DONE]\n\n"
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
        logger.error(f"[{request_id}] Xiaomi streaming error: {e}")
        err, done = yield_sse_error(model, f"[Xiaomi Error: {e}]")
        yield err; yield done
    finally:
        if owns_session:
            await session.close()


# --- OpenCode (Zen + Go APIs — OpenAI-compatible) ---

def _resolve_opencode(model: str):
    """Resolve OpenCode model prefix to (api_model, base_url, api_key, plan_name).

    Guards against double-prefixed models like 'opencode:xiaomi:mimo-v2-pro' —
    if the stripped model still contains another provider prefix, that's a config
    error on the SkyrimNet side.
    """
    if model.lower().startswith("opencode-go:"):
        return model[len("opencode-go:"):], OPENCODE_GO_URL, opencode_go_api_key or opencode_api_key, "Go"
    return model[len("opencode:"):], OPENCODE_ZEN_URL, opencode_api_key or opencode_go_api_key, "Zen"


async def call_opencode_direct(system_prompt: Optional[str], messages: list, model: str, max_tokens: int, **extra_params) -> str:
    """Forward request to OpenCode API (OpenAI-compatible), collect full response."""
    api_model, base_url, api_key, plan = _resolve_opencode(model)
    if not api_key:
        raise HTTPException(status_code=503, detail=f"OpenCode {plan} API key not configured — add it via /config/opencode-key")
    endpoint = f"{base_url}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "User-Agent": "opencode/1.3.10"}

    payload = {"model": api_model, "messages": build_oai_messages(system_prompt, messages), "max_tokens": max_tokens}
    payload.update({k: v for k, v in extra_params.items() if v is not None})

    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] -> OpenCode ({api_model}, {len(messages)} msgs)")
    start = time.time()

    session = third_party_session or auth.session or create_session()
    owns_session = session is not third_party_session and session is not auth.session
    try:
        async with session.post(endpoint, json=payload, headers=headers) as resp:
            elapsed = time.time() - start
            if resp.status in (401, 403):
                raise HTTPException(status_code=401, detail="OpenCode auth failed — check API key")
            if resp.status == 429:
                raise HTTPException(status_code=429, detail="OpenCode rate limit exceeded")
            if resp.status != 200:
                error_body = await resp.text()
                logger.error(f"[{request_id}] OpenCode {resp.status}: {error_body[:300]}")
                raise HTTPException(status_code=resp.status, detail=error_body[:200])
            data = await resp.json()
            choices = data.get("choices", [])
            if not choices:
                raise HTTPException(status_code=500, detail="OpenCode returned empty response")
            content = _extract_oai_content(choices[0].get("message", {}))
            if content is None:
                if _is_reasoning_truncated(data):
                    logger.info(f"[{request_id}] OpenCode reasoning truncated, retrying without max_tokens")
                    retry_payload = dict(payload)
                    retry_payload.pop("max_tokens", None)
                    async with session.post(endpoint, json=retry_payload, headers=headers) as retry_resp:
                        if retry_resp.status == 200:
                            retry_data = await retry_resp.json()
                            content = _extract_oai_content(retry_data.get("choices", [{}])[0].get("message", {}))
                            if content:
                                logger.info(f"[{request_id}] <- {len(content)} chars ({time.time() - start:.1f}s, OpenCode, retry)")
                                return content
                raise HTTPException(status_code=500, detail="OpenCode returned empty content")
            logger.info(f"[{request_id}] <- {len(content)} chars ({elapsed:.1f}s, OpenCode)")
            return content
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="OpenCode request timed out")
    finally:
        if owns_session:
            await session.close()


async def call_opencode_streaming(system_prompt: Optional[str], messages: list, model: str, max_tokens: int, **extra_params):
    """Forward request to OpenCode API with streaming, passthrough SSE directly."""
    api_model, base_url, api_key, plan = _resolve_opencode(model)
    if not api_key:
        err, done = yield_sse_error(model, f"[OpenCode {plan} Error: API key not configured]")
        yield err; yield done
        return

    endpoint = f"{base_url}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "User-Agent": "opencode/1.3.10"}

    # Omit max_tokens for streaming — reasoning models exhaust it on chain-of-thought
    payload = {"model": api_model, "messages": build_oai_messages(system_prompt, messages), "stream": True}
    payload.update({k: v for k, v in extra_params.items() if v is not None})

    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] -> OpenCode ({api_model}, {len(messages)} msgs, stream)")
    start = time.time()

    session = third_party_session or auth.session or create_session()
    owns_session = session is not third_party_session and session is not auth.session
    try:
        async with session.post(endpoint, json=payload, headers=headers) as resp:
            if resp.status != 200:
                error_body = await resp.text()
                logger.error(f"[{request_id}] OpenCode {resp.status}: {error_body[:300]}")
                err, done = yield_sse_error(model, f"[OpenCode Error {resp.status}]")
                yield err; yield done
                return

            async for event in passthrough_sse(resp, request_id, "OpenCode", start):
                yield event
            yield "data: [DONE]\n\n"
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
        logger.error(f"[{request_id}] OpenCode streaming error: {e}")
        err, done = yield_sse_error(model, f"[OpenCode Error: {e}]")
        yield err; yield done
    finally:
        if owns_session:
            await session.close()


# --- Qwen Code (OpenAI-compatible via portal.qwen.ai) ---

async def call_qwen_direct(system_prompt: Optional[str], messages: list, model: str, max_tokens: int, **extra_params) -> str:
    """Forward request to Qwen (OpenAI-compatible), collect full response."""
    # Re-read credentials file in case the CLI refreshed the token
    qwen_auth.reload_from_file()
    if not qwen_auth.is_ready:
        raise HTTPException(status_code=503, detail="Qwen auth not ready -- run Qwen Code CLI login first")

    api_model = model[len("qwen:"):]
    endpoint = f"{QWEN_BASE_URL}/chat/completions"
    headers = qwen_auth.get_auth_headers()

    payload = {"model": api_model, "messages": build_oai_messages(system_prompt, messages)}
    payload.update({k: v for k, v in extra_params.items() if v is not None})

    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] -> Qwen ({api_model}, {len(messages)} msgs)")
    start = time.time()

    session = qwen_auth.session or create_session()
    try:
        async with session.post(endpoint, json=payload, headers=headers) as resp:
            elapsed = time.time() - start
            if resp.status in (400, 401, 403):
                # Token may be stale — try reloading from file and retry once
                if qwen_auth.reload_from_file():
                    headers = qwen_auth.get_auth_headers()
                    async with session.post(endpoint, json=payload, headers=headers) as retry_resp:
                        if retry_resp.status == 200:
                            data = await retry_resp.json()
                            content = _extract_oai_content(data.get("choices", [{}])[0].get("message", {}))
                            if content:
                                logger.info(f"[{request_id}] <- {len(content)} chars ({elapsed:.1f}s, Qwen)")
                                return content
                error_body = await resp.text()
                raise HTTPException(status_code=resp.status, detail=f"Qwen error — {error_body[:200]}. Run Qwen Code CLI to refresh auth.")
            if resp.status == 429:
                raise HTTPException(status_code=429, detail="Qwen rate limit exceeded")
            if resp.status != 200:
                error_body = await resp.text()
                logger.error(f"[{request_id}] Qwen {resp.status}: {error_body[:300]}")
                raise HTTPException(status_code=resp.status, detail=error_body[:200])
            data = await resp.json()
            choices = data.get("choices", [])
            if not choices:
                raise HTTPException(status_code=500, detail="Qwen returned empty response")
            content = _extract_oai_content(choices[0].get("message", {}))
            if content is None:
                raise HTTPException(status_code=500, detail="Qwen returned empty content")
            logger.info(f"[{request_id}] <- {len(content)} chars ({elapsed:.1f}s, Qwen)")
            return content
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Qwen request timed out")
    finally:
        if not qwen_auth.session:
            await session.close()


async def call_qwen_streaming(system_prompt: Optional[str], messages: list, model: str, max_tokens: int, **extra_params):
    """Forward request to Qwen with streaming, passthrough SSE directly."""
    qwen_auth.reload_from_file()
    if not qwen_auth.is_ready:
        err, done = yield_sse_error(model, "[Qwen Error: auth not ready — run Qwen Code CLI login]")
        yield err; yield done
        return

    api_model = model[len("qwen:"):]
    endpoint = f"{QWEN_BASE_URL}/chat/completions"
    headers = qwen_auth.get_auth_headers()

    payload = {"model": api_model, "messages": build_oai_messages(system_prompt, messages), "stream": True}
    payload.update({k: v for k, v in extra_params.items() if v is not None})

    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] -> Qwen ({api_model}, {len(messages)} msgs, stream)")
    start = time.time()

    session = qwen_auth.session or create_session()
    owns_session = not qwen_auth.session
    try:
        async with session.post(endpoint, json=payload, headers=headers) as resp:
            if resp.status in (400, 401, 403):
                # Token may be stale — reload from file and retry
                if qwen_auth.reload_from_file():
                    headers = qwen_auth.get_auth_headers()
                    async with session.post(endpoint, json=payload, headers=headers) as retry_resp:
                        if retry_resp.status == 200:
                            async for event in passthrough_sse(retry_resp, request_id, "Qwen", start):
                                yield event
                            yield "data: [DONE]\n\n"
                            return
                err, done = yield_sse_error(model, "[Qwen Error: auth failed]")
                yield err; yield done
                return

            if resp.status != 200:
                error_body = await resp.text()
                logger.error(f"[{request_id}] Qwen {resp.status}: {error_body[:300]}")
                err, done = yield_sse_error(model, f"[Qwen Error {resp.status}]")
                yield err; yield done
                return

            async for event in passthrough_sse(resp, request_id, "Qwen", start):
                yield event
            yield "data: [DONE]\n\n"
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
        logger.error(f"[{request_id}] Qwen streaming error: {e}")
        err, done = yield_sse_error(model, f"[Qwen Error: {e}]")
        yield err; yield done
    finally:
        if owns_session:
            await session.close()


# --- Fireworks (OpenAI-compatible via api.fireworks.ai) ---

FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"

# Short name → full Fireworks API model path.  Allows users to send e.g.
# "fireworks:kimi-k2p5-turbo" instead of the full "fireworks:accounts/fireworks/routers/kimi-k2p5-turbo".
_FIREWORKS_MODEL_ALIASES: dict[str, str] = {
    "kimi-k2p5-turbo": "accounts/fireworks/routers/kimi-k2p5-turbo",
}


_FIREWORKS_UNSUPPORTED_PARAMS = {"reasoning", "top_k", "provider"}


def _resolve_fireworks_model(model: str) -> str:
    """Strip the 'fireworks:' prefix and expand short aliases to full API paths."""
    raw = model[len("fireworks:"):]
    return _FIREWORKS_MODEL_ALIASES.get(raw, raw)


def _fireworks_payload_fixup(payload: dict, extra_params: dict) -> None:
    """Apply Fireworks-specific payload adjustments.

    - Strips unsupported params (reasoning, top_k, provider).
    - When the caller sends reasoning.enabled=false, disables thinking via the
      Anthropic-compatible thinking param to prevent chain-of-thought leaking
      into the content stream.
    """
    payload.update({k: v for k, v in extra_params.items()
                    if v is not None and k not in _FIREWORKS_UNSUPPORTED_PARAMS})
    # Translate OpenAI-style reasoning param to Fireworks thinking param
    reasoning = extra_params.get("reasoning")
    if isinstance(reasoning, dict) and not reasoning.get("enabled", True):
        payload["thinking"] = {"type": "disabled"}


async def _call_fireworks_direct_via_streaming(system_prompt: Optional[str], messages: list, model: str, max_tokens: int, **extra_params) -> str:
    """Collect streaming response from Fireworks for large max_tokens requests."""
    content_parts = []
    async for event in call_fireworks_streaming(system_prompt, messages, model, max_tokens, **extra_params):
        if event.startswith("data: ") and not event.startswith("data: [DONE]"):
            try:
                data = json.loads(event[6:])
                delta = data.get("choices", [{}])[0].get("delta", {})
                if delta.get("content"):
                    content_parts.append(delta["content"])
            except (json.JSONDecodeError, IndexError, KeyError):
                pass
    return "".join(content_parts)


async def call_fireworks_direct(system_prompt: Optional[str], messages: list, model: str, max_tokens: int, **extra_params) -> str:
    """Forward request to Fireworks (OpenAI-compatible), collect full response.

    Fireworks requires stream=true when max_tokens > 4096, so we use streaming
    internally and collect the response in that case.
    """
    if not fireworks_api_key:
        raise HTTPException(status_code=503, detail="Fireworks API key not configured — add it via /config/fireworks-key")

    # Fireworks requires streaming for max_tokens > 4096
    if max_tokens > 4096:
        return await _call_fireworks_direct_via_streaming(system_prompt, messages, model, max_tokens, **extra_params)

    api_model = _resolve_fireworks_model(model)
    endpoint = f"{FIREWORKS_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {fireworks_api_key}", "Content-Type": "application/json"}

    payload = {"model": api_model, "messages": build_oai_messages(system_prompt, messages), "max_tokens": max_tokens}
    _fireworks_payload_fixup(payload, extra_params)

    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] -> Fireworks ({api_model}, {len(messages)} msgs)")
    start = time.time()

    session = third_party_session or auth.session or create_session()
    owns_session = session is not third_party_session and session is not auth.session
    try:
        async with session.post(endpoint, json=payload, headers=headers) as resp:
            elapsed = time.time() - start
            if resp.status in (401, 403):
                raise HTTPException(status_code=401, detail="Fireworks auth failed — check API key")
            if resp.status == 429:
                raise HTTPException(status_code=429, detail="Fireworks rate limit exceeded")
            if resp.status != 200:
                error_body = await resp.text()
                logger.error(f"[{request_id}] Fireworks {resp.status}: {error_body[:300]}")
                raise HTTPException(status_code=resp.status, detail=error_body[:200])
            data = await resp.json()
            choices = data.get("choices", [])
            if not choices:
                logger.warning(f"[{request_id}] Fireworks returned empty choices array")
                raise HTTPException(status_code=500, detail="Fireworks returned empty response")
            content = _extract_oai_content(choices[0].get("message", {}))
            if content is None:
                if _is_reasoning_truncated(data):
                    logger.info(f"[{request_id}] Fireworks reasoning truncated, retrying without max_tokens")
                    retry_payload = dict(payload)
                    retry_payload.pop("max_tokens", None)
                    async with session.post(endpoint, json=retry_payload, headers=headers) as retry_resp:
                        if retry_resp.status == 200:
                            retry_data = await retry_resp.json()
                            content = _extract_oai_content(retry_data.get("choices", [{}])[0].get("message", {}))
                            if content:
                                logger.info(f"[{request_id}] <- {len(content)} chars ({time.time() - start:.1f}s, Fireworks, retry)")
                                return content
                logger.warning(f"[{request_id}] Fireworks returned null content, full response: {str(data)[:500]}")
                raise HTTPException(status_code=500, detail="Fireworks returned empty content")
            logger.info(f"[{request_id}] <- {len(content)} chars ({elapsed:.1f}s, Fireworks)")
            return content
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Fireworks request timed out")
    finally:
        if owns_session:
            await session.close()


async def call_fireworks_streaming(system_prompt: Optional[str], messages: list, model: str, max_tokens: int, **extra_params):
    """Forward request to Fireworks with streaming, passthrough SSE directly."""
    if not fireworks_api_key:
        err, done = yield_sse_error(model, "[Fireworks Error: API key not configured]")
        yield err; yield done
        return

    api_model = _resolve_fireworks_model(model)
    endpoint = f"{FIREWORKS_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {fireworks_api_key}", "Content-Type": "application/json"}

    payload = {"model": api_model, "messages": build_oai_messages(system_prompt, messages), "stream": True}
    _fireworks_payload_fixup(payload, extra_params)

    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] -> Fireworks ({api_model}, {len(messages)} msgs, stream)")
    start = time.time()

    session = third_party_session or auth.session or create_session()
    owns_session = session is not third_party_session and session is not auth.session
    try:
        async with session.post(endpoint, json=payload, headers=headers) as resp:
            if resp.status != 200:
                error_body = await resp.text()
                logger.error(f"[{request_id}] Fireworks {resp.status}: {error_body[:300]}")
                err, done = yield_sse_error(model, f"[Fireworks Error {resp.status}]")
                yield err; yield done
                return

            saw_done = False
            async for event in passthrough_sse(resp, request_id, "Fireworks", start):
                if event.strip().startswith("data: [DONE]"):
                    saw_done = True
                yield event

            if not saw_done:
                yield "data: [DONE]\n\n"
            return
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
        logger.error(f"[{request_id}] Fireworks streaming error: {e}")
        err, done = yield_sse_error(model, f"[Fireworks Error: {e}]")
        yield err; yield done
    finally:
        if owns_session:
            await session.close()


# --- Antigravity Auth Loading ---

ANTIGRAVITY_AUTH_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "antigravity-auth.json")


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
    global antigravity_auth

    if not os.path.exists(ANTIGRAVITY_AUTH_FILE):
        logger.info("No Antigravity auth file found -- visit /config/antigravity-login")
        return

    try:
        with open(ANTIGRAVITY_AUTH_FILE, "r", encoding="utf-8") as f:
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
            decrypted = _decrypt_auth_account(data)
            for field in _ENCRYPTED_AUTH_FIELDS:
                val = data.get(field, "")
                if isinstance(val, str) and val and not val.startswith(_ENC_PREFIX):
                    migrated = True
            account = AntigravityAccount.from_dict(decrypted)
            if account.refresh_token:
                if await account.refresh_if_needed():
                    account.session = create_session()
                    antigravity_auth._legacy_account = account
                    logger.info(f"Antigravity auth loaded for {account.email} (legacy mode)")
                else:
                    antigravity_auth._legacy_account = account
                    logger.warning("Antigravity token refresh failed -- re-login required")
        if migrated:
            _save_antigravity_auth()
            logger.info("Migrated antigravity-auth.json tokens to encrypted storage")
    except Exception as e:
        logger.error(f"Failed to load Antigravity auth: {e}")


def _save_antigravity_auth():
    """Save Antigravity auth to cached file in multi-account format, encrypting tokens."""
    accounts_data = []

    # Include legacy account if it has a refresh token
    if antigravity_auth._legacy_account.refresh_token:
        accounts_data.append(_encrypt_auth_account(antigravity_auth._legacy_account.to_dict()))

    # Include all multi-account accounts
    for account in antigravity_auth.accounts:
        # Avoid duplicates with legacy account
        if account.email != antigravity_auth._legacy_account.email:
            accounts_data.append(_encrypt_auth_account(account.to_dict()))

    # If only one account, save in legacy format for backward compatibility
    if len(accounts_data) == 1:
        with open(ANTIGRAVITY_AUTH_FILE, "w", encoding="utf-8") as f:
            json.dump(accounts_data[0], f, indent=2)
    else:
        data = {"accounts": accounts_data}
        with open(ANTIGRAVITY_AUTH_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


# --- OpenCode Auth Loading ---

def load_opencode_key():
    """Load OpenCode API keys from ~/.local/share/opencode/auth.json if not already configured.

    Zen and Go plans share the same API key, so a single key is sufficient for both.
    If only one key is available (from config.json or auth.json), it is used for both plans.
    """
    global opencode_api_key, opencode_go_api_key

    if not os.path.exists(OPENCODE_AUTH_FILE):
        if not opencode_api_key and not opencode_go_api_key:
            logger.info("No OpenCode auth file found -- set up OpenCode first or add key via /config/opencode-key")
        # Even without auth file, share whichever key we already have from config.json
        if opencode_api_key and not opencode_go_api_key:
            opencode_go_api_key = opencode_api_key
            logger.info("OpenCode Go using shared Zen API key from config")
        elif opencode_go_api_key and not opencode_api_key:
            opencode_api_key = opencode_go_api_key
            logger.info("OpenCode Zen using shared Go API key from config")
        return

    try:
        with open(OPENCODE_AUTH_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Load Zen key
        if not opencode_api_key:
            oc_entry = data.get("opencode", {})
            if oc_entry.get("type") == "api" and oc_entry.get("key"):
                opencode_api_key = oc_entry["key"]
                logger.info("OpenCode Zen API key loaded from auth file")

        # Load Go key
        if not opencode_go_api_key:
            go_entry = data.get("opencode-go", {})
            if go_entry.get("type") == "api" and go_entry.get("key"):
                opencode_go_api_key = go_entry["key"]
                logger.info("OpenCode Go API key loaded from auth file")

        # Zen and Go share the same key — fill in whichever is missing
        if opencode_api_key and not opencode_go_api_key:
            opencode_go_api_key = opencode_api_key
            logger.info("OpenCode Go using shared Zen API key")
        elif opencode_go_api_key and not opencode_api_key:
            opencode_api_key = opencode_go_api_key
            logger.info("OpenCode Zen using shared Go API key")
    except Exception as e:
        logger.error(f"Failed to load OpenCode auth: {e}")


# --- Qwen Code Auth Loading ---

async def load_qwen_auth():
    """Load Qwen auth from ~/.qwen/oauth_creds.json (written by 'qwen code' login).

    The Qwen CLI manages token lifecycle via device flow OAuth.  We just read
    whatever token is on disk — the CLI writes it after each successful auth.
    On each request we also re-read the file to pick up refreshed tokens.
    """
    if not os.path.exists(QWEN_CREDS_FILE):
        logger.info(f"No Qwen credentials file at {QWEN_CREDS_FILE} -- run Qwen Code CLI login")
        return

    if qwen_auth.reload_from_file():
        qwen_auth.session = create_session()
        logger.info("Qwen auth loaded from credentials file")
    else:
        logger.warning("Qwen credentials file found but no valid access_token")


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


def _get_gemini_model_id(model: str) -> str:
    """Map gcli- proxy model name to the Code Assist API model ID."""
    m = model.lower()
    if m.startswith("gcli-"):
        m = m[5:]  # strip prefix

    # Map to valid Code Assist model IDs
    _GCLI_MODEL_MAP = {
        "gemini-2.5-pro": "gemini-2.5-pro",
        "gemini-2.5-pro-preview-05-06": "gemini-2.5-pro",
        "gemini-2.5-flash": "gemini-2.5-flash",
        "gemini-2.5-flash-preview-04-17": "gemini-2.5-flash",
        "gemini-2.5-flash-lite": "gemini-2.5-flash-lite",
        "gemini-3-pro-preview": "gemini-3-pro-preview",
        "gemini-3.1-pro-preview": "gemini-3.1-pro-preview",
        "gemini-3-flash-preview": "gemini-3-flash-preview",
        "gemini-3.1-flash-lite-preview": "gemini-3.1-flash-lite-preview",
        # Legacy models - best-effort fallback
        "gemini-2.0-flash": "gemini-2.5-flash",
        "gemini-2.0-flash-exp": "gemini-2.5-flash",
        "gemini-1.5-pro": "gemini-2.5-pro",
        "gemini-1.5-flash": "gemini-2.5-flash",
    }
    return _GCLI_MODEL_MAP.get(m, m)


async def call_gemini_direct(
    system_prompt: Optional[str],
    messages: list,
    model: str,
    max_tokens: int,
) -> str:
    """Call Gemini Code Assist API (cloudcode-pa.googleapis.com) with stored OAuth token."""
    if not gemini_auth.is_ready or gemini_auth.is_expired():
        if not await gemini_auth.refresh_if_needed():
            raise HTTPException(status_code=503, detail="Gemini auth not ready -- run 'gemini auth login'")

    gemini_model = _get_gemini_model_id(model)
    request_contents = _convert_messages_to_antigravity(messages, system_prompt)

    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] -> Gemini CLI {gemini_model} ({len(messages)} msgs)")
    start = time.time()

    payload = {
        "model": gemini_model,
        "request": {
            **request_contents,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": 0.7,
            },
        },
    }
    if gemini_auth.project_id:
        payload["project"] = gemini_auth.project_id

    url = f"{GEMINI_CODE_ASSIST_ENDPOINT}/{GEMINI_CODE_ASSIST_API_VERSION}:generateContent"
    headers = gemini_auth.get_auth_headers()
    session = gemini_auth.session or create_session()
    owns_session = not gemini_auth.session

    async def _do_post():
        return await session.post(url, json=payload, headers=headers)

    try:
        async with await _do_post() as resp:
            if resp.status == 401:
                if await gemini_auth.refresh_if_needed():
                    headers.update(gemini_auth.get_auth_headers())
                    async with await _do_post() as retry_resp:
                        if retry_resp.status != 200:
                            error_body = await retry_resp.text()
                            raise HTTPException(status_code=retry_resp.status, detail=f"Gemini API error: {error_body[:300]}")
                        data = await retry_resp.json()
                else:
                    raise HTTPException(status_code=401, detail="Gemini auth expired and refresh failed")
            elif resp.status != 200:
                error_body = await resp.text()
                logger.error(f"[{request_id}] Gemini {resp.status}: {error_body[:300]}")
                raise HTTPException(status_code=resp.status, detail=f"Gemini API error: {error_body[:300]}")
            else:
                data = await resp.json()

        # Code Assist wraps the response: {response: {candidates: [...]}}
        inner = data.get("response", data)
        elapsed = time.time() - start
        text = _extract_antigravity_text(inner)
        logger.info(f"[{request_id}] <- {len(text)} chars in {elapsed:.2f}s")
        return text
    finally:
        if owns_session:
            await session.close()


async def call_gemini_streaming(
    system_prompt: Optional[str],
    messages: list,
    model: str,
    max_tokens: int,
):
    """Call Gemini API with SSE streaming."""
    if not gemini_auth.is_ready or gemini_auth.is_expired():
        if not await gemini_auth.refresh_if_needed():
            yield 'data: {"error": "Gemini auth not ready -- run gemini auth login"}\n\n'
            yield "data: [DONE]\n\n"
            return

    gemini_model = _get_gemini_model_id(model)
    request_contents = _convert_messages_to_antigravity(messages, system_prompt)

    request_id = uuid.uuid4().hex[:8]
    cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    created = int(time.time())
    logger.info(f"[{request_id}] -> Gemini CLI {gemini_model} ({len(messages)} msgs, stream)")
    start = time.time()
    total_chars = 0

    role_chunk = {
        "id": cmpl_id, "object": "chat.completion.chunk", "created": created, "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(role_chunk)}\n\n"

    payload = {
        "model": gemini_model,
        "request": {
            **request_contents,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": 0.7,
            },
        },
    }
    if gemini_auth.project_id:
        payload["project"] = gemini_auth.project_id

    url = f"{GEMINI_CODE_ASSIST_ENDPOINT}/{GEMINI_CODE_ASSIST_API_VERSION}:streamGenerateContent?alt=sse"
    headers = gemini_auth.get_auth_headers()
    headers["Accept"] = "text/event-stream"
    session = gemini_auth.session or create_session()
    owns_session = not gemini_auth.session

    try:
        async with session.post(url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                error_body = await resp.text()
                logger.error(f"[{request_id}] Gemini stream {resp.status}: {error_body[:300]}")
                err_chunk = {
                    "id": cmpl_id, "object": "chat.completion.chunk", "created": created, "model": model,
                    "choices": [{"index": 0, "delta": {"content": f"[Gemini Error {resp.status}]"}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(err_chunk)}\n\n"
                yield "data: [DONE]\n\n"
                return

            buffer = ""
            async for raw_chunk in resp.content.iter_any():
                buffer += raw_chunk.decode("utf-8", errors="replace")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if not data_str:
                        continue
                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    # Code Assist wraps response: {response: {candidates: [...]}}
                    inner = event.get("response", event)
                    candidates = inner.get("candidates", [])
                    for candidate in candidates:
                        content = candidate.get("content", {})
                        parts = content.get("parts", [])
                        for part in parts:
                            if part.get("thought") or part.get("thoughtSignature"):
                                continue
                            text = part.get("text", "")
                            if text:
                                total_chars += len(text)
                                oai_chunk = {
                                    "id": cmpl_id, "object": "chat.completion.chunk",
                                    "created": created, "model": model,
                                    "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                                }
                                yield f"data: {json.dumps(oai_chunk)}\n\n"

            # Flush remaining buffer
            if buffer.strip():
                line = buffer.strip()
                if line.startswith("data:"):
                    data_str = line[5:].strip()
                    if data_str:
                        try:
                            event = json.loads(data_str)
                            inner = event.get("response", event)
                            for candidate in inner.get("candidates", []):
                                for part in candidate.get("content", {}).get("parts", []):
                                    if part.get("thought") or part.get("thoughtSignature"):
                                        continue
                                    text = part.get("text", "")
                                    if text:
                                        total_chars += len(text)
                                        oai_chunk = {
                                            "id": cmpl_id, "object": "chat.completion.chunk",
                                            "created": created, "model": model,
                                            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                                        }
                                        yield f"data: {json.dumps(oai_chunk)}\n\n"
                        except json.JSONDecodeError:
                            pass

            elapsed = time.time() - start
            logger.info(f"[{request_id}] <- {total_chars} chars streamed in {elapsed:.2f}s")

    except Exception as e:
        logger.error(f"[{request_id}] Gemini streaming error: {e}")
        err_chunk = {
            "id": cmpl_id, "object": "chat.completion.chunk", "created": created, "model": model,
            "choices": [{"index": 0, "delta": {"content": f"[Gemini streaming error: {e}]"}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(err_chunk)}\n\n"
    finally:
        if owns_session:
            await session.close()

    stop_chunk = {
        "id": cmpl_id, "object": "chat.completion.chunk", "created": created, "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(stop_chunk)}\n\n"
    yield "data: [DONE]\n\n"


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
                await resp.release()
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
                got_finish = False
                async for raw_chunk in resp.content.iter_any():
                    buffer += raw_chunk.decode("utf-8", errors="replace")
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
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

                            if candidate.get("finishReason"):
                                got_finish = True

                # Flush remaining buffer
                if buffer.strip():
                    line = buffer.strip()
                    if line.startswith("data:"):
                        data_str = line[5:].strip()
                        if data_str:
                            try:
                                event = json.loads(data_str)
                                response = event.get("response", event)
                                for candidate in response.get("candidates", []):
                                    for part in candidate.get("content", {}).get("parts", []):
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
                            except json.JSONDecodeError:
                                pass

                # Single stop chunk + DONE
                stop_chunk = {
                    "id": cmpl_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                }
                yield f"data: {json.dumps(stop_chunk)}\n\n"
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

class VisionContent(BaseModel):
    """Individual content item for vision messages (text or image_url)."""
    type: str
    text: Optional[str] = None
    image_url: Optional[dict] = None


class ChatMessage(BaseModel):
    role: str
    content: Union[str, list[VisionContent]]


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
    global ollama_session, third_party_session
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

    # Persistent Ollama session — shared across all Ollama requests
    ollama_session = create_session()
    third_party_session = create_session()

    # Load Antigravity auth from cached file
    await load_antigravity_auth()
    # Load Gemini CLI auth from credentials file
    await load_gemini_auth()
    # Load OpenCode API key from auth file (if not in config.json)
    load_opencode_key()
    # Load Qwen Code auth from credentials file
    await load_qwen_auth()
    # Start background auth refresh loop for Claude
    global _claude_refresh_task
    if auth.is_ready and CLAUDE_PATH:
        _claude_refresh_task = asyncio.create_task(_claude_auth_refresh_loop())
        logger.info("Claude auth auto-refresh enabled (every 45 min)")

    # Start background stats persistence loop (flushes to stats.json every 60s)
    global _stats_persist_task
    _stats_persist_task = asyncio.create_task(_stats_persist_loop())
    logger.info(f"Stats persistence enabled (flushing to {STATS_FILE} every {int(_stats_persist_interval_s)}s)")

    # Freeze all module-level objects so the GC skips scanning them on every cycle.
    gc.freeze()
    yield
    # Cancel background refresh task
    if _claude_refresh_task and not _claude_refresh_task.done():
        _claude_refresh_task.cancel()
    # Cancel stats persist loop and do a final flush before shutdown
    if _stats_persist_task and not _stats_persist_task.done():
        _stats_persist_task.cancel()
    try:
        await asyncio.to_thread(_save_stats_to_disk)
        logger.info("Final stats flush complete")
    except Exception as e:
        logger.warning(f"Final stats flush failed: {e}")
    if auth.session:
        await auth.session.close()
    if ollama_session:
        await ollama_session.close()
    if third_party_session:
        await third_party_session.close()
    if codex_auth.session:
        await codex_auth.session.close()
    if gemini_auth.session:
        await gemini_auth.session.close()
    if qwen_auth.session:
        await qwen_auth.session.close()
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

_STREAMING_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


@app.post("/v1/chat/completions")
@app.post("/api/chat/completions")
@app.post("/chat/completions")
async def chat_completions(req: ChatRequest):
    try:
        return await _chat_completions_inner(req)
    except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError,
            aiohttp.ClientOSError, aiohttp.ServerDisconnectedError):
        logger.debug("Client disconnected mid-request — suppressing traceback")
        raise HTTPException(status_code=499, detail="Client disconnected")


async def _chat_completions_inner(req: ChatRequest):
    # Parse model list and pick via round-robin
    model_field = req.model or DEFAULT_MODEL
    models = parse_model_list(model_field)
    model = normalize_model_name(pick_model_round_robin(models) if len(models) > 1 else models[0])
    use_ollama = is_ollama_model(model)
    use_openrouter = not use_ollama and is_openrouter_model(model)
    use_codex = not use_ollama and is_codex_model(model)
    use_antigravity = not use_ollama and is_antigravity_model(model)
    use_gemini_cli = not use_ollama and is_gemini_cli_model(model)
    use_zai = not use_ollama and is_zai_model(model)
    use_xiaomi = not use_ollama and is_xiaomi_model(model)
    use_opencode = not use_ollama and is_opencode_model(model)
    use_qwen = not use_ollama and is_qwen_model(model)
    use_fireworks = not use_ollama and is_fireworks_model(model)
    use_nvidia = not use_ollama and is_nvidia_model(model)

    if use_ollama:
        pass  # No auth check — local needs none, cloud key checked inside call function
    elif use_openrouter:
        pass  # No auth check needed for OpenRouter
    elif use_zai:
        if not zai_api_key:
            raise HTTPException(status_code=503, detail="Z.AI API key not configured — add it via /config/zai-key")
    elif use_xiaomi:
        if not xiaomi_api_key:
            raise HTTPException(status_code=503, detail="Xiaomi API key not configured — add it via /config/xiaomi-key")
    elif use_opencode:
        _, _, resolved_key, plan = _resolve_opencode(model)
        if not resolved_key:
            raise HTTPException(status_code=503, detail=f"OpenCode {plan} API key not configured — add it via /config/opencode-key")
    elif use_fireworks:
        if not fireworks_api_key:
            raise HTTPException(status_code=503, detail="Fireworks API key not configured — add it via /config/fireworks-key")
    elif use_nvidia:
        if not nvidia_api_key:
            raise HTTPException(status_code=503, detail="NVIDIA NIM API key not configured — add it via /config/nvidia-key")
    elif use_qwen:
        if not qwen_auth.is_ready:
            raise HTTPException(status_code=503, detail="Qwen auth not ready -- run Qwen Code CLI login first")
    elif use_codex:
        if not codex_auth.is_ready:
            raise HTTPException(status_code=503, detail="Codex auth not ready -- ensure you have run 'codex login'")
    elif use_antigravity:
        if not antigravity_auth.is_ready:
            raise HTTPException(status_code=503, detail="Antigravity auth not ready -- visit /config/antigravity-login")
    elif use_gemini_cli:
        if not gemini_auth.is_ready:
            raise HTTPException(status_code=503, detail="Gemini auth not ready -- run 'gemini auth login'")
    else:
        if not auth.is_ready:
            raise HTTPException(status_code=503, detail="Claude auth not ready -- warming up")

    system_prompt, merged, oai_messages = _normalize_chat_messages(req.messages)

    if not merged:
        raise HTTPException(status_code=400, detail="No user message provided")

    max_tokens = req.max_tokens or 4096

    # Whitelist extra params — only forward standard OpenAI sampling parameters.
    # SkyrimNet sends provider-specific fields (provider, reasoning, etc.) that
    # cause 401/400 errors on providers that don't understand them.
    extra_params = {k: v for k, v in (req.model_extra or {}).items()
                    if v is not None and k in _CHAT_ALLOWED_EXTRA}

    def _stream(gen):
        """Wrap a streaming generator with timeout routing if enabled."""
        if timeout_routing_enabled:
            return _with_timeout_routing(gen, system_prompt, merged, oai_messages, model, max_tokens)
        return gen

    async def _tracked_stream(gen):
        """Wrap a streaming generator to record stats."""
        start = time.time()
        had_error = True  # assume error until we see a clean finish
        try:
            async for chunk in gen:
                yield chunk
                if isinstance(chunk, str) and "[DONE]" in chunk:
                    had_error = False
        except Exception:
            raise
        finally:
            request_stats.record(model, "streaming", time.time() - start, not had_error)

    async def _call_direct(coro):
        """Await a direct call and record stats. Direct (non-streaming) calls are
        NOT bounded by timeout routing — summary/memory workflows legitimately take
        longer than the streaming TTS budget."""
        start = time.time()
        try:
            result = await coro
            request_stats.record(model, "direct", time.time() - start, True)
            return result
        except Exception:
            request_stats.record(model, "direct", time.time() - start, False)
            raise

    # Route to correct provider using the booleans already computed above.
    if use_ollama:
        direct_fn, stream_fn, use_extra = call_ollama_direct, call_ollama_streaming, True
    elif use_openrouter:
        direct_fn, stream_fn, use_extra = call_openrouter_direct, call_openrouter_streaming, True
    elif use_codex:
        direct_fn, stream_fn, use_extra = call_codex_direct, call_codex_streaming, False
    elif use_antigravity:
        direct_fn, stream_fn, use_extra = call_antigravity_direct, call_antigravity_streaming, False
    elif use_gemini_cli:
        direct_fn, stream_fn, use_extra = call_gemini_direct, call_gemini_streaming, False
    elif use_zai:
        direct_fn, stream_fn, use_extra = call_zai_direct, call_zai_streaming, True
    elif use_xiaomi:
        direct_fn, stream_fn, use_extra = call_xiaomi_direct, call_xiaomi_streaming, True
    elif use_opencode:
        direct_fn, stream_fn, use_extra = call_opencode_direct, call_opencode_streaming, True
    elif use_qwen:
        direct_fn, stream_fn, use_extra = call_qwen_direct, call_qwen_streaming, True
    elif use_fireworks:
        direct_fn, stream_fn, use_extra = call_fireworks_direct, call_fireworks_streaming, True
    elif use_nvidia:
        direct_fn, stream_fn, use_extra = call_nvidia_direct, call_nvidia_streaming, True
    else:
        direct_fn, stream_fn, use_extra = call_api_direct, call_api_streaming, False

    call_system_prompt = None if _model_uses_oai_messages(model) else system_prompt
    call_messages = oai_messages if _model_uses_oai_messages(model) else merged
    kwargs = extra_params if use_extra else {}
    if req.stream:
        return StreamingResponse(
            _tracked_stream(_stream(stream_fn(call_system_prompt, call_messages, model, max_tokens, **kwargs))),
            media_type="text/event-stream",
            headers=_STREAMING_HEADERS,
        )
    response = await _call_direct(direct_fn(call_system_prompt, call_messages, model, max_tokens, **kwargs))

    if not response:
        raise HTTPException(status_code=500, detail="Empty response")

    prompt_chars = sum(len(msg.content) for msg in req.messages)
    prompt_chars += max(0, len(req.messages) - 1)
    prompt_tokens = prompt_chars // 4
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


@app.post("/config/fireworks-key")
async def set_fireworks_key(request: Request):
    global fireworks_api_key
    data = await request.json()
    key = data.get("key", "").strip()
    cfg = _load_config()
    if not key:
        fireworks_api_key = None
        cfg.pop("fireworks_api_key", None)
        _save_config(cfg)
        logger.info("Fireworks API key cleared")
        return {"status": "cleared"}
    fireworks_api_key = key
    cfg["fireworks_api_key"] = key
    _save_config(cfg)
    logger.info("Fireworks API key configured and saved to config.json")
    return {"status": "saved"}


@app.post("/config/nvidia-key")
async def set_nvidia_key(request: Request):
    global nvidia_api_key
    data = await request.json()
    key = data.get("key", "").strip()
    cfg = _load_config()
    if not key:
        nvidia_api_key = None
        cfg.pop("nvidia_api_key", None)
        _save_config(cfg)
        logger.info("NVIDIA NIM API key cleared")
        return {"status": "cleared"}
    nvidia_api_key = key
    cfg["nvidia_api_key"] = key
    _save_config(cfg)
    logger.info("NVIDIA NIM API key configured and saved to config.json")
    return {"status": "saved"}


@app.post("/config/zai-key")
async def set_zai_key(request: Request):
    global zai_api_key
    data = await request.json()
    key = data.get("key", "").strip()
    cfg = _load_config()
    if not key:
        zai_api_key = None
        cfg.pop("zai_api_key", None)
        _save_config(cfg)
        logger.info("Z.AI API key cleared")
        return {"status": "cleared"}
    zai_api_key = key
    cfg["zai_api_key"] = key
    _save_config(cfg)
    logger.info("Z.AI API key configured and saved to config.json")
    return {"status": "saved"}


@app.post("/config/xiaomi-key")
async def set_xiaomi_key(request: Request):
    global xiaomi_api_key
    data = await request.json()
    key = data.get("key", "").strip()
    cfg = _load_config()
    if not key:
        xiaomi_api_key = None
        cfg.pop("xiaomi_api_key", None)
        _save_config(cfg)
        logger.info("Xiaomi API key cleared")
        return {"status": "cleared"}
    xiaomi_api_key = key
    cfg["xiaomi_api_key"] = key
    _save_config(cfg)
    logger.info("Xiaomi API key configured and saved to config.json")
    return {"status": "saved"}


@app.post("/config/opencode-key")
async def set_opencode_key(request: Request):
    global opencode_api_key, opencode_go_api_key
    data = await request.json()
    key = data.get("key", "").strip()
    cfg = _load_config()
    if not key:
        opencode_api_key = None
        opencode_go_api_key = None
        cfg.pop("opencode_api_key", None)
        cfg.pop("opencode_go_api_key", None)
        _save_config(cfg)
        logger.info("OpenCode API key cleared (both plans)")
        return {"status": "cleared"}
    opencode_api_key = key
    opencode_go_api_key = key
    cfg["opencode_api_key"] = key
    _save_config(cfg)
    logger.info("OpenCode API key configured (shared across Zen + Go)")
    return {"status": "saved"}


@app.post("/config/opencode-go-key")
async def set_opencode_go_key(request: Request):
    global opencode_api_key, opencode_go_api_key
    data = await request.json()
    key = data.get("key", "").strip()
    cfg = _load_config()
    if not key:
        opencode_api_key = None
        opencode_go_api_key = None
        cfg.pop("opencode_api_key", None)
        cfg.pop("opencode_go_api_key", None)
        _save_config(cfg)
        logger.info("OpenCode API key cleared (both plans)")
        return {"status": "cleared"}
    opencode_api_key = key
    opencode_go_api_key = key
    cfg["opencode_api_key"] = key
    _save_config(cfg)
    logger.info("OpenCode API key configured (shared across Zen + Go)")
    return {"status": "saved"}


@app.post("/config/timeout-routing")
async def set_timeout_routing(request: Request):
    global timeout_routing_enabled
    data = await request.json()
    enabled = bool(data.get("enabled"))
    timeout_routing_enabled = enabled
    cfg = _load_config()
    cfg["timeout_routing_enabled"] = enabled
    _save_config(cfg)
    logger.info(f"Timeout routing {'enabled' if enabled else 'disabled'}")
    return {"status": "ok", "timeout_routing_enabled": enabled}


@app.post("/config/timeout-cutoff")
async def set_timeout_cutoff(request: Request):
    global timeout_cutoff_seconds
    data = await request.json()
    seconds = float(data.get("seconds", 6.0))
    if seconds < 1.0 or seconds > 30.0:
        raise HTTPException(status_code=400, detail="Timeout must be between 1 and 30 seconds")
    timeout_cutoff_seconds = seconds
    cfg = _load_config()
    cfg["timeout_cutoff_seconds"] = seconds
    _save_config(cfg)
    logger.info(f"TTFT cutoff set to {seconds}s")
    return {"status": "ok", "timeout_cutoff_seconds": seconds}


@app.post("/config/max-total")
async def set_max_total(request: Request):
    global max_total_seconds
    data = await request.json()
    seconds = float(data.get("seconds", 9.0))
    if seconds < 1.0 or seconds > 60.0:
        raise HTTPException(status_code=400, detail="Max total must be between 1 and 60 seconds")
    max_total_seconds = seconds
    cfg = _load_config()
    cfg["max_total_seconds"] = seconds
    _save_config(cfg)
    logger.info(f"Max total cutoff set to {seconds}s")
    return {"status": "ok", "max_total_seconds": seconds}


@app.get("/config/timeout-routing")
async def get_timeout_routing():
    return {
        "timeout_routing_enabled": timeout_routing_enabled,
        "timeout_cutoff_seconds": timeout_cutoff_seconds,
        "timeout_seconds": timeout_cutoff_seconds,  # legacy alias
        "max_total_seconds": max_total_seconds,
        "stats": model_stats.get_stats(),
    }


@app.get("/stats")
async def get_request_stats():
    """Full request stats: per-model, per-mode (streaming/direct), and global."""
    return request_stats.get_stats()


@app.get("/config/ollama-models")
async def get_ollama_models():
    """Run 'ollama list' and return installed model names."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "ollama", "list",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
        lines = stdout.decode(errors="replace").splitlines()
        models = []
        for line in lines[1:]:  # skip header row
            name = line.split()[0] if line.split() else ""
            if name:
                models.append(name)
        return {"models": models}
    except (FileNotFoundError, asyncio.TimeoutError, Exception):
        return {"models": []}


@app.get("/config/zai-models")
async def get_zai_models():
    """Fetch available models from Z.AI API."""
    if not zai_api_key:
        return {"models": []}
    zai_models = await _fetch_zai_models()
    # Strip the "zai:" prefix since the dashboard JS adds it
    models = [m["id"][len("zai:"):] for m in zai_models if m.get("id", "").startswith("zai:")]
    return {"models": models}


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


async def _fetch_claude_models() -> list[dict]:
    """Fetch available models from Anthropic API using captured auth headers."""
    if not auth.is_ready:
        return []
    session = auth.session or create_session()
    headers = {k: v for k, v in auth.headers.items() if k.lower() != "content-type"}
    try:
        async with session.get(
            "https://api.anthropic.com/v1/models",
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=5, connect=2),
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                return [
                    {"id": m["id"], "object": "model", "owned_by": "anthropic"}
                    for m in data.get("data", [])
                    if m.get("id")
                ]
            logger.debug(f"Anthropic /v1/models returned {resp.status}")
    except Exception as e:
        logger.debug(f"Claude model fetch failed: {e}")
    return []


async def _fetch_codex_models() -> list[dict]:
    """Fetch available models from OpenAI API using captured Codex OAuth token."""
    if not codex_auth.is_ready:
        return []
    await codex_auth.refresh_if_needed()
    if not codex_auth.is_ready:
        return []
    session = codex_auth.session or create_session()
    headers = codex_auth.get_auth_headers()
    try:
        async with session.get(
            "https://api.openai.com/v1/models",
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=5, connect=2),
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                return [
                    {"id": m["id"], "object": "model", "owned_by": "openai"}
                    for m in data.get("data", [])
                    if m.get("id")
                ]
            logger.debug(f"OpenAI /v1/models returned {resp.status}")
    except Exception as e:
        logger.debug(f"Codex model fetch failed: {e}")
    return []


async def _fetch_zai_models() -> list[dict]:
    """Fetch available models from Z.AI API."""
    if not zai_api_key:
        return []
    session = third_party_session or auth.session or create_session()
    owns_session = session is not third_party_session and session is not auth.session
    headers = {"Authorization": f"Bearer {zai_api_key}"}
    try:
        async with session.get(
            f"{ZAI_BASE_URL}/models",
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=5, connect=2),
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                models = data.get("data", [])
                if not models and isinstance(data, list):
                    models = data
                return [
                    {"id": f"zai:{m['id']}", "object": "model", "owned_by": "zhipuai"}
                    for m in models
                    if isinstance(m, dict) and m.get("id")
                ]
            logger.debug(f"Z.AI /models returned {resp.status}")
    except Exception as e:
        logger.debug(f"Z.AI model fetch failed: {e}")
    finally:
        if owns_session:
            await session.close()
    return []


@app.get("/v1/models")
@app.get("/api/models")
async def list_models():
    """List only models from providers that are currently authenticated and active."""
    data = []

    # Claude — only if auth is ready
    if auth.is_ready:
        claude_models = await _fetch_claude_models()
        if claude_models:
            data.extend(claude_models)
        else:
            data.extend([
                {"id": "claude-opus-4-6", "object": "model", "owned_by": "anthropic"},
                {"id": "claude-sonnet-4-6", "object": "model", "owned_by": "anthropic"},
                {"id": "claude-sonnet-4-5-20250929", "object": "model", "owned_by": "anthropic"},
                {"id": "claude-haiku-4-5-20251001", "object": "model", "owned_by": "anthropic"},
            ])

    # Codex/OpenAI — only if auth is ready
    if codex_auth.is_ready:
        codex_models = await _fetch_codex_models()
        if codex_models:
            data.extend(codex_models)
        else:
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

    # Antigravity — only if auth is ready
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

    # Ollama — probe local or cloud; skip if unreachable
    try:
        if ollama_api_key:
            endpoint = "https://ollama.com/v1/models"
            headers = {"Authorization": f"Bearer {ollama_api_key}"}
        else:
            endpoint = "http://localhost:11434/v1/models"
            headers = {}
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=2, connect=1)
        ) as s:
            async with s.get(endpoint, headers=headers) as ollama_resp:
                if ollama_resp.status == 200:
                    ollama_data = await ollama_resp.json()
                    for m in ollama_data.get("data", []):
                        model_id = m.get("id", "")
                        if model_id:
                            data.append({"id": f"ollama:{model_id}", "object": "model", "owned_by": "ollama"})
    except Exception:
        pass  # Ollama not running or unreachable — skip

    # Gemini — only if auth is ready. No dynamic listing: the Code Assist OAuth
    # token can't authenticate against generativelanguage.googleapis.com/v1beta/models,
    # so we serve a hardcoded list of known gcli model ids.
    if gemini_auth.is_ready:
        data.extend([
            {"id": "gcli-gemini-2.5-pro", "object": "model", "owned_by": "google"},
            {"id": "gcli-gemini-2.5-flash", "object": "model", "owned_by": "google"},
            {"id": "gcli-gemini-2.5-flash-lite", "object": "model", "owned_by": "google"},
            {"id": "gcli-gemini-3-pro-preview", "object": "model", "owned_by": "google"},
            {"id": "gcli-gemini-3.1-pro-preview", "object": "model", "owned_by": "google"},
            {"id": "gcli-gemini-3-flash-preview", "object": "model", "owned_by": "google"},
            {"id": "gcli-gemini-3.1-flash-lite-preview", "object": "model", "owned_by": "google"},
        ])

    # OpenRouter — only if API key is configured
    if openrouter_api_key:
        data.append({"id": "openrouter/*", "object": "model", "owned_by": "openrouter"})

    # Z.AI — dynamic fetch with static fallback
    if zai_api_key:
        zai_models = await _fetch_zai_models()
        if zai_models:
            data.extend(zai_models)
        else:
            data.extend([
                {"id": "zai:glm-4.5-air", "object": "model", "owned_by": "zhipuai"},
                {"id": "zai:glm-4.5", "object": "model", "owned_by": "zhipuai"},
                {"id": "zai:glm-4.6", "object": "model", "owned_by": "zhipuai"},
                {"id": "zai:glm-4.7", "object": "model", "owned_by": "zhipuai"},
                {"id": "zai:glm-5", "object": "model", "owned_by": "zhipuai"},
                {"id": "zai:glm-5-turbo", "object": "model", "owned_by": "zhipuai"},
            ])

    # Xiaomi MiMo — only if API key is configured
    if xiaomi_api_key:
        data.extend([
            {"id": "xiaomi:mimo-v2-pro", "object": "model", "owned_by": "xiaomi"},
            {"id": "xiaomi:mimo-v2-omni", "object": "model", "owned_by": "xiaomi"},
            {"id": "xiaomi:mimo-v2-tts", "object": "model", "owned_by": "xiaomi"},
            {"id": "xiaomi:mimo-v2-flash", "object": "model", "owned_by": "xiaomi"},
        ])

    # OpenCode — Go plan gets full model list (dynamic), Zen plan is free-tier only (hardcoded)
    if opencode_api_key:
        data.extend([
            {"id": "opencode:big-pickle", "object": "model", "owned_by": "opencode"},
            {"id": "opencode:minimax-m2.5-free", "object": "model", "owned_by": "opencode"},
            {"id": "opencode:nemotron-3-super-free", "object": "model", "owned_by": "opencode"},
        ])
    if opencode_go_api_key:
        try:
            import subprocess
            result = subprocess.run(
                ["opencode", "models", "opencode-go"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    line = line.strip()
                    if line.startswith("opencode-go/"):
                        proxy_id = f"opencode-go:{line.split('/', 1)[1]}"
                        data.append({"id": proxy_id, "object": "model", "owned_by": "opencode"})
        except Exception:
            data.extend([
                {"id": "opencode-go:glm-5", "object": "model", "owned_by": "opencode"},
                {"id": "opencode-go:glm-5.1", "object": "model", "owned_by": "opencode"},
                {"id": "opencode-go:kimi-k2.5", "object": "model", "owned_by": "opencode"},
                {"id": "opencode-go:mimo-v2-omni", "object": "model", "owned_by": "opencode"},
                {"id": "opencode-go:mimo-v2-pro", "object": "model", "owned_by": "opencode"},
                {"id": "opencode-go:minimax-m2.5", "object": "model", "owned_by": "opencode"},
                {"id": "opencode-go:minimax-m2.7", "object": "model", "owned_by": "opencode"},
            ])

    # Qwen Code — only if auth is ready
    if qwen_auth.is_ready:
        data.extend([
            {"id": "qwen:coder-model", "object": "model", "owned_by": "alibaba"},
        ])

    # Fireworks — only if API key is configured
    if fireworks_api_key:
        data.extend([
            {"id": "fireworks:kimi-k2p5-turbo", "object": "model", "owned_by": "fireworks"},
            {"id": "fireworks:accounts/fireworks/routers/kimi-k2p5-turbo", "object": "model", "owned_by": "fireworks"},
        ])

    # NVIDIA NIM — only if API key is configured
    if nvidia_api_key:
        for alias, full_path in _NVIDIA_MODEL_ALIASES.items():
            data.append({"id": f"nvidia:{alias}", "object": "model", "owned_by": "nvidia"})
            if alias != full_path:
                data.append({"id": f"nvidia:{full_path}", "object": "model", "owned_by": "nvidia"})

    return {"object": "list", "data": data}


@app.get("/health")
async def health():
    accounts_info = antigravity_auth.get_all_accounts_info()
    return {
        "status": "healthy" if (auth.is_ready or codex_auth.is_ready or antigravity_auth.is_ready or gemini_auth.is_ready or openrouter_api_key or ollama_api_key or zai_api_key or xiaomi_api_key or opencode_api_key or opencode_go_api_key or qwen_auth.is_ready or fireworks_api_key or nvidia_api_key) else "warming_up",
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
        "gemini_cli": {
            "auth_cached": gemini_auth.is_ready,
            "token_expired": gemini_auth.is_expired() if gemini_auth.is_ready else None,
            "has_refresh_token": gemini_auth.refresh_token is not None,
        },
        "openrouter_configured": openrouter_api_key is not None,
        "ollama_configured": ollama_api_key is not None,
        "zai_configured": zai_api_key is not None,
        "xiaomi_configured": xiaomi_api_key is not None,
        "opencode_zen_configured": opencode_api_key is not None,
        "opencode_go_configured": opencode_go_api_key is not None,
        "fireworks_configured": fireworks_api_key is not None,
        "nvidia_configured": nvidia_api_key is not None,
        "qwen": {
            "auth_cached": qwen_auth.is_ready,
            "token_expired": qwen_auth.is_expired() if qwen_auth.is_ready else None,
            "has_refresh_token": qwen_auth.refresh_token is not None,
        },
    }


@app.get("/config/vision-models")
async def list_vision_models():
    """Return vision-capable models based on currently available providers.

    Models are filtered by provider availability and returned in a format
    suitable for the dashboard vision test dropdown.
    """
    models = []

    # Claude models (vision capable)
    if auth.is_ready:
        models.extend([
            {"id": "claude-opus-4-6", "name": "Claude Opus 4.6", "provider": "claude"},
            {"id": "claude-sonnet-4-6", "name": "Claude Sonnet 4.6", "provider": "claude"},
            {"id": "claude-sonnet-4-5-20250929", "name": "Claude Sonnet 4.5", "provider": "claude"},
        ])

    # OpenRouter models (vision capable via API key)
    if openrouter_api_key:
        models.extend([
            {"id": "openai/gpt-4o", "name": "GPT-4o", "provider": "openrouter"},
            {"id": "openai/gpt-4o-mini", "name": "GPT-4o Mini", "provider": "openrouter"},
            {"id": "anthropic/claude-3-5-sonnet", "name": "Claude 3.5 Sonnet", "provider": "openrouter"},
            {"id": "anthropic/claude-3-7-sonnet", "name": "Claude 3.7 Sonnet", "provider": "openrouter"},
            {"id": "anthropic/claude-3-opus", "name": "Claude 3 Opus", "provider": "openrouter"},
            {"id": "google/gemini-2.5-pro-preview", "name": "Gemini 2.5 Pro", "provider": "openrouter"},
            {"id": "google/gemini-2.5-flash-preview", "name": "Gemini 2.5 Flash", "provider": "openrouter"},
            {"id": "google/gemini-2.0-flash", "name": "Gemini 2.0 Flash", "provider": "openrouter"},
        ])

    # Gemini CLI models (vision capable)
    if gemini_auth.is_ready:
        models.extend([
            {"id": "gcli-gemini-2.5-pro", "name": "Gemini 2.5 Pro", "provider": "gemini-cli"},
            {"id": "gcli-gemini-2.5-flash", "name": "Gemini 2.5 Flash", "provider": "gemini-cli"},
            {"id": "gcli-gemini-3-pro-preview", "name": "Gemini 3 Pro", "provider": "gemini-cli"},
            {"id": "gcli-gemini-3-flash-preview", "name": "Gemini 3 Flash", "provider": "gemini-cli"},
        ])

    # Antigravity models (vision capable Gemini models)
    if antigravity_auth.is_ready:
        models.extend([
            {"id": "antigravity-gemini-3.1-pro", "name": "Gemini 3.1 Pro", "provider": "antigravity"},
            {"id": "antigravity-gemini-3-pro", "name": "Gemini 3 Pro", "provider": "antigravity"},
            {"id": "antigravity-gemini-3-flash", "name": "Gemini 3 Flash", "provider": "antigravity"},
            {"id": "antigravity-gemini-2.5-pro", "name": "Gemini 2.5 Pro", "provider": "antigravity"},
            {"id": "antigravity-gemini-2.5-flash", "name": "Gemini 2.5 Flash", "provider": "antigravity"},
        ])

    # Ollama vision models - query CLI for installed models
    ollama_vision_models = []
    try:
        # Check if ollama CLI is available
        ollama_cli = shutil.which("ollama")
        if ollama_cli:
            proc = await asyncio.create_subprocess_exec(
                "ollama", "list",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=5.0
            )
            if proc.returncode == 0:
                output = stdout.decode().strip()
                # Parse table output: NAME ID SIZE MODIFIED
                for line in output.split('\n')[1:]:  # Skip header
                    parts = line.split()
                    if parts:
                        model_id = parts[0].strip()
                        # Include known vision-capable models
                        if model_id and any(x in model_id.lower() for x in ["vision", "llava", "moondream", "cogito", "bakllava", "vl"]):
                            ollama_vision_models.append({
                                "id": f"ollama:{model_id}",
                                "name": model_id,
                                "provider": "ollama"
                            })
    except Exception:
        pass

    # Static fallback list for common vision models
    common_ollama_vision = [
        {"id": "ollama:llava", "name": "llava", "provider": "ollama"},
        {"id": "ollama:llava-phi3", "name": "llava-phi3", "provider": "ollama"},
        {"id": "ollama:llava-llama3", "name": "llava-llama3", "provider": "ollama"},
        {"id": "ollama:llama3.2-vision", "name": "llama3.2-vision", "provider": "ollama"},
        {"id": "ollama:bakllava", "name": "bakllava", "provider": "ollama"},
        {"id": "ollama:moondream", "name": "moondream", "provider": "ollama"},
        {"id": "ollama:granite3.2-vision", "name": "granite3.2-vision", "provider": "ollama"},
    ]
    # Merge CLI-discovered models with static fallback (avoiding duplicates)
    existing_ids = {m["id"] for m in ollama_vision_models}
    for m in common_ollama_vision:
        if m["id"] not in existing_ids:
            ollama_vision_models.append(m)
    models.extend(ollama_vision_models)

    # Fireworks models (vision capable) - fetched dynamically if possible
    fireworks_vision_models = []
    if fireworks_api_key:
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5, connect=2)) as s:
                headers = {"Authorization": f"Bearer {fireworks_api_key}"}
                async with s.get("https://api.fireworks.ai/inference/v1/models", headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for m in data.get("data", []):
                            model_id = m.get("id", "")
                            # Vision models typically have 'vision' or 'vl' in the name
                            if model_id and any(x in model_id.lower() for x in ["vision", "-vl-", "qwen2-vl", "llava"]):
                                fireworks_vision_models.append({
                                    "id": f"fireworks:{model_id}",
                                    "name": m.get("name", model_id),
                                    "provider": "fireworks"
                                })
        except Exception:
            pass

        # Static fallback for Fireworks vision models
        # Note: Fireworks uses _resolve_fireworks_model() which strips 'fireworks:' prefix
        # and looks up aliases. Short names like 'kimi-k2p5-turbo' get expanded to full paths.
        if not fireworks_vision_models:
            fireworks_vision_models = [
                # These use the short alias format that gets expanded by _FIREWORKS_MODEL_ALIASES
                {"id": "fireworks:kimi-k2p5-turbo", "name": "Kimi K2.5 Turbo (Vision)", "provider": "fireworks"},
            ]
    models.extend(fireworks_vision_models)

    return {"models": models}


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    _any_ready = auth.is_ready or codex_auth.is_ready or antigravity_auth.is_ready or gemini_auth.is_ready or openrouter_api_key or ollama_api_key or zai_api_key or xiaomi_api_key or opencode_api_key or opencode_go_api_key or qwen_auth.is_ready or fireworks_api_key or nvidia_api_key
    status = "Ready" if _any_ready else "Warming up..."
    status_color = "#4ade80" if _any_ready else "#facc15"
    template_size = len(json.dumps(auth.body_template)) if auth.body_template else 0

    or_status = "Configured (saved)" if openrouter_api_key else "Not set"
    or_color = "#4ade80" if openrouter_api_key else "#64748b"

    ollama_status = "Cloud (key configured)" if ollama_api_key else "Local (localhost:11434)"
    ollama_color = "#4ade80" if ollama_api_key else "#64748b"

    zai_status = "Configured (saved)" if zai_api_key else "Not set"
    zai_color = "#4ade80" if zai_api_key else "#64748b"

    xiaomi_status = "Configured (saved)" if xiaomi_api_key else "Not set"
    xiaomi_color = "#4ade80" if xiaomi_api_key else "#64748b"

    _oc_ready = opencode_api_key or opencode_go_api_key
    opencode_status = f"{'Zen' if opencode_api_key else ''}{' + ' if opencode_api_key and opencode_go_api_key else ''}{'Go' if opencode_go_api_key else ''}" if _oc_ready else "Not set"
    opencode_color = "#4ade80" if _oc_ready else "#64748b"

    qwen_status = "Ready" if qwen_auth.is_ready else ("Credentials found" if os.path.exists(QWEN_CREDS_FILE) else "Not authenticated")
    qwen_color = "#4ade80" if qwen_auth.is_ready else ("#facc15" if os.path.exists(QWEN_CREDS_FILE) else "#64748b")

    fireworks_status = "Configured (saved)" if fireworks_api_key else "Not set"
    fireworks_color = "#4ade80" if fireworks_api_key else "#64748b"

    nvidia_status = "Configured (saved)" if nvidia_api_key else "Not set"
    nvidia_color = "#4ade80" if nvidia_api_key else "#64748b"

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
        ("claude-sonnet-4-6", "Sonnet 4.6", "Default"),
        ("claude-sonnet-4-5-20250929", "Sonnet 4.5", "Previous"),
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

    gemini_cli_status = "Ready" if gemini_auth.is_ready else ("Credentials found" if os.path.exists(GEMINI_CREDS_FILE) else "Not authenticated")
    gemini_cli_color = "#4ade80" if gemini_auth.is_ready else ("#facc15" if os.path.exists(GEMINI_CREDS_FILE) else "#64748b")
    gemini_cli_models = [
        ("gcli-gemini-2.5-pro", "Gemini 2.5 Pro", "Latest"),
        ("gcli-gemini-2.5-flash", "Gemini 2.5 Flash", "Fast"),
        ("gcli-gemini-2.5-flash-lite", "Gemini 2.5 Flash Lite", "Efficient"),
        ("gcli-gemini-3-pro-preview", "Gemini 3 Pro", "Preview"),
        ("gcli-gemini-3.1-pro-preview", "Gemini 3.1 Pro", "Preview"),
        ("gcli-gemini-3-flash-preview", "Gemini 3 Flash", "Preview"),
        ("gcli-gemini-3.1-flash-lite-preview", "Gemini 3.1 Flash Lite", "Preview"),
    ]
    gemini_cli_rows = "".join(
        f'<tr><td style="font-family:monospace;color:#93c5fd">{mid}</td><td>{name}</td><td style="color:#9ca3af">{desc}</td></tr>'
        for mid, name, desc in gemini_cli_models
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
      <div><span class="label">Endpoint</span><div class="endpoint">https://macbook-pro.taila2c1ae.ts.net:8539/v1/chat/completions</div></div>
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

    <!-- Gemini CLI Provider -->
    <div class="provider-card {'provider-ready' if gemini_auth.is_ready else 'provider-notready'}">
      <div class="provider-header">
        <h3 style="margin:0;font-size:1rem;color:#f1f5f9">🟡 Gemini CLI (Google)</h3>
        <span class="provider-status {'status-ready' if gemini_auth.is_ready else ('status-notready' if os.path.exists(GEMINI_CREDS_FILE) else 'status-offline')}">{gemini_cli_status}</span>
      </div>
      <div style="margin-bottom:8px"><span class="label">Auth Method:</span> <span class="value">~/.gemini/oauth_creds.json</span></div>
      <div style="margin-bottom:8px"><span class="label">Prefix:</span> <span class="value">gcli-model-name</span></div>
      {'<div style="color:#64748b;font-size:0.8rem;margin-bottom:8px">Run <code style="color:#67e8f9">gemini auth login</code> then restart proxy.</div>' if not gemini_auth.is_ready else ''}
      <table style="font-size:0.8rem">
        <thead><tr><th>Model ID</th><th>Name</th><th>Notes</th></tr></thead>
        <tbody>{gemini_cli_rows}</tbody>
      </table>
    </div>

    <!-- Ollama Provider -->
    <div class="provider-card" style="border-color:{ollama_color}40">
      <div class="provider-header">
        <h3 style="margin:0;font-size:1rem;color:#f1f5f9">🦙 Ollama</h3>
        <span class="provider-status" style="background:{ollama_color}20;color:{ollama_color}">{ollama_status}</span>
      </div>
      <div style="margin-bottom:8px"><span class="label">Prefix:</span> <span class="value">ollama:model-name</span></div>
      <div style="color:#64748b;font-size:0.8rem">
        Pull models with <code style="color:#67e8f9">ollama pull model-name</code><br>
        Set API key below for Ollama Cloud.
      </div>
    </div>

    <!-- Z.AI Provider -->
    <div class="provider-card" style="border-color:{zai_color}40">
      <div class="provider-header">
        <h3 style="margin:0;font-size:1rem;color:#f1f5f9">⚡ Z.AI</h3>
        <span class="provider-status" style="background:{zai_color}20;color:{zai_color}">{zai_status}</span>
      </div>
      <div style="margin-bottom:8px"><span class="label">Prefix:</span> <span class="value">zai:model-name</span></div>
      <div style="margin-bottom:8px"><span class="label">Base URL:</span> <span class="value">api.z.ai/api/coding/paas/v4</span></div>
      <div style="color:#64748b;font-size:0.8rem">
        GLM models via Z.AI coding plan.<br>
        e.g. <code style="color:#67e8f9">zai:glm-4.7</code>, <code style="color:#67e8f9">zai:glm-5</code>
      </div>
    </div>

    <!-- Xiaomi MiMo Provider -->
    <div class="provider-card" style="border-color:{xiaomi_color}40">
      <div class="provider-header">
        <h3 style="margin:0;font-size:1rem;color:#f1f5f9">🟠 Xiaomi MiMo</h3>
        <span class="provider-status" style="background:{xiaomi_color}20;color:{xiaomi_color}">{xiaomi_status}</span>
      </div>
      <div style="margin-bottom:8px"><span class="label">Prefix:</span> <span class="value">xiaomi:model-name</span></div>
      <div style="margin-bottom:8px"><span class="label">Base URL:</span> <span class="value">token-plan-sgp.xiaomimimo.com/v1</span></div>
      <div style="color:#64748b;font-size:0.8rem">
        Pro/Omni/TTS → token-plan-sgp endpoint.<br>
        Flash → api.xiaomimimo.com (open-source model).<br>
        e.g. <code style="color:#67e8f9">xiaomi:mimo-v2-pro</code>, <code style="color:#67e8f9">xiaomi:mimo-v2-omni</code>
      </div>
    </div>

    <!-- OpenCode Provider -->
    <div class="provider-card" style="border-color:{opencode_color}40">
      <div class="provider-header">
        <h3 style="margin:0;font-size:1rem;color:#f1f5f9">🌐 OpenCode</h3>
        <span class="provider-status" style="background:{opencode_color}20;color:{opencode_color}">{opencode_status}</span>
      </div>
      <div style="margin-bottom:8px"><span class="label">Zen:</span> <span class="value">opencode:model-name</span></div>
      <div style="margin-bottom:8px"><span class="label">Go:</span> <span class="value">opencode-go:model-name</span></div>
      <div style="color:#64748b;font-size:0.8rem">
        Zen: 40+ curated models (Claude, Gemini, GPT, GLM, Qwen).<br>
        Go: Kimi, GLM, MiMo, Minimax.<br>
        e.g. <code style="color:#67e8f9">opencode-go:kimi-k2.5</code>, <code style="color:#67e8f9">opencode:glm-5</code>
      </div>
    </div>

    <!-- Qwen Code Provider -->
    <div class="provider-card {'provider-ready' if qwen_auth.is_ready else 'provider-notready'}">
      <div class="provider-header">
        <h3 style="margin:0;font-size:1rem;color:#f1f5f9">🟣 Qwen Code</h3>
        <span class="provider-status {'status-ready' if qwen_auth.is_ready else ('status-notready' if os.path.exists(QWEN_CREDS_FILE) else 'status-offline')}">{qwen_status}</span>
      </div>
      <div style="margin-bottom:8px"><span class="label">Auth Method:</span> <span class="value">~/.qwen/oauth_creds.json</span></div>
      <div style="margin-bottom:8px"><span class="label">Prefix:</span> <span class="value">qwen:model-name</span></div>
      {'<div style="color:#64748b;font-size:0.8rem;margin-bottom:8px">Run <code style="color:#67e8f9">qwen code</code> and log in, then restart proxy.</div>' if not qwen_auth.is_ready else ''}
      <div style="color:#64748b;font-size:0.8rem">
        Free tier: 1,000 req/day via Qwen OAuth.<br>
        e.g. <code style="color:#67e8f9">qwen:coder-model</code>
      </div>
    </div>

    <!-- Fireworks Provider -->
    <div class="provider-card" style="border-color:{fireworks_color}40">
      <div class="provider-header">
        <h3 style="margin:0;font-size:1rem;color:#f1f5f9">🔥 Fireworks</h3>
        <span class="provider-status" style="background:{fireworks_color}20;color:{fireworks_color}">{fireworks_status}</span>
      </div>
      <div style="margin-bottom:8px"><span class="label">Prefix:</span> <span class="value">fireworks:model-name</span></div>
      <div style="margin-bottom:8px"><span class="label">Base URL:</span> <span class="value">api.fireworks.ai/inference/v1</span></div>
      <div style="color:#64748b;font-size:0.8rem">
        OpenAI-compatible API for hosted models.<br>
        e.g. <code style="color:#67e8f9">fireworks:accounts/fireworks/routers/kimi-k2p5-turbo</code>
      </div>
    </div>

    <!-- NVIDIA NIM Provider -->
    <div class="provider-card" style="border-color:{nvidia_color}40">
      <div class="provider-header">
        <h3 style="margin:0;font-size:1rem;color:#f1f5f9">🟢 NVIDIA NIM</h3>
        <span class="provider-status" style="background:{nvidia_color}20;color:{nvidia_color}">{nvidia_status}</span>
      </div>
      <div style="margin-bottom:8px"><span class="label">Prefix:</span> <span class="value">nvidia:model-name</span></div>
      <div style="margin-bottom:8px"><span class="label">Base URL:</span> <span class="value">integrate.api.nvidia.com/v1</span></div>
      <div style="color:#64748b;font-size:0.8rem">
        Free-tier OpenAI-compatible API for NVIDIA-hosted models.<br>
        e.g. <code style="color:#67e8f9">nvidia:deepseek-r1</code>, <code style="color:#67e8f9">nvidia:llama-3.3-70b</code>
      </div>
    </div>
  </div>

  <!-- Ollama Cloud Key -->
  <div class="card">
    <h3 style="margin:0 0 8px; font-size:1rem; color:#f1f5f9">🦙 Ollama Cloud Key</h3>
    <div style="display:flex; gap:8px; align-items:center">
      <input type="password" id="ollamaKey" placeholder="API key"
             style="flex:1; background:#0f172a; color:#e2e8f0; border:1px solid #334155;
                    border-radius:6px; padding:8px 12px; font-family:monospace; font-size:0.85rem">
      <button onclick="saveOllamaKey()" style="margin-top:0">Save</button>
      <span id="ollamaStatus" style="color:{ollama_color}; font-size:0.85rem; font-weight:600">{ollama_status}</span>
    </div>
    <div style="color:#64748b; font-size:0.8rem; margin-top:8px">
      Leave empty to use local Ollama at <code style="color:#67e8f9">localhost:11434</code>
    </div>
  </div>

  <!-- Fireworks Key -->
  <div class="card">
    <h3 style="margin:0 0 8px; font-size:1rem; color:#f1f5f9">🔥 Fireworks API Key</h3>
    <div style="display:flex; gap:8px; align-items:center">
      <input type="password" id="fireworksKey" placeholder="API key (fw_...)"
             style="flex:1; background:#0f172a; color:#e2e8f0; border:1px solid #334155;
                    border-radius:6px; padding:8px 12px; font-family:monospace; font-size:0.85rem">
      <button onclick="saveFireworksKey()" style="margin-top:0">Save</button>
      <span id="fireworksStatus" style="color:{fireworks_color}; font-size:0.85rem; font-weight:600">{fireworks_status}</span>
    </div>
    <div style="color:#64748b; font-size:0.8rem; margin-top:8px">
      API key from <code style="color:#67e8f9">console.fireworks.ai</code>
    </div>
  </div>

  <!-- NVIDIA NIM Key -->
  <div class="card">
    <h3 style="margin:0 0 8px; font-size:1rem; color:#f1f5f9">🟢 NVIDIA NIM API Key</h3>
    <div style="display:flex; gap:8px; align-items:center">
      <input type="password" id="nvidiaKey" placeholder="API key (nvapi-...)"
             style="flex:1; background:#0f172a; color:#e2e8f0; border:1px solid #334155;
                    border-radius:6px; padding:8px 12px; font-family:monospace; font-size:0.85rem">
      <button onclick="saveNvidiaKey()" style="margin-top:0">Save</button>
      <span id="nvidiaStatus" style="color:{nvidia_color}; font-size:0.85rem; font-weight:600">{nvidia_status}</span>
    </div>
    <div style="color:#64748b; font-size:0.8rem; margin-top:8px">
      API key from <code style="color:#67e8f9">build.nvidia.com</code>
    </div>
  </div>

  <!-- Z.AI Key -->
  <div class="card">
    <h3 style="margin:0 0 8px; font-size:1rem; color:#f1f5f9">⚡ Z.AI API Key</h3>
    <div style="display:flex; gap:8px; align-items:center">
      <input type="password" id="zaiKey" placeholder="API key ({{ID}}.{{secret}})"
             style="flex:1; background:#0f172a; color:#e2e8f0; border:1px solid #334155;
                    border-radius:6px; padding:8px 12px; font-family:monospace; font-size:0.85rem">
      <button onclick="saveZaiKey()" style="margin-top:0">Save</button>
      <span id="zaiStatus" style="color:{zai_color}; font-size:0.85rem; font-weight:600">{zai_status}</span>
    </div>
    <div style="color:#64748b; font-size:0.8rem; margin-top:8px">
      Coding plan key from <code style="color:#67e8f9">z.ai/manage-apikey/apikey-list</code>
    </div>
  </div>

  <!-- Xiaomi MiMo Key -->
  <div class="card">
    <h3 style="margin:0 0 8px; font-size:1rem; color:#f1f5f9">🟠 Xiaomi MiMo API Key</h3>
    <div style="display:flex; gap:8px; align-items:center">
      <input type="password" id="xiaomiKey" placeholder="API key"
             style="flex:1; background:#0f172a; color:#e2e8f0; border:1px solid #334155;
                    border-radius:6px; padding:8px 12px; font-family:monospace; font-size:0.85rem">
      <button onclick="saveXiaomiKey()" style="margin-top:0">Save</button>
      <span id="xiaomiStatus" style="color:{xiaomi_color}; font-size:0.85rem; font-weight:600">{xiaomi_status}</span>
    </div>
    <div style="color:#64748b; font-size:0.8rem; margin-top:8px">
      Token plan key from <code style="color:#67e8f9">platform.xiaomimimo.com</code> — SGP endpoint
    </div>
  </div>

  <!-- OpenCode Zen Key -->
  <div class="card">
    <h3 style="margin:0 0 8px; font-size:1rem; color:#f1f5f9">🌐 OpenCode Zen API Key</h3>
    <div style="display:flex; gap:8px; align-items:center">
      <input type="password" id="opencodeKey" placeholder="API key (sk-...)"
             style="flex:1; background:#0f172a; color:#e2e8f0; border:1px solid #334155;
                    border-radius:6px; padding:8px 12px; font-family:monospace; font-size:0.85rem">
      <button onclick="saveOpencodeKey()" style="margin-top:0">Save</button>
      <span id="opencodeStatus" style="color:{opencode_color}; font-size:0.85rem; font-weight:600">{opencode_status}</span>
    </div>
    <div style="color:#64748b; font-size:0.8rem; margin-top:8px">
      Auto-loaded from <code style="color:#67e8f9">~/.local/share/opencode/auth.json</code> if present
    </div>
  </div>

  <!-- OpenCode Go Key -->
  <div class="card">
    <h3 style="margin:0 0 8px; font-size:1rem; color:#f1f5f9">🌐 OpenCode Go API Key</h3>
    <div style="display:flex; gap:8px; align-items:center">
      <input type="password" id="opencodeGoKey" placeholder="API key (sk-...)"
             style="flex:1; background:#0f172a; color:#e2e8f0; border:1px solid #334155;
                    border-radius:6px; padding:8px 12px; font-family:monospace; font-size:0.85rem">
      <button onclick="saveOpencodeGoKey()" style="margin-top:0">Save</button>
      <span id="opencodeGoStatus" style="color:{'#4ade80' if opencode_go_api_key else '#64748b'}; font-size:0.85rem; font-weight:600">{'Configured' if opencode_go_api_key else 'Not set'}</span>
    </div>
    <div style="color:#64748b; font-size:0.8rem; margin-top:8px">
      Go plan key — auto-loaded from <code style="color:#67e8f9">~/.local/share/opencode/auth.json</code> if present
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

  <!-- Timeout Routing -->
  <div class="card">
    <h3 style="margin:0 0 8px; font-size:1rem; color:#f1f5f9">⚡ Timeout Routing <span style="font-size:0.75rem;font-weight:400;color:#64748b">(streaming only)</span></h3>
    <p style="margin:0 0 10px;color:#94a3b8;font-size:0.85rem">
      <b>TTFT cutoff</b>: if no content token arrives in time, the stream is cancelled and re-routed to the most reliable model.
      <b>Max total</b>: hard wall-clock ceiling for the whole stream — anything past this is cut mid-generation with <code>[DONE]</code>.
      Non-streaming (direct) calls are NOT bounded — memory/summary workflows need the headroom.
    </p>
    <div style="display:flex; gap:12px; align-items:center; flex-wrap:wrap">
      <label style="display:flex;gap:8px;align-items:center;cursor:pointer">
        <input type="checkbox" id="timeoutRoutingToggle" {'checked' if timeout_routing_enabled else ''}
               onchange="setTimeoutRouting(this.checked)"
               style="width:16px;height:16px;cursor:pointer">
        <span style="color:#f1f5f9;font-size:0.9rem">Enabled</span>
      </label>
      <span id="timeoutRoutingStatus" style="color:{'#4ade80' if timeout_routing_enabled else '#64748b'};font-size:0.85rem;font-weight:600">
        {'Active' if timeout_routing_enabled else 'Disabled'}
      </span>
      <label style="display:flex;gap:6px;align-items:center">
        <span style="color:#94a3b8;font-size:0.85rem">TTFT cutoff:</span>
        <input type="number" id="timeoutCutoffInput" value="{timeout_cutoff_seconds}" min="1" max="30" step="0.5"
               style="width:60px;background:#1e293b;border:1px solid #334155;color:#f1f5f9;padding:4px 6px;border-radius:4px;font-size:0.85rem">
        <span style="color:#94a3b8;font-size:0.85rem">s</span>
        <button onclick="setTimeoutCutoff()" style="margin-top:0;background:#334155;padding:4px 10px;font-size:0.8rem">Set</button>
      </label>
      <label style="display:flex;gap:6px;align-items:center">
        <span style="color:#94a3b8;font-size:0.85rem">Max total:</span>
        <input type="number" id="maxTotalInput" value="{max_total_seconds}" min="1" max="60" step="0.5"
               style="width:60px;background:#1e293b;border:1px solid #334155;color:#f1f5f9;padding:4px 6px;border-radius:4px;font-size:0.85rem">
        <span style="color:#94a3b8;font-size:0.85rem">s</span>
        <button onclick="setMaxTotal()" style="margin-top:0;background:#334155;padding:4px 10px;font-size:0.8rem">Set</button>
      </label>
      <button onclick="loadRoutingStats()" style="margin-top:0;background:#334155">View Stats</button>
      <button onclick="loadRequestStats()" style="margin-top:0;background:#334155">Request Stats</button>
    </div>
    <pre id="routingStats" style="display:none;margin-top:10px;background:#0f172a;padding:10px;border-radius:6px;font-size:0.8rem;color:#94a3b8;overflow:auto"></pre>
    <pre id="requestStats" style="display:none;margin-top:10px;background:#0f172a;padding:10px;border-radius:6px;font-size:0.8rem;color:#94a3b8;overflow:auto"></pre>
  </div>

  <!-- Quick Test -->
  <div class="card">
    <h3 style="margin:0 0 8px; font-size:1rem; color:#f1f5f9">Quick Test</h3>

    <!-- Model Selection -->
    <div style="margin-bottom:12px">
      <div style="display:flex; gap:12px; align-items:center; margin-bottom:8px">
        <label style="display:flex;align-items:center;gap:6px;cursor:pointer">
          <input type="radio" name="quickModelSource" value="dropdown" checked
                 onchange="toggleQuickModelSource()"
                 style="cursor:pointer">
          <span style="color:#e2e8f0;font-size:0.9rem">Select from list</span>
        </label>
        <label style="display:flex;align-items:center;gap:6px;cursor:pointer">
          <input type="radio" name="quickModelSource" value="custom"
                 onchange="toggleQuickModelSource()"
                 style="cursor:pointer">
          <span style="color:#e2e8f0;font-size:0.9rem">Custom model ID</span>
        </label>
      </div>

      <select id="modelSelect">
        <optgroup label="Claude Models">
          <option value="claude-sonnet-4-6">claude-sonnet-4-6 (default)</option>
          <option value="claude-opus-4-6">claude-opus-4-6</option>
          <option value="claude-sonnet-4-5-20250929">claude-sonnet-4-5-20250929</option>
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
        <optgroup label="Gemini CLI Models">
          <option value="gcli-gemini-2.5-pro">gcli-gemini-2.5-pro</option>
          <option value="gcli-gemini-2.5-flash">gcli-gemini-2.5-flash</option>
          <option value="gcli-gemini-2.5-flash-lite">gcli-gemini-2.5-flash-lite</option>
          <option value="gcli-gemini-3-pro-preview">gcli-gemini-3-pro-preview</option>
          <option value="gcli-gemini-3.1-pro-preview">gcli-gemini-3.1-pro-preview</option>
          <option value="gcli-gemini-3-flash-preview">gcli-gemini-3-flash-preview</option>
          <option value="gcli-gemini-3.1-flash-lite-preview">gcli-gemini-3.1-flash-lite-preview</option>
        </optgroup>
        <optgroup label="Ollama Models" id="ollamaOptgroup">
          <option disabled value="">Loading…</option>
        </optgroup>
        <optgroup label="Z.AI Models" id="zaiOptgroup">
          <option disabled value="">Loading…</option>
        </optgroup>
        <optgroup label="Xiaomi MiMo Models">
          <option value="xiaomi:mimo-v2-pro">xiaomi:mimo-v2-pro</option>
          <option value="xiaomi:mimo-v2-omni">xiaomi:mimo-v2-omni</option>
          <option value="xiaomi:mimo-v2-tts">xiaomi:mimo-v2-tts</option>
          <option value="xiaomi:mimo-v2-flash">xiaomi:mimo-v2-flash (platform)</option>
        </optgroup>
        <optgroup label="OpenCode Zen Models (Free)">
          <option value="opencode:big-pickle">opencode:big-pickle</option>
          <option value="opencode:minimax-m2.5-free">opencode:minimax-m2.5-free</option>
          <option value="opencode:nemotron-3-super-free">opencode:nemotron-3-super-free</option>
        </optgroup>
        <optgroup label="OpenCode Go Models">
          <option value="opencode-go:glm-5">opencode-go:glm-5</option>
          <option value="opencode-go:glm-5.1">opencode-go:glm-5.1</option>
          <option value="opencode-go:kimi-k2.5">opencode-go:kimi-k2.5</option>
          <option value="opencode-go:mimo-v2-omni">opencode-go:mimo-v2-omni</option>
          <option value="opencode-go:mimo-v2-pro">opencode-go:mimo-v2-pro</option>
          <option value="opencode-go:minimax-m2.5">opencode-go:minimax-m2.5</option>
          <option value="opencode-go:minimax-m2.7">opencode-go:minimax-m2.7</option>
        </optgroup>
        <optgroup label="Qwen Code Models">
          <option value="qwen:coder-model">qwen:coder-model</option>
        </optgroup>
        <optgroup label="Fireworks Models">
          <option value="fireworks:accounts/fireworks/routers/kimi-k2p5-turbo">fireworks:kimi-k2p5-turbo</option>
        </optgroup>
        <optgroup label="NVIDIA NIM Models (Free)">
          <option value="nvidia:kimi-k2">nvidia:kimi-k2</option>
          <option value="nvidia:kimi-k2-0905">nvidia:kimi-k2-0905</option>
          <option value="nvidia:kimi-k2-thinking">nvidia:kimi-k2-thinking</option>
          <option value="nvidia:glm-4.7">nvidia:glm-4.7</option>
          <option value="nvidia:deepseek-v3.2">nvidia:deepseek-v3.2</option>
          <option value="nvidia:deepseek-v3.1">nvidia:deepseek-v3.1</option>
          <option value="nvidia:deepseek-v3.1-terminus">nvidia:deepseek-v3.1-terminus</option>
          <option value="nvidia:step-3.5-flash">nvidia:step-3.5-flash</option>
          <option value="nvidia:llama-4-maverick">nvidia:llama-4-maverick</option>
          <option value="nvidia:gemma-3-27b">nvidia:gemma-3-27b</option>
          <option value="nvidia:mistral-large-3">nvidia:mistral-large-3</option>
          <option value="nvidia:devstral-2">nvidia:devstral-2</option>
          <option value="nvidia:magistral-small">nvidia:magistral-small</option>
          <option value="nvidia:mistral-nemotron">nvidia:mistral-nemotron</option>
          <option value="nvidia:qwen3-coder-480b">nvidia:qwen3-coder-480b</option>
          <option value="nvidia:seed-oss-36b">nvidia:seed-oss-36b</option>
          <option value="nvidia:phi-3.5-mini">nvidia:phi-3.5-mini</option>
        </optgroup>
      </select>

      <input type="text" id="quickModelInput" placeholder="Enter custom model ID (e.g., claude-opus-4-6)"
             style="display:none; width:100%; background:#0f172a; color:#e2e8f0; border:1px solid #334155;
                    border-radius:6px; padding:8px 12px; font-family:monospace; font-size:0.85rem; box-sizing:border-box">

      <div id="quickModelDisplay" style="color:#4ade80; font-size:0.85rem; font-family:monospace; margin-top:4px">
        Using: <span id="quickCurrentModelId">-select a model-</span>
      </div>
    </div>
    <textarea id="sysPrompt" rows="2" placeholder="System prompt">You are Lydia, a Nord housecarl sworn to protect the Dragonborn. Stay in character. One sentence only.</textarea>
    <textarea id="userMsg" rows="1" placeholder="User message" style="margin-top:6px">What do you think of dragons?</textarea>
    <button onclick="testChat()" id="testBtn">Send</button>
    <div id="response" style="display:none"></div>
    <div id="timing" class="timing"></div>
  </div>

  <!-- Vision Test -->
  <div class="card">
    <h3 style="margin:0 0 8px; font-size:1rem; color:#f1f5f9">Vision Test</h3>

    <!-- Model Selection -->
    <div style="margin-bottom:12px">
      <div style="display:flex; gap:12px; align-items:center; margin-bottom:8px">
        <label style="display:flex;align-items:center;gap:6px;cursor:pointer">
          <input type="radio" name="visionModelSource" value="dropdown" checked
                 onchange="toggleVisionModelSource()"
                 style="cursor:pointer">
          <span style="color:#e2e8f0;font-size:0.9rem">Select from list</span>
        </label>
        <label style="display:flex;align-items:center;gap:6px;cursor:pointer">
          <input type="radio" name="visionModelSource" value="custom"
                 onchange="toggleVisionModelSource()"
                 style="cursor:pointer">
          <span style="color:#e2e8f0;font-size:0.9rem">Custom model ID</span>
        </label>
      </div>

      <select id="visionModelSelect" style="width:100%; margin-bottom:8px">
        <option disabled value="">Loading available vision models…</option>
      </select>

      <input type="text" id="visionModelInput" placeholder="Enter custom model ID (e.g., ollama:llava)"
             style="display:none; width:100%; background:#0f172a; color:#e2e8f0; border:1px solid #334155;
                    border-radius:6px; padding:8px 12px; font-family:monospace; font-size:0.85rem; margin-bottom:8px; box-sizing:border-box">

      <div id="visionModelDisplay" style="color:#4ade80; font-size:0.85rem; font-family:monospace; margin-top:4px">
        Using: <span id="visionCurrentModelId">-select a model-</span>
      </div>
    </div>

    <textarea id="visionPrompt" rows="2" placeholder="Describe what you want to know about the image">What do you see in this image? Describe it briefly.</textarea>

    <!-- Image Selection -->
    <div style="margin-top:12px; padding:12px; background:#0f172a; border-radius:6px; border:1px solid #334155">
      <div style="font-size:0.85rem; color:#94a3b8; margin-bottom:8px">Test Image:</div>
      <div style="display:flex; gap:12px; align-items:center; flex-wrap:wrap">
        <label style="display:flex;align-items:center;gap:6px;cursor:pointer">
          <input type="radio" name="visionImageSource" value="random" checked
                 onchange="toggleVisionImageSource()"
                 style="cursor:pointer">
          <span style="color:#e2e8f0;font-size:0.85rem">Random test image</span>
        </label>
        <label style="display:flex;align-items:center;gap:6px;cursor:pointer">
          <input type="radio" name="visionImageSource" value="upload"
                 onchange="toggleVisionImageSource()"
                 style="cursor:pointer">
          <span style="color:#e2e8f0;font-size:0.85rem">Upload image</span>
        </label>
      </div>

      <div id="visionRandomPreview" style="margin-top:10px">
        <div style="width:120px; height:120px; border:2px dashed #334155; border-radius:6px; display:flex; align-items:center; justify-content:center; background:#1e293b; position:relative; overflow:hidden">
          <canvas id="visionRandomCanvas" width="120" height="120"></canvas>
        </div>
        <button onclick="generateRandomVisionImage()" type="button" style="margin-top:8px; background:#334155; padding:6px 12px; font-size:0.8rem">Generate new image</button>
      </div>

      <div id="visionUploadSection" style="display:none; margin-top:10px">
        <input type="file" id="visionImageFile" accept="image/*"
               style="color:#e2e8f0; font-size:0.85rem">
        <div style="color:#64748b; font-size:0.75rem; margin-top:6px">Supports: PNG, JPG, GIF, WebP (max 4MB)</div>
      </div>
    </div>

    <button onclick="testVision()" id="visionTestBtn" style="margin-top:12px">Analyze Image</button>
    <div id="visionResponse" style="display:none; margin-top:12px; padding:12px; background:#0f172a; border-radius:6px; border:1px solid #334155; font-size:0.9rem; min-height:40px; white-space:pre-wrap"></div>
    <div id="visionTiming" class="timing"></div>
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

async function saveOllamaKey() {{
  const key = document.getElementById('ollamaKey').value.trim();
  const status = document.getElementById('ollamaStatus');
  try {{
    await fetch('/config/ollama-key', {{
      method: 'POST', headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{key: key}})
    }});
    status.textContent = key ? 'Cloud (key configured)' : 'Local (localhost:11434)';
    status.style.color = key ? '#4ade80' : '#64748b';
    document.getElementById('ollamaKey').value = '';
  }} catch(e) {{
    status.textContent = 'Error';
    status.style.color = '#f87171';
  }}
}}

async function saveFireworksKey() {{
  const key = document.getElementById('fireworksKey').value.trim();
  const status = document.getElementById('fireworksStatus');
  try {{
    await fetch('/config/fireworks-key', {{
      method: 'POST', headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{key: key}})
    }});
    status.textContent = key ? 'Configured (saved)' : 'Not set';
    status.style.color = key ? '#4ade80' : '#64748b';
    document.getElementById('fireworksKey').value = '';
  }} catch(e) {{
    status.textContent = 'Error';
    status.style.color = '#f87171';
  }}
}}

async function saveNvidiaKey() {{
  const key = document.getElementById('nvidiaKey').value.trim();
  const status = document.getElementById('nvidiaStatus');
  try {{
    await fetch('/config/nvidia-key', {{
      method: 'POST', headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{key: key}})
    }});
    status.textContent = key ? 'Configured (saved)' : 'Not set';
    status.style.color = key ? '#4ade80' : '#64748b';
    document.getElementById('nvidiaKey').value = '';
  }} catch(e) {{
    status.textContent = 'Error';
    status.style.color = '#f87171';
  }}
}}

async function saveZaiKey() {{
  const key = document.getElementById('zaiKey').value.trim();
  const status = document.getElementById('zaiStatus');
  try {{
    await fetch('/config/zai-key', {{
      method: 'POST', headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{key: key}})
    }});
    status.textContent = key ? 'Configured (saved)' : 'Not set';
    status.style.color = key ? '#4ade80' : '#64748b';
    document.getElementById('zaiKey').value = '';
  }} catch(e) {{
    status.textContent = 'Error';
    status.style.color = '#f87171';
  }}
}}

async function saveXiaomiKey() {{
  const key = document.getElementById('xiaomiKey').value.trim();
  const status = document.getElementById('xiaomiStatus');
  try {{
    await fetch('/config/xiaomi-key', {{
      method: 'POST', headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{key: key}})
    }});
    status.textContent = key ? 'Configured (saved)' : 'Not set';
    status.style.color = key ? '#4ade80' : '#64748b';
    document.getElementById('xiaomiKey').value = '';
  }} catch(e) {{
    status.textContent = 'Error';
    status.style.color = '#f87171';
  }}
}}

async function saveOpencodeKey() {{
  const key = document.getElementById('opencodeKey').value.trim();
  const status = document.getElementById('opencodeStatus');
  try {{
    await fetch('/config/opencode-key', {{
      method: 'POST', headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{key: key}})
    }});
    status.textContent = key ? 'Configured (saved)' : 'Not set';
    status.style.color = key ? '#4ade80' : '#64748b';
    document.getElementById('opencodeKey').value = '';
  }} catch(e) {{
    status.textContent = 'Error';
    status.style.color = '#f87171';
  }}
}}

async function saveOpencodeGoKey() {{
  const key = document.getElementById('opencodeGoKey').value.trim();
  const status = document.getElementById('opencodeGoStatus');
  try {{
    await fetch('/config/opencode-go-key', {{
      method: 'POST', headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{key: key}})
    }});
    status.textContent = key ? 'Configured (saved)' : 'Not set';
    status.style.color = key ? '#4ade80' : '#64748b';
    document.getElementById('opencodeGoKey').value = '';
  }} catch(e) {{
    status.textContent = 'Error';
    status.style.color = '#f87171';
  }}
}}

async function testChat() {{
  const btn = document.getElementById('testBtn');
  const resp = document.getElementById('response');
  const timing = document.getElementById('timing');

  // Get model from dropdown or custom input
  const source = document.querySelector('input[name="quickModelSource"]:checked').value;
  let model = '';
  if (source === 'dropdown') {{
    model = document.getElementById('modelSelect').value;
  }} else {{
    model = document.getElementById('quickModelInput').value.trim();
  }}

  if (!model) {{
    resp.style.display = 'block';
    resp.textContent = 'Please select or enter a model ID';
    return;
  }}

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

async function loadOllamaModels() {{
  const group = document.getElementById('ollamaOptgroup');
  try {{
    const r = await fetch('/config/ollama-models');
    const data = await r.json();
    group.innerHTML = '';
    if (data.models && data.models.length > 0) {{
      data.models.forEach(function(m) {{
        const opt = document.createElement('option');
        opt.value = 'ollama:' + m;
        opt.textContent = 'ollama:' + m;
        group.appendChild(opt);
      }});
    }} else {{
      const opt = document.createElement('option');
      opt.disabled = true;
      opt.value = '';
      opt.textContent = 'No local models (run: ollama pull <model>)';
      group.appendChild(opt);
    }}
  }} catch(e) {{
    group.innerHTML = '<option disabled value="">Ollama not running</option>';
  }}
}}
async function loadZaiModels() {{
  const group = document.getElementById('zaiOptgroup');
  try {{
    const r = await fetch('/config/zai-models');
    const data = await r.json();
    group.innerHTML = '';
    if (data.models && data.models.length > 0) {{
      data.models.forEach(function(m) {{
        const opt = document.createElement('option');
        opt.value = 'zai:' + m;
        opt.textContent = 'zai:' + m;
        group.appendChild(opt);
      }});
    }} else {{
      const opt = document.createElement('option');
      opt.disabled = true;
      opt.value = '';
      opt.textContent = 'No Z.AI models (check API key)';
      group.appendChild(opt);
    }}
  }} catch(e) {{
    group.innerHTML = '<option disabled value="">Z.AI unavailable</option>';
  }}
}}
document.addEventListener('DOMContentLoaded', loadOllamaModels);
document.addEventListener('DOMContentLoaded', loadZaiModels);
document.addEventListener('DOMContentLoaded', loadVisionModels);
document.addEventListener('DOMContentLoaded', function() {{
  // Initialize vision test UI
  toggleVisionModelSource();
  generateRandomVisionImage();
  toggleVisionImageSource();
}});

async function loadVisionModels() {{
  const select = document.getElementById('visionModelSelect');
  try {{
    const r = await fetch('/config/vision-models');
    const data = await r.json();
    select.innerHTML = '';

    // Group models by provider
    const byProvider = {{}};
    if (data.models && data.models.length > 0) {{
      data.models.forEach(function(m) {{
        const provider = m.provider || 'Other';
        if (!byProvider[provider]) byProvider[provider] = [];
        byProvider[provider].push(m);
      }});
    }} else {{
      select.innerHTML = '<option disabled value="">No vision models available</option>';
      updateVisionModelDisplay();
      return;
    }}

    // Create optgroups
    const providerOrder = ['claude', 'openrouter', 'gemini-cli', 'antigravity', 'ollama', 'fireworks', 'nvidia'];
    const providerLabels = {{
      'claude': 'Claude',
      'openrouter': 'OpenRouter',
      'gemini-cli': 'Gemini CLI',
      'antigravity': 'Antigravity',
      'ollama': 'Ollama',
      'fireworks': 'Fireworks',
      'nvidia': 'NVIDIA NIM'
    }};

    providerOrder.forEach(function(provider) {{
      if (byProvider[provider] && byProvider[provider].length > 0) {{
        const optgroup = document.createElement('optgroup');
        optgroup.label = providerLabels[provider] || provider;
        byProvider[provider].forEach(function(m) {{
          const opt = document.createElement('option');
          opt.value = m.id;
          opt.textContent = m.name;
          optgroup.appendChild(opt);
        }});
        select.appendChild(optgroup);
      }}
    }});

    // Add any other providers not in the order list
    Object.keys(byProvider).forEach(function(provider) {{
      if (!providerOrder.includes(provider) && byProvider[provider].length > 0) {{
        const optgroup = document.createElement('optgroup');
        optgroup.label = providerLabels[provider] || provider;
        byProvider[provider].forEach(function(m) {{
          const opt = document.createElement('option');
          opt.value = m.id;
          opt.textContent = m.name;
          optgroup.appendChild(opt);
        }});
        select.appendChild(optgroup);
      }}
    }});

    // Set first model as selected and update display
    if (select.options.length > 0) {{
      select.selectedIndex = 0;
    }}
    updateVisionModelDisplay();

    // Add change listener
    select.addEventListener('change', updateVisionModelDisplay);

  }} catch(e) {{
    select.innerHTML = '<option disabled value="">Failed to load vision models</option>';
    console.error('Failed to load vision models:', e);
  }}
}}

function toggleVisionModelSource() {{
  const source = document.querySelector('input[name="visionModelSource"]:checked').value;
  const select = document.getElementById('visionModelSelect');
  const input = document.getElementById('visionModelInput');

  if (source === 'dropdown') {{
    select.style.display = 'block';
    input.style.display = 'none';
  }} else {{
    select.style.display = 'none';
    input.style.display = 'block';
    input.focus();
  }}
  updateVisionModelDisplay();
}}

function updateVisionModelDisplay() {{
  const source = document.querySelector('input[name="visionModelSource"]:checked').value;
  const display = document.getElementById('visionCurrentModelId');

  if (source === 'dropdown') {{
    const select = document.getElementById('visionModelSelect');
    display.textContent = select.value || '-select a model-';
  }} else {{
    const input = document.getElementById('visionModelInput');
    display.textContent = input.value.trim() || '-enter custom model ID-';
  }}
}}

// Update display when custom input changes
document.addEventListener('DOMContentLoaded', function() {{
  const input = document.getElementById('visionModelInput');
  if (input) {{
    input.addEventListener('input', updateVisionModelDisplay);
  }}
}});

function toggleQuickModelSource() {{
  const source = document.querySelector('input[name="quickModelSource"]:checked').value;
  const select = document.getElementById('modelSelect');
  const input = document.getElementById('quickModelInput');

  if (source === 'dropdown') {{
    select.style.display = 'block';
    input.style.display = 'none';
  }} else {{
    select.style.display = 'none';
    input.style.display = 'block';
    input.focus();
  }}
  updateQuickModelDisplay();
}}

function updateQuickModelDisplay() {{
  const source = document.querySelector('input[name="quickModelSource"]:checked').value;
  const display = document.getElementById('quickCurrentModelId');

  if (source === 'dropdown') {{
    const select = document.getElementById('modelSelect');
    display.textContent = select.value || '-select a model-';
  }} else {{
    const input = document.getElementById('quickModelInput');
    display.textContent = input.value.trim() || '-enter custom model ID-';
  }}
}}

// Update Quick Test display on page load and input changes
document.addEventListener('DOMContentLoaded', function() {{
  toggleQuickModelSource();

  const input = document.getElementById('quickModelInput');
  if (input) {{
    input.addEventListener('input', updateQuickModelDisplay);
  }}

  const select = document.getElementById('modelSelect');
  if (select) {{
    select.addEventListener('change', updateQuickModelDisplay);
  }}
}});

function toggleVisionImageSource() {{
  const source = document.querySelector('input[name="visionImageSource"]:checked').value;
  const randomSection = document.getElementById('visionRandomPreview');
  const uploadSection = document.getElementById('visionUploadSection');

  if (source === 'random') {{
    randomSection.style.display = 'block';
    uploadSection.style.display = 'none';
    generateRandomVisionImage();
  }} else {{
    randomSection.style.display = 'none';
    uploadSection.style.display = 'block';
  }}
}}

function generateRandomVisionImage() {{
  const canvas = document.getElementById('visionRandomCanvas');
  const ctx = canvas.getContext('2d');

  // Generate random colors
  const hue1 = Math.floor(Math.random() * 360);
  const hue2 = (hue1 + 120 + Math.floor(Math.random() * 120)) % 360;
  const hue3 = (hue2 + 120 + Math.floor(Math.random() * 120)) % 360;

  // Fill background
  ctx.fillStyle = `hsl(${{hue1}}, 60%, 20%)`;
  ctx.fillRect(0, 0, 120, 120);

  // Draw random shapes
  for (let i = 0; i < 8; i++) {{
    ctx.fillStyle = `hsl(${{Math.random() * 360}}, 70%, ${{40 + Math.random() * 40}}%)`;
    const x = Math.random() * 120;
    const y = Math.random() * 120;
    const size = 10 + Math.random() * 40;

    if (Math.random() > 0.5) {{
      // Circle
      ctx.beginPath();
      ctx.arc(x, y, size / 2, 0, Math.PI * 2);
      ctx.fill();
    }} else {{
      // Rectangle
      ctx.fillRect(x - size/2, y - size/2, size, size);
    }}
  }}

  // Add some lines
  for (let i = 0; i < 5; i++) {{
    ctx.strokeStyle = `hsl(${{Math.random() * 360}}, 80%, 60%)`;
    ctx.lineWidth = 2 + Math.random() * 4;
    ctx.beginPath();
    ctx.moveTo(Math.random() * 120, Math.random() * 120);
    ctx.lineTo(Math.random() * 120, Math.random() * 120);
    ctx.stroke();
  }}

  // Store the base64 for later use
  canvas.dataset.base64 = canvas.toDataURL('image/png');
}}

function getVisionImageBase64() {{
  const source = document.querySelector('input[name="visionImageSource"]:checked').value;

  if (source === 'random') {{
    const canvas = document.getElementById('visionRandomCanvas');
    return canvas.dataset.base64 || canvas.toDataURL('image/png');
  }} else {{
    const fileInput = document.getElementById('visionImageFile');
    return new Promise((resolve, reject) => {{
      if (!fileInput.files || fileInput.files.length === 0) {{
        reject(new Error('Please select an image file'));
        return;
      }}
      const file = fileInput.files[0];
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(new Error('Error reading image file'));
      reader.readAsDataURL(file);
    }});
  }}
}}

async function testVision() {{
  const btn = document.getElementById('visionTestBtn');
  const resp = document.getElementById('visionResponse');
  const timing = document.getElementById('visionTiming');

  // Get model from dropdown or custom input
  const source = document.querySelector('input[name="visionModelSource"]:checked').value;
  let model = '';
  if (source === 'dropdown') {{
    model = document.getElementById('visionModelSelect').value;
  }} else {{
    model = document.getElementById('visionModelInput').value.trim();
  }}

  if (!model) {{
    resp.style.display = 'block';
    resp.textContent = 'Please select or enter a model ID';
    return;
  }}

  // Check image source
  const imageSource = document.querySelector('input[name="visionImageSource"]:checked').value;
  if (imageSource === 'upload') {{
    const fileInput = document.getElementById('visionImageFile');
    if (!fileInput.files || fileInput.files.length === 0) {{
      resp.style.display = 'block';
      resp.textContent = 'Please select an image file';
      return;
    }}
  }}

  btn.disabled = true; btn.textContent = 'Analyzing...';
  resp.style.display = 'block'; resp.textContent = 'Processing image...';
  timing.textContent = '';

  const start = Date.now();

  try {{
    // Get image base64
    let base64Image;
    if (imageSource === 'random') {{
      base64Image = getVisionImageBase64();
    }} else {{
      base64Image = await getVisionImageBase64();
    }}

    const prompt = document.getElementById('visionPrompt').value || 'What do you see in this image?';

    const r = await fetch('/v1/chat/completions', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{
        model: model,
        messages: [
          {{role: 'user', content: [
            {{type: 'text', text: prompt}},
            {{type: 'image_url', image_url: {{url: base64Image}}}}
          ]}}
        ]
      }})
    }});

    const data = await r.json();
    const elapsed = ((Date.now() - start) / 1000).toFixed(1);

    if (data.choices && data.choices[0] && data.choices[0].message) {{
      resp.textContent = data.choices[0].message.content;
      timing.textContent = model + ' · ' + elapsed + 's';
    }} else if (data.error) {{
      resp.textContent = 'Error: ' + (data.error.message || JSON.stringify(data.error));
      timing.textContent = '';
    }} else {{
      resp.textContent = JSON.stringify(data, null, 2);
    }}
  }} catch(e) {{
    resp.textContent = 'Error: ' + e.message;
    timing.textContent = '';
  }}

  btn.disabled = false; btn.textContent = 'Analyze Image';
}}

async function setTimeoutRouting(enabled) {{
  const status = document.getElementById('timeoutRoutingStatus');
  try {{
    const resp = await fetch('/config/timeout-routing', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{enabled}}),
    }});
    const data = await resp.json();
    status.textContent = data.timeout_routing_enabled ? 'Active' : 'Disabled';
    status.style.color = data.timeout_routing_enabled ? '#4ade80' : '#64748b';
  }} catch(e) {{
    status.textContent = 'Error: ' + e.message;
  }}
}}

async function loadRoutingStats() {{
  const pre = document.getElementById('routingStats');
  pre.style.display = 'block';
  pre.textContent = 'Loading...';
  try {{
    const resp = await fetch('/config/timeout-routing');
    const data = await resp.json();
    if (Object.keys(data.stats).length === 0) {{
      pre.textContent = 'No data yet — stats accumulate as streaming requests are made.';
    }} else {{
      pre.textContent = JSON.stringify(data.stats, null, 2);
    }}
  }} catch(e) {{
    pre.textContent = 'Error: ' + e.message;
  }}
}}

async function setTimeoutCutoff() {{
  const input = document.getElementById('timeoutCutoffInput');
  const seconds = parseFloat(input.value);
  if (isNaN(seconds) || seconds < 1 || seconds > 30) {{
    alert('TTFT cutoff must be between 1 and 30 seconds');
    return;
  }}
  try {{
    const resp = await fetch('/config/timeout-cutoff', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{seconds}}),
    }});
    const data = await resp.json();
    input.value = data.timeout_cutoff_seconds;
  }} catch(e) {{
    alert('Error: ' + e.message);
  }}
}}

async function setMaxTotal() {{
  const input = document.getElementById('maxTotalInput');
  const seconds = parseFloat(input.value);
  if (isNaN(seconds) || seconds < 1 || seconds > 60) {{
    alert('Max total must be between 1 and 60 seconds');
    return;
  }}
  try {{
    const resp = await fetch('/config/max-total', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{seconds}}),
    }});
    const data = await resp.json();
    input.value = data.max_total_seconds;
  }} catch(e) {{
    alert('Error: ' + e.message);
  }}
}}

async function loadRequestStats() {{
  const pre = document.getElementById('requestStats');
  pre.style.display = 'block';
  pre.textContent = 'Loading...';
  try {{
    const resp = await fetch('/stats');
    const data = await resp.json();
    if (Object.keys(data.by_model).length === 0) {{
      pre.textContent = 'No request data yet — stats accumulate as requests are made.';
    }} else {{
      pre.textContent = JSON.stringify(data, null, 2);
    }}
  }} catch(e) {{
    pre.textContent = 'Error: ' + e.message;
  }}
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


def _setup_tailscale_serve(ports: list[int]):
    """Configure Tailscale HTTPS reverse proxy for the given local ports.

    Uses `tailscale serve --bg` to map each port to HTTPS on the machine's
    Tailscale FQDN.  Requires Tailscale to be running and `tailscale serve`
    to be available (MagicDNS + HTTPS enabled on the tailnet).
    """
    import shutil
    import subprocess

    if not shutil.which("tailscale"):
        logger.warning("tailscale CLI not found — skipping HTTPS setup")
        return

    # Get this machine's Tailscale FQDN
    try:
        result = subprocess.run(
            ["tailscale", "status", "--json"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            logger.warning("tailscale status failed — is Tailscale running?")
            return
        import json as _json
        ts_status = _json.loads(result.stdout)
        fqdn = ts_status.get("Self", {}).get("DNSName", "").rstrip(".")
        if not fqdn:
            logger.warning("Could not determine Tailscale FQDN")
            return
    except Exception as e:
        logger.warning(f"Failed to query Tailscale status: {e}")
        return

    for port in ports:
        try:
            result = subprocess.run(
                ["tailscale", "serve", "--bg", "--https", str(port),
                 f"http://127.0.0.1:{port}"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                logger.info(f"Tailscale HTTPS: https://{fqdn}:{port} -> http://127.0.0.1:{port}")
            else:
                logger.warning(f"tailscale serve failed for port {port}: {result.stderr.strip()}")
        except Exception as e:
            logger.warning(f"Failed to set up Tailscale serve for port {port}: {e}")

    logger.info(f"Tailscale FQDN: {fqdn}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Codex Proxy — multi-provider LLM gateway")
    parser.add_argument("--mode", choices=["proxy", "mcp"], default="proxy",
                        help="proxy: SkyrimNet-compatible OpenAI proxy on port 8539 (default). "
                             "mcp: MCP tool server for general use.")
    parser.add_argument("--transport", choices=["stdio", "sse"], default="stdio",
                        help="MCP transport: stdio for Claude Desktop/CLI (default), "
                             "sse for HTTP SSE on port 8432.")
    parser.add_argument("--tailscale", action="store_true",
                        help="Set up Tailscale HTTPS reverse proxy (requires Tailscale with HTTPS enabled)")
    args = parser.parse_args()

    if args.mode == "mcp":
        from mcp_server import run_mcp_server
        if args.tailscale and args.transport == "sse":
            _setup_tailscale_serve([8432])
        run_mcp_server(transport=args.transport)
    else:
        if args.tailscale:
            _setup_tailscale_serve([8539])
        uvicorn.run(app, host="0.0.0.0", port=8539, http="httptools", access_log=False)
