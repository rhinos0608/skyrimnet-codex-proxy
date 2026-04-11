"""Ollama OpenAI-compatible provider (Cloud + local daemon)."""

import asyncio
import logging
import time
import uuid
from typing import Optional

import aiohttp
from fastapi import HTTPException

from proxy_internal.message_normalize import (
    _extract_oai_content,
    _has_reasoning_without_content,
    _is_reasoning_truncated,
    build_oai_messages,
)
from proxy_internal.sse_utils import yield_sse_error

logger = logging.getLogger("proxy")


_OLLAMA_UNSUPPORTED_PARAMS = {"top_k"}


def _ollama_payload_fixup(payload: dict, extra_params: dict) -> None:
    """Apply Ollama-specific payload adjustments for OpenAI-compatible calls."""
    payload.update({
        k: v for k, v in extra_params.items()
        if v is not None and k not in _OLLAMA_UNSUPPORTED_PARAMS
    })


async def call_ollama_direct(system_prompt: Optional[str], messages: list, model: str, max_tokens: int, **extra_params) -> str:
    """Forward request to Ollama (OpenAI-compatible), collect full response."""
    import proxy
    api_model = model[len("ollama:"):]
    if proxy.ollama_api_key:
        endpoint = "https://ollama.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {proxy.ollama_api_key}", "Content-Type": "application/json"}
    else:
        endpoint = "http://localhost:11434/v1/chat/completions"
        headers = {"Content-Type": "application/json"}

    payload = {"model": api_model, "messages": build_oai_messages(system_prompt, messages), "max_tokens": max_tokens}
    _ollama_payload_fixup(payload, extra_params)

    request_id = uuid.uuid4().hex[:8]
    target = "Ollama Cloud" if proxy.ollama_api_key else f"Ollama local ({api_model})"
    logger.info(f"[{request_id}] -> {target} ({len(messages)} msgs)")
    start = time.time()

    session = proxy.ollama_session or proxy.auth.session or proxy.create_session()
    owns_session = not proxy.ollama_session and not proxy.auth.session

    async def _do_call() -> str:
        # ClientConnectorError means the local Ollama daemon is down — short
        # circuit so _with_retry doesn't waste attempts on a dead socket.
        try:
            async with session.post(endpoint, json=payload, headers=headers) as resp:
                if resp.status in (401, 403):
                    raise HTTPException(status_code=401, detail="Ollama Cloud auth failed — check API key")
                if resp.status == 429:
                    raise HTTPException(status_code=429, detail="Ollama rate limit exceeded")
                if resp.status != 200:
                    error_body = await resp.text()
                    logger.error(f"[{request_id}] Ollama {resp.status}: {error_body[:300]}")
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
                return text
        except aiohttp.ClientConnectorError:
            # Local daemon down — not transient, surface immediately without retry.
            raise HTTPException(status_code=503, detail="Ollama not running at localhost:11434")

    try:
        text = await proxy._with_retry(
            _do_call,
            operation=f"Ollama {api_model}",
            request_id=request_id,
        )
        elapsed = time.time() - start
        logger.info(f"[{request_id}] <- {len(text)} chars ({elapsed:.1f}s, Ollama)")
        return text
    finally:
        if owns_session:
            await session.close()


async def call_ollama_streaming(system_prompt: Optional[str], messages: list, model: str, max_tokens: int, **extra_params):
    """Forward request to Ollama with streaming, passthrough SSE directly.

    Retry policy: pre-first-byte retries via ``_open_stream_with_retry``.  A
    ``ClientConnectorError`` (local daemon down) is NOT retried — surfaced as
    an SSE error chunk immediately.
    """
    import proxy
    api_model = model[len("ollama:"):]
    if proxy.ollama_api_key:
        endpoint = "https://ollama.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {proxy.ollama_api_key}", "Content-Type": "application/json"}
    else:
        endpoint = "http://localhost:11434/v1/chat/completions"
        headers = {"Content-Type": "application/json"}

    payload = {"model": api_model, "messages": build_oai_messages(system_prompt, messages), "max_tokens": max_tokens, "stream": True}
    _ollama_payload_fixup(payload, extra_params)

    request_id = uuid.uuid4().hex[:8]
    target = "Ollama Cloud" if proxy.ollama_api_key else f"Ollama local ({api_model})"
    logger.info(f"[{request_id}] -> {target} ({len(messages)} msgs, stream)")
    start = time.time()

    session = proxy.ollama_session or proxy.auth.session or proxy.create_session()
    owns_session = not proxy.ollama_session and not proxy.auth.session
    resp_cm = None
    try:
        try:
            resp, resp_cm = await proxy._open_stream_with_retry(
                session,
                endpoint,
                json_payload=payload,
                headers=headers,
                operation="Ollama",
                request_id=request_id,
                retry_connect_errors=False,
            )
        except aiohttp.ClientConnectorError:
            err, done = yield_sse_error(model, "[Ollama Error: not running at localhost:11434]")
            yield err; yield done
            return
        except HTTPException as e:
            err, done = yield_sse_error(model, f"[Ollama Error {e.status_code}]")
            yield err; yield done
            return

        try:
            # Ollama /v1 returns OpenAI-format SSE — passthrough directly
            async for event in proxy.passthrough_sse(resp, request_id, "Ollama", start):
                yield event
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            logger.error(f"[{request_id}] Ollama streaming error: {e}")
            err, done = yield_sse_error(model, f"[Ollama Error: {e}]")
            yield err; yield done
    finally:
        if resp_cm is not None:
            try:
                await resp_cm.__aexit__(None, None, None)
            except Exception:
                pass
        if owns_session:
            await session.close()
