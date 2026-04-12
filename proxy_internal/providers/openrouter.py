"""OpenRouter OpenAI-compatible provider."""

import asyncio
import logging
import time
import uuid
from typing import Optional

import aiohttp
from fastapi import HTTPException

from proxy_internal.message_normalize import build_oai_messages
from proxy_internal.sse_utils import yield_sse_error

logger = logging.getLogger("proxy")

_reasoning_strip_warned = False


async def call_openrouter_direct(system_prompt: Optional[str], messages: list, model: str, max_tokens: int, **extra_params) -> str:
    """Forward request to OpenRouter (OpenAI-compatible), collect full response."""
    import proxy
    if not proxy.openrouter_api_key:
        raise HTTPException(status_code=500, detail="OpenRouter API key not configured")

    payload = {"model": model, "messages": build_oai_messages(system_prompt, messages), "max_tokens": max_tokens}
    global _reasoning_strip_warned
    if not _reasoning_strip_warned and proxy.reasoning_override_enabled:
        _reasoning_strip_warned = True
        logger.info(
            "Reasoning override active — 'thinking' param stripped for OpenRouter "
            "(upstream does not support it); 'reasoning_effort' forwarded"
        )
    payload.update({k: v for k, v in extra_params.items()
                    if v is not None and k not in ("thinking", "reasoning")})
    headers = {"Authorization": f"Bearer {proxy.openrouter_api_key}", "Content-Type": "application/json"}

    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] -> OpenRouter {model} ({len(messages)} msgs)")
    start = time.time()

    session = proxy.third_party_session or proxy.auth.session or proxy.create_session()
    owns_session = session is not proxy.third_party_session and session is not proxy.auth.session

    async def _do_call() -> str:
        async with session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json=payload, headers=headers,
        ) as resp:
            if resp.status != 200:
                error_body = await resp.text()
                logger.error(f"[{request_id}] OpenRouter {resp.status}: {error_body[:300]}")
                raise HTTPException(status_code=resp.status, detail=error_body[:200])
            data = await resp.json()
            choices = data.get("choices") or []
            if not choices:
                raise HTTPException(status_code=502, detail="OpenRouter returned no choices")
            text = (choices[0].get("message") or {}).get("content")
            if text is None:
                raise HTTPException(status_code=502, detail="OpenRouter returned no content")
            return text

    try:
        text = await proxy._with_retry(
            _do_call,
            operation=f"OpenRouter {model}",
            request_id=request_id,
        )
        elapsed = time.time() - start
        logger.info(f"[{request_id}] <- {len(text)} chars ({elapsed:.1f}s, OpenRouter)")
        return text
    finally:
        if owns_session:
            await session.close()


async def call_openrouter_streaming(system_prompt: Optional[str], messages: list, model: str, max_tokens: int, **extra_params):
    """Forward request to OpenRouter with streaming, passthrough SSE directly.

    Retry policy: the pre-first-byte phase (connect + status check) is retried
    via ``_open_stream_with_retry``.  Once bytes are flowing we cannot cleanly
    resume — mid-stream errors are surfaced as SSE error chunks.
    """
    import proxy
    if not proxy.openrouter_api_key:
        yield 'data: {"error": "OpenRouter API key not configured"}\n\n'
        yield "data: [DONE]\n\n"
        return

    payload = {"model": model, "messages": build_oai_messages(system_prompt, messages), "max_tokens": max_tokens, "stream": True}
    # OpenRouter proxies to many upstreams — most don't understand thinking/reasoning.
    # Strip them to avoid 400 errors; response-side scrubbing handles any reasoning tokens.
    global _reasoning_strip_warned
    if not _reasoning_strip_warned and proxy.reasoning_override_enabled:
        _reasoning_strip_warned = True
        logger.info(
            "Reasoning override active — 'thinking' param stripped for OpenRouter "
            "(upstream does not support it); 'reasoning_effort' forwarded"
        )
    payload.update({k: v for k, v in extra_params.items()
                    if v is not None and k not in ("thinking", "reasoning")})
    headers = {"Authorization": f"Bearer {proxy.openrouter_api_key}", "Content-Type": "application/json"}

    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] -> OpenRouter {model} ({len(messages)} msgs, stream)")
    start = time.time()

    session = proxy.third_party_session or proxy.auth.session or proxy.create_session()
    owns_session = session is not proxy.third_party_session and session is not proxy.auth.session
    resp_cm = None
    try:
        try:
            resp, resp_cm = await proxy._open_stream_with_retry(
                session,
                "https://openrouter.ai/api/v1/chat/completions",
                json_payload=payload,
                headers=headers,
                operation="OpenRouter",
                request_id=request_id,
            )
        except HTTPException as e:
            err, done = yield_sse_error(model, f"[OpenRouter Error {e.status_code}]")
            yield err; yield done
            return

        try:
            # OpenRouter returns OpenAI-format SSE — passthrough directly
            saw_done = False
            async for event in proxy.passthrough_sse(resp, request_id, "OpenRouter", start):
                if event.strip().startswith("data: [DONE]"):
                    saw_done = True
                yield event
            if not saw_done:
                yield "data: [DONE]\n\n"
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            logger.error(f"[{request_id}] OpenRouter streaming error: {e}")
            err, done = yield_sse_error(model, f"[OpenRouter Error: {e}]")
            yield err; yield done
    finally:
        if resp_cm is not None:
            try:
                await resp_cm.__aexit__(None, None, None)
            except Exception:
                pass
        if owns_session:
            await session.close()
