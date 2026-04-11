"""Fireworks AI OpenAI-compatible provider."""

import asyncio
import json
import logging
import time
import uuid
from typing import Optional

import aiohttp
from fastapi import HTTPException

from proxy_internal.message_normalize import (
    _extract_oai_content,
    _is_reasoning_truncated,
    build_oai_messages,
)
from proxy_internal.sse_utils import yield_sse_error

logger = logging.getLogger("proxy")


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
    import proxy
    content_parts = []
    async for event in proxy.call_fireworks_streaming(system_prompt, messages, model, max_tokens, **extra_params):
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
    import proxy
    if not proxy.fireworks_api_key:
        raise HTTPException(status_code=503, detail="Fireworks API key not configured — add it via /config/fireworks-key")

    # Fireworks requires streaming for max_tokens > 4096
    if max_tokens > 4096:
        return await _call_fireworks_direct_via_streaming(system_prompt, messages, model, max_tokens, **extra_params)

    api_model = _resolve_fireworks_model(model)
    endpoint = f"{proxy.FIREWORKS_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {proxy.fireworks_api_key}", "Content-Type": "application/json"}

    payload = {"model": api_model, "messages": build_oai_messages(system_prompt, messages), "max_tokens": max_tokens}
    _fireworks_payload_fixup(payload, extra_params)

    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] -> Fireworks ({api_model}, {len(messages)} msgs)")
    start = time.time()

    session = proxy.third_party_session or proxy.auth.session or proxy.create_session()
    owns_session = session is not proxy.third_party_session and session is not proxy.auth.session

    async def _do_call() -> str:
        async with session.post(endpoint, json=payload, headers=headers) as resp:
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
                                return content
                logger.warning(f"[{request_id}] Fireworks returned null content, full response: {str(data)[:500]}")
                raise HTTPException(status_code=500, detail="Fireworks returned empty content")
            return content

    try:
        try:
            content = await proxy._with_retry(_do_call, operation=f"Fireworks {api_model}", request_id=request_id)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Fireworks request timed out")
        elapsed = time.time() - start
        logger.info(f"[{request_id}] <- {len(content)} chars ({elapsed:.1f}s, Fireworks)")
        return content
    finally:
        if owns_session:
            await session.close()


async def call_fireworks_streaming(system_prompt: Optional[str], messages: list, model: str, max_tokens: int, **extra_params):
    """Forward request to Fireworks with streaming, passthrough SSE directly."""
    import proxy
    if not proxy.fireworks_api_key:
        err, done = yield_sse_error(model, "[Fireworks Error: API key not configured]")
        yield err; yield done
        return

    api_model = _resolve_fireworks_model(model)
    endpoint = f"{proxy.FIREWORKS_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {proxy.fireworks_api_key}", "Content-Type": "application/json"}

    payload = {"model": api_model, "messages": build_oai_messages(system_prompt, messages), "stream": True}
    _fireworks_payload_fixup(payload, extra_params)

    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] -> Fireworks ({api_model}, {len(messages)} msgs, stream)")
    start = time.time()

    session = proxy.third_party_session or proxy.auth.session or proxy.create_session()
    owns_session = session is not proxy.third_party_session and session is not proxy.auth.session
    resp_cm = None
    try:
        try:
            resp, resp_cm = await proxy._open_stream_with_retry(
                session, endpoint,
                json_payload=payload, headers=headers,
                operation="Fireworks", request_id=request_id,
            )
        except HTTPException as e:
            err, done = yield_sse_error(model, f"[Fireworks Error {e.status_code}]")
            yield err; yield done
            return

        try:
            saw_done = False
            async for event in proxy.passthrough_sse(resp, request_id, "Fireworks", start):
                if event.strip().startswith("data: [DONE]"):
                    saw_done = True
                yield event

            if not saw_done:
                yield "data: [DONE]\n\n"
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            logger.error(f"[{request_id}] Fireworks streaming error: {e}")
            err, done = yield_sse_error(model, f"[Fireworks Error: {e}]")
            yield err; yield done
    finally:
        if resp_cm is not None:
            try:
                await resp_cm.__aexit__(None, None, None)
            except Exception:
                pass
        if owns_session:
            await session.close()
