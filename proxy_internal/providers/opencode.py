"""OpenCode (Zen + Go) OpenAI-compatible provider."""

import asyncio
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


def _resolve_opencode(model: str):
    """Resolve OpenCode model prefix to (api_model, base_url, api_key, plan_name).

    Guards against double-prefixed models like 'opencode:xiaomi:mimo-v2-pro' —
    if the stripped model still contains another provider prefix, that's a config
    error on the SkyrimNet side.
    """
    import proxy
    if model.lower().startswith("opencode-go:"):
        return model[len("opencode-go:"):], proxy.OPENCODE_GO_URL, proxy.opencode_go_api_key or proxy.opencode_api_key, "Go"
    return model[len("opencode:"):], proxy.OPENCODE_ZEN_URL, proxy.opencode_api_key or proxy.opencode_go_api_key, "Zen"


async def call_opencode_direct(system_prompt: Optional[str], messages: list, model: str, max_tokens: int, **extra_params) -> str:
    """Forward request to OpenCode API (OpenAI-compatible), collect full response."""
    import proxy
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

    session = proxy.third_party_session or proxy.auth.session or proxy.create_session()
    owns_session = session is not proxy.third_party_session and session is not proxy.auth.session

    async def _do_call() -> str:
        async with session.post(endpoint, json=payload, headers=headers) as resp:
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
                                return content
                raise HTTPException(status_code=500, detail="OpenCode returned empty content")
            return content

    try:
        try:
            content = await proxy._with_retry(_do_call, operation=f"OpenCode {api_model}", request_id=request_id)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="OpenCode request timed out")
        elapsed = time.time() - start
        logger.info(f"[{request_id}] <- {len(content)} chars ({elapsed:.1f}s, OpenCode)")
        return content
    finally:
        if owns_session:
            await session.close()


async def call_opencode_streaming(system_prompt: Optional[str], messages: list, model: str, max_tokens: int, **extra_params):
    """Forward request to OpenCode API with streaming, passthrough SSE directly."""
    import proxy
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

    session = proxy.third_party_session or proxy.auth.session or proxy.create_session()
    owns_session = session is not proxy.third_party_session and session is not proxy.auth.session
    resp_cm = None
    try:
        try:
            resp, resp_cm = await proxy._open_stream_with_retry(
                session, endpoint,
                json_payload=payload, headers=headers,
                operation="OpenCode", request_id=request_id,
            )
        except HTTPException as e:
            err, done = yield_sse_error(model, f"[OpenCode Error {e.status_code}]")
            yield err; yield done
            return

        try:
            async for event in proxy.passthrough_sse(resp, request_id, "OpenCode", start):
                yield event
            yield "data: [DONE]\n\n"
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            logger.error(f"[{request_id}] OpenCode streaming error: {e}")
            err, done = yield_sse_error(model, f"[OpenCode Error: {e}]")
            yield err; yield done
    finally:
        if resp_cm is not None:
            try:
                await resp_cm.__aexit__(None, None, None)
            except Exception:
                pass
        if owns_session:
            await session.close()
