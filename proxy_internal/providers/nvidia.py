"""NVIDIA NIM OpenAI-compatible provider."""

import asyncio
import logging
import time
import uuid
from typing import Optional

import aiohttp
from fastapi import HTTPException

from proxy_internal.message_normalize import (
    _extract_oai_content,
    build_oai_messages,
)
from proxy_internal.sse_utils import yield_sse_error

logger = logging.getLogger("proxy")


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


_NVIDIA_UNSUPPORTED_PARAMS = {"reasoning", "thinking", "top_k", "provider"}

_reasoning_strip_warned = False


def _nvidia_payload_fixup(payload: dict, extra_params: dict) -> None:
    """Apply NVIDIA NIM-specific payload adjustments.

    - Strips unsupported params (reasoning, top_k, provider).
    """
    import proxy
    global _reasoning_strip_warned
    if not _reasoning_strip_warned and proxy.reasoning_override_enabled:
        _reasoning_strip_warned = True
        logger.info(
            "Reasoning override active — 'thinking' param stripped for NVIDIA NIM "
            "(upstream does not support it); 'reasoning_effort' forwarded"
        )
    payload.update({k: v for k, v in extra_params.items()
                    if v is not None and k not in _NVIDIA_UNSUPPORTED_PARAMS})


async def call_nvidia_direct(system_prompt: Optional[str], messages: list, model: str, max_tokens: int, **extra_params) -> str:
    """Forward request to NVIDIA NIM (OpenAI-compatible), collect full response."""
    import proxy
    if not proxy.nvidia_api_key:
        raise HTTPException(status_code=503, detail="NVIDIA NIM API key not configured — add it via /config/nvidia-key")

    api_model = _resolve_nvidia_model(model)
    endpoint = f"{proxy.NVIDIA_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {proxy.nvidia_api_key}", "Content-Type": "application/json"}

    payload = {"model": api_model, "messages": build_oai_messages(system_prompt, messages), "max_tokens": max_tokens}
    _nvidia_payload_fixup(payload, extra_params)

    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] -> NVIDIA NIM ({api_model}, {len(messages)} msgs)")
    start = time.time()

    session = proxy.third_party_session or proxy.auth.session or proxy.create_session()
    owns_session = session is not proxy.third_party_session and session is not proxy.auth.session

    async def _do_call() -> str:
        async with session.post(endpoint, json=payload, headers=headers) as resp:
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
            return content

    try:
        try:
            content = await proxy._with_retry(_do_call, operation=f"NVIDIA NIM {api_model}", request_id=request_id)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="NVIDIA NIM request timed out")
        elapsed = time.time() - start
        logger.info(f"[{request_id}] <- {len(content)} chars ({elapsed:.1f}s, NVIDIA NIM)")
        return content
    finally:
        if owns_session:
            await session.close()


async def call_nvidia_streaming(system_prompt: Optional[str], messages: list, model: str, max_tokens: int, **extra_params):
    """Forward request to NVIDIA NIM with streaming, passthrough SSE directly."""
    import proxy
    if not proxy.nvidia_api_key:
        err, done = yield_sse_error(model, "[NVIDIA NIM Error: API key not configured]")
        yield err; yield done
        return

    api_model = _resolve_nvidia_model(model)
    endpoint = f"{proxy.NVIDIA_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {proxy.nvidia_api_key}", "Content-Type": "application/json"}

    payload = {"model": api_model, "messages": build_oai_messages(system_prompt, messages), "stream": True}
    _nvidia_payload_fixup(payload, extra_params)

    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] -> NVIDIA NIM ({api_model}, {len(messages)} msgs, stream)")
    start = time.time()

    session = proxy.third_party_session or proxy.auth.session or proxy.create_session()
    owns_session = session is not proxy.third_party_session and session is not proxy.auth.session
    resp_cm = None
    try:
        try:
            resp, resp_cm = await proxy._open_stream_with_retry(
                session, endpoint,
                json_payload=payload, headers=headers,
                operation="NVIDIA NIM", request_id=request_id,
            )
        except HTTPException as e:
            err, done = yield_sse_error(model, f"[NVIDIA NIM Error {e.status_code}]")
            yield err; yield done
            return

        try:
            saw_done = False
            async for event in proxy.passthrough_sse(resp, request_id, "NVIDIA NIM", start):
                if event.strip().startswith("data: [DONE]"):
                    saw_done = True
                yield event

            if not saw_done:
                yield "data: [DONE]\n\n"
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            logger.error(f"[{request_id}] NVIDIA NIM streaming error: {e}")
            err, done = yield_sse_error(model, f"[NVIDIA NIM Error: {e}]")
            yield err; yield done
    finally:
        if resp_cm is not None:
            try:
                await resp_cm.__aexit__(None, None, None)
            except Exception:
                pass
        if owns_session:
            await session.close()
