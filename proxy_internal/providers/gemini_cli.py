"""Gemini Code Assist (Gemini CLI OAuth) provider."""

import json
import logging
import time
import uuid
from typing import Optional

from fastapi import HTTPException

from proxy_internal.providers.antigravity import (
    _convert_messages_to_antigravity,
    _extract_antigravity_text,
)

logger = logging.getLogger("proxy")


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
    import proxy
    # Retry policy: not wrapped — see _with_retry docstring.
    if not proxy.gemini_auth.is_ready or proxy.gemini_auth.is_expired():
        if not await proxy.gemini_auth.refresh_if_needed():
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
    if proxy.gemini_auth.project_id:
        payload["project"] = proxy.gemini_auth.project_id

    url = f"{proxy.GEMINI_CODE_ASSIST_ENDPOINT}/{proxy.GEMINI_CODE_ASSIST_API_VERSION}:generateContent"
    headers = proxy.gemini_auth.get_auth_headers()
    session = proxy.gemini_auth.session or proxy.create_session()
    owns_session = not proxy.gemini_auth.session

    async def _do_post():
        return await session.post(url, json=payload, headers=headers)

    try:
        async with await _do_post() as resp:
            if resp.status == 401:
                if await proxy.gemini_auth.refresh_if_needed():
                    headers.update(proxy.gemini_auth.get_auth_headers())
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
    import proxy
    _mcp = getattr(proxy, "MCP_MODE", False)
    # Retry policy: not wrapped — see _with_retry docstring.
    if not proxy.gemini_auth.is_ready or proxy.gemini_auth.is_expired():
        if not await proxy.gemini_auth.refresh_if_needed():
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
    if proxy.gemini_auth.project_id:
        payload["project"] = proxy.gemini_auth.project_id

    url = f"{proxy.GEMINI_CODE_ASSIST_ENDPOINT}/{proxy.GEMINI_CODE_ASSIST_API_VERSION}:streamGenerateContent?alt=sse"
    headers = proxy.gemini_auth.get_auth_headers()
    headers["Accept"] = "text/event-stream"
    session = proxy.gemini_auth.session or proxy.create_session()
    owns_session = not proxy.gemini_auth.session

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
                            # Skip thought blocks in proxy mode (MCP mode preserves reasoning)
                            if not _mcp and (part.get("thought") or part.get("thoughtSignature")):
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
                                    if not _mcp and (part.get("thought") or part.get("thoughtSignature")):
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
