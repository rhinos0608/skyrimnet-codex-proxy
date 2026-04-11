"""Claude native (Anthropic) provider — uses MITM-captured auth headers."""

import asyncio
import json
import logging
import time
import uuid
from typing import Optional

import aiohttp
from fastapi import HTTPException

from proxy_internal._json import json_dumps
from proxy_internal.message_normalize import _sanitize_claude_template
from proxy_internal.sse_utils import yield_sse_error

logger = logging.getLogger("proxy")


def _build_api_body(system_prompt: Optional[str], messages: list, model: str) -> dict:
    """Build Anthropic API request body from template."""
    import proxy
    # Shallow copy — we replace system/messages/model in-place so the template is never mutated.
    body = _sanitize_claude_template(proxy.auth.body_template)

    # 1. Build system array: billing block (cached) + optional user system prompt
    system: list = []
    if proxy._cached_billing_block is not None:
        system.append(proxy._cached_billing_block)
    if system_prompt:
        system.append({"type": "text", "text": system_prompt})
    body["system"] = system

    # 2. Build messages: auth blocks (cached) prepended to first user message
    new_messages = []
    for i, m in enumerate(messages):
        if i == 0 and m["role"] == "user":
            content = proxy._cached_auth_blocks + [{"type": "text", "text": m["content"]}]
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
    import proxy
    # Retry policy: not wrapped — see _with_retry docstring.
    body = _build_api_body(system_prompt, messages, model)
    headers = dict(proxy.auth.headers)
    body_bytes = json_dumps(body)
    headers["Content-Length"] = str(len(body_bytes))

    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] -> {model} ({len(messages)} msgs)")
    start = time.time()

    session = proxy.auth.session or proxy.create_session()
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
                        proxy.auth.headers = None
                        proxy.auth.body_template = None
                        refreshed = await proxy.recapture_claude_auth()
                        if refreshed and attempt < 2:
                            headers = dict(proxy.auth.headers)
                            headers["Content-Length"] = str(len(body_bytes))
                            session = proxy.auth.session or session
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
        if not proxy.auth.session:
            await session.close()


async def call_api_streaming(system_prompt: Optional[str], messages: list, model: str, max_tokens: int):
    """Direct API call, yields OpenAI-format SSE chunks as they arrive."""
    import proxy
    # Retry policy: not wrapped — see _with_retry docstring.
    body = _build_api_body(system_prompt, messages, model)
    headers = dict(proxy.auth.headers)
    body_bytes = json_dumps(body)
    headers["Content-Length"] = str(len(body_bytes))

    request_id = uuid.uuid4().hex[:8]
    cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    created = int(time.time())
    logger.info(f"[{request_id}] -> {model} ({len(messages)} msgs, stream)")
    start = time.time()
    total_chars = 0

    session = proxy.auth.session or proxy.create_session()
    owns_session = not proxy.auth.session
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
                        proxy.auth.headers = None
                        proxy.auth.body_template = None
                        refreshed = await proxy.recapture_claude_auth()
                        if refreshed and attempt < 2:
                            headers = dict(proxy.auth.headers)
                            headers["Content-Length"] = str(len(body_bytes))
                            session = proxy.auth.session or session
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
