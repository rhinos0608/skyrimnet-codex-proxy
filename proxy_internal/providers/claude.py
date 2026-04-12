"""Claude native (Anthropic) provider — uses MITM-captured auth headers."""

import asyncio
import copy
import json
import logging
import time
import uuid
from typing import Optional, Union

import aiohttp
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from proxy_internal._json import json_dumps
from proxy_internal.message_normalize import _sanitize_claude_template
from proxy_internal.sse_utils import yield_sse_error

logger = logging.getLogger("proxy")

# Body fields Anthropic rejects from third-party clients — the MITM template
# strips them server-side during auth capture, and we mirror that here so the
# passthrough body matches what the Claude CLI would actually send.
_PASSTHROUGH_STRIP_FIELDS = {"metadata"}


def _build_api_body(system_prompt: Optional[str], messages: list, model: str, max_tokens: int = 0) -> dict:
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

    # 3. Model, streaming, and thinking
    body["model"] = model
    body["stream"] = True
    # When the dashboard reasoning override is active, inject the override
    # thinking config. Otherwise strip thinking to prevent chain-of-thought
    # tokens that get scrubbed server-side, leaving 0-char responses.
    if proxy.reasoning_override_enabled:
        body["thinking"] = proxy._override_thinking_payload(max_tokens)
    else:
        body.pop("thinking", None)
    return body


async def call_api_direct(system_prompt: Optional[str], messages: list, model: str, max_tokens: int) -> str:
    """Direct API call, collects full response (non-streaming to caller)."""
    import proxy
    # Retry policy: not wrapped — see _with_retry docstring.
    body = _build_api_body(system_prompt, messages, model, max_tokens)
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
    body = _build_api_body(system_prompt, messages, model, max_tokens)
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


# ---------------------------------------------------------------------------
# /v1/messages native passthrough (Route B)
#
# Used by the /v1/messages endpoint when the caller supplies ``tools`` and
# the requested model routes to Claude (the default provider). The structured
# OAI path (``_anthropic_messages_structured``) cannot be used for Claude
# because Anthropic models do not speak the OpenAI tool_calls shape, and the
# legacy text-only fallback drops structured tool dispatch entirely. This
# helper sits between the two: it preserves the caller's Anthropic-shaped
# body verbatim (including tools / tool_use / tool_result blocks) and
# forwards it to api.anthropic.com with the MITM-captured billing + auth
# blocks injected so the upstream actually bills the Claude Max subscription.
# ---------------------------------------------------------------------------


def _inject_billing_and_auth(body: dict) -> dict:
    """Prepend cached billing + system-reminder auth blocks into ``body``.

    Works on an Anthropic-shaped ``body`` (already has ``system`` as string or
    list + ``messages`` as a list of role/content dicts). The cached billing
    block is prepended to ``system`` so Anthropic's routing treats this as a
    first-party Claude Code call; the cached auth blocks (system-reminder
    text) are prepended to the first user message's content array so the
    warm-up fingerprint matches what the CLI would send.
    """
    import proxy

    out = copy.deepcopy(body)

    # Strip fields Anthropic rejects from third-party clients.
    for field in _PASSTHROUGH_STRIP_FIELDS:
        out.pop(field, None)

    # 1. Normalise system -> list[content_block] so we can prepend billing.
    billing = proxy._cached_billing_block
    system = out.get("system")
    system_list: list = []
    if isinstance(system, list):
        system_list = list(system)
    elif isinstance(system, str) and system:
        system_list = [{"type": "text", "text": system}]
    if billing is not None:
        system_list = [billing] + system_list
    if system_list:
        out["system"] = system_list
    else:
        out.pop("system", None)

    # 2. Prepend auth blocks to the first user message's content array.
    auth_blocks = proxy._cached_auth_blocks or []
    if auth_blocks:
        messages = out.get("messages") or []
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict) or msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, str):
                new_content = list(auth_blocks) + [{"type": "text", "text": content}]
            elif isinstance(content, list):
                new_content = list(auth_blocks) + list(content)
            else:
                # Unknown shape — don't try to patch, let Anthropic 400 cleanly.
                break
            messages[i] = {**msg, "content": new_content}
            break
        out["messages"] = messages

    return out


async def call_claude_messages_passthrough(body: dict, stream: bool):
    """Forward an Anthropic-shaped ``/v1/messages`` body to api.anthropic.com.

    Used by the dashboard's /v1/messages endpoint when tools are requested
    against a Claude-routed model. Returns a ``StreamingResponse`` when
    ``stream=True`` (forwards SSE bytes verbatim, no re-encoding) or the
    parsed JSON dict when ``stream=False``.
    """
    import proxy

    if not proxy.auth.is_ready:
        raise HTTPException(status_code=503, detail="Claude auth not ready")

    patched = _inject_billing_and_auth(body)
    # Apply reasoning override if active — Route B should respect the same
    # dashboard toggle as the normal Claude path.
    if proxy.reasoning_override_enabled:
        passthrough_max = patched.get("max_tokens", 0)
        patched["thinking"] = proxy._override_thinking_payload(passthrough_max)
    patched["stream"] = bool(stream)

    headers = dict(proxy.auth.headers or {})
    headers.pop("Content-Length", None)  # aiohttp will recompute after serialisation
    body_bytes = json_dumps(patched)
    headers["Content-Length"] = str(len(body_bytes))

    request_id = uuid.uuid4().hex[:8]
    model = patched.get("model", "?")
    msg_count = len(patched.get("messages") or [])
    logger.info(
        f"[{request_id}] -> {model} (/v1/messages passthrough, {msg_count} msgs, "
        f"stream={stream})"
    )
    start = time.time()

    session = proxy.auth.session or proxy.create_session()
    owns_session = proxy.auth.session is None
    url = "https://api.anthropic.com/v1/messages?beta=true"

    if not stream:
        last_error = None
        try:
            for attempt in range(3):
                async with session.post(url, data=body_bytes, headers=headers) as resp:
                    raw = await resp.read()
                    elapsed = time.time() - start
                    if resp.status != 200:
                        err_text = raw.decode("utf-8", errors="replace")
                        logger.error(
                            f"[{request_id}] passthrough {resp.status}: {err_text[:300]}"
                        )
                        if resp.status in (401, 403) or "credential" in err_text.lower():
                            logger.warning("Auth expired -- triggering auto-refresh (passthrough)")
                            proxy.auth.headers = None
                            proxy.auth.body_template = None
                            refreshed = await proxy.recapture_claude_auth()
                            if refreshed and attempt < 2:
                                headers = dict(proxy.auth.headers or {})
                                headers.pop("Content-Length", None)
                                body_bytes = json_dumps(patched)
                                headers["Content-Length"] = str(len(body_bytes))
                                session = proxy.auth.session or session
                                start = time.time()
                                continue
                        if resp.status >= 500 and attempt < 2:
                            logger.info(f"[{request_id}] Retrying passthrough after {resp.status} (attempt {attempt + 1}/3)")
                            last_error = HTTPException(status_code=resp.status, detail=err_text[:500])
                            await asyncio.sleep(2 ** attempt)
                            start = time.time()
                            continue
                        raise HTTPException(status_code=resp.status, detail=err_text[:500])
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        logger.error(f"[{request_id}] passthrough returned non-JSON body")
                        raise HTTPException(status_code=502, detail="Upstream returned non-JSON body")
                    logger.info(
                        f"[{request_id}] <- passthrough done ({elapsed:.1f}s, {len(raw)} bytes)"
                    )
                    return data
            raise last_error
        finally:
            if owns_session:
                await session.close()

    async def _stream_iter():
        sess = session
        _headers = dict(headers)
        _body_bytes = body_bytes
        _start = start
        try:
            for attempt in range(3):
                async with sess.post(url, data=_body_bytes, headers=_headers) as resp:
                    if resp.status != 200:
                        err_body = await resp.read()
                        err_text = err_body.decode("utf-8", errors="replace")
                        logger.error(
                            f"[{request_id}] passthrough stream {resp.status}: {err_text[:300]}"
                        )
                        if resp.status in (401, 403) or "credential" in err_text.lower():
                            logger.warning("Auth expired -- triggering auto-refresh (passthrough stream)")
                            proxy.auth.headers = None
                            proxy.auth.body_template = None
                            refreshed = await proxy.recapture_claude_auth()
                            if refreshed and attempt < 2:
                                _headers = dict(proxy.auth.headers or {})
                                _headers.pop("Content-Length", None)
                                _body_bytes = json_dumps(patched)
                                _headers["Content-Length"] = str(len(_body_bytes))
                                sess = proxy.auth.session or sess
                                _start = time.time()
                                continue
                        if resp.status >= 500 and attempt < 2:
                            logger.info(f"[{request_id}] Retrying passthrough stream after {resp.status} (attempt {attempt + 1}/3)")
                            await asyncio.sleep(2 ** attempt)
                            _start = time.time()
                            continue
                        # Emit an Anthropic-format SSE error + message_stop so the
                        # caller sees a clean close rather than a hung connection.
                        error_evt = {
                            "type": "error",
                            "error": {
                                "type": "upstream_error",
                                "message": err_text[:500],
                            },
                        }
                        yield f"event: error\ndata: {json.dumps(error_evt)}\n\n".encode("utf-8")
                        yield b"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
                        return
                    # Verbatim byte passthrough — the body is already Anthropic SSE.
                    total = 0
                    async for chunk in resp.content.iter_any():
                        total += len(chunk)
                        yield chunk
                    elapsed = time.time() - _start
                    logger.info(
                        f"[{request_id}] <- passthrough stream done ({elapsed:.1f}s, {total} bytes)"
                    )
                    return
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            logger.error(f"[{request_id}] passthrough stream error: {e}")
            error_evt = {"type": "error", "error": {"type": "network_error", "message": str(e)}}
            yield f"event: error\ndata: {json.dumps(error_evt)}\n\n".encode("utf-8")
            yield b"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
        finally:
            if owns_session:
                await sess.close()

    return StreamingResponse(
        _stream_iter(),
        media_type="text/event-stream",
        headers=proxy._STREAMING_HEADERS,
    )
