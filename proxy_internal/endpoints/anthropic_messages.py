"""Anthropic /v1/messages bridge endpoints."""
import json
import logging
import time
import uuid

import aiohttp
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

router = APIRouter()
logger = logging.getLogger("proxy")


@router.post("/v1/messages")
async def anthropic_messages(request: Request):
    """Anthropic Messages API compatibility endpoint.

    Two code paths:
      1. **Structured tool-use path** — taken when the caller supplies ``tools``
         AND the model routes to an OpenAI-compatible provider.  Preserves
         ``tool_use`` / ``tool_result`` blocks on both request and response so
         clients like Claude Code can dispatch tools end-to-end.
      2. **Legacy text-only path** — taken when no tools are requested (or the
         request is fed to a non-OAI provider).  Flattens everything through
         ``_chat_completions_inner`` and wraps the result as an Anthropic
         Message.  Requests with ``tools`` targeting a non-OAI provider are
         soft-fallback'd onto this path with a warning logged: tool definitions
         are dropped and the model produces a plain text reply, which keeps
         clients like CCS working when they tier-route to claude-*/gpt-5.*/
         antigravity-* models that can't dispatch tools natively through this
         proxy.
    """
    import proxy

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")

    request_model = body.get("model") or proxy.DEFAULT_MODEL
    resolved_model = proxy.normalize_model_name(request_model)
    is_stream = bool(body.get("stream"))
    has_tools = bool(body.get("tools"))

    # Route 1: structured tool_use passthrough.
    if has_tools:
        try:
            resolved = await proxy._resolve_oai_compatible(resolved_model)
        except HTTPException:
            # Explicit resolver failures (missing API key, etc.) still propagate.
            raise
        if resolved is not None:
            return await proxy._anthropic_messages_structured(body, resolved, resolved_model, is_stream)
        # Soft-fallback: the requested model isn't OAI-compatible so we can't
        # dispatch tools natively.  Flatten tools into the system prompt via the
        # legacy text-only pipeline so the conversation keeps flowing under
        # clients like CCS that tier-route to claude-*/gpt-5.*/antigravity-*
        # models.  The assistant won't emit structured tool_use blocks — the
        # caller gets a text response summarising what it would do.
        logger.warning(
            "[/v1/messages] Soft-fallback: tools present but model %r does not "
            "route to an OAI-compatible provider (%s). Flattening through the "
            "text-only path; tool dispatch is not available on this model.",
            resolved_model, ", ".join(proxy._OAI_COMPATIBLE_PROVIDERS),
        )
        # Keep ``tools`` in the body: _anthropic_request_to_chat_request calls
        # _anthropic_tools_to_system_hint() which folds the tool catalogue into
        # the system prompt as a human-readable bullet list, so the model still
        # knows what capabilities exist even though it can't emit structured
        # tool_use blocks.  We only drop ``tool_choice``, which is an OpenAI-
        # specific dispatch directive whose semantics don't apply on the text
        # path.  tool_use / tool_result blocks inside ``messages`` are already
        # flattened to readable text by _anthropic_flatten_content().
        body.pop("tool_choice", None)

    chat_req = proxy._anthropic_request_to_chat_request(body)

    # Rough prompt-token estimate for the message_start usage field.
    try:
        prompt_chars = sum(len(m.content) for m in chat_req.messages if isinstance(m.content, str))
    except Exception:
        prompt_chars = 0
    prompt_tokens_hint = prompt_chars // 4

    try:
        inner_result = await proxy._chat_completions_inner(chat_req)
    except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError,
            aiohttp.ClientOSError, aiohttp.ServerDisconnectedError):
        logger.debug("Client disconnected mid-/v1/messages — suppressing traceback")
        raise HTTPException(status_code=499, detail="Client disconnected")

    if is_stream:
        if not isinstance(inner_result, StreamingResponse):
            # Provider did not honour stream=True; wrap the dict result into a
            # single-shot Anthropic stream so the client still gets valid events.
            async def _oneshot():
                text = ""
                if isinstance(inner_result, dict):
                    choices = inner_result.get("choices") or []
                    if choices:
                        text = (choices[0].get("message") or {}).get("content") or ""
                # Re-emit as a minimal OpenAI SSE for our wrapper to parse.
                chunk = {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request_model,
                    "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                done_chunk = {
                    "id": chunk["id"], "object": "chat.completion.chunk",
                    "created": chunk["created"], "model": request_model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(done_chunk)}\n\n"
                yield "data: [DONE]\n\n"
            source_iter = _oneshot()
        else:
            source_iter = inner_result.body_iterator

        return StreamingResponse(
            proxy._anthropic_stream_from_openai(source_iter, request_model, prompt_tokens_hint),
            media_type="text/event-stream",
            headers=proxy._STREAMING_HEADERS,
        )

    if not isinstance(inner_result, dict):
        raise HTTPException(status_code=500, detail="Unexpected upstream response type")
    return proxy._openai_completion_to_anthropic_message(inner_result, request_model)


@router.post("/v1/messages/count_tokens")
async def anthropic_count_tokens(request: Request):
    """Stub for Anthropic's token-count endpoint.

    Claude Code calls this during startup to size the prompt budget.  We return
    a coarse chars/4 estimate so the client can proceed — exact token counts
    would require per-model tokenisers this proxy does not ship.
    """
    import proxy

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    total_chars = 0
    system_text = proxy._anthropic_flatten_system(body.get("system")) or ""
    total_chars += len(system_text)
    for msg in body.get("messages") or []:
        if isinstance(msg, dict):
            total_chars += len(proxy._anthropic_flatten_content(msg.get("content")))
    # Tool definitions add prompt weight too.
    tools = body.get("tools") or []
    if isinstance(tools, list):
        for tool in tools:
            if isinstance(tool, dict):
                total_chars += len(tool.get("name", "")) + len(tool.get("description") or "")
                schema = tool.get("input_schema")
                if schema:
                    try:
                        total_chars += len(json.dumps(schema))
                    except (TypeError, ValueError):
                        pass
    return {"input_tokens": max(1, total_chars // 4)}
