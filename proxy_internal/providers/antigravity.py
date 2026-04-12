"""Antigravity (Google IDE) provider with multi-account OAuth fallback."""

import json
import logging
import time
import uuid
from typing import Optional

from fastapi import HTTPException

logger = logging.getLogger("proxy")


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


def _extract_antigravity_text(data: dict) -> str:
    """Extract text from Antigravity response."""
    import proxy
    text_parts = []
    response = data.get("response", data)
    candidates = response.get("candidates", [])
    _mcp = getattr(proxy, "MCP_MODE", False)
    for candidate in candidates:
        content = candidate.get("content", {})
        parts = content.get("parts", [])
        for part in parts:
            # Skip thought blocks in proxy mode (NPC callers don't see reasoning).
            # In MCP mode, include thought text so general-purpose callers get reasoning.
            if not _mcp and (part.get("thought") or part.get("thoughtSignature")):
                continue
            text = part.get("text", "")
            if text:
                text_parts.append(text)
    return "".join(text_parts)


async def call_antigravity_direct(
    system_prompt: Optional[str],
    messages: list,
    model: str,
    max_tokens: int,
) -> str:
    """Call Antigravity API directly with multi-account fallback on 503 errors."""
    import proxy
    # Retry policy: not wrapped — see _with_retry docstring.
    if not proxy.antigravity_auth.is_ready:
        await proxy.antigravity_auth.refresh_if_needed()
        if not proxy.antigravity_auth.is_ready:
            raise HTTPException(status_code=503, detail="Antigravity auth not ready -- run /config/antigravity-login first")

    ag_model = _get_antigravity_model_id(model)
    request_contents = _convert_messages_to_antigravity(messages, system_prompt)

    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] -> Antigravity {ag_model} ({len(messages)} msgs)")
    start = time.time()

    # Get all available accounts for fallback
    all_accounts = proxy.antigravity_auth._get_active_accounts()
    if not all_accounts:
        raise HTTPException(status_code=503, detail="No Antigravity accounts available")

    last_error = None
    for account in all_accounts:
        if not account.is_ready:
            if not await account.refresh_if_needed():
                continue
            if account.is_ready and not account.session:
                account.session = proxy.create_session()

        if not account.is_ready:
            continue

        # Build payload with this account's project_id
        payload = {
            "project": account.project_id or proxy.ANTIGRAVITY_DEFAULT_PROJECT_ID,
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
        session = account.session or proxy.create_session()
        owns_session = not account.session

        try:
            async with session.post(
                f"{proxy.ANTIGRAVITY_ENDPOINT}/v1internal:generateContent",
                json=payload,
                headers=headers,
            ) as resp:
                elapsed = time.time() - start

                if resp.status != 200:
                    error_body = await resp.text()
                    logger.error(f"[{request_id}] Antigravity {resp.status} (account: {account.email}): {error_body[:300]}")

                    # Handle specific error codes - fallback for transient/recoverable errors
                    if resp.status in (503, 429, 403):  # Service unavailable, rate limited, or permission denied
                        proxy.antigravity_auth.mark_account_error(account, resp.status)
                        last_error = HTTPException(status_code=resp.status, detail=error_body[:200])
                        logger.info(f"[{request_id}] Falling back to next account due to {resp.status}")
                        continue  # Try next account

                    if resp.status == 401:
                        # Token expired, try refresh and retry once
                        if await account.refresh_if_needed():
                            headers = account.get_auth_headers()
                            async with session.post(
                                f"{proxy.ANTIGRAVITY_ENDPOINT}/v1internal:generateContent",
                                json=payload,
                                headers=headers,
                            ) as retry_resp:
                                if retry_resp.status == 200:
                                    data = await retry_resp.json()
                                    proxy.antigravity_auth.mark_account_success(account)
                                    return _extract_antigravity_text(data)
                                elif retry_resp.status in (503, 429, 403):
                                    proxy.antigravity_auth.mark_account_error(account, retry_resp.status)
                                    last_error = HTTPException(status_code=retry_resp.status, detail=error_body[:200])
                                    continue  # Try next account
                        proxy.antigravity_auth.mark_account_error(account, resp.status)
                        last_error = HTTPException(status_code=resp.status, detail=error_body[:200])
                        continue  # Try next account

                    # For other errors, raise immediately
                    raise HTTPException(status_code=resp.status, detail=error_body[:200])

                data = await resp.json()
                response_text = _extract_antigravity_text(data)
                proxy.antigravity_auth.mark_account_success(account)
                logger.info(f"[{request_id}] <- {len(response_text)} chars ({elapsed:.1f}s, Antigravity, account: {account.email})")
                return response_text
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[{request_id}] Antigravity request error for {account.email}: {e}")
            proxy.antigravity_auth.mark_account_error(account, 500)
            last_error = HTTPException(status_code=500, detail=str(e))
        finally:
            if owns_session:
                await session.close()

    # All accounts failed
    if last_error:
        raise last_error
    raise HTTPException(status_code=503, detail="All Antigravity accounts failed")


async def call_antigravity_streaming(
    system_prompt: Optional[str],
    messages: list,
    model: str,
    max_tokens: int,
):
    """Call Antigravity API with streaming and multi-account fallback on initial errors."""
    import proxy
    _mcp = getattr(proxy, "MCP_MODE", False)
    # Retry policy: not wrapped — see _with_retry docstring.
    if not proxy.antigravity_auth.is_ready:
        await proxy.antigravity_auth.refresh_if_needed()
        if not proxy.antigravity_auth.is_ready:
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
    all_accounts = proxy.antigravity_auth._get_active_accounts()
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
                account.session = proxy.create_session()

        if not account.is_ready:
            continue

        # Build payload with this account's project_id
        payload = {
            "project": account.project_id or proxy.ANTIGRAVITY_DEFAULT_PROJECT_ID,
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

        session = account.session or proxy.create_session()
        owns_session = not account.session

        async def make_stream_request(headers_to_use):
            """Make the streaming request, return (resp, should_retry) tuple."""
            resp = await session.post(
                f"{proxy.ANTIGRAVITY_ENDPOINT}/v1internal:streamGenerateContent?alt=sse",
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
                        proxy.antigravity_auth.mark_account_error(account, resp.status)
                        last_error_status = resp.status
                        last_error_msg = error_body[:200]
                        logger.info(f"[{request_id}] Falling back to next account due to {resp.status}")
                        continue  # Try next account

                    if resp.status == 401:
                        proxy.antigravity_auth.mark_account_error(account, resp.status)
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
                                # Skip thought blocks in proxy mode (MCP mode preserves reasoning)
                                if not _mcp and (part.get("thought") or part.get("thoughtSignature")):
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
                                        if not _mcp and (part.get("thought") or part.get("thoughtSignature")):
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

                proxy.antigravity_auth.mark_account_success(account)
                elapsed = time.time() - start
                logger.info(f"[{request_id}] <- {total_chars} chars ({elapsed:.1f}s, Antigravity stream, account: {account.email})")
                return  # Success, exit the generator
        except Exception as e:
            logger.error(f"[{request_id}] Antigravity stream error for {account.email}: {e}")
            proxy.antigravity_auth.mark_account_error(account, 500)
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
