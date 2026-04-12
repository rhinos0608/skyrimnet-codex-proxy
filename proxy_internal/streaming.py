"""Shared streaming helpers for upstream SSE passthrough."""

import asyncio
import json
import logging
import time
from typing import Optional

import aiohttp

from proxy_internal.sse_utils import make_sse_content_chunk

logger = logging.getLogger("proxy")

_REASONING_FIELDS = {"reasoning", "reasoning_content", "reasoning_details"}

_REWRITE_PROMPT = (
    "The following is the internal chain-of-thought reasoning of a game NPC character. "
    "Based on this reasoning, write a short, natural in-character dialogue response "
    "(1-3 sentences maximum). Stay in character. Output ONLY the NPC's spoken dialogue "
    "— no narration, no stage directions, no meta-commentary.\n\n"
    "Reasoning:\n{reasoning}"
)

_REWRITE_MODEL = "fireworks:kimi-k2p5-turbo"
_REWRITE_TIMEOUT = 5.0


def _normalize_reasoning_text(value) -> str:
    """Flatten provider-specific reasoning payloads into plain text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [_normalize_reasoning_text(item) for item in value]
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            return value["text"]
        parts = [_normalize_reasoning_text(item) for item in value.values()]
        flattened = "\n".join(part for part in parts if part)
        return flattened or json.dumps(value, ensure_ascii=False)
    return str(value)


def _event_has_visible_content(event: str) -> bool:
    if not (event.startswith("data: ") and "[DONE]" not in event and '"content"' in event):
        return False
    try:
        data = json.loads(event[6:])
    except json.JSONDecodeError:
        return False
    return bool(data.get("choices", [{}])[0].get("delta", {}).get("content"))


def _is_done_event(event: str) -> bool:
    return event.strip().startswith("data: [DONE]")


def _content_len(event: str) -> int:
    if not _event_has_visible_content(event):
        return 0
    try:
        data = json.loads(event[6:])
        return len(data.get("choices", [{}])[0].get("delta", {}).get("content") or "")
    except (json.JSONDecodeError, KeyError, IndexError):
        return 0


def _scrub_event(event_bytes: bytes) -> tuple[Optional[str], Optional[str]]:
    """Return (sanitized_event, hidden_reasoning_text)."""
    if not event_bytes:
        return None, None

    event = event_bytes.decode("utf-8", errors="replace").strip()
    if not event:
        return None, None
    if not event.startswith("data: ") or _is_done_event(event):
        return event + "\n\n", None

    if b'"reasoning' not in event_bytes:
        return event + "\n\n", None

    try:
        data = json.loads(event[6:])
        delta = data.get("choices", [{}])[0].get("delta", {})
        content = delta.get("content")
        reasoning_text = None
        modified = False
        for field in ("reasoning_content", "reasoning", "reasoning_details"):
            if field not in delta:
                continue
            value = delta.pop(field)
            if value is not None:
                modified = True
                if reasoning_text is None:
                    reasoning_text = _normalize_reasoning_text(value)
        if content:
            return (f"data: {json.dumps(data)}\n\n" if modified else event + "\n\n"), None
        if modified:
            return None, reasoning_text
        return event + "\n\n", None
    except (json.JSONDecodeError, KeyError, IndexError):
        return event + "\n\n", None


async def passthrough_sse(resp, request_id: str, provider_name: str, start: float):
    """Passthrough OpenAI-format SSE from an upstream response, yielding events.

    Used by OpenRouter, Ollama, Z.AI, OpenCode, Qwen — all OpenAI-compatible SSE.
    In proxy mode: strips reasoning fields from mixed chunks and drops reasoning-only chunks.
    In MCP mode: passes events through verbatim (reasoning preserved for general-purpose callers).
    """
    import proxy as _proxy
    _mcp = getattr(_proxy, "MCP_MODE", False)

    buffer = bytearray()
    total_content = 0
    warned_reasoning_drop = False

    async for raw_chunk in resp.content.iter_any():
        buffer.extend(raw_chunk)
        while True:
            idx = buffer.find(b"\n\n")
            if idx < 0:
                break
            event_bytes = bytes(buffer[:idx]).strip()
            del buffer[:idx + 2]
            if _mcp:
                emitted = event_bytes.decode("utf-8", errors="replace").strip() + "\n\n" if event_bytes else None
                reasoning_text = None
            else:
                emitted, reasoning_text = _scrub_event(event_bytes)
            if reasoning_text and not warned_reasoning_drop:
                warned_reasoning_drop = True
                logger.warning(f"[{request_id}] {provider_name} returned reasoning-only chunks, dropping hidden reasoning")
            if emitted is None:
                continue
            total_content += _content_len(emitted)
            yield emitted

    if buffer.strip():
        event_bytes = bytes(buffer).strip()
        if _mcp:
            emitted = event_bytes.decode("utf-8", errors="replace").strip() + "\n\n"
            reasoning_text = None
        else:
            emitted, reasoning_text = _scrub_event(event_bytes)
        if reasoning_text and not warned_reasoning_drop:
            warned_reasoning_drop = True
            logger.warning(f"[{request_id}] {provider_name} returned reasoning-only chunks, dropping hidden reasoning")
        if emitted is not None:
            total_content += _content_len(emitted)
            yield emitted

    elapsed = time.time() - start
    if total_content == 0:
        logger.warning(f"[{request_id}] <- stream done ({elapsed:.1f}s, {provider_name}, 0 chars) — model returned only reasoning tokens, no content")
    else:
        logger.info(f"[{request_id}] <- stream done ({elapsed:.1f}s, {provider_name}, {total_content} chars)")


async def _rewrite_reasoning_to_dialogue(
    reasoning_text: str, system_prompt: Optional[str], request_id: str, model: str,
) -> Optional[str]:
    """Use a fast LLM to rewrite reasoning text into in-character NPC dialogue."""
    import proxy
    if not proxy.fireworks_api_key:
        logger.warning(f"[{request_id}] reasoning rewrite skipped — Fireworks API key not configured")
        return None

    from proxy_internal.providers.fireworks import _resolve_fireworks_model

    api_model = _resolve_fireworks_model(_REWRITE_MODEL)
    endpoint = f"{proxy.FIREWORKS_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {proxy.fireworks_api_key}", "Content-Type": "application/json"}

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": _REWRITE_PROMPT.format(reasoning=reasoning_text[:4000])})

    payload = {"model": api_model, "messages": messages, "max_tokens": 300, "thinking": {"type": "disabled"}}

    session = proxy.third_party_session or proxy.auth.session or proxy.create_session()
    owns_session = session is not proxy.third_party_session and session is not proxy.auth.session

    try:
        t0 = time.time()
        async with asyncio.timeout(_REWRITE_TIMEOUT):
            async with session.post(endpoint, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    logger.warning(f"[{request_id}] reasoning rewrite failed: Fireworks {resp.status}")
                    return None
                data = await resp.json()
        elapsed = time.time() - t0
        choices = data.get("choices", [])
        if not choices:
            logger.warning(f"[{request_id}] reasoning rewrite returned empty choices")
            return None
        content = choices[0].get("message", {}).get("content", "")
        if not content:
            logger.warning(f"[{request_id}] reasoning rewrite returned empty content")
            return None
        logger.info(f"[{request_id}] reasoning rewrite via {_REWRITE_MODEL} ({elapsed:.1f}s, {len(content)} chars)")
        return content.strip()
    except asyncio.TimeoutError:
        logger.warning(f"[{request_id}] reasoning rewrite timed out ({_REWRITE_TIMEOUT}s)")
        return None
    except (aiohttp.ClientError, OSError) as e:
        logger.warning(f"[{request_id}] reasoning rewrite error: {e}")
        return None
    finally:
        if owns_session:
            await session.close()


async def passthrough_sse_with_rewrite(
    resp, request_id: str, provider_name: str, start: float,
    *,
    system_prompt: Optional[str] = None,
    model: str = "",
):
    """Rewrite hidden reasoning when it is the only output, without leaking it."""
    buffer = bytearray()
    total_content = 0
    control_buffer: list[str] = []
    accumulated_reasoning: list[str] = []
    warned_reasoning_buffer = False
    saw_visible_content = False

    async for raw_chunk in resp.content.iter_any():
        buffer.extend(raw_chunk)
        while True:
            idx = buffer.find(b"\n\n")
            if idx < 0:
                break
            event_bytes = bytes(buffer[:idx]).strip()
            del buffer[:idx + 2]
            emitted, reasoning_text = _scrub_event(event_bytes)

            if reasoning_text:
                accumulated_reasoning.append(reasoning_text)
                if not warned_reasoning_buffer:
                    warned_reasoning_buffer = True
                    logger.warning(f"[{request_id}] {provider_name} returned reasoning-only chunks, buffering hidden reasoning for rewrite")

            if emitted is None:
                continue

            if saw_visible_content:
                total_content += _content_len(emitted)
                yield emitted
                continue

            if _event_has_visible_content(emitted):
                saw_visible_content = True
                for buffered in control_buffer:
                    if not _is_done_event(buffered):
                        yield buffered
                control_buffer.clear()
                total_content += _content_len(emitted)
                yield emitted
                continue

            control_buffer.append(emitted)

    if buffer.strip():
        emitted, reasoning_text = _scrub_event(bytes(buffer).strip())
        if reasoning_text:
            accumulated_reasoning.append(reasoning_text)
            if not warned_reasoning_buffer:
                warned_reasoning_buffer = True
                logger.warning(f"[{request_id}] {provider_name} returned reasoning-only chunks, buffering hidden reasoning for rewrite")
        if emitted is not None:
            if saw_visible_content:
                total_content += _content_len(emitted)
                yield emitted
            elif _event_has_visible_content(emitted):
                saw_visible_content = True
                for buffered in control_buffer:
                    if not _is_done_event(buffered):
                        yield buffered
                control_buffer.clear()
                total_content += _content_len(emitted)
                yield emitted
            else:
                control_buffer.append(emitted)

    elapsed = time.time() - start
    if saw_visible_content:
        logger.info(f"[{request_id}] <- stream done ({elapsed:.1f}s, {provider_name}, {total_content} chars)")
        return

    if accumulated_reasoning:
        rewritten = await _rewrite_reasoning_to_dialogue("".join(accumulated_reasoning), system_prompt, request_id, model)
        if rewritten:
            total_content += len(rewritten)
            yield make_sse_content_chunk(model, rewritten)

    for buffered in control_buffer:
        if _is_done_event(buffered):
            yield buffered

    if total_content == 0:
        logger.warning(f"[{request_id}] <- stream done ({elapsed:.1f}s, {provider_name}, 0 chars) — model returned only reasoning tokens, no content")
    else:
        logger.info(f"[{request_id}] <- stream done ({elapsed:.1f}s, {provider_name}, {total_content} chars)")
