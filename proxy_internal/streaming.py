"""Shared streaming helpers for upstream SSE passthrough."""

import json
import logging
import time
from typing import Optional

logger = logging.getLogger("proxy")

_REASONING_FIELDS = {"reasoning", "reasoning_content", "reasoning_details"}


async def passthrough_sse(resp, request_id: str, provider_name: str, start: float):
    """Passthrough OpenAI-format SSE from an upstream response, yielding events.

    Used by OpenRouter, Ollama, Z.AI, OpenCode, Qwen — all OpenAI-compatible SSE.
    Strips reasoning fields (reasoning, reasoning_content, reasoning_details) from
    SSE deltas so downstream consumers only see plain content.  Reasoning-only
    chunks (content empty/null while reasoning present) are silently dropped.
    """
    buffer = bytearray()
    total_content = 0

    def _emit_event(event_bytes: bytes) -> tuple[Optional[str], int]:
        if not event_bytes:
            return None, 0

        event = event_bytes.decode("utf-8", errors="replace").strip()
        if not event:
            return None, 0
        if not event.startswith("data: ") or event.startswith("data: [DONE]"):
            return event + "\n\n", 0

        # Fast path: most chunks do not carry reasoning fields, so skip JSON
        # parsing entirely unless the event looks like a reasoning chunk.
        if b'"reasoning' not in event_bytes:
            return event + "\n\n", 0

        try:
            data = json.loads(event[6:])
            delta = data.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content")
            modified = False
            for rf in _REASONING_FIELDS:
                if delta.pop(rf, None) is not None:
                    modified = True
            if content:
                return (f"data: {json.dumps(data)}\n\n" if modified else event + "\n\n"), len(content)
            if modified:
                return None, 0
            return event + "\n\n", 0
        except (json.JSONDecodeError, KeyError, IndexError):
            return event + "\n\n", 0

    async for raw_chunk in resp.content.iter_any():
        buffer.extend(raw_chunk)
        while True:
            idx = buffer.find(b"\n\n")
            if idx < 0:
                break
            event_bytes = bytes(buffer[:idx]).strip()
            del buffer[:idx + 2]
            emitted, content_len = _emit_event(event_bytes)
            total_content += content_len
            if emitted is not None:
                yield emitted
    if buffer.strip():
        emitted, content_len = _emit_event(bytes(buffer).strip())
        total_content += content_len
        if emitted is not None:
            yield emitted
    elapsed = time.time() - start
    logger.info(f"[{request_id}] <- stream done ({elapsed:.1f}s, {provider_name}, {total_content} chars)")
