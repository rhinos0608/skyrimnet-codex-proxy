"""SSE chunk builders for OpenAI-compatible and Anthropic streaming responses."""

import json
import time
import uuid

from proxy_internal._json import json_dumps


def make_sse_error_chunk(model: str, error_msg: str) -> str:
    """Build a single SSE error chunk in OpenAI chat.completion.chunk format."""
    cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    chunk = {
        "id": cmpl_id, "object": "chat.completion.chunk",
        "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "delta": {"content": error_msg}, "finish_reason": None}],
    }
    return f"data: {json.dumps(chunk)}\n\n"


def make_sse_content_chunk(model: str, content: str) -> str:
    """Build a single SSE content chunk in OpenAI chat.completion.chunk format."""
    cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    chunk = {
        "id": cmpl_id, "object": "chat.completion.chunk",
        "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }
    return f"data: {json.dumps(chunk)}\n\n"


def yield_sse_error(model: str, error_msg: str):
    """Return (error_chunk, done_marker) strings for SSE error responses."""
    return make_sse_error_chunk(model, error_msg), "data: [DONE]\n\n"


def _format_anthropic_sse(event: str, data: dict) -> bytes:
    """Format an Anthropic SSE event (named event + JSON data line).

    Returns ``bytes`` directly rather than a ``str``.  ``json_dumps`` is
    orjson-backed when available and always yields bytes; previously we
    decoded those bytes into a UTF-8 string just so ``StreamingResponse``
    could re-encode them back to bytes on the way out to the client.  In
    the per-token streaming hot path that decode + re-encode cycle is pure
    overhead — FastAPI's ``StreamingResponse`` accepts both ``str`` and
    ``bytes`` chunks, so callers don't need to change.
    """
    return b"event: " + event.encode("ascii") + b"\ndata: " + json_dumps(data) + b"\n\n"
