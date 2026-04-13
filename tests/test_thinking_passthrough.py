"""Tests for thinking token passthrough when dashboard reasoning override is active."""

import json
import pytest

from proxy_internal import streaming as streaming_module


def _make_mixed_event(content_text="hello", reasoning_text="thinking"):
    """Create an SSE event where the same delta has both content and reasoning_content."""
    data = {"choices": [{"delta": {"content": content_text, "reasoning_content": reasoning_text}}]}
    return f"data: {json.dumps(data)}\n\n".encode()


class _FakeContent:
    def __init__(self, chunks):
        self._chunks = chunks

    async def iter_any(self):
        for chunk in self._chunks:
            yield chunk


class _FakeResponse:
    def __init__(self, chunks):
        self.content = _FakeContent(chunks)


async def _collect(agen):
    out = []
    async for chunk in agen:
        out.append(chunk)
    return out


def _make_reasoning_event(reasoning_text="private trace", field="reasoning_content"):
    data = {"choices": [{"delta": {field: reasoning_text}}]}
    return f"data: {json.dumps(data)}\n\n".encode()


def _make_content_event(content_text="hello"):
    data = {"choices": [{"delta": {"content": content_text}}]}
    return f"data: {json.dumps(data)}\n\n".encode()


@pytest.mark.asyncio
async def test_override_on_preserves_reasoning(monkeypatch):
    """When reasoning_override_enabled is True, reasoning fields should NOT be scrubbed."""
    import proxy
    monkeypatch.setattr(proxy, "MCP_MODE", False)
    monkeypatch.setattr(proxy, "reasoning_override_enabled", True)

    resp = _FakeResponse([
        _make_reasoning_event("deep thinking here"),
        _make_content_event("spoken words"),
        b"data: [DONE]\n\n",
    ])

    chunks = await _collect(
        streaming_module.passthrough_sse(resp, "req-override", "TestProvider", 0.0)
    )

    combined = "".join(chunks)
    assert "deep thinking here" in combined
    assert "spoken words" in combined
    assert "[DONE]" in combined


@pytest.mark.asyncio
async def test_override_off_scrubs_reasoning(monkeypatch):
    """When override is OFF and not MCP, reasoning fields ARE scrubbed."""
    import proxy
    monkeypatch.setattr(proxy, "MCP_MODE", False)
    monkeypatch.setattr(proxy, "reasoning_override_enabled", False)

    resp = _FakeResponse([
        _make_reasoning_event("secret thoughts"),
        _make_content_event("visible reply"),
        b"data: [DONE]\n\n",
    ])

    chunks = await _collect(
        streaming_module.passthrough_sse(resp, "req-no-override", "TestProvider", 0.0)
    )

    combined = "".join(chunks)
    assert "secret thoughts" not in combined
    assert "visible reply" in combined
    assert "[DONE]" in combined


@pytest.mark.asyncio
async def test_mcp_mode_preserves_reasoning(monkeypatch):
    """MCP mode always preserves reasoning (existing behavior)."""
    import proxy
    monkeypatch.setattr(proxy, "MCP_MODE", True)
    monkeypatch.setattr(proxy, "reasoning_override_enabled", False)

    resp = _FakeResponse([
        _make_reasoning_event("mcp reasoning"),
        _make_content_event("mcp content"),
        b"data: [DONE]\n\n",
    ])

    chunks = await _collect(
        streaming_module.passthrough_sse(resp, "req-mcp", "TestProvider", 0.0)
    )

    combined = "".join(chunks)
    assert "mcp reasoning" in combined
    assert "mcp content" in combined


@pytest.mark.asyncio
async def test_override_logs_info_once(monkeypatch, caplog):
    """When override is active and reasoning is present, an INFO log should appear once."""
    import logging
    import proxy
    monkeypatch.setattr(proxy, "MCP_MODE", False)
    monkeypatch.setattr(proxy, "reasoning_override_enabled", True)

    resp = _FakeResponse([
        _make_reasoning_event("thought 1"),
        _make_reasoning_event("thought 2"),
        _make_content_event("reply"),
        b"data: [DONE]\n\n",
    ])

    with caplog.at_level(logging.INFO, logger="proxy"):
        chunks = await _collect(
            streaming_module.passthrough_sse(resp, "req-log", "TestProvider", 0.0)
        )

    override_msgs = [r for r in caplog.records if "dashboard reasoning override" in r.message]
    assert len(override_msgs) == 1


@pytest.mark.asyncio
async def test_override_preserves_reasoning_in_tail_buffer(monkeypatch):
    """Reasoning in the tail buffer (incomplete final chunk) is also preserved with override."""
    import proxy
    monkeypatch.setattr(proxy, "MCP_MODE", False)
    monkeypatch.setattr(proxy, "reasoning_override_enabled", True)

    # Send a reasoning event WITHOUT trailing \n\n so it stays in the tail buffer
    resp = _FakeResponse([
        _make_reasoning_event("tail reasoning"),
    ])

    chunks = await _collect(
        streaming_module.passthrough_sse(resp, "req-tail", "TestProvider", 0.0)
    )

    combined = "".join(chunks)
    assert "tail reasoning" in combined


@pytest.mark.asyncio
async def test_override_beats_rewrite_in_dispatch(monkeypatch):
    """When both reasoning_override_enabled AND reasoning_rewrite_enabled are True,
    the dispatch function should use the original passthrough (raw pass-through),
    NOT the rewrite path."""
    import proxy

    monkeypatch.setattr(proxy, "MCP_MODE", False)
    monkeypatch.setattr(proxy, "reasoning_override_enabled", True)
    monkeypatch.setattr(proxy, "reasoning_rewrite_enabled", True)

    # Set a rewrite context so that branch would trigger if override didn't block it
    token = proxy._rewrite_ctx.set({"system_prompt": "You are an NPC.", "model": "test"})

    dispatch_target = []

    original_passthrough = proxy._passthrough_sse_original

    def fake_original(resp, request_id, provider_name, start, **kw):
        dispatch_target.append("original")
        return original_passthrough(resp, request_id, provider_name, start, **kw)

    def fake_rewrite(resp, request_id, provider_name, start, **kw):
        dispatch_target.append("rewrite")

    monkeypatch.setattr(proxy, "_passthrough_sse_original", fake_original)
    monkeypatch.setattr(proxy, "passthrough_sse_with_rewrite", fake_rewrite)

    resp = _FakeResponse([
        _make_reasoning_event("deep thought"),
        _make_content_event("reply"),
        b"data: [DONE]\n\n",
    ])

    try:
        # Call the dispatch function directly
        gen = proxy._passthrough_sse_dispatch(resp, "req-dispatch", "TestProvider", 0.0)
        chunks = await _collect(gen)
    finally:
        proxy._rewrite_ctx.reset(token)

    assert dispatch_target == ["original"], f"Expected original passthrough, got {dispatch_target}"
    combined = "".join(chunks)
    assert "deep thought" in combined
    assert "reply" in combined


@pytest.mark.asyncio
async def test_mixed_content_and_reasoning_preserved_with_override(monkeypatch):
    """When override is active and a single delta has both content and reasoning_content,
    both fields should be preserved in the output."""
    import proxy
    monkeypatch.setattr(proxy, "MCP_MODE", False)
    monkeypatch.setattr(proxy, "reasoning_override_enabled", True)

    resp = _FakeResponse([
        _make_mixed_event(content_text="spoken words", reasoning_text="inner thoughts"),
        b"data: [DONE]\n\n",
    ])

    chunks = await _collect(
        streaming_module.passthrough_sse(resp, "req-mixed", "TestProvider", 0.0)
    )

    combined = "".join(chunks)
    assert "spoken words" in combined
    assert "inner thoughts" in combined
