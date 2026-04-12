"""Tests for reasoning sanitization and rewrite behavior in streaming helpers."""

import pytest

from proxy_internal import streaming as streaming_module


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


@pytest.mark.asyncio
async def test_passthrough_sse_drops_reasoning_only_chunks():
    resp = _FakeResponse([
        b'data: {"choices":[{"delta":{"reasoning_content":"private trace"}}]}\n\n',
        b"data: [DONE]\n\n",
    ])

    chunks = await _collect(
        streaming_module.passthrough_sse(resp, "req1", "Provider", 0.0)
    )

    combined = "".join(chunks)
    assert "private trace" not in combined
    assert "[DONE]" in combined


@pytest.mark.asyncio
async def test_passthrough_sse_with_rewrite_failure_never_leaks_reasoning(monkeypatch):
    resp = _FakeResponse([
        b'data: {"choices":[{"delta":{"reasoning_content":"private trace"}}]}\n\n',
        b"data: [DONE]\n\n",
    ])

    async def _fail(*args, **kwargs):
        return None

    monkeypatch.setattr(streaming_module, "_rewrite_reasoning_to_dialogue", _fail)

    chunks = await _collect(
        streaming_module.passthrough_sse_with_rewrite(
            resp,
            "req2",
            "Provider",
            0.0,
            system_prompt="Stay in character.",
            model="fireworks:kimi-k2p5-turbo",
        )
    )

    combined = "".join(chunks)
    assert "private trace" not in combined


@pytest.mark.asyncio
async def test_passthrough_sse_with_rewrite_does_not_flush_reasoning_before_real_content(monkeypatch):
    resp = _FakeResponse([
        b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n',
        b'data: {"choices":[{"delta":{"reasoning_content":"private trace"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"final answer"}}]}\n\n',
        b"data: [DONE]\n\n",
    ])

    async def _unused(*args, **kwargs):
        raise AssertionError("rewrite should not run once real content arrives")

    monkeypatch.setattr(streaming_module, "_rewrite_reasoning_to_dialogue", _unused)

    chunks = await _collect(
        streaming_module.passthrough_sse_with_rewrite(
            resp,
            "req3",
            "Provider",
            0.0,
            system_prompt="Stay in character.",
            model="fireworks:kimi-k2p5-turbo",
        )
    )

    combined = "".join(chunks)
    assert "private trace" not in combined
    assert "final answer" in combined
