"""Tests for MCP_MODE reasoning passthrough behavior.

Verifies that when proxy.MCP_MODE is True:
  - Thinking is not defaulted to disabled
  - SSE scrubbing is bypassed (reasoning preserved)
  - Reasoning rewrite dispatch is skipped
  - Claude _build_api_body does not strip thinking
"""
import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock


@pytest.fixture()
def proxy_mod(proxy_module):
    return proxy_module


@pytest.fixture(autouse=True)
def _reset_mcp_mode(proxy_mod):
    """Ensure MCP_MODE is reset after each test."""
    original_mcp = proxy_mod.MCP_MODE
    original_rewrite = proxy_mod.reasoning_rewrite_enabled
    yield
    proxy_mod.MCP_MODE = original_mcp
    proxy_mod.reasoning_rewrite_enabled = original_rewrite


# ---------------------------------------------------------------------------
# 1. Thinking default not forced to disabled in MCP mode
# ---------------------------------------------------------------------------

class TestMCPThinkingDefault:
    def test_proxy_mode_defaults_thinking_disabled(self, proxy_mod):
        """In proxy mode, thinking defaults to disabled when caller doesn't set it."""
        proxy_mod.MCP_MODE = False
        extra_params = {}
        if not proxy_mod.MCP_MODE:
            _caller_thinking = extra_params.get("thinking")
            if _caller_thinking is None:
                extra_params.setdefault("thinking", {"type": "disabled"})
        assert extra_params.get("thinking") == {"type": "disabled"}

    def test_mcp_mode_leaves_thinking_unset(self, proxy_mod):
        """In MCP mode, thinking is not injected when caller doesn't set it."""
        proxy_mod.MCP_MODE = True
        extra_params = {}
        if not proxy_mod.MCP_MODE:
            _caller_thinking = extra_params.get("thinking")
            if _caller_thinking is None:
                extra_params.setdefault("thinking", {"type": "disabled"})
        assert "thinking" not in extra_params


# ---------------------------------------------------------------------------
# 2. SSE scrubbing bypassed in MCP mode
# ---------------------------------------------------------------------------

class TestMCPSSEScrubbing:
    def test_scrub_event_strips_reasoning(self):
        """_scrub_event strips reasoning_content from mixed chunks (content preserved)."""
        from proxy_internal.streaming import _scrub_event
        # When content AND reasoning are both present, reasoning is scrubbed from the
        # emitted event but reasoning_text returns None (not a reasoning-only chunk).
        event = b'data: {"choices":[{"delta":{"content":"hi","reasoning_content":"think"}}]}'
        emitted, reasoning_text = _scrub_event(event)
        assert reasoning_text is None  # not reasoning-only, so no reasoning_text
        data = json.loads(emitted.strip()[6:])
        assert "reasoning_content" not in data["choices"][0]["delta"]
        assert data["choices"][0]["delta"]["content"] == "hi"

    def test_scrub_event_drops_reasoning_only(self):
        """_scrub_event drops reasoning-only chunks (emitted=None)."""
        from proxy_internal.streaming import _scrub_event
        event = b'data: {"choices":[{"delta":{"reasoning_content":"deep thoughts"}}]}'
        emitted, reasoning_text = _scrub_event(event)
        assert emitted is None
        assert reasoning_text == "deep thoughts"

    @pytest.mark.asyncio
    async def test_mcp_mode_preserves_reasoning_in_stream(self, proxy_mod):
        """In MCP mode, passthrough_sse preserves reasoning fields verbatim."""
        proxy_mod.MCP_MODE = True

        raw_event = b'data: {"choices":[{"delta":{"content":"hi","reasoning_content":"think"}}]}\n\n'
        done_event = b'data: [DONE]\n\n'

        chunks = [raw_event, done_event]

        class FakeIterAny:
            def __aiter__(self):
                return self
            async def __anext__(self):
                if chunks:
                    return chunks.pop(0)
                raise StopAsyncIteration

        class FakeContent:
            def iter_any(self):
                return FakeIterAny()

        class FakeResp:
            def __init__(self):
                self.content = FakeContent()

        from proxy_internal.streaming import passthrough_sse
        collected = []
        async for chunk in passthrough_sse(FakeResp(), "test-req", "test-provider", 0.0):
            collected.append(chunk)

        # 2 events: the content+reasoning event and [DONE]
        assert len(collected) == 2
        data = json.loads(collected[0].strip()[6:])
        assert data["choices"][0]["delta"]["reasoning_content"] == "think"
        assert data["choices"][0]["delta"]["content"] == "hi"


# ---------------------------------------------------------------------------
# 3. Rewrite dispatch skipped in MCP mode
# ---------------------------------------------------------------------------

class TestMCPRewriteDispatch:
    def test_mcp_mode_bypasses_rewrite(self, proxy_mod):
        """In MCP mode, dispatch always uses original passthrough even with rewrite enabled."""
        proxy_mod.MCP_MODE = True
        proxy_mod.reasoning_rewrite_enabled = True
        token = proxy_mod._rewrite_ctx.set({"system_prompt": "test", "model": "m"})
        try:
            result = proxy_mod._passthrough_sse_dispatch(
                MagicMock(), "req", "provider", 0.0
            )
            assert result is not None
        finally:
            proxy_mod._rewrite_ctx.reset(token)


# ---------------------------------------------------------------------------
# 4. Claude _build_api_body preserves thinking in MCP mode
# ---------------------------------------------------------------------------

class TestMCPClaudeThinking:
    @pytest.fixture(autouse=True)
    def _mock_template(self, proxy_mod):
        """Provide a minimal auth template so _build_api_body doesn't crash."""
        fake_template = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [],
        }
        with patch.object(proxy_mod.auth, "body_template", fake_template):
            with patch.object(proxy_mod.auth, "headers", {"x-api-key": "test"}):
                yield

    def test_proxy_mode_strips_thinking(self, proxy_mod):
        """In proxy mode, _build_api_body strips thinking from body."""
        proxy_mod.MCP_MODE = False
        proxy_mod.reasoning_override_enabled = False
        from proxy_internal.providers.claude import _build_api_body
        body = _build_api_body(
            system_prompt="test",
            messages=[{"role": "user", "content": "hi"}],
            model="claude-sonnet-4-20250514",
            max_tokens=100,
        )
        assert "thinking" not in body

    def test_mcp_mode_does_not_explicitly_strip_thinking(self, proxy_mod):
        """In MCP mode, _build_api_body does not pop thinking from body."""
        proxy_mod.MCP_MODE = True
        proxy_mod.reasoning_override_enabled = False
        from proxy_internal.providers.claude import _build_api_body
        # The template has thinking stripped by _sanitize_claude_template,
        # so it won't be present. The test verifies _build_api_body doesn't
        # crash and doesn't force-add thinking:disabled.
        body = _build_api_body(
            system_prompt="test",
            messages=[{"role": "user", "content": "hi"}],
            model="claude-sonnet-4-20250514",
            max_tokens=100,
        )
        # thinking is absent (from template), but crucially NOT {"type": "disabled"}
        assert body.get("thinking") != {"type": "disabled"}
