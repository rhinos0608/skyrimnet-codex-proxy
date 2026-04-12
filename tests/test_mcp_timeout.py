"""Tests for _logged_call_with_timeout — a timeout wrapper around _logged_call.

The function under test does not exist yet; these tests are written TDD-style
so they will fail until the implementation is added to mcp_server.py.

Expectations:
- `_logged_call_with_timeout` has the same signature as `_logged_call` plus an
  optional `timeout` parameter (default: MCP_TOOL_TIMEOUT seconds).
- On normal completion it returns exactly what the underlying call_fn returned.
- On timeout it returns an error string instead of raising, because MCP tool
  handlers are expected to return strings.
- MCP_TOOL_TIMEOUT constant should be 300 seconds (matching aiohttp default).
"""
import asyncio
import sys
import os
import pytest

# Ensure project root is importable without running proxy startup side-effects.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_mcp_server():
    """Import mcp_server with shutil.which mocked to avoid CLI subprocess spawning."""
    from unittest.mock import patch
    if "mcp_server" in sys.modules:
        return sys.modules["mcp_server"]
    with patch("shutil.which", return_value=None):
        import mcp_server  # noqa: F401
    return sys.modules["mcp_server"]


async def _stub_call_fn(system_prompt, messages, model, max_tokens, **kwargs):
    """Minimal call_fn stub that returns immediately."""
    return "stub result"


async def _slow_call_fn(system_prompt, messages, model, max_tokens, **kwargs):
    """call_fn stub that sleeps long enough to trigger a timeout."""
    await asyncio.sleep(10)
    return "should never reach here"


_MESSAGES = [{"role": "user", "content": "Hello"}]
_SYSTEM = "You are a helpful assistant."
_MODEL = "claude-sonnet-4-6"
_MAX_TOKENS = 256


# ---------------------------------------------------------------------------
# Test 1: normal completion within timeout
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_logged_call_with_timeout_normal_completion():
    """A call_fn that completes in ~0.1 s with a 5 s timeout should succeed."""
    mcp = _get_mcp_server()

    async def fast_call_fn(system_prompt, messages, model, max_tokens, **kwargs):
        await asyncio.sleep(0.1)
        return "fast result"

    result = await mcp._logged_call_with_timeout(
        "test_tool",
        "claude",
        _MODEL,
        fast_call_fn,
        _SYSTEM,
        _MESSAGES,
        _MAX_TOKENS,
        timeout=5.0,
    )

    assert result == "fast result"


# ---------------------------------------------------------------------------
# Test 2: call_fn exceeds timeout — returns error string, does not raise
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_logged_call_with_timeout_exceeds_timeout():
    """A call_fn that would take 10 s but timeout=0.5 s should return an error string."""
    mcp = _get_mcp_server()

    result = await mcp._logged_call_with_timeout(
        "test_tool",
        "claude",
        _MODEL,
        _slow_call_fn,
        _SYSTEM,
        _MESSAGES,
        _MAX_TOKENS,
        timeout=0.5,
    )

    assert isinstance(result, str)
    assert "timed out" in result.lower()
    assert "0.5" in result


# ---------------------------------------------------------------------------
# Test 3: default timeout constant is 300 seconds
# ---------------------------------------------------------------------------

def test_logged_call_with_timeout_default_timeout():
    """MCP_TOOL_TIMEOUT should equal 300 seconds."""
    mcp = _get_mcp_server()

    assert hasattr(mcp, "MCP_TOOL_TIMEOUT"), (
        "mcp_server.py must define a MCP_TOOL_TIMEOUT constant"
    )
    assert mcp.MCP_TOOL_TIMEOUT == 300, (
        f"Expected MCP_TOOL_TIMEOUT=300, got {mcp.MCP_TOOL_TIMEOUT}"
    )


# ---------------------------------------------------------------------------
# Test 4: wrapper preserves the exact return value of the underlying call
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_logged_call_with_timeout_preserves_result():
    """The wrapper must return the exact same value that call_fn returns."""
    mcp = _get_mcp_server()

    expected = "The dragon Alduin soars over Helgen."

    async def known_result_fn(system_prompt, messages, model, max_tokens, **kwargs):
        return expected

    result = await mcp._logged_call_with_timeout(
        "npc_chat",
        "claude",
        _MODEL,
        known_result_fn,
        _SYSTEM,
        _MESSAGES,
        _MAX_TOKENS,
        timeout=5.0,
    )

    assert result == expected
