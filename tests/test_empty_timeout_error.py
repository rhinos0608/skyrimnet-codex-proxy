"""Tests that timeout and client errors produce non-empty error messages.

Verifies the `str(e) or type(e).__name__` pattern in _call_oai_compatible_direct
and _stream_oai_compatible prevents blank error strings.
"""
import asyncio
import aiohttp
import pytest
from fastapi import HTTPException
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture()
def resolved():
    """Minimal resolved dict for the OAI-compatible helpers."""
    session = AsyncMock(spec=aiohttp.ClientSession)
    return {
        "session": session,
        "endpoint_url": "http://localhost:9999/v1/chat/completions",
        "headers": {"Authorization": "Bearer test"},
        "provider_name": "TestProvider",
        "api_model": "test-model",
    }


@pytest.fixture()
def payload():
    return {
        "messages": [{"role": "user", "content": "hello"}],
        "model": "test-model",
    }


@pytest.mark.asyncio
async def test_timeout_error_direct_has_nonempty_message(proxy_module, resolved, payload):
    """asyncio.TimeoutError in _call_oai_compatible_direct produces a non-empty detail."""
    resolved["session"].post = MagicMock(side_effect=asyncio.TimeoutError())

    with pytest.raises(HTTPException) as exc_info:
        await proxy_module._call_oai_compatible_direct(resolved, payload, "req-t1")

    detail = exc_info.value.detail
    assert detail, "Error detail should not be empty"
    assert "TimeoutError" in detail, f"Expected 'TimeoutError' in detail, got: {detail}"


@pytest.mark.asyncio
async def test_timeout_error_stream_has_nonempty_message(proxy_module, resolved, payload):
    """asyncio.TimeoutError in _stream_oai_compatible produces a non-empty detail."""
    resolved["session"].post = MagicMock(side_effect=asyncio.TimeoutError())

    with pytest.raises(HTTPException) as exc_info:
        async for _ in proxy_module._stream_oai_compatible(resolved, payload, "req-t2"):
            pass

    detail = exc_info.value.detail
    assert detail, "Error detail should not be empty"
    assert "TimeoutError" in detail, f"Expected 'TimeoutError' in detail, got: {detail}"


@pytest.mark.asyncio
async def test_client_error_direct_has_message(proxy_module, resolved, payload):
    """aiohttp.ClientError subclass with a message preserves that message."""
    resolved["session"].post = MagicMock(
        side_effect=aiohttp.ClientConnectionError("Connection refused")
    )

    with pytest.raises(HTTPException) as exc_info:
        await proxy_module._call_oai_compatible_direct(resolved, payload, "req-c1")

    detail = exc_info.value.detail
    assert detail, "Error detail should not be empty"
    assert "Connection refused" in detail, f"Expected 'Connection refused' in detail, got: {detail}"


@pytest.mark.asyncio
async def test_client_error_stream_has_message(proxy_module, resolved, payload):
    """aiohttp.ClientError subclass with a message preserves that message in streaming."""
    resolved["session"].post = MagicMock(
        side_effect=aiohttp.ClientConnectionError("Connection refused")
    )

    with pytest.raises(HTTPException) as exc_info:
        async for _ in proxy_module._stream_oai_compatible(resolved, payload, "req-c2"):
            pass

    detail = exc_info.value.detail
    assert detail, "Error detail should not be empty"
    assert "Connection refused" in detail, f"Expected 'Connection refused' in detail, got: {detail}"


@pytest.mark.asyncio
async def test_os_error_direct_fallback(proxy_module, resolved, payload):
    """OSError with empty str still produces a non-empty detail via type name."""
    resolved["session"].post = MagicMock(side_effect=OSError())

    with pytest.raises(HTTPException) as exc_info:
        await proxy_module._call_oai_compatible_direct(resolved, payload, "req-o1")

    detail = exc_info.value.detail
    assert detail, "Error detail should not be empty"
    # OSError() has an empty str, so the fallback should use the class name
    assert "OSError" in detail, f"Expected 'OSError' in detail, got: {detail}"
