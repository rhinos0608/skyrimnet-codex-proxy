"""Tests for provider call functions: OpenRouter, Z.AI direct/streaming, and Claude error paths."""
import pytest
import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers (mirrors test_ollama.py patterns)
# ---------------------------------------------------------------------------

def _make_mock_response(status=200, json_data=None, text_data=""):
    """Build a mock aiohttp response usable as an async context manager."""
    mock_resp = AsyncMock()
    mock_resp.status = status
    mock_resp.json = AsyncMock(return_value=json_data or {})
    mock_resp.text = AsyncMock(return_value=text_data)
    mock_resp.read = AsyncMock(return_value=text_data.encode("utf-8") if isinstance(text_data, str) else text_data)
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    return mock_resp


def _make_mock_session(resp):
    """Build a mock aiohttp session whose .post() returns the given response."""
    session = MagicMock()
    session.post = MagicMock(return_value=resp)
    session.close = AsyncMock()
    return session


async def _async_bytes_iter(chunks):
    """Async generator that yields byte chunks, simulating aiohttp iter_any."""
    for chunk in chunks:
        yield chunk


def _make_streaming_response(status=200, sse_chunks=None, text_data=""):
    """Build a mock response with streaming content support."""
    mock_resp = AsyncMock()
    mock_resp.status = status
    mock_resp.text = AsyncMock(return_value=text_data)
    mock_resp.read = AsyncMock(return_value=text_data.encode("utf-8") if isinstance(text_data, str) else text_data)
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    if sse_chunks is not None:
        mock_content = MagicMock()
        mock_content.iter_any.return_value = _async_bytes_iter(sse_chunks)
        mock_resp.content = mock_content

    return mock_resp


def _make_mock_codex_proc(stdout=b"", stderr=b"", returncode=0):
    """Build a mock subprocess result for Codex CLI tests."""
    proc = MagicMock()
    proc.returncode = returncode
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    return proc


# ---------------------------------------------------------------------------
# OpenRouter Direct (call_openrouter_direct)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_openrouter_direct_no_key_raises_500(proxy_module):
    """Raises HTTPException 500 when openrouter_api_key is not set."""
    from fastapi import HTTPException

    with patch.object(proxy_module, "openrouter_api_key", None):
        with pytest.raises(HTTPException) as exc:
            await proxy_module.call_openrouter_direct(
                None, [{"role": "user", "content": "hi"}], "openai/gpt-4o", 100
            )
    assert exc.value.status_code == 500


@pytest.mark.asyncio
async def test_openrouter_direct_sends_bearer_auth(proxy_module):
    """Sends Bearer authorization header with the API key."""
    resp = _make_mock_response(200, {"choices": [{"message": {"content": "Hello"}}]})
    session = _make_mock_session(resp)

    with patch.object(proxy_module, "openrouter_api_key", "sk-test"), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch("aiohttp.ClientSession", return_value=session):
        await proxy_module.call_openrouter_direct(
            None, [{"role": "user", "content": "hi"}], "openai/gpt-4o", 100
        )

    hdrs = session.post.call_args[1]["headers"]
    assert hdrs["Authorization"] == "Bearer sk-test"


@pytest.mark.asyncio
async def test_openrouter_direct_sends_to_correct_url(proxy_module):
    """Posts to the OpenRouter chat completions endpoint."""
    resp = _make_mock_response(200, {"choices": [{"message": {"content": "ok"}}]})
    session = _make_mock_session(resp)

    with patch.object(proxy_module, "openrouter_api_key", "sk-test"), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch("aiohttp.ClientSession", return_value=session):
        await proxy_module.call_openrouter_direct(
            None, [{"role": "user", "content": "hi"}], "openai/gpt-4o", 100
        )

    url = session.post.call_args[0][0]
    assert url == "https://openrouter.ai/api/v1/chat/completions"


@pytest.mark.asyncio
async def test_openrouter_direct_includes_system_prompt(proxy_module):
    """System prompt is prepended as a system message."""
    resp = _make_mock_response(200, {"choices": [{"message": {"content": "ok"}}]})
    session = _make_mock_session(resp)

    with patch.object(proxy_module, "openrouter_api_key", "sk-test"), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch("aiohttp.ClientSession", return_value=session):
        await proxy_module.call_openrouter_direct(
            "You are helpful.", [{"role": "user", "content": "hi"}], "openai/gpt-4o", 100
        )

    payload = session.post.call_args[1]["json"]
    assert payload["messages"][0] == {"role": "system", "content": "You are helpful."}
    assert payload["messages"][1] == {"role": "user", "content": "hi"}


@pytest.mark.asyncio
async def test_openrouter_direct_returns_content(proxy_module):
    """Returns the text from choices[0].message.content."""
    resp = _make_mock_response(200, {"choices": [{"message": {"content": "The answer is 42"}}]})
    session = _make_mock_session(resp)

    with patch.object(proxy_module, "openrouter_api_key", "sk-test"), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch("aiohttp.ClientSession", return_value=session):
        result = await proxy_module.call_openrouter_direct(
            None, [{"role": "user", "content": "hi"}], "openai/gpt-4o", 100
        )

    assert result == "The answer is 42"


@pytest.mark.asyncio
async def test_openrouter_direct_non_200_raises(proxy_module):
    """Non-200 status raises HTTPException with the upstream status code."""
    from fastapi import HTTPException

    resp = _make_mock_response(status=429, text_data="Rate limited")
    session = _make_mock_session(resp)

    with patch.object(proxy_module, "openrouter_api_key", "sk-test"), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch("aiohttp.ClientSession", return_value=session):
        with pytest.raises(HTTPException) as exc:
            await proxy_module.call_openrouter_direct(
                None, [{"role": "user", "content": "hi"}], "openai/gpt-4o", 100
            )
    assert exc.value.status_code == 429


@pytest.mark.asyncio
async def test_openrouter_direct_uses_existing_session(proxy_module):
    """When auth.session exists, uses it instead of creating a new one."""
    resp = _make_mock_response(200, {"choices": [{"message": {"content": "ok"}}]})
    existing_session = _make_mock_session(resp)

    with patch.object(proxy_module, "openrouter_api_key", "sk-test"), \
         patch.object(proxy_module, "auth", MagicMock(session=existing_session)), \
         patch("aiohttp.ClientSession") as mock_create:
        await proxy_module.call_openrouter_direct(
            None, [{"role": "user", "content": "hi"}], "openai/gpt-4o", 100
        )

    # Should not have created a new session
    mock_create.assert_not_called()
    # Session should NOT be closed when it belongs to auth
    existing_session.close.assert_not_awaited()


# ---------------------------------------------------------------------------
# OpenRouter Streaming (call_openrouter_streaming)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_openrouter_streaming_no_key_yields_error_and_done(proxy_module):
    """When no API key, yields error message and [DONE]."""
    with patch.object(proxy_module, "openrouter_api_key", None):
        chunks = []
        async for chunk in proxy_module.call_openrouter_streaming(
            None, [{"role": "user", "content": "hi"}], "openai/gpt-4o", 100
        ):
            chunks.append(chunk)

    combined = "".join(chunks)
    assert "[DONE]" in combined
    assert "error" in combined.lower() or "not configured" in combined.lower()


@pytest.mark.asyncio
async def test_openrouter_streaming_passthrough(proxy_module):
    """SSE bytes from OpenRouter are passed through."""
    sse_bytes = b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\ndata: [DONE]\n\n'
    mock_resp = _make_streaming_response(200, [sse_bytes])
    session = _make_mock_session(mock_resp)

    with patch.object(proxy_module, "openrouter_api_key", "sk-test"), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch("aiohttp.ClientSession", return_value=session):
        chunks = []
        async for chunk in proxy_module.call_openrouter_streaming(
            None, [{"role": "user", "content": "hi"}], "openai/gpt-4o", 100
        ):
            chunks.append(chunk)

    combined = "".join(chunks)
    assert "hello" in combined


@pytest.mark.asyncio
async def test_openrouter_streaming_error_status_yields_error_and_done(proxy_module):
    """Non-200 status yields an error chunk and [DONE]."""
    mock_resp = _make_streaming_response(500, text_data="Internal Server Error")
    session = _make_mock_session(mock_resp)

    with patch.object(proxy_module, "openrouter_api_key", "sk-test"), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch("aiohttp.ClientSession", return_value=session):
        chunks = []
        async for chunk in proxy_module.call_openrouter_streaming(
            None, [{"role": "user", "content": "hi"}], "openai/gpt-4o", 100
        ):
            chunks.append(chunk)

    combined = "".join(chunks)
    assert "[DONE]" in combined
    assert "OpenRouter Error 500" in combined


# ---------------------------------------------------------------------------
# Codex CLI (call_codex_direct)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_call_codex_direct_forces_low_reasoning_effort(proxy_module):
    """Codex provider should run in fast mode via low reasoning effort."""
    proc = _make_mock_codex_proc(
        stdout=b'{"msg":{"type":"agent_message","message":"ok"}}\n',
        stderr=b"",
        returncode=0,
    )

    with patch.object(proxy_module, "CODEX_PATH", "/usr/local/bin/codex"), \
         patch.object(proxy_module, "_get_codex_command", return_value=("codex", [])), \
         patch.object(proxy_module, "_create_isolated_codex_home",
                      return_value=("/tmp/codex-home", {"CODEX_HOME": "/tmp/codex-home/.codex"})), \
         patch.object(proxy_module, "_cleanup_isolated_home"), \
         patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=proc) as mock_exec:
        result = await proxy_module.call_codex_direct(
            "Follow instructions carefully.",
            [{"role": "user", "content": "hi"}],
            "gpt-5.3-codex",
            100,
        )

    assert result == "ok"
    args = mock_exec.call_args.args
    assert "-c" in args
    idx = args.index("-c")
    assert args[idx + 1] == 'model_reasoning_effort="low"'


@pytest.mark.asyncio
async def test_call_codex_streaming_forces_low_reasoning_effort(proxy_module):
    """Streaming Codex provider should also run in fast mode."""
    proc = _make_mock_codex_proc(
        stdout=b'{"msg":{"type":"agent_message","message":"ok"}}\n',
        stderr=b"",
        returncode=0,
    )

    with patch.object(proxy_module, "CODEX_PATH", "/usr/local/bin/codex"), \
         patch.object(proxy_module, "_get_codex_command", return_value=("codex", [])), \
         patch.object(proxy_module, "_create_isolated_codex_home",
                      return_value=("/tmp/codex-home", {"CODEX_HOME": "/tmp/codex-home/.codex"})), \
         patch.object(proxy_module, "_cleanup_isolated_home"), \
         patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=proc) as mock_exec:
        chunks = []
        async for chunk in proxy_module.call_codex_streaming(
            "Follow instructions carefully.",
            [{"role": "user", "content": "hi"}],
            "gpt-5.3-codex",
            100,
        ):
            chunks.append(chunk)

    assert any("ok" in chunk for chunk in chunks)
    args = mock_exec.call_args.args
    assert "-c" in args
    idx = args.index("-c")
    assert args[idx + 1] == 'model_reasoning_effort="low"'


@pytest.mark.asyncio
async def test_openrouter_streaming_connection_error_yields_error(proxy_module):
    """ClientError during streaming yields error chunk + [DONE]."""
    import aiohttp

    session = MagicMock()
    conn_key = MagicMock()
    session.post = MagicMock(side_effect=aiohttp.ClientConnectorError(
        conn_key, OSError("Connection refused")
    ))
    session.close = AsyncMock()

    with patch.object(proxy_module, "openrouter_api_key", "sk-test"), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch("aiohttp.ClientSession", return_value=session):
        chunks = []
        async for chunk in proxy_module.call_openrouter_streaming(
            None, [{"role": "user", "content": "hi"}], "openai/gpt-4o", 100
        ):
            chunks.append(chunk)

    combined = "".join(chunks)
    assert "[DONE]" in combined
    assert "OpenRouter Error" in combined or "error" in combined.lower()


# ---------------------------------------------------------------------------
# Z.AI Direct (call_zai_direct)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_zai_direct_no_key_raises_503(proxy_module):
    """Raises HTTPException 503 when zai_api_key is not set."""
    from fastapi import HTTPException

    with patch.object(proxy_module, "zai_api_key", None):
        with pytest.raises(HTTPException) as exc:
            await proxy_module.call_zai_direct(
                None, [{"role": "user", "content": "hi"}], "zai:claude-sonnet-4-6", 100
            )
    assert exc.value.status_code == 503


@pytest.mark.asyncio
async def test_zai_direct_strips_prefix(proxy_module):
    """Strips 'zai:' prefix from model name before sending to API."""
    resp = _make_mock_response(200, {"choices": [{"message": {"content": "ok"}}]})
    session = _make_mock_session(resp)

    with patch.object(proxy_module, "zai_api_key", "sk-zai-test"), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch("aiohttp.ClientSession", return_value=session):
        await proxy_module.call_zai_direct(
            None, [{"role": "user", "content": "hi"}], "zai:claude-sonnet-4-6", 100
        )

    payload = session.post.call_args[1]["json"]
    assert payload["model"] == "claude-sonnet-4-6"


@pytest.mark.asyncio
async def test_zai_direct_sends_to_correct_endpoint(proxy_module):
    """Posts to the Z.AI chat completions endpoint."""
    resp = _make_mock_response(200, {"choices": [{"message": {"content": "ok"}}]})
    session = _make_mock_session(resp)

    with patch.object(proxy_module, "zai_api_key", "sk-zai-test"), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch("aiohttp.ClientSession", return_value=session):
        await proxy_module.call_zai_direct(
            None, [{"role": "user", "content": "hi"}], "zai:gpt-4o", 100
        )

    url = session.post.call_args[0][0]
    assert url == "https://api.z.ai/api/coding/paas/v4/chat/completions"


@pytest.mark.asyncio
async def test_zai_direct_returns_content(proxy_module):
    """Returns the text from choices[0].message.content."""
    resp = _make_mock_response(200, {"choices": [{"message": {"content": "Z.AI says hello"}}]})
    session = _make_mock_session(resp)

    with patch.object(proxy_module, "zai_api_key", "sk-zai-test"), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch("aiohttp.ClientSession", return_value=session):
        result = await proxy_module.call_zai_direct(
            None, [{"role": "user", "content": "hi"}], "zai:gpt-4o", 100
        )

    assert result == "Z.AI says hello"


@pytest.mark.asyncio
async def test_zai_direct_401_raises_401(proxy_module):
    """401 from Z.AI raises HTTPException 401."""
    from fastapi import HTTPException

    resp = _make_mock_response(status=401, text_data="Unauthorized")
    session = _make_mock_session(resp)

    with patch.object(proxy_module, "zai_api_key", "sk-zai-test"), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch("aiohttp.ClientSession", return_value=session):
        with pytest.raises(HTTPException) as exc:
            await proxy_module.call_zai_direct(
                None, [{"role": "user", "content": "hi"}], "zai:gpt-4o", 100
            )
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_zai_direct_403_raises_401(proxy_module):
    """403 from Z.AI also raises HTTPException 401 (auth failed)."""
    from fastapi import HTTPException

    resp = _make_mock_response(status=403, text_data="Forbidden")
    session = _make_mock_session(resp)

    with patch.object(proxy_module, "zai_api_key", "sk-zai-test"), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch("aiohttp.ClientSession", return_value=session):
        with pytest.raises(HTTPException) as exc:
            await proxy_module.call_zai_direct(
                None, [{"role": "user", "content": "hi"}], "zai:gpt-4o", 100
            )
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_zai_direct_429_raises_429(proxy_module):
    """429 from Z.AI raises HTTPException 429."""
    from fastapi import HTTPException

    resp = _make_mock_response(status=429, text_data="Rate limit exceeded")
    session = _make_mock_session(resp)

    with patch.object(proxy_module, "zai_api_key", "sk-zai-test"), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch("aiohttp.ClientSession", return_value=session):
        with pytest.raises(HTTPException) as exc:
            await proxy_module.call_zai_direct(
                None, [{"role": "user", "content": "hi"}], "zai:gpt-4o", 100
            )
    assert exc.value.status_code == 429


@pytest.mark.asyncio
async def test_zai_direct_sends_bearer_auth(proxy_module):
    """Sends Bearer authorization header with the Z.AI API key."""
    resp = _make_mock_response(200, {"choices": [{"message": {"content": "ok"}}]})
    session = _make_mock_session(resp)

    with patch.object(proxy_module, "zai_api_key", "sk-zai-test"), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch("aiohttp.ClientSession", return_value=session):
        await proxy_module.call_zai_direct(
            None, [{"role": "user", "content": "hi"}], "zai:gpt-4o", 100
        )

    hdrs = session.post.call_args[1]["headers"]
    assert hdrs["Authorization"] == "Bearer sk-zai-test"


@pytest.mark.asyncio
async def test_zai_direct_includes_system_prompt(proxy_module):
    """System prompt is prepended as a system message."""
    resp = _make_mock_response(200, {"choices": [{"message": {"content": "ok"}}]})
    session = _make_mock_session(resp)

    with patch.object(proxy_module, "zai_api_key", "sk-zai-test"), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch("aiohttp.ClientSession", return_value=session):
        await proxy_module.call_zai_direct(
            "Be concise.", [{"role": "user", "content": "hi"}], "zai:gpt-4o", 100
        )

    payload = session.post.call_args[1]["json"]
    assert payload["messages"][0] == {"role": "system", "content": "Be concise."}
    assert payload["messages"][1] == {"role": "user", "content": "hi"}


# ---------------------------------------------------------------------------
# Z.AI Streaming (call_zai_streaming)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_zai_streaming_no_key_yields_error_and_done(proxy_module):
    """When no API key, yields error chunk and [DONE]."""
    with patch.object(proxy_module, "zai_api_key", None):
        chunks = []
        async for chunk in proxy_module.call_zai_streaming(
            None, [{"role": "user", "content": "hi"}], "zai:gpt-4o", 100
        ):
            chunks.append(chunk)

    combined = "".join(chunks)
    assert "[DONE]" in combined
    assert "not configured" in combined.lower() or "error" in combined.lower()


@pytest.mark.asyncio
async def test_zai_streaming_passthrough(proxy_module):
    """SSE bytes from Z.AI are passed through."""
    sse_bytes = b'data: {"choices":[{"delta":{"content":"world"}}]}\n\ndata: [DONE]\n\n'
    mock_resp = _make_streaming_response(200, [sse_bytes])
    session = _make_mock_session(mock_resp)

    with patch.object(proxy_module, "zai_api_key", "sk-zai-test"), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch("aiohttp.ClientSession", return_value=session):
        chunks = []
        async for chunk in proxy_module.call_zai_streaming(
            None, [{"role": "user", "content": "hi"}], "zai:gpt-4o", 100
        ):
            chunks.append(chunk)

    combined = "".join(chunks)
    assert "world" in combined


@pytest.mark.asyncio
async def test_zai_streaming_error_status_yields_error_and_done(proxy_module):
    """Non-200 status yields an error chunk and [DONE]."""
    mock_resp = _make_streaming_response(502, text_data="Bad Gateway")
    session = _make_mock_session(mock_resp)

    with patch.object(proxy_module, "zai_api_key", "sk-zai-test"), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch("aiohttp.ClientSession", return_value=session):
        chunks = []
        async for chunk in proxy_module.call_zai_streaming(
            None, [{"role": "user", "content": "hi"}], "zai:gpt-4o", 100
        ):
            chunks.append(chunk)

    combined = "".join(chunks)
    assert "[DONE]" in combined
    assert "Z.AI Error 502" in combined


@pytest.mark.asyncio
async def test_zai_streaming_strips_prefix(proxy_module):
    """Strips 'zai:' prefix from model name in streaming payload."""
    sse_bytes = b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'
    mock_resp = _make_streaming_response(200, [sse_bytes])
    session = _make_mock_session(mock_resp)

    with patch.object(proxy_module, "zai_api_key", "sk-zai-test"), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch("aiohttp.ClientSession", return_value=session):
        chunks = []
        async for chunk in proxy_module.call_zai_streaming(
            None, [{"role": "user", "content": "hi"}], "zai:claude-sonnet-4-6", 100
        ):
            chunks.append(chunk)

    payload = session.post.call_args[1]["json"]
    assert payload["model"] == "claude-sonnet-4-6"
    assert payload["stream"] is True


@pytest.mark.asyncio
async def test_zai_streaming_connection_error_yields_error(proxy_module):
    """ClientError during streaming yields error chunk + [DONE]."""
    import aiohttp

    session = MagicMock()
    conn_key = MagicMock()
    session.post = MagicMock(side_effect=aiohttp.ClientConnectorError(
        conn_key, OSError("Connection refused")
    ))
    session.close = AsyncMock()

    with patch.object(proxy_module, "zai_api_key", "sk-zai-test"), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch("aiohttp.ClientSession", return_value=session):
        chunks = []
        async for chunk in proxy_module.call_zai_streaming(
            None, [{"role": "user", "content": "hi"}], "zai:gpt-4o", 100
        ):
            chunks.append(chunk)

    combined = "".join(chunks)
    assert "[DONE]" in combined
    assert "Z.AI Error" in combined or "error" in combined.lower()


@pytest.mark.asyncio
async def test_zai_streaming_empty_body_retries_then_falls_back_to_direct(proxy_module):
    """If streaming returns an empty body, retry then fall back to direct JSON."""
    # Three 200 responses with an empty iter_any, then a direct JSON response.
    resp1 = _make_streaming_response(200, [])
    resp2 = _make_streaming_response(200, [])
    resp3 = _make_streaming_response(200, [])
    direct_resp = _make_mock_response(200, {"choices": [{"message": {"content": "Fallback"}}]})

    session = MagicMock()
    session.post = MagicMock(side_effect=[resp1, resp2, resp3, direct_resp])
    session.close = AsyncMock()

    with patch.object(proxy_module, "zai_api_key", "sk-zai-test"), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch("aiohttp.ClientSession", return_value=session), \
         patch("asyncio.sleep", new=AsyncMock()):
        chunks = []
        async for chunk in proxy_module.call_zai_streaming(
            None, [{"role": "user", "content": "hi"}], "zai:gpt-4o", 100
        ):
            chunks.append(chunk)

    assert session.post.call_count == 4
    combined = "".join(chunks)
    assert "[DONE]" in combined
    assert "Fallback" in combined


@pytest.mark.asyncio
async def test_zai_streaming_json_fallback(proxy_module):
    """If Z.AI returns JSON despite stream=True, convert it to SSE."""
    mock_resp = _make_streaming_response(200)
    mock_resp.headers = {"Content-Type": "application/json"}
    mock_resp.json = AsyncMock(return_value={"choices": [{"message": {"content": "Hello"}}]})
    session = _make_mock_session(mock_resp)

    with patch.object(proxy_module, "zai_api_key", "sk-zai-test"), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch("aiohttp.ClientSession", return_value=session):
        chunks = []
        async for chunk in proxy_module.call_zai_streaming(
            None, [{"role": "user", "content": "hi"}], "zai:gpt-4o", 100
        ):
            chunks.append(chunk)

    combined = "".join(chunks)
    assert "Hello" in combined
    assert "[DONE]" in combined


# ---------------------------------------------------------------------------
# Claude Streaming error paths (call_api_streaming)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_claude_streaming_non_200_yields_error_and_done(proxy_module):
    """When API returns non-200 (not auth, not 500+), yields error chunk + [DONE]."""
    resp = _make_mock_response(status=400, text_data="Bad Request")
    session = _make_mock_session(resp)

    auth_mock = MagicMock()
    auth_mock.session = session
    auth_mock.headers = {
        "x-api-key": "test",
        "anthropic-version": "2025-01-01",
        "Content-Type": "application/json",
    }
    auth_mock.body_template = {"model": "claude-sonnet-4-6", "max_tokens": 4096}

    with patch.object(proxy_module, "auth", auth_mock), \
         patch.object(proxy_module, "_cached_billing_block", None), \
         patch.object(proxy_module, "_cached_auth_blocks", []):
        chunks = []
        async for chunk in proxy_module.call_api_streaming(
            "You are helpful.", [{"role": "user", "content": "hi"}], "claude-sonnet-4-6", 4096
        ):
            chunks.append(chunk)

    combined = "".join(chunks)
    assert "[DONE]" in combined
    assert "API Error 400" in combined


@pytest.mark.asyncio
async def test_claude_streaming_uses_existing_session(proxy_module):
    """When auth.session exists, uses it (no new session created)."""
    resp = _make_mock_response(status=400, text_data="Bad Request")
    existing_session = _make_mock_session(resp)

    auth_mock = MagicMock()
    auth_mock.session = existing_session
    auth_mock.headers = {
        "x-api-key": "test",
        "anthropic-version": "2025-01-01",
        "Content-Type": "application/json",
    }
    auth_mock.body_template = {"model": "claude-sonnet-4-6", "max_tokens": 4096}

    with patch.object(proxy_module, "auth", auth_mock), \
         patch.object(proxy_module, "_cached_billing_block", None), \
         patch.object(proxy_module, "_cached_auth_blocks", []), \
         patch("aiohttp.ClientSession") as mock_create:
        chunks = []
        async for chunk in proxy_module.call_api_streaming(
            None, [{"role": "user", "content": "hi"}], "claude-sonnet-4-6", 4096
        ):
            chunks.append(chunk)

    # Should not have created a new session
    mock_create.assert_not_called()
    # Existing session should have been used
    existing_session.post.assert_called_once()


@pytest.mark.asyncio
async def test_claude_streaming_posts_to_anthropic_endpoint(proxy_module):
    """Sends request to the Anthropic messages API endpoint."""
    resp = _make_mock_response(status=400, text_data="Bad Request")
    session = _make_mock_session(resp)

    auth_mock = MagicMock()
    auth_mock.session = session
    auth_mock.headers = {
        "x-api-key": "test",
        "anthropic-version": "2025-01-01",
        "Content-Type": "application/json",
    }
    auth_mock.body_template = {"model": "claude-sonnet-4-6", "max_tokens": 4096}

    with patch.object(proxy_module, "auth", auth_mock), \
         patch.object(proxy_module, "_cached_billing_block", None), \
         patch.object(proxy_module, "_cached_auth_blocks", []):
        chunks = []
        async for chunk in proxy_module.call_api_streaming(
            None, [{"role": "user", "content": "hi"}], "claude-sonnet-4-6", 4096
        ):
            chunks.append(chunk)

    url = session.post.call_args[0][0]
    assert url == "https://api.anthropic.com/v1/messages?beta=true"


@pytest.mark.asyncio
async def test_claude_streaming_success_yields_role_content_done(proxy_module):
    """Successful stream yields role chunk, content, stop chunk, and [DONE]."""
    # Simulate Claude SSE stream with content_block_delta events
    sse_lines = (
        b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_1"}}\n\n'
        b'event: content_block_start\ndata: {"type":"content_block_start","index":0}\n\n'
        b'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}\n\n'
        b'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"type":"text_delta","text":" world"}}\n\n'
        b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}\n\n'
        b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
    )

    mock_resp = _make_streaming_response(200, [sse_lines])
    session = _make_mock_session(mock_resp)

    auth_mock = MagicMock()
    auth_mock.session = session
    auth_mock.headers = {
        "x-api-key": "test",
        "anthropic-version": "2025-01-01",
        "Content-Type": "application/json",
    }
    auth_mock.body_template = {"model": "claude-sonnet-4-6", "max_tokens": 4096}

    with patch.object(proxy_module, "auth", auth_mock), \
         patch.object(proxy_module, "_cached_billing_block", None), \
         patch.object(proxy_module, "_cached_auth_blocks", []):
        chunks = []
        async for chunk in proxy_module.call_api_streaming(
            None, [{"role": "user", "content": "hi"}], "claude-sonnet-4-6", 4096
        ):
            chunks.append(chunk)

    combined = "".join(chunks)
    assert "[DONE]" in combined
    assert "Hello" in combined
    assert " world" in combined
    # Should contain role chunk at the start
    assert '"role": "assistant"' in combined or '"role":"assistant"' in combined
    # Should contain finish_reason stop
    assert "stop" in combined
