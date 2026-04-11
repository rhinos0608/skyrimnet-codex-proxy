"""Tests for NVIDIA NIM provider: detection, model resolution, call functions."""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Detection + model resolution
# ---------------------------------------------------------------------------

def test_is_nvidia_model_simple(proxy_module):
    assert proxy_module.is_nvidia_model("nvidia:deepseek-r1") is True


def test_is_nvidia_model_case_insensitive(proxy_module):
    assert proxy_module.is_nvidia_model("NVIDIA:llama-3.3-70b") is True


def test_is_nvidia_model_false_claude(proxy_module):
    assert proxy_module.is_nvidia_model("claude-sonnet-4-6") is False


def test_is_nvidia_model_false_openrouter(proxy_module):
    assert proxy_module.is_nvidia_model("openai/gpt-4o") is False


def test_is_nvidia_model_false_ollama(proxy_module):
    assert proxy_module.is_nvidia_model("ollama:llama3.2") is False


def test_nvidia_with_slash_not_openrouter(proxy_module):
    """nvidia: models containing '/' must NOT be detected as OpenRouter."""
    assert proxy_module.is_nvidia_model("nvidia:moonshotai/kimi-k2-instruct-0905") is True
    assert proxy_module.is_openrouter_model("nvidia:moonshotai/kimi-k2-instruct-0905") is False


def test_nvidia_full_path_passthrough(proxy_module):
    """Full org/model paths after nvidia: prefix pass through to the API."""
    assert proxy_module._resolve_nvidia_model("nvidia:moonshotai/kimi-k2-instruct-0905") == "moonshotai/kimi-k2-instruct-0905"


def test_resolve_nvidia_model_alias(proxy_module):
    """Short aliases expand to full API model paths."""
    assert proxy_module._resolve_nvidia_model("nvidia:deepseek-v3.2") == "deepseek-ai/deepseek-v3.2"


def test_resolve_nvidia_model_full_path(proxy_module):
    """Full API paths pass through unchanged."""
    assert proxy_module._resolve_nvidia_model("nvidia:deepseek-ai/deepseek-r1") == "deepseek-ai/deepseek-r1"


def test_resolve_nvidia_model_unknown_alias(proxy_module):
    """Unknown short names pass through as-is."""
    assert proxy_module._resolve_nvidia_model("nvidia:some-future-model") == "some-future-model"


@pytest.mark.parametrize("alias, expected", [
    ("nvidia:kimi-k2", "moonshotai/kimi-k2-instruct"),
    ("nvidia:kimi-k2-0905", "moonshotai/kimi-k2-instruct-0905"),
    ("nvidia:kimi-k2-thinking", "moonshotai/kimi-k2-thinking"),
    ("nvidia:deepseek-v3.1", "deepseek-ai/deepseek-v3.1"),
    ("nvidia:deepseek-v3.2", "deepseek-ai/deepseek-v3.2"),
    ("nvidia:mistral-large-3", "mistralai/mistral-large-3-675b-instruct-2512"),
    ("nvidia:devstral-2", "mistralai/devstral-2-123b-instruct-2512"),
    ("nvidia:glm-4.7", "z-ai/glm4.7"),
    ("nvidia:step-3.5-flash", "stepfun-ai/step-3.5-flash"),
    ("nvidia:llama-4-maverick", "meta/llama-4-maverick-17b-128e-instruct"),
    ("nvidia:gemma-3-27b", "google/gemma-3-27b-it"),
    ("nvidia:seed-oss-36b", "bytedance/seed-oss-36b-instruct"),
])
def test_resolve_nvidia_new_aliases(proxy_module, alias, expected):
    """New model aliases resolve to full NVIDIA NIM API paths."""
    assert proxy_module._resolve_nvidia_model(alias) == expected


def test_nvidia_in_oai_messages(proxy_module):
    """NVIDIA NIM models should use OpenAI-format messages."""
    assert proxy_module._model_uses_oai_messages("nvidia:deepseek-r1") is True


# ---------------------------------------------------------------------------
# Helpers (mirrors test_providers.py patterns)
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


# ---------------------------------------------------------------------------
# NVIDIA NIM Direct (call_nvidia_direct)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_nvidia_direct_no_key_raises_503(proxy_module):
    """Raises HTTPException 503 when nvidia_api_key is not set."""
    from fastapi import HTTPException

    with patch.object(proxy_module, "nvidia_api_key", None):
        with pytest.raises(HTTPException) as exc:
            await proxy_module.call_nvidia_direct(
                None, [{"role": "user", "content": "hi"}], "nvidia:deepseek-r1", 100
            )
    assert exc.value.status_code == 503


@pytest.mark.asyncio
async def test_nvidia_direct_sends_bearer_auth(proxy_module):
    """Sends Bearer authorization header with the NVIDIA API key."""
    resp = _make_mock_response(200, {"choices": [{"message": {"content": "Hello"}}]})
    session = _make_mock_session(resp)

    with patch.object(proxy_module, "nvidia_api_key", "nvapi-test-key"), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch("aiohttp.ClientSession", return_value=session):
        await proxy_module.call_nvidia_direct(
            None, [{"role": "user", "content": "hi"}], "nvidia:deepseek-r1", 100
        )

    hdrs = session.post.call_args[1]["headers"]
    assert hdrs["Authorization"] == "Bearer nvapi-test-key"


@pytest.mark.asyncio
async def test_nvidia_direct_sends_to_correct_url(proxy_module):
    """Posts to the NVIDIA NIM chat completions endpoint."""
    resp = _make_mock_response(200, {"choices": [{"message": {"content": "ok"}}]})
    session = _make_mock_session(resp)

    with patch.object(proxy_module, "nvidia_api_key", "nvapi-test-key"), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch("aiohttp.ClientSession", return_value=session):
        await proxy_module.call_nvidia_direct(
            None, [{"role": "user", "content": "hi"}], "nvidia:deepseek-r1", 100
        )

    url = session.post.call_args[0][0]
    assert url == "https://integrate.api.nvidia.com/v1/chat/completions"


@pytest.mark.asyncio
async def test_nvidia_direct_resolves_alias(proxy_module):
    """Short alias 'nvidia:deepseek-v3.2' resolves to full API path in the payload."""
    resp = _make_mock_response(200, {"choices": [{"message": {"content": "ok"}}]})
    session = _make_mock_session(resp)

    with patch.object(proxy_module, "nvidia_api_key", "nvapi-test-key"), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch("aiohttp.ClientSession", return_value=session):
        await proxy_module.call_nvidia_direct(
            None, [{"role": "user", "content": "hi"}], "nvidia:deepseek-v3.2", 100
        )

    payload = session.post.call_args[1]["json"]
    assert payload["model"] == "deepseek-ai/deepseek-v3.2"


@pytest.mark.asyncio
async def test_nvidia_direct_returns_content(proxy_module):
    """Returns the text from choices[0].message.content."""
    resp = _make_mock_response(200, {"choices": [{"message": {"content": "The answer is 42"}}]})
    session = _make_mock_session(resp)

    with patch.object(proxy_module, "nvidia_api_key", "nvapi-test-key"), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch("aiohttp.ClientSession", return_value=session):
        result = await proxy_module.call_nvidia_direct(
            None, [{"role": "user", "content": "hi"}], "nvidia:deepseek-r1", 100
        )

    assert result == "The answer is 42"


@pytest.mark.asyncio
async def test_nvidia_direct_401_raises_401(proxy_module):
    """401 from NVIDIA NIM raises HTTPException 401."""
    from fastapi import HTTPException

    resp = _make_mock_response(status=401, text_data="Unauthorized")
    session = _make_mock_session(resp)

    with patch.object(proxy_module, "nvidia_api_key", "nvapi-test-key"), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch("aiohttp.ClientSession", return_value=session):
        with pytest.raises(HTTPException) as exc:
            await proxy_module.call_nvidia_direct(
                None, [{"role": "user", "content": "hi"}], "nvidia:deepseek-r1", 100
            )
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_nvidia_direct_429_raises_429(proxy_module):
    """429 from NVIDIA NIM raises HTTPException 429."""
    from fastapi import HTTPException

    resp = _make_mock_response(status=429, text_data="Rate limit exceeded")
    session = _make_mock_session(resp)

    with patch.object(proxy_module, "nvidia_api_key", "nvapi-test-key"), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch("aiohttp.ClientSession", return_value=session):
        with pytest.raises(HTTPException) as exc:
            await proxy_module.call_nvidia_direct(
                None, [{"role": "user", "content": "hi"}], "nvidia:deepseek-r1", 100
            )
    assert exc.value.status_code == 429


@pytest.mark.asyncio
async def test_nvidia_direct_strips_unsupported_params(proxy_module):
    """Unsupported params (reasoning, top_k, provider) are not forwarded."""
    resp = _make_mock_response(200, {"choices": [{"message": {"content": "ok"}}]})
    session = _make_mock_session(resp)

    with patch.object(proxy_module, "nvidia_api_key", "nvapi-test-key"), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch("aiohttp.ClientSession", return_value=session):
        await proxy_module.call_nvidia_direct(
            None, [{"role": "user", "content": "hi"}], "nvidia:deepseek-r1", 100,
            reasoning={"enabled": False}, top_k=50, provider="nvidia"
        )

    payload = session.post.call_args[1]["json"]
    assert "reasoning" not in payload
    assert "top_k" not in payload
    assert "provider" not in payload


# ---------------------------------------------------------------------------
# NVIDIA NIM Streaming (call_nvidia_streaming)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_nvidia_streaming_no_key_yields_error_and_done(proxy_module):
    """When no API key, yields error message and [DONE]."""
    with patch.object(proxy_module, "nvidia_api_key", None):
        chunks = []
        async for chunk in proxy_module.call_nvidia_streaming(
            None, [{"role": "user", "content": "hi"}], "nvidia:deepseek-r1", 100
        ):
            chunks.append(chunk)

    combined = "".join(chunks)
    assert "[DONE]" in combined
    assert "not configured" in combined.lower() or "error" in combined.lower()


@pytest.mark.asyncio
async def test_nvidia_streaming_passthrough(proxy_module):
    """SSE bytes from NVIDIA NIM are passed through."""
    sse_bytes = b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\ndata: [DONE]\n\n'
    mock_resp = _make_streaming_response(200, [sse_bytes])
    session = _make_mock_session(mock_resp)

    with patch.object(proxy_module, "nvidia_api_key", "nvapi-test-key"), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch("aiohttp.ClientSession", return_value=session):
        chunks = []
        async for chunk in proxy_module.call_nvidia_streaming(
            None, [{"role": "user", "content": "hi"}], "nvidia:deepseek-r1", 100
        ):
            chunks.append(chunk)

    combined = "".join(chunks)
    assert "hello" in combined


@pytest.mark.asyncio
async def test_nvidia_streaming_error_status_yields_error_and_done(proxy_module):
    """Non-200 status yields an error chunk and [DONE]."""
    mock_resp = _make_streaming_response(500, text_data="Internal Server Error")
    session = _make_mock_session(mock_resp)

    with patch.object(proxy_module, "nvidia_api_key", "nvapi-test-key"), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch("aiohttp.ClientSession", return_value=session):
        chunks = []
        async for chunk in proxy_module.call_nvidia_streaming(
            None, [{"role": "user", "content": "hi"}], "nvidia:deepseek-r1", 100
        ):
            chunks.append(chunk)

    combined = "".join(chunks)
    assert "[DONE]" in combined
    assert "NVIDIA NIM Error 500" in combined


@pytest.mark.asyncio
async def test_nvidia_streaming_resolves_alias(proxy_module):
    """Short alias 'nvidia:deepseek-v3.2' resolves in streaming payload."""
    sse_bytes = b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'
    mock_resp = _make_streaming_response(200, [sse_bytes])
    session = _make_mock_session(mock_resp)

    with patch.object(proxy_module, "nvidia_api_key", "nvapi-test-key"), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch("aiohttp.ClientSession", return_value=session):
        chunks = []
        async for chunk in proxy_module.call_nvidia_streaming(
            None, [{"role": "user", "content": "hi"}], "nvidia:deepseek-v3.2", 100
        ):
            chunks.append(chunk)

    payload = session.post.call_args[1]["json"]
    assert payload["model"] == "deepseek-ai/deepseek-v3.2"
    assert payload["stream"] is True


@pytest.mark.asyncio
async def test_nvidia_streaming_connection_error_yields_error(proxy_module):
    """ClientError during streaming yields error chunk + [DONE]."""
    import aiohttp

    session = MagicMock()
    conn_key = MagicMock()
    session.post = MagicMock(side_effect=aiohttp.ClientConnectorError(
        conn_key, OSError("Connection refused")
    ))
    session.close = AsyncMock()

    with patch.object(proxy_module, "nvidia_api_key", "nvapi-test-key"), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch("aiohttp.ClientSession", return_value=session):
        chunks = []
        async for chunk in proxy_module.call_nvidia_streaming(
            None, [{"role": "user", "content": "hi"}], "nvidia:deepseek-r1", 100
        ):
            chunks.append(chunk)

    combined = "".join(chunks)
    assert "[DONE]" in combined
    assert "NVIDIA NIM Error" in combined or "error" in combined.lower()