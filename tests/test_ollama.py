"""Tests for Ollama provider: detection, routing, call functions, endpoints."""
import pytest
import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock


# ---------------------------------------------------------------------------
# Detection + routing
# ---------------------------------------------------------------------------

def test_is_ollama_model_simple(proxy_module):
    assert proxy_module.is_ollama_model("ollama:llama3.2") is True

def test_is_ollama_model_with_tag(proxy_module):
    assert proxy_module.is_ollama_model("ollama:mistral:7b") is True

def test_is_ollama_model_with_slash_in_name(proxy_module):
    """Slash inside ollama model name must still be detected as Ollama, not OpenRouter."""
    assert proxy_module.is_ollama_model("ollama:namespace/model") is True

def test_is_ollama_model_case_insensitive(proxy_module):
    assert proxy_module.is_ollama_model("OLLAMA:llama3.2") is True

def test_is_ollama_model_false_openrouter(proxy_module):
    assert proxy_module.is_ollama_model("openai/gpt-4o") is False

def test_is_ollama_model_false_claude(proxy_module):
    assert proxy_module.is_ollama_model("claude-sonnet-4-6") is False

def test_is_ollama_model_false_codex(proxy_module):
    assert proxy_module.is_ollama_model("gpt-5.4") is False


def test_routing_ollama_slash_model_not_routed_to_openrouter(proxy_module):
    """ollama:ns/model contains '/' but must route to Ollama only."""
    model = "ollama:namespace/model"
    use_ollama = proxy_module.is_ollama_model(model)
    use_openrouter = not use_ollama and proxy_module.is_openrouter_model(model)
    assert use_ollama is True
    assert use_openrouter is False

def test_routing_openrouter_unaffected(proxy_module):
    model = "openai/gpt-4o"
    use_ollama = proxy_module.is_ollama_model(model)
    use_openrouter = not use_ollama and proxy_module.is_openrouter_model(model)
    assert use_ollama is False
    assert use_openrouter is True

def test_routing_codex_unaffected(proxy_module):
    model = "gpt-5.4"
    use_ollama = proxy_module.is_ollama_model(model)
    use_codex = not use_ollama and proxy_module.is_codex_model(model)
    assert use_ollama is False
    assert use_codex is True

def test_routing_antigravity_unaffected(proxy_module):
    model = "antigravity-gemini-2.5-pro"
    use_ollama = proxy_module.is_ollama_model(model)
    use_antigravity = not use_ollama and proxy_module.is_antigravity_model(model)
    assert use_ollama is False
    assert use_antigravity is True


# ---------------------------------------------------------------------------
# call_ollama_direct
# ---------------------------------------------------------------------------

def _make_mock_response(status=200, json_data=None, text_data=""):
    """Build a mock aiohttp response usable as an async context manager."""
    mock_resp = AsyncMock()
    mock_resp.status = status
    mock_resp.json = AsyncMock(return_value=json_data or {})
    mock_resp.text = AsyncMock(return_value=text_data)
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    return mock_resp


def _make_mock_session(resp):
    """Build a mock aiohttp session whose .post() returns the given response."""
    session = MagicMock()
    session.post = MagicMock(return_value=resp)
    session.close = AsyncMock()
    return session


@pytest.mark.asyncio
async def test_call_ollama_direct_strips_prefix_local(proxy_module):
    """Strips 'ollama:' prefix before sending model name to local endpoint."""
    resp = _make_mock_response(200, {"choices": [{"message": {"content": "Hi"}}]})
    session = _make_mock_session(resp)

    with patch.object(proxy_module, "ollama_api_key", None), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch("aiohttp.ClientSession", return_value=session):
        result = await proxy_module.call_ollama_direct(
            None, [{"role": "user", "content": "hi"}], "ollama:llama3.2", 100
        )

    assert result == "Hi"
    url = session.post.call_args[0][0]
    assert url == "http://localhost:11434/v1/chat/completions"
    payload = session.post.call_args[1]["json"]
    assert payload["model"] == "llama3.2"  # prefix stripped


@pytest.mark.asyncio
async def test_call_ollama_direct_cloud_endpoint_and_auth(proxy_module):
    """When key is set, uses cloud endpoint with Bearer token."""
    resp = _make_mock_response(200, {"choices": [{"message": {"content": "Cloud hi"}}]})
    session = _make_mock_session(resp)

    with patch.object(proxy_module, "ollama_api_key", "sk-ollama-test"), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch("aiohttp.ClientSession", return_value=session):
        result = await proxy_module.call_ollama_direct(
            None, [{"role": "user", "content": "hi"}], "ollama:llama3.2", 100
        )

    assert result == "Cloud hi"
    url = session.post.call_args[0][0]
    assert url == "https://ollama.com/v1/chat/completions"
    hdrs = session.post.call_args[1]["headers"]
    assert hdrs["Authorization"] == "Bearer sk-ollama-test"


@pytest.mark.asyncio
async def test_call_ollama_direct_strips_unsupported_top_k(proxy_module):
    """Ollama OpenAI-compat requests should not forward unsupported top_k."""
    resp = _make_mock_response(200, {"choices": [{"message": {"content": "Cloud hi"}}]})
    session = _make_mock_session(resp)

    with patch.object(proxy_module, "ollama_api_key", "sk-ollama-test"), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch("aiohttp.ClientSession", return_value=session):
        result = await proxy_module.call_ollama_direct(
            None,
            [{"role": "user", "content": "hi"}],
            "ollama:llama3.2",
            100,
            temperature=0.2,
            top_k=40,
        )

    assert result == "Cloud hi"
    payload = session.post.call_args[1]["json"]
    assert payload["temperature"] == 0.2
    assert "top_k" not in payload


@pytest.mark.asyncio
async def test_call_ollama_direct_reasoning_only_response_raises_502(proxy_module):
    """A reasoning-only Ollama response should not crash the proxy with KeyError."""
    from fastapi import HTTPException

    resp = _make_mock_response(
        200,
        {
            "choices": [{
                "finish_reason": "stop",
                "message": {"reasoning": "internal trace"},
            }]
        },
    )
    session = _make_mock_session(resp)

    with patch.object(proxy_module, "ollama_api_key", None), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch("aiohttp.ClientSession", return_value=session):
        with pytest.raises(HTTPException) as exc:
            await proxy_module.call_ollama_direct(
                None, [{"role": "user", "content": "hi"}], "ollama:llama3.2", 100
            )

    assert exc.value.status_code == 502


@pytest.mark.asyncio
async def test_call_ollama_direct_retries_reasoning_truncation_without_max_tokens(proxy_module):
    """A reasoning-truncated Ollama response should retry once without max_tokens."""
    first_resp = _make_mock_response(
        200,
        {
            "choices": [{
                "finish_reason": "length",
                "message": {"reasoning_content": "internal trace"},
            }]
        },
    )
    second_resp = _make_mock_response(
        200,
        {"choices": [{"message": {"content": "final answer"}}]},
    )
    session = MagicMock()
    session.post = MagicMock(side_effect=[first_resp, second_resp])
    session.close = AsyncMock()

    with patch.object(proxy_module, "ollama_api_key", None), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch("aiohttp.ClientSession", return_value=session):
        result = await proxy_module.call_ollama_direct(
            None, [{"role": "user", "content": "hi"}], "ollama:llama3.2", 100
        )

    assert result == "final answer"
    first_payload = session.post.call_args_list[0].kwargs["json"]
    second_payload = session.post.call_args_list[1].kwargs["json"]
    assert first_payload["max_tokens"] == 100
    assert "max_tokens" not in second_payload


@pytest.mark.asyncio
async def test_call_ollama_direct_retries_reasoning_only_stop_without_max_tokens(proxy_module):
    """Some Ollama reasoning responses stop without content; retry once without max_tokens."""
    first_resp = _make_mock_response(
        200,
        {
            "choices": [{
                "finish_reason": "stop",
                "message": {"reasoning_content": "internal trace"},
            }]
        },
    )
    second_resp = _make_mock_response(
        200,
        {"choices": [{"message": {"content": "final answer"}}]},
    )
    session = MagicMock()
    session.post = MagicMock(side_effect=[first_resp, second_resp])
    session.close = AsyncMock()

    with patch.object(proxy_module, "ollama_api_key", None), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch("aiohttp.ClientSession", return_value=session):
        result = await proxy_module.call_ollama_direct(
            None, [{"role": "user", "content": "hi"}], "ollama:llama3.2", 100
        )

    assert result == "final answer"
    first_payload = session.post.call_args_list[0].kwargs["json"]
    second_payload = session.post.call_args_list[1].kwargs["json"]
    assert first_payload["max_tokens"] == 100
    assert "max_tokens" not in second_payload


@pytest.mark.asyncio
async def test_call_ollama_streaming_appends_done_when_passthrough_omits_it(proxy_module):
    """Ollama streaming should add a clean terminator if passthrough does not."""
    mock_resp = _make_mock_response(200, {})
    session = _make_mock_session(mock_resp)

    async def fake_passthrough(*args, **kwargs):
        yield 'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'

    with patch.object(proxy_module, "ollama_api_key", None), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch.object(proxy_module, "create_session", return_value=session), \
         patch.object(proxy_module, "_open_stream_with_retry", new_callable=AsyncMock, return_value=(mock_resp, mock_resp)), \
         patch.object(proxy_module, "passthrough_sse", side_effect=fake_passthrough):
        chunks = []
        async for chunk in proxy_module.call_ollama_streaming(
            None, [{"role": "user", "content": "hi"}], "ollama:llama3.2", 100
        ):
            chunks.append(chunk)

    assert chunks[-1] == "data: [DONE]\n\n"
    assert "".join(chunks).count("[DONE]") == 1


@pytest.mark.asyncio
async def test_call_ollama_direct_unreachable_raises_503(proxy_module):
    """ClientConnectorError -> HTTPException 503."""
    import aiohttp
    from fastapi import HTTPException

    session = MagicMock()
    conn_key = MagicMock()
    session.post = MagicMock(side_effect=aiohttp.ClientConnectorError(
        conn_key, OSError("Connection refused")
    ))
    session.close = AsyncMock()

    with patch.object(proxy_module, "ollama_api_key", None), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch("aiohttp.ClientSession", return_value=session):
        with pytest.raises(HTTPException) as exc:
            await proxy_module.call_ollama_direct(
                None, [{"role": "user", "content": "hi"}], "ollama:llama3.2", 100
            )
    assert exc.value.status_code == 503


# ---------------------------------------------------------------------------
# call_ollama_streaming
# ---------------------------------------------------------------------------

async def _async_bytes_iter(chunks):
    """Async generator that yields byte chunks, simulating aiohttp iter_any."""
    for chunk in chunks:
        yield chunk


@pytest.mark.asyncio
async def test_call_ollama_streaming_passthrough(proxy_module):
    """SSE bytes from Ollama are passed through verbatim."""
    sse_bytes = b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\ndata: [DONE]\n\n'

    mock_content = MagicMock()
    mock_content.iter_any.return_value = _async_bytes_iter([sse_bytes])

    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.content = mock_content
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    session = MagicMock()
    session.post = MagicMock(return_value=mock_resp)
    session.close = AsyncMock()

    with patch.object(proxy_module, "ollama_api_key", None), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch("aiohttp.ClientSession", return_value=session):
        chunks = []
        async for chunk in proxy_module.call_ollama_streaming(
            None, [{"role": "user", "content": "hi"}], "ollama:llama3.2", 100
        ):
            chunks.append(chunk)

    combined = "".join(chunks)
    assert "hello" in combined


@pytest.mark.asyncio
async def test_call_ollama_streaming_strips_unsupported_top_k(proxy_module):
    """Streaming requests should also drop unsupported top_k for Ollama."""
    sse_bytes = b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\ndata: [DONE]\n\n'

    mock_content = MagicMock()
    mock_content.iter_any.return_value = _async_bytes_iter([sse_bytes])

    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.content = mock_content
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    session = MagicMock()
    session.post = MagicMock(return_value=mock_resp)
    session.close = AsyncMock()

    with patch.object(proxy_module, "ollama_api_key", "sk-ollama-test"), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch("aiohttp.ClientSession", return_value=session):
        chunks = []
        async for chunk in proxy_module.call_ollama_streaming(
            None,
            [{"role": "user", "content": "hi"}],
            "ollama:llama3.2",
            100,
            reasoning={"effort": "low"},
            top_k=20,
        ):
            chunks.append(chunk)

    combined = "".join(chunks)
    payload = session.post.call_args[1]["json"]
    assert "hello" in combined
    # Both reasoning and thinking are stripped — Ollama doesn't support them
    assert "reasoning" not in payload
    assert "thinking" not in payload
    assert "top_k" not in payload


@pytest.mark.asyncio
async def test_call_ollama_streaming_unreachable_yields_error_and_done(proxy_module):
    """ClientConnectorError -> SSE error chunk + [DONE], no exception raised."""
    import aiohttp

    session = MagicMock()
    conn_key = MagicMock()
    session.post = MagicMock(side_effect=aiohttp.ClientConnectorError(
        conn_key, OSError("refused")
    ))
    session.close = AsyncMock()

    with patch.object(proxy_module, "ollama_api_key", None), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch("aiohttp.ClientSession", return_value=session):
        chunks = []
        async for chunk in proxy_module.call_ollama_streaming(
            None, [{"role": "user", "content": "hi"}], "ollama:llama3.2", 100
        ):
            chunks.append(chunk)

    combined = "".join(chunks)
    assert "[DONE]" in combined
    assert "Ollama" in combined or "error" in combined.lower() or "503" in combined


# ---------------------------------------------------------------------------
# /config/ollama-key endpoint
# ---------------------------------------------------------------------------

def test_set_ollama_key_saves(test_client, proxy_module):
    """POST with non-empty key saves it and returns status=saved."""
    with patch.object(proxy_module, "_save_config"), \
         patch.object(proxy_module, "_load_config", return_value={}):
        resp = test_client.post("/config/ollama-key", json={"key": "sk-test-key"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "saved"
    assert proxy_module.ollama_api_key == "sk-test-key"


def test_set_ollama_key_clears(test_client, proxy_module):
    """POST with empty key clears it and returns status=cleared."""
    proxy_module.ollama_api_key = "existing-key"
    with patch.object(proxy_module, "_save_config"), \
         patch.object(proxy_module, "_load_config", return_value={"ollama_api_key": "existing-key"}):
        resp = test_client.post("/config/ollama-key", json={"key": ""})
    assert resp.status_code == 200
    assert resp.json()["status"] == "cleared"
    assert proxy_module.ollama_api_key is None
