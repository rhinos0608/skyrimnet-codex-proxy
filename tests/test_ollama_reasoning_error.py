"""Tests for improved error messaging when Ollama returns reasoning but no content."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException


async def _passthrough_retry(fn, **kw):
    return await fn()


def _make_oai_response(content=None, reasoning=None, finish_reason="stop"):
    """Build a minimal OpenAI-compatible response dict."""
    message = {"role": "assistant"}
    if content is not None:
        message["content"] = content
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {
        "choices": [{"message": message, "finish_reason": finish_reason}],
    }


def _mock_response(status, json_data):
    resp = AsyncMock()
    resp.status = status
    resp.json = AsyncMock(return_value=json_data)
    resp.text = AsyncMock(return_value="error")
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)
    return resp


@pytest.mark.asyncio
async def test_reasoning_without_content_after_retry_gives_descriptive_error(proxy_module):
    """When both original and retry return reasoning but no content, error should mention reasoning."""
    original_resp = _make_oai_response(content=None, reasoning="Let me think about this...")
    retry_resp_data = _make_oai_response(content=None, reasoning="Still thinking...")

    mock_session = MagicMock()
    mock_session.post = MagicMock(side_effect=[
        _mock_response(200, original_resp),
        _mock_response(200, retry_resp_data),
    ])

    with patch.multiple(
        "proxy",
        ollama_api_key="fake-key",
        ollama_session=mock_session,
        _with_retry=_passthrough_retry,
    ):
        with pytest.raises(HTTPException) as exc_info:
            from proxy_internal.providers.ollama import call_ollama_direct
            await call_ollama_direct(
                system_prompt="You are helpful.",
                messages=[{"role": "user", "content": "Hello"}],
                model="ollama:deepseek-r1",
                max_tokens=1024,
            )

        assert exc_info.value.status_code == 502
        assert "reasoning tokens but no content" in exc_info.value.detail
        assert mock_session.post.call_count == 2


@pytest.mark.asyncio
async def test_truly_empty_response_keeps_generic_message(proxy_module):
    """When response has no reasoning and no content, use the generic error."""
    empty_resp = _make_oai_response(content=None, reasoning=None)

    mock_session = MagicMock()
    # No retry will happen for empty response without reasoning, so only one call
    mock_session.post = MagicMock(return_value=_mock_response(200, empty_resp))

    with patch.multiple(
        "proxy",
        ollama_api_key="fake-key",
        ollama_session=mock_session,
        _with_retry=_passthrough_retry,
    ):
        with pytest.raises(HTTPException) as exc_info:
            from proxy_internal.providers.ollama import call_ollama_direct
            await call_ollama_direct(
                system_prompt="You are helpful.",
                messages=[{"role": "user", "content": "Hello"}],
                model="ollama:llama3",
                max_tokens=1024,
            )

        assert exc_info.value.status_code == 502
        assert exc_info.value.detail == "Ollama returned no content"
        assert "reasoning" not in exc_info.value.detail
        assert mock_session.post.call_count == 1


@pytest.mark.asyncio
async def test_reasoning_retry_succeeds_returns_content(proxy_module):
    """When first response has reasoning-only but retry succeeds with content, return content."""
    original_resp = _make_oai_response(content=None, reasoning="Let me think about this...")
    retry_resp_data = _make_oai_response(content="Here is the answer.", reasoning="Thought about it.")

    mock_session = MagicMock()
    mock_session.post = MagicMock(side_effect=[
        _mock_response(200, original_resp),
        _mock_response(200, retry_resp_data),
    ])

    with patch.multiple(
        "proxy",
        ollama_api_key="fake-key",
        ollama_session=mock_session,
        _with_retry=_passthrough_retry,
    ):
        from proxy_internal.providers.ollama import call_ollama_direct
        result = await call_ollama_direct(
            system_prompt="You are helpful.",
            messages=[{"role": "user", "content": "Hello"}],
            model="ollama:deepseek-r1",
            max_tokens=1024,
        )

    assert result == "Here is the answer."
    assert mock_session.post.call_count == 2
