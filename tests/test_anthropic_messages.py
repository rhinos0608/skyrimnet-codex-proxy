"""Tests for the Anthropic Messages API compatibility layer (/v1/messages)."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Unit tests — request translation helpers
# ---------------------------------------------------------------------------


class TestFlattenSystem:

    def test_none_returns_none(self, proxy_module):
        assert proxy_module._anthropic_flatten_system(None) is None

    def test_empty_string_returns_none(self, proxy_module):
        assert proxy_module._anthropic_flatten_system("") is None

    def test_plain_string_passthrough(self, proxy_module):
        assert proxy_module._anthropic_flatten_system("hello") == "hello"

    def test_list_of_text_blocks_concatenates(self, proxy_module):
        system = [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ]
        assert proxy_module._anthropic_flatten_system(system) == "first\n\nsecond"

    def test_list_with_empty_text_blocks_filtered(self, proxy_module):
        system = [
            {"type": "text", "text": ""},
            {"type": "text", "text": "real"},
        ]
        assert proxy_module._anthropic_flatten_system(system) == "real"

    def test_empty_list_returns_none(self, proxy_module):
        assert proxy_module._anthropic_flatten_system([]) is None


class TestFlattenContent:

    def test_string_content_passthrough(self, proxy_module):
        assert proxy_module._anthropic_flatten_content("hi there") == "hi there"

    def test_none_content_empty_string(self, proxy_module):
        assert proxy_module._anthropic_flatten_content(None) == ""

    def test_text_blocks_joined(self, proxy_module):
        content = [
            {"type": "text", "text": "alpha"},
            {"type": "text", "text": "beta"},
        ]
        assert proxy_module._anthropic_flatten_content(content) == "alpha\n\nbeta"

    def test_tool_use_block_rendered(self, proxy_module):
        content = [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {"cmd": "ls"}}]
        rendered = proxy_module._anthropic_flatten_content(content)
        assert "tool_use" in rendered
        assert "Bash" in rendered
        assert "ls" in rendered

    def test_tool_result_block_rendered_with_text_content(self, proxy_module):
        content = [{"type": "tool_result", "tool_use_id": "t1", "content": "output text"}]
        rendered = proxy_module._anthropic_flatten_content(content)
        assert "tool_result" in rendered
        assert "t1" in rendered
        assert "output text" in rendered

    def test_tool_result_block_rendered_with_nested_blocks(self, proxy_module):
        content = [{
            "type": "tool_result",
            "tool_use_id": "t2",
            "content": [{"type": "text", "text": "nested output"}],
        }]
        rendered = proxy_module._anthropic_flatten_content(content)
        assert "nested output" in rendered

    def test_tool_result_error_flag_shown(self, proxy_module):
        content = [{
            "type": "tool_result", "tool_use_id": "t3",
            "content": "boom", "is_error": True,
        }]
        rendered = proxy_module._anthropic_flatten_content(content)
        assert "error" in rendered

    def test_image_block_becomes_placeholder(self, proxy_module):
        content = [
            {"type": "text", "text": "look:"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "xxx"}},
        ]
        rendered = proxy_module._anthropic_flatten_content(content)
        assert "[image]" in rendered
        assert "look:" in rendered

    def test_thinking_block_dropped(self, proxy_module):
        content = [
            {"type": "thinking", "thinking": "internal"},
            {"type": "text", "text": "external"},
        ]
        rendered = proxy_module._anthropic_flatten_content(content)
        assert "internal" not in rendered
        assert rendered == "external"


class TestToolsToSystemHint:

    def test_no_tools_returns_none(self, proxy_module):
        assert proxy_module._anthropic_tools_to_system_hint(None) is None
        assert proxy_module._anthropic_tools_to_system_hint([]) is None

    def test_tools_rendered_as_bullet_list(self, proxy_module):
        tools = [
            {"name": "Bash", "description": "Run shell commands"},
            {"name": "Read", "description": "Read files"},
        ]
        hint = proxy_module._anthropic_tools_to_system_hint(tools)
        assert "Bash" in hint
        assert "Run shell commands" in hint
        assert "Read" in hint

    def test_tool_without_description_still_listed(self, proxy_module):
        tools = [{"name": "Custom"}]
        hint = proxy_module._anthropic_tools_to_system_hint(tools)
        assert "Custom" in hint

    def test_unnamed_tool_skipped(self, proxy_module):
        tools = [{"description": "no name"}, {"name": "Ok"}]
        hint = proxy_module._anthropic_tools_to_system_hint(tools)
        assert "no name" not in hint
        assert "Ok" in hint


class TestRequestToChatRequest:

    def test_basic_request(self, proxy_module):
        body = {
            "model": "ollama:llama3",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 500,
        }
        chat_req = proxy_module._anthropic_request_to_chat_request(body)
        assert chat_req.model == "ollama:llama3"
        assert chat_req.max_tokens == 500
        assert chat_req.stream is False
        assert len(chat_req.messages) == 1
        assert chat_req.messages[0].role == "user"
        assert chat_req.messages[0].content == "hello"

    def test_system_field_prepended_as_system_message(self, proxy_module):
        body = {
            "model": "ollama:llama3",
            "system": "You are helpful",
            "messages": [{"role": "user", "content": "hi"}],
        }
        chat_req = proxy_module._anthropic_request_to_chat_request(body)
        assert chat_req.messages[0].role == "system"
        assert chat_req.messages[0].content == "You are helpful"
        assert chat_req.messages[1].role == "user"

    def test_tools_appended_to_system_message(self, proxy_module):
        body = {
            "model": "ollama:llama3",
            "system": "Base",
            "tools": [{"name": "Bash", "description": "Shell"}],
            "messages": [{"role": "user", "content": "do it"}],
        }
        chat_req = proxy_module._anthropic_request_to_chat_request(body)
        assert chat_req.messages[0].role == "system"
        assert "Base" in chat_req.messages[0].content
        assert "Bash" in chat_req.messages[0].content

    def test_empty_messages_raises_400(self, proxy_module):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc:
            proxy_module._anthropic_request_to_chat_request({"messages": []})
        assert exc.value.status_code == 400

    def test_assistant_only_history_raises_400(self, proxy_module):
        """Anthropic requires at least one user message in the history."""
        from fastapi import HTTPException
        body = {
            "messages": [{"role": "assistant", "content": "prior"}],
        }
        with pytest.raises(HTTPException) as exc:
            proxy_module._anthropic_request_to_chat_request(body)
        assert exc.value.status_code == 400

    def test_non_dict_body_raises_400(self, proxy_module):
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            proxy_module._anthropic_request_to_chat_request("not a dict")

    def test_stop_sequences_mapped_to_stop(self, proxy_module):
        body = {
            "messages": [{"role": "user", "content": "hi"}],
            "stop_sequences": ["###", "END"],
        }
        chat_req = proxy_module._anthropic_request_to_chat_request(body)
        extra = chat_req.model_extra or {}
        assert extra.get("stop") == ["###", "END"]

    def test_stream_flag_propagated(self, proxy_module):
        body = {
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }
        chat_req = proxy_module._anthropic_request_to_chat_request(body)
        assert chat_req.stream is True

    def test_messages_with_content_block_lists(self, proxy_module):
        body = {
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "part A"},
                    {"type": "text", "text": "part B"},
                ],
            }],
        }
        chat_req = proxy_module._anthropic_request_to_chat_request(body)
        assert "part A" in chat_req.messages[0].content
        assert "part B" in chat_req.messages[0].content


# ---------------------------------------------------------------------------
# Unit tests — response translation
# ---------------------------------------------------------------------------


class TestOpenAIToAnthropicResponse:

    def test_basic_text_response(self, proxy_module):
        oai = {
            "choices": [{
                "message": {"role": "assistant", "content": "hello world"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
        }
        result = proxy_module._openai_completion_to_anthropic_message(oai, "ollama:llama3")
        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["model"] == "ollama:llama3"
        assert result["content"] == [{"type": "text", "text": "hello world"}]
        assert result["stop_reason"] == "end_turn"
        assert result["usage"]["input_tokens"] == 12
        assert result["usage"]["output_tokens"] == 3
        assert result["id"].startswith("msg_")

    @pytest.mark.parametrize("finish,expected", [
        ("stop", "end_turn"),
        ("length", "max_tokens"),
        ("content_filter", "stop_sequence"),
        ("tool_calls", "tool_use"),
        ("function_call", "tool_use"),
        ("weird_unknown", "end_turn"),
    ])
    def test_stop_reason_mapping(self, proxy_module, finish, expected):
        oai = {
            "choices": [{"message": {"content": "x"}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        result = proxy_module._openai_completion_to_anthropic_message(oai, "m")
        assert result["stop_reason"] == expected

    def test_missing_usage_approximates_output_tokens(self, proxy_module):
        oai = {"choices": [{"message": {"content": "abcd" * 10}, "finish_reason": "stop"}]}
        result = proxy_module._openai_completion_to_anthropic_message(oai, "m")
        assert result["usage"]["output_tokens"] >= 1
        assert result["usage"]["input_tokens"] == 0

    def test_empty_content_returns_empty_text_block(self, proxy_module):
        oai = {
            "choices": [{"message": {"content": None}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 0, "total_tokens": 5},
        }
        result = proxy_module._openai_completion_to_anthropic_message(oai, "m")
        assert result["content"] == [{"type": "text", "text": ""}]


# ---------------------------------------------------------------------------
# Unit tests — streaming translation
# ---------------------------------------------------------------------------


class TestStreamingTranslator:

    @pytest.mark.asyncio
    async def test_basic_text_stream(self, proxy_module):
        async def fake_openai_stream():
            yield 'data: {"choices":[{"delta":{"content":"Hel"},"finish_reason":null}]}\n\n'
            yield 'data: {"choices":[{"delta":{"content":"lo"},"finish_reason":null}]}\n\n'
            yield 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
            yield "data: [DONE]\n\n"

        events = []
        async for chunk in proxy_module._anthropic_stream_from_openai(
            fake_openai_stream(), "ollama:llama3", prompt_tokens_hint=7
        ):
            events.append(chunk)

        joined = "".join(events)
        assert "event: message_start" in joined
        assert "event: content_block_start" in joined
        assert "event: content_block_delta" in joined
        assert "Hel" in joined
        assert "lo" in joined
        assert "event: content_block_stop" in joined
        assert "event: message_delta" in joined
        assert '"stop_reason":"end_turn"' in joined
        assert "event: message_stop" in joined

    @pytest.mark.asyncio
    async def test_length_finish_reason_maps_to_max_tokens(self, proxy_module):
        async def fake_openai_stream():
            yield 'data: {"choices":[{"delta":{"content":"x"},"finish_reason":null}]}\n\n'
            yield 'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\n'
            yield "data: [DONE]\n\n"

        events = []
        async for chunk in proxy_module._anthropic_stream_from_openai(
            fake_openai_stream(), "m", 0
        ):
            events.append(chunk)
        joined = "".join(events)
        assert '"stop_reason":"max_tokens"' in joined

    @pytest.mark.asyncio
    async def test_raw_error_chunk_surfaces_as_text_delta(self, proxy_module):
        """A provider may emit {"error": "..."} with no choices array — translate it."""
        async def fake_error_stream():
            yield 'data: {"error": "API key missing"}\n\n'
            yield "data: [DONE]\n\n"

        events = []
        async for chunk in proxy_module._anthropic_stream_from_openai(
            fake_error_stream(), "m", 0
        ):
            events.append(chunk)
        joined = "".join(events)
        assert "API key missing" in joined
        assert "upstream error" in joined
        assert "content_block_delta" in joined

    @pytest.mark.asyncio
    async def test_structured_error_chunk_surfaces(self, proxy_module):
        """Errors in {"error": {"message": "..."}} form should also surface."""
        async def fake_error_stream():
            yield 'data: {"error": {"message": "quota exceeded", "type": "rate_limit"}}\n\n'
            yield "data: [DONE]\n\n"

        events = []
        async for chunk in proxy_module._anthropic_stream_from_openai(
            fake_error_stream(), "m", 0
        ):
            events.append(chunk)
        assert "quota exceeded" in "".join(events)

    @pytest.mark.asyncio
    async def test_upstream_exception_still_emits_message_stop(self, proxy_module):
        """If the upstream generator blows up, the client must still see a
        clean content_block_stop / message_delta / message_stop terminator
        instead of an indefinite hang."""
        async def exploding_stream():
            yield 'data: {"choices":[{"delta":{"content":"start"},"finish_reason":null}]}\n\n'
            raise RuntimeError("kaboom")

        events = []
        async for chunk in proxy_module._anthropic_stream_from_openai(
            exploding_stream(), "m", 0
        ):
            events.append(chunk)
        joined = "".join(events)
        assert "start" in joined
        assert "stream error" in joined
        assert "kaboom" in joined
        # Critical: the closing events must all be present.
        assert "event: content_block_stop" in joined
        assert "event: message_delta" in joined
        assert "event: message_stop" in joined

    @pytest.mark.asyncio
    async def test_empty_upstream_still_emits_closing_events(self, proxy_module):
        async def fake_empty_stream():
            yield "data: [DONE]\n\n"

        events = []
        async for chunk in proxy_module._anthropic_stream_from_openai(
            fake_empty_stream(), "m", 0
        ):
            events.append(chunk)
        joined = "".join(events)
        # Must still produce message_start / content_block_start / ...stop / message_delta / message_stop
        assert "message_start" in joined
        assert "content_block_start" in joined
        assert "content_block_stop" in joined
        assert "message_delta" in joined
        assert "message_stop" in joined


# ---------------------------------------------------------------------------
# Integration tests — /v1/messages end-to-end via TestClient
# ---------------------------------------------------------------------------


class TestMessagesEndpoint:

    def test_non_streaming_ollama(self, test_client, proxy_module):
        """POST /v1/messages with ollama model returns Anthropic message dict."""
        with patch.object(proxy_module, "call_ollama_direct",
                          new_callable=AsyncMock, return_value="assistant reply"), \
             patch.object(proxy_module, "request_stats", MagicMock(record=MagicMock())):
            resp = test_client.post("/v1/messages", json={
                "model": "ollama:llama3.2",
                "system": "You are a helper",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 200,
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "message"
        assert data["role"] == "assistant"
        assert data["content"] == [{"type": "text", "text": "assistant reply"}]
        assert data["model"] == "ollama:llama3.2"
        assert data["stop_reason"] == "end_turn"
        assert data["id"].startswith("msg_")
        assert "input_tokens" in data["usage"]
        assert "output_tokens" in data["usage"]

    def test_non_streaming_forwards_system_prompt(self, test_client, proxy_module):
        """System field should reach the underlying provider's messages list."""
        with patch.object(proxy_module, "call_ollama_direct",
                          new_callable=AsyncMock, return_value="ok") as mock_call, \
             patch.object(proxy_module, "request_stats", MagicMock(record=MagicMock())):
            test_client.post("/v1/messages", json={
                "model": "ollama:llama3.2",
                "system": "ROLE=helper",
                "messages": [{"role": "user", "content": "hi"}],
            })
        args = mock_call.call_args.args
        # For ollama (OpenAI-message path), system_prompt is None and the
        # system message lives inside the messages list.
        _system_prompt, messages, _model, _max_tokens = args
        assert any(
            m["role"] == "system" and "ROLE=helper" in m["content"]
            for m in messages
        )

    def test_streaming_returns_sse(self, test_client, proxy_module):
        async def fake_stream(*args, **kwargs):
            yield 'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}\n\n'
            yield 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
            yield "data: [DONE]\n\n"

        with patch.object(proxy_module, "call_ollama_streaming",
                          side_effect=lambda *a, **k: fake_stream()), \
             patch.object(proxy_module, "timeout_routing_enabled", False), \
             patch.object(proxy_module, "request_stats", MagicMock(record=MagicMock())):
            resp = test_client.post("/v1/messages", json={
                "model": "ollama:llama3.2",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            })
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        body = resp.text
        assert "event: message_start" in body
        assert "event: content_block_delta" in body
        assert "hi" in body
        assert "event: message_stop" in body

    def test_missing_user_message_returns_400(self, test_client, proxy_module):
        resp = test_client.post("/v1/messages", json={
            "model": "ollama:llama3.2",
            "messages": [{"role": "assistant", "content": "prior"}],
        })
        assert resp.status_code == 400

    def test_invalid_json_returns_400(self, test_client, proxy_module):
        resp = test_client.post(
            "/v1/messages",
            content=b"not-json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400


class TestCountTokensEndpoint:

    def test_basic_count(self, test_client, proxy_module):
        resp = test_client.post("/v1/messages/count_tokens", json={
            "model": "ollama:llama3.2",
            "messages": [{"role": "user", "content": "hello world"}],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "input_tokens" in data
        assert data["input_tokens"] >= 1

    def test_counts_system_and_tools(self, test_client, proxy_module):
        resp = test_client.post("/v1/messages/count_tokens", json={
            "model": "ollama:llama3.2",
            "system": "a" * 40,
            "tools": [{"name": "Bash", "description": "run shell",
                       "input_schema": {"type": "object", "properties": {"cmd": {"type": "string"}}}}],
            "messages": [{"role": "user", "content": "b" * 40}],
        })
        assert resp.status_code == 200
        data = resp.json()
        # 40 + 40 + schema text should push well above a single token
        assert data["input_tokens"] >= 10

    def test_invalid_json_returns_400(self, test_client, proxy_module):
        resp = test_client.post(
            "/v1/messages/count_tokens",
            content=b"nope",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400
