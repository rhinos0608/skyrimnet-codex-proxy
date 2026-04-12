"""Tests for utility functions: parse_model_list, pick_model_round_robin,
normalize_model_name, and _make_streaming_gen routing dispatch."""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock


# ---------------------------------------------------------------------------
# parse_model_list
# ---------------------------------------------------------------------------

def test_parse_model_list_single(proxy_module):
    assert proxy_module.parse_model_list("claude-sonnet-4-6") == ["claude-sonnet-4-6"]


def test_parse_model_list_comma_separated(proxy_module):
    result = proxy_module.parse_model_list("model-a,model-b,model-c")
    assert result == ["model-a", "model-b", "model-c"]


def test_parse_model_list_whitespace_handling(proxy_module):
    result = proxy_module.parse_model_list("  model-a , model-b , model-c  ")
    assert result == ["model-a", "model-b", "model-c"]


def test_parse_model_list_empty_string(proxy_module):
    assert proxy_module.parse_model_list("") == []


def test_parse_model_list_only_whitespace(proxy_module):
    assert proxy_module.parse_model_list("   ") == []


def test_parse_model_list_trailing_comma(proxy_module):
    result = proxy_module.parse_model_list("model-a,model-b,")
    assert result == ["model-a", "model-b"]


def test_parse_model_list_leading_comma(proxy_module):
    result = proxy_module.parse_model_list(",model-a,model-b")
    assert result == ["model-a", "model-b"]


def test_parse_model_list_multiple_commas(proxy_module):
    result = proxy_module.parse_model_list("model-a,,model-b,,,model-c")
    assert result == ["model-a", "model-b", "model-c"]


# ---------------------------------------------------------------------------
# pick_model_round_robin
# ---------------------------------------------------------------------------

def test_pick_model_round_robin_cycles(proxy_module):
    """Models are selected in round-robin order."""
    saved = proxy_module._round_robin_counter
    try:
        proxy_module._round_robin_counter = 0
        models = ["a", "b", "c"]
        picks = [proxy_module.pick_model_round_robin(models) for _ in range(3)]
        assert picks == ["a", "b", "c"]
    finally:
        proxy_module._round_robin_counter = saved


def test_pick_model_round_robin_wraps_around(proxy_module):
    """After exhausting the list, selection wraps back to the beginning."""
    saved = proxy_module._round_robin_counter
    try:
        proxy_module._round_robin_counter = 0
        models = ["x", "y"]
        picks = [proxy_module.pick_model_round_robin(models) for _ in range(5)]
        assert picks == ["x", "y", "x", "y", "x"]
    finally:
        proxy_module._round_robin_counter = saved


def test_pick_model_round_robin_single_model(proxy_module):
    """Single-model list always returns the same model."""
    saved = proxy_module._round_robin_counter
    try:
        proxy_module._round_robin_counter = 0
        models = ["only"]
        picks = [proxy_module.pick_model_round_robin(models) for _ in range(3)]
        assert picks == ["only", "only", "only"]
    finally:
        proxy_module._round_robin_counter = saved


def test_pick_model_round_robin_increments_counter(proxy_module):
    """Each call increments the global counter by 1."""
    saved = proxy_module._round_robin_counter
    try:
        proxy_module._round_robin_counter = 0
        proxy_module.pick_model_round_robin(["a", "b"])
        assert proxy_module._round_robin_counter == 1
        proxy_module.pick_model_round_robin(["a", "b"])
        assert proxy_module._round_robin_counter == 2
    finally:
        proxy_module._round_robin_counter = saved


def test_pick_model_round_robin_large_counter(proxy_module):
    """Modulo arithmetic works correctly when counter is already large."""
    saved = proxy_module._round_robin_counter
    try:
        proxy_module._round_robin_counter = 1000000
        models = ["a", "b", "c"]
        result = proxy_module.pick_model_round_robin(models)
        expected = models[1000000 % 3]
        assert result == expected
    finally:
        proxy_module._round_robin_counter = saved


# ---------------------------------------------------------------------------
# normalize_model_name
# ---------------------------------------------------------------------------

def test_normalize_model_name_claude_sonnet_46(proxy_module):
    assert proxy_module.normalize_model_name("claude-sonnet-4.6") == "claude-sonnet-4-6"


def test_normalize_model_name_claude_opus_46(proxy_module):
    assert proxy_module.normalize_model_name("claude-opus-4.6") == "claude-opus-4-6"


def test_normalize_model_name_claude_haiku_45(proxy_module):
    assert proxy_module.normalize_model_name("claude-haiku-4.5") == "claude-haiku-4-5-20251001"


def test_normalize_model_name_claude_sonnet_45(proxy_module):
    assert proxy_module.normalize_model_name("claude-sonnet-4.5") == "claude-sonnet-4-5-20250929"


def test_normalize_model_name_unknown_passes_through(proxy_module):
    assert proxy_module.normalize_model_name("gpt-5.4") == "gpt-5.4"


def test_normalize_model_name_already_canonical(proxy_module):
    assert proxy_module.normalize_model_name("claude-sonnet-4-6") == "claude-sonnet-4-6"


def test_normalize_model_name_arbitrary_string(proxy_module):
    assert proxy_module.normalize_model_name("some-random-model") == "some-random-model"


def test_chat_allowed_extra_includes_thinking(proxy_module):
    assert "thinking" in proxy_module._CHAT_ALLOWED_EXTRA


def test_default_thinking_disabled_when_caller_omits(proxy_module):
    """When the caller sends no thinking param, the proxy should inject disabled."""
    req = proxy_module.ChatRequest(
        model="ollama:llama3.2",
        messages=[{"role": "user", "content": "hello"}],
    )
    extra = {k: v for k, v in (req.model_extra or {}).items() if v is not None and k in proxy_module._CHAT_ALLOWED_EXTRA}
    # Caller didn't send thinking — proxy should inject it
    _caller_thinking = extra.get("thinking")
    if _caller_thinking is None:
        extra.setdefault("thinking", {"type": "disabled"})
    assert extra["thinking"] == {"type": "disabled"}


def test_caller_thinking_enabled_preserved(proxy_module):
    """When the caller explicitly enables thinking, the proxy must not override."""
    req = proxy_module.ChatRequest(
        model="ollama:llama3.2",
        messages=[{"role": "user", "content": "hello"}],
        thinking={"type": "enabled", "budget_tokens": 10000},
    )
    extra = {k: v for k, v in (req.model_extra or {}).items() if v is not None and k in proxy_module._CHAT_ALLOWED_EXTRA}
    _caller_thinking = extra.get("thinking")
    if _caller_thinking is None:
        extra.setdefault("thinking", {"type": "disabled"})
    assert extra["thinking"] == {"type": "enabled", "budget_tokens": 10000}


# ---------------------------------------------------------------------------
# _make_streaming_gen routing dispatch
# ---------------------------------------------------------------------------

_SYSTEM = "You are helpful."
_MESSAGES = [{"role": "user", "content": "hi"}]
_MAX_TOKENS = 100


def test_make_streaming_gen_routes_ollama(proxy_module):
    sentinel = MagicMock()
    with patch.object(proxy_module, "call_ollama_streaming", return_value=sentinel) as mock_fn:
        result = proxy_module._make_streaming_gen(_SYSTEM, _MESSAGES, "ollama:llama3.2", _MAX_TOKENS)
    assert result is sentinel
    mock_fn.assert_called_once_with(None, _MESSAGES, "ollama:llama3.2", _MAX_TOKENS)


def test_make_streaming_gen_routes_openrouter(proxy_module):
    sentinel = MagicMock()
    with patch.object(proxy_module, "call_openrouter_streaming", return_value=sentinel) as mock_fn:
        result = proxy_module._make_streaming_gen(_SYSTEM, _MESSAGES, "openai/gpt-4o", _MAX_TOKENS)
    assert result is sentinel
    mock_fn.assert_called_once_with(None, _MESSAGES, "openai/gpt-4o", _MAX_TOKENS)


def test_make_streaming_gen_routes_codex(proxy_module):
    sentinel = MagicMock()
    with patch.object(proxy_module, "call_codex_streaming", return_value=sentinel) as mock_fn:
        result = proxy_module._make_streaming_gen(_SYSTEM, _MESSAGES, "codex-mini", _MAX_TOKENS)
    assert result is sentinel
    mock_fn.assert_called_once_with(_SYSTEM, _MESSAGES, "codex-mini", _MAX_TOKENS)


def test_make_streaming_gen_routes_antigravity(proxy_module):
    sentinel = MagicMock()
    with patch.object(proxy_module, "call_antigravity_streaming", return_value=sentinel) as mock_fn:
        result = proxy_module._make_streaming_gen(_SYSTEM, _MESSAGES, "antigravity-gemini-2.5-pro", _MAX_TOKENS)
    assert result is sentinel
    mock_fn.assert_called_once_with(_SYSTEM, _MESSAGES, "antigravity-gemini-2.5-pro", _MAX_TOKENS)


def test_make_streaming_gen_routes_gemini_cli(proxy_module):
    sentinel = MagicMock()
    with patch.object(proxy_module, "call_gemini_streaming", return_value=sentinel) as mock_fn:
        result = proxy_module._make_streaming_gen(_SYSTEM, _MESSAGES, "gcli-gemini-2.5-pro", _MAX_TOKENS)
    assert result is sentinel
    mock_fn.assert_called_once_with(_SYSTEM, _MESSAGES, "gcli-gemini-2.5-pro", _MAX_TOKENS)


def test_make_streaming_gen_routes_zai(proxy_module):
    sentinel = MagicMock()
    with patch.object(proxy_module, "call_zai_streaming", return_value=sentinel) as mock_fn:
        result = proxy_module._make_streaming_gen(_SYSTEM, _MESSAGES, "zai:grok-3", _MAX_TOKENS)
    assert result is sentinel
    mock_fn.assert_called_once_with(None, _MESSAGES, "zai:grok-3", _MAX_TOKENS)


def test_make_streaming_gen_uses_inline_oai_messages_for_fallback(proxy_module):
    sentinel = MagicMock()
    oai_messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "hi"},
    ]
    with patch.object(proxy_module, "call_ollama_streaming", return_value=sentinel) as mock_fn:
        result = proxy_module._make_streaming_gen(
            _SYSTEM,
            _MESSAGES,
            "ollama:llama3.2",
            _MAX_TOKENS,
            oai_messages=oai_messages,
        )
    assert result is sentinel
    mock_fn.assert_called_once_with(None, oai_messages, "ollama:llama3.2", _MAX_TOKENS)


def test_make_streaming_gen_preserves_extra_params_for_oai_fallback(proxy_module):
    sentinel = MagicMock()
    extra_params = {"thinking": {"type": "disabled"}}
    with patch.object(proxy_module, "call_ollama_streaming", return_value=sentinel) as mock_fn:
        result = proxy_module._make_streaming_gen(
            _SYSTEM,
            _MESSAGES,
            "ollama:llama3.2",
            _MAX_TOKENS,
            extra_params=extra_params,
        )
    assert result is sentinel
    mock_fn.assert_called_once_with(
        None,
        _MESSAGES,
        "ollama:llama3.2",
        _MAX_TOKENS,
        thinking={"type": "disabled"},
    )


def test_make_streaming_gen_routes_claude_default(proxy_module):
    sentinel = MagicMock()
    with patch.object(proxy_module, "call_api_streaming", return_value=sentinel) as mock_fn:
        result = proxy_module._make_streaming_gen(_SYSTEM, _MESSAGES, "claude-sonnet-4-6", _MAX_TOKENS)
    assert result is sentinel
    mock_fn.assert_called_once_with(_SYSTEM, _MESSAGES, "claude-sonnet-4-6", _MAX_TOKENS)


def test_make_streaming_gen_normalizes_alias_before_routing(proxy_module):
    """Model aliases are resolved before dispatch, e.g. 'claude-sonnet-4.6' -> 'claude-sonnet-4-6'."""
    sentinel = MagicMock()
    with patch.object(proxy_module, "call_api_streaming", return_value=sentinel) as mock_fn:
        result = proxy_module._make_streaming_gen(_SYSTEM, _MESSAGES, "claude-sonnet-4.6", _MAX_TOKENS)
    assert result is sentinel
    # The normalized name should be passed to the call function
    mock_fn.assert_called_once_with(_SYSTEM, _MESSAGES, "claude-sonnet-4-6", _MAX_TOKENS)


def test_make_streaming_gen_ollama_with_slash_not_openrouter(proxy_module):
    """ollama:ns/model must route to Ollama, not OpenRouter (even though it contains '/')."""
    sentinel = MagicMock()
    with patch.object(proxy_module, "call_ollama_streaming", return_value=sentinel) as mock_ollama, \
         patch.object(proxy_module, "call_openrouter_streaming") as mock_openrouter:
        result = proxy_module._make_streaming_gen(_SYSTEM, _MESSAGES, "ollama:namespace/model", _MAX_TOKENS)
    assert result is sentinel
    mock_ollama.assert_called_once()
    mock_openrouter.assert_not_called()


# ---------------------------------------------------------------------------
# build_oai_messages
# ---------------------------------------------------------------------------

def test_build_oai_messages_with_system_prompt(proxy_module):
    """System prompt should be prepended as a system message."""
    msgs = [{"role": "user", "content": "hello"}]
    result = proxy_module.build_oai_messages("You are helpful.", msgs)
    assert result[0] == {"role": "system", "content": "You are helpful."}
    assert result[1] == {"role": "user", "content": "hello"}
    assert len(result) == 2


def test_build_oai_messages_without_system_prompt(proxy_module):
    """When system_prompt is None, no system message should be included."""
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    result = proxy_module.build_oai_messages(None, msgs)
    assert len(result) == 2
    assert result[0] == {"role": "user", "content": "hi"}
    assert result[1] == {"role": "assistant", "content": "hello"}


def test_build_oai_messages_empty_messages_with_system_prompt(proxy_module):
    """Empty messages list with a system prompt yields only the system message."""
    result = proxy_module.build_oai_messages("Be concise.", [])
    assert result == [{"role": "system", "content": "Be concise."}]


def test_build_oai_messages_multiple_messages(proxy_module):
    """Multiple messages are preserved in order after the system prompt."""
    msgs = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
        {"role": "user", "content": "third"},
    ]
    result = proxy_module.build_oai_messages("system", msgs)
    assert len(result) == 4
    assert result[0] == {"role": "system", "content": "system"}
    assert result[1] == {"role": "user", "content": "first"}
    assert result[2] == {"role": "assistant", "content": "second"}
    assert result[3] == {"role": "user", "content": "third"}


def test_sanitize_claude_template_strips_control_fields(proxy_module):
    """Claude auth templates should drop tools/system/control-plane baggage."""
    parsed = {
        "model": "claude-sonnet-4-5-20250929",
        "system": [{"type": "text", "text": "hidden system"}],
        "tools": [{"name": "search"}],
        "thinking": {"type": "enabled"},
        "context_management": {"foo": "bar"},
        "tool_choice": {"type": "auto"},
        "messages": [{"role": "user", "content": "hello"}],
    }

    sanitized = proxy_module._sanitize_claude_template(parsed)

    assert sanitized["model"] == "claude-sonnet-4-5-20250929"
    assert sanitized["messages"] == [{"role": "user", "content": "hello"}]
    assert "system" not in sanitized
    assert "tools" not in sanitized
    assert "thinking" not in sanitized
    assert "context_management" not in sanitized
    assert "tool_choice" not in sanitized


def test_build_api_body_strips_template_baggage(proxy_module):
    """Claude request bodies should not retain captured template baggage."""
    saved_template = proxy_module.auth.body_template
    saved_billing = proxy_module._cached_billing_block
    saved_auth_blocks = list(proxy_module._cached_auth_blocks)
    try:
        proxy_module.auth.body_template = {
            "model": "claude-sonnet-4-5-20250929",
            "system": [{"type": "text", "text": "template system"}],
            "tools": [{"name": "search"}],
            "thinking": {"type": "enabled"},
            "context_management": {"foo": "bar"},
            "tool_choice": {"type": "auto"},
            "messages": [{"role": "user", "content": [{"type": "text", "text": "template"}]}],
        }
        proxy_module._cached_billing_block = {"type": "text", "text": "billing"}
        proxy_module._cached_auth_blocks = [{"type": "text", "text": "<system-reminder>"}]

        body = proxy_module._build_api_body("User prompt", [{"role": "user", "content": "hello"}], "claude-sonnet-4-5-20250929")

        assert body["system"] == [
            {"type": "text", "text": "billing"},
            {"type": "text", "text": "User prompt"},
        ]
        assert body["messages"][0]["content"][0] == {"type": "text", "text": "<system-reminder>"}
        assert body["messages"][0]["content"][1] == {"type": "text", "text": "hello"}
        assert "tools" not in body
        assert "thinking" not in body
        assert "context_management" not in body
        assert "tool_choice" not in body
    finally:
        proxy_module.auth.body_template = saved_template
        proxy_module._cached_billing_block = saved_billing
        proxy_module._cached_auth_blocks = saved_auth_blocks



# ---------------------------------------------------------------------------
# make_sse_error_chunk
# ---------------------------------------------------------------------------

def test_make_sse_error_chunk_valid_sse_format(proxy_module):
    """Result must start with 'data: ' and end with double newline."""
    result = proxy_module.make_sse_error_chunk("test-model", "something went wrong")
    assert result.startswith("data: ")
    assert result.endswith("\n\n")


def test_make_sse_error_chunk_contains_error_message(proxy_module):
    """Parsed JSON payload must contain the error message in delta content."""
    import json
    result = proxy_module.make_sse_error_chunk("test-model", "something went wrong")
    payload = json.loads(result[len("data: "):].strip())
    assert payload["choices"][0]["delta"]["content"] == "something went wrong"


def test_make_sse_error_chunk_correct_model(proxy_module):
    """Parsed JSON payload must contain the correct model name."""
    import json
    result = proxy_module.make_sse_error_chunk("my-model", "err")
    payload = json.loads(result[len("data: "):].strip())
    assert payload["model"] == "my-model"


def test_make_sse_error_chunk_object_type(proxy_module):
    """Object type must be 'chat.completion.chunk'."""
    import json
    result = proxy_module.make_sse_error_chunk("m", "e")
    payload = json.loads(result[len("data: "):].strip())
    assert payload["object"] == "chat.completion.chunk"


# ---------------------------------------------------------------------------
# yield_sse_error
# ---------------------------------------------------------------------------

def test_yield_sse_error_returns_tuple(proxy_module):
    """yield_sse_error returns a two-element tuple (error_chunk, done_marker)."""
    error_chunk, done_marker = proxy_module.yield_sse_error("model", "err")
    assert isinstance(error_chunk, str)
    assert isinstance(done_marker, str)


def test_yield_sse_error_done_marker(proxy_module):
    """The done_marker is always the standard SSE DONE sentinel."""
    _, done_marker = proxy_module.yield_sse_error("model", "err")
    assert done_marker == "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# passthrough_sse
# ---------------------------------------------------------------------------

async def _async_bytes_iter(chunks):
    """Helper: async generator yielding byte chunks."""
    for chunk in chunks:
        yield chunk


@pytest.mark.asyncio
async def test_passthrough_sse_passes_through_events(proxy_module):
    """Complete SSE events are yielded as-is."""
    import time

    mock_content = MagicMock()
    mock_content.iter_any.return_value = _async_bytes_iter([
        b"data: {\"hello\": 1}\n\n",
        b"data: {\"hello\": 2}\n\n",
    ])
    mock_resp = MagicMock()
    mock_resp.content = mock_content

    events = []
    async for event in proxy_module.passthrough_sse(mock_resp, "req-1", "test", time.time()):
        events.append(event)

    assert events[0] == "data: {\"hello\": 1}\n\n"
    assert events[1] == "data: {\"hello\": 2}\n\n"


@pytest.mark.asyncio
async def test_passthrough_sse_handles_partial_chunks(proxy_module):
    """Chunks split across iter_any calls are buffered and reassembled."""
    import time

    mock_content = MagicMock()
    # The event "data: {\"x\": 1}\n\n" is split across two raw chunks
    mock_content.iter_any.return_value = _async_bytes_iter([
        b"data: {\"x\":",
        b" 1}\n\n",
    ])
    mock_resp = MagicMock()
    mock_resp.content = mock_content

    events = []
    async for event in proxy_module.passthrough_sse(mock_resp, "req-2", "test", time.time()):
        events.append(event)

    assert len(events) == 1
    assert events[0] == "data: {\"x\": 1}\n\n"


@pytest.mark.asyncio
async def test_passthrough_sse_flushes_remaining_buffer(proxy_module):
    """Any remaining data in the buffer after iteration is flushed."""
    import time

    mock_content = MagicMock()
    # Trailing data without a final \n\n should still be yielded
    mock_content.iter_any.return_value = _async_bytes_iter([
        b"data: {\"a\": 1}\n\n",
        b"data: {\"b\": 2}",
    ])
    mock_resp = MagicMock()
    mock_resp.content = mock_content

    events = []
    async for event in proxy_module.passthrough_sse(mock_resp, "req-3", "test", time.time()):
        events.append(event)

    assert events[0] == "data: {\"a\": 1}\n\n"
    # The trailing partial event should be flushed
    assert events[1] == "data: {\"b\": 2}\n\n"


@pytest.mark.asyncio
async def test_passthrough_sse_drops_reasoning_only_chunks(proxy_module):
    """Reasoning-only deltas must be suppressed, never re-emitted as content."""
    import time

    mock_content = MagicMock()
    mock_content.iter_any.return_value = _async_bytes_iter([
        b'data: {"choices":[{"delta":{"reasoning":"internal","reasoning_details":"trace"}}]}\n\n',
    ])
    mock_resp = MagicMock()
    mock_resp.content = mock_content

    events = []
    async for event in proxy_module.passthrough_sse(mock_resp, "req-4", "test", time.time()):
        events.append(event)

    assert events == []


@pytest.mark.asyncio
async def test_passthrough_sse_strips_reasoning_fields_from_mixed_chunks(proxy_module):
    """Mixed content + reasoning chunks should keep content and drop reasoning fields."""
    import json
    import time

    mock_content = MagicMock()
    mock_content.iter_any.return_value = _async_bytes_iter([
        b'data: {"choices":[{"delta":{"content":"hello","reasoning":"internal","reasoning_content":"trace"}}]}\n\n',
    ])
    mock_resp = MagicMock()
    mock_resp.content = mock_content

    events = []
    async for event in proxy_module.passthrough_sse(mock_resp, "req-5", "test", time.time()):
        events.append(event)

    assert len(events) == 1
    payload = json.loads(events[0][len("data: "):].strip())
    delta = payload["choices"][0]["delta"]
    assert delta == {"content": "hello"}


@pytest.mark.asyncio
async def test_passthrough_sse_with_rewrite_reasoning_only(proxy_module):
    """Reasoning-only stream should be rewritten via fast LLM."""
    import json
    import time

    # Import the real function directly from streaming module
    from proxy_internal.streaming import passthrough_sse_with_rewrite

    mock_content = MagicMock()
    mock_content.iter_any.return_value = _async_bytes_iter([
        b'data: {"choices":[{"delta":{"reasoning":"Let me think about this..."}}]}\n\n',
        b'data: {"choices":[{"delta":{"reasoning":"The player asked about dragons."}}]}\n\n',
    ])
    mock_resp = MagicMock()
    mock_resp.content = mock_content

    # Mock the rewrite function
    with patch("proxy_internal.streaming._rewrite_reasoning_to_dialogue", new_callable=AsyncMock) as mock_rewrite:
        mock_rewrite.return_value = "Ah, dragons! Aye, I've heard tales of those beasts."

        events = []
        async for event in passthrough_sse_with_rewrite(
            mock_resp, "req-rw1", "TestProvider", time.time(),
            system_prompt="You are a Skyrim innkeeper.", model="test:model",
        ):
            events.append(event)

        # Should have called the rewrite with accumulated reasoning
        mock_rewrite.assert_called_once()
        call_args = mock_rewrite.call_args
        assert "Let me think about this..." in call_args[0][0]
        assert "The player asked about dragons." in call_args[0][0]
        assert call_args[0][1] == "You are a Skyrim innkeeper."

        # Should emit one content chunk with rewritten text
        assert len(events) == 1
        payload = json.loads(events[0][len("data: "):].strip())
        assert "Ah, dragons" in payload["choices"][0]["delta"]["content"]


@pytest.mark.asyncio
async def test_passthrough_sse_with_rewrite_reasoning_only_preserves_done(proxy_module):
    """Reasoning-only rewrite must still preserve the upstream stream terminator."""
    import time

    from proxy_internal.streaming import passthrough_sse_with_rewrite

    mock_content = MagicMock()
    mock_content.iter_any.return_value = _async_bytes_iter([
        b'data: {"choices":[{"delta":{"reasoning":"thinking..."}}]}\n\n',
        b"data: [DONE]\n\n",
    ])
    mock_resp = MagicMock()
    mock_resp.content = mock_content

    with patch("proxy_internal.streaming._rewrite_reasoning_to_dialogue", new_callable=AsyncMock) as mock_rewrite:
        mock_rewrite.return_value = "Stay close."

        events = []
        async for event in passthrough_sse_with_rewrite(
            mock_resp, "req-rw1b", "TestProvider", time.time(),
            system_prompt="You are an NPC.", model="test:model",
        ):
            events.append(event)

        assert len(events) == 2
        assert "Stay close." in events[0]
        assert events[1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_passthrough_sse_with_rewrite_normalizes_structured_reasoning(proxy_module):
    """Structured reasoning_details payloads must be normalized before rewrite."""
    import time

    from proxy_internal.streaming import passthrough_sse_with_rewrite

    mock_content = MagicMock()
    mock_content.iter_any.return_value = _async_bytes_iter([
        (
            b'data: {"choices":[{"delta":{"reasoning_details":'
            b'[{"type":"text","text":"First thought."},{"type":"text","text":"Second thought."}]}}]}\n\n'
        ),
        b"data: [DONE]\n\n",
    ])
    mock_resp = MagicMock()
    mock_resp.content = mock_content

    with patch("proxy_internal.streaming._rewrite_reasoning_to_dialogue", new_callable=AsyncMock) as mock_rewrite:
        mock_rewrite.return_value = "Move."

        events = []
        async for event in passthrough_sse_with_rewrite(
            mock_resp, "req-rw1c", "TestProvider", time.time(),
            system_prompt="You are an NPC.", model="test:model",
        ):
            events.append(event)

        mock_rewrite.assert_called_once()
        reasoning_text = mock_rewrite.call_args.args[0]
        assert "First thought." in reasoning_text
        assert "Second thought." in reasoning_text
        assert events[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_passthrough_sse_with_rewrite_mixed_content(proxy_module):
    """Hidden reasoning must stay hidden when real content later arrives."""
    import json
    import time

    from proxy_internal.streaming import passthrough_sse_with_rewrite

    mock_content = MagicMock()
    mock_content.iter_any.return_value = _async_bytes_iter([
        b'data: {"choices":[{"delta":{"reasoning":"thinking..."}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"Hello, traveler!"}}]}\n\n',
    ])
    mock_resp = MagicMock()
    mock_resp.content = mock_content

    with patch("proxy_internal.streaming._rewrite_reasoning_to_dialogue", new_callable=AsyncMock) as mock_rewrite:
        events = []
        async for event in passthrough_sse_with_rewrite(
            mock_resp, "req-rw2", "TestProvider", time.time(),
            system_prompt="You are an NPC.", model="test:model",
        ):
            events.append(event)

        # Rewrite should NOT be called — real content was present
        mock_rewrite.assert_not_called()

        assert len(events) == 1
        payload = json.loads(events[0][len("data: "):].strip())
        assert payload["choices"][0]["delta"]["content"] == "Hello, traveler!"


@pytest.mark.asyncio
async def test_passthrough_sse_with_rewrite_real_content_only(proxy_module):
    """Stream with only real content should pass through without rewrite."""
    import time

    from proxy_internal.streaming import passthrough_sse_with_rewrite

    mock_content = MagicMock()
    mock_content.iter_any.return_value = _async_bytes_iter([
        b'data: {"choices":[{"delta":{"content":"Greetings!"}}]}\n\n',
    ])
    mock_resp = MagicMock()
    mock_resp.content = mock_content

    with patch("proxy_internal.streaming._rewrite_reasoning_to_dialogue", new_callable=AsyncMock) as mock_rewrite:
        events = []
        async for event in passthrough_sse_with_rewrite(
            mock_resp, "req-rw3", "TestProvider", time.time(),
            model="test:model",
        ):
            events.append(event)

        mock_rewrite.assert_not_called()
        assert len(events) == 1


@pytest.mark.asyncio
async def test_passthrough_sse_with_rewrite_failure_fallback(proxy_module):
    """When rewrite fails, the proxy must fail closed rather than leak reasoning."""
    import time

    from proxy_internal.streaming import passthrough_sse_with_rewrite

    mock_content = MagicMock()
    mock_content.iter_any.return_value = _async_bytes_iter([
        b'data: {"choices":[{"delta":{"reasoning":"Let me think..."}}]}\n\n',
    ])
    mock_resp = MagicMock()
    mock_resp.content = mock_content

    with patch("proxy_internal.streaming._rewrite_reasoning_to_dialogue", new_callable=AsyncMock) as mock_rewrite:
        mock_rewrite.return_value = None  # Rewrite fails

        events = []
        async for event in passthrough_sse_with_rewrite(
            mock_resp, "req-rw4", "TestProvider", time.time(),
            system_prompt="You are an NPC.", model="test:model",
        ):
            events.append(event)

        assert events == []


@pytest.mark.asyncio
async def test_streaming_request_uses_original_system_prompt_for_rewrite_context_and_resets(proxy_module):
    """Rewrite context should use the original system prompt and be reset after streaming."""
    seen = []

    async def fake_stream(system_prompt, messages, model, max_tokens, **kwargs):
        seen.append(dict(proxy_module._rewrite_ctx.get()))
        yield "data: [DONE]\n\n"

    req = proxy_module.ChatRequest(
        model="ollama:llama3.2",
        stream=True,
        messages=[
            {"role": "system", "content": "You are a Skyrim innkeeper."},
            {"role": "user", "content": "Hello"},
        ],
    )

    with patch.object(proxy_module, "call_ollama_streaming", side_effect=fake_stream), \
         patch.object(proxy_module, "timeout_routing_enabled", False):
        response = await proxy_module._chat_completions_inner(req)
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk)

    assert seen == [{"system_prompt": "You are a Skyrim innkeeper.", "model": "ollama:llama3.2"}]
    assert proxy_module._rewrite_ctx.get() == {}
    assert chunks == ["data: [DONE]\n\n"]
