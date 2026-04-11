"""Tests for Anthropic /v1/messages tool_use / tool_result passthrough.

Covers:
  - Request translation (Anthropic → OpenAI) including tool defs, tool_choice,
    tool_use blocks, tool_result blocks, mixed text+tool_use, malformed input.
  - Non-streaming response translation (OpenAI → Anthropic) with/without
    tool_calls.
  - Streaming response translation including incremental tool_call arguments,
    mixed text-then-tool, multiple tools, and text-only regression.
  - End-to-end /v1/messages routing through TestClient for OAI-compat model,
    rejection for non-OAI model, and regression for text-only model.
"""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class _StubSession:
    """Placeholder session object so _resolve_oai_compatible doesn't allocate a
    real aiohttp ClientSession during tests (which would require an event loop)."""
    async def close(self):
        pass


# ----------------------------------------------------------------------------
# Request translation: _anthropic_to_oai_structured + helpers
# ----------------------------------------------------------------------------

class TestRequestTranslation:

    def test_tool_defs_translated_to_openai_shape(self, proxy_module):
        body = {
            "model": "ollama:llama3.2",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 200,
            "tools": [
                {
                    "name": "Bash",
                    "description": "Run a shell command",
                    "input_schema": {
                        "type": "object",
                        "properties": {"cmd": {"type": "string"}},
                        "required": ["cmd"],
                    },
                }
            ],
        }
        payload = proxy_module._anthropic_to_oai_structured(body)
        assert payload["model"] == "ollama:llama3.2"
        assert payload["max_tokens"] == 200
        assert len(payload["tools"]) == 1
        t = payload["tools"][0]
        assert t["type"] == "function"
        assert t["function"]["name"] == "Bash"
        assert t["function"]["description"] == "Run a shell command"
        assert t["function"]["parameters"]["properties"]["cmd"]["type"] == "string"
        assert "tool_choice" not in payload

    def test_tool_choice_auto(self, proxy_module):
        body = {
            "model": "ollama:llama3.2",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "Bash", "description": "x",
                       "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "auto"},
        }
        payload = proxy_module._anthropic_to_oai_structured(body)
        assert payload["tool_choice"] == "auto"

    def test_tool_choice_any_maps_to_required(self, proxy_module):
        body = {
            "model": "ollama:llama3.2",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "Bash", "description": "x",
                       "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "any"},
        }
        payload = proxy_module._anthropic_to_oai_structured(body)
        assert payload["tool_choice"] == "required"

    def test_tool_choice_specific_tool(self, proxy_module):
        body = {
            "model": "ollama:llama3.2",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "Bash", "description": "x",
                       "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "tool", "name": "Bash"},
        }
        payload = proxy_module._anthropic_to_oai_structured(body)
        assert payload["tool_choice"] == {"type": "function",
                                          "function": {"name": "Bash"}}

    def test_assistant_tool_use_block_becomes_tool_calls(self, proxy_module):
        body = {
            "model": "ollama:llama3.2",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "list files"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "toolu_abc123",
                         "name": "Bash", "input": {"cmd": "ls"}},
                    ],
                },
            ],
        }
        payload = proxy_module._anthropic_to_oai_structured(body)
        msgs = payload["messages"]
        assert msgs[0] == {"role": "user", "content": "list files"}
        asst = msgs[1]
        assert asst["role"] == "assistant"
        assert asst["content"] is None
        assert len(asst["tool_calls"]) == 1
        tc = asst["tool_calls"][0]
        assert tc["id"] == "toolu_abc123"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "Bash"
        assert json.loads(tc["function"]["arguments"]) == {"cmd": "ls"}

    def test_user_tool_result_becomes_separate_tool_message(self, proxy_module):
        body = {
            "model": "ollama:llama3.2",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "ls"},
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "toolu_1",
                                 "name": "Bash", "input": {"cmd": "ls"}}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_1",
                         "content": "file1\nfile2\n"},
                    ],
                },
            ],
        }
        payload = proxy_module._anthropic_to_oai_structured(body)
        msgs = payload["messages"]
        tool_msg = msgs[-1]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "toolu_1"
        assert "file1" in tool_msg["content"]

    def test_multiple_tool_use_and_result_in_one_turn(self, proxy_module):
        body = {
            "model": "ollama:llama3.2",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "start"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "t1", "name": "Bash",
                         "input": {"cmd": "a"}},
                        {"type": "tool_use", "id": "t2", "name": "Bash",
                         "input": {"cmd": "b"}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "t1",
                         "content": "out-a"},
                        {"type": "tool_result", "tool_use_id": "t2",
                         "content": "out-b"},
                    ],
                },
            ],
        }
        payload = proxy_module._anthropic_to_oai_structured(body)
        msgs = payload["messages"]
        asst = [m for m in msgs if m["role"] == "assistant"][0]
        assert len(asst["tool_calls"]) == 2
        tool_msgs = [m for m in msgs if m["role"] == "tool"]
        assert len(tool_msgs) == 2
        assert {t["tool_call_id"] for t in tool_msgs} == {"t1", "t2"}
        assert tool_msgs[0]["content"] == "out-a"
        assert tool_msgs[1]["content"] == "out-b"

    def test_mixed_text_and_tool_use_in_assistant_message(self, proxy_module):
        body = {
            "model": "ollama:llama3.2",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Let me check."},
                        {"type": "tool_use", "id": "t1", "name": "Bash",
                         "input": {"cmd": "pwd"}},
                    ],
                },
            ],
        }
        payload = proxy_module._anthropic_to_oai_structured(body)
        asst = [m for m in payload["messages"] if m["role"] == "assistant"][0]
        assert asst["content"] == "Let me check."
        assert len(asst["tool_calls"]) == 1
        assert asst["tool_calls"][0]["function"]["name"] == "Bash"

    def test_malformed_tool_def_raises_400(self, proxy_module):
        from fastapi import HTTPException
        body = {
            "model": "ollama:llama3.2",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"description": "missing name"}],
        }
        with pytest.raises(HTTPException) as exc:
            proxy_module._anthropic_to_oai_structured(body)
        assert exc.value.status_code == 400
        assert "name" in exc.value.detail.lower()

    def test_malformed_tool_choice_raises_400(self, proxy_module):
        from fastapi import HTTPException
        body = {
            "model": "ollama:llama3.2",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "Bash", "description": "x",
                       "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "tool"},  # missing "name"
        }
        with pytest.raises(HTTPException) as exc:
            proxy_module._anthropic_to_oai_structured(body)
        assert exc.value.status_code == 400

    def test_system_string_becomes_system_message(self, proxy_module):
        body = {
            "model": "ollama:llama3.2",
            "max_tokens": 100,
            "system": "You are a helper.",
            "messages": [{"role": "user", "content": "hi"}],
        }
        payload = proxy_module._anthropic_to_oai_structured(body)
        assert payload["messages"][0] == {"role": "system",
                                          "content": "You are a helper."}

    def test_system_block_list_flattened(self, proxy_module):
        body = {
            "model": "ollama:llama3.2",
            "max_tokens": 100,
            "system": [
                {"type": "text", "text": "Line one."},
                {"type": "text", "text": "Line two."},
            ],
            "messages": [{"role": "user", "content": "hi"}],
        }
        payload = proxy_module._anthropic_to_oai_structured(body)
        sys_msg = payload["messages"][0]
        assert sys_msg["role"] == "system"
        assert "Line one." in sys_msg["content"]
        assert "Line two." in sys_msg["content"]

    def test_stop_sequences_mapped(self, proxy_module):
        body = {
            "model": "ollama:llama3.2",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
            "stop_sequences": ["\n\n", "STOP"],
        }
        payload = proxy_module._anthropic_to_oai_structured(body)
        assert payload["stop"] == ["\n\n", "STOP"]

    def test_missing_model_raises_400(self, proxy_module):
        from fastapi import HTTPException
        body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 10}
        with pytest.raises(HTTPException) as exc:
            proxy_module._anthropic_to_oai_structured(body)
        assert exc.value.status_code == 400

    # ---- Fix 1: system message duplication / merging ----

    def _system_messages(self, payload: dict) -> list[dict]:
        return [m for m in payload["messages"] if m.get("role") == "system"]

    def test_top_level_and_in_array_system_merge_into_one(self, proxy_module):
        """Top-level ``system`` + in-array system message collapse into a
        single OAI system entry with both texts (top-level first)."""
        body = {
            "model": "ollama:llama3.2",
            "max_tokens": 50,
            "system": "You are helpful.",
            "messages": [
                {"role": "system", "content": "Also be brief."},
                {"role": "user", "content": "hi"},
            ],
        }
        payload = proxy_module._anthropic_to_oai_structured(body)
        sys_msgs = self._system_messages(payload)
        assert len(sys_msgs) == 1
        assert sys_msgs[0] is payload["messages"][0]
        # top-level first, in-array second, separated by blank line
        assert sys_msgs[0]["content"] == "You are helpful.\n\nAlso be brief."

    def test_two_in_array_system_messages_collapse_preserving_order(self, proxy_module):
        body = {
            "model": "ollama:llama3.2",
            "max_tokens": 50,
            "messages": [
                {"role": "system", "content": "first rule"},
                {"role": "system", "content": "second rule"},
                {"role": "user", "content": "hi"},
            ],
        }
        payload = proxy_module._anthropic_to_oai_structured(body)
        sys_msgs = self._system_messages(payload)
        assert len(sys_msgs) == 1
        assert sys_msgs[0]["content"] == "first rule\n\nsecond rule"
        # and it's at index 0, before the user message
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1]["role"] == "user"

    def test_only_top_level_system_single_message_regression(self, proxy_module):
        body = {
            "model": "ollama:llama3.2",
            "max_tokens": 50,
            "system": "just top level",
            "messages": [{"role": "user", "content": "hi"}],
        }
        payload = proxy_module._anthropic_to_oai_structured(body)
        sys_msgs = self._system_messages(payload)
        assert len(sys_msgs) == 1
        assert sys_msgs[0]["content"] == "just top level"

    def test_only_single_in_array_system_message(self, proxy_module):
        body = {
            "model": "ollama:llama3.2",
            "max_tokens": 50,
            "messages": [
                {"role": "system", "content": "only me"},
                {"role": "user", "content": "hi"},
            ],
        }
        payload = proxy_module._anthropic_to_oai_structured(body)
        sys_msgs = self._system_messages(payload)
        assert len(sys_msgs) == 1
        assert sys_msgs[0]["content"] == "only me"


# ----------------------------------------------------------------------------
# Non-streaming response translation
# ----------------------------------------------------------------------------

class TestNonStreamingResponseTranslation:

    def test_tool_calls_emit_tool_use_blocks(self, proxy_module):
        data = {
            "id": "chatcmpl-xyz",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "I'll run it.",
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {
                                    "name": "Bash",
                                    "arguments": '{"cmd": "ls"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 5},
        }
        msg = proxy_module._openai_completion_to_anthropic_message(data, "ollama:llama3.2")
        assert msg["type"] == "message"
        assert msg["role"] == "assistant"
        assert msg["model"] == "ollama:llama3.2"
        assert msg["stop_reason"] == "tool_use"
        types = [b["type"] for b in msg["content"]]
        assert "text" in types
        assert "tool_use" in types
        tu = [b for b in msg["content"] if b["type"] == "tool_use"][0]
        assert tu["id"] == "call_abc"  # upstream id preserved verbatim
        assert tu["name"] == "Bash"
        assert tu["input"] == {"cmd": "ls"}
        assert msg["usage"]["input_tokens"] == 12
        assert msg["usage"]["output_tokens"] == 5

    def test_tool_calls_without_text_emits_only_tool_use(self, proxy_module):
        """When upstream returns content=None with tool_calls, the text block
        is suppressed entirely so clients see clean structured output."""
        data = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "call_1", "type": "function",
                            "function": {"name": "Foo", "arguments": "{}"},
                        }],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
        msg = proxy_module._openai_completion_to_anthropic_message(data, "ollama:x")
        types = [b["type"] for b in msg["content"]]
        assert types == ["tool_use"]
        assert msg["stop_reason"] == "tool_use"

    def test_malformed_tool_arguments_json_still_parses(self, proxy_module):
        """If the upstream arguments string isn't valid JSON, we keep the raw
        string in a _raw_arguments field rather than exploding."""
        data = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "call_bad", "type": "function",
                            "function": {"name": "Foo",
                                         "arguments": "not-json"},
                        }],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
        msg = proxy_module._openai_completion_to_anthropic_message(data, "ollama:x")
        tu = [b for b in msg["content"] if b["type"] == "tool_use"][0]
        assert "_raw_arguments" in tu["input"]
        assert tu["input"]["_raw_arguments"] == "not-json"

    def test_response_without_tool_calls_regression(self, proxy_module):
        """Plain text responses still emit a single text block, preserving
        the legacy /v1/messages contract."""
        data = {
            "id": "chatcmpl-1",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        }
        msg = proxy_module._openai_completion_to_anthropic_message(data, "ollama:llama3.2")
        assert msg["stop_reason"] == "end_turn"
        assert msg["content"] == [{"type": "text", "text": "Hello!"}]


# ----------------------------------------------------------------------------
# Streaming response translation
# ----------------------------------------------------------------------------

async def _collect(aiter):
    out = []
    async for chunk in aiter:
        out.append(chunk)
    return out


def _parse_sse(events: list) -> list[tuple[str, dict]]:
    parsed = []
    for raw in events:
        # _format_anthropic_sse now yields bytes directly (perf fix) so we
        # decode here before the line-split.  Callers may also hand us plain
        # strings (single-shot wrapper path) — accept both.
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        event_type = None
        data = None
        for line in raw.split("\n"):
            if line.startswith("event: "):
                event_type = line[7:].strip()
            elif line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                except json.JSONDecodeError:
                    data = None
        if event_type is not None:
            parsed.append((event_type, data))
    return parsed


async def _make_iter(chunks):
    for c in chunks:
        yield c


class TestStreamingResponseTranslation:

    @pytest.mark.asyncio
    async def test_tool_call_stream_produces_tool_use_blocks(self, proxy_module):
        # Dict chunks from _stream_oai_compatible simulated directly.
        chunks = [
            {"choices": [{"index": 0, "delta": {"role": "assistant"},
                          "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {
                "tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                "function": {"name": "Bash", "arguments": ""}}]
            }, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {
                "tool_calls": [{"index": 0, "function": {"arguments": '{"cmd":'}}]
            }, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {
                "tool_calls": [{"index": 0, "function": {"arguments": '"ls"}'}}]
            }, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        ]
        events = await _collect(
            proxy_module._anthropic_stream_from_openai(_make_iter(chunks), "ollama:x", 0)
        )
        parsed = _parse_sse(events)
        event_types = [p[0] for p in parsed]
        assert event_types[0] == "message_start"
        assert "content_block_start" in event_types
        assert "content_block_delta" in event_types
        assert "content_block_stop" in event_types
        assert "message_delta" in event_types
        assert event_types[-1] == "message_stop"

        # One of the content_block_start events should be a tool_use
        tool_starts = [
            p for p in parsed
            if p[0] == "content_block_start"
            and p[1]["content_block"]["type"] == "tool_use"
        ]
        assert len(tool_starts) == 1
        assert tool_starts[0][1]["content_block"]["name"] == "Bash"
        assert tool_starts[0][1]["content_block"]["id"] == "call_1"

        # Accumulated input_json_delta should reassemble the tool arguments
        deltas = [p[1]["delta"]["partial_json"] for p in parsed
                  if p[0] == "content_block_delta"
                  and p[1]["delta"]["type"] == "input_json_delta"]
        assert "".join(deltas) == '{"cmd":"ls"}'

        # stop_reason on message_delta reflects tool_use
        md = [p for p in parsed if p[0] == "message_delta"][0]
        assert md[1]["delta"]["stop_reason"] == "tool_use"

    @pytest.mark.asyncio
    async def test_mixed_text_then_tool_call(self, proxy_module):
        chunks = [
            {"choices": [{"index": 0, "delta": {"content": "Let me "},
                          "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {"content": "check."},
                          "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {
                "tool_calls": [{"index": 0, "id": "call_x", "type": "function",
                                "function": {"name": "Bash", "arguments": "{}"}}]
            }, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        ]
        events = await _collect(
            proxy_module._anthropic_stream_from_openai(_make_iter(chunks), "ollama:x", 0)
        )
        parsed = _parse_sse(events)
        starts = [p for p in parsed if p[0] == "content_block_start"]
        # First: initial text block (always pre-opened). Second: tool_use.
        assert len(starts) == 2
        assert starts[0][1]["content_block"]["type"] == "text"
        assert starts[1][1]["content_block"]["type"] == "tool_use"
        # Ensure the text block closes before the tool_use opens.
        start_idxs = [i for i, p in enumerate(parsed) if p[0] == "content_block_start"]
        stops = [i for i, p in enumerate(parsed) if p[0] == "content_block_stop"]
        assert stops[0] < start_idxs[1]
        # Text delta should be present before the tool_use
        text_deltas = [p[1]["delta"]["text"] for p in parsed
                       if p[0] == "content_block_delta"
                       and p[1]["delta"]["type"] == "text_delta"]
        assert "".join(text_deltas).startswith("Let me ")
        md = [p for p in parsed if p[0] == "message_delta"][0]
        assert md[1]["delta"]["stop_reason"] == "tool_use"

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_in_one_stream(self, proxy_module):
        chunks = [
            {"choices": [{"index": 0, "delta": {
                "tool_calls": [{"index": 0, "id": "c1", "type": "function",
                                "function": {"name": "Bash", "arguments": "{}"}}]
            }, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {
                "tool_calls": [{"index": 1, "id": "c2", "type": "function",
                                "function": {"name": "Read", "arguments": ""}}]
            }, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {
                "tool_calls": [{"index": 1,
                                "function": {"arguments": '{"path":"x"}'}}]
            }, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        ]
        events = await _collect(
            proxy_module._anthropic_stream_from_openai(_make_iter(chunks), "ollama:x", 0)
        )
        parsed = _parse_sse(events)
        tool_starts = [p for p in parsed if p[0] == "content_block_start"
                       and p[1]["content_block"]["type"] == "tool_use"]
        assert len(tool_starts) == 2
        names = {s[1]["content_block"]["name"] for s in tool_starts}
        assert names == {"Bash", "Read"}
        stops = [p for p in parsed if p[0] == "content_block_stop"]
        # 1 text block + 2 tool_use blocks = 3 stops
        assert len(stops) == 3
        md = [p for p in parsed if p[0] == "message_delta"][0]
        assert md[1]["delta"]["stop_reason"] == "tool_use"

    @pytest.mark.asyncio
    async def test_tool_call_args_before_name_are_buffered_and_flushed(self, proxy_module):
        """Regression: some upstreams emit a tool_call delta carrying only
        ``function.arguments`` BEFORE the delta that reveals
        ``function.name``.  The pre-name args must be buffered and flushed
        as an input_json_delta immediately after content_block_start so no
        bytes are silently dropped."""
        chunks = [
            # First delta: NO name, just a fragment of arguments.
            {"choices": [{"index": 0, "delta": {
                "tool_calls": [{"index": 0, "id": "call_pre",
                                "function": {"arguments": '{"a":'}}]
            }, "finish_reason": None}]},
            # Second delta: name appears.  At this moment the content block
            # should start AND the buffered pre-name args should be flushed.
            {"choices": [{"index": 0, "delta": {
                "tool_calls": [{"index": 0,
                                "function": {"name": "Bash"}}]
            }, "finish_reason": None}]},
            # Third delta: normal streaming args after the block is open.
            {"choices": [{"index": 0, "delta": {
                "tool_calls": [{"index": 0,
                                "function": {"arguments": '1}'}}]
            }, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {},
                          "finish_reason": "tool_calls"}]},
        ]
        events = await _collect(
            proxy_module._anthropic_stream_from_openai(_make_iter(chunks), "ollama:x", 0)
        )
        parsed = _parse_sse(events)

        # Exactly one tool_use content block.
        tool_starts = [
            p for p in parsed
            if p[0] == "content_block_start"
            and p[1]["content_block"]["type"] == "tool_use"
        ]
        assert len(tool_starts) == 1
        tool_start_idx = next(
            i for i, p in enumerate(parsed)
            if p[0] == "content_block_start"
            and p[1]["content_block"]["type"] == "tool_use"
        )

        # The content_block_delta that immediately follows the tool_use
        # content_block_start must carry the buffered pre-name fragment.
        next_delta = parsed[tool_start_idx + 1]
        assert next_delta[0] == "content_block_delta"
        assert next_delta[1]["delta"]["type"] == "input_json_delta"
        assert next_delta[1]["delta"]["partial_json"] == '{"a":'

        # Full reassembled arguments preserved end-to-end.
        deltas = [p[1]["delta"]["partial_json"] for p in parsed
                  if p[0] == "content_block_delta"
                  and p[1]["delta"]["type"] == "input_json_delta"]
        assert "".join(deltas) == '{"a":1}'

    @pytest.mark.asyncio
    async def test_text_only_dict_stream_regression(self, proxy_module):
        """Dict-chunk path must still produce a clean text-only response
        when no tool_calls are emitted."""
        chunks = [
            {"choices": [{"index": 0, "delta": {"content": "Hi"},
                          "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {"content": " there"},
                          "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ]
        events = await _collect(
            proxy_module._anthropic_stream_from_openai(_make_iter(chunks), "ollama:x", 0)
        )
        parsed = _parse_sse(events)
        types = [p[0] for p in parsed]
        assert types[0] == "message_start"
        assert "content_block_start" in types
        assert "content_block_delta" in types
        assert "content_block_stop" in types
        md = [p for p in parsed if p[0] == "message_delta"][0]
        assert md[1]["delta"]["stop_reason"] == "end_turn"
        assert types[-1] == "message_stop"
        # No tool_use blocks emitted
        tool_starts = [p for p in parsed if p[0] == "content_block_start"
                       and p[1]["content_block"]["type"] == "tool_use"]
        assert tool_starts == []


# ----------------------------------------------------------------------------
# Resolver
# ----------------------------------------------------------------------------

class TestResolver:

    @pytest.mark.asyncio
    async def test_returns_none_for_claude(self, proxy_module):
        assert await proxy_module._resolve_oai_compatible("claude-sonnet-4-6") is None

    @pytest.mark.asyncio
    async def test_returns_none_for_codex(self, proxy_module):
        assert await proxy_module._resolve_oai_compatible("gpt-5.3-codex") is None

    @pytest.mark.asyncio
    async def test_resolves_ollama(self, proxy_module):
        with patch.object(proxy_module, "ollama_session", _StubSession()):
            resolved = await proxy_module._resolve_oai_compatible("ollama:llama3.2")
        assert resolved is not None
        assert resolved["api_model"] == "llama3.2"
        assert "chat/completions" in resolved["endpoint_url"]
        assert resolved["provider_name"] == "Ollama"

    @pytest.mark.asyncio
    async def test_resolves_openrouter(self, proxy_module):
        with patch.object(proxy_module, "third_party_session", _StubSession()), \
             patch.object(proxy_module, "openrouter_api_key", "sk-test"):
            resolved = await proxy_module._resolve_oai_compatible(
                "openai/gpt-4o-mini"
            )
        assert resolved is not None
        assert resolved["provider_name"] == "OpenRouter"
        assert "openrouter.ai" in resolved["endpoint_url"]


# ----------------------------------------------------------------------------
# End-to-end TestClient integration
# ----------------------------------------------------------------------------

class TestMessagesEndpointStructured:

    def test_tools_on_oai_model_returns_tool_use(self, test_client, proxy_module):
        fake_response = {
            "id": "chatcmpl-1",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1", "type": "function",
                        "function": {"name": "Bash",
                                     "arguments": '{"cmd":"ls"}'},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4},
        }
        with patch.object(proxy_module, "ollama_session", _StubSession()), \
             patch.object(proxy_module, "_call_oai_compatible_direct",
                          new=AsyncMock(return_value=fake_response)), \
             patch.object(proxy_module, "request_stats",
                          MagicMock(record=MagicMock())):
            resp = test_client.post(
                "/v1/messages",
                json={
                    "model": "ollama:llama3.2",
                    "max_tokens": 256,
                    "messages": [{"role": "user", "content": "list files"}],
                    "tools": [{
                        "name": "Bash",
                        "description": "Run a shell command",
                        "input_schema": {
                            "type": "object",
                            "properties": {"cmd": {"type": "string"}},
                        },
                    }],
                },
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["type"] == "message"
        assert body["role"] == "assistant"
        assert body["stop_reason"] == "tool_use"
        tool_blocks = [b for b in body["content"] if b["type"] == "tool_use"]
        assert len(tool_blocks) == 1
        assert tool_blocks[0]["name"] == "Bash"
        assert tool_blocks[0]["input"] == {"cmd": "ls"}

    def test_tools_on_non_oai_model_soft_fallback(self, test_client, proxy_module, caplog):
        """Tools + non-OAI model now soft-falls-back onto the legacy text-only
        path with a warning, instead of returning HTTP 400.  This keeps clients
        like CCS (which tier-route to claude-*/gpt-5.*/antigravity-* models)
        working — they just won't get structured tool dispatch on that turn."""
        mock_auth = MagicMock(is_ready=True, session=MagicMock(), headers={},
                              body_template={"messages": []})
        with patch.object(proxy_module, "auth", mock_auth), \
             patch.object(proxy_module, "call_api_direct",
                          new_callable=AsyncMock,
                          return_value="I would run that command for you."), \
             patch.object(proxy_module, "request_stats", MagicMock(record=MagicMock())), \
             caplog.at_level("WARNING", logger=proxy_module.logger.name):
            resp = test_client.post(
                "/v1/messages",
                json={
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "hi"}],
                    "tools": [{
                        "name": "Bash",
                        "description": "Run a command",
                        "input_schema": {"type": "object"},
                    }],
                },
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["type"] == "message"
        assert body["content"][0]["type"] == "text"
        assert body["content"][0]["text"] == "I would run that command for you."
        # A WARNING log should have been emitted explaining the fallback.
        warning_text = " ".join(r.getMessage() for r in caplog.records
                                if r.levelname == "WARNING")
        assert ("Soft-fallback" in warning_text
                or "soft-fallback" in warning_text
                or "does not route to an OAI-compatible provider" in warning_text)

    def test_tools_on_non_oai_model_soft_fallback_streaming(self, test_client, proxy_module, caplog):
        """Streaming variant of the soft-fallback: tools + claude-* + stream=True
        must produce an Anthropic SSE stream (via the text-only pipeline) and
        emit the Soft-fallback warning."""
        async def _fake_claude_stream(system_prompt, messages, model, max_tokens):
            # Emit a minimal OpenAI SSE stream with two text deltas, like the
            # real call_api_streaming() does.
            import json as _json
            chunk1 = {
                "id": "chatcmpl-test", "object": "chat.completion.chunk",
                "created": 1, "model": model,
                "choices": [{"index": 0, "delta": {"content": "Hello "},
                             "finish_reason": None}],
            }
            chunk2 = {
                "id": "chatcmpl-test", "object": "chat.completion.chunk",
                "created": 1, "model": model,
                "choices": [{"index": 0, "delta": {"content": "world"},
                             "finish_reason": None}],
            }
            done = {
                "id": "chatcmpl-test", "object": "chat.completion.chunk",
                "created": 1, "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {_json.dumps(chunk1)}\n\n"
            yield f"data: {_json.dumps(chunk2)}\n\n"
            yield f"data: {_json.dumps(done)}\n\n"
            yield "data: [DONE]\n\n"

        mock_auth = MagicMock(is_ready=True, session=MagicMock(), headers={},
                              body_template={"messages": []})
        with patch.object(proxy_module, "auth", mock_auth), \
             patch.object(proxy_module, "call_api_streaming",
                          return_value=_fake_claude_stream(None, [], "claude-sonnet-4-6", 100)), \
             patch.object(proxy_module, "timeout_routing_enabled", False), \
             patch.object(proxy_module, "request_stats", MagicMock(record=MagicMock())), \
             caplog.at_level("WARNING", logger=proxy_module.logger.name):
            resp = test_client.post(
                "/v1/messages",
                json={
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 100,
                    "stream": True,
                    "messages": [{"role": "user", "content": "hi"}],
                    "tools": [{
                        "name": "Bash",
                        "description": "Run a command",
                        "input_schema": {"type": "object"},
                    }],
                },
            )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        body = resp.text
        assert "event: message_start" in body
        assert "event: content_block_delta" in body
        assert "event: message_stop" in body
        warning_text = " ".join(r.getMessage() for r in caplog.records
                                if r.levelname == "WARNING")
        assert ("Soft-fallback" in warning_text
                or "soft-fallback" in warning_text
                or "does not route to an OAI-compatible provider" in warning_text)

    def test_without_tools_still_uses_text_fallback(self, test_client, proxy_module):
        """Regression: a tools-free /v1/messages POST must still go through
        the legacy text-only pipeline unchanged, even for OAI-compat models."""
        with patch.object(proxy_module, "call_ollama_direct",
                          new_callable=AsyncMock,
                          return_value="Hello!"), \
             patch.object(proxy_module, "request_stats",
                          MagicMock(record=MagicMock())):
            resp = test_client.post(
                "/v1/messages",
                json={
                    "model": "ollama:llama3.2",
                    "max_tokens": 128,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["type"] == "message"
        assert body["stop_reason"] == "end_turn"
        assert any(b["type"] == "text" and b["text"] == "Hello!"
                   for b in body["content"])

    def test_streaming_tool_use_end_to_end(self, test_client, proxy_module):
        async def _fake_stream(resolved, payload, request_id):
            for c in [
                {"choices": [{"index": 0, "delta": {
                    "tool_calls": [{"index": 0, "id": "call_s", "type": "function",
                                    "function": {"name": "Bash",
                                                 "arguments": '{"cmd":"pwd"}'}}]
                }, "finish_reason": None}]},
                {"choices": [{"index": 0, "delta": {},
                              "finish_reason": "tool_calls"}]},
            ]:
                yield c

        with patch.object(proxy_module, "ollama_session", _StubSession()), \
             patch.object(proxy_module, "_stream_oai_compatible",
                          new=_fake_stream), \
             patch.object(proxy_module, "request_stats",
                          MagicMock(record=MagicMock())):
            resp = test_client.post(
                "/v1/messages",
                json={
                    "model": "ollama:llama3.2",
                    "max_tokens": 128,
                    "stream": True,
                    "messages": [{"role": "user", "content": "where am i?"}],
                    "tools": [{
                        "name": "Bash",
                        "description": "shell",
                        "input_schema": {"type": "object"},
                    }],
                },
            )
        assert resp.status_code == 200
        body = resp.text
        assert "event: message_start" in body
        assert "event: content_block_start" in body
        assert '"type":"tool_use"' in body or '"type": "tool_use"' in body
        assert '"name":"Bash"' in body or '"name": "Bash"' in body
        assert "event: message_stop" in body
