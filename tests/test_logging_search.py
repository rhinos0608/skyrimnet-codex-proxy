"""Tests for conversation logging, cheap summarizer, and web search integration."""
import json
import os
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from orchestrator import (
    ConversationLogger, conversation_log, summarize_text,
    search_and_summarize, _make_summary, StepResult,
    _SUMMARIZER_PREFERENCE,
)


# ---------------------------------------------------------------------------
# ConversationLogger
# ---------------------------------------------------------------------------

class TestConversationLogger:
    def test_log_user_input(self, tmp_path):
        path = str(tmp_path / "test.jsonl")
        log = ConversationLogger(path)
        log.log_user_input("chat_claude", "Hello world", "claude", "claude-sonnet")

        with open(path) as f:
            entry = json.loads(f.readline())
        assert entry["turn"] == "user_input"
        assert entry["tool"] == "chat_claude"
        assert entry["message"] == "Hello world"
        assert entry["provider"] == "claude"
        assert "_ts" in entry

    def test_log_model_output(self, tmp_path):
        path = str(tmp_path / "test.jsonl")
        log = ConversationLogger(path)
        log.log_model_output("chat_claude", "claude", "claude-sonnet",
                             "short summary", 5000, 1.23)

        with open(path) as f:
            entry = json.loads(f.readline())
        assert entry["turn"] == "model_output"
        assert entry["summary"] == "short summary"
        assert entry["full_length"] == 5000
        assert entry["latency_s"] == 1.23

    def test_log_sub_agent_turn(self, tmp_path):
        path = str(tmp_path / "test.jsonl")
        log = ConversationLogger(path)
        log.log_sub_agent_turn("task-123", 0, "codex_cli", "gpt-5",
                               "did the thing", 2.5)

        with open(path) as f:
            entry = json.loads(f.readline())
        assert entry["turn"] == "sub_agent_turn"
        assert entry["task_id"] == "task-123"
        assert entry["step_index"] == 0

    def test_log_web_search(self, tmp_path):
        path = str(tmp_path / "test.jsonl")
        log = ConversationLogger(path)
        log.log_web_search("python async", 5, 3000, 500,
                           "ollama", "llama3.1", 1.5)

        with open(path) as f:
            entry = json.loads(f.readline())
        assert entry["turn"] == "web_search_result"
        assert entry["query"] == "python async"
        assert entry["result_count"] == 5
        assert entry["summarizer_provider"] == "ollama"

    def test_truncates_long_messages(self, tmp_path):
        path = str(tmp_path / "test.jsonl")
        log = ConversationLogger(path)
        long_msg = "x" * 5000
        log.log_user_input("test", long_msg)

        with open(path) as f:
            entry = json.loads(f.readline())
        assert len(entry["message"]) == 2000

    def test_multiple_entries_append(self, tmp_path):
        path = str(tmp_path / "test.jsonl")
        log = ConversationLogger(path)
        log.log_user_input("t1", "msg1")
        log.log_user_input("t2", "msg2")

        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 2

    def test_write_failure_non_fatal(self):
        log = ConversationLogger("/nonexistent/dir/file.jsonl")
        # Should not raise
        log.log_user_input("test", "msg")


# ---------------------------------------------------------------------------
# Cheap model summarizer
# ---------------------------------------------------------------------------

class TestSummarizeText:
    @pytest.mark.asyncio
    async def test_short_text_passthrough(self):
        result = await summarize_text("short text")
        assert result.provider == "passthrough"
        assert result.full == "short text"

    @pytest.mark.asyncio
    async def test_empty_text_passthrough(self):
        result = await summarize_text("")
        assert result.provider == "passthrough"

    @pytest.mark.asyncio
    async def test_no_providers_truncates(self):
        long = "word " * 200
        result = await summarize_text(long, providers=None)
        assert result.provider == "truncation"
        assert len(result.full) <= 3000

    @pytest.mark.asyncio
    async def test_routes_to_cheapest(self):
        """Verify providers are sorted by cheapness preference."""
        expensive = {"name": "claude", "default_model": "m1",
                     "call_fn": AsyncMock(return_value="expensive result"),
                     "accepts_extra": False}
        cheap = {"name": "ollama", "default_model": "m2",
                 "call_fn": AsyncMock(return_value="cheap result"),
                 "accepts_extra": False}

        long_text = "word " * 200
        with patch("orchestrator._get_proxy") as mock_gp:
            mock_gp.return_value.request_stats.record = lambda *a, **kw: None
            result = await summarize_text(long_text, providers=[expensive, cheap])

        assert result.provider == "ollama"
        assert result.full == "cheap result"
        # Cheap provider was tried first, expensive never called
        cheap["call_fn"].assert_called_once()
        expensive["call_fn"].assert_not_called()

    def test_preference_order(self):
        assert _SUMMARIZER_PREFERENCE[0] == "ollama"
        assert "claude" in _SUMMARIZER_PREFERENCE
        assert _SUMMARIZER_PREFERENCE.index("ollama") < _SUMMARIZER_PREFERENCE.index("claude")


# ---------------------------------------------------------------------------
# search_and_summarize
# ---------------------------------------------------------------------------

class TestSearchAndSummarize:
    @pytest.mark.asyncio
    async def test_search_failure_returns_error(self):
        with patch("orchestrator.search_web", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = {"error": "search down", "results": []}
            result = await search_and_summarize("test query", providers=[])

        assert result["result_count"] == 0
        assert "Search failed" in result["summarized"]

    @pytest.mark.asyncio
    async def test_search_success_summarizes(self):
        mock_results = {
            "results": [
                {"title": "Result 1", "url": "https://example.com/1",
                 "description": "A detailed description of the first result that contains enough text to exceed the summarizer passthrough threshold of two hundred characters."},
                {"title": "Result 2", "url": "https://example.com/2",
                 "description": "Another detailed description for the second result which also has enough content to push the total well past the minimum."},
            ]
        }

        provider = {"name": "ollama", "default_model": "m1",
                    "call_fn": AsyncMock(return_value="Summarized: two results about example.com"),
                    "accepts_extra": False}

        with patch("orchestrator.search_web", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = mock_results
            with patch("orchestrator._get_proxy") as mock_gp:
                mock_gp.return_value.request_stats.record = lambda *a, **kw: None
                result = await search_and_summarize("test", providers=[provider], count=2)

        assert result["result_count"] == 2
        assert result["query"] == "test"
        assert "Summarized" in result["summarized"]
        assert result["summarizer"] == "ollama/m1"


# ---------------------------------------------------------------------------
# MCP server web_search tool registration
# ---------------------------------------------------------------------------

class TestWebSearchToolRegistration:
    def test_web_search_registered(self, proxy_module):
        import mcp_server
        server = mcp_server.create_mcp_server()
        assert "web_search" in server._tool_manager._tools

    def test_total_tool_count(self, proxy_module):
        import mcp_server
        server = mcp_server.create_mcp_server()
        assert len(server._tool_manager._tools) == 23


# ---------------------------------------------------------------------------
# _logged_call helper
# ---------------------------------------------------------------------------

class TestLoggedCall:
    @pytest.mark.asyncio
    async def test_logs_input_and_output(self, tmp_path):
        import orchestrator
        original_path = orchestrator.conversation_log._path
        try:
            log_path = str(tmp_path / "test.jsonl")
            orchestrator.conversation_log._path = log_path

            import mcp_server
            mock_fn = AsyncMock(return_value="response text")
            result = await mcp_server._logged_call(
                "test_tool", "test_provider", "test_model",
                mock_fn, None, [{"role": "user", "content": "hello"}], 4096,
            )

            assert result == "response text"

            with open(log_path) as f:
                lines = f.readlines()
            assert len(lines) == 2

            input_entry = json.loads(lines[0])
            assert input_entry["turn"] == "user_input"
            assert input_entry["message"] == "hello"

            output_entry = json.loads(lines[1])
            assert output_entry["turn"] == "model_output"
            assert output_entry["provider"] == "test_provider"
            assert output_entry["full_length"] == len("response text")
        finally:
            orchestrator.conversation_log._path = original_path
