"""Tests for orchestrator module -- TaskStore, StepResult, Orchestrator, fallback."""
import asyncio
import json
import os
import pytest
from unittest.mock import AsyncMock, patch

from orchestrator import (
    TaskStore, TaskStatus, Task, TaskStep, StepResult,
    Orchestrator, call_with_stats, call_with_fallback,
    _make_summary, _append_checkpoint, CHECKPOINT_FILE,
)


# ---------------------------------------------------------------------------
# _make_summary
# ---------------------------------------------------------------------------

class TestMakeSummary:
    def test_short_text(self):
        assert _make_summary("Hello world") == "Hello world"

    def test_empty_text(self):
        assert _make_summary("") == "(empty response)"

    def test_none_text(self):
        assert _make_summary(None) == "(empty response)"

    def test_first_paragraph(self):
        text = "First paragraph.\n\nSecond paragraph with more detail."
        assert _make_summary(text) == "First paragraph."

    def test_truncation(self):
        long_text = "word " * 200  # ~1000 chars
        result = _make_summary(long_text, max_chars=50)
        assert len(result) <= 55  # some slack for "..."
        assert result.endswith("...")


# ---------------------------------------------------------------------------
# StepResult
# ---------------------------------------------------------------------------

class TestStepResult:
    def test_to_dict(self):
        sr = StepResult(
            summary="short",
            full="full result text",
            provider="claude",
            model="claude-sonnet",
            latency_s=1.234,
        )
        d = sr.to_dict()
        assert d["summary"] == "short"
        assert d["full"] == "full result text"
        assert d["provider"] == "claude"
        assert d["latency_s"] == 1.23


# ---------------------------------------------------------------------------
# TaskStore
# ---------------------------------------------------------------------------

class TestTaskStore:
    def setup_method(self):
        self.store = TaskStore(max_tasks=10)

    def test_create_task(self):
        task = self.store.create("test", "cli")
        assert task.id.startswith("task-")
        assert task.status == TaskStatus.PENDING
        assert task.description == "test"
        assert task.category == "cli"

    def test_create_with_steps(self):
        task = self.store.create("test", "chat",
                                  steps=[{"description": "s1"}, {"description": "s2"}])
        assert len(task.steps) == 2
        assert task.steps[0].description == "s1"
        assert task.steps[1].description == "s2"

    def test_lifecycle_pending_to_completed(self):
        task = self.store.create("test", "cli")
        assert task.status == TaskStatus.PENDING

        self.store.start(task.id)
        assert task.status == TaskStatus.RUNNING
        assert task.started_at is not None

        self.store.complete(task.id, "result text", "summary")
        assert task.status == TaskStatus.COMPLETED
        assert task.result == "result text"
        assert task.summary == "summary"
        assert task.finished_at is not None

    def test_lifecycle_pending_to_failed(self):
        task = self.store.create("test", "cli")
        self.store.start(task.id)
        self.store.fail(task.id, "something broke")
        assert task.status == TaskStatus.FAILED
        assert task.error == "something broke"

    def test_step_lifecycle(self):
        task = self.store.create("test", "cli",
                                  steps=[{"description": "step1"}])
        self.store.start(task.id)
        step = task.steps[0]

        self.store.start_step(task.id, step.id)
        assert step.status == TaskStatus.RUNNING

        self.store.complete_step(task.id, step.id, "step result", "step summary")
        assert step.status == TaskStatus.COMPLETED
        assert step.result == "step result"
        assert step.summary == "step summary"
        assert step.latency_s is not None

    def test_step_fail(self):
        task = self.store.create("test", "cli",
                                  steps=[{"description": "step1"}])
        self.store.start(task.id)
        step = task.steps[0]
        self.store.start_step(task.id, step.id)
        self.store.fail_step(task.id, step.id, "step error", provider="claude")
        assert step.status == TaskStatus.FAILED
        assert step.error == "step error"
        assert step.provider == "claude"

    def test_get_nonexistent(self):
        assert self.store.get("nonexistent") is None

    def test_list_active(self):
        t1 = self.store.create("active", "cli")
        self.store.start(t1.id)
        t2 = self.store.create("done", "cli")
        self.store.start(t2.id)
        self.store.complete(t2.id, "done")

        active = self.store.list_active()
        assert len(active) == 1
        assert active[0].id == t1.id

    def test_list_all(self):
        self.store.create("t1", "cli")
        self.store.create("t2", "cli")
        assert len(self.store.list_all()) == 2

    def test_eviction(self):
        store = TaskStore(max_tasks=3)
        # Fill with completed tasks
        for i in range(3):
            t = store.create(f"task{i}", "cli")
            store.start(t.id)
            store.complete(t.id, "done")

        # Should evict oldest
        t4 = store.create("task3", "cli")
        assert len(store.list_all()) == 3
        assert t4.id in [t.id for t in store.list_all()]

    def test_to_dict_summary_vs_full(self):
        task = self.store.create("test", "cli",
                                  steps=[{"description": "s1"}])
        self.store.start(task.id)
        self.store.start_step(task.id, task.steps[0].id)
        self.store.complete_step(task.id, task.steps[0].id, "long result text")
        self.store.complete(task.id, "full combined result")

        summary_d = task.to_dict()
        assert "result" not in summary_d
        assert "result_length" in summary_d

        full_d = task.to_full_dict()
        assert "result" in full_d
        assert full_d["result"] == "full combined result"

    def test_auto_summary_on_complete(self):
        task = self.store.create("test", "cli")
        self.store.start(task.id)
        self.store.complete(task.id, "First paragraph here.\n\nSecond paragraph.")
        assert task.summary == "First paragraph here."


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

class TestCheckpoint:
    def test_append_checkpoint(self, tmp_path):
        import orchestrator
        original = orchestrator.CHECKPOINT_FILE
        try:
            orchestrator.CHECKPOINT_FILE = str(tmp_path / "test.jsonl")
            _append_checkpoint({"event": "test", "data": 123})
            with open(orchestrator.CHECKPOINT_FILE) as f:
                lines = f.readlines()
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data["event"] == "test"
            assert "_ts" in data
        finally:
            orchestrator.CHECKPOINT_FILE = original


# ---------------------------------------------------------------------------
# call_with_stats
# ---------------------------------------------------------------------------

class TestCallWithStats:
    @pytest.mark.asyncio
    async def test_records_success(self):
        mock_fn = AsyncMock(return_value="response text")
        with patch("orchestrator._get_proxy") as mock_gp:
            mock_gp.return_value.request_stats.record = lambda *a, **kw: None
            result = await call_with_stats(
                mock_fn, "test_provider", None, [{"role": "user", "content": "hi"}],
                "test-model", 4096,
            )
        assert isinstance(result, StepResult)
        assert result.full == "response text"
        assert result.provider == "test_provider"
        assert result.model == "test-model"
        assert result.latency_s >= 0

    @pytest.mark.asyncio
    async def test_records_failure(self):
        mock_fn = AsyncMock(side_effect=RuntimeError("provider down"))
        with patch("orchestrator._get_proxy") as mock_gp:
            mock_gp.return_value.request_stats.record = lambda *a, **kw: None
            with pytest.raises(RuntimeError, match="provider down"):
                await call_with_stats(
                    mock_fn, "test_provider", None, [], "test-model", 4096,
                )


# ---------------------------------------------------------------------------
# call_with_fallback
# ---------------------------------------------------------------------------

class TestCallWithFallback:
    @pytest.mark.asyncio
    async def test_first_provider_succeeds(self):
        providers = [
            {"name": "p1", "default_model": "m1", "call_fn": AsyncMock(return_value="ok"),
             "accepts_extra": False},
        ]
        with patch("orchestrator._get_proxy") as mock_gp:
            mock_gp.return_value.request_stats.record = lambda *a, **kw: None
            result = await call_with_fallback(
                providers, None, [{"role": "user", "content": "hi"}], None, 4096,
            )
        assert isinstance(result, StepResult)
        assert result.full == "ok"
        assert result.provider == "p1"

    @pytest.mark.asyncio
    async def test_fallback_on_failure(self):
        providers = [
            {"name": "p1", "default_model": "m1",
             "call_fn": AsyncMock(side_effect=RuntimeError("fail")),
             "accepts_extra": False},
            {"name": "p2", "default_model": "m2",
             "call_fn": AsyncMock(return_value="backup ok"),
             "accepts_extra": False},
        ]
        with patch("orchestrator._get_proxy") as mock_gp:
            mock_gp.return_value.request_stats.record = lambda *a, **kw: None
            result = await call_with_fallback(
                providers, None, [{"role": "user", "content": "hi"}], None, 4096,
            )
        assert result.full == "backup ok"
        assert result.provider == "p2"

    @pytest.mark.asyncio
    async def test_all_fail_raises(self):
        providers = [
            {"name": "p1", "default_model": "m1",
             "call_fn": AsyncMock(side_effect=RuntimeError("fail1")),
             "accepts_extra": False},
            {"name": "p2", "default_model": "m2",
             "call_fn": AsyncMock(side_effect=RuntimeError("fail2")),
             "accepts_extra": False},
        ]
        with patch("orchestrator._get_proxy") as mock_gp:
            mock_gp.return_value.request_stats.record = lambda *a, **kw: None
            with pytest.raises(RuntimeError, match="fail2"):
                await call_with_fallback(
                    providers, None, [{"role": "user", "content": "hi"}], None, 4096,
                )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class TestOrchestrator:
    def setup_method(self):
        self.store = TaskStore()
        self.orch = Orchestrator(self.store)

    @pytest.mark.asyncio
    async def test_execute_single_success(self):
        provider = {
            "name": "test", "default_model": "m1",
            "call_fn": AsyncMock(return_value="result"),
            "accepts_extra": False,
        }
        with patch("orchestrator._get_proxy") as mock_gp:
            mock_gp.return_value.request_stats.record = lambda *a, **kw: None
            task = await self.orch.execute_single(
                "test task", "cli", "do something", [provider],
            )
        assert task.status == TaskStatus.COMPLETED
        assert task.result == "result"
        assert task.summary is not None
        assert task.steps[0].provider == "test"

    @pytest.mark.asyncio
    async def test_execute_single_failure(self):
        provider = {
            "name": "test", "default_model": "m1",
            "call_fn": AsyncMock(side_effect=RuntimeError("boom")),
            "accepts_extra": False,
        }
        with patch("orchestrator._get_proxy") as mock_gp:
            mock_gp.return_value.request_stats.record = lambda *a, **kw: None
            task = await self.orch.execute_single(
                "test task", "cli", "do something", [provider],
            )
        assert task.status == TaskStatus.FAILED
        assert "boom" in task.error

    @pytest.mark.asyncio
    async def test_execute_sequential(self):
        provider = {
            "name": "test", "default_model": "m1",
            "call_fn": AsyncMock(side_effect=["result1", "result2"]),
            "accepts_extra": False,
        }
        steps = [
            {"description": "Step 1", "message": "do step 1"},
            {"description": "Step 2", "message": "do step 2"},
        ]
        with patch("orchestrator._get_proxy") as mock_gp:
            mock_gp.return_value.request_stats.record = lambda *a, **kw: None
            task = await self.orch.execute_sequential(
                "multi-step", "cli", steps, [provider],
            )
        assert task.status == TaskStatus.COMPLETED
        assert len(task.steps) == 2
        assert task.steps[0].status == TaskStatus.COMPLETED
        assert task.steps[1].status == TaskStatus.COMPLETED
        assert task.summary is not None

    @pytest.mark.asyncio
    async def test_execute_sequential_step_failure_stops(self):
        provider = {
            "name": "test", "default_model": "m1",
            "call_fn": AsyncMock(side_effect=[RuntimeError("step1 failed")]),
            "accepts_extra": False,
        }
        steps = [
            {"description": "Step 1", "message": "do step 1"},
            {"description": "Step 2", "message": "do step 2"},
        ]
        with patch("orchestrator._get_proxy") as mock_gp:
            mock_gp.return_value.request_stats.record = lambda *a, **kw: None
            task = await self.orch.execute_sequential(
                "multi-step", "cli", steps, [provider],
            )
        assert task.status == TaskStatus.FAILED
        assert task.steps[0].status == TaskStatus.FAILED
        assert task.steps[1].status == TaskStatus.PENDING  # never started

    @pytest.mark.asyncio
    async def test_execute_parallel(self):
        provider = {
            "name": "test", "default_model": "m1",
            "call_fn": AsyncMock(side_effect=["result1", "result2"]),
            "accepts_extra": False,
        }
        steps = [
            {"description": "Task A", "message": "do A"},
            {"description": "Task B", "message": "do B"},
        ]
        with patch("orchestrator._get_proxy") as mock_gp:
            mock_gp.return_value.request_stats.record = lambda *a, **kw: None
            task = await self.orch.execute_parallel(
                "parallel work", "cli", steps, [provider],
            )
        assert task.status == TaskStatus.COMPLETED
        assert len(task.steps) == 2

    @pytest.mark.asyncio
    async def test_execute_parallel_partial_failure(self):
        call_fn = AsyncMock(side_effect=["result1", RuntimeError("fail")])
        provider = {
            "name": "test", "default_model": "m1",
            "call_fn": call_fn,
            "accepts_extra": False,
        }
        steps = [
            {"description": "Task A", "message": "do A"},
            {"description": "Task B", "message": "do B"},
        ]
        with patch("orchestrator._get_proxy") as mock_gp:
            mock_gp.return_value.request_stats.record = lambda *a, **kw: None
            task = await self.orch.execute_parallel(
                "parallel work", "cli", steps, [provider],
            )
        # Should still complete since one step succeeded
        assert task.status == TaskStatus.COMPLETED
        assert "failed" in task.summary.lower() or "Failed" in task.summary

    @pytest.mark.asyncio
    async def test_sequential_pipes_summaries(self):
        """Verify summaries (not full results) are piped between steps."""
        captured_messages = []

        async def capture_fn(system_prompt, messages, model, max_tokens, **kw):
            captured_messages.append(messages[0]["content"])
            return f"result for call {len(captured_messages)}"

        provider = {
            "name": "test", "default_model": "m1",
            "call_fn": capture_fn,
            "accepts_extra": False,
        }
        steps = [
            {"description": "Step 1", "message": "first step"},
            {"description": "Step 2", "message": "second step"},
        ]
        with patch("orchestrator._get_proxy") as mock_gp:
            mock_gp.return_value.request_stats.record = lambda *a, **kw: None
            await self.orch.execute_sequential(
                "test", "cli", steps, [provider],
            )

        # Second call should have "Previous step summaries" with summary of step 1
        assert len(captured_messages) == 2
        assert "Previous step summaries" in captured_messages[1]
        assert "Step 1" in captured_messages[1]


# ---------------------------------------------------------------------------
# MCP server integration
# ---------------------------------------------------------------------------

class TestMCPOrchestratorTools:
    def test_orchestration_tools_registered(self, proxy_module):
        import mcp_server
        server = mcp_server.create_mcp_server()
        tool_names = set(server._tool_manager._tools.keys())
        assert "orchestrate_sequential" in tool_names
        assert "orchestrate_parallel" in tool_names
        assert "task_status" in tool_names
        assert "task_result" in tool_names
        assert "task_list" in tool_names

    def test_total_tool_count(self, proxy_module):
        import mcp_server
        server = mcp_server.create_mcp_server()
        tools = server._tool_manager._tools
        # 6 chat + 5 cli + 3 list + 2 unified + 5 orchestration = 21
        assert len(tools) == 23
