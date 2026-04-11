"""
Orchestrator for CLI agent coordination and state management.

Provides:
  - TaskStore: in-memory task state machine with JSONL checkpoint for replay
  - StepResult: structured output with summary + full result for aggregation
  - Orchestrator: dispatches work to providers with fallback, collects results
  - Stats wrapper: records latency/success for all MCP-initiated calls

Designed to be imported by mcp_server.py and wired into the FastMCP instance.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger("orchestrator")

# Lazy import to avoid circular dependency at module load time.
# proxy is always available when call_with_stats runs (mcp_server imports both).
_proxy = None

def _get_proxy():
    global _proxy
    if _proxy is None:
        import proxy
        _proxy = proxy
    return _proxy

# File paths live next to config.json
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_FILE = os.path.join(_BASE_DIR, ".task-checkpoint.jsonl")
CONVERSATION_LOG_FILE = os.path.join(_BASE_DIR, ".conversation-log.jsonl")


# ---------------------------------------------------------------------------
# Structured step result -- summary + full for aggregation
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    """Structured output from a provider call.

    The summary is short enough for an orchestrator's context window when
    aggregating across many steps.  The full result is available for deep
    inspection when needed.
    """
    summary: str
    full: str
    provider: str
    model: str
    latency_s: float

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "full": self.full,
            "provider": self.provider,
            "model": self.model,
            "latency_s": round(self.latency_s, 2),
        }


def _make_summary(text: str, max_chars: int = 300) -> str:
    """Extract a short summary from a provider response.

    Uses the first paragraph or first N chars, whichever is shorter.
    """
    if not text:
        return "(empty response)"
    # Take first paragraph
    first_para = text.split("\n\n")[0].strip()
    if len(first_para) <= max_chars:
        return first_para
    return first_para[:max_chars].rsplit(" ", 1)[0] + "..."


# ---------------------------------------------------------------------------
# Task State Machine
# ---------------------------------------------------------------------------

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskStep:
    """A single step within a multi-step task."""
    id: str
    description: str
    provider: Optional[str] = None
    model: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    summary: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    latency_s: Optional[float] = None

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "description": self.description,
            "status": self.status.value,
        }
        if self.provider:
            d["provider"] = self.provider
        if self.model:
            d["model"] = self.model
        if self.summary:
            d["summary"] = self.summary
        if self.result:
            d["result_length"] = len(self.result)
        if self.error:
            d["error"] = self.error
        if self.latency_s is not None:
            d["latency_s"] = round(self.latency_s, 2)
        return d

    def to_full_dict(self) -> dict:
        """Full dict including complete result text (for resource queries)."""
        d = self.to_dict()
        if self.result:
            d["result"] = self.result
        return d


@dataclass
class Task:
    """A tracked orchestration task with one or more steps."""
    id: str
    description: str
    category: str  # "chat" or "cli"
    status: TaskStatus = TaskStatus.PENDING
    steps: list[TaskStep] = field(default_factory=list)
    result: Optional[str] = None
    summary: Optional[str] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Summary dict (step results are summaries only)."""
        d = {
            "id": self.id,
            "description": self.description,
            "category": self.category,
            "status": self.status.value,
            "steps": [s.to_dict() for s in self.steps],
            "created_at": self.created_at,
        }
        if self.summary:
            d["summary"] = self.summary
        if self.result:
            d["result_length"] = len(self.result)
        if self.error:
            d["error"] = self.error
        if self.started_at:
            d["started_at"] = self.started_at
        if self.finished_at:
            d["finished_at"] = self.finished_at
            d["total_latency_s"] = round(self.finished_at - self.started_at, 2)
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    def to_full_dict(self) -> dict:
        """Full dict including complete results (for resource queries)."""
        d = self.to_dict()
        if self.result:
            d["result"] = self.result
        d["steps"] = [s.to_full_dict() for s in self.steps]
        return d


# ---------------------------------------------------------------------------
# JSONL Checkpoint
# ---------------------------------------------------------------------------

def _append_checkpoint(event: dict):
    """Append a task event to the JSONL checkpoint file."""
    try:
        event["_ts"] = time.time()
        with open(CHECKPOINT_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=str) + "\n")
    except Exception as e:
        logger.debug(f"Checkpoint write failed (non-fatal): {e}")


# ---------------------------------------------------------------------------
# Conversation Logger
# ---------------------------------------------------------------------------

class ConversationLogger:
    """Append-only JSONL logger for conversation turns.

    Turn types:
      - user_input: message sent by the user/caller
      - model_output: response from a provider
      - sub_agent_turn: an orchestrator step or sub-agent call
      - web_search_result: search results (raw + summarized)
    """

    def __init__(self, path: str = CONVERSATION_LOG_FILE):
        self._path = path

    def _write(self, entry: dict):
        try:
            entry["_ts"] = time.time()
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as e:
            logger.debug(f"Conversation log write failed (non-fatal): {e}")

    def log_user_input(self, tool: str, message: str, provider: str = None,
                       model: str = None, metadata: dict = None):
        self._write({
            "turn": "user_input",
            "tool": tool,
            "message": message[:2000],
            "provider": provider,
            "model": model,
            **(metadata or {}),
        })

    def log_model_output(self, tool: str, provider: str, model: str,
                         summary: str, full_length: int, latency_s: float,
                         metadata: dict = None):
        self._write({
            "turn": "model_output",
            "tool": tool,
            "provider": provider,
            "model": model,
            "summary": summary[:500],
            "full_length": full_length,
            "latency_s": round(latency_s, 2),
            **(metadata or {}),
        })

    def log_sub_agent_turn(self, task_id: str, step_index: int,
                           provider: str, model: str,
                           summary: str, latency_s: float):
        self._write({
            "turn": "sub_agent_turn",
            "task_id": task_id,
            "step_index": step_index,
            "provider": provider,
            "model": model,
            "summary": summary[:500],
            "latency_s": round(latency_s, 2),
        })

    def log_web_search(self, query: str, result_count: int,
                       raw_length: int, summarized_length: int,
                       summarizer_provider: str = None,
                       summarizer_model: str = None,
                       latency_s: float = 0):
        self._write({
            "turn": "web_search_result",
            "query": query[:500],
            "result_count": result_count,
            "raw_length": raw_length,
            "summarized_length": summarized_length,
            "summarizer_provider": summarizer_provider,
            "summarizer_model": summarizer_model,
            "latency_s": round(latency_s, 2),
        })


# Singleton logger
conversation_log = ConversationLogger()


# ---------------------------------------------------------------------------
# Cheap model summarizer
# ---------------------------------------------------------------------------

# Provider preference order for summarization (cheapest/fastest first).
# Names must match PROVIDERS registry in mcp_server.py.
_SUMMARIZER_PREFERENCE = ["ollama", "xiaomi", "zai", "openrouter", "antigravity", "claude"]


async def summarize_text(text: str, instruction: str = None,
                         providers: list[dict] = None,
                         max_tokens: int = 1024) -> StepResult:
    """Compress/summarize text using the cheapest available provider.

    Args:
        text: The text to summarize.
        instruction: Custom instruction for the summarizer. Defaults to
            a generic compression prompt.
        providers: Available provider list (healthy providers from mcp_server).
            If None, falls back to _get_proxy() direct call on default model.
        max_tokens: Max tokens for the summary response.

    Returns:
        StepResult with summary and full summarized text.
    """
    if not text or len(text) < 200:
        return StepResult(
            summary=text or "",
            full=text or "",
            provider="passthrough",
            model="none",
            latency_s=0,
        )

    prompt = instruction or (
        "Summarize the following text concisely. Preserve key facts, names, "
        "URLs, dates, and technical details. Remove redundancy and filler. "
        "Output only the summary, no preamble."
    )
    messages = [{"role": "user", "content": f"{prompt}\n\n---\n\n{text[:8000]}"}]

    if not providers:
        # Fallback: no provider list available, return truncated
        return StepResult(
            summary=_make_summary(text),
            full=text[:3000],
            provider="truncation",
            model="none",
            latency_s=0,
        )

    # Sort providers by cheapness preference
    def _sort_key(p):
        name = p["name"]
        try:
            return _SUMMARIZER_PREFERENCE.index(name)
        except ValueError:
            return len(_SUMMARIZER_PREFERENCE)

    sorted_providers = sorted(providers, key=_sort_key)

    return await call_with_fallback(
        sorted_providers,
        None,  # no system prompt
        messages,
        None,  # use provider default model
        max_tokens,
    )


# ---------------------------------------------------------------------------
# Search MCP client
# ---------------------------------------------------------------------------

_SEARCH_MCP_CMD = "node"
_SEARCH_MCP_ARGS = [os.path.expanduser("~/search-mcp/dist/index.js")]

# Persistent client session — initialized on first use, reused across calls.
_search_session: Any = None
_search_ctx: Any = None  # context manager refs for cleanup


async def _get_search_session():
    """Get or create a persistent MCP client session to the search-mcp server."""
    global _search_session, _search_ctx
    if _search_session is not None:
        return _search_session

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command=_SEARCH_MCP_CMD, args=_SEARCH_MCP_ARGS)
    # stdio_client is an async context manager; we hold it open for the process lifetime
    _search_ctx = stdio_client(params)
    read_stream, write_stream = await _search_ctx.__aenter__()
    _search_session = ClientSession(read_stream, write_stream)
    await _search_session.__aenter__()
    await _search_session.initialize()
    logger.info("Search MCP client connected")
    return _search_session


async def close_search_session():
    """Cleanly shut down the search MCP client."""
    global _search_session, _search_ctx
    if _search_session:
        try:
            await _search_session.__aexit__(None, None, None)
        except Exception:
            pass
        _search_session = None
    if _search_ctx:
        try:
            await _search_ctx.__aexit__(None, None, None)
        except Exception:
            pass
        _search_ctx = None


async def search_web(query: str, count: int = 5) -> dict:
    """Call the search-mcp web_search tool.

    Returns the parsed JSON result or an error dict.
    """
    try:
        session = await _get_search_session()
        result = await session.call_tool("web_search", {"query": query, "count": count})
        # result.content is a list of TextContent blocks
        if result.content and hasattr(result.content[0], "text"):
            return json.loads(result.content[0].text)
        return {"error": "Empty search result", "results": []}
    except Exception as e:
        logger.warning(f"Search MCP call failed: {e}")
        return {"error": str(e), "results": []}


async def search_and_summarize(
    query: str,
    providers: list[dict] = None,
    count: int = 5,
    max_summary_tokens: int = 1024,
) -> dict:
    """Search the web via search-mcp, then summarize results through a cheap model.

    Returns:
        {
            "query": str,
            "result_count": int,
            "raw_text": str,        # concatenated search results
            "summarized": str,      # cheap-model compressed version
            "summarizer": str,      # provider/model used for compression
            "latency_s": float,
        }
    """
    start = time.time()
    raw = await search_web(query, count)

    if "error" in raw and not raw.get("results"):
        return {
            "query": query,
            "result_count": 0,
            "raw_text": "",
            "summarized": f"Search failed: {raw['error']}",
            "summarizer": "none",
            "latency_s": round(time.time() - start, 2),
        }

    # Build raw text from results
    results = raw.get("results", raw if isinstance(raw, list) else [])
    if not isinstance(results, list):
        results = []

    raw_lines = []
    for i, r in enumerate(results, 1):
        if isinstance(r, dict):
            title = r.get("title", "")
            url = r.get("url", "")
            desc = r.get("description", "")
            raw_lines.append(f"[{i}] {title}\n    {url}\n    {desc}")
    raw_text = "\n\n".join(raw_lines)

    # Summarize through cheap model
    sr = await summarize_text(
        raw_text,
        instruction=(
            f"Summarize these web search results for the query: '{query}'. "
            "Preserve the most relevant facts, URLs, and key findings. "
            "Be concise but complete. Output only the summary."
        ),
        providers=providers,
        max_tokens=max_summary_tokens,
    )

    total_latency = time.time() - start

    # Log the search turn
    conversation_log.log_web_search(
        query=query,
        result_count=len(results),
        raw_length=len(raw_text),
        summarized_length=len(sr.full),
        summarizer_provider=sr.provider,
        summarizer_model=sr.model,
        latency_s=total_latency,
    )

    return {
        "query": query,
        "result_count": len(results),
        "raw_text": raw_text,
        "summarized": sr.full,
        "summarizer": f"{sr.provider}/{sr.model}",
        "latency_s": round(total_latency, 2),
    }


# ---------------------------------------------------------------------------
# TaskStore
# ---------------------------------------------------------------------------

class TaskStore:
    """In-memory task store with JSONL checkpoint for replay on restart."""

    def __init__(self, max_tasks: int = 200):
        self._tasks: dict[str, Task] = {}
        self._max_tasks = max_tasks

    def create(self, description: str, category: str, steps: list[dict] = None,
               metadata: dict = None) -> Task:
        self._evict_if_needed()
        task_id = f"task-{uuid.uuid4().hex[:12]}"
        task = Task(
            id=task_id,
            description=description,
            category=category,
            metadata=metadata or {},
        )
        if steps:
            for i, step_def in enumerate(steps):
                step = TaskStep(
                    id=f"{task_id}-s{i}",
                    description=step_def.get("description", f"Step {i + 1}"),
                    provider=step_def.get("provider"),
                    model=step_def.get("model"),
                )
                task.steps.append(step)
        self._tasks[task_id] = task
        _append_checkpoint({"event": "create", "task_id": task_id,
                            "description": description, "category": category,
                            "step_count": len(task.steps)})
        logger.info(f"Task created: {task_id} ({description})")
        return task

    def get(self, task_id: str) -> Optional[Task]:
        return self._tasks.get(task_id)

    def list_active(self) -> list[Task]:
        return [t for t in self._tasks.values()
                if t.status in (TaskStatus.PENDING, TaskStatus.RUNNING)]

    def list_all(self) -> list[Task]:
        return list(self._tasks.values())

    def start(self, task_id: str) -> Optional[Task]:
        task = self._tasks.get(task_id)
        if task and task.status == TaskStatus.PENDING:
            task.status = TaskStatus.RUNNING
            task.started_at = time.time()
            _append_checkpoint({"event": "start", "task_id": task_id})
        return task

    def complete(self, task_id: str, result: str, summary: str = None) -> Optional[Task]:
        task = self._tasks.get(task_id)
        if task and task.status == TaskStatus.RUNNING:
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.summary = summary or _make_summary(result)
            task.finished_at = time.time()
            _append_checkpoint({"event": "complete", "task_id": task_id,
                                "summary": task.summary,
                                "latency_s": round(task.finished_at - task.started_at, 2)})
            logger.info(f"Task completed: {task_id} ({task.finished_at - task.started_at:.1f}s)")
        return task

    def fail(self, task_id: str, error: str) -> Optional[Task]:
        task = self._tasks.get(task_id)
        if task and task.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
            task.status = TaskStatus.FAILED
            task.error = error
            task.finished_at = time.time()
            _append_checkpoint({"event": "fail", "task_id": task_id, "error": error})
            logger.warning(f"Task failed: {task_id} - {error}")
        return task

    def start_step(self, task_id: str, step_id: str) -> Optional[TaskStep]:
        task = self._tasks.get(task_id)
        if not task:
            return None
        for step in task.steps:
            if step.id == step_id and step.status == TaskStatus.PENDING:
                step.status = TaskStatus.RUNNING
                step.started_at = time.time()
                return step
        return None

    def complete_step(self, task_id: str, step_id: str,
                      result: str, summary: str = None) -> Optional[TaskStep]:
        task = self._tasks.get(task_id)
        if not task:
            return None
        for step in task.steps:
            if step.id == step_id and step.status == TaskStatus.RUNNING:
                step.status = TaskStatus.COMPLETED
                step.result = result
                step.summary = summary or _make_summary(result)
                step.finished_at = time.time()
                step.latency_s = step.finished_at - step.started_at
                _append_checkpoint({"event": "step_complete", "task_id": task_id,
                                    "step_id": step_id, "summary": step.summary,
                                    "provider": step.provider, "model": step.model,
                                    "latency_s": round(step.latency_s, 2)})
                return step
        return None

    def fail_step(self, task_id: str, step_id: str, error: str,
                  provider: str = None, model: str = None) -> Optional[TaskStep]:
        task = self._tasks.get(task_id)
        if not task:
            return None
        for step in task.steps:
            if step.id == step_id and step.status == TaskStatus.RUNNING:
                step.status = TaskStatus.FAILED
                step.error = error
                step.finished_at = time.time()
                step.latency_s = step.finished_at - step.started_at
                if provider:
                    step.provider = provider
                if model:
                    step.model = model
                _append_checkpoint({"event": "step_fail", "task_id": task_id,
                                    "step_id": step_id, "error": error})
                return step
        return None

    def _evict_if_needed(self):
        if len(self._tasks) < self._max_tasks:
            return
        completed = sorted(
            [t for t in self._tasks.values()
             if t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)],
            key=lambda t: t.created_at,
        )
        while len(self._tasks) >= self._max_tasks and completed:
            old = completed.pop(0)
            del self._tasks[old.id]


# Singleton store
task_store = TaskStore()


# ---------------------------------------------------------------------------
# Stats-recording wrapper for direct calls
# ---------------------------------------------------------------------------

async def call_with_stats(
    call_fn,
    provider_name: str,
    system_prompt: Optional[str],
    messages: list,
    model: str,
    max_tokens: int,
    **kwargs,
) -> StepResult:
    """Wrap a provider call_*_direct function with latency/success recording.

    Returns a StepResult with summary + full result for aggregation.
    """
    start = time.time()
    try:
        result = await call_fn(system_prompt, messages, model, max_tokens, **kwargs)
        latency = time.time() - start
        _get_proxy().request_stats.record(model, "direct", latency, True)
        logger.debug(f"[stats] {provider_name}/{model} OK {latency:.2f}s")
        return StepResult(
            summary=_make_summary(result),
            full=result,
            provider=provider_name,
            model=model,
            latency_s=latency,
        )
    except Exception:
        latency = time.time() - start
        _get_proxy().request_stats.record(model, "direct", latency, False)
        logger.debug(f"[stats] {provider_name}/{model} FAIL {latency:.2f}s")
        raise


# ---------------------------------------------------------------------------
# Provider fallback executor
# ---------------------------------------------------------------------------

async def call_with_fallback(
    providers: list[dict],
    system_prompt: Optional[str],
    messages: list,
    model: Optional[str],
    max_tokens: int,
    **kwargs,
) -> StepResult:
    """Try providers in order until one succeeds.

    Returns StepResult with structured summary + full result.
    Raises the last exception if all providers fail.
    """
    last_error = None
    for provider in providers:
        use_model = model or provider["default_model"]
        call_kwargs = kwargs if provider.get("accepts_extra") else {}
        try:
            return await call_with_stats(
                provider["call_fn"],
                provider["name"],
                system_prompt,
                messages,
                use_model,
                max_tokens,
                **call_kwargs,
            )
        except Exception as e:
            last_error = e
            logger.warning(f"Provider {provider['name']} failed: {e}")
            continue
    raise last_error


# ---------------------------------------------------------------------------
# Orchestrator: multi-step task execution
# ---------------------------------------------------------------------------

class Orchestrator:
    """Executes multi-step tasks across providers with fallback and state tracking.

    Results are structured with summaries for efficient aggregation:
    - Each step produces a StepResult with summary (short) and full (complete)
    - The orchestrator pipes summaries between steps to avoid context collapse
    - Full results are available via task resources for deep inspection
    """

    def __init__(self, store: TaskStore):
        self.store = store

    async def execute_single(
        self,
        description: str,
        category: str,
        message: str,
        providers: list[dict],
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        **kwargs,
    ) -> Task:
        """Execute a single-step task with automatic provider fallback."""
        task = self.store.create(
            description=description,
            category=category,
            steps=[{"description": description}],
        )
        self.store.start(task.id)
        step = task.steps[0]
        self.store.start_step(task.id, step.id)

        try:
            sr = await call_with_fallback(
                providers, system_prompt, [{"role": "user", "content": message}],
                model, max_tokens, **kwargs,
            )
            step.provider = sr.provider
            step.model = sr.model
            self.store.complete_step(task.id, step.id, sr.full, sr.summary)
            conversation_log.log_sub_agent_turn(task.id, 0, sr.provider, sr.model,
                                                sr.summary, sr.latency_s)
            self.store.complete(task.id, sr.full, sr.summary)
        except Exception as e:
            self.store.fail_step(task.id, step.id, str(e))
            self.store.fail(task.id, str(e))

        return task

    async def execute_sequential(
        self,
        description: str,
        category: str,
        steps: list[dict],
        providers: list[dict],
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        progress_callback=None,
        **kwargs,
    ) -> Task:
        """Execute multiple steps sequentially, piping summaries forward.

        Each step dict should have:
          - description: str
          - message: str (the prompt for this step)
          - provider: Optional[str] (specific provider name)
          - model: Optional[str]

        Previous step SUMMARIES (not full results) are appended to context
        to avoid context window collapse on long multi-step workflows.
        Full results are stored and queryable via task resources.
        """
        step_defs = [{"description": s.get("description", f"Step {i+1}"),
                       "provider": s.get("provider"),
                       "model": s.get("model")} for i, s in enumerate(steps)]

        task = self.store.create(
            description=description,
            category=category,
            steps=step_defs,
        )
        self.store.start(task.id)

        step_results: list[StepResult] = []

        for i, (step_def, step) in enumerate(zip(steps, task.steps)):
            if progress_callback:
                await progress_callback(i, len(steps),
                                        f"Running step {i+1}/{len(steps)}: {step.description}")

            self.store.start_step(task.id, step.id)

            # Build step prompt with summaries from previous steps
            step_message = step_def.get("message", step_def.get("description", ""))
            if step_results:
                context_block = "\n\n---\nPrevious step summaries:\n" + "\n".join(
                    f"- Step {j+1} ({sr.provider}): {sr.summary}"
                    for j, sr in enumerate(step_results)
                )
                step_message = step_message + context_block

            # Prefer the specified provider, fall back to others
            step_providers = providers
            if step_def.get("provider"):
                specific = [p for p in providers if p["name"] == step_def["provider"]]
                if specific:
                    step_providers = specific + [p for p in providers if p["name"] != step_def["provider"]]

            try:
                sr = await call_with_fallback(
                    step_providers,
                    system_prompt,
                    [{"role": "user", "content": step_message}],
                    step_def.get("model"),
                    max_tokens,
                    **kwargs,
                )
                step.provider = sr.provider
                step.model = sr.model
                self.store.complete_step(task.id, step.id, sr.full, sr.summary)
                conversation_log.log_sub_agent_turn(task.id, i, sr.provider, sr.model,
                                                    sr.summary, sr.latency_s)
                step_results.append(sr)
            except Exception as e:
                self.store.fail_step(task.id, step.id, str(e))
                self.store.fail(task.id, f"Step {i+1} failed: {e}")
                return task

        if progress_callback:
            await progress_callback(len(steps), len(steps), "All steps completed")

        # Build combined result with full content + overall summary from step summaries
        combined = "\n\n---\n\n".join(
            f"## Step {i+1}: {task.steps[i].description}\n\n{sr.full}"
            for i, sr in enumerate(step_results)
        )
        overall_summary = " | ".join(
            f"Step {i+1}: {sr.summary[:100]}" for i, sr in enumerate(step_results)
        )
        self.store.complete(task.id, combined, overall_summary)
        return task

    async def execute_parallel(
        self,
        description: str,
        category: str,
        steps: list[dict],
        providers: list[dict],
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        progress_callback=None,
        **kwargs,
    ) -> Task:
        """Execute multiple steps in parallel across available providers.

        All steps run concurrently. Task completes when all steps finish.
        Individual step failures don't block other steps.
        """
        step_defs = [{"description": s.get("description", f"Step {i+1}"),
                       "provider": s.get("provider"),
                       "model": s.get("model")} for i, s in enumerate(steps)]

        task = self.store.create(
            description=description,
            category=category,
            steps=step_defs,
        )
        self.store.start(task.id)

        async def _run_step(i: int, step_def: dict, step: TaskStep) -> Optional[StepResult]:
            self.store.start_step(task.id, step.id)
            step_message = step_def.get("message", step_def.get("description", ""))

            step_providers = providers
            if step_def.get("provider"):
                specific = [p for p in providers if p["name"] == step_def["provider"]]
                if specific:
                    step_providers = specific + [p for p in providers if p["name"] != step_def["provider"]]

            try:
                sr = await call_with_fallback(
                    step_providers,
                    system_prompt,
                    [{"role": "user", "content": step_message}],
                    step_def.get("model"),
                    max_tokens,
                    **kwargs,
                )
                step.provider = sr.provider
                step.model = sr.model
                self.store.complete_step(task.id, step.id, sr.full, sr.summary)
                conversation_log.log_sub_agent_turn(task.id, i, sr.provider, sr.model,
                                                    sr.summary, sr.latency_s)
                return sr
            except Exception as e:
                self.store.fail_step(task.id, step.id, str(e))
                return None

        # Run all steps concurrently
        coros = [_run_step(i, sd, s) for i, (sd, s) in enumerate(zip(steps, task.steps))]
        results = await asyncio.gather(*coros, return_exceptions=True)

        if progress_callback:
            await progress_callback(len(steps), len(steps), "All parallel steps completed")

        successful = [(i, r) for i, r in enumerate(results)
                      if isinstance(r, StepResult)]
        failed_idxs = [i for i, r in enumerate(results) if not isinstance(r, StepResult)]

        if not successful:
            self.store.fail(task.id, "All parallel steps failed")
            return task

        combined = "\n\n---\n\n".join(
            f"## Step {i+1}: {task.steps[i].description}\n\n{sr.full}"
            for i, sr in successful
        )
        if failed_idxs:
            combined += f"\n\n---\n\n**Note:** Steps {', '.join(str(i+1) for i in failed_idxs)} failed."

        overall_summary = " | ".join(
            f"Step {i+1}: {sr.summary[:100]}" for i, sr in successful
        )
        if failed_idxs:
            overall_summary += f" | Failed: steps {', '.join(str(i+1) for i in failed_idxs)}"

        self.store.complete(task.id, combined, overall_summary)
        return task


# Singleton orchestrator
orchestrator = Orchestrator(task_store)
