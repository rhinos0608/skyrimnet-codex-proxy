"""
MCP Server mode for the Codex Proxy.

Exposes healthy LLM providers as MCP tools, separated into two categories:
  - Chat tools: API-based providers for questions & research (non-agentic tasks)
  - CLI tools: CLI-based providers for agentic tasks on the local system

Orchestrator mode adds:
  - Task state management (create, track, query multi-step workflows)
  - Provider fallback (auto-retry with next healthy provider on failure)
  - Sequential and parallel multi-step task execution
  - MCP resources for querying task state
  - Stats recording for all direct calls

Run:  python proxy.py --mode mcp
      (starts MCP SSE server on port 8432)
"""

import asyncio
import gc
import json
import logging
from contextlib import asynccontextmanager
from typing import Optional

from mcp.server import FastMCP
from mcp.server.fastmcp import Context
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger("mcp_server")

# ---------------------------------------------------------------------------
# Import everything we need from the main proxy module.  proxy.py declares
# auth caches, API keys, call functions, and model-detection helpers at
# module level -- they're all available after `import proxy`.
# ---------------------------------------------------------------------------
import proxy
from orchestrator import (
    task_store, orchestrator, call_with_stats, call_with_fallback,
    conversation_log, summarize_text, search_and_summarize, close_search_session,
)

# ---------------------------------------------------------------------------
# Provider registry -- defines every provider, its health check, category,
# and the call function used to invoke it.
# ---------------------------------------------------------------------------

CHAT = "chat"  # API-based, for questions & research
CLI = "cli"    # CLI-based, for agentic local tasks


def _is_claude_healthy() -> bool:
    return proxy.auth.is_ready


def _is_codex_healthy() -> bool:
    return proxy.codex_auth.is_ready and proxy.CODEX_PATH is not None


def _is_openrouter_healthy() -> bool:
    return proxy.openrouter_api_key is not None


_ollama_models: list[str] = []
_ollama_healthy: bool = False
_ollama_local_available: bool = False


async def _probe_ollama():
    """Query Ollama for available models. Probes local first, then cloud if configured."""
    global _ollama_models, _ollama_healthy, _ollama_local_available
    import aiohttp
    all_models: list[str] = []
    local_ok = False

    # Always try local first — local models are free and fast
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=3, connect=2)
        ) as s:
            async with s.get("http://localhost:11434/api/tags") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    local_models = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
                    all_models.extend(local_models)
                    local_ok = True
    except Exception:
        pass

    # Also try cloud if key configured
    if proxy.ollama_api_key:
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=3, connect=2)
            ) as s:
                async with s.get("https://ollama.com/v1/models",
                                 headers={"Authorization": f"Bearer {proxy.ollama_api_key}"}) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        cloud_models = [m.get("id", "") for m in data.get("data", []) if m.get("id")]
                        # Add cloud models not already in local list
                        local_set = set(all_models)
                        for m in cloud_models:
                            if m not in local_set:
                                all_models.append(m)
        except Exception:
            pass

    _ollama_models = all_models
    _ollama_local_available = local_ok
    _ollama_healthy = len(all_models) > 0

    # Update PROVIDERS default model
    if _ollama_models:
        for p in PROVIDERS:
            if p["name"] == "ollama":
                p["default_model"] = f"ollama:{_ollama_models[0]}"
                break

    if _ollama_healthy:
        src = "local" if local_ok else "cloud"
        logger.info(f"Ollama probe: {len(all_models)} models ({src})")
    else:
        logger.debug("Ollama probe: no models found")


def _is_ollama_healthy() -> bool:
    return _ollama_healthy


def _get_ollama_default_model() -> str:
    """Return the first available Ollama model, prefixed with 'ollama:'."""
    if _ollama_models:
        return f"ollama:{_ollama_models[0]}"
    return "ollama:llama3.1"  # fallback if probe hasn't run yet


def get_ollama_models() -> list[str]:
    """Return currently known Ollama models (prefixed with 'ollama:')."""
    return [f"ollama:{m}" for m in _ollama_models]


def _is_antigravity_healthy() -> bool:
    return proxy.antigravity_auth.is_ready


def _is_gemini_cli_healthy() -> bool:
    return proxy.gemini_auth.is_ready


def _is_zai_healthy() -> bool:
    return proxy.zai_api_key is not None


def _is_xiaomi_healthy() -> bool:
    return proxy.xiaomi_api_key is not None


def _is_opencode_zen_healthy() -> bool:
    return proxy.opencode_api_key is not None


def _is_opencode_go_healthy() -> bool:
    return proxy.opencode_go_api_key is not None or proxy.opencode_api_key is not None


def _is_qwen_healthy() -> bool:
    return proxy.qwen_auth.is_ready


PROVIDERS = [
    # --- Chat providers (API-based, non-agentic) ---
    {
        "name": "claude",
        "category": CHAT,
        "health_fn": _is_claude_healthy,
        "call_fn": proxy.call_api_direct,
        "default_model": proxy.DEFAULT_MODEL,
        "accepts_extra": False,
        "description": "Claude (Anthropic) — paid with heavy rate limits. Use when deep thinking or complex reasoning is necessary. For lighter tasks, prefer other providers to conserve quota.",
    },
    {
        "name": "openrouter",
        "category": CHAT,
        "health_fn": _is_openrouter_healthy,
        "call_fn": proxy.call_openrouter_direct,
        "default_model": "anthropic/claude-sonnet-4",
        "accepts_extra": True,
        "description": "Send a chat message via OpenRouter. Access hundreds of models (use provider/model format).",
    },
    {
        "name": "ollama",
        "category": CHAT,
        "health_fn": _is_ollama_healthy,
        "call_fn": proxy.call_ollama_direct,
        "default_model": "ollama:llama3.1",
        "accepts_extra": True,
        "description": "Send a chat message to Ollama (local or cloud). Use 'ollama:model' format.",
    },
    {
        "name": "antigravity",
        "category": CHAT,
        "health_fn": _is_antigravity_healthy,
        "call_fn": proxy.call_antigravity_direct,
        "default_model": "gemini-2.5-pro",
        "accepts_extra": False,
        "description": "Antigravity (Google IDE API) — supports Gemini models and Claude endpoints (e.g. antigravity-claude-sonnet). Use Claude endpoints as a Sonnet fallback when main Claude is rate-limited.",
    },
    {
        "name": "zai",
        "category": CHAT,
        "health_fn": _is_zai_healthy,
        "call_fn": proxy.call_zai_direct,
        "default_model": "zai:glm-4.5-air",
        "accepts_extra": True,
        "description": "Send a chat message to Z.AI.",
    },
    {
        "name": "xiaomi",
        "category": CHAT,
        "health_fn": _is_xiaomi_healthy,
        "call_fn": proxy.call_xiaomi_direct,
        "default_model": "xiaomi:mimo-v2-pro",
        "accepts_extra": True,
        "description": "Send a chat message to Xiaomi MiMo models.",
    },
    # --- CLI providers (agentic, local system tasks) ---
    {
        "name": "codex_cli",
        "category": CLI,
        "health_fn": _is_codex_healthy,
        "call_fn": proxy.call_codex_direct,
        "default_model": proxy.DEFAULT_CODEX_MODEL,
        "accepts_extra": False,
        "description": "Codex CLI (OpenAI) — the workhorse. Highest rate limits, strong capability, good for most agentic tasks. Prefer this as your default CLI provider.",
    },
    {
        "name": "gemini_cli",
        "category": CLI,
        "health_fn": _is_gemini_cli_healthy,
        "call_fn": proxy.call_gemini_direct,
        "default_model": "gcli-gemini-2.5-pro",
        "accepts_extra": False,
        "description": "Gemini CLI — unreliable rate limits and serving issues. Avoid as primary. Use only when Codex and Claude are unavailable.",
    },
    {
        "name": "qwen_cli",
        "category": CLI,
        "health_fn": _is_qwen_healthy,
        "call_fn": proxy.call_qwen_direct,
        "default_model": "qwen:coder-model",
        "accepts_extra": True,
        "description": "Qwen Code CLI — free but unreliable. Good for low-stakes tasks or when all paid providers are rate-limited.",
    },
    {
        "name": "opencode_zen",
        "category": CLI,
        "health_fn": _is_opencode_zen_healthy,
        "call_fn": proxy.call_opencode_direct,
        "default_model": "opencode:default",
        "accepts_extra": True,
        "description": "Execute an agentic task via OpenCode Zen CLI. Code-focused agent for local development.",
    },
    {
        "name": "opencode_go",
        "category": CLI,
        "health_fn": _is_opencode_go_healthy,
        "call_fn": proxy.call_opencode_direct,
        "default_model": "opencode-go:default",
        "accepts_extra": True,
        "description": "Execute an agentic task via OpenCode Go CLI. Code-focused agent for local development.",
    },
]


# ---------------------------------------------------------------------------
# Health helpers
# ---------------------------------------------------------------------------

def get_healthy_providers() -> list[dict]:
    return [p for p in PROVIDERS if p["health_fn"]()]


def get_healthy_chat_providers() -> list[dict]:
    return [p for p in PROVIDERS if p["category"] == CHAT and p["health_fn"]()]


def get_healthy_cli_providers() -> list[dict]:
    return [p for p in PROVIDERS if p["category"] == CLI and p["health_fn"]()]


# ---------------------------------------------------------------------------
# Annotation presets
# ---------------------------------------------------------------------------

_READONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)
_CHAT_ANN = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=True)
_CLI_ANN = ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=True)


# ---------------------------------------------------------------------------
# Logged call helper -- wraps a direct provider call with conversation logging
# ---------------------------------------------------------------------------

import time as _time

async def _logged_call(tool_name: str, provider_name: str, model: str,
                       call_fn, system_prompt, messages: list,
                       max_tokens: int, ctx=None, **kwargs) -> str:
    """Call a provider, log the user_input and model_output turns, return result."""
    user_text = messages[0]["content"] if messages else ""
    conversation_log.log_user_input(tool_name, user_text, provider_name, model)

    if ctx:
        await ctx.report_progress(0, 1)

    start = _time.time()
    result = await call_fn(system_prompt, messages, model, max_tokens, **kwargs)
    latency = _time.time() - start

    from orchestrator import _make_summary
    conversation_log.log_model_output(
        tool_name, provider_name, model,
        _make_summary(result), len(result), latency,
    )

    if ctx:
        await ctx.report_progress(1, 1)
    return result


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def mcp_lifespan(server: FastMCP):
    """Initialize proxy auth/sessions before the MCP server starts accepting connections."""
    claude_runner = None
    codex_runner = None

    try:
        claude_runner = await proxy.start_interceptor(proxy.MCP_INTERCEPTOR_PORT)
    except Exception as e:
        logger.warning(f"Claude auth capture failed: {e}. Claude provider unavailable.")

    try:
        codex_runner = await proxy.start_codex_interceptor()
    except Exception as e:
        logger.warning(f"Codex auth capture failed: {e}. Codex provider unavailable.")

    # Ensure a shared session exists even if Claude auth fails.
    # All providers fall back to auth.session for connection pooling.
    if not proxy.auth.session:
        proxy.auth.session = proxy.create_session()

    try:
        proxy.ollama_session = proxy.create_session()
    except Exception as e:
        logger.warning(f"Ollama session creation failed: {e}")

    try:
        await proxy.load_antigravity_auth()
    except Exception as e:
        logger.warning(f"Antigravity auth load failed: {e}")

    try:
        await proxy.load_gemini_auth()
    except Exception as e:
        logger.warning(f"Gemini auth load failed: {e}")

    try:
        proxy.load_opencode_key()
    except Exception as e:
        logger.warning(f"OpenCode key load failed: {e}")

    try:
        await proxy.load_qwen_auth()
    except Exception as e:
        logger.warning(f"Qwen auth load failed: {e}")

    refresh_task = None
    if proxy.auth.is_ready and proxy.CLAUDE_PATH:
        refresh_task = asyncio.create_task(proxy._claude_auth_refresh_loop())
        logger.info("Claude auth auto-refresh enabled (every 45 min)")

    # Probe Ollama for available models
    await _probe_ollama()

    # Background task: re-probe Ollama every 60s (models can be pulled/removed)
    async def _ollama_probe_loop():
        while True:
            await asyncio.sleep(60)
            try:
                await _probe_ollama()
            except Exception:
                pass

    ollama_probe_task = asyncio.create_task(_ollama_probe_loop())

    gc.freeze()

    healthy = get_healthy_providers()
    chat_p = [p for p in healthy if p["category"] == CHAT]
    cli_p = [p for p in healthy if p["category"] == CLI]
    logger.info(f"MCP server ready -- Chat ({len(chat_p)}): {', '.join(p['name'] for p in chat_p)}")
    logger.info(f"                    CLI  ({len(cli_p)}): {', '.join(p['name'] for p in cli_p)}")
    if _ollama_models:
        logger.info(f"  Ollama models: {', '.join(_ollama_models[:10])}")

    yield

    # Shutdown
    ollama_probe_task.cancel()
    if refresh_task and not refresh_task.done():
        refresh_task.cancel()
    if proxy.auth.session:
        await proxy.auth.session.close()
    if proxy.ollama_session:
        await proxy.ollama_session.close()
    if proxy.codex_auth.session:
        await proxy.codex_auth.session.close()
    if proxy.gemini_auth.session:
        await proxy.gemini_auth.session.close()
    if proxy.qwen_auth.session:
        await proxy.qwen_auth.session.close()
    for account in proxy.antigravity_auth.accounts:
        if account.session:
            await account.session.close()
    if proxy.antigravity_auth._legacy_account.session:
        await proxy.antigravity_auth._legacy_account.session.close()
    await close_search_session()
    if claude_runner:
        await claude_runner.cleanup()
    if codex_runner:
        await codex_runner.cleanup()


# ---------------------------------------------------------------------------
# Smart Ollama routing — prefers local when model exists locally
# ---------------------------------------------------------------------------

async def _call_ollama_smart(system_prompt, messages, model, max_tokens, **kwargs):
    """Route Ollama calls to local when the model is available locally, cloud otherwise.

    proxy.call_ollama_direct always goes to cloud when ollama_api_key is set.
    This wrapper checks if the model exists in our local probe results first.
    """
    api_model = model[len("ollama:"):] if model.lower().startswith("ollama:") else model

    # If local Ollama is up and has this model, call local directly
    if _ollama_local_available and api_model in _ollama_models:
        import aiohttp
        endpoint = "http://localhost:11434/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": api_model,
            "messages": proxy.build_oai_messages(system_prompt, messages),
            "max_tokens": max_tokens,
        }
        payload.update({k: v for k, v in kwargs.items() if v is not None})

        session = proxy.ollama_session or proxy.auth.session or proxy.create_session()
        async with session.post(endpoint, json=payload, headers=headers) as resp:
            if resp.status != 200:
                error = await resp.text()
                raise Exception(f"Ollama local {resp.status}: {error[:200]}")
            data = await resp.json()
            choices = data.get("choices", [])
            if not choices:
                raise Exception("Ollama returned empty choices")
            return choices[0].get("message", {}).get("content", "")

    # Fall through to standard proxy routing (cloud or local based on key)
    return await proxy.call_ollama_direct(system_prompt, messages, model, max_tokens, **kwargs)


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------

def _get_tailscale_fqdn() -> Optional[str]:
    """Get the Tailscale FQDN for this machine, if available."""
    import shutil
    import subprocess
    if not shutil.which("tailscale"):
        return None
    try:
        result = subprocess.run(
            ["tailscale", "status", "--json"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            ts_status = json.loads(result.stdout)
            return ts_status.get("Self", {}).get("DNSName", "").rstrip(".") or None
    except Exception:
        pass
    return None


def create_mcp_server() -> FastMCP:
    # Build transport security: allow localhost + Tailscale FQDN
    allowed_hosts = ["127.0.0.1:*", "localhost:*", "[::1]:*"]
    allowed_origins = ["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"]

    ts_fqdn = _get_tailscale_fqdn()
    if ts_fqdn:
        allowed_hosts.append(f"{ts_fqdn}:*")
        allowed_origins.append(f"https://{ts_fqdn}:*")
        allowed_origins.append(f"http://{ts_fqdn}:*")
        logger.info(f"MCP transport security: allowing Tailscale host {ts_fqdn}")

    transport_security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=allowed_hosts,
        allowed_origins=allowed_origins,
    )

    # Bind to 127.0.0.1 — Tailscale serve handles external HTTPS and
    # reverse-proxies to localhost.  Binding 0.0.0.0 conflicts with
    # Tailscale's own listener on the same port.
    server = FastMCP(
        name="codex-proxy",
        host="127.0.0.1",
        port=8432,
        lifespan=mcp_lifespan,
        transport_security=transport_security,
        instructions=(
            "Codex Proxy MCP server -- multi-provider LLM gateway with orchestration.\n\n"
            "CHAT tools (chat_claude, chat_openrouter, chat_ollama, chat_antigravity, "
            "chat_zai, chat_xiaomi) are for questions, research, and analysis.\n\n"
            "CLI tool selection guide (agentic tasks):\n"
            "- cli_codex: WORKHORSE. Highest rate limits, strong capability. Use as default.\n"
            "- cli_qwen: Free but unreliable. Use for low-stakes tasks or as fallback.\n"
            "- cli_gemini: Unreliable rate limits and serving. Avoid as primary.\n"
            "- cli_opencode_zen / cli_opencode_go: Alternative code-focused agents.\n"
            "For heavy thinking, use chat_claude (paid, heavy rate limits) or "
            "chat_antigravity with a Claude model (e.g. antigravity-claude-sonnet) as a Sonnet fallback.\n\n"
            "ORCHESTRATION tools:\n"
            "- orchestrate_sequential: Run multi-step workflows with context piped between steps\n"
            "- orchestrate_parallel: Run independent tasks concurrently across providers\n"
            "- task_status: Check the state of a running/completed task\n"
            "- task_list: List all active tasks\n\n"
            "SEARCH: web_search queries the web via search-mcp (Brave/SearXNG). Results "
            "are auto-summarized through the cheapest available model before returning.\n\n"
            "The unified 'chat' and 'cli' tools auto-select the best healthy provider "
            "with automatic fallback on failure. All turns are logged to "
            ".conversation-log.jsonl. Query task://{task_id} resources for detailed results."
        ),
    )

    # -------------------------------------------------------------------
    # Health endpoint (custom route, not a tool)
    # -------------------------------------------------------------------

    @server.custom_route("/health", methods=["GET"])
    async def health_check(request: Request) -> JSONResponse:
        result = {}
        for p in PROVIDERS:
            result[p["name"]] = {
                "healthy": p["health_fn"](),
                "category": p["category"],
                "default_model": p["default_model"],
            }
        return JSONResponse(result)

    # -------------------------------------------------------------------
    # Meta tools: list available providers
    # -------------------------------------------------------------------

    @server.tool(annotations=_READONLY)
    async def list_chat_providers(ctx: Context = None) -> str:
        """List all currently healthy chat providers for questions and research (non-agentic tasks)."""
        providers = get_healthy_chat_providers()
        if not providers:
            return "No chat providers are currently available."
        lines = [f"- **{p['name']}** (model: `{p['default_model']}`): {p['description']}" for p in providers]
        return "Available chat providers:\n" + "\n".join(lines)

    @server.tool(annotations=_READONLY)
    async def list_cli_providers(ctx: Context = None) -> str:
        """List all currently healthy CLI providers for agentic tasks on the local system."""
        providers = get_healthy_cli_providers()
        if not providers:
            return "No CLI providers are currently available."
        lines = [f"- **{p['name']}** (model: `{p['default_model']}`): {p['description']}" for p in providers]
        return "Available CLI providers:\n" + "\n".join(lines)

    @server.tool(annotations=_READONLY)
    async def list_all_providers(ctx: Context = None) -> str:
        """List all currently healthy providers grouped by category (chat vs CLI)."""
        chat = get_healthy_chat_providers()
        cli = get_healthy_cli_providers()
        lines = []
        if chat:
            lines.append("## Chat Providers (questions & research)")
            for p in chat:
                lines.append(f"- **{p['name']}** (`{p['default_model']}`): {p['description']}")
        else:
            lines.append("## Chat Providers\nNone available.")
        lines.append("")
        if cli:
            lines.append("## CLI Providers (agentic local tasks)")
            for p in cli:
                lines.append(f"- **{p['name']}** (`{p['default_model']}`): {p['description']}")
        else:
            lines.append("## CLI Providers\nNone available.")
        return "\n".join(lines)

    @server.tool(annotations=_READONLY)
    async def list_ollama_models(ctx: Context = None) -> str:
        """List models currently available on the connected Ollama instance.
        Re-probes Ollama live to get the freshest list."""
        await _probe_ollama()
        if not _ollama_models:
            return "No Ollama models available. Is Ollama running?"
        lines = [f"- `ollama:{m}`" for m in _ollama_models]
        return f"Ollama models ({len(_ollama_models)}):\n" + "\n".join(lines)

    # -------------------------------------------------------------------
    # Chat provider tools
    # -------------------------------------------------------------------

    @server.tool(annotations=_CHAT_ANN)
    async def chat_claude(message: str, system_prompt: Optional[str] = None,
                          model: Optional[str] = None, max_tokens: int = 4096,
                          ctx: Context = None) -> str:
        """Claude (Anthropic) — paid with heavy rate limits.
        Use when deep thinking or complex reasoning is truly necessary. For lighter tasks, prefer other providers to conserve quota.
        This is a CHAT provider -- use for non-agentic tasks like research and Q&A."""
        if not _is_claude_healthy():
            return "Error: Claude provider is not currently available (auth not ready)."
        m = model or proxy.DEFAULT_MODEL
        try:
            return await _logged_call("chat_claude", "claude", m, proxy.call_api_direct,
                                      system_prompt, [{"role": "user", "content": message}], max_tokens, ctx)
        except Exception as e:
            return f"Error calling Claude: {e}"

    @server.tool(annotations=_CHAT_ANN)
    async def chat_openrouter(message: str, model: str = "anthropic/claude-sonnet-4",
                              system_prompt: Optional[str] = None, max_tokens: int = 4096,
                              temperature: Optional[float] = None, ctx: Context = None) -> str:
        """Send a chat message via OpenRouter to access hundreds of models.
        Use provider/model format (e.g. 'anthropic/claude-sonnet-4', 'google/gemini-2.5-pro').
        This is a CHAT provider -- use for non-agentic tasks like research and Q&A."""
        if not _is_openrouter_healthy():
            return "Error: OpenRouter provider is not available (API key not configured)."
        kw = {"temperature": temperature} if temperature is not None else {}
        try:
            return await _logged_call("chat_openrouter", "openrouter", model, proxy.call_openrouter_direct,
                                      system_prompt, [{"role": "user", "content": message}], max_tokens, ctx, **kw)
        except Exception as e:
            return f"Error calling OpenRouter: {e}"

    @server.tool(annotations=_CHAT_ANN)
    async def chat_ollama(message: str, model: Optional[str] = None,
                          system_prompt: Optional[str] = None, max_tokens: int = 4096,
                          temperature: Optional[float] = None, ctx: Context = None) -> str:
        """Send a chat message to Ollama (local or cloud). Use 'ollama:model' format.
        If model is omitted, uses the first model available on the Ollama instance.
        Prefers local Ollama when the model is available locally.
        This is a CHAT provider -- use for non-agentic tasks like research and Q&A."""
        if not _is_ollama_healthy():
            return "Error: Ollama is not reachable or has no models. Check that Ollama is running."
        m = model or _get_ollama_default_model()
        kw = {"temperature": temperature} if temperature is not None else {}
        try:
            return await _logged_call("chat_ollama", "ollama", m, _call_ollama_smart,
                                      system_prompt, [{"role": "user", "content": message}], max_tokens, ctx, **kw)
        except Exception as e:
            return f"Error calling Ollama: {e}"

    @server.tool(annotations=_CHAT_ANN)
    async def chat_antigravity(message: str, model: str = "gemini-2.5-pro",
                               system_prompt: Optional[str] = None, max_tokens: int = 4096,
                               ctx: Context = None) -> str:
        """Antigravity (Google IDE API) — supports Gemini models and Claude endpoints.
        Use Claude models (e.g. 'antigravity-claude-sonnet') as a Sonnet fallback when main Claude is rate-limited.
        This is a CHAT provider -- use for non-agentic tasks like research and Q&A."""
        if not _is_antigravity_healthy():
            return "Error: Antigravity provider is not available (auth not ready)."
        try:
            return await _logged_call("chat_antigravity", "antigravity", model, proxy.call_antigravity_direct,
                                      system_prompt, [{"role": "user", "content": message}], max_tokens, ctx)
        except Exception as e:
            return f"Error calling Antigravity: {e}"

    @server.tool(annotations=_CHAT_ANN)
    async def chat_zai(message: str, model: str = "zai:glm-4.5-air",
                       system_prompt: Optional[str] = None, max_tokens: int = 4096,
                       temperature: Optional[float] = None, ctx: Context = None) -> str:
        """Send a chat message to Z.AI.
        This is a CHAT provider -- use for non-agentic tasks like research and Q&A."""
        if not _is_zai_healthy():
            return "Error: Z.AI provider is not available (API key not configured)."
        kw = {"temperature": temperature} if temperature is not None else {}
        try:
            return await _logged_call("chat_zai", "zai", model, proxy.call_zai_direct,
                                      system_prompt, [{"role": "user", "content": message}], max_tokens, ctx, **kw)
        except Exception as e:
            return f"Error calling Z.AI: {e}"

    @server.tool(annotations=_CHAT_ANN)
    async def chat_xiaomi(message: str, model: str = "xiaomi:mimo-v2-pro",
                          system_prompt: Optional[str] = None, max_tokens: int = 4096,
                          temperature: Optional[float] = None, ctx: Context = None) -> str:
        """Send a chat message to Xiaomi MiMo models.
        This is a CHAT provider -- use for non-agentic tasks like research and Q&A."""
        if not _is_xiaomi_healthy():
            return "Error: Xiaomi provider is not available (API key not configured)."
        kw = {"temperature": temperature} if temperature is not None else {}
        try:
            return await _logged_call("chat_xiaomi", "xiaomi", model, proxy.call_xiaomi_direct,
                                      system_prompt, [{"role": "user", "content": message}], max_tokens, ctx, **kw)
        except Exception as e:
            return f"Error calling Xiaomi: {e}"

    # -------------------------------------------------------------------
    # CLI provider tools (agentic, local system)
    # -------------------------------------------------------------------

    @server.tool(annotations=_CLI_ANN)
    async def cli_codex(task: str, system_prompt: Optional[str] = None,
                        model: Optional[str] = None, max_tokens: int = 4096,
                        ctx: Context = None) -> str:
        """Codex CLI (OpenAI) — the workhorse CLI provider.
        Highest rate limits and strong all-around capability. Use as your default for most agentic tasks.
        Can run code, edit files, and perform complex multi-step operations.
        This is a CLI provider -- use for agentic tasks that interact with the local filesystem."""
        if not _is_codex_healthy():
            return "Error: Codex CLI provider is not available (not installed or auth not ready)."
        m = model or proxy.DEFAULT_CODEX_MODEL
        try:
            if ctx:
                await ctx.info("Starting Codex CLI task...")
            return await _logged_call("cli_codex", "codex_cli", m, proxy.call_codex_direct,
                                      system_prompt, [{"role": "user", "content": task}], max_tokens, ctx)
        except Exception as e:
            return f"Error calling Codex CLI: {e}"

    @server.tool(annotations=_CLI_ANN)
    async def cli_gemini(task: str, system_prompt: Optional[str] = None,
                         model: str = "gcli-gemini-2.5-pro", max_tokens: int = 4096,
                         ctx: Context = None) -> str:
        """Gemini CLI — unreliable rate limits and serving issues.
        Avoid as a primary provider. Use only when Codex and Claude are both unavailable.
        This is a CLI provider -- use for agentic tasks that interact with the local filesystem."""
        if not _is_gemini_cli_healthy():
            return "Error: Gemini CLI provider is not available (auth not ready)."
        try:
            if ctx:
                await ctx.info("Starting Gemini CLI task...")
            return await _logged_call("cli_gemini", "gemini_cli", model, proxy.call_gemini_direct,
                                      system_prompt, [{"role": "user", "content": task}], max_tokens, ctx)
        except Exception as e:
            return f"Error calling Gemini CLI: {e}"

    @server.tool(annotations=_CLI_ANN)
    async def cli_qwen(task: str, system_prompt: Optional[str] = None,
                       model: str = "qwen:coder-model", max_tokens: int = 4096,
                       temperature: Optional[float] = None, ctx: Context = None) -> str:
        """Qwen Code CLI — free but unreliable.
        Good for low-stakes tasks or when all paid providers are rate-limited. Do not rely on for critical work.
        This is a CLI provider -- use for agentic tasks that interact with the local filesystem."""
        if not _is_qwen_healthy():
            return "Error: Qwen CLI provider is not available (auth not ready)."
        kw = {"temperature": temperature} if temperature is not None else {}
        try:
            if ctx:
                await ctx.info("Starting Qwen CLI task...")
            return await _logged_call("cli_qwen", "qwen_cli", model, proxy.call_qwen_direct,
                                      system_prompt, [{"role": "user", "content": task}], max_tokens, ctx, **kw)
        except Exception as e:
            return f"Error calling Qwen CLI: {e}"

    @server.tool(annotations=_CLI_ANN)
    async def cli_opencode_zen(task: str, model: str = "opencode:default",
                               system_prompt: Optional[str] = None, max_tokens: int = 4096,
                               temperature: Optional[float] = None, ctx: Context = None) -> str:
        """Execute an agentic task via OpenCode Zen CLI. Code-focused agent for local development.
        This is a CLI provider -- use for agentic tasks that interact with the local filesystem."""
        if not _is_opencode_zen_healthy():
            return "Error: OpenCode Zen provider is not available (API key not configured)."
        kw = {"temperature": temperature} if temperature is not None else {}
        try:
            if ctx:
                await ctx.info("Starting OpenCode Zen task...")
            return await _logged_call("cli_opencode_zen", "opencode_zen", model, proxy.call_opencode_direct,
                                      system_prompt, [{"role": "user", "content": task}], max_tokens, ctx, **kw)
        except Exception as e:
            return f"Error calling OpenCode Zen: {e}"

    @server.tool(annotations=_CLI_ANN)
    async def cli_opencode_go(task: str, model: str = "opencode-go:default",
                              system_prompt: Optional[str] = None, max_tokens: int = 4096,
                              temperature: Optional[float] = None, ctx: Context = None) -> str:
        """Execute an agentic task via OpenCode Go CLI. Code-focused agent for local development.
        This is a CLI provider -- use for agentic tasks that interact with the local filesystem."""
        if not _is_opencode_go_healthy():
            return "Error: OpenCode Go provider is not available (API key not configured)."
        kw = {"temperature": temperature} if temperature is not None else {}
        try:
            if ctx:
                await ctx.info("Starting OpenCode Go task...")
            return await _logged_call("cli_opencode_go", "opencode_go", model, proxy.call_opencode_direct,
                                      system_prompt, [{"role": "user", "content": task}], max_tokens, ctx, **kw)
        except Exception as e:
            return f"Error calling OpenCode Go: {e}"

    # -------------------------------------------------------------------
    # Unified dispatch tools
    # -------------------------------------------------------------------

    @server.tool(annotations=_CHAT_ANN)
    async def chat(
        message: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
        ctx: Context = None,
    ) -> str:
        """Send a chat message to the best available chat provider (or a specific one).
        Use this for questions, research, analysis -- non-agentic tasks.

        Args:
            message: The message to send.
            provider: Optional provider name (claude, openrouter, ollama, antigravity, zai, xiaomi). Auto-selects if omitted.
            model: Optional model override. Uses provider default if omitted.
            system_prompt: Optional system prompt.
            max_tokens: Maximum response tokens (default 4096).
            temperature: Optional sampling temperature.
        """
        chat_providers = get_healthy_chat_providers()
        if not chat_providers:
            return "Error: No chat providers are currently available."

        target = None
        if provider:
            for p in chat_providers:
                if p["name"] == provider:
                    target = p
                    break
            if not target:
                available = ", ".join(p["name"] for p in chat_providers)
                return f"Error: Provider '{provider}' is not available. Healthy chat providers: {available}"
        else:
            target = chat_providers[0]

        use_model = model or target["default_model"]
        messages = [{"role": "user", "content": message}]
        kwargs = {}
        if temperature is not None and target["accepts_extra"]:
            kwargs["temperature"] = temperature

        # Only fallback to other providers when auto-selecting (no explicit provider).
        # When a specific provider is requested, fail clearly instead of silently routing elsewhere.
        if provider:
            fallback_providers = [target]
        else:
            fallback_providers = [target] + [p for p in chat_providers if p["name"] != target["name"]]

        try:
            if ctx:
                await ctx.report_progress(0, 1)
            sr = await call_with_fallback(fallback_providers, system_prompt, messages,
                                          use_model, max_tokens, **kwargs)
            if ctx:
                await ctx.report_progress(1, 1)
            return sr.full
        except Exception as e:
            if provider:
                return f"Error calling {provider}: {e}"
            return f"Error: All chat providers failed. Last error: {e}"

    @server.tool(annotations=_CLI_ANN)
    async def cli(
        task: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        ctx: Context = None,
    ) -> str:
        """Execute an agentic task on the local system using the best available CLI provider (or a specific one).
        Automatically falls back to other healthy CLI providers on failure.

        Args:
            task: Description of the task to perform.
            provider: Optional provider name (codex_cli, gemini_cli, qwen_cli, opencode_zen, opencode_go). Auto-selects if omitted.
            model: Optional model override. Uses provider default if omitted.
            system_prompt: Optional system prompt.
            max_tokens: Maximum response tokens (default 4096).
        """
        cli_providers = get_healthy_cli_providers()
        if not cli_providers:
            return "Error: No CLI providers are currently available."

        target = None
        if provider:
            for p in cli_providers:
                if p["name"] == provider:
                    target = p
                    break
            if not target:
                available = ", ".join(p["name"] for p in cli_providers)
                return f"Error: Provider '{provider}' is not available. Healthy CLI providers: {available}"
        else:
            target = cli_providers[0]

        use_model = model or target["default_model"]
        messages = [{"role": "user", "content": task}]

        # Only fallback when auto-selecting. Explicit provider = no silent rerouting.
        if provider:
            fallback_providers = [target]
        else:
            fallback_providers = [target] + [p for p in cli_providers if p["name"] != target["name"]]

        try:
            if ctx:
                await ctx.info(f"Starting {target['name']} task...")
                await ctx.report_progress(0, 1)
            sr = await call_with_fallback(fallback_providers, system_prompt, messages,
                                          use_model, max_tokens)
            if ctx:
                await ctx.report_progress(1, 1)
            return sr.full
        except Exception as e:
            if provider:
                return f"Error calling {provider}: {e}"
            return f"Error: All CLI providers failed. Last error: {e}"

    # -------------------------------------------------------------------
    # Orchestration tools
    # -------------------------------------------------------------------

    _ORCH_ANN = ToolAnnotations(readOnlyHint=False, destructiveHint=True,
                                 idempotentHint=False, openWorldHint=True)

    @server.tool(annotations=_ORCH_ANN)
    async def orchestrate_sequential(
        description: str,
        steps: str,
        category: str = "cli",
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        ctx: Context = None,
    ) -> str:
        """Execute a multi-step workflow sequentially, piping context between steps.
        Each step's summary is passed to the next step as context (not the full result,
        to avoid context window collapse). Full results are stored and queryable via
        the task_result tool.

        Args:
            description: High-level description of the workflow.
            steps: JSON array of step objects. Each step: {"description": "...", "message": "...", "provider": "optional", "model": "optional"}.
            category: "chat" or "cli" (determines which providers are used).
            system_prompt: Optional system prompt applied to all steps.
            max_tokens: Max tokens per step (default 4096).
        """
        try:
            step_list = json.loads(steps)
        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON in steps parameter: {e}"

        if not isinstance(step_list, list) or not step_list:
            return "Error: steps must be a non-empty JSON array."

        providers = get_healthy_chat_providers() if category == "chat" else get_healthy_cli_providers()
        if not providers:
            return f"Error: No healthy {category} providers available."

        async def _progress(current, total, msg):
            if ctx:
                await ctx.report_progress(current, total)
                await ctx.info(msg)

        task = await orchestrator.execute_sequential(
            description=description,
            category=category,
            steps=step_list,
            providers=providers,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            progress_callback=_progress,
        )

        return json.dumps(task.to_dict(), indent=2)

    @server.tool(annotations=_ORCH_ANN)
    async def orchestrate_parallel(
        description: str,
        steps: str,
        category: str = "cli",
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        ctx: Context = None,
    ) -> str:
        """Execute independent tasks in parallel across available providers.
        All steps run concurrently. Individual failures don't block other steps.

        Args:
            description: High-level description of the parallel workflow.
            steps: JSON array of step objects. Each: {"description": "...", "message": "...", "provider": "optional", "model": "optional"}.
            category: "chat" or "cli" (determines which providers are used).
            system_prompt: Optional system prompt applied to all steps.
            max_tokens: Max tokens per step (default 4096).
        """
        try:
            step_list = json.loads(steps)
        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON in steps parameter: {e}"

        if not isinstance(step_list, list) or not step_list:
            return "Error: steps must be a non-empty JSON array."

        providers = get_healthy_chat_providers() if category == "chat" else get_healthy_cli_providers()
        if not providers:
            return f"Error: No healthy {category} providers available."

        async def _progress(current, total, msg):
            if ctx:
                await ctx.report_progress(current, total)
                await ctx.info(msg)

        task = await orchestrator.execute_parallel(
            description=description,
            category=category,
            steps=step_list,
            providers=providers,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            progress_callback=_progress,
        )

        return json.dumps(task.to_dict(), indent=2)

    @server.tool(annotations=_READONLY)
    async def task_status(task_id: str, ctx: Context = None) -> str:
        """Get the current status and summary of a task.
        Returns the task summary (not full results) for efficient context use.
        Use task_result for full results."""
        task = task_store.get(task_id)
        if not task:
            return f"Error: Task '{task_id}' not found."
        return json.dumps(task.to_dict(), indent=2)

    @server.tool(annotations=_READONLY)
    async def task_result(task_id: str, step_index: Optional[int] = None,
                          ctx: Context = None) -> str:
        """Get the full result of a completed task or a specific step.
        Use this when you need the complete output, not just the summary.

        Args:
            task_id: The task ID to query.
            step_index: Optional 0-based step index. If omitted, returns the full combined result.
        """
        task = task_store.get(task_id)
        if not task:
            return f"Error: Task '{task_id}' not found."
        if task.status != "completed":
            return f"Error: Task is {task.status}, not completed."

        if step_index is not None:
            if step_index < 0 or step_index >= len(task.steps):
                return f"Error: Step index {step_index} out of range (0-{len(task.steps)-1})."
            step = task.steps[step_index]
            return step.result or f"Error: Step {step_index} has no result."

        return task.result or "Error: Task has no result."

    @server.tool(annotations=_READONLY)
    async def task_list(active_only: bool = True, ctx: Context = None) -> str:
        """List tasks. Returns summaries for efficient context use.

        Args:
            active_only: If true, only show pending/running tasks. If false, show all.
        """
        tasks = task_store.list_active() if active_only else task_store.list_all()
        if not tasks:
            return "No tasks found."
        return json.dumps([t.to_dict() for t in tasks], indent=2)

    # -------------------------------------------------------------------
    # Web search tool (connects to search-mcp via MCP client)
    # -------------------------------------------------------------------

    _SEARCH_ANN = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=True)

    @server.tool(annotations=_SEARCH_ANN)
    async def web_search(
        query: str,
        count: int = 5,
        summarize: bool = True,
        ctx: Context = None,
    ) -> str:
        """Search the web and return results. When summarize=True (default), results are
        compressed through a cheap/fast model before being returned, keeping context
        usage efficient. Raw results are logged alongside the summary.

        Uses the search-mcp server (Brave/SearXNG backends) under the hood.

        Args:
            query: The search query.
            count: Number of results to fetch (default 5).
            summarize: If True, compress results through cheapest available model. If False, return raw.
        """
        conversation_log.log_user_input("web_search", query)

        if ctx:
            await ctx.info(f"Searching: {query}")
            await ctx.report_progress(0, 2)

        if summarize:
            chat_providers = get_healthy_chat_providers()
            result = await search_and_summarize(query, providers=chat_providers, count=count)

            if ctx:
                await ctx.report_progress(2, 2)

            return json.dumps({
                "query": result["query"],
                "result_count": result["result_count"],
                "summarized": result["summarized"],
                "summarizer": result["summarizer"],
                "latency_s": result["latency_s"],
            }, indent=2)
        else:
            from orchestrator import search_web
            raw = await search_web(query, count)
            results = raw.get("results", raw if isinstance(raw, list) else [])

            conversation_log.log_web_search(
                query=query, result_count=len(results),
                raw_length=len(json.dumps(raw)), summarized_length=0,
            )

            if ctx:
                await ctx.report_progress(2, 2)

            return json.dumps(raw, indent=2)

    # -------------------------------------------------------------------
    # MCP Resources -- expose task state for querying
    # -------------------------------------------------------------------

    @server.resource("task://active")
    async def resource_active_tasks() -> str:
        """Active (pending/running) tasks."""
        tasks = task_store.list_active()
        return json.dumps([t.to_dict() for t in tasks], indent=2)

    @server.resource("task://all")
    async def resource_all_tasks() -> str:
        """All tasks (including completed/failed)."""
        tasks = task_store.list_all()
        return json.dumps([t.to_dict() for t in tasks], indent=2)

    @server.resource("task://stats")
    async def resource_task_stats() -> str:
        """Task execution statistics."""
        all_tasks = task_store.list_all()
        completed = [t for t in all_tasks if t.status == "completed"]
        failed = [t for t in all_tasks if t.status == "failed"]
        active = [t for t in all_tasks if t.status in ("pending", "running")]
        latencies = [t.finished_at - t.started_at for t in completed
                     if t.finished_at and t.started_at]
        return json.dumps({
            "total": len(all_tasks),
            "completed": len(completed),
            "failed": len(failed),
            "active": len(active),
            "avg_latency_s": round(sum(latencies) / len(latencies), 2) if latencies else None,
        }, indent=2)

    return server


def _kill_stale_port(port: int) -> bool:
    """Kill any stale process occupying a port. Returns True if a process was killed."""
    import os
    import subprocess as _sp
    import time
    try:
        result = _sp.run(
            ["lsof", "-ti", f"tcp:{port}"], capture_output=True, text=True, timeout=5,
        )
        pids = result.stdout.strip().split()
        if not pids:
            return False
        my_pid = str(os.getpid())
        killed = False
        for pid in pids:
            if pid and pid != my_pid:
                logger.warning(f"Killing stale process {pid} on port {port}")
                os.kill(int(pid), 9)
                killed = True
        if killed:
            time.sleep(0.3)
        return killed
    except Exception as e:
        logger.debug(f"Could not kill stale process on port {port}: {e}")
        return False


def _ensure_port_free(port: int, host: str = "0.0.0.0", max_attempts: int = 3):
    """Ensure a port is free by test-binding, killing stale processes if needed."""
    import socket
    import time
    for attempt in range(1, max_attempts + 1):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            sock.close()
            return  # port is free
        except OSError:
            sock.close()
            if attempt == max_attempts:
                logger.error(f"Port {port} still in use after {max_attempts} attempts")
                raise
            logger.warning(f"Port {port} in use (attempt {attempt}/{max_attempts}), killing stale process...")
            _kill_stale_port(port)
            time.sleep(1)


def run_mcp_server(transport: str = "stdio"):
    """Start the MCP server.

    Args:
        transport: "stdio" for Claude Desktop / CLI integration (default),
                   "sse" for HTTP SSE on port 8432.
    """
    server = create_mcp_server()
    if transport == "sse":
        port = server.settings.port
        _ensure_port_free(port, server.settings.host)
        logger.info(f"Starting MCP server on port {port} (SSE transport)")
        server.run(transport="sse")
    else:
        logger.info("Starting MCP server (stdio transport)")
        server.run(transport="stdio")
