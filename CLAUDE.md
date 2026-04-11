# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenAI-compatible API proxy that routes `/v1/chat/completions` requests through multiple LLM providers (Claude, Codex, OpenRouter, Ollama, Antigravity, Gemini CLI, Z.AI, OpenCode, Qwen Code) to power AI-driven NPC conversations in [SkyrimNet](https://github.com/MinLL/SkyrimNet-GamePlugin). Single-file FastAPI app (`proxy.py`, ~5000 lines) running on port 8000.

## Commands

```bash
# Run the proxy
python proxy.py

# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_ollama.py -v

# Run a single test
pytest tests/test_ollama.py::test_is_ollama_model_simple -v

# Install dependencies
pip install -r requirements.txt
```

No linter or formatter is configured for this project.

## Architecture

### Request Flow

```
Game (SkyrimNet) → POST /v1/chat/completions → proxy.py → Provider API
```

The proxy translates all provider responses into OpenAI-compatible format (both streaming SSE and non-streaming JSON).
Caller-supplied `system` messages are preserved for OpenAI-compatible upstreams and collapsed into `system_prompt` only for providers that require a separate system channel.

### Provider Routing

Model name determines routing via detection functions in `proxy.py`:

| Prefix/Pattern | Provider | Detection Function |
|---|---|---|
| `ollama:*` | Ollama | `is_ollama_model()` |
| `gpt-5.*`, `codex-*` | Codex (OpenAI) | `is_codex_model()` |
| Contains `/` | OpenRouter | `is_openrouter_model()` |
| `antigravity-*` | Antigravity (Google IDE API) | `is_antigravity_model()` |
| `gcli-*` | Gemini CLI | `is_gemini_cli_model()` |
| `zai:*` | Z.AI | `is_zai_model()` |
| `opencode:*`, `opencode-go:*` | OpenCode (Zen/Go) | `is_opencode_model()` |
| `qwen:*` | Qwen Code | `is_qwen_model()` |
| `fireworks:*` | Fireworks | `is_fireworks_model()` |
| `nvidia:*` | NVIDIA NIM | `is_nvidia_model()` |
| Everything else | Claude (Anthropic) | Default |

Each provider has paired `call_<provider>_direct()` and `call_<provider>_streaming()` functions.

### Auth & Startup

- **Claude**: MITM interceptor captures authenticated headers from `claude --print` at startup. Uses port 9999 for proxy mode, port 9997 for MCP mode. Auth refreshed every 45 minutes.
- **Codex**: OAuth tokens read from `~/.codex/auth.json`, auto-refreshed on expiry. The proxy runs Codex in fast mode with `model_reasoning_effort="low"` for both direct and streaming requests.
- **Antigravity**: Multi-account Google OAuth stored encrypted in `antigravity-auth.json`. Automatic fallback on account errors.
- **Gemini CLI**: Credentials loaded from `~/.gemini/oauth_creds.json`.
- **OpenCode**: API keys from `~/.local/share/opencode/auth.json` or `config.json`. Zen at `opencode.ai/zen/v1`, Go at `opencode.ai/zen/go/v1`. Models queried dynamically from `opencode models` CLI.
- **Qwen Code**: OAuth tokens loaded from `~/.qwen/oauth_creds.json`. Auto-refreshed via `chat.qwen.ai` token endpoint.
- **Fireworks**: API key stored encrypted in `config.json`. Base URL `api.fireworks.ai/inference/v1`.
- **NVIDIA NIM**: API key stored encrypted in `config.json`. Base URL `integrate.api.nvidia.com/v1`. Free-tier models with short aliases (e.g. `nvidia:deepseek-r1` → `deepseek-ai/deepseek-r1`).
- **OpenRouter/Ollama/Z.AI**: API keys stored encrypted in `config.json`.

### Encryption

All secrets encrypted at rest using `cryptography.fernet.Fernet`. Key stored in `.proxy.key` (auto-generated). Plaintext values in `config.json` are auto-encrypted on next save.

### Key Subsystems

- **`ModelStatsTracker`**: Per-model TTFT, success rates, rolling window of 50. Powers `get_reliable_model()` for timeout routing fallback.
- **`RequestStatsTracker`**: Per-model request counts, error rates, latency percentiles. Exposed via `/request-stats`.
- **Model rotation**: Comma-separated model lists cycle round-robin across requests (`parse_model_list()`, `pick_model_round_robin()`).
- **Timeout routing**: Optional — if TTFT exceeds cutoff (default 6s), falls back to most reliable alternative model. Streaming only.

### Performance Notes

- Persistent `aiohttp.ClientSession` reused across requests (avoids TCP/TLS setup per request)
- GC tuned: `gc.set_threshold(10000, 20, 20)` + `gc.freeze()` at startup
- `orjson` used when available (falls back to stdlib `json`)
- Cached template parts avoid deepcopy overhead on hot path

## Testing

Tests use `pytest` + `pytest-asyncio`. The `conftest.py` mocks `shutil.which` to prevent CLI subprocess spawning during import. `TestClient` is instantiated without lifespan to keep tests fast and isolated.

## Web Dashboard

`GET /` serves an inline HTML dashboard with provider status, model tables, quick test form, API key configuration, and request stats.
