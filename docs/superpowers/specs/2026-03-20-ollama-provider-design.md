# Ollama Provider Integration — Design Spec

**Date:** 2026-03-20
**Status:** Approved

---

## Overview

Add Ollama as a first-class provider in the SkyrimNet proxy, supporting both local Ollama (`http://localhost:11434`) and Ollama Cloud (`https://ollama.com/v1`) via Bearer token. The round-robin system routes requests to the correct provider based on model name — no changes to the existing round-robin logic are needed.

---

## Model Detection

Models prefixed with `ollama:` route to Ollama. The prefix is stripped before sending to the API.

```
ollama:llama3.2       → sends "llama3.2" to Ollama
ollama:mistral:7b     → sends "mistral:7b" to Ollama
ollama:qwen3:8b       → sends "qwen3:8b" to Ollama
```

Detection function:

```python
def is_ollama_model(model: str) -> bool:
    return model.lower().startswith("ollama:")
```

---

## Endpoint Selection

| Condition | Endpoint | Auth |
|-----------|----------|------|
| No `ollama_api_key` configured | `http://localhost:11434/v1/chat/completions` | None |
| `ollama_api_key` configured | `https://ollama.com/v1/chat/completions` | `Authorization: Bearer <key>` |

---

## Round-Robin Routing (all providers)

Model names in the round-robin list are detected in this priority order:

1. `ollama:*` → Ollama
2. `provider/model` (contains `/`) → OpenRouter
3. `gpt-5.*` / `codex-*` → Codex
4. `antigravity-*` / `gemini-3*` / `gemini-2.5*` / `gpt-oss-*` → Antigravity
5. All others → Claude (Anthropic)

Example rotation:

```
ollama:llama3.2, gpt-5.4, claude-sonnet-4-6, antigravity-gemini-2.5-pro
```

- Turn 1 → `llama3.2` on Ollama (local or cloud)
- Turn 2 → `gpt-5.4` on Codex
- Turn 3 → `claude-sonnet-4-6` on Claude
- Turn 4 → `antigravity-gemini-2.5-pro` on Antigravity

---

## Changes to `proxy.py`

### 1. Global config

```python
ollama_api_key: Optional[str] = _cfg.get("ollama_api_key") or None
```

Loaded from `config.json` at startup, same pattern as `openrouter_api_key`.

### 2. `is_ollama_model(model: str) -> bool`

Added alongside the other `is_*_model` functions.

### 3. `call_ollama_direct(...)` and `call_ollama_streaming(...)`

Both functions:
- Determine endpoint: localhost or Ollama Cloud based on `ollama_api_key`
- Set `Authorization: Bearer <key>` header if key is present
- Send OpenAI-compatible JSON payload (no format translation needed — Ollama's `/v1/` endpoint is OpenAI-compatible)
- Strip `ollama:` prefix from model name before sending
- Reuse `auth.session` or create a new session

### 4. Routing in `chat_completions`

`use_ollama = is_ollama_model(model)` added alongside the other `use_*` flags.
Ollama branch inserted before the Claude `else` branch.

### 5. `/config/ollama-key` POST endpoint

Same structure as `/config/openrouter-key`. Saves/clears `ollama_api_key` in `config.json`.

### 6. Dashboard

- Ollama status card: shows "Local (localhost:11434)" or "Cloud (key configured)"
- API key input form for Ollama Cloud key (below local status)

### 7. `/v1/models`

Attempt a quick ping to `http://localhost:11434/v1/models`; if successful, include the returned models (prefixed with `ollama:`) in the response. If unreachable, skip silently.

---

## Error Handling

- If local Ollama is unreachable: return HTTP 503 with `"Ollama not running at localhost:11434"`
- If Ollama Cloud key is set but request fails (401/403): return HTTP 401 with `"Ollama Cloud auth failed — check API key"`
- No fallback between local and cloud (explicit configuration)

---

## No Changes Required

- `parse_model_list()` — unchanged
- `pick_model_round_robin()` — unchanged
- All other provider call functions — unchanged
