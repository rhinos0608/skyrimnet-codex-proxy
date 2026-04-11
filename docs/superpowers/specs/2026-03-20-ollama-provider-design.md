# Ollama Provider Integration — Design Spec

**Date:** 2026-03-20
**Status:** Approved (rev 5)

---

## Overview

Add Ollama as a first-class provider in the SkyrimNet proxy, supporting both local Ollama (`http://localhost:11434`) and Ollama Cloud (`https://ollama.com/v1`) via Bearer token. No changes to `parse_model_list` or `pick_model_round_robin` needed.

---

## Model Detection

Models prefixed with `ollama:` route to Ollama. All detection functions are called on the **original, unstripped `model` variable**. The `ollama:` prefix is stripped **inside** the call functions only.

```
ollama:llama3.2       → sends "llama3.2" to Ollama API
ollama:mistral:7b     → sends "mistral:7b" to Ollama API
ollama:namespace/model → sends "namespace/model" to Ollama API (slash in name is valid)
```

The `"model"` field in both streaming chunks and non-streaming responses echoes the **full original model string** (e.g., `"ollama:llama3.2"`), consistent with all other providers which return their model string unchanged.

```python
def is_ollama_model(model: str) -> bool:
    return model.lower().startswith("ollama:")
```

---

## Routing Priority

**Rewrite** all four existing detection lines in `chat_completions` (do not add a fifth — replace all four):

```python
use_ollama      = is_ollama_model(model)
use_openrouter  = not use_ollama and is_openrouter_model(model)
use_codex       = not use_ollama and is_codex_model(model)
use_antigravity = not use_ollama and is_antigravity_model(model)
# else → Claude
```

This ensures `ollama:namespace/model` (which contains `/`) is never misrouted to OpenRouter.

The `if/elif` routing block order: Ollama → OpenRouter → Codex → Antigravity → Claude.

---

## Endpoint Selection

| Condition | Endpoint | Auth |
|-----------|----------|------|
| No `ollama_api_key` configured | `http://localhost:11434/v1/chat/completions` | None |
| `ollama_api_key` configured | `https://ollama.com/v1/chat/completions` | `Authorization: Bearer <key>` |

No pre-flight auth check in the routing block (same pattern as OpenRouter).

---

## Changes to `proxy.py`

### 1. Global config

```python
ollama_api_key: Optional[str] = _cfg.get("ollama_api_key") or None
```

JSON key in `config.json`: `"ollama_api_key"`. Loaded at startup alongside `openrouter_api_key`.

### 2. `is_ollama_model(model: str) -> bool`

Added immediately before `is_openrouter_model` in source order.

### 3. `call_ollama_direct(system_prompt, messages, model, max_tokens, **extra_params) -> str`

- Strip prefix: `api_model = model[len("ollama:"):]`
- Select endpoint and auth header based on `ollama_api_key`
- Build `oai_messages` same way as `call_openrouter_direct`
- Payload: `{"model": api_model, "messages": oai_messages, "max_tokens": max_tokens, **extra_params}`
- Session: `session = auth.session or create_session()`. `auth.session` may be `None` (Ollama-only deployment with no Claude auth) — this is normal and expected. In that case `create_session()` provides an owned session. Cleanup: `if not auth.session: await session.close()` in `finally`
- Parse `choices[0].message.content` defensively. If `content` is empty while reasoning fields are present and `finish_reason == "length"`, retry once without `max_tokens`. If Ollama still returns no usable content, surface that as an upstream error instead of raising `KeyError`.
- Transport and HTTP error handling (catch in this order):
  - `aiohttp.ClientConnectorError` (connection refused, local not running) → `HTTPException(503, "Ollama not running at localhost:11434")`
  - `asyncio.TimeoutError` → `HTTPException(504, "Ollama request timed out")`
  - HTTP 401/403 → `HTTPException(401, "Ollama Cloud auth failed — check API key")`
  - HTTP 429 → `HTTPException(429, "Ollama rate limit exceeded")`
  - Other non-200 → `HTTPException(resp.status, f"Ollama error: {text[:200]}")`

### 4. `call_ollama_streaming(system_prompt, messages, model, max_tokens, **extra_params)`

- Same endpoint/auth/prefix-strip logic as `call_ollama_direct`
- Payload includes `"stream": True`, forwards `**extra_params`
- Session: `owns_session = not auth.session` variable; `if owns_session: await session.close()` in `finally`
- **Passthrough** of Ollama's SSE chunks directly to the client, same pattern as `call_openrouter_streaming` (not parsed and re-emitted like Claude)
- On `aiohttp.ClientConnectorError`, `asyncio.TimeoutError`, or non-200: yield SSE error chunk then `data: [DONE]\n\n`

### 5. Routing in `chat_completions`

Rewrite the four `use_*` lines as specified in Routing Priority. Add the Ollama branch at the top of the `if/elif` chain:

```python
if use_ollama:
    if req.stream:
        return StreamingResponse(call_ollama_streaming(...), media_type="text/event-stream", ...)
    response = await call_ollama_direct(...)
elif use_openrouter:
    ...
```

### 6. `/config/ollama-key` POST endpoint

Same structure as `/config/openrouter-key`. Accepts `{"key": "..."}`. Empty string or missing key clears `ollama_api_key` (sets to `None`, removes from `config.json`) — this switches the endpoint back to local mode. Non-empty key saves to `config.json` and updates the global.

### 7. `/health` endpoint

Add `"ollama_configured": ollama_api_key is not None` to the health response JSON alongside existing fields.

### 8. Dashboard

Ollama status card with two states:

- Key configured → `"Cloud (key configured)"`, color `#4ade80`
- No key → `"Local (localhost:11434)"`, color `#64748b`

API key input form below the card, same HTML pattern as the OpenRouter form, posting to `/config/ollama-key`.

### 9. `/v1/models`

Ping the OpenAI-compat models endpoint on local Ollama using a tight timeout:

```python
async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2, connect=1)) as s:
    async with s.get("http://localhost:11434/v1/models") as resp:
        if resp.status == 200:
            data = await resp.json()
            # OpenAI-compat format: {"object": "list", "data": [{"id": "model-name", ...}]}
            for m in data.get("data", []):
                ollama_models.append({
                    "id": f"ollama:{m['id']}",
                    "object": "model",
                    "owned_by": "ollama",
                })
```

On any failure (connection error, timeout, non-200), skip silently — no Ollama models added.

### 10. No persistent session / no lifespan change

No `ollama_session` global is added. Ollama uses the per-request `auth.session or create_session()` pattern. The `lifespan` function is unchanged.

---

## Error Handling Summary

| Scenario | Non-streaming | Streaming |
|----------|--------------|-----------|
| Local unreachable (`ClientConnectorError`) | HTTP 503 | SSE error chunk + DONE |
| Timeout (`asyncio.TimeoutError`) | HTTP 504 | SSE error chunk + DONE |
| Cloud auth fail (401/403) | HTTP 401 | SSE error chunk + DONE |
| Rate limit (429) | HTTP 429 | SSE error chunk + DONE |
| Reasoning truncated (`finish_reason="length"` with no content) | Retry once without `max_tokens` | N/A |
| No usable content after parsing/retry | HTTP 502 | SSE error chunk + DONE |
| Other non-200 | HTTP `resp.status` | SSE error chunk + DONE |

---

## No Changes Required

- `parse_model_list()` — unchanged
- `pick_model_round_robin()` — unchanged
- All other provider call functions — unchanged
- `lifespan` — unchanged
