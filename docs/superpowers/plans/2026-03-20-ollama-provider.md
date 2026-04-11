# Ollama Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Ollama as a fifth provider (local `http://localhost:11434` or cloud `https://ollama.com/v1`) with `ollama:` model prefix routing, API key config, dashboard card, and dynamic model listing.

**Architecture:** All changes are in `proxy.py` (single file, ~2600 lines). New code follows existing patterns exactly: `is_ollama_model()` for detection, `call_ollama_direct/streaming()` mirroring OpenRouter call functions, and a rewrite of the four `use_*` routing flags so Ollama is checked first (preventing `ollama:ns/model` from being misrouted to OpenRouter). No new files except tests.

**Tech Stack:** Python 3.10+, FastAPI, aiohttp, pytest + pytest-asyncio (add to dev deps), unittest.mock

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `proxy.py` | Modify | All provider logic lives here |
| `tests/conftest.py` | Create | Shared fixture that imports proxy with mocked CLI deps |
| `tests/test_ollama.py` | Create | Unit + integration tests |
| `requirements.txt` | Modify | Add pytest, pytest-asyncio, httpx (for TestClient) |

---

### Task 1: Test infrastructure + detection function + global config

**Files:**
- Create: `tests/__init__.py`, `tests/conftest.py`, `tests/test_ollama.py`
- Modify: `requirements.txt`, `proxy.py:465-467`, `proxy.py:482-496`

- [ ] **Step 1: Add test deps to requirements**

Append to `requirements.txt`:
```
# Dev
pytest>=8.0.0
pytest-asyncio>=0.23.0
httpx>=0.27.0
```

(`httpx` is required by FastAPI's TestClient.)

- [ ] **Step 2: Create tests directory and conftest**

```bash
mkdir tests
```

Create `tests/__init__.py` (empty file).

Create `tests/conftest.py`:
```python
"""Shared test fixtures for proxy tests.

Strategy: import proxy with shutil.which mocked to None so no CLI auth
capture runs at startup. The FastAPI app object is available on proxy.app.
All module-level globals (auth, codex_auth, etc.) are initialised but inert.
"""
import sys, os
import pytest
from unittest.mock import patch

# Ensure proxy is importable from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


@pytest.fixture(scope="session", autouse=True)
def _import_proxy_once():
    """Import proxy exactly once per test session with CLI tools mocked out."""
    with patch("shutil.which", return_value=None):
        import proxy  # noqa: F401 — side-effect: registers on sys.modules


@pytest.fixture()
def proxy_module():
    """Return the already-imported proxy module."""
    import proxy
    return proxy


@pytest.fixture()
def test_client(proxy_module):
    """FastAPI TestClient that does NOT run the lifespan (no subprocess spawning)."""
    from fastapi.testclient import TestClient
    # lifespan=False skips the startup/shutdown events (prevents CLI auth subprocesses)
    with TestClient(proxy_module.app, raise_server_exceptions=True, lifespan=False) as client:
        yield client
```

- [ ] **Step 3: Write failing tests for `is_ollama_model` and routing**

Create `tests/test_ollama.py`:
```python
"""Tests for Ollama provider: detection, routing, call functions, endpoints."""
import pytest
import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock


# ---------------------------------------------------------------------------
# Detection + routing
# ---------------------------------------------------------------------------

def test_is_ollama_model_simple(proxy_module):
    assert proxy_module.is_ollama_model("ollama:llama3.2") is True

def test_is_ollama_model_with_tag(proxy_module):
    assert proxy_module.is_ollama_model("ollama:mistral:7b") is True

def test_is_ollama_model_with_slash_in_name(proxy_module):
    """Slash inside ollama model name must still be detected as Ollama, not OpenRouter."""
    assert proxy_module.is_ollama_model("ollama:namespace/model") is True

def test_is_ollama_model_case_insensitive(proxy_module):
    assert proxy_module.is_ollama_model("OLLAMA:llama3.2") is True

def test_is_ollama_model_false_openrouter(proxy_module):
    assert proxy_module.is_ollama_model("openai/gpt-4o") is False

def test_is_ollama_model_false_claude(proxy_module):
    assert proxy_module.is_ollama_model("claude-sonnet-4-6") is False

def test_is_ollama_model_false_codex(proxy_module):
    assert proxy_module.is_ollama_model("gpt-5.4") is False


def test_routing_ollama_slash_model_not_routed_to_openrouter(proxy_module):
    """ollama:ns/model contains '/' but must route to Ollama only."""
    model = "ollama:namespace/model"
    use_ollama = proxy_module.is_ollama_model(model)
    use_openrouter = not use_ollama and proxy_module.is_openrouter_model(model)
    assert use_ollama is True
    assert use_openrouter is False

def test_routing_openrouter_unaffected(proxy_module):
    model = "openai/gpt-4o"
    use_ollama = proxy_module.is_ollama_model(model)
    use_openrouter = not use_ollama and proxy_module.is_openrouter_model(model)
    assert use_ollama is False
    assert use_openrouter is True

def test_routing_codex_unaffected(proxy_module):
    model = "gpt-5.4"
    use_ollama = proxy_module.is_ollama_model(model)
    use_codex = not use_ollama and proxy_module.is_codex_model(model)
    assert use_ollama is False
    assert use_codex is True

def test_routing_antigravity_unaffected(proxy_module):
    model = "antigravity-gemini-2.5-pro"
    use_ollama = proxy_module.is_ollama_model(model)
    use_antigravity = not use_ollama and proxy_module.is_antigravity_model(model)
    assert use_ollama is False
    assert use_antigravity is True
```

- [ ] **Step 4: Run tests to confirm failure (function not yet added)**

```bash
cd C:\skyrimnet-codex-proxy && python -m pytest tests/test_ollama.py -v -k "is_ollama or routing"
```
Expected: `AttributeError: module 'proxy' has no attribute 'is_ollama_model'`

- [ ] **Step 5: Add `is_ollama_model` to `proxy.py`**

In `proxy.py`, find line 484:
```python
def is_openrouter_model(model: str) -> bool:
```

Insert immediately before it:
```python
def is_ollama_model(model: str) -> bool:
    """Ollama models use 'ollama:model' or 'ollama:model:tag' prefix."""
    return model.lower().startswith("ollama:")


```

- [ ] **Step 6: Add `ollama_api_key` global**

In `proxy.py`, find (lines 465-467):
```python
openrouter_api_key: Optional[str] = _cfg.get("openrouter_api_key") or None
if openrouter_api_key:
    logger.info("OpenRouter API key loaded from config.json")
```

Add immediately after:
```python
ollama_api_key: Optional[str] = _cfg.get("ollama_api_key") or None
if ollama_api_key:
    logger.info("Ollama Cloud API key loaded from config.json")
```

- [ ] **Step 7: Run tests**

```bash
python -m pytest tests/test_ollama.py -v -k "is_ollama or routing"
```
Expected: all 11 tests PASS

- [ ] **Step 8: Commit**

```bash
git add tests/ requirements.txt proxy.py && git commit -m "feat: add is_ollama_model, ollama_api_key global, and test infrastructure"
```

---

### Task 2: `call_ollama_direct`

**Files:**
- Modify: `proxy.py` (insert after `call_openrouter_streaming`, before `# --- Antigravity Auth Loading ---`)
- Modify: `tests/test_ollama.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_ollama.py`:
```python
# ---------------------------------------------------------------------------
# call_ollama_direct
# ---------------------------------------------------------------------------

def _make_mock_response(status=200, json_data=None, text_data=""):
    """Build a mock aiohttp response usable as an async context manager."""
    mock_resp = AsyncMock()
    mock_resp.status = status
    mock_resp.json = AsyncMock(return_value=json_data or {})
    mock_resp.text = AsyncMock(return_value=text_data)
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    return mock_resp


def _make_mock_session(resp):
    """Build a mock aiohttp session whose .post() returns the given response."""
    session = MagicMock()
    session.post = MagicMock(return_value=resp)
    session.close = AsyncMock()
    return session


@pytest.mark.asyncio
async def test_call_ollama_direct_strips_prefix_local(proxy_module):
    """Strips 'ollama:' prefix before sending model name to local endpoint."""
    resp = _make_mock_response(200, {"choices": [{"message": {"content": "Hi"}}]})
    session = _make_mock_session(resp)

    with patch.object(proxy_module, "ollama_api_key", None), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch("aiohttp.ClientSession", return_value=session):
        result = await proxy_module.call_ollama_direct(
            None, [{"role": "user", "content": "hi"}], "ollama:llama3.2", 100
        )

    assert result == "Hi"
    url = session.post.call_args[0][0]
    assert url == "http://localhost:11434/v1/chat/completions"
    payload = session.post.call_args[1]["json"]
    assert payload["model"] == "llama3.2"  # prefix stripped


@pytest.mark.asyncio
async def test_call_ollama_direct_cloud_endpoint_and_auth(proxy_module):
    """When key is set, uses cloud endpoint with Bearer token."""
    resp = _make_mock_response(200, {"choices": [{"message": {"content": "Cloud hi"}}]})
    session = _make_mock_session(resp)

    with patch.object(proxy_module, "ollama_api_key", "sk-ollama-test"), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch("aiohttp.ClientSession", return_value=session):
        result = await proxy_module.call_ollama_direct(
            None, [{"role": "user", "content": "hi"}], "ollama:llama3.2", 100
        )

    assert result == "Cloud hi"
    url = session.post.call_args[0][0]
    assert url == "https://ollama.com/v1/chat/completions"
    hdrs = session.post.call_args[1]["headers"]
    assert hdrs["Authorization"] == "Bearer sk-ollama-test"


@pytest.mark.asyncio
async def test_call_ollama_direct_retries_reasoning_truncation_without_max_tokens(proxy_module):
    """Reasoning-only truncation should retry once without max_tokens."""
    first_resp = _make_mock_response(
        200,
        {
            "choices": [{
                "finish_reason": "length",
                "message": {"reasoning_content": "internal trace"},
            }]
        },
    )
    second_resp = _make_mock_response(200, {"choices": [{"message": {"content": "final answer"}}]})
    session = MagicMock()
    session.post = MagicMock(side_effect=[first_resp, second_resp])
    session.close = AsyncMock()

    with patch.object(proxy_module, "ollama_api_key", None), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch("aiohttp.ClientSession", return_value=session):
        result = await proxy_module.call_ollama_direct(
            None, [{"role": "user", "content": "hi"}], "ollama:llama3.2", 100
        )

    assert result == "final answer"
    assert session.post.call_args_list[0].kwargs["json"]["max_tokens"] == 100
    assert "max_tokens" not in session.post.call_args_list[1].kwargs["json"]


@pytest.mark.asyncio
async def test_call_ollama_direct_unreachable_raises_503(proxy_module):
    """ClientConnectorError → HTTPException 503."""
    import aiohttp
    from fastapi import HTTPException

    session = MagicMock()
    conn_key = MagicMock()
    session.post = MagicMock(side_effect=aiohttp.ClientConnectorError(
        conn_key, OSError("Connection refused")
    ))
    session.close = AsyncMock()

    with patch.object(proxy_module, "ollama_api_key", None), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch("aiohttp.ClientSession", return_value=session):
        with pytest.raises(HTTPException) as exc:
            await proxy_module.call_ollama_direct(
                None, [{"role": "user", "content": "hi"}], "ollama:llama3.2", 100
            )
    assert exc.value.status_code == 503
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/test_ollama.py -v -k "call_ollama_direct"
```
Expected: `AttributeError: module 'proxy' has no attribute 'call_ollama_direct'`

- [ ] **Step 3: Implement `call_ollama_direct`**

In `proxy.py`, find the comment `# --- Antigravity Auth Loading ---` (around line 1336) and insert before it:

```python
# --- Ollama API calls ---

async def call_ollama_direct(system_prompt: Optional[str], messages: list, model: str, max_tokens: int, **extra_params) -> str:
    """Forward request to Ollama (OpenAI-compatible), collect full response."""
    api_model = model[len("ollama:"):]
    if ollama_api_key:
        endpoint = "https://ollama.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {ollama_api_key}", "Content-Type": "application/json"}
    else:
        endpoint = "http://localhost:11434/v1/chat/completions"
        headers = {"Content-Type": "application/json"}

    oai_messages = []
    if system_prompt:
        oai_messages.append({"role": "system", "content": system_prompt})
    for m in messages:
        oai_messages.append({"role": m["role"], "content": m["content"]})

    payload = {"model": api_model, "messages": oai_messages, "max_tokens": max_tokens}
    payload.update({k: v for k, v in extra_params.items() if v is not None})

    request_id = uuid.uuid4().hex[:8]
    target = "Ollama Cloud" if ollama_api_key else f"Ollama local ({api_model})"
    logger.info(f"[{request_id}] -> {target} ({len(messages)} msgs)")
    start = time.time()

    session = auth.session or create_session()
    try:
        async with session.post(endpoint, json=payload, headers=headers) as resp:
            elapsed = time.time() - start
            if resp.status in (401, 403):
                raise HTTPException(status_code=401, detail="Ollama Cloud auth failed — check API key")
            if resp.status == 429:
                raise HTTPException(status_code=429, detail="Ollama rate limit exceeded")
            if resp.status != 200:
                error_body = await resp.text()
                logger.error(f"[{request_id}] Ollama {resp.status}: {error_body[:300]}")
                raise HTTPException(status_code=resp.status, detail=error_body[:200])
            data = await resp.json()
            text = _extract_oai_content(data["choices"][0]["message"])
            if text is None and _is_reasoning_truncated(data):
                retry_payload = dict(payload)
                retry_payload.pop("max_tokens", None)
                async with session.post(endpoint, json=retry_payload, headers=headers) as retry_resp:
                    retry_data = await retry_resp.json()
                    text = _extract_oai_content(retry_data["choices"][0]["message"])
            if text is None:
                raise HTTPException(status_code=502, detail="Ollama returned no content")
            logger.info(f"[{request_id}] <- {len(text)} chars ({elapsed:.1f}s, Ollama)")
            return text
    except aiohttp.ClientConnectorError:
        raise HTTPException(status_code=503, detail="Ollama not running at localhost:11434")
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Ollama request timed out")
    finally:
        if not auth.session:
            await session.close()


```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_ollama.py -v -k "call_ollama_direct"
```
Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add proxy.py tests/test_ollama.py && git commit -m "feat: add call_ollama_direct"
```

---

### Task 3: `call_ollama_streaming`

**Files:**
- Modify: `proxy.py` (insert immediately after `call_ollama_direct`)
- Modify: `tests/test_ollama.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_ollama.py`:
```python
# ---------------------------------------------------------------------------
# call_ollama_streaming
# ---------------------------------------------------------------------------

async def _async_bytes_iter(chunks):
    """Async generator that yields byte chunks, simulating aiohttp iter_any."""
    for chunk in chunks:
        yield chunk


@pytest.mark.asyncio
async def test_call_ollama_streaming_passthrough(proxy_module):
    """SSE bytes from Ollama are passed through verbatim."""
    sse_bytes = b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\ndata: [DONE]\n\n'

    mock_content = MagicMock()
    mock_content.iter_any.return_value = _async_bytes_iter([sse_bytes])

    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.content = mock_content
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    session = MagicMock()
    session.post = MagicMock(return_value=mock_resp)
    session.close = AsyncMock()

    with patch.object(proxy_module, "ollama_api_key", None), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch("aiohttp.ClientSession", return_value=session):
        chunks = []
        async for chunk in proxy_module.call_ollama_streaming(
            None, [{"role": "user", "content": "hi"}], "ollama:llama3.2", 100
        ):
            chunks.append(chunk)

    combined = "".join(chunks)
    assert "hello" in combined


@pytest.mark.asyncio
async def test_call_ollama_streaming_unreachable_yields_error_and_done(proxy_module):
    """ClientConnectorError → SSE error chunk + [DONE], no exception raised."""
    import aiohttp

    session = MagicMock()
    conn_key = MagicMock()
    session.post = MagicMock(side_effect=aiohttp.ClientConnectorError(
        conn_key, OSError("refused")
    ))
    session.close = AsyncMock()

    with patch.object(proxy_module, "ollama_api_key", None), \
         patch.object(proxy_module, "auth", MagicMock(session=None)), \
         patch("aiohttp.ClientSession", return_value=session):
        chunks = []
        async for chunk in proxy_module.call_ollama_streaming(
            None, [{"role": "user", "content": "hi"}], "ollama:llama3.2", 100
        ):
            chunks.append(chunk)

    combined = "".join(chunks)
    assert "[DONE]" in combined
    # Must contain an error message in the SSE payload
    assert "Ollama" in combined or "error" in combined.lower() or "503" in combined
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/test_ollama.py -v -k "streaming"
```
Expected: `AttributeError: module 'proxy' has no attribute 'call_ollama_streaming'`

- [ ] **Step 3: Implement `call_ollama_streaming`**

Insert immediately after `call_ollama_direct` (before `# --- Antigravity Auth Loading ---`):

```python
async def call_ollama_streaming(system_prompt: Optional[str], messages: list, model: str, max_tokens: int, **extra_params):
    """Forward request to Ollama with streaming, passthrough SSE directly."""
    api_model = model[len("ollama:"):]
    if ollama_api_key:
        endpoint = "https://ollama.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {ollama_api_key}", "Content-Type": "application/json"}
    else:
        endpoint = "http://localhost:11434/v1/chat/completions"
        headers = {"Content-Type": "application/json"}

    oai_messages = []
    if system_prompt:
        oai_messages.append({"role": "system", "content": system_prompt})
    for m in messages:
        oai_messages.append({"role": m["role"], "content": m["content"]})

    payload = {"model": api_model, "messages": oai_messages, "max_tokens": max_tokens, "stream": True}
    payload.update({k: v for k, v in extra_params.items() if v is not None})

    request_id = uuid.uuid4().hex[:8]
    target = "Ollama Cloud" if ollama_api_key else f"Ollama local ({api_model})"
    logger.info(f"[{request_id}] -> {target} ({len(messages)} msgs, stream)")
    start = time.time()

    session = auth.session or create_session()
    owns_session = not auth.session
    try:
        async with session.post(endpoint, json=payload, headers=headers) as resp:
            if resp.status != 200:
                error_body = await resp.text()
                logger.error(f"[{request_id}] Ollama {resp.status}: {error_body[:300]}")
                cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
                err_chunk = {"id": cmpl_id, "object": "chat.completion.chunk",
                             "created": int(time.time()), "model": model,
                             "choices": [{"index": 0, "delta": {"content": f"[Ollama Error {resp.status}]"}, "finish_reason": None}]}
                yield f"data: {json.dumps(err_chunk)}\n\n"
                yield "data: [DONE]\n\n"
                return

            # Ollama /v1 returns OpenAI-format SSE — passthrough directly
            buffer = ""
            async for raw_chunk in resp.content.iter_any():
                buffer += raw_chunk.decode("utf-8", errors="replace")
                while "\n\n" in buffer:
                    event, buffer = buffer.split("\n\n", 1)
                    event = event.strip()
                    if event:
                        yield event + "\n\n"

            if buffer.strip():
                yield buffer.strip() + "\n\n"

            elapsed = time.time() - start
            logger.info(f"[{request_id}] <- stream done ({elapsed:.1f}s, Ollama)")
    except aiohttp.ClientConnectorError:
        cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
        err_chunk = {"id": cmpl_id, "object": "chat.completion.chunk",
                     "created": int(time.time()), "model": model,
                     "choices": [{"index": 0, "delta": {"content": "[Ollama Error: not running at localhost:11434]"}, "finish_reason": None}]}
        yield f"data: {json.dumps(err_chunk)}\n\n"
        yield "data: [DONE]\n\n"
    except asyncio.TimeoutError:
        cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
        err_chunk = {"id": cmpl_id, "object": "chat.completion.chunk",
                     "created": int(time.time()), "model": model,
                     "choices": [{"index": 0, "delta": {"content": "[Ollama Error: request timed out]"}, "finish_reason": None}]}
        yield f"data: {json.dumps(err_chunk)}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        if owns_session:
            await session.close()


```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_ollama.py -v -k "streaming"
```
Expected: 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add proxy.py tests/test_ollama.py && git commit -m "feat: add call_ollama_streaming"
```

---

### Task 4: Update routing in `chat_completions`

**Files:**
- Modify: `proxy.py:1885-1899` (use_* flags and auth check block)
- Modify: `proxy.py:1928` (routing if/elif chain)

- [ ] **Step 1: Rewrite the three `use_*` detection lines (add `use_ollama` first)**

Find (lines 1885-1887):
```python
    use_openrouter = is_openrouter_model(model)
    use_codex = is_codex_model(model)
    use_antigravity = is_antigravity_model(model)
```

Replace with:
```python
    use_ollama = is_ollama_model(model)
    use_openrouter = not use_ollama and is_openrouter_model(model)
    use_codex = not use_ollama and is_codex_model(model)
    use_antigravity = not use_ollama and is_antigravity_model(model)
```

- [ ] **Step 2: Add Ollama branch to the auth-check block**

Find (line 1889):
```python
    if use_openrouter:
        pass  # No auth check needed for OpenRouter
```

Replace with:
```python
    if use_ollama:
        pass  # No auth check — local needs none, cloud key checked inside call function
    elif use_openrouter:
        pass  # No auth check needed for OpenRouter
```

- [ ] **Step 3: Insert Ollama branch into the routing if/elif chain**

Find (line 1928):
```python
    # Route to correct provider
    if use_openrouter:
```

Replace with:
```python
    # Route to correct provider
    if use_ollama:
        if req.stream:
            return StreamingResponse(
                call_ollama_streaming(system_prompt, merged, model, max_tokens, **extra_params),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        response = await call_ollama_direct(system_prompt, merged, model, max_tokens, **extra_params)
    elif use_openrouter:
```

- [ ] **Step 4: Verify clean import**

```bash
python -c "import proxy; print('OK')"
```
Expected: `OK`

- [ ] **Step 5: Run full test suite**

```bash
python -m pytest tests/ -v
```
Expected: all tests PASS

- [ ] **Step 6: Smoke-test routing logic manually**

```bash
python -c "
import proxy
m = 'ollama:ns/model'
uo = proxy.is_ollama_model(m)
uor = not uo and proxy.is_openrouter_model(m)
print(f'ollama={uo}, openrouter={uor}')   # ollama=True, openrouter=False

m2 = 'openai/gpt-4o'
uo2 = proxy.is_ollama_model(m2)
uor2 = not uo2 and proxy.is_openrouter_model(m2)
print(f'ollama={uo2}, openrouter={uor2}') # ollama=False, openrouter=True
"
```

- [ ] **Step 7: Commit**

```bash
git add proxy.py && git commit -m "feat: add Ollama routing to chat_completions"
```

---

### Task 5: `/config/ollama-key` endpoint

**Files:**
- Modify: `proxy.py` (insert after `set_openrouter_key`)
- Modify: `tests/test_ollama.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_ollama.py`:
```python
# ---------------------------------------------------------------------------
# /config/ollama-key endpoint
# ---------------------------------------------------------------------------

def test_set_ollama_key_saves(test_client, proxy_module):
    """POST with non-empty key saves it and returns status=saved."""
    with patch.object(proxy_module, "_save_config"), \
         patch.object(proxy_module, "_load_config", return_value={}):
        resp = test_client.post("/config/ollama-key", json={"key": "sk-test-key"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "saved"
    assert proxy_module.ollama_api_key == "sk-test-key"


def test_set_ollama_key_clears(test_client, proxy_module):
    """POST with empty key clears it and returns status=cleared."""
    proxy_module.ollama_api_key = "existing-key"  # set up state
    with patch.object(proxy_module, "_save_config"), \
         patch.object(proxy_module, "_load_config", return_value={"ollama_api_key": "existing-key"}):
        resp = test_client.post("/config/ollama-key", json={"key": ""})
    assert resp.status_code == 200
    assert resp.json()["status"] == "cleared"
    assert proxy_module.ollama_api_key is None
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/test_ollama.py -v -k "set_ollama_key"
```
Expected: 404 Not Found

- [ ] **Step 3: Implement the endpoint**

In `proxy.py`, find the end of `set_openrouter_key` (after `return {"status": "saved"}` around line 2003). Insert immediately after:

```python

@app.post("/config/ollama-key")
async def set_ollama_key(request: Request):
    global ollama_api_key
    data = await request.json()
    key = data.get("key", "").strip()
    cfg = _load_config()
    if not key:
        ollama_api_key = None
        cfg.pop("ollama_api_key", None)
        _save_config(cfg)
        logger.info("Ollama API key cleared — using local endpoint (localhost:11434)")
        return {"status": "cleared"}
    ollama_api_key = key
    cfg["ollama_api_key"] = key
    _save_config(cfg)
    logger.info("Ollama Cloud API key configured and saved to config.json")
    return {"status": "saved"}
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_ollama.py -v -k "set_ollama_key"
```
Expected: 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add proxy.py tests/test_ollama.py && git commit -m "feat: add /config/ollama-key endpoint"
```

---

### Task 6: `/health` and `/v1/models` updates

**Files:**
- Modify: `proxy.py:2285` (`/health`)
- Modify: `proxy.py:2258` (`/v1/models`)

- [ ] **Step 1: Update `/health`**

Find (line 2285):
```python
        "openrouter_configured": openrouter_api_key is not None,
```

Replace with:
```python
        "openrouter_configured": openrouter_api_key is not None,
        "ollama_configured": ollama_api_key is not None,
```

- [ ] **Step 2: Update `/v1/models` to probe local Ollama**

In `list_models()`, find (lines 2258-2259):
```python
    if openrouter_api_key:
        data.append({"id": "openrouter/*", "object": "model", "owned_by": "openrouter"})
```

Insert before it:
```python
    # Probe local Ollama with a short timeout — skip silently if unreachable
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=2, connect=1)
        ) as s:
            async with s.get("http://localhost:11434/v1/models") as ollama_resp:
                if ollama_resp.status == 200:
                    ollama_data = await ollama_resp.json()
                    for m in ollama_data.get("data", []):
                        model_id = m.get("id", "")
                        if model_id:
                            data.append({"id": f"ollama:{model_id}", "object": "model", "owned_by": "ollama"})
    except Exception:
        pass  # Ollama not running or unreachable — skip silently

```

- [ ] **Step 3: Verify health endpoint has the new field**

```bash
python -c "
from fastapi.testclient import TestClient
import proxy
with TestClient(proxy.app) as client:
    resp = client.get('/health')
    d = resp.json()
    assert 'ollama_configured' in d, 'Missing ollama_configured'
    print('health OK:', d['ollama_configured'])
"
```
Expected: `health OK: False`

- [ ] **Step 4: Commit**

```bash
git add proxy.py && git commit -m "feat: update /health and /v1/models for Ollama"
```

---

### Task 7: Dashboard updates

**Files:**
- Modify: `proxy.py` (`dashboard()` function, lines 2289-2580)

- [ ] **Step 1: Add Ollama status variables**

Find (lines 2295-2296):
```python
    or_status = "Configured (saved)" if openrouter_api_key else "Not set"
    or_color = "#4ade80" if openrouter_api_key else "#64748b"
```

Add immediately after:
```python
    ollama_status = "Cloud (key configured)" if ollama_api_key else "Local (localhost:11434)"
    ollama_color = "#4ade80" if ollama_api_key else "#64748b"
```

- [ ] **Step 2: Add Ollama card to the provider grid**

The `<div class="provider-grid">` contains three provider cards. Find the closing `</div>` of the Antigravity card (it ends just before `</div>` that closes the grid). Add the Ollama card as a fourth card inside the grid div, after the Antigravity card's closing `</div>`:

```html
    <!-- Ollama Provider -->
    <div class="provider-card" style="border-color:{ollama_color}40">
      <div class="provider-header">
        <h3 style="margin:0;font-size:1rem;color:#f1f5f9">🦙 Ollama</h3>
        <span class="provider-status" style="background:{ollama_color}20;color:{ollama_color}">{ollama_status}</span>
      </div>
      <div style="margin-bottom:8px"><span class="label">Prefix:</span> <span class="value">ollama:model-name</span></div>
      <div style="color:#64748b;font-size:0.8rem">
        Pull models with <code style="color:#67e8f9">ollama pull model-name</code><br>
        Set API key below for Ollama Cloud.
      </div>
    </div>
```

To find the exact insertion point, look for this pattern in the dashboard HTML string:
```
  </div>

  <!-- OpenRouter -->
```
The Ollama card goes between the `</div>` (closing provider-grid) and `<!-- OpenRouter -->`. Actually it goes **inside** the provider-grid div. Look for:
```html
    </div>
  </div>

  <!-- OpenRouter -->
```
The first `</div>` closes the Antigravity card. The second `</div>` closes the provider-grid. Insert the Ollama card between them (after Antigravity's `</div>`, before the grid's `</div>`).

- [ ] **Step 3: Add Ollama Cloud key form**

Find `<!-- OpenRouter -->` in the dashboard HTML. Insert a new card immediately before it:

```html
  <!-- Ollama Cloud Key -->
  <div class="card">
    <h3 style="margin:0 0 8px; font-size:1rem; color:#f1f5f9">🦙 Ollama Cloud Key</h3>
    <div style="display:flex; gap:8px; align-items:center">
      <input type="password" id="ollamaKey" placeholder="API key"
             style="flex:1; background:#0f172a; color:#e2e8f0; border:1px solid #334155;
                    border-radius:6px; padding:8px 12px; font-family:monospace; font-size:0.85rem">
      <button onclick="saveOllamaKey()" style="margin-top:0">Save</button>
      <span id="ollamaStatus" style="color:{ollama_color}; font-size:0.85rem; font-weight:600">{ollama_status}</span>
    </div>
    <div style="color:#64748b; font-size:0.8rem; margin-top:8px">
      Leave empty to use local Ollama at <code style="color:#67e8f9">localhost:11434</code>
    </div>
  </div>

```

- [ ] **Step 4: Add Ollama options to Quick Test dropdown**

Find `</optgroup>` that closes the Antigravity optgroup in the `<select id="modelSelect">`. Add after it:

```html
        <optgroup label="Ollama Models">
          <option value="ollama:llama3.2">ollama:llama3.2</option>
          <option value="ollama:mistral">ollama:mistral</option>
          <option value="ollama:qwen3:8b">ollama:qwen3:8b</option>
        </optgroup>
```

- [ ] **Step 5: Add `saveOllamaKey` JavaScript**

Find `async function saveOrKey()` in the `<script>` block. Add the Ollama equivalent immediately after the closing `}}` of `saveOrKey`:

```javascript
async function saveOllamaKey() {{
  const key = document.getElementById('ollamaKey').value.trim();
  const status = document.getElementById('ollamaStatus');
  try {{
    await fetch('/config/ollama-key', {{
      method: 'POST', headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{key: key}})
    }});
    status.textContent = key ? 'Cloud (key configured)' : 'Local (localhost:11434)';
    status.style.color = key ? '#4ade80' : '#64748b';
    document.getElementById('ollamaKey').value = '';
  }} catch(e) {{
    status.textContent = 'Error';
    status.style.color = '#f87171';
  }}
}}
```

Note: the `{{` and `}}` double-brace escaping is required because the dashboard HTML is inside a Python f-string.

- [ ] **Step 6: Verify dashboard renders with all Ollama additions**

```bash
python -c "
from fastapi.testclient import TestClient
import proxy
with TestClient(proxy.app) as client:
    resp = client.get('/')
assert resp.status_code == 200
html = resp.text
assert 'Ollama' in html, 'No Ollama in dashboard'
assert 'ollama:llama3.2' in html, 'No ollama model in dropdown'
assert 'saveOllamaKey' in html, 'No saveOllamaKey JS'
assert 'ollamaKey' in html, 'No ollamaKey input'
assert '/config/ollama-key' in html, 'No endpoint in JS'
print('Dashboard OK')
"
```
Expected: `Dashboard OK`

- [ ] **Step 7: Run full test suite**

```bash
python -m pytest tests/ -v
```
Expected: all tests PASS

- [ ] **Step 8: Commit**

```bash
git add proxy.py && git commit -m "feat: add Ollama dashboard card and API key form"
```

---

### Task 8: Final checks and README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Full test suite**

```bash
python -m pytest tests/ -v
```
Expected: all tests PASS, no warnings about unclosed sessions

- [ ] **Step 2: Verify proxy starts cleanly**

```bash
python -c "
import proxy
print('is_ollama_model:', proxy.is_ollama_model('ollama:llama3.2'))
print('ollama_api_key:', proxy.ollama_api_key)
print('call_ollama_direct:', callable(proxy.call_ollama_direct))
print('call_ollama_streaming:', callable(proxy.call_ollama_streaming))
"
```
Expected:
```
is_ollama_model: True
ollama_api_key: None
call_ollama_direct: True
call_ollama_streaming: True
```

- [ ] **Step 3: Manual end-to-end test (if local Ollama is available)**

If Ollama is installed and `llama3.2` is pulled (`ollama pull llama3.2`):
```bash
# Terminal 1
python proxy.py

# Terminal 2
curl -s http://localhost:8539/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"ollama:llama3.2\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi in one word\"}],\"stream\":false}" \
  | python -m json.tool
```
Expected: JSON with `"model": "ollama:llama3.2"` and non-empty content.

- [ ] **Step 4: Update README — Supported Models table**

Add a row to the table under `### Supported Models`:
```markdown
| `ollama:model-name` | Ollama | Local model (e.g. `ollama:llama3.2`, `ollama:mistral:7b`) |
```

- [ ] **Step 5: Update README — add Ollama Support section**

Add after the `### Codex Support` section:
```markdown
### Ollama Support

To use local Ollama models:

1. Install Ollama from [ollama.com](https://ollama.com) and start it
2. Pull a model: `ollama pull llama3.2`
3. Use `ollama:model-name` as the model in SkyrimNet (e.g., `ollama:llama3.2`)

To use Ollama Cloud:

1. Open the dashboard at `http://127.0.0.1:8539` and enter your Ollama Cloud API key
2. Use the same `ollama:model-name` prefix — the proxy routes to cloud automatically when the key is set
3. Clear the key to switch back to local
```

- [ ] **Step 6: Final commit**

```bash
git add README.md && git commit -m "docs: add Ollama provider to README"
```
