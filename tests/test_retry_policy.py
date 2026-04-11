"""Tests for the shared retry helper, config endpoints, and provider integration.

Ported from the retry-policy worktree; adapted to master's fixtures
(``proxy_module`` + ``test_client`` from tests/conftest.py) and extended with
two new provider integration cases (Ollama direct retry, Fireworks 401
fail-fast).
"""
import asyncio
import json
from unittest.mock import patch, MagicMock, AsyncMock

import aiohttp
import pytest
from fastapi import HTTPException


# ---------- _with_retry unit tests ----------

@pytest.mark.asyncio
async def test_retry_helper_success_no_retry(proxy_module):
    """A successful call should not retry and the helper should return its result."""
    calls = 0

    async def fn():
        nonlocal calls
        calls += 1
        return "ok"

    with patch.object(proxy_module, "max_retries", 1):
        result = await proxy_module._with_retry(fn, operation="test", request_id="t")
    assert result == "ok"
    assert calls == 1


@pytest.mark.asyncio
async def test_retry_helper_retries_once_on_retryable_5xx(proxy_module):
    """max_retries=1 + first call 500 + second call success -> 2 attempts total."""
    calls = 0

    async def fn():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise HTTPException(status_code=500, detail="upstream boom")
        return "ok"

    with patch.object(proxy_module, "max_retries", 1):
        result = await proxy_module._with_retry(
            fn, operation="test", request_id="t", base_delay_s=0.0
        )
    assert result == "ok"
    assert calls == 2


@pytest.mark.asyncio
async def test_retry_helper_raises_after_exhausting_retries(proxy_module):
    """max_retries=1 + both attempts fail retryably -> raises the last exception."""
    calls = 0

    async def fn():
        nonlocal calls
        calls += 1
        raise HTTPException(status_code=503, detail="still down")

    with patch.object(proxy_module, "max_retries", 1):
        with pytest.raises(HTTPException) as exc_info:
            await proxy_module._with_retry(
                fn, operation="test", request_id="t", base_delay_s=0.0
            )
    assert exc_info.value.status_code == 503
    assert calls == 2


@pytest.mark.asyncio
async def test_retry_helper_no_retry_on_4xx_except_429(proxy_module):
    """A 400 should abort immediately; no retry."""
    calls = 0

    async def fn():
        nonlocal calls
        calls += 1
        raise HTTPException(status_code=400, detail="bad request")

    with patch.object(proxy_module, "max_retries", 3):
        with pytest.raises(HTTPException) as exc_info:
            await proxy_module._with_retry(
                fn, operation="test", request_id="t", base_delay_s=0.0
            )
    assert exc_info.value.status_code == 400
    assert calls == 1


@pytest.mark.asyncio
async def test_retry_helper_retries_on_429(proxy_module):
    """429 is retryable by default."""
    calls = 0

    async def fn():
        nonlocal calls
        calls += 1
        if calls < 2:
            raise HTTPException(status_code=429, detail="slow down")
        return "ok"

    with patch.object(proxy_module, "max_retries", 2):
        result = await proxy_module._with_retry(
            fn, operation="test", request_id="t", base_delay_s=0.0
        )
    assert result == "ok"
    assert calls == 2


@pytest.mark.asyncio
async def test_retry_helper_no_retry_on_401(proxy_module):
    """401 is in _NO_RETRY_STATUSES — fail fast even if max_retries is large."""
    calls = 0

    async def fn():
        nonlocal calls
        calls += 1
        raise HTTPException(status_code=401, detail="unauthorized")

    with patch.object(proxy_module, "max_retries", 5):
        with pytest.raises(HTTPException) as exc_info:
            await proxy_module._with_retry(
                fn, operation="test", request_id="t", base_delay_s=0.0
            )
    assert exc_info.value.status_code == 401
    assert calls == 1


@pytest.mark.asyncio
async def test_retry_helper_no_retry_on_403(proxy_module):
    """403 is in _NO_RETRY_STATUSES — fail fast even if max_retries is large."""
    calls = 0

    async def fn():
        nonlocal calls
        calls += 1
        raise HTTPException(status_code=403, detail="forbidden")

    with patch.object(proxy_module, "max_retries", 5):
        with pytest.raises(HTTPException):
            await proxy_module._with_retry(
                fn, operation="test", request_id="t", base_delay_s=0.0
            )
    assert calls == 1


@pytest.mark.asyncio
async def test_retry_helper_max_retries_zero(proxy_module):
    """max_retries=0 means single attempt, no retry."""
    calls = 0

    async def fn():
        nonlocal calls
        calls += 1
        raise HTTPException(status_code=500, detail="boom")

    with patch.object(proxy_module, "max_retries", 0):
        with pytest.raises(HTTPException):
            await proxy_module._with_retry(
                fn, operation="test", request_id="t", base_delay_s=0.0
            )
    assert calls == 1


@pytest.mark.asyncio
async def test_retry_helper_max_retries_three(proxy_module):
    """max_retries=3 means up to 4 attempts total."""
    calls = 0

    async def fn():
        nonlocal calls
        calls += 1
        raise HTTPException(status_code=502, detail="bad gateway")

    with patch.object(proxy_module, "max_retries", 3):
        with pytest.raises(HTTPException):
            await proxy_module._with_retry(
                fn, operation="test", request_id="t", base_delay_s=0.0
            )
    assert calls == 4


@pytest.mark.asyncio
async def test_retry_helper_retries_on_network_exception(proxy_module):
    """asyncio.TimeoutError is a retryable network exception."""
    calls = 0

    async def fn():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise asyncio.TimeoutError()
        return "ok"

    with patch.object(proxy_module, "max_retries", 1):
        result = await proxy_module._with_retry(
            fn, operation="test", request_id="t", base_delay_s=0.0
        )
    assert result == "ok"
    assert calls == 2


@pytest.mark.asyncio
async def test_retry_helper_no_retry_on_unknown_exception(proxy_module):
    """An arbitrary non-retryable exception aborts immediately."""
    calls = 0

    async def fn():
        nonlocal calls
        calls += 1
        raise ValueError("nope")

    with patch.object(proxy_module, "max_retries", 5):
        with pytest.raises(ValueError):
            await proxy_module._with_retry(
                fn, operation="test", request_id="t", base_delay_s=0.0
            )
    assert calls == 1


# ---------- Fix 3: _with_retry OSError narrowing ----------
#
# Bare ``OSError`` is no longer caught — only the genuine network-flavoured
# ``ConnectionError`` subfamily (ConnectionResetError, ConnectionAbortedError,
# BrokenPipeError).  PermissionError / FileNotFoundError / etc. must fail
# fast because they're always fatal configuration problems, never transient.

@pytest.mark.asyncio
async def test_retry_helper_permission_error_fails_fast(proxy_module):
    """PermissionError is an OSError but NOT a ConnectionError — do not retry."""
    calls = 0

    async def fn():
        nonlocal calls
        calls += 1
        raise PermissionError("denied")

    with patch.object(proxy_module, "max_retries", 5):
        with pytest.raises(PermissionError):
            await proxy_module._with_retry(
                fn, operation="test", request_id="t", base_delay_s=0.0
            )
    assert calls == 1


@pytest.mark.asyncio
async def test_retry_helper_file_not_found_fails_fast(proxy_module):
    """FileNotFoundError is an OSError but NOT a ConnectionError — do not retry."""
    calls = 0

    async def fn():
        nonlocal calls
        calls += 1
        raise FileNotFoundError("no such file")

    with patch.object(proxy_module, "max_retries", 5):
        with pytest.raises(FileNotFoundError):
            await proxy_module._with_retry(
                fn, operation="test", request_id="t", base_delay_s=0.0
            )
    assert calls == 1


@pytest.mark.asyncio
async def test_retry_helper_retries_connection_reset_error(proxy_module):
    """ConnectionResetError (a genuine transport hiccup) should still retry."""
    calls = 0

    async def fn():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConnectionResetError("peer went away")
        return "ok"

    with patch.object(proxy_module, "max_retries", 1):
        result = await proxy_module._with_retry(
            fn, operation="test", request_id="t", base_delay_s=0.0
        )
    assert result == "ok"
    assert calls == 2


@pytest.mark.asyncio
async def test_retry_helper_retries_broken_pipe_error(proxy_module):
    """BrokenPipeError is a ConnectionError subclass — should retry."""
    calls = 0

    async def fn():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise BrokenPipeError("pipe closed")
        return "ok"

    with patch.object(proxy_module, "max_retries", 1):
        result = await proxy_module._with_retry(
            fn, operation="test", request_id="t", base_delay_s=0.0
        )
    assert result == "ok"
    assert calls == 2


# ---------- Config endpoint tests ----------

@pytest.fixture
def isolated_config(proxy_module):
    """Patch config load/save to avoid touching the real config.json."""
    store = {}
    with patch.object(proxy_module, "_load_config", lambda: dict(store)), \
         patch.object(proxy_module, "_save_config", lambda d: store.update(d)):
        yield store


def test_get_max_retries_returns_current_value(proxy_module, test_client):
    with patch.object(proxy_module, "max_retries", 2):
        r = test_client.get("/config/max-retries")
    assert r.status_code == 200
    assert r.json() == {"max_retries": 2}


def test_post_max_retries_updates_and_persists(proxy_module, test_client, isolated_config):
    original = proxy_module.max_retries
    try:
        r = test_client.post("/config/max-retries", json={"max_retries": 3})
        assert r.status_code == 200
        assert r.json() == {"status": "saved", "max_retries": 3}
        assert proxy_module.max_retries == 3
        assert isolated_config.get("max_retries") == 3
    finally:
        proxy_module.max_retries = original


def test_post_max_retries_rejects_negative(proxy_module, test_client, isolated_config):
    original = proxy_module.max_retries
    try:
        r = test_client.post("/config/max-retries", json={"max_retries": -1})
        assert r.status_code == 400
        assert proxy_module.max_retries == original
    finally:
        proxy_module.max_retries = original


def test_post_max_retries_rejects_too_large(proxy_module, test_client, isolated_config):
    original = proxy_module.max_retries
    try:
        r = test_client.post("/config/max-retries", json={"max_retries": 11})
        assert r.status_code == 400
        assert proxy_module.max_retries == original
    finally:
        proxy_module.max_retries = original


def test_post_max_retries_rejects_non_integer(proxy_module, test_client, isolated_config):
    original = proxy_module.max_retries
    try:
        r = test_client.post("/config/max-retries", json={"max_retries": "abc"})
        assert r.status_code == 400
        assert proxy_module.max_retries == original
    finally:
        proxy_module.max_retries = original


def test_post_max_retries_rejects_missing(proxy_module, test_client, isolated_config):
    r = test_client.post("/config/max-retries", json={})
    assert r.status_code == 400


def test_post_max_retries_accepts_zero_and_ten(proxy_module, test_client, isolated_config):
    original = proxy_module.max_retries
    try:
        r = test_client.post("/config/max-retries", json={"max_retries": 0})
        assert r.status_code == 200
        assert proxy_module.max_retries == 0

        r = test_client.post("/config/max-retries", json={"max_retries": 10})
        assert r.status_code == 200
        assert proxy_module.max_retries == 10
    finally:
        proxy_module.max_retries = original


# ---------- Provider integration: call_openrouter_direct ----------

class _FakeResponse:
    """Minimal async context manager mimicking aiohttp response."""
    def __init__(self, status, body):
        self.status = status
        self._body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def text(self):
        return self._body

    async def json(self):
        return json.loads(self._body)


class _FakeSession:
    """Records posts and returns scripted responses."""
    def __init__(self, responses):
        self._responses = list(responses)
        self.posts = 0

    def post(self, url, **kwargs):
        self.posts += 1
        return self._responses.pop(0)


@pytest.fixture
def fast_sleep():
    """Neutralise asyncio.sleep so retry-backoff doesn't slow tests."""
    original_sleep = asyncio.sleep

    async def _fast(_):
        await original_sleep(0)

    with patch("asyncio.sleep", _fast):
        yield


@pytest.mark.asyncio
async def test_openrouter_direct_retries_on_500_then_succeeds(proxy_module, fast_sleep):
    """500 then 200 -> helper retries once and returns the second body."""
    success_body = json.dumps({
        "choices": [{"message": {"content": "hello world"}}]
    })
    fake_session = _FakeSession([
        _FakeResponse(500, "upstream exploded"),
        _FakeResponse(200, success_body),
    ])

    with patch.object(proxy_module, "openrouter_api_key", "sk-or-fake"), \
         patch.object(proxy_module, "max_retries", 1), \
         patch.object(proxy_module, "third_party_session", fake_session), \
         patch.object(proxy_module, "auth", MagicMock(session=None)):
        result = await proxy_module.call_openrouter_direct(
            system_prompt=None,
            messages=[{"role": "user", "content": "hi"}],
            model="openai/gpt-4o",
            max_tokens=10,
        )
    assert result == "hello world"
    assert fake_session.posts == 2


@pytest.mark.asyncio
async def test_openrouter_direct_does_not_retry_with_max_retries_zero(proxy_module, fast_sleep):
    """max_retries=0 + single 500 -> raise immediately, no retry."""
    fake_session = _FakeSession([
        _FakeResponse(500, "upstream exploded"),
    ])

    with patch.object(proxy_module, "openrouter_api_key", "sk-or-fake"), \
         patch.object(proxy_module, "max_retries", 0), \
         patch.object(proxy_module, "third_party_session", fake_session), \
         patch.object(proxy_module, "auth", MagicMock(session=None)):
        with pytest.raises(HTTPException) as exc_info:
            await proxy_module.call_openrouter_direct(
                system_prompt=None,
                messages=[{"role": "user", "content": "hi"}],
                model="openai/gpt-4o",
                max_tokens=10,
            )
    assert exc_info.value.status_code == 500
    assert fake_session.posts == 1


@pytest.mark.asyncio
async def test_openrouter_direct_does_not_retry_on_401(proxy_module, fast_sleep):
    """401 is no-retry even with a large max_retries budget."""
    fake_session = _FakeSession([
        _FakeResponse(401, "unauthorized"),
    ])

    with patch.object(proxy_module, "openrouter_api_key", "sk-or-fake"), \
         patch.object(proxy_module, "max_retries", 5), \
         patch.object(proxy_module, "third_party_session", fake_session), \
         patch.object(proxy_module, "auth", MagicMock(session=None)):
        with pytest.raises(HTTPException) as exc_info:
            await proxy_module.call_openrouter_direct(
                system_prompt=None,
                messages=[{"role": "user", "content": "hi"}],
                model="openai/gpt-4o",
                max_tokens=10,
            )
    assert exc_info.value.status_code == 401
    assert fake_session.posts == 1


# ---------- New provider integration tests (added in master) ----------

@pytest.mark.asyncio
async def test_ollama_direct_retries_on_500_then_succeeds(proxy_module, fast_sleep):
    """Ollama: 500 then 200 with max_retries=1 -> succeeds on second attempt."""
    success_body = json.dumps({
        "choices": [{"message": {"content": "ollama says hi"}}]
    })
    fake_session = _FakeSession([
        _FakeResponse(500, "ollama boom"),
        _FakeResponse(200, success_body),
    ])

    with patch.object(proxy_module, "ollama_api_key", None), \
         patch.object(proxy_module, "ollama_session", fake_session), \
         patch.object(proxy_module, "max_retries", 1), \
         patch.object(proxy_module, "auth", MagicMock(session=None)):
        result = await proxy_module.call_ollama_direct(
            system_prompt=None,
            messages=[{"role": "user", "content": "hi"}],
            model="ollama:llama3",
            max_tokens=10,
        )
    assert result == "ollama says hi"
    assert fake_session.posts == 2


@pytest.mark.asyncio
async def test_fireworks_direct_does_not_retry_on_401(proxy_module, fast_sleep):
    """Fireworks: a 401 is auth failure and must fail fast, no retry."""
    fake_session = _FakeSession([
        _FakeResponse(401, "unauthorized"),
    ])

    with patch.object(proxy_module, "fireworks_api_key", "fw-fake"), \
         patch.object(proxy_module, "max_retries", 5), \
         patch.object(proxy_module, "third_party_session", fake_session), \
         patch.object(proxy_module, "auth", MagicMock(session=None)):
        with pytest.raises(HTTPException) as exc_info:
            await proxy_module.call_fireworks_direct(
                system_prompt=None,
                messages=[{"role": "user", "content": "hi"}],
                model="fireworks:kimi-k2p5-turbo",
                max_tokens=10,
            )
    assert exc_info.value.status_code == 401
    assert fake_session.posts == 1


# ---------- _load_max_retries clamping ----------

def test_load_max_retries_default(proxy_module):
    assert proxy_module._load_max_retries({}) == 1


def test_load_max_retries_clamps_negative(proxy_module):
    assert proxy_module._load_max_retries({"max_retries": -5}) == 0


def test_load_max_retries_clamps_huge(proxy_module):
    assert proxy_module._load_max_retries({"max_retries": 9999}) == 10


def test_load_max_retries_non_integer_falls_back_to_default(proxy_module):
    assert proxy_module._load_max_retries({"max_retries": "abc"}) == 1
    assert proxy_module._load_max_retries({"max_retries": None}) == 1
