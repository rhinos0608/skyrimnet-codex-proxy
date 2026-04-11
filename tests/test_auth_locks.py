"""Concurrent-refresh safety tests for auth cache classes.

Each auth cache that refreshes tokens now uses an ``asyncio.Lock`` with a
double-checked ``is_expired()`` guard. These tests verify:

1. When many coroutines hit ``refresh_if_needed`` at once while the cache is
   expired, the underlying refresh path runs **exactly once**, and every caller
   sees a valid token afterward.
2. When the cache is NOT expired, ``refresh_if_needed`` returns immediately
   without touching the underlying refresh path.

The tests construct each cache class directly (no module-level patching) so
they are hermetic and don't mutate the shared proxy module state used by the
rest of the test suite.
"""
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_expired(cache, field: str = "expires_at") -> None:
    """Force is_expired() to return True by setting expires_at into the past."""
    setattr(cache, field, datetime.now() - timedelta(hours=1))


def _make_fresh(cache, field: str = "expires_at") -> None:
    """Force is_expired() to return False by setting expires_at well into the future."""
    setattr(cache, field, datetime.now() + timedelta(hours=1))


# ---------------------------------------------------------------------------
# CodexAuthCache
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_codex_auth_cache_concurrent_refresh_runs_once(proxy_module):
    cache = proxy_module.CodexAuthCache()
    cache.refresh_token = "rt-codex"
    _make_expired(cache)

    call_count = 0

    async def fake_do_refresh():
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.05)  # simulate network latency
        cache.access_token = "new-access-token"
        _make_fresh(cache)
        return True

    cache._do_refresh = fake_do_refresh

    results = await asyncio.gather(*[cache.refresh_if_needed() for _ in range(10)])

    assert call_count == 1, f"Expected exactly 1 refresh call, got {call_count}"
    assert all(results), "All concurrent callers should see success"
    assert cache.access_token == "new-access-token"


@pytest.mark.asyncio
async def test_codex_auth_cache_not_expired_skips_refresh(proxy_module):
    cache = proxy_module.CodexAuthCache()
    cache.refresh_token = "rt-codex"
    cache.access_token = "still-valid"
    _make_fresh(cache)

    mock_refresh = AsyncMock(return_value=True)
    cache._do_refresh = mock_refresh

    result = await cache.refresh_if_needed()

    assert result is True
    mock_refresh.assert_not_called()
    assert cache.access_token == "still-valid"


# ---------------------------------------------------------------------------
# GeminiAuthCache
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_gemini_auth_cache_concurrent_refresh_runs_once(proxy_module):
    cache = proxy_module.GeminiAuthCache()
    cache.refresh_token = "rt-gemini"
    _make_expired(cache)

    call_count = 0

    async def fake_do_refresh():
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.05)
        cache.access_token = "new-gemini-token"
        _make_fresh(cache)
        return True

    cache._do_refresh = fake_do_refresh

    results = await asyncio.gather(*[cache.refresh_if_needed() for _ in range(10)])

    assert call_count == 1, f"Expected exactly 1 refresh call, got {call_count}"
    assert all(results)
    assert cache.access_token == "new-gemini-token"


@pytest.mark.asyncio
async def test_gemini_auth_cache_not_expired_skips_refresh(proxy_module):
    cache = proxy_module.GeminiAuthCache()
    cache.refresh_token = "rt-gemini"
    cache.access_token = "still-valid"
    _make_fresh(cache)

    mock_refresh = AsyncMock(return_value=True)
    cache._do_refresh = mock_refresh

    result = await cache.refresh_if_needed()

    assert result is True
    mock_refresh.assert_not_called()
    assert cache.access_token == "still-valid"


# ---------------------------------------------------------------------------
# QwenAuthCache
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_qwen_auth_cache_concurrent_refresh_runs_once(proxy_module):
    cache = proxy_module.QwenAuthCache()
    _make_expired(cache)

    call_count = 0

    def fake_reload():
        nonlocal call_count
        call_count += 1
        cache.access_token = "new-qwen-token"
        _make_fresh(cache)
        return True

    cache.reload_from_file = fake_reload

    # 10 concurrent callers — the first one to acquire the lock runs the
    # reload; the remaining 9 hit the double-check and return without calling
    # reload_from_file a second time.
    results = await asyncio.gather(*[cache.refresh_if_needed() for _ in range(10)])

    assert call_count == 1, f"Expected exactly 1 reload call, got {call_count}"
    assert all(results)
    assert cache.access_token == "new-qwen-token"


@pytest.mark.asyncio
async def test_qwen_auth_cache_not_expired_skips_refresh(proxy_module):
    cache = proxy_module.QwenAuthCache()
    cache.access_token = "still-valid"
    _make_fresh(cache)

    from unittest.mock import MagicMock
    mock_reload = MagicMock(return_value=True)
    cache.reload_from_file = mock_reload

    result = await cache.refresh_if_needed()

    assert result is True
    mock_reload.assert_not_called()
    assert cache.access_token == "still-valid"


# ---------------------------------------------------------------------------
# AntigravityAccount  (refresh lives on the per-account object)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_antigravity_account_concurrent_refresh_runs_once(proxy_module):
    account = proxy_module.AntigravityAccount()
    account.refresh_token = "rt-antigravity"
    account.email = "test@example.com"
    _make_expired(account)

    call_count = 0

    async def fake_do_refresh():
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.05)
        account.access_token = "new-antigravity-token"
        _make_fresh(account)
        return True

    account._do_refresh = fake_do_refresh

    results = await asyncio.gather(*[account.refresh_if_needed() for _ in range(10)])

    assert call_count == 1, f"Expected exactly 1 refresh call, got {call_count}"
    assert all(results)
    assert account.access_token == "new-antigravity-token"


@pytest.mark.asyncio
async def test_antigravity_account_not_expired_skips_refresh(proxy_module):
    account = proxy_module.AntigravityAccount()
    account.refresh_token = "rt-antigravity"
    account.access_token = "still-valid"
    account.email = "test@example.com"
    _make_fresh(account)

    mock_refresh = AsyncMock(return_value=True)
    account._do_refresh = mock_refresh

    result = await account.refresh_if_needed()

    assert result is True
    mock_refresh.assert_not_called()
    assert account.access_token == "still-valid"


# ---------------------------------------------------------------------------
# AntigravityAuthCache wraps per-account refresh — verify its fan-out also
# benefits from the per-account lock (each account refreshes at most once).
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_antigravity_auth_cache_fanout_locks_per_account(proxy_module):
    cache = proxy_module.AntigravityAuthCache()

    acct1 = proxy_module.AntigravityAccount()
    acct1.refresh_token = "rt-1"
    acct1.email = "a@example.com"
    _make_expired(acct1)

    acct2 = proxy_module.AntigravityAccount()
    acct2.refresh_token = "rt-2"
    acct2.email = "b@example.com"
    _make_expired(acct2)

    cache.accounts = [acct1, acct2]

    counts = {"a": 0, "b": 0}

    async def make_fake(tag, acct):
        async def fake_do_refresh():
            counts[tag] += 1
            await asyncio.sleep(0.05)
            acct.access_token = f"token-{tag}"
            _make_fresh(acct)
            return True
        return fake_do_refresh

    acct1._do_refresh = await make_fake("a", acct1)
    acct2._do_refresh = await make_fake("b", acct2)

    # Avoid create_session() side-effects inside refresh_if_needed by
    # pre-populating sessions with sentinels.
    acct1.session = object()
    acct2.session = object()

    # 5 concurrent cache-level refreshes — each should refresh each account
    # exactly once, since the per-account lock dedupes concurrent callers.
    results = await asyncio.gather(*[cache.refresh_if_needed() for _ in range(5)])

    assert all(results), "Cache refresh should succeed when accounts are ready"
    assert counts["a"] == 1, f"Account A refreshed {counts['a']} times, expected 1"
    assert counts["b"] == 1, f"Account B refreshed {counts['b']} times, expected 1"
    assert acct1.access_token == "token-a"
    assert acct2.access_token == "token-b"
