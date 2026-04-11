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

# ---------------------------------------------------------------------------
# Fix 5: Qwen refresh_if_needed offloads reload_from_file to a thread
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_qwen_refresh_if_needed_runs_reload_in_executor(proxy_module):
    """``reload_from_file`` is synchronous disk I/O; ``refresh_if_needed``
    must schedule it via ``asyncio.to_thread`` so the event loop stays
    responsive.  We verify the result propagates correctly and the event
    loop yields at least once during the reload (a direct sync call would
    not yield)."""
    cache = proxy_module.QwenAuthCache()
    _make_expired(cache)

    loop = asyncio.get_running_loop()
    call_threads: list[int] = []

    def fake_reload():
        import threading
        call_threads.append(threading.get_ident())
        # simulate small disk latency
        import time as _time
        _time.sleep(0.02)
        cache.access_token = "qwen-token-from-thread"
        _make_fresh(cache)
        return True

    cache.reload_from_file = fake_reload

    main_thread_id = __import__("threading").get_ident()
    result = await cache.refresh_if_needed()

    assert result is True
    assert cache.access_token == "qwen-token-from-thread"
    # The sync reload should have executed on a worker thread, NOT on the
    # main thread — proving we didn't block the loop.
    assert len(call_threads) == 1
    assert call_threads[0] != main_thread_id


# ---------------------------------------------------------------------------
# Fix 4: AntigravityAuthCache.refresh_if_needed snapshots accounts list
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_antigravity_refresh_concurrent_add_remove_account(proxy_module):
    """A concurrent ``add_account``/``remove_account`` call mid-refresh
    must not raise or cause a silent skip.  The refresh should still
    complete successfully for every account that was present at entry."""
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

    refresh_events: list[str] = []

    async def slow_refresh_a():
        refresh_events.append("a-start")
        await asyncio.sleep(0.05)  # window during which add/remove may fire
        acct1.access_token = "tok-a"
        _make_fresh(acct1)
        refresh_events.append("a-done")
        return True

    async def slow_refresh_b():
        refresh_events.append("b-start")
        await asyncio.sleep(0.05)
        acct2.access_token = "tok-b"
        _make_fresh(acct2)
        refresh_events.append("b-done")
        return True

    acct1._do_refresh = slow_refresh_a
    acct2._do_refresh = slow_refresh_b

    class _StubSession:
        async def close(self):
            pass

    # Pre-populate sessions to avoid create_session() during refresh, and
    # give them an async close() so remove_account can cleanly dispose.
    acct1.session = _StubSession()
    acct2.session = _StubSession()

    async def mutate_accounts():
        # Let refresh_if_needed start iterating and grab its snapshot.
        await asyncio.sleep(0.01)
        # Simulate a user adding a new account mid-refresh.
        acct3 = proxy_module.AntigravityAccount()
        acct3.refresh_token = "rt-3"
        acct3.email = "c@example.com"
        _make_fresh(acct3)
        acct3.access_token = "tok-c-preexisting"
        acct3.session = _StubSession()
        cache.add_account(acct3)
        # And removing one of the in-flight accounts.
        cache.remove_account("b@example.com")

    # Run both concurrently — snapshot means neither mutation disrupts
    # the in-flight refresh.
    result, _ = await asyncio.gather(
        cache.refresh_if_needed(),
        mutate_accounts(),
    )

    assert result is True
    # Both snapshot accounts completed their refresh despite the mutation.
    assert "a-done" in refresh_events
    assert "b-done" in refresh_events
    assert acct1.access_token == "tok-a"
    assert acct2.access_token == "tok-b"


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
