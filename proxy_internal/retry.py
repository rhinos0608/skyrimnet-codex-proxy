"""Shared retry policy for OpenAI-compatible upstreams."""

import asyncio
import logging
import sys

import aiohttp
from fastapi import HTTPException

logger = logging.getLogger("proxy")


def _current_max_retries() -> int:
    """Resolve max_retries from the top-level proxy module at call time.

    Reading via sys.modules lets ``patch.object(proxy, "max_retries", N)``
    in tests take effect without rebinding any local name captured at
    import time.
    """
    proxy_module = sys.modules.get("proxy")
    if proxy_module is None:
        return 1
    return getattr(proxy_module, "max_retries", 1)


_DEFAULT_RETRY_STATUSES: frozenset = frozenset({429, 500, 502, 503, 504})
_NO_RETRY_STATUSES: frozenset = frozenset({401, 403})


async def _with_retry(
    fn,
    *,
    operation: str,
    request_id: str,
    retry_on_status: frozenset = _DEFAULT_RETRY_STATUSES,
    base_delay_s: float = 0.5,
):
    """Call ``fn()`` (async, zero-arg) up to ``max_retries + 1`` times.

    Retries on:
      - network exceptions: ``aiohttp.ClientConnectorError``,
        ``aiohttp.ServerDisconnectedError``, ``asyncio.TimeoutError``,
        ``ConnectionError`` (parent of ``ConnectionResetError``,
        ``ConnectionAbortedError``, and ``BrokenPipeError``)
      - ``HTTPException`` whose ``status_code`` is in ``retry_on_status``
        (default: 429, 500, 502, 503, 504)

    Does NOT retry:
      - ``HTTPException`` with status_code in ``_NO_RETRY_STATUSES``
        (401/403 — auth is broken, fail fast)
      - any other 4xx
      - **bare ``OSError``** is deliberately NOT retried — a
        ``PermissionError`` or ``FileNotFoundError`` is always fatal
        (misconfiguration, not a transient upstream hiccup) and retrying
        them would just mask the root cause.  Only the network-flavoured
        ``ConnectionError`` subfamily above is retried.
      - any other exception type

    Between attempts, sleeps ``base_delay_s * 2**attempt`` seconds (exponential
    backoff, no jitter). On the final attempt the last exception is re-raised.

    Note: the backoff schedule is computed from ``attempt`` alone (no wall-
    clock arithmetic), so there is no need for ``time.monotonic()`` inside
    this helper.  If future changes add elapsed-time accounting, prefer
    ``time.monotonic()`` over ``time.time()`` so NTP adjustments can't cause
    negative delays.
    """
    attempts = _current_max_retries() + 1
    last_exc = None
    for attempt in range(attempts):
        try:
            return await fn()
        except HTTPException as e:
            last_exc = e
            if e.status_code in _NO_RETRY_STATUSES:
                raise
            if e.status_code not in retry_on_status:
                raise
            if attempt + 1 >= attempts:
                raise
            delay = base_delay_s * (2 ** attempt)
            logger.warning(
                f"[{request_id}] {operation} HTTP {e.status_code} — retry "
                f"{attempt + 1}/{_current_max_retries()} in {delay:.1f}s"
            )
            await asyncio.sleep(delay)
        except (aiohttp.ClientConnectorError, aiohttp.ServerDisconnectedError,
                asyncio.TimeoutError, ConnectionError) as e:
            # ConnectionError covers ConnectionResetError, ConnectionAbortedError,
            # and BrokenPipeError — all genuine transport hiccups.  Note we do
            # NOT catch bare OSError here (see docstring).
            last_exc = e
            if attempt + 1 >= attempts:
                raise
            delay = base_delay_s * (2 ** attempt)
            logger.warning(
                f"[{request_id}] {operation} {type(e).__name__} — retry "
                f"{attempt + 1}/{_current_max_retries()} in {delay:.1f}s"
            )
            await asyncio.sleep(delay)
    # Unreachable — loop either returns or raises. Keep type-checkers happy.
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"{operation}: retry loop exited without result")


async def _open_stream_with_retry(
    session,
    url: str,
    *,
    json_payload: dict,
    headers: dict,
    operation: str,
    request_id: str,
    retry_on_status: frozenset = _DEFAULT_RETRY_STATUSES,
    base_delay_s: float = 0.5,
    retry_connect_errors: bool = True,
):
    """Open a streaming POST, retrying only the pre-first-byte phase.

    Returns a tuple ``(resp, resp_cm)`` of the live aiohttp response and its
    context-manager handle — the caller is responsible for ``await
    resp_cm.__aexit__(None, None, None)`` once the body has been consumed.

    If every attempt fails with a retryable status / network error, the final
    ``HTTPException`` (or network exception wrapped as HTTPException 502) is
    raised so the caller can surface an SSE error chunk via
    ``yield_sse_error``.  401/403 fail fast; 4xx other than 429 fails fast.

    Pass ``retry_connect_errors=False`` to let ``aiohttp.ClientConnectorError``
    propagate immediately (used by Ollama where a connect error means the
    local daemon is down, not a transient failure).
    """
    attempts = _current_max_retries() + 1
    for attempt in range(attempts):
        try:
            resp_cm = session.post(url, json=json_payload, headers=headers)
            resp = await resp_cm.__aenter__()
        except aiohttp.ClientConnectorError:
            if not retry_connect_errors:
                raise
            if attempt + 1 >= attempts:
                raise HTTPException(status_code=502, detail=f"{operation} connect error")
            delay = base_delay_s * (2 ** attempt)
            logger.warning(
                f"[{request_id}] {operation} stream ClientConnectorError — retry "
                f"{attempt + 1}/{_current_max_retries()} in {delay:.1f}s"
            )
            await asyncio.sleep(delay)
            continue
        except (aiohttp.ServerDisconnectedError, asyncio.TimeoutError,
                ConnectionError) as e:
            # Mirror _with_retry's narrowing: only genuine transport errors
            # (ConnectionError = ConnectionResetError / ConnectionAbortedError
            # / BrokenPipeError) are retried.  Bare OSError (e.g.
            # PermissionError, FileNotFoundError) is fatal and must propagate.
            if attempt + 1 >= attempts:
                raise HTTPException(status_code=502, detail=f"{operation} network error: {type(e).__name__}")
            delay = base_delay_s * (2 ** attempt)
            logger.warning(
                f"[{request_id}] {operation} stream {type(e).__name__} — retry "
                f"{attempt + 1}/{_current_max_retries()} in {delay:.1f}s"
            )
            await asyncio.sleep(delay)
            continue
        if resp.status == 200:
            return resp, resp_cm
        # Non-200: classify and maybe retry
        error_body = await resp.text()
        status = resp.status
        logger.error(f"[{request_id}] {operation} {status}: {error_body[:300]}")
        try:
            await resp_cm.__aexit__(None, None, None)
        except Exception:
            pass
        if (
            status in _NO_RETRY_STATUSES
            or status not in retry_on_status
            or attempt + 1 >= attempts
        ):
            raise HTTPException(status_code=status, detail=error_body[:200])
        delay = base_delay_s * (2 ** attempt)
        logger.warning(
            f"[{request_id}] {operation} stream HTTP {status} — retry "
            f"{attempt + 1}/{_current_max_retries()} in {delay:.1f}s"
        )
        await asyncio.sleep(delay)
    # Unreachable
    raise RuntimeError(f"{operation}: stream retry loop exited without result")
