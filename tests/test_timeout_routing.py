"""Tests for _with_timeout_routing and its shared deadline helper.

The important invariant: the total wall-clock cap (`max_total_seconds`) must
apply to BOTH the original stream AND any fallback stream kicked off after a
Phase 1 TTFT timeout. Prior to the _stream_under_deadline refactor, the
fallback path iterated the fallback generator raw with no deadline, so a slow
fallback could blow past the cap (observed in production as 22s ollama calls
under an 8.5s cap).
"""
import asyncio
import time

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helper async generators used as fake provider streams
# ---------------------------------------------------------------------------


async def _never_content_gen():
    """Yields role-only chunks forever — trips the TTFT phase 1 timeout."""
    while True:
        yield 'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
        await asyncio.sleep(0.02)


async def _forever_content_gen():
    """First chunk has real content (passes phase 1), then streams forever."""
    yield 'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
    while True:
        yield 'data: {"choices":[{"delta":{"content":"..."}}]}\n\n'
        await asyncio.sleep(0.02)


async def _short_content_gen():
    """Well-behaved stream: one content chunk, then clean [DONE]."""
    yield 'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
    yield 'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
    yield "data: [DONE]\n\n"


async def _drain(agen, hard_cap):
    """Iterate an async generator into a list, refusing to wait longer than
    `hard_cap` seconds so a broken deadline doesn't hang the test suite."""
    chunks = []
    start = time.time()

    async def _loop():
        async for chunk in agen:
            chunks.append(chunk)

    await asyncio.wait_for(_loop(), timeout=hard_cap)
    return chunks, time.time() - start


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTimeoutRoutingTotalCap:

    @pytest.mark.asyncio
    async def test_fallback_is_deadline_bounded(self, proxy_module):
        """Regression test for the 22.3s ollama:gemma4 call.

        When Phase 1 TTFT times out and the fallback model also runs long,
        the wrapper must still cut at total_timeout and emit a synthetic [DONE]
        so the client doesn't retry.
        """
        ttft = 0.2
        total = 0.5

        primary = _never_content_gen()
        fallback = _forever_content_gen()

        with patch.object(proxy_module, "_pick_fallback_model", return_value="fake-fallback"), \
             patch.object(proxy_module, "_make_streaming_gen", return_value=fallback), \
             patch.object(proxy_module, "model_stats", MagicMock(record=MagicMock())):
            wrapped = proxy_module._with_timeout_routing(
                primary, None, [], [], "fake-primary", 256,
                ttft_timeout=ttft, total_timeout=total,
            )
            chunks, elapsed = await _drain(wrapped, hard_cap=total + 1.5)

        # Must have honoured the total cap (small slack for scheduling jitter).
        assert elapsed < total + 0.5, (
            f"wrapper ran {elapsed:.2f}s, expected <= {total + 0.5:.2f}s — "
            "fallback path is not deadline-bounded"
        )

        joined = "".join(chunks)
        assert "[DONE]" in joined, "missing synthetic [DONE] — client would retry"
        assert '"finish_reason": "stop"' in joined, "missing synthetic stop chunk"

    @pytest.mark.asyncio
    async def test_phase2_cuts_slow_original_stream(self, proxy_module):
        """Phase 2 still caps the original stream after a successful TTFT.

        ttft is generous so phase 1 passes on the first chunk; total is tight
        so phase 2 kicks in. Guards against regressions in the shared helper.
        """
        ttft = 2.0
        total = 0.3

        primary = _forever_content_gen()

        with patch.object(proxy_module, "model_stats", MagicMock(record=MagicMock())):
            wrapped = proxy_module._with_timeout_routing(
                primary, None, [], [], "fake-primary", 256,
                ttft_timeout=ttft, total_timeout=total,
            )
            chunks, elapsed = await _drain(wrapped, hard_cap=total + 1.5)

        assert elapsed < total + 0.5, (
            f"phase 2 ran {elapsed:.2f}s, expected <= {total + 0.5:.2f}s"
        )
        joined = "".join(chunks)
        assert "[DONE]" in joined
        assert '"finish_reason": "stop"' in joined

    @pytest.mark.asyncio
    async def test_clean_stream_passes_through_untouched(self, proxy_module):
        """A stream that terminates naturally before any deadline must flow
        through without a synthetic stop chunk injected."""
        gen = _short_content_gen()

        with patch.object(proxy_module, "model_stats", MagicMock(record=MagicMock())):
            wrapped = proxy_module._with_timeout_routing(
                gen, None, [], [], "fake-primary", 256,
                ttft_timeout=2.0, total_timeout=2.0,
            )
            chunks, _ = await _drain(wrapped, hard_cap=3.0)

        joined = "".join(chunks)
        assert '"content": "hi"' in joined or '"content":"hi"' in joined
        assert joined.count("[DONE]") == 1  # no duplicate synthetic [DONE]
        # The upstream [DONE] passes through; no synthetic stop chunk is added
        # (the upstream never emitted a finish_reason, so none should appear).
        assert "finish_reason" not in joined
