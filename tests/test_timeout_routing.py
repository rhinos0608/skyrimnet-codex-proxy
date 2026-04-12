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


async def _errors_before_content_gen():
    """Raises mid-iteration before yielding any real content — simulates a
    provider that 500s or a broken session that fails on first read."""
    # Yield once so the consumer's __anext__ actually advances the frame;
    # the exception is raised on the *next* pull, matching how real aiohttp
    # stream failures surface (you get some bytes, then a read error).
    yield 'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
    raise RuntimeError("simulated provider read error")


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


class TestTimeoutRoutingCascade:
    """After a Phase 1 TTFT timeout, the wrapper must cascade through multiple
    fallbacks instead of giving up after one. Regression for the 2026-04-12
    production log where `ollama:gemma4:31b-cloud` stalled, its only fallback
    `fireworks:kimi-k2p5-turbo` also stalled, and the client received an empty
    synthetic [DONE] — which caused SkyrimNet to retry and generate twice."""

    @pytest.mark.asyncio
    async def test_cascade_recovers_when_first_fallback_stalls(self, proxy_module):
        """Primary times out, fallback #1 never produces content within TTFT,
        fallback #2 returns real content. Client should see fb2's content."""
        ttft = 0.15
        total = 2.0  # generous so only per-fallback TTFT kicks in

        primary = _never_content_gen()
        fb1 = _never_content_gen()
        fb2 = _short_content_gen()

        gen_iter = iter([fb1, fb2])
        name_iter = iter(["fb1", "fb2"])

        def _pick(*args, **kwargs):
            return next(name_iter, None)

        def _make(*args, **kwargs):
            return next(gen_iter)

        with patch.object(proxy_module, "_pick_fallback_model", side_effect=_pick), \
             patch.object(proxy_module, "_make_streaming_gen", side_effect=_make), \
             patch.object(proxy_module, "model_stats", MagicMock(record=MagicMock())):
            wrapped = proxy_module._with_timeout_routing(
                primary, None, [], [], "fake-primary", 256,
                ttft_timeout=ttft, total_timeout=total,
            )
            chunks, _ = await _drain(wrapped, hard_cap=total + 1.0)

        joined = "".join(chunks)
        assert '"content": "hi"' in joined or '"content":"hi"' in joined, (
            "cascade did not recover — fb2's content never reached the client"
        )
        assert joined.count("[DONE]") == 1, "expected exactly one [DONE] terminator"

    @pytest.mark.asyncio
    async def test_cascade_exhausts_when_all_fallbacks_stall(self, proxy_module):
        """When every fallback stalls in Phase 1, the wrapper must emit a
        synthetic stop+[DONE] (not leak a truncated stream). This is the exact
        failure mode from the 2026-04-12 production log."""
        ttft = 0.1
        total = 1.0

        primary = _never_content_gen()
        fallbacks = [_never_content_gen() for _ in range(4)]
        gen_iter = iter(fallbacks)
        name_iter = iter(["fb1", "fb2", "fb3", "fb4"])

        def _pick(*args, **kwargs):
            return next(name_iter, None)

        def _make(*args, **kwargs):
            return next(gen_iter)

        with patch.object(proxy_module, "_pick_fallback_model", side_effect=_pick), \
             patch.object(proxy_module, "_make_streaming_gen", side_effect=_make), \
             patch.object(proxy_module, "model_stats", MagicMock(record=MagicMock())):
            wrapped = proxy_module._with_timeout_routing(
                primary, None, [], [], "fake-primary", 256,
                ttft_timeout=ttft, total_timeout=total,
            )
            chunks, elapsed = await _drain(wrapped, hard_cap=total + 1.5)

        assert elapsed < total + 0.5, (
            f"cascade exceeded total cap ({elapsed:.2f}s) — per-fallback TTFT "
            "bound is not honouring the global deadline"
        )
        joined = "".join(chunks)
        assert "[DONE]" in joined, "missing synthetic [DONE] — client would retry"
        assert '"finish_reason": "stop"' in joined, "missing synthetic stop chunk"

    @pytest.mark.asyncio
    async def test_cascade_skips_fallback_that_errors_before_content(self, proxy_module):
        """If a fallback raises before yielding real content, the cascade must
        record the failure and try the next candidate — not bubble the
        exception out (which would leave the client with no [DONE])."""
        ttft = 0.5
        total = 3.0

        primary = _never_content_gen()
        fb1 = _errors_before_content_gen()
        fb2 = _short_content_gen()

        gen_iter = iter([fb1, fb2])
        name_iter = iter(["fb1", "fb2"])

        def _pick(*args, **kwargs):
            return next(name_iter, None)

        def _make(*args, **kwargs):
            return next(gen_iter)

        with patch.object(proxy_module, "_pick_fallback_model", side_effect=_pick), \
             patch.object(proxy_module, "_make_streaming_gen", side_effect=_make), \
             patch.object(proxy_module, "model_stats", MagicMock(record=MagicMock())):
            wrapped = proxy_module._with_timeout_routing(
                primary, None, [], [], "fake-primary", 256,
                ttft_timeout=ttft, total_timeout=total,
            )
            chunks, _ = await _drain(wrapped, hard_cap=total + 1.0)

        joined = "".join(chunks)
        assert '"content": "hi"' in joined or '"content":"hi"' in joined, (
            "cascade did not recover from fb1's pre-content error"
        )
        assert joined.count("[DONE]") == 1

    @pytest.mark.asyncio
    async def test_cascade_threads_extra_params_to_fallbacks(self, proxy_module):
        """Fallback stream construction must preserve suppression params."""
        ttft = 0.1
        total = 1.0
        extra_params = {"thinking": {"type": "disabled"}}

        primary = _never_content_gen()
        fallback = _short_content_gen()

        with patch.object(proxy_module, "_pick_fallback_model", return_value="fb1"), \
             patch.object(proxy_module, "_make_streaming_gen", return_value=fallback) as mock_make, \
             patch.object(proxy_module, "model_stats", MagicMock(record=MagicMock())):
            wrapped = proxy_module._with_timeout_routing(
                primary, "sys", [{"role": "user", "content": "hi"}],
                [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
                "fake-primary", 256,
                ttft_timeout=ttft, total_timeout=total,
                extra_params=extra_params,
            )
            chunks, _ = await _drain(wrapped, hard_cap=total + 1.0)

        joined = "".join(chunks)
        assert '"content": "hi"' in joined or '"content":"hi"' in joined
        mock_make.assert_called_once_with(
            "sys",
            [{"role": "user", "content": "hi"}],
            "fb1",
            256,
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
            extra_params=extra_params,
        )

    @pytest.mark.asyncio
    async def test_cascade_preserves_extra_params_for_fallbacks(self, proxy_module):
        """Fallback generators must inherit request controls like thinking or
        reasoning suppression instead of dropping them during timeout routing."""
        ttft = 0.1
        total = 1.0

        primary = _never_content_gen()
        captured = []

        def _pick(*args, **kwargs):
            return "fb1"

        def _make(system_prompt, messages, model, max_tokens, oai_messages=None, extra_params=None):
            captured.append(extra_params)
            return _short_content_gen()

        with patch.object(proxy_module, "_pick_fallback_model", side_effect=_pick), \
             patch.object(proxy_module, "_make_streaming_gen", side_effect=_make), \
             patch.object(proxy_module, "model_stats", MagicMock(record=MagicMock())):
            wrapped = proxy_module._with_timeout_routing(
                primary,
                None,
                [],
                [],
                "fake-primary",
                256,
                extra_params={"thinking": {"type": "disabled"}},
                ttft_timeout=ttft,
                total_timeout=total,
            )
            chunks, _ = await _drain(wrapped, hard_cap=total + 1.0)

        assert captured == [{"thinking": {"type": "disabled"}}]
        assert '"content": "hi"' in "".join(chunks) or '"content":"hi"' in "".join(chunks)

    @pytest.mark.asyncio
    async def test_cascade_preserves_extra_params_for_fallback(self, proxy_module):
        """Fallback streams must inherit suppression flags from the original request."""
        ttft = 0.1
        total = 1.0

        primary = _never_content_gen()
        captured = {}

        def _make(system_prompt, messages, model, max_tokens, oai_messages=None, extra_params=None):
            captured["model"] = model
            captured["extra_params"] = extra_params
            return _short_content_gen()

        with patch.object(proxy_module, "_pick_fallback_model", return_value="fb1"), \
             patch.object(proxy_module, "_make_streaming_gen", side_effect=_make), \
             patch.object(proxy_module, "model_stats", MagicMock(record=MagicMock())):
            wrapped = proxy_module._with_timeout_routing(
                primary, None, [], [], "fake-primary", 256,
                ttft_timeout=ttft,
                total_timeout=total,
                extra_params={"thinking": {"type": "disabled"}},
            )
            chunks, _ = await _drain(wrapped, hard_cap=total + 1.0)

        assert captured["model"] == "fb1"
        assert captured["extra_params"] == {"thinking": {"type": "disabled"}}
        assert "[DONE]" in "".join(chunks)
