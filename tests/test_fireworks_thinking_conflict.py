"""Tests that Fireworks payload never sends both 'thinking' and 'reasoning_effort'.

Fireworks API rejects requests that specify both fields simultaneously (400:
"cannot specify both 'thinking' and 'reasoning_effort'").  These tests verify
the mutual-exclusion invariant at the payload-fixup level and end-to-end.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Unit tests: _fireworks_payload_fixup directly
# ---------------------------------------------------------------------------


def test_fireworks_fixup_strips_thinking_when_reasoning_effort_present(proxy_module):
    """When both thinking and reasoning_effort are in extra_params, payload
    must NOT contain 'thinking' — only 'reasoning_effort' should survive."""
    from proxy_internal.providers.fireworks import _fireworks_payload_fixup

    payload = {"model": "test", "messages": []}
    extra_params = {
        "thinking": {"type": "enabled", "budget_tokens": 8192},
        "reasoning_effort": "high",
        "temperature": 0.7,
    }
    _fireworks_payload_fixup(payload, extra_params)

    assert "reasoning_effort" in payload
    assert payload["reasoning_effort"] == "high"
    assert "thinking" not in payload, (
        f"payload must not contain 'thinking' when 'reasoning_effort' is present; "
        f"got thinking={payload.get('thinking')}"
    )


def test_fireworks_fixup_strips_thinking_alone(proxy_module):
    """When only thinking (no reasoning_effort) is in extra_params, it should
    still be stripped — Fireworks doesn't support the Anthropic thinking param."""
    from proxy_internal.providers.fireworks import _fireworks_payload_fixup

    payload = {"model": "test", "messages": []}
    extra_params = {
        "thinking": {"type": "enabled", "budget_tokens": 8192},
    }
    _fireworks_payload_fixup(payload, extra_params)

    assert "thinking" not in payload


def test_fireworks_fixup_reasoning_disabled_sets_no_thinking_when_effort_present(proxy_module):
    """When reasoning.enabled=false AND reasoning_effort is present, the payload
    must not get 'thinking' injected — use reasoning_effort only."""
    from proxy_internal.providers.fireworks import _fireworks_payload_fixup

    payload = {"model": "test", "messages": []}
    extra_params = {
        "reasoning": {"enabled": False},
        "reasoning_effort": "medium",
    }
    _fireworks_payload_fixup(payload, extra_params)

    assert "thinking" not in payload
    assert "reasoning_effort" in payload


def test_fireworks_fixup_reasoning_disabled_without_effort(proxy_module):
    """When reasoning.enabled=false and NO reasoning_effort, set thinking=disabled."""
    from proxy_internal.providers.fireworks import _fireworks_payload_fixup

    payload = {"model": "test", "messages": []}
    extra_params = {
        "reasoning": {"enabled": False},
    }
    _fireworks_payload_fixup(payload, extra_params)

    assert payload.get("thinking") == {"type": "disabled"}


def test_fireworks_fixup_thinking_disabled_without_effort(proxy_module):
    """When thinking=disabled and NO reasoning_effort, set thinking=disabled."""
    from proxy_internal.providers.fireworks import _fireworks_payload_fixup

    payload = {"model": "test", "messages": []}
    extra_params = {
        "thinking": {"type": "disabled"},
    }
    _fireworks_payload_fixup(payload, extra_params)

    assert payload.get("thinking") == {"type": "disabled"}


def test_fireworks_fixup_never_has_both_thinking_and_effort(proxy_module):
    """Exhaustive check: no combination of extra_params produces both fields."""
    from proxy_internal.providers.fireworks import _fireworks_payload_fixup

    combos = [
        # (thinking, reasoning_effort, reasoning)
        ({"type": "enabled", "budget_tokens": 8192}, "high", None),
        ({"type": "enabled", "budget_tokens": 8192}, "medium", None),
        ({"type": "enabled", "budget_tokens": 8192}, "low", None),
        ({"type": "disabled"}, "high", None),
        ({"type": "disabled"}, None, None),
        ({"type": "enabled", "budget_tokens": 8192}, None, {"enabled": False}),
        ({"type": "disabled"}, "low", {"enabled": False}),
        (None, "high", None),
        (None, None, {"enabled": False}),
    ]

    for thinking, effort, reasoning in combos:
        payload = {"model": "test", "messages": []}
        extra = {}
        if thinking is not None:
            extra["thinking"] = thinking
        if effort is not None:
            extra["reasoning_effort"] = effort
        if reasoning is not None:
            extra["reasoning"] = reasoning

        _fireworks_payload_fixup(payload, extra)

        has_thinking = "thinking" in payload
        has_effort = "reasoning_effort" in payload
        assert not (has_thinking and has_effort), (
            f"Both 'thinking' and 'reasoning_effort' in payload for "
            f"thinking={thinking}, effort={effort}, reasoning={reasoning}"
        )


# ---------------------------------------------------------------------------
# Production (end-to-end) test: reasoning override → Fireworks payload
# ---------------------------------------------------------------------------


async def _fake_fireworks_stream(*args, **kwargs):
    yield 'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
    yield "data: [DONE]\n\n"


def _extract_provider_kwargs(mock_fn):
    assert mock_fn.call_count == 1
    _, kwargs = mock_fn.call_args
    return kwargs


def test_fireworks_pipeline_kwargs_are_cleaned_by_fixup(proxy_module, test_client):
    """Reasoning override ON → Fireworks receives both in kwargs, but the
    payload fixup strips 'thinking' before it reaches the upstream API.
    This verifies the kwargs pass through (pipeline level) and the fixup
    removes the conflict (provider level)."""
    from proxy_internal.providers.fireworks import _fireworks_payload_fixup

    mock_stream = MagicMock(side_effect=_fake_fireworks_stream)
    with patch.object(proxy_module, "reasoning_override_enabled", True), \
         patch.object(proxy_module, "reasoning_override_level", "high"), \
         patch.object(proxy_module, "fireworks_api_key", "fake-key"), \
         patch.object(proxy_module, "call_fireworks_streaming", mock_stream), \
         patch.object(proxy_module, "timeout_routing_enabled", False), \
         patch.object(proxy_module, "request_stats", MagicMock(record=MagicMock())):
        resp = test_client.post("/v1/chat/completions", json={
            "model": "fireworks:accounts/fireworks/models/kimi-k2p5-turbo",
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 32768,
            "stream": True,
        })
    assert resp.status_code == 200
    # The pipeline passes both — that's expected (provider fixup handles it)
    kw = _extract_provider_kwargs(mock_stream)
    assert "thinking" in kw, "pipeline should pass thinking through"
    assert "reasoning_effort" in kw, "pipeline should pass reasoning_effort through"

    # But the payload fixup must resolve the conflict
    payload = {"model": "test", "messages": []}
    _fireworks_payload_fixup(payload, kw)
    has_thinking = "thinking" in payload
    has_effort = "reasoning_effort" in payload
    assert not (has_thinking and has_effort), (
        f"Payload after fixup has both thinking={payload.get('thinking')} and "
        f"reasoning_effort={payload.get('reasoning_effort')}"
    )
