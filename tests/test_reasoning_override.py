"""Tests for the dashboard reasoning override + Codex effort helper.

The override lets dashboard operators flip thinking/reasoning back ON for
every request, overriding the caller's value. This is the primary fix for
CCS-routed Claude Code, which otherwise lands here with ``thinking`` absent
and gets silently downgraded to ``thinking=disabled`` by the request
pipeline.
"""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# _override_thinking_payload — budget mapping per level
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("level,expected_budget", [
    ("low", 2048),
    ("medium", 8192),
    ("high", 24576),
])
def test_override_thinking_payload_budget_per_level(proxy_module, level, expected_budget):
    with patch.object(proxy_module, "reasoning_override_level", level):
        payload = proxy_module._override_thinking_payload()
    assert payload == {"type": "enabled", "budget_tokens": expected_budget}


@pytest.mark.parametrize("level,max_tokens,expected_budget", [
    # max_tokens=0 -> unclamped (default path)
    ("low", 0, 2048),
    ("medium", 0, 8192),
    ("high", 0, 24576),
    # max_tokens > budget -> unclamped
    ("low", 50000, 2048),
    ("medium", 50000, 8192),
    ("high", 50000, 24576),
    # max_tokens < budget -> clamps to max_tokens - 1 (floor 1024)
    ("high", 2000, 1999),  # 24576 clamped to max(1024, 1999) = 1999
    ("high", 1024, 1024),  # 24576 clamped to max(1024, 1023) = 1024
    ("medium", 1024, 1024),  # 8192 clamped to max(1024, 1023) = 1024
    # max_tokens below floor -> floor at 1024
    ("high", 500, 1024),  # max(1024, 499) = 1024
    ("medium", 100, 1024),  # max(1024, 99) = 1024
    ("low", 1025, 1024),  # max(1024, 1024) = 1024, min(2048, 1024) = 1024
])
def test_override_thinking_payload_clamp_edge_cases(proxy_module, level, max_tokens, expected_budget):
    with patch.object(proxy_module, "reasoning_override_level", level):
        payload = proxy_module._override_thinking_payload(max_tokens)
    assert payload == {"type": "enabled", "budget_tokens": expected_budget}


def test_override_thinking_payload_unknown_level_falls_back_to_medium(proxy_module):
    with patch.object(proxy_module, "reasoning_override_level", "bogus"):
        payload = proxy_module._override_thinking_payload()
    # Unknown level -> 8192 (medium) by table default.
    assert payload == {"type": "enabled", "budget_tokens": 8192}


# ---------------------------------------------------------------------------
# _active_codex_reasoning_effort — override vs constant
# ---------------------------------------------------------------------------


def test_codex_effort_uses_constant_when_override_disabled(proxy_module):
    with patch.object(proxy_module, "reasoning_override_enabled", False):
        assert proxy_module._active_codex_reasoning_effort() == proxy_module.CODEX_FAST_REASONING_EFFORT


@pytest.mark.parametrize("level", ["low", "medium", "high"])
def test_codex_effort_uses_override_when_enabled(proxy_module, level):
    with patch.object(proxy_module, "reasoning_override_enabled", True), \
         patch.object(proxy_module, "reasoning_override_level", level):
        assert proxy_module._active_codex_reasoning_effort() == level


# ---------------------------------------------------------------------------
# Request-pipeline precedence (unit): caller wins when override is off,
# override wins when it's on.
#
# These reproduce the injection block inline as a fast sanity check.
# The _production tests below exercise the real _chat_completions_inner path.
# ---------------------------------------------------------------------------


def _apply_thinking_precedence(proxy_module, extra_params: dict) -> dict:
    """Mirror the inlined logic at proxy.py:_chat_completions_inner."""
    if proxy_module.reasoning_override_enabled:
        extra_params["thinking"] = proxy_module._override_thinking_payload()
        extra_params["reasoning_effort"] = proxy_module.reasoning_override_level
    else:
        _caller_thinking = extra_params.get("thinking")
        if _caller_thinking is None:
            extra_params.setdefault("thinking", {"type": "disabled"})
    return extra_params


def test_precedence_unit_override_off_caller_omits_defaults_to_disabled(proxy_module):
    with patch.object(proxy_module, "reasoning_override_enabled", False):
        out = _apply_thinking_precedence(proxy_module, {})
    assert out["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in out


def test_precedence_unit_override_off_caller_wins(proxy_module):
    caller = {"type": "enabled", "budget_tokens": 10000}
    with patch.object(proxy_module, "reasoning_override_enabled", False):
        out = _apply_thinking_precedence(proxy_module, {"thinking": dict(caller)})
    assert out["thinking"] == caller


def test_precedence_unit_override_on_beats_caller(proxy_module):
    caller = {"type": "enabled", "budget_tokens": 10000}
    with patch.object(proxy_module, "reasoning_override_enabled", True), \
         patch.object(proxy_module, "reasoning_override_level", "high"):
        out = _apply_thinking_precedence(proxy_module, {"thinking": dict(caller)})
    assert out["thinking"] == {"type": "enabled", "budget_tokens": 24576}
    assert out["reasoning_effort"] == "high"


def test_precedence_unit_override_on_beats_caller_disabled(proxy_module):
    """Even an explicit thinking=disabled from the caller loses to the override."""
    with patch.object(proxy_module, "reasoning_override_enabled", True), \
         patch.object(proxy_module, "reasoning_override_level", "low"):
        out = _apply_thinking_precedence(proxy_module, {"thinking": {"type": "disabled"}})
    assert out["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert out["reasoning_effort"] == "low"


# ---------------------------------------------------------------------------
# Request-pipeline precedence (production): exercise the REAL code path in
# _chat_completions_inner by POSTing to /v1/chat/completions with a mocked
# provider. Uses Fireworks because use_extra=True so kwargs (including
# ``thinking``) are forwarded to the provider function.
# ---------------------------------------------------------------------------


async def _fake_fireworks_stream(*args, **kwargs):
    """Minimal async generator that satisfies the streaming pipeline."""
    yield 'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
    yield "data: [DONE]\n\n"


def _extract_provider_kwargs(mock_fn):
    """Extract the **kwargs dict from the mock provider's call args.

    Provider streaming functions are called as:
        stream_fn(system_prompt, messages, model, max_tokens, **kwargs)
    so positional args are [0..3] and keyword args are the rest.
    """
    assert mock_fn.call_count == 1, f"Expected 1 call, got {mock_fn.call_count}"
    _, kwargs = mock_fn.call_args
    return kwargs


def test_precedence_override_off_caller_omits_production(proxy_module, test_client):
    """Override OFF, no thinking in request -> provider sees thinking=disabled."""
    mock_stream = MagicMock(side_effect=_fake_fireworks_stream)
    with patch.object(proxy_module, "reasoning_override_enabled", False), \
         patch.object(proxy_module, "fireworks_api_key", "fake-key"), \
         patch.object(proxy_module, "call_fireworks_streaming", mock_stream), \
         patch.object(proxy_module, "timeout_routing_enabled", False), \
         patch.object(proxy_module, "request_stats", MagicMock(record=MagicMock())):
        resp = test_client.post("/v1/chat/completions", json={
            "model": "fireworks:accounts/fireworks/models/kimi-k2p5-turbo",
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 4096,
            "stream": True,
        })
    assert resp.status_code == 200
    kw = _extract_provider_kwargs(mock_stream)
    assert kw["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in kw


def test_precedence_override_off_caller_wins_production(proxy_module, test_client):
    """Override OFF, caller sends thinking -> provider sees caller's value."""
    caller_thinking = {"type": "enabled", "budget_tokens": 10000}
    mock_stream = MagicMock(side_effect=_fake_fireworks_stream)
    with patch.object(proxy_module, "reasoning_override_enabled", False), \
         patch.object(proxy_module, "fireworks_api_key", "fake-key"), \
         patch.object(proxy_module, "call_fireworks_streaming", mock_stream), \
         patch.object(proxy_module, "timeout_routing_enabled", False), \
         patch.object(proxy_module, "request_stats", MagicMock(record=MagicMock())):
        resp = test_client.post("/v1/chat/completions", json={
            "model": "fireworks:accounts/fireworks/models/kimi-k2p5-turbo",
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 4096,
            "stream": True,
            "thinking": caller_thinking,
        })
    assert resp.status_code == 200
    kw = _extract_provider_kwargs(mock_stream)
    assert kw["thinking"] == caller_thinking


def test_precedence_override_on_beats_caller_production(proxy_module, test_client):
    """Override ON at high, caller sends thinking -> provider sees override."""
    caller_thinking = {"type": "enabled", "budget_tokens": 10000}
    mock_stream = MagicMock(side_effect=_fake_fireworks_stream)
    # Use large max_tokens so the budget clamp does not reduce the value.
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
            "thinking": caller_thinking,
        })
    assert resp.status_code == 200
    kw = _extract_provider_kwargs(mock_stream)
    assert kw["thinking"] == {"type": "enabled", "budget_tokens": 24576}
    assert kw["reasoning_effort"] == "high"


def test_precedence_override_on_beats_disabled_production(proxy_module, test_client):
    """Override ON at low, caller sends thinking=disabled -> provider sees override."""
    mock_stream = MagicMock(side_effect=_fake_fireworks_stream)
    # Use large max_tokens so the budget clamp does not reduce the value.
    with patch.object(proxy_module, "reasoning_override_enabled", True), \
         patch.object(proxy_module, "reasoning_override_level", "low"), \
         patch.object(proxy_module, "fireworks_api_key", "fake-key"), \
         patch.object(proxy_module, "call_fireworks_streaming", mock_stream), \
         patch.object(proxy_module, "timeout_routing_enabled", False), \
         patch.object(proxy_module, "request_stats", MagicMock(record=MagicMock())):
        resp = test_client.post("/v1/chat/completions", json={
            "model": "fireworks:accounts/fireworks/models/kimi-k2p5-turbo",
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 32768,
            "stream": True,
            "thinking": {"type": "disabled"},
        })
    assert resp.status_code == 200
    kw = _extract_provider_kwargs(mock_stream)
    assert kw["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert kw["reasoning_effort"] == "low"


# ---------------------------------------------------------------------------
# Config loader: invalid level on disk falls back to "medium"
# ---------------------------------------------------------------------------


def test_invalid_level_rejected_by_config_endpoint(proxy_module, test_client):
    """POST /config/reasoning-level rejects an invalid level string."""
    original = proxy_module.reasoning_override_level
    try:
        r = test_client.post("/config/reasoning-level", json={"level": "totally-bogus"})
        assert r.status_code == 400
        # Module-level attribute unchanged.
        assert proxy_module.reasoning_override_level == original
    finally:
        proxy_module.reasoning_override_level = original


# ---------------------------------------------------------------------------
# /config/reasoning-override + /config/reasoning-level endpoints
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_config(proxy_module):
    """Patch config load/save so we don't touch the real config.json."""
    store = {}
    with patch.object(proxy_module, "_load_config", lambda: dict(store)), \
         patch.object(proxy_module, "_save_config", lambda d: store.update(d)):
        yield store


def test_post_reasoning_override_enables(proxy_module, test_client, isolated_config):
    original = proxy_module.reasoning_override_enabled
    try:
        r = test_client.post("/config/reasoning-override", json={"enabled": True})
        assert r.status_code == 200
        data = r.json()
        assert data["reasoning_override_enabled"] is True
        assert proxy_module.reasoning_override_enabled is True
        assert isolated_config.get("reasoning_override_enabled") is True
    finally:
        proxy_module.reasoning_override_enabled = original


def test_post_reasoning_override_disables(proxy_module, test_client, isolated_config):
    original = proxy_module.reasoning_override_enabled
    try:
        proxy_module.reasoning_override_enabled = True
        r = test_client.post("/config/reasoning-override", json={"enabled": False})
        assert r.status_code == 200
        assert r.json()["reasoning_override_enabled"] is False
        assert proxy_module.reasoning_override_enabled is False
    finally:
        proxy_module.reasoning_override_enabled = original


def test_post_reasoning_override_rejects_missing(proxy_module, test_client, isolated_config):
    r = test_client.post("/config/reasoning-override", json={})
    assert r.status_code == 400


def test_post_reasoning_level_accepts_valid(proxy_module, test_client, isolated_config):
    original = proxy_module.reasoning_override_level
    try:
        for level in ("low", "medium", "high"):
            r = test_client.post("/config/reasoning-level", json={"level": level})
            assert r.status_code == 200
            assert r.json()["reasoning_override_level"] == level
            assert proxy_module.reasoning_override_level == level
    finally:
        proxy_module.reasoning_override_level = original


def test_post_reasoning_level_rejects_invalid(proxy_module, test_client, isolated_config):
    original = proxy_module.reasoning_override_level
    try:
        r = test_client.post("/config/reasoning-level", json={"level": "extreme"})
        assert r.status_code == 400
        assert proxy_module.reasoning_override_level == original
    finally:
        proxy_module.reasoning_override_level = original


def test_post_reasoning_level_rejects_non_string(proxy_module, test_client, isolated_config):
    r = test_client.post("/config/reasoning-level", json={"level": 5})
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# _is_claude_default_model — /v1/messages routing decision
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", [
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    "claude-haiku-4-5-20251001",
    "some-unknown-string",  # defaults to Claude
])
def test_is_claude_default_model_true_for_claude(proxy_module, model):
    assert proxy_module._is_claude_default_model(model) is True


@pytest.mark.parametrize("model", [
    "ollama:llama3.2",
    "openai/gpt-4o",
    "gpt-5.1-codex",
    "antigravity-gemini-2.5-flash",
    "gcli-gemini-2.5-flash",
    "zai:glm-4.5",
    "fireworks:kimi-k2p5-turbo",
    "nvidia:deepseek-r1",
    "qwen:qwen3-coder-plus",
    "opencode:big-pickle",
    "xiaomi:mimo-1-50b",
])
def test_is_claude_default_model_false_for_other_providers(proxy_module, model):
    assert proxy_module._is_claude_default_model(model) is False


# ---------------------------------------------------------------------------
# /config/reasoning-rewrite endpoint
# ---------------------------------------------------------------------------


def test_post_reasoning_rewrite_enables(proxy_module, test_client, isolated_config):
    original = proxy_module.reasoning_rewrite_enabled
    try:
        r = test_client.post("/config/reasoning-rewrite", json={"enabled": True})
        assert r.status_code == 200
        assert r.json()["reasoning_rewrite_enabled"] is True
        assert proxy_module.reasoning_rewrite_enabled is True
    finally:
        proxy_module.reasoning_rewrite_enabled = original


def test_post_reasoning_rewrite_disables(proxy_module, test_client, isolated_config):
    original = proxy_module.reasoning_rewrite_enabled
    try:
        proxy_module.reasoning_rewrite_enabled = True
        r = test_client.post("/config/reasoning-rewrite", json={"enabled": False})
        assert r.status_code == 200
        assert r.json()["reasoning_rewrite_enabled"] is False
    finally:
        proxy_module.reasoning_rewrite_enabled = original


def test_post_reasoning_rewrite_rejects_missing(proxy_module, test_client, isolated_config):
    r = test_client.post("/config/reasoning-rewrite", json={})
    assert r.status_code == 400
