"""Tests for reasoning/thinking parameter forwarding in _anthropic_to_oai_structured()."""
import pytest
from unittest.mock import patch


@pytest.fixture()
def structured(proxy_module):
    """Return the _anthropic_to_oai_structured function."""
    return proxy_module._anthropic_to_oai_structured


def _make_body(**overrides):
    """Minimal valid Anthropic /v1/messages body."""
    base = {
        "model": "fireworks:accounts/fireworks/models/llama-v3p1-8b-instruct",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Hello"}],
    }
    base.update(overrides)
    return base


class TestOverrideEnabled:
    """When reasoning_override_enabled is True, reasoning_effort must be injected."""

    def test_injects_reasoning_effort(self, proxy_module, structured):
        with patch.object(proxy_module, "reasoning_override_enabled", True), \
             patch.object(proxy_module, "reasoning_override_level", "high"):
            payload = structured(_make_body())
        assert payload["reasoning_effort"] == "high"

    def test_override_takes_precedence_over_caller_thinking(self, proxy_module, structured):
        """Even if the caller sends thinking, override wins."""
        body = _make_body(thinking={"type": "enabled", "budget_tokens": 32768})
        with patch.object(proxy_module, "reasoning_override_enabled", True), \
             patch.object(proxy_module, "reasoning_override_level", "low"):
            payload = structured(body)
        assert payload["reasoning_effort"] == "low"

    def test_no_thinking_dict_in_payload(self, proxy_module, structured):
        """OAI providers should never see an Anthropic thinking dict."""
        with patch.object(proxy_module, "reasoning_override_enabled", True), \
             patch.object(proxy_module, "reasoning_override_level", "medium"):
            payload = structured(_make_body())
        assert "thinking" not in payload


class TestCallerThinkingTranslation:
    """When override is OFF but caller sends thinking.type=enabled, translate to reasoning_effort."""

    def test_low_budget(self, proxy_module, structured):
        body = _make_body(thinking={"type": "enabled", "budget_tokens": 2048})
        with patch.object(proxy_module, "reasoning_override_enabled", False):
            payload = structured(body)
        assert payload["reasoning_effort"] == "low"

    def test_medium_budget(self, proxy_module, structured):
        body = _make_body(thinking={"type": "enabled", "budget_tokens": 8192})
        with patch.object(proxy_module, "reasoning_override_enabled", False):
            payload = structured(body)
        assert payload["reasoning_effort"] == "medium"

    def test_high_budget(self, proxy_module, structured):
        body = _make_body(thinking={"type": "enabled", "budget_tokens": 32768})
        with patch.object(proxy_module, "reasoning_override_enabled", False):
            payload = structured(body)
        assert payload["reasoning_effort"] == "high"

    def test_boundary_4096_is_low(self, proxy_module, structured):
        body = _make_body(thinking={"type": "enabled", "budget_tokens": 4096})
        with patch.object(proxy_module, "reasoning_override_enabled", False):
            payload = structured(body)
        assert payload["reasoning_effort"] == "low"

    def test_boundary_16384_is_medium(self, proxy_module, structured):
        body = _make_body(thinking={"type": "enabled", "budget_tokens": 16384})
        with patch.object(proxy_module, "reasoning_override_enabled", False):
            payload = structured(body)
        assert payload["reasoning_effort"] == "medium"

    def test_no_thinking_dict_forwarded(self, proxy_module, structured):
        body = _make_body(thinking={"type": "enabled", "budget_tokens": 10000})
        with patch.object(proxy_module, "reasoning_override_enabled", False):
            payload = structured(body)
        assert "thinking" not in payload


class TestNoThinking:
    """When override is OFF and no caller thinking, payload should have no reasoning params."""

    def test_no_reasoning_effort(self, proxy_module, structured):
        with patch.object(proxy_module, "reasoning_override_enabled", False):
            payload = structured(_make_body())
        assert "reasoning_effort" not in payload
        assert "thinking" not in payload

    def test_thinking_disabled_ignored(self, proxy_module, structured):
        """thinking.type=disabled should not produce reasoning_effort."""
        body = _make_body(thinking={"type": "disabled"})
        with patch.object(proxy_module, "reasoning_override_enabled", False):
            payload = structured(body)
        assert "reasoning_effort" not in payload

    def test_thinking_enabled_no_budget_tokens(self, proxy_module, structured):
        """thinking={type: enabled} with NO budget_tokens should not inject reasoning_effort."""
        body = _make_body(thinking={"type": "enabled"})
        with patch.object(proxy_module, "reasoning_override_enabled", False):
            payload = structured(body)
        assert "reasoning_effort" not in payload

    def test_thinking_string_not_dict(self, proxy_module, structured):
        """thinking='enabled' (string, not dict) should not inject reasoning_effort."""
        body = _make_body(thinking="enabled")
        with patch.object(proxy_module, "reasoning_override_enabled", False):
            payload = structured(body)
        assert "reasoning_effort" not in payload
