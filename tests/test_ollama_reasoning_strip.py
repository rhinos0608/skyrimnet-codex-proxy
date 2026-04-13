"""Tests that reasoning-related params are stripped from Ollama payloads."""

from proxy_internal.providers.ollama import _ollama_payload_fixup, _OLLAMA_UNSUPPORTED_PARAMS


def _make_payload():
    return {
        "model": "llama3",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 512,
    }


def test_reasoning_effort_stripped():
    """reasoning_effort must be removed — Ollama API doesn't support it."""
    payload = _make_payload()
    extra = {"reasoning_effort": "high", "temperature": 0.7}
    _ollama_payload_fixup(payload, extra)
    assert "reasoning_effort" not in payload


def test_thinking_and_reasoning_stripped():
    """thinking and reasoning dicts must also be stripped."""
    payload = _make_payload()
    extra = {
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "reasoning": {"effort": "high"},
        "reasoning_effort": "medium",
    }
    _ollama_payload_fixup(payload, extra)
    assert "thinking" not in payload
    assert "reasoning" not in payload
    assert "reasoning_effort" not in payload


def test_regular_params_preserved():
    """Standard params like temperature, top_p must survive fixup."""
    payload = _make_payload()
    extra = {"temperature": 0.5, "top_p": 0.9, "reasoning_effort": "low"}
    _ollama_payload_fixup(payload, extra)
    assert payload["temperature"] == 0.5
    assert payload["top_p"] == 0.9
    assert payload["model"] == "llama3"
    assert payload["max_tokens"] == 512
    assert "reasoning_effort" not in payload


def test_unsupported_params_set_includes_reasoning_effort():
    """Verify the constant itself contains reasoning_effort."""
    assert "reasoning_effort" in _OLLAMA_UNSUPPORTED_PARAMS
