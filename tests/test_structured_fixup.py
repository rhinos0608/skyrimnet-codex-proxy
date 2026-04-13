"""Tests for _structured_payload_fixup — provider-specific param stripping in the structured path."""

import copy
import pytest


@pytest.fixture()
def fixup(proxy_module):
    return proxy_module._structured_payload_fixup


# --- Base payload used across tests ---

def _base_payload():
    return {
        "model": "some-model",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1024,
        "reasoning_effort": "medium",
        "thinking": {"type": "enabled", "budget_tokens": 5000},
        "reasoning": {"enabled": True},
        "top_k": 40,
        "provider": {"order": ["fallback"]},
        "temperature": 0.7,
    }


# --- Always-strip: thinking & reasoning removed for every provider ---

@pytest.mark.parametrize("provider", [
    "Ollama", "Fireworks", "OpenRouter", "Z.AI", "Xiaomi",
    "OpenCode Zen", "Qwen", "NVIDIA NIM",
])
def test_thinking_and_reasoning_always_stripped(fixup, provider):
    payload = _base_payload()
    fixup(payload, provider)
    assert "thinking" not in payload
    assert "reasoning" not in payload


@pytest.mark.parametrize("provider", [
    "Fireworks", "OpenRouter", "Z.AI", "Xiaomi",
    "OpenCode Zen", "Qwen", "NVIDIA NIM",
])
def test_reasoning_effort_preserved(fixup, provider):
    """reasoning_effort is a standard OpenAI param — most providers should keep it."""
    payload = _base_payload()
    fixup(payload, provider)
    assert payload["reasoning_effort"] == "medium"


def test_ollama_strips_reasoning_effort(fixup):
    """Ollama doesn't support reasoning_effort — it must be stripped."""
    payload = _base_payload()
    fixup(payload, "Ollama")
    assert "reasoning_effort" not in payload


# --- Fireworks thinking=disabled re-injection ---

def test_fireworks_preserves_thinking_disabled(fixup):
    """When caller sends thinking={type: disabled} with no reasoning_effort,
    the disabled marker should survive stripping."""
    payload = {
        "model": "fireworks:deepseek-r1",
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": {"type": "disabled"},
    }
    fixup(payload, "Fireworks")
    assert payload["thinking"] == {"type": "disabled"}


def test_fireworks_strips_thinking_when_reasoning_effort_present(fixup):
    """reasoning_effort takes precedence — thinking should be fully stripped."""
    payload = {
        "model": "fireworks:deepseek-r1",
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": {"type": "disabled"},
        "reasoning_effort": "medium",
    }
    fixup(payload, "Fireworks")
    assert "thinking" not in payload


def test_fireworks_reasoning_disabled_injects_thinking_disabled(fixup):
    """When caller sends reasoning={enabled: false} with no reasoning_effort,
    it should be translated into thinking={type: disabled}."""
    payload = {
        "model": "fireworks:deepseek-r1",
        "messages": [{"role": "user", "content": "hi"}],
        "reasoning": {"enabled": False},
    }
    fixup(payload, "Fireworks")
    assert payload["thinking"] == {"type": "disabled"}
    assert "reasoning" not in payload


# --- Provider-specific extra strips ---

def test_ollama_strips_top_k(fixup):
    payload = _base_payload()
    fixup(payload, "Ollama")
    assert "top_k" not in payload


def test_fireworks_strips_top_k_and_provider(fixup):
    payload = _base_payload()
    fixup(payload, "Fireworks")
    assert "top_k" not in payload
    assert "provider" not in payload


def test_nvidia_strips_top_k_and_provider(fixup):
    payload = _base_payload()
    fixup(payload, "NVIDIA NIM")
    assert "top_k" not in payload
    assert "provider" not in payload


# --- Providers that keep top_k and provider ---

@pytest.mark.parametrize("provider", ["OpenRouter", "Z.AI", "Xiaomi", "OpenCode Zen", "Qwen"])
def test_other_providers_keep_top_k_and_provider(fixup, provider):
    payload = _base_payload()
    fixup(payload, provider)
    assert payload.get("top_k") == 40
    assert payload.get("provider") == {"order": ["fallback"]}


# --- Non-reasoning fields are never touched ---

def test_standard_fields_untouched(fixup):
    payload = _base_payload()
    fixup(payload, "Fireworks")
    assert payload["model"] == "some-model"
    assert payload["messages"] == [{"role": "user", "content": "hi"}]
    assert payload["max_tokens"] == 1024
    assert payload["temperature"] == 0.7


# --- Idempotent when fields already absent ---

def test_fixup_no_op_on_clean_payload(fixup):
    payload = {"model": "x", "messages": [], "max_tokens": 512}
    original = copy.deepcopy(payload)
    fixup(payload, "Ollama")
    assert payload == original


# --- Unknown provider treated conservatively (only always-strip) ---

def test_unknown_provider_strips_only_thinking_and_reasoning(fixup):
    payload = _base_payload()
    fixup(payload, "SomeNewProvider")
    assert "thinking" not in payload
    assert "reasoning" not in payload
    # Everything else kept
    assert payload.get("top_k") == 40
    assert payload.get("provider") == {"order": ["fallback"]}
    assert payload["reasoning_effort"] == "medium"
