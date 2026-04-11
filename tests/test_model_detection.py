"""Tests for model detection functions and normalize_model_name().

Covers all 7 provider detection functions: is_ollama_model, is_openrouter_model,
is_codex_model, is_antigravity_model, is_gemini_cli_model, is_zai_model, and
the routing priority chain ensuring correct provider selection when model names
could match multiple detectors.
"""
import pytest


# ---------------------------------------------------------------------------
# is_ollama_model
# ---------------------------------------------------------------------------

class TestIsOllamaModel:
    def test_simple_model(self, proxy_module):
        assert proxy_module.is_ollama_model("ollama:llama3.2") is True

    def test_with_tag(self, proxy_module):
        assert proxy_module.is_ollama_model("ollama:mistral:7b") is True

    def test_with_slash(self, proxy_module):
        assert proxy_module.is_ollama_model("ollama:namespace/model") is True

    def test_case_insensitive_upper(self, proxy_module):
        assert proxy_module.is_ollama_model("OLLAMA:llama3") is True

    def test_case_insensitive_mixed(self, proxy_module):
        assert proxy_module.is_ollama_model("Ollama:Llama3") is True

    def test_rejects_openrouter(self, proxy_module):
        assert proxy_module.is_ollama_model("openai/gpt-4o") is False

    def test_rejects_claude(self, proxy_module):
        assert proxy_module.is_ollama_model("claude-sonnet-4-6") is False

    def test_rejects_codex(self, proxy_module):
        assert proxy_module.is_ollama_model("codex-mini-latest") is False

    def test_rejects_antigravity(self, proxy_module):
        assert proxy_module.is_ollama_model("antigravity-gemini-2.5-pro") is False

    def test_rejects_gemini_cli(self, proxy_module):
        assert proxy_module.is_ollama_model("gcli-gemini-2.5-flash") is False

    def test_rejects_zai(self, proxy_module):
        assert proxy_module.is_ollama_model("zai:gpt-4o") is False

    def test_rejects_empty_string(self, proxy_module):
        assert proxy_module.is_ollama_model("") is False

    def test_rejects_partial_prefix(self, proxy_module):
        assert proxy_module.is_ollama_model("ollam:llama3") is False

    def test_rejects_prefix_without_colon(self, proxy_module):
        assert proxy_module.is_ollama_model("ollamallama3") is False


# ---------------------------------------------------------------------------
# is_openrouter_model
# ---------------------------------------------------------------------------

class TestIsOpenrouterModel:
    def test_standard_model(self, proxy_module):
        assert proxy_module.is_openrouter_model("openai/gpt-4o") is True

    def test_nested_namespace(self, proxy_module):
        assert proxy_module.is_openrouter_model("meta-llama/llama-3-70b") is True

    def test_anthropic_model(self, proxy_module):
        assert proxy_module.is_openrouter_model("anthropic/claude-3.5-sonnet") is True

    def test_rejects_claude(self, proxy_module):
        assert proxy_module.is_openrouter_model("claude-sonnet-4-6") is False

    def test_rejects_codex(self, proxy_module):
        assert proxy_module.is_openrouter_model("codex-mini-latest") is False

    def test_rejects_ollama(self, proxy_module):
        assert proxy_module.is_openrouter_model("ollama:llama3") is False

    def test_rejects_antigravity(self, proxy_module):
        assert proxy_module.is_openrouter_model("antigravity-gemini-2.5-pro") is False

    def test_rejects_gemini_cli(self, proxy_module):
        assert proxy_module.is_openrouter_model("gcli-gemini-2.5-flash") is False

    def test_rejects_zai(self, proxy_module):
        assert proxy_module.is_openrouter_model("zai:gpt-4o") is False

    def test_rejects_empty_string(self, proxy_module):
        assert proxy_module.is_openrouter_model("") is False

    def test_slash_at_start(self, proxy_module):
        """Leading slash still contains '/' so technically matches."""
        assert proxy_module.is_openrouter_model("/model") is True

    def test_slash_at_end(self, proxy_module):
        assert proxy_module.is_openrouter_model("model/") is True


# ---------------------------------------------------------------------------
# is_codex_model
# ---------------------------------------------------------------------------

class TestIsCodexModel:
    def test_gpt5_dot(self, proxy_module):
        assert proxy_module.is_codex_model("gpt-5.4") is True

    def test_gpt5_dash(self, proxy_module):
        assert proxy_module.is_codex_model("gpt-5-mini") is True

    def test_codex_prefix(self, proxy_module):
        assert proxy_module.is_codex_model("codex-mini-latest") is True

    def test_case_insensitive_gpt5(self, proxy_module):
        assert proxy_module.is_codex_model("GPT-5.4") is True

    def test_case_insensitive_codex(self, proxy_module):
        assert proxy_module.is_codex_model("CODEX-MINI-LATEST") is True

    def test_case_insensitive_mixed(self, proxy_module):
        assert proxy_module.is_codex_model("Codex-Mini") is True

    def test_rejects_gpt4(self, proxy_module):
        assert proxy_module.is_codex_model("gpt-4o") is False

    def test_rejects_claude(self, proxy_module):
        assert proxy_module.is_codex_model("claude-sonnet-4-6") is False

    def test_rejects_ollama(self, proxy_module):
        assert proxy_module.is_codex_model("ollama:llama3") is False

    def test_rejects_openrouter(self, proxy_module):
        assert proxy_module.is_codex_model("openai/gpt-4o") is False

    def test_rejects_antigravity(self, proxy_module):
        assert proxy_module.is_codex_model("antigravity-gemini-2.5-pro") is False

    def test_rejects_gemini_cli(self, proxy_module):
        assert proxy_module.is_codex_model("gcli-gemini-2.5-flash") is False

    def test_rejects_zai(self, proxy_module):
        assert proxy_module.is_codex_model("zai:gpt-4o") is False

    def test_rejects_empty_string(self, proxy_module):
        assert proxy_module.is_codex_model("") is False

    def test_rejects_gpt5_without_separator(self, proxy_module):
        """'gpt-5x' should not match because there is no dot or dash after '5'."""
        assert proxy_module.is_codex_model("gpt-5x") is False

    def test_rejects_codex_without_dash(self, proxy_module):
        assert proxy_module.is_codex_model("codexmini") is False


# ---------------------------------------------------------------------------
# is_antigravity_model
# ---------------------------------------------------------------------------

class TestIsAntigravityModel:
    def test_antigravity_prefix(self, proxy_module):
        assert proxy_module.is_antigravity_model("antigravity-gemini-2.5-pro") is True

    def test_gemini_3_prefix(self, proxy_module):
        assert proxy_module.is_antigravity_model("gemini-3-flash") is True

    def test_gemini_25_prefix(self, proxy_module):
        assert proxy_module.is_antigravity_model("gemini-2.5-flash") is True

    def test_gpt_oss_prefix(self, proxy_module):
        assert proxy_module.is_antigravity_model("gpt-oss-mini") is True

    def test_case_insensitive_antigravity(self, proxy_module):
        assert proxy_module.is_antigravity_model("ANTIGRAVITY-GEMINI-2.5-PRO") is True

    def test_case_insensitive_gemini3(self, proxy_module):
        assert proxy_module.is_antigravity_model("GEMINI-3-FLASH") is True

    def test_case_insensitive_gemini25(self, proxy_module):
        assert proxy_module.is_antigravity_model("Gemini-2.5-Flash") is True

    def test_case_insensitive_gpt_oss(self, proxy_module):
        assert proxy_module.is_antigravity_model("GPT-OSS-mini") is True

    def test_rejects_gemini_2_non_25(self, proxy_module):
        """gemini-2.0 should not match antigravity (only 2.5+ and 3+)."""
        assert proxy_module.is_antigravity_model("gemini-2.0-flash") is False

    def test_rejects_claude(self, proxy_module):
        assert proxy_module.is_antigravity_model("claude-sonnet-4-6") is False

    def test_rejects_ollama(self, proxy_module):
        assert proxy_module.is_antigravity_model("ollama:llama3") is False

    def test_rejects_openrouter(self, proxy_module):
        assert proxy_module.is_antigravity_model("openai/gpt-4o") is False

    def test_rejects_codex(self, proxy_module):
        assert proxy_module.is_antigravity_model("codex-mini-latest") is False

    def test_rejects_gemini_cli(self, proxy_module):
        assert proxy_module.is_antigravity_model("gcli-gemini-2.5-flash") is False

    def test_rejects_zai(self, proxy_module):
        assert proxy_module.is_antigravity_model("zai:gpt-4o") is False

    def test_rejects_empty_string(self, proxy_module):
        assert proxy_module.is_antigravity_model("") is False


# ---------------------------------------------------------------------------
# is_gemini_cli_model
# ---------------------------------------------------------------------------

class TestIsGeminiCliModel:
    def test_standard_model(self, proxy_module):
        assert proxy_module.is_gemini_cli_model("gcli-gemini-2.5-flash") is True

    def test_another_model(self, proxy_module):
        assert proxy_module.is_gemini_cli_model("gcli-gemini-2.5-pro") is True

    def test_case_insensitive(self, proxy_module):
        assert proxy_module.is_gemini_cli_model("GCLI-gemini-2.5-flash") is True

    def test_case_insensitive_mixed(self, proxy_module):
        assert proxy_module.is_gemini_cli_model("Gcli-Model") is True

    def test_rejects_claude(self, proxy_module):
        assert proxy_module.is_gemini_cli_model("claude-sonnet-4-6") is False

    def test_rejects_ollama(self, proxy_module):
        assert proxy_module.is_gemini_cli_model("ollama:llama3") is False

    def test_rejects_openrouter(self, proxy_module):
        assert proxy_module.is_gemini_cli_model("openai/gpt-4o") is False

    def test_rejects_codex(self, proxy_module):
        assert proxy_module.is_gemini_cli_model("codex-mini-latest") is False

    def test_rejects_antigravity(self, proxy_module):
        assert proxy_module.is_gemini_cli_model("antigravity-gemini-2.5-pro") is False

    def test_rejects_zai(self, proxy_module):
        assert proxy_module.is_gemini_cli_model("zai:gpt-4o") is False

    def test_rejects_empty_string(self, proxy_module):
        assert proxy_module.is_gemini_cli_model("") is False

    def test_rejects_gcli_without_dash(self, proxy_module):
        assert proxy_module.is_gemini_cli_model("gclimodel") is False


# ---------------------------------------------------------------------------
# is_zai_model
# ---------------------------------------------------------------------------

class TestIsZaiModel:
    def test_standard_model(self, proxy_module):
        assert proxy_module.is_zai_model("zai:gpt-4o") is True

    def test_another_model(self, proxy_module):
        assert proxy_module.is_zai_model("zai:claude-sonnet") is True

    def test_case_insensitive(self, proxy_module):
        assert proxy_module.is_zai_model("ZAI:gpt-4o") is True

    def test_case_insensitive_mixed(self, proxy_module):
        assert proxy_module.is_zai_model("Zai:Model") is True

    def test_rejects_claude(self, proxy_module):
        assert proxy_module.is_zai_model("claude-sonnet-4-6") is False

    def test_rejects_ollama(self, proxy_module):
        assert proxy_module.is_zai_model("ollama:llama3") is False

    def test_rejects_openrouter(self, proxy_module):
        assert proxy_module.is_zai_model("openai/gpt-4o") is False

    def test_rejects_codex(self, proxy_module):
        assert proxy_module.is_zai_model("codex-mini-latest") is False

    def test_rejects_antigravity(self, proxy_module):
        assert proxy_module.is_zai_model("antigravity-gemini-2.5-pro") is False

    def test_rejects_gemini_cli(self, proxy_module):
        assert proxy_module.is_zai_model("gcli-gemini-2.5-flash") is False

    def test_rejects_empty_string(self, proxy_module):
        assert proxy_module.is_zai_model("") is False

    def test_rejects_zai_without_colon(self, proxy_module):
        assert proxy_module.is_zai_model("zaimodel") is False


# ---------------------------------------------------------------------------
# Routing priority chain
# ---------------------------------------------------------------------------

class TestRoutingPriority:
    """Verify that the routing priority (Ollama checked first, etc.) works
    correctly even for ambiguous model names like ollama:namespace/model
    which contains a slash that would match OpenRouter."""

    def _classify(self, proxy_module, model: str) -> str:
        """Simulate the routing priority chain from proxy.py and return
        the provider name that would handle this model."""
        if proxy_module.is_ollama_model(model):
            return "ollama"
        if proxy_module.is_codex_model(model):
            return "codex"
        if proxy_module.is_antigravity_model(model):
            return "antigravity"
        if proxy_module.is_gemini_cli_model(model):
            return "gemini_cli"
        if proxy_module.is_zai_model(model):
            return "zai"
        if proxy_module.is_openrouter_model(model):
            return "openrouter"
        return "claude"

    def test_ollama_with_slash_routes_to_ollama(self, proxy_module):
        """ollama:namespace/model contains '/' but must route to Ollama, not OpenRouter."""
        assert self._classify(proxy_module, "ollama:namespace/model") == "ollama"

    def test_plain_slash_routes_to_openrouter(self, proxy_module):
        assert self._classify(proxy_module, "openai/gpt-4o") == "openrouter"

    def test_codex_routes_correctly(self, proxy_module):
        assert self._classify(proxy_module, "gpt-5.4") == "codex"

    def test_codex_dash_routes_correctly(self, proxy_module):
        assert self._classify(proxy_module, "gpt-5-mini") == "codex"

    def test_codex_prefix_routes_correctly(self, proxy_module):
        assert self._classify(proxy_module, "codex-mini-latest") == "codex"

    def test_antigravity_routes_correctly(self, proxy_module):
        assert self._classify(proxy_module, "antigravity-gemini-2.5-pro") == "antigravity"

    def test_gemini25_routes_to_antigravity(self, proxy_module):
        assert self._classify(proxy_module, "gemini-2.5-flash") == "antigravity"

    def test_gemini3_routes_to_antigravity(self, proxy_module):
        assert self._classify(proxy_module, "gemini-3-flash") == "antigravity"

    def test_gpt_oss_routes_to_antigravity(self, proxy_module):
        assert self._classify(proxy_module, "gpt-oss-mini") == "antigravity"

    def test_gemini_cli_routes_correctly(self, proxy_module):
        assert self._classify(proxy_module, "gcli-gemini-2.5-flash") == "gemini_cli"

    def test_zai_routes_correctly(self, proxy_module):
        assert self._classify(proxy_module, "zai:gpt-4o") == "zai"

    def test_claude_default(self, proxy_module):
        assert self._classify(proxy_module, "claude-sonnet-4-6") == "claude"

    def test_unknown_model_defaults_to_claude(self, proxy_module):
        assert self._classify(proxy_module, "some-random-model") == "claude"

    def test_empty_string_defaults_to_claude(self, proxy_module):
        assert self._classify(proxy_module, "") == "claude"

    def test_zai_with_slash_routes_to_zai_not_openrouter(self, proxy_module):
        """zai:org/model contains '/' but zai: prefix is checked before openrouter."""
        assert self._classify(proxy_module, "zai:org/model") == "zai"

    def test_ollama_with_codex_name_routes_to_ollama(self, proxy_module):
        """ollama:codex-mini should route to Ollama, not Codex."""
        assert self._classify(proxy_module, "ollama:codex-mini") == "ollama"


# ---------------------------------------------------------------------------
# normalize_model_name
# ---------------------------------------------------------------------------

class TestNormalizeModelName:
    def test_sonnet_dot_to_hyphen(self, proxy_module):
        assert proxy_module.normalize_model_name("claude-sonnet-4.6") == "claude-sonnet-4-6"

    def test_opus_dot_to_hyphen(self, proxy_module):
        assert proxy_module.normalize_model_name("claude-opus-4.6") == "claude-opus-4-6"

    def test_haiku_alias(self, proxy_module):
        assert proxy_module.normalize_model_name("claude-haiku-4.5") == "claude-haiku-4-5-20251001"

    def test_sonnet_45_alias(self, proxy_module):
        assert proxy_module.normalize_model_name("claude-sonnet-4.5") == "claude-sonnet-4-5-20250929"

    def test_unknown_model_passes_through(self, proxy_module):
        assert proxy_module.normalize_model_name("some-random-model") == "some-random-model"

    def test_empty_string_passes_through(self, proxy_module):
        assert proxy_module.normalize_model_name("") == ""

    def test_already_canonical_name_unchanged(self, proxy_module):
        assert proxy_module.normalize_model_name("claude-sonnet-4-6") == "claude-sonnet-4-6"

    def test_ollama_model_passes_through(self, proxy_module):
        assert proxy_module.normalize_model_name("ollama:llama3") == "ollama:llama3"

    def test_openrouter_model_passes_through(self, proxy_module):
        assert proxy_module.normalize_model_name("openai/gpt-4o") == "openai/gpt-4o"

    def test_codex_model_passes_through(self, proxy_module):
        assert proxy_module.normalize_model_name("codex-mini-latest") == "codex-mini-latest"
