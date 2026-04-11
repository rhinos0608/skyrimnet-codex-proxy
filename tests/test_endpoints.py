"""Tests for HTTP endpoints in proxy.py: health, stats, config, and chat completions."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Helper: build standard auth mocks that make all providers "not ready"
# ---------------------------------------------------------------------------

def _all_providers_down(proxy_module):
    """Return a dict of patch.object() context managers that disable every provider."""
    return {
        "auth": patch.object(
            proxy_module, "auth",
            MagicMock(is_ready=False, session=None, headers=None, body_template=None),
        ),
        "codex_auth": patch.object(
            proxy_module, "codex_auth",
            MagicMock(is_ready=False, session=None, is_expired=MagicMock(return_value=None)),
        ),
        "antigravity_auth": patch.object(
            proxy_module, "antigravity_auth",
            MagicMock(
                is_ready=False, email=None, project_id=None,
                accounts=[], _legacy_account=MagicMock(session=None),
                get_all_accounts_info=MagicMock(return_value=[]),
                is_expired=MagicMock(return_value=None),
            ),
        ),
        "gemini_auth": patch.object(
            proxy_module, "gemini_auth",
            MagicMock(is_ready=False, session=None, refresh_token=None,
                      is_expired=MagicMock(return_value=None)),
        ),
        "openrouter_key": patch.object(proxy_module, "openrouter_api_key", None),
        "ollama_key": patch.object(proxy_module, "ollama_api_key", None),
        "zai_key": patch.object(proxy_module, "zai_api_key", None),
        "xiaomi_key": patch.object(proxy_module, "xiaomi_api_key", None),
        "opencode_key": patch.object(proxy_module, "opencode_api_key", None),
        "opencode_go_key": patch.object(proxy_module, "opencode_go_api_key", None),
        "qwen_auth": patch.object(
            proxy_module, "qwen_auth",
            MagicMock(is_ready=False, session=None, refresh_token=None,
                      is_expired=MagicMock(return_value=None)),
        ),
        "fireworks_key": patch.object(proxy_module, "fireworks_api_key", None),
        "nvidia_key": patch.object(proxy_module, "nvidia_api_key", None),
    }


# ===========================================================================
# GET /health
# ===========================================================================


class TestHealthEndpoint:

    def test_all_providers_down_returns_warming_up(self, test_client, proxy_module):
        """When no auth-based provider is ready, status should be 'warming_up'."""
        patches = _all_providers_down(proxy_module)
        with patches["auth"], patches["codex_auth"], patches["antigravity_auth"], \
             patches["gemini_auth"], patches["openrouter_key"], \
             patches["ollama_key"], patches["zai_key"], patches["xiaomi_key"], \
             patches["opencode_key"], patches["opencode_go_key"], patches["qwen_auth"], \
             patches["fireworks_key"], patches["nvidia_key"]:
            resp = test_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "warming_up"

    def test_one_provider_ready_returns_healthy(self, test_client, proxy_module):
        """When at least one auth provider is ready, status should be 'healthy'."""
        patches = _all_providers_down(proxy_module)
        # Override claude auth to be ready
        patches["auth"] = patch.object(
            proxy_module, "auth",
            MagicMock(is_ready=True, session=MagicMock(), headers={}, body_template={}),
        )
        with patches["auth"], patches["codex_auth"], patches["antigravity_auth"], \
             patches["gemini_auth"], patches["openrouter_key"], \
             patches["ollama_key"], patches["zai_key"], patches["xiaomi_key"], \
             patches["opencode_key"], patches["opencode_go_key"], patches["qwen_auth"], \
             patches["fireworks_key"], patches["nvidia_key"]:
            resp = test_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"

    def test_response_has_correct_structure(self, test_client, proxy_module):
        """Health response must contain all expected top-level keys."""
        patches = _all_providers_down(proxy_module)
        with patches["auth"], patches["codex_auth"], patches["antigravity_auth"], \
             patches["gemini_auth"], patches["openrouter_key"], \
             patches["ollama_key"], patches["zai_key"], patches["xiaomi_key"], \
             patches["opencode_key"], patches["opencode_go_key"], patches["qwen_auth"], \
             patches["fireworks_key"], patches["nvidia_key"]:
            resp = test_client.get("/health")
        data = resp.json()
        expected_keys = {
            "status", "claude", "codex", "antigravity", "gemini_cli",
            "openrouter_configured", "ollama_configured", "zai_configured",
            "xiaomi_configured", "opencode_zen_configured", "opencode_go_configured",
            "qwen", "fireworks_configured", "nvidia_configured",
        }
        assert expected_keys.issubset(data.keys())


# ===========================================================================
# GET /stats
# ===========================================================================


class TestStatsEndpoint:

    def test_returns_expected_structure(self, test_client, proxy_module):
        """GET /stats should return global and by_model keys."""
        mock_stats = MagicMock()
        mock_stats.get_stats.return_value = {
            "global": {"total_requests": 0, "total_errors": 0, "error_rate": 0},
            "by_model": {},
        }
        with patch.object(proxy_module, "request_stats", mock_stats):
            resp = test_client.get("/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "global" in data
        assert "by_model" in data


# ===========================================================================
# POST /config/openrouter-key
# ===========================================================================


class TestOpenRouterKeyEndpoint:

    def test_set_key_returns_saved(self, test_client, proxy_module):
        """Setting a non-empty key returns status=saved and updates the global."""
        with patch.object(proxy_module, "_save_config"), \
             patch.object(proxy_module, "_load_config", return_value={}):
            resp = test_client.post("/config/openrouter-key", json={"key": "sk-or-test"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "saved"
        assert proxy_module.openrouter_api_key == "sk-or-test"

    def test_clear_key_returns_cleared(self, test_client, proxy_module):
        """Sending an empty key clears the global and returns status=cleared."""
        proxy_module.openrouter_api_key = "existing-key"
        with patch.object(proxy_module, "_save_config"), \
             patch.object(proxy_module, "_load_config", return_value={"openrouter_api_key": "existing-key"}):
            resp = test_client.post("/config/openrouter-key", json={"key": ""})
        assert resp.status_code == 200
        assert resp.json()["status"] == "cleared"
        assert proxy_module.openrouter_api_key is None


# ===========================================================================
# POST /config/zai-key
# ===========================================================================


class TestZaiKeyEndpoint:

    def test_set_key_returns_saved(self, test_client, proxy_module):
        """Setting a non-empty key returns status=saved and updates the global."""
        with patch.object(proxy_module, "_save_config"), \
             patch.object(proxy_module, "_load_config", return_value={}):
            resp = test_client.post("/config/zai-key", json={"key": "zai-test-key"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "saved"
        assert proxy_module.zai_api_key == "zai-test-key"

    def test_clear_key_returns_cleared(self, test_client, proxy_module):
        """Sending an empty key clears the global and returns status=cleared."""
        proxy_module.zai_api_key = "existing-zai"
        with patch.object(proxy_module, "_save_config"), \
             patch.object(proxy_module, "_load_config", return_value={"zai_api_key": "existing-zai"}):
            resp = test_client.post("/config/zai-key", json={"key": ""})
        assert resp.status_code == 200
        assert resp.json()["status"] == "cleared"
        assert proxy_module.zai_api_key is None


# ===========================================================================
# POST /config/timeout-routing
# ===========================================================================


class TestTimeoutRoutingEndpoint:

    def test_enable_timeout_routing(self, test_client, proxy_module):
        """Enabling timeout routing returns status=ok with enabled=True."""
        with patch.object(proxy_module, "_save_config"), \
             patch.object(proxy_module, "_load_config", return_value={}):
            resp = test_client.post("/config/timeout-routing", json={"enabled": True})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["timeout_routing_enabled"] is True
        assert proxy_module.timeout_routing_enabled is True

    def test_disable_timeout_routing(self, test_client, proxy_module):
        """Disabling timeout routing returns status=ok with enabled=False."""
        with patch.object(proxy_module, "_save_config"), \
             patch.object(proxy_module, "_load_config", return_value={}):
            resp = test_client.post("/config/timeout-routing", json={"enabled": False})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["timeout_routing_enabled"] is False
        assert proxy_module.timeout_routing_enabled is False


# ===========================================================================
# POST /config/timeout-cutoff
# ===========================================================================


class TestTimeoutCutoffEndpoint:

    def test_valid_value_accepted(self, test_client, proxy_module):
        """A value between 1 and 30 is saved successfully."""
        with patch.object(proxy_module, "_save_config"), \
             patch.object(proxy_module, "_load_config", return_value={}):
            resp = test_client.post("/config/timeout-cutoff", json={"seconds": 10.0})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["timeout_cutoff_seconds"] == 10.0
        assert proxy_module.timeout_cutoff_seconds == 10.0

    def test_below_minimum_rejected(self, test_client, proxy_module):
        """A value below 1.0 returns HTTP 400."""
        with patch.object(proxy_module, "_save_config"), \
             patch.object(proxy_module, "_load_config", return_value={}):
            resp = test_client.post("/config/timeout-cutoff", json={"seconds": 0.5})
        assert resp.status_code == 400

    def test_above_maximum_rejected(self, test_client, proxy_module):
        """A value above 30.0 returns HTTP 400."""
        with patch.object(proxy_module, "_save_config"), \
             patch.object(proxy_module, "_load_config", return_value={}):
            resp = test_client.post("/config/timeout-cutoff", json={"seconds": 31.0})
        assert resp.status_code == 400


# ===========================================================================
# GET /config/timeout-routing
# ===========================================================================


class TestGetTimeoutRoutingEndpoint:

    def test_returns_current_config(self, test_client, proxy_module):
        """GET returns timeout_routing_enabled, timeout_seconds, and stats."""
        with patch.object(proxy_module, "timeout_routing_enabled", True), \
             patch.object(proxy_module, "timeout_cutoff_seconds", 8.0), \
             patch.object(proxy_module, "model_stats", MagicMock(get_stats=MagicMock(return_value={}))):
            resp = test_client.get("/config/timeout-routing")
        assert resp.status_code == 200
        data = resp.json()
        assert data["timeout_routing_enabled"] is True
        assert data["timeout_seconds"] == 8.0
        assert "stats" in data


# ===========================================================================
# POST /v1/chat/completions
# ===========================================================================


class TestChatCompletionsEndpoint:

    def test_missing_auth_returns_503(self, test_client, proxy_module):
        """When default Claude auth is not ready, returns 503."""
        patches = _all_providers_down(proxy_module)
        with patches["auth"], patches["codex_auth"], patches["antigravity_auth"], \
             patches["gemini_auth"], patches["openrouter_key"], \
             patches["ollama_key"], patches["zai_key"], patches["xiaomi_key"], \
             patches["opencode_key"], patches["opencode_go_key"], patches["qwen_auth"], \
             patches["fireworks_key"], patches["nvidia_key"]:
            resp = test_client.post("/v1/chat/completions", json={
                "model": "claude-sonnet-4-6",
                "messages": [{"role": "user", "content": "Hello"}],
            })
        assert resp.status_code == 503

    def test_empty_messages_returns_400(self, test_client, proxy_module):
        """When no user/assistant messages are provided, returns 400."""
        # Use ollama model so no auth check is needed
        resp = test_client.post("/v1/chat/completions", json={
            "model": "ollama:llama3.2",
            "messages": [{"role": "system", "content": "You are helpful"}],
        })
        assert resp.status_code == 400

    def test_non_streaming_ollama_returns_openai_format(self, test_client, proxy_module):
        """Non-streaming ollama request returns a well-formed OpenAI-style response."""
        with patch.object(proxy_module, "call_ollama_direct",
                          new_callable=AsyncMock, return_value="test response"), \
             patch.object(proxy_module, "request_stats", MagicMock(record=MagicMock())):
            resp = test_client.post("/v1/chat/completions", json={
                "model": "ollama:llama3.2",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
            })
        assert resp.status_code == 200
        data = resp.json()
        # Verify OpenAI response structure
        assert data["object"] == "chat.completion"
        assert "id" in data
        assert "created" in data
        assert data["model"] == "ollama:llama3.2"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["content"] == "test response"
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["finish_reason"] == "stop"
        assert "usage" in data
        assert "prompt_tokens" in data["usage"]
        assert "completion_tokens" in data["usage"]
        assert "total_tokens" in data["usage"]

    def test_ollama_preserves_user_system_messages(self, test_client, proxy_module):
        """OpenAI-compatible providers should receive user-supplied system messages intact."""
        with patch.object(proxy_module, "call_ollama_direct",
                          new_callable=AsyncMock, return_value="test response") as mock_call, \
             patch.object(proxy_module, "request_stats", MagicMock(record=MagicMock())):
            resp = test_client.post("/v1/chat/completions", json={
                "model": "ollama:llama3.2",
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Hi"},
                ],
                "stream": False,
            })
        assert resp.status_code == 200
        system_prompt, messages, _, _ = mock_call.call_args.args
        assert system_prompt is None
        assert messages == [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
        ]

    def test_claude_keeps_system_prompt_on_separate_provider_path(self, test_client, proxy_module):
        """Non-OpenAI providers should still receive system_prompt separately."""
        mock_auth = MagicMock(is_ready=True, session=MagicMock(), headers={},
                              body_template={"messages": []})
        with patch.object(proxy_module, "auth", mock_auth), \
             patch.object(proxy_module, "call_api_direct",
                          new_callable=AsyncMock, return_value="claude reply") as mock_call, \
             patch.object(proxy_module, "request_stats", MagicMock(record=MagicMock())):
            resp = test_client.post("/v1/chat/completions", json={
                "model": "claude-sonnet-4-6",
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Hi"},
                ],
                "stream": False,
            })
        assert resp.status_code == 200
        system_prompt, messages, _, _ = mock_call.call_args.args
        assert system_prompt == "You are helpful"
        assert messages == [{"role": "user", "content": "Hi"}]

    def test_streaming_returns_streaming_response(self, test_client, proxy_module):
        """Streaming request returns a response with text/event-stream media type."""
        async def fake_stream(*args, **kwargs):
            yield 'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
            yield "data: [DONE]\n\n"

        with patch.object(proxy_module, "call_ollama_streaming",
                          return_value=fake_stream()), \
             patch.object(proxy_module, "timeout_routing_enabled", False), \
             patch.object(proxy_module, "request_stats", MagicMock(record=MagicMock())):
            resp = test_client.post("/v1/chat/completions", json={
                "model": "ollama:llama3.2",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            })
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

    def test_routes_to_ollama_provider(self, test_client, proxy_module):
        """Ollama-prefixed model calls call_ollama_direct, not other providers."""
        with patch.object(proxy_module, "call_ollama_direct",
                          new_callable=AsyncMock, return_value="ollama reply") as mock_call, \
             patch.object(proxy_module, "request_stats", MagicMock(record=MagicMock())):
            resp = test_client.post("/v1/chat/completions", json={
                "model": "ollama:llama3.2",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
            })
        assert resp.status_code == 200
        mock_call.assert_called_once()

    def test_routes_to_openrouter_provider(self, test_client, proxy_module):
        """Slash-containing model routes to OpenRouter."""
        with patch.object(proxy_module, "call_openrouter_direct",
                          new_callable=AsyncMock, return_value="openrouter reply") as mock_call, \
             patch.object(proxy_module, "request_stats", MagicMock(record=MagicMock())):
            resp = test_client.post("/v1/chat/completions", json={
                "model": "openai/gpt-4o",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
            })
        assert resp.status_code == 200
        mock_call.assert_called_once()

    def test_routes_to_claude_provider(self, test_client, proxy_module):
        """Default model routes to Claude (call_api_direct)."""
        mock_auth = MagicMock(is_ready=True, session=MagicMock(), headers={},
                              body_template={"messages": []})
        with patch.object(proxy_module, "auth", mock_auth), \
             patch.object(proxy_module, "call_api_direct",
                          new_callable=AsyncMock, return_value="claude reply") as mock_call, \
             patch.object(proxy_module, "request_stats", MagicMock(record=MagicMock())):
            resp = test_client.post("/v1/chat/completions", json={
                "model": "claude-sonnet-4-6",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
            })
        assert resp.status_code == 200
        mock_call.assert_called_once()
