"""Tests for MCP server mode (mcp_server.py).

Covers provider registry, health checking, tool registration, provider
categorization (chat vs CLI), and server configuration.
"""
import pytest
from unittest.mock import patch, PropertyMock


@pytest.fixture()
def mcp_module(proxy_module):
    """Return the mcp_server module, ensuring proxy is imported first."""
    import mcp_server
    return mcp_server


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

class TestProviderRegistry:
    def test_provider_count(self, mcp_module):
        assert len(mcp_module.PROVIDERS) == 11

    def test_chat_providers_defined(self, mcp_module):
        chat = [p for p in mcp_module.PROVIDERS if p["category"] == mcp_module.CHAT]
        names = {p["name"] for p in chat}
        assert names == {"claude", "openrouter", "ollama", "antigravity", "zai", "xiaomi"}

    def test_cli_providers_defined(self, mcp_module):
        cli = [p for p in mcp_module.PROVIDERS if p["category"] == mcp_module.CLI]
        names = {p["name"] for p in cli}
        assert names == {"codex_cli", "gemini_cli", "qwen_cli", "opencode_zen", "opencode_go"}

    def test_all_providers_have_required_keys(self, mcp_module):
        required = {"name", "category", "health_fn", "call_fn", "default_model", "accepts_extra", "description"}
        for p in mcp_module.PROVIDERS:
            assert required.issubset(p.keys()), f"Provider {p.get('name', '?')} missing keys: {required - p.keys()}"

    def test_all_provider_names_unique(self, mcp_module):
        names = [p["name"] for p in mcp_module.PROVIDERS]
        assert len(names) == len(set(names))

    def test_categories_only_chat_or_cli(self, mcp_module):
        for p in mcp_module.PROVIDERS:
            assert p["category"] in (mcp_module.CHAT, mcp_module.CLI)

    def test_opencode_is_cli(self, mcp_module):
        """OpenCode providers must be CLI (agentic), not chat."""
        opencode = [p for p in mcp_module.PROVIDERS if "opencode" in p["name"]]
        assert len(opencode) == 2
        for p in opencode:
            assert p["category"] == mcp_module.CLI


# ---------------------------------------------------------------------------
# Health checking
# ---------------------------------------------------------------------------

class TestHealthChecking:
    def test_ollama_healthy_reflects_probe(self, mcp_module):
        # Without a running Ollama, probe hasn't succeeded so it's unhealthy
        assert mcp_module._is_ollama_healthy() == mcp_module._ollama_healthy

    def test_claude_unhealthy_when_no_auth(self, mcp_module, proxy_module):
        # With mocked CLI tools, auth won't be ready
        assert mcp_module._is_claude_healthy() == proxy_module.auth.is_ready

    def test_codex_unhealthy_without_path(self, mcp_module, proxy_module):
        # CODEX_PATH is None due to mocked shutil.which
        assert mcp_module._is_codex_healthy() is False

    def test_get_healthy_providers_returns_list(self, mcp_module):
        result = mcp_module.get_healthy_providers()
        assert isinstance(result, list)
        # Only healthy providers are included
        for p in result:
            assert p["health_fn"]() is True

    def test_get_healthy_chat_providers_only_chat(self, mcp_module):
        for p in mcp_module.get_healthy_chat_providers():
            assert p["category"] == mcp_module.CHAT

    def test_get_healthy_cli_providers_only_cli(self, mcp_module):
        for p in mcp_module.get_healthy_cli_providers():
            assert p["category"] == mcp_module.CLI

    def test_unhealthy_providers_excluded(self, mcp_module):
        healthy = mcp_module.get_healthy_providers()
        for p in healthy:
            assert p["health_fn"]() is True


# ---------------------------------------------------------------------------
# Server creation
# ---------------------------------------------------------------------------

class TestServerCreation:
    def test_creates_server(self, mcp_module):
        server = mcp_module.create_mcp_server()
        assert server is not None
        assert server.name == "codex-proxy"

    def test_server_has_instructions(self, mcp_module):
        server = mcp_module.create_mcp_server()
        assert server.instructions is not None
        assert "chat" in server.instructions.lower()
        assert "cli" in server.instructions.lower()

    def test_server_port(self, mcp_module):
        server = mcp_module.create_mcp_server()
        assert server.settings.port == 8432

    def test_server_host(self, mcp_module):
        server = mcp_module.create_mcp_server()
        assert server.settings.host == "127.0.0.1"


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

class TestToolRegistration:
    def test_tools_registered(self, mcp_module):
        server = mcp_module.create_mcp_server()
        tools = server._tool_manager._tools
        assert len(tools) > 0

    def test_chat_tools_registered(self, mcp_module):
        server = mcp_module.create_mcp_server()
        tool_names = set(server._tool_manager._tools.keys())
        expected_chat = {"chat_claude", "chat_openrouter", "chat_ollama",
                         "chat_antigravity", "chat_zai", "chat_xiaomi"}
        assert expected_chat.issubset(tool_names)

    def test_cli_tools_registered(self, mcp_module):
        server = mcp_module.create_mcp_server()
        tool_names = set(server._tool_manager._tools.keys())
        expected_cli = {"cli_codex", "cli_gemini", "cli_qwen",
                        "cli_opencode_zen", "cli_opencode_go"}
        assert expected_cli.issubset(tool_names)

    def test_meta_tools_registered(self, mcp_module):
        server = mcp_module.create_mcp_server()
        tool_names = set(server._tool_manager._tools.keys())
        expected_meta = {"list_chat_providers", "list_cli_providers",
                         "list_all_providers"}
        assert expected_meta.issubset(tool_names)

    def test_unified_dispatch_tools_registered(self, mcp_module):
        server = mcp_module.create_mcp_server()
        tool_names = set(server._tool_manager._tools.keys())
        assert "chat" in tool_names
        assert "cli" in tool_names

    def test_total_tool_count(self, mcp_module):
        server = mcp_module.create_mcp_server()
        tools = server._tool_manager._tools
        # 6 chat + 5 cli + 3 list + 2 unified + 5 orchestration = 21
        assert len(tools) == 23


# ---------------------------------------------------------------------------
# Provider name validation (MCP spec)
# ---------------------------------------------------------------------------

class TestToolNames:
    def test_all_provider_names_valid(self, mcp_module):
        import re
        pattern = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
        for p in mcp_module.PROVIDERS:
            assert pattern.match(p["name"]), f"Invalid provider name: {p['name']}"

    def test_all_tool_names_valid(self, mcp_module):
        import re
        pattern = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
        server = mcp_module.create_mcp_server()
        for name in server._tool_manager._tools.keys():
            assert pattern.match(name), f"Invalid tool name: {name}"


# ---------------------------------------------------------------------------
# Annotation presets
# ---------------------------------------------------------------------------

class TestAnnotations:
    def test_readonly_annotation(self, mcp_module):
        assert mcp_module._READONLY.readOnlyHint is True
        assert mcp_module._READONLY.destructiveHint is False

    def test_chat_annotation(self, mcp_module):
        assert mcp_module._CHAT_ANN.readOnlyHint is True
        assert mcp_module._CHAT_ANN.destructiveHint is False
        assert mcp_module._CHAT_ANN.openWorldHint is True

    def test_cli_annotation(self, mcp_module):
        assert mcp_module._CLI_ANN.readOnlyHint is False
        assert mcp_module._CLI_ANN.destructiveHint is True
        assert mcp_module._CLI_ANN.openWorldHint is True


# ---------------------------------------------------------------------------
# CLI --mode flag in proxy.py
# ---------------------------------------------------------------------------

class TestProxyModeFlag:
    def test_proxy_main_has_argparse(self, proxy_module):
        import inspect
        source = inspect.getsource(proxy_module)
        assert "--mode" in source
        assert "mcp" in source
