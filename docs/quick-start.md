# Quick Start Guide

## Prerequisites

- **Python 3.10+**
- **At least one provider** authenticated (Claude CLI, Codex CLI, API keys, etc.)
- **Dependencies**: `pip install -r requirements.txt`

---

## 1. Proxy Mode (Default)

The proxy serves an OpenAI-compatible API on port 8539 for SkyrimNet and other clients.

```bash
python proxy.py
```

**What happens on startup:**
1. MITM interceptor starts on port 9999 (internal, captures Claude CLI auth)
2. Codex interceptor starts on port 9998 (internal, captures Codex CLI auth)
3. Loads API keys and OAuth tokens for all other providers
4. API server starts on `http://0.0.0.0:8539`

**Endpoints:**
- Dashboard: `http://localhost:8539`
- Chat API: `http://localhost:8539/v1/chat/completions`
- Models: `http://localhost:8539/v1/models`
- Health: `http://localhost:8539/health`

---

## 2. MCP Server Mode

Exposes all LLM providers as MCP tools for Claude Desktop, Claude Code, or any MCP client.

### STDIO transport (local, for Claude Desktop / Claude Code)

```bash
python proxy.py --mode mcp --transport stdio
```

This is the default MCP transport. The server communicates over stdin/stdout -- no network port needed.

### SSE transport (networked, for remote MCP clients)

```bash
python proxy.py --mode mcp --transport sse
```

Starts an SSE server on `http://0.0.0.0:8432`. MCP clients connect to `http://<host>:8432/sse`.

---

## 3. Running Proxy + MCP Simultaneously

Proxy mode and MCP mode can now run simultaneously. They use different interceptor ports:

| Mode | Interceptor Port | Command |
|---|---|---|
| Proxy mode | 9999 | `python proxy.py` |
| MCP mode (stdio) | 9997 | `python proxy.py --mode mcp` |
| MCP mode (SSE) | 9997 | `python proxy.py --mode mcp --transport sse` |

The Claude MITM interceptor runs during startup to capture authentication headers.

---

## 4. Tailscale HTTPS

Expose the proxy and/or MCP server over HTTPS on your Tailscale network. Tailscale handles TLS certificates automatically -- no certs to manage.

### Prerequisites

- [Tailscale](https://tailscale.com) installed and running
- MagicDNS enabled on your tailnet
- HTTPS enabled in Tailscale admin console (Settings > DNS > HTTPS Certificates)

### Automatic setup (via --tailscale flag)

```bash
# Proxy with Tailscale HTTPS
python proxy.py --tailscale

# MCP SSE server with Tailscale HTTPS
python proxy.py --mode mcp --transport sse --tailscale
```

The `--tailscale` flag runs `tailscale serve --bg` to create a reverse proxy from your Tailscale FQDN to the local HTTP port. This happens before the server starts.

The MCP server automatically detects the Tailscale FQDN at startup and adds it to the allowed hosts for DNS rebinding protection. No extra configuration needed -- remote MCP clients connecting via Tailscale will be accepted.

### Manual setup

If you prefer to configure Tailscale independently:

```bash
# Set up HTTPS for the proxy (port 8539)
tailscale serve --bg --https 8539 http://127.0.0.1:8539

# Set up HTTPS for the MCP SSE server (port 8432)
tailscale serve --bg --https 8432 http://127.0.0.1:8432

# Check status
tailscale serve status

# Remove a specific port
tailscale serve --https=8539 off
tailscale serve --https=8432 off

# Remove all serve configs
tailscale serve reset
```

### Resulting URLs

Once configured, services are available over HTTPS on your tailnet:

| Service | URL |
|---|---|
| Proxy API | `https://<hostname>.<tailnet>.ts.net:8539/v1/chat/completions` |
| Dashboard | `https://<hostname>.<tailnet>.ts.net:8539` |
| MCP (SSE) | `https://<hostname>.<tailnet>.ts.net:8432/sse` |

Find your hostname with:

```bash
tailscale status
```

These URLs are accessible from any device on your tailnet (other computers, phones, etc.) with valid TLS certificates.

### MCP client configuration (SSE over Tailscale)

For remote MCP clients connecting over Tailscale HTTPS:

```json
{
  "mcpServers": {
    "codex-proxy": {
      "transport": "sse",
      "url": "https://<hostname>.<tailnet>.ts.net:8432/sse"
    }
  }
}
```

### SkyrimNet configuration (over Tailscale)

For a gaming PC on the same tailnet:

- **API Endpoint**: `https://<hostname>.<tailnet>.ts.net:8539/v1/chat/completions`
- **API Key**: (not required)
- **Model**: `claude-sonnet-4-5-20250929` or any supported model

---

## 5. MCP Client Configuration

### Claude Code (CLI) -- STDIO

Add to `.claude/settings.json` or project `.mcp.json`:

```json
{
  "mcpServers": {
    "codex-proxy": {
      "command": "python3",
      "args": [
        "/path/to/proxy.py",
        "--mode", "mcp",
        "--transport", "stdio"
      ]
    }
  }
}
```

### Claude Desktop -- STDIO

In Claude Desktop's MCP server settings:

- **Server Name**: `codex-proxy`
- **Transport Type**: STDIO
- **Command**: `python3` (or full path like `/opt/homebrew/opt/python@3.11/libexec/bin/python3`)
- **Arguments** (one per line):
  1. `/path/to/proxy.py`
  2. `--mode`
  3. `mcp`
  4. `--transport`
  5. `stdio`

Or paste raw JSON via the `{...}` button:

```json
{
  "command": "python3",
  "args": [
    "/path/to/proxy.py",
    "--mode", "mcp",
    "--transport", "stdio"
  ]
}
```

### Any MCP Client -- SSE (networked)

First start the server:

```bash
python proxy.py --mode mcp --transport sse
```

Then configure the client:

```json
{
  "mcpServers": {
    "codex-proxy": {
      "transport": "sse",
      "url": "http://localhost:8432/sse"
    }
  }
}
```

---

## Summary of Ports

| Port | Service | When |
|---|---|---|
| 8539 | Proxy HTTP API | `--mode proxy` (default) |
| 8432 | MCP SSE server | `--mode mcp --transport sse` |
| 9999 | Claude MITM interceptor | Proxy mode only (internal, startup) |
| 9997 | Claude MITM interceptor | MCP mode only (internal, startup) |
| 9998 | Codex MITM interceptor | Both modes (internal, startup only) |

---

## Command Reference

```bash
# Proxy mode (default)
python proxy.py

# Proxy mode with Tailscale HTTPS
python proxy.py --tailscale

# MCP mode, stdio transport (for Claude Desktop/CLI)
python proxy.py --mode mcp --transport stdio

# MCP mode, SSE transport (networked)
python proxy.py --mode mcp --transport sse

# MCP mode, SSE transport with Tailscale HTTPS
python proxy.py --mode mcp --transport sse --tailscale
```
