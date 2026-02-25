# Claude SkyrimNet Proxy

An OpenAI-compatible API proxy that routes requests through a Claude Max subscription, OpenAI Codex CLI subscription, or OpenRouter. Designed for use with [SkyrimNet](https://github.com/MinLL/SkyrimNet-GamePlugin) to power AI-driven NPC conversations in Skyrim.

## How It Works

```
SkyrimNet (game) --> POST /v1/chat/completions --> Proxy (port 8000) --> Anthropic API
                                                                    \-> OpenAI Codex CLI
                                                                    \-> OpenRouter API
```

### Claude Provider
1. **Startup**: The proxy spawns a single `claude --print` command to capture authenticated headers and a minimal request template from the Claude CLI.
2. **Per-request**: Uses the cached auth to make direct API calls to `api.anthropic.com` via a persistent connection. No subprocess is spawned per request.
3. **Response**: Translates Anthropic's streaming SSE format into OpenAI-compatible SSE chunks (`chat.completion.chunk`) that SkyrimNet expects.

### Codex Provider
1. **Startup**: Reads OAuth tokens from `~/.codex/auth.json` (created by `codex login`).
2. **Per-request**: Spawns an isolated Codex CLI subprocess with a clean HOME directory (no global config bloat).
3. **Response**: Parses JSONL output from Codex CLI and converts to OpenAI-compatible format.

### Optimizations

- **Direct API calls** with cached auth (~2s vs ~9s with subprocess per request)
- **Persistent TCP/TLS session** reused across all requests
- **Tool definitions stripped** from template (saves ~60KB per request)
- **Extended thinking disabled** to minimize latency
- **Clean temp directory capture** minimizes template bloat (~1KB vs ~16KB)
- **Isolated Codex HOME** prevents config/instruction bloat

## Requirements

- **Python 3.10+**
- **Claude CLI** (optional) installed and authenticated with a [Claude Max subscription](https://claude.ai/pricing) (`claude` must be on your PATH)
- **Codex CLI** (optional) installed and authenticated (`codex login`)
- **Python packages**:

```
pip install fastapi uvicorn aiohttp pydantic
```

At least one of Claude CLI or Codex CLI must be available.

## Usage

### Start the proxy

```bash
python proxy.py
```

Or use the included batch file on Windows:

```bash
start-proxy.bat
```

The proxy will:
1. Start an interceptor on port 9999 (internal, used only during startup for Claude)
2. Capture auth from the Claude CLI (~5 seconds, if available)
3. Load Codex auth from `~/.codex/auth.json` (if available)
4. Start the API server on `http://127.0.0.1:8000`

### Dashboard

Open `http://127.0.0.1:8000` in a browser to see the status dashboard with a quick test form.

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | OpenAI-compatible chat completions (streaming + non-streaming) |
| `/v1/models` | GET | List available models |
| `/config/openrouter-key` | POST | Set OpenRouter API key (JSON body: `{"key": "sk-or-..."}`) |
| `/health` | GET | Health check |
| `/` | GET | Web dashboard |

### Configure SkyrimNet

In your SkyrimNet configuration, set:
- **API Endpoint**: `http://localhost:8000/v1/chat/completions`
- **API Key**: (leave empty or set any value — not required)
- **Model**: `claude-sonnet-4-5-20250929` (recommended), `gpt-5.2-codex`, or any supported model

### Supported Models

| Model ID | Provider | Notes |
|----------|----------|-------|
| `claude-opus-4-6` | Anthropic | Most capable, highest latency |
| `claude-sonnet-4-5-20250929` | Anthropic | Best balance (default for Claude) |
| `claude-haiku-4-5-20251001` | Anthropic | Fastest, least capable |
| `gpt-5.2` | Codex | Default Codex model |
| `gpt-5.2-codex` | Codex | Codex CLI model |
| `gpt-5.1-codex-max` | Codex | High-capability Codex |
| `gpt-5.1-codex-mini` | Codex | Fast Codex variant |
| `provider/model` | OpenRouter | Any model via [OpenRouter](https://openrouter.ai) (requires API key) |

### Codex Support

To use Codex models:

1. Install the Codex CLI and authenticate:
   ```bash
   codex login
   ```
2. This creates `~/.codex/auth.json` with your OAuth tokens
3. Restart the proxy
4. Use any model starting with `gpt-5.` or `codex-`

Codex models spawn a subprocess per request (unlike Claude's direct API calls), so expect slightly higher latency.

### OpenRouter Support

You can use any model available on [OpenRouter](https://openrouter.ai) by setting an API key:

1. Get an API key from [openrouter.ai/keys](https://openrouter.ai/keys)
2. Open the dashboard at `http://127.0.0.1:8000` and paste the key in the OpenRouter section
3. Use `provider/model` format in the Model field (e.g. `openai/gpt-4o`, `google/gemini-2.0-flash-001`)

### Model Rotation

You can comma-separate multiple models to rotate between them round-robin:

```
claude-sonnet-4-5-20250929, gpt-5.2-codex, openai/gpt-4o
```

Each request cycles to the next model in the list. Models are routed based on naming:
- `gpt-5.*` or `codex-*` → Codex CLI
- `provider/model` (contains `/`) → OpenRouter
- All others → Anthropic/Claude

## Important Legal Disclaimer

> **This project operates in a gray area with respect to Anthropic's and OpenAI's Terms of Service.**
>
> This proxy uses the Claude CLI's authenticated session to make direct API calls, bypassing the standard Claude Code interface. While it uses a legitimately paid Claude Max subscription, the method of accessing the API may not be explicitly authorized by Anthropic's [Terms of Service](https://www.anthropic.com/legal/consumer-terms) or [Acceptable Use Policy](https://www.anthropic.com/legal/aup).
>
> Similarly, Codex auth capture reads OAuth tokens from the Codex CLI's stored credentials, which may violate OpenAI's Terms of Service.
>
> **Before using this proxy, you should:**
> - Read Anthropic's and OpenAI's current Terms of Service in full
> - Understand that this access method could be restricted or blocked at any time
> - Be aware that your account could potentially be affected
> - Accept all responsibility for your use of this software
>
> **This software is provided as-is, with no warranty or guarantee of continued functionality.** The authors are not responsible for any consequences of using this proxy, including but not limited to account suspension or service disruption.

## Troubleshooting

### "Auth not ready" / Startup fails
- Ensure `claude` CLI is on your PATH: `claude --version`
- Ensure you're logged in: `claude` (should start without auth errors)
- Check that port 9999 is not in use

### "Codex auth not ready"
- Ensure `codex` CLI is on your PATH: `codex --version`
- Run `codex login` first to create `~/.codex/auth.json`
- Restart the proxy after logging in

### SkyrimNet NPC not responding
- Check the proxy console for errors
- Verify SkyrimNet is configured with the correct endpoint (`http://localhost:8000/v1/chat/completions`)
- Ensure `stream: true` is being sent (SkyrimNet uses streaming by default)

### "credential only authorized for Claude Code" error
- The cached auth has expired. Restart the proxy to re-capture.

## Changelog

### 2025-02-25 — Codex Provider Addition
- Added Codex CLI as a provider option alongside Claude and OpenRouter
- Models: `gpt-5.2`, `gpt-5.2-codex`, `gpt-5.1-codex-max`, `gpt-5.1-codex-mini`
- Reads OAuth tokens from `~/.codex/auth.json` (created by `codex login`)
- Uses isolated HOME directory per request to avoid config bloat
- Supports both streaming and non-streaming modes

### Previous Releases (by [@galanx](https://github.com/galanx))
See [galanx/Claude-SkyrimNet-Proxy](https://github.com/galanx/Claude-SkyrimNet-Proxy) for original release history:
- OpenRouter support with round-robin model rotation
- Dashboard UI for configuring OpenRouter API key
- OpenAI-compatible proxy with Claude Max subscription auth
- Direct API calls with persistent session
- Streaming + non-streaming support
- Web dashboard with quick test form

## Attribution

This project builds on [Claude-SkyrimNet-Proxy](https://github.com/galanx/Claude-SkyrimNet-Proxy) by [@galanx](https://github.com/galanx), with the addition of Codex as a provider option.

## License

MIT