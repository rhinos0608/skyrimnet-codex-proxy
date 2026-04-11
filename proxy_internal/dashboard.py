"""Dashboard HTML template rendering."""

from dataclasses import dataclass


@dataclass
class DashboardContext:
    """Pre-computed state for dashboard template rendering.

    All fields are strings / primitives that can be directly substituted
    into the HTML f-string template — no further conditionals required at
    render time.
    """
    # Overall status badge
    status: str
    status_color: str

    # Per-provider status + color pairs (used in inline style attributes)
    claude_status: str
    claude_color: str
    codex_status: str
    codex_color: str
    gemini_cli_status: str
    gemini_cli_color: str
    qwen_status: str
    qwen_color: str
    ollama_status: str
    ollama_color: str
    zai_status: str
    zai_color: str
    xiaomi_status: str
    xiaomi_color: str
    opencode_status: str
    opencode_color: str
    fireworks_status: str
    fireworks_color: str
    nvidia_status: str
    nvidia_color: str
    or_status: str
    or_color: str
    ag_status: str
    ag_color: str
    ag_account_count: int

    # Pre-rendered HTML blocks
    claude_rows: str
    codex_rows: str
    antigravity_rows: str
    gemini_cli_rows: str
    ag_accounts_html: str

    # Provider card class hints (so template avoids inline conditionals)
    claude_card_class: str
    claude_status_class: str
    codex_card_class: str
    codex_status_class: str
    ag_card_class: str
    ag_status_class: str
    gemini_cli_card_class: str
    gemini_cli_status_class: str
    qwen_card_class: str
    qwen_status_class: str

    # Inline hint snippets
    gemini_cli_hint: str
    qwen_hint: str

    # OpenCode Go status (separate save-panel label)
    opencode_go_status_color: str
    opencode_go_status_text: str

    # Runtime config values (raw)
    timeout_routing_enabled: bool
    timeout_cutoff_seconds: float
    max_total_seconds: float
    max_retries: int
    timeout_routing_checked: str
    timeout_routing_color: str
    timeout_routing_label: str


def render_dashboard(ctx: DashboardContext) -> str:
    """Render the dashboard HTML from pre-computed state."""
    return f"""<!DOCTYPE html>
<html><head><title>Claude SkyrimNet Proxy</title>
<style>
  body {{ background:#0f172a; color:#e2e8f0; font-family:system-ui,sans-serif; max-width:800px; margin:40px auto; padding:0 20px }}
  h1 {{ color:#f8fafc; font-size:1.5rem; margin-bottom:4px }}
  .subtitle {{ color:#64748b; font-size:0.9rem; margin-bottom:30px }}
  .status {{ display:inline-block; padding:4px 12px; border-radius:12px; font-size:0.85rem; font-weight:600;
             background:{ctx.status_color}20; color:{ctx.status_color}; border:1px solid {ctx.status_color}40 }}
  .card {{ background:#1e293b; border-radius:8px; padding:20px; margin:16px 0; border:1px solid #334155 }}
  .provider-card {{ background:#1e293b; border-radius:8px; padding:16px; border:1px solid #334155 }}
  .provider-ready {{ border-color:#4ade8040 }}
  .provider-notready {{ border-color:#facc1540 }}
  table {{ width:100%; border-collapse:collapse }}
  th {{ text-align:left; color:#94a3b8; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.05em; padding:8px 12px; border-bottom:1px solid #334155 }}
  td {{ padding:6px 12px; border-bottom:1px solid #1e293b }}
  .label {{ color:#94a3b8; font-size:0.85rem }}
  .value {{ color:#f1f5f9; font-family:monospace; font-size:0.85rem }}
  .endpoint {{ background:#0f172a; padding:10px 14px; border-radius:6px; font-family:monospace; font-size:0.85rem; color:#67e8f9; margin:8px 0; border:1px solid #334155 }}
  textarea {{ width:100%; background:#0f172a; color:#e2e8f0; border:1px solid #334155; border-radius:6px; padding:10px; font-family:monospace; font-size:0.85rem; resize:vertical; box-sizing:border-box }}
  button {{ background:#3b82f6; color:white; border:none; padding:8px 20px; border-radius:6px; cursor:pointer; font-size:0.85rem; margin-top:8px }}
  button:hover {{ background:#2563eb }}
  button:disabled {{ background:#475569; cursor:wait }}
  #response {{ margin-top:12px; padding:12px; background:#0f172a; border-radius:6px; border:1px solid #334155; font-size:0.9rem; min-height:40px; white-space:pre-wrap }}
  .timing {{ color:#4ade80; font-size:0.8rem; margin-top:6px }}
  select {{ background:#0f172a; color:#e2e8f0; border:1px solid #334155; border-radius:6px; padding:8px 12px; font-family:monospace; font-size:0.85rem }}
  .provider-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; margin:16px 0 }}
  .provider-header {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:12px }}
  .provider-status {{ padding:4px 10px; border-radius:12px; font-size:0.75rem; font-weight:600 }}
  .status-ready {{ background:#4ade8020; color:#4ade80 }}
  .status-notready {{ background:#facc1520; color:#facc15 }}
  .status-offline {{ background:#64748b20; color:#64748b }}
</style></head>
<body>
  <h1>Claude SkyrimNet Proxy</h1>
  <div class="subtitle">OpenAI-compatible proxy using Claude Max or Codex CLI subscription</div>
  <span class="status">{ctx.status}</span>

  <div class="card" style="border-color:#f87171;background:#450a0a20">
    <h3 style="margin:0 0 8px;font-size:1rem;color:#f87171">⚠️ Terms of Service Warning</h3>
    <p style="margin:0;color:#fca5a5;font-size:0.85rem">
      The Codex provider intercepts OAuth tokens from the official Codex CLI. The Antigravity provider uses
      Google OAuth to access the Antigravity IDE API. These approaches exist in a
      <strong>gray area of Terms of Service</strong>. Use at your own risk.
    </p>
  </div>

  <div class="card">
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px">
      <div><span class="label">Endpoint</span><div class="endpoint">https://macbook-pro.taila2c1ae.ts.net:8539/v1/chat/completions</div></div>
      <div><span class="label">API Key</span><div class="endpoint">not required</div></div>
    </div>
  </div>

  <!-- Provider Slots -->
  <div class="provider-grid">
    <!-- Claude Provider -->
    <div class="provider-card {ctx.claude_card_class}">
      <div class="provider-header">
        <h3 style="margin:0;font-size:1rem;color:#f1f5f9">🟠 Claude (Anthropic)</h3>
        <span class="provider-status {ctx.claude_status_class}">{ctx.claude_status}</span>
      </div>
      <div style="margin-bottom:8px"><span class="label">Auth Method:</span> <span class="value">MITM Interceptor</span></div>
      <table style="font-size:0.8rem">
        <thead><tr><th>Model ID</th><th>Name</th><th>Notes</th></tr></thead>
        <tbody>{ctx.claude_rows}</tbody>
      </table>
    </div>

    <!-- Codex Provider -->
    <div class="provider-card {ctx.codex_card_class}">
      <div class="provider-header">
        <h3 style="margin:0;font-size:1rem;color:#f1f5f9">🟢 Codex (OpenAI)</h3>
        <span class="provider-status {ctx.codex_status_class}">{ctx.codex_status}</span>
      </div>
      <div style="margin-bottom:8px"><span class="label">Auth Method:</span> <span class="value">Isolated HOME</span></div>
      <table style="font-size:0.8rem">
        <thead><tr><th>Model ID</th><th>Name</th><th>Notes</th></tr></thead>
        <tbody>{ctx.codex_rows}</tbody>
      </table>
    </div>

    <!-- Antigravity Provider -->
    <div class="provider-card {ctx.ag_card_class}">
      <div class="provider-header">
        <h3 style="margin:0;font-size:1rem;color:#f1f5f9">🔵 Antigravity (Google)</h3>
        <span class="provider-status {ctx.ag_status_class}">{ctx.ag_status}</span>
      </div>
      <div style="margin-bottom:8px">
        <span class="label">Auth Method:</span> <span class="value">Google OAuth</span>
        <span style="margin-left:12px" class="label">Accounts:</span> <span class="value">{ctx.ag_account_count}</span>
      </div>
      {ctx.ag_accounts_html}
      <div style="margin:8px 0">
        <a href="/config/antigravity-login" style="color:#67e8f9;font-size:0.85rem" target="_blank">+ Add Google Account →</a>
      </div>
      <table style="font-size:0.8rem">
        <thead><tr><th>Model ID</th><th>Name</th><th>Notes</th></tr></thead>
        <tbody>{ctx.antigravity_rows}</tbody>
      </table>
    </div>

    <!-- Gemini CLI Provider -->
    <div class="provider-card {ctx.gemini_cli_card_class}">
      <div class="provider-header">
        <h3 style="margin:0;font-size:1rem;color:#f1f5f9">🟡 Gemini CLI (Google)</h3>
        <span class="provider-status {ctx.gemini_cli_status_class}">{ctx.gemini_cli_status}</span>
      </div>
      <div style="margin-bottom:8px"><span class="label">Auth Method:</span> <span class="value">~/.gemini/oauth_creds.json</span></div>
      <div style="margin-bottom:8px"><span class="label">Prefix:</span> <span class="value">gcli-model-name</span></div>
      {ctx.gemini_cli_hint}
      <table style="font-size:0.8rem">
        <thead><tr><th>Model ID</th><th>Name</th><th>Notes</th></tr></thead>
        <tbody>{ctx.gemini_cli_rows}</tbody>
      </table>
    </div>

    <!-- Ollama Provider -->
    <div class="provider-card" style="border-color:{ctx.ollama_color}40">
      <div class="provider-header">
        <h3 style="margin:0;font-size:1rem;color:#f1f5f9">🦙 Ollama</h3>
        <span class="provider-status" style="background:{ctx.ollama_color}20;color:{ctx.ollama_color}">{ctx.ollama_status}</span>
      </div>
      <div style="margin-bottom:8px"><span class="label">Prefix:</span> <span class="value">ollama:model-name</span></div>
      <div style="color:#64748b;font-size:0.8rem">
        Pull models with <code style="color:#67e8f9">ollama pull model-name</code><br>
        Set API key below for Ollama Cloud.
      </div>
    </div>

    <!-- Z.AI Provider -->
    <div class="provider-card" style="border-color:{ctx.zai_color}40">
      <div class="provider-header">
        <h3 style="margin:0;font-size:1rem;color:#f1f5f9">⚡ Z.AI</h3>
        <span class="provider-status" style="background:{ctx.zai_color}20;color:{ctx.zai_color}">{ctx.zai_status}</span>
      </div>
      <div style="margin-bottom:8px"><span class="label">Prefix:</span> <span class="value">zai:model-name</span></div>
      <div style="margin-bottom:8px"><span class="label">Base URL:</span> <span class="value">api.z.ai/api/coding/paas/v4</span></div>
      <div style="color:#64748b;font-size:0.8rem">
        GLM models via Z.AI coding plan.<br>
        e.g. <code style="color:#67e8f9">zai:glm-4.7</code>, <code style="color:#67e8f9">zai:glm-5</code>
      </div>
    </div>

    <!-- Xiaomi MiMo Provider -->
    <div class="provider-card" style="border-color:{ctx.xiaomi_color}40">
      <div class="provider-header">
        <h3 style="margin:0;font-size:1rem;color:#f1f5f9">🟠 Xiaomi MiMo</h3>
        <span class="provider-status" style="background:{ctx.xiaomi_color}20;color:{ctx.xiaomi_color}">{ctx.xiaomi_status}</span>
      </div>
      <div style="margin-bottom:8px"><span class="label">Prefix:</span> <span class="value">xiaomi:model-name</span></div>
      <div style="margin-bottom:8px"><span class="label">Base URL:</span> <span class="value">token-plan-sgp.xiaomimimo.com/v1</span></div>
      <div style="color:#64748b;font-size:0.8rem">
        Pro/Omni/TTS → token-plan-sgp endpoint.<br>
        Flash → api.xiaomimimo.com (open-source model).<br>
        e.g. <code style="color:#67e8f9">xiaomi:mimo-v2-pro</code>, <code style="color:#67e8f9">xiaomi:mimo-v2-omni</code>
      </div>
    </div>

    <!-- OpenCode Provider -->
    <div class="provider-card" style="border-color:{ctx.opencode_color}40">
      <div class="provider-header">
        <h3 style="margin:0;font-size:1rem;color:#f1f5f9">🌐 OpenCode</h3>
        <span class="provider-status" style="background:{ctx.opencode_color}20;color:{ctx.opencode_color}">{ctx.opencode_status}</span>
      </div>
      <div style="margin-bottom:8px"><span class="label">Zen:</span> <span class="value">opencode:model-name</span></div>
      <div style="margin-bottom:8px"><span class="label">Go:</span> <span class="value">opencode-go:model-name</span></div>
      <div style="color:#64748b;font-size:0.8rem">
        Zen: 40+ curated models (Claude, Gemini, GPT, GLM, Qwen).<br>
        Go: Kimi, GLM, MiMo, Minimax.<br>
        e.g. <code style="color:#67e8f9">opencode-go:kimi-k2.5</code>, <code style="color:#67e8f9">opencode:glm-5</code>
      </div>
    </div>

    <!-- Qwen Code Provider -->
    <div class="provider-card {ctx.qwen_card_class}">
      <div class="provider-header">
        <h3 style="margin:0;font-size:1rem;color:#f1f5f9">🟣 Qwen Code</h3>
        <span class="provider-status {ctx.qwen_status_class}">{ctx.qwen_status}</span>
      </div>
      <div style="margin-bottom:8px"><span class="label">Auth Method:</span> <span class="value">~/.qwen/oauth_creds.json</span></div>
      <div style="margin-bottom:8px"><span class="label">Prefix:</span> <span class="value">qwen:model-name</span></div>
      {ctx.qwen_hint}
      <div style="color:#64748b;font-size:0.8rem">
        Free tier: 1,000 req/day via Qwen OAuth.<br>
        e.g. <code style="color:#67e8f9">qwen:coder-model</code>
      </div>
    </div>

    <!-- Fireworks Provider -->
    <div class="provider-card" style="border-color:{ctx.fireworks_color}40">
      <div class="provider-header">
        <h3 style="margin:0;font-size:1rem;color:#f1f5f9">🔥 Fireworks</h3>
        <span class="provider-status" style="background:{ctx.fireworks_color}20;color:{ctx.fireworks_color}">{ctx.fireworks_status}</span>
      </div>
      <div style="margin-bottom:8px"><span class="label">Prefix:</span> <span class="value">fireworks:model-name</span></div>
      <div style="margin-bottom:8px"><span class="label">Base URL:</span> <span class="value">api.fireworks.ai/inference/v1</span></div>
      <div style="color:#64748b;font-size:0.8rem">
        OpenAI-compatible API for hosted models.<br>
        e.g. <code style="color:#67e8f9">fireworks:accounts/fireworks/routers/kimi-k2p5-turbo</code>
      </div>
    </div>

    <!-- NVIDIA NIM Provider -->
    <div class="provider-card" style="border-color:{ctx.nvidia_color}40">
      <div class="provider-header">
        <h3 style="margin:0;font-size:1rem;color:#f1f5f9">🟢 NVIDIA NIM</h3>
        <span class="provider-status" style="background:{ctx.nvidia_color}20;color:{ctx.nvidia_color}">{ctx.nvidia_status}</span>
      </div>
      <div style="margin-bottom:8px"><span class="label">Prefix:</span> <span class="value">nvidia:model-name</span></div>
      <div style="margin-bottom:8px"><span class="label">Base URL:</span> <span class="value">integrate.api.nvidia.com/v1</span></div>
      <div style="color:#64748b;font-size:0.8rem">
        Free-tier OpenAI-compatible API for NVIDIA-hosted models.<br>
        e.g. <code style="color:#67e8f9">nvidia:deepseek-r1</code>, <code style="color:#67e8f9">nvidia:llama-3.3-70b</code>
      </div>
    </div>
  </div>

  <!-- Ollama Cloud Key -->
  <div class="card">
    <h3 style="margin:0 0 8px; font-size:1rem; color:#f1f5f9">🦙 Ollama Cloud Key</h3>
    <div style="display:flex; gap:8px; align-items:center">
      <input type="password" id="ollamaKey" placeholder="API key"
             style="flex:1; background:#0f172a; color:#e2e8f0; border:1px solid #334155;
                    border-radius:6px; padding:8px 12px; font-family:monospace; font-size:0.85rem">
      <button onclick="saveOllamaKey()" style="margin-top:0">Save</button>
      <span id="ollamaStatus" style="color:{ctx.ollama_color}; font-size:0.85rem; font-weight:600">{ctx.ollama_status}</span>
    </div>
    <div style="color:#64748b; font-size:0.8rem; margin-top:8px">
      Leave empty to use local Ollama at <code style="color:#67e8f9">localhost:11434</code>
    </div>
  </div>

  <!-- Fireworks Key -->
  <div class="card">
    <h3 style="margin:0 0 8px; font-size:1rem; color:#f1f5f9">🔥 Fireworks API Key</h3>
    <div style="display:flex; gap:8px; align-items:center">
      <input type="password" id="fireworksKey" placeholder="API key (fw_...)"
             style="flex:1; background:#0f172a; color:#e2e8f0; border:1px solid #334155;
                    border-radius:6px; padding:8px 12px; font-family:monospace; font-size:0.85rem">
      <button onclick="saveFireworksKey()" style="margin-top:0">Save</button>
      <span id="fireworksStatus" style="color:{ctx.fireworks_color}; font-size:0.85rem; font-weight:600">{ctx.fireworks_status}</span>
    </div>
    <div style="color:#64748b; font-size:0.8rem; margin-top:8px">
      API key from <code style="color:#67e8f9">console.fireworks.ai</code>
    </div>
  </div>

  <!-- NVIDIA NIM Key -->
  <div class="card">
    <h3 style="margin:0 0 8px; font-size:1rem; color:#f1f5f9">🟢 NVIDIA NIM API Key</h3>
    <div style="display:flex; gap:8px; align-items:center">
      <input type="password" id="nvidiaKey" placeholder="API key (nvapi-...)"
             style="flex:1; background:#0f172a; color:#e2e8f0; border:1px solid #334155;
                    border-radius:6px; padding:8px 12px; font-family:monospace; font-size:0.85rem">
      <button onclick="saveNvidiaKey()" style="margin-top:0">Save</button>
      <span id="nvidiaStatus" style="color:{ctx.nvidia_color}; font-size:0.85rem; font-weight:600">{ctx.nvidia_status}</span>
    </div>
    <div style="color:#64748b; font-size:0.8rem; margin-top:8px">
      API key from <code style="color:#67e8f9">build.nvidia.com</code>
    </div>
  </div>

  <!-- Z.AI Key -->
  <div class="card">
    <h3 style="margin:0 0 8px; font-size:1rem; color:#f1f5f9">⚡ Z.AI API Key</h3>
    <div style="display:flex; gap:8px; align-items:center">
      <input type="password" id="zaiKey" placeholder="API key ({{ID}}.{{secret}})"
             style="flex:1; background:#0f172a; color:#e2e8f0; border:1px solid #334155;
                    border-radius:6px; padding:8px 12px; font-family:monospace; font-size:0.85rem">
      <button onclick="saveZaiKey()" style="margin-top:0">Save</button>
      <span id="zaiStatus" style="color:{ctx.zai_color}; font-size:0.85rem; font-weight:600">{ctx.zai_status}</span>
    </div>
    <div style="color:#64748b; font-size:0.8rem; margin-top:8px">
      Coding plan key from <code style="color:#67e8f9">z.ai/manage-apikey/apikey-list</code>
    </div>
  </div>

  <!-- Xiaomi MiMo Key -->
  <div class="card">
    <h3 style="margin:0 0 8px; font-size:1rem; color:#f1f5f9">🟠 Xiaomi MiMo API Key</h3>
    <div style="display:flex; gap:8px; align-items:center">
      <input type="password" id="xiaomiKey" placeholder="API key"
             style="flex:1; background:#0f172a; color:#e2e8f0; border:1px solid #334155;
                    border-radius:6px; padding:8px 12px; font-family:monospace; font-size:0.85rem">
      <button onclick="saveXiaomiKey()" style="margin-top:0">Save</button>
      <span id="xiaomiStatus" style="color:{ctx.xiaomi_color}; font-size:0.85rem; font-weight:600">{ctx.xiaomi_status}</span>
    </div>
    <div style="color:#64748b; font-size:0.8rem; margin-top:8px">
      Token plan key from <code style="color:#67e8f9">platform.xiaomimimo.com</code> — SGP endpoint
    </div>
  </div>

  <!-- OpenCode Zen Key -->
  <div class="card">
    <h3 style="margin:0 0 8px; font-size:1rem; color:#f1f5f9">🌐 OpenCode Zen API Key</h3>
    <div style="display:flex; gap:8px; align-items:center">
      <input type="password" id="opencodeKey" placeholder="API key (sk-...)"
             style="flex:1; background:#0f172a; color:#e2e8f0; border:1px solid #334155;
                    border-radius:6px; padding:8px 12px; font-family:monospace; font-size:0.85rem">
      <button onclick="saveOpencodeKey()" style="margin-top:0">Save</button>
      <span id="opencodeStatus" style="color:{ctx.opencode_color}; font-size:0.85rem; font-weight:600">{ctx.opencode_status}</span>
    </div>
    <div style="color:#64748b; font-size:0.8rem; margin-top:8px">
      Auto-loaded from <code style="color:#67e8f9">~/.local/share/opencode/auth.json</code> if present
    </div>
  </div>

  <!-- OpenCode Go Key -->
  <div class="card">
    <h3 style="margin:0 0 8px; font-size:1rem; color:#f1f5f9">🌐 OpenCode Go API Key</h3>
    <div style="display:flex; gap:8px; align-items:center">
      <input type="password" id="opencodeGoKey" placeholder="API key (sk-...)"
             style="flex:1; background:#0f172a; color:#e2e8f0; border:1px solid #334155;
                    border-radius:6px; padding:8px 12px; font-family:monospace; font-size:0.85rem">
      <button onclick="saveOpencodeGoKey()" style="margin-top:0">Save</button>
      <span id="opencodeGoStatus" style="color:{ctx.opencode_go_status_color}; font-size:0.85rem; font-weight:600">{ctx.opencode_go_status_text}</span>
    </div>
    <div style="color:#64748b; font-size:0.8rem; margin-top:8px">
      Go plan key — auto-loaded from <code style="color:#67e8f9">~/.local/share/opencode/auth.json</code> if present
    </div>
  </div>

  <!-- OpenRouter -->
  <div class="card">
    <h3 style="margin:0 0 8px; font-size:1rem; color:#f1f5f9">🔵 OpenRouter</h3>
    <div style="display:flex; gap:8px; align-items:center">
      <input type="password" id="orKey" placeholder="API key (sk-or-...)"
             style="flex:1; background:#0f172a; color:#e2e8f0; border:1px solid #334155;
                    border-radius:6px; padding:8px 12px; font-family:monospace; font-size:0.85rem">
      <button onclick="saveOrKey()" style="margin-top:0">Save</button>
      <span id="orStatus" style="color:{ctx.or_color}; font-size:0.85rem; font-weight:600">{ctx.or_status}</span>
    </div>
    <div style="color:#64748b; font-size:0.8rem; margin-top:8px">
      Use <code style="color:#67e8f9">provider/model</code> IDs (e.g. <code style="color:#67e8f9">openai/gpt-4o</code>)
    </div>
  </div>

  <!-- Timeout Routing -->
  <div class="card">
    <h3 style="margin:0 0 8px; font-size:1rem; color:#f1f5f9">⚡ Timeout Routing <span style="font-size:0.75rem;font-weight:400;color:#64748b">(streaming only)</span></h3>
    <p style="margin:0 0 10px;color:#94a3b8;font-size:0.85rem">
      <b>TTFT cutoff</b>: if no content token arrives in time, the stream is cancelled and re-routed to the most reliable model.
      <b>Max total</b>: hard wall-clock ceiling for the whole stream — anything past this is cut mid-generation with <code>[DONE]</code>.
      Non-streaming (direct) calls are NOT bounded — memory/summary workflows need the headroom.
    </p>
    <div style="display:flex; gap:12px; align-items:center; flex-wrap:wrap">
      <label style="display:flex;gap:8px;align-items:center;cursor:pointer">
        <input type="checkbox" id="timeoutRoutingToggle" {ctx.timeout_routing_checked}
               onchange="setTimeoutRouting(this.checked)"
               style="width:16px;height:16px;cursor:pointer">
        <span style="color:#f1f5f9;font-size:0.9rem">Enabled</span>
      </label>
      <span id="timeoutRoutingStatus" style="color:{ctx.timeout_routing_color};font-size:0.85rem;font-weight:600">
        {ctx.timeout_routing_label}
      </span>
      <label style="display:flex;gap:6px;align-items:center">
        <span style="color:#94a3b8;font-size:0.85rem">TTFT cutoff:</span>
        <input type="number" id="timeoutCutoffInput" value="{ctx.timeout_cutoff_seconds}" min="1" max="30" step="0.5"
               style="width:60px;background:#1e293b;border:1px solid #334155;color:#f1f5f9;padding:4px 6px;border-radius:4px;font-size:0.85rem">
        <span style="color:#94a3b8;font-size:0.85rem">s</span>
        <button onclick="setTimeoutCutoff()" style="margin-top:0;background:#334155;padding:4px 10px;font-size:0.8rem">Set</button>
      </label>
      <label style="display:flex;gap:6px;align-items:center">
        <span style="color:#94a3b8;font-size:0.85rem">Max total:</span>
        <input type="number" id="maxTotalInput" value="{ctx.max_total_seconds}" min="1" max="60" step="0.5"
               style="width:60px;background:#1e293b;border:1px solid #334155;color:#f1f5f9;padding:4px 6px;border-radius:4px;font-size:0.85rem">
        <span style="color:#94a3b8;font-size:0.85rem">s</span>
        <button onclick="setMaxTotal()" style="margin-top:0;background:#334155;padding:4px 10px;font-size:0.8rem">Set</button>
      </label>
      <button onclick="loadRoutingStats()" style="margin-top:0;background:#334155">View Stats</button>
      <button onclick="loadRequestStats()" style="margin-top:0;background:#334155">Request Stats</button>
    </div>
    <pre id="routingStats" style="display:none;margin-top:10px;background:#0f172a;padding:10px;border-radius:6px;font-size:0.8rem;color:#94a3b8;overflow:auto"></pre>
    <pre id="requestStats" style="display:none;margin-top:10px;background:#0f172a;padding:10px;border-radius:6px;font-size:0.8rem;color:#94a3b8;overflow:auto"></pre>
  </div>

  <!-- Retry Policy -->
  <div class="card">
    <h3 style="margin:0 0 8px; font-size:1rem; color:#f1f5f9">🔁 Retry Policy</h3>
    <p style="margin:0 0 10px;color:#94a3b8;font-size:0.85rem">
      Retries on 429/5xx and transient network errors. <code>0</code> disables retry; max <code>10</code>.
      Applies to OAI-compatible upstreams (OpenRouter, Ollama, Z.AI, Xiaomi, OpenCode, Qwen, Fireworks, NVIDIA NIM).
      Claude/Codex/Gemini CLI/Antigravity have their own mitigations and are not wrapped.
    </p>
    <div style="display:flex; gap:12px; align-items:center; flex-wrap:wrap">
      <label style="display:flex;gap:6px;align-items:center">
        <span style="color:#94a3b8;font-size:0.85rem">Max retries (0–10):</span>
        <input type="number" id="maxRetriesInput" value="{ctx.max_retries}" min="0" max="10" step="1"
               style="width:60px;background:#1e293b;border:1px solid #334155;color:#f1f5f9;padding:4px 6px;border-radius:4px;font-size:0.85rem">
        <button onclick="setMaxRetries()" style="margin-top:0;background:#334155;padding:4px 10px;font-size:0.8rem">Save</button>
      </label>
      <span id="maxRetriesStatus" style="color:#64748b;font-size:0.85rem;font-weight:600"></span>
    </div>
  </div>

  <!-- Quick Test -->
  <div class="card">
    <h3 style="margin:0 0 8px; font-size:1rem; color:#f1f5f9">Quick Test</h3>

    <!-- Model Selection -->
    <div style="margin-bottom:12px">
      <div style="display:flex; gap:12px; align-items:center; margin-bottom:8px">
        <label style="display:flex;align-items:center;gap:6px;cursor:pointer">
          <input type="radio" name="quickModelSource" value="dropdown" checked
                 onchange="toggleQuickModelSource()"
                 style="cursor:pointer">
          <span style="color:#e2e8f0;font-size:0.9rem">Select from list</span>
        </label>
        <label style="display:flex;align-items:center;gap:6px;cursor:pointer">
          <input type="radio" name="quickModelSource" value="custom"
                 onchange="toggleQuickModelSource()"
                 style="cursor:pointer">
          <span style="color:#e2e8f0;font-size:0.9rem">Custom model ID</span>
        </label>
      </div>

      <select id="modelSelect">
        <optgroup label="Claude Models">
          <option value="claude-sonnet-4-6">claude-sonnet-4-6 (default)</option>
          <option value="claude-opus-4-6">claude-opus-4-6</option>
          <option value="claude-sonnet-4-5-20250929">claude-sonnet-4-5-20250929</option>
          <option value="claude-haiku-4-5-20251001">claude-haiku-4-5-20251001</option>
        </optgroup>
        <optgroup label="Codex Models">
          <option value="gpt-5.4">gpt-5.4</option>
          <option value="gpt-5.2">gpt-5.2</option>
          <option value="gpt-5.2-codex">gpt-5.2-codex</option>
          <option value="gpt-5.1-codex-max">gpt-5.1-codex-max</option>
          <option value="gpt-5.1-codex-mini">gpt-5.1-codex-mini</option>
          <option value="gpt-5.1">gpt-5.1</option>
          <option value="gpt-5.1-codex">gpt-5.1-codex</option>
          <option value="gpt-5-codex">gpt-5-codex</option>
          <option value="gpt-5-codex-mini">gpt-5-codex-mini</option>
          <option value="gpt-5">gpt-5</option>
        </optgroup>
        <optgroup label="Antigravity Models">
          <option value="antigravity-gemini-3.1-pro">antigravity-gemini-3.1-pro</option>
          <option value="antigravity-gemini-3-pro">antigravity-gemini-3-pro</option>
          <option value="antigravity-gemini-3-flash">antigravity-gemini-3-flash</option>
          <option value="antigravity-gemini-2.5-pro">antigravity-gemini-2.5-pro</option>
          <option value="antigravity-gemini-2.5-flash">antigravity-gemini-2.5-flash</option>
          <option value="antigravity-claude-sonnet-4-6">antigravity-claude-sonnet-4-6</option>
          <option value="antigravity-claude-opus-4-6-thinking">antigravity-claude-opus-4-6-thinking</option>
        </optgroup>
        <optgroup label="Gemini CLI Models">
          <option value="gcli-gemini-2.5-pro">gcli-gemini-2.5-pro</option>
          <option value="gcli-gemini-2.5-flash">gcli-gemini-2.5-flash</option>
          <option value="gcli-gemini-2.5-flash-lite">gcli-gemini-2.5-flash-lite</option>
          <option value="gcli-gemini-3-pro-preview">gcli-gemini-3-pro-preview</option>
          <option value="gcli-gemini-3.1-pro-preview">gcli-gemini-3.1-pro-preview</option>
          <option value="gcli-gemini-3-flash-preview">gcli-gemini-3-flash-preview</option>
          <option value="gcli-gemini-3.1-flash-lite-preview">gcli-gemini-3.1-flash-lite-preview</option>
        </optgroup>
        <optgroup label="Ollama Models" id="ollamaOptgroup">
          <option disabled value="">Loading…</option>
        </optgroup>
        <optgroup label="Z.AI Models" id="zaiOptgroup">
          <option disabled value="">Loading…</option>
        </optgroup>
        <optgroup label="Xiaomi MiMo Models">
          <option value="xiaomi:mimo-v2-pro">xiaomi:mimo-v2-pro</option>
          <option value="xiaomi:mimo-v2-omni">xiaomi:mimo-v2-omni</option>
          <option value="xiaomi:mimo-v2-tts">xiaomi:mimo-v2-tts</option>
          <option value="xiaomi:mimo-v2-flash">xiaomi:mimo-v2-flash (platform)</option>
        </optgroup>
        <optgroup label="OpenCode Zen Models (Free)">
          <option value="opencode:big-pickle">opencode:big-pickle</option>
          <option value="opencode:minimax-m2.5-free">opencode:minimax-m2.5-free</option>
          <option value="opencode:nemotron-3-super-free">opencode:nemotron-3-super-free</option>
        </optgroup>
        <optgroup label="OpenCode Go Models">
          <option value="opencode-go:glm-5">opencode-go:glm-5</option>
          <option value="opencode-go:glm-5.1">opencode-go:glm-5.1</option>
          <option value="opencode-go:kimi-k2.5">opencode-go:kimi-k2.5</option>
          <option value="opencode-go:mimo-v2-omni">opencode-go:mimo-v2-omni</option>
          <option value="opencode-go:mimo-v2-pro">opencode-go:mimo-v2-pro</option>
          <option value="opencode-go:minimax-m2.5">opencode-go:minimax-m2.5</option>
          <option value="opencode-go:minimax-m2.7">opencode-go:minimax-m2.7</option>
        </optgroup>
        <optgroup label="Qwen Code Models">
          <option value="qwen:coder-model">qwen:coder-model</option>
        </optgroup>
        <optgroup label="Fireworks Models">
          <option value="fireworks:accounts/fireworks/routers/kimi-k2p5-turbo">fireworks:kimi-k2p5-turbo</option>
        </optgroup>
        <optgroup label="NVIDIA NIM Models (Free)">
          <option value="nvidia:kimi-k2">nvidia:kimi-k2</option>
          <option value="nvidia:kimi-k2-0905">nvidia:kimi-k2-0905</option>
          <option value="nvidia:kimi-k2-thinking">nvidia:kimi-k2-thinking</option>
          <option value="nvidia:glm-4.7">nvidia:glm-4.7</option>
          <option value="nvidia:deepseek-v3.2">nvidia:deepseek-v3.2</option>
          <option value="nvidia:deepseek-v3.1">nvidia:deepseek-v3.1</option>
          <option value="nvidia:deepseek-v3.1-terminus">nvidia:deepseek-v3.1-terminus</option>
          <option value="nvidia:step-3.5-flash">nvidia:step-3.5-flash</option>
          <option value="nvidia:llama-4-maverick">nvidia:llama-4-maverick</option>
          <option value="nvidia:gemma-3-27b">nvidia:gemma-3-27b</option>
          <option value="nvidia:mistral-large-3">nvidia:mistral-large-3</option>
          <option value="nvidia:devstral-2">nvidia:devstral-2</option>
          <option value="nvidia:magistral-small">nvidia:magistral-small</option>
          <option value="nvidia:mistral-nemotron">nvidia:mistral-nemotron</option>
          <option value="nvidia:qwen3-coder-480b">nvidia:qwen3-coder-480b</option>
          <option value="nvidia:seed-oss-36b">nvidia:seed-oss-36b</option>
          <option value="nvidia:phi-3.5-mini">nvidia:phi-3.5-mini</option>
        </optgroup>
      </select>

      <input type="text" id="quickModelInput" placeholder="Enter custom model ID (e.g., claude-opus-4-6)"
             style="display:none; width:100%; background:#0f172a; color:#e2e8f0; border:1px solid #334155;
                    border-radius:6px; padding:8px 12px; font-family:monospace; font-size:0.85rem; box-sizing:border-box">

      <div id="quickModelDisplay" style="color:#4ade80; font-size:0.85rem; font-family:monospace; margin-top:4px">
        Using: <span id="quickCurrentModelId">-select a model-</span>
      </div>
    </div>
    <textarea id="sysPrompt" rows="2" placeholder="System prompt">You are Lydia, a Nord housecarl sworn to protect the Dragonborn. Stay in character. One sentence only.</textarea>
    <textarea id="userMsg" rows="1" placeholder="User message" style="margin-top:6px">What do you think of dragons?</textarea>
    <button onclick="testChat()" id="testBtn">Send</button>
    <div id="response" style="display:none"></div>
    <div id="timing" class="timing"></div>
  </div>

  <!-- Vision Test -->
  <div class="card">
    <h3 style="margin:0 0 8px; font-size:1rem; color:#f1f5f9">Vision Test</h3>

    <!-- Model Selection -->
    <div style="margin-bottom:12px">
      <div style="display:flex; gap:12px; align-items:center; margin-bottom:8px">
        <label style="display:flex;align-items:center;gap:6px;cursor:pointer">
          <input type="radio" name="visionModelSource" value="dropdown" checked
                 onchange="toggleVisionModelSource()"
                 style="cursor:pointer">
          <span style="color:#e2e8f0;font-size:0.9rem">Select from list</span>
        </label>
        <label style="display:flex;align-items:center;gap:6px;cursor:pointer">
          <input type="radio" name="visionModelSource" value="custom"
                 onchange="toggleVisionModelSource()"
                 style="cursor:pointer">
          <span style="color:#e2e8f0;font-size:0.9rem">Custom model ID</span>
        </label>
      </div>

      <select id="visionModelSelect" style="width:100%; margin-bottom:8px">
        <option disabled value="">Loading available vision models…</option>
      </select>

      <input type="text" id="visionModelInput" placeholder="Enter custom model ID (e.g., ollama:llava)"
             style="display:none; width:100%; background:#0f172a; color:#e2e8f0; border:1px solid #334155;
                    border-radius:6px; padding:8px 12px; font-family:monospace; font-size:0.85rem; margin-bottom:8px; box-sizing:border-box">

      <div id="visionModelDisplay" style="color:#4ade80; font-size:0.85rem; font-family:monospace; margin-top:4px">
        Using: <span id="visionCurrentModelId">-select a model-</span>
      </div>
    </div>

    <textarea id="visionPrompt" rows="2" placeholder="Describe what you want to know about the image">What do you see in this image? Describe it briefly.</textarea>

    <!-- Image Selection -->
    <div style="margin-top:12px; padding:12px; background:#0f172a; border-radius:6px; border:1px solid #334155">
      <div style="font-size:0.85rem; color:#94a3b8; margin-bottom:8px">Test Image:</div>
      <div style="display:flex; gap:12px; align-items:center; flex-wrap:wrap">
        <label style="display:flex;align-items:center;gap:6px;cursor:pointer">
          <input type="radio" name="visionImageSource" value="random" checked
                 onchange="toggleVisionImageSource()"
                 style="cursor:pointer">
          <span style="color:#e2e8f0;font-size:0.85rem">Random test image</span>
        </label>
        <label style="display:flex;align-items:center;gap:6px;cursor:pointer">
          <input type="radio" name="visionImageSource" value="upload"
                 onchange="toggleVisionImageSource()"
                 style="cursor:pointer">
          <span style="color:#e2e8f0;font-size:0.85rem">Upload image</span>
        </label>
      </div>

      <div id="visionRandomPreview" style="margin-top:10px">
        <div style="width:120px; height:120px; border:2px dashed #334155; border-radius:6px; display:flex; align-items:center; justify-content:center; background:#1e293b; position:relative; overflow:hidden">
          <canvas id="visionRandomCanvas" width="120" height="120"></canvas>
        </div>
        <button onclick="generateRandomVisionImage()" type="button" style="margin-top:8px; background:#334155; padding:6px 12px; font-size:0.8rem">Generate new image</button>
      </div>

      <div id="visionUploadSection" style="display:none; margin-top:10px">
        <input type="file" id="visionImageFile" accept="image/*"
               style="color:#e2e8f0; font-size:0.85rem">
        <div style="color:#64748b; font-size:0.75rem; margin-top:6px">Supports: PNG, JPG, GIF, WebP (max 4MB)</div>
      </div>
    </div>

    <button onclick="testVision()" id="visionTestBtn" style="margin-top:12px">Analyze Image</button>
    <div id="visionResponse" style="display:none; margin-top:12px; padding:12px; background:#0f172a; border-radius:6px; border:1px solid #334155; font-size:0.9rem; min-height:40px; white-space:pre-wrap"></div>
    <div id="visionTiming" class="timing"></div>
  </div>

<script>
async function saveOrKey() {{
  const key = document.getElementById('orKey').value.trim();
  const status = document.getElementById('orStatus');
  try {{
    await fetch('/config/openrouter-key', {{
      method: 'POST', headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{key: key}})
    }});
    status.textContent = key ? 'Saved' : 'Not set';
    status.style.color = key ? '#4ade80' : '#64748b';
    document.getElementById('orKey').value = '';
  }} catch(e) {{
    status.textContent = 'Error';
    status.style.color = '#f87171';
  }}
}}

async function saveOllamaKey() {{
  const key = document.getElementById('ollamaKey').value.trim();
  const status = document.getElementById('ollamaStatus');
  try {{
    await fetch('/config/ollama-key', {{
      method: 'POST', headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{key: key}})
    }});
    status.textContent = key ? 'Cloud (key configured)' : 'Local (localhost:11434)';
    status.style.color = key ? '#4ade80' : '#64748b';
    document.getElementById('ollamaKey').value = '';
  }} catch(e) {{
    status.textContent = 'Error';
    status.style.color = '#f87171';
  }}
}}

async function saveFireworksKey() {{
  const key = document.getElementById('fireworksKey').value.trim();
  const status = document.getElementById('fireworksStatus');
  try {{
    await fetch('/config/fireworks-key', {{
      method: 'POST', headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{key: key}})
    }});
    status.textContent = key ? 'Configured (saved)' : 'Not set';
    status.style.color = key ? '#4ade80' : '#64748b';
    document.getElementById('fireworksKey').value = '';
  }} catch(e) {{
    status.textContent = 'Error';
    status.style.color = '#f87171';
  }}
}}

async function saveNvidiaKey() {{
  const key = document.getElementById('nvidiaKey').value.trim();
  const status = document.getElementById('nvidiaStatus');
  try {{
    await fetch('/config/nvidia-key', {{
      method: 'POST', headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{key: key}})
    }});
    status.textContent = key ? 'Configured (saved)' : 'Not set';
    status.style.color = key ? '#4ade80' : '#64748b';
    document.getElementById('nvidiaKey').value = '';
  }} catch(e) {{
    status.textContent = 'Error';
    status.style.color = '#f87171';
  }}
}}

async function saveZaiKey() {{
  const key = document.getElementById('zaiKey').value.trim();
  const status = document.getElementById('zaiStatus');
  try {{
    await fetch('/config/zai-key', {{
      method: 'POST', headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{key: key}})
    }});
    status.textContent = key ? 'Configured (saved)' : 'Not set';
    status.style.color = key ? '#4ade80' : '#64748b';
    document.getElementById('zaiKey').value = '';
  }} catch(e) {{
    status.textContent = 'Error';
    status.style.color = '#f87171';
  }}
}}

async function saveXiaomiKey() {{
  const key = document.getElementById('xiaomiKey').value.trim();
  const status = document.getElementById('xiaomiStatus');
  try {{
    await fetch('/config/xiaomi-key', {{
      method: 'POST', headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{key: key}})
    }});
    status.textContent = key ? 'Configured (saved)' : 'Not set';
    status.style.color = key ? '#4ade80' : '#64748b';
    document.getElementById('xiaomiKey').value = '';
  }} catch(e) {{
    status.textContent = 'Error';
    status.style.color = '#f87171';
  }}
}}

async function saveOpencodeKey() {{
  const key = document.getElementById('opencodeKey').value.trim();
  const status = document.getElementById('opencodeStatus');
  try {{
    await fetch('/config/opencode-key', {{
      method: 'POST', headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{key: key}})
    }});
    status.textContent = key ? 'Configured (saved)' : 'Not set';
    status.style.color = key ? '#4ade80' : '#64748b';
    document.getElementById('opencodeKey').value = '';
  }} catch(e) {{
    status.textContent = 'Error';
    status.style.color = '#f87171';
  }}
}}

async function saveOpencodeGoKey() {{
  const key = document.getElementById('opencodeGoKey').value.trim();
  const status = document.getElementById('opencodeGoStatus');
  try {{
    await fetch('/config/opencode-go-key', {{
      method: 'POST', headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{key: key}})
    }});
    status.textContent = key ? 'Configured (saved)' : 'Not set';
    status.style.color = key ? '#4ade80' : '#64748b';
    document.getElementById('opencodeGoKey').value = '';
  }} catch(e) {{
    status.textContent = 'Error';
    status.style.color = '#f87171';
  }}
}}

async function testChat() {{
  const btn = document.getElementById('testBtn');
  const resp = document.getElementById('response');
  const timing = document.getElementById('timing');

  // Get model from dropdown or custom input
  const source = document.querySelector('input[name="quickModelSource"]:checked').value;
  let model = '';
  if (source === 'dropdown') {{
    model = document.getElementById('modelSelect').value;
  }} else {{
    model = document.getElementById('quickModelInput').value.trim();
  }}

  if (!model) {{
    resp.style.display = 'block';
    resp.textContent = 'Please select or enter a model ID';
    return;
  }}

  btn.disabled = true; btn.textContent = 'Waiting...';
  resp.style.display = 'block'; resp.textContent = '...';
  timing.textContent = '';
  const start = Date.now();
  try {{
    const r = await fetch('/v1/chat/completions', {{
      method: 'POST', headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{
        model: model,
        messages: [
          {{role: 'system', content: document.getElementById('sysPrompt').value}},
          {{role: 'user', content: document.getElementById('userMsg').value}}
        ]
      }})
    }});
    const data = await r.json();
    const elapsed = ((Date.now() - start) / 1000).toFixed(1);
    if (data.choices) {{
      resp.textContent = data.choices[0].message.content;
      timing.textContent = model + ' · ' + elapsed + 's';
    }} else {{
      resp.textContent = JSON.stringify(data, null, 2);
    }}
  }} catch(e) {{
    resp.textContent = 'Error: ' + e.message;
  }}
  btn.disabled = false; btn.textContent = 'Send';
}}

async function loadOllamaModels() {{
  const group = document.getElementById('ollamaOptgroup');
  try {{
    const r = await fetch('/config/ollama-models');
    const data = await r.json();
    group.innerHTML = '';
    if (data.models && data.models.length > 0) {{
      data.models.forEach(function(m) {{
        const opt = document.createElement('option');
        opt.value = 'ollama:' + m;
        opt.textContent = 'ollama:' + m;
        group.appendChild(opt);
      }});
    }} else {{
      const opt = document.createElement('option');
      opt.disabled = true;
      opt.value = '';
      opt.textContent = 'No local models (run: ollama pull <model>)';
      group.appendChild(opt);
    }}
  }} catch(e) {{
    group.innerHTML = '<option disabled value="">Ollama not running</option>';
  }}
}}
async function loadZaiModels() {{
  const group = document.getElementById('zaiOptgroup');
  try {{
    const r = await fetch('/config/zai-models');
    const data = await r.json();
    group.innerHTML = '';
    if (data.models && data.models.length > 0) {{
      data.models.forEach(function(m) {{
        const opt = document.createElement('option');
        opt.value = 'zai:' + m;
        opt.textContent = 'zai:' + m;
        group.appendChild(opt);
      }});
    }} else {{
      const opt = document.createElement('option');
      opt.disabled = true;
      opt.value = '';
      opt.textContent = 'No Z.AI models (check API key)';
      group.appendChild(opt);
    }}
  }} catch(e) {{
    group.innerHTML = '<option disabled value="">Z.AI unavailable</option>';
  }}
}}
document.addEventListener('DOMContentLoaded', loadOllamaModels);
document.addEventListener('DOMContentLoaded', loadZaiModels);
document.addEventListener('DOMContentLoaded', loadVisionModels);
document.addEventListener('DOMContentLoaded', function() {{
  // Initialize vision test UI
  toggleVisionModelSource();
  generateRandomVisionImage();
  toggleVisionImageSource();
}});

async function loadVisionModels() {{
  const select = document.getElementById('visionModelSelect');
  try {{
    const r = await fetch('/config/vision-models');
    const data = await r.json();
    select.innerHTML = '';

    // Group models by provider
    const byProvider = {{}};
    if (data.models && data.models.length > 0) {{
      data.models.forEach(function(m) {{
        const provider = m.provider || 'Other';
        if (!byProvider[provider]) byProvider[provider] = [];
        byProvider[provider].push(m);
      }});
    }} else {{
      select.innerHTML = '<option disabled value="">No vision models available</option>';
      updateVisionModelDisplay();
      return;
    }}

    // Create optgroups
    const providerOrder = ['claude', 'openrouter', 'gemini-cli', 'antigravity', 'ollama', 'fireworks', 'nvidia'];
    const providerLabels = {{
      'claude': 'Claude',
      'openrouter': 'OpenRouter',
      'gemini-cli': 'Gemini CLI',
      'antigravity': 'Antigravity',
      'ollama': 'Ollama',
      'fireworks': 'Fireworks',
      'nvidia': 'NVIDIA NIM'
    }};

    providerOrder.forEach(function(provider) {{
      if (byProvider[provider] && byProvider[provider].length > 0) {{
        const optgroup = document.createElement('optgroup');
        optgroup.label = providerLabels[provider] || provider;
        byProvider[provider].forEach(function(m) {{
          const opt = document.createElement('option');
          opt.value = m.id;
          opt.textContent = m.name;
          optgroup.appendChild(opt);
        }});
        select.appendChild(optgroup);
      }}
    }});

    // Add any other providers not in the order list
    Object.keys(byProvider).forEach(function(provider) {{
      if (!providerOrder.includes(provider) && byProvider[provider].length > 0) {{
        const optgroup = document.createElement('optgroup');
        optgroup.label = providerLabels[provider] || provider;
        byProvider[provider].forEach(function(m) {{
          const opt = document.createElement('option');
          opt.value = m.id;
          opt.textContent = m.name;
          optgroup.appendChild(opt);
        }});
        select.appendChild(optgroup);
      }}
    }});

    // Set first model as selected and update display
    if (select.options.length > 0) {{
      select.selectedIndex = 0;
    }}
    updateVisionModelDisplay();

    // Add change listener
    select.addEventListener('change', updateVisionModelDisplay);

  }} catch(e) {{
    select.innerHTML = '<option disabled value="">Failed to load vision models</option>';
    console.error('Failed to load vision models:', e);
  }}
}}

function toggleVisionModelSource() {{
  const source = document.querySelector('input[name="visionModelSource"]:checked').value;
  const select = document.getElementById('visionModelSelect');
  const input = document.getElementById('visionModelInput');

  if (source === 'dropdown') {{
    select.style.display = 'block';
    input.style.display = 'none';
  }} else {{
    select.style.display = 'none';
    input.style.display = 'block';
    input.focus();
  }}
  updateVisionModelDisplay();
}}

function updateVisionModelDisplay() {{
  const source = document.querySelector('input[name="visionModelSource"]:checked').value;
  const display = document.getElementById('visionCurrentModelId');

  if (source === 'dropdown') {{
    const select = document.getElementById('visionModelSelect');
    display.textContent = select.value || '-select a model-';
  }} else {{
    const input = document.getElementById('visionModelInput');
    display.textContent = input.value.trim() || '-enter custom model ID-';
  }}
}}

// Update display when custom input changes
document.addEventListener('DOMContentLoaded', function() {{
  const input = document.getElementById('visionModelInput');
  if (input) {{
    input.addEventListener('input', updateVisionModelDisplay);
  }}
}});

function toggleQuickModelSource() {{
  const source = document.querySelector('input[name="quickModelSource"]:checked').value;
  const select = document.getElementById('modelSelect');
  const input = document.getElementById('quickModelInput');

  if (source === 'dropdown') {{
    select.style.display = 'block';
    input.style.display = 'none';
  }} else {{
    select.style.display = 'none';
    input.style.display = 'block';
    input.focus();
  }}
  updateQuickModelDisplay();
}}

function updateQuickModelDisplay() {{
  const source = document.querySelector('input[name="quickModelSource"]:checked').value;
  const display = document.getElementById('quickCurrentModelId');

  if (source === 'dropdown') {{
    const select = document.getElementById('modelSelect');
    display.textContent = select.value || '-select a model-';
  }} else {{
    const input = document.getElementById('quickModelInput');
    display.textContent = input.value.trim() || '-enter custom model ID-';
  }}
}}

// Update Quick Test display on page load and input changes
document.addEventListener('DOMContentLoaded', function() {{
  toggleQuickModelSource();

  const input = document.getElementById('quickModelInput');
  if (input) {{
    input.addEventListener('input', updateQuickModelDisplay);
  }}

  const select = document.getElementById('modelSelect');
  if (select) {{
    select.addEventListener('change', updateQuickModelDisplay);
  }}
}});

function toggleVisionImageSource() {{
  const source = document.querySelector('input[name="visionImageSource"]:checked').value;
  const randomSection = document.getElementById('visionRandomPreview');
  const uploadSection = document.getElementById('visionUploadSection');

  if (source === 'random') {{
    randomSection.style.display = 'block';
    uploadSection.style.display = 'none';
    generateRandomVisionImage();
  }} else {{
    randomSection.style.display = 'none';
    uploadSection.style.display = 'block';
  }}
}}

function generateRandomVisionImage() {{
  const canvas = document.getElementById('visionRandomCanvas');
  const ctx = canvas.getContext('2d');

  // Generate random colors
  const hue1 = Math.floor(Math.random() * 360);
  const hue2 = (hue1 + 120 + Math.floor(Math.random() * 120)) % 360;
  const hue3 = (hue2 + 120 + Math.floor(Math.random() * 120)) % 360;

  // Fill background
  ctx.fillStyle = `hsl(${{hue1}}, 60%, 20%)`;
  ctx.fillRect(0, 0, 120, 120);

  // Draw random shapes
  for (let i = 0; i < 8; i++) {{
    ctx.fillStyle = `hsl(${{Math.random() * 360}}, 70%, ${{40 + Math.random() * 40}}%)`;
    const x = Math.random() * 120;
    const y = Math.random() * 120;
    const size = 10 + Math.random() * 40;

    if (Math.random() > 0.5) {{
      // Circle
      ctx.beginPath();
      ctx.arc(x, y, size / 2, 0, Math.PI * 2);
      ctx.fill();
    }} else {{
      // Rectangle
      ctx.fillRect(x - size/2, y - size/2, size, size);
    }}
  }}

  // Add some lines
  for (let i = 0; i < 5; i++) {{
    ctx.strokeStyle = `hsl(${{Math.random() * 360}}, 80%, 60%)`;
    ctx.lineWidth = 2 + Math.random() * 4;
    ctx.beginPath();
    ctx.moveTo(Math.random() * 120, Math.random() * 120);
    ctx.lineTo(Math.random() * 120, Math.random() * 120);
    ctx.stroke();
  }}

  // Store the base64 for later use
  canvas.dataset.base64 = canvas.toDataURL('image/png');
}}

function getVisionImageBase64() {{
  const source = document.querySelector('input[name="visionImageSource"]:checked').value;

  if (source === 'random') {{
    const canvas = document.getElementById('visionRandomCanvas');
    return canvas.dataset.base64 || canvas.toDataURL('image/png');
  }} else {{
    const fileInput = document.getElementById('visionImageFile');
    return new Promise((resolve, reject) => {{
      if (!fileInput.files || fileInput.files.length === 0) {{
        reject(new Error('Please select an image file'));
        return;
      }}
      const file = fileInput.files[0];
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(new Error('Error reading image file'));
      reader.readAsDataURL(file);
    }});
  }}
}}

async function testVision() {{
  const btn = document.getElementById('visionTestBtn');
  const resp = document.getElementById('visionResponse');
  const timing = document.getElementById('visionTiming');

  // Get model from dropdown or custom input
  const source = document.querySelector('input[name="visionModelSource"]:checked').value;
  let model = '';
  if (source === 'dropdown') {{
    model = document.getElementById('visionModelSelect').value;
  }} else {{
    model = document.getElementById('visionModelInput').value.trim();
  }}

  if (!model) {{
    resp.style.display = 'block';
    resp.textContent = 'Please select or enter a model ID';
    return;
  }}

  // Check image source
  const imageSource = document.querySelector('input[name="visionImageSource"]:checked').value;
  if (imageSource === 'upload') {{
    const fileInput = document.getElementById('visionImageFile');
    if (!fileInput.files || fileInput.files.length === 0) {{
      resp.style.display = 'block';
      resp.textContent = 'Please select an image file';
      return;
    }}
  }}

  btn.disabled = true; btn.textContent = 'Analyzing...';
  resp.style.display = 'block'; resp.textContent = 'Processing image...';
  timing.textContent = '';

  const start = Date.now();

  try {{
    // Get image base64
    let base64Image;
    if (imageSource === 'random') {{
      base64Image = getVisionImageBase64();
    }} else {{
      base64Image = await getVisionImageBase64();
    }}

    const prompt = document.getElementById('visionPrompt').value || 'What do you see in this image?';

    const r = await fetch('/v1/chat/completions', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{
        model: model,
        messages: [
          {{role: 'user', content: [
            {{type: 'text', text: prompt}},
            {{type: 'image_url', image_url: {{url: base64Image}}}}
          ]}}
        ]
      }})
    }});

    const data = await r.json();
    const elapsed = ((Date.now() - start) / 1000).toFixed(1);

    if (data.choices && data.choices[0] && data.choices[0].message) {{
      resp.textContent = data.choices[0].message.content;
      timing.textContent = model + ' · ' + elapsed + 's';
    }} else if (data.error) {{
      resp.textContent = 'Error: ' + (data.error.message || JSON.stringify(data.error));
      timing.textContent = '';
    }} else {{
      resp.textContent = JSON.stringify(data, null, 2);
    }}
  }} catch(e) {{
    resp.textContent = 'Error: ' + e.message;
    timing.textContent = '';
  }}

  btn.disabled = false; btn.textContent = 'Analyze Image';
}}

async function setTimeoutRouting(enabled) {{
  const status = document.getElementById('timeoutRoutingStatus');
  try {{
    const resp = await fetch('/config/timeout-routing', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{enabled}}),
    }});
    const data = await resp.json();
    status.textContent = data.timeout_routing_enabled ? 'Active' : 'Disabled';
    status.style.color = data.timeout_routing_enabled ? '#4ade80' : '#64748b';
  }} catch(e) {{
    status.textContent = 'Error: ' + e.message;
  }}
}}

async function loadRoutingStats() {{
  const pre = document.getElementById('routingStats');
  pre.style.display = 'block';
  pre.textContent = 'Loading...';
  try {{
    const resp = await fetch('/config/timeout-routing');
    const data = await resp.json();
    if (Object.keys(data.stats).length === 0) {{
      pre.textContent = 'No data yet — stats accumulate as streaming requests are made.';
    }} else {{
      pre.textContent = JSON.stringify(data.stats, null, 2);
    }}
  }} catch(e) {{
    pre.textContent = 'Error: ' + e.message;
  }}
}}

async function setTimeoutCutoff() {{
  const input = document.getElementById('timeoutCutoffInput');
  const seconds = parseFloat(input.value);
  if (isNaN(seconds) || seconds < 1 || seconds > 30) {{
    alert('TTFT cutoff must be between 1 and 30 seconds');
    return;
  }}
  try {{
    const resp = await fetch('/config/timeout-cutoff', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{seconds}}),
    }});
    const data = await resp.json();
    input.value = data.timeout_cutoff_seconds;
  }} catch(e) {{
    alert('Error: ' + e.message);
  }}
}}

async function setMaxTotal() {{
  const input = document.getElementById('maxTotalInput');
  const seconds = parseFloat(input.value);
  if (isNaN(seconds) || seconds < 1 || seconds > 60) {{
    alert('Max total must be between 1 and 60 seconds');
    return;
  }}
  try {{
    const resp = await fetch('/config/max-total', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{seconds}}),
    }});
    const data = await resp.json();
    input.value = data.max_total_seconds;
  }} catch(e) {{
    alert('Error: ' + e.message);
  }}
}}

async function setMaxRetries() {{
  const input = document.getElementById('maxRetriesInput');
  const status = document.getElementById('maxRetriesStatus');
  const n = parseInt(input.value, 10);
  if (isNaN(n) || n < 0 || n > 10) {{
    status.textContent = 'Must be 0..10';
    status.style.color = '#f87171';
    return;
  }}
  try {{
    const resp = await fetch('/config/max-retries', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{max_retries: n}}),
    }});
    if (resp.ok) {{
      const data = await resp.json();
      input.value = data.max_retries;
      status.textContent = 'Saved (' + data.max_retries + ')';
      status.style.color = '#4ade80';
    }} else {{
      const err = await resp.json();
      status.textContent = 'Error: ' + (err.detail || resp.status);
      status.style.color = '#f87171';
    }}
  }} catch(e) {{
    status.textContent = 'Error: ' + e.message;
    status.style.color = '#f87171';
  }}
}}

async function loadRequestStats() {{
  const pre = document.getElementById('requestStats');
  pre.style.display = 'block';
  pre.textContent = 'Loading...';
  try {{
    const resp = await fetch('/stats');
    const data = await resp.json();
    if (Object.keys(data.by_model).length === 0) {{
      pre.textContent = 'No request data yet — stats accumulate as requests are made.';
    }} else {{
      pre.textContent = JSON.stringify(data, null, 2);
    }}
  }} catch(e) {{
    pre.textContent = 'Error: ' + e.message;
  }}
}}

async function removeAccount(email) {{
  if (!confirm('Remove account ' + email + '?')) return;
  try {{
    const resp = await fetch('/config/antigravity-account/' + encodeURIComponent(email), {{
      method: 'DELETE'
    }});
    const data = await resp.json();
    if (data.status === 'removed') {{
      alert('Account removed: ' + email);
      location.reload();
    }} else {{
      alert('Account not found: ' + email);
    }}
  }} catch(e) {{
    alert('Error: ' + e.message);
  }}
}}
</script>
</body></html>"""
