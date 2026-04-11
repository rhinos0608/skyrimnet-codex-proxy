"""Model listing endpoints (/v1/models, /api/models, /config/*-models)."""
import asyncio
import shutil
import subprocess

import aiohttp
from fastapi import APIRouter

router = APIRouter()


@router.get("/config/ollama-models")
async def get_ollama_models():
    """Run 'ollama list' and return installed model names."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "ollama", "list",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
        lines = stdout.decode(errors="replace").splitlines()
        models = []
        for line in lines[1:]:  # skip header row
            name = line.split()[0] if line.split() else ""
            if name:
                models.append(name)
        return {"models": models}
    except (FileNotFoundError, asyncio.TimeoutError, Exception):
        return {"models": []}


@router.get("/config/zai-models")
async def get_zai_models():
    """Fetch available models from Z.AI API."""
    import proxy
    if not proxy.zai_api_key:
        return {"models": []}
    zai_models = await proxy._fetch_zai_models()
    # Strip the "zai:" prefix since the dashboard JS adds it
    models = [m["id"][len("zai:"):] for m in zai_models if m.get("id", "").startswith("zai:")]
    return {"models": models}


@router.get("/v1/models")
@router.get("/api/models")
async def list_models():
    """List only models from providers that are currently authenticated and active."""
    import proxy
    data = []

    # Claude — only if auth is ready
    if proxy.auth.is_ready:
        claude_models = await proxy._fetch_claude_models()
        if claude_models:
            data.extend(claude_models)
        else:
            data.extend([
                {"id": "claude-opus-4-6", "object": "model", "owned_by": "anthropic"},
                {"id": "claude-sonnet-4-6", "object": "model", "owned_by": "anthropic"},
                {"id": "claude-sonnet-4-5-20250929", "object": "model", "owned_by": "anthropic"},
                {"id": "claude-haiku-4-5-20251001", "object": "model", "owned_by": "anthropic"},
            ])

    # Codex/OpenAI — only if auth is ready
    if proxy.codex_auth.is_ready:
        codex_models = await proxy._fetch_codex_models()
        if codex_models:
            data.extend(codex_models)
        else:
            data.extend([
                {"id": "gpt-5.4", "object": "model", "owned_by": "openai"},
                {"id": "gpt-5.2", "object": "model", "owned_by": "openai"},
                {"id": "gpt-5.2-codex", "object": "model", "owned_by": "openai"},
                {"id": "gpt-5.1-codex-max", "object": "model", "owned_by": "openai"},
                {"id": "gpt-5.1-codex-mini", "object": "model", "owned_by": "openai"},
                {"id": "gpt-5.1", "object": "model", "owned_by": "openai"},
                {"id": "gpt-5.1-codex", "object": "model", "owned_by": "openai"},
                {"id": "gpt-5-codex", "object": "model", "owned_by": "openai"},
                {"id": "gpt-5-codex-mini", "object": "model", "owned_by": "openai"},
                {"id": "gpt-5", "object": "model", "owned_by": "openai"},
            ])

    # Antigravity — only if auth is ready
    if proxy.antigravity_auth.is_ready:
        data.extend([
            {"id": "antigravity-gemini-3.1-pro", "object": "model", "owned_by": "google"},
            {"id": "antigravity-gemini-3-pro", "object": "model", "owned_by": "google"},
            {"id": "antigravity-gemini-3-flash", "object": "model", "owned_by": "google"},
            {"id": "antigravity-gemini-2.5-pro", "object": "model", "owned_by": "google"},
            {"id": "antigravity-gemini-2.5-flash", "object": "model", "owned_by": "google"},
            {"id": "antigravity-claude-sonnet-4-6", "object": "model", "owned_by": "anthropic"},
            {"id": "antigravity-claude-opus-4-6-thinking", "object": "model", "owned_by": "anthropic"},
            {"id": "antigravity-gpt-oss-120b-medium", "object": "model", "owned_by": "other"},
        ])

    # Ollama — probe local or cloud; skip if unreachable
    try:
        if proxy.ollama_api_key:
            endpoint = "https://ollama.com/v1/models"
            headers = {"Authorization": f"Bearer {proxy.ollama_api_key}"}
        else:
            endpoint = "http://localhost:11434/v1/models"
            headers = {}
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=2, connect=1)
        ) as s:
            async with s.get(endpoint, headers=headers) as ollama_resp:
                if ollama_resp.status == 200:
                    ollama_data = await ollama_resp.json()
                    for m in ollama_data.get("data", []):
                        model_id = m.get("id", "")
                        if model_id:
                            data.append({"id": f"ollama:{model_id}", "object": "model", "owned_by": "ollama"})
    except Exception:
        pass  # Ollama not running or unreachable — skip

    # Gemini — only if auth is ready. No dynamic listing: the Code Assist OAuth
    # token can't authenticate against generativelanguage.googleapis.com/v1beta/models,
    # so we serve a hardcoded list of known gcli model ids.
    if proxy.gemini_auth.is_ready:
        data.extend([
            {"id": "gcli-gemini-2.5-pro", "object": "model", "owned_by": "google"},
            {"id": "gcli-gemini-2.5-flash", "object": "model", "owned_by": "google"},
            {"id": "gcli-gemini-2.5-flash-lite", "object": "model", "owned_by": "google"},
            {"id": "gcli-gemini-3-pro-preview", "object": "model", "owned_by": "google"},
            {"id": "gcli-gemini-3.1-pro-preview", "object": "model", "owned_by": "google"},
            {"id": "gcli-gemini-3-flash-preview", "object": "model", "owned_by": "google"},
            {"id": "gcli-gemini-3.1-flash-lite-preview", "object": "model", "owned_by": "google"},
        ])

    # OpenRouter — only if API key is configured
    if proxy.openrouter_api_key:
        data.append({"id": "openrouter/*", "object": "model", "owned_by": "openrouter"})

    # Z.AI — dynamic fetch with static fallback
    if proxy.zai_api_key:
        zai_models = await proxy._fetch_zai_models()
        if zai_models:
            data.extend(zai_models)
        else:
            data.extend([
                {"id": "zai:glm-4.5-air", "object": "model", "owned_by": "zhipuai"},
                {"id": "zai:glm-4.5", "object": "model", "owned_by": "zhipuai"},
                {"id": "zai:glm-4.6", "object": "model", "owned_by": "zhipuai"},
                {"id": "zai:glm-4.7", "object": "model", "owned_by": "zhipuai"},
                {"id": "zai:glm-5", "object": "model", "owned_by": "zhipuai"},
                {"id": "zai:glm-5-turbo", "object": "model", "owned_by": "zhipuai"},
            ])

    # Xiaomi MiMo — only if API key is configured
    if proxy.xiaomi_api_key:
        data.extend([
            {"id": "xiaomi:mimo-v2-pro", "object": "model", "owned_by": "xiaomi"},
            {"id": "xiaomi:mimo-v2-omni", "object": "model", "owned_by": "xiaomi"},
            {"id": "xiaomi:mimo-v2-tts", "object": "model", "owned_by": "xiaomi"},
            {"id": "xiaomi:mimo-v2-flash", "object": "model", "owned_by": "xiaomi"},
        ])

    # OpenCode — Go plan gets full model list (dynamic), Zen plan is free-tier only (hardcoded)
    if proxy.opencode_api_key:
        data.extend([
            {"id": "opencode:big-pickle", "object": "model", "owned_by": "opencode"},
            {"id": "opencode:minimax-m2.5-free", "object": "model", "owned_by": "opencode"},
            {"id": "opencode:nemotron-3-super-free", "object": "model", "owned_by": "opencode"},
        ])
    if proxy.opencode_go_api_key:
        try:
            result = subprocess.run(
                ["opencode", "models", "opencode-go"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    line = line.strip()
                    if line.startswith("opencode-go/"):
                        proxy_id = f"opencode-go:{line.split('/', 1)[1]}"
                        data.append({"id": proxy_id, "object": "model", "owned_by": "opencode"})
        except Exception:
            data.extend([
                {"id": "opencode-go:glm-5", "object": "model", "owned_by": "opencode"},
                {"id": "opencode-go:glm-5.1", "object": "model", "owned_by": "opencode"},
                {"id": "opencode-go:kimi-k2.5", "object": "model", "owned_by": "opencode"},
                {"id": "opencode-go:mimo-v2-omni", "object": "model", "owned_by": "opencode"},
                {"id": "opencode-go:mimo-v2-pro", "object": "model", "owned_by": "opencode"},
                {"id": "opencode-go:minimax-m2.5", "object": "model", "owned_by": "opencode"},
                {"id": "opencode-go:minimax-m2.7", "object": "model", "owned_by": "opencode"},
            ])

    # Qwen Code — only if auth is ready
    if proxy.qwen_auth.is_ready:
        data.extend([
            {"id": "qwen:coder-model", "object": "model", "owned_by": "alibaba"},
        ])

    # Fireworks — only if API key is configured
    if proxy.fireworks_api_key:
        data.extend([
            {"id": "fireworks:kimi-k2p5-turbo", "object": "model", "owned_by": "fireworks"},
            {"id": "fireworks:accounts/fireworks/routers/kimi-k2p5-turbo", "object": "model", "owned_by": "fireworks"},
        ])

    # NVIDIA NIM — only if API key is configured
    if proxy.nvidia_api_key:
        for alias, full_path in proxy._NVIDIA_MODEL_ALIASES.items():
            data.append({"id": f"nvidia:{alias}", "object": "model", "owned_by": "nvidia"})
            if alias != full_path:
                data.append({"id": f"nvidia:{full_path}", "object": "model", "owned_by": "nvidia"})

    return {"object": "list", "data": data}


@router.get("/config/vision-models")
async def list_vision_models():
    """Return vision-capable models based on currently available providers.

    Models are filtered by provider availability and returned in a format
    suitable for the dashboard vision test dropdown.
    """
    import proxy
    models = []

    # Claude models (vision capable)
    if proxy.auth.is_ready:
        models.extend([
            {"id": "claude-opus-4-6", "name": "Claude Opus 4.6", "provider": "claude"},
            {"id": "claude-sonnet-4-6", "name": "Claude Sonnet 4.6", "provider": "claude"},
            {"id": "claude-sonnet-4-5-20250929", "name": "Claude Sonnet 4.5", "provider": "claude"},
        ])

    # OpenRouter models (vision capable via API key)
    if proxy.openrouter_api_key:
        models.extend([
            {"id": "openai/gpt-4o", "name": "GPT-4o", "provider": "openrouter"},
            {"id": "openai/gpt-4o-mini", "name": "GPT-4o Mini", "provider": "openrouter"},
            {"id": "anthropic/claude-3-5-sonnet", "name": "Claude 3.5 Sonnet", "provider": "openrouter"},
            {"id": "anthropic/claude-3-7-sonnet", "name": "Claude 3.7 Sonnet", "provider": "openrouter"},
            {"id": "anthropic/claude-3-opus", "name": "Claude 3 Opus", "provider": "openrouter"},
            {"id": "google/gemini-2.5-pro-preview", "name": "Gemini 2.5 Pro", "provider": "openrouter"},
            {"id": "google/gemini-2.5-flash-preview", "name": "Gemini 2.5 Flash", "provider": "openrouter"},
            {"id": "google/gemini-2.0-flash", "name": "Gemini 2.0 Flash", "provider": "openrouter"},
        ])

    # Gemini CLI models (vision capable)
    if proxy.gemini_auth.is_ready:
        models.extend([
            {"id": "gcli-gemini-2.5-pro", "name": "Gemini 2.5 Pro", "provider": "gemini-cli"},
            {"id": "gcli-gemini-2.5-flash", "name": "Gemini 2.5 Flash", "provider": "gemini-cli"},
            {"id": "gcli-gemini-3-pro-preview", "name": "Gemini 3 Pro", "provider": "gemini-cli"},
            {"id": "gcli-gemini-3-flash-preview", "name": "Gemini 3 Flash", "provider": "gemini-cli"},
        ])

    # Antigravity models (vision capable Gemini models)
    if proxy.antigravity_auth.is_ready:
        models.extend([
            {"id": "antigravity-gemini-3.1-pro", "name": "Gemini 3.1 Pro", "provider": "antigravity"},
            {"id": "antigravity-gemini-3-pro", "name": "Gemini 3 Pro", "provider": "antigravity"},
            {"id": "antigravity-gemini-3-flash", "name": "Gemini 3 Flash", "provider": "antigravity"},
            {"id": "antigravity-gemini-2.5-pro", "name": "Gemini 2.5 Pro", "provider": "antigravity"},
            {"id": "antigravity-gemini-2.5-flash", "name": "Gemini 2.5 Flash", "provider": "antigravity"},
        ])

    # Ollama vision models - query CLI for installed models
    ollama_vision_models = []
    try:
        # Check if ollama CLI is available
        ollama_cli = shutil.which("ollama")
        if ollama_cli:
            proc = await asyncio.create_subprocess_exec(
                "ollama", "list",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=5.0
            )
            if proc.returncode == 0:
                output = stdout.decode().strip()
                # Parse table output: NAME ID SIZE MODIFIED
                for line in output.split('\n')[1:]:  # Skip header
                    parts = line.split()
                    if parts:
                        model_id = parts[0].strip()
                        # Include known vision-capable models
                        if model_id and any(x in model_id.lower() for x in ["vision", "llava", "moondream", "cogito", "bakllava", "vl"]):
                            ollama_vision_models.append({
                                "id": f"ollama:{model_id}",
                                "name": model_id,
                                "provider": "ollama"
                            })
    except Exception:
        pass

    # Static fallback list for common vision models
    common_ollama_vision = [
        {"id": "ollama:llava", "name": "llava", "provider": "ollama"},
        {"id": "ollama:llava-phi3", "name": "llava-phi3", "provider": "ollama"},
        {"id": "ollama:llava-llama3", "name": "llava-llama3", "provider": "ollama"},
        {"id": "ollama:llama3.2-vision", "name": "llama3.2-vision", "provider": "ollama"},
        {"id": "ollama:bakllava", "name": "bakllava", "provider": "ollama"},
        {"id": "ollama:moondream", "name": "moondream", "provider": "ollama"},
        {"id": "ollama:granite3.2-vision", "name": "granite3.2-vision", "provider": "ollama"},
    ]
    # Merge CLI-discovered models with static fallback (avoiding duplicates)
    existing_ids = {m["id"] for m in ollama_vision_models}
    for m in common_ollama_vision:
        if m["id"] not in existing_ids:
            ollama_vision_models.append(m)
    models.extend(ollama_vision_models)

    # Fireworks models (vision capable) - fetched dynamically if possible
    fireworks_vision_models = []
    if proxy.fireworks_api_key:
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5, connect=2)) as s:
                headers = {"Authorization": f"Bearer {proxy.fireworks_api_key}"}
                async with s.get("https://api.fireworks.ai/inference/v1/models", headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for m in data.get("data", []):
                            model_id = m.get("id", "")
                            # Vision models typically have 'vision' or 'vl' in the name
                            if model_id and any(x in model_id.lower() for x in ["vision", "-vl-", "qwen2-vl", "llava"]):
                                fireworks_vision_models.append({
                                    "id": f"fireworks:{model_id}",
                                    "name": m.get("name", model_id),
                                    "provider": "fireworks"
                                })
        except Exception:
            pass

        # Static fallback for Fireworks vision models
        # Note: Fireworks uses _resolve_fireworks_model() which strips 'fireworks:' prefix
        # and looks up aliases. Short names like 'kimi-k2p5-turbo' get expanded to full paths.
        if not fireworks_vision_models:
            fireworks_vision_models = [
                # These use the short alias format that gets expanded by _FIREWORKS_MODEL_ALIASES
                {"id": "fireworks:kimi-k2p5-turbo", "name": "Kimi K2.5 Turbo (Vision)", "provider": "fireworks"},
            ]
    models.extend(fireworks_vision_models)

    return {"models": models}
