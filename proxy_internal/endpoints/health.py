"""Health check endpoint."""
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health():
    import proxy  # lazy
    accounts_info = proxy.antigravity_auth.get_all_accounts_info()
    return {
        "status": "healthy" if (
            proxy.auth.is_ready
            or proxy.codex_auth.is_ready
            or proxy.antigravity_auth.is_ready
            or proxy.gemini_auth.is_ready
            or proxy.openrouter_api_key
            or proxy.ollama_api_key
            or proxy.zai_api_key
            or proxy.xiaomi_api_key
            or proxy.opencode_api_key
            or proxy.opencode_go_api_key
            or proxy.qwen_auth.is_ready
            or proxy.fireworks_api_key
            or proxy.nvidia_api_key
        ) else "warming_up",
        "claude": {
            "path": proxy.CLAUDE_PATH,
            "auth_cached": proxy.auth.is_ready,
        },
        "codex": {
            "path": proxy.CODEX_PATH,
            "auth_cached": proxy.codex_auth.is_ready,
            "token_expired": proxy.codex_auth.is_expired() if proxy.codex_auth.is_ready else None,
        },
        "antigravity": {
            "auth_cached": proxy.antigravity_auth.is_ready,
            "email": proxy.antigravity_auth.email,
            "project_id": proxy.antigravity_auth.project_id,
            "token_expired": proxy.antigravity_auth.is_expired() if proxy.antigravity_auth.is_ready else None,
            "accounts": accounts_info,
            "account_count": len(accounts_info),
        },
        "gemini_cli": {
            "auth_cached": proxy.gemini_auth.is_ready,
            "token_expired": proxy.gemini_auth.is_expired() if proxy.gemini_auth.is_ready else None,
            "has_refresh_token": proxy.gemini_auth.refresh_token is not None,
        },
        "openrouter_configured": proxy.openrouter_api_key is not None,
        "ollama_configured": proxy.ollama_api_key is not None,
        "zai_configured": proxy.zai_api_key is not None,
        "xiaomi_configured": proxy.xiaomi_api_key is not None,
        "opencode_zen_configured": proxy.opencode_api_key is not None,
        "opencode_go_configured": proxy.opencode_go_api_key is not None,
        "fireworks_configured": proxy.fireworks_api_key is not None,
        "nvidia_configured": proxy.nvidia_api_key is not None,
        "qwen": {
            "auth_cached": proxy.qwen_auth.is_ready,
            "token_expired": proxy.qwen_auth.is_expired() if proxy.qwen_auth.is_ready else None,
            "has_refresh_token": proxy.qwen_auth.refresh_token is not None,
        },
    }
