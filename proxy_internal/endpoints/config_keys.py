"""Config endpoints for setting provider API keys."""
import logging

from fastapi import APIRouter, Request

router = APIRouter()
logger = logging.getLogger("proxy")


@router.post("/config/openrouter-key")
async def set_openrouter_key(request: Request):
    import proxy
    data = await request.json()
    key = data.get("key", "").strip()
    cfg = proxy._load_config()
    if not key:
        proxy.openrouter_api_key = None
        cfg.pop("openrouter_api_key", None)
        proxy._save_config(cfg)
        logger.info("OpenRouter API key cleared")
        return {"status": "cleared"}
    proxy.openrouter_api_key = key
    cfg["openrouter_api_key"] = key
    proxy._save_config(cfg)
    logger.info("OpenRouter API key configured and saved to config.json")
    return {"status": "saved"}


@router.post("/config/ollama-key")
async def set_ollama_key(request: Request):
    import proxy
    data = await request.json()
    key = data.get("key", "").strip()
    cfg = proxy._load_config()
    if not key:
        proxy.ollama_api_key = None
        cfg.pop("ollama_api_key", None)
        proxy._save_config(cfg)
        logger.info("Ollama API key cleared — using local endpoint (localhost:11434)")
        return {"status": "cleared"}
    proxy.ollama_api_key = key
    cfg["ollama_api_key"] = key
    proxy._save_config(cfg)
    logger.info("Ollama Cloud API key configured and saved to config.json")
    return {"status": "saved"}


@router.post("/config/fireworks-key")
async def set_fireworks_key(request: Request):
    import proxy
    data = await request.json()
    key = data.get("key", "").strip()
    cfg = proxy._load_config()
    if not key:
        proxy.fireworks_api_key = None
        cfg.pop("fireworks_api_key", None)
        proxy._save_config(cfg)
        logger.info("Fireworks API key cleared")
        return {"status": "cleared"}
    proxy.fireworks_api_key = key
    cfg["fireworks_api_key"] = key
    proxy._save_config(cfg)
    logger.info("Fireworks API key configured and saved to config.json")
    return {"status": "saved"}


@router.post("/config/nvidia-key")
async def set_nvidia_key(request: Request):
    import proxy
    data = await request.json()
    key = data.get("key", "").strip()
    cfg = proxy._load_config()
    if not key:
        proxy.nvidia_api_key = None
        cfg.pop("nvidia_api_key", None)
        proxy._save_config(cfg)
        logger.info("NVIDIA NIM API key cleared")
        return {"status": "cleared"}
    proxy.nvidia_api_key = key
    cfg["nvidia_api_key"] = key
    proxy._save_config(cfg)
    logger.info("NVIDIA NIM API key configured and saved to config.json")
    return {"status": "saved"}


@router.post("/config/zai-key")
async def set_zai_key(request: Request):
    import proxy
    data = await request.json()
    key = data.get("key", "").strip()
    cfg = proxy._load_config()
    if not key:
        proxy.zai_api_key = None
        cfg.pop("zai_api_key", None)
        proxy._save_config(cfg)
        logger.info("Z.AI API key cleared")
        return {"status": "cleared"}
    proxy.zai_api_key = key
    cfg["zai_api_key"] = key
    proxy._save_config(cfg)
    logger.info("Z.AI API key configured and saved to config.json")
    return {"status": "saved"}


@router.post("/config/xiaomi-key")
async def set_xiaomi_key(request: Request):
    import proxy
    data = await request.json()
    key = data.get("key", "").strip()
    cfg = proxy._load_config()
    if not key:
        proxy.xiaomi_api_key = None
        cfg.pop("xiaomi_api_key", None)
        proxy._save_config(cfg)
        logger.info("Xiaomi API key cleared")
        return {"status": "cleared"}
    proxy.xiaomi_api_key = key
    cfg["xiaomi_api_key"] = key
    proxy._save_config(cfg)
    logger.info("Xiaomi API key configured and saved to config.json")
    return {"status": "saved"}


@router.post("/config/opencode-key")
async def set_opencode_key(request: Request):
    import proxy
    data = await request.json()
    key = data.get("key", "").strip()
    cfg = proxy._load_config()
    if not key:
        proxy.opencode_api_key = None
        proxy.opencode_go_api_key = None
        cfg.pop("opencode_api_key", None)
        cfg.pop("opencode_go_api_key", None)
        proxy._save_config(cfg)
        logger.info("OpenCode API key cleared (both plans)")
        return {"status": "cleared"}
    proxy.opencode_api_key = key
    proxy.opencode_go_api_key = key
    cfg["opencode_api_key"] = key
    proxy._save_config(cfg)
    logger.info("OpenCode API key configured (shared across Zen + Go)")
    return {"status": "saved"}


@router.post("/config/opencode-go-key")
async def set_opencode_go_key(request: Request):
    import proxy
    data = await request.json()
    key = data.get("key", "").strip()
    cfg = proxy._load_config()
    if not key:
        proxy.opencode_api_key = None
        proxy.opencode_go_api_key = None
        cfg.pop("opencode_api_key", None)
        cfg.pop("opencode_go_api_key", None)
        proxy._save_config(cfg)
        logger.info("OpenCode API key cleared (both plans)")
        return {"status": "cleared"}
    proxy.opencode_api_key = key
    proxy.opencode_go_api_key = key
    cfg["opencode_api_key"] = key
    proxy._save_config(cfg)
    logger.info("OpenCode API key configured (shared across Zen + Go)")
    return {"status": "saved"}
