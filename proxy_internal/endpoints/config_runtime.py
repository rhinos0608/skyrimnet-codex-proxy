"""Runtime config POST endpoints (timeouts + retries)."""
import logging

from fastapi import APIRouter, HTTPException, Request

router = APIRouter()
logger = logging.getLogger("proxy")


@router.post("/config/timeout-routing")
async def set_timeout_routing(request: Request):
    import proxy
    data = await request.json()
    enabled = bool(data.get("enabled"))
    proxy.timeout_routing_enabled = enabled
    cfg = proxy._load_config()
    cfg["timeout_routing_enabled"] = enabled
    proxy._save_config(cfg)
    logger.info(f"Timeout routing {'enabled' if enabled else 'disabled'}")
    return {"status": "ok", "timeout_routing_enabled": enabled}


@router.post("/config/timeout-cutoff")
async def set_timeout_cutoff(request: Request):
    import proxy
    data = await request.json()
    seconds = float(data.get("seconds", 6.0))
    if seconds < 1.0 or seconds > 30.0:
        raise HTTPException(status_code=400, detail="Timeout must be between 1 and 30 seconds")
    proxy.timeout_cutoff_seconds = seconds
    cfg = proxy._load_config()
    cfg["timeout_cutoff_seconds"] = seconds
    proxy._save_config(cfg)
    logger.info(f"TTFT cutoff set to {seconds}s")
    return {"status": "ok", "timeout_cutoff_seconds": seconds}


@router.post("/config/max-total")
async def set_max_total(request: Request):
    import proxy
    data = await request.json()
    seconds = float(data.get("seconds", 9.0))
    if seconds < 1.0 or seconds > 60.0:
        raise HTTPException(status_code=400, detail="Max total must be between 1 and 60 seconds")
    proxy.max_total_seconds = seconds
    cfg = proxy._load_config()
    cfg["max_total_seconds"] = seconds
    proxy._save_config(cfg)
    logger.info(f"Max total cutoff set to {seconds}s")
    return {"status": "ok", "max_total_seconds": seconds}


_VALID_REASONING_LEVELS = ("low", "medium", "high")


@router.post("/config/reasoning-override")
async def set_reasoning_override(request: Request):
    """Toggle the dashboard reasoning override.

    When enabled, every request's ``thinking``/``reasoning_effort`` is forced
    to the configured level regardless of what the caller supplied. Useful for
    CCS-routed Claude Code sessions that otherwise arrive with reasoning off.
    """
    import proxy
    data = await request.json()
    if "enabled" not in data:
        raise HTTPException(status_code=400, detail="enabled is required")
    enabled = bool(data.get("enabled"))
    proxy.reasoning_override_enabled = enabled
    cfg = proxy._load_config()
    cfg["reasoning_override_enabled"] = enabled
    proxy._save_config(cfg)
    logger.info(
        f"Reasoning override {'enabled' if enabled else 'disabled'} "
        f"(level={proxy.reasoning_override_level})"
    )
    return {
        "status": "ok",
        "reasoning_override_enabled": enabled,
        "reasoning_override_level": proxy.reasoning_override_level,
    }


@router.post("/config/reasoning-level")
async def set_reasoning_level(request: Request):
    """Set the reasoning effort level used when the override is active.

    Level maps to (a) Anthropic ``thinking.budget_tokens`` and (b) OpenAI-style
    ``reasoning_effort`` — both are injected into ``extra_params`` so whichever
    field the downstream provider honours is set.
    """
    import proxy
    data = await request.json()
    level = data.get("level")
    if not isinstance(level, str) or level not in _VALID_REASONING_LEVELS:
        raise HTTPException(
            status_code=400,
            detail=f"level must be one of {_VALID_REASONING_LEVELS}",
        )
    proxy.reasoning_override_level = level
    cfg = proxy._load_config()
    cfg["reasoning_override_level"] = level
    proxy._save_config(cfg)
    logger.info(f"Reasoning override level set to {level}")
    return {
        "status": "ok",
        "reasoning_override_enabled": proxy.reasoning_override_enabled,
        "reasoning_override_level": level,
    }


@router.post("/config/reasoning-rewrite")
async def set_reasoning_rewrite(request: Request):
    """Toggle the reasoning rewrite feature.

    When enabled, if an upstream returns only reasoning tokens with no visible
    content, the proxy makes a secondary call to rewrite the reasoning into
    NPC dialogue. Uses Fireworks API credits.
    """
    import proxy
    data = await request.json()
    if "enabled" not in data:
        raise HTTPException(status_code=400, detail="enabled is required")
    enabled = bool(data.get("enabled"))
    proxy.reasoning_rewrite_enabled = enabled
    cfg = proxy._load_config()
    cfg["reasoning_rewrite_enabled"] = enabled
    proxy._save_config(cfg)
    logger.info(f"Reasoning rewrite {'enabled' if enabled else 'disabled'}")
    return {"status": "ok", "reasoning_rewrite_enabled": enabled}


@router.post("/config/max-retries")
async def set_max_retries(request: Request):
    """Update the retry budget. Clamps to 0..10 and persists to config.json."""
    import proxy
    data = await request.json()
    raw = data.get("max_retries")
    if raw is None:
        raise HTTPException(status_code=400, detail="max_retries is required")
    try:
        n = int(raw)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="max_retries must be an integer")
    if n < 0 or n > 10:
        raise HTTPException(status_code=400, detail="max_retries must be between 0 and 10")
    proxy.max_retries = n
    cfg = proxy._load_config()
    cfg["max_retries"] = n
    proxy._save_config(cfg)
    logger.info(f"Retry policy updated: max_retries={n}")
    return {"status": "saved", "max_retries": n}
