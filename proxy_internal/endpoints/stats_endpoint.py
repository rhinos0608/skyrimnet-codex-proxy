"""Stats and read-only runtime config endpoints."""
from fastapi import APIRouter

router = APIRouter()


@router.get("/stats")
async def get_request_stats():
    """Full request stats: per-model, per-mode (streaming/direct), and global."""
    import proxy
    return proxy.request_stats.get_stats()


@router.get("/config/timeout-routing")
async def get_timeout_routing():
    import proxy
    return {
        "timeout_routing_enabled": proxy.timeout_routing_enabled,
        "timeout_cutoff_seconds": proxy.timeout_cutoff_seconds,
        "timeout_seconds": proxy.timeout_cutoff_seconds,  # legacy alias
        "max_total_seconds": proxy.max_total_seconds,
        "stats": proxy.model_stats.get_stats(),
    }


@router.get("/config/max-retries")
async def get_max_retries():
    """Return the current retry budget applied to OAI-compatible upstreams."""
    import proxy
    return {"max_retries": proxy.max_retries}
