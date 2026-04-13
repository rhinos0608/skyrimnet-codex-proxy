"""Tailscale HTTPS reverse-proxy helper (optional)."""

import json as _json
import logging
import shutil
import subprocess
from typing import Iterable, Optional

logger = logging.getLogger("proxy")


def get_tailscale_fqdn() -> Optional[str]:
    """Return this machine's Tailscale FQDN (e.g. ``myhost.tailnet123.ts.net``).

    Returns ``None`` when the CLI is missing, Tailscale is not running, or the
    FQDN cannot be determined.
    """
    if not shutil.which("tailscale"):
        return None
    try:
        result = subprocess.run(
            ["tailscale", "status", "--json"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return (
                _json.loads(result.stdout)
                .get("Self", {})
                .get("DNSName", "")
                .rstrip(".") or None
            )
    except Exception as exc:
        logger.debug(f"tailscale FQDN lookup failed: {exc}")
    return None


def _setup_tailscale_serve(ports: list[int]):
    """Configure Tailscale HTTPS reverse proxy for the given local ports.

    Uses `tailscale serve --bg` to map each port to HTTPS on the machine's
    Tailscale FQDN.  Requires Tailscale to be running and `tailscale serve`
    to be available (MagicDNS + HTTPS enabled on the tailnet).
    """
    fqdn = get_tailscale_fqdn()
    if not fqdn:
        logger.warning("Could not determine Tailscale FQDN — skipping HTTPS setup")
        return

    for port in ports:
        try:
            result = subprocess.run(
                ["tailscale", "serve", "--bg", "--https", str(port),
                 f"http://127.0.0.1:{port}"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                logger.info(f"Tailscale HTTPS: https://{fqdn}:{port} -> http://127.0.0.1:{port}")
            else:
                logger.warning(f"tailscale serve failed for port {port}: {result.stderr.strip()}")
        except Exception as e:
            logger.warning(f"Failed to set up Tailscale serve for port {port}: {e}")

    logger.info(f"Tailscale FQDN: {fqdn}")
