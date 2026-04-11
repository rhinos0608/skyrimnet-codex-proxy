"""Tailscale HTTPS reverse-proxy helper (optional)."""

import logging
import shutil
import subprocess
from typing import Iterable

logger = logging.getLogger("proxy")


def _setup_tailscale_serve(ports: list[int]):
    """Configure Tailscale HTTPS reverse proxy for the given local ports.

    Uses `tailscale serve --bg` to map each port to HTTPS on the machine's
    Tailscale FQDN.  Requires Tailscale to be running and `tailscale serve`
    to be available (MagicDNS + HTTPS enabled on the tailnet).
    """
    import shutil
    import subprocess

    if not shutil.which("tailscale"):
        logger.warning("tailscale CLI not found — skipping HTTPS setup")
        return

    # Get this machine's Tailscale FQDN
    try:
        result = subprocess.run(
            ["tailscale", "status", "--json"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            logger.warning("tailscale status failed — is Tailscale running?")
            return
        import json as _json
        ts_status = _json.loads(result.stdout)
        fqdn = ts_status.get("Self", {}).get("DNSName", "").rstrip(".")
        if not fqdn:
            logger.warning("Could not determine Tailscale FQDN")
            return
    except Exception as e:
        logger.warning(f"Failed to query Tailscale status: {e}")
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
