"""Shared test fixtures for proxy tests.

Strategy: import proxy with shutil.which mocked to None so no CLI auth
capture runs at startup. The FastAPI app object is available on proxy.app.
All module-level globals (auth, codex_auth, etc.) are initialised but inert.
"""
import sys, os
import pytest
from unittest.mock import patch

# Ensure proxy is importable from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


@pytest.fixture(scope="session", autouse=True)
def _import_proxy_once():
    """Import proxy exactly once per test session with CLI tools mocked out."""
    with patch("shutil.which", return_value=None):
        import proxy  # noqa: F401 — side-effect: registers on sys.modules


@pytest.fixture()
def proxy_module():
    """Return the already-imported proxy module."""
    import proxy
    return proxy


@pytest.fixture()
def test_client(proxy_module):
    """FastAPI TestClient that does NOT run the lifespan (no subprocess spawning)."""
    from fastapi.testclient import TestClient
    # lifespan=False skips the startup/shutdown events (prevents CLI auth subprocesses)
    with TestClient(proxy_module.app, raise_server_exceptions=True, lifespan=False) as client:
        yield client
