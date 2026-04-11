"""Fernet-based at-rest encryption for config secrets."""

import logging
import os

from cryptography.fernet import Fernet

logger = logging.getLogger("proxy")

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
KEY_FILE = os.path.join(_REPO_ROOT, ".proxy.key")

_ENC_PREFIX = "enc:"


def _get_fernet() -> Fernet:
    """Return a Fernet instance, generating a key file on first use."""
    if not os.path.exists(KEY_FILE):
        key = Fernet.generate_key()
        with open(KEY_FILE, "wb") as f:
            f.write(key)
        # Restrict permissions (best-effort on Windows)
        try:
            os.chmod(KEY_FILE, 0o600)
        except OSError:
            pass
        logger.info("Generated new encryption key (.proxy.key)")
    with open(KEY_FILE, "rb") as f:
        return Fernet(f.read().strip())


def _encrypt_value(fernet: Fernet, value: str) -> str:
    """Encrypt a plaintext string, returning an 'enc:...' token."""
    return _ENC_PREFIX + fernet.encrypt(value.encode("utf-8")).decode("ascii")


def _decrypt_value(fernet: Fernet, value: str) -> str:
    """Decrypt an 'enc:...' token back to plaintext. Returns as-is if not encrypted."""
    if not isinstance(value, str) or not value.startswith(_ENC_PREFIX):
        return value
    return fernet.decrypt(value[len(_ENC_PREFIX):].encode("ascii")).decode("utf-8")
