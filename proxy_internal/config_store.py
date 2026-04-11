"""Load/save the on-disk proxy config, with encrypted secret fields."""

import json
import logging
import os
from typing import Optional

from proxy_internal.encryption import _get_fernet, _encrypt_value, _decrypt_value, _ENC_PREFIX

logger = logging.getLogger("proxy")

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CONFIG_FILE = os.path.join(_REPO_ROOT, "config.json")

# Fields in config.json that contain secrets and should be encrypted at rest.
_ENCRYPTED_CONFIG_FIELDS = {"openrouter_api_key", "ollama_api_key", "zai_api_key", "xiaomi_api_key", "opencode_api_key", "opencode_go_api_key", "fireworks_api_key", "nvidia_api_key"}


def _load_config() -> dict:
    """Load persisted config from disk, decrypting secret fields."""
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    fernet = _get_fernet()
    migrated = False
    for field in _ENCRYPTED_CONFIG_FIELDS:
        val = data.get(field)
        if isinstance(val, str) and val:
            if val.startswith(_ENC_PREFIX):
                data[field] = _decrypt_value(fernet, val)
            else:
                # Plaintext on disk — encrypt it in place for next load.
                migrated = True
    if migrated:
        _save_config(data)
    return data


def _save_config(data: dict) -> None:
    """Persist config dict to disk, encrypting secret fields.

    Writes atomically via write-then-rename so a crash or a concurrent save
    cannot leave ``config.json`` half-written — a torn write would corrupt
    the Fernet ciphertexts and lock the user out of every stored API key.
    """
    fernet = _get_fernet()
    out = dict(data)
    for field in _ENCRYPTED_CONFIG_FIELDS:
        val = out.get(field)
        if isinstance(val, str) and val and not val.startswith(_ENC_PREFIX):
            out[field] = _encrypt_value(fernet, val)
    tmp_path = f"{CONFIG_FILE}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass  # fsync unsupported on some filesystems — best effort
    os.replace(tmp_path, CONFIG_FILE)


def _load_max_retries(cfg: Optional[dict] = None) -> int:
    """Clamp max_retries from config into the legal 0..10 range.

    Accepts either an already-loaded config dict or ``None`` to re-read from
    disk.  Default is 1 (one try, one retry).
    """
    if cfg is None:
        cfg = _load_config()
    raw = cfg.get("max_retries", 1)
    try:
        n = int(raw)
    except (TypeError, ValueError):
        n = 1
    return max(0, min(10, n))
