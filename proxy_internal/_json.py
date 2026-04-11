"""Fast JSON serialisation with an orjson-backed fast path."""

import json as _stdlib_json

try:
    import orjson as _json_lib

    def json_dumps(obj: object) -> bytes:
        return _json_lib.dumps(obj)
except ImportError:
    def json_dumps(obj: object) -> bytes:  # type: ignore[misc]
        return _stdlib_json.dumps(obj).encode("utf-8")
