"""Tests for CORS configuration.

The proxy must only allow cross-origin requests from localhost origins,
preventing external websites from accessing config key endpoints.
"""
import pytest


# All allowed origins configured in proxy.py CORS middleware
ALLOWED_ORIGINS = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:8539",
    "http://127.0.0.1:8539",
    "http://localhost:8432",
    "http://127.0.0.1:8432",
]

# Origins that must be rejected
REJECTED_ORIGINS = [
    "https://evil.com",
    "http://evil.com",
    "https://example.com",
    "http://192.168.1.100:8539",
    "http://0.0.0.0:8539",
    "http://somehost:8000",
    "null",
]


class TestCorsAllowedOrigins:
    """Verify that allowed localhost origins receive CORS headers."""

    @pytest.mark.parametrize("origin", ALLOWED_ORIGINS)
    def test_allowed_origin_gets_cors_headers(self, test_client, origin):
        """A preflight OPTIONS request from a localhost origin must receive
        Access-Control-Allow-Origin matching the request origin."""
        resp = test_client.options(
            "/config/openrouter-key",
            headers={"Origin": origin, "Access-Control-Request-Method": "POST"},
        )
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == origin

    @pytest.mark.parametrize("origin", ALLOWED_ORIGINS)
    def test_allowed_origin_simple_request(self, test_client, origin):
        """A simple GET request from a localhost origin must receive
        Access-Control-Allow-Origin header."""
        resp = test_client.get("/health", headers={"Origin": origin})
        assert resp.headers.get("access-control-allow-origin") == origin


class TestCorsRejectedOrigins:
    """Verify that non-localhost origins do NOT receive CORS headers."""

    @pytest.mark.parametrize("origin", REJECTED_ORIGINS)
    def test_rejected_origin_no_preflight_cors(self, test_client, origin):
        """A preflight OPTIONS request from a non-localhost origin must NOT
        receive Access-Control-Allow-Origin."""
        resp = test_client.options(
            "/config/openrouter-key",
            headers={"Origin": origin, "Access-Control-Request-Method": "POST"},
        )
        # Starlette CORSMiddleware returns 200 for unknown origins but strips
        # the allow-origin header.
        assert resp.headers.get("access-control-allow-origin") != origin

    @pytest.mark.parametrize("origin", REJECTED_ORIGINS)
    def test_rejected_origin_no_simple_cors(self, test_client, origin):
        """A simple GET request from a non-localhost origin must NOT
        receive Access-Control-Allow-Origin."""
        resp = test_client.get("/health", headers={"Origin": origin})
        assert resp.headers.get("access-control-allow-origin") != origin


class TestCorsWildcardAbsent:
    """Ensure the wildcard '*' is never returned as an allowed origin."""

    def test_no_wildcard_on_preflight(self, test_client):
        """Preflight to a config endpoint must never return '*'."""
        resp = test_client.options(
            "/config/openrouter-key",
            headers={
                "Origin": "https://evil.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert resp.headers.get("access-control-allow-origin") != "*"

    def test_no_wildcard_on_get(self, test_client):
        """GET to health must never return '*' for any origin."""
        resp = test_client.get("/health", headers={"Origin": "https://evil.com"})
        assert resp.headers.get("access-control-allow-origin") != "*"


class TestCorsGameEndpoint:
    """The /v1/chat/completions endpoint must also respect CORS restrictions."""

    @pytest.mark.parametrize("origin", REJECTED_ORIGINS)
    def test_game_endpoint_rejects_external_origin(self, test_client, origin):
        """External origins must not get CORS headers on the game API."""
        resp = test_client.options(
            "/v1/chat/completions",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
            },
        )
        assert resp.headers.get("access-control-allow-origin") != origin

    def test_game_endpoint_allows_localhost(self, test_client):
        """Localhost origins must get CORS headers on the game API."""
        resp = test_client.options(
            "/v1/chat/completions",
            headers={
                "Origin": "http://localhost:8539",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert resp.headers.get("access-control-allow-origin") == "http://localhost:8539"
