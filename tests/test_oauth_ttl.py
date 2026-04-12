"""Tests for OAuth state TTL/cleanup logic.

Verifies that _cleanup_expired_oauth_states() removes entries older than
10 minutes and preserves recent entries. Also checks that cleanup is called
during the OAuth flow.
"""
import time

import pytest


class TestCleanupExpiredOAuthStates:
    """Direct unit tests for _cleanup_expired_oauth_states()."""

    def test_removes_expired_entries(self, proxy_module):
        """Entries older than 10 minutes (600s) should be removed."""
        now = time.time()
        proxy_module._oauth_states["old-state"] = {
            "verifier": "v1",
            "project_id": "",
            "created_at": now - 601,  # just over 10 minutes
        }
        proxy_module._oauth_states["older-state"] = {
            "verifier": "v2",
            "project_id": "p1",
            "created_at": now - 1200,
        }

        proxy_module._cleanup_expired_oauth_states()

        assert "old-state" not in proxy_module._oauth_states
        assert "older-state" not in proxy_module._oauth_states

    def test_preserves_recent_entries(self, proxy_module):
        """Entries younger than 10 minutes should be preserved."""
        now = time.time()
        proxy_module._oauth_states["new-state"] = {
            "verifier": "v3",
            "project_id": "p2",
            "created_at": now - 300,  # 5 minutes ago
        }
        proxy_module._oauth_states["fresh-state"] = {
            "verifier": "v4",
            "project_id": "",
            "created_at": now,
        }

        proxy_module._cleanup_expired_oauth_states()

        assert "new-state" in proxy_module._oauth_states
        assert "fresh-state" in proxy_module._oauth_states

    def test_handles_empty_dict(self, proxy_module):
        """Cleanup on an empty dict should not raise."""
        proxy_module._oauth_states.clear()
        proxy_module._cleanup_expired_oauth_states()
        assert proxy_module._oauth_states == {}

    def test_handles_missing_created_at(self, proxy_module):
        """Entries without created_at should be treated as very old (epoch 0)."""
        now = time.time()
        proxy_module._oauth_states["no-timestamp"] = {
            "verifier": "v5",
            "project_id": "",
            # no created_at key
        }
        proxy_module._oauth_states["recent"] = {
            "verifier": "v6",
            "project_id": "",
            "created_at": now - 100,
        }

        proxy_module._cleanup_expired_oauth_states()

        # Missing created_at defaults to 0 via .get("created_at", 0), so
        # now - 0 > 600 is always true for any reasonable now value.
        assert "no-timestamp" not in proxy_module._oauth_states
        assert "recent" in proxy_module._oauth_states

    def test_boundary_exactly_600_seconds(self, proxy_module):
        """Entry exactly 600 seconds old should be removed (>= 600)."""
        now = time.time()
        proxy_module._oauth_states["boundary"] = {
            "verifier": "v7",
            "project_id": "",
            "created_at": now - 600,
        }

        proxy_module._cleanup_expired_oauth_states()

        assert "boundary" not in proxy_module._oauth_states


class TestCleanupCalledDuringOAuthFlow:
    """Verify that cleanup is invoked during the OAuth flow."""

    def test_callback_handler_calls_cleanup(self, proxy_module):
        """The callback_handler in start_oauth_callback_server should call cleanup."""
        import inspect
        source = inspect.getsource(proxy_module.start_oauth_callback_server)
        assert "_cleanup_expired_oauth_states()" in source, (
            "callback_handler should call _cleanup_expired_oauth_states()"
        )

    def test_antigravity_login_calls_cleanup(self):
        """The antigravity_login endpoint should call cleanup before adding entries."""
        from proxy_internal.endpoints import antigravity_oauth
        import inspect
        source = inspect.getsource(antigravity_oauth.antigravity_login)
        assert "_cleanup_expired_oauth_states()" in source, (
            "antigravity_login should call _cleanup_expired_oauth_states()"
        )

    def test_antigravity_login_includes_created_at(self):
        """The antigravity_login endpoint should include created_at in state entries."""
        from proxy_internal.endpoints import antigravity_oauth
        import inspect
        source = inspect.getsource(antigravity_oauth.antigravity_login)
        assert '"created_at"' in source or "'created_at'" in source, (
            "antigravity_login should include created_at timestamp in OAuth state entries"
        )

    def test_cleanup_function_exists(self, proxy_module):
        """The cleanup helper function should exist on the proxy module."""
        assert callable(getattr(proxy_module, "_cleanup_expired_oauth_states", None)), (
            "proxy module should expose _cleanup_expired_oauth_states()"
        )
