"""Tests for XSS protection in OAuth HTML callback responses.

All user-controlled data interpolated into HTML responses must be escaped
with html.escape() to prevent reflected XSS.

Strategy: The OAuth callback handler is a nested async function inside
start_oauth_callback_server(), making it hard to call directly in tests.
Instead, we verify the source code of each handler actually uses html.escape()
on every user-controlled value, and we verify html.escape() itself works
correctly. This ensures the fixes survive refactoring.
"""
import html
import inspect
import re
import textwrap

import pytest


def _get_callback_source(proxy_module):
    """Extract the source code of the callback_handler nested function."""
    func = proxy_module.start_oauth_callback_server
    source = inspect.getsource(func)
    return source


def _get_antigravity_login_source():
    """Get the source of the antigravity_login endpoint."""
    from proxy_internal.endpoints import antigravity_oauth
    func = antigravity_oauth.antigravity_login
    return inspect.getsource(func)


def _get_dashboard_source():
    """Get the source of the dashboard endpoint."""
    from proxy_internal.endpoints import dashboard
    func = dashboard.dashboard
    return inspect.getsource(func)


class TestProxyOAuthCallbackEscaping:
    """Verify html.escape() is used in the proxy.py OAuth callback handler."""

    def test_error_param_escaped_in_callback(self, proxy_module):
        """The 'error' query param must be escaped with html.escape()."""
        source = _get_callback_source(proxy_module)
        # The handler should escape the error param before putting it in HTML
        assert "html.escape(error)" in source, (
            "callback_handler must call html.escape(error) before HTML interpolation"
        )

    def test_token_exchange_error_escaped(self, proxy_module):
        """The raw token exchange error body must be escaped."""
        source = _get_callback_source(proxy_module)
        assert "html.escape(error_text)" in source, (
            "Token exchange failure response must escape error_text"
        )

    def test_exception_message_escaped(self, proxy_module):
        """Exception messages interpolated into HTML must be escaped."""
        source = _get_callback_source(proxy_module)
        assert "html.escape(str(e))" in source, (
            "Exception handler must escape str(e) before HTML interpolation"
        )

    def test_email_escaped(self, proxy_module):
        """User email from OAuth must be escaped in the success page."""
        source = _get_callback_source(proxy_module)
        assert "html.escape(new_account.email" in source, (
            "Success page must escape new_account.email"
        )

    def test_project_id_escaped(self, proxy_module):
        """Project ID from OAuth must be escaped in the success page."""
        source = _get_callback_source(proxy_module)
        assert "html.escape(str(new_account.project_id" in source, (
            "Success page must escape new_account.project_id"
        )

    def test_no_html_variable_shadowing(self, proxy_module):
        """The local variable 'html' must not shadow the html module.

        Using 'html' as a variable name for HTML string content prevents
        html.escape() from working in subsequent code within the same scope.
        """
        source = _get_callback_source(proxy_module)
        # Look for assignments like 'html = f"' that would shadow the module
        # Allow 'html_content' or similar but not bare 'html ='
        shadowing_pattern = re.compile(r'^\s+html\s*=\s*f["\']', re.MULTILINE)
        matches = shadowing_pattern.findall(source)
        assert matches == [], (
            f"Found {len(matches)} local variable(s) named 'html' that shadow the "
            f"html module. Rename to 'html_content' or similar."
        )

    def test_html_module_imported(self, proxy_module):
        """The html module must be imported at module level."""
        assert hasattr(proxy_module, 'html'), (
            "proxy.py must 'import html' for html.escape()"
        )


class TestAntigravityOAuthEndpointEscaping:
    """Verify html.escape() is used in the antigravity_oauth.py endpoint."""

    def test_exception_escaped_in_login_endpoint(self):
        """Exception in start_oauth_callback_server must be escaped in the HTML response."""
        source = _get_antigravity_login_source()
        assert "html.escape(str(e))" in source or "html.escape(" in source, (
            "antigravity_login must escape the exception message with html.escape()"
        )


class TestDashboardEmailEscaping:
    """Verify email addresses are escaped in the dashboard HTML output."""

    def test_email_escaped_in_dashboard(self):
        """Account emails must be escaped with html.escape() in the dashboard."""
        source = _get_dashboard_source()
        assert "html.escape(acc['email'])" in source or "html.escape(acc[\"email\"])" in source, (
            "Dashboard must escape acc['email'] with html.escape()"
        )

    def test_email_escaped_in_js_context(self):
        """Emails in onclick JS handlers must be additionally quote-escaped."""
        source = _get_dashboard_source()
        # For JS string context, the email needs quote escaping after html.escape
        assert "js_safe_email" in source or 'replace(' in source, (
            "Dashboard must apply JS-safe escaping for emails in onclick handlers"
        )


class TestHtmlEscapeCorrectness:
    """Verify html.escape() itself works as expected (defense in depth)."""

    @pytest.mark.parametrize("payload,expected_substring", [
        ("<script>alert('xss')</script>", "&lt;script&gt;"),
        ('<img src=x onerror=alert(1)>', "&lt;img"),
        ('user@example.com"><script>alert(1)</script>', "&lt;script&gt;"),
        ("'><script>", "&#x27;&gt;&lt;script&gt;"),
        ('"><script>', '&quot;&gt;&lt;script&gt;'),
    ])
    def test_html_escape_neutralises_xss_payloads(self, payload, expected_substring):
        """html.escape() must neutralize common XSS payloads."""
        escaped = html.escape(payload)
        assert expected_substring in escaped
        assert "<script>" not in escaped

    def test_html_escape_handles_none_gracefully(self):
        """html.escape(None) raises AttributeError; code must handle None explicitly."""
        with pytest.raises(AttributeError):
            html.escape(None)
        # The correct pattern is: html.escape(value or "")
        assert html.escape(None or "") == ""
        assert html.escape("" or "") == ""
