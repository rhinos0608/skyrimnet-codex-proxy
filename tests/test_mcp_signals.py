"""Tests for MCP server signal handling and graceful shutdown.

These tests are written TDD-style: they describe the intended behavior of
``install_signal_handlers()`` in ``mcp_server``, which does not yet exist.
All four tests are expected to FAIL until the implementation is added.

Covered behaviors:
  - SIGPIPE is set to SIG_IGN after install_signal_handlers()
  - Sending SIGTERM to self sets the supplied shutdown_event
  - The SIGTERM handler logs an INFO message about graceful shutdown
  - Writing to a broken pipe does not raise BrokenPipeError when SIGPIPE is SIG_IGN
"""

import asyncio
import logging
import os
import signal
import sys
import threading

import pytest


# ---------------------------------------------------------------------------
# Module import helper
# ---------------------------------------------------------------------------

def _get_mcp_server():
    """Import mcp_server, skipping it on platforms that don't support SIGPIPE."""
    import importlib
    return importlib.import_module("mcp_server")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def shutdown_event():
    """A fresh threading.Event for each test."""
    return threading.Event()


@pytest.fixture(autouse=True)
def _restore_signal_handlers():
    """Restore SIGTERM and SIGPIPE to their original handlers after each test
    so that one test's install_signal_handlers() call doesn't bleed into the next."""
    original_sigterm = signal.getsignal(signal.SIGTERM)
    original_sigpipe = signal.getsignal(signal.SIGPIPE) if hasattr(signal, "SIGPIPE") else None
    yield
    signal.signal(signal.SIGTERM, original_sigterm)
    if hasattr(signal, "SIGPIPE") and original_sigpipe is not None:
        signal.signal(signal.SIGPIPE, original_sigpipe)


# ---------------------------------------------------------------------------
# Test 1: SIGPIPE is ignored after install_signal_handlers()
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not hasattr(signal, "SIGPIPE"), reason="SIGPIPE not available on this platform")
def test_sigpipe_ignored():
    """After install_signal_handlers(), SIGPIPE must be set to SIG_IGN.

    Rationale: Claude Desktop may close the stdio pipe at any time. Without
    SIG_IGN the default action (process termination) fires silently.
    """
    mcp_server = _get_mcp_server()

    mcp_server.install_signal_handlers()

    assert signal.getsignal(signal.SIGPIPE) is signal.SIG_IGN, (
        "Expected SIGPIPE to be SIG_IGN after install_signal_handlers(), "
        f"got {signal.getsignal(signal.SIGPIPE)!r}"
    )


# ---------------------------------------------------------------------------
# Test 2: SIGTERM sets the shutdown event
# ---------------------------------------------------------------------------

def test_sigterm_sets_shutdown_event(shutdown_event):
    """Delivering SIGTERM to the process must set the supplied shutdown_event.

    install_signal_handlers() should accept a shutdown_event and wire the
    SIGTERM handler to call shutdown_event.set().
    """
    mcp_server = _get_mcp_server()

    mcp_server.install_signal_handlers(shutdown_event=shutdown_event)

    assert not shutdown_event.is_set(), "Precondition: event must not be set before SIGTERM"

    with pytest.raises(SystemExit):
        os.kill(os.getpid(), signal.SIGTERM)

    assert shutdown_event.is_set(), (
        "shutdown_event should be set after SIGTERM is delivered, "
        "but it is still clear"
    )


# ---------------------------------------------------------------------------
# Test 3: SIGTERM handler emits an INFO log message
# ---------------------------------------------------------------------------

def test_sigterm_handler_logs(shutdown_event, caplog):
    """The SIGTERM handler must log an INFO-level message about graceful shutdown.

    This ensures operators can see why the process stopped in logs rather
    than having it disappear silently.
    """
    mcp_server = _get_mcp_server()

    with caplog.at_level(logging.INFO):
        mcp_server.install_signal_handlers(shutdown_event=shutdown_event)
        with pytest.raises(SystemExit):
            os.kill(os.getpid(), signal.SIGTERM)

    log_text = caplog.text.lower()
    assert any(
        keyword in log_text
        for keyword in ("sigterm", "shutdown", "graceful", "termination", "stopping")
    ), (
        "Expected an INFO log mentioning SIGTERM or graceful shutdown, "
        f"but caplog contained: {caplog.text!r}"
    )


# ---------------------------------------------------------------------------
# Test 4: Broken-pipe write does not crash when SIGPIPE is SIG_IGN
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not hasattr(signal, "SIGPIPE"), reason="SIGPIPE not available on this platform")
def test_broken_pipe_does_not_crash():
    """A write to a closed pipe must raise OSError/BrokenPipeError and NOT
    kill the process when SIGPIPE is SIG_IGN.

    Before the fix the default SIGPIPE disposition terminates the process.
    With SIG_IGN the kernel converts it to a normal EPIPE errno, so Python
    raises BrokenPipeError (a subclass of OSError). The MCP server must be
    able to catch that and carry on — this test verifies the signal disposition
    allows that recovery path instead of silent process death.
    """
    mcp_server = _get_mcp_server()

    mcp_server.install_signal_handlers()

    # Create a pipe, close the read end, then write to the write end.
    # With SIG_IGN this must raise BrokenPipeError rather than terminate.
    r_fd, w_fd = os.pipe()
    os.close(r_fd)
    try:
        with pytest.raises((BrokenPipeError, OSError)):
            os.write(w_fd, b"ping")
    finally:
        try:
            os.close(w_fd)
        except OSError:
            pass  # already closed by the broken-pipe write
