"""Tests for thread safety of stats persistence in proxy.py.

Verifies that the _stats_lock threading.Lock exists and is used inside
_save_stats_to_disk() to protect to_dict() calls from concurrent record()
mutations.
"""
import threading
import time

import pytest


class TestStatsLockExists:
    def test_stats_lock_is_threading_lock(self, proxy_module):
        """_stats_lock must be a threading.Lock instance at module level."""
        lock = proxy_module._stats_lock
        assert isinstance(lock, type(threading.Lock())), (
            f"_stats_lock is {type(lock)}, expected threading.Lock"
        )


class TestSaveStatsUsesLock:
    def test_lock_held_during_save(self, proxy_module):
        """_save_stats_to_disk must acquire _stats_lock before calling to_dict().

        We verify this by recording the lock state inside to_dict() via a
        monkey-patched wrapper that asserts the lock is held.
        """
        import json
        import os
        import tempfile

        lock = proxy_module._stats_lock
        original_model_to_dict = proxy_module.model_stats.to_dict
        original_request_to_dict = proxy_module.request_stats.to_dict

        lock_was_held = {"model": False, "request": False}

        def checked_model_to_dict():
            lock_was_held["model"] = lock.locked()
            return original_model_to_dict()

        def checked_request_to_dict():
            lock_was_held["request"] = lock.locked()
            return original_request_to_dict()

        # Patch to_dict methods on the live instances
        proxy_module.model_stats.to_dict = checked_model_to_dict
        proxy_module.request_stats.to_dict = checked_request_to_dict

        try:
            # Point STATS_FILE at a temp file so we don't clobber real data
            original_stats_file = proxy_module.STATS_FILE
            fd, tmp = tempfile.mkstemp(suffix=".stats")
            os.close(fd)
            proxy_module.STATS_FILE = tmp

            try:
                proxy_module._save_stats_to_disk()
            finally:
                proxy_module.STATS_FILE = original_stats_file
                if os.path.exists(tmp):
                    os.remove(tmp)
        finally:
            proxy_module.model_stats.to_dict = original_model_to_dict
            proxy_module.request_stats.to_dict = original_request_to_dict

        assert lock_was_held["model"], "model_stats.to_dict() was called without _stats_lock held"
        assert lock_was_held["request"], "request_stats.to_dict() was called without _stats_lock held"


class TestConcurrentRecordAndSnapshot:
    def test_to_dict_while_recording(self, proxy_module):
        """Basic concurrency test: record() from one thread while another
        calls to_dict() under the lock.  No exceptions or crashes expected."""
        tracker = proxy_module.ModelStatsTracker(window=200)
        errors = []

        def recorder():
            for i in range(500):
                try:
                    tracker.record("concurrent-model", float(i) * 0.01, i % 7 != 0)
                except Exception as e:
                    errors.append(e)

        def snapshotter():
            lock = proxy_module._stats_lock
            for _ in range(200):
                with lock:
                    try:
                        d = tracker.to_dict()
                        # Basic sanity: result should be a dict with expected keys
                        assert "window" in d
                        assert "records" in d
                    except Exception as e:
                        errors.append(e)

        t1 = threading.Thread(target=recorder)
        t2 = threading.Thread(target=snapshotter)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert not errors, f"Errors during concurrent access: {errors}"
