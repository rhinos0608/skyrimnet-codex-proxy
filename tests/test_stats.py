"""Tests for ModelStatsTracker and RequestStatsTracker classes in proxy.py."""
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# ModelStatsTracker
# ---------------------------------------------------------------------------


class TestModelStatsTrackerEmpty:
    def test_empty_tracker_returns_empty_stats(self, proxy_module):
        tracker = proxy_module.ModelStatsTracker()
        assert tracker.get_stats() == {}


class TestModelStatsTrackerRecord:
    def test_record_single_model(self, proxy_module):
        tracker = proxy_module.ModelStatsTracker()
        tracker.record("model-a", 1.5, True)
        tracker.record("model-a", 2.0, True)

        stats = tracker.get_stats()
        assert "model-a" in stats
        assert stats["model-a"]["samples"] == 2
        assert stats["model-a"]["success_rate"] == 1.0
        assert stats["model-a"]["median_ttft_s"] == 1.75

    def test_multiple_models_tracked_independently(self, proxy_module):
        tracker = proxy_module.ModelStatsTracker()
        tracker.record("model-a", 1.0, True)
        tracker.record("model-b", 5.0, True)

        stats = tracker.get_stats()
        assert "model-a" in stats
        assert "model-b" in stats
        assert stats["model-a"]["median_ttft_s"] == 1.0
        assert stats["model-b"]["median_ttft_s"] == 5.0

    def test_success_rate_with_mixed_results(self, proxy_module):
        tracker = proxy_module.ModelStatsTracker()
        # 3 successes, 2 failures -> 0.6 success rate
        tracker.record("model-a", 1.0, True)
        tracker.record("model-a", 2.0, True)
        tracker.record("model-a", 3.0, True)
        tracker.record("model-a", 4.0, False)
        tracker.record("model-a", 5.0, False)

        stats = tracker.get_stats()
        assert stats["model-a"]["success_rate"] == 0.6
        # median_ttft only uses successful TTFTs: [1.0, 2.0, 3.0] -> median 2.0
        assert stats["model-a"]["median_ttft_s"] == 2.0

    def test_all_failures_gives_none_median(self, proxy_module):
        tracker = proxy_module.ModelStatsTracker()
        tracker.record("model-a", 1.0, False)
        tracker.record("model-a", 2.0, False)

        stats = tracker.get_stats()
        assert stats["model-a"]["success_rate"] == 0.0
        assert stats["model-a"]["median_ttft_s"] is None

    def test_p90_requires_at_least_10_samples(self, proxy_module):
        tracker = proxy_module.ModelStatsTracker()
        for i in range(9):
            tracker.record("model-a", float(i), True)
        stats = tracker.get_stats()
        assert stats["model-a"]["p90_ttft_s"] is None

        tracker.record("model-a", 9.0, True)
        stats = tracker.get_stats()
        assert stats["model-a"]["p90_ttft_s"] is not None

    def test_rolling_window_respects_maxlen(self, proxy_module):
        tracker = proxy_module.ModelStatsTracker(window=5)
        for i in range(10):
            tracker.record("model-a", float(i), True)

        stats = tracker.get_stats()
        # Only the last 5 records should be kept
        assert stats["model-a"]["samples"] == 5
        # TTFTs should be [5.0, 6.0, 7.0, 8.0, 9.0], median = 7.0
        assert stats["model-a"]["median_ttft_s"] == 7.0


class TestModelStatsGetReliableModel:
    def _make_mock_request_stats(self, proxy_module, success_rate=1.0,
                                  median_latency=1.0, sample_count=0):
        """Create a mock RequestStatsTracker with configurable return values."""
        mock = MagicMock(spec=proxy_module.RequestStatsTracker)
        mock.get_model_reliability.return_value = (success_rate, median_latency)
        mock.get_model_sample_count.return_value = sample_count
        return mock

    def test_excludes_specified_model(self, proxy_module):
        tracker = proxy_module.ModelStatsTracker()
        # Only one model with enough records — but it's the excluded one
        for _ in range(10):
            tracker.record("model-a", 1.0, True)
        mock_rs = self._make_mock_request_stats(proxy_module)

        with patch.object(proxy_module, "request_stats", mock_rs):
            result = tracker.get_reliable_model("model-a")
        assert result is None

    def test_requires_minimum_5_records(self, proxy_module):
        tracker = proxy_module.ModelStatsTracker()
        # model-a: only 4 records (below threshold)
        for _ in range(4):
            tracker.record("model-a", 1.0, True)
        mock_rs = self._make_mock_request_stats(proxy_module)

        with patch.object(proxy_module, "request_stats", mock_rs):
            result = tracker.get_reliable_model("other-model")
        assert result is None

    def test_requires_success_rate_at_least_0_7(self, proxy_module):
        tracker = proxy_module.ModelStatsTracker()
        # 3 successes + 7 failures -> 30% success rate (below 0.7)
        for _ in range(3):
            tracker.record("model-a", 1.0, True)
        for _ in range(7):
            tracker.record("model-a", 1.0, False)
        mock_rs = self._make_mock_request_stats(proxy_module, success_rate=0.3,
                                                 sample_count=0)

        with patch.object(proxy_module, "request_stats", mock_rs):
            result = tracker.get_reliable_model("other-model")
        assert result is None

    def test_returns_none_when_no_candidates(self, proxy_module):
        tracker = proxy_module.ModelStatsTracker()
        mock_rs = self._make_mock_request_stats(proxy_module)

        with patch.object(proxy_module, "request_stats", mock_rs):
            result = tracker.get_reliable_model("any")
        assert result is None

    def test_picks_best_model(self, proxy_module):
        tracker = proxy_module.ModelStatsTracker()

        # model-fast: fast, all successes
        for _ in range(10):
            tracker.record("model-fast", 0.5, True)

        # model-slow: slow, all successes
        for _ in range(10):
            tracker.record("model-slow", 8.0, True)

        mock_rs = self._make_mock_request_stats(proxy_module, success_rate=1.0,
                                                 median_latency=1.0,
                                                 sample_count=0)

        with patch.object(proxy_module, "request_stats", mock_rs):
            result = tracker.get_reliable_model("excluded-model")
        # model-fast should win due to much better speed_score
        assert result == "model-fast"

    def test_blends_request_stats_when_enough_samples(self, proxy_module):
        tracker = proxy_module.ModelStatsTracker()

        # model-a: all TTFT successes, fast
        for _ in range(10):
            tracker.record("model-a", 1.0, True)

        # model-b: all TTFT successes, equally fast
        for _ in range(10):
            tracker.record("model-b", 1.0, True)

        # model-a has poor request_stats reliability (req_samples >= 5 triggers blending)
        mock_rs_a = MagicMock(spec=proxy_module.RequestStatsTracker)

        def reliability_side_effect(model):
            if model == "model-a":
                return (0.5, 2.0)  # poor request success rate
            return (1.0, 1.0)  # model-b: perfect

        def sample_count_side_effect(model):
            return 10  # both have enough samples to trigger blending

        mock_rs_a.get_model_reliability.side_effect = reliability_side_effect
        mock_rs_a.get_model_sample_count.side_effect = sample_count_side_effect

        with patch.object(proxy_module, "request_stats", mock_rs_a):
            result = tracker.get_reliable_model("excluded")
        # model-b should win because model-a's blended success_rate is
        # 1.0 * 0.4 + 0.5 * 0.6 = 0.7 while model-b's is
        # 1.0 * 0.4 + 1.0 * 0.6 = 1.0
        assert result == "model-b"

    def test_skips_model_with_no_successful_ttfts(self, proxy_module):
        tracker = proxy_module.ModelStatsTracker()
        # All failures -> no ttfts list -> should be skipped
        for _ in range(10):
            tracker.record("model-a", 1.0, False)
        mock_rs = self._make_mock_request_stats(proxy_module)

        with patch.object(proxy_module, "request_stats", mock_rs):
            result = tracker.get_reliable_model("other")
        assert result is None

    def test_sample_confidence_contributes_to_load_balanced_score(self, proxy_module):
        """Both fast-tier models should be reachable via load-balanced selection,
        and the score math should still give the higher-sample_confidence model
        a measurable edge in the weights.

        Under the load-balanced selection, get_reliable_model() returns a random
        pick weighted by score among models with median_ttft < 3s and
        success_rate >= 0.8. The sample_confidence weight (0.1) only contributes
        a ~7% score difference here, so we verify reachability across many trials
        rather than asserting strict majority (which is within statistical noise).
        """
        tracker = proxy_module.ModelStatsTracker()

        # model-a: few samples (5), fast — low sample_confidence
        for _ in range(5):
            tracker.record("model-a", 1.0, True)

        # model-b: many samples (25), same speed — high sample_confidence
        for _ in range(25):
            tracker.record("model-b", 1.0, True)

        mock_rs = self._make_mock_request_stats(proxy_module, success_rate=1.0,
                                                 median_latency=1.0,
                                                 sample_count=0)

        picks = {"model-a": 0, "model-b": 0}
        with patch.object(proxy_module, "request_stats", mock_rs):
            for _ in range(500):
                result = tracker.get_reliable_model("excluded")
                picks[result] = picks.get(result, 0) + 1

        # Both models must be reachable — this is the load-balancing guarantee.
        assert picks["model-a"] > 0
        assert picks["model-b"] > 0

    def test_load_balances_across_multiple_fast_models(self, proxy_module):
        """Multiple models that all beat the 3s / 0.8 fast-tier threshold should
        each be picked over many iterations. This protects the single fastest
        model from being hammered and rate-limited."""
        tracker = proxy_module.ModelStatsTracker()

        # Three equally-fast, equally-reliable models.
        for model in ("model-x", "model-y", "model-z"):
            for _ in range(10):
                tracker.record(model, 1.0, True)

        mock_rs = self._make_mock_request_stats(proxy_module, success_rate=1.0,
                                                 median_latency=1.0,
                                                 sample_count=0)

        picks = {"model-x": 0, "model-y": 0, "model-z": 0}
        with patch.object(proxy_module, "request_stats", mock_rs):
            for _ in range(600):
                result = tracker.get_reliable_model("excluded")
                picks[result] = picks.get(result, 0) + 1

        # All three must see meaningful traffic (>5% each).
        for model in picks:
            assert picks[model] > 30, f"{model} picked only {picks[model]}/600 times"

    def test_slow_model_excluded_from_fast_tier(self, proxy_module):
        """A model with median TTFT >= 3s should NOT be in the fast tier;
        with no fast-tier candidates, the slow model is still returned via
        the best-effort fallback path."""
        tracker = proxy_module.ModelStatsTracker()

        # Only one model, and it's slow (median 5s > 3s threshold).
        for _ in range(10):
            tracker.record("model-slow", 5.0, True)

        mock_rs = self._make_mock_request_stats(proxy_module, success_rate=1.0,
                                                 median_latency=1.0,
                                                 sample_count=0)

        with patch.object(proxy_module, "request_stats", mock_rs):
            result = tracker.get_reliable_model("excluded")
        # Falls through to best-effort tier since no fast candidates.
        assert result == "model-slow"


# ---------------------------------------------------------------------------
# RequestStatsTracker
# ---------------------------------------------------------------------------


class TestRequestStatsTrackerEmpty:
    def test_empty_tracker_returns_zero_counts(self, proxy_module):
        tracker = proxy_module.RequestStatsTracker()
        stats = tracker.get_stats()

        assert stats["global"]["total_requests"] == 0
        assert stats["global"]["total_errors"] == 0
        assert stats["global"]["error_rate"] == 0
        assert stats["by_model"] == {}

    def test_empty_sample_count(self, proxy_module):
        tracker = proxy_module.RequestStatsTracker()
        assert tracker.get_model_sample_count("nonexistent") == 0


class TestRequestStatsTrackerRecord:
    def test_record_and_retrieve(self, proxy_module):
        tracker = proxy_module.RequestStatsTracker()
        tracker.record("model-a", "streaming", 1.5, True)
        tracker.record("model-a", "streaming", 2.5, True)

        stats = tracker.get_stats()
        assert stats["global"]["total_requests"] == 2
        assert stats["global"]["total_errors"] == 0

        model_stats = stats["by_model"]["model-a"]["streaming"]
        assert model_stats["requests"] == 2
        assert model_stats["errors"] == 0
        assert model_stats["error_rate"] == 0
        assert model_stats["median_latency_s"] == 2.0
        assert model_stats["avg_latency_s"] == 2.0

    def test_tracks_streaming_and_direct_separately(self, proxy_module):
        tracker = proxy_module.RequestStatsTracker()
        tracker.record("model-a", "streaming", 1.0, True)
        tracker.record("model-a", "streaming", 2.0, True)
        tracker.record("model-a", "direct", 5.0, True)

        stats = tracker.get_stats()
        streaming = stats["by_model"]["model-a"]["streaming"]
        direct = stats["by_model"]["model-a"]["direct"]

        assert streaming["requests"] == 2
        assert direct["requests"] == 1
        assert streaming["median_latency_s"] == 1.5
        assert direct["median_latency_s"] == 5.0

    def test_error_counting(self, proxy_module):
        tracker = proxy_module.RequestStatsTracker()
        tracker.record("model-a", "streaming", 1.0, True)
        tracker.record("model-a", "streaming", 2.0, False)
        tracker.record("model-a", "streaming", 3.0, False)

        stats = tracker.get_stats()
        assert stats["global"]["total_requests"] == 3
        assert stats["global"]["total_errors"] == 2
        assert stats["global"]["error_rate"] == round(2 / 3, 3)

        model_stats = stats["by_model"]["model-a"]["streaming"]
        assert model_stats["errors"] == 2
        assert model_stats["error_rate"] == round(2 / 3, 3)

    def test_p90_latency_requires_10_samples(self, proxy_module):
        tracker = proxy_module.RequestStatsTracker()
        for i in range(9):
            tracker.record("model-a", "direct", float(i), True)

        stats = tracker.get_stats()
        assert stats["by_model"]["model-a"]["direct"]["p90_latency_s"] is None

        tracker.record("model-a", "direct", 9.0, True)
        stats = tracker.get_stats()
        assert stats["by_model"]["model-a"]["direct"]["p90_latency_s"] is not None

    def test_multiple_models(self, proxy_module):
        tracker = proxy_module.RequestStatsTracker()
        tracker.record("model-a", "streaming", 1.0, True)
        tracker.record("model-b", "direct", 2.0, False)

        stats = tracker.get_stats()
        assert stats["global"]["total_requests"] == 2
        assert stats["global"]["total_errors"] == 1
        assert "model-a" in stats["by_model"]
        assert "model-b" in stats["by_model"]


class TestRequestStatsTrackerReliability:
    def test_no_data_returns_zero_and_999(self, proxy_module):
        tracker = proxy_module.RequestStatsTracker()
        success_rate, median_lat = tracker.get_model_reliability("nonexistent")
        assert success_rate == 0
        assert median_lat == 999

    def test_reliability_across_modes(self, proxy_module):
        tracker = proxy_module.RequestStatsTracker()
        tracker.record("model-a", "streaming", 1.0, True)
        tracker.record("model-a", "streaming", 2.0, True)
        tracker.record("model-a", "direct", 3.0, False)

        success_rate, median_lat = tracker.get_model_reliability("model-a")
        # 2 successes out of 3 total
        assert success_rate == pytest.approx(2 / 3)
        # median of [1.0, 2.0, 3.0] = 2.0
        assert median_lat == 2.0

    def test_reliability_all_successes(self, proxy_module):
        tracker = proxy_module.RequestStatsTracker()
        tracker.record("model-a", "streaming", 1.0, True)
        tracker.record("model-a", "direct", 2.0, True)

        success_rate, median_lat = tracker.get_model_reliability("model-a")
        assert success_rate == 1.0
        assert median_lat == 1.5


class TestRequestStatsTrackerSampleCount:
    def test_aggregates_across_modes(self, proxy_module):
        tracker = proxy_module.RequestStatsTracker()
        tracker.record("model-a", "streaming", 1.0, True)
        tracker.record("model-a", "streaming", 2.0, True)
        tracker.record("model-a", "direct", 3.0, True)

        assert tracker.get_model_sample_count("model-a") == 3

    def test_nonexistent_model_returns_zero(self, proxy_module):
        tracker = proxy_module.RequestStatsTracker()
        assert tracker.get_model_sample_count("nonexistent") == 0

    def test_counts_errors_in_sample_count(self, proxy_module):
        tracker = proxy_module.RequestStatsTracker()
        tracker.record("model-a", "streaming", 1.0, True)
        tracker.record("model-a", "streaming", 2.0, False)

        assert tracker.get_model_sample_count("model-a") == 2


class TestRequestStatsTrackerWindow:
    def test_latency_window_respects_maxlen(self, proxy_module):
        tracker = proxy_module.RequestStatsTracker(window=5)
        for i in range(10):
            tracker.record("model-a", "streaming", float(i), True)

        stats = tracker.get_stats()
        # count tracks all 10 records, but latencies deque only keeps last 5
        model_stats = stats["by_model"]["model-a"]["streaming"]
        assert model_stats["requests"] == 10
        # median of [5.0, 6.0, 7.0, 8.0, 9.0] = 7.0
        assert model_stats["median_latency_s"] == 7.0
