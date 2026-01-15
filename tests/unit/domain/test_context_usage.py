"""Unit tests for ContextUsage token tracking and pressure calculation."""

import pytest

from src.domain.lifecycle import TRACKING_DISABLED, ContextUsage


class TestContextUsageUpdateFromRequest:
    """Tests for update_from_request() per-request value storage."""

    def test_update_replaces_values(self) -> None:
        """update_from_request() should replace, not accumulate."""
        usage = ContextUsage()

        usage.update_from_request(
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=200,
            cache_creation_tokens=1000,
        )
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.cache_read_tokens == 200
        assert usage.cache_creation_tokens == 1000

        # Second call replaces previous values
        usage.update_from_request(
            input_tokens=5,
            output_tokens=30,
            cache_read_tokens=500,
            cache_creation_tokens=2000,
        )
        assert usage.input_tokens == 5
        assert usage.output_tokens == 30
        assert usage.cache_read_tokens == 500
        assert usage.cache_creation_tokens == 2000

    def test_update_noop_when_disabled(self) -> None:
        """update_from_request() should be a no-op when tracking is disabled."""
        usage = ContextUsage()
        usage.disable_tracking()

        usage.update_from_request(
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=200,
            cache_creation_tokens=1000,
        )

        assert usage.input_tokens == TRACKING_DISABLED
        assert usage.output_tokens == 0
        assert usage.cache_read_tokens == 0
        assert usage.cache_creation_tokens == 0

    def test_update_clamps_negative_values(self) -> None:
        """update_from_request() should clamp negative values to 0."""
        usage = ContextUsage()

        usage.update_from_request(
            input_tokens=-50,
            output_tokens=-30,
            cache_read_tokens=-20,
            cache_creation_tokens=-100,
        )

        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.cache_read_tokens == 0
        assert usage.cache_creation_tokens == 0


class TestContextUsagePromptTokens:
    """Tests for prompt_tokens property."""

    def test_prompt_tokens_sums_input_cache_read_cache_creation(self) -> None:
        """prompt_tokens should be input + cache_read + cache_creation."""
        usage = ContextUsage(
            input_tokens=100,
            output_tokens=500,  # Should NOT be included
            cache_read_tokens=10_000,
            cache_creation_tokens=5_000,
        )

        assert usage.prompt_tokens == 15_100  # 100 + 10000 + 5000


class TestContextUsagePressureRatio:
    """Tests for pressure_ratio() calculation."""

    def test_pressure_ratio_includes_all_prompt_tokens(self) -> None:
        """pressure_ratio() should use input + cache_read + cache_creation."""
        usage = ContextUsage(
            input_tokens=2,
            output_tokens=5_000,  # Should NOT be included
            cache_read_tokens=12_000,
            cache_creation_tokens=185_000,
        )

        # Prompt = 2 + 12000 + 185000 = 197002
        # Pressure = 197002 / 200000 = 0.98501
        ratio = usage.pressure_ratio(200_000)
        assert ratio == pytest.approx(0.98501)

    def test_pressure_ratio_excludes_output_tokens(self) -> None:
        """Output tokens should not affect pressure (already in future prompts)."""
        usage = ContextUsage(
            input_tokens=1_000,
            cache_read_tokens=50_000,
            cache_creation_tokens=49_000,
        )

        base_ratio = usage.pressure_ratio(200_000)  # 100000 / 200000 = 0.5

        # Adding output tokens should not change pressure
        usage.output_tokens = 50_000
        assert usage.pressure_ratio(200_000) == base_ratio

    def test_disabled_tracking_returns_zero_pressure(self) -> None:
        """pressure_ratio() should return 0.0 when tracking is disabled."""
        usage = ContextUsage()
        usage.disable_tracking()

        assert usage.pressure_ratio(200_000) == 0.0

    def test_pressure_ratio_with_zero_limit(self) -> None:
        """pressure_ratio() should return 0.0 when limit is 0 or negative."""
        usage = ContextUsage(input_tokens=1000, cache_read_tokens=5000)

        assert usage.pressure_ratio(0) == 0.0
        assert usage.pressure_ratio(-100) == 0.0

    def test_pressure_ratio_can_exceed_one(self) -> None:
        """pressure_ratio() can exceed 1.0 when over the limit."""
        usage = ContextUsage(
            input_tokens=10_000,
            cache_read_tokens=100_000,
            cache_creation_tokens=140_000,
        )

        # 250K tokens / 200K limit = 1.25
        ratio = usage.pressure_ratio(200_000)
        assert ratio == pytest.approx(1.25)


class TestContextUsageReset:
    """Tests for reset() behavior."""

    def test_reset_clears_counters(self) -> None:
        """reset() should clear all counters to zero."""
        usage = ContextUsage(
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=200,
            cache_creation_tokens=1000,
        )

        usage.reset()

        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.cache_read_tokens == 0
        assert usage.cache_creation_tokens == 0

    def test_reset_preserves_disabled_state(self) -> None:
        """reset() should keep tracking disabled if it was disabled."""
        usage = ContextUsage(
            output_tokens=50, cache_read_tokens=200, cache_creation_tokens=500
        )
        usage.disable_tracking()

        usage.reset()

        assert usage.tracking_disabled is True
        assert usage.input_tokens == TRACKING_DISABLED
        assert usage.output_tokens == 0
        assert usage.cache_read_tokens == 0
        assert usage.cache_creation_tokens == 0


class TestContextUsageIsTrackingEnabled:
    """Tests for is_tracking_enabled() method."""

    def test_is_tracking_enabled(self) -> None:
        """is_tracking_enabled() should return correct state."""
        usage = ContextUsage()
        assert usage.is_tracking_enabled() is True

        usage.disable_tracking()
        assert usage.is_tracking_enabled() is False
