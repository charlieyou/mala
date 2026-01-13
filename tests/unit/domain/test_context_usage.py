"""Unit tests for ContextUsage token tracking and pressure calculation."""

import pytest

from src.domain.lifecycle import ContextUsage, TRACKING_DISABLED


class TestContextUsageAddTurn:
    """Tests for add_turn() accumulation."""

    def test_add_turn_accumulates_tokens(self) -> None:
        """add_turn() should accumulate tokens across multiple calls."""
        usage = ContextUsage()

        usage.add_turn(input_tokens=100, output_tokens=50, cache_read_tokens=200)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.cache_read_tokens == 200

        usage.add_turn(input_tokens=150, output_tokens=30, cache_read_tokens=100)
        assert usage.input_tokens == 250
        assert usage.output_tokens == 80
        assert usage.cache_read_tokens == 300

    def test_add_turn_noop_when_disabled(self) -> None:
        """add_turn() should be a no-op when tracking is disabled."""
        usage = ContextUsage()
        usage.disable_tracking()

        usage.add_turn(input_tokens=100, output_tokens=50, cache_read_tokens=200)

        assert usage.input_tokens == TRACKING_DISABLED
        assert usage.output_tokens == 0
        assert usage.cache_read_tokens == 0

    def test_add_turn_clamps_negative_values(self) -> None:
        """add_turn() should clamp negative values to 0."""
        usage = ContextUsage(input_tokens=100, output_tokens=50, cache_read_tokens=50)

        usage.add_turn(input_tokens=-50, output_tokens=-30, cache_read_tokens=-20)

        # Negative values clamped to 0, so totals unchanged
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.cache_read_tokens == 50


class TestContextUsagePressureRatio:
    """Tests for pressure_ratio() calculation."""

    def test_pressure_ratio_excludes_cache_read(self) -> None:
        """pressure_ratio() should exclude cache_read_tokens from calculation."""
        usage = ContextUsage(
            input_tokens=1_000, output_tokens=500, cache_read_tokens=10_000
        )

        # Pressure should be based on input + output only (1500 / 200000)
        ratio = usage.pressure_ratio(200_000)
        assert ratio == pytest.approx(0.0075)

        # Verify cache_read NOT included: if it were, ratio would be ~0.0575
        assert ratio < 0.01

    def test_disabled_tracking_returns_zero_pressure(self) -> None:
        """pressure_ratio() should return 0.0 when tracking is disabled."""
        usage = ContextUsage()
        usage.disable_tracking()

        assert usage.pressure_ratio(200_000) == 0.0

    def test_pressure_ratio_with_zero_limit(self) -> None:
        """pressure_ratio() should return 0.0 when limit is 0 or negative."""
        usage = ContextUsage(input_tokens=1000, output_tokens=500)

        assert usage.pressure_ratio(0) == 0.0
        assert usage.pressure_ratio(-100) == 0.0

    def test_pressure_ratio_can_exceed_one(self) -> None:
        """pressure_ratio() can exceed 1.0 when over the limit."""
        usage = ContextUsage(input_tokens=150_000, output_tokens=100_000)

        # 250K tokens / 200K limit = 1.25
        ratio = usage.pressure_ratio(200_000)
        assert ratio == pytest.approx(1.25)


class TestContextUsageReset:
    """Tests for reset() behavior."""

    def test_reset_clears_counters(self) -> None:
        """reset() should clear all counters to zero."""
        usage = ContextUsage(input_tokens=100, output_tokens=50, cache_read_tokens=200)

        usage.reset()

        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.cache_read_tokens == 0

    def test_reset_preserves_disabled_state(self) -> None:
        """reset() should keep tracking disabled if it was disabled."""
        usage = ContextUsage(output_tokens=50, cache_read_tokens=200)
        usage.disable_tracking()

        usage.reset()

        assert usage.tracking_disabled is True
        assert usage.input_tokens == TRACKING_DISABLED
        assert usage.output_tokens == 0
        assert usage.cache_read_tokens == 0


class TestContextUsageIsTrackingEnabled:
    """Tests for is_tracking_enabled() method."""

    def test_is_tracking_enabled(self) -> None:
        """is_tracking_enabled() should return correct state."""
        usage = ContextUsage()
        assert usage.is_tracking_enabled() is True

        usage.disable_tracking()
        assert usage.is_tracking_enabled() is False
