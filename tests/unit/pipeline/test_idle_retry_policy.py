"""Unit tests for idle_retry_policy module."""

import pytest

from src.pipeline.idle_retry_policy import RetryConfig


@pytest.mark.unit
class TestRetryConfigValidation:
    """Tests for RetryConfig __post_init__ validation."""

    def test_default_config_valid(self) -> None:
        """Default RetryConfig should pass validation."""
        config = RetryConfig()
        assert config.max_idle_retries == 2
        assert config.idle_retry_backoff == (0.0, 5.0, 15.0)

    def test_negative_max_retries_raises(self) -> None:
        """Negative max_idle_retries should raise ValueError."""
        with pytest.raises(ValueError, match="max_idle_retries must be non-negative"):
            RetryConfig(max_idle_retries=-1)

    def test_backoff_too_short_raises(self) -> None:
        """Backoff shorter than max_idle_retries + 1 should raise ValueError."""
        with pytest.raises(
            ValueError,
            match=r"idle_retry_backoff must have at least 4 entries.*got 2",
        ):
            RetryConfig(max_idle_retries=3, idle_retry_backoff=(0.0, 5.0))

    def test_backoff_exact_length_valid(self) -> None:
        """Backoff with exactly max_idle_retries + 1 entries should be valid."""
        config = RetryConfig(max_idle_retries=2, idle_retry_backoff=(0.0, 5.0, 10.0))
        assert config.max_idle_retries == 2
        assert len(config.idle_retry_backoff) == 3

    def test_backoff_longer_than_needed_valid(self) -> None:
        """Backoff with more entries than needed should be valid."""
        config = RetryConfig(
            max_idle_retries=1, idle_retry_backoff=(0.0, 5.0, 10.0, 20.0)
        )
        assert config.max_idle_retries == 1
        assert len(config.idle_retry_backoff) == 4

    def test_zero_retries_with_single_backoff_valid(self) -> None:
        """Zero retries with a single backoff value should be valid."""
        config = RetryConfig(max_idle_retries=0, idle_retry_backoff=(0.0,))
        assert config.max_idle_retries == 0
        assert config.idle_retry_backoff == (0.0,)

    def test_zero_retries_with_empty_backoff_raises(self) -> None:
        """Zero retries with empty backoff should raise ValueError."""
        with pytest.raises(
            ValueError,
            match=r"idle_retry_backoff must have at least 1 entries.*got 0",
        ):
            RetryConfig(max_idle_retries=0, idle_retry_backoff=())
