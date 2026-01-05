"""Unit tests for idle_retry_policy module."""

import pytest

from src.pipeline.idle_retry_policy import RetryConfig


@pytest.mark.unit
class TestRetryConfigValidation:
    """Tests for RetryConfig __post_init__ validation."""

    def test_default_config_valid(self) -> None:
        """Default RetryConfig with zero retries should pass validation."""
        config = RetryConfig(max_idle_retries=0)
        assert config.max_idle_retries == 0
        assert config.idle_retry_backoff == (0.0, 5.0, 15.0)

    def test_negative_max_retries_raises(self) -> None:
        """Negative max_idle_retries should raise ValueError."""
        with pytest.raises(ValueError, match="max_idle_retries must be non-negative"):
            RetryConfig(max_idle_retries=-1)

    def test_empty_resume_prompt_with_retries_raises(self) -> None:
        """Empty idle_resume_prompt with positive max_idle_retries should raise ValueError."""
        with pytest.raises(
            ValueError,
            match=r"idle_resume_prompt must be non-empty when max_idle_retries > 0",
        ):
            RetryConfig(max_idle_retries=3, idle_resume_prompt="")

    def test_backoff_exact_length_valid(self) -> None:
        """Backoff with exactly max_idle_retries + 1 entries should be valid."""
        config = RetryConfig(
            max_idle_retries=2,
            idle_retry_backoff=(0.0, 5.0, 10.0),
            idle_resume_prompt="Continue {issue_id}",
        )
        assert config.max_idle_retries == 2
        assert len(config.idle_retry_backoff) == 3

    def test_backoff_longer_than_needed_valid(self) -> None:
        """Backoff with more entries than needed should be valid."""
        config = RetryConfig(
            max_idle_retries=1,
            idle_retry_backoff=(0.0, 5.0, 10.0, 20.0),
            idle_resume_prompt="Continue {issue_id}",
        )
        assert config.max_idle_retries == 1
        assert len(config.idle_retry_backoff) == 4

    def test_backoff_shorter_than_retries_valid(self) -> None:
        """Backoff shorter than max_idle_retries is valid (reuses last entry)."""
        # 3 retries but only 2 backoff entries - last entry will be reused
        config = RetryConfig(
            max_idle_retries=3,
            idle_retry_backoff=(0.0, 5.0),
            idle_resume_prompt="Continue {issue_id}",
        )
        assert config.max_idle_retries == 3
        assert len(config.idle_retry_backoff) == 2

    def test_zero_retries_with_single_backoff_valid(self) -> None:
        """Zero retries with a single backoff value should be valid."""
        config = RetryConfig(max_idle_retries=0, idle_retry_backoff=(0.0,))
        assert config.max_idle_retries == 0
        assert config.idle_retry_backoff == (0.0,)

    def test_zero_retries_with_empty_backoff_valid(self) -> None:
        """Zero retries with empty backoff should be valid (no retries needed)."""
        config = RetryConfig(max_idle_retries=0, idle_retry_backoff=())
        assert config.max_idle_retries == 0
        assert config.idle_retry_backoff == ()
