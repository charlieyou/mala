"""Idle timeout retry policy types for AgentSessionRunner.

This module contains the dataclasses for idle timeout retry handling,
used by IdleTimeoutRetryPolicy.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetryConfig:
    """Configuration for idle timeout retry behavior.

    The backoff sequence must have at least max_idle_retries + 1 entries
    to provide a backoff value for each retry attempt (including retry 0).
    """

    max_idle_retries: int = 2
    idle_retry_backoff: tuple[float, ...] = (0.0, 5.0, 15.0)

    def __post_init__(self) -> None:
        """Validate configuration invariants."""
        if self.max_idle_retries < 0:
            raise ValueError("max_idle_retries must be non-negative")
        required_backoff_len = self.max_idle_retries + 1
        if len(self.idle_retry_backoff) < required_backoff_len:
            raise ValueError(
                f"idle_retry_backoff must have at least {required_backoff_len} entries "
                f"for max_idle_retries={self.max_idle_retries}, got {len(self.idle_retry_backoff)}"
            )


@dataclass
class IterationResult:
    """Result from a single session iteration."""

    success: bool
    session_id: str | None = None
    should_continue: bool = True
    error_message: str | None = None
