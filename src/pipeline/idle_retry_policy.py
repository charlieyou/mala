"""Idle timeout retry policy types for AgentSessionRunner.

This module contains the dataclasses for idle timeout retry handling,
used by IdleTimeoutRetryPolicy.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetryConfig:
    """Configuration for idle timeout retry behavior."""

    max_idle_retries: int = 2
    idle_retry_backoff: tuple[float, ...] = (0.0, 5.0, 15.0)


@dataclass
class IterationResult:
    """Result from a single session iteration."""

    success: bool
    session_id: str | None = None
    should_continue: bool = True
    error_message: str | None = None
