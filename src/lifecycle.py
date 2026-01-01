"""Backward-compatibility shim for src.lifecycle.

This module re-exports all public symbols from src.domain.lifecycle.
New code should import directly from src.domain.lifecycle.
"""

from src.domain.lifecycle import (
    Effect,
    GateOutcome,
    ImplementerLifecycle,
    LifecycleConfig,
    LifecycleContext,
    LifecycleState,
    RetryState,
    ReviewIssue,
    ReviewOutcome,
    TransitionResult,
)

__all__ = [
    "Effect",
    "GateOutcome",
    "ImplementerLifecycle",
    "LifecycleConfig",
    "LifecycleContext",
    "LifecycleState",
    "RetryState",
    "ReviewIssue",
    "ReviewOutcome",
    "TransitionResult",
]
