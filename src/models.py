"""Backward-compatibility shim for src.models.

This module re-exports all public types from src.core.models.
New code should import directly from src.core.models.
"""

from src.core.models import (
    EpicVerdict,
    EpicVerificationResult,
    IssueResolution,
    ResolutionOutcome,
    RetryConfig,
    UnmetCriterion,
    ValidationArtifacts,
)

__all__ = [
    "EpicVerdict",
    "EpicVerificationResult",
    "IssueResolution",
    "ResolutionOutcome",
    "RetryConfig",
    "UnmetCriterion",
    "ValidationArtifacts",
]
