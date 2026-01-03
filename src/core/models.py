"""Shared domain-agnostic dataclasses for mala.

This module provides shared types that are used across multiple domains
(logging, validation, orchestrator) to avoid circular dependencies.

Types:
- ResolutionOutcome: Enum for issue resolution outcomes
- IssueResolution: Records how an issue was resolved
- ValidationArtifacts: Record of validation outputs for observability
- UnmetCriterion: Individual gap identified during epic verification
- EpicVerdict: Result of verifying an epic against its acceptance criteria
- EpicVerificationResult: Summary of a verification run across multiple epics
- RetryConfig: Configuration for retry behavior with exponential backoff
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pathlib import Path


class ResolutionOutcome(Enum):
    """Outcome of issue resolution."""

    SUCCESS = "success"
    NO_CHANGE = "no_change"
    OBSOLETE = "obsolete"
    ALREADY_COMPLETE = "already_complete"


@dataclass(frozen=True)
class IssueResolution:
    """Records how an issue was resolved.

    Attributes:
        outcome: The resolution outcome (success, no_change, obsolete).
        rationale: Explanation for the resolution.
    """

    outcome: ResolutionOutcome
    rationale: str


@dataclass
class ValidationArtifacts:
    """Record of validation outputs for observability.

    Attributes:
        log_dir: Directory containing validation logs.
        worktree_path: Path to the worktree (if any).
        worktree_state: State of the worktree ("kept" or "removed").
        coverage_report: Path to coverage report.
        e2e_fixture_path: Path to E2E fixture directory.
    """

    log_dir: Path
    worktree_path: Path | None = None
    worktree_state: Literal["kept", "removed"] | None = None
    coverage_report: Path | None = None
    e2e_fixture_path: Path | None = None


@dataclass
class UnmetCriterion:
    """Individual gap identified during epic verification.

    Attributes:
        criterion: The acceptance criterion not met.
        evidence: Why it's considered unmet.
        priority: Issue priority matching Cerberus levels (0-3).
            P0/P1 are blocking, P2/P3 are non-blocking (informational).
        criterion_hash: SHA256 of criterion text, for deduplication.
    """

    criterion: str
    evidence: str
    priority: int  # 0=P0 (blocking), 1=P1 (blocking), 2=P2 (non-blocking), 3=P3 (non-blocking)
    criterion_hash: str


@dataclass
class EpicVerdict:
    """Result of verifying an epic against its acceptance criteria.

    Attributes:
        passed: Whether all acceptance criteria were met.
        unmet_criteria: List of criteria that were not satisfied.
        confidence: Model confidence in the verdict (0.0 to 1.0).
        reasoning: Explanation of the verification outcome.
    """

    passed: bool
    unmet_criteria: list[UnmetCriterion]
    confidence: float
    reasoning: str


@dataclass
class EpicVerificationResult:
    """Summary of a verification run across multiple epics.

    Attributes:
        verified_count: Number of epics verified.
        passed_count: Number that passed verification.
        failed_count: Number that failed verification.
        verdicts: Mapping of epic_id to its verdict.
        remediation_issues_created: Issue IDs created for remediation.
    """

    verified_count: int
    passed_count: int
    failed_count: int
    verdicts: dict[str, EpicVerdict]
    remediation_issues_created: list[str]


@dataclass
class RetryConfig:
    """Configuration for retry behavior with exponential backoff.

    Attributes:
        max_retries: Total retry attempts.
        initial_delay_ms: First retry delay in milliseconds.
        backoff_multiplier: Exponential backoff multiplier.
        max_delay_ms: Cap on retry delay in milliseconds.
        timeout_ms: Per-attempt timeout in milliseconds.
    """

    max_retries: int = 2
    initial_delay_ms: int = 1000
    backoff_multiplier: float = 2.0
    max_delay_ms: int = 30000
    timeout_ms: int = 120000
