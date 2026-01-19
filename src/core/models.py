"""Shared domain-agnostic dataclasses for mala.

This module provides shared types that are used across multiple domains
(logging, validation, orchestrator) to avoid circular dependencies.

Types:
- LockEventType: Enum for lock event types (acquired, waiting, released)
- LockEvent: A lock event from an agent
- ResolutionOutcome: Enum for issue resolution outcomes
- IssueResolution: Records how an issue was resolved
- ValidationArtifacts: Record of validation outputs for observability
- UnmetCriterion: Individual gap identified during epic verification
- EpicVerdict: Result of verifying an epic against its acceptance criteria
- EpicVerificationResult: Summary of a verification run across multiple epics
- RetryConfig: Configuration for retry behavior with exponential backoff
- WatchConfig: Configuration for watch mode behavior
- WatchState: Runtime state for watch mode
- RunResult: Result of a coordinator run loop
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from src.core.session_end_result import SessionEndResult


class OrderPreference(Enum):
    """Issue ordering preference for orchestrator.

    Attributes:
        FOCUS: Only work on one epic at a time (strict single-epic mode).
        EPIC_PRIORITY: Group tasks by epic, order groups by priority (default).
        ISSUE_PRIORITY: Use global priority ordering across all epics.
        INPUT: Preserve user-specified input order (only valid with --scope ids:).
    """

    FOCUS = "focus"
    EPIC_PRIORITY = "epic-priority"
    ISSUE_PRIORITY = "issue-priority"
    INPUT = "input"


class LockEventType(Enum):
    """Types of lock events emitted by agents."""

    ACQUIRED = "acquired"
    WAITING = "waiting"
    RELEASED = "released"


@dataclass
class LockEvent:
    """A lock event from an agent.

    Attributes:
        event_type: Type of lock event.
        agent_id: ID of the agent that emitted this event.
        lock_path: Path to the lock file.
        timestamp: Unix timestamp when the event occurred.
    """

    event_type: LockEventType
    agent_id: str
    lock_path: str
    timestamp: float


class ResolutionOutcome(Enum):
    """Outcome of issue resolution."""

    SUCCESS = "success"
    NO_CHANGE = "no_change"
    OBSOLETE = "obsolete"
    ALREADY_COMPLETE = "already_complete"
    DOCS_ONLY = "docs_only"


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
        reasoning: Explanation of the verification outcome.
    """

    passed: bool
    unmet_criteria: list[UnmetCriterion]
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
        ineligibility_reason: Human-readable explanation if epic was not eligible.
        interrupted: Whether the verification was interrupted by SIGINT.
    """

    verified_count: int
    passed_count: int
    failed_count: int
    verdicts: dict[str, EpicVerdict]
    remediation_issues_created: list[str]
    ineligibility_reason: str | None = None
    interrupted: bool = False


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


@dataclass
class WatchConfig:
    """Configuration for watch mode polling behavior.

    Attributes:
        enabled: Whether watch mode is active.
        poll_interval_seconds: Seconds between poll cycles (internal, not CLI-exposed).
    """

    enabled: bool = False
    poll_interval_seconds: float = 60.0


@dataclass
class PeriodicValidationConfig:
    """Configuration for periodic validation triggering.

    Attributes:
        validate_every: Run validation after this many issues complete.
            None means disabled.
    """

    validate_every: int | None = None


@dataclass
class WatchState:
    """Runtime state for watch mode (internal to Coordinator).

    Attributes:
        completed_count: Total issues completed in this watch session.
        last_validation_at: Completed count at last validation.
        next_validation_threshold: Run validation when completed_count reaches this.
            Must be initialized from PeriodicValidationConfig.validate_every to stay in sync.
        consecutive_poll_failures: Count of consecutive poll failures.
        last_idle_log_time: Timestamp of last idle log message.
        was_idle: Whether the previous poll found no ready issues.
    """

    # No default - must be explicitly set from PeriodicValidationConfig.validate_every
    next_validation_threshold: int
    completed_count: int = 0
    last_validation_at: int = 0
    consecutive_poll_failures: int = 0
    last_idle_log_time: float = 0.0
    was_idle: bool = False


@dataclass
class RunResult:
    """Result of a coordinator run loop.

    Attributes:
        issues_spawned: Number of issues spawned during the run.
        exit_code: Exit code (0=success, 1=validation_failed, 2=invalid_args, 3=abort, 130=interrupted).
        exit_reason: Human-readable exit reason.
    """

    issues_spawned: int
    exit_code: int
    exit_reason: str


@dataclass
class ReviewInput:
    """Input for running a code review.

    Bundles all data needed to run a single review check.

    Attributes:
        issue_id: The issue being reviewed.
        repo_path: Path to the repository.
        issue_description: Issue description for scope verification.
        commit_shas: List of commit SHAs to review directly.
        claude_session_id: Optional Claude session ID for external review context.
        author_context: Optional author-provided context for the reviewer.
        previous_findings: Optional list of issues from the previous review attempt.
            When provided, these are included in the context so the reviewer can
            see what was disputed and verify claims before re-flagging.
        diff_content: Optional pre-computed diff content for cumulative reviews.
            When provided, the reviewer should use this diff instead of computing
            it from commit_shas. This supports cumulative review workflows where
            the diff is explicitly generated from a baseline..HEAD range.
        session_end_result: Optional SessionEndResult from session_end trigger.
            Per spec R5, this is informational evidence for the reviewer.
            Review does NOT auto-fail based on session_end status.
    """

    issue_id: str
    repo_path: Path
    commit_shas: list[str]
    issue_description: str | None = None
    claude_session_id: str | None = None
    author_context: str | None = None
    previous_findings: Sequence[Any] | None = None
    diff_content: str | None = None
    session_end_result: SessionEndResult | None = None
