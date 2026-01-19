"""Validation protocols for quality gate checking.

This module defines protocols for validation specifications, evidence tracking,
and quality gate checking, enabling the orchestrator to verify agent work
meets quality requirements.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from collections.abc import Sequence  # noqa: TC003 - runtime needed

if TYPE_CHECKING:
    from pathlib import Path

    from src.core.models import EpicVerificationResult
    from src.core.session_end_result import SessionEndResult, SessionEndRetryState

    from .issue import IssueResolutionProtocol


@runtime_checkable
class ValidationSpecProtocol(Protocol):
    """Protocol for validation specification.

    Matches the shape of validation.spec.ValidationSpec for structural typing.
    Only includes attributes/methods that protocols.py method signatures use.
    """

    commands: Sequence[Any]
    """List of validation commands to run."""

    scope: Any
    """The validation scope (per-session or global)."""


@runtime_checkable
class ValidationEvidenceProtocol(Protocol):
    """Protocol for validation evidence from agent runs.

    Matches the shape of evidence_check.ValidationEvidence for structural typing.
    """

    commands_ran: dict[Any, bool]
    """Mapping of CommandKind to whether it ran."""

    failed_commands: list[str]
    """List of validation commands that failed."""

    custom_commands_ran: dict[str, bool]
    """Mapping of custom command name to whether it ran."""

    custom_commands_failed: dict[str, bool]
    """Mapping of custom command name to whether it failed (exited non-zero)."""

    def has_any_evidence(self) -> bool:
        """Check if any validation command ran."""
        ...

    def to_evidence_dict(self) -> dict[str, bool]:
        """Convert evidence to a serializable dict keyed by CommandKind value."""
        ...


@runtime_checkable
class CommitResultProtocol(Protocol):
    """Protocol for commit existence check results.

    Matches the shape of evidence_check.CommitResult for structural typing.
    """

    exists: bool
    """Whether a matching commit exists."""

    commit_hash: str | None
    """The commit hash if found."""

    message: str | None
    """The commit message if found."""


@runtime_checkable
class GateResultProtocol(Protocol):
    """Protocol for quality gate check results.

    Matches the shape of evidence_check.GateResult for structural typing.
    """

    passed: bool
    """Whether the quality gate passed."""

    failure_reasons: list[str]
    """List of reasons why the gate failed."""

    commit_hash: str | None
    """The commit hash if found."""

    validation_evidence: ValidationEvidenceProtocol | None
    """Evidence of validation commands executed."""

    no_progress: bool
    """Whether no progress was detected."""

    resolution: IssueResolutionProtocol | None
    """Issue resolution if applicable."""


@runtime_checkable
class GateChecker(Protocol):
    """Protocol for quality gate checking.

    Provides methods for verifying agent work meets quality requirements.
    The orchestrator uses this after each agent attempt to determine if
    the issue was successfully resolved.

    The canonical implementation is EvidenceCheck, which conforms to this
    protocol. Test implementations can verify specific conditions for isolation.

    Methods match EvidenceCheck's API exactly so EvidenceCheck conforms to this
    protocol without adaptation.
    """

    def check_with_resolution(
        self,
        issue_id: str,
        log_path: Path,
        baseline_timestamp: int | None = None,
        log_offset: int = 0,
        spec: ValidationSpecProtocol | None = None,
    ) -> GateResultProtocol:
        """Run quality gate check with support for no-op/obsolete resolutions.

        This method is scope-aware and handles special resolution outcomes:
        - ISSUE_NO_CHANGE: Issue already addressed, no commit needed
        - ISSUE_OBSOLETE: Issue no longer relevant, no commit needed
        - ISSUE_ALREADY_COMPLETE: Work done in previous run

        Args:
            issue_id: The issue ID to verify.
            log_path: Path to the JSONL log file from agent session.
            baseline_timestamp: Unix timestamp for commit freshness check.
            log_offset: Byte offset to start parsing from.
            spec: ValidationSpec for scope-aware evidence checking (required).

        Returns:
            GateResultProtocol with pass/fail, failure reasons, and resolution.

        Raises:
            ValueError: If spec is not provided.
        """
        ...

    def get_log_end_offset(self, log_path: Path, start_offset: int = 0) -> int:
        """Get the byte offset at the end of a log file.

        This is a lightweight method for getting the current file position
        after reading from a given offset. Use this when you only need the
        offset for retry scoping, not the evidence itself.

        Args:
            log_path: Path to the JSONL log file.
            start_offset: Byte offset to start from (default 0).

        Returns:
            The byte offset at the end of the file, or start_offset if file
            doesn't exist or can't be read.
        """
        ...

    def check_no_progress(
        self,
        log_path: Path,
        log_offset: int,
        previous_commit_hash: str | None,
        current_commit_hash: str | None,
        spec: ValidationSpecProtocol | None = None,
        check_validation_evidence: bool = True,
    ) -> bool:
        """Check if no progress was made since the last attempt.

        No progress is detected when ALL of these are true:
        - The commit hash hasn't changed (or both are None)
        - No uncommitted changes in the working tree
        - (Optionally) No new validation evidence was found after the log offset

        Args:
            log_path: Path to the JSONL log file from agent session.
            log_offset: Byte offset marking the end of the previous attempt.
            previous_commit_hash: Commit hash from the previous attempt.
            current_commit_hash: Commit hash from this attempt.
            spec: Optional ValidationSpec for spec-driven evidence detection.
            check_validation_evidence: If True (default), also check for new validation
                evidence. Set to False for review retries where only commit/working-tree
                changes should gate progress.

        Returns:
            True if no progress was made, False if progress was detected.
        """
        ...

    def parse_validation_evidence_with_spec(
        self, log_path: Path, spec: ValidationSpecProtocol, offset: int = 0
    ) -> ValidationEvidenceProtocol:
        """Parse JSONL log for validation evidence using spec-defined patterns.

        Args:
            log_path: Path to the JSONL log file.
            spec: ValidationSpec defining detection patterns.
            offset: Byte offset to start parsing from (default 0).

        Returns:
            ValidationEvidenceProtocol with flags indicating which validations ran.
        """
        ...

    def check_commit_exists(
        self, issue_id: str, baseline_timestamp: int | None = None
    ) -> CommitResultProtocol:
        """Check if a commit with bd-<issue_id> exists in recent history.

        Searches commits from the last 30 days to accommodate long-running
        work that may span multiple days.

        Args:
            issue_id: The issue ID to search for (without bd- prefix).
            baseline_timestamp: Unix timestamp. If provided, only accepts
                commits created after this time.

        Returns:
            CommitResultProtocol indicating whether a matching commit exists.
        """
        ...


@runtime_checkable
class UnmetCriterionProtocol(Protocol):
    """Protocol for unmet criteria during epic verification.

    Matches the shape of models.UnmetCriterion for structural typing.
    """

    criterion: str
    """The acceptance criterion not met."""

    evidence: str
    """Why it's considered unmet."""

    priority: int
    """Issue priority matching Cerberus levels (0-3). P0/P1 blocking, P2/P3 informational."""

    criterion_hash: str
    """SHA256 of criterion text, for deduplication."""


@runtime_checkable
class EpicVerdictProtocol(Protocol):
    """Protocol for epic verification verdicts.

    Matches the shape of models.EpicVerdict for structural typing.
    """

    passed: bool
    """Whether all acceptance criteria were met."""

    unmet_criteria: Sequence[UnmetCriterionProtocol]
    """List of criteria that were not satisfied."""

    reasoning: str
    """Explanation of the verification outcome."""


@runtime_checkable
class EpicVerifierProtocol(Protocol):
    """Protocol for epic verification to avoid importing concrete EpicVerifier.

    Provides the async interface for verifying and closing epics based on
    their acceptance criteria. The orchestrator uses this to trigger epic
    verification when child issues complete.

    The canonical implementation is EpicVerifier in src/infra/epic_verifier.py.
    """

    async def verify_and_close_epic(
        self,
        epic_id: str,
        human_override: bool = False,
    ) -> EpicVerificationResult:
        """Verify and close a single specific epic if eligible.

        This method checks if the specified epic is eligible (all children closed),
        then verifies it if eligible, and closes it if verification passes.

        Args:
            epic_id: The epic ID to verify and close.
            human_override: If True, bypass verification and close directly.

        Returns:
            EpicVerificationResult with verification outcome.
        """
        ...


@runtime_checkable
class EpicVerificationModel(Protocol):
    """Protocol for model-agnostic epic verification.

    Provides an interface for verifying whether an epic's acceptance
    criteria are met. The model explores the repository using its tools
    to find and verify the implementation of each criterion.

    The canonical implementation is ClaudeEpicVerificationModel in
    src/epic_verifier.py. Test implementations can return predetermined
    verdicts for isolation.
    """

    async def verify(
        self,
        epic_context: str,
    ) -> EpicVerdictProtocol:
        """Verify if the epic's acceptance criteria are met.

        The model explores the repository using its tools to find and verify
        the implementation of each acceptance criterion.

        Args:
            epic_context: Combined epic content including description, plan,
                and spec file content. This is injected into the prompt's
                EPIC_CONTEXT section.

        Returns:
            Structured verdict with pass/fail and unmet criteria details.
        """
        ...


@runtime_checkable
class GateOutcomeProtocol(Protocol):
    """Protocol for gate check outcome.

    Matches the shape of domain.lifecycle.GateOutcome for structural typing.
    This allows IGateRunner to reference the return type without importing
    from domain.
    """

    @property
    def passed(self) -> bool:
        """Whether the gate check passed."""
        ...

    @property
    def failure_reasons(self) -> list[str]:
        """Reasons for failure (empty if passed)."""
        ...

    @property
    def commit_hash(self) -> str | None:
        """Commit hash if a commit was found."""
        ...

    @property
    def no_progress(self) -> bool:
        """Whether no progress was detected since last attempt."""
        ...

    @property
    def resolution(self) -> IssueResolutionProtocol | None:
        """Issue resolution if a resolution marker was found."""
        ...


@runtime_checkable
class RetryStateProtocol(Protocol):
    """Protocol for retry state.

    Matches the shape of domain.lifecycle.RetryState for structural typing.
    This allows protocols to reference retry state without importing from domain.
    """

    gate_attempt: int
    """Current gate attempt number (1-indexed)."""

    session_end_attempt: int
    """Current session_end attempt number."""

    review_attempt: int
    """Current review attempt number."""

    log_offset: int
    """Byte offset in log file for resuming."""

    previous_commit_hash: str | None
    """Commit hash from the previous attempt."""

    baseline_timestamp: int
    """Timestamp for baseline comparisons."""


@runtime_checkable
class IGateRunner(Protocol):
    """Protocol for gate checking operations.

    This protocol defines methods for running quality gate checks and
    session-end validation. It replaces the on_gate_check and on_session_end_check
    callbacks from SessionCallbacks.

    The canonical implementation is SessionCallbackFactory in
    src/pipeline/session_callback_factory.py.
    """

    async def run_gate_check(
        self,
        issue_id: str,
        log_path: Path,
        retry_state: RetryStateProtocol,
    ) -> tuple[GateOutcomeProtocol, int]:
        """Run quality gate check.

        Args:
            issue_id: The issue ID being checked.
            log_path: Path to the JSONL log file from agent session.
            retry_state: Current retry state with attempt counts and log offset.

        Returns:
            Tuple of (GateOutcome indicating pass/fail/retry, new log offset).
        """
        ...

    async def run_session_end_check(
        self,
        issue_id: str,
        log_path: Path,
        retry_state: SessionEndRetryState,
    ) -> SessionEndResult:
        """Run session-end validation check.

        Args:
            issue_id: The issue ID being checked.
            log_path: Path to the JSONL log file from agent session.
            retry_state: Current session-end retry state.

        Returns:
            SessionEndResult with validation outcome.
        """
        ...
