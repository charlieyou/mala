"""Protocol definitions for pipeline stage abstractions.

This module defines Protocol classes that enable dependency injection and
testability for the MalaOrchestrator's pipeline stages. Each protocol
represents a stage boundary that the orchestrator interacts with.

Design principles:
- Protocols use structural typing (typing.Protocol) for flexibility
- Methods match exactly what the orchestrator actually calls
- Result types are defined as local Protocol types to avoid import-time dependencies
- BeadsClient, ReviewRunner, and QualityGate conform to these protocols

Usage:
    These protocols enable:
    1. Mock implementations for unit testing the orchestrator
    2. Alternative implementations (e.g., in-memory issue tracker)
    3. Clear contracts between orchestrator and its dependencies
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from collections.abc import Callable
from pathlib import Path

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
    from types import TracebackType
    from typing import Self


# =============================================================================
# Type Aliases
# =============================================================================
# Centralized type aliases used across multiple modules.
# =============================================================================

# Factory function for creating MCP servers configuration.
# Parameters: agent_id, repo_path, optional emit_lock_event callback
# Returns: Dict mapping server names to server configurations
McpServerFactory = Callable[[str, Path, Callable | None], dict[str, object]]


# =============================================================================
# Local Protocol Types
# =============================================================================
# These Protocol types replace TYPE_CHECKING imports from domain/infra modules
# to satisfy the "Layered Architecture" contract. They define the structural
# shape that protocols.py needs without creating import-time dependencies.
#
# Each Protocol matches only the attributes/methods that protocols.py actually
# uses, following the Interface Segregation Principle.
#
# Note: These protocols use plain attributes (not @property) to be compatible
# with dataclass implementations. Protocol structural typing matches attributes
# regardless of whether they're defined as properties or plain attributes.
# =============================================================================


@runtime_checkable
class JsonlEntryProtocol(Protocol):
    """Protocol for parsed JSONL log entries with byte offset tracking.

    Matches the shape of session_log_parser.JsonlEntry for structural typing.
    """

    data: dict[str, Any]
    """The parsed JSON object from this line."""

    entry: object | None
    """The typed LogEntry if successfully parsed, None otherwise."""

    line_len: int
    """Length of the raw line in bytes (for offset tracking)."""

    offset: int
    """Byte offset where this line started in the file."""


@runtime_checkable
class ValidationSpecProtocol(Protocol):
    """Protocol for validation specification.

    Matches the shape of validation.spec.ValidationSpec for structural typing.
    Only includes attributes/methods that protocols.py method signatures use.
    """

    commands: Sequence[Any]
    """List of validation commands to run."""

    scope: Any
    """The validation scope (per-issue or run-level)."""


@runtime_checkable
class ValidationEvidenceProtocol(Protocol):
    """Protocol for validation evidence from agent runs.

    Matches the shape of quality_gate.ValidationEvidence for structural typing.
    """

    commands_ran: dict[Any, bool]
    """Mapping of CommandKind to whether it ran."""

    failed_commands: list[str]
    """List of validation commands that failed."""

    def has_any_evidence(self) -> bool:
        """Check if any validation command ran."""
        ...

    def to_evidence_dict(self) -> dict[str, bool]:
        """Convert evidence to a serializable dict keyed by CommandKind value."""
        ...


@runtime_checkable
class CommitResultProtocol(Protocol):
    """Protocol for commit existence check results.

    Matches the shape of quality_gate.CommitResult for structural typing.
    """

    exists: bool
    """Whether a matching commit exists."""

    commit_hash: str | None
    """The commit hash if found."""

    message: str | None
    """The commit message if found."""


@runtime_checkable
class IssueResolutionProtocol(Protocol):
    """Protocol for issue resolution records.

    Matches the shape of models.IssueResolution for structural typing.
    """

    outcome: Any
    """The resolution outcome (success, no_change, obsolete, etc.)."""

    rationale: str
    """Explanation for the resolution."""


@runtime_checkable
class GateResultProtocol(Protocol):
    """Protocol for quality gate check results.

    Matches the shape of quality_gate.GateResult for structural typing.
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
class ReviewIssueProtocol(Protocol):
    """Protocol for review issues found during code review.

    Matches the shape of cerberus_review.ReviewIssue for structural typing.
    """

    file: str
    """File path where the issue was found."""

    line_start: int
    """Starting line number."""

    line_end: int
    """Ending line number."""

    priority: int | None
    """Issue priority (0=P0, 1=P1, etc.)."""

    title: str
    """Issue title."""

    body: str
    """Issue body/description."""

    reviewer: str
    """Which reviewer found this issue."""


@runtime_checkable
class ReviewResultProtocol(Protocol):
    """Protocol for code review results.

    Matches the shape of cerberus_review.ReviewResult for structural typing.
    """

    passed: bool
    """Whether the review passed."""

    issues: Sequence[ReviewIssueProtocol]
    """List of issues found during review."""

    parse_error: str | None
    """Parse error message if JSON parsing failed."""

    fatal_error: bool
    """Whether this is a fatal error (should not retry)."""

    review_log_path: Path | None
    """Path to review session logs."""


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

    confidence: float
    """Model confidence in the verdict (0.0 to 1.0)."""

    reasoning: str
    """Explanation of the verification outcome."""


@runtime_checkable
class DeadlockInfoProtocol(Protocol):
    """Protocol for deadlock detection information.

    Matches the shape of domain.deadlock.DeadlockInfo for structural typing.
    """

    cycle: list[str]
    """List of agent IDs forming the deadlock cycle."""

    victim_id: str
    """Agent ID selected to be killed (youngest in cycle)."""

    victim_issue_id: str | None
    """Issue ID the victim was working on."""

    blocked_on: str
    """Lock path the victim was waiting for."""

    blocker_id: str
    """Agent ID holding the lock the victim needs."""

    blocker_issue_id: str | None
    """Issue ID the blocker was working on."""


@runtime_checkable
class LockEventProtocol(Protocol):
    """Protocol for lock events.

    Matches the shape of core.models.LockEvent for structural typing.
    """

    event_type: Any
    """Type of lock event (LockEventType enum value)."""

    agent_id: str
    """ID of the agent that emitted this event."""

    lock_path: str
    """Path to the lock file."""

    timestamp: float
    """Unix timestamp when the event occurred."""


@runtime_checkable
class DeadlockMonitorProtocol(Protocol):
    """Protocol for deadlock monitor.

    Matches the interface of domain.deadlock.DeadlockMonitor for structural typing.
    Only includes the handle_event method used by hooks.
    """

    async def handle_event(self, event: Any) -> Any:  # noqa: ANN401
        """Process a lock event and check for deadlocks.

        Args:
            event: The lock event to process (LockEvent).

        Returns:
            DeadlockInfo if a deadlock is detected, None otherwise.
        """
        ...


@runtime_checkable
class LogProvider(Protocol):
    """Protocol for abstracting SDK log storage and schema.

    Provides methods for accessing session logs without hardcoding filesystem
    paths or Claude SDK's internal log format. This enables:
    - Testing with mock log providers that return synthetic events
    - Future support for remote log storage or SDK API access
    - Isolation from SDK log format changes

    The canonical implementation is FileSystemLogProvider, which reads JSONL
    logs from the Claude SDK's ~/.claude/projects/{encoded-path}/ directory.
    Test implementations can return in-memory events for isolation.

    Methods:
        get_log_path: Get the filesystem path for a session log.
        iter_events: Iterate over typed log entries from a session.
        get_end_offset: Get the byte offset at the end of a log file.
    """

    def get_log_path(self, repo_path: Path, session_id: str) -> Path:
        """Get the filesystem path for a session's log file.

        This method computes the expected log file path based on the repo
        and session. The path may or may not exist yet.

        Args:
            repo_path: Path to the repository the session was run in.
            session_id: Claude SDK session ID (UUID from ResultMessage).

        Returns:
            Path to the JSONL log file.
        """
        ...

    def iter_events(
        self, log_path: Path, offset: int = 0
    ) -> Iterator[JsonlEntryProtocol]:
        """Iterate over parsed JSONL entries from a log file.

        Reads the file starting from the given byte offset and yields
        structured entries. This enables incremental parsing across
        retry attempts.

        Args:
            log_path: Path to the JSONL log file.
            offset: Byte offset to start reading from (default 0).

        Yields:
            JsonlEntryProtocol objects for each successfully parsed JSON line.
            The entry field contains the typed LogEntry if parsing succeeded.

        Note:
            - Lines that fail UTF-8 decoding are silently skipped
            - Empty lines are silently skipped
            - Lines that fail JSON parsing are silently skipped
            - If file doesn't exist, yields nothing
        """
        ...

    def get_end_offset(self, log_path: Path, start_offset: int = 0) -> int:
        """Get the byte offset at the end of a log file.

        This is a lightweight method for getting the current file position.
        Use this when you only need the offset for retry scoping, not the
        parsed entries themselves.

        Args:
            log_path: Path to the JSONL log file.
            start_offset: Byte offset to start from (default 0).

        Returns:
            The byte offset at the end of the file, or start_offset if file
            doesn't exist or can't be read.
        """
        ...

    def extract_bash_commands(self, entry: JsonlEntryProtocol) -> list[tuple[str, str]]:
        """Extract Bash tool_use commands from an entry.

        Args:
            entry: A JsonlEntryProtocol from iter_events.

        Returns:
            List of (tool_id, command) tuples for Bash tool_use blocks.
            Returns empty list if entry is not an assistant message.
        """
        ...

    def extract_tool_results(self, entry: JsonlEntryProtocol) -> list[tuple[str, bool]]:
        """Extract tool_result entries from an entry.

        Args:
            entry: A JsonlEntryProtocol from iter_events.

        Returns:
            List of (tool_use_id, is_error) tuples for tool_result blocks.
            Returns empty list if entry is not a user message.
        """
        ...

    def extract_assistant_text_blocks(self, entry: JsonlEntryProtocol) -> list[str]:
        """Extract text content from assistant message blocks.

        Args:
            entry: A JsonlEntryProtocol from iter_events.

        Returns:
            List of text strings from text blocks in assistant messages.
            Returns empty list if entry is not an assistant message.
        """
        ...


@runtime_checkable
class IssueProvider(Protocol):
    """Protocol for issue tracking operations.

    Provides methods for fetching, claiming, closing, and marking issues.
    The orchestrator uses this to manage issue lifecycle during parallel
    processing.

    The canonical implementation is BeadsClient, which wraps the bd CLI.
    Test implementations can use in-memory state for isolation.

    Methods match BeadsClient's async API exactly so BeadsClient conforms
    to this protocol without adaptation.
    """

    async def get_ready_async(
        self,
        exclude_ids: set[str] | None = None,
        epic_id: str | None = None,
        only_ids: list[str] | None = None,
        suppress_warn_ids: set[str] | None = None,
        prioritize_wip: bool = False,
        focus: bool = True,
        orphans_only: bool = False,
    ) -> list[str]:
        """Get list of ready issue IDs, sorted by priority.

        Args:
            exclude_ids: Set of issue IDs to exclude from results.
            epic_id: Optional epic ID to filter by - only return children.
            only_ids: Optional list of issue IDs to include exclusively.
            suppress_warn_ids: Set of issue IDs to suppress from warnings.
            prioritize_wip: If True, sort in_progress issues first.
            focus: If True, group tasks by parent epic.
            orphans_only: If True, only return issues with no parent epic.

        Returns:
            List of issue IDs sorted by priority (lower = higher priority).
        """
        ...

    async def claim_async(self, issue_id: str) -> bool:
        """Claim an issue by setting status to in_progress.

        Args:
            issue_id: The issue ID to claim.

        Returns:
            True if successfully claimed, False otherwise.
        """
        ...

    async def close_async(self, issue_id: str) -> bool:
        """Close an issue by setting status to closed.

        Args:
            issue_id: The issue ID to close.

        Returns:
            True if successfully closed, False otherwise.
        """
        ...

    async def reopen_issue_async(self, issue_id: str) -> bool:
        """Reopen an issue by setting status to ready.

        Used by deadlock resolution to reset victim issues so they can be
        picked up again after the blocker completes.

        Args:
            issue_id: The issue ID to reopen.

        Returns:
            True if successfully reopened, False otherwise.
        """
        ...

    async def mark_needs_followup_async(
        self, issue_id: str, reason: str, log_path: Path | None = None
    ) -> bool:
        """Mark an issue as needing follow-up.

        Called when the quality gate fails and the issue needs manual
        intervention or a follow-up task.

        Args:
            issue_id: The issue ID to mark.
            reason: Description of why the quality gate failed.
            log_path: Optional path to the JSONL log file from the attempt.

        Returns:
            True if successfully marked, False otherwise.
        """
        ...

    async def add_dependency_async(self, issue_id: str, depends_on_id: str) -> bool:
        """Add a dependency between two issues.

        Creates a "blocks" relationship where depends_on_id blocks issue_id.
        Used by deadlock resolution to record that a victim issue depends on
        the blocker's issue.

        Args:
            issue_id: The issue that depends on another.
            depends_on_id: The issue that blocks issue_id.

        Returns:
            True if dependency added successfully, False otherwise.
        """
        ...

    async def get_issue_description_async(self, issue_id: str) -> str | None:
        """Get the description of an issue.

        Args:
            issue_id: The issue ID to get description for.

        Returns:
            The issue description string, or None if not found.
        """
        ...

    async def close_eligible_epics_async(self) -> bool:
        """Auto-close epics where all children are complete.

        Returns:
            True if any epics were closed, False otherwise.
        """
        ...

    async def commit_issues_async(self) -> bool:
        """Commit .beads/issues.jsonl if it has changes.

        Returns:
            True if commit succeeded, False otherwise.
        """
        ...

    async def reset_async(
        self, issue_id: str, log_path: Path | None = None, error: str | None = None
    ) -> bool:
        """Reset an issue back to ready status.

        Called when an implementation attempt fails and the issue should be
        made available for retry.

        Args:
            issue_id: The issue ID to reset.
            log_path: Optional path to the JSONL log file from the attempt.
            error: Optional error message describing the failure.

        Returns:
            True if successfully reset, False otherwise.
        """
        ...

    async def get_epic_children_async(self, epic_id: str) -> set[str]:
        """Get all child issue IDs of an epic.

        Args:
            epic_id: The epic ID to get children for.

        Returns:
            Set of child issue IDs, or empty set if not found or on error.
        """
        ...

    async def get_parent_epic_async(self, issue_id: str) -> str | None:
        """Get the parent epic ID for an issue.

        Args:
            issue_id: The issue ID to find the parent epic for.

        Returns:
            The parent epic ID, or None if no parent epic exists (orphan).
        """
        ...

    async def create_issue_async(
        self,
        title: str,
        description: str,
        priority: str,
        tags: list[str] | None = None,
        parent_id: str | None = None,
    ) -> str | None:
        """Create a new issue for tracking.

        Used to create tracking issues for low-priority review findings (P2/P3)
        that should be addressed later but don't block the current work.

        Args:
            title: Issue title.
            description: Issue description (supports markdown).
            priority: Priority string (P1, P2, P3, etc.).
            tags: Optional list of tags to apply.
            parent_id: Optional parent epic ID to attach this issue to.

        Returns:
            Created issue ID, or None on failure.
        """
        ...

    async def find_issue_by_tag_async(self, tag: str) -> str | None:
        """Find an existing issue with the given tag.

        Used for deduplication when creating tracking issues.

        Args:
            tag: The tag to search for.

        Returns:
            Issue ID if found, None otherwise.
        """
        ...

    async def update_issue_description_async(
        self, issue_id: str, description: str
    ) -> bool:
        """Update an issue's description.

        Used for appending new findings to existing tracking issues.

        Args:
            issue_id: The issue ID to update.
            description: New description content (replaces existing).

        Returns:
            True if successfully updated, False otherwise.
        """
        ...

    async def update_issue_async(
        self,
        issue_id: str,
        *,
        title: str | None = None,
        priority: str | None = None,
    ) -> bool:
        """Update an issue's title and/or priority.

        Used for updating tracking issues when new findings change
        the count or highest priority.

        Args:
            issue_id: The issue ID to update.
            title: New title (optional).
            priority: New priority string like "P2" (optional).

        Returns:
            True if successfully updated, False otherwise.
        """
        ...

    async def get_blocked_count_async(self) -> int | None:
        """Get count of issues that exist but aren't ready.

        Used by watch mode to report how many issues are blocked on
        dependencies or other conditions.

        Returns:
            Count of blocked issues, or None if unknown/unsupported.
        """
        ...


@runtime_checkable
class CodeReviewer(Protocol):
    """Protocol for code review operations.

    Provides a callable interface for reviewing commits and returning
    structured results. The orchestrator uses this to run post-commit
    code reviews via the Cerberus review-gate CLI.

    The canonical implementation is DefaultReviewer in cerberus_review.py.
    Test implementations can return predetermined results for isolation.
    """

    async def __call__(
        self,
        diff_range: str,
        context_file: Path | None = None,
        timeout: int = 300,
        claude_session_id: str | None = None,
        *,
        commit_shas: Sequence[str] | None = None,
    ) -> ReviewResultProtocol:
        """Run code review on a diff range.

        Args:
            diff_range: Git diff range to review (e.g., "baseline..HEAD").
            context_file: Optional path to file with issue description context.
            timeout: Timeout in seconds for the review operation.
            claude_session_id: Optional Claude session ID for review attribution.
            commit_shas: Optional list of commit SHAs to review directly.
                When provided, reviewers should scope to these commits only.

        Returns:
            ReviewResultProtocol with review outcome. On parse failure,
            returns passed=False with parse_error set.
        """
        ...


@runtime_checkable
class GateChecker(Protocol):
    """Protocol for quality gate checking.

    Provides methods for verifying agent work meets quality requirements.
    The orchestrator uses this after each agent attempt to determine if
    the issue was successfully resolved.

    The canonical implementation is QualityGate, which conforms to this
    protocol. Test implementations can verify specific conditions for isolation.

    Methods match QualityGate's API exactly so QualityGate conforms to this
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
class EpicVerificationModel(Protocol):
    """Protocol for model-agnostic epic verification.

    Provides an interface for verifying whether code changes satisfy
    an epic's acceptance criteria. The initial implementation uses
    Claude via SDK, but this protocol allows swapping to other models
    (Codex, Gemini, local models) without changing the verifier.

    The canonical implementation is ClaudeEpicVerificationModel in
    src/epic_verifier.py. Test implementations can return predetermined
    verdicts for isolation.
    """

    async def verify(
        self,
        epic_criteria: str,
        commit_range: str,
        commit_list: str,
        spec_content: str | None,
    ) -> EpicVerdictProtocol:
        """Verify if the commit scope satisfies the epic's acceptance criteria.

        Args:
            epic_criteria: The epic's acceptance criteria text.
            commit_range: Commit range hint covering child issue commits.
            commit_list: Authoritative list of commit SHAs to inspect.
            spec_content: Optional content of linked spec file.

        Returns:
            Structured verdict with pass/fail and unmet criteria details.
        """
        ...


# =============================================================================
# SDK Client Protocol
# =============================================================================


@runtime_checkable
class SDKClientProtocol(Protocol):
    """Protocol for Claude SDK client abstraction.

    Enables the pipeline layer to use SDK clients without importing
    claude_agent_sdk directly. The canonical implementation is
    ClaudeSDKClient, wrapped by SDKClientFactory in infra.

    This protocol captures the async context manager and streaming
    interface used by AgentSessionRunner.
    """

    async def __aenter__(self) -> Self:
        """Enter async context."""
        ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit async context."""
        ...

    async def query(self, prompt: str, session_id: str | None = None) -> None:
        """Send a query to the agent.

        Args:
            prompt: The prompt text to send.
            session_id: Optional session ID for continuation.
        """
        ...

    def receive_response(self) -> AsyncIterator[object]:
        """Get an async iterator of response messages.

        Returns:
            AsyncIterator yielding AssistantMessage, ResultMessage, etc.
        """
        ...

    async def disconnect(self) -> None:
        """Disconnect the client."""
        ...


@runtime_checkable
class SDKClientFactoryProtocol(Protocol):
    """Protocol for SDK client factory.

    Enables dependency injection of the factory into pipeline components,
    allowing tests to provide mock factories.
    """

    def create(self, options: object) -> SDKClientProtocol:
        """Create a new SDK client with the given options.

        Args:
            options: ClaudeAgentOptions for the client.

        Returns:
            SDKClientProtocol instance.
        """
        ...

    def create_options(
        self,
        *,
        cwd: str,
        permission_mode: str = "bypassPermissions",
        model: str = "opus",
        system_prompt: dict[str, str] | None = None,
        setting_sources: list[str] | None = None,
        mcp_servers: object | None = None,
        disallowed_tools: list[str] | None = None,
        env: dict[str, str] | None = None,
        hooks: dict[str, list[object]] | None = None,
        resume: str | None = None,
    ) -> object:
        """Create SDK options without requiring SDK import in caller.

        Args:
            cwd: Working directory for the agent.
            permission_mode: Permission mode.
            model: Model to use.
            system_prompt: System prompt configuration.
            setting_sources: List of setting sources.
            mcp_servers: List of MCP server configurations.
            disallowed_tools: List of tools to disallow.
            env: Environment variables for the agent.
            hooks: Hook configurations keyed by event type.
            resume: Session ID to resume from. When set, the SDK loads
                the prior conversation context before processing the query.

        Returns:
            ClaudeAgentOptions instance.
        """
        ...

    def create_hook_matcher(
        self,
        matcher: object | None,
        hooks: list[object],
    ) -> object:
        """Create a HookMatcher for SDK hook registration.

        Args:
            matcher: Optional matcher configuration.
            hooks: List of hook callables.

        Returns:
            HookMatcher instance.
        """
        ...

    def with_resume(self, options: object, resume: str | None) -> object:
        """Create a copy of options with a different resume session ID.

        This is used to resume a prior session when retrying after idle timeout
        or review failures. The SDK's resume feature loads the prior conversation
        context before processing the next query.

        Args:
            options: Existing ClaudeAgentOptions to clone.
            resume: Session ID to resume from, or None to start fresh.

        Returns:
            New ClaudeAgentOptions with the resume field set.
        """
        ...


# =============================================================================
# Command Runner Protocols
# =============================================================================


@runtime_checkable
class CommandResultProtocol(Protocol):
    """Protocol for command execution results.

    Matches the interface of src.infra.tools.command_runner.CommandResult
    for structural typing without import-time dependencies.
    """

    ok: bool
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    duration_seconds: float

    def stdout_tail(self, max_chars: int = 800, max_lines: int = 20) -> str:
        """Get truncated stdout.

        Args:
            max_chars: Maximum number of characters.
            max_lines: Maximum number of lines.

        Returns:
            Truncated stdout string.
        """
        ...

    def stderr_tail(self, max_chars: int = 800, max_lines: int = 20) -> str:
        """Get truncated stderr.

        Args:
            max_chars: Maximum number of characters.
            max_lines: Maximum number of lines.

        Returns:
            Truncated stderr string.
        """
        ...


@runtime_checkable
class CommandRunnerPort(Protocol):
    """Protocol for abstracting command execution.

    Enables dependency injection of command runners into domain modules,
    allowing the core layer to define the interface without depending on
    the infrastructure implementation.

    The canonical implementation is CommandRunner in
    src/infra/tools/command_runner.py.
    """

    def run(
        self,
        cmd: list[str] | str,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        use_process_group: bool | None = None,
        shell: bool = False,
        cwd: Path | None = None,
    ) -> CommandResultProtocol:
        """Run a command synchronously.

        Args:
            cmd: Command to run. Can be a list of strings or a shell string.
            env: Environment variables to set (merged with os.environ).
            timeout: Timeout for command execution in seconds.
            use_process_group: Whether to use process group for termination.
            shell: If True, run command through shell.
            cwd: Override working directory for this command.

        Returns:
            CommandResultProtocol with execution details.
        """
        ...

    async def run_async(
        self,
        cmd: list[str] | str,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        use_process_group: bool | None = None,
        shell: bool = False,
        cwd: Path | None = None,
    ) -> CommandResultProtocol:
        """Run a command asynchronously.

        Args:
            cmd: Command to run. Can be a list of strings or a shell string.
            env: Environment variables to set (merged with os.environ).
            timeout: Timeout for command execution in seconds.
            use_process_group: Whether to use process group for termination.
            shell: If True, run command through shell.
            cwd: Override working directory for this command.

        Returns:
            CommandResultProtocol with execution details.
        """
        ...


@runtime_checkable
class EnvConfigPort(Protocol):
    """Protocol for abstracting environment configuration.

    Enables dependency injection of environment config into domain modules,
    allowing the core layer to define the interface without depending on
    the infrastructure implementation.

    The canonical implementation is in src/infra/tools/env.py.
    """

    @property
    def scripts_dir(self) -> Path:
        """Path to the scripts directory (e.g., test-mutex.sh)."""
        ...

    @property
    def cache_dir(self) -> Path:
        """Path to the mala cache directory."""
        ...

    @property
    def lock_dir(self) -> Path:
        """Path to the lock directory for multi-agent coordination."""
        ...

    def find_cerberus_bin_path(self) -> Path | None:
        """Find the cerberus plugin bin directory.

        Returns:
            Path to cerberus bin directory, or None if not found.
        """
        ...


@runtime_checkable
class LockManagerPort(Protocol):
    """Protocol for abstracting file-based locking operations.

    Enables dependency injection of lock managers into domain modules,
    allowing the core layer to define the interface without depending on
    the infrastructure implementation.

    The canonical implementation is in src/infra/tools/locking.py.
    """

    def lock_path(self, filepath: str, repo_namespace: str | None = None) -> Path:
        """Get the lock file path for a given filepath.

        Args:
            filepath: Path to the file to lock.
            repo_namespace: Optional repo namespace for cross-repo disambiguation.

        Returns:
            Path to the lock file.
        """
        ...

    def try_lock(
        self, filepath: str, agent_id: str, repo_namespace: str | None = None
    ) -> bool:
        """Try to acquire a lock without blocking.

        Args:
            filepath: Path to the file to lock.
            agent_id: Identifier of the agent requesting the lock.
            repo_namespace: Optional repo namespace for cross-repo disambiguation.

        Returns:
            True if lock was acquired, False if already held by another agent.
        """
        ...

    def wait_for_lock(
        self,
        filepath: str,
        agent_id: str,
        repo_namespace: str | None = None,
        timeout_seconds: float = 30.0,
        poll_interval_ms: int = 100,
    ) -> bool:
        """Wait for and acquire a lock on a file.

        Args:
            filepath: Path to the file to lock.
            agent_id: Identifier of the agent requesting the lock.
            repo_namespace: Optional repo namespace for cross-repo disambiguation.
            timeout_seconds: Maximum time to wait for the lock in seconds.
            poll_interval_ms: Polling interval in milliseconds.

        Returns:
            True if lock was acquired, False if timeout.
        """
        ...

    def release_lock(
        self, filepath: str, agent_id: str, repo_namespace: str | None = None
    ) -> bool:
        """Release a lock on a file.

        Only releases the lock if it is held by the specified agent_id.
        This prevents accidental or malicious release of locks held by
        other agents.

        Args:
            filepath: Path to the file to unlock.
            agent_id: Identifier of the agent releasing the lock.
            repo_namespace: Optional repo namespace for cross-repo disambiguation.

        Returns:
            True if lock was released, False if lock was not held by agent_id.
        """
        ...


@runtime_checkable
class LoggerPort(Protocol):
    """Protocol for console/terminal logging with colored output.

    Enables dependency injection of loggers into domain modules,
    allowing the core layer to define the interface without depending on
    the infrastructure implementation.

    The canonical implementation is in src/infra/io/log_output/console.py.
    """

    def log(
        self,
        message: str,
        *,
        level: str = "info",
        color: str | None = None,
    ) -> None:
        """Log a message to the console.

        Args:
            message: The message to log.
            level: Log level (e.g., "info", "debug", "error").
            color: Optional color name (e.g., "cyan", "green", "red").
        """
        ...


# =============================================================================
# Event Sink Protocol
# =============================================================================


@dataclass
class EventRunConfig:
    """Configuration snapshot for a run, passed to on_run_started.

    Mirrors the relevant fields from MalaOrchestrator for event reporting.
    """

    repo_path: str
    max_agents: int | None
    timeout_minutes: int | None
    max_issues: int | None
    max_gate_retries: int
    max_review_retries: int
    epic_id: str | None = None
    only_ids: list[str] | None = None
    braintrust_enabled: bool = False
    braintrust_disabled_reason: str | None = None  # e.g., "add BRAINTRUST_API_KEY..."
    review_enabled: bool = True  # Cerberus code review enabled
    review_disabled_reason: str | None = None
    prioritize_wip: bool = False
    orphans_only: bool = False
    cli_args: dict[str, object] | None = None


@runtime_checkable
class MalaEventSink(Protocol):
    """Protocol for receiving orchestrator events.

    Implementations handle presentation (console, logging, metrics) while
    the orchestrator focuses on coordination logic. Each method corresponds
    to a semantic event in the orchestration flow.

    All methods are synchronous and should be non-blocking. Implementations
    that need async behavior should queue events internally.
    """

    # -------------------------------------------------------------------------
    # Run lifecycle
    # -------------------------------------------------------------------------

    def on_run_started(self, config: EventRunConfig) -> None:
        """Called when the orchestrator run begins.

        Args:
            config: Run configuration snapshot.
        """
        ...

    def on_run_completed(
        self,
        success_count: int,
        total_count: int,
        run_validation_passed: bool,
        abort_reason: str | None = None,
    ) -> None:
        """Called when the orchestrator run completes.

        Args:
            success_count: Number of issues completed successfully.
            total_count: Total number of issues processed.
            run_validation_passed: Whether Gate 4 (run-level validation) passed.
            abort_reason: If run was aborted, the reason string.
        """
        ...

    def on_ready_issues(self, issue_ids: list[str]) -> None:
        """Called when ready issues are fetched.

        Args:
            issue_ids: List of issue IDs ready for processing.
        """
        ...

    def on_waiting_for_agents(self, count: int) -> None:
        """Called when waiting for agents to complete.

        Args:
            count: Number of active agents being waited on.
        """
        ...

    def on_no_more_issues(self, reason: str) -> None:
        """Called when there are no more issues to process.

        Args:
            reason: Reason string (e.g., "limit_reached", "none_ready").
        """
        ...

    # -------------------------------------------------------------------------
    # Agent lifecycle
    # -------------------------------------------------------------------------

    def on_agent_started(self, agent_id: str, issue_id: str) -> None:
        """Called when an agent is spawned for an issue.

        Args:
            agent_id: Unique agent identifier.
            issue_id: Issue being worked on.
        """
        ...

    def on_agent_completed(
        self,
        agent_id: str,
        issue_id: str,
        success: bool,
        duration_seconds: float,
        summary: str,
    ) -> None:
        """Called when an agent completes (success or failure).

        Args:
            agent_id: Unique agent identifier.
            issue_id: Issue that was worked on.
            success: Whether the agent succeeded.
            duration_seconds: Total execution time.
            summary: Result summary or error message.
        """
        ...

    def on_claim_failed(self, agent_id: str, issue_id: str) -> None:
        """Called when claiming an issue fails.

        Args:
            agent_id: Agent that attempted the claim.
            issue_id: Issue that could not be claimed.
        """
        ...

    # -------------------------------------------------------------------------
    # SDK message streaming
    # -------------------------------------------------------------------------

    def on_tool_use(
        self,
        agent_id: str,
        tool_name: str,
        description: str = "",
        arguments: dict[str, Any] | None = None,
    ) -> None:
        """Called when an agent invokes a tool.

        Args:
            agent_id: Agent invoking the tool.
            tool_name: Name of the tool being called.
            description: Brief description of the action.
            arguments: Tool arguments (may be truncated for display).
        """
        ...

    def on_agent_text(self, agent_id: str, text: str) -> None:
        """Called when an agent emits text output.

        Args:
            agent_id: Agent emitting text.
            text: Text content (may be truncated for display).
        """
        ...

    # -------------------------------------------------------------------------
    # Quality gate events
    # -------------------------------------------------------------------------

    def on_gate_started(
        self,
        agent_id: str | None,
        attempt: int,
        max_attempts: int,
        issue_id: str | None = None,
    ) -> None:
        """Called when a quality gate check begins.

        Args:
            agent_id: Agent ID (None for run-level gate).
            attempt: Current attempt number (1-indexed).
            max_attempts: Maximum retry attempts.
            issue_id: Issue being validated (for display).
        """
        ...

    def on_gate_passed(
        self,
        agent_id: str | None,
        issue_id: str | None = None,
    ) -> None:
        """Called when a quality gate passes.

        Args:
            agent_id: Agent ID (None for run-level gate).
            issue_id: Issue being validated (for display).
        """
        ...

    def on_gate_failed(
        self,
        agent_id: str | None,
        attempt: int,
        max_attempts: int,
        issue_id: str | None = None,
    ) -> None:
        """Called when a quality gate fails after all retries.

        Args:
            agent_id: Agent ID (None for run-level gate).
            attempt: Final attempt number.
            max_attempts: Maximum retry attempts.
            issue_id: Issue being validated (for display).
        """
        ...

    def on_gate_retry(
        self,
        agent_id: str,
        attempt: int,
        max_attempts: int,
        issue_id: str | None = None,
    ) -> None:
        """Called when retrying a quality gate after failure.

        Args:
            agent_id: Agent being retried.
            attempt: Current retry attempt number.
            max_attempts: Maximum retry attempts.
            issue_id: Issue being validated (for display).
        """
        ...

    def on_gate_result(
        self,
        agent_id: str | None,
        passed: bool,
        failure_reasons: list[str] | None = None,
        issue_id: str | None = None,
    ) -> None:
        """Called when a quality gate check completes with its result.

        This provides the detailed gate result including failure reasons,
        complementing the simpler on_gate_passed/on_gate_failed events.

        Args:
            agent_id: Agent ID (None for run-level gate).
            passed: Whether the gate passed.
            failure_reasons: List of failure reasons (if failed).
            issue_id: Issue being validated (for display).
        """
        ...

    # -------------------------------------------------------------------------
    # Codex review events
    # -------------------------------------------------------------------------

    def on_review_started(
        self,
        agent_id: str,
        attempt: int,
        max_attempts: int,
        issue_id: str | None = None,
    ) -> None:
        """Called when a Codex review begins.

        Args:
            agent_id: Agent being reviewed.
            attempt: Current attempt number.
            max_attempts: Maximum review attempts.
            issue_id: Issue being reviewed (for display).
        """
        ...

    def on_review_passed(
        self,
        agent_id: str,
        issue_id: str | None = None,
    ) -> None:
        """Called when a Codex review passes.

        Args:
            agent_id: Agent that passed review.
            issue_id: Issue being reviewed (for display).
        """
        ...

    def on_review_retry(
        self,
        agent_id: str,
        attempt: int,
        max_attempts: int,
        error_count: int | None = None,
        parse_error: str | None = None,
        issue_id: str | None = None,
    ) -> None:
        """Called when retrying a Codex review after issues found.

        Args:
            agent_id: Agent being retried.
            attempt: Current retry attempt number.
            max_attempts: Maximum review attempts.
            error_count: Number of errors found (if available).
            parse_error: Parse error message (if review failed to parse).
            issue_id: Issue being reviewed (for display).
        """
        ...

    def on_review_warning(
        self,
        message: str,
        agent_id: str | None = None,
        issue_id: str | None = None,
    ) -> None:
        """Called for review-related warnings (e.g., verdict mismatch).

        Args:
            message: Warning message.
            agent_id: Associated agent (if any).
            issue_id: Issue being reviewed (for display).
        """
        ...

    # -------------------------------------------------------------------------
    # Fixer agent events
    # -------------------------------------------------------------------------

    def on_fixer_started(
        self,
        attempt: int,
        max_attempts: int,
    ) -> None:
        """Called when a fixer agent is spawned.

        Args:
            attempt: Current fixer attempt number.
            max_attempts: Maximum fixer attempts.
        """
        ...

    def on_fixer_completed(self, result: str) -> None:
        """Called when a fixer agent completes.

        Args:
            result: Brief result description.
        """
        ...

    def on_fixer_failed(self, reason: str) -> None:
        """Called when a fixer agent fails.

        Args:
            reason: Failure reason (e.g., "timeout", "error").
        """
        ...

    # -------------------------------------------------------------------------
    # Issue lifecycle
    # -------------------------------------------------------------------------

    def on_issue_closed(self, agent_id: str, issue_id: str) -> None:
        """Called when an issue is closed after successful completion.

        Args:
            agent_id: Agent that completed the issue.
            issue_id: Issue that was closed.
        """
        ...

    def on_issue_completed(
        self,
        agent_id: str,
        issue_id: str,
        success: bool,
        duration_seconds: float,
        summary: str,
    ) -> None:
        """Called when an issue implementation completes (success or failure).

        This is the primary issue completion event, distinct from on_agent_completed
        which tracks the agent lifecycle. Use this for issue-level tracking.

        Args:
            agent_id: Agent that worked on the issue.
            issue_id: Issue that was completed.
            success: Whether the issue was successfully implemented.
            duration_seconds: Total time spent on the issue.
            summary: Result summary or error message.
        """
        ...

    def on_epic_closed(self, agent_id: str) -> None:
        """Called when a parent epic is auto-closed.

        Args:
            agent_id: Agent that triggered the epic closure.
        """
        ...

    def on_validation_started(
        self,
        agent_id: str,
        issue_id: str | None = None,
    ) -> None:
        """Called when per-issue validation begins.

        Args:
            agent_id: Agent being validated.
            issue_id: Issue being validated (for display).
        """
        ...

    def on_validation_result(
        self,
        agent_id: str,
        passed: bool,
        issue_id: str | None = None,
    ) -> None:
        """Called when per-issue validation completes.

        Args:
            agent_id: Agent that was validated.
            passed: Whether validation passed.
            issue_id: Issue being validated (for display).
        """
        ...

    def on_validation_step_running(
        self,
        step_name: str,
        agent_id: str | None = None,
    ) -> None:
        """Called when a validation step starts.

        Args:
            step_name: Name of the validation step (e.g., "ruff", "pytest").
            agent_id: Associated agent (if any).
        """
        ...

    def on_validation_step_skipped(
        self,
        step_name: str,
        reason: str,
        agent_id: str | None = None,
    ) -> None:
        """Called when a validation step is skipped.

        Args:
            step_name: Name of the validation step.
            reason: Reason for skipping (e.g., "cache hit", "no changes").
            agent_id: Associated agent (if any).
        """
        ...

    def on_validation_step_passed(
        self,
        step_name: str,
        duration_seconds: float,
        agent_id: str | None = None,
    ) -> None:
        """Called when a validation step succeeds.

        Args:
            step_name: Name of the validation step.
            duration_seconds: Time taken to complete the step.
            agent_id: Associated agent (if any).
        """
        ...

    def on_validation_step_failed(
        self,
        step_name: str,
        exit_code: int,
        agent_id: str | None = None,
    ) -> None:
        """Called when a validation step fails.

        Args:
            step_name: Name of the validation step.
            exit_code: Exit code from the step.
            agent_id: Associated agent (if any).
        """
        ...

    # -------------------------------------------------------------------------
    # Warnings and diagnostics
    # -------------------------------------------------------------------------

    def on_warning(self, message: str, agent_id: str | None = None) -> None:
        """Called for warning conditions.

        Args:
            message: Warning message.
            agent_id: Associated agent (if any).
        """
        ...

    def on_log_timeout(self, agent_id: str, log_path: str) -> None:
        """Called when waiting for a log file times out.

        Args:
            agent_id: Agent waiting for the log.
            log_path: Path to the missing log file.
        """
        ...

    def on_locks_cleaned(self, agent_id: str, count: int) -> None:
        """Called when stale locks are cleaned up for an agent.

        Args:
            agent_id: Agent whose locks were cleaned.
            count: Number of locks cleaned.
        """
        ...

    def on_locks_released(self, count: int) -> None:
        """Called when remaining locks are released at run end.

        Args:
            count: Number of locks released.
        """
        ...

    def on_issues_committed(self) -> None:
        """Called when .beads/issues.jsonl is committed."""
        ...

    def on_run_metadata_saved(self, path: str) -> None:
        """Called when run metadata is saved.

        Args:
            path: Path to the saved metadata file.
        """
        ...

    def on_run_level_validation_disabled(self) -> None:
        """Called when run-level validation is disabled."""
        ...

    def on_abort_requested(self, reason: str) -> None:
        """Called when a fatal error triggers a run abort.

        Args:
            reason: Description of the fatal error.
        """
        ...

    def on_tasks_aborting(self, count: int, reason: str) -> None:
        """Called when active tasks are being aborted.

        Args:
            count: Number of active tasks being aborted.
            reason: Reason for the abort.
        """
        ...

    # -------------------------------------------------------------------------
    # Epic verification lifecycle
    # -------------------------------------------------------------------------

    def on_epic_verification_started(self, epic_id: str) -> None:
        """Called when epic verification begins.

        Args:
            epic_id: The epic being verified.
        """
        ...

    def on_epic_verification_passed(self, epic_id: str, confidence: float) -> None:
        """Called when epic verification passes.

        Args:
            epic_id: The epic that passed verification.
            confidence: Confidence score (0.0 to 1.0).
        """
        ...

    def on_epic_verification_failed(
        self,
        epic_id: str,
        unmet_count: int,
        remediation_ids: list[str],
    ) -> None:
        """Called when epic verification fails with unmet criteria.

        Args:
            epic_id: The epic that failed verification.
            unmet_count: Number of unmet criteria.
            remediation_ids: IDs of created remediation issues.
        """
        ...

    def on_epic_remediation_created(
        self,
        epic_id: str,
        issue_id: str,
        criterion: str,
    ) -> None:
        """Called when a remediation issue is created for an unmet criterion.

        Args:
            epic_id: The epic the remediation is for.
            issue_id: The created issue ID.
            criterion: The unmet criterion text (may be truncated).
        """
        ...

    # -------------------------------------------------------------------------
    # Pipeline module events
    # -------------------------------------------------------------------------

    def on_lifecycle_state(self, agent_id: str, state: str) -> None:
        """Called when lifecycle state changes (verbose/debug).

        Args:
            agent_id: Agent whose lifecycle changed.
            state: New lifecycle state name.
        """
        ...

    def on_log_waiting(self, agent_id: str) -> None:
        """Called when waiting for session log file.

        Args:
            agent_id: Agent waiting for log.
        """
        ...

    def on_log_ready(self, agent_id: str) -> None:
        """Called when session log file is ready.

        Args:
            agent_id: Agent whose log is ready.
        """
        ...

    def on_review_skipped_no_progress(self, agent_id: str) -> None:
        """Called when review is skipped due to no progress.

        Args:
            agent_id: Agent whose review was skipped.
        """
        ...

    def on_fixer_text(self, attempt: int, text: str) -> None:
        """Called when fixer agent emits text output.

        Args:
            attempt: Current fixer attempt number.
            text: Text content.
        """
        ...

    def on_fixer_tool_use(
        self,
        attempt: int,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> None:
        """Called when fixer agent invokes a tool.

        Args:
            attempt: Current fixer attempt number.
            tool_name: Name of the tool being called.
            arguments: Tool arguments.
        """
        ...

    def on_deadlock_detected(self, info: DeadlockInfoProtocol) -> None:
        """Called when a deadlock is detected and resolved.

        Args:
            info: Information about the detected deadlock, including the cycle
                of agents, the victim selected for cancellation, and the
                blocker holding the needed resource.
        """
        ...

    def on_watch_idle(self, wait_seconds: float, issues_blocked: int | None) -> None:
        """Called when watch mode enters idle sleep.

        Args:
            wait_seconds: Duration of the upcoming sleep.
            issues_blocked: Count of blocked issues, or None if unknown.
        """
        ...
