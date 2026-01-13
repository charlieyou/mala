"""Quality gate for verifying agent work before marking success.

Implements Track A4 from 2025-12-26-coordination-plan.md:
- Verify commit message contains bd-<issue_id>
- Verify validation commands ran (parse JSONL logs)
- On failure: mark needs-followup with failure context

Evidence Detection:
    Production code should use parse_validation_evidence_with_spec() or
    check_with_resolution(..., spec=spec) to derive detection patterns from
    the ValidationSpec. This ensures spec command changes automatically update
    evidence expectations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from .validation.spec import (
    CommandKind,
    IssueResolution,
    ResolutionOutcome,
    ValidationScope,
    build_validation_spec,
)
from .validation.validation_gating import should_trigger_validation

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from src.core.protocols.infra import CommandRunnerPort
    from src.core.protocols.issue import IssueResolutionProtocol
    from src.core.protocols.log import JsonlEntryProtocol, LogProvider
    from src.core.protocols.validation import ValidationEvidenceProtocol

    from .validation.spec import ValidationSpec


__all__ = [
    "CommitResult",
    "EvidenceCheck",
    "GateResult",
    "ValidationEvidence",
]

# Command kinds that should not be required by the quality gate.
# SETUP commands like `uv sync` are useful for local setup, but should not
# block gate passing if omitted or failed.
EVIDENCE_CHECK_IGNORED_KINDS: set[CommandKind] = {CommandKind.SETUP}

# Regex for parsing custom command markers from tool_result content.
# Matches: [custom:<name>:start], [custom:<name>:pass],
#          [custom:<name>:fail exit=<code>], [custom:<name>:timeout]
# Group 1: command name (alphanumeric, underscore, hyphen - valid YAML dict keys)
# Group 2: marker type (start|pass|fail exit=\d+|timeout)
CUSTOM_MARKER_PATTERN = re.compile(
    r"\[custom:([\w-]+):(start|pass|fail exit=\d+|timeout)\]"
)


@dataclass
class ValidationEvidence:
    """Evidence of validation commands executed during agent run.

    This class is spec-driven: evidence is stored by CommandKind rather than
    using hardcoded tool-specific boolean flags. This allows adding new
    validation commands without code changes to the evidence structure.

    Backward Compatibility:
        Properties like `pytest_ran`, `ruff_check_ran`, etc. are provided
        for backward compatibility with existing code that references these
        directly. Internally, all evidence is stored in `commands_ran`.
    """

    # Spec-driven evidence storage: CommandKind -> ran boolean
    commands_ran: dict[CommandKind, bool] = field(default_factory=dict)

    # Track which validation commands failed (exited non-zero)
    failed_commands: list[str] = field(default_factory=list)

    # Custom command evidence: command_name -> ran/failed boolean
    custom_commands_ran: dict[str, bool] = field(default_factory=dict)
    custom_commands_failed: dict[str, bool] = field(default_factory=dict)

    # Backward-compatible properties for external consumers
    @property
    def pytest_ran(self) -> bool:
        """Whether pytest (TEST) command ran."""
        return self.commands_ran.get(CommandKind.TEST, False)

    @pytest_ran.setter
    def pytest_ran(self, value: bool) -> None:
        """Set pytest (TEST) evidence."""
        self.commands_ran[CommandKind.TEST] = value

    @property
    def ruff_check_ran(self) -> bool:
        """Whether ruff check (LINT) command ran."""
        return self.commands_ran.get(CommandKind.LINT, False)

    @ruff_check_ran.setter
    def ruff_check_ran(self, value: bool) -> None:
        """Set ruff check (LINT) evidence."""
        self.commands_ran[CommandKind.LINT] = value

    @property
    def ruff_format_ran(self) -> bool:
        """Whether ruff format (FORMAT) command ran."""
        return self.commands_ran.get(CommandKind.FORMAT, False)

    @ruff_format_ran.setter
    def ruff_format_ran(self, value: bool) -> None:
        """Set ruff format (FORMAT) evidence."""
        self.commands_ran[CommandKind.FORMAT] = value

    @property
    def ty_check_ran(self) -> bool:
        """Whether ty check (TYPECHECK) command ran."""
        return self.commands_ran.get(CommandKind.TYPECHECK, False)

    @ty_check_ran.setter
    def ty_check_ran(self, value: bool) -> None:
        """Set ty check (TYPECHECK) evidence."""
        self.commands_ran[CommandKind.TYPECHECK] = value

    def has_any_evidence(self) -> bool:
        """Check if any validation command ran.

        Used for progress detection to determine if new validation
        activity occurred since the last check. Includes both built-in
        commands (via commands_ran) and custom commands (via custom_commands_ran).
        """
        return any(self.commands_ran.values()) or any(self.custom_commands_ran.values())

    def has_minimum_validation(self) -> bool:
        """Check if minimum required validation was performed.

        Requires the full validation suite:
        - pytest (run tests)
        - ruff check (lint)
        - ruff format (format)
        - ty check (type check)
        """
        return (
            self.pytest_ran
            and self.ruff_check_ran
            and self.ruff_format_ran
            and self.ty_check_ran
        )

    def missing_commands(self) -> list[str]:
        """List validation commands that didn't run."""
        missing = []
        if not self.pytest_ran:
            missing.append("pytest")
        if not self.ruff_check_ran:
            missing.append("ruff check")
        if not self.ruff_format_ran:
            missing.append("ruff format")
        if not self.ty_check_ran:
            missing.append("ty check")
        return missing

    def to_evidence_dict(self) -> dict[str, bool]:
        """Convert evidence to a serializable dict keyed by CommandKind value.

        This is the spec-driven alternative to accessing individual properties.
        Returns a dict with keys like "test", "lint", "format", "typecheck"
        based on what commands were detected.

        Use this method when building metadata to avoid hardcoded property access.

        Returns:
            Dict mapping CommandKind.value strings to their ran status.
        """
        return {kind.value: ran for kind, ran in self.commands_ran.items()}


def get_required_evidence_kinds(spec: ValidationSpec) -> set[CommandKind]:
    """Get the set of command kinds required by a ValidationSpec.

    This derives the expected evidence from the spec, ensuring scope-aware
    evidence requirements. For example, per-session scope specs won't have
    E2E commands, so E2E evidence won't be required.

    Args:
        spec: The ValidationSpec to extract requirements from.

    Returns:
        Set of CommandKind values that must have evidence.
    """
    return {
        cmd.kind
        for cmd in spec.commands
        if cmd.kind not in EVIDENCE_CHECK_IGNORED_KINDS
    }


def check_evidence_against_spec(
    evidence: ValidationEvidence, spec: ValidationSpec
) -> tuple[bool, list[str], list[str]]:
    """Check if evidence satisfies a ValidationSpec's requirements.

    This is fully spec-driven: evidence requirements are controlled by
    spec.evidence_required. If evidence_required is empty, no evidence
    checking is performed (gate passes immediately). Otherwise, only
    commands whose names are in evidence_required are checked.

    This is scope-aware: a per-session spec won't require E2E evidence because
    per-session specs don't include E2E commands in evidence_required.

    For custom commands (CommandKind.CUSTOM), checks:
    - "not run" (no markers for spec'd command) → missing
    - "ran and passed" → OK
    - "ran and failed" with allow_fail=True → advisory failure (doesn't block gate)
    - "ran and failed" with allow_fail=False → strict failure (blocks gate)

    Args:
        evidence: The parsed validation evidence.
        spec: The ValidationSpec defining what's required.

    Returns:
        Tuple of (passed, missing_commands, failed_strict) where:
        - missing_commands lists commands that didn't run
        - failed_strict lists strict (allow_fail=False) custom commands that failed
    """
    # Early return if no evidence required (BREAKING CHANGE: empty means pass)
    if not spec.evidence_required:
        return (True, [], [])

    # Build set of required command names for O(1) lookup
    required_keys = set(spec.evidence_required)

    missing: list[str] = []
    failed_strict: list[str] = []

    # Build kind-to-name mapping from spec (spec-driven display names)
    # Also track all command names present in spec.commands
    kind_to_name: dict[CommandKind, str] = {}
    spec_command_names: set[str] = set()
    for cmd in spec.commands:
        spec_command_names.add(cmd.name)
        # Use first command name for each kind as the display name
        if cmd.kind not in kind_to_name:
            kind_to_name[cmd.kind] = cmd.name

    # Check each required kind from the spec (non-CUSTOM kinds)
    # Only check kinds whose display names are in evidence_required
    for kind in get_required_evidence_kinds(spec):
        if kind == CommandKind.CUSTOM:
            # Custom commands are checked individually by name below
            continue
        name = kind_to_name.get(kind, kind.value)
        if name not in required_keys:
            # This command is not in evidence_required, skip it
            continue
        ran = evidence.commands_ran.get(kind, False)
        if not ran:
            missing.append(name)

    # Check custom commands individually by name
    # Only check commands whose names are in evidence_required
    for cmd in spec.commands:
        if cmd.kind != CommandKind.CUSTOM:
            continue
        name = cmd.name
        if name not in required_keys:
            # This command is not in evidence_required, skip it
            continue
        ran = evidence.custom_commands_ran.get(name, False)
        if not ran:
            missing.append(name)
        elif evidence.custom_commands_failed.get(name, False):
            # Command ran but failed
            if not cmd.allow_fail:
                # Strict failure - blocks gate
                failed_strict.append(name)
            # Advisory failure (allow_fail=True) - doesn't block gate, just noted

    # Check for required keys that have no corresponding command in spec.commands
    # These are treated as missing evidence (config bug or stale config)
    for key in required_keys:
        if key not in spec_command_names:
            missing.append(key)

    passed = len(missing) == 0 and len(failed_strict) == 0
    return passed, missing, failed_strict


@dataclass
class CommitResult:
    """Result of checking for a matching commit."""

    exists: bool
    commit_hash: str | None = None
    message: str | None = None


@dataclass
class GateResult:
    """Result of quality gate check."""

    passed: bool
    failure_reasons: list[str] = field(default_factory=list)
    commit_hash: str | None = None
    validation_evidence: ValidationEvidence | ValidationEvidenceProtocol | None = None
    no_progress: bool = False
    resolution: IssueResolution | IssueResolutionProtocol | None = None


class EvidenceCheck:
    """Quality gate for verifying agent work meets requirements.

    Uses LogProvider for JSONL log parsing, keeping this class
    focused on policy checking and validation logic.
    """

    # Patterns for detecting issue resolution markers in log text
    RESOLUTION_PATTERNS: ClassVar[dict[str, re.Pattern[str]]] = {
        "no_change": re.compile(r"ISSUE_NO_CHANGE:\s*(.*)$", re.MULTILINE),
        "obsolete": re.compile(r"ISSUE_OBSOLETE:\s*(.*)$", re.MULTILINE),
        "already_complete": re.compile(
            r"ISSUE_ALREADY_COMPLETE:\s*(.*)$", re.MULTILINE
        ),
        "docs_only": re.compile(r"ISSUE_DOCS_ONLY:\s*(.*)$", re.MULTILINE),
    }

    # Map pattern names to resolution outcomes
    PATTERN_TO_OUTCOME: ClassVar[dict[str, ResolutionOutcome]] = {
        "no_change": ResolutionOutcome.NO_CHANGE,
        "obsolete": ResolutionOutcome.OBSOLETE,
        "already_complete": ResolutionOutcome.ALREADY_COMPLETE,
        "docs_only": ResolutionOutcome.DOCS_ONLY,
    }

    # Pattern to extract issue ID from ALREADY_COMPLETE rationale
    # Matches: "bd-issue-123", "bd-mala-xyz", etc. in rationale text
    RATIONALE_ISSUE_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"\bbd-([a-zA-Z0-9_-]+)\b"
    )

    def __init__(
        self,
        repo_path: Path,
        log_provider: LogProvider,
        command_runner: CommandRunnerPort,
    ):
        """Initialize quality gate.

        Args:
            repo_path: Path to the repository for git operations.
            log_provider: LogProvider for reading session logs.
            command_runner: CommandRunnerPort for running git commands.
        """
        self.repo_path = repo_path
        self._log_provider = log_provider
        self._command_runner = command_runner

    def _match_resolution_pattern(self, text: str) -> IssueResolution | None:
        """Check text against all resolution patterns.

        Args:
            text: Text content to search for patterns.

        Returns:
            IssueResolution if a pattern matches, None otherwise.
        """
        for name, pattern in self.RESOLUTION_PATTERNS.items():
            match = pattern.search(text)
            if match:
                return IssueResolution(
                    outcome=self.PATTERN_TO_OUTCOME[name],
                    rationale=match.group(1).strip(),
                )
        return None

    def _match_spec_pattern_with_kinds(
        self,
        command: str,
        evidence: ValidationEvidence,
        kind_patterns: dict[CommandKind, list[re.Pattern[str]]],
    ) -> list[CommandKind]:
        """Check command against spec-defined patterns and return all matched kinds.

        A command may match multiple kinds (e.g., "ruff" matches both LINT and FORMAT
        patterns). This method returns all matching kinds for proper evidence tracking.

        Args:
            command: The bash command string.
            evidence: ValidationEvidence to update.
            kind_patterns: Mapping of CommandKind to detection patterns.

        Returns:
            List of matched CommandKinds (may be empty if no match).
        """
        matched_kinds: list[CommandKind] = []
        for kind, patterns in kind_patterns.items():
            for pattern in patterns:
                if pattern.search(command):
                    # Spec-driven: record any CommandKind directly
                    evidence.commands_ran[kind] = True
                    matched_kinds.append(kind)
                    break  # Found match for this kind, try next kind
        return matched_kinds

    def _build_spec_patterns(
        self, spec: ValidationSpec
    ) -> dict[CommandKind, list[re.Pattern[str]]]:
        """Build pattern mapping from a ValidationSpec.

        Args:
            spec: The ValidationSpec defining commands and their detection patterns.

        Returns:
            Mapping of CommandKind to list of detection patterns.
        """
        kind_patterns: dict[CommandKind, list[re.Pattern[str]]] = {}
        for cmd in spec.commands:
            if cmd.kind not in kind_patterns:
                kind_patterns[cmd.kind] = []
            if cmd.detection_pattern is not None:
                kind_patterns[cmd.kind].append(cmd.detection_pattern)
        return kind_patterns

    def _parse_custom_markers(
        self,
        content: str,
        state: dict[str, tuple[bool, str | None]],
    ) -> None:
        """Parse custom command markers from tool result content.

        Updates state in-place to track marker occurrences. For each command name,
        tracks whether a start marker was seen and the latest terminal marker.

        A new `start` marker resets the terminal state to None, so incomplete
        retries (start without terminal) are correctly treated as failures.
        Latest terminal marker wins within a single attempt.

        Args:
            content: Tool result content to scan for markers.
            state: Mutable dict tracking (has_start, latest_terminal) per command name.
        """
        for match in CUSTOM_MARKER_PATTERN.finditer(content):
            name = match.group(1)
            marker_type = match.group(2)

            # Get or initialize state for this command
            has_start, terminal = state.get(name, (False, None))

            if marker_type == "start":
                # New attempt: reset terminal state so incomplete retry is detected
                has_start = True
                terminal = None
            else:
                # Terminal marker: pass, fail exit=N, timeout
                # Latest wins (allows retries to override earlier failures)
                terminal = marker_type

            state[name] = (has_start, terminal)

    def _iter_jsonl_entries(
        self, log_path: Path, offset: int = 0
    ) -> Iterator[JsonlEntryProtocol]:
        """Iterate over parsed JSONL entries from a log file.

        Delegates to LogProvider.iter_events().

        Args:
            log_path: Path to the JSONL log file.
            offset: Byte offset to start reading from (default 0).

        Yields:
            JsonlEntryProtocol objects for each successfully parsed JSON line.
        """
        return self._log_provider.iter_events(log_path, offset)

    def parse_issue_resolution(self, log_path: Path) -> IssueResolution | None:
        """Parse JSONL log file for issue resolution markers.

        Looks for ISSUE_NO_CHANGE or ISSUE_OBSOLETE markers with rationale.

        Args:
            log_path: Path to the JSONL log file from agent session.

        Returns:
            IssueResolution if a marker was found, None otherwise.
        """
        resolution, _ = self.parse_issue_resolution_from_offset(log_path, offset=0)
        return resolution

    def parse_issue_resolution_from_offset(
        self, log_path: Path, offset: int = 0
    ) -> tuple[IssueResolution | None, int]:
        """Parse JSONL log file for issue resolution markers starting at offset.

        Only parses assistant messages to prevent user prompts from triggering
        resolution markers.

        Args:
            log_path: Path to the JSONL log file from agent session.
            offset: Byte offset to start reading from (default 0 = beginning).

        Returns:
            Tuple of (IssueResolution or None, new_offset).
        """
        if not log_path.exists():
            return None, 0

        try:
            for entry in self._iter_jsonl_entries(log_path, offset):
                for text in self._log_provider.extract_assistant_text_blocks(entry):
                    resolution = self._match_resolution_pattern(text)
                    if resolution:
                        return resolution, entry.offset + entry.line_len
            # No match found - return EOF position (matches original f.tell())
            return None, self.get_log_end_offset(log_path, offset)
        except OSError:
            return None, 0

    def check_working_tree_clean(self) -> tuple[bool, str]:
        """Check if the git working tree is clean (no uncommitted changes).

        Returns:
            Tuple of (is_clean, status_output). On git failure, returns
            (False, error_message) to treat unknown state as dirty.
        """
        result = self._command_runner.run(["git", "status", "--porcelain"])
        # Treat git failures as dirty/unknown state
        if not result.ok:
            error_msg = result.stderr.strip() or "git status failed"
            return False, f"git error: {error_msg}"
        output = result.stdout.strip()
        return len(output) == 0, output

    def parse_validation_evidence_with_spec(
        self, log_path: Path, spec: ValidationSpec, offset: int = 0
    ) -> ValidationEvidence:
        """Parse JSONL log for validation evidence using spec-defined patterns."""
        evidence = ValidationEvidence()
        if not log_path.exists():
            return evidence

        kind_patterns = self._build_spec_patterns(spec)
        # Track tool_id → list of (CommandKind, display_name) for proper failure tracking
        # A command may match multiple kinds (e.g., "ruff" matches LINT and FORMAT)
        tool_id_to_info: dict[str, list[tuple[CommandKind, str]]] = {}
        # Track failures per CommandKind (latest status wins for retries of same command)
        kind_failed: dict[CommandKind, tuple[bool, str]] = {}
        # Track custom command markers: name → (has_start, latest_terminal_marker)
        # Terminal markers: "pass", "fail exit=N", "timeout"
        # None means no terminal marker seen yet
        custom_marker_state: dict[str, tuple[bool, str | None]] = {}

        for entry in self._iter_jsonl_entries(log_path, offset):
            for tool_id, command in self._log_provider.extract_bash_commands(entry):
                matched_kinds = self._match_spec_pattern_with_kinds(
                    command, evidence, kind_patterns
                )
                if matched_kinds:
                    # Store full command for display in failure messages
                    tool_id_to_info[tool_id] = [
                        (kind, command) for kind in matched_kinds
                    ]
            for tool_use_id, is_error in self._log_provider.extract_tool_results(entry):
                if tool_use_id in tool_id_to_info:
                    for kind, full_cmd in tool_id_to_info[tool_use_id]:
                        # Latest status for this CommandKind wins (allows retries to succeed)
                        kind_failed[kind] = (is_error, full_cmd)

            # Extract tool result content for custom command marker parsing
            for (
                _tool_use_id,
                content,
            ) in self._log_provider.extract_tool_result_content(entry):
                self._parse_custom_markers(content, custom_marker_state)

        # Build failed_commands from kinds that failed, using full command strings
        # Filter out ignored kinds (e.g., SETUP) so they don't block the gate
        # Filter out CUSTOM kinds - custom command failures use marker/allow_fail path
        # Deduplicate: multiple kinds (LINT, FORMAT) may map to the same command
        evidence.failed_commands = list(
            dict.fromkeys(
                full_cmd
                for kind, (is_failed, full_cmd) in kind_failed.items()
                if is_failed
                and kind not in EVIDENCE_CHECK_IGNORED_KINDS
                and kind != CommandKind.CUSTOM
            )
        )

        # Populate custom command evidence from marker state
        # A command "ran" if it has any terminal marker OR has start-only (which is a failure)
        # A command "failed" if terminal marker is fail/timeout OR has start-only
        for name, (has_start, terminal) in custom_marker_state.items():
            if terminal is not None:
                # Terminal marker present: command ran
                evidence.custom_commands_ran[name] = True
                # Check if terminal marker indicates failure
                evidence.custom_commands_failed[name] = (
                    terminal.startswith("fail") or terminal == "timeout"
                )
            elif has_start:
                # Start-only without terminal: command ran but failed (incomplete)
                evidence.custom_commands_ran[name] = True
                evidence.custom_commands_failed[name] = True

        return evidence

    def get_log_end_offset(self, log_path: Path, start_offset: int = 0) -> int:
        """Get the byte offset at the end of a log file.

        Delegates to LogProvider.get_end_offset().

        Args:
            log_path: Path to the JSONL log file.
            start_offset: Byte offset to start from (default 0).

        Returns:
            The byte offset at the end of the file, or start_offset if file
            doesn't exist or can't be read.
        """
        return self._log_provider.get_end_offset(log_path, start_offset)

    def check_no_progress(
        self,
        log_path: Path,
        log_offset: int,
        previous_commit_hash: str | None,
        current_commit_hash: str | None,
        spec: ValidationSpec | None = None,
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
            previous_commit_hash: Commit hash from the previous attempt (None if no commit).
            current_commit_hash: Commit hash from this attempt (None if no commit).
            spec: Optional ValidationSpec for spec-driven evidence detection.
                If not provided, builds a default per-session spec.
            check_validation_evidence: If True (default), also check for new validation
                evidence. Set to False for review retries where only commit/working-tree
                changes should gate progress.

        Returns:
            True if no progress was made, False if progress was detected.
        """
        # Check if commit changed
        commit_changed = previous_commit_hash != current_commit_hash

        # A new commit from None is progress (first successful commit)
        if previous_commit_hash is None and current_commit_hash is not None:
            return False

        # If commit changed, that's progress
        if commit_changed:
            return False

        # Check for uncommitted working tree changes
        if self._has_working_tree_changes():
            return False

        # Skip validation evidence check if not requested (for review retries)
        if not check_validation_evidence:
            # No commit change and no working tree changes = no progress
            return True

        # Build default spec if not provided
        # Note: We don't pass repo_path here to ensure Python validation commands
        # are always included for progress detection. The spec-driven parsing
        # ensures consistency with the production evidence parsing patterns.
        if spec is None:
            spec = build_validation_spec(
                self.repo_path,
                scope=ValidationScope.PER_SESSION,
            )

        # Check for new validation evidence after the offset using spec-driven parsing
        evidence = self.parse_validation_evidence_with_spec(log_path, spec, log_offset)

        # Any new validation evidence counts as progress (spec-driven)
        if evidence.has_any_evidence():
            return False

        # No commit change, no working tree changes, and no new evidence = no progress
        return True

    def _has_working_tree_changes(self) -> bool:
        """Check if the working tree has uncommitted changes.

        Returns:
            True if there are staged or unstaged changes, or if git status
            fails (conservative assumption that changes may exist).
        """
        # Use git status --porcelain to detect any changes
        # This includes staged, unstaged, and untracked files
        result = self._command_runner.run(["git", "status", "--porcelain"], timeout=5.0)
        if not result.ok:
            # If git status fails, assume changes exist (conservative default)
            # This prevents false "no progress" conclusions when git state is unknown
            return True

        # Any output means there are changes
        return bool(result.stdout.strip())

    def extract_issue_from_rationale(self, rationale: str) -> str | None:
        """Extract issue ID from ALREADY_COMPLETE rationale.

        For duplicate issues, the agent may reference a different issue ID
        in the rationale (e.g., "Work committed in 238e17f (bd-mala-xyz: ...)").
        This extracts that referenced issue ID so we can verify the correct commit.

        Args:
            rationale: The rationale text from ALREADY_COMPLETE resolution.

        Returns:
            The extracted issue ID (without bd- prefix), or None if not found.
        """
        match = self.RATIONALE_ISSUE_PATTERN.search(rationale)
        if match:
            return match.group(1)
        return None

    def check_commit_exists(
        self, issue_id: str, baseline_timestamp: int | None = None
    ) -> CommitResult:
        """Check if a commit with bd-<issue_id> exists in recent history.

        Searches commits from the last 30 days to accommodate long-running
        work that may span multiple days.

        Args:
            issue_id: The issue ID to search for (without bd- prefix).
            baseline_timestamp: Unix timestamp. If provided, only accepts commits
                created after this time (to reject stale commits from previous runs).

        Returns:
            CommitResult indicating whether a matching commit exists.
        """
        # Search for commits with bd-<issue_id> in the message
        # Use git log with grep to find matching commits
        pattern = f"bd-{issue_id}"

        # Include commit timestamp in format for baseline comparison
        format_str = "%h %ct %s" if baseline_timestamp is not None else "%h %s"

        result = self._command_runner.run(
            [
                "git",
                "log",
                f"--format={format_str}",
                "--grep",
                pattern,
                "-n",
                "1",
                "--since=30 days ago",
            ]
        )

        if not result.ok:
            return CommitResult(exists=False)

        output = result.stdout.strip()
        if not output:
            return CommitResult(exists=False)

        # Parse the output based on format
        if baseline_timestamp is not None:
            # Format: "hash timestamp message"
            parts = output.split(" ", 2)
            if len(parts) < 2:
                return CommitResult(exists=False)

            commit_hash = parts[0]
            try:
                commit_timestamp = int(parts[1])
            except ValueError:
                return CommitResult(exists=False)

            message = parts[2] if len(parts) > 2 else None

            # Reject commits created before the baseline
            if commit_timestamp < baseline_timestamp:
                return CommitResult(exists=False)

            return CommitResult(
                exists=True,
                commit_hash=commit_hash,
                message=message,
            )
        else:
            # Original format: "hash message"
            parts = output.split(" ", 1)
            commit_hash = parts[0] if parts else None
            message = parts[1] if len(parts) > 1 else None

            return CommitResult(
                exists=True,
                commit_hash=commit_hash,
                message=message,
            )

    def get_commit_files(self, commit_hash: str) -> list[str] | None:
        """Get list of files changed in a commit.

        Args:
            commit_hash: The commit hash to get files from.

        Returns:
            List of file paths changed in the commit (relative to repo root).
            None if git command fails (callers should treat as error).
            Empty list if commit has no file changes.
        """
        # Use -m to show file changes for merge commits (otherwise empty)
        result = self._command_runner.run(
            [
                "git",
                "diff-tree",
                "-m",
                "--no-commit-id",
                "--name-only",
                "-r",
                commit_hash,
            ]
        )

        if not result.ok:
            return None

        output = result.stdout.strip()
        if not output:
            return []

        return output.split("\n")

    def check_with_resolution(
        self,
        issue_id: str,
        log_path: Path,
        baseline_timestamp: int | None = None,
        log_offset: int = 0,
        spec: ValidationSpec | None = None,
    ) -> GateResult:
        """Run quality gate check with support for no-op/obsolete resolutions.

        This method is scope-aware and handles special resolution outcomes:
        - ISSUE_NO_CHANGE: Issue already addressed, no commit needed
        - ISSUE_OBSOLETE: Issue no longer relevant, no commit needed
        - ISSUE_ALREADY_COMPLETE: Work done in previous run, verify commit exists
        - ISSUE_DOCS_ONLY: Documentation-only changes, skip validation

        For no-op/obsolete resolutions:
        - Gate 2 (commit check) is skipped
        - Gate 3 (validation evidence) is skipped
        - Requires clean working tree and rationale

        For already_complete resolutions:
        - Gate 2 (commit check) runs WITHOUT baseline timestamp (accepts stale commits)
        - Gate 3 (validation evidence) is skipped
        - Requires rationale and valid pre-existing commit

        For docs_only resolutions:
        - Gate 2 (commit check) runs normally
        - Gate 3 (validation evidence) is skipped
        - Requires rationale and commit must not trigger validation
        - Uses should_trigger_validation() for consistency with validation gating:
          * mala.yaml changes always trigger (rejected for DOCS_ONLY)
          * empty code_patterns means all files trigger (rejected for DOCS_ONLY)
          * code_patterns + config_files + setup_files patterns all checked
        - Fails if commit hash or changed files cannot be determined (fails closed)

        When a ValidationSpec is provided, evidence requirements are derived
        from the spec rather than using hardcoded defaults. This ensures:
        - Per-session scope never requires E2E evidence
        - Disabled validations don't cause failures

        Args:
            issue_id: The issue ID to verify.
            log_path: Path to the JSONL log file from agent session.
            baseline_timestamp: Unix timestamp for commit freshness check.
            log_offset: Byte offset to start parsing from.
            spec: ValidationSpec for scope-aware evidence checking. Required.

        Returns:
            GateResult with pass/fail, failure reasons, and resolution if applicable.

        Raises:
            ValueError: If spec is not provided.
        """
        if spec is None:
            raise ValueError("spec is required for check_with_resolution")

        failure_reasons: list[str] = []

        # First, check for resolution markers
        resolution, _ = self.parse_issue_resolution_from_offset(
            log_path, offset=log_offset
        )

        if resolution is not None:
            # No-op or obsolete resolution - verify requirements
            if resolution.outcome in (
                ResolutionOutcome.NO_CHANGE,
                ResolutionOutcome.OBSOLETE,
            ):
                # Require rationale
                if not resolution.rationale.strip():
                    failure_reasons.append(
                        f"{resolution.outcome.value.upper()} resolution requires a rationale"
                    )
                    return GateResult(
                        passed=False,
                        failure_reasons=failure_reasons,
                        resolution=resolution,
                    )

                # No-op/obsolete with rationale passes
                # (skip working tree check - parallel agents may have uncommitted changes)
                return GateResult(
                    passed=True,
                    resolution=resolution,
                )

            # Already complete resolution - verify pre-existing commit
            if resolution.outcome == ResolutionOutcome.ALREADY_COMPLETE:
                # Require rationale
                if not resolution.rationale.strip():
                    failure_reasons.append(
                        "ALREADY_COMPLETE resolution requires a rationale"
                    )
                    return GateResult(
                        passed=False,
                        failure_reasons=failure_reasons,
                        resolution=resolution,
                    )

                # For duplicate issues, the rationale may reference a different issue ID
                # (e.g., "Work committed in 238e17f (bd-mala-xyz: ...)").
                # Extract and use that ID if present, otherwise fall back to current issue.
                referenced_id = self.extract_issue_from_rationale(resolution.rationale)
                check_issue_id = referenced_id or issue_id

                # Verify commit exists WITHOUT baseline check (accepts stale commits)
                commit_result = self.check_commit_exists(
                    check_issue_id, baseline_timestamp=None
                )
                if not commit_result.exists:
                    if referenced_id and referenced_id != issue_id:
                        failure_reasons.append(
                            f"ALREADY_COMPLETE resolution references bd-{referenced_id} "
                            "but no matching commit was found"
                        )
                    else:
                        failure_reasons.append(
                            f"ALREADY_COMPLETE resolution requires a commit with bd-{issue_id} "
                            "but none was found"
                        )
                    return GateResult(
                        passed=False,
                        failure_reasons=failure_reasons,
                        resolution=resolution,
                    )

                # Already complete with rationale and valid commit passes
                # (skip validation evidence - was validated in prior run)
                return GateResult(
                    passed=True,
                    commit_hash=commit_result.commit_hash,
                    resolution=resolution,
                )

            # Docs-only resolution - verify commit exists and contains no code files
            if resolution.outcome == ResolutionOutcome.DOCS_ONLY:
                # Require rationale
                if not resolution.rationale.strip():
                    failure_reasons.append("DOCS_ONLY resolution requires a rationale")
                    return GateResult(
                        passed=False,
                        failure_reasons=failure_reasons,
                        resolution=resolution,
                    )

                # Verify commit exists
                commit_result = self.check_commit_exists(issue_id, baseline_timestamp)
                if not commit_result.exists:
                    if baseline_timestamp is not None:
                        failure_reasons.append(
                            f"No commit with bd-{issue_id} found after run baseline "
                            f"(stale commits from previous runs are rejected)"
                        )
                    else:
                        failure_reasons.append(
                            f"No commit with bd-{issue_id} found in the last 30 days"
                        )
                    return GateResult(
                        passed=False,
                        failure_reasons=failure_reasons,
                        resolution=resolution,
                    )

                # Fail closed: if commit hash is missing, we can't verify files
                if not commit_result.commit_hash:
                    failure_reasons.append(
                        "DOCS_ONLY resolution failed: commit exists but hash "
                        "could not be determined"
                    )
                    return GateResult(
                        passed=False,
                        failure_reasons=failure_reasons,
                        resolution=resolution,
                    )

                # Get files changed in this commit
                commit_files = self.get_commit_files(commit_result.commit_hash)

                # Fail closed: if we can't determine changed files, reject DOCS_ONLY
                if commit_files is None:
                    failure_reasons.append(
                        "DOCS_ONLY resolution failed: could not determine "
                        "files changed in commit (git diff-tree failed)"
                    )
                    return GateResult(
                        passed=False,
                        failure_reasons=failure_reasons,
                        commit_hash=commit_result.commit_hash,
                        resolution=resolution,
                    )

                # Use should_trigger_validation to check if any changed files
                # would trigger validation. This reuses the same logic as
                # validation gating, including:
                # - mala.yaml changes always trigger
                # - empty code_patterns means match all files
                # - config_files and setup_files patterns
                if should_trigger_validation(commit_files, spec):
                    failure_reasons.append(
                        "DOCS_ONLY resolution but commit contains files that "
                        f"trigger validation: {', '.join(commit_files[:5])}"
                        + (
                            f" (and {len(commit_files) - 5} more)"
                            if len(commit_files) > 5
                            else ""
                        )
                    )
                    return GateResult(
                        passed=False,
                        failure_reasons=failure_reasons,
                        commit_hash=commit_result.commit_hash,
                        resolution=resolution,
                    )

                # Docs-only with rationale and no triggering files passes
                # (skip validation evidence - only documentation changes)
                return GateResult(
                    passed=True,
                    commit_hash=commit_result.commit_hash,
                    resolution=resolution,
                )

        # Normal flow - require commit and validation evidence
        commit_result = self.check_commit_exists(issue_id, baseline_timestamp)
        if not commit_result.exists:
            if baseline_timestamp is not None:
                failure_reasons.append(
                    f"No commit with bd-{issue_id} found after run baseline "
                    f"(stale commits from previous runs are rejected)"
                )
            else:
                failure_reasons.append(
                    f"No commit with bd-{issue_id} found in the last 30 days"
                )
            return GateResult(
                passed=False,
                failure_reasons=failure_reasons,
            )

        # Gate 3: Check validation evidence (spec-driven)
        evidence = self.parse_validation_evidence_with_spec(log_path, spec, log_offset)

        passed, missing, failed_strict = check_evidence_against_spec(evidence, spec)

        # Check for missing validation commands
        if missing:
            failure_reasons.append(
                f"Missing validation evidence for: {', '.join(missing)}"
            )

        # Check for strict custom command failures (allow_fail=False)
        if failed_strict:
            failure_reasons.append(
                f"Custom command(s) failed: {', '.join(failed_strict)}"
            )

        # Check for failed built-in validation commands
        if evidence.failed_commands:
            passed = False
            failure_reasons.append(
                f"Validation command(s) failed: {', '.join(evidence.failed_commands)}"
            )

        return GateResult(
            passed=passed,
            failure_reasons=failure_reasons,
            commit_hash=commit_result.commit_hash,
            validation_evidence=evidence,
        )
