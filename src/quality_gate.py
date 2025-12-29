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

from src.session_log_parser import JsonlEntry, SessionLogParser
from src.tools.command_runner import run_command
from src.validation.spec import (
    CommandKind,
    IssueResolution,
    ResolutionOutcome,
    ValidationScope,
    build_validation_spec,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from src.validation.spec import ValidationSpec


# Re-export JsonlEntry for backward compatibility
__all__ = [
    "CommitResult",
    "GateResult",
    "JsonlEntry",
    "QualityGate",
    "ValidationEvidence",
]


@dataclass
class ValidationEvidence:
    """Evidence of validation commands executed during agent run."""

    pytest_ran: bool = False
    ruff_check_ran: bool = False
    ruff_format_ran: bool = False
    ty_check_ran: bool = False

    # Track which validation commands failed (exited non-zero)
    failed_commands: list[str] = field(default_factory=list)

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


def get_required_evidence_kinds(spec: ValidationSpec) -> set[CommandKind]:
    """Get the set of command kinds required by a ValidationSpec.

    This derives the expected evidence from the spec, ensuring scope-aware
    evidence requirements. For example, per-issue scope specs won't have
    E2E commands, so E2E evidence won't be required.

    Args:
        spec: The ValidationSpec to extract requirements from.

    Returns:
        Set of CommandKind values that must have evidence.
    """
    return {cmd.kind for cmd in spec.commands}


def check_evidence_against_spec(
    evidence: ValidationEvidence, spec: ValidationSpec
) -> tuple[bool, list[str]]:
    """Check if evidence satisfies a ValidationSpec's requirements.

    This is scope-aware: a per-issue spec won't require E2E evidence because
    per-issue specs don't include E2E commands.

    Args:
        evidence: The parsed validation evidence.
        spec: The ValidationSpec defining what's required.

    Returns:
        Tuple of (passed, missing_commands) where missing_commands lists
        human-readable names of commands that didn't run.
    """
    required = get_required_evidence_kinds(spec)
    missing: list[str] = []

    # Map CommandKind to evidence flags and display names
    kind_to_evidence: dict[CommandKind, tuple[bool, str]] = {
        CommandKind.TEST: (evidence.pytest_ran, "pytest"),
        CommandKind.LINT: (evidence.ruff_check_ran, "ruff check"),
        CommandKind.FORMAT: (evidence.ruff_format_ran, "ruff format"),
        CommandKind.TYPECHECK: (evidence.ty_check_ran, "ty check"),
        # E2E is checked separately since it has special handling
    }

    for kind in required:
        if kind in kind_to_evidence:
            ran, name = kind_to_evidence[kind]
            if not ran:
                missing.append(name)

    return len(missing) == 0, missing


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
    validation_evidence: ValidationEvidence | None = None
    no_progress: bool = False
    resolution: IssueResolution | None = None


class QualityGate:
    """Quality gate for verifying agent work meets requirements.

    Uses SessionLogParser for JSONL log parsing, keeping this class
    focused on policy checking and validation logic.
    """

    # Patterns for detecting issue resolution markers in log text
    RESOLUTION_PATTERNS: ClassVar[dict[str, re.Pattern[str]]] = {
        "no_change": re.compile(r"ISSUE_NO_CHANGE:\s*(.*)$", re.MULTILINE),
        "obsolete": re.compile(r"ISSUE_OBSOLETE:\s*(.*)$", re.MULTILINE),
        "already_complete": re.compile(
            r"ISSUE_ALREADY_COMPLETE:\s*(.*)$", re.MULTILINE
        ),
    }

    # Map pattern names to resolution outcomes
    PATTERN_TO_OUTCOME: ClassVar[dict[str, ResolutionOutcome]] = {
        "no_change": ResolutionOutcome.NO_CHANGE,
        "obsolete": ResolutionOutcome.OBSOLETE,
        "already_complete": ResolutionOutcome.ALREADY_COMPLETE,
    }

    def __init__(self, repo_path: Path):
        """Initialize quality gate.

        Args:
            repo_path: Path to the repository for git operations.
        """
        self.repo_path = repo_path
        self._parser = SessionLogParser()

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

    def _match_spec_pattern(
        self,
        command: str,
        evidence: ValidationEvidence,
        kind_patterns: dict[CommandKind, list[re.Pattern[str]]],
        kind_to_name: dict[CommandKind, str],
    ) -> str | None:
        """Check command against spec-defined patterns and update evidence.

        Checks ALL patterns independently (a command may match multiple).
        Returns the last matched command name for failure tracking.

        Args:
            command: The bash command string.
            evidence: ValidationEvidence to update.
            kind_patterns: Mapping of CommandKind to detection patterns.
            kind_to_name: Mapping of CommandKind to human-readable names.

        Returns:
            Last matched command name (for failure tracking), None if no match.
        """
        last_match: str | None = None
        for kind, patterns in kind_patterns.items():
            for pattern in patterns:
                if pattern.search(command):
                    if kind == CommandKind.TEST:
                        evidence.pytest_ran = True
                    elif kind == CommandKind.LINT:
                        evidence.ruff_check_ran = True
                    elif kind == CommandKind.FORMAT:
                        evidence.ruff_format_ran = True
                    elif kind == CommandKind.TYPECHECK:
                        evidence.ty_check_ran = True
                    last_match = kind_to_name.get(kind)
                    break  # Found match for this kind, try next kind
        return last_match

    # Map CommandKind to human-readable names for failure reporting
    KIND_TO_NAME: ClassVar[dict[CommandKind, str]] = {
        CommandKind.TEST: "pytest",
        CommandKind.LINT: "ruff check",
        CommandKind.FORMAT: "ruff format",
        CommandKind.TYPECHECK: "ty check",
    }

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

    def _iter_jsonl_entries(
        self, log_path: Path, offset: int = 0
    ) -> Iterator[JsonlEntry]:
        """Iterate over parsed JSONL entries from a log file.

        Delegates to SessionLogParser.iter_jsonl_entries().

        Args:
            log_path: Path to the JSONL log file.
            offset: Byte offset to start reading from (default 0).

        Yields:
            JsonlEntry objects for each successfully parsed JSON line.
        """
        return self._parser.iter_jsonl_entries(log_path, offset)

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
                for text in self._parser.extract_assistant_text_blocks(entry):
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
        result = run_command(
            ["git", "status", "--porcelain"],
            cwd=self.repo_path,
        )
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
        tool_id_to_command: dict[str, str] = {}
        command_failed: dict[str, bool] = {}

        for entry in self._iter_jsonl_entries(log_path, offset):
            for tool_id, command in self._parser.extract_bash_commands(entry):
                cmd_name = self._match_spec_pattern(
                    command, evidence, kind_patterns, self.KIND_TO_NAME
                )
                if cmd_name:
                    tool_id_to_command[tool_id] = cmd_name
            for tool_use_id, is_error in self._parser.extract_tool_results(entry):
                if tool_use_id in tool_id_to_command:
                    command_failed[tool_id_to_command[tool_use_id]] = is_error

        evidence.failed_commands = [c for c, f in command_failed.items() if f]
        return evidence

    def get_log_end_offset(self, log_path: Path, start_offset: int = 0) -> int:
        """Get the byte offset at the end of a log file.

        Delegates to SessionLogParser.get_log_end_offset().

        Args:
            log_path: Path to the JSONL log file.
            start_offset: Byte offset to start from (default 0).

        Returns:
            The byte offset at the end of the file, or start_offset if file
            doesn't exist or can't be read.
        """
        return self._parser.get_log_end_offset(log_path, start_offset)

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
                If not provided, builds a default per-issue spec.
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
                scope=ValidationScope.PER_ISSUE,
            )

        # Check for new validation evidence after the offset using spec-driven parsing
        evidence = self.parse_validation_evidence_with_spec(log_path, spec, log_offset)

        # Any new validation evidence counts as progress
        has_new_evidence = (
            evidence.pytest_ran
            or evidence.ruff_check_ran
            or evidence.ruff_format_ran
            or evidence.ty_check_ran
        )

        if has_new_evidence:
            return False

        # No commit change, no working tree changes, and no new evidence = no progress
        return True

    def _has_working_tree_changes(self) -> bool:
        """Check if the working tree has uncommitted changes.

        Returns:
            True if there are staged or unstaged changes, False otherwise.
        """
        # Use git status --porcelain to detect any changes
        # This includes staged, unstaged, and untracked files
        result = run_command(
            ["git", "status", "--porcelain"],
            cwd=self.repo_path,
            timeout_seconds=5.0,
        )
        if not result.ok:
            # If git status fails, assume no changes (safe default)
            return False

        # Any output means there are changes
        return bool(result.stdout.strip())

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

        result = run_command(
            [
                "git",
                "log",
                f"--format={format_str}",
                "--grep",
                pattern,
                "-n",
                "1",
                "--since=30 days ago",
            ],
            cwd=self.repo_path,
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

        For no-op/obsolete resolutions:
        - Gate 2 (commit check) is skipped
        - Gate 3 (validation evidence) is skipped
        - Requires clean working tree and rationale

        For already_complete resolutions:
        - Gate 2 (commit check) runs WITHOUT baseline timestamp (accepts stale commits)
        - Gate 3 (validation evidence) is skipped
        - Requires rationale and valid pre-existing commit

        When a ValidationSpec is provided, evidence requirements are derived
        from the spec rather than using hardcoded defaults. This ensures:
        - Per-issue scope never requires E2E evidence
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

                # Require clean working tree
                is_clean, status_output = self.check_working_tree_clean()
                if not is_clean:
                    failure_reasons.append(
                        f"Working tree has uncommitted changes for {resolution.outcome.value} resolution: {status_output}"
                    )
                    return GateResult(
                        passed=False,
                        failure_reasons=failure_reasons,
                        resolution=resolution,
                    )

                # No-op/obsolete with rationale and clean tree passes
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

                # Verify commit exists WITHOUT baseline check (accepts stale commits)
                commit_result = self.check_commit_exists(
                    issue_id, baseline_timestamp=None
                )
                if not commit_result.exists:
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
                    f"No commit with bd-{issue_id} found in last 30 days"
                )

        # Check validation evidence from the given offset (scopes to current attempt)
        evidence = self.parse_validation_evidence_with_spec(log_path, spec, log_offset)
        passed, missing = check_evidence_against_spec(evidence, spec)
        if not passed:
            failure_reasons.append(f"Missing validation commands: {', '.join(missing)}")

        # Check for validation commands that failed (exited non-zero)
        if evidence.failed_commands:
            failure_reasons.append(
                f"Validation commands failed (non-zero exit): {', '.join(evidence.failed_commands)}"
            )

        return GateResult(
            passed=len(failure_reasons) == 0,
            failure_reasons=failure_reasons,
            commit_hash=commit_result.commit_hash,
            validation_evidence=evidence,
            resolution=resolution,
        )
