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

    VALIDATION_PATTERNS, parse_validation_evidence(), and
    parse_validation_evidence_from_offset() exist for backward compatibility
    and are deprecated for new code. The legacy check() method is also
    deprecated in favor of check_with_resolution().
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from src.validation.command_runner import run_command
from src.validation.spec import (
    CommandKind,
    IssueResolution,
    ResolutionOutcome,
    ValidationScope,
    build_validation_spec,
)

if TYPE_CHECKING:
    from pathlib import Path

    from src.validation.spec import ValidationSpec


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
    """Quality gate for verifying agent work meets requirements."""

    # DEPRECATED: Use detection_pattern from ValidationSpec commands instead.
    # These hardcoded patterns exist for backward compatibility when parsing
    # evidence without a spec. New code should use parse_validation_evidence_with_spec()
    # or check_with_resolution(..., spec=spec) to derive patterns from the spec.
    VALIDATION_PATTERNS: ClassVar[dict[str, re.Pattern[str]]] = {
        "pytest": re.compile(r"\b(uv\s+run\s+)?pytest\b"),
        "ruff_check": re.compile(r"\b(uvx\s+)?ruff\s+check\b"),
        "ruff_format": re.compile(r"\b(uvx\s+)?ruff\s+format\b"),
        "ty_check": re.compile(r"\b(uvx\s+)?ty\s+check\b"),
    }

    # Patterns for detecting issue resolution markers in log text
    RESOLUTION_PATTERNS: ClassVar[dict[str, re.Pattern[str]]] = {
        "no_change": re.compile(r"ISSUE_NO_CHANGE:\s*(.*)$", re.MULTILINE),
        "obsolete": re.compile(r"ISSUE_OBSOLETE:\s*(.*)$", re.MULTILINE),
        "already_complete": re.compile(
            r"ISSUE_ALREADY_COMPLETE:\s*(.*)$", re.MULTILINE
        ),
    }

    def __init__(self, repo_path: Path):
        """Initialize quality gate.

        Args:
            repo_path: Path to the repository for git operations.
        """
        self.repo_path = repo_path

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
            # Read file in binary mode for accurate byte offset tracking
            with open(log_path, "rb") as f:
                f.seek(offset)
                current_offset = offset

                for line_bytes in f:
                    line_len = len(line_bytes)
                    try:
                        line = line_bytes.decode("utf-8").strip()
                    except UnicodeDecodeError:
                        current_offset += line_len
                        continue

                    if not line:
                        current_offset += line_len
                        continue

                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        current_offset += line_len
                        continue

                    # Only parse assistant messages to prevent user prompt injection
                    entry_type = entry.get("type", "")
                    entry_role = entry.get("message", {}).get("role", "")
                    if entry_type != "assistant" and entry_role != "assistant":
                        current_offset += line_len
                        continue

                    # Look for text blocks in assistant messages
                    message = entry.get("message", {})
                    message_content = message.get("content", [])

                    for block in message_content:
                        if not isinstance(block, dict):
                            continue
                        if block.get("type") != "text":
                            continue

                        text = block.get("text", "")

                        # Check for no_change marker
                        match = self.RESOLUTION_PATTERNS["no_change"].search(text)
                        if match:
                            rationale = match.group(1).strip()
                            return (
                                IssueResolution(
                                    outcome=ResolutionOutcome.NO_CHANGE,
                                    rationale=rationale,
                                ),
                                current_offset + line_len,
                            )

                        # Check for obsolete marker
                        match = self.RESOLUTION_PATTERNS["obsolete"].search(text)
                        if match:
                            rationale = match.group(1).strip()
                            return (
                                IssueResolution(
                                    outcome=ResolutionOutcome.OBSOLETE,
                                    rationale=rationale,
                                ),
                                current_offset + line_len,
                            )

                        # Check for already_complete marker
                        match = self.RESOLUTION_PATTERNS["already_complete"].search(
                            text
                        )
                        if match:
                            rationale = match.group(1).strip()
                            return (
                                IssueResolution(
                                    outcome=ResolutionOutcome.ALREADY_COMPLETE,
                                    rationale=rationale,
                                ),
                                current_offset + line_len,
                            )

                    current_offset += line_len

                # Return final position
                return None, f.tell()

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

    def parse_validation_evidence(self, log_path: Path) -> ValidationEvidence:
        """Parse JSONL log file for validation command evidence.

        DEPRECATED: Use parse_validation_evidence_with_spec() instead to derive
        detection patterns from the ValidationSpec. This method uses hardcoded
        VALIDATION_PATTERNS which may drift from spec command definitions.

        Args:
            log_path: Path to the JSONL log file from agent session.

        Returns:
            ValidationEvidence with flags for each detected command.
        """
        evidence, _ = self.parse_validation_evidence_from_offset(log_path, offset=0)
        return evidence

    def parse_validation_evidence_from_offset(
        self, log_path: Path, offset: int = 0
    ) -> tuple[ValidationEvidence, int]:
        """Parse JSONL log file for validation command evidence starting at offset.

        DEPRECATED: Use parse_validation_evidence_with_spec() instead for evidence
        checking. This method is retained for offset tracking (getting the new file
        position) but uses hardcoded VALIDATION_PATTERNS which may drift from spec
        command definitions.

        This allows scoping evidence to a specific attempt by starting from
        the byte offset where the previous attempt ended.

        Also checks tool_result entries for failures (is_error=true), tracking
        which validation commands exited non-zero. If a command is retried and
        succeeds, the failure is cleared (tracks latest status per command).

        Args:
            log_path: Path to the JSONL log file from agent session.
            offset: Byte offset to start reading from (default 0 = beginning).

        Returns:
            Tuple of (ValidationEvidence, new_offset) where new_offset is the
            file position after parsing (for use in subsequent calls).
        """
        evidence = ValidationEvidence()

        if not log_path.exists():
            return evidence, 0

        # Map tool_use_id to validation command name for tracking failures
        tool_id_to_command: dict[str, str] = {}
        # Track latest status per command (True = failed, False = succeeded)
        command_failed: dict[str, bool] = {}

        try:
            # Read file in binary mode for accurate byte offset tracking
            # (must match parse_issue_resolution_from_offset's binary mode)
            with open(log_path, "rb") as f:
                f.seek(offset)
                current_offset = offset

                for line_bytes in f:
                    line_len = len(line_bytes)
                    try:
                        line = line_bytes.decode("utf-8").strip()
                    except UnicodeDecodeError:
                        current_offset += line_len
                        continue

                    if not line:
                        current_offset += line_len
                        continue

                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        current_offset += line_len
                        continue

                    entry_type = entry.get("type", "")
                    message = entry.get("message", {})
                    content = message.get("content", [])

                    # Process assistant messages for tool_use entries
                    if entry_type == "assistant":
                        for block in content:
                            if not isinstance(block, dict):
                                continue
                            if block.get("type") != "tool_use":
                                continue
                            if block.get("name") != "Bash":
                                continue

                            tool_id = block.get("id", "")
                            input_data = block.get("input", {})
                            command = input_data.get("command", "")

                            # Check against validation patterns and record tool_id mapping
                            if self.VALIDATION_PATTERNS["pytest"].search(command):
                                evidence.pytest_ran = True
                                tool_id_to_command[tool_id] = "pytest"
                            if self.VALIDATION_PATTERNS["ruff_check"].search(command):
                                evidence.ruff_check_ran = True
                                tool_id_to_command[tool_id] = "ruff check"
                            if self.VALIDATION_PATTERNS["ruff_format"].search(command):
                                evidence.ruff_format_ran = True
                                tool_id_to_command[tool_id] = "ruff format"
                            if self.VALIDATION_PATTERNS["ty_check"].search(command):
                                evidence.ty_check_ran = True
                                tool_id_to_command[tool_id] = "ty check"

                    # Process user messages for tool_result entries (check for failures)
                    elif entry_type == "user":
                        for block in content:
                            if not isinstance(block, dict):
                                continue
                            if block.get("type") != "tool_result":
                                continue

                            tool_use_id = block.get("tool_use_id", "")
                            is_error = block.get("is_error", False)

                            # Track latest status for validation commands
                            if tool_use_id in tool_id_to_command:
                                cmd_name = tool_id_to_command[tool_use_id]
                                command_failed[cmd_name] = is_error

                    current_offset += line_len

                # Compute failed_commands from final status of each command
                evidence.failed_commands = [
                    cmd for cmd, failed in command_failed.items() if failed
                ]

                # Return final position
                return evidence, f.tell()

        except OSError:
            # File read error - return empty evidence
            return evidence, 0

    def parse_validation_evidence_with_spec(
        self, log_path: Path, spec: ValidationSpec, offset: int = 0
    ) -> ValidationEvidence:
        """Parse JSONL log for validation evidence using spec-defined patterns.

        This method derives detection patterns from the ValidationSpec, ensuring
        that evidence parsing is driven from spec command definitions. When spec
        commands change, evidence detection updates automatically.

        For commands without a detection_pattern, falls back to hardcoded
        VALIDATION_PATTERNS for backward compatibility.

        Also checks tool_result entries for failures (is_error=true), tracking
        which validation commands exited non-zero.

        Args:
            log_path: Path to the JSONL log file from agent session.
            spec: The ValidationSpec defining commands and their detection patterns.
            offset: Byte offset to start reading from (default 0 = beginning).

        Returns:
            ValidationEvidence with flags for each detected command.
        """
        evidence = ValidationEvidence()

        if not log_path.exists():
            return evidence

        # Build mapping from CommandKind to detection patterns
        # Use spec patterns when available, fall back to hardcoded patterns
        kind_patterns: dict[CommandKind, list[re.Pattern[str]]] = {}
        for cmd in spec.commands:
            if cmd.kind not in kind_patterns:
                kind_patterns[cmd.kind] = []
            if cmd.detection_pattern is not None:
                kind_patterns[cmd.kind].append(cmd.detection_pattern)
            else:
                # Fall back to hardcoded patterns
                fallback = self._get_fallback_pattern(cmd.kind)
                if fallback is not None:
                    kind_patterns[cmd.kind].append(fallback)

        # Map CommandKind to human-readable names for failure reporting
        kind_to_name: dict[CommandKind, str] = {
            CommandKind.TEST: "pytest",
            CommandKind.LINT: "ruff check",
            CommandKind.FORMAT: "ruff format",
            CommandKind.TYPECHECK: "ty check",
        }

        # Map tool_use_id to command name for tracking failures
        tool_id_to_command: dict[str, str] = {}
        # Track latest status per command (True = failed, False = succeeded)
        command_failed: dict[str, bool] = {}

        try:
            with open(log_path, "rb") as f:
                f.seek(offset)

                for line_bytes in f:
                    try:
                        line = line_bytes.decode("utf-8").strip()
                    except UnicodeDecodeError:
                        continue

                    if not line:
                        continue

                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    entry_type = entry.get("type", "")
                    message = entry.get("message", {})
                    content = message.get("content", [])

                    # Process assistant messages for tool_use entries
                    if entry_type == "assistant":
                        for block in content:
                            if not isinstance(block, dict):
                                continue
                            if block.get("type") != "tool_use":
                                continue
                            if block.get("name") != "Bash":
                                continue

                            tool_id = block.get("id", "")
                            input_data = block.get("input", {})
                            command = input_data.get("command", "")

                            # Check against spec-defined patterns
                            for kind, patterns in kind_patterns.items():
                                for pattern in patterns:
                                    if pattern.search(command):
                                        # Map CommandKind to evidence flags
                                        if kind == CommandKind.TEST:
                                            evidence.pytest_ran = True
                                        elif kind == CommandKind.LINT:
                                            evidence.ruff_check_ran = True
                                        elif kind == CommandKind.FORMAT:
                                            evidence.ruff_format_ran = True
                                        elif kind == CommandKind.TYPECHECK:
                                            evidence.ty_check_ran = True
                                        # Record tool_id mapping for failure tracking
                                        if kind in kind_to_name:
                                            tool_id_to_command[tool_id] = kind_to_name[
                                                kind
                                            ]
                                        break  # Found match for this kind

                    # Process user messages for tool_result entries (check for failures)
                    elif entry_type == "user":
                        for block in content:
                            if not isinstance(block, dict):
                                continue
                            if block.get("type") != "tool_result":
                                continue

                            tool_use_id = block.get("tool_use_id", "")
                            is_error = block.get("is_error", False)

                            # Track latest status for validation commands
                            if tool_use_id in tool_id_to_command:
                                cmd_name = tool_id_to_command[tool_use_id]
                                command_failed[cmd_name] = is_error

        except OSError:
            pass  # File read error - return empty evidence

        # Compute failed_commands from final status of each command
        evidence.failed_commands = [
            cmd for cmd, failed in command_failed.items() if failed
        ]

        return evidence

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
        if not log_path.exists():
            return start_offset

        try:
            with open(log_path, "rb") as f:
                f.seek(0, 2)  # Seek to end
                return f.tell()
        except OSError:
            return start_offset

    def _get_fallback_pattern(self, kind: CommandKind) -> re.Pattern[str] | None:
        """Get fallback pattern for a CommandKind when spec pattern is missing.

        This is a backward compatibility fallback. All ValidationSpec commands
        built by build_validation_spec() include detection_pattern, so this
        fallback should rarely be needed. If you're adding new command kinds,
        ensure build_validation_spec() sets detection_pattern for them.
        """
        fallback_map = {
            CommandKind.TEST: self.VALIDATION_PATTERNS["pytest"],
            CommandKind.LINT: self.VALIDATION_PATTERNS["ruff_check"],
            CommandKind.FORMAT: self.VALIDATION_PATTERNS["ruff_format"],
            CommandKind.TYPECHECK: self.VALIDATION_PATTERNS["ty_check"],
        }
        return fallback_map.get(kind)

    def check_no_progress(
        self,
        log_path: Path,
        log_offset: int,
        previous_commit_hash: str | None,
        current_commit_hash: str | None,
        spec: ValidationSpec | None = None,
    ) -> bool:
        """Check if no progress was made since the last attempt.

        No progress is detected when:
        - The commit hash hasn't changed (or both are None)
        - No new validation evidence was found after the log offset

        Args:
            log_path: Path to the JSONL log file from agent session.
            log_offset: Byte offset marking the end of the previous attempt.
            previous_commit_hash: Commit hash from the previous attempt (None if no commit).
            current_commit_hash: Commit hash from this attempt (None if no commit).
            spec: Optional ValidationSpec for spec-driven evidence detection.
                If not provided, builds a default per-issue spec.

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

        # No commit change and no new evidence = no progress
        return True

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

    def check(
        self, issue_id: str, log_path: Path, baseline_timestamp: int | None = None
    ) -> GateResult:
        """Run full quality gate check.

        DEPRECATED: Use check_with_resolution(..., spec=spec) instead. This method
        does not support no-change/obsolete/already-complete resolutions.

        Internally delegates to check_with_resolution with a default per-issue spec
        to ensure spec-driven evidence parsing.

        Verifies:
        1. A commit exists with bd-<issue_id> in the message
        2. Validation commands (pytest, ruff) were executed

        Args:
            issue_id: The issue ID to verify.
            log_path: Path to the JSONL log file from agent session.
            baseline_timestamp: Unix timestamp. If provided, only accepts commits
                created after this time (to reject stale commits from previous runs).

        Returns:
            GateResult with pass/fail and failure reasons.
        """
        # Build a default per-issue spec to use spec-driven evidence parsing
        # Note: We don't pass repo_path here to ensure Python validation commands
        # are always expected. Production callers should use check_with_resolution
        # with an explicit spec for repo-type-aware validation.
        spec = build_validation_spec(
            scope=ValidationScope.PER_ISSUE,
        )
        return self.check_with_resolution(
            issue_id=issue_id,
            log_path=log_path,
            baseline_timestamp=baseline_timestamp,
            spec=spec,
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
            spec: Optional ValidationSpec for scope-aware evidence checking.

        Returns:
            GateResult with pass/fail, failure reasons, and resolution if applicable.
        """
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
        # Use spec-driven parsing when spec is provided
        if spec is not None:
            evidence = self.parse_validation_evidence_with_spec(
                log_path, spec, log_offset
            )
            passed, missing = check_evidence_against_spec(evidence, spec)
            if not passed:
                failure_reasons.append(
                    f"Missing validation commands: {', '.join(missing)}"
                )
        else:
            # Fallback to hardcoded patterns (backward compat)
            evidence, _ = self.parse_validation_evidence_from_offset(
                log_path, log_offset
            )
            if not evidence.has_minimum_validation():
                missing = evidence.missing_commands()
                failure_reasons.append(
                    f"Missing validation commands: {', '.join(missing)}"
                )

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
