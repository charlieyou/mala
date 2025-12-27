"""Quality gate for verifying agent work before marking success.

Implements Track A4 from 2025-12-26-coordination-plan.md:
- Verify commit message contains bd-<issue_id>
- Verify validation commands ran (parse JSONL logs)
- On failure: mark needs-followup with failure context
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from src.validation.spec import (
    CommandKind,
    IssueResolution,
    ResolutionOutcome,
)

if TYPE_CHECKING:
    from pathlib import Path

    from src.validation.spec import ValidationSpec


@dataclass
class ValidationEvidence:
    """Evidence of validation commands executed during agent run."""

    uv_sync_ran: bool = False
    pytest_ran: bool = False
    ruff_check_ran: bool = False
    ruff_format_ran: bool = False
    ty_check_ran: bool = False

    def has_minimum_validation(self) -> bool:
        """Check if minimum required validation was performed.

        Requires the full validation suite:
        - uv sync (ensure dependencies are current)
        - pytest (run tests)
        - ruff check (lint)
        - ruff format (format)
        - ty check (type check)
        """
        return (
            self.uv_sync_ran
            and self.pytest_ran
            and self.ruff_check_ran
            and self.ruff_format_ran
            and self.ty_check_ran
        )

    def missing_commands(self) -> list[str]:
        """List validation commands that didn't run."""
        missing = []
        if not self.uv_sync_ran:
            missing.append("uv sync")
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
        CommandKind.DEPS: (evidence.uv_sync_ran, "uv sync"),
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

    # Patterns for detecting validation commands in Bash tool calls
    VALIDATION_PATTERNS: ClassVar[dict[str, re.Pattern[str]]] = {
        "uv_sync": re.compile(r"\buv\s+sync\b"),
        "pytest": re.compile(r"\b(uv\s+run\s+)?pytest\b"),
        "ruff_check": re.compile(r"\b(uvx\s+)?ruff\s+check\b"),
        "ruff_format": re.compile(r"\b(uvx\s+)?ruff\s+format\b"),
        "ty_check": re.compile(r"\b(uvx\s+)?ty\s+check\b"),
    }

    # Patterns for detecting issue resolution markers in log text
    RESOLUTION_PATTERNS: ClassVar[dict[str, re.Pattern[str]]] = {
        "no_change": re.compile(r"ISSUE_NO_CHANGE:\s*(.*)$", re.MULTILINE),
        "obsolete": re.compile(r"ISSUE_OBSOLETE:\s*(.*)$", re.MULTILINE),
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
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=self.repo_path,
        )
        # Treat git failures as dirty/unknown state
        if result.returncode != 0:
            error_msg = result.stderr.strip() or "git status failed"
            return False, f"git error: {error_msg}"
        output = result.stdout.strip()
        return len(output) == 0, output

    def parse_validation_evidence(self, log_path: Path) -> ValidationEvidence:
        """Parse JSONL log file for validation command evidence.

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

        This allows scoping evidence to a specific attempt by starting from
        the byte offset where the previous attempt ended.

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

        try:
            with open(log_path) as f:
                # Seek to the starting offset
                f.seek(offset)

                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Look for Bash tool_use entries
                    message = entry.get("message", {})
                    content = message.get("content", [])

                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        if block.get("type") != "tool_use":
                            continue
                        if block.get("name") != "Bash":
                            continue

                        # Extract command from input
                        input_data = block.get("input", {})
                        command = input_data.get("command", "")

                        # Check against validation patterns
                        if self.VALIDATION_PATTERNS["uv_sync"].search(command):
                            evidence.uv_sync_ran = True
                        if self.VALIDATION_PATTERNS["pytest"].search(command):
                            evidence.pytest_ran = True
                        if self.VALIDATION_PATTERNS["ruff_check"].search(command):
                            evidence.ruff_check_ran = True
                        if self.VALIDATION_PATTERNS["ruff_format"].search(command):
                            evidence.ruff_format_ran = True
                        if self.VALIDATION_PATTERNS["ty_check"].search(command):
                            evidence.ty_check_ran = True

                # Get the new offset (current file position)
                new_offset = f.tell()

        except OSError:
            # File read error - return empty evidence
            return evidence, 0

        return evidence, new_offset

    def check_no_progress(
        self,
        log_path: Path,
        log_offset: int,
        previous_commit_hash: str | None,
        current_commit_hash: str | None,
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

        # Check for new validation evidence after the offset
        evidence, _ = self.parse_validation_evidence_from_offset(
            log_path, offset=log_offset
        )

        # Any new validation evidence counts as progress
        has_new_evidence = (
            evidence.uv_sync_ran
            or evidence.pytest_ran
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

        result = subprocess.run(
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
            capture_output=True,
            text=True,
            cwd=self.repo_path,
        )

        if result.returncode != 0:
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
        failure_reasons: list[str] = []

        # Check for matching commit
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

        # Check validation evidence
        evidence = self.parse_validation_evidence(log_path)
        if not evidence.has_minimum_validation():
            missing = evidence.missing_commands()
            failure_reasons.append(f"Missing validation commands: {', '.join(missing)}")

        return GateResult(
            passed=len(failure_reasons) == 0,
            failure_reasons=failure_reasons,
            commit_hash=commit_result.commit_hash,
            validation_evidence=evidence,
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

        For no-op/obsolete resolutions:
        - Gate 2 (commit check) is skipped
        - Gate 3 (validation evidence) is skipped
        - Requires clean working tree and rationale

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
        evidence, _ = self.parse_validation_evidence_from_offset(log_path, log_offset)

        # Use spec-aware checking if spec provided, otherwise use default
        if spec is not None:
            passed, missing = check_evidence_against_spec(evidence, spec)
            if not passed:
                failure_reasons.append(
                    f"Missing validation commands: {', '.join(missing)}"
                )
        else:
            # Fallback to hardcoded default (backward compat)
            if not evidence.has_minimum_validation():
                missing = evidence.missing_commands()
                failure_reasons.append(
                    f"Missing validation commands: {', '.join(missing)}"
                )

        return GateResult(
            passed=len(failure_reasons) == 0,
            failure_reasons=failure_reasons,
            commit_hash=commit_result.commit_hash,
            validation_evidence=evidence,
            resolution=resolution,
        )
