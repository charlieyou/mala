"""Quality gate for verifying agent work before marking success.

Implements Track A4 from coordination-plan-codex.md:
- Verify commit message contains bd-<issue_id>
- Verify validation commands ran (parse JSONL logs)
- On failure: mark needs-followup with failure context
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


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

        Requires at least:
        - pytest OR (ruff_check AND ruff_format)

        ty_check is optional since not all projects use it.
        uv_sync is recommended but not required.
        """
        has_tests = self.pytest_ran
        has_lint = self.ruff_check_ran and self.ruff_format_ran
        return has_tests or has_lint

    def missing_commands(self) -> list[str]:
        """List validation commands that didn't run."""
        missing = []
        if not self.pytest_ran:
            missing.append("pytest")
        if not self.ruff_check_ran:
            missing.append("ruff check")
        if not self.ruff_format_ran:
            missing.append("ruff format")
        return missing


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


class QualityGate:
    """Quality gate for verifying agent work meets requirements."""

    # Patterns for detecting validation commands in Bash tool calls
    VALIDATION_PATTERNS = {
        "uv_sync": re.compile(r"\buv\s+sync\b"),
        "pytest": re.compile(r"\b(uv\s+run\s+)?pytest\b"),
        "ruff_check": re.compile(r"\b(uvx\s+)?ruff\s+check\b"),
        "ruff_format": re.compile(r"\b(uvx\s+)?ruff\s+format\b"),
        "ty_check": re.compile(r"\b(uvx\s+)?ty\s+check\b"),
    }

    def __init__(self, repo_path: Path):
        """Initialize quality gate.

        Args:
            repo_path: Path to the repository for git operations.
        """
        self.repo_path = repo_path

    def parse_validation_evidence(self, log_path: Path) -> ValidationEvidence:
        """Parse JSONL log file for validation command evidence.

        Args:
            log_path: Path to the JSONL log file from agent session.

        Returns:
            ValidationEvidence with flags for each detected command.
        """
        evidence = ValidationEvidence()

        if not log_path.exists():
            return evidence

        try:
            with open(log_path) as f:
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

        except OSError:
            # File read error - return empty evidence
            pass

        return evidence

    def check_commit_exists(self, issue_id: str) -> CommitResult:
        """Check if a commit with bd-<issue_id> exists in recent history.

        Searches commits from the last 30 days to accommodate long-running
        work that may span multiple days.

        Args:
            issue_id: The issue ID to search for (without bd- prefix).

        Returns:
            CommitResult indicating whether a matching commit exists.
        """
        # Search for commits with bd-<issue_id> in the message
        # Use git log with grep to find matching commits
        pattern = f"bd-{issue_id}"

        result = subprocess.run(
            [
                "git",
                "log",
                "--oneline",
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

        # Parse the commit hash from oneline format: "abc1234 message"
        parts = output.split(" ", 1)
        commit_hash = parts[0] if parts else None
        message = parts[1] if len(parts) > 1 else None

        return CommitResult(
            exists=True,
            commit_hash=commit_hash,
            message=message,
        )

    def check(self, issue_id: str, log_path: Path) -> GateResult:
        """Run full quality gate check.

        Verifies:
        1. A commit exists with bd-<issue_id> in the message
        2. Validation commands (pytest, ruff) were executed

        Args:
            issue_id: The issue ID to verify.
            log_path: Path to the JSONL log file from agent session.

        Returns:
            GateResult with pass/fail and failure reasons.
        """
        failure_reasons: list[str] = []

        # Check for matching commit
        commit_result = self.check_commit_exists(issue_id)
        if not commit_result.exists:
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
