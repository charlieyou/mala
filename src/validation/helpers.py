"""Shared helper functions for validation runners.

These utilities are used by both LegacyValidationRunner and SpecValidationRunner
to avoid code duplication.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def tail(text: str, max_chars: int = 800, max_lines: int = 20) -> str:
    """Truncate text to last N lines and M characters.

    Args:
        text: The text to truncate.
        max_chars: Maximum number of characters to keep.
        max_lines: Maximum number of lines to keep.

    Returns:
        The truncated text.
    """
    if not text:
        return ""
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    clipped = "\n".join(lines)
    if len(clipped) > max_chars:
        return clipped[-max_chars:]
    return clipped


def decode_timeout_output(data: bytes | str | None) -> str:
    """Decode TimeoutExpired stdout/stderr which may be bytes, str, or None.

    Args:
        data: The output data from TimeoutExpired exception.

    Returns:
        Decoded and truncated string.
    """
    if data is None:
        return ""
    if isinstance(data, str):
        return tail(data)
    return tail(data.decode())


def format_step_output(
    stdout_tail: str,
    stderr_tail: str,
    max_chars: int = 300,
    max_lines: int = 6,
) -> str:
    """Format step output for error messages.

    Prefers stderr over stdout when both are available.

    Args:
        stdout_tail: Truncated stdout.
        stderr_tail: Truncated stderr.
        max_chars: Maximum characters for output.
        max_lines: Maximum lines for output.

    Returns:
        Formatted output string.
    """
    parts = []
    if stderr_tail:
        parts.append(
            f"stderr: {tail(stderr_tail, max_chars=max_chars, max_lines=max_lines)}"
        )
    if stdout_tail and not stderr_tail:
        parts.append(
            f"stdout: {tail(stdout_tail, max_chars=max_chars, max_lines=max_lines)}"
        )
    return " | ".join(parts)


def check_e2e_prereqs(env: dict[str, str]) -> str | None:
    """Check prerequisites for E2E validation.

    Args:
        env: Environment variables (unused, kept for API compatibility).

    Returns:
        Error message if prereqs not met, None otherwise.
    """
    if not shutil.which("mala"):
        return "E2E prereq missing: mala CLI not found in PATH"
    if not shutil.which("bd"):
        return "E2E prereq missing: bd CLI not found in PATH"
    # Note: MORPH_API_KEY is intentionally NOT checked here.
    # E2E validation should not fail just because the key is missing.
    # Morph-specific tests will skip when the key is absent.
    return None


def write_fixture_repo(repo_path: Path) -> None:
    """Create a minimal fixture repository for E2E testing.

    Creates a simple Python project with a failing test that the
    implementer agent needs to fix.

    Args:
        repo_path: Path to create the fixture repository in.
    """
    (repo_path / "tests").mkdir(parents=True, exist_ok=True)
    (repo_path / "app.py").write_text(
        "def add(a: int, b: int) -> int:\n    return a - b\n"
    )
    (repo_path / "tests" / "test_app.py").write_text(
        "from app import add\n\n\ndef test_add():\n    assert add(2, 2) == 4\n"
    )
    (repo_path / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                'name = "mala-e2e-fixture"',
                'version = "0.0.0"',
                'description = "Fixture repo for mala e2e validation"',
                'requires-python = ">=3.11"',
                "dependencies = []",
                "",
                "[project.optional-dependencies]",
                'dev = ["pytest>=8.0.0", "pytest-cov>=4.1.0"]',
            ]
        )
        + "\n"
    )


def init_fixture_repo(repo_path: Path) -> str | None:
    """Initialize a fixture repository with git and beads.

    Args:
        repo_path: Path to the fixture repository.

    Returns:
        Error message if initialization failed, None on success.
    """
    for cmd in (
        ["git", "init"],
        ["git", "config", "user.email", "mala-e2e@example.com"],
        ["git", "config", "user.name", "Mala E2E"],
        ["git", "add", "."],
        ["git", "commit", "-m", "initial"],
        ["bd", "init"],
        ["bd", "create", "Fix failing add() test", "-p", "1"],
    ):
        result = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            reason = (
                f"E2E fixture setup failed: {' '.join(cmd)} (exit {result.returncode})"
            )
            if stderr:
                reason = f"{reason}: {tail(stderr)}"
            return reason
    return None


def get_ready_issue_id(repo_path: Path) -> str | None:
    """Get the first ready issue ID from a repository.

    Args:
        repo_path: Path to the repository.

    Returns:
        Issue ID if found, None otherwise.
    """
    result = subprocess.run(
        ["bd", "ready", "--json"],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    try:
        issues = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    for issue in issues:
        issue_id = issue.get("id")
        if isinstance(issue_id, str):
            return issue_id
    return None


def annotate_issue(repo_path: Path, issue_id: str) -> None:
    """Add test plan notes to an issue.

    Args:
        repo_path: Path to the repository.
        issue_id: Issue ID to annotate.
    """
    notes = "\n".join(
        [
            "Context:",
            "- Tests are failing for add() in app.py",
            "",
            "Acceptance Criteria:",
            "- Fix add() so tests pass",
            "- Run full validation suite",
            "",
            "Test Plan:",
            "- uv run pytest",
        ]
    )
    subprocess.run(
        ["bd", "update", issue_id, "--notes", notes],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )
