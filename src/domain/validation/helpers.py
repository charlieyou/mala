"""Shared helper functions for validation runners.

These utilities are used by SpecValidationRunner and related modules.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.protocols.infra import CommandRunnerPort


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
    if not shutil.which("br"):
        return "E2E prereq missing: br CLI not found in PATH"
    return None


def _generate_fixture_programmatically(repo_path: Path) -> None:
    """Generate E2E fixture files programmatically.

    This is a fallback when the fixture template directory is not available
    (e.g., in installed packages where only src/ is included).

    Args:
        repo_path: Path to create the fixture repository in.
    """
    # Create src/app.py with a bug (subtracts instead of adds)
    src_dir = repo_path / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "app.py").write_text(
        "def add(a: int, b: int) -> int:\n    return a - b\n"
    )

    # Create tests/test_app.py
    tests_dir = repo_path / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "test_app.py").write_text(
        """import pytest

from app import add


@pytest.mark.unit
def test_add():
    assert add(2, 2) == 4
"""
    )

    # Create pyproject.toml
    (repo_path / "pyproject.toml").write_text(
        """[project]
name = "mala-e2e-fixture"
version = "0.0.0"
description = "Fixture repo for mala e2e validation"
requires-python = ">=3.11"
dependencies = []

[project.optional-dependencies]
dev = ["pytest>=8.0.0", "pytest-cov>=4.1.0", "pytest-xdist>=3.8.0"]

[tool.pytest.ini_options]
pythonpath = ["src"]
markers = [
    "unit: fast, isolated tests (default)",
    "integration: tests that exercise multiple components",
]

[dependency-groups]
dev = [
    "pytest>=9.0.2",
    "pytest-cov>=6.0.0",
    "pytest-xdist>=3.8.0",
]
"""
    )

    # Create mala.yaml
    (repo_path / "mala.yaml").write_text(
        """preset: python-uv

# Explicit run_end trigger with validation commands
validation_triggers:
  run_end:
    failure_mode: continue
    commands:
      - ref: test
        command: "uv run pytest --cov=src --cov-report=xml:coverage.xml --cov-fail-under=0 -o cache_dir=/tmp/pytest-${AGENT_ID:-default}"

coverage:
  format: xml
  file: coverage.xml
  threshold: 0
"""
    )


def write_fixture_repo(repo_path: Path, repo_root: Path | None = None) -> None:
    """Create a minimal fixture repository for E2E testing.

    Creates a simple Python project with a failing test that the
    implementer agent needs to fix. Uses src/ layout for compatibility
    with mala's coverage checking (--cov=src).

    Fixture files are sourced from tests/fixtures/e2e-fixture if available,
    otherwise generated programmatically (for installed packages).

    Args:
        repo_path: Path to create the fixture repository in.
        repo_root: Optional repo root to source the fixture template from.
    """
    root = repo_root or Path(__file__).resolve().parents[3]
    fixture_root = root / "tests" / "fixtures" / "e2e-fixture"
    if fixture_root.exists():
        shutil.copytree(fixture_root, repo_path, dirs_exist_ok=True)
    else:
        # Fallback to programmatic generation when fixture template is not available
        # (e.g., in installed packages where only src/ is included in the wheel)
        _generate_fixture_programmatically(repo_path)


def init_fixture_repo(
    repo_path: Path,
    command_runner: CommandRunnerPort,
) -> str | None:
    """Initialize a fixture repository with git and beads.

    Args:
        repo_path: Path to the fixture repository.
        command_runner: CommandRunnerPort for running commands.

    Returns:
        Error message if initialization failed, None on success.
    """
    runner = command_runner

    for cmd in (
        ["git", "init"],
        ["git", "config", "user.email", "mala-e2e@example.com"],
        ["git", "config", "user.name", "Mala E2E"],
        ["git", "add", "."],
        ["git", "commit", "-m", "initial"],
        ["br", "init"],
        ["br", "create", "Fix failing add() test", "-p", "1"],
    ):
        result = runner.run(cmd, cwd=repo_path)
        if not result.ok:
            stderr = result.stderr.strip()
            reason = (
                f"E2E fixture setup failed: {' '.join(cmd)} (exit {result.returncode})"
            )
            if stderr:
                reason = f"{reason}: {tail(stderr)}"
            return reason
    return None


def get_ready_issue_id(
    repo_path: Path,
    command_runner: CommandRunnerPort,
) -> str | None:
    """Get the first ready issue ID from a repository.

    Args:
        repo_path: Path to the repository.
        command_runner: CommandRunnerPort for running commands.

    Returns:
        Issue ID if found, None otherwise.
    """
    runner = command_runner

    result = runner.run(["br", "ready", "--json"], cwd=repo_path)
    if not result.ok:
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


def annotate_issue(
    repo_path: Path,
    issue_id: str,
    command_runner: CommandRunnerPort,
) -> None:
    """Add test plan notes to an issue.

    Args:
        repo_path: Path to the repository.
        issue_id: Issue ID to annotate.
        command_runner: CommandRunnerPort for running commands.
    """
    runner = command_runner

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
    runner.run(["br", "update", issue_id, "--notes", notes], cwd=repo_path)
