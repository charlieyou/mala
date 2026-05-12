"""E2E smoke checks for the Cerberus v2 binary and reviewer adapter."""

from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from typing import TYPE_CHECKING

import pytest

from src.infra.clients.cerberus_review import DefaultReviewer

if TYPE_CHECKING:
    from pathlib import Path


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        shutil.which("cerberus") is None,
        reason="cerberus binary not found in PATH",
    ),
]


def _run_git(repo_path: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _setup_git_repo(repo_path: Path) -> str:
    """Initialize a git repo with one reviewable commit and return its SHA."""
    _run_git(repo_path, "init")
    _run_git(repo_path, "config", "user.email", "test@test.com")
    _run_git(repo_path, "config", "user.name", "Test")
    _run_git(repo_path, "config", "commit.gpgsign", "false")

    (repo_path / "main.py").write_text("# Initial file\n", encoding="utf-8")
    _run_git(repo_path, "add", "main.py")
    _run_git(repo_path, "commit", "-m", "Initial commit")

    (repo_path / "main.py").write_text(
        "# Initial file\n\ndef hello():\n    print('hello')\n",
        encoding="utf-8",
    )
    _run_git(repo_path, "add", "main.py")
    _run_git(repo_path, "commit", "-m", "Add hello function")
    return _run_git(repo_path, "rev-parse", "HEAD")


def _setup_git_repo_with_commits(repo_path: Path) -> list[str]:
    """Initialize a git repo with three commits and return their SHAs."""
    _run_git(repo_path, "init")
    _run_git(repo_path, "config", "user.email", "test@test.com")
    _run_git(repo_path, "config", "user.name", "Test")
    _run_git(repo_path, "config", "commit.gpgsign", "false")

    (repo_path / "main.py").write_text("# Initial file\n", encoding="utf-8")
    _run_git(repo_path, "add", "main.py")
    _run_git(repo_path, "commit", "-m", "Initial commit")
    first_sha = _run_git(repo_path, "rev-parse", "HEAD")

    (repo_path / "main.py").write_text(
        "# Initial file\n\ndef hello():\n    print('hello')\n",
        encoding="utf-8",
    )
    _run_git(repo_path, "add", "main.py")
    _run_git(repo_path, "commit", "-m", "Add hello function")
    second_sha = _run_git(repo_path, "rev-parse", "HEAD")

    (repo_path / "main.py").write_text(
        "# Initial file\n\n"
        "def hello():\n"
        "    print('hello')\n\n"
        "def other_agent():\n"
        "    print('other agent')\n",
        encoding="utf-8",
    )
    _run_git(repo_path, "add", "main.py")
    _run_git(repo_path, "commit", "-m", "Add other agent change")
    third_sha = _run_git(repo_path, "rev-parse", "HEAD")

    return [first_sha, second_sha, third_sha]


def _cerberus_env() -> dict[str, str]:
    return {"PATH": os.environ.get("PATH", "")}


def _reviewer(repo_path: Path) -> DefaultReviewer:
    return DefaultReviewer(
        repo_path=repo_path,
        state_root=repo_path / ".mala" / "cerberus",
        project_key=f"test-{uuid.uuid4()}",
        spawn_args=("--mode=fast",),
        env=_cerberus_env(),
    )


def test_cerberus_binary_available() -> None:
    """The v2 cerberus binary is available for real e2e runs."""
    result = subprocess.run(
        ["cerberus", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0


@pytest.mark.asyncio
async def test_review_gate_full_flow(tmp_path: Path) -> None:
    """DefaultReviewer can drive the real Cerberus v2 review flow."""
    head_sha = _setup_git_repo(tmp_path)
    result = await _reviewer(tmp_path)(
        claude_session_id=f"test-{uuid.uuid4()}",
        timeout=180,
        commit_shas=[head_sha],
    )

    assert result.fatal_error is False, f"Fatal error: {result.parse_error}"


@pytest.mark.asyncio
async def test_review_gate_commits_scope(tmp_path: Path) -> None:
    """Review should accept and run against explicit commit scopes."""
    commits = _setup_git_repo_with_commits(tmp_path)
    result = await _reviewer(tmp_path)(
        claude_session_id=f"test-{uuid.uuid4()}",
        timeout=180,
        commit_shas=[commits[1]],
    )

    assert result.fatal_error is False, f"Fatal error: {result.parse_error}"


@pytest.mark.asyncio
async def test_review_gate_empty_commit_shortcircuit(tmp_path: Path) -> None:
    """Empty commit list should short-circuit to PASS without spawning reviewers."""
    _run_git(tmp_path, "init")
    _run_git(tmp_path, "config", "user.email", "test@test.com")
    _run_git(tmp_path, "config", "user.name", "Test")
    _run_git(tmp_path, "config", "commit.gpgsign", "false")
    (tmp_path / "main.py").write_text("# File\n", encoding="utf-8")
    _run_git(tmp_path, "add", "main.py")
    _run_git(tmp_path, "commit", "-m", "Initial")

    result = await _reviewer(tmp_path)(
        claude_session_id=f"test-{uuid.uuid4()}",
        commit_shas=[],
    )

    assert result.passed is True
    assert result.parse_error is None
    assert result.issues == []
