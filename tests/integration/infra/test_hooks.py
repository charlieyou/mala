"""Integration tests for hooks - real git repo interactions.

These tests use real git subprocess calls to verify hook behavior
with actual git repositories.
"""

import subprocess
from pathlib import Path

import pytest

from src.infra.hooks.lint_cache import _get_git_state

pytestmark = pytest.mark.integration


class TestGetGitState:
    """Integration tests for _get_git_state with real git repos."""

    def test_returns_hash_for_git_repo(self, tmp_path: Path) -> None:
        """_get_git_state returns a hash string for a valid git repo."""
        # Create a git repo
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        (tmp_path / "file.py").write_text("print('hello')")
        subprocess.run(
            ["git", "add", "."], cwd=tmp_path, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "initial"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

        result = _get_git_state(tmp_path)

        assert result is not None
        assert len(result) == 16  # SHA-256 hash truncated to 16 chars

    def test_returns_none_for_non_git_directory(self, tmp_path: Path) -> None:
        """_get_git_state returns None for a non-git directory."""
        result = _get_git_state(tmp_path)

        assert result is None

    def test_hash_changes_after_file_modification(self, tmp_path: Path) -> None:
        """_get_git_state returns different hash after uncommitted changes."""
        # Create a git repo
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        test_file = tmp_path / "file.py"
        test_file.write_text("print('hello')")
        subprocess.run(
            ["git", "add", "."], cwd=tmp_path, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "initial"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

        hash_before = _get_git_state(tmp_path)

        # Modify file (uncommitted change)
        test_file.write_text("print('world')")

        hash_after = _get_git_state(tmp_path)

        assert hash_before is not None
        assert hash_after is not None
        assert hash_before != hash_after
