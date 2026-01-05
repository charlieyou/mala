"""Unit tests for LintCache - skipping redundant lint commands.

Note: Integration tests that use real git subprocess calls are in
tests/integration/infra/test_lint_cache.py.
"""

from pathlib import Path

import pytest

from src.domain.validation.lint_cache import LintCache, LintCacheEntry, LintCacheKey
from src.infra.tools.command_runner import CommandResult
from tests.fakes.command_runner import FakeCommandRunner


class TestLintCacheKey:
    """Tests for LintCacheKey."""

    def test_to_dict_roundtrip(self) -> None:
        """Test that to_dict and from_dict are inverses."""
        key = LintCacheKey(command_name="ruff check", working_dir="/my/repo")
        data = key.to_dict()
        restored = LintCacheKey.from_dict(data)
        assert restored == key


class TestLintCacheEntry:
    """Tests for LintCacheEntry."""

    def test_to_dict_roundtrip(self) -> None:
        """Test that to_dict and from_dict are inverses."""
        entry = LintCacheEntry(
            head_sha="abc123",
            has_uncommitted=True,
            files_hash="def456",
        )
        data = entry.to_dict()
        restored = LintCacheEntry.from_dict(data)
        assert restored == entry

    def test_to_dict_with_none_files_hash(self) -> None:
        """Test serialization when files_hash is None."""
        entry = LintCacheEntry(
            head_sha="abc123",
            has_uncommitted=False,
            files_hash=None,
        )
        data = entry.to_dict()
        assert data["files_hash"] is None
        restored = LintCacheEntry.from_dict(data)
        assert restored.files_hash is None


class TestLintCacheWithMocks:
    """Tests for LintCache using mocks for git commands."""

    @pytest.fixture
    def cache_dir(self, tmp_path: Path) -> Path:
        """Provide a temporary cache directory."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        return cache_dir

    @pytest.fixture
    def repo_path(self, tmp_path: Path) -> Path:
        """Provide a fake repo path."""
        repo = tmp_path / "repo"
        repo.mkdir()
        return repo

    @pytest.fixture
    def failing_command_runner(self) -> FakeCommandRunner:
        """Provide a command runner that simulates git failures (non-git repo)."""
        runner = FakeCommandRunner()
        # Simulate git commands failing (non-git repo)
        runner.responses[("git", "rev-parse", "HEAD")] = CommandResult(
            command=["git", "rev-parse", "HEAD"],
            returncode=128,
            stdout="",
            stderr="fatal: not a git repository",
        )
        return runner

    def test_handles_non_git_repo(
        self,
        cache_dir: Path,
        repo_path: Path,
        failing_command_runner: FakeCommandRunner,
    ) -> None:
        """Test behavior when repo is not a git repository."""
        # git commands will fail since we configured the fake to return errors
        cache = LintCache(
            cache_dir=cache_dir,
            repo_path=repo_path,
            command_runner=failing_command_runner,
        )

        # mark_passed should not crash, but also shouldn't cache anything
        # since git state can't be determined
        cache.mark_passed("ruff check")

        # should_skip must return False when git state is unavailable
        assert cache.should_skip("ruff check") is False

        # Verify no cache file was written (mark_passed returns early)
        assert not (cache_dir / "lint_cache.json").exists()
