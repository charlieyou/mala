"""Unit tests for LintCache - skipping redundant lint commands.

Note: Integration tests that use real git subprocess calls are in
tests/integration/infra/test_lint_cache.py.
"""

from pathlib import Path

import pytest

from src.domain.validation.lint_cache import LintCache, LintCacheEntry, LintCacheKey
from src.infra.tools.command_runner import CommandRunner


class TestLintCacheKey:
    """Tests for LintCacheKey."""

    def test_to_dict_roundtrip(self) -> None:
        """Test that to_dict and from_dict are inverses."""
        key = LintCacheKey(command_name="ruff check", working_dir="/my/repo")
        data = key.to_dict()
        restored = LintCacheKey.from_dict(data)
        assert restored == key

    def test_frozen(self) -> None:
        """Test that LintCacheKey is immutable."""
        key = LintCacheKey(command_name="ruff check", working_dir="/my/repo")
        with pytest.raises(AttributeError):
            key.command_name = "something else"  # type: ignore[misc]


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
    def command_runner(self, repo_path: Path) -> CommandRunner:
        """Provide a command runner for tests."""
        return CommandRunner(cwd=repo_path)

    def test_handles_non_git_repo(
        self, cache_dir: Path, repo_path: Path, command_runner: CommandRunner
    ) -> None:
        """Test behavior when repo is not a git repository."""
        # git commands will fail since it's not a real git repo
        cache = LintCache(
            cache_dir=cache_dir, repo_path=repo_path, command_runner=command_runner
        )

        # Should not crash, just return False (can't verify state)
        cache.mark_passed("ruff check")
        # With empty head_sha, cache will still work based on what we get
        # The key point is it doesn't crash
        result = cache.should_skip("ruff check")
        assert isinstance(result, bool)
