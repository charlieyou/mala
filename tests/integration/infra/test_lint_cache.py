"""Integration tests for LintCache - real git repo interactions.

These tests use real git subprocess calls to verify LintCache behavior
with actual git repositories.
"""

import subprocess
from pathlib import Path

import pytest

from src.domain.validation.lint_cache import LintCache
from src.infra.tools.command_runner import CommandRunner

pytestmark = pytest.mark.integration


class TestLintCacheWithRealGit:
    """Integration tests for LintCache using real git repos."""

    @pytest.fixture
    def cache_dir(self, tmp_path: Path) -> Path:
        """Provide a temporary cache directory."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        return cache_dir

    @pytest.fixture
    def git_repo(self, tmp_path: Path) -> Path:
        """Create a minimal git repo for testing."""
        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=repo,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=repo,
            check=True,
            capture_output=True,
        )
        # Create an initial commit
        (repo / "file.py").write_text("# initial\n")
        subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "initial"],
            cwd=repo,
            check=True,
            capture_output=True,
        )
        return repo

    @pytest.fixture
    def command_runner(self, git_repo: Path) -> CommandRunner:
        """Provide a command runner for tests."""
        return CommandRunner(cwd=git_repo)

    def test_should_skip_returns_true_after_mark_passed(
        self, cache_dir: Path, git_repo: Path, command_runner: CommandRunner
    ) -> None:
        """Test that should_skip returns True after marking command as passed."""
        cache = LintCache(
            cache_dir=cache_dir, repo_path=git_repo, command_runner=command_runner
        )

        # First run - should not skip
        assert cache.should_skip("ruff check") is False

        # Mark as passed
        cache.mark_passed("ruff check")

        # Second run - should skip
        assert cache.should_skip("ruff check") is True

    def test_should_skip_returns_false_after_new_commit(
        self, cache_dir: Path, git_repo: Path, command_runner: CommandRunner
    ) -> None:
        """Test that should_skip returns False after a new commit."""
        cache = LintCache(
            cache_dir=cache_dir, repo_path=git_repo, command_runner=command_runner
        )

        # Mark as passed
        cache.mark_passed("ruff check")
        assert cache.should_skip("ruff check") is True

        # Make a new commit
        (git_repo / "new_file.py").write_text("# new\n")
        subprocess.run(
            ["git", "add", "."], cwd=git_repo, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "new commit"],
            cwd=git_repo,
            check=True,
            capture_output=True,
        )

        # Should not skip after new commit
        assert cache.should_skip("ruff check") is False

    def test_should_skip_returns_false_after_uncommitted_change(
        self, cache_dir: Path, git_repo: Path, command_runner: CommandRunner
    ) -> None:
        """Test that should_skip returns False after uncommitted changes."""
        cache = LintCache(
            cache_dir=cache_dir, repo_path=git_repo, command_runner=command_runner
        )

        # Mark as passed (clean state)
        cache.mark_passed("ruff check")
        assert cache.should_skip("ruff check") is True

        # Make an uncommitted change
        (git_repo / "file.py").write_text("# modified\n")

        # Should not skip after uncommitted change
        assert cache.should_skip("ruff check") is False

    def test_cache_persists_across_instances(
        self, cache_dir: Path, git_repo: Path, command_runner: CommandRunner
    ) -> None:
        """Test that cache entries persist to disk and can be loaded."""
        cache1 = LintCache(
            cache_dir=cache_dir, repo_path=git_repo, command_runner=command_runner
        )
        cache1.mark_passed("ruff check")

        # Create new instance - should load from disk
        cache2 = LintCache(
            cache_dir=cache_dir, repo_path=git_repo, command_runner=command_runner
        )
        assert cache2.should_skip("ruff check") is True

    def test_invalidate_removes_entry(
        self, cache_dir: Path, git_repo: Path, command_runner: CommandRunner
    ) -> None:
        """Test that invalidate removes the cache entry."""
        cache = LintCache(
            cache_dir=cache_dir, repo_path=git_repo, command_runner=command_runner
        )
        cache.mark_passed("ruff check")
        assert cache.should_skip("ruff check") is True

        cache.invalidate("ruff check")
        assert cache.should_skip("ruff check") is False

    def test_clear_removes_all_entries(
        self, cache_dir: Path, git_repo: Path, command_runner: CommandRunner
    ) -> None:
        """Test that clear removes all cache entries."""
        cache = LintCache(
            cache_dir=cache_dir, repo_path=git_repo, command_runner=command_runner
        )
        cache.mark_passed("ruff check")
        cache.mark_passed("ty check")

        cache.clear()

        assert cache.should_skip("ruff check") is False
        assert cache.should_skip("ty check") is False

    def test_different_commands_cached_separately(
        self, cache_dir: Path, git_repo: Path, command_runner: CommandRunner
    ) -> None:
        """Test that different commands have separate cache entries."""
        cache = LintCache(
            cache_dir=cache_dir, repo_path=git_repo, command_runner=command_runner
        )

        cache.mark_passed("ruff check")

        assert cache.should_skip("ruff check") is True
        assert cache.should_skip("ty check") is False

    def test_handles_corrupted_cache_file(
        self, cache_dir: Path, git_repo: Path, command_runner: CommandRunner
    ) -> None:
        """Test that corrupted cache file is handled gracefully."""
        cache_file = cache_dir / "lint_cache.json"
        cache_file.write_text("not valid json {{{")

        # Should not raise, just start with empty cache
        cache = LintCache(
            cache_dir=cache_dir, repo_path=git_repo, command_runner=command_runner
        )
        assert cache.should_skip("ruff check") is False

    def test_uncommitted_changes_tracked_by_content(
        self, cache_dir: Path, git_repo: Path, command_runner: CommandRunner
    ) -> None:
        """Test that different uncommitted changes invalidate cache."""
        cache = LintCache(
            cache_dir=cache_dir, repo_path=git_repo, command_runner=command_runner
        )

        # Make uncommitted change and mark as passed
        (git_repo / "file.py").write_text("# change 1\n")
        cache.mark_passed("ruff check")
        assert cache.should_skip("ruff check") is True

        # Different uncommitted change - should not skip
        (git_repo / "file.py").write_text("# change 2\n")
        assert cache.should_skip("ruff check") is False
