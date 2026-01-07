"""Unit tests for EpicScopeAnalyzer."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from src.infra.epic_scope import EpicScopeAnalyzer, ScopedCommits
from src.infra.tools.command_runner import CommandResult

if TYPE_CHECKING:
    from pathlib import Path


# ============================================================================
# Fixtures
# ============================================================================


def make_mock_runner() -> Mock:
    """Create a mock CommandRunnerPort."""
    mock = Mock()
    mock.run.return_value = CommandResult(
        command=["mock"], returncode=0, stdout="", stderr=""
    )
    return mock


@pytest.fixture
def analyzer(tmp_path: Path) -> EpicScopeAnalyzer:
    """Create an EpicScopeAnalyzer with a temporary repo path."""
    return EpicScopeAnalyzer(repo_path=tmp_path, runner=make_mock_runner())


# ============================================================================
# _compute_commit_list tests
# ============================================================================


class TestComputeCommitList:
    """Tests for _compute_commit_list method."""

    @pytest.mark.asyncio
    async def test_searches_with_issue_prefix(
        self, analyzer: EpicScopeAnalyzer
    ) -> None:
        """Should search for commits using bd-<issue_id>: prefix."""
        commands_run: list[list[str]] = []

        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            commands_run.append(cmd)
            return CommandResult(command=cmd, returncode=0, stdout="")

        analyzer._runner.run_async = mock_run_async  # type: ignore[method-assign]

        await analyzer._compute_commit_list({"child-1"})

        # Verify git log was called with grep pattern
        log_cmds = [c for c in commands_run if "log" in c]
        assert len(log_cmds) == 1
        assert "--grep=bd-child-1:" in log_cmds[0]

    @pytest.mark.asyncio
    async def test_skips_merge_commits(self, analyzer: EpicScopeAnalyzer) -> None:
        """Should use --no-merges flag to skip merge commits."""
        commands_run: list[list[str]] = []

        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            commands_run.append(cmd)
            return CommandResult(command=cmd, returncode=0, stdout="")

        analyzer._runner.run_async = mock_run_async  # type: ignore[method-assign]

        await analyzer._compute_commit_list({"child-1"})

        # Verify --no-merges was used
        log_cmds = [c for c in commands_run if "log" in c]
        assert len(log_cmds) > 0
        assert "--no-merges" in log_cmds[0]

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_commits(
        self, analyzer: EpicScopeAnalyzer
    ) -> None:
        """Should return empty list when no commits found."""

        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            return CommandResult(command=cmd, returncode=0, stdout="")

        analyzer._runner.run_async = mock_run_async  # type: ignore[method-assign]

        commits = await analyzer._compute_commit_list({"child-1"})
        assert commits == []

    @pytest.mark.asyncio
    async def test_handles_multiple_commits_per_session(
        self, analyzer: EpicScopeAnalyzer
    ) -> None:
        """Should include all commits matching an issue prefix."""

        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            if "log" in cmd:
                return CommandResult(
                    command=cmd,
                    returncode=0,
                    stdout="commit1\ncommit2\ncommit3",
                )
            return CommandResult(command=cmd, returncode=0, stdout="")

        analyzer._runner.run_async = mock_run_async  # type: ignore[method-assign]

        commits = await analyzer._compute_commit_list({"child-1"})
        assert commits == ["commit1", "commit2", "commit3"]

    @pytest.mark.asyncio
    async def test_deduplicates_commits(self, analyzer: EpicScopeAnalyzer) -> None:
        """Should deduplicate commits that appear in multiple issues."""

        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            if "log" in cmd:
                # Batched call returns commits from both issues, with shared one duplicated
                return CommandResult(
                    command=cmd,
                    returncode=0,
                    stdout="shared-commit\nunique-commit-1\nshared-commit\nunique-commit-2",
                )
            return CommandResult(command=cmd, returncode=0, stdout="")

        analyzer._runner.run_async = mock_run_async  # type: ignore[method-assign]

        commits = await analyzer._compute_commit_list({"child-1", "child-2"})
        # Should have 3 unique commits, not 4
        assert len(commits) == 3
        assert "shared-commit" in commits

    @pytest.mark.asyncio
    async def test_includes_blocker_commits(self, analyzer: EpicScopeAnalyzer) -> None:
        """Should include commits from blocker issues."""
        commands_run: list[list[str]] = []

        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            commands_run.append(cmd)
            if "log" in cmd:
                # Batched call with both patterns returns both commits
                return CommandResult(
                    command=cmd, returncode=0, stdout="child-commit\nblocker-commit"
                )
            return CommandResult(command=cmd, returncode=0, stdout="")

        analyzer._runner.run_async = mock_run_async  # type: ignore[method-assign]

        commits = await analyzer._compute_commit_list(
            {"child-1"}, blocker_ids={"blocker-1"}
        )
        assert "child-commit" in commits
        assert "blocker-commit" in commits
        # Verify both patterns are in the batched command
        log_cmd = next(c for c in commands_run if "log" in c)
        assert "--grep=bd-child-1:" in log_cmd
        assert "--grep=bd-blocker-1:" in log_cmd


# ============================================================================
# _summarize_commit_range tests
# ============================================================================


class TestSummarizeCommitRange:
    """Tests for _summarize_commit_range method."""

    @pytest.mark.asyncio
    async def test_returns_none_for_empty_commits(
        self, analyzer: EpicScopeAnalyzer
    ) -> None:
        """Should return None when commits list is empty."""
        result = await analyzer._summarize_commit_range([])
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_single_commit(self, analyzer: EpicScopeAnalyzer) -> None:
        """Should return single commit SHA when only one commit."""

        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            if "show" in cmd and "--format=%H %ct" in cmd:
                # Batched format: SHA timestamp
                return CommandResult(
                    command=cmd, returncode=0, stdout="abc123 1234567890"
                )
            return CommandResult(command=cmd, returncode=0, stdout="")

        analyzer._runner.run_async = mock_run_async  # type: ignore[method-assign]

        result = await analyzer._summarize_commit_range(["abc123"])
        assert result == "abc123"

    @pytest.mark.asyncio
    async def test_returns_range_for_multiple_commits(
        self, analyzer: EpicScopeAnalyzer
    ) -> None:
        """Should return range format for multiple commits."""

        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            if "show" in cmd and "--format=%H %ct" in cmd:
                # Batched format: all commits with timestamps in one response
                return CommandResult(
                    command=cmd,
                    returncode=0,
                    stdout="abc123 1000\ndef456 2000\nghi789 1500",
                )
            if "rev-parse" in cmd:
                return CommandResult(command=cmd, returncode=0, stdout="parent-sha")
            return CommandResult(command=cmd, returncode=0, stdout="")

        analyzer._runner.run_async = mock_run_async  # type: ignore[method-assign]

        result = await analyzer._summarize_commit_range(["abc123", "def456", "ghi789"])
        # abc123 is oldest (1000), def456 is newest (2000)
        assert result == "abc123^..def456"

    @pytest.mark.asyncio
    async def test_handles_root_commit(self, analyzer: EpicScopeAnalyzer) -> None:
        """Should handle root commit without parent."""

        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            if "show" in cmd and "--format=%H %ct" in cmd:
                # Batched format: all commits with timestamps in one response
                return CommandResult(
                    command=cmd,
                    returncode=0,
                    stdout="root123 1000\nchild456 2000",
                )
            if "rev-parse" in cmd:
                # Root commit has no parent
                return CommandResult(command=cmd, returncode=1, stdout="", stderr="")
            return CommandResult(command=cmd, returncode=0, stdout="")

        analyzer._runner.run_async = mock_run_async  # type: ignore[method-assign]

        result = await analyzer._summarize_commit_range(["root123", "child456"])
        # Should use base..tip format for root commit
        assert result == "root123..child456"

    @pytest.mark.asyncio
    async def test_returns_none_when_timestamps_unavailable(
        self, analyzer: EpicScopeAnalyzer
    ) -> None:
        """Should return None when timestamps cannot be retrieved."""

        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            if "show" in cmd:
                # Return non-numeric output
                return CommandResult(
                    command=cmd, returncode=1, stdout="", stderr="error"
                )
            return CommandResult(command=cmd, returncode=0, stdout="")

        analyzer._runner.run_async = mock_run_async  # type: ignore[method-assign]

        result = await analyzer._summarize_commit_range(["abc123", "def456"])
        assert result is None


# ============================================================================
# _format_commit_summary tests
# ============================================================================


class TestFormatCommitSummary:
    """Tests for _format_commit_summary method."""

    @pytest.mark.asyncio
    async def test_returns_no_commits_message(
        self, analyzer: EpicScopeAnalyzer
    ) -> None:
        """Should return 'No commits found.' for empty list."""
        result = await analyzer._format_commit_summary([])
        assert result == "No commits found."

    @pytest.mark.asyncio
    async def test_formats_commits_with_subjects(
        self, analyzer: EpicScopeAnalyzer
    ) -> None:
        """Should format commits with SHA and subject."""

        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            if "show" in cmd and "--format=%H %s" in cmd:
                # Batched format: all commits with subjects in one response
                return CommandResult(
                    command=cmd,
                    returncode=0,
                    stdout="abc123 Fix bug in parser\ndef456 Add new feature",
                )
            return CommandResult(command=cmd, returncode=0, stdout="")

        analyzer._runner.run_async = mock_run_async  # type: ignore[method-assign]

        result = await analyzer._format_commit_summary(["abc123", "def456"])
        assert "- abc123 Fix bug in parser" in result
        assert "- def456 Add new feature" in result

    @pytest.mark.asyncio
    async def test_truncates_long_lists(self, analyzer: EpicScopeAnalyzer) -> None:
        """Should truncate commit list when exceeding max_commits."""

        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            if "show" in cmd:
                return CommandResult(command=cmd, returncode=0, stdout="sha subject")
            return CommandResult(command=cmd, returncode=0, stdout="")

        analyzer._runner.run_async = mock_run_async  # type: ignore[method-assign]

        # Create 10 commits, limit to 5
        commits = [f"commit{i}" for i in range(10)]
        result = await analyzer._format_commit_summary(commits, max_commits=5)
        assert "5 more commits omitted" in result

    @pytest.mark.asyncio
    async def test_falls_back_to_sha_on_error(
        self, analyzer: EpicScopeAnalyzer
    ) -> None:
        """Should fall back to SHA when git show fails."""

        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            if "show" in cmd:
                return CommandResult(
                    command=cmd, returncode=1, stdout="", stderr="error"
                )
            return CommandResult(command=cmd, returncode=0, stdout="")

        analyzer._runner.run_async = mock_run_async  # type: ignore[method-assign]

        result = await analyzer._format_commit_summary(["abc123"])
        assert "- abc123" in result


# ============================================================================
# compute_scoped_commits integration tests
# ============================================================================


class TestComputeScopedCommits:
    """Tests for compute_scoped_commits public method."""

    @pytest.mark.asyncio
    async def test_returns_scoped_commits_result(
        self, analyzer: EpicScopeAnalyzer
    ) -> None:
        """Should return ScopedCommits with all fields populated."""

        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            if "log" in cmd:
                return CommandResult(command=cmd, returncode=0, stdout="abc123\ndef456")
            if "show" in cmd and "--format=%H %ct" in cmd:
                # Batched format: all commits with timestamps
                return CommandResult(
                    command=cmd, returncode=0, stdout="abc123 1000\ndef456 2000"
                )
            if "show" in cmd and "--format=%H %s" in cmd:
                # Batched format: all commits with subjects
                return CommandResult(
                    command=cmd,
                    returncode=0,
                    stdout="abc123 First commit\ndef456 Second commit",
                )
            if "rev-parse" in cmd:
                return CommandResult(command=cmd, returncode=0, stdout="parent")
            return CommandResult(command=cmd, returncode=0, stdout="")

        analyzer._runner.run_async = mock_run_async  # type: ignore[method-assign]

        result = await analyzer.compute_scoped_commits({"child-1"})

        assert isinstance(result, ScopedCommits)
        assert result.commit_shas == ["abc123", "def456"]
        assert result.commit_range == "abc123^..def456"
        assert "First commit" in result.commit_summary
        assert "Second commit" in result.commit_summary

    @pytest.mark.asyncio
    async def test_handles_no_commits(self, analyzer: EpicScopeAnalyzer) -> None:
        """Should handle case with no commits found."""

        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            return CommandResult(command=cmd, returncode=0, stdout="")

        analyzer._runner.run_async = mock_run_async  # type: ignore[method-assign]

        result = await analyzer.compute_scoped_commits({"child-1"})

        assert result.commit_shas == []
        assert result.commit_range is None
        assert result.commit_summary == "No commits found."
