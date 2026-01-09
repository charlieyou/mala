from pathlib import Path

import pytest

from src.infra import git_utils
from src.infra import git_utils as infra_git_utils
from src.infra.git_utils import DiffStat
from src.infra.tools.command_runner import CommandResult


@pytest.mark.asyncio
async def test_get_git_commit_async_success(monkeypatch: pytest.MonkeyPatch) -> None:
    async def mock_run_command_async(
        cmd: list[str], cwd: Path, timeout_seconds: float | None = None
    ) -> CommandResult:
        return CommandResult(command=cmd, returncode=0, stdout="deadbeef\n")

    # Patch in the infra module where the function is actually used
    monkeypatch.setattr(infra_git_utils, "run_command_async", mock_run_command_async)

    result = await git_utils.get_git_commit_async(Path("."))

    assert result == "deadbeef"


@pytest.mark.asyncio
async def test_get_git_branch_async_success(monkeypatch: pytest.MonkeyPatch) -> None:
    async def mock_run_command_async(
        cmd: list[str], cwd: Path, timeout_seconds: float | None = None
    ) -> CommandResult:
        return CommandResult(command=cmd, returncode=0, stdout="main\n")

    # Patch in the infra module where the function is actually used
    monkeypatch.setattr(infra_git_utils, "run_command_async", mock_run_command_async)

    result = await git_utils.get_git_branch_async(Path("."))

    assert result == "main"


@pytest.mark.asyncio
async def test_get_git_commit_async_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    async def mock_run_command_async(
        cmd: list[str], cwd: Path, timeout_seconds: float | None = None
    ) -> CommandResult:
        # Simulate timeout - command runner returns timed_out=True and exit code 124
        return CommandResult(
            command=cmd, returncode=124, stdout="", stderr="", timed_out=True
        )

    # Patch in the infra module where the function is actually used
    monkeypatch.setattr(infra_git_utils, "run_command_async", mock_run_command_async)

    result = await git_utils.get_git_commit_async(Path("."), timeout=0.01)

    assert result == ""


# --- Tests for is_commit_reachable ---


@pytest.mark.asyncio
async def test_is_commit_reachable_returns_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """is_commit_reachable returns True when commit exists locally."""

    async def mock_run_command_async(
        cmd: list[str], cwd: Path, timeout_seconds: float | None = None
    ) -> CommandResult:
        # git cat-file -e returns 0 for existing objects
        assert cmd == ["git", "cat-file", "-e", "abc1234"]
        return CommandResult(command=cmd, returncode=0, stdout="")

    monkeypatch.setattr(infra_git_utils, "run_command_async", mock_run_command_async)

    result = await git_utils.is_commit_reachable(Path("/repo"), "abc1234")

    assert result is True


@pytest.mark.asyncio
async def test_is_commit_reachable_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """is_commit_reachable returns False when commit doesn't exist locally."""

    async def mock_run_command_async(
        cmd: list[str], cwd: Path, timeout_seconds: float | None = None
    ) -> CommandResult:
        # git cat-file -e returns non-zero for missing objects
        assert cmd == ["git", "cat-file", "-e", "missing123"]
        return CommandResult(
            command=cmd,
            returncode=128,
            stdout="",
            stderr="fatal: Not a valid object name missing123",
        )

    monkeypatch.setattr(infra_git_utils, "run_command_async", mock_run_command_async)

    result = await git_utils.is_commit_reachable(Path("/repo"), "missing123")

    assert result is False


# --- Tests for get_baseline_for_issue ---


class MockCommandRunner:
    """Mock CommandRunner for testing."""

    def __init__(self, responses: list[CommandResult]) -> None:
        self.responses = responses
        self.call_index = 0
        self.calls: list[list[str]] = []

    async def run_async(self, cmd: list[str]) -> CommandResult:
        self.calls.append(cmd)
        result = self.responses[self.call_index]
        self.call_index += 1
        return result


class TestGetBaselineForIssue:
    """Tests for get_baseline_for_issue() function.

    The function should:
    - Return parent of first commit with "bd-{issue_id}:" prefix if exists
    - Return None if no commits exist for the issue
    - Handle edge cases like root commits, merge commits, rebased history
    """

    @pytest.mark.asyncio
    async def test_fresh_issue_no_prior_commits(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fresh issue with no commits should return None."""
        mock_runner = MockCommandRunner(
            responses=[
                # git log returns empty (no matching commits)
                CommandResult(command=["git", "log"], returncode=0, stdout=""),
            ]
        )

        # Patch in the infra module where CommandRunner is actually used
        monkeypatch.setattr(
            infra_git_utils, "CommandRunner", lambda cwd, timeout_seconds: mock_runner
        )

        result = await git_utils.get_baseline_for_issue(Path("/repo"), "mala-123")

        assert result is None
        # Verify git log was called with correct grep pattern
        assert len(mock_runner.calls) == 1
        # re.escape() escapes hyphens too, so pattern is mala\-123
        assert r"--grep=^bd-mala\-123:" in mock_runner.calls[0]

    @pytest.mark.asyncio
    async def test_resumed_issue_with_prior_commits(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Resumed issue should return parent of first matching commit."""
        mock_runner = MockCommandRunner(
            responses=[
                # git log returns two commits for this issue
                CommandResult(
                    command=["git", "log"],
                    returncode=0,
                    stdout="abc1234 bd-mala-123: First commit\ndef5678 bd-mala-123: Second commit",
                ),
                # git rev-parse returns parent of first commit
                CommandResult(
                    command=["git", "rev-parse"], returncode=0, stdout="parent99\n"
                ),
            ]
        )

        # Patch in the infra module where CommandRunner is actually used
        monkeypatch.setattr(
            infra_git_utils, "CommandRunner", lambda cwd, timeout_seconds: mock_runner
        )

        result = await git_utils.get_baseline_for_issue(Path("/repo"), "mala-123")

        assert result == "parent99"
        # Verify rev-parse was called with correct commit^
        assert len(mock_runner.calls) == 2
        assert "abc1234^" in mock_runner.calls[1]

    @pytest.mark.asyncio
    async def test_root_commit_no_parent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If first commit is root (no parent), should return None."""
        mock_runner = MockCommandRunner(
            responses=[
                # git log returns a commit
                CommandResult(
                    command=["git", "log"],
                    returncode=0,
                    stdout="abc1234 bd-mala-123: Root commit",
                ),
                # git rev-parse fails (root commit has no parent)
                CommandResult(
                    command=["git", "rev-parse"], returncode=128, stdout="", stderr=""
                ),
            ]
        )

        # Patch in the infra module where CommandRunner is actually used
        monkeypatch.setattr(
            infra_git_utils, "CommandRunner", lambda cwd, timeout_seconds: mock_runner
        )

        result = await git_utils.get_baseline_for_issue(Path("/repo"), "mala-123")

        assert result is None


class TestGetIssueCommitsAsync:
    """Tests for get_issue_commits_async() function."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_commits(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_runner = MockCommandRunner(
            responses=[
                CommandResult(command=["git", "log"], returncode=0, stdout=""),
            ]
        )
        # Patch in the infra module where CommandRunner is actually used
        monkeypatch.setattr(
            infra_git_utils, "CommandRunner", lambda cwd, timeout_seconds: mock_runner
        )

        result = await git_utils.get_issue_commits_async(Path("/repo"), "mala-123")

        assert result == []
        assert r"--grep=^bd-mala\-123:" in mock_runner.calls[0]

    @pytest.mark.asyncio
    async def test_returns_commits_in_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_runner = MockCommandRunner(
            responses=[
                CommandResult(
                    command=["git", "log"],
                    returncode=0,
                    stdout="aaa111\nbbb222\nccc333\n",
                ),
            ]
        )
        # Patch in the infra module where CommandRunner is actually used
        monkeypatch.setattr(
            infra_git_utils, "CommandRunner", lambda cwd, timeout_seconds: mock_runner
        )

        result = await git_utils.get_issue_commits_async(Path("/repo"), "mala-123")

        assert result == ["aaa111", "bbb222", "ccc333"]

    @pytest.mark.asyncio
    async def test_since_timestamp_is_applied(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_runner = MockCommandRunner(
            responses=[
                CommandResult(
                    command=["git", "log"],
                    returncode=0,
                    stdout="aaa111\n",
                ),
            ]
        )
        # Patch in the infra module where CommandRunner is actually used
        monkeypatch.setattr(
            infra_git_utils, "CommandRunner", lambda cwd, timeout_seconds: mock_runner
        )

        await git_utils.get_issue_commits_async(
            Path("/repo"), "mala-123", since_timestamp=1703502000
        )

        assert "--since=@1703502000" in mock_runner.calls[0]

    @pytest.mark.asyncio
    async def test_timeout_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Timeout should return None gracefully."""
        mock_runner = MockCommandRunner(
            responses=[
                # Timeout on git log
                CommandResult(
                    command=["git", "log"],
                    returncode=124,
                    stdout="",
                    stderr="",
                    timed_out=True,
                ),
            ]
        )

        # Patch in the infra module where CommandRunner is actually used
        monkeypatch.setattr(
            infra_git_utils, "CommandRunner", lambda cwd, timeout_seconds: mock_runner
        )

        result = await git_utils.get_baseline_for_issue(
            Path("/repo"), "mala-123", timeout=0.01
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_single_commit_with_parent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Single commit for issue should return its parent."""
        mock_runner = MockCommandRunner(
            responses=[
                # git log returns one commit
                CommandResult(
                    command=["git", "log"],
                    returncode=0,
                    stdout="aaa1111 bd-mala-abc: Single commit",
                ),
                # git rev-parse returns parent
                CommandResult(
                    command=["git", "rev-parse"], returncode=0, stdout="beforeaaa\n"
                ),
            ]
        )

        # Patch in the infra module where CommandRunner is actually used
        monkeypatch.setattr(
            infra_git_utils, "CommandRunner", lambda cwd, timeout_seconds: mock_runner
        )

        result = await git_utils.get_baseline_for_issue(Path("/repo"), "mala-abc")

        assert result == "beforeaaa"

    @pytest.mark.asyncio
    async def test_issue_id_with_regex_metacharacters(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Issue IDs with regex metacharacters should be escaped properly.

        Without escaping, an issue like "mala-g3h.1" would match "mala-g3hX1"
        because "." is a regex wildcard.
        """
        mock_runner = MockCommandRunner(
            responses=[
                CommandResult(
                    command=["git", "log"],
                    returncode=0,
                    stdout="abc1234 bd-mala-g3h.1: Fix the bug",
                ),
                CommandResult(
                    command=["git", "rev-parse"], returncode=0, stdout="parent123\n"
                ),
            ]
        )

        # Patch in the infra module where CommandRunner is actually used
        monkeypatch.setattr(
            infra_git_utils, "CommandRunner", lambda cwd, timeout_seconds: mock_runner
        )

        result = await git_utils.get_baseline_for_issue(Path("/repo"), "mala-g3h.1")

        assert result == "parent123"
        # Verify the grep pattern has escaped metacharacters
        assert len(mock_runner.calls) >= 1
        # re.escape() escapes both dot and hyphen: mala\-g3h\.1
        assert r"--grep=^bd-mala\-g3h\.1:" in mock_runner.calls[0]

    @pytest.mark.asyncio
    async def test_merge_commit_still_finds_first_issue_commit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Merge commits in history should not affect finding the first issue commit.

        Even if the issue has merge commits, we should find the first commit
        with the bd-{issue_id}: prefix and return its parent.
        """
        mock_runner = MockCommandRunner(
            responses=[
                CommandResult(
                    command=["git", "log"],
                    returncode=0,
                    stdout="first11 bd-mala-merge: Initial\n"
                    "merge22 bd-mala-merge: Merge branch 'feature'\n"
                    "third33 bd-mala-merge: More work",
                ),
                CommandResult(
                    command=["git", "rev-parse"], returncode=0, stdout="beforefirst\n"
                ),
            ]
        )

        # Patch in the infra module where CommandRunner is actually used
        monkeypatch.setattr(
            infra_git_utils, "CommandRunner", lambda cwd, timeout_seconds: mock_runner
        )

        result = await git_utils.get_baseline_for_issue(Path("/repo"), "mala-merge")

        # Should get parent of first commit, not any merge commit
        assert result == "beforefirst"

    @pytest.mark.asyncio
    async def test_rebased_history_uses_new_first_commit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After a rebase, the first commit hash changes.

        The function should find the new first commit after rebase
        and return its parent, which is the correct baseline.
        """
        mock_runner = MockCommandRunner(
            responses=[
                CommandResult(
                    command=["git", "log"],
                    returncode=0,
                    stdout="newrebase1 bd-mala-rebase: Original work (rebased)\n"
                    "newrebase2 bd-mala-rebase: Follow-up (rebased)",
                ),
                CommandResult(
                    command=["git", "rev-parse"], returncode=0, stdout="rebasebase\n"
                ),
            ]
        )

        # Patch in the infra module where CommandRunner is actually used
        monkeypatch.setattr(
            infra_git_utils, "CommandRunner", lambda cwd, timeout_seconds: mock_runner
        )

        result = await git_utils.get_baseline_for_issue(Path("/repo"), "mala-rebase")

        # Should get parent of the new (rebased) first commit
        assert result == "rebasebase"


class TestGetDiffStat:
    """Tests for get_diff_stat() function."""

    @pytest.mark.asyncio
    async def test_returns_stat_for_changes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should parse insertions and deletions from git diff --numstat output."""
        # numstat format: "added\tremoved\tfilename"
        numstat_output = "5\t5\tsrc/foo.py\n5\t0\tsrc/bar.py\n"
        mock_runner = MockCommandRunner(
            responses=[
                CommandResult(
                    command=["git", "diff", "--numstat"],
                    returncode=0,
                    stdout=numstat_output,
                ),
            ]
        )
        monkeypatch.setattr(
            infra_git_utils, "CommandRunner", lambda cwd, timeout_seconds: mock_runner
        )

        result = await git_utils.get_diff_stat(Path("/repo"), "abc123", "def456")

        assert result.total_lines == 15  # 5+5 + 5+0
        assert result.files_changed == ["src/foo.py", "src/bar.py"]
        assert mock_runner.calls[0] == ["git", "diff", "--numstat", "abc123", "def456"]

    @pytest.mark.asyncio
    async def test_empty_diff_returns_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Empty diff should return DiffStat with zero lines and empty files."""
        mock_runner = MockCommandRunner(
            responses=[
                CommandResult(
                    command=["git", "diff", "--numstat"],
                    returncode=0,
                    stdout="",
                ),
            ]
        )
        monkeypatch.setattr(
            infra_git_utils, "CommandRunner", lambda cwd, timeout_seconds: mock_runner
        )

        result = await git_utils.get_diff_stat(Path("/repo"), "abc123")

        assert result == DiffStat(total_lines=0, files_changed=[])

    @pytest.mark.asyncio
    async def test_insertions_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should handle diffs with only insertions."""
        numstat_output = "20\t0\tsrc/new.py\n"
        mock_runner = MockCommandRunner(
            responses=[
                CommandResult(
                    command=["git", "diff", "--numstat"],
                    returncode=0,
                    stdout=numstat_output,
                ),
            ]
        )
        monkeypatch.setattr(
            infra_git_utils, "CommandRunner", lambda cwd, timeout_seconds: mock_runner
        )

        result = await git_utils.get_diff_stat(Path("/repo"), "abc123")

        assert result.total_lines == 20
        assert result.files_changed == ["src/new.py"]

    @pytest.mark.asyncio
    async def test_deletions_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should handle diffs with only deletions."""
        numstat_output = "0\t15\tsrc/old.py\n"
        mock_runner = MockCommandRunner(
            responses=[
                CommandResult(
                    command=["git", "diff", "--numstat"],
                    returncode=0,
                    stdout=numstat_output,
                ),
            ]
        )
        monkeypatch.setattr(
            infra_git_utils, "CommandRunner", lambda cwd, timeout_seconds: mock_runner
        )

        result = await git_utils.get_diff_stat(Path("/repo"), "abc123")

        assert result.total_lines == 15
        assert result.files_changed == ["src/old.py"]

    @pytest.mark.asyncio
    async def test_raises_on_invalid_commit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should raise ValueError when git command fails."""
        mock_runner = MockCommandRunner(
            responses=[
                CommandResult(
                    command=["git", "diff", "--numstat"],
                    returncode=128,
                    stdout="",
                    stderr="fatal: bad revision 'invalid'",
                ),
            ]
        )
        monkeypatch.setattr(
            infra_git_utils, "CommandRunner", lambda cwd, timeout_seconds: mock_runner
        )

        with pytest.raises(ValueError, match="git diff --numstat failed"):
            await git_utils.get_diff_stat(Path("/repo"), "invalid")

    @pytest.mark.asyncio
    async def test_uses_head_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should default to HEAD when to_commit not specified."""
        mock_runner = MockCommandRunner(
            responses=[
                CommandResult(
                    command=["git", "diff", "--numstat"],
                    returncode=0,
                    stdout="",
                ),
            ]
        )
        monkeypatch.setattr(
            infra_git_utils, "CommandRunner", lambda cwd, timeout_seconds: mock_runner
        )

        await git_utils.get_diff_stat(Path("/repo"), "abc123")

        assert mock_runner.calls[0] == ["git", "diff", "--numstat", "abc123", "HEAD"]

    @pytest.mark.asyncio
    async def test_handles_binary_files(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should handle binary files (shown as - for added/removed)."""
        numstat_output = "10\t5\tsrc/code.py\n-\t-\timage.png\n"
        mock_runner = MockCommandRunner(
            responses=[
                CommandResult(
                    command=["git", "diff", "--numstat"],
                    returncode=0,
                    stdout=numstat_output,
                ),
            ]
        )
        monkeypatch.setattr(
            infra_git_utils, "CommandRunner", lambda cwd, timeout_seconds: mock_runner
        )

        result = await git_utils.get_diff_stat(Path("/repo"), "abc123")

        assert result.total_lines == 15  # Binary files don't add to line count
        assert result.files_changed == ["src/code.py", "image.png"]

    @pytest.mark.asyncio
    async def test_handles_renames(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should extract new filename from rename format."""
        numstat_output = "0\t0\told.py => new.py\n5\t3\tdir/{old.py => new.py}\n"
        mock_runner = MockCommandRunner(
            responses=[
                CommandResult(
                    command=["git", "diff", "--numstat"],
                    returncode=0,
                    stdout=numstat_output,
                ),
            ]
        )
        monkeypatch.setattr(
            infra_git_utils, "CommandRunner", lambda cwd, timeout_seconds: mock_runner
        )

        result = await git_utils.get_diff_stat(Path("/repo"), "abc123")

        assert result.files_changed == ["new.py", "dir/new.py"]

    @pytest.mark.asyncio
    async def test_handles_quoted_paths(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should strip quotes from paths with spaces or special chars."""
        numstat_output = '5\t3\t"path with spaces.py"\n2\t1\tnormal.py\n'
        mock_runner = MockCommandRunner(
            responses=[
                CommandResult(
                    command=["git", "diff", "--numstat"],
                    returncode=0,
                    stdout=numstat_output,
                ),
            ]
        )
        monkeypatch.setattr(
            infra_git_utils, "CommandRunner", lambda cwd, timeout_seconds: mock_runner
        )

        result = await git_utils.get_diff_stat(Path("/repo"), "abc123")

        assert result.files_changed == ["path with spaces.py", "normal.py"]

    @pytest.mark.asyncio
    async def test_handles_quoted_paths_with_escaped_characters(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should unescape special characters in quoted paths.

        Git escapes special characters inside quoted paths, e.g., a file named
        foo"bar.py is output as "foo\"bar.py".
        """
        # Git escapes: \" for ", \\ for \, \t for tab, etc.
        numstat_output = (
            '5\t3\t"foo\\"bar.py"\n2\t1\t"path\\\\with\\\\backslashes.py"\n'
        )
        mock_runner = MockCommandRunner(
            responses=[
                CommandResult(
                    command=["git", "diff", "--numstat"],
                    returncode=0,
                    stdout=numstat_output,
                ),
            ]
        )
        monkeypatch.setattr(
            infra_git_utils, "CommandRunner", lambda cwd, timeout_seconds: mock_runner
        )

        result = await git_utils.get_diff_stat(Path("/repo"), "abc123")

        assert result.files_changed == ['foo"bar.py', "path\\with\\backslashes.py"]

    @pytest.mark.asyncio
    async def test_handles_renames_with_non_arrow_braces(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should preserve brace segments without arrow when parsing renames."""
        # Git output: src/{v1}/{old.py => new.py} - the {v1} should be preserved
        numstat_output = "5\t3\tsrc/{v1}/{old.py => new.py}\n"
        mock_runner = MockCommandRunner(
            responses=[
                CommandResult(
                    command=["git", "diff", "--numstat"],
                    returncode=0,
                    stdout=numstat_output,
                ),
            ]
        )
        monkeypatch.setattr(
            infra_git_utils, "CommandRunner", lambda cwd, timeout_seconds: mock_runner
        )

        result = await git_utils.get_diff_stat(Path("/repo"), "abc123")

        assert result.files_changed == ["src/v1/new.py"]


class TestGetDiffContent:
    """Tests for get_diff_content() function."""

    @pytest.mark.asyncio
    async def test_returns_unified_diff(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should return the raw unified diff output."""
        diff_content = (
            "diff --git a/src/foo.py b/src/foo.py\n"
            "index abc123..def456 100644\n"
            "--- a/src/foo.py\n"
            "+++ b/src/foo.py\n"
            "@@ -1,3 +1,4 @@\n"
            " line1\n"
            "+new line\n"
            " line2\n"
            " line3\n"
        )
        mock_runner = MockCommandRunner(
            responses=[
                CommandResult(
                    command=["git", "diff"],
                    returncode=0,
                    stdout=diff_content,
                ),
            ]
        )
        monkeypatch.setattr(
            infra_git_utils, "CommandRunner", lambda cwd, timeout_seconds: mock_runner
        )

        result = await git_utils.get_diff_content(Path("/repo"), "abc123", "def456")

        assert result == diff_content
        assert mock_runner.calls[0] == ["git", "diff", "abc123", "def456"]

    @pytest.mark.asyncio
    async def test_empty_diff_returns_empty_string(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Empty diff should return empty string."""
        mock_runner = MockCommandRunner(
            responses=[
                CommandResult(
                    command=["git", "diff"],
                    returncode=0,
                    stdout="",
                ),
            ]
        )
        monkeypatch.setattr(
            infra_git_utils, "CommandRunner", lambda cwd, timeout_seconds: mock_runner
        )

        result = await git_utils.get_diff_content(Path("/repo"), "abc123")

        assert result == ""

    @pytest.mark.asyncio
    async def test_raises_on_invalid_commit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should raise ValueError when git command fails."""
        mock_runner = MockCommandRunner(
            responses=[
                CommandResult(
                    command=["git", "diff"],
                    returncode=128,
                    stdout="",
                    stderr="fatal: bad revision 'invalid'",
                ),
            ]
        )
        monkeypatch.setattr(
            infra_git_utils, "CommandRunner", lambda cwd, timeout_seconds: mock_runner
        )

        with pytest.raises(ValueError, match="git diff failed"):
            await git_utils.get_diff_content(Path("/repo"), "invalid")

    @pytest.mark.asyncio
    async def test_uses_head_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should default to HEAD when to_commit not specified."""
        mock_runner = MockCommandRunner(
            responses=[
                CommandResult(
                    command=["git", "diff"],
                    returncode=0,
                    stdout="",
                ),
            ]
        )
        monkeypatch.setattr(
            infra_git_utils, "CommandRunner", lambda cwd, timeout_seconds: mock_runner
        )

        await git_utils.get_diff_content(Path("/repo"), "abc123")

        assert mock_runner.calls[0] == ["git", "diff", "abc123", "HEAD"]

    @pytest.mark.asyncio
    async def test_uses_two_argument_form_not_range(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should use 'git diff A B' not 'git diff A..B' range syntax."""
        mock_runner = MockCommandRunner(
            responses=[
                CommandResult(
                    command=["git", "diff"],
                    returncode=0,
                    stdout="",
                ),
            ]
        )
        monkeypatch.setattr(
            infra_git_utils, "CommandRunner", lambda cwd, timeout_seconds: mock_runner
        )

        await git_utils.get_diff_content(Path("/repo"), "abc123", "def456")

        # Verify command uses two separate arguments, not range syntax
        cmd = mock_runner.calls[0]
        assert cmd == ["git", "diff", "abc123", "def456"]
        # Ensure no range syntax like ".." is present
        assert not any(".." in arg for arg in cmd)
