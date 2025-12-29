from pathlib import Path

import pytest

from src import git_utils
from src.tools.command_runner import CommandResult


@pytest.mark.asyncio
async def test_get_git_commit_async_success(monkeypatch: pytest.MonkeyPatch) -> None:
    async def mock_run_command_async(
        cmd: list[str], cwd: Path, timeout_seconds: float | None = None
    ) -> CommandResult:
        return CommandResult(command=cmd, returncode=0, stdout="deadbeef\n")

    monkeypatch.setattr(git_utils, "run_command_async", mock_run_command_async)

    result = await git_utils.get_git_commit_async(Path("."))

    assert result == "deadbeef"


@pytest.mark.asyncio
async def test_get_git_branch_async_success(monkeypatch: pytest.MonkeyPatch) -> None:
    async def mock_run_command_async(
        cmd: list[str], cwd: Path, timeout_seconds: float | None = None
    ) -> CommandResult:
        return CommandResult(command=cmd, returncode=0, stdout="main\n")

    monkeypatch.setattr(git_utils, "run_command_async", mock_run_command_async)

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

    monkeypatch.setattr(git_utils, "run_command_async", mock_run_command_async)

    result = await git_utils.get_git_commit_async(Path("."), timeout=0.01)

    assert result == ""


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

        monkeypatch.setattr(
            git_utils, "CommandRunner", lambda cwd, timeout_seconds: mock_runner
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

        monkeypatch.setattr(
            git_utils, "CommandRunner", lambda cwd, timeout_seconds: mock_runner
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

        monkeypatch.setattr(
            git_utils, "CommandRunner", lambda cwd, timeout_seconds: mock_runner
        )

        result = await git_utils.get_baseline_for_issue(Path("/repo"), "mala-123")

        assert result is None

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

        monkeypatch.setattr(
            git_utils, "CommandRunner", lambda cwd, timeout_seconds: mock_runner
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

        monkeypatch.setattr(
            git_utils, "CommandRunner", lambda cwd, timeout_seconds: mock_runner
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

        monkeypatch.setattr(
            git_utils, "CommandRunner", lambda cwd, timeout_seconds: mock_runner
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

        monkeypatch.setattr(
            git_utils, "CommandRunner", lambda cwd, timeout_seconds: mock_runner
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

        monkeypatch.setattr(
            git_utils, "CommandRunner", lambda cwd, timeout_seconds: mock_runner
        )

        result = await git_utils.get_baseline_for_issue(Path("/repo"), "mala-rebase")

        # Should get parent of the new (rebased) first commit
        assert result == "rebasebase"
