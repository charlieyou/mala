import asyncio
from pathlib import Path

import pytest

from src import git_utils


class DummyResult:
    def __init__(self, stdout: str, returncode: int = 0) -> None:
        self.stdout = stdout
        self.returncode = returncode


@pytest.mark.asyncio
async def test_get_git_commit_async_success(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy = DummyResult("deadbeef\n")
    monkeypatch.setattr(git_utils.subprocess, "run", lambda *args, **kwargs: dummy)

    result = await git_utils.get_git_commit_async(Path("."))

    assert result == "deadbeef"


@pytest.mark.asyncio
async def test_get_git_branch_async_success(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy = DummyResult("main\n")
    monkeypatch.setattr(git_utils.subprocess, "run", lambda *args, **kwargs: dummy)

    result = await git_utils.get_git_branch_async(Path("."))

    assert result == "main"


@pytest.mark.asyncio
async def test_get_git_commit_async_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    async def slow() -> str:
        await asyncio.sleep(0.05)
        return "ignored"

    monkeypatch.setattr(git_utils.asyncio, "to_thread", lambda *args, **kwargs: slow())

    result = await git_utils.get_git_commit_async(Path("."), timeout=0.01)

    assert result == ""


# --- Tests for get_baseline_for_issue ---


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
        # git log --oneline --reverse --grep returns empty
        calls: list[tuple[list[str], Path]] = []

        def mock_run(cmd: list[str], **kwargs: object) -> DummyResult:
            calls.append((cmd, kwargs.get("cwd")))  # type: ignore[arg-type]
            if "log" in cmd:
                return DummyResult("", returncode=0)  # No matching commits
            return DummyResult("")

        monkeypatch.setattr(git_utils.subprocess, "run", mock_run)

        result = await git_utils.get_baseline_for_issue(Path("/repo"), "mala-123")

        assert result is None
        # Verify git log was called with correct grep pattern
        log_call = [c for c in calls if "log" in c[0]]
        assert len(log_call) == 1
        # re.escape() escapes hyphens too, so pattern is mala\-123
        assert r"--grep=^bd-mala\-123:" in log_call[0][0]

    @pytest.mark.asyncio
    async def test_resumed_issue_with_prior_commits(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Resumed issue should return parent of first matching commit."""
        calls: list[tuple[list[str], Path]] = []

        def mock_run(cmd: list[str], **kwargs: object) -> DummyResult:
            calls.append((cmd, kwargs.get("cwd")))  # type: ignore[arg-type]
            if "log" in cmd:
                # Two commits for this issue
                return DummyResult(
                    "abc1234 bd-mala-123: First commit\ndef5678 bd-mala-123: Second commit"
                )
            elif "rev-parse" in cmd:
                # Parent of first commit
                return DummyResult("parent99\n")
            return DummyResult("")

        monkeypatch.setattr(git_utils.subprocess, "run", mock_run)

        result = await git_utils.get_baseline_for_issue(Path("/repo"), "mala-123")

        assert result == "parent99"
        # Verify rev-parse was called with correct commit^
        revparse_call = [c for c in calls if "rev-parse" in c[0]]
        assert len(revparse_call) == 1
        assert "abc1234^" in revparse_call[0][0]

    @pytest.mark.asyncio
    async def test_root_commit_no_parent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If first commit is root (no parent), should return None."""

        def mock_run(cmd: list[str], **kwargs: object) -> DummyResult:
            if "log" in cmd:
                return DummyResult("abc1234 bd-mala-123: Root commit")
            elif "rev-parse" in cmd:
                # Root commit has no parent - rev-parse fails
                return DummyResult("", returncode=128)
            return DummyResult("")

        monkeypatch.setattr(git_utils.subprocess, "run", mock_run)

        result = await git_utils.get_baseline_for_issue(Path("/repo"), "mala-123")

        assert result is None

    @pytest.mark.asyncio
    async def test_timeout_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Timeout should return None gracefully."""

        async def slow() -> str | None:
            await asyncio.sleep(0.1)
            return "ignored"

        monkeypatch.setattr(
            git_utils.asyncio, "to_thread", lambda *args, **kwargs: slow()
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

        def mock_run(cmd: list[str], **kwargs: object) -> DummyResult:
            if "log" in cmd:
                return DummyResult("aaa1111 bd-mala-abc: Single commit")
            elif "rev-parse" in cmd:
                return DummyResult("beforeaaa\n")
            return DummyResult("")

        monkeypatch.setattr(git_utils.subprocess, "run", mock_run)

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
        calls: list[tuple[list[str], Path]] = []

        def mock_run(cmd: list[str], **kwargs: object) -> DummyResult:
            calls.append((cmd, kwargs.get("cwd")))  # type: ignore[arg-type]
            if "log" in cmd:
                return DummyResult("abc1234 bd-mala-g3h.1: Fix the bug")
            elif "rev-parse" in cmd:
                return DummyResult("parent123\n")
            return DummyResult("")

        monkeypatch.setattr(git_utils.subprocess, "run", mock_run)

        result = await git_utils.get_baseline_for_issue(Path("/repo"), "mala-g3h.1")

        assert result == "parent123"
        # Verify the grep pattern has escaped metacharacters
        log_call = [c for c in calls if "log" in c[0]]
        assert len(log_call) == 1
        # re.escape() escapes both dot and hyphen: mala\-g3h\.1
        assert r"--grep=^bd-mala\-g3h\.1:" in log_call[0][0]

    @pytest.mark.asyncio
    async def test_merge_commit_still_finds_first_issue_commit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Merge commits in history should not affect finding the first issue commit.

        Even if the issue has merge commits, we should find the first commit
        with the bd-{issue_id}: prefix and return its parent.
        """

        def mock_run(cmd: list[str], **kwargs: object) -> DummyResult:
            if "log" in cmd:
                # Multiple commits including what could be merge commits
                # The first one (chronologically) is what we want
                return DummyResult(
                    "first11 bd-mala-merge: Initial\n"
                    "merge22 bd-mala-merge: Merge branch 'feature'\n"
                    "third33 bd-mala-merge: More work"
                )
            elif "rev-parse" in cmd:
                # Return parent of first commit
                return DummyResult("beforefirst\n")
            return DummyResult("")

        monkeypatch.setattr(git_utils.subprocess, "run", mock_run)

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

        def mock_run(cmd: list[str], **kwargs: object) -> DummyResult:
            if "log" in cmd:
                # After rebase, commit hashes are different
                # but the prefix pattern still matches
                return DummyResult(
                    "newrebase1 bd-mala-rebase: Original work (rebased)\n"
                    "newrebase2 bd-mala-rebase: Follow-up (rebased)"
                )
            elif "rev-parse" in cmd:
                # Parent after rebase is different from original
                return DummyResult("rebasebase\n")
            return DummyResult("")

        monkeypatch.setattr(git_utils.subprocess, "run", mock_run)

        result = await git_utils.get_baseline_for_issue(Path("/repo"), "mala-rebase")

        # Should get parent of the new (rebased) first commit
        assert result == "rebasebase"
