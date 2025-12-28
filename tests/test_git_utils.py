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
        assert "--grep=^bd-mala-123:" in log_call[0][0]

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
