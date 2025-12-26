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
