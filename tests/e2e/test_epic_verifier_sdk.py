"""E2E test that runs epic verification through the Claude Agent SDK."""

from __future__ import annotations

from typing import TYPE_CHECKING

import subprocess

import pytest

from src.infra.epic_verifier import ClaudeEpicVerificationModel
from tests.e2e.claude_auth import is_claude_cli_available, has_valid_oauth_credentials

pytestmark = [pytest.mark.e2e]

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def require_claude_cli_auth() -> None:
    if not is_claude_cli_available():
        pytest.skip("Claude Code CLI not installed")
    if not has_valid_oauth_credentials():
        pytest.skip(
            "Claude Code CLI not logged in or token expired - run `claude` and login"
        )


@pytest.mark.asyncio
async def test_epic_verifier_runs_via_sdk(tmp_path: Path) -> None:
    model = ClaudeEpicVerificationModel(repo_path=tmp_path)

    criteria = """Acceptance Criteria:\n- The helper function add returns a + b\n"""
    (tmp_path / "src").mkdir()
    file_path = tmp_path / "src" / "math_utils.py"
    file_path.write_text("def add(a: int, b: int) -> int:\n    return a + b\n")

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
    subprocess.run(
        ["git", "config", "commit.gpgsign", "false"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Add math utils"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    verdict = await model.verify(criteria)

    assert isinstance(verdict.passed, bool)
    assert verdict.reasoning
    assert "Failed to parse" not in verdict.reasoning
