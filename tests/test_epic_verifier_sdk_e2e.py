"""E2E test that runs epic verification through the Claude Agent SDK."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.epic_verifier import ClaudeEpicVerificationModel
from tests.claude_auth import is_claude_cli_available, has_valid_oauth_credentials

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
    diff = """diff --git a/src/math_utils.py b/src/math_utils.py\nnew file mode 100644\nindex 0000000..1111111\n--- /dev/null\n+++ b/src/math_utils.py\n@@\n+def add(a: int, b: int) -> int:\n+    return a + b\n"""

    verdict = await model.verify(criteria, diff, None)

    assert isinstance(verdict.passed, bool)
    assert 0.0 <= verdict.confidence <= 1.0
    assert verdict.reasoning
    assert "Failed to parse" not in verdict.reasoning
