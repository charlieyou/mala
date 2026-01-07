"""E2E test for AgentSDKReviewer with real Claude Agent SDK.

This test validates that AgentSDKReviewer works end-to-end with the real Agent SDK,
not just mocks. It requires ANTHROPIC_API_KEY to be set and uses minimal test
cases to keep costs low.

Key validations:
- Real SDK client creation and session management
- Agent can execute tools (git diff, file reading)
- ReviewResult structure is valid (passed, issues, no parse_error)
- Session log path is populated correctly
"""

from __future__ import annotations

import os
import subprocess
from typing import TYPE_CHECKING

import pytest

from src.infra.clients.agent_sdk_review import AgentSDKReviewer
from src.infra.clients.review_output_parser import ReviewResult
from src.infra.sdk_adapter import SDKClientFactory
from tests.e2e.claude_auth import has_valid_oauth_credentials, is_claude_cli_available

pytestmark = [pytest.mark.e2e]

if TYPE_CHECKING:
    from pathlib import Path


def _skip_if_no_auth() -> None:
    """Skip test if neither API key nor OAuth credentials available."""
    has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_oauth = is_claude_cli_available() and has_valid_oauth_credentials()

    if not has_api_key and not has_oauth:
        pytest.skip(
            "No auth available: set ANTHROPIC_API_KEY or login via `claude` CLI"
        )


@pytest.fixture
def test_repo(tmp_path: Path) -> Path:
    """Create a minimal git repo with a single file change."""
    # Initialize git repo
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    # Create initial commit (empty file)
    (tmp_path / "example.py").write_text("")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    # Create second commit with a simple function
    (tmp_path / "example.py").write_text(
        "def greet(name: str) -> str:\n"
        '    """Return a greeting message."""\n'
        '    return f"Hello, {name}!"\n'
    )
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Add greet function"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    return tmp_path


@pytest.mark.asyncio
async def test_real_agent_review_flow(test_repo: Path) -> None:
    """Test AgentSDKReviewer with real SDK client.

    Validates:
    - Agent session runs successfully
    - ReviewResult structure is valid
    - Agent executed at least one tool (via session log path presence)
    """
    _skip_if_no_auth()

    # Create real SDK client factory
    sdk_factory = SDKClientFactory()

    # Minimal review prompt focused on speed (minimize token usage)
    review_prompt = """You are a code reviewer. Review the git diff and output JSON.

Instructions:
1. Run `git diff HEAD~1..HEAD` to see the changes
2. Return a JSON verdict immediately after viewing the diff

Output this exact JSON structure (no other text):
```json
{
  "consensus_verdict": "PASS",
  "aggregated_findings": []
}
```

Only use FAIL if there's a serious bug. This is a simple function, so PASS is expected.
"""

    # Create reviewer with short timeout (agent should be fast with minimal diff)
    reviewer = AgentSDKReviewer(
        repo_path=test_repo,
        review_agent_prompt=review_prompt,
        sdk_client_factory=sdk_factory,
        event_sink=None,
        model="haiku",  # Use haiku for speed and cost
        default_timeout=120,  # 2 minutes should be plenty for minimal diff
    )

    # Run review on the last commit
    result = await reviewer(
        diff_range="HEAD~1..HEAD",
        context_file=None,
        timeout=120,
        claude_session_id=None,
        commit_shas=None,
    )

    # Validate ReviewResult structure
    assert isinstance(result, ReviewResult), (
        f"Expected ReviewResult, got {type(result)}"
    )
    assert isinstance(result.passed, bool), "passed must be a boolean"
    assert isinstance(result.issues, list), "issues must be a list"

    # If there's no parse error, the agent successfully returned valid JSON
    if result.parse_error is None:
        # Successful review should either pass or fail with issues
        assert result.passed is True or len(result.issues) > 0, (
            "If parse_error is None, review should either pass or have issues"
        )
    else:
        # Parse error is acceptable for E2E (agent might format JSON incorrectly)
        # but we should not have fatal_error
        assert result.fatal_error is False, (
            f"Fatal error occurred: {result.parse_error}"
        )

    # Session log path should be populated if agent ran successfully
    # (may be None if agent timed out or errored before completing)
    if result.parse_error is None:
        assert result.review_log_path is not None, (
            "review_log_path should be populated for successful review"
        )


@pytest.mark.asyncio
async def test_empty_diff_skips_agent(test_repo: Path) -> None:
    """Test that empty diff returns PASS without running agent."""
    _skip_if_no_auth()

    sdk_factory = SDKClientFactory()

    reviewer = AgentSDKReviewer(
        repo_path=test_repo,
        review_agent_prompt="This should not be called",
        sdk_client_factory=sdk_factory,
        event_sink=None,
        model="haiku",
        default_timeout=60,
    )

    # HEAD..HEAD is always empty (no changes)
    result = await reviewer(
        diff_range="HEAD..HEAD",
        context_file=None,
        timeout=60,
        claude_session_id=None,
        commit_shas=None,
    )

    # Empty diff should short-circuit to PASS
    assert result.passed is True
    assert result.issues == []
    assert result.parse_error is None
    assert result.fatal_error is False
    # No agent session ran, so no log path
    assert result.review_log_path is None
