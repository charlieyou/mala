"""E2E test for AgentSDKReviewer with real Claude Agent SDK.

This test validates that AgentSDKReviewer works end-to-end with the real Agent SDK,
not just mocks. It requires a working Claude Code authentication setup and uses
minimal test cases to keep costs low.

Key validations:
- Real SDK client creation and session management
- Agent can execute tools (git diff, file reading)
- ReviewResult structure is valid (passed, issues, no parse_error)
- Session log contains evidence of tool usage
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import pytest

from src.infra.clients.agent_sdk_review import AgentSDKReviewer
from src.infra.clients.review_output_parser import ReviewResult
from src.infra.io.session_log_parser import SessionLogParser
from src.infra.sdk_adapter import SDKClientFactory

pytestmark = [pytest.mark.e2e]

if TYPE_CHECKING:
    from pathlib import Path


def _session_log_contains_tool_use(log_path: Path) -> bool:
    """Check if session log contains evidence of tool usage.

    Parses the session log and looks for any tool_use blocks (Bash, Read, etc.).

    Args:
        log_path: Path to the session log file.

    Returns:
        True if at least one tool_use block is found in the log.
    """
    if not log_path.exists():
        return False

    parser = SessionLogParser()
    for entry in parser.iter_jsonl_entries(log_path):
        # Check for Bash commands (includes git diff)
        bash_commands = parser.extract_bash_commands(entry)
        if bash_commands:
            return True

        # Check for any tool results (indicates tools were used and completed)
        tool_results = parser.extract_tool_results(entry)
        if tool_results:
            return True

    return False


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
    # Disable GPG signing to avoid failures when global commit.gpgsign=true
    subprocess.run(
        ["git", "config", "commit.gpgsign", "false"],
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
    # Create real SDK client factory
    sdk_factory = SDKClientFactory()

    # Minimal review prompt focused on speed (minimize token usage)
    review_prompt = """You are a code reviewer. Review the listed commits and output JSON.

Instructions:
1. Run `git show <commit_sha>` for the commit provided
2. Return a JSON verdict immediately after viewing the commit

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
        default_timeout=180,  # Buffer for slow SDK sessions
    )

    # Run review on the last commit
    last_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=test_repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    result = await reviewer(
        context_file=None,
        timeout=180,
        claude_session_id=None,
        commit_shas=[last_commit],
    )

    # Validate ReviewResult structure
    assert isinstance(result, ReviewResult), (
        f"Expected ReviewResult, got {type(result)}"
    )
    assert isinstance(result.passed, bool), "passed must be a boolean"
    assert isinstance(result.issues, list), "issues must be a list"

    # E2E test requires no parse errors - agent must return valid JSON
    assert result.parse_error is None, (
        f"Agent returned invalid JSON: {result.parse_error}"
    )
    assert result.fatal_error is False, "Review should not have fatal errors"

    # Session log path must be populated for successful review
    assert result.review_log_path is not None, (
        "review_log_path should be populated for successful review"
    )

    # Verify agent executed at least one tool (e.g., git diff)
    assert _session_log_contains_tool_use(result.review_log_path), (
        f"Session log at {result.review_log_path} should contain tool usage evidence"
    )


@pytest.mark.asyncio
async def test_empty_commit_list_skips_agent(test_repo: Path) -> None:
    """Test that empty commit list returns PASS without running agent."""
    sdk_factory = SDKClientFactory()

    reviewer = AgentSDKReviewer(
        repo_path=test_repo,
        review_agent_prompt="This should not be called",
        sdk_client_factory=sdk_factory,
        event_sink=None,
        model="haiku",
        default_timeout=60,
    )

    result = await reviewer(
        context_file=None,
        timeout=60,
        claude_session_id=None,
        commit_shas=[],
    )

    # Empty commit list should short-circuit to PASS
    assert result.passed is True
    assert result.issues == []
    assert result.parse_error is None
    assert result.fatal_error is False
    # No agent session ran, so no log path
    assert result.review_log_path is None
