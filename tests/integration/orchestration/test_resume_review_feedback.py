"""Integration test for resume with review feedback flow.

This test verifies the full data flow for resuming sessions with stored review issues:
AgentSessionRunner → IssueResult → IssueRun → SessionInfo → _build_resume_prompt
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

from src.infra.io.log_output.run_metadata import (
    SessionInfo,
    extract_session_from_run,
    parse_timestamp,
)
from src.infra.tools.env import encode_repo_path, get_claude_log_path
from src.orchestration.orchestrator import StoredReviewIssue, _build_resume_prompt
from src.pipeline.agent_session_runner import AgentSessionOutput
from src.pipeline.issue_result import IssueResult
from tests.fakes.gate_checker import FakeGateChecker
from tests.fakes.issue_provider import FakeIssue, FakeIssueProvider
from tests.fakes.sdk_client import FakeSDKClientFactory

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from src.domain.prompts import PromptProvider
    from src.domain.validation.config import PromptValidationCommands
    from src.orchestration.orchestrator import MalaOrchestrator


# --- Unit tests for dataclass fields ---


def test_issue_result_has_last_review_issues_field() -> None:
    """IssueResult should have last_review_issues field with None default."""
    result = IssueResult(
        issue_id="test-1",
        agent_id="agent-1",
        success=True,
        summary="Done",
    )
    assert result.last_review_issues is None

    # Can be set to list of dicts
    result_with_issues = IssueResult(
        issue_id="test-2",
        agent_id="agent-2",
        success=True,
        summary="Done",
        last_review_issues=[{"file": "a.py", "title": "Issue 1"}],
    )
    assert result_with_issues.last_review_issues == [
        {"file": "a.py", "title": "Issue 1"}
    ]


def test_agent_session_output_has_last_review_issues_field() -> None:
    """AgentSessionOutput should have last_review_issues field with None default."""
    output = AgentSessionOutput(
        success=True,
        summary="Done",
    )
    assert output.last_review_issues is None

    output_with_issues = AgentSessionOutput(
        success=True,
        summary="Done",
        last_review_issues=[{"file": "b.py", "title": "Issue 2"}],
    )
    assert output_with_issues.last_review_issues == [
        {"file": "b.py", "title": "Issue 2"}
    ]


def test_agent_session_output_has_session_end_result_field() -> None:
    """AgentSessionOutput should have session_end_result field with None default."""
    from datetime import UTC, datetime
    from src.core.session_end_result import SessionEndResult

    output = AgentSessionOutput(
        success=True,
        summary="Done",
    )
    assert output.session_end_result is None

    now = datetime.now(tz=UTC)
    session_end = SessionEndResult(
        status="pass",
        started_at=now,
        finished_at=now,
    )
    output_with_session_end = AgentSessionOutput(
        success=True,
        summary="Done",
        session_end_result=session_end,
    )
    assert output_with_session_end.session_end_result is not None
    assert output_with_session_end.session_end_result.status == "pass"


def test_session_info_has_last_review_issues_field(tmp_path: Path) -> None:
    """SessionInfo should have last_review_issues field with None default."""
    info = SessionInfo(
        run_id="run-1",
        session_id="session-1",
        issue_id="issue-1",
        run_started_at="2024-01-01T00:00:00Z",
        started_at_ts=0.0,
        status="success",
        log_path=None,
        metadata_path=tmp_path / "run.json",
        repo_path=str(tmp_path),
    )
    assert info.last_review_issues is None

    info_with_issues = SessionInfo(
        run_id="run-2",
        session_id="session-2",
        issue_id="issue-2",
        run_started_at="2024-01-01T00:00:00Z",
        started_at_ts=0.0,
        status="success",
        log_path=None,
        metadata_path=tmp_path / "run.json",
        repo_path=str(tmp_path),
        last_review_issues=[{"file": "c.py", "title": "Issue 3"}],
    )
    assert info_with_issues.last_review_issues == [{"file": "c.py", "title": "Issue 3"}]


def test_extract_session_from_run_extracts_last_review_issues(tmp_path: Path) -> None:
    """extract_session_from_run should extract last_review_issues from metadata."""
    review_issues = [
        {"file": "src/main.py", "line_start": 10, "title": "Fix this"},
    ]
    run_data: dict[str, Any] = {
        "run_id": "run-123",
        "started_at": "2024-01-01T12:00:00Z",
        "issues": {
            "issue-abc": {
                "session_id": "session-456",
                "status": "success",
                "last_review_issues": review_issues,
            }
        },
    }
    metadata_path = tmp_path / "run.json"
    metadata_path.write_text(json.dumps(run_data))

    session = extract_session_from_run(run_data, metadata_path, "issue-abc")

    assert session is not None
    assert session.last_review_issues == review_issues


def test_extract_session_from_run_handles_missing_review_issues(tmp_path: Path) -> None:
    """extract_session_from_run returns None for missing last_review_issues."""
    run_data: dict[str, Any] = {
        "run_id": "run-123",
        "started_at": "2024-01-01T12:00:00Z",
        "issues": {
            "issue-abc": {
                "session_id": "session-456",
                "status": "success",
            }
        },
    }
    metadata_path = tmp_path / "run.json"
    metadata_path.write_text(json.dumps(run_data))

    session = extract_session_from_run(run_data, metadata_path, "issue-abc")

    assert session is not None
    assert session.last_review_issues is None


# --- StoredReviewIssue tests ---


def test_stored_review_issue_from_dict_complete() -> None:
    """StoredReviewIssue.from_dict handles complete data."""
    data = {
        "file": "src/main.py",
        "line_start": 10,
        "line_end": 20,
        "priority": 1,
        "title": "Fix the bug",
        "body": "Details here",
        "reviewer": "cerberus",
    }
    issue = StoredReviewIssue.from_dict(data)

    assert issue.file == "src/main.py"
    assert issue.line_start == 10
    assert issue.line_end == 20
    assert issue.priority == 1
    assert issue.title == "Fix the bug"
    assert issue.body == "Details here"
    assert issue.reviewer == "cerberus"


def test_stored_review_issue_from_dict_missing_keys() -> None:
    """StoredReviewIssue.from_dict uses safe defaults for missing keys."""
    data: dict[str, Any] = {}
    issue = StoredReviewIssue.from_dict(data)

    assert issue.file == "unknown"
    assert issue.line_start == 0
    assert issue.line_end == 0  # Defaults to line_start
    assert issue.priority is None
    assert issue.title == "Unknown issue"
    assert issue.body == ""
    assert issue.reviewer == "unknown"


def test_stored_review_issue_from_dict_invalid_types() -> None:
    """StoredReviewIssue.from_dict handles invalid types gracefully."""
    data = {
        "file": 123,  # Should be string
        "line_start": "not_a_number",  # Should be int
        "line_end": None,  # Missing, should default to line_start
        "priority": None,  # Valid None
        "title": None,  # Should become string
        "body": None,  # Should become string
        "reviewer": None,  # Should become string
    }
    issue = StoredReviewIssue.from_dict(data)

    assert issue.file == "123"  # str() conversion
    assert issue.line_start == 0  # Invalid type defaults to 0
    assert issue.line_end == 0  # Defaults to line_start
    assert issue.priority is None
    assert issue.title == "None"  # str(None)
    assert issue.body == "None"  # str(None)
    assert issue.reviewer == "None"  # str(None)


def test_stored_review_issue_line_end_defaults_to_line_start() -> None:
    """StoredReviewIssue.from_dict defaults line_end to line_start."""
    data = {
        "file": "test.py",
        "line_start": 42,
        # line_end missing
        "title": "Issue",
        "body": "",
        "reviewer": "test",
    }
    issue = StoredReviewIssue.from_dict(data)

    assert issue.line_start == 42
    assert issue.line_end == 42


# --- _build_resume_prompt behavior tests ---


def test_build_resume_prompt_returns_formatted_prompt(tmp_path: Path) -> None:
    """_build_resume_prompt returns formatted prompt when issues present."""
    from src.domain.prompts import PromptProvider
    from src.domain.validation.config import PromptValidationCommands

    prompts = PromptProvider(
        implementer_prompt="impl",
        review_followup_prompt=(
            "Attempt {attempt}/{max_attempts}\n"
            "Issues: {review_issues}\n"
            "Issue ID: {issue_id}"
        ),
        gate_followup_prompt="gate",
        fixer_prompt="fixer",
        idle_resume_prompt="idle",
        checkpoint_request_prompt="checkpoint",
        continuation_prompt="continuation",
    )
    validation_commands = PromptValidationCommands(
        lint="lint",
        format="format",
        typecheck="typecheck",
        test="test",
        custom_commands=(),
    )
    review_issues = [{"file": "a.py", "title": "Issue", "line_start": 1}]

    result = _build_resume_prompt(
        review_issues,
        prompts,
        validation_commands,
        issue_id="test-123",
        max_review_retries=3,
        repo_path=tmp_path,
        prior_run_id="run-prior",
    )

    assert result is not None
    assert "Attempt 1/3" in result
    assert "Issue ID: test-123" in result


def test_build_resume_prompt_returns_none_when_no_issues(tmp_path: Path) -> None:
    """_build_resume_prompt returns None when no issues."""
    from src.domain.prompts import PromptProvider
    from src.domain.validation.config import PromptValidationCommands

    prompts = PromptProvider(
        implementer_prompt="impl",
        review_followup_prompt="followup",
        gate_followup_prompt="gate",
        fixer_prompt="fixer",
        idle_resume_prompt="idle",
        checkpoint_request_prompt="checkpoint",
        continuation_prompt="continuation",
    )
    validation_commands = PromptValidationCommands(
        lint="lint",
        format="format",
        typecheck="typecheck",
        test="test",
        custom_commands=(),
    )

    result = _build_resume_prompt(
        review_issues=[],
        prompts=prompts,
        validation_commands=validation_commands,
        issue_id="test-123",
        max_review_retries=3,
        repo_path=tmp_path,
        prior_run_id="run-xyz",
    )

    assert result is None


# --- Integration test (expected to fail until T005) ---


def _read_latest_run_metadata(runs_dir: Path, repo_path: Path) -> dict[str, Any]:
    """Read the most recent run metadata file for a repo."""
    repo_runs_dir = runs_dir / encode_repo_path(repo_path)
    files = list(repo_runs_dir.glob("*.json"))
    assert files, "expected at least one run metadata file"
    best: tuple[float, str, Path] | None = None
    for path in files:
        data = json.loads(path.read_text())
        started_at = data.get("started_at") or ""
        run_id = data.get("run_id") or ""
        timestamp = parse_timestamp(started_at)
        key = (timestamp, run_id, path)
        if best is None or key > best:
            best = key
    assert best is not None
    return json.loads(best[2].read_text())


def _make_result_message(session_id: str, *, result: str) -> object:
    """Create a fake ResultMessage for testing."""
    from claude_agent_sdk import ResultMessage

    return ResultMessage(
        subtype="result",
        session_id=session_id,
        result=result,
        duration_ms=1000,
        duration_api_ms=800,
        is_error=False,
        num_turns=1,
        total_cost_usd=0.01,
        usage=None,
    )


async def _run_with_fake_git(orchestrator: MalaOrchestrator) -> None:
    """Run orchestrator with mocked git operations."""
    import contextlib

    patches = [
        patch(
            "src.orchestration.orchestrator.get_git_branch_async",
            return_value="main",
        ),
        patch(
            "src.orchestration.orchestrator.get_git_commit_async",
            return_value="abc123",
        ),
        patch(
            "src.infra.git_utils.get_git_commit_async",
            return_value="abc123",
        ),
    ]

    with contextlib.ExitStack() as stack:
        for patcher in patches:
            stack.enter_context(patcher)
        await orchestrator.run()


@pytest.mark.asyncio
async def test_resume_with_review_feedback_uses_review_followup_prompt(
    tmp_path: Path,
    make_orchestrator: Callable[..., MalaOrchestrator],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When resuming with stored review issues, should use review_followup prompt.

    This integration test verifies the full flow:
    1. Run 1: Session completes with review issues stored in metadata
    2. Run 2: Resume picks up stored issues and uses review_followup prompt
    """
    issue_id = "issue-123"
    runs_dir = tmp_path / "runs"
    claude_dir = tmp_path / "claude"
    monkeypatch.setenv("MALA_RUNS_DIR", str(runs_dir))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(claude_dir))

    # --- Run 1: Initial run that will store review issues ---
    session_id_run1 = "session-1"
    log_path_1 = get_claude_log_path(tmp_path, session_id_run1)
    log_path_1.parent.mkdir(parents=True, exist_ok=True)
    log_path_1.write_text('{"type": "result"}\n')

    gate_checker_run1 = FakeGateChecker()
    gate_checker_run1.gate_result.commit_hash = None  # Skip review

    sdk_factory_run1 = FakeSDKClientFactory()
    sdk_factory_run1.configure_next_client(
        result_message=_make_result_message(session_id_run1, result="Done")
    )

    orchestrator_run1 = make_orchestrator(
        repo_path=tmp_path,
        max_agents=1,
        issue_provider=FakeIssueProvider(
            {issue_id: FakeIssue(id=issue_id, description="Run 1")}
        ),
        gate_checker=gate_checker_run1,
        runs_dir=runs_dir,
        disable_validations={"global-validate"},
    )
    orchestrator_run1._sdk_client_factory = sdk_factory_run1

    await _run_with_fake_git(orchestrator_run1)

    # Manually inject review issues into the stored metadata (simulating T003)
    run1_metadata = _read_latest_run_metadata(runs_dir, tmp_path)
    stored_review_issues = [
        {
            "file": "src/main.py",
            "line_start": 10,
            "line_end": 15,
            "priority": 0,
            "title": "Fix critical bug",
            "body": "This needs immediate attention",
            "reviewer": "cerberus",
        },
        {
            "file": "src/utils.py",
            "line_start": 20,
            "line_end": 20,
            "priority": 1,
            "title": "Improve error handling",
            "body": "Add try/except",
            "reviewer": "cerberus",
        },
    ]
    run1_metadata["issues"][issue_id]["last_review_issues"] = stored_review_issues

    # Write back the modified metadata
    repo_runs_dir = runs_dir / encode_repo_path(tmp_path)
    metadata_files = list(repo_runs_dir.glob("*.json"))
    assert metadata_files
    # Find the file we read
    for mf in metadata_files:
        data = json.loads(mf.read_text())
        if data.get("run_id") == run1_metadata.get("run_id"):
            mf.write_text(json.dumps(run1_metadata))
            break

    # --- Run 2: Resume should use review_followup prompt ---
    session_id_run2 = "session-2"
    log_path_2 = get_claude_log_path(tmp_path, session_id_run2)
    log_path_2.parent.mkdir(parents=True, exist_ok=True)
    log_path_2.write_text('{"type": "result"}\n')

    gate_checker_run2 = FakeGateChecker()
    gate_checker_run2.gate_result.commit_hash = None

    sdk_factory_run2 = FakeSDKClientFactory()
    sdk_factory_run2.configure_next_client(
        result_message=_make_result_message(session_id_run2, result="Done")
    )

    orchestrator_run2 = make_orchestrator(
        repo_path=tmp_path,
        max_agents=1,
        issue_provider=FakeIssueProvider(
            {
                issue_id: FakeIssue(
                    id=issue_id, status="in_progress", description="Run 2"
                )
            }
        ),
        gate_checker=gate_checker_run2,
        runs_dir=runs_dir,
        include_wip=True,  # Enable resume
        disable_validations={"global-validate"},
    )
    orchestrator_run2._sdk_client_factory = sdk_factory_run2

    # Check if _build_resume_prompt is called with the right data
    import src.orchestration.orchestrator as orch_module

    build_resume_prompt_calls: list[
        tuple[
            list[dict[str, Any]],
            PromptProvider,
            PromptValidationCommands,
            str,
            int,
            Path,
            str,
        ]
    ] = []
    original_build_resume = _build_resume_prompt

    def mock_build_resume(
        review_issues: list[dict[str, Any]],
        prompts: object,
        validation_commands: PromptValidationCommands,
        issue_id: str,
        max_review_retries: int,
        repo_path: Path,
        prior_run_id: str,
    ) -> str | None:
        build_resume_prompt_calls.append(
            (
                review_issues,
                prompts,
                validation_commands,
                issue_id,
                max_review_retries,
                repo_path,
                prior_run_id,
            )  # type: ignore[arg-type]
        )
        return original_build_resume(
            review_issues,
            prompts,  # type: ignore[arg-type]
            validation_commands,
            issue_id,
            max_review_retries,
            repo_path,
            prior_run_id,
        )

    with patch.object(orch_module, "_build_resume_prompt", mock_build_resume):
        await _run_with_fake_git(orchestrator_run2)

    # Verify that when the session was resumed, _build_resume_prompt was called
    # with the stored review issues.
    assert len(build_resume_prompt_calls) > 0, (
        "_build_resume_prompt was not called during resume."
    )

    # Verify the call was made with the correct review issues
    call_args = build_resume_prompt_calls[0]
    assert call_args[0] == stored_review_issues, (
        "Expected _build_resume_prompt to be called with the stored review issues"
    )
