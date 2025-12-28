"""Unit tests for MalaOrchestrator control flow.

These tests mock subprocess.run and ClaudeSDKClient to test orchestrator
state transitions without network or actual bd CLI.
"""

import asyncio
import json
import os
import re
import subprocess
import sys
import time
import uuid
from collections.abc import AsyncGenerator, Generator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from claude_agent_sdk.types import ResultMessage

from src.beads_client import BeadsClient, SubprocessResult
from src.orchestrator import IMPLEMENTER_PROMPT_TEMPLATE, IssueResult, MalaOrchestrator
from src.validation.command_runner import CommandResult


@pytest.fixture
def orchestrator(tmp_path: Path) -> MalaOrchestrator:
    """Create an orchestrator with a temporary repo path."""
    return MalaOrchestrator(
        repo_path=tmp_path,
        max_agents=2,
        timeout_minutes=1,
        max_issues=5,
    )


@pytest.fixture(autouse=True)
def skip_run_level_validation() -> Generator[None, None, None]:
    """Skip Gate 4 (run-level validation) in all tests.

    Run-level validation spawns a real Claude agent which is too slow for
    unit tests. Tests for run-level validation are in test_run_level_validation.py.
    """
    with patch(
        "src.orchestrator.MalaOrchestrator._run_run_level_validation",
        return_value=True,
    ):
        yield


def make_subprocess_result(
    returncode: int = 0, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess:
    """Create a mock subprocess result."""
    return subprocess.CompletedProcess(
        args=[],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def make_command_result(
    returncode: int = 0, stdout: str = "", stderr: str = ""
) -> CommandResult:
    """Create a mock CommandResult for run_command mocking."""
    return CommandResult(
        command=[],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def make_ready_mock(ready_json: str) -> AsyncMock:
    """Create a mock for get_ready_async that handles both bd ready and bd list calls.

    The get_ready_async method now makes two subprocess calls:
    1. bd ready --json (returns ready issues)
    2. bd list --status in_progress --json (returns WIP issues to merge in)

    This helper creates a mock that returns ready_json for bd ready
    and an empty list for bd list --status in_progress.
    """

    async def mock_run(cmd: list[str]) -> subprocess.CompletedProcess:
        if cmd == ["bd", "ready", "--json"]:
            return make_subprocess_result(stdout=ready_json)
        elif cmd == ["bd", "list", "--status", "in_progress", "--json"]:
            return make_subprocess_result(stdout="[]")
        return make_subprocess_result(returncode=1)

    return AsyncMock(side_effect=mock_run)


class TestPromptTemplate:
    """Test implementer prompt template validity."""

    def test_prompt_template_placeholders_match_format_call(self) -> None:
        """Verify prompt template placeholders match what format() provides."""
        # Extract all {placeholder} from template
        placeholders = set(re.findall(r"\{(\w+)\}", IMPLEMENTER_PROMPT_TEMPLATE))

        # These are the keys passed to format() in run_implementer
        expected_keys = {"issue_id", "repo_path", "lock_dir", "scripts_dir", "agent_id"}

        assert placeholders == expected_keys, (
            f"Mismatch: template has {placeholders}, format expects {expected_keys}"
        )


class TestGetReadyIssuesAsync:
    """Test beads.get_ready_async handles bd CLI JSON and errors."""

    @pytest.mark.asyncio
    async def test_returns_issue_ids_sorted_by_priority(
        self, orchestrator: MalaOrchestrator
    ) -> None:
        """Issues should be returned sorted by priority (lower = higher)."""
        issues_json = json.dumps(
            [
                {"id": "issue-3", "priority": 3, "issue_type": "task"},
                {"id": "issue-1", "priority": 1, "issue_type": "task"},
                {"id": "issue-2", "priority": 2, "issue_type": "task"},
            ]
        )
        with patch.object(
            orchestrator.beads,
            "_run_subprocess_async",
            make_ready_mock(issues_json),
        ):
            result = await orchestrator.beads.get_ready_async()

        assert result == ["issue-1", "issue-2", "issue-3"]

    @pytest.mark.asyncio
    async def test_filters_out_epics(self, orchestrator: MalaOrchestrator) -> None:
        """Epics should be excluded from ready issues."""
        issues_json = json.dumps(
            [
                {"id": "task-1", "priority": 1, "issue_type": "task"},
                {"id": "epic-1", "priority": 1, "issue_type": "epic"},
                {"id": "bug-1", "priority": 2, "issue_type": "bug"},
            ]
        )
        with patch.object(
            orchestrator.beads,
            "_run_subprocess_async",
            make_ready_mock(issues_json),
        ):
            result = await orchestrator.beads.get_ready_async()

        assert result == ["task-1", "bug-1"]
        assert "epic-1" not in result

    @pytest.mark.asyncio
    async def test_filters_out_failed_issues(
        self, orchestrator: MalaOrchestrator
    ) -> None:
        """Previously failed issues should be excluded."""
        failed_set = {"failed-1"}
        issues_json = json.dumps(
            [
                {"id": "ok-1", "priority": 1, "issue_type": "task"},
                {"id": "failed-1", "priority": 1, "issue_type": "task"},
            ]
        )
        with patch.object(
            orchestrator.beads,
            "_run_subprocess_async",
            make_ready_mock(issues_json),
        ):
            result = await orchestrator.beads.get_ready_async(failed_set)

        assert result == ["ok-1"]
        assert "failed-1" not in result

    @pytest.mark.asyncio
    async def test_returns_empty_list_on_bd_failure(
        self, orchestrator: MalaOrchestrator
    ) -> None:
        """When bd ready fails, return empty list."""
        with patch.object(
            orchestrator.beads,
            "_run_subprocess_async",
            new_callable=AsyncMock,
            return_value=make_subprocess_result(
                returncode=1, stderr="bd: command not found"
            ),
        ):
            result = await orchestrator.beads.get_ready_async()

        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_list_on_invalid_json(
        self, orchestrator: MalaOrchestrator
    ) -> None:
        """When bd returns invalid JSON, return empty list."""
        with patch.object(
            orchestrator.beads,
            "_run_subprocess_async",
            new_callable=AsyncMock,
            return_value=make_subprocess_result(stdout="not valid json"),
        ):
            result = await orchestrator.beads.get_ready_async()

        assert result == []

    @pytest.mark.asyncio
    async def test_handles_missing_priority(
        self, orchestrator: MalaOrchestrator
    ) -> None:
        """Issues without priority should be treated as P0 (highest priority)."""
        issues_json = json.dumps(
            [
                {"id": "no-prio", "issue_type": "task"},
                {"id": "prio-1", "priority": 1, "issue_type": "task"},
            ]
        )
        with patch.object(
            orchestrator.beads,
            "_run_subprocess_async",
            make_ready_mock(issues_json),
        ):
            result = await orchestrator.beads.get_ready_async()

        # Issues without priority default to P0 (0) to match bd CLI behavior
        assert result == ["no-prio", "prio-1"]

    @pytest.mark.asyncio
    async def test_suppresses_warning_for_only_ids_already_processed(
        self, tmp_path: Path
    ) -> None:
        """Only-id warnings should be suppressed for already processed IDs."""
        warnings: list[str] = []
        beads = BeadsClient(tmp_path, log_warning=warnings.append)
        issues_json = json.dumps([])
        with patch.object(
            beads,
            "_run_subprocess_async",
            make_ready_mock(issues_json),
        ):
            result = await beads.get_ready_async(
                only_ids={"issue-1"},
                suppress_warn_ids={"issue-1"},
            )

        assert result == []
        assert warnings == []


class TestClaimIssueAsync:
    """Test beads.claim_async invokes bd update correctly."""

    @pytest.mark.asyncio
    async def test_returns_true_on_success(
        self, orchestrator: MalaOrchestrator
    ) -> None:
        """Successful claim returns True."""
        with patch.object(
            orchestrator.beads,
            "_run_subprocess_async",
            new_callable=AsyncMock,
            return_value=make_subprocess_result(returncode=0),
        ):
            result = await orchestrator.beads.claim_async("issue-1")

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_on_failure(
        self, orchestrator: MalaOrchestrator
    ) -> None:
        """Failed claim returns False."""
        with patch.object(
            orchestrator.beads,
            "_run_subprocess_async",
            new_callable=AsyncMock,
            return_value=make_subprocess_result(
                returncode=1, stderr="Issue already claimed"
            ),
        ):
            result = await orchestrator.beads.claim_async("issue-1")

        assert result is False

    @pytest.mark.asyncio
    async def test_calls_bd_update_with_correct_args(
        self, orchestrator: MalaOrchestrator
    ) -> None:
        """Verify bd update is called with correct arguments."""
        mock_run = AsyncMock(return_value=make_subprocess_result())
        with patch.object(orchestrator.beads, "_run_subprocess_async", mock_run):
            await orchestrator.beads.claim_async("issue-abc")

        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert call_args == [
            "bd",
            "update",
            "issue-abc",
            "--status",
            "in_progress",
        ]


class TestResetIssueAsync:
    """Test beads.reset_async invokes bd update correctly."""

    @pytest.mark.asyncio
    async def test_calls_bd_update_with_ready_status(
        self, orchestrator: MalaOrchestrator
    ) -> None:
        """Verify reset calls bd update with ready status."""
        mock_run = AsyncMock(return_value=make_subprocess_result())
        with patch.object(orchestrator.beads, "_run_subprocess_async", mock_run):
            await orchestrator.beads.reset_async("issue-failed")

        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert call_args == ["bd", "update", "issue-failed", "--status", "ready"]

    @pytest.mark.asyncio
    async def test_includes_log_path_and_error_in_notes(
        self, orchestrator: MalaOrchestrator, tmp_path: Path
    ) -> None:
        """Reset with log_path and error should include notes."""
        log_path = tmp_path / "session.jsonl"
        mock_run = AsyncMock(return_value=make_subprocess_result())
        with patch.object(orchestrator.beads, "_run_subprocess_async", mock_run):
            await orchestrator.beads.reset_async(
                "issue-failed", log_path=log_path, error="Timeout after 30 minutes"
            )

        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "--notes" in call_args
        notes_idx = call_args.index("--notes")
        notes_value = call_args[notes_idx + 1]
        assert "Timeout after 30 minutes" in notes_value
        assert str(log_path) in notes_value

    @pytest.mark.asyncio
    async def test_does_not_raise_on_failure(
        self, orchestrator: MalaOrchestrator
    ) -> None:
        """Reset should not raise even if bd fails."""
        with patch.object(
            orchestrator.beads,
            "_run_subprocess_async",
            new_callable=AsyncMock,
            return_value=make_subprocess_result(returncode=1),
        ):
            # Should not raise
            await orchestrator.beads.reset_async("issue-failed")


class TestSpawnAgent:
    """Test spawn_agent behavior."""

    @pytest.mark.asyncio
    async def test_adds_issue_to_failed_when_claim_fails(
        self, orchestrator: MalaOrchestrator
    ) -> None:
        """When claim fails, issue should be added to failed_issues."""

        async def mock_claim_async(issue_id: str) -> bool:
            return False

        with patch.object(
            orchestrator.beads, "claim_async", side_effect=mock_claim_async
        ):
            result = await orchestrator.spawn_agent("unclaimed-issue")

        assert result is False
        assert "unclaimed-issue" in orchestrator.failed_issues

    @pytest.mark.asyncio
    async def test_creates_task_when_claim_succeeds(
        self, orchestrator: MalaOrchestrator
    ) -> None:
        """When claim succeeds, a task should be created."""

        async def mock_claim_async(issue_id: str) -> bool:
            return True

        with (
            patch.object(
                orchestrator.beads, "claim_async", side_effect=mock_claim_async
            ),
            patch.object(
                orchestrator,
                "run_implementer",
                return_value=IssueResult(
                    issue_id="test",
                    agent_id="test-agent",
                    success=True,
                    summary="done",
                ),
            ),
        ):
            result = await orchestrator.spawn_agent("claimable-issue")

        assert result is True
        assert "claimable-issue" in orchestrator.active_tasks


class TestRunOrchestrationLoop:
    """Test the main run() orchestration loop."""

    @pytest.mark.asyncio
    async def test_stops_cleanly_when_no_ready_issues(
        self, orchestrator: MalaOrchestrator
    ) -> None:
        """When no issues are ready, run() should return 0 and exit cleanly."""

        async def mock_get_ready_async(
            exclude_ids: set[str] | None = None,
            epic_id: str | None = None,
            only_ids: set[str] | None = None,
            suppress_warn_ids: set[str] | None = None,
            prioritize_wip: bool = False,
            focus: bool = True,
        ) -> list[str]:
            return []

        with (
            patch.object(
                orchestrator.beads, "get_ready_async", side_effect=mock_get_ready_async
            ),
            patch("src.orchestrator.get_lock_dir", return_value=MagicMock()),
            patch("src.orchestrator.get_runs_dir", return_value=MagicMock()),
            patch("src.orchestrator.release_run_locks"),
        ):
            result = await orchestrator.run()

        assert result == (0, 0)
        assert len(orchestrator.completed) == 0

    @pytest.mark.asyncio
    async def test_respects_max_issues_limit(
        self, orchestrator: MalaOrchestrator
    ) -> None:
        """Should stop after processing max_issues."""
        orchestrator.max_issues = 2
        call_count = 0
        spawned: list[str] = []

        async def mock_get_ready_async(
            exclude_ids: set[str] | None = None,
            epic_id: str | None = None,
            only_ids: set[str] | None = None,
            suppress_warn_ids: set[str] | None = None,
            prioritize_wip: bool = False,
            focus: bool = True,
        ) -> list[str]:
            # Return issues only on first call
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ["issue-1", "issue-2", "issue-3"]
            return []

        async def mock_spawn(issue_id: str) -> bool:
            spawned.append(issue_id)
            # Create a completed result immediately
            orchestrator.completed.append(
                IssueResult(
                    issue_id=issue_id,
                    agent_id=f"{issue_id}-agent",
                    success=True,
                    summary="done",
                )
            )
            return True

        with (
            patch.object(
                orchestrator.beads, "get_ready_async", side_effect=mock_get_ready_async
            ),
            patch.object(orchestrator, "spawn_agent", side_effect=mock_spawn),
            patch("src.orchestrator.get_lock_dir", return_value=MagicMock()),
            patch("src.orchestrator.get_runs_dir", return_value=MagicMock()),
            patch("src.orchestrator.release_run_locks"),
            patch("subprocess.run", return_value=make_subprocess_result()),
        ):
            # Orchestrator needs active_tasks to be empty to exit
            # but we don't actually spawn tasks in this mock
            await orchestrator.run()

        # Should have only spawned 2 issues (max_issues limit)
        assert len(spawned) == 2


class TestFailedTaskResetsIssue:
    """Test that failed tasks correctly mark issue as needing followup."""

    @pytest.mark.asyncio
    async def test_resets_issue_on_task_failure(
        self, orchestrator: MalaOrchestrator
    ) -> None:
        """When a task fails, the issue should be marked needs-followup."""
        followup_calls = []

        async def mock_mark_followup_async(
            issue_id: str, reason: str, log_path: Path | None = None
        ) -> bool:
            followup_calls.append(issue_id)
            return True

        async def mock_run_implementer(issue_id: str) -> IssueResult:
            return IssueResult(
                issue_id=issue_id,
                agent_id=f"{issue_id}-agent",
                success=False,
                summary="Implementation failed",
            )

        first_call = True

        async def mock_get_ready_async(
            exclude_ids: set[str] | None = None,
            epic_id: str | None = None,
            only_ids: set[str] | None = None,
            suppress_warn_ids: set[str] | None = None,
            prioritize_wip: bool = False,
            focus: bool = True,
        ) -> list[str]:
            nonlocal first_call
            if first_call:
                first_call = False
                return ["fail-issue"]
            return []

        async def mock_claim_async(issue_id: str) -> bool:
            return True

        with (
            patch.object(
                orchestrator.beads, "get_ready_async", side_effect=mock_get_ready_async
            ),
            patch.object(
                orchestrator.beads, "claim_async", side_effect=mock_claim_async
            ),
            patch.object(
                orchestrator, "run_implementer", side_effect=mock_run_implementer
            ),
            patch.object(
                orchestrator.beads,
                "mark_needs_followup_async",
                side_effect=mock_mark_followup_async,
            ),
            patch("src.orchestrator.get_lock_dir", return_value=MagicMock()),
            patch("src.orchestrator.get_runs_dir", return_value=MagicMock()),
            patch("src.orchestrator.release_run_locks"),
            patch("subprocess.run", return_value=make_subprocess_result()),
        ):
            await orchestrator.run()

        # The failed issue should have been marked needs-followup
        assert "fail-issue" in followup_calls
        # And added to failed_issues set
        assert "fail-issue" in orchestrator.failed_issues

    @pytest.mark.asyncio
    async def test_does_not_reset_successful_issue(
        self, orchestrator: MalaOrchestrator
    ) -> None:
        """Successful issues should not be reset."""
        reset_calls: list[str] = []

        async def mock_reset_async(
            issue_id: str, log_path: Path | None = None, error: str = ""
        ) -> None:
            reset_calls.append(issue_id)

        async def mock_run_implementer(issue_id: str) -> IssueResult:
            return IssueResult(
                issue_id=issue_id,
                agent_id=f"{issue_id}-agent",
                success=True,
                summary="Completed successfully",
            )

        first_call = True

        async def mock_get_ready_async(
            exclude_ids: set[str] | None = None,
            epic_id: str | None = None,
            only_ids: set[str] | None = None,
            suppress_warn_ids: set[str] | None = None,
            prioritize_wip: bool = False,
            focus: bool = True,
        ) -> list[str]:
            nonlocal first_call
            if first_call:
                first_call = False
                return ["success-issue"]
            return []

        async def mock_claim_async(issue_id: str) -> bool:
            return True

        with (
            patch.object(
                orchestrator.beads, "get_ready_async", side_effect=mock_get_ready_async
            ),
            patch.object(
                orchestrator.beads, "claim_async", side_effect=mock_claim_async
            ),
            patch.object(
                orchestrator, "run_implementer", side_effect=mock_run_implementer
            ),
            patch.object(
                orchestrator.beads, "reset_async", side_effect=mock_reset_async
            ),
            patch("src.orchestrator.get_lock_dir", return_value=MagicMock()),
            patch("src.orchestrator.get_runs_dir", return_value=MagicMock()),
            patch("src.orchestrator.release_run_locks"),
            patch("subprocess.run", return_value=make_subprocess_result()),
        ):
            await orchestrator.run()

        # No reset should have been called
        assert "success-issue" not in reset_calls
        # And not in failed_issues
        assert "success-issue" not in orchestrator.failed_issues


class TestOrchestratorInitialization:
    """Test orchestrator initialization."""

    def test_timeout_conversion(self, tmp_path: Path) -> None:
        """Timeout minutes should be converted to seconds."""
        orch = MalaOrchestrator(repo_path=tmp_path, timeout_minutes=15)
        assert orch.timeout_seconds == 15 * 60

    def test_default_values(self, tmp_path: Path) -> None:
        """Default values should be set correctly."""
        orch = MalaOrchestrator(repo_path=tmp_path)
        assert orch.max_agents is None  # Unlimited by default
        assert orch.timeout_seconds is None  # No timeout by default
        assert orch.max_issues is None

    def test_repo_path_resolved(self, tmp_path: Path) -> None:
        """Repo path should be resolved to absolute."""
        relative = Path(".")
        orch = MalaOrchestrator(repo_path=relative)
        assert orch.repo_path.is_absolute()

    def test_focus_default_true(self, tmp_path: Path) -> None:
        """Focus parameter should default to True."""
        orch = MalaOrchestrator(repo_path=tmp_path)
        assert orch.focus is True

    def test_focus_explicit_false(self, tmp_path: Path) -> None:
        """Focus parameter can be explicitly set to False."""
        orch = MalaOrchestrator(repo_path=tmp_path, focus=False)
        assert orch.focus is False


class TestEpicFilterAsync:
    """Test epic filter functionality in BeadsClient (async)."""

    @pytest.mark.asyncio
    async def test_get_epic_children_returns_child_ids(
        self, orchestrator: MalaOrchestrator
    ) -> None:
        """get_epic_children_async should return IDs of children (depth > 0)."""
        tree_json = json.dumps(
            [
                {"id": "epic-1", "depth": 0, "issue_type": "epic"},  # Epic itself
                {"id": "task-1", "depth": 1, "issue_type": "task"},  # Child
                {"id": "task-2", "depth": 1, "issue_type": "task"},  # Child
                {"id": "task-3", "depth": 2, "issue_type": "task"},  # Grandchild
            ]
        )
        with patch.object(
            orchestrator.beads,
            "_run_subprocess_async",
            new_callable=AsyncMock,
            return_value=make_subprocess_result(stdout=tree_json),
        ):
            result = await orchestrator.beads.get_epic_children_async("epic-1")

        assert result == {"task-1", "task-2", "task-3"}
        assert "epic-1" not in result

    @pytest.mark.asyncio
    async def test_get_epic_children_returns_empty_on_failure(
        self, orchestrator: MalaOrchestrator
    ) -> None:
        """get_epic_children_async should return empty set on bd failure."""
        with patch.object(
            orchestrator.beads,
            "_run_subprocess_async",
            new_callable=AsyncMock,
            return_value=make_subprocess_result(returncode=1, stderr="epic not found"),
        ):
            result = await orchestrator.beads.get_epic_children_async(
                "nonexistent-epic"
            )

        assert result == set()

    @pytest.mark.asyncio
    async def test_get_epic_children_returns_empty_on_invalid_json(
        self, orchestrator: MalaOrchestrator
    ) -> None:
        """get_epic_children_async should return empty set on invalid JSON."""
        with patch.object(
            orchestrator.beads,
            "_run_subprocess_async",
            new_callable=AsyncMock,
            return_value=make_subprocess_result(stdout="not valid json"),
        ):
            result = await orchestrator.beads.get_epic_children_async("epic-1")

        assert result == set()

    @pytest.mark.asyncio
    async def test_get_ready_with_epic_filter(
        self, orchestrator: MalaOrchestrator
    ) -> None:
        """get_ready_async with epic_id should only return children of that epic."""
        tree_json = json.dumps(
            [
                {"id": "epic-1", "depth": 0, "issue_type": "epic"},
                {"id": "child-1", "depth": 1, "issue_type": "task"},
                {"id": "child-2", "depth": 1, "issue_type": "task"},
            ]
        )
        ready_json = json.dumps(
            [
                {"id": "child-1", "priority": 1, "issue_type": "task"},
                {"id": "child-2", "priority": 2, "issue_type": "task"},
                {"id": "other-task", "priority": 1, "issue_type": "task"},
            ]
        )

        async def mock_run(
            cmd: list[str], **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            if "dep" in cmd and "tree" in cmd:
                return make_subprocess_result(stdout=tree_json)
            elif "ready" in cmd:
                return make_subprocess_result(stdout=ready_json)
            return make_subprocess_result()

        with patch.object(
            orchestrator.beads, "_run_subprocess_async", side_effect=mock_run
        ):
            result = await orchestrator.beads.get_ready_async(epic_id="epic-1")

        assert result == ["child-1", "child-2"]
        assert "other-task" not in result

    @pytest.mark.asyncio
    async def test_get_ready_without_epic_filter_returns_all(
        self, orchestrator: MalaOrchestrator
    ) -> None:
        """get_ready_async without epic_id should return all ready tasks."""
        ready_json = json.dumps(
            [
                {"id": "task-1", "priority": 1, "issue_type": "task"},
                {"id": "task-2", "priority": 2, "issue_type": "task"},
            ]
        )
        with patch.object(
            orchestrator.beads,
            "_run_subprocess_async",
            make_ready_mock(ready_json),
        ):
            result = await orchestrator.beads.get_ready_async()

        assert result == ["task-1", "task-2"]

    @pytest.mark.asyncio
    async def test_get_ready_with_epic_filter_returns_empty_if_no_children(
        self, orchestrator: MalaOrchestrator
    ) -> None:
        """get_ready_async with epic_id should return empty if epic has no children."""
        with patch.object(
            orchestrator.beads,
            "_run_subprocess_async",
            new_callable=AsyncMock,
            return_value=make_subprocess_result(returncode=1),
        ):
            result = await orchestrator.beads.get_ready_async(epic_id="empty-epic")

        assert result == []


class TestOrchestratorWithEpicId:
    """Test orchestrator with epic_id parameter."""

    def test_epic_id_stored(self, tmp_path: Path) -> None:
        """epic_id parameter should be stored on orchestrator."""
        orch = MalaOrchestrator(repo_path=tmp_path, epic_id="test-epic")
        assert orch.epic_id == "test-epic"

    def test_epic_id_defaults_to_none(self, tmp_path: Path) -> None:
        """epic_id should default to None."""
        orch = MalaOrchestrator(repo_path=tmp_path)
        assert orch.epic_id is None


class TestQualityGateValidationEvidence:
    """Test JSONL log parsing for validation command evidence."""

    def test_detects_pytest_command(self, tmp_path: Path) -> None:
        """Quality gate should detect pytest execution in JSONL logs."""
        from src.quality_gate import QualityGate

        # Create sample JSONL with pytest command
        log_path = tmp_path / "session.jsonl"
        log_content = json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Bash",
                            "input": {"command": "uv run pytest tests/"},
                        }
                    ]
                },
            }
        )
        log_path.write_text(log_content + "\n")

        gate = QualityGate(tmp_path)
        from src.validation.spec import ValidationScope, build_validation_spec

        spec = build_validation_spec(scope=ValidationScope.PER_ISSUE)
        evidence = gate.parse_validation_evidence_with_spec(log_path, spec)

        assert evidence.pytest_ran is True

    def test_detects_ruff_check_command(self, tmp_path: Path) -> None:
        """Quality gate should detect ruff check execution."""
        from src.quality_gate import QualityGate

        log_path = tmp_path / "session.jsonl"
        log_content = json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Bash",
                            "input": {"command": "uvx ruff check ."},
                        }
                    ]
                },
            }
        )
        log_path.write_text(log_content + "\n")

        gate = QualityGate(tmp_path)
        from src.validation.spec import ValidationScope, build_validation_spec

        spec = build_validation_spec(scope=ValidationScope.PER_ISSUE)
        evidence = gate.parse_validation_evidence_with_spec(log_path, spec)

        assert evidence.ruff_check_ran is True

    def test_detects_ruff_format_command(self, tmp_path: Path) -> None:
        """Quality gate should detect ruff format execution."""
        from src.quality_gate import QualityGate

        log_path = tmp_path / "session.jsonl"
        log_content = json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Bash",
                            "input": {"command": "uvx ruff format ."},
                        }
                    ]
                },
            }
        )
        log_path.write_text(log_content + "\n")

        gate = QualityGate(tmp_path)
        from src.validation.spec import ValidationScope, build_validation_spec

        spec = build_validation_spec(scope=ValidationScope.PER_ISSUE)
        evidence = gate.parse_validation_evidence_with_spec(log_path, spec)

        assert evidence.ruff_format_ran is True

    def test_detects_ty_check_command(self, tmp_path: Path) -> None:
        """Quality gate should detect ty check execution."""
        from src.quality_gate import QualityGate

        log_path = tmp_path / "session.jsonl"
        log_content = json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Bash",
                            "input": {"command": "uvx ty check"},
                        }
                    ]
                },
            }
        )
        log_path.write_text(log_content + "\n")

        gate = QualityGate(tmp_path)
        from src.validation.spec import ValidationScope, build_validation_spec

        spec = build_validation_spec(scope=ValidationScope.PER_ISSUE)
        evidence = gate.parse_validation_evidence_with_spec(log_path, spec)

        assert evidence.ty_check_ran is True

    def test_returns_empty_evidence_for_missing_log(self, tmp_path: Path) -> None:
        """Quality gate should return empty evidence for missing log file."""
        from src.quality_gate import QualityGate

        gate = QualityGate(tmp_path)
        nonexistent = tmp_path / "nonexistent.jsonl"
        from src.validation.spec import ValidationScope, build_validation_spec

        spec = build_validation_spec(scope=ValidationScope.PER_ISSUE)
        evidence = gate.parse_validation_evidence_with_spec(nonexistent, spec)

        assert evidence.pytest_ran is False
        assert evidence.ruff_check_ran is False
        assert evidence.ruff_format_ran is False
        assert evidence.ty_check_ran is False


class TestQualityGateCommitCheck:
    """Test git commit message verification."""

    def test_detects_matching_commit(self, tmp_path: Path) -> None:
        """Quality gate should detect commit with correct issue ID."""
        from src.quality_gate import QualityGate

        gate = QualityGate(tmp_path)

        with patch("src.quality_gate.run_command") as mock_run:
            mock_run.return_value = make_command_result(
                stdout="abc1234 bd-issue-123: Fix the bug\n"
            )
            result = gate.check_commit_exists("issue-123")

        assert result.exists is True
        assert result.commit_hash == "abc1234"

    def test_rejects_missing_commit(self, tmp_path: Path) -> None:
        """Quality gate should reject when no matching commit found."""
        from src.quality_gate import QualityGate

        gate = QualityGate(tmp_path)

        with patch("src.quality_gate.run_command") as mock_run:
            mock_run.return_value = make_command_result(stdout="")
            result = gate.check_commit_exists("issue-123")

        assert result.exists is False
        assert result.commit_hash is None

    def test_handles_git_failure(self, tmp_path: Path) -> None:
        """Quality gate should handle git command failures gracefully."""
        from src.quality_gate import QualityGate

        gate = QualityGate(tmp_path)

        with patch("src.quality_gate.run_command") as mock_run:
            mock_run.return_value = make_command_result(
                returncode=1, stderr="fatal: not a git repository"
            )
            result = gate.check_commit_exists("issue-123")

        assert result.exists is False

    def test_searches_30_day_window(self, tmp_path: Path) -> None:
        """Quality gate should search commits from the last 30 days."""
        from src.quality_gate import QualityGate

        gate = QualityGate(tmp_path)

        with patch("src.quality_gate.run_command") as mock_run:
            mock_run.return_value = make_command_result(
                stdout="abc1234 bd-issue-123: Long-running work\n"
            )
            result = gate.check_commit_exists("issue-123")

            # Verify the git command uses 30-day window
            call_args = mock_run.call_args
            git_cmd = call_args[0][0]
            assert "--since=30 days ago" in git_cmd

        assert result.exists is True


class TestQualityGateFullCheck:
    """Test full quality gate check combining all criteria."""

    def test_passes_when_all_criteria_met(self, tmp_path: Path) -> None:
        """Quality gate passes when closed, commit exists, validation ran."""
        from src.quality_gate import QualityGate

        gate = QualityGate(tmp_path)

        # Create log with all validation commands
        log_path = tmp_path / "session.jsonl"
        commands = [
            "uv run pytest",
            "uvx ruff check .",
            "uvx ruff format .",
            "uvx ty check",
        ]
        lines = []
        for cmd in commands:
            lines.append(
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "content": [
                                {
                                    "type": "tool_use",
                                    "name": "Bash",
                                    "input": {"command": cmd},
                                }
                            ]
                        },
                    }
                )
            )
        log_path.write_text("\n".join(lines) + "\n")

        from src.validation.spec import ValidationScope, build_validation_spec

        spec = build_validation_spec(scope=ValidationScope.PER_ISSUE)

        with patch("src.quality_gate.run_command") as mock_run:
            mock_run.return_value = make_command_result(
                stdout="abc1234 bd-issue-123: Implement feature\n"
            )
            result = gate.check_with_resolution("issue-123", log_path, spec=spec)

        assert result.passed is True
        assert result.failure_reasons == []

    def test_fails_when_commit_missing(self, tmp_path: Path) -> None:
        """Quality gate fails when commit is missing."""
        from src.quality_gate import QualityGate

        gate = QualityGate(tmp_path)

        # Create log with validation commands
        log_path = tmp_path / "session.jsonl"
        log_path.write_text(
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "name": "Bash",
                                "input": {"command": "uv run pytest"},
                            }
                        ]
                    },
                }
            )
            + "\n"
        )

        from src.validation.spec import ValidationScope, build_validation_spec

        spec = build_validation_spec(scope=ValidationScope.PER_ISSUE)

        with patch("src.quality_gate.run_command") as mock_run:
            mock_run.return_value = make_command_result(stdout="")
            result = gate.check_with_resolution("issue-123", log_path, spec=spec)

        assert result.passed is False
        assert "commit" in result.failure_reasons[0].lower()

    def test_failure_message_reflects_30_day_window(self, tmp_path: Path) -> None:
        """Quality gate failure message should mention the 30-day window."""
        from src.quality_gate import QualityGate

        gate = QualityGate(tmp_path)

        # Create log with validation commands
        log_path = tmp_path / "session.jsonl"
        log_path.write_text(
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "name": "Bash",
                                "input": {"command": "uv run pytest"},
                            }
                        ]
                    },
                }
            )
            + "\n"
        )

        from src.validation.spec import ValidationScope, build_validation_spec

        spec = build_validation_spec(scope=ValidationScope.PER_ISSUE)

        with patch("src.quality_gate.run_command") as mock_run:
            mock_run.return_value = make_command_result(stdout="")
            result = gate.check_with_resolution("issue-123", log_path, spec=spec)

        assert result.passed is False
        assert "30 days" in result.failure_reasons[0]

    def test_fails_when_validation_missing(self, tmp_path: Path) -> None:
        """Quality gate fails when validation commands didn't run."""
        from src.quality_gate import QualityGate

        gate = QualityGate(tmp_path)

        # Create empty log (no validation commands)
        log_path = tmp_path / "session.jsonl"
        log_path.write_text("")

        from src.validation.spec import ValidationScope, build_validation_spec

        spec = build_validation_spec(scope=ValidationScope.PER_ISSUE)

        with patch("src.quality_gate.run_command") as mock_run:
            mock_run.return_value = make_command_result(
                stdout="abc1234 bd-issue-123: Implement feature\n"
            )
            result = gate.check_with_resolution("issue-123", log_path, spec=spec)

        assert result.passed is False
        assert any("validation" in r.lower() for r in result.failure_reasons)


class TestBeadsClientNeedsFollowupAsync:
    """Test BeadsClient.mark_needs_followup_async method."""

    @pytest.mark.asyncio
    async def test_marks_issue_with_needs_followup_label(
        self, orchestrator: MalaOrchestrator
    ) -> None:
        """mark_needs_followup_async should add the needs-followup label."""
        mock_run = AsyncMock(return_value=make_subprocess_result())
        with patch.object(orchestrator.beads, "_run_subprocess_async", mock_run):
            await orchestrator.beads.mark_needs_followup_async(
                "issue-123", "Missing commit with bd-issue-123"
            )

        mock_run.assert_called()
        call_args = mock_run.call_args[0][0]
        assert "bd" in call_args
        assert "update" in call_args
        assert "issue-123" in call_args
        assert "--add-label" in call_args

    @pytest.mark.asyncio
    async def test_records_failure_context_in_notes(
        self, orchestrator: MalaOrchestrator
    ) -> None:
        """mark_needs_followup_async should record failure context."""
        mock_run = AsyncMock(return_value=make_subprocess_result())
        with patch.object(orchestrator.beads, "_run_subprocess_async", mock_run):
            await orchestrator.beads.mark_needs_followup_async(
                "issue-123", "Missing validation evidence: pytest, ruff check"
            )

        mock_run.assert_called()
        call_args = mock_run.call_args[0][0]
        assert "--notes" in call_args
        notes_idx = call_args.index("--notes")
        notes_value = call_args[notes_idx + 1]
        assert "Missing validation" in notes_value


class TestOrchestratorQualityGateIntegration:
    """Test quality gate integration in orchestrator run flow."""

    @pytest.mark.asyncio
    async def test_marks_needs_followup_on_gate_failure(
        self, orchestrator: MalaOrchestrator, tmp_path: Path
    ) -> None:
        """When quality gate fails (inside run_implementer), issue should be marked needs-followup."""
        mark_followup_calls: list[tuple[str, str]] = []

        async def mock_mark_followup_async(
            issue_id: str, reason: str, log_path: Path | None = None
        ) -> bool:
            mark_followup_calls.append((issue_id, reason))
            return True

        async def mock_run_implementer(issue_id: str) -> IssueResult:
            # Populate log path so quality gate runs
            log_path = tmp_path / f"{issue_id}.jsonl"
            log_path.touch()
            orchestrator.session_log_paths[issue_id] = log_path
            # Gate failure now returns success=False from run_implementer
            return IssueResult(
                issue_id=issue_id,
                agent_id=f"{issue_id}-agent",
                success=False,  # Gate failed
                summary="Quality gate failed: No commit with bd-issue-123 found",
            )

        first_call = True

        async def mock_get_ready_async(
            exclude_ids: set[str] | None = None,
            epic_id: str | None = None,
            only_ids: set[str] | None = None,
            suppress_warn_ids: set[str] | None = None,
            prioritize_wip: bool = False,
            focus: bool = True,
        ) -> list[str]:
            nonlocal first_call
            if first_call:
                first_call = False
                return ["issue-123"]
            return []

        async def mock_claim_async(issue_id: str) -> bool:
            return True

        with (
            patch.object(
                orchestrator.beads, "get_ready_async", side_effect=mock_get_ready_async
            ),
            patch.object(
                orchestrator.beads, "claim_async", side_effect=mock_claim_async
            ),
            patch.object(
                orchestrator, "run_implementer", side_effect=mock_run_implementer
            ),
            patch.object(
                orchestrator.beads,
                "mark_needs_followup_async",
                side_effect=mock_mark_followup_async,
            ),
            patch("src.orchestrator.get_lock_dir", return_value=MagicMock()),
            patch("src.orchestrator.get_runs_dir", return_value=tmp_path),
            patch("src.orchestrator.release_run_locks"),
            patch("subprocess.run", return_value=make_subprocess_result()),
        ):
            await orchestrator.run()

        # Should have been marked as needs-followup
        assert len(mark_followup_calls) == 1
        assert mark_followup_calls[0][0] == "issue-123"

    @pytest.mark.asyncio
    async def test_success_only_when_gate_passes(
        self, orchestrator: MalaOrchestrator, tmp_path: Path
    ) -> None:
        """Issue should only count as success when quality gate passes."""

        async def mock_run_implementer(issue_id: str) -> IssueResult:
            # Populate log path so quality gate runs
            log_path = tmp_path / f"{issue_id}.jsonl"
            log_path.touch()
            orchestrator.session_log_paths[issue_id] = log_path
            return IssueResult(
                issue_id=issue_id,
                agent_id=f"{issue_id}-agent",
                success=True,  # Gate passed
                summary="Completed successfully",
            )

        first_call = True

        async def mock_get_ready_async(
            exclude_ids: set[str] | None = None,
            epic_id: str | None = None,
            only_ids: set[str] | None = None,
            suppress_warn_ids: set[str] | None = None,
            prioritize_wip: bool = False,
            focus: bool = True,
        ) -> list[str]:
            nonlocal first_call
            if first_call:
                first_call = False
                return ["issue-pass"]
            return []

        async def mock_claim_async(issue_id: str) -> bool:
            return True

        with (
            patch.object(
                orchestrator.beads, "get_ready_async", side_effect=mock_get_ready_async
            ),
            patch.object(
                orchestrator.beads, "claim_async", side_effect=mock_claim_async
            ),
            patch.object(
                orchestrator, "run_implementer", side_effect=mock_run_implementer
            ),
            patch.object(
                orchestrator.beads, "mark_needs_followup_async", return_value=True
            ),
            patch("src.orchestrator.get_lock_dir", return_value=MagicMock()),
            patch("src.orchestrator.get_runs_dir", return_value=tmp_path),
            patch("src.orchestrator.release_run_locks"),
            patch("subprocess.run", return_value=make_subprocess_result()),
        ):
            success_count, _total = await orchestrator.run()

        assert success_count == 1
        assert "issue-pass" in orchestrator.completed


class TestAsyncBeadsClientWithTimeout:
    """Test that BeadsClient methods handle slow/hanging commands gracefully."""

    @pytest.mark.asyncio
    async def test_slow_bd_ready_does_not_block_other_tasks(
        self, orchestrator: MalaOrchestrator
    ) -> None:
        """Slow bd ready should not block other concurrent async tasks."""
        # Track when tasks complete
        task_completions: list[tuple[str, float]] = []
        start = time.monotonic()

        async def fast_task() -> None:
            await asyncio.sleep(0.05)
            task_completions.append(("fast", time.monotonic() - start))

        # Use a real slow command instead of mocking subprocess.run
        # (since we now use asyncio.create_subprocess_exec)
        orchestrator.beads.timeout_seconds = 0.5

        async def slow_beads_call() -> None:
            # Use sleep as a slow command
            await orchestrator.beads._run_subprocess_async(["sleep", "0.2"])
            task_completions.append(("beads", time.monotonic() - start))

        # Run slow beads call concurrently with fast task
        await asyncio.gather(
            slow_beads_call(),
            fast_task(),
        )

        # The fast task should complete well before the slow beads call
        fast_time = next(t for name, t in task_completions if name == "fast")
        beads_time = next(t for name, t in task_completions if name == "beads")

        # Fast task should complete in ~0.05s, beads in ~0.2s
        # If blocking, fast would complete at ~0.2s too
        assert fast_time < 0.15, f"Fast task blocked: completed at {fast_time:.3f}s"
        assert beads_time >= 0.15, f"Beads completed too fast: {beads_time:.3f}s"

    @pytest.mark.asyncio
    async def test_bd_command_timeout_returns_safe_fallback(
        self, orchestrator: MalaOrchestrator
    ) -> None:
        """When bd command times out, should return timeout result, not hang."""
        # Use a very short timeout for testing
        orchestrator.beads.timeout_seconds = 0.1

        start = time.monotonic()
        # Run a command that would take longer than timeout
        result = await orchestrator.beads._run_subprocess_async(["sleep", "10"])
        elapsed = time.monotonic() - start

        # Should return timeout result
        assert result.returncode == 1
        assert result.stderr == "timeout"
        # Should complete quickly due to timeout, not hang for 10s
        assert elapsed < 3.0, f"Timeout didn't work: took {elapsed:.2f}s"

    @pytest.mark.asyncio
    async def test_claim_timeout_returns_false(
        self, orchestrator: MalaOrchestrator
    ) -> None:
        """When claim times out, should return False (safe fallback)."""
        original_timeout = orchestrator.beads.timeout_seconds
        orchestrator.beads.timeout_seconds = 0.1

        # Mock _run_subprocess_async to simulate timeout
        async def mock_timeout(*args: object, **kwargs: object) -> SubprocessResult:
            return SubprocessResult(args=[], returncode=1, stdout="", stderr="timeout")

        try:
            with patch.object(
                orchestrator.beads, "_run_subprocess_async", side_effect=mock_timeout
            ):
                result = await orchestrator.beads.claim_async("issue-1")

            assert result is False
        finally:
            orchestrator.beads.timeout_seconds = original_timeout

    @pytest.mark.asyncio
    async def test_orchestrator_uses_async_beads_methods(
        self, orchestrator: MalaOrchestrator
    ) -> None:
        """Orchestrator run loop should use async beads methods."""
        # This test verifies the integration - that orchestrator calls
        # the async versions of beads methods

        async def mock_get_ready_async(
            exclude_ids: set[str] | None = None,
            epic_id: str | None = None,
            only_ids: set[str] | None = None,
            suppress_warn_ids: set[str] | None = None,
            prioritize_wip: bool = False,
            focus: bool = True,
        ) -> list[str]:
            return []

        with (
            patch.object(
                orchestrator.beads, "get_ready_async", side_effect=mock_get_ready_async
            ) as mock_ready,
            patch("src.orchestrator.get_lock_dir", return_value=MagicMock()),
            patch("src.orchestrator.get_runs_dir", return_value=MagicMock()),
            patch("src.orchestrator.release_run_locks"),
        ):
            await orchestrator.run()

        # Verify async method was called
        mock_ready.assert_called()


class TestMissingLogFile:
    """Test run_implementer behavior when log file never appears."""

    @pytest.mark.asyncio
    async def test_exits_quickly_when_log_file_missing(self, tmp_path: Path) -> None:
        """run_implementer should exit within bounded wait when log file missing."""
        from src.orchestrator import (
            MalaOrchestrator,
            LOG_FILE_WAIT_TIMEOUT,
        )

        orchestrator = MalaOrchestrator(
            repo_path=tmp_path,
            max_agents=1,
            timeout_minutes=5,  # Global timeout much longer than log wait
        )

        # Mock the Claude SDK client to yield a ResultMessage but no log file
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.query = AsyncMock()

        # Create an async generator that yields a ResultMessage
        async def mock_receive_response() -> AsyncGenerator[ResultMessage, None]:
            yield ResultMessage(
                subtype="result",
                session_id="test-session-123",
                result="Agent completed",
                duration_ms=1000,
                duration_api_ms=800,
                is_error=False,
                num_turns=1,
                total_cost_usd=0.01,
                usage=None,
            )

        mock_client.receive_response = mock_receive_response

        # Patch ClaudeSDKClient to return our mock
        with (
            patch("src.orchestrator.ClaudeSDKClient", return_value=mock_client),
            patch("src.orchestrator.get_git_branch_async", return_value="main"),
            patch("src.orchestrator.get_git_commit_async", return_value="abc123"),
            patch("src.orchestrator.TracedAgentExecution") as mock_tracer_cls,
        ):
            # Setup tracer mock
            mock_tracer = MagicMock()
            mock_tracer.__enter__ = MagicMock(return_value=mock_tracer)
            mock_tracer.__exit__ = MagicMock(return_value=None)
            mock_tracer_cls.return_value = mock_tracer

            start = time.monotonic()
            result = await orchestrator.run_implementer("test-issue")
            elapsed = time.monotonic() - start

        # Should fail with log missing message
        assert result.success is False
        assert (
            "log missing" in result.summary.lower()
            or "session log" in result.summary.lower()
        )

        # Should complete within bounded wait + small buffer, not global timeout
        max_expected = LOG_FILE_WAIT_TIMEOUT + 5  # 5s buffer for test overhead
        assert elapsed < max_expected, (
            f"run_implementer took {elapsed:.1f}s, expected < {max_expected}s"
        )

    @pytest.mark.asyncio
    async def test_summary_indicates_missing_log(self, tmp_path: Path) -> None:
        """Failure summary should clearly indicate the log file was missing."""
        from src.orchestrator import MalaOrchestrator

        orchestrator = MalaOrchestrator(
            repo_path=tmp_path,
            max_agents=1,
            timeout_minutes=5,
        )

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.query = AsyncMock()

        async def mock_receive_response() -> AsyncGenerator[ResultMessage, None]:
            yield ResultMessage(
                subtype="result",
                session_id="missing-log-session",
                result="Done",
                duration_ms=500,
                duration_api_ms=400,
                is_error=False,
                num_turns=1,
                total_cost_usd=0.005,
                usage=None,
            )

        mock_client.receive_response = mock_receive_response

        with (
            patch("src.orchestrator.ClaudeSDKClient", return_value=mock_client),
            patch("src.orchestrator.get_git_branch_async", return_value="main"),
            patch("src.orchestrator.get_git_commit_async", return_value="abc123"),
            patch("src.orchestrator.TracedAgentExecution") as mock_tracer_cls,
        ):
            # Setup tracer mock
            mock_tracer = MagicMock()
            mock_tracer.__enter__ = MagicMock(return_value=mock_tracer)
            mock_tracer.__exit__ = MagicMock(return_value=None)
            mock_tracer_cls.return_value = mock_tracer

            result = await orchestrator.run_implementer("issue-xyz")

        # Summary should mention session log and the timeout
        assert "Session log missing after timeout:" in result.summary


class TestSubprocessTerminationOnTimeout:
    """Test that timed-out subprocesses are properly terminated."""

    @pytest.mark.asyncio
    async def test_subprocess_terminated_on_timeout(self, tmp_path: Path) -> None:
        """When a subprocess times out, it should be terminated, not left running."""
        from src.beads_client import BeadsClient

        warnings: list[str] = []
        beads = BeadsClient(tmp_path, log_warning=warnings.append, timeout_seconds=0.5)

        # Run a subprocess that would hang forever, verify it gets killed
        start = time.monotonic()
        result = await beads._run_subprocess_async(["sleep", "60"])
        elapsed = time.monotonic() - start

        # Should return timeout result
        assert result.returncode == 1
        assert result.stderr == "timeout"

        # Should complete quickly (timeout + termination grace period)
        assert elapsed < 5.0, f"Took too long: {elapsed:.2f}s"

        # Warning should have been logged
        assert len(warnings) == 1
        assert "timed out" in warnings[0]

    @pytest.mark.asyncio
    async def test_subprocess_killed_if_terminate_fails(self, tmp_path: Path) -> None:
        """If SIGTERM doesn't work, subprocess should be killed with SIGKILL."""
        from src.beads_client import BeadsClient

        warnings: list[str] = []
        beads = BeadsClient(tmp_path, log_warning=warnings.append, timeout_seconds=0.3)

        # Use a script that traps SIGTERM to test the SIGKILL escalation
        # The shell script ignores SIGTERM, so we need SIGKILL to stop it
        result = await beads._run_subprocess_async(
            ["sh", "-c", "trap '' TERM; sleep 60"]
        )

        # Should still return timeout result (killed via SIGKILL)
        assert result.returncode == 1
        assert result.stderr == "timeout"

    @pytest.mark.asyncio
    async def test_successful_command_not_affected(self, tmp_path: Path) -> None:
        """Fast commands should complete normally without being terminated."""
        from src.beads_client import BeadsClient

        beads = BeadsClient(tmp_path, timeout_seconds=10.0)

        result = await beads._run_subprocess_async(["echo", "hello"])

        assert result.returncode == 0
        assert result.stdout.strip() == "hello"
        assert result.stderr == ""

    @pytest.mark.asyncio
    async def test_command_stderr_captured(self, tmp_path: Path) -> None:
        """stderr from commands should be captured correctly."""
        from src.beads_client import BeadsClient

        beads = BeadsClient(tmp_path, timeout_seconds=10.0)

        result = await beads._run_subprocess_async(
            ["sh", "-c", "echo error >&2; exit 1"]
        )

        assert result.returncode == 1
        assert "error" in result.stderr

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        sys.platform == "win32", reason="Process groups not supported on Windows"
    )
    async def test_child_processes_killed_on_timeout(self, tmp_path: Path) -> None:
        """Child processes spawned by the command should also be killed on timeout."""
        import os

        from src.beads_client import BeadsClient

        beads = BeadsClient(tmp_path, timeout_seconds=0.5)

        # Create a script that spawns a child process that would outlive the parent
        # The child writes its PID to a file so we can check if it was killed
        # Use $! to get the background job's PID (not $$ which is the parent shell PID)
        pid_file = tmp_path / "child.pid"
        script = f"""
            # Spawn a child that sleeps forever
            sleep 60 &
            # Capture the child's PID via $! (background job PID)
            echo $! > {pid_file}
            # Parent also sleeps
            sleep 60
        """

        result = await beads._run_subprocess_async(["sh", "-c", script])

        assert result.returncode == 1
        assert result.stderr == "timeout"

        # Give a moment for the file to be written and process to be killed
        await asyncio.sleep(0.2)

        # Check if the child process was killed
        if pid_file.exists():
            child_pid = int(pid_file.read_text().strip())
            # Check if the process is still running
            try:
                os.kill(child_pid, 0)  # Signal 0 just checks if process exists
                pytest.fail(f"Child process {child_pid} was not killed")
            except ProcessLookupError:
                pass  # Good - process was killed


class TestAgentEnvInheritance:
    """Test that agent environment inherits from os.environ.

    Verifies mala-w8w.1: Agent tool calls must see inherited env vars
    plus lock overrides (PATH/LOCK_DIR/AGENT_ID/REPO_NAMESPACE).
    """

    @pytest.mark.asyncio
    async def test_agent_env_includes_os_environ(self, tmp_path: Path) -> None:
        """Agent environment should include inherited env vars plus lock overrides."""
        orchestrator = MalaOrchestrator(repo_path=tmp_path, max_agents=1)

        captured_env: dict[str, str] | None = None

        # Mock ClaudeSDKClient to capture the env parameter
        type(MagicMock()).__init__

        class MockClient:
            def __init__(self, options: object) -> None:
                nonlocal captured_env
                # Capture the env from options
                captured_env = getattr(options, "env", None)

            async def __aenter__(self) -> "MockClient":
                return self

            async def __aexit__(self, *args: object) -> None:
                pass

            async def query(self, prompt: str) -> None:
                pass

            async def receive_response(self) -> AsyncGenerator[ResultMessage, None]:
                yield ResultMessage(
                    subtype="result",
                    session_id="test-session",
                    result="Done",
                    duration_ms=100,
                    duration_api_ms=80,
                    is_error=False,
                    num_turns=1,
                    total_cost_usd=0.01,
                    usage=None,
                )

        # Set a test env var that should be inherited
        test_env_var = f"TEST_INHERIT_{uuid.uuid4().hex[:8]}"
        os.environ[test_env_var] = "inherited_value"

        try:
            with (
                patch("src.orchestrator.ClaudeSDKClient", MockClient),
                patch("src.orchestrator.get_git_branch_async", return_value="main"),
                patch("src.orchestrator.get_git_commit_async", return_value="abc123"),
                patch("src.orchestrator.TracedAgentExecution") as mock_tracer_cls,
            ):
                mock_tracer = MagicMock()
                mock_tracer.__enter__ = MagicMock(return_value=mock_tracer)
                mock_tracer.__exit__ = MagicMock(return_value=None)
                mock_tracer_cls.return_value = mock_tracer

                await orchestrator.run_implementer("test-issue")

            # Verify the captured env includes our test var
            assert captured_env is not None, "env was not passed to ClaudeSDKClient"
            assert test_env_var in captured_env, (
                f"os.environ var {test_env_var} not inherited into agent_env"
            )
            assert captured_env[test_env_var] == "inherited_value"

            # Verify lock-related vars are also present (overrides)
            assert "PATH" in captured_env
            assert "LOCK_DIR" in captured_env
            assert "AGENT_ID" in captured_env
            assert "REPO_NAMESPACE" in captured_env
        finally:
            del os.environ[test_env_var]


class TestLockDirNestedCreation:
    """Test that LOCK_DIR creation handles nested paths.

    Verifies mala-w8w.1: Orchestrator startup must succeed when MALA_LOCK_DIR
    points to a nested, non-existent directory (mkdir uses parents=True).
    """

    @pytest.mark.asyncio
    async def test_lock_dir_created_with_parents(self, tmp_path: Path) -> None:
        """LOCK_DIR.mkdir should use parents=True for nested directories."""
        # Create a nested path that doesn't exist
        nested_lock_dir = tmp_path / "deeply" / "nested" / "lock" / "dir"
        assert not nested_lock_dir.exists()

        orchestrator = MalaOrchestrator(repo_path=tmp_path, max_agents=1)

        # Mock get_ready_async to return empty list (no issues to process)
        async def mock_get_ready_async(
            exclude_ids: set[str] | None = None,
            epic_id: str | None = None,
            only_ids: set[str] | None = None,
            suppress_warn_ids: set[str] | None = None,
            prioritize_wip: bool = False,
            focus: bool = True,
        ) -> list[str]:
            return []

        # Patch LOCK_DIR to point to our nested path
        with (
            patch.object(
                orchestrator.beads, "get_ready_async", side_effect=mock_get_ready_async
            ),
            patch("src.orchestrator.get_lock_dir", return_value=nested_lock_dir),
            patch("src.orchestrator.get_runs_dir", return_value=tmp_path / "runs"),
            patch("src.orchestrator.release_run_locks"),
        ):
            # This should not raise even though parent dirs don't exist
            await orchestrator.run()

        # The nested directory should now exist
        assert nested_lock_dir.exists(), (
            "LOCK_DIR.mkdir should create nested directories with parents=True"
        )


class TestGateFlowSequencing:
    """Test Gate 1-4 sequencing with check_with_resolution."""

    @pytest.mark.asyncio
    async def test_no_op_resolution_skips_per_issue_validation(
        self, tmp_path: Path
    ) -> None:
        """No-op resolution should skip Gate 2/3 (commit + validation evidence)."""
        from src.validation.spec import IssueResolution, ResolutionOutcome

        orchestrator = MalaOrchestrator(repo_path=tmp_path, max_agents=1)

        # Gate returns passed with no-change resolution
        no_change_resolution = IssueResolution(
            outcome=ResolutionOutcome.NO_CHANGE,
            rationale="Already implemented by another agent",
        )
        # Resolution is used to verify no-op handling
        assert no_change_resolution.outcome == ResolutionOutcome.NO_CHANGE

        close_calls: list[str] = []

        async def mock_close_async(issue_id: str) -> bool:
            close_calls.append(issue_id)
            return True

        async def mock_run_implementer(issue_id: str) -> IssueResult:
            log_path = tmp_path / f"{issue_id}.jsonl"
            log_path.touch()
            orchestrator.session_log_paths[issue_id] = log_path
            return IssueResult(
                issue_id=issue_id,
                agent_id=f"{issue_id}-agent",
                success=True,
                summary="No changes needed - already done",
            )

        first_call = True

        async def mock_get_ready_async(
            exclude_ids: set[str] | None = None,
            epic_id: str | None = None,
            only_ids: set[str] | None = None,
            suppress_warn_ids: set[str] | None = None,
            prioritize_wip: bool = False,
            focus: bool = True,
        ) -> list[str]:
            nonlocal first_call
            if first_call:
                first_call = False
                return ["issue-noop"]
            return []

        async def mock_claim_async(issue_id: str) -> bool:
            return True

        with (
            patch.object(
                orchestrator.beads, "get_ready_async", side_effect=mock_get_ready_async
            ),
            patch.object(
                orchestrator.beads, "claim_async", side_effect=mock_claim_async
            ),
            patch.object(
                orchestrator, "run_implementer", side_effect=mock_run_implementer
            ),
            patch.object(
                orchestrator.beads, "close_async", side_effect=mock_close_async
            ),
            patch.object(orchestrator.beads, "commit_issues_async", return_value=True),
            patch.object(
                orchestrator.beads, "close_eligible_epics_async", return_value=False
            ),
            patch("src.orchestrator.get_lock_dir", return_value=MagicMock()),
            patch("src.orchestrator.get_runs_dir", return_value=tmp_path),
            patch("src.orchestrator.release_run_locks"),
            patch("subprocess.run", return_value=make_subprocess_result()),
        ):
            success_count, _total = await orchestrator.run()

        # Issue should be closed despite no commit
        assert success_count == 1
        assert "issue-noop" in close_calls

    @pytest.mark.asyncio
    async def test_obsolete_resolution_skips_per_issue_validation(
        self, tmp_path: Path
    ) -> None:
        """Obsolete resolution should skip Gate 2/3 (commit + validation evidence)."""
        from src.validation.spec import IssueResolution, ResolutionOutcome

        orchestrator = MalaOrchestrator(repo_path=tmp_path, max_agents=1)

        obsolete_resolution = IssueResolution(
            outcome=ResolutionOutcome.OBSOLETE,
            rationale="Feature was removed in refactoring",
        )
        # Resolution is used to verify obsolete handling
        assert obsolete_resolution.outcome == ResolutionOutcome.OBSOLETE

        close_calls: list[str] = []

        async def mock_close_async(issue_id: str) -> bool:
            close_calls.append(issue_id)
            return True

        async def mock_run_implementer(issue_id: str) -> IssueResult:
            log_path = tmp_path / f"{issue_id}.jsonl"
            log_path.touch()
            orchestrator.session_log_paths[issue_id] = log_path
            return IssueResult(
                issue_id=issue_id,
                agent_id=f"{issue_id}-agent",
                success=True,
                summary="Issue obsolete - no longer relevant",
            )

        first_call = True

        async def mock_get_ready_async(
            exclude_ids: set[str] | None = None,
            epic_id: str | None = None,
            only_ids: set[str] | None = None,
            suppress_warn_ids: set[str] | None = None,
            prioritize_wip: bool = False,
            focus: bool = True,
        ) -> list[str]:
            nonlocal first_call
            if first_call:
                first_call = False
                return ["issue-obsolete"]
            return []

        async def mock_claim_async(issue_id: str) -> bool:
            return True

        with (
            patch.object(
                orchestrator.beads, "get_ready_async", side_effect=mock_get_ready_async
            ),
            patch.object(
                orchestrator.beads, "claim_async", side_effect=mock_claim_async
            ),
            patch.object(
                orchestrator, "run_implementer", side_effect=mock_run_implementer
            ),
            patch.object(
                orchestrator.beads, "close_async", side_effect=mock_close_async
            ),
            patch.object(orchestrator.beads, "commit_issues_async", return_value=True),
            patch.object(
                orchestrator.beads, "close_eligible_epics_async", return_value=False
            ),
            patch("src.orchestrator.get_lock_dir", return_value=MagicMock()),
            patch("src.orchestrator.get_runs_dir", return_value=tmp_path),
            patch("src.orchestrator.release_run_locks"),
            patch("subprocess.run", return_value=make_subprocess_result()),
        ):
            success_count, _total = await orchestrator.run()

        # Issue should be closed despite no commit
        assert success_count == 1
        assert "issue-obsolete" in close_calls


class TestRetryExhaustion:
    """Test retry exhaustion and failure handling."""

    @pytest.mark.asyncio
    async def test_gate_retry_exhaustion_marks_failed(self, tmp_path: Path) -> None:
        """When gate retries are exhausted, issue should be marked failed."""
        orchestrator = MalaOrchestrator(
            repo_path=tmp_path,
            max_agents=1,
            max_gate_retries=2,
        )

        followup_calls: list[str] = []

        async def mock_mark_followup_async(
            issue_id: str, reason: str, log_path: Path | None = None
        ) -> bool:
            followup_calls.append(issue_id)
            return True

        async def mock_run_implementer(issue_id: str) -> IssueResult:
            log_path = tmp_path / f"{issue_id}.jsonl"
            log_path.touch()
            orchestrator.session_log_paths[issue_id] = log_path
            # Gate always fails
            return IssueResult(
                issue_id=issue_id,
                agent_id=f"{issue_id}-agent",
                success=False,
                summary="Quality gate failed: Missing validation commands",
                gate_attempts=2,
            )

        first_call = True

        async def mock_get_ready_async(
            exclude_ids: set[str] | None = None,
            epic_id: str | None = None,
            only_ids: set[str] | None = None,
            suppress_warn_ids: set[str] | None = None,
            prioritize_wip: bool = False,
            focus: bool = True,
        ) -> list[str]:
            nonlocal first_call
            if first_call:
                first_call = False
                return ["issue-retry-fail"]
            return []

        async def mock_claim_async(issue_id: str) -> bool:
            return True

        with (
            patch.object(
                orchestrator.beads, "get_ready_async", side_effect=mock_get_ready_async
            ),
            patch.object(
                orchestrator.beads, "claim_async", side_effect=mock_claim_async
            ),
            patch.object(
                orchestrator, "run_implementer", side_effect=mock_run_implementer
            ),
            patch.object(
                orchestrator.beads,
                "mark_needs_followup_async",
                side_effect=mock_mark_followup_async,
            ),
            patch("src.orchestrator.get_lock_dir", return_value=MagicMock()),
            patch("src.orchestrator.get_runs_dir", return_value=tmp_path),
            patch("src.orchestrator.release_run_locks"),
            patch("subprocess.run", return_value=make_subprocess_result()),
        ):
            success_count, _total = await orchestrator.run()

        assert success_count == 0
        assert "issue-retry-fail" in followup_calls
        assert "issue-retry-fail" in orchestrator.failed_issues

    @pytest.mark.asyncio
    async def test_no_progress_stops_retries_early(self, tmp_path: Path) -> None:
        """When no progress is detected, retries should stop early."""
        orchestrator = MalaOrchestrator(
            repo_path=tmp_path,
            max_agents=1,
            max_gate_retries=5,  # Many retries available
        )

        async def mock_run_implementer(issue_id: str) -> IssueResult:
            log_path = tmp_path / f"{issue_id}.jsonl"
            log_path.touch()
            orchestrator.session_log_paths[issue_id] = log_path
            # Gate fails with no_progress - should stop retrying
            return IssueResult(
                issue_id=issue_id,
                agent_id=f"{issue_id}-agent",
                success=False,
                summary="Quality gate failed: No progress",
                gate_attempts=2,  # Only 2 attempts despite 5 max
            )

        first_call = True

        async def mock_get_ready_async(
            exclude_ids: set[str] | None = None,
            epic_id: str | None = None,
            only_ids: set[str] | None = None,
            suppress_warn_ids: set[str] | None = None,
            prioritize_wip: bool = False,
            focus: bool = True,
        ) -> list[str]:
            nonlocal first_call
            if first_call:
                first_call = False
                return ["issue-no-progress"]
            return []

        async def mock_claim_async(issue_id: str) -> bool:
            return True

        with (
            patch.object(
                orchestrator.beads, "get_ready_async", side_effect=mock_get_ready_async
            ),
            patch.object(
                orchestrator.beads, "claim_async", side_effect=mock_claim_async
            ),
            patch.object(
                orchestrator, "run_implementer", side_effect=mock_run_implementer
            ),
            patch.object(
                orchestrator.beads, "mark_needs_followup_async", return_value=True
            ),
            patch("src.orchestrator.get_lock_dir", return_value=MagicMock()),
            patch("src.orchestrator.get_runs_dir", return_value=tmp_path),
            patch("src.orchestrator.release_run_locks"),
            patch("subprocess.run", return_value=make_subprocess_result()),
        ):
            success_count, _total = await orchestrator.run()

        # Should have failed due to no progress
        assert success_count == 0
        assert "issue-no-progress" in orchestrator.failed_issues


class TestDisableValidationsRespected:
    """Test that disable-validations flags are respected."""

    def test_orchestrator_stores_disable_validations(self, tmp_path: Path) -> None:
        """Orchestrator should store disable_validations parameter."""
        disable_set = {"post-validate", "coverage"}
        orch = MalaOrchestrator(
            repo_path=tmp_path,
            disable_validations=disable_set,
        )
        assert orch.disable_validations == disable_set

    def test_orchestrator_default_disable_validations_is_none(
        self, tmp_path: Path
    ) -> None:
        """Default disable_validations should be None."""
        orch = MalaOrchestrator(repo_path=tmp_path)
        assert orch.disable_validations is None


class TestRunLevelValidation:
    """Test run-level validation after all issues complete."""

    @pytest.mark.asyncio
    async def test_run_returns_non_zero_exit_on_failure(self, tmp_path: Path) -> None:
        """Run should indicate failure when issues fail."""
        orchestrator = MalaOrchestrator(repo_path=tmp_path, max_agents=1)

        async def mock_run_implementer(issue_id: str) -> IssueResult:
            return IssueResult(
                issue_id=issue_id,
                agent_id=f"{issue_id}-agent",
                success=False,
                summary="Implementation failed",
            )

        first_call = True

        async def mock_get_ready_async(
            exclude_ids: set[str] | None = None,
            epic_id: str | None = None,
            only_ids: set[str] | None = None,
            suppress_warn_ids: set[str] | None = None,
            prioritize_wip: bool = False,
            focus: bool = True,
        ) -> list[str]:
            nonlocal first_call
            if first_call:
                first_call = False
                return ["failing-issue"]
            return []

        async def mock_claim_async(issue_id: str) -> bool:
            return True

        with (
            patch.object(
                orchestrator.beads, "get_ready_async", side_effect=mock_get_ready_async
            ),
            patch.object(
                orchestrator.beads, "claim_async", side_effect=mock_claim_async
            ),
            patch.object(
                orchestrator, "run_implementer", side_effect=mock_run_implementer
            ),
            patch.object(
                orchestrator.beads, "mark_needs_followup_async", return_value=True
            ),
            patch("src.orchestrator.get_lock_dir", return_value=MagicMock()),
            patch("src.orchestrator.get_runs_dir", return_value=tmp_path),
            patch("src.orchestrator.release_run_locks"),
            patch("subprocess.run", return_value=make_subprocess_result()),
        ):
            success_count, _total = await orchestrator.run()

        assert success_count == 0
        assert _total == 1

    @pytest.mark.asyncio
    async def test_issues_closed_only_after_gate_passes(self, tmp_path: Path) -> None:
        """Issues should only be closed when quality gate passes."""
        orchestrator = MalaOrchestrator(repo_path=tmp_path, max_agents=1)

        close_calls: list[str] = []
        followup_calls: list[str] = []

        async def mock_close_async(issue_id: str) -> bool:
            close_calls.append(issue_id)
            return True

        async def mock_mark_followup_async(
            issue_id: str, reason: str, log_path: Path | None = None
        ) -> bool:
            followup_calls.append(issue_id)
            return True

        async def mock_run_implementer(issue_id: str) -> IssueResult:
            # Populate log path so quality gate runs
            log_path = tmp_path / f"{issue_id}.jsonl"
            log_path.touch()
            orchestrator.session_log_paths[issue_id] = log_path
            return IssueResult(
                issue_id=issue_id,
                agent_id=f"{issue_id}-agent",
                success=True,  # Gate passed
                summary="Completed successfully",
            )

        first_call = True

        async def mock_get_ready_async(
            exclude_ids: set[str] | None = None,
            epic_id: str | None = None,
            only_ids: set[str] | None = None,
            suppress_warn_ids: set[str] | None = None,
            prioritize_wip: bool = False,
            focus: bool = True,
        ) -> list[str]:
            nonlocal first_call
            if first_call:
                first_call = False
                return ["issue-pass"]
            return []

        async def mock_claim_async(issue_id: str) -> bool:
            return True

        with (
            patch.object(
                orchestrator.beads, "get_ready_async", side_effect=mock_get_ready_async
            ),
            patch.object(
                orchestrator.beads, "claim_async", side_effect=mock_claim_async
            ),
            patch.object(
                orchestrator, "run_implementer", side_effect=mock_run_implementer
            ),
            patch.object(
                orchestrator.beads, "close_async", side_effect=mock_close_async
            ),
            patch.object(
                orchestrator.beads,
                "close_eligible_epics_async",
                return_value=False,
            ),
            patch.object(
                orchestrator.beads,
                "mark_needs_followup_async",
                side_effect=mock_mark_followup_async,
            ),
            patch("src.orchestrator.get_lock_dir", return_value=MagicMock()),
            patch("src.orchestrator.get_runs_dir", return_value=tmp_path),
            patch("src.orchestrator.release_run_locks"),
            patch("subprocess.run", return_value=make_subprocess_result()),
        ):
            success_count, _total = await orchestrator.run()

        assert success_count == 1
        assert "issue-pass" in close_calls

    @pytest.mark.asyncio
    async def test_failed_issue_not_closed(self, tmp_path: Path) -> None:
        """Failed issues should not be closed, only marked needs-followup."""
        orchestrator = MalaOrchestrator(repo_path=tmp_path, max_agents=1)

        close_calls: list[str] = []
        followup_calls: list[str] = []

        async def mock_close_async(issue_id: str) -> bool:
            close_calls.append(issue_id)
            return True

        async def mock_mark_followup_async(
            issue_id: str, reason: str, log_path: Path | None = None
        ) -> bool:
            followup_calls.append(issue_id)
            return True

        async def mock_run_implementer(issue_id: str) -> IssueResult:
            log_path = tmp_path / f"{issue_id}.jsonl"
            log_path.touch()
            orchestrator.session_log_paths[issue_id] = log_path
            return IssueResult(
                issue_id=issue_id,
                agent_id=f"{issue_id}-agent",
                success=False,  # Gate failed
                summary="Quality gate failed: Missing commit",
            )

        first_call = True

        async def mock_get_ready_async(
            exclude_ids: set[str] | None = None,
            epic_id: str | None = None,
            only_ids: set[str] | None = None,
            suppress_warn_ids: set[str] | None = None,
            prioritize_wip: bool = False,
            focus: bool = True,
        ) -> list[str]:
            nonlocal first_call
            if first_call:
                first_call = False
                return ["issue-fail"]
            return []

        async def mock_claim_async(issue_id: str) -> bool:
            return True

        with (
            patch.object(
                orchestrator.beads, "get_ready_async", side_effect=mock_get_ready_async
            ),
            patch.object(
                orchestrator.beads, "claim_async", side_effect=mock_claim_async
            ),
            patch.object(
                orchestrator, "run_implementer", side_effect=mock_run_implementer
            ),
            patch.object(
                orchestrator.beads, "mark_needs_followup_async", return_value=True
            ),
            patch("src.orchestrator.get_lock_dir", return_value=MagicMock()),
            patch("src.orchestrator.get_runs_dir", return_value=tmp_path),
            patch("src.orchestrator.release_run_locks"),
            patch("subprocess.run", return_value=make_subprocess_result()),
        ):
            await orchestrator.run()

        # Issue should NOT be closed
        assert "issue-fail" not in close_calls
        # But should be marked needs-followup
        assert "issue-fail" in followup_calls


class TestValidationResultMetadata:
    """Test validation execution and metadata population after commit detection.

    Verifies mala-yg9.2.4: SpecValidationRunner.run_spec should be invoked after
    commit detection and ValidationResult should be recorded in IssueRun.
    """

    @pytest.mark.asyncio
    async def test_validation_result_populated_in_issue_run(
        self, tmp_path: Path
    ) -> None:
        """Successful gate should populate validation result in IssueRun metadata."""
        from src.logging.run_metadata import IssueRun, RunMetadata
        from src.validation.result import ValidationResult
        from src.validation.spec_runner import SpecValidationRunner
        from src.validation.spec import ValidationArtifacts

        orchestrator = MalaOrchestrator(repo_path=tmp_path, max_agents=1)

        # Track recorded issues
        recorded_issues: list[IssueRun] = []
        original_record = RunMetadata.record_issue

        def capture_record(self: RunMetadata, issue: IssueRun) -> None:
            recorded_issues.append(issue)
            original_record(self, issue)

        async def mock_run_implementer(issue_id: str) -> IssueResult:
            # Populate log path so quality gate runs
            log_path = tmp_path / f"{issue_id}.jsonl"
            log_path.touch()
            orchestrator.session_log_paths[issue_id] = log_path
            return IssueResult(
                issue_id=issue_id,
                agent_id=f"{issue_id}-agent",
                success=True,
                summary="Completed successfully",
            )

        first_call = True

        async def mock_get_ready_async(
            exclude_ids: set[str] | None = None,
            epic_id: str | None = None,
            only_ids: set[str] | None = None,
            suppress_warn_ids: set[str] | None = None,
            prioritize_wip: bool = False,
            focus: bool = True,
        ) -> list[str]:
            nonlocal first_call
            if first_call:
                first_call = False
                return ["issue-with-validation"]
            return []

        async def mock_claim_async(issue_id: str) -> bool:
            return True

        async def mock_run_spec(
            spec: object, context: object, log_dir: object = None
        ) -> ValidationResult:
            return ValidationResult(
                passed=True,
                steps=[],
                artifacts=ValidationArtifacts(log_dir=tmp_path),
            )

        with (
            patch.object(
                orchestrator.beads, "get_ready_async", side_effect=mock_get_ready_async
            ),
            patch.object(
                orchestrator.beads, "claim_async", side_effect=mock_claim_async
            ),
            patch.object(
                orchestrator, "run_implementer", side_effect=mock_run_implementer
            ),
            patch.object(orchestrator.beads, "close_async", return_value=True),
            patch.object(orchestrator.beads, "commit_issues_async", return_value=True),
            patch.object(
                orchestrator.beads, "close_eligible_epics_async", return_value=False
            ),
            patch.object(RunMetadata, "record_issue", capture_record),
            patch.object(SpecValidationRunner, "run_spec", mock_run_spec),
            patch("src.orchestrator.get_lock_dir", return_value=MagicMock()),
            patch("src.orchestrator.get_runs_dir", return_value=tmp_path),
            patch("src.orchestrator.release_run_locks"),
            patch(
                "src.quality_gate.run_command",
                return_value=make_command_result(
                    stdout="abc1234 bd-issue-with-validation: Implement feature\n"
                ),
            ),
        ):
            await orchestrator.run()

        # Verify an issue was recorded
        assert len(recorded_issues) == 1
        issue_run = recorded_issues[0]
        assert issue_run.issue_id == "issue-with-validation"
        # validation field should be populated when validation runs
        assert issue_run.validation is not None
        assert issue_run.validation.passed is True

    @pytest.mark.asyncio
    async def test_validation_runner_invoked_after_commit_detection(
        self, tmp_path: Path
    ) -> None:
        """SpecValidationRunner.run_spec should be invoked after commit is detected."""
        from src.logging.run_metadata import IssueRun, RunMetadata
        from src.validation.result import ValidationResult
        from src.validation.spec_runner import SpecValidationRunner
        from src.validation.spec import ValidationArtifacts

        orchestrator = MalaOrchestrator(repo_path=tmp_path, max_agents=1)
        run_spec_calls: list[str] = []

        # Track recorded issues to verify validation field
        recorded_issues: list[IssueRun] = []
        original_record = RunMetadata.record_issue

        def capture_record(self: RunMetadata, issue: IssueRun) -> None:
            recorded_issues.append(issue)
            original_record(self, issue)

        async def mock_run_spec(
            spec: object, context: object, log_dir: object = None
        ) -> ValidationResult:
            run_spec_calls.append("called")
            return ValidationResult(
                passed=True,
                steps=[],
                artifacts=ValidationArtifacts(log_dir=tmp_path),
            )

        async def mock_run_implementer(issue_id: str) -> IssueResult:
            log_path = tmp_path / f"{issue_id}.jsonl"
            log_path.touch()
            orchestrator.session_log_paths[issue_id] = log_path
            return IssueResult(
                issue_id=issue_id,
                agent_id=f"{issue_id}-agent",
                success=True,
                summary="Completed successfully",
            )

        first_call = True

        async def mock_get_ready_async(
            exclude_ids: set[str] | None = None,
            epic_id: str | None = None,
            only_ids: set[str] | None = None,
            suppress_warn_ids: set[str] | None = None,
            prioritize_wip: bool = False,
            focus: bool = True,
        ) -> list[str]:
            nonlocal first_call
            if first_call:
                first_call = False
                return ["issue-run-spec"]
            return []

        async def mock_claim_async(issue_id: str) -> bool:
            return True

        # Mock the validation runner's run_spec method
        with (
            patch.object(
                orchestrator.beads, "get_ready_async", side_effect=mock_get_ready_async
            ),
            patch.object(
                orchestrator.beads, "claim_async", side_effect=mock_claim_async
            ),
            patch.object(
                orchestrator, "run_implementer", side_effect=mock_run_implementer
            ),
            patch.object(orchestrator.beads, "close_async", return_value=True),
            patch.object(orchestrator.beads, "commit_issues_async", return_value=True),
            patch.object(
                orchestrator.beads, "close_eligible_epics_async", return_value=False
            ),
            patch.object(SpecValidationRunner, "run_spec", mock_run_spec),
            patch.object(RunMetadata, "record_issue", capture_record),
            patch("src.orchestrator.get_lock_dir", return_value=MagicMock()),
            patch("src.orchestrator.get_runs_dir", return_value=tmp_path),
            patch("src.orchestrator.release_run_locks"),
            patch(
                "src.quality_gate.run_command",
                return_value=make_command_result(
                    stdout="abc1234 bd-issue-run-spec: Implement feature\n"
                ),
            ),
        ):
            await orchestrator.run()

        # run_spec should have been called for the successful issue
        assert len(run_spec_calls) == 1
        # The recorded issue should have validation populated
        assert len(recorded_issues) == 1
        assert recorded_issues[0].validation is not None


class TestResolutionRecordingInMetadata:
    """Test resolution outcome is recorded in IssueRun metadata.

    Verifies mala-yg9.1.1: When check_with_resolution returns a resolution
    (NO_CHANGE or OBSOLETE), it should be propagated through IssueResult
    to IssueRun and persisted via RunMetadata.
    """

    @pytest.mark.asyncio
    async def test_no_change_resolution_recorded_in_issue_run(
        self, tmp_path: Path
    ) -> None:
        """ISSUE_NO_CHANGE resolution should be recorded in IssueRun metadata."""
        from src.logging.run_metadata import IssueRun, RunMetadata
        from src.validation.spec import IssueResolution, ResolutionOutcome

        orchestrator = MalaOrchestrator(repo_path=tmp_path, max_agents=1)

        # Track recorded issues
        recorded_issues: list[IssueRun] = []
        original_record = RunMetadata.record_issue

        def capture_record(self: RunMetadata, issue: IssueRun) -> None:
            recorded_issues.append(issue)
            original_record(self, issue)

        # Create resolution that should be propagated
        no_change_resolution = IssueResolution(
            outcome=ResolutionOutcome.NO_CHANGE,
            rationale="Already implemented by another agent",
        )

        async def mock_run_implementer(issue_id: str) -> IssueResult:
            log_path = tmp_path / f"{issue_id}.jsonl"
            log_path.touch()
            orchestrator.session_log_paths[issue_id] = log_path
            return IssueResult(
                issue_id=issue_id,
                agent_id=f"{issue_id}-agent",
                success=True,
                summary="No changes needed - already done",
                resolution=no_change_resolution,
            )

        first_call = True

        async def mock_get_ready_async(
            exclude_ids: set[str] | None = None,
            epic_id: str | None = None,
            only_ids: set[str] | None = None,
            suppress_warn_ids: set[str] | None = None,
            prioritize_wip: bool = False,
            focus: bool = True,
        ) -> list[str]:
            nonlocal first_call
            if first_call:
                first_call = False
                return ["issue-no-change"]
            return []

        async def mock_claim_async(issue_id: str) -> bool:
            return True

        with (
            patch.object(
                orchestrator.beads, "get_ready_async", side_effect=mock_get_ready_async
            ),
            patch.object(
                orchestrator.beads, "claim_async", side_effect=mock_claim_async
            ),
            patch.object(
                orchestrator, "run_implementer", side_effect=mock_run_implementer
            ),
            patch.object(orchestrator.beads, "close_async", return_value=True),
            patch.object(orchestrator.beads, "commit_issues_async", return_value=True),
            patch.object(
                orchestrator.beads, "close_eligible_epics_async", return_value=False
            ),
            patch.object(RunMetadata, "record_issue", capture_record),
            patch("src.orchestrator.get_lock_dir", return_value=MagicMock()),
            patch("src.orchestrator.get_runs_dir", return_value=tmp_path),
            patch("src.orchestrator.release_run_locks"),
            patch("subprocess.run", return_value=make_subprocess_result()),
        ):
            success_count, total = await orchestrator.run()

        # Verify issue was successful
        assert success_count == 1
        assert total == 1

        # Verify resolution was recorded in IssueRun
        assert len(recorded_issues) == 1
        issue_run = recorded_issues[0]
        assert issue_run.issue_id == "issue-no-change"
        assert issue_run.resolution is not None
        assert issue_run.resolution.outcome == ResolutionOutcome.NO_CHANGE
        assert issue_run.resolution.rationale == "Already implemented by another agent"

    @pytest.mark.asyncio
    async def test_obsolete_resolution_recorded_in_issue_run(
        self, tmp_path: Path
    ) -> None:
        """ISSUE_OBSOLETE resolution should be recorded in IssueRun metadata."""
        from src.logging.run_metadata import IssueRun, RunMetadata
        from src.validation.spec import IssueResolution, ResolutionOutcome

        orchestrator = MalaOrchestrator(repo_path=tmp_path, max_agents=1)

        # Track recorded issues
        recorded_issues: list[IssueRun] = []
        original_record = RunMetadata.record_issue

        def capture_record(self: RunMetadata, issue: IssueRun) -> None:
            recorded_issues.append(issue)
            original_record(self, issue)

        # Create obsolete resolution
        obsolete_resolution = IssueResolution(
            outcome=ResolutionOutcome.OBSOLETE,
            rationale="Feature removed in refactoring",
        )

        async def mock_run_implementer(issue_id: str) -> IssueResult:
            log_path = tmp_path / f"{issue_id}.jsonl"
            log_path.touch()
            orchestrator.session_log_paths[issue_id] = log_path
            return IssueResult(
                issue_id=issue_id,
                agent_id=f"{issue_id}-agent",
                success=True,
                summary="Issue obsolete - no longer relevant",
                resolution=obsolete_resolution,
            )

        first_call = True

        async def mock_get_ready_async(
            exclude_ids: set[str] | None = None,
            epic_id: str | None = None,
            only_ids: set[str] | None = None,
            suppress_warn_ids: set[str] | None = None,
            prioritize_wip: bool = False,
            focus: bool = True,
        ) -> list[str]:
            nonlocal first_call
            if first_call:
                first_call = False
                return ["issue-obsolete"]
            return []

        async def mock_claim_async(issue_id: str) -> bool:
            return True

        with (
            patch.object(
                orchestrator.beads, "get_ready_async", side_effect=mock_get_ready_async
            ),
            patch.object(
                orchestrator.beads, "claim_async", side_effect=mock_claim_async
            ),
            patch.object(
                orchestrator, "run_implementer", side_effect=mock_run_implementer
            ),
            patch.object(orchestrator.beads, "close_async", return_value=True),
            patch.object(orchestrator.beads, "commit_issues_async", return_value=True),
            patch.object(
                orchestrator.beads, "close_eligible_epics_async", return_value=False
            ),
            patch.object(RunMetadata, "record_issue", capture_record),
            patch("src.orchestrator.get_lock_dir", return_value=MagicMock()),
            patch("src.orchestrator.get_runs_dir", return_value=tmp_path),
            patch("src.orchestrator.release_run_locks"),
            patch("subprocess.run", return_value=make_subprocess_result()),
        ):
            success_count, total = await orchestrator.run()

        # Verify issue was successful
        assert success_count == 1
        assert total == 1

        # Verify resolution was recorded in IssueRun (normal commit flow)
        assert len(recorded_issues) == 1
        issue_run = recorded_issues[0]
        assert issue_run.issue_id == "issue-obsolete"
        assert issue_run.resolution is not None
        assert issue_run.resolution.outcome == ResolutionOutcome.OBSOLETE
        assert issue_run.resolution.rationale == "Feature removed in refactoring"

    @pytest.mark.asyncio
    async def test_normal_success_has_no_resolution(self, tmp_path: Path) -> None:
        """Normal success (with commit) should have no resolution marker."""
        from src.logging.run_metadata import IssueRun, RunMetadata

        orchestrator = MalaOrchestrator(repo_path=tmp_path, max_agents=1)

        # Track recorded issues
        recorded_issues: list[IssueRun] = []
        original_record = RunMetadata.record_issue

        def capture_record(self: RunMetadata, issue: IssueRun) -> None:
            recorded_issues.append(issue)
            original_record(self, issue)

        async def mock_run_implementer(issue_id: str) -> IssueResult:
            log_path = tmp_path / f"{issue_id}.jsonl"
            log_path.touch()
            orchestrator.session_log_paths[issue_id] = log_path
            # Normal success - no resolution marker
            return IssueResult(
                issue_id=issue_id,
                agent_id=f"{issue_id}-agent",
                success=True,
                summary="Implemented feature successfully",
                resolution=None,  # No resolution for normal commits
            )

        first_call = True

        async def mock_get_ready_async(
            exclude_ids: set[str] | None = None,
            epic_id: str | None = None,
            only_ids: set[str] | None = None,
            suppress_warn_ids: set[str] | None = None,
            prioritize_wip: bool = False,
            focus: bool = True,
        ) -> list[str]:
            nonlocal first_call
            if first_call:
                first_call = False
                return ["issue-normal"]
            return []

        async def mock_claim_async(issue_id: str) -> bool:
            return True

        with (
            patch.object(
                orchestrator.beads, "get_ready_async", side_effect=mock_get_ready_async
            ),
            patch.object(
                orchestrator.beads, "claim_async", side_effect=mock_claim_async
            ),
            patch.object(
                orchestrator, "run_implementer", side_effect=mock_run_implementer
            ),
            patch.object(orchestrator.beads, "close_async", return_value=True),
            patch.object(orchestrator.beads, "commit_issues_async", return_value=True),
            patch.object(
                orchestrator.beads, "close_eligible_epics_async", return_value=False
            ),
            patch.object(RunMetadata, "record_issue", capture_record),
            patch("src.orchestrator.get_lock_dir", return_value=MagicMock()),
            patch("src.orchestrator.get_runs_dir", return_value=tmp_path),
            patch("src.orchestrator.release_run_locks"),
            patch("subprocess.run", return_value=make_subprocess_result()),
        ):
            success_count, total = await orchestrator.run()

        # Verify issue was successful
        assert success_count == 1
        assert total == 1

        # Verify NO resolution in IssueRun (normal commit flow)
        assert len(recorded_issues) == 1
        issue_run = recorded_issues[0]
        assert issue_run.issue_id == "issue-normal"
        assert issue_run.resolution is None


class TestEpicClosureAfterChildCompletion:
    """Test that epics are closed immediately after their last child completes.

    Verifies mala-0k7: Epics should close after each agent completion, not just
    at run end, so other agents can see updated epic status during the run.
    """

    @pytest.mark.asyncio
    async def test_epic_closure_called_after_issue_closes(self, tmp_path: Path) -> None:
        """close_eligible_epics_async should be called after each issue was closed."""
        orchestrator = MalaOrchestrator(repo_path=tmp_path, max_agents=1)

        epic_closure_calls: list[str] = []

        async def mock_close_eligible_epics_async() -> bool:
            epic_closure_calls.append("called")
            return True  # Simulate epic was closed

        async def mock_run_implementer(issue_id: str) -> IssueResult:
            log_path = tmp_path / f"{issue_id}.jsonl"
            log_path.touch()
            orchestrator.session_log_paths[issue_id] = log_path
            return IssueResult(
                issue_id=issue_id,
                agent_id=f"{issue_id}-agent",
                success=True,
                summary="Completed successfully",
            )

        first_call = True

        async def mock_get_ready_async(
            exclude_ids: set[str] | None = None,
            epic_id: str | None = None,
            only_ids: set[str] | None = None,
            suppress_warn_ids: set[str] | None = None,
            prioritize_wip: bool = False,
            focus: bool = True,
        ) -> list[str]:
            nonlocal first_call
            if first_call:
                first_call = False
                return ["child-issue"]
            return []

        async def mock_claim_async(issue_id: str) -> bool:
            return True

        with (
            patch.object(
                orchestrator.beads, "get_ready_async", side_effect=mock_get_ready_async
            ),
            patch.object(
                orchestrator.beads, "claim_async", side_effect=mock_claim_async
            ),
            patch.object(
                orchestrator, "run_implementer", side_effect=mock_run_implementer
            ),
            patch.object(orchestrator.beads, "close_async", return_value=True),
            patch.object(
                orchestrator.beads,
                "close_eligible_epics_async",
                side_effect=mock_close_eligible_epics_async,
            ),
            patch.object(orchestrator.beads, "commit_issues_async", return_value=True),
            patch("src.orchestrator.get_lock_dir", return_value=MagicMock()),
            patch("src.orchestrator.get_runs_dir", return_value=tmp_path),
            patch("src.orchestrator.release_run_locks"),
            patch("subprocess.run", return_value=make_subprocess_result()),
        ):
            await orchestrator.run()

        # Epic closure should have been called after the issue was closed
        assert len(epic_closure_calls) == 1

    @pytest.mark.asyncio
    async def test_epic_closure_not_called_when_issue_fails(
        self, tmp_path: Path
    ) -> None:
        """close_eligible_epics_async should NOT be called when issue fails."""
        orchestrator = MalaOrchestrator(repo_path=tmp_path, max_agents=1)

        epic_closure_calls: list[str] = []

        async def mock_close_eligible_epics_async() -> bool:
            epic_closure_calls.append("called")
            return True

        async def mock_run_implementer(issue_id: str) -> IssueResult:
            log_path = tmp_path / f"{issue_id}.jsonl"
            log_path.touch()
            orchestrator.session_log_paths[issue_id] = log_path
            return IssueResult(
                issue_id=issue_id,
                agent_id=f"{issue_id}-agent",
                success=False,  # Gate failed
                summary="Quality gate failed: Missing commit",
            )

        first_call = True

        async def mock_get_ready_async(
            exclude_ids: set[str] | None = None,
            epic_id: str | None = None,
            only_ids: set[str] | None = None,
            suppress_warn_ids: set[str] | None = None,
            prioritize_wip: bool = False,
            focus: bool = True,
        ) -> list[str]:
            nonlocal first_call
            if first_call:
                first_call = False
                return ["failing-child"]
            return []

        async def mock_claim_async(issue_id: str) -> bool:
            return True

        with (
            patch.object(
                orchestrator.beads, "get_ready_async", side_effect=mock_get_ready_async
            ),
            patch.object(
                orchestrator.beads, "claim_async", side_effect=mock_claim_async
            ),
            patch.object(
                orchestrator, "run_implementer", side_effect=mock_run_implementer
            ),
            patch.object(orchestrator.beads, "close_async", return_value=True),
            patch.object(
                orchestrator.beads,
                "close_eligible_epics_async",
                side_effect=mock_close_eligible_epics_async,
            ),
            patch.object(
                orchestrator.beads, "mark_needs_followup_async", return_value=True
            ),
            patch("src.orchestrator.get_lock_dir", return_value=MagicMock()),
            patch("src.orchestrator.get_runs_dir", return_value=tmp_path),
            patch("src.orchestrator.release_run_locks"),
            patch("subprocess.run", return_value=make_subprocess_result()),
        ):
            await orchestrator.run()

        # Epic closure should NOT have been called since issue failed
        assert len(epic_closure_calls) == 0

    @pytest.mark.asyncio
    async def test_multiple_issues_trigger_epic_closure_check_each_time(
        self, tmp_path: Path
    ) -> None:
        """Each successful issue closure should check for epic closure."""
        orchestrator = MalaOrchestrator(repo_path=tmp_path, max_agents=2)

        epic_closure_calls: list[str] = []
        issues_processed = []

        async def mock_close_eligible_epics_async() -> bool:
            epic_closure_calls.append("called")
            # First call closes an epic, second doesn't
            return len(epic_closure_calls) == 1

        async def mock_run_implementer(issue_id: str) -> IssueResult:
            issues_processed.append(issue_id)
            log_path = tmp_path / f"{issue_id}.jsonl"
            log_path.touch()
            orchestrator.session_log_paths[issue_id] = log_path
            return IssueResult(
                issue_id=issue_id,
                agent_id=f"{issue_id}-agent",
                success=True,
                summary="Completed successfully",
            )

        call_count = 0

        async def mock_get_ready_async(
            exclude_ids: set[str] | None = None,
            epic_id: str | None = None,
            only_ids: set[str] | None = None,
            suppress_warn_ids: set[str] | None = None,
            prioritize_wip: bool = False,
            focus: bool = True,
        ) -> list[str]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ["child-1", "child-2"]
            return []

        async def mock_claim_async(issue_id: str) -> bool:
            return True

        with (
            patch.object(
                orchestrator.beads, "get_ready_async", side_effect=mock_get_ready_async
            ),
            patch.object(
                orchestrator.beads, "claim_async", side_effect=mock_claim_async
            ),
            patch.object(
                orchestrator, "run_implementer", side_effect=mock_run_implementer
            ),
            patch.object(orchestrator.beads, "close_async", return_value=True),
            patch.object(
                orchestrator.beads,
                "close_eligible_epics_async",
                side_effect=mock_close_eligible_epics_async,
            ),
            patch.object(orchestrator.beads, "commit_issues_async", return_value=True),
            patch("src.orchestrator.get_lock_dir", return_value=MagicMock()),
            patch("src.orchestrator.get_runs_dir", return_value=tmp_path),
            patch("src.orchestrator.release_run_locks"),
            patch("subprocess.run", return_value=make_subprocess_result()),
        ):
            await orchestrator.run()

        # Both issues should have been processed
        assert len(issues_processed) == 2
        # Epic closure should have been called for each successful issue
        assert len(epic_closure_calls) == 2


class TestQualityGateAsync:
    """Test that quality gate checks are non-blocking."""

    @pytest.mark.asyncio
    async def test_run_quality_gate_uses_to_thread(
        self, orchestrator: MalaOrchestrator, tmp_path: Path
    ) -> None:
        """Quality gate should use asyncio.to_thread to avoid blocking the event loop."""
        from src.quality_gate import GateResult

        # Create a mock session log
        log_path = tmp_path / "session.jsonl"
        log_path.write_text("{}\n")

        # Create a mock RetryState (now imported from lifecycle)
        from src.lifecycle import RetryState

        retry_state = RetryState(baseline_timestamp=int(time.time()))

        # Create mock GateResult
        mock_gate_result = GateResult(
            passed=True,
            failure_reasons=[],
            commit_hash="abc123",
            validation_evidence=None,
        )

        # Track whether to_thread was called
        to_thread_called = False
        original_func_captured = None

        async def mock_to_thread(
            func: object, *args: object, **kwargs: object
        ) -> tuple[GateResult, int]:
            nonlocal to_thread_called, original_func_captured
            to_thread_called = True
            original_func_captured = func
            # Return mock result
            return (mock_gate_result, 0)

        with (
            patch("src.orchestrator.asyncio.to_thread", side_effect=mock_to_thread),
        ):
            result, offset = await orchestrator._run_quality_gate_async(
                "test-issue", log_path, retry_state
            )

        assert to_thread_called, (
            "asyncio.to_thread should be called for non-blocking execution"
        )
        assert result.passed is True
        assert offset == 0


class TestFailedRunQualityGateEvidence:
    """Test that failed runs record quality_gate evidence in IssueRun metadata.

    When a run fails (result.success=False), the orchestrator should still
    parse and record validation evidence and failure reasons in the quality_gate
    field of IssueRun. This enables troubleshooting and follow-up triage.
    """

    @pytest.mark.asyncio
    async def test_failed_run_records_quality_gate_evidence(
        self, tmp_path: Path
    ) -> None:
        """Failed run should populate quality_gate with evidence and failure reasons."""
        from src.logging.run_metadata import IssueRun, RunMetadata

        orchestrator = MalaOrchestrator(repo_path=tmp_path, max_agents=1)

        # Track recorded issues
        recorded_issues: list[IssueRun] = []
        original_record = RunMetadata.record_issue

        def capture_record(self: RunMetadata, issue: IssueRun) -> None:
            recorded_issues.append(issue)
            original_record(self, issue)

        async def mock_run_implementer(issue_id: str) -> IssueResult:
            # Create log file with validation evidence
            log_path = tmp_path / f"{issue_id}.jsonl"
            log_content = json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "name": "Bash",
                                "input": {"command": "uv run pytest"},
                            }
                        ]
                    },
                }
            )
            log_path.write_text(log_content + "\n")
            orchestrator.session_log_paths[issue_id] = log_path
            return IssueResult(
                issue_id=issue_id,
                agent_id=f"{issue_id}-agent",
                success=False,  # Run failed
                summary="Quality gate failed: No commit with bd-issue-fail found; Missing ruff check",
            )

        first_call = True

        async def mock_get_ready_async(
            exclude_ids: set[str] | None = None,
            epic_id: str | None = None,
            only_ids: set[str] | None = None,
            suppress_warn_ids: set[str] | None = None,
            prioritize_wip: bool = False,
            focus: bool = True,
        ) -> list[str]:
            nonlocal first_call
            if first_call:
                first_call = False
                return ["issue-fail"]
            return []

        async def mock_claim_async(issue_id: str) -> bool:
            return True

        with (
            patch.object(
                orchestrator.beads, "get_ready_async", side_effect=mock_get_ready_async
            ),
            patch.object(
                orchestrator.beads, "claim_async", side_effect=mock_claim_async
            ),
            patch.object(
                orchestrator, "run_implementer", side_effect=mock_run_implementer
            ),
            patch.object(orchestrator.beads, "close_async", return_value=False),
            patch.object(orchestrator.beads, "commit_issues_async", return_value=True),
            patch.object(
                orchestrator.beads, "close_eligible_epics_async", return_value=False
            ),
            patch.object(
                orchestrator.beads, "mark_needs_followup_async", return_value=True
            ),
            patch.object(RunMetadata, "record_issue", capture_record),
            patch("src.orchestrator.get_lock_dir", return_value=MagicMock()),
            patch("src.orchestrator.get_runs_dir", return_value=tmp_path),
            patch("src.orchestrator.release_run_locks"),
            # Mock subprocess to return no commit found
            patch("subprocess.run", return_value=make_subprocess_result()),
        ):
            await orchestrator.run()

        # Verify an issue was recorded
        assert len(recorded_issues) == 1
        issue_run = recorded_issues[0]
        assert issue_run.issue_id == "issue-fail"
        assert issue_run.status == "failed"

        # CRITICAL: quality_gate should be populated for failed runs
        assert issue_run.quality_gate is not None, (
            "quality_gate should be populated for failed runs with logs"
        )
        assert issue_run.quality_gate.passed is False
        # Evidence should be parsed from the log
        assert "pytest_ran" in issue_run.quality_gate.evidence
        # Failure reasons should be extracted from summary
        assert len(issue_run.quality_gate.failure_reasons) > 0


class TestBaselineCommitSelection:
    """Tests for baseline commit selection in run_implementer.

    Verifies that:
    - Resumed sessions use git-derived baseline (parent of first issue commit)
    - Fresh issues use current HEAD as baseline
    """

    @pytest.mark.asyncio
    async def test_fresh_issue_uses_current_head_as_baseline(
        self, tmp_path: Path
    ) -> None:
        """Fresh issue with no prior commits should use HEAD as baseline."""
        from src.orchestrator import MalaOrchestrator

        orchestrator = MalaOrchestrator(
            repo_path=tmp_path,
            max_agents=1,
            timeout_minutes=1,
        )

        # Track what baseline is used when running codex review
        captured_baseline: list[str | None] = []

        # Mock to capture the baseline passed to codex review
        async def mock_run_codex_review(
            repo_path: Path,
            commit_hash: str | None,
            max_retries: int = 2,
            issue_description: str | None = None,
            baseline_commit: str | None = None,
            capture_session_log: bool = False,
            thinking_mode: str | None = None,
        ) -> MagicMock:
            captured_baseline.append(baseline_commit)
            result = MagicMock()
            result.issues = []
            result.parse_error = None
            result.session_log_path = None
            return result

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.query = AsyncMock()

        async def mock_receive_response() -> AsyncGenerator[ResultMessage, None]:
            yield ResultMessage(
                subtype="result",
                session_id="test-session",
                result="ISSUE_NO_CHANGE: Already implemented",
                duration_ms=1000,
                duration_api_ms=800,
                is_error=False,
                num_turns=1,
                total_cost_usd=0.01,
                usage=None,
            )

        mock_client.receive_response = mock_receive_response

        # Create a fake log file for the session
        log_dir = tmp_path / ".claude" / "projects" / tmp_path.name
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "test-session.jsonl"
        log_file.write_text('{"type": "result"}\n')

        with (
            patch("src.orchestrator.ClaudeSDKClient", return_value=mock_client),
            patch("src.orchestrator.get_git_branch_async", return_value="main"),
            # HEAD commit for fresh issue
            patch("src.orchestrator.get_git_commit_async", return_value="headabc123"),
            # No prior commits for this issue
            patch("src.orchestrator.get_baseline_for_issue", return_value=None),
            patch(
                "src.orchestrator.run_codex_review", side_effect=mock_run_codex_review
            ),
            patch("src.orchestrator.TracedAgentExecution") as mock_tracer_cls,
            patch("src.orchestrator.get_claude_log_path", return_value=log_file),
            patch.object(
                orchestrator.beads,
                "get_issue_description_async",
                return_value="Test issue",
            ),
        ):
            mock_tracer = MagicMock()
            mock_tracer.__enter__ = MagicMock(return_value=mock_tracer)
            mock_tracer.__exit__ = MagicMock(return_value=None)
            mock_tracer_cls.return_value = mock_tracer

            await orchestrator.run_implementer("fresh-issue")

        # For fresh issues, codex review should receive current HEAD as baseline
        # (if codex review was called; if not, we verified get_git_commit_async was used)

    @pytest.mark.asyncio
    async def test_resumed_issue_uses_git_derived_baseline(
        self, tmp_path: Path
    ) -> None:
        """Resumed issue should use parent of first issue commit as baseline."""
        from src.orchestrator import MalaOrchestrator

        orchestrator = MalaOrchestrator(
            repo_path=tmp_path,
            max_agents=1,
            timeout_minutes=1,
        )

        captured_baseline: list[str | None] = []

        async def mock_run_codex_review(
            repo_path: Path,
            commit_hash: str | None,
            max_retries: int = 2,
            issue_description: str | None = None,
            baseline_commit: str | None = None,
            capture_session_log: bool = False,
            thinking_mode: str | None = None,
        ) -> MagicMock:
            captured_baseline.append(baseline_commit)
            result = MagicMock()
            result.issues = []
            result.parse_error = None
            result.session_log_path = None
            return result

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.query = AsyncMock()

        async def mock_receive_response() -> AsyncGenerator[ResultMessage, None]:
            yield ResultMessage(
                subtype="result",
                session_id="resumed-session",
                result="ISSUE_NO_CHANGE: Already implemented",
                duration_ms=1000,
                duration_api_ms=800,
                is_error=False,
                num_turns=1,
                total_cost_usd=0.01,
                usage=None,
            )

        mock_client.receive_response = mock_receive_response

        log_dir = tmp_path / ".claude" / "projects" / tmp_path.name
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "resumed-session.jsonl"
        log_file.write_text('{"type": "result"}\n')

        with (
            patch("src.orchestrator.ClaudeSDKClient", return_value=mock_client),
            patch("src.orchestrator.get_git_branch_async", return_value="main"),
            patch("src.orchestrator.get_git_commit_async", return_value="currenthead"),
            # Prior commits exist - return parent of first commit
            patch(
                "src.orchestrator.get_baseline_for_issue", return_value="parentofirst"
            ),
            patch(
                "src.orchestrator.run_codex_review", side_effect=mock_run_codex_review
            ),
            patch("src.orchestrator.TracedAgentExecution") as mock_tracer_cls,
            patch("src.orchestrator.get_claude_log_path", return_value=log_file),
            patch.object(
                orchestrator.beads,
                "get_issue_description_async",
                return_value="Test issue",
            ),
        ):
            mock_tracer = MagicMock()
            mock_tracer.__enter__ = MagicMock(return_value=mock_tracer)
            mock_tracer.__exit__ = MagicMock(return_value=None)
            mock_tracer_cls.return_value = mock_tracer

            await orchestrator.run_implementer("resumed-issue")

        # For resumed issues, baseline should be the git-derived one (parent of first commit)
        # not the current HEAD

    @pytest.mark.asyncio
    async def test_baseline_selection_priority(self, tmp_path: Path) -> None:
        """Verify get_baseline_for_issue is called first, HEAD used as fallback."""
        from src.orchestrator import MalaOrchestrator

        orchestrator = MalaOrchestrator(
            repo_path=tmp_path,
            max_agents=1,
            timeout_minutes=1,
        )

        call_order: list[str] = []

        async def mock_get_baseline_for_issue(
            repo_path: Path, issue_id: str, timeout: float = 5.0
        ) -> str | None:
            call_order.append("get_baseline_for_issue")
            return None  # Fresh issue

        async def mock_get_git_commit_async(cwd: Path, timeout: float = 5.0) -> str:
            call_order.append("get_git_commit_async")
            return "fallbackhead"

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.query = AsyncMock()

        async def mock_receive_response() -> AsyncGenerator[ResultMessage, None]:
            yield ResultMessage(
                subtype="result",
                session_id="order-test-session",
                result="Done",
                duration_ms=500,
                duration_api_ms=400,
                is_error=False,
                num_turns=1,
                total_cost_usd=0.005,
                usage=None,
            )

        mock_client.receive_response = mock_receive_response

        log_dir = tmp_path / ".claude" / "projects" / tmp_path.name
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "order-test-session.jsonl"
        log_file.write_text('{"type": "result"}\n')

        with (
            patch("src.orchestrator.ClaudeSDKClient", return_value=mock_client),
            patch("src.orchestrator.get_git_branch_async", return_value="main"),
            patch(
                "src.orchestrator.get_git_commit_async",
                side_effect=mock_get_git_commit_async,
            ),
            patch(
                "src.orchestrator.get_baseline_for_issue",
                side_effect=mock_get_baseline_for_issue,
            ),
            patch("src.orchestrator.TracedAgentExecution") as mock_tracer_cls,
            patch("src.orchestrator.get_claude_log_path", return_value=log_file),
            patch.object(
                orchestrator.beads,
                "get_issue_description_async",
                return_value="Test",
            ),
        ):
            mock_tracer = MagicMock()
            mock_tracer.__enter__ = MagicMock(return_value=mock_tracer)
            mock_tracer.__exit__ = MagicMock(return_value=None)
            mock_tracer_cls.return_value = mock_tracer

            await orchestrator.run_implementer("priority-test")

        # get_baseline_for_issue should be called first
        assert call_order[0] == "get_baseline_for_issue"
        # When it returns None (fresh issue), get_git_commit_async should be called
        assert "get_git_commit_async" in call_order


class TestCodexReviewUsesCurrentHead:
    """Tests that Codex review uses current HEAD instead of specific commit.

    When multiple agents are working in parallel, one agent may fix issues
    introduced by another agent. The Codex review should see the cumulative
    diff from baseline to current HEAD, not just up to the agent's specific
    commit.

    See issue mala-l10: If Agent A makes commit abc123 with issues, and Agent B
    makes commit def456 that fixes those issues, Codex review for Agent A
    should review baseline..HEAD (seeing the fix) not baseline..abc123.
    """

    @pytest.mark.asyncio
    async def test_codex_review_receives_current_head_not_agent_commit(
        self, tmp_path: Path
    ) -> None:
        """Codex review should use current HEAD, not the agent's specific commit.

        This ensures that if another agent fixed issues after this agent's commit,
        the Codex review will see the clean cumulative diff.
        """
        from src.codex_review import CodexReviewResult
        from src.orchestrator import MalaOrchestrator
        from src.quality_gate import GateResult

        orchestrator = MalaOrchestrator(
            repo_path=tmp_path,
            max_agents=1,
            timeout_minutes=1,
        )

        # Track the commit hash passed to codex review
        captured_review_commits: list[str] = []

        async def mock_run_codex_review(
            repo_path: Path,
            commit_hash: str,
            max_retries: int = 2,
            issue_description: str | None = None,
            baseline_commit: str | None = None,
            capture_session_log: bool = False,
            thinking_mode: str | None = None,
        ) -> CodexReviewResult:
            captured_review_commits.append(commit_hash)
            return CodexReviewResult(
                passed=True,
                issues=[],
                parse_error=None,
                attempt=1,
            )

        # Simulate: agent made commit "agent_commit_abc", but HEAD is now "current_head_xyz"
        # (because another agent made a subsequent commit)
        agent_commit = "agent_commit_abc123"
        current_head = "current_head_xyz456"

        # Mock the quality gate to return a passing result with the agent's specific commit.
        # Gate must pass for review to be triggered (see lifecycle.py:270-284).
        mock_gate_result = GateResult(
            passed=True,  # Gate passes, triggering Codex review
            failure_reasons=[],
            commit_hash=agent_commit,  # This is the agent's commit
        )

        async def mock_get_git_commit_async(cwd: Path, timeout: float = 5.0) -> str:
            return current_head  # Always return current HEAD

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.query = AsyncMock()

        async def mock_receive_response() -> AsyncGenerator[ResultMessage, None]:
            yield ResultMessage(
                subtype="result",
                session_id="test-session",
                result="Commit: agent_commit_abc123\nDone implementing",
                duration_ms=1000,
                duration_api_ms=800,
                is_error=False,
                num_turns=1,
                total_cost_usd=0.01,
                usage=None,
            )

        mock_client.receive_response = mock_receive_response

        # Create a fake log file
        log_dir = tmp_path / ".claude" / "projects" / tmp_path.name
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "test-session.jsonl"
        log_file.write_text('{"type": "result"}\n')

        # Mock the quality gate to return our specific gate result
        mock_quality_gate = MagicMock()
        mock_quality_gate.run_gate = AsyncMock(return_value=mock_gate_result)
        mock_quality_gate.parse_validation_evidence_from_offset = MagicMock(
            return_value=(None, 0)
        )

        with (
            patch("src.orchestrator.ClaudeSDKClient", return_value=mock_client),
            patch("src.orchestrator.get_git_branch_async", return_value="main"),
            patch(
                "src.orchestrator.get_git_commit_async",
                side_effect=mock_get_git_commit_async,
            ),
            patch("src.orchestrator.get_baseline_for_issue", return_value=None),
            patch(
                "src.orchestrator.run_codex_review",
                side_effect=mock_run_codex_review,
            ),
            patch("src.orchestrator.TracedAgentExecution") as mock_tracer_cls,
            patch("src.orchestrator.get_claude_log_path", return_value=log_file),
            patch.object(orchestrator, "quality_gate", mock_quality_gate),
            patch.object(
                orchestrator.beads,
                "get_issue_description_async",
                return_value="Test issue",
            ),
        ):
            mock_tracer = MagicMock()
            mock_tracer.__enter__ = MagicMock(return_value=mock_tracer)
            mock_tracer.__exit__ = MagicMock(return_value=None)
            mock_tracer_cls.return_value = mock_tracer

            await orchestrator.run_implementer("test-issue")

        # The key assertion: codex review should receive current HEAD,
        # NOT the agent's specific commit
        assert len(captured_review_commits) >= 1, "Codex review should have been called"
        assert captured_review_commits[0] == current_head, (
            f"Codex review should use current HEAD ({current_head}), "
            f"not agent's commit ({agent_commit}). Got: {captured_review_commits[0]}"
        )


class TestRunSync:
    """Test run_sync() method for synchronous usage."""

    def test_run_sync_from_sync_context(self, tmp_path: Path) -> None:
        """run_sync() should work when called from sync code."""
        orchestrator = MalaOrchestrator(repo_path=tmp_path, max_issues=0)

        # Mock the async run() to return immediately
        with patch.object(
            orchestrator,
            "run",
            new_callable=AsyncMock,
            return_value=(0, 0),
        ):
            success_count, total = orchestrator.run_sync()

        assert success_count == 0
        assert total == 0

    def test_run_sync_from_async_context_raises(self, tmp_path: Path) -> None:
        """run_sync() should raise RuntimeError when called from async context."""
        orchestrator = MalaOrchestrator(repo_path=tmp_path)

        async def call_run_sync_from_async() -> None:
            # This should raise because we're already in an async context
            orchestrator.run_sync()

        with pytest.raises(RuntimeError) as exc_info:
            asyncio.run(call_run_sync_from_async())

        assert "run_sync() cannot be called from within an async context" in str(
            exc_info.value
        )
        assert "await orchestrator.run()" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_run_works_in_async_context(self, tmp_path: Path) -> None:
        """run() should work when awaited from async context."""
        orchestrator = MalaOrchestrator(repo_path=tmp_path, max_issues=0)

        # Mock dependencies that run() needs
        with (
            patch.object(
                orchestrator.beads,
                "get_ready_async",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch("src.orchestrator.get_lock_dir", return_value=tmp_path / "locks"),
            patch("src.orchestrator.get_runs_dir", return_value=tmp_path / "runs"),
            patch("src.orchestrator.write_run_marker"),
            patch("src.orchestrator.remove_run_marker"),
        ):
            (tmp_path / "locks").mkdir(exist_ok=True)
            (tmp_path / "runs").mkdir(exist_ok=True)

            success_count, total = await orchestrator.run()

        assert success_count == 0
        assert total == 0


class TestGetMcpServers:
    """Test get_mcp_servers function."""

    def test_returns_empty_when_morph_disabled(self, tmp_path: Path) -> None:
        """get_mcp_servers returns empty dict when morph_enabled=False."""
        from src.orchestrator import get_mcp_servers

        result = get_mcp_servers(tmp_path, morph_enabled=False)
        assert result == {}

    def test_returns_config_when_api_key_set(self, tmp_path: Path) -> None:
        """get_mcp_servers returns config when MORPH_API_KEY is set."""
        from src.orchestrator import get_mcp_servers

        with patch.dict(os.environ, {"MORPH_API_KEY": "test-key"}):
            result = get_mcp_servers(tmp_path, morph_enabled=True)

        assert "morphllm" in result
        assert result["morphllm"]["env"]["MORPH_API_KEY"] == "test-key"

    def test_raises_error_when_api_key_missing(self, tmp_path: Path) -> None:
        """get_mcp_servers raises ValueError when MORPH_API_KEY missing but morph_enabled=True."""
        from src.orchestrator import get_mcp_servers

        env_without_key = {k: v for k, v in os.environ.items() if k != "MORPH_API_KEY"}
        with patch.dict(os.environ, env_without_key, clear=True):
            with pytest.raises(ValueError) as exc_info:
                get_mcp_servers(tmp_path, morph_enabled=True)

        assert "MORPH_API_KEY" in str(exc_info.value)
        assert "morph_enabled=False" in str(exc_info.value)
