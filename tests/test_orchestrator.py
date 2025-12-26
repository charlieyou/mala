"""Unit tests for MalaOrchestrator control flow.

These tests mock subprocess.run and ClaudeSDKClient to test orchestrator
state transitions without network or actual bd CLI.
"""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.beads_client import BeadsClient
from src.orchestrator import IssueResult, MalaOrchestrator


@pytest.fixture
def orchestrator(tmp_path: Path) -> MalaOrchestrator:
    """Create an orchestrator with a temporary repo path."""
    return MalaOrchestrator(
        repo_path=tmp_path,
        max_agents=2,
        timeout_minutes=1,
        max_issues=5,
    )


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


class TestGetReadyIssues:
    """Test beads.get_ready handles bd CLI JSON and errors."""

    def test_returns_issue_ids_sorted_by_priority(self, orchestrator: MalaOrchestrator):
        """Issues should be returned sorted by priority (lower = higher)."""
        issues_json = json.dumps(
            [
                {"id": "issue-3", "priority": 3, "issue_type": "task"},
                {"id": "issue-1", "priority": 1, "issue_type": "task"},
                {"id": "issue-2", "priority": 2, "issue_type": "task"},
            ]
        )
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = make_subprocess_result(stdout=issues_json)
            result = orchestrator.beads.get_ready()

        assert result == ["issue-1", "issue-2", "issue-3"]

    def test_filters_out_epics(self, orchestrator: MalaOrchestrator):
        """Epics should be excluded from ready issues."""
        issues_json = json.dumps(
            [
                {"id": "task-1", "priority": 1, "issue_type": "task"},
                {"id": "epic-1", "priority": 1, "issue_type": "epic"},
                {"id": "bug-1", "priority": 2, "issue_type": "bug"},
            ]
        )
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = make_subprocess_result(stdout=issues_json)
            result = orchestrator.beads.get_ready()

        assert result == ["task-1", "bug-1"]
        assert "epic-1" not in result

    def test_filters_out_failed_issues(self, orchestrator: MalaOrchestrator):
        """Previously failed issues should be excluded."""
        failed_set = {"failed-1"}
        issues_json = json.dumps(
            [
                {"id": "ok-1", "priority": 1, "issue_type": "task"},
                {"id": "failed-1", "priority": 1, "issue_type": "task"},
            ]
        )
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = make_subprocess_result(stdout=issues_json)
            result = orchestrator.beads.get_ready(failed_set)

        assert result == ["ok-1"]
        assert "failed-1" not in result

    def test_returns_empty_list_on_bd_failure(self, orchestrator: MalaOrchestrator):
        """When bd ready fails, return empty list."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = make_subprocess_result(
                returncode=1, stderr="bd: command not found"
            )
            result = orchestrator.beads.get_ready()

        assert result == []

    def test_returns_empty_list_on_invalid_json(self, orchestrator: MalaOrchestrator):
        """When bd returns invalid JSON, return empty list."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = make_subprocess_result(stdout="not valid json")
            result = orchestrator.beads.get_ready()

        assert result == []

    def test_handles_missing_priority(self, orchestrator: MalaOrchestrator):
        """Issues without priority should be sorted last (priority 999)."""
        issues_json = json.dumps(
            [
                {"id": "no-prio", "issue_type": "task"},
                {"id": "prio-1", "priority": 1, "issue_type": "task"},
            ]
        )
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = make_subprocess_result(stdout=issues_json)
            result = orchestrator.beads.get_ready()

        assert result == ["prio-1", "no-prio"]

    def test_suppresses_warning_for_only_ids_already_processed(
        self, tmp_path: Path
    ):
        """Only-id warnings should be suppressed for already processed IDs."""
        warnings: list[str] = []
        beads = BeadsClient(tmp_path, log_warning=warnings.append)
        issues_json = json.dumps([])
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = make_subprocess_result(stdout=issues_json)
            result = beads.get_ready(
                only_ids={"issue-1"},
                suppress_warn_ids={"issue-1"},
            )

        assert result == []
        assert warnings == []


class TestClaimIssue:
    """Test beads.claim invokes bd update correctly."""

    def test_returns_true_on_success(self, orchestrator: MalaOrchestrator):
        """Successful claim returns True."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = make_subprocess_result(returncode=0)
            result = orchestrator.beads.claim("issue-1")

        assert result is True

    def test_returns_false_on_failure(self, orchestrator: MalaOrchestrator):
        """Failed claim returns False."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = make_subprocess_result(
                returncode=1, stderr="Issue already claimed"
            )
            result = orchestrator.beads.claim("issue-1")

        assert result is False

    def test_calls_bd_update_with_correct_args(self, orchestrator: MalaOrchestrator):
        """Verify bd update is called with correct arguments."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = make_subprocess_result()
            orchestrator.beads.claim("issue-abc")

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[0][0] == [
            "bd",
            "update",
            "issue-abc",
            "--status",
            "in_progress",
        ]
        assert call_args[1]["cwd"] == orchestrator.repo_path


class TestResetIssue:
    """Test beads.reset invokes bd update correctly."""

    def test_calls_bd_update_with_ready_status(self, orchestrator: MalaOrchestrator):
        """Verify reset calls bd update with ready status."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = make_subprocess_result()
            orchestrator.beads.reset("issue-failed")

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[0][0] == ["bd", "update", "issue-failed", "--status", "ready"]

    def test_does_not_raise_on_failure(self, orchestrator: MalaOrchestrator):
        """Reset should not raise even if bd fails."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = make_subprocess_result(returncode=1)
            # Should not raise
            orchestrator.beads.reset("issue-failed")


class TestSpawnAgent:
    """Test spawn_agent behavior."""

    @pytest.mark.asyncio
    async def test_adds_issue_to_failed_when_claim_fails(
        self, orchestrator: MalaOrchestrator
    ):
        """When claim fails, issue should be added to failed_issues."""
        with patch.object(orchestrator.beads, "claim", return_value=False):
            result = await orchestrator.spawn_agent("unclaimed-issue")

        assert result is False
        assert "unclaimed-issue" in orchestrator.failed_issues

    @pytest.mark.asyncio
    async def test_creates_task_when_claim_succeeds(
        self, orchestrator: MalaOrchestrator
    ):
        """When claim succeeds, a task should be created."""
        with (
            patch.object(orchestrator.beads, "claim", return_value=True),
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
    ):
        """When no issues are ready, run() should return 0 and exit cleanly."""
        with (
            patch.object(orchestrator.beads, "get_ready", return_value=[]),
            patch("src.orchestrator.LOCK_DIR", MagicMock()),
            patch("src.orchestrator.JSONL_LOG_DIR", MagicMock()),
            patch("src.orchestrator.release_all_locks"),
        ):
            result = await orchestrator.run()

        assert result == (0, 0)
        assert len(orchestrator.completed) == 0

    @pytest.mark.asyncio
    async def test_respects_max_issues_limit(self, orchestrator: MalaOrchestrator):
        """Should stop after processing max_issues."""
        orchestrator.max_issues = 2
        call_count = 0
        spawned = []

        def mock_get_ready(
            failed=None, epic_id=None, only_ids=None, suppress_warn_ids=None
        ):
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
            patch.object(orchestrator.beads, "get_ready", side_effect=mock_get_ready),
            patch.object(orchestrator, "spawn_agent", side_effect=mock_spawn),
            patch("src.orchestrator.LOCK_DIR", MagicMock()),
            patch("src.orchestrator.JSONL_LOG_DIR", MagicMock()),
            patch("src.orchestrator.release_all_locks"),
            patch("subprocess.run", return_value=make_subprocess_result()),
        ):
            # Orchestrator needs active_tasks to be empty to exit
            # but we don't actually spawn tasks in this mock
            await orchestrator.run()

        # Should have only spawned 2 issues (max_issues limit)
        assert len(spawned) == 2


class TestFailedTaskResetsIssue:
    """Test that failed tasks correctly reset issue status."""

    @pytest.mark.asyncio
    async def test_resets_issue_on_task_failure(self, orchestrator: MalaOrchestrator):
        """When a task fails, the issue should be reset to ready status."""
        reset_calls = []

        def mock_reset(issue_id: str):
            reset_calls.append(issue_id)

        async def mock_run_implementer(issue_id: str) -> IssueResult:
            return IssueResult(
                issue_id=issue_id,
                agent_id=f"{issue_id}-agent",
                success=False,
                summary="Implementation failed",
            )

        first_call = True

        def mock_get_ready(
            failed=None, epic_id=None, only_ids=None, suppress_warn_ids=None
        ):
            nonlocal first_call
            if first_call:
                first_call = False
                return ["fail-issue"]
            return []

        with (
            patch.object(orchestrator.beads, "get_ready", side_effect=mock_get_ready),
            patch.object(orchestrator.beads, "claim", return_value=True),
            patch.object(
                orchestrator, "run_implementer", side_effect=mock_run_implementer
            ),
            patch.object(orchestrator.beads, "reset", side_effect=mock_reset),
            patch("src.orchestrator.LOCK_DIR", MagicMock()),
            patch("src.orchestrator.JSONL_LOG_DIR", MagicMock()),
            patch("src.orchestrator.release_all_locks"),
            patch("subprocess.run", return_value=make_subprocess_result()),
        ):
            await orchestrator.run()

        # The failed issue should have been reset
        assert "fail-issue" in reset_calls
        # And added to failed_issues set
        assert "fail-issue" in orchestrator.failed_issues

    @pytest.mark.asyncio
    async def test_does_not_reset_successful_issue(
        self, orchestrator: MalaOrchestrator
    ):
        """Successful issues should not be reset."""
        reset_calls = []

        def mock_reset(issue_id: str):
            reset_calls.append(issue_id)

        async def mock_run_implementer(issue_id: str) -> IssueResult:
            return IssueResult(
                issue_id=issue_id,
                agent_id=f"{issue_id}-agent",
                success=True,
                summary="Completed successfully",
            )

        first_call = True

        def mock_get_ready(
            failed=None, epic_id=None, only_ids=None, suppress_warn_ids=None
        ):
            nonlocal first_call
            if first_call:
                first_call = False
                return ["success-issue"]
            return []

        with (
            patch.object(orchestrator.beads, "get_ready", side_effect=mock_get_ready),
            patch.object(orchestrator.beads, "claim", return_value=True),
            patch.object(
                orchestrator, "run_implementer", side_effect=mock_run_implementer
            ),
            patch.object(orchestrator.beads, "reset", side_effect=mock_reset),
            patch("src.orchestrator.LOCK_DIR", MagicMock()),
            patch("src.orchestrator.JSONL_LOG_DIR", MagicMock()),
            patch("src.orchestrator.release_all_locks"),
            patch("subprocess.run", return_value=make_subprocess_result()),
        ):
            await orchestrator.run()

        # No reset should have been called
        assert "success-issue" not in reset_calls
        # And not in failed_issues
        assert "success-issue" not in orchestrator.failed_issues


class TestOrchestratorInitialization:
    """Test orchestrator initialization."""

    def test_timeout_conversion(self, tmp_path: Path):
        """Timeout minutes should be converted to seconds."""
        orch = MalaOrchestrator(repo_path=tmp_path, timeout_minutes=15)
        assert orch.timeout_seconds == 15 * 60

    def test_default_values(self, tmp_path: Path):
        """Default values should be set correctly."""
        orch = MalaOrchestrator(repo_path=tmp_path)
        assert orch.max_agents is None  # Unlimited by default
        assert orch.timeout_seconds == 30 * 60
        assert orch.max_issues is None

    def test_repo_path_resolved(self, tmp_path: Path):
        """Repo path should be resolved to absolute."""
        relative = Path(".")
        orch = MalaOrchestrator(repo_path=relative)
        assert orch.repo_path.is_absolute()


class TestEpicFilter:
    """Test epic filter functionality in BeadsClient."""

    def test_get_epic_children_returns_child_ids(self, orchestrator: MalaOrchestrator):
        """get_epic_children should return IDs of children (depth > 0)."""
        tree_json = json.dumps(
            [
                {"id": "epic-1", "depth": 0, "issue_type": "epic"},  # Epic itself
                {"id": "task-1", "depth": 1, "issue_type": "task"},  # Child
                {"id": "task-2", "depth": 1, "issue_type": "task"},  # Child
                {"id": "task-3", "depth": 2, "issue_type": "task"},  # Grandchild
            ]
        )
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = make_subprocess_result(stdout=tree_json)
            result = orchestrator.beads.get_epic_children("epic-1")

        assert result == {"task-1", "task-2", "task-3"}
        assert "epic-1" not in result

    def test_get_epic_children_returns_empty_on_failure(
        self, orchestrator: MalaOrchestrator
    ):
        """get_epic_children should return empty set on bd failure."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = make_subprocess_result(
                returncode=1, stderr="epic not found"
            )
            result = orchestrator.beads.get_epic_children("nonexistent-epic")

        assert result == set()

    def test_get_epic_children_returns_empty_on_invalid_json(
        self, orchestrator: MalaOrchestrator
    ):
        """get_epic_children should return empty set on invalid JSON."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = make_subprocess_result(stdout="not valid json")
            result = orchestrator.beads.get_epic_children("epic-1")

        assert result == set()

    def test_get_ready_with_epic_filter(self, orchestrator: MalaOrchestrator):
        """get_ready with epic_id should only return children of that epic."""
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

        def mock_run(cmd, **kwargs):
            if "dep" in cmd and "tree" in cmd:
                return make_subprocess_result(stdout=tree_json)
            elif "ready" in cmd:
                return make_subprocess_result(stdout=ready_json)
            return make_subprocess_result()

        with patch("subprocess.run", side_effect=mock_run):
            result = orchestrator.beads.get_ready(epic_id="epic-1")

        assert result == ["child-1", "child-2"]
        assert "other-task" not in result

    def test_get_ready_without_epic_filter_returns_all(
        self, orchestrator: MalaOrchestrator
    ):
        """get_ready without epic_id should return all ready tasks."""
        ready_json = json.dumps(
            [
                {"id": "task-1", "priority": 1, "issue_type": "task"},
                {"id": "task-2", "priority": 2, "issue_type": "task"},
            ]
        )
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = make_subprocess_result(stdout=ready_json)
            result = orchestrator.beads.get_ready()

        assert result == ["task-1", "task-2"]

    def test_get_ready_with_epic_filter_returns_empty_if_no_children(
        self, orchestrator: MalaOrchestrator
    ):
        """get_ready with epic_id should return empty if epic has no children."""
        with patch("subprocess.run") as mock_run:
            # Epic tree returns empty (epic not found or no children)
            mock_run.return_value = make_subprocess_result(returncode=1)
            result = orchestrator.beads.get_ready(epic_id="empty-epic")

        assert result == []


class TestOrchestratorWithEpicId:
    """Test orchestrator with epic_id parameter."""

    def test_epic_id_stored(self, tmp_path: Path):
        """epic_id parameter should be stored on orchestrator."""
        orch = MalaOrchestrator(repo_path=tmp_path, epic_id="test-epic")
        assert orch.epic_id == "test-epic"

    def test_epic_id_defaults_to_none(self, tmp_path: Path):
        """epic_id should default to None."""
        orch = MalaOrchestrator(repo_path=tmp_path)
        assert orch.epic_id is None
