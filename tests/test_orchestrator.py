"""Unit tests for MalaOrchestrator control flow.

These tests mock subprocess.run and ClaudeSDKClient to test orchestrator
state transitions without network or actual bd CLI.
"""

import asyncio
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.beads_client import BeadsClient
from src.orchestrator import IMPLEMENTER_PROMPT_TEMPLATE, IssueResult, MalaOrchestrator


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


class TestPromptTemplate:
    """Test implementer prompt template validity."""

    def test_prompt_template_placeholders_match_format_call(self):
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
    ):
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
            new_callable=AsyncMock,
            return_value=make_subprocess_result(stdout=issues_json),
        ):
            result = await orchestrator.beads.get_ready_async()

        assert result == ["issue-1", "issue-2", "issue-3"]

    @pytest.mark.asyncio
    async def test_filters_out_epics(self, orchestrator: MalaOrchestrator):
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
            new_callable=AsyncMock,
            return_value=make_subprocess_result(stdout=issues_json),
        ):
            result = await orchestrator.beads.get_ready_async()

        assert result == ["task-1", "bug-1"]
        assert "epic-1" not in result

    @pytest.mark.asyncio
    async def test_filters_out_failed_issues(self, orchestrator: MalaOrchestrator):
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
            new_callable=AsyncMock,
            return_value=make_subprocess_result(stdout=issues_json),
        ):
            result = await orchestrator.beads.get_ready_async(failed_set)

        assert result == ["ok-1"]
        assert "failed-1" not in result

    @pytest.mark.asyncio
    async def test_returns_empty_list_on_bd_failure(
        self, orchestrator: MalaOrchestrator
    ):
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
    ):
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
    async def test_handles_missing_priority(self, orchestrator: MalaOrchestrator):
        """Issues without priority should be sorted last (priority 999)."""
        issues_json = json.dumps(
            [
                {"id": "no-prio", "issue_type": "task"},
                {"id": "prio-1", "priority": 1, "issue_type": "task"},
            ]
        )
        with patch.object(
            orchestrator.beads,
            "_run_subprocess_async",
            new_callable=AsyncMock,
            return_value=make_subprocess_result(stdout=issues_json),
        ):
            result = await orchestrator.beads.get_ready_async()

        assert result == ["prio-1", "no-prio"]

    @pytest.mark.asyncio
    async def test_suppresses_warning_for_only_ids_already_processed(
        self, tmp_path: Path
    ):
        """Only-id warnings should be suppressed for already processed IDs."""
        warnings: list[str] = []
        beads = BeadsClient(tmp_path, log_warning=warnings.append)
        issues_json = json.dumps([])
        with patch.object(
            beads,
            "_run_subprocess_async",
            new_callable=AsyncMock,
            return_value=make_subprocess_result(stdout=issues_json),
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
    async def test_returns_true_on_success(self, orchestrator: MalaOrchestrator):
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
    async def test_returns_false_on_failure(self, orchestrator: MalaOrchestrator):
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
    ):
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
    ):
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
    ):
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
    async def test_does_not_raise_on_failure(self, orchestrator: MalaOrchestrator):
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
    ):
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
    ):
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
    ):
        """When no issues are ready, run() should return 0 and exit cleanly."""

        async def mock_get_ready_async(
            exclude_ids=None, epic_id=None, only_ids=None, suppress_warn_ids=None
        ):
            return []

        with (
            patch.object(
                orchestrator.beads, "get_ready_async", side_effect=mock_get_ready_async
            ),
            patch("src.orchestrator.LOCK_DIR", MagicMock()),
            patch("src.orchestrator.RUNS_DIR", MagicMock()),
            patch("src.orchestrator.release_run_locks"),
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

        async def mock_get_ready_async(
            exclude_ids=None, epic_id=None, only_ids=None, suppress_warn_ids=None
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
            patch.object(
                orchestrator.beads, "get_ready_async", side_effect=mock_get_ready_async
            ),
            patch.object(orchestrator, "spawn_agent", side_effect=mock_spawn),
            patch("src.orchestrator.LOCK_DIR", MagicMock()),
            patch("src.orchestrator.RUNS_DIR", MagicMock()),
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
    async def test_resets_issue_on_task_failure(self, orchestrator: MalaOrchestrator):
        """When a task fails, the issue should be marked needs-followup."""
        followup_calls = []

        async def mock_mark_followup_async(issue_id: str, reason: str, log_path=None):
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
            exclude_ids=None, epic_id=None, only_ids=None, suppress_warn_ids=None
        ):
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
            patch("src.orchestrator.LOCK_DIR", MagicMock()),
            patch("src.orchestrator.RUNS_DIR", MagicMock()),
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
    ):
        """Successful issues should not be reset."""
        reset_calls = []

        async def mock_reset_async(issue_id: str, log_path=None, error=""):
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
            exclude_ids=None, epic_id=None, only_ids=None, suppress_warn_ids=None
        ):
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
            patch("src.orchestrator.LOCK_DIR", MagicMock()),
            patch("src.orchestrator.RUNS_DIR", MagicMock()),
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

    def test_timeout_conversion(self, tmp_path: Path):
        """Timeout minutes should be converted to seconds."""
        orch = MalaOrchestrator(repo_path=tmp_path, timeout_minutes=15)
        assert orch.timeout_seconds == 15 * 60

    def test_default_values(self, tmp_path: Path):
        """Default values should be set correctly."""
        orch = MalaOrchestrator(repo_path=tmp_path)
        assert orch.max_agents is None  # Unlimited by default
        assert orch.timeout_seconds is None  # No timeout by default
        assert orch.max_issues is None

    def test_repo_path_resolved(self, tmp_path: Path):
        """Repo path should be resolved to absolute."""
        relative = Path(".")
        orch = MalaOrchestrator(repo_path=relative)
        assert orch.repo_path.is_absolute()


class TestEpicFilterAsync:
    """Test epic filter functionality in BeadsClient (async)."""

    @pytest.mark.asyncio
    async def test_get_epic_children_returns_child_ids(
        self, orchestrator: MalaOrchestrator
    ):
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
    ):
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
    ):
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
    async def test_get_ready_with_epic_filter(self, orchestrator: MalaOrchestrator):
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

        async def mock_run(cmd, **kwargs):
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
    ):
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
            new_callable=AsyncMock,
            return_value=make_subprocess_result(stdout=ready_json),
        ):
            result = await orchestrator.beads.get_ready_async()

        assert result == ["task-1", "task-2"]

    @pytest.mark.asyncio
    async def test_get_ready_with_epic_filter_returns_empty_if_no_children(
        self, orchestrator: MalaOrchestrator
    ):
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

    def test_epic_id_stored(self, tmp_path: Path):
        """epic_id parameter should be stored on orchestrator."""
        orch = MalaOrchestrator(repo_path=tmp_path, epic_id="test-epic")
        assert orch.epic_id == "test-epic"

    def test_epic_id_defaults_to_none(self, tmp_path: Path):
        """epic_id should default to None."""
        orch = MalaOrchestrator(repo_path=tmp_path)
        assert orch.epic_id is None


class TestQualityGateValidationEvidence:
    """Test JSONL log parsing for validation command evidence."""

    def test_detects_pytest_command(self, tmp_path: Path):
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
        evidence = gate.parse_validation_evidence(log_path)

        assert evidence.pytest_ran is True

    def test_detects_ruff_check_command(self, tmp_path: Path):
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
        evidence = gate.parse_validation_evidence(log_path)

        assert evidence.ruff_check_ran is True

    def test_detects_ruff_format_command(self, tmp_path: Path):
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
        evidence = gate.parse_validation_evidence(log_path)

        assert evidence.ruff_format_ran is True

    def test_detects_ty_check_command(self, tmp_path: Path):
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
        evidence = gate.parse_validation_evidence(log_path)

        assert evidence.ty_check_ran is True

    def test_detects_uv_sync_command(self, tmp_path: Path):
        """Quality gate should detect uv sync execution."""
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
                            "input": {"command": "uv sync"},
                        }
                    ]
                },
            }
        )
        log_path.write_text(log_content + "\n")

        gate = QualityGate(tmp_path)
        evidence = gate.parse_validation_evidence(log_path)

        assert evidence.uv_sync_ran is True

    def test_returns_empty_evidence_for_missing_log(self, tmp_path: Path):
        """Quality gate should return empty evidence for missing log file."""
        from src.quality_gate import QualityGate

        gate = QualityGate(tmp_path)
        nonexistent = tmp_path / "nonexistent.jsonl"
        evidence = gate.parse_validation_evidence(nonexistent)

        assert evidence.pytest_ran is False
        assert evidence.ruff_check_ran is False
        assert evidence.ruff_format_ran is False
        assert evidence.ty_check_ran is False
        assert evidence.uv_sync_ran is False


class TestQualityGateCommitCheck:
    """Test git commit message verification."""

    def test_detects_matching_commit(self, tmp_path: Path):
        """Quality gate should detect commit with correct issue ID."""
        from src.quality_gate import QualityGate

        gate = QualityGate(tmp_path)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = make_subprocess_result(
                stdout="abc1234 bd-issue-123: Fix the bug\n"
            )
            result = gate.check_commit_exists("issue-123")

        assert result.exists is True
        assert result.commit_hash == "abc1234"

    def test_rejects_missing_commit(self, tmp_path: Path):
        """Quality gate should reject when no matching commit found."""
        from src.quality_gate import QualityGate

        gate = QualityGate(tmp_path)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = make_subprocess_result(stdout="")
            result = gate.check_commit_exists("issue-123")

        assert result.exists is False
        assert result.commit_hash is None

    def test_handles_git_failure(self, tmp_path: Path):
        """Quality gate should handle git command failures gracefully."""
        from src.quality_gate import QualityGate

        gate = QualityGate(tmp_path)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = make_subprocess_result(
                returncode=1, stderr="fatal: not a git repository"
            )
            result = gate.check_commit_exists("issue-123")

        assert result.exists is False

    def test_searches_30_day_window(self, tmp_path: Path):
        """Quality gate should search commits from the last 30 days."""
        from src.quality_gate import QualityGate

        gate = QualityGate(tmp_path)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = make_subprocess_result(
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

    def test_passes_when_all_criteria_met(self, tmp_path: Path):
        """Quality gate passes when closed, commit exists, validation ran."""
        from src.quality_gate import QualityGate

        gate = QualityGate(tmp_path)

        # Create log with all validation commands
        log_path = tmp_path / "session.jsonl"
        commands = ["uv sync", "uv run pytest", "uvx ruff check .", "uvx ruff format ."]
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

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = make_subprocess_result(
                stdout="abc1234 bd-issue-123: Implement feature\n"
            )
            result = gate.check("issue-123", log_path)

        assert result.passed is True
        assert result.failure_reasons == []

    def test_fails_when_commit_missing(self, tmp_path: Path):
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

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = make_subprocess_result(stdout="")
            result = gate.check("issue-123", log_path)

        assert result.passed is False
        assert "commit" in result.failure_reasons[0].lower()

    def test_failure_message_reflects_30_day_window(self, tmp_path: Path):
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

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = make_subprocess_result(stdout="")
            result = gate.check("issue-123", log_path)

        assert result.passed is False
        assert "30 days" in result.failure_reasons[0]

    def test_fails_when_validation_missing(self, tmp_path: Path):
        """Quality gate fails when validation commands didn't run."""
        from src.quality_gate import QualityGate

        gate = QualityGate(tmp_path)

        # Create empty log (no validation commands)
        log_path = tmp_path / "session.jsonl"
        log_path.write_text("")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = make_subprocess_result(
                stdout="abc1234 bd-issue-123: Implement feature\n"
            )
            result = gate.check("issue-123", log_path)

        assert result.passed is False
        assert any("validation" in r.lower() for r in result.failure_reasons)


class TestBeadsClientNeedsFollowupAsync:
    """Test BeadsClient.mark_needs_followup_async method."""

    @pytest.mark.asyncio
    async def test_marks_issue_with_needs_followup_label(
        self, orchestrator: MalaOrchestrator
    ):
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
    ):
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
    ):
        """When quality gate fails (inside run_implementer), issue should be marked needs-followup."""
        mark_followup_calls = []

        async def mock_mark_followup_async(issue_id: str, reason: str, log_path=None):
            mark_followup_calls.append((issue_id, reason))
            return True

        async def mock_run_implementer(issue_id: str):
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
            exclude_ids=None, epic_id=None, only_ids=None, suppress_warn_ids=None
        ):
            nonlocal first_call
            if first_call:
                first_call = False
                return ["issue-123"]
            return []

        async def mock_claim_async(issue_id: str) -> bool:
            return True

        async def mock_commit_issues_async() -> bool:
            return True

        async def mock_close_eligible_epics_async() -> bool:
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
            patch("src.orchestrator.LOCK_DIR", MagicMock()),
            patch("src.orchestrator.RUNS_DIR", tmp_path),
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
    ):
        """Issue should only count as success when quality gate passes."""
        from src.quality_gate import GateResult

        # Gate passes
        mock_gate_result = GateResult(passed=True, failure_reasons=[])

        async def mock_run_implementer(issue_id: str):
            # Populate log path so quality gate runs
            log_path = tmp_path / f"{issue_id}.jsonl"
            log_path.touch()
            orchestrator.session_log_paths[issue_id] = log_path
            return IssueResult(
                issue_id=issue_id,
                agent_id=f"{issue_id}-agent",
                success=True,
                summary="done",
            )

        first_call = True

        async def mock_get_ready_async(
            exclude_ids=None, epic_id=None, only_ids=None, suppress_warn_ids=None
        ):
            nonlocal first_call
            if first_call:
                first_call = False
                return ["issue-ok"]
            return []

        async def mock_claim_async(issue_id: str) -> bool:
            return True

        async def mock_commit_issues_async() -> bool:
            return True

        async def mock_close_eligible_epics_async() -> bool:
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
                orchestrator.quality_gate, "check", return_value=mock_gate_result
            ),
            patch.object(
                orchestrator.beads,
                "commit_issues_async",
                side_effect=mock_commit_issues_async,
            ),
            patch.object(
                orchestrator.beads,
                "close_eligible_epics_async",
                side_effect=mock_close_eligible_epics_async,
            ),
            patch("src.orchestrator.LOCK_DIR", MagicMock()),
            patch("src.orchestrator.RUNS_DIR", tmp_path),
            patch("src.orchestrator.release_run_locks"),
            patch("subprocess.run", return_value=make_subprocess_result()),
        ):
            success_count, total = await orchestrator.run()

        assert success_count == 1


class TestAsyncBeadsClientWithTimeout:
    """Test that BeadsClient methods handle slow/hanging commands gracefully."""

    @pytest.mark.asyncio
    async def test_slow_bd_ready_does_not_block_other_tasks(
        self, orchestrator: MalaOrchestrator
    ):
        """Slow bd ready should not block other concurrent async tasks."""
        # Track when tasks complete
        task_completions: list[tuple[str, float]] = []
        start = time.monotonic()

        async def fast_task():
            await asyncio.sleep(0.05)
            task_completions.append(("fast", time.monotonic() - start))

        # Use a real slow command instead of mocking subprocess.run
        # (since we now use asyncio.create_subprocess_exec)
        orchestrator.beads.timeout_seconds = 0.5

        async def slow_beads_call():
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
    ):
        """When bd command times out, should return safe fallback, not hang."""
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
    async def test_claim_timeout_returns_false(self, orchestrator: MalaOrchestrator):
        """When claim times out, should return False (safe fallback)."""
        from src.beads_client import SubprocessResult

        original_timeout = orchestrator.beads.timeout_seconds
        orchestrator.beads.timeout_seconds = 0.1

        # Mock _run_subprocess_async to simulate timeout
        async def mock_timeout(*args, **kwargs):
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
    ):
        """Orchestrator run loop should use async beads methods."""
        # This test verifies the integration - that orchestrator calls
        # the async versions of beads methods

        async def mock_get_ready_async(
            exclude_ids=None, epic_id=None, only_ids=None, suppress_warn_ids=None
        ):
            return []

        with (
            patch.object(
                orchestrator.beads, "get_ready_async", side_effect=mock_get_ready_async
            ) as mock_ready,
            patch("src.orchestrator.LOCK_DIR", MagicMock()),
            patch("src.orchestrator.RUNS_DIR", MagicMock()),
            patch("src.orchestrator.release_run_locks"),
        ):
            await orchestrator.run()

        # Verify async method was called
        mock_ready.assert_called()


class TestMissingLogFile:
    """Test run_implementer behavior when log file never appears."""

    @pytest.mark.asyncio
    async def test_exits_quickly_when_log_file_missing(self, tmp_path: Path):
        """run_implementer should exit within bounded wait when log file missing."""
        from unittest.mock import AsyncMock, MagicMock
        from claude_agent_sdk.types import ResultMessage

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
        async def mock_receive_response():
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
    async def test_summary_indicates_missing_log(self, tmp_path: Path):
        """Failure summary should clearly indicate the log file was missing."""
        from unittest.mock import AsyncMock, MagicMock
        from claude_agent_sdk.types import ResultMessage

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

        async def mock_receive_response():
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
            mock_tracer = MagicMock()
            mock_tracer.__enter__ = MagicMock(return_value=mock_tracer)
            mock_tracer.__exit__ = MagicMock(return_value=None)
            mock_tracer_cls.return_value = mock_tracer

            result = await orchestrator.run_implementer("issue-xyz")

        # Summary should mention session log and the timeout
        assert "Session log missing" in result.summary
        assert "30s" in result.summary  # Default timeout


class TestSubprocessTerminationOnTimeout:
    """Test that timed-out subprocesses are properly terminated."""

    @pytest.mark.asyncio
    async def test_subprocess_terminated_on_timeout(self, tmp_path: Path):
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
    async def test_subprocess_killed_if_terminate_fails(self, tmp_path: Path):
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
    async def test_successful_command_not_affected(self, tmp_path: Path):
        """Fast commands should complete normally without being terminated."""
        from src.beads_client import BeadsClient

        beads = BeadsClient(tmp_path, timeout_seconds=10.0)

        result = await beads._run_subprocess_async(["echo", "hello"])

        assert result.returncode == 0
        assert result.stdout.strip() == "hello"
        assert result.stderr == ""

    @pytest.mark.asyncio
    async def test_command_stderr_captured(self, tmp_path: Path):
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
    async def test_child_processes_killed_on_timeout(self, tmp_path: Path):
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
