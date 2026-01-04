"""Unit tests for OrchestratorState dataclass."""

from pathlib import Path

from src.orchestration.orchestrator_state import OrchestratorState
from src.pipeline.issue_result import IssueResult


class TestOrchestratorStateInitialization:
    """Tests for OrchestratorState initialization with empty collections."""

    def test_default_initialization(self) -> None:
        """State initializes with empty collections by default."""
        state = OrchestratorState()

        assert state.agent_ids == {}
        assert state.completed == []
        assert state.active_session_log_paths == {}
        assert state.deadlock_cleaned_agents == set()

    def test_fields_are_independent_instances(self) -> None:
        """Each state instance has independent mutable collections."""
        state1 = OrchestratorState()
        state2 = OrchestratorState()

        state1.agent_ids["issue-1"] = "agent-1"
        state1.completed.append(
            IssueResult(
                issue_id="issue-1",
                agent_id="agent-1",
                success=True,
                summary="Done",
            )
        )
        state1.active_session_log_paths["agent-1"] = Path("/tmp/log1")
        state1.deadlock_cleaned_agents.add("agent-1")

        # state2 should remain empty
        assert state2.agent_ids == {}
        assert state2.completed == []
        assert state2.active_session_log_paths == {}
        assert state2.deadlock_cleaned_agents == set()


class TestOrchestratorStateUsage:
    """Tests for typical usage patterns of OrchestratorState."""

    def test_agent_ids_mapping(self) -> None:
        """Can map issue IDs to agent IDs."""
        state = OrchestratorState()

        state.agent_ids["mala-abc.1"] = "mala-abc.1-12345678"
        state.agent_ids["mala-abc.2"] = "mala-abc.2-87654321"

        assert state.agent_ids["mala-abc.1"] == "mala-abc.1-12345678"
        assert len(state.agent_ids) == 2

    def test_completed_results_list(self) -> None:
        """Can track completed issue results."""
        state = OrchestratorState()
        result = IssueResult(
            issue_id="mala-xyz.1",
            agent_id="mala-xyz.1-abcdef",
            success=True,
            summary="Implemented feature",
            duration_seconds=120.5,
        )

        state.completed.append(result)

        assert len(state.completed) == 1
        assert state.completed[0].issue_id == "mala-xyz.1"
        assert state.completed[0].success is True

    def test_active_session_log_paths(self) -> None:
        """Can track active session log paths for agents."""
        state = OrchestratorState()

        state.active_session_log_paths["agent-1"] = Path(
            "/home/user/.claude/logs/session1.jsonl"
        )
        state.active_session_log_paths["agent-2"] = Path(
            "/home/user/.claude/logs/session2.jsonl"
        )

        assert len(state.active_session_log_paths) == 2
        assert state.active_session_log_paths["agent-1"].name == "session1.jsonl"

    def test_deadlock_cleaned_agents_set(self) -> None:
        """Can track agents cleaned during deadlock resolution."""
        state = OrchestratorState()

        state.deadlock_cleaned_agents.add("agent-1")
        state.deadlock_cleaned_agents.add("agent-2")
        state.deadlock_cleaned_agents.add("agent-1")  # duplicate

        assert len(state.deadlock_cleaned_agents) == 2
        assert "agent-1" in state.deadlock_cleaned_agents
