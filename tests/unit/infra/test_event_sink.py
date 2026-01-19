"""Tests for event_sink module."""

from dataclasses import dataclass

import pytest

from src.infra.io.base_sink import BaseEventSink, NullEventSink
from src.infra.io.console_sink import ConsoleEventSink
from src.core.protocols.events import EventRunConfig, MalaEventSink


@dataclass
class FakeDeadlockInfo:
    """Fake implementation of DeadlockInfoProtocol for testing."""

    cycle: list[str]
    victim_id: str
    victim_issue_id: str | None
    blocked_on: str
    blocker_id: str
    blocker_issue_id: str | None


class TestBaseEventSink:
    """Tests for BaseEventSink implementation."""

    def test_implements_protocol(self) -> None:
        """BaseEventSink implements MalaEventSink protocol."""
        sink = BaseEventSink()
        # Protocol structural subtyping - if it quacks like a duck...
        # This verifies all required methods exist with compatible signatures
        assert isinstance(sink, MalaEventSink)

    def test_protocol_coverage(self) -> None:
        """Verify all protocol methods are implemented in BaseEventSink.

        This test ensures protocol and implementation stay in sync.
        """
        # Get protocol-declared methods from __protocol_attrs__ or fallback to __dict__
        # This excludes ABCMeta helpers like 'register' that appear in dir()
        if hasattr(MalaEventSink, "__protocol_attrs__"):
            protocol_methods = MalaEventSink.__protocol_attrs__
        else:
            protocol_methods = {
                name
                for name in MalaEventSink.__dict__
                if not name.startswith("_")
                and callable(getattr(MalaEventSink, name, None))
            }

        # Get all public methods from BaseEventSink
        sink_methods = {
            name
            for name in dir(BaseEventSink)
            if not name.startswith("_") and callable(getattr(BaseEventSink, name, None))
        }

        # All protocol methods should be in the sink
        missing = protocol_methods - sink_methods
        assert not missing, f"BaseEventSink missing protocol methods: {missing}"

    def test_all_methods_are_no_ops(self) -> None:
        """All BaseEventSink methods execute without side effects."""
        sink = BaseEventSink()
        config = EventRunConfig(
            repo_path="/tmp/repo",
            max_agents=2,
            timeout_minutes=30,
            max_issues=10,
            max_gate_retries=3,
            max_review_retries=2,
        )

        # Run lifecycle - all should return None (no side effects)
        assert sink.on_run_started(config) is None
        assert sink.on_run_completed(5, 10, True) is None
        assert sink.on_ready_issues(["issue-1", "issue-2"]) is None
        assert sink.on_waiting_for_agents(3) is None
        assert sink.on_no_more_issues("none_ready") is None

        # Agent lifecycle
        assert sink.on_agent_started("agent-1", "issue-1") is None
        assert sink.on_agent_completed("agent-1", "issue-1", True, 60.0, "Done") is None
        assert sink.on_claim_failed("agent-1", "issue-1") is None

        # SDK message streaming
        assert (
            sink.on_tool_use("agent-1", "Read", "file.py", {"path": "file.py"}) is None
        )
        assert sink.on_agent_text("agent-1", "Working on it...") is None

        # Quality gate events
        assert sink.on_gate_started("agent-1", 1, 3) is None
        assert sink.on_gate_started(None, 1, 3) is None  # global
        assert sink.on_gate_passed("agent-1") is None
        assert sink.on_gate_passed(None) is None  # global
        assert sink.on_gate_failed("agent-1", 3, 3) is None
        assert sink.on_gate_retry("agent-1", 2, 3) is None
        assert sink.on_gate_result("agent-1", True) is None
        assert (
            sink.on_gate_result("agent-1", False, ["lint failed", "tests failed"])
            is None
        )
        assert sink.on_gate_result(None, True) is None  # global

        # Codex review events
        assert sink.on_review_started("agent-1", 1, 2) is None
        assert sink.on_review_passed("agent-1") is None
        assert sink.on_review_retry("agent-1", 2, 2, error_count=5) is None
        assert sink.on_review_retry("agent-1", 2, 2, parse_error="Invalid JSON") is None

        # Fixer agent events
        assert sink.on_fixer_started(1, 3) is None
        assert sink.on_fixer_completed("Fixed lint errors") is None
        assert sink.on_fixer_failed("timeout") is None

        # Issue lifecycle
        assert sink.on_issue_closed("agent-1", "issue-1") is None
        assert (
            sink.on_issue_completed("agent-1", "issue-1", True, 120.5, "Done") is None
        )
        assert (
            sink.on_issue_completed("agent-1", "issue-2", False, 60.0, "Failed") is None
        )
        assert sink.on_epic_closed("agent-1") is None
        assert sink.on_validation_started("agent-1") is None
        assert sink.on_validation_result("agent-1", True) is None

        # Warnings and diagnostics
        assert sink.on_warning("Something went wrong") is None
        assert sink.on_warning("Agent issue", agent_id="agent-1") is None
        assert sink.on_log_timeout("agent-1", "/tmp/log.jsonl") is None
        assert sink.on_locks_cleaned("agent-1", 3) is None
        assert sink.on_locks_released(5) is None
        assert sink.on_issues_committed() is None
        assert sink.on_run_metadata_saved("/tmp/run.json") is None
        assert sink.on_global_validation_disabled() is None
        assert sink.on_abort_requested("Fatal error occurred") is None
        assert sink.on_tasks_aborting(3, "Unrecoverable error") is None

        # Epic verification lifecycle
        assert sink.on_epic_verification_started("epic-1") is None
        assert sink.on_epic_verification_passed("epic-1") is None
        assert (
            sink.on_epic_verification_failed("epic-1", 2, ["issue-1", "issue-2"])
            is None
        )
        assert (
            sink.on_epic_remediation_created("epic-1", "issue-4", "Criterion text here")
            is None
        )

        # Deadlock detection
        deadlock_info = FakeDeadlockInfo(
            cycle=["agent-1", "agent-2"],
            victim_id="agent-2",
            victim_issue_id="issue-2",
            blocked_on="/tmp/lock",
            blocker_id="agent-1",
            blocker_issue_id="issue-1",
        )
        assert sink.on_deadlock_detected(deadlock_info) is None

        # Watch mode events
        assert sink.on_watch_idle(30.0, None) is None
        assert sink.on_watch_idle(60.0, 5) is None

        # Session end lifecycle
        assert sink.on_session_end_started("issue-1") is None
        assert sink.on_session_end_completed("issue-1", "pass") is None
        assert sink.on_session_end_skipped("issue-1", "gate_failed") is None


class TestNullEventSink:
    """Tests for NullEventSink implementation."""

    def test_implements_protocol(self) -> None:
        """NullEventSink implements MalaEventSink protocol."""
        sink = NullEventSink()
        # Protocol structural subtyping - if it quacks like a duck...
        # This verifies all required methods exist with compatible signatures
        assert isinstance(sink, MalaEventSink)

    def test_inherits_from_base_event_sink(self) -> None:
        """NullEventSink inherits from BaseEventSink."""
        sink = NullEventSink()
        assert isinstance(sink, BaseEventSink)

    def test_all_methods_are_no_ops(self) -> None:
        """All NullEventSink methods execute without side effects."""
        sink = NullEventSink()
        config = EventRunConfig(
            repo_path="/tmp/repo",
            max_agents=2,
            timeout_minutes=30,
            max_issues=10,
            max_gate_retries=3,
            max_review_retries=2,
        )

        # Run lifecycle - all should return None (no side effects)
        assert sink.on_run_started(config) is None
        assert sink.on_run_completed(5, 10, True) is None
        assert sink.on_ready_issues(["issue-1", "issue-2"]) is None
        assert sink.on_waiting_for_agents(3) is None
        assert sink.on_no_more_issues("none_ready") is None

        # Agent lifecycle
        assert sink.on_agent_started("agent-1", "issue-1") is None
        assert sink.on_agent_completed("agent-1", "issue-1", True, 60.0, "Done") is None
        assert sink.on_claim_failed("agent-1", "issue-1") is None

        # SDK message streaming
        assert (
            sink.on_tool_use("agent-1", "Read", "file.py", {"path": "file.py"}) is None
        )
        assert sink.on_agent_text("agent-1", "Working on it...") is None

        # Quality gate events
        assert sink.on_gate_started("agent-1", 1, 3) is None
        assert sink.on_gate_started(None, 1, 3) is None  # global
        assert sink.on_gate_passed("agent-1") is None
        assert sink.on_gate_passed(None) is None  # global
        assert sink.on_gate_failed("agent-1", 3, 3) is None
        assert sink.on_gate_retry("agent-1", 2, 3) is None
        assert sink.on_gate_result("agent-1", True) is None
        assert (
            sink.on_gate_result("agent-1", False, ["lint failed", "tests failed"])
            is None
        )
        assert sink.on_gate_result(None, True) is None  # global

        # Codex review events
        assert sink.on_review_started("agent-1", 1, 2) is None
        assert sink.on_review_passed("agent-1") is None
        assert sink.on_review_retry("agent-1", 2, 2, error_count=5) is None
        assert sink.on_review_retry("agent-1", 2, 2, parse_error="Invalid JSON") is None

        # Fixer agent events
        assert sink.on_fixer_started(1, 3) is None
        assert sink.on_fixer_completed("Fixed lint errors") is None
        assert sink.on_fixer_failed("timeout") is None

        # Issue lifecycle
        assert sink.on_issue_closed("agent-1", "issue-1") is None
        assert (
            sink.on_issue_completed("agent-1", "issue-1", True, 120.5, "Done") is None
        )
        assert (
            sink.on_issue_completed("agent-1", "issue-2", False, 60.0, "Failed") is None
        )
        assert sink.on_epic_closed("agent-1") is None
        assert sink.on_validation_started("agent-1") is None
        assert sink.on_validation_result("agent-1", True) is None

        # Warnings and diagnostics
        assert sink.on_warning("Something went wrong") is None
        assert sink.on_warning("Agent issue", agent_id="agent-1") is None
        assert sink.on_log_timeout("agent-1", "/tmp/log.jsonl") is None
        assert sink.on_locks_cleaned("agent-1", 3) is None
        assert sink.on_locks_released(5) is None
        assert sink.on_issues_committed() is None
        assert sink.on_run_metadata_saved("/tmp/run.json") is None
        assert sink.on_global_validation_disabled() is None
        assert sink.on_abort_requested("Fatal error occurred") is None
        assert sink.on_tasks_aborting(3, "Unrecoverable error") is None

        # Epic verification lifecycle
        assert sink.on_epic_verification_started("epic-1") is None
        assert sink.on_epic_verification_passed("epic-1") is None
        assert (
            sink.on_epic_verification_failed("epic-1", 2, ["issue-1", "issue-2"])
            is None
        )
        assert (
            sink.on_epic_remediation_created("epic-1", "issue-4", "Criterion text here")
            is None
        )

        # Deadlock detection
        deadlock_info = FakeDeadlockInfo(
            cycle=["agent-1", "agent-2"],
            victim_id="agent-2",
            victim_issue_id="issue-2",
            blocked_on="/tmp/lock",
            blocker_id="agent-1",
            blocker_issue_id="issue-1",
        )
        assert sink.on_deadlock_detected(deadlock_info) is None

        # Watch mode events
        assert sink.on_watch_idle(30.0, None) is None
        assert sink.on_watch_idle(60.0, 5) is None

        # Session end lifecycle
        assert sink.on_session_end_started("issue-1") is None
        assert sink.on_session_end_completed("issue-1", "pass") is None
        assert sink.on_session_end_skipped("issue-1", "gate_failed") is None

    def test_can_be_called_multiple_times(self) -> None:
        """NullEventSink methods can be called repeatedly."""
        sink = NullEventSink()

        # Simulate a typical orchestrator flow
        config = EventRunConfig(
            repo_path="/tmp/repo",
            max_agents=2,
            timeout_minutes=30,
            max_issues=10,
            max_gate_retries=3,
            max_review_retries=2,
        )

        sink.on_run_started(config)

        # Spawn multiple agents
        for i in range(5):
            agent_id = f"agent-{i}"
            issue_id = f"issue-{i}"
            sink.on_agent_started(agent_id, issue_id)
            sink.on_tool_use(agent_id, "Read", "file.py")
            sink.on_agent_text(agent_id, "Processing...")
            sink.on_gate_started(agent_id, 1, 3)
            sink.on_gate_passed(agent_id)
            sink.on_agent_completed(agent_id, issue_id, True, 30.0, "Done")
            sink.on_issue_closed(agent_id, issue_id)

        sink.on_run_completed(5, 5, True)

    def test_protocol_coverage(self) -> None:
        """Verify all protocol methods are implemented in NullEventSink.

        This test ensures protocol and implementation stay in sync.
        """
        # Get protocol-declared methods from __protocol_attrs__ or fallback to __dict__
        # This excludes ABCMeta helpers like 'register' that appear in dir()
        if hasattr(MalaEventSink, "__protocol_attrs__"):
            protocol_methods = MalaEventSink.__protocol_attrs__
        else:
            protocol_methods = {
                name
                for name in MalaEventSink.__dict__
                if not name.startswith("_")
                and callable(getattr(MalaEventSink, name, None))
            }

        # Get all public methods from NullEventSink
        sink_methods = {
            name
            for name in dir(NullEventSink)
            if not name.startswith("_") and callable(getattr(NullEventSink, name, None))
        }

        # All protocol methods should be in the sink
        missing = protocol_methods - sink_methods
        assert not missing, f"NullEventSink missing protocol methods: {missing}"

    def test_on_validation_started_returns_none(self) -> None:
        """on_validation_started returns None (no-op)."""
        sink = NullEventSink()
        assert sink.on_validation_started("agent-1") is None
        assert sink.on_validation_started("agent-1", issue_id="issue-1") is None

    def test_methods_accept_issue_id_param(self) -> None:
        """Methods with issue_id parameter accept it without error."""
        sink = NullEventSink()

        # Gate methods with issue_id
        assert sink.on_gate_started("agent-1", 1, 3, issue_id="issue-1") is None
        assert sink.on_gate_started(None, 1, 3, issue_id=None) is None
        assert sink.on_gate_passed("agent-1", issue_id="issue-1") is None
        assert sink.on_gate_passed(None, issue_id=None) is None
        assert sink.on_gate_failed("agent-1", 3, 3, issue_id="issue-1") is None
        assert sink.on_gate_retry("agent-1", 2, 3, issue_id="issue-1") is None
        assert (
            sink.on_gate_result("agent-1", False, ["lint"], issue_id="issue-1") is None
        )

        # Review methods with issue_id
        assert sink.on_review_started("agent-1", 1, 2, issue_id="issue-1") is None
        assert sink.on_review_passed("agent-1", issue_id="issue-1") is None
        assert (
            sink.on_review_retry("agent-1", 2, 2, error_count=5, issue_id="issue-1")
            is None
        )

        # Validation methods with issue_id
        assert sink.on_validation_started("agent-1", issue_id="issue-1") is None
        assert sink.on_validation_result("agent-1", True, issue_id="issue-1") is None


class TestConsoleEventSink:
    """Tests for ConsoleEventSink implementation."""

    def test_implements_protocol(self) -> None:
        """ConsoleEventSink implements MalaEventSink protocol."""
        sink = ConsoleEventSink()
        # Protocol structural subtyping
        assert isinstance(sink, MalaEventSink)

    def test_protocol_coverage(self) -> None:
        """Verify all protocol methods are implemented in ConsoleEventSink."""
        # Get protocol-declared methods from __protocol_attrs__ or fallback to __dict__
        # This excludes ABCMeta helpers like 'register' that appear in dir()
        if hasattr(MalaEventSink, "__protocol_attrs__"):
            protocol_methods = MalaEventSink.__protocol_attrs__
        else:
            protocol_methods = {
                name
                for name in MalaEventSink.__dict__
                if not name.startswith("_")
                and callable(getattr(MalaEventSink, name, None))
            }

        # Get all public methods from ConsoleEventSink
        sink_methods = {
            name
            for name in dir(ConsoleEventSink)
            if not name.startswith("_")
            and callable(getattr(ConsoleEventSink, name, None))
        }

        # All protocol methods should be in the sink
        missing = protocol_methods - sink_methods
        assert not missing, f"ConsoleEventSink missing protocol methods: {missing}"

    def test_all_methods_execute_without_error(self) -> None:
        """All ConsoleEventSink methods execute without raising exceptions.

        Tests that the sink's methods can be called with valid arguments.
        Actual logging output is tested via integration tests.
        """
        sink = ConsoleEventSink()
        config = EventRunConfig(
            repo_path="/tmp/repo",
            max_agents=2,
            timeout_minutes=30,
            max_issues=10,
            max_gate_retries=3,
            max_review_retries=2,
        )

        # Run lifecycle - should not raise
        sink.on_run_started(config)
        sink.on_run_completed(5, 10, True)
        sink.on_ready_issues(["issue-1"])
        sink.on_no_more_issues("none_ready")
        sink.on_waiting_for_agents(3)

        # Agent lifecycle
        sink.on_agent_started("agent-1", "issue-1")
        sink.on_agent_completed("agent-1", "issue-1", True, 60.0, "Done")
        sink.on_claim_failed("agent-1", "issue-1")

        # SDK message streaming
        sink.on_tool_use("agent-1", "Read", "file.py", {"path": "file.py"})
        sink.on_agent_text("agent-1", "Processing...")

        # Quality gate events
        sink.on_gate_started("agent-1", 1, 3, issue_id="issue-123")
        sink.on_gate_started(None, 1, 3)
        sink.on_gate_passed("agent-1", issue_id="issue-456")
        sink.on_gate_passed(None)
        sink.on_gate_failed("agent-1", 3, 3, issue_id="issue-789")
        sink.on_gate_retry("agent-1", 2, 3, issue_id="issue-abc")
        sink.on_gate_result("agent-1", True)
        sink.on_gate_result("agent-1", False, ["lint failed"], issue_id="issue-mno")
        sink.on_gate_result(None, True)

        # Codex review events
        sink.on_review_started("agent-1", 1, 2, issue_id="issue-123")
        sink.on_review_passed("agent-1", issue_id="issue-def")
        sink.on_review_retry("agent-1", 2, 2, error_count=5, issue_id="issue-ghi")
        sink.on_review_retry("agent-1", 2, 2, parse_error="Invalid JSON")

        # Fixer agent events
        sink.on_fixer_started(1, 3)
        sink.on_fixer_completed("Fixed lint errors")
        sink.on_fixer_failed("timeout")

        # Issue lifecycle
        sink.on_issue_closed("agent-1", "issue-1")
        sink.on_issue_completed("agent-1", "issue-1", True, 120.5, "Done")
        sink.on_issue_completed("agent-1", "issue-2", False, 60.0, "Failed")
        sink.on_epic_closed("agent-1")
        sink.on_validation_started("agent-1", issue_id="issue-123")
        sink.on_validation_result("agent-1", True, issue_id="issue-pqr")
        sink.on_validation_result("agent-1", False, issue_id="issue-jkl")

        # Warnings and diagnostics
        sink.on_warning("Something went wrong")
        sink.on_warning("Agent issue", agent_id="agent-1")
        sink.on_log_timeout("agent-1", "/tmp/log.jsonl")
        sink.on_locks_cleaned("agent-1", 3)
        sink.on_locks_released(5)
        sink.on_issues_committed()
        sink.on_run_metadata_saved("/tmp/run.json")
        sink.on_global_validation_disabled()
        sink.on_abort_requested("Fatal error occurred")
        sink.on_tasks_aborting(3, "Unrecoverable error")

        # Epic verification lifecycle
        sink.on_epic_verification_started("epic-1")
        sink.on_epic_verification_passed("epic-1")
        sink.on_epic_verification_failed("epic-1", 2, ["issue-1", "issue-2"])
        sink.on_epic_remediation_created("epic-123", "issue-1", "A" * 100)
        sink.on_epic_remediation_created("epic-123", "issue-2", "Short criterion")

        # Deadlock detection
        deadlock_info = FakeDeadlockInfo(
            cycle=["agent-1", "agent-2", "agent-3"],
            victim_id="agent-3",
            victim_issue_id="issue-3",
            blocked_on="/tmp/locks/file.py",
            blocker_id="agent-1",
            blocker_issue_id="issue-1",
        )
        sink.on_deadlock_detected(deadlock_info)

        # Deadlock with None issue IDs
        deadlock_info_none = FakeDeadlockInfo(
            cycle=["agent-1", "agent-2"],
            victim_id="agent-2",
            victim_issue_id=None,
            blocked_on="/tmp/lock",
            blocker_id="agent-1",
            blocker_issue_id=None,
        )
        sink.on_deadlock_detected(deadlock_info_none)

        # Watch mode events
        sink.on_watch_idle(30.0, None)
        sink.on_watch_idle(45.5, 0)
        sink.on_watch_idle(60.0, 5)

        # Session end lifecycle
        sink.on_session_end_started("issue-1")
        sink.on_session_end_completed("issue-1", "pass")
        sink.on_session_end_completed("issue-2", "fail")
        sink.on_session_end_skipped("issue-3", "gate_failed")
        sink.on_session_end_skipped("issue-4", "not_configured")

    def test_on_abort_requested_includes_abort_in_message(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """on_abort_requested message should include 'ABORT' keyword for visibility."""
        sink = ConsoleEventSink()

        sink.on_abort_requested("Fatal error occurred")

        captured = capsys.readouterr()
        assert "ABORT" in captured.out

    def test_on_epic_remediation_truncates_long_criterion(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """on_epic_remediation_created should truncate long criteria to 80 chars."""
        sink = ConsoleEventSink()

        # Create a criterion longer than 80 characters
        long_criterion = "A" * 100

        sink.on_epic_remediation_created("epic-123", "issue-1", long_criterion)

        captured = capsys.readouterr()
        # Should contain truncated text with ellipsis, not the full 100 A's
        assert "A" * 80 not in captured.out or "..." in captured.out
        # Should not contain the full 100-char string
        assert long_criterion not in captured.out

    def test_on_gate_failed_includes_red_color(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """on_gate_failed should use RED color for failed status."""
        sink = ConsoleEventSink()

        sink.on_gate_failed("agent-1", 3, 3, issue_id="issue-789")

        captured = capsys.readouterr()
        # Should include the word "failed" (red color codes are ANSI escape sequences)
        assert "failed" in captured.out

    def test_on_run_started_with_triggers_logs_trigger_summary(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """on_run_started with triggers should log trigger summary."""
        from src.core.protocols.events import TriggerSummary, ValidationTriggersSummary

        sink = ConsoleEventSink()
        triggers = ValidationTriggersSummary(
            session_end=TriggerSummary(
                enabled=True,
                failure_mode="remediate",
                command_count=2,
            ),
            periodic=TriggerSummary(
                enabled=True,
                failure_mode="continue",
                command_count=1,
            ),
        )
        config = EventRunConfig(
            repo_path="/tmp/repo",
            max_agents=2,
            timeout_minutes=30,
            max_issues=10,
            max_gate_retries=3,
            max_review_retries=2,
            validation_triggers=triggers,
        )

        sink.on_run_started(config)

        captured = capsys.readouterr()
        # Should include trigger summary in output
        assert "Triggers:" in captured.out
        assert "session_end" in captured.out
        assert "remediate" in captured.out
        assert "periodic" in captured.out
        assert "continue" in captured.out

    def test_on_run_started_without_triggers_shows_verbose_message(self) -> None:
        """on_run_started without triggers should only show verbose message."""
        sink = ConsoleEventSink()
        config = EventRunConfig(
            repo_path="/tmp/repo",
            max_agents=2,
            timeout_minutes=30,
            max_issues=10,
            max_gate_retries=3,
            max_review_retries=2,
        )

        # Should execute without error - verbose message won't show without verbose mode
        sink.on_run_started(config)

    def test_on_session_end_started_log_format(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """on_session_end_started logs correct format."""
        sink = ConsoleEventSink()

        sink.on_session_end_started("issue-123")

        captured = capsys.readouterr()
        assert "[issue-123]" in captured.out
        assert "[session_end] started" in captured.out

    def test_on_session_end_completed_log_format(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """on_session_end_completed logs correct format with result."""
        sink = ConsoleEventSink()

        sink.on_session_end_completed("issue-456", "pass")

        captured = capsys.readouterr()
        assert "[issue-456]" in captured.out
        assert "[session_end] completed" in captured.out
        assert "result=" in captured.out
        assert "pass" in captured.out

    def test_on_session_end_skipped_log_format(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """on_session_end_skipped logs correct format with reason."""
        sink = ConsoleEventSink()

        sink.on_session_end_skipped("issue-789", "gate_failed")

        captured = capsys.readouterr()
        assert "[issue-789]" in captured.out
        assert "[session_end] skipped" in captured.out
        assert "reason=gate_failed" in captured.out
