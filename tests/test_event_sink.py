"""Tests for event_sink module."""

from unittest.mock import MagicMock, patch

from src.infra.io.event_sink import (
    MalaEventSink,
    NullEventSink,
    EventRunConfig,
    ConsoleEventSink,
)


class TestEventRunConfig:
    """Tests for EventRunConfig dataclass."""

    def test_minimal_config(self) -> None:
        """EventRunConfig can be created with required fields only."""
        config = EventRunConfig(
            repo_path="/tmp/repo",
            max_agents=2,
            timeout_minutes=30,
            max_issues=10,
            max_gate_retries=3,
            max_review_retries=2,
        )
        assert config.repo_path == "/tmp/repo"
        assert config.max_agents == 2
        assert config.timeout_minutes == 30
        assert config.max_issues == 10
        assert config.max_gate_retries == 3
        assert config.max_review_retries == 2
        # Defaults
        assert config.epic_id is None
        assert config.only_ids is None
        assert config.braintrust_enabled is False
        assert config.review_enabled is True
        assert config.morph_enabled is True
        assert config.morph_disallowed_tools is None
        assert config.prioritize_wip is False
        assert config.cli_args is None

    def test_full_config(self) -> None:
        """EventRunConfig can be created with all optional fields."""
        config = EventRunConfig(
            repo_path="/tmp/repo",
            max_agents=4,
            timeout_minutes=60,
            max_issues=20,
            max_gate_retries=5,
            max_review_retries=3,
            epic_id="epic-123",
            only_ids=["issue-1", "issue-2"],
            braintrust_enabled=True,
            review_enabled=False,
            morph_enabled=False,
            morph_disallowed_tools=["Edit", "Write"],
            prioritize_wip=True,
            cli_args={"verbose": True},
        )
        assert config.epic_id == "epic-123"
        assert config.only_ids == ["issue-1", "issue-2"]
        assert config.braintrust_enabled is True
        assert config.review_enabled is False
        assert config.morph_enabled is False
        assert config.morph_disallowed_tools == ["Edit", "Write"]
        assert config.prioritize_wip is True
        assert config.cli_args == {"verbose": True}

    def test_none_values_allowed(self) -> None:
        """EventRunConfig accepts None for optional int fields."""
        config = EventRunConfig(
            repo_path="/tmp/repo",
            max_agents=None,
            timeout_minutes=None,
            max_issues=None,
            max_gate_retries=3,
            max_review_retries=2,
        )
        assert config.max_agents is None
        assert config.timeout_minutes is None
        assert config.max_issues is None


class TestNullEventSink:
    """Tests for NullEventSink implementation."""

    def test_implements_protocol(self) -> None:
        """NullEventSink implements MalaEventSink protocol."""
        sink = NullEventSink()
        # Protocol structural subtyping - if it quacks like a duck...
        # This verifies all required methods exist with compatible signatures
        assert isinstance(sink, MalaEventSink)

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
        assert sink.on_gate_started(None, 1, 3) is None  # run-level
        assert sink.on_gate_passed("agent-1") is None
        assert sink.on_gate_passed(None) is None  # run-level
        assert sink.on_gate_failed("agent-1", 3, 3) is None
        assert sink.on_gate_retry("agent-1", 2, 3) is None
        assert sink.on_gate_result("agent-1", True) is None
        assert (
            sink.on_gate_result("agent-1", False, ["lint failed", "tests failed"])
            is None
        )
        assert sink.on_gate_result(None, True) is None  # run-level

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
        assert sink.on_run_level_validation_disabled() is None
        assert sink.on_abort_requested("Fatal error occurred") is None
        assert sink.on_tasks_aborting(3, "Unrecoverable error") is None

        # Epic verification lifecycle
        assert sink.on_epic_verification_started("epic-1") is None
        assert sink.on_epic_verification_passed("epic-1", 0.95) is None
        assert (
            sink.on_epic_verification_failed("epic-1", 2, ["issue-1", "issue-2"])
            is None
        )
        assert (
            sink.on_epic_verification_human_review(
                "epic-1", "Low confidence", "issue-3"
            )
            is None
        )
        assert (
            sink.on_epic_remediation_created("epic-1", "issue-4", "Criterion text here")
            is None
        )

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
        # Get all public methods from the protocol (excluding dunder methods)
        protocol_methods = {
            name
            for name in dir(MalaEventSink)
            if not name.startswith("_") and callable(getattr(MalaEventSink, name, None))
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
        # Get all public methods from the protocol (excluding dunder methods)
        protocol_methods = {
            name
            for name in dir(MalaEventSink)
            if not name.startswith("_") and callable(getattr(MalaEventSink, name, None))
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

    @patch("src.infra.io.event_sink.log")
    def test_on_run_started_logs(self, mock_log: MagicMock) -> None:
        """on_run_started calls log with run configuration."""
        sink = ConsoleEventSink()
        config = EventRunConfig(
            repo_path="/tmp/repo",
            max_agents=2,
            timeout_minutes=30,
            max_issues=10,
            max_gate_retries=3,
            max_review_retries=2,
        )
        sink.on_run_started(config)
        assert mock_log.called
        # Verify one of the log calls contains the repo path
        all_call_args = [str(call) for call in mock_log.call_args_list]
        assert any("/tmp/repo" in args for args in all_call_args)

    @patch("src.infra.io.event_sink.log_tool")
    def test_on_tool_use_delegates_to_log_tool(self, mock_log_tool: MagicMock) -> None:
        """on_tool_use delegates to log_tool helper."""
        sink = ConsoleEventSink()
        sink.on_tool_use("agent-1", "Read", "file.py", {"path": "file.py"})
        mock_log_tool.assert_called_once_with(
            "Read", "file.py", agent_id="agent-1", arguments={"path": "file.py"}
        )

    @patch("src.infra.io.event_sink.log_agent_text")
    def test_on_agent_text_delegates_to_log_agent_text(
        self, mock_log_agent_text: MagicMock
    ) -> None:
        """on_agent_text delegates to log_agent_text helper."""
        sink = ConsoleEventSink()
        sink.on_agent_text("agent-1", "Processing...")
        mock_log_agent_text.assert_called_once_with("Processing...", "agent-1")

    @patch("src.infra.io.event_sink.log")
    def test_representative_methods_execute(self, mock_log: MagicMock) -> None:
        """Representative methods from each category execute without error."""
        sink = ConsoleEventSink()
        config = EventRunConfig(
            repo_path="/tmp/repo",
            max_agents=2,
            timeout_minutes=30,
            max_issues=10,
            max_gate_retries=3,
            max_review_retries=2,
        )

        # Run lifecycle
        sink.on_run_started(config)
        sink.on_run_completed(5, 10, True)
        sink.on_ready_issues(["issue-1"])
        sink.on_no_more_issues("none_ready")

        # Agent lifecycle
        sink.on_agent_started("agent-1", "issue-1")
        sink.on_agent_completed("agent-1", "issue-1", True, 60.0, "Done")
        sink.on_claim_failed("agent-1", "issue-1")

        # Quality gate events
        sink.on_gate_passed("agent-1")
        sink.on_gate_passed(None)  # run-level
        sink.on_gate_failed("agent-1", 3, 3)
        sink.on_gate_retry("agent-1", 2, 3)
        sink.on_gate_result("agent-1", False, ["lint failed"])

        # Codex review events
        sink.on_review_passed("agent-1")
        sink.on_review_retry("agent-1", 2, 2, error_count=5)
        sink.on_review_retry("agent-1", 2, 2, parse_error="Invalid JSON")

        # Fixer agent events
        sink.on_fixer_started(1, 3)
        sink.on_fixer_completed("Fixed lint errors")
        sink.on_fixer_failed("timeout")

        # Issue lifecycle
        sink.on_issue_closed("agent-1", "issue-1")
        sink.on_issue_completed("agent-1", "issue-1", True, 120.5, "Done")
        sink.on_epic_closed("agent-1")

        # Warnings and diagnostics
        sink.on_warning("Something went wrong")
        sink.on_warning("Agent issue", agent_id="agent-1")
        sink.on_log_timeout("agent-1", "/tmp/log.jsonl")
        sink.on_abort_requested("Fatal error occurred")
        sink.on_tasks_aborting(3, "Unrecoverable error")

        # All should have executed without raising exceptions
        assert mock_log.called

    @patch("src.infra.io.event_sink.log")
    def test_on_abort_requested_logs_fatal_error(self, mock_log: MagicMock) -> None:
        """on_abort_requested logs fatal error with abort message."""
        sink = ConsoleEventSink()
        sink.on_abort_requested("Database connection lost")

        mock_log.assert_called_once()
        call_args = mock_log.call_args
        assert "Fatal error" in call_args[0][1]
        assert "Database connection lost" in call_args[0][1]
        assert "Aborting run" in call_args[0][1]

    @patch("src.infra.io.event_sink.log")
    def test_on_tasks_aborting_logs_count_and_reason(self, mock_log: MagicMock) -> None:
        """on_tasks_aborting logs task count and reason."""
        sink = ConsoleEventSink()
        sink.on_tasks_aborting(5, "Timeout exceeded")

        mock_log.assert_called_once()
        call_args = mock_log.call_args
        assert "5" in call_args[0][1]
        assert "active task" in call_args[0][1]
        assert "Timeout exceeded" in call_args[0][1]

    @patch("src.infra.io.event_sink.log")
    def test_on_epic_verification_started_logs_epic_id(
        self, mock_log: MagicMock
    ) -> None:
        """on_epic_verification_started logs the epic being verified."""
        sink = ConsoleEventSink()
        sink.on_epic_verification_started("epic-123")

        mock_log.assert_called_once()
        call_args = mock_log.call_args
        assert "Verifying epic" in call_args[0][1]
        assert "epic-123" in call_args[0][1]

    @patch("src.infra.io.event_sink.log")
    def test_on_epic_verification_passed_logs_confidence(
        self, mock_log: MagicMock
    ) -> None:
        """on_epic_verification_passed logs epic id and confidence."""
        sink = ConsoleEventSink()
        sink.on_epic_verification_passed("epic-123", 0.95)

        mock_log.assert_called_once()
        call_args = mock_log.call_args
        assert "epic-123" in call_args[0][1]
        assert "verified" in call_args[0][1]
        assert "95%" in call_args[0][1]

    @patch("src.infra.io.event_sink.log")
    def test_on_epic_verification_failed_logs_details(
        self, mock_log: MagicMock
    ) -> None:
        """on_epic_verification_failed logs unmet count and remediation issues."""
        sink = ConsoleEventSink()
        sink.on_epic_verification_failed("epic-123", 2, ["issue-1", "issue-2"])

        mock_log.assert_called_once()
        call_args = mock_log.call_args
        assert "epic-123" in call_args[0][1]
        assert "2 unmet criteria" in call_args[0][1]
        assert "issue-1" in call_args[0][1]
        assert "issue-2" in call_args[0][1]

    @patch("src.infra.io.event_sink.log")
    def test_on_epic_verification_human_review_logs_reason(
        self, mock_log: MagicMock
    ) -> None:
        """on_epic_verification_human_review logs reason and review issue."""
        sink = ConsoleEventSink()
        sink.on_epic_verification_human_review("epic-123", "Low confidence", "review-1")

        mock_log.assert_called_once()
        call_args = mock_log.call_args
        assert "epic-123" in call_args[0][1]
        assert "human review" in call_args[0][1]
        assert "Low confidence" in call_args[0][1]
        assert "review-1" in call_args[0][1]

    @patch("src.infra.io.event_sink.log_verbose")
    def test_on_epic_remediation_created_logs_criterion(
        self, mock_log_verbose: MagicMock
    ) -> None:
        """on_epic_remediation_created logs issue id and truncated criterion."""
        sink = ConsoleEventSink()
        long_criterion = "A" * 100  # Long text to test truncation
        sink.on_epic_remediation_created("epic-123", "issue-1", long_criterion)

        mock_log_verbose.assert_called_once()
        call_args = mock_log_verbose.call_args
        assert "epic-123" in call_args[0][1]
        assert "issue-1" in call_args[0][1]
        # Should be truncated (60 chars + ...)
        assert "..." in call_args[0][1]

    @patch("src.infra.io.event_sink.log_verbose")
    def test_on_epic_remediation_created_short_criterion_not_truncated(
        self, mock_log_verbose: MagicMock
    ) -> None:
        """on_epic_remediation_created does not truncate short criterion."""
        sink = ConsoleEventSink()
        short_criterion = "Short criterion text"
        sink.on_epic_remediation_created("epic-123", "issue-1", short_criterion)

        mock_log_verbose.assert_called_once()
        call_args = mock_log_verbose.call_args
        assert short_criterion in call_args[0][1]
        # Short text should not have "..." appended unless it's cut off
        assert "..." not in call_args[0][1]

    @patch("src.infra.io.event_sink.log")
    def test_validation_started_logs_message_with_issue_id(
        self, mock_log: MagicMock
    ) -> None:
        """on_validation_started logs with issue_id parameter."""
        sink = ConsoleEventSink()
        sink.on_validation_started("agent-1", issue_id="issue-123")

        mock_log.assert_called_once()
        call_args = mock_log.call_args
        # Check the message contains "Starting validation..."
        assert "Starting validation" in call_args[0][1]
        # Check issue_id is passed as a keyword argument
        assert call_args.kwargs.get("issue_id") == "issue-123"

    @patch("src.infra.io.event_sink.log")
    def test_on_gate_started_uses_log_not_verbose(self, mock_log: MagicMock) -> None:
        """on_gate_started uses log() not log_verbose() for normal visibility."""
        sink = ConsoleEventSink()
        sink.on_gate_started("agent-1", 1, 3, issue_id="issue-123")

        # log() should be called (not log_verbose which wouldn't call log unless verbose)
        mock_log.assert_called_once()
        call_args = mock_log.call_args
        assert "Quality gate" in call_args[0][1]
        assert "attempt 1/3" in call_args[0][1]
        assert call_args.kwargs.get("issue_id") == "issue-123"

    @patch("src.infra.io.event_sink.log")
    def test_on_review_started_uses_log_not_verbose(self, mock_log: MagicMock) -> None:
        """on_review_started uses log() not log_verbose() for normal visibility."""
        sink = ConsoleEventSink()
        sink.on_review_started("agent-1", 1, 2, issue_id="issue-123")

        # log() should be called (not log_verbose which wouldn't call log unless verbose)
        mock_log.assert_called_once()
        call_args = mock_log.call_args
        assert "Review" in call_args[0][1]
        assert "attempt 1/2" in call_args[0][1]
        assert call_args.kwargs.get("issue_id") == "issue-123"

    @patch("src.infra.io.event_sink.log")
    def test_gate_passed_passes_issue_id(self, mock_log: MagicMock) -> None:
        """on_gate_passed passes issue_id to log()."""
        sink = ConsoleEventSink()
        sink.on_gate_passed("agent-1", issue_id="issue-456")

        mock_log.assert_called_once()
        assert mock_log.call_args.kwargs.get("issue_id") == "issue-456"

    @patch("src.infra.io.event_sink.log")
    def test_gate_failed_passes_issue_id(self, mock_log: MagicMock) -> None:
        """on_gate_failed passes issue_id to log()."""
        sink = ConsoleEventSink()
        sink.on_gate_failed("agent-1", 3, 3, issue_id="issue-789")

        mock_log.assert_called_once()
        assert mock_log.call_args.kwargs.get("issue_id") == "issue-789"

    @patch("src.infra.io.event_sink.log")
    def test_gate_retry_passes_issue_id(self, mock_log: MagicMock) -> None:
        """on_gate_retry passes issue_id to log()."""
        sink = ConsoleEventSink()
        sink.on_gate_retry("agent-1", 2, 3, issue_id="issue-abc")

        mock_log.assert_called_once()
        assert mock_log.call_args.kwargs.get("issue_id") == "issue-abc"

    @patch("src.infra.io.event_sink.log")
    def test_review_passed_passes_issue_id(self, mock_log: MagicMock) -> None:
        """on_review_passed passes issue_id to log()."""
        sink = ConsoleEventSink()
        sink.on_review_passed("agent-1", issue_id="issue-def")

        mock_log.assert_called_once()
        assert mock_log.call_args.kwargs.get("issue_id") == "issue-def"

    @patch("src.infra.io.event_sink.log")
    def test_review_retry_passes_issue_id(self, mock_log: MagicMock) -> None:
        """on_review_retry passes issue_id to log()."""
        sink = ConsoleEventSink()
        sink.on_review_retry("agent-1", 2, 2, error_count=5, issue_id="issue-ghi")

        mock_log.assert_called_once()
        assert mock_log.call_args.kwargs.get("issue_id") == "issue-ghi"

    @patch("src.infra.io.event_sink.log")
    def test_validation_result_failed_passes_issue_id(
        self, mock_log: MagicMock
    ) -> None:
        """on_validation_result (failed) passes issue_id to log()."""
        sink = ConsoleEventSink()
        sink.on_validation_result("agent-1", passed=False, issue_id="issue-jkl")

        mock_log.assert_called_once()
        assert mock_log.call_args.kwargs.get("issue_id") == "issue-jkl"

    @patch("src.infra.io.event_sink.log")
    def test_gate_result_passes_issue_id(self, mock_log: MagicMock) -> None:
        """on_gate_result passes issue_id to log()."""
        sink = ConsoleEventSink()
        sink.on_gate_result(
            "agent-1", passed=False, failure_reasons=["lint"], issue_id="issue-mno"
        )

        mock_log.assert_called_once()
        assert mock_log.call_args.kwargs.get("issue_id") == "issue-mno"

    @patch("src.infra.io.event_sink.log")
    def test_validation_result_passed_passes_issue_id(
        self, mock_log: MagicMock
    ) -> None:
        """on_validation_result (passed) passes issue_id to log()."""
        sink = ConsoleEventSink()
        sink.on_validation_result("agent-1", passed=True, issue_id="issue-pqr")

        mock_log.assert_called_once()
        assert mock_log.call_args.kwargs.get("issue_id") == "issue-pqr"
