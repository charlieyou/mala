"""Tests for event_sink module."""

from src.event_sink import MalaEventSink, NullEventSink, RunConfig


class TestRunConfig:
    """Tests for RunConfig dataclass."""

    def test_minimal_config(self) -> None:
        """RunConfig can be created with required fields only."""
        config = RunConfig(
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
        assert config.codex_review_enabled is True
        assert config.codex_thinking_mode is None
        assert config.morph_enabled is True
        assert config.morph_disallowed_tools is None
        assert config.prioritize_wip is False
        assert config.cli_args is None

    def test_full_config(self) -> None:
        """RunConfig can be created with all optional fields."""
        config = RunConfig(
            repo_path="/tmp/repo",
            max_agents=4,
            timeout_minutes=60,
            max_issues=20,
            max_gate_retries=5,
            max_review_retries=3,
            epic_id="epic-123",
            only_ids=["issue-1", "issue-2"],
            braintrust_enabled=True,
            codex_review_enabled=False,
            codex_thinking_mode="minimal",
            morph_enabled=False,
            morph_disallowed_tools=["Edit", "Write"],
            prioritize_wip=True,
            cli_args={"verbose": True},
        )
        assert config.epic_id == "epic-123"
        assert config.only_ids == ["issue-1", "issue-2"]
        assert config.braintrust_enabled is True
        assert config.codex_review_enabled is False
        assert config.codex_thinking_mode == "minimal"
        assert config.morph_enabled is False
        assert config.morph_disallowed_tools == ["Edit", "Write"]
        assert config.prioritize_wip is True
        assert config.cli_args == {"verbose": True}

    def test_none_values_allowed(self) -> None:
        """RunConfig accepts None for optional int fields."""
        config = RunConfig(
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
        config = RunConfig(
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

    def test_can_be_called_multiple_times(self) -> None:
        """NullEventSink methods can be called repeatedly."""
        sink = NullEventSink()

        # Simulate a typical orchestrator flow
        config = RunConfig(
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
