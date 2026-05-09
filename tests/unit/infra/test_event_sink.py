"""Tests for event_sink module."""

from dataclasses import dataclass
from typing import Literal

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


class TestNullEventSink:
    """Tests for NullEventSink implementation."""

    def test_inherits_from_base_event_sink(self) -> None:
        """NullEventSink inherits from BaseEventSink."""
        sink = NullEventSink()
        assert isinstance(sink, BaseEventSink)

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


class TestConsoleEventSink:
    """Tests for ConsoleEventSink implementation."""

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


class TestCoderAttribute:
    """Tests for the ``coder`` span attribute (T015).

    Verifies that issue-, agent-session-, fixer-, and session-end-scoped sink
    methods accept the optional ``coder`` keyword and that absence of it does
    not break legacy call sites.
    """

    @pytest.mark.parametrize("coder", ["claude", "amp", "codex"])
    def test_base_sink_accepts_coder_on_relevant_spans(
        self, coder: Literal["claude", "amp", "codex"]
    ) -> None:
        """BaseEventSink methods accept ``coder`` for every supported value."""
        sink = BaseEventSink()

        # Agent session scope
        assert sink.on_agent_started("a-1", "i-1", coder=coder) is None
        assert (
            sink.on_agent_completed("a-1", "i-1", True, 1.0, "ok", coder=coder) is None
        )

        # Issue lifecycle scope
        assert sink.on_issue_closed("a-1", "i-1", coder=coder) is None
        assert (
            sink.on_issue_completed("a-1", "i-1", True, 1.0, "ok", coder=coder) is None
        )

        # Fixer session scope
        assert sink.on_fixer_started(1, 3, coder=coder) is None
        assert sink.on_fixer_completed("fixed", coder=coder) is None
        assert sink.on_fixer_failed("timeout", coder=coder) is None

        # Session end scope
        assert sink.on_session_end_started("i-1", coder=coder) is None
        assert sink.on_session_end_completed("i-1", "pass", coder=coder) is None
        assert sink.on_session_end_skipped("i-1", "gate_failed", coder=coder) is None

    def test_legacy_calls_without_coder_still_work(self) -> None:
        """Existing call sites that omit ``coder`` continue to work (back-compat)."""
        sink = BaseEventSink()

        assert sink.on_agent_started("a-1", "i-1") is None
        assert sink.on_agent_completed("a-1", "i-1", True, 1.0, "ok") is None
        assert sink.on_issue_closed("a-1", "i-1") is None
        assert sink.on_issue_completed("a-1", "i-1", True, 1.0, "ok") is None
        assert sink.on_fixer_started(1, 3) is None
        assert sink.on_fixer_completed("fixed") is None
        assert sink.on_fixer_failed("timeout") is None
        assert sink.on_session_end_started("i-1") is None
        assert sink.on_session_end_completed("i-1", "pass") is None
        assert sink.on_session_end_skipped("i-1", "gate_failed") is None

    @pytest.mark.parametrize("coder", ["claude", "amp", "codex"])
    def test_console_sink_renders_coder_on_per_issue_header(
        self,
        coder: Literal["claude", "amp", "codex"],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Console sink surfaces the active coder on the per-issue claim header."""
        sink = ConsoleEventSink()

        sink.on_agent_started("agent-1", "issue-42", coder=coder)

        captured = capsys.readouterr()
        assert "issue-42" in captured.out
        assert f"coder={coder}" in captured.out

    def test_console_sink_omits_coder_when_not_provided(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Without ``coder``, console output remains byte-equivalent to today."""
        sink = ConsoleEventSink()

        sink.on_agent_started("agent-1", "issue-42")

        captured = capsys.readouterr()
        assert "issue-42" in captured.out
        assert "coder=" not in captured.out

    @pytest.mark.parametrize("coder", ["claude", "amp", "codex"])
    def test_console_sink_renders_coder_on_issue_completed(
        self,
        coder: Literal["claude", "amp", "codex"],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """on_issue_completed renders the coder so the per-issue summary shows it."""
        sink = ConsoleEventSink()

        sink.on_issue_completed("agent-1", "issue-42", True, 12.5, "done", coder=coder)

        captured = capsys.readouterr()
        assert "issue-42" in captured.out
        assert f"coder={coder}" in captured.out

    @pytest.mark.parametrize("coder", ["claude", "amp", "codex"])
    def test_console_sink_renders_coder_on_fixer_started(
        self,
        coder: Literal["claude", "amp", "codex"],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """on_fixer_started renders the coder so dashboards can scope by fixer coder."""
        sink = ConsoleEventSink()

        sink.on_fixer_started(1, 3, coder=coder)

        captured = capsys.readouterr()
        assert "FIXER" in captured.out
        assert f"coder={coder}" in captured.out
