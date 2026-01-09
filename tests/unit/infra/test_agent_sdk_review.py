"""Tests for AgentSDKReviewer.

Tests the Agent SDK code reviewer implementation including:
- Empty commit list short-circuiting
- Successful review scenarios (PASS/FAIL/NEEDS_WORK)
- JSON parsing with fallbacks
- Error handling (timeout, SDK failures)
- Context file loading
- Priority conversion
- Telemetry warnings
- overrides_disabled_setting behavior
- Interrupt handling
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

from src.infra.clients.agent_sdk_review import AgentSDKReviewer
from tests.fakes.event_sink import FakeEventSink
from tests.fakes.sdk_client import FakeSDKClientFactory

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path


def _make_review_json(
    verdict: str = "PASS",
    issues: list[dict[str, Any]] | None = None,
) -> str:
    """Create a valid review JSON response."""
    return json.dumps(
        {
            "consensus_verdict": verdict,
            "aggregated_findings": issues or [],
        }
    )


def _make_issue(
    file: str = "src/test.py",
    line_start: int = 10,
    line_end: int = 12,
    priority: str | int = "P1",
    title: str = "Test finding",
    body: str = "Test body",
    reviewer: str = "agent_sdk",
) -> dict[str, Any]:
    """Create a valid issue dict."""
    return {
        "file_path": file,
        "line_start": line_start,
        "line_end": line_end,
        "priority": priority,
        "title": title,
        "body": body,
        "reviewer": reviewer,
    }


class TestEmptyDiffSkipsAgentSession:
    """Test that empty diffs short-circuit before agent session."""

    async def test_empty_commit_sha_list_skips_agent_session(
        self, tmp_path: Path
    ) -> None:
        """Empty commit SHA list [] returns PASS without running agent session."""
        factory = FakeSDKClientFactory()
        reviewer = AgentSDKReviewer(
            repo_path=tmp_path,
            review_agent_prompt="Review the code",
            sdk_client_factory=factory,
        )

        # Empty list should short-circuit without calling git diff
        result = await reviewer(commit_shas=[])

        # Should pass without issues
        assert result.passed is True
        assert result.issues == []

        # No SDK client should have been created
        assert len(factory.clients) == 0


class TestSuccessfulReview:
    """Test successful review scenarios."""

    async def test_successful_review_pass(self, tmp_path: Path) -> None:
        """Review that passes returns correct result."""
        factory = FakeSDKClientFactory()
        # Configure SDK client to return PASS verdict
        client = factory.configure_next_client(
            messages=[
                MagicMock(
                    subtype="assistant",
                    content=[{"type": "text", "text": _make_review_json("PASS")}],
                )
            ]
        )

        reviewer = AgentSDKReviewer(
            repo_path=tmp_path,
            review_agent_prompt="Review the code",
            sdk_client_factory=factory,
        )

        result = await reviewer(
            commit_shas=["abc123"], claude_session_id="test-session"
        )

        assert result.passed is True
        assert result.issues == []
        assert result.parse_error is None
        assert result.fatal_error is False
        # review_log_path should be set from SDK session
        assert result.review_log_path is not None
        assert "fake-session" in str(result.review_log_path)

        # Verify SDK client was used
        assert len(factory.clients) == 1
        assert len(client.queries) == 1

    async def test_successful_review_fail(self, tmp_path: Path) -> None:
        """Review that fails returns issues."""
        factory = FakeSDKClientFactory()
        issue = _make_issue(priority="P1", title="Critical bug")
        factory.configure_next_client(
            messages=[
                MagicMock(
                    subtype="assistant",
                    content=[
                        {"type": "text", "text": _make_review_json("FAIL", [issue])}
                    ],
                )
            ]
        )

        reviewer = AgentSDKReviewer(
            repo_path=tmp_path,
            review_agent_prompt="Review the code",
            sdk_client_factory=factory,
        )

        result = await reviewer(
            commit_shas=["abc123"], claude_session_id="test-session"
        )

        assert result.passed is False
        assert len(result.issues) == 1
        assert result.issues[0].title == "Critical bug"
        assert result.issues[0].priority == 1  # Converted from "P1"

    async def test_successful_review_needs_work(self, tmp_path: Path) -> None:
        """NEEDS_WORK verdict is treated as failure."""
        factory = FakeSDKClientFactory()
        factory.configure_next_client(
            messages=[
                MagicMock(
                    subtype="assistant",
                    content=[{"type": "text", "text": _make_review_json("NEEDS_WORK")}],
                )
            ]
        )

        reviewer = AgentSDKReviewer(
            repo_path=tmp_path,
            review_agent_prompt="Review the code",
            sdk_client_factory=factory,
        )

        result = await reviewer(
            commit_shas=["abc123"], claude_session_id="test-session"
        )

        assert result.passed is False


class TestMalformedJsonResponse:
    """Test handling of malformed JSON responses."""

    async def test_malformed_json_response(self, tmp_path: Path) -> None:
        """Malformed JSON returns parse error."""
        factory = FakeSDKClientFactory()
        factory.configure_next_client(
            messages=[
                MagicMock(
                    subtype="assistant",
                    content=[{"type": "text", "text": "not valid json at all"}],
                )
            ]
        )

        event_sink = FakeEventSink()
        reviewer = AgentSDKReviewer(
            repo_path=tmp_path,
            review_agent_prompt="Review the code",
            sdk_client_factory=factory,
            event_sink=event_sink,
        )

        result = await reviewer(
            commit_shas=["abc123"], claude_session_id="test-session"
        )

        assert result.passed is False
        assert result.parse_error is not None
        assert result.fatal_error is False

    async def test_json_in_code_block_extracted(self, tmp_path: Path) -> None:
        """JSON inside markdown code block is extracted."""
        factory = FakeSDKClientFactory()
        json_content = _make_review_json("PASS")
        wrapped = f"Here is my review:\n```json\n{json_content}\n```"
        factory.configure_next_client(
            messages=[
                MagicMock(
                    subtype="assistant",
                    content=[{"type": "text", "text": wrapped}],
                )
            ]
        )

        reviewer = AgentSDKReviewer(
            repo_path=tmp_path,
            review_agent_prompt="Review the code",
            sdk_client_factory=factory,
        )

        result = await reviewer(
            commit_shas=["abc123"], claude_session_id="test-session"
        )

        assert result.passed is True
        assert result.parse_error is None


class TestTimeoutHandling:
    """Test timeout handling."""

    async def test_timeout_handling(self, tmp_path: Path) -> None:
        """Timeout during agent session returns appropriate error."""
        factory = FakeSDKClientFactory()

        # Configure client to raise TimeoutError
        factory.configure_next_client(query_error=TimeoutError("Agent timed out"))

        event_sink = FakeEventSink()
        reviewer = AgentSDKReviewer(
            repo_path=tmp_path,
            review_agent_prompt="Review the code",
            sdk_client_factory=factory,
            event_sink=event_sink,
        )

        result = await reviewer(
            commit_shas=["abc123"],
            claude_session_id="test-session",
            timeout=10,
        )

        assert result.passed is False
        assert result.parse_error is not None
        assert (
            "timeout" in result.parse_error.lower()
            or "timed out" in result.parse_error.lower()
        )
        assert result.fatal_error is False


class TestSDKClientCreationFailure:
    """Test SDK client creation failure handling."""

    async def test_sdk_client_creation_failure(self, tmp_path: Path) -> None:
        """SDK client creation failure returns error result."""
        factory = MagicMock()
        factory.create_options.return_value = {}
        factory.create.side_effect = RuntimeError("SDK initialization failed")

        event_sink = FakeEventSink()
        reviewer = AgentSDKReviewer(
            repo_path=tmp_path,
            review_agent_prompt="Review the code",
            sdk_client_factory=factory,
            event_sink=event_sink,
        )

        result = await reviewer(
            commit_shas=["abc123"], claude_session_id="test-session"
        )

        assert result.passed is False
        assert result.parse_error is not None
        assert (
            "SDK" in result.parse_error
            or "initialization" in result.parse_error.lower()
        )
        assert result.fatal_error is False


class TestContextFileLoading:
    """Test context file loading."""

    async def test_context_file_loading(self, tmp_path: Path) -> None:
        """Context file content is included in review query."""
        # Create context file
        context_file = tmp_path / "context.txt"
        context_file.write_text("Issue: Fix the bug in authentication")

        factory = FakeSDKClientFactory()
        client = factory.configure_next_client(
            messages=[
                MagicMock(
                    subtype="assistant",
                    content=[{"type": "text", "text": _make_review_json("PASS")}],
                )
            ]
        )

        reviewer = AgentSDKReviewer(
            repo_path=tmp_path,
            review_agent_prompt="Review the code",
            sdk_client_factory=factory,
        )

        result = await reviewer(
            commit_shas=["abc123"],
            context_file=context_file,
            claude_session_id="test-session",
        )

        assert result.passed is True
        # Verify context was included in query
        assert len(client.queries) == 1
        query_text = client.queries[0][0]
        assert "authentication" in query_text


class TestPriorityConversion:
    """Test priority string/int conversion."""

    async def test_priority_conversion_string_p1(self, tmp_path: Path) -> None:
        """Priority string 'P1' is converted to int 1."""
        factory = FakeSDKClientFactory()
        issue = _make_issue(priority="P1")
        factory.configure_next_client(
            messages=[
                MagicMock(
                    subtype="assistant",
                    content=[
                        {"type": "text", "text": _make_review_json("FAIL", [issue])}
                    ],
                )
            ]
        )

        reviewer = AgentSDKReviewer(
            repo_path=tmp_path,
            review_agent_prompt="Review the code",
            sdk_client_factory=factory,
        )

        result = await reviewer(
            commit_shas=["abc123"], claude_session_id="test-session"
        )

        assert result.issues[0].priority == 1

    async def test_priority_conversion_int(self, tmp_path: Path) -> None:
        """Priority int 2 is kept as int 2."""
        factory = FakeSDKClientFactory()
        issue = _make_issue(priority=2)
        factory.configure_next_client(
            messages=[
                MagicMock(
                    subtype="assistant",
                    content=[
                        {"type": "text", "text": _make_review_json("FAIL", [issue])}
                    ],
                )
            ]
        )

        reviewer = AgentSDKReviewer(
            repo_path=tmp_path,
            review_agent_prompt="Review the code",
            sdk_client_factory=factory,
        )

        result = await reviewer(
            commit_shas=["abc123"], claude_session_id="test-session"
        )

        assert result.issues[0].priority == 2


class TestP0Priority:
    """Test P0 priority support."""

    async def test_p0_priority_string(self, tmp_path: Path) -> None:
        """Priority string 'P0' is converted to int 0."""
        factory = FakeSDKClientFactory()
        issue = _make_issue(priority="P0", title="Critical blocking issue")
        factory.configure_next_client(
            messages=[
                MagicMock(
                    subtype="assistant",
                    content=[
                        {"type": "text", "text": _make_review_json("FAIL", [issue])}
                    ],
                )
            ]
        )

        reviewer = AgentSDKReviewer(
            repo_path=tmp_path,
            review_agent_prompt="Review the code",
            sdk_client_factory=factory,
        )

        result = await reviewer(
            commit_shas=["abc123"], claude_session_id="test-session"
        )

        assert result.passed is False
        assert len(result.issues) == 1
        assert result.issues[0].priority == 0

    async def test_p0_priority_int(self, tmp_path: Path) -> None:
        """Priority int 0 is kept as int 0."""
        factory = FakeSDKClientFactory()
        issue = _make_issue(priority=0, title="Critical blocking issue")
        factory.configure_next_client(
            messages=[
                MagicMock(
                    subtype="assistant",
                    content=[
                        {"type": "text", "text": _make_review_json("FAIL", [issue])}
                    ],
                )
            ]
        )

        reviewer = AgentSDKReviewer(
            repo_path=tmp_path,
            review_agent_prompt="Review the code",
            sdk_client_factory=factory,
        )

        result = await reviewer(
            commit_shas=["abc123"], claude_session_id="test-session"
        )

        assert result.passed is False
        assert len(result.issues) == 1
        assert result.issues[0].priority == 0


class TestMissingRequiredFields:
    """Test parse error handling for missing required fields."""

    async def test_missing_consensus_verdict(self, tmp_path: Path) -> None:
        """Missing consensus_verdict returns parse_error."""
        factory = FakeSDKClientFactory()
        # Return JSON without consensus_verdict
        invalid_json = json.dumps({"aggregated_findings": []})
        factory.configure_next_client(
            messages=[
                MagicMock(
                    subtype="assistant",
                    content=[{"type": "text", "text": invalid_json}],
                )
            ]
        )

        event_sink = FakeEventSink()
        reviewer = AgentSDKReviewer(
            repo_path=tmp_path,
            review_agent_prompt="Review the code",
            sdk_client_factory=factory,
            event_sink=event_sink,
        )

        result = await reviewer(
            commit_shas=["abc123"], claude_session_id="test-session"
        )

        assert result.passed is False
        assert result.parse_error is not None
        assert "verdict" in result.parse_error.lower()

    async def test_wrong_field_names_verdict_findings(self, tmp_path: Path) -> None:
        """Using wrong field names (verdict/findings) returns parse_error."""
        factory = FakeSDKClientFactory()
        # Return JSON with wrong field names
        invalid_json = json.dumps({"verdict": "PASS", "findings": []})
        factory.configure_next_client(
            messages=[
                MagicMock(
                    subtype="assistant",
                    content=[{"type": "text", "text": invalid_json}],
                )
            ]
        )

        event_sink = FakeEventSink()
        reviewer = AgentSDKReviewer(
            repo_path=tmp_path,
            review_agent_prompt="Review the code",
            sdk_client_factory=factory,
            event_sink=event_sink,
        )

        result = await reviewer(
            commit_shas=["abc123"], claude_session_id="test-session"
        )

        # Should fail because consensus_verdict is missing (empty string)
        assert result.passed is False
        assert result.parse_error is not None
        assert "verdict" in result.parse_error.lower()

    async def test_aggregated_findings_not_a_list(self, tmp_path: Path) -> None:
        """aggregated_findings not being a list returns parse_error."""
        factory = FakeSDKClientFactory()
        invalid_json = json.dumps(
            {"consensus_verdict": "PASS", "aggregated_findings": "not a list"}
        )
        factory.configure_next_client(
            messages=[
                MagicMock(
                    subtype="assistant",
                    content=[{"type": "text", "text": invalid_json}],
                )
            ]
        )

        event_sink = FakeEventSink()
        reviewer = AgentSDKReviewer(
            repo_path=tmp_path,
            review_agent_prompt="Review the code",
            sdk_client_factory=factory,
            event_sink=event_sink,
        )

        result = await reviewer(
            commit_shas=["abc123"], claude_session_id="test-session"
        )

        assert result.passed is False
        assert result.parse_error is not None
        assert "aggregated_findings" in result.parse_error


class TestTelemetryWarning:
    """Test telemetry warning on errors."""

    async def test_telemetry_warning_on_error(self, tmp_path: Path) -> None:
        """Non-fatal errors emit on_review_warning to event sink."""
        factory = FakeSDKClientFactory()
        # Return malformed JSON to trigger warning
        factory.configure_next_client(
            messages=[
                MagicMock(
                    subtype="assistant",
                    content=[{"type": "text", "text": "invalid json response"}],
                )
            ]
        )

        event_sink = FakeEventSink()
        reviewer = AgentSDKReviewer(
            repo_path=tmp_path,
            review_agent_prompt="Review the code",
            sdk_client_factory=factory,
            event_sink=event_sink,
        )

        result = await reviewer(
            commit_shas=["abc123"], claude_session_id="test-session"
        )

        assert result.passed is False
        # Check that warning was emitted
        warnings = event_sink.get_events("review_warning")
        assert len(warnings) >= 1


class TestStructuredOutput:
    """Test structured output handling."""

    async def test_structured_output_parsed(self, tmp_path: Path) -> None:
        """Structured output in result message is parsed and used."""
        factory = FakeSDKClientFactory()
        structured = {"consensus_verdict": "PASS", "aggregated_findings": []}
        result_message = MagicMock(
            type="result",
            subtype="success",
            session_id="fake-session",
            structured_output=structured,
        )
        factory.configure_next_client(result_message=result_message)

        reviewer = AgentSDKReviewer(
            repo_path=tmp_path,
            review_agent_prompt="Review the code",
            sdk_client_factory=factory,
        )

        result = await reviewer(
            commit_shas=["abc123"], claude_session_id="test-session"
        )

        assert result.passed is True
        assert result.parse_error is None

        # Structured output format should be passed into SDK options
        assert factory.created_options, "Expected SDK options to be created"
        output_format = factory.created_options[0].get("output_format")
        assert isinstance(output_format, dict)
        assert output_format.get("type") == "json_schema"


class TestSDKFlowConfiguration:
    """Test that SDK options include correct flow identifier for subprocess logging."""

    async def test_sdk_options_include_mala_sdk_flow_reviewer(
        self, tmp_path: Path
    ) -> None:
        """SDK options should set MALA_SDK_FLOW=reviewer for subprocess logging.

        The sdk_transport.py module logs 'sdk_subprocess_spawned pid=%s pgid=%s flow=%s'
        when spawning a subprocess, using the MALA_SDK_FLOW env var for the flow value.
        This test verifies AgentSDKReviewer sets the correct flow for PID capture/monitoring.
        """
        factory = FakeSDKClientFactory()
        factory.configure_next_client(
            messages=[
                MagicMock(
                    subtype="assistant",
                    content=[{"type": "text", "text": _make_review_json("PASS")}],
                )
            ]
        )

        reviewer = AgentSDKReviewer(
            repo_path=tmp_path,
            review_agent_prompt="Review the code",
            sdk_client_factory=factory,
        )

        await reviewer(commit_shas=["abc123"], claude_session_id="test-session")

        # MALA_SDK_FLOW should be set to "reviewer" in SDK options
        # This ensures sdk_transport.py logs: sdk_subprocess_spawned pid=%s pgid=%s flow=reviewer
        assert factory.created_options, "Expected SDK options to be created"
        env = factory.created_options[0].get("env")
        assert env == {"MALA_SDK_FLOW": "reviewer"}


class TestOverridesDisabledSetting:
    """Test overrides_disabled_setting behavior."""

    def test_overrides_disabled_setting_returns_false(self, tmp_path: Path) -> None:
        """AgentSDKReviewer.overrides_disabled_setting() returns False."""
        factory = FakeSDKClientFactory()
        reviewer = AgentSDKReviewer(
            repo_path=tmp_path,
            review_agent_prompt="Review the code",
            sdk_client_factory=factory,
        )

        assert reviewer.overrides_disabled_setting() is False


class TestInterruptHandling:
    """Test interrupt event handling in AgentSDKReviewer."""

    async def test_reviewer_checks_interrupt_before_starting(
        self, tmp_path: Path
    ) -> None:
        """Reviewer should check interrupt_event before starting review."""
        factory = FakeSDKClientFactory()
        reviewer = AgentSDKReviewer(
            repo_path=tmp_path,
            review_agent_prompt="Review the code",
            sdk_client_factory=factory,
        )

        # Set interrupt event before calling
        interrupt_event = asyncio.Event()
        interrupt_event.set()

        # Should return early without calling SDK
        result = await reviewer(
            commit_shas=["abc123"],
            interrupt_event=interrupt_event,
        )

        # Should return failed result with interrupted=True (Finding 2)
        assert result.passed is False
        assert result.parse_error is None
        assert result.interrupted is True
        assert result.fatal_error is False

        # No SDK client should have been created
        assert len(factory.clients) == 0

    async def test_reviewer_returns_interrupted_when_event_set(
        self, tmp_path: Path
    ) -> None:
        """Reviewer should return appropriate result when interrupted."""
        factory = FakeSDKClientFactory()
        reviewer = AgentSDKReviewer(
            repo_path=tmp_path,
            review_agent_prompt="Review the code",
            sdk_client_factory=factory,
        )

        # Set interrupt event
        interrupt_event = asyncio.Event()
        interrupt_event.set()

        result = await reviewer(
            commit_shas=["abc123"],
            interrupt_event=interrupt_event,
        )

        # Should not run agent session
        assert len(factory.clients) == 0

        # Should return with appropriate fields (Finding 2: interrupted=True, not parse_error)
        assert result.passed is False
        assert result.issues == []
        assert result.parse_error is None
        assert result.interrupted is True

    async def test_reviewer_accepts_interrupt_event_parameter(
        self, tmp_path: Path
    ) -> None:
        """Reviewer __call__ should accept interrupt_event parameter."""
        factory = FakeSDKClientFactory()
        factory.configure_next_client(
            messages=[
                MagicMock(
                    subtype="assistant",
                    content=[{"type": "text", "text": _make_review_json("PASS")}],
                )
            ]
        )

        reviewer = AgentSDKReviewer(
            repo_path=tmp_path,
            review_agent_prompt="Review the code",
            sdk_client_factory=factory,
        )

        # Unset event should not interrupt
        interrupt_event = asyncio.Event()

        result = await reviewer(
            commit_shas=["abc123"],
            claude_session_id="test-session",
            interrupt_event=interrupt_event,
        )

        # Should complete normally
        assert result.passed is True
        assert result.interrupted is False
        assert len(factory.clients) == 1

    async def test_reviewer_checks_interrupt_during_iteration(
        self, tmp_path: Path
    ) -> None:
        """Reviewer should check interrupt_event during SDK session iteration (Finding 4)."""
        # Create an interrupt event
        interrupt_event = asyncio.Event()

        # Create a custom client that sets interrupt during iteration
        class InterruptingFakeClient:
            """Fake client that sets interrupt event during message iteration."""

            def __init__(self, interrupt_event: asyncio.Event) -> None:
                self._interrupt_event = interrupt_event
                self.queries: list[tuple[str, str | None]] = []

            async def __aenter__(self) -> "InterruptingFakeClient":  # noqa: UP037
                return self

            async def __aexit__(self, *args: object) -> None:
                pass

            async def query(self, prompt: str, session_id: str | None = None) -> None:
                self.queries.append((prompt, session_id))

            async def receive_response(self) -> AsyncIterator[object]:
                """Yield messages, setting interrupt after first one."""
                # First message
                yield MagicMock(
                    subtype="assistant",
                    content=[{"type": "text", "text": "Starting review..."}],
                )
                # Set interrupt before second message
                self._interrupt_event.set()
                # Second message (should be skipped due to interrupt check)
                yield MagicMock(
                    subtype="assistant",
                    content=[{"type": "text", "text": _make_review_json("PASS")}],
                )

        # Create a factory that returns our custom client
        custom_client = InterruptingFakeClient(interrupt_event)
        factory = MagicMock()
        factory.create_options.return_value = {}
        factory.create.return_value = custom_client

        reviewer = AgentSDKReviewer(
            repo_path=tmp_path,
            review_agent_prompt="Review the code",
            sdk_client_factory=factory,
        )

        result = await reviewer(
            commit_shas=["abc123"],
            claude_session_id="test-session",
            interrupt_event=interrupt_event,
        )

        # Should return interrupted=True because interrupt was set during iteration
        assert result.passed is False
        assert result.interrupted is True
        assert result.parse_error is None
