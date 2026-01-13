"""Unit tests for SessionCallbackFactory protocol adapters."""

from pathlib import Path
from unittest.mock import MagicMock

from src.pipeline.session_callback_factory import (
    SessionCallbackFactory,
    SessionRunContext,
)
from src.core.session_end_result import SessionEndResult, SessionEndRetryState


def _create_minimal_context() -> SessionRunContext:
    """Create a SessionRunContext with minimal lambda stubs."""
    return SessionRunContext(
        log_provider_getter=lambda: MagicMock(),
        evidence_check_getter=lambda: MagicMock(),
        on_session_log_path=lambda issue_id, path: None,
        on_review_log_path=lambda issue_id, path: None,
        interrupt_event_getter=lambda: None,
        get_base_sha=lambda issue_id: None,
        get_run_metadata=lambda: None,
        on_abort=lambda reason: None,
        abort_event_getter=lambda: None,
    )


def _create_minimal_factory() -> SessionCallbackFactory:
    """Create a SessionCallbackFactory with minimal mock dependencies."""
    return SessionCallbackFactory(
        gate_async_runner=MagicMock(),
        review_runner=MagicMock(),
        context=_create_minimal_context(),
        event_sink=MagicMock(return_value=MagicMock()),
        repo_path=Path("/test/repo"),
        get_per_session_spec=MagicMock(return_value=None),
        is_verbose=MagicMock(return_value=False),
    )


class TestSessionEndCheckViaAdapter:
    """Tests for run_session_end_check via protocol adapter."""

    def test_adapters_are_present_in_build_adapters(self) -> None:
        """build_adapters returns adapters with gate_runner, review_runner, session_lifecycle."""
        factory = _create_minimal_factory()
        adapters = factory.build_adapters("test-issue")

        assert adapters.gate_runner is not None
        assert adapters.review_runner is not None
        assert adapters.session_lifecycle is not None

    async def test_returns_skipped_when_not_configured(self) -> None:
        """run_session_end_check returns skipped when session_end not configured."""
        factory = _create_minimal_factory()
        adapters = factory.build_adapters("test-issue")

        result = await adapters.gate_runner.run_session_end_check(
            "test-issue",
            Path("/test/log.txt"),
            SessionEndRetryState(),
        )

        assert isinstance(result, SessionEndResult)
        assert result.status == "skipped"
        assert result.reason == "not_configured"

    async def test_adapter_accepts_correct_signature(self) -> None:
        """Adapter accepts (issue_id, log_path, retry_state) parameters."""
        factory = _create_minimal_factory()
        adapters = factory.build_adapters("test-issue")

        retry_state = SessionEndRetryState(
            attempt=2,
            max_retries=3,
            log_offset=1024,
            previous_commit_hash="abc123",
        )

        result = await adapters.gate_runner.run_session_end_check(
            "my-issue-id",
            Path("/path/to/session.log"),
            retry_state,
        )

        # Should return skipped when not configured
        assert result.status == "skipped"
        assert result.reason == "not_configured"
