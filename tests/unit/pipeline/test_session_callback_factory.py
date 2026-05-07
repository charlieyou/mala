"""Unit tests for SessionCallbackFactory protocol adapters."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.pipeline.session_callback_factory import (
    SessionCallbackFactory,
    SessionRunContext,
)
from src.core.session_end_result import SessionEndResult, SessionEndRetryState


def _create_minimal_context() -> SessionRunContext:
    """Create a SessionRunContext with minimal lambda stubs."""
    return SessionRunContext(
        evidence_provider_getter=lambda: MagicMock(),
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


class TestReviewAdapterEmptyDiffSkip:
    """Tests for review adapter empty cumulative diff handling."""

    @pytest.mark.asyncio
    async def test_skips_review_when_issue_commits_have_no_net_tree_changes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Empty cumulative issue diffs pass without invoking external review."""
        from src.infra import git_utils as infra_git_utils

        async def fake_get_issue_commits_async(
            _repo_path: Path,
            _issue_id: str,
        ) -> list[str]:
            return ["first-commit", "revert-commit"]

        async def fake_has_tree_changes(
            _repo_path: Path,
            from_commit: str,
            to_commit: str,
        ) -> bool:
            assert from_commit == "base-commit"
            assert to_commit == "revert-commit"
            return False

        async def fake_get_commits_between_async(
            _repo_path: Path,
            from_commit: str,
            to_commit: str,
        ) -> list[str]:
            assert from_commit == "base-commit"
            assert to_commit == "revert-commit"
            return ["first-commit", "revert-commit"]

        monkeypatch.setattr(
            infra_git_utils, "get_issue_commits_async", fake_get_issue_commits_async
        )
        monkeypatch.setattr(
            infra_git_utils, "get_commits_between_async", fake_get_commits_between_async
        )
        monkeypatch.setattr(infra_git_utils, "has_tree_changes", fake_has_tree_changes)

        review_runner = MagicMock()
        review_runner.config = MagicMock()
        review_runner.run_review = AsyncMock(
            side_effect=AssertionError("external review should be skipped")
        )
        context = SessionRunContext(
            evidence_provider_getter=lambda: MagicMock(),
            evidence_check_getter=lambda: MagicMock(),
            on_session_log_path=lambda issue_id, path: None,
            on_review_log_path=lambda issue_id, path: None,
            interrupt_event_getter=lambda: None,
            get_base_sha=lambda issue_id: "base-commit",
            get_run_metadata=lambda: None,
            on_abort=lambda reason: None,
            abort_event_getter=lambda: None,
        )
        factory = SessionCallbackFactory(
            gate_async_runner=MagicMock(),
            review_runner=review_runner,
            context=context,
            event_sink=MagicMock(return_value=MagicMock()),
            repo_path=Path("/test/repo"),
            get_per_session_spec=MagicMock(return_value=None),
            is_verbose=MagicMock(return_value=False),
        )

        adapters = factory.build_adapters("test-issue")
        result = await adapters.review_runner.run_review(
            "test-issue",
            "description",
            "session-id",
            MagicMock(),
            None,
            None,
            None,
        )

        assert result.passed is True
        assert result.issues == []
        assert result.parse_error is None
        assert result.fatal_error is False

    @pytest.mark.asyncio
    async def test_falls_back_to_review_when_empty_diff_check_errors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Git diff errors do not suppress external review."""
        from src.infra import git_utils as infra_git_utils

        async def fake_get_issue_commits_async(
            _repo_path: Path,
            _issue_id: str,
        ) -> list[str]:
            return ["commit"]

        async def fake_has_tree_changes(
            _repo_path: Path,
            _from_commit: str,
            _to_commit: str,
        ) -> bool:
            raise ValueError("bad revision")

        async def fake_get_commits_between_async(
            _repo_path: Path,
            _from_commit: str,
            _to_commit: str,
        ) -> list[str]:
            return ["commit"]

        monkeypatch.setattr(
            infra_git_utils, "get_issue_commits_async", fake_get_issue_commits_async
        )
        monkeypatch.setattr(
            infra_git_utils, "get_commits_between_async", fake_get_commits_between_async
        )
        monkeypatch.setattr(infra_git_utils, "has_tree_changes", fake_has_tree_changes)

        review_result = MagicMock(
            passed=True,
            issues=[],
            parse_error=None,
            fatal_error=False,
            review_log_path=None,
        )
        review_output = MagicMock(
            result=review_result,
            session_log_path=None,
            interrupted=False,
        )
        review_runner = MagicMock()
        review_runner.config = MagicMock()
        review_runner.run_review = AsyncMock(return_value=review_output)
        context = SessionRunContext(
            evidence_provider_getter=lambda: MagicMock(),
            evidence_check_getter=lambda: MagicMock(),
            on_session_log_path=lambda issue_id, path: None,
            on_review_log_path=lambda issue_id, path: None,
            interrupt_event_getter=lambda: None,
            get_base_sha=lambda issue_id: "base-commit",
            get_run_metadata=lambda: None,
            on_abort=lambda reason: None,
            abort_event_getter=lambda: None,
        )
        factory = SessionCallbackFactory(
            gate_async_runner=MagicMock(),
            review_runner=review_runner,
            context=context,
            event_sink=MagicMock(return_value=MagicMock()),
            repo_path=Path("/test/repo"),
            get_per_session_spec=MagicMock(return_value=None),
            is_verbose=MagicMock(return_value=False),
        )

        adapters = factory.build_adapters("test-issue")
        result = await adapters.review_runner.run_review(
            "test-issue",
            "description",
            "session-id",
            MagicMock(),
            None,
            None,
            None,
        )

        assert result is review_result

    @pytest.mark.asyncio
    async def test_falls_back_to_review_when_issue_commits_are_interleaved(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Do not pre-skip if the base-to-tip range includes unrelated commits."""
        from src.infra import git_utils as infra_git_utils

        async def fake_get_issue_commits_async(
            _repo_path: Path,
            _issue_id: str,
        ) -> list[str]:
            return ["issue-commit"]

        async def fake_get_commits_between_async(
            _repo_path: Path,
            _from_commit: str,
            _to_commit: str,
        ) -> list[str]:
            return ["unrelated-commit", "issue-commit"]

        async def fake_has_tree_changes(
            _repo_path: Path,
            _from_commit: str,
            _to_commit: str,
        ) -> bool:
            raise AssertionError(
                "should not check tree changes for interleaved commits"
            )

        monkeypatch.setattr(
            infra_git_utils, "get_issue_commits_async", fake_get_issue_commits_async
        )
        monkeypatch.setattr(
            infra_git_utils, "get_commits_between_async", fake_get_commits_between_async
        )
        monkeypatch.setattr(infra_git_utils, "has_tree_changes", fake_has_tree_changes)

        review_result = MagicMock(
            passed=True,
            issues=[],
            parse_error=None,
            fatal_error=False,
            review_log_path=None,
        )
        review_output = MagicMock(
            result=review_result,
            session_log_path=None,
            interrupted=False,
        )
        review_runner = MagicMock()
        review_runner.config = MagicMock()
        review_runner.run_review = AsyncMock(return_value=review_output)
        context = SessionRunContext(
            evidence_provider_getter=lambda: MagicMock(),
            evidence_check_getter=lambda: MagicMock(),
            on_session_log_path=lambda issue_id, path: None,
            on_review_log_path=lambda issue_id, path: None,
            interrupt_event_getter=lambda: None,
            get_base_sha=lambda issue_id: "base-commit",
            get_run_metadata=lambda: None,
            on_abort=lambda reason: None,
            abort_event_getter=lambda: None,
        )
        factory = SessionCallbackFactory(
            gate_async_runner=MagicMock(),
            review_runner=review_runner,
            context=context,
            event_sink=MagicMock(return_value=MagicMock()),
            repo_path=Path("/test/repo"),
            get_per_session_spec=MagicMock(return_value=None),
            is_verbose=MagicMock(return_value=False),
        )

        adapters = factory.build_adapters("test-issue")
        result = await adapters.review_runner.run_review(
            "test-issue",
            "description",
            "session-id",
            MagicMock(),
            None,
            None,
            None,
        )

        assert result is review_result


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
