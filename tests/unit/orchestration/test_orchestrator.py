"""Unit tests for MalaOrchestrator control flow.

These tests use FakeIssueProvider to test orchestrator
state transitions without network or actual bd CLI.
"""

import asyncio
import json
import os
import re
import subprocess
import time
import uuid
from collections.abc import AsyncGenerator, Callable, Generator, Sequence
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from claude_agent_sdk.types import ResultMessage

from src.orchestration.orchestrator import (
    MalaOrchestrator,
)
from src.pipeline.issue_result import IssueResult
from src.domain.prompts import load_prompts
from src.infra.tools.env import PROMPTS_DIR
from src.infra.tools.command_runner import CommandRunner

from src.core.models import OrderPreference
from src.core.protocols.log import LogProvider
from tests.fakes.issue_provider import FakeIssueProvider, FakeIssue


@pytest.fixture
def orchestrator(
    tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
) -> MalaOrchestrator:
    """Create an orchestrator with a temporary repo path.

    Note: Tests that need to inject FakeIssueProvider should use make_orchestrator
    directly with issue_provider parameter instead of this fixture.
    """
    return make_orchestrator(
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

    def test_prompt_template_placeholders_match_format_call(self) -> None:
        """Verify prompt template placeholders match what format() provides."""
        # Extract all {placeholder} from template
        prompts = load_prompts(PROMPTS_DIR)
        placeholders = set(re.findall(r"\{(\w+)\}", prompts.implementer_prompt))

        # These are the keys passed to format() in run_implementer
        expected_keys = {
            "issue_id",
            "repo_path",
            "lock_dir",
            "agent_id",
            "lint_command",
            "format_command",
            "typecheck_command",
            "custom_commands_section",
            "test_command",
            "issue_description",
        }

        assert placeholders == expected_keys, (
            f"Mismatch: template has {placeholders}, format expects {expected_keys}"
        )


class TestSpawnAgent:
    """Test spawn_agent behavior."""

    @pytest.mark.asyncio
    async def test_adds_issue_to_failed_when_claim_fails(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """When claim fails, issue should be added to failed_issues."""
        fake_issues = FakeIssueProvider()  # Empty - claim will fail
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            issue_provider=fake_issues,
            runs_dir=runs_dir,
            lock_releaser=lambda _: 0,
        )

        result = await orchestrator.spawn_agent("unclaimed-issue")

        assert result is None
        assert "unclaimed-issue" in orchestrator.failed_issues

    @pytest.mark.asyncio
    async def test_creates_task_when_claim_succeeds(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """When claim succeeds, a task should be created."""
        fake_issues = FakeIssueProvider(
            {"claimable-issue": FakeIssue(id="claimable-issue", priority=1)}
        )
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            issue_provider=fake_issues,
            runs_dir=runs_dir,
            lock_releaser=lambda _: 0,
        )

        # Replace run_implementer to return immediately
        async def fake_run_implementer(
            issue_id: str, *, flow: str = "implementer"
        ) -> IssueResult:
            return IssueResult(
                issue_id=issue_id,
                agent_id="test-agent",
                success=True,
                summary="done",
            )

        original_run_implementer = orchestrator.run_implementer
        orchestrator.run_implementer = fake_run_implementer  # type: ignore[method-assign]
        try:
            task = await orchestrator.spawn_agent("claimable-issue")
            # spawn_agent returns the Task on success (caller is responsible for registration)
            assert task is not None
            assert isinstance(task, asyncio.Task)
            # Issue should be claimed
            assert "claimable-issue" in fake_issues.claimed
            # Await the task to ensure it completes before restoring run_implementer
            await task
        finally:
            orchestrator.run_implementer = original_run_implementer  # type: ignore[method-assign]


class TestRunOrchestrationLoop:
    """Test the main run() orchestration loop."""

    @pytest.mark.asyncio
    async def test_stops_cleanly_when_no_ready_issues(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """When no issues are ready, run() should return 0 and exit cleanly."""
        fake_issues = FakeIssueProvider()  # Empty - no issues
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=2,
            timeout_minutes=1,
            max_issues=5,
            issue_provider=fake_issues,
            runs_dir=runs_dir,
            lock_releaser=lambda _: 0,
        )

        result = await orchestrator.run()

        assert result == (0, 0)
        assert len(orchestrator._state.completed) == 0

    @pytest.mark.asyncio
    async def test_respects_max_issues_limit(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """Should stop after processing max_issues."""
        fake_issues = FakeIssueProvider(
            {
                "issue-1": FakeIssue(id="issue-1", priority=1),
                "issue-2": FakeIssue(id="issue-2", priority=2),
                "issue-3": FakeIssue(id="issue-3", priority=3),
            }
        )
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=2,
            timeout_minutes=1,
            max_issues=2,
            issue_provider=fake_issues,
            runs_dir=runs_dir,
            lock_releaser=lambda _: 0,
        )

        # Use a fake spawn that completes immediately
        spawned: list[str] = []

        original_spawn = orchestrator.spawn_agent

        async def tracking_spawn(issue_id: str) -> asyncio.Task[IssueResult] | None:
            spawned.append(issue_id)

            # Complete immediately with success
            async def work() -> IssueResult:
                return IssueResult(
                    issue_id=issue_id,
                    agent_id=f"{issue_id}-agent",
                    success=True,
                    summary="done",
                )

            return asyncio.create_task(work())

        # Replace method directly (not using patch)
        orchestrator.spawn_agent = tracking_spawn  # type: ignore[method-assign]
        try:
            await orchestrator.run()
        finally:
            orchestrator.spawn_agent = original_spawn  # type: ignore[method-assign]

        # Should have only spawned 2 issues (max_issues limit)
        assert len(spawned) == 2


class TestFailedTaskResetsIssue:
    """Test that failed tasks correctly mark issue as needing followup."""

    @pytest.mark.asyncio
    async def test_resets_issue_on_task_failure(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """When a task fails, the issue should be marked needs-followup."""
        fake_issues = FakeIssueProvider(
            {"fail-issue": FakeIssue(id="fail-issue", priority=1)}
        )
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            timeout_minutes=1,
            issue_provider=fake_issues,
            runs_dir=runs_dir,
            lock_releaser=lambda _: 0,
        )

        async def fake_run_implementer(
            issue_id: str, *, flow: str = "implementer"
        ) -> IssueResult:
            return IssueResult(
                issue_id=issue_id,
                agent_id=f"{issue_id}-agent",
                success=False,
                summary="Implementation failed",
            )

        original_run_implementer = orchestrator.run_implementer
        orchestrator.run_implementer = fake_run_implementer  # type: ignore[method-assign]
        try:
            await orchestrator.run()
        finally:
            orchestrator.run_implementer = original_run_implementer  # type: ignore[method-assign]

        # The failed issue should have been marked needs-followup
        assert len(fake_issues.followup_calls) == 1
        assert fake_issues.followup_calls[0][0] == "fail-issue"
        # And added to failed_issues set
        assert "fail-issue" in orchestrator.failed_issues

    @pytest.mark.asyncio
    async def test_does_not_reset_successful_issue(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """Successful issues should not be reset."""
        fake_issues = FakeIssueProvider(
            {"success-issue": FakeIssue(id="success-issue", priority=1)}
        )
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            timeout_minutes=1,
            issue_provider=fake_issues,
            runs_dir=runs_dir,
            lock_releaser=lambda _: 0,
        )

        async def fake_run_implementer(
            issue_id: str, *, flow: str = "implementer"
        ) -> IssueResult:
            return IssueResult(
                issue_id=issue_id,
                agent_id=f"{issue_id}-agent",
                success=True,
                summary="Completed successfully",
            )

        original_run_implementer = orchestrator.run_implementer
        orchestrator.run_implementer = fake_run_implementer  # type: ignore[method-assign]
        try:
            await orchestrator.run()
        finally:
            orchestrator.run_implementer = original_run_implementer  # type: ignore[method-assign]

        # Issue should be closed, not reset
        assert "success-issue" in fake_issues.closed
        # No reset should have been called
        assert len(fake_issues.reset_calls) == 0
        # And not in failed_issues
        assert "success-issue" not in orchestrator.failed_issues


class TestOrchestratorInitialization:
    """Test orchestrator initialization."""

    def test_timeout_conversion(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """Timeout minutes should be converted to seconds."""
        orch = make_orchestrator(repo_path=tmp_path, timeout_minutes=15)
        assert orch.timeout_seconds == 15 * 60

    def test_default_values(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """Default values should be set correctly."""
        orch = make_orchestrator(repo_path=tmp_path)
        assert orch.max_agents is None  # Unlimited by default
        # Default timeout of 60 min protects against hung MCP subprocesses
        assert orch.timeout_seconds == 60 * 60
        assert orch.max_issues is None

    def test_repo_path_resolved(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """Repo path should be resolved to absolute."""
        # Use tmp_path subdirectory to avoid loading mala.yaml from cwd
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        orch = make_orchestrator(repo_path=subdir)
        assert orch.repo_path.is_absolute()
        assert orch.repo_path == subdir

    def test_focus_default_true(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """Focus parameter should default to True."""
        orch = make_orchestrator(repo_path=tmp_path)
        assert orch.focus is True

    def test_focus_explicit_false(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """Focus parameter can be explicitly set to False."""
        orch = make_orchestrator(repo_path=tmp_path, focus=False)
        assert orch.focus is False

    def test_telemetry_provider_default_null(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """Default telemetry provider is NullTelemetryProvider."""
        from src.infra.telemetry import NullTelemetryProvider

        orch = make_orchestrator(repo_path=tmp_path)
        assert isinstance(orch.telemetry_provider, NullTelemetryProvider)

    def test_telemetry_provider_injection(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """Telemetry provider can be injected for testing."""
        from src.infra.telemetry import NullTelemetryProvider

        custom_provider = NullTelemetryProvider()
        orch = make_orchestrator(
            repo_path=tmp_path,
            telemetry_provider=custom_provider,
        )
        # Injected provider should be used, not the default
        assert orch.telemetry_provider is custom_provider


class TestOrchestratorWithEpicId:
    """Test orchestrator with epic_id parameter."""

    def test_epic_id_stored(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """epic_id parameter should be stored on orchestrator."""
        orch = make_orchestrator(repo_path=tmp_path, epic_id="test-epic")
        assert orch.epic_id == "test-epic"

    def test_epic_id_defaults_to_none(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """epic_id should default to None."""
        orch = make_orchestrator(repo_path=tmp_path)
        assert orch.epic_id is None


class TestOrchestratorEvidenceCheckIntegration:
    """Test quality gate integration in orchestrator run flow.

    These tests verify that success=False triggers followup marking
    and success=True triggers issue closure.
    """

    @pytest.mark.asyncio
    async def test_marks_needs_followup_on_gate_failure(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """When run_implementer returns success=False, issue should be marked needs-followup."""
        fake_issues = FakeIssueProvider(
            {"issue-123": FakeIssue(id="issue-123", priority=1)}
        )
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            max_issues=1,
            issue_provider=fake_issues,
            runs_dir=runs_dir,
            lock_releaser=lambda _: 0,
        )

        log_path = tmp_path / "issue-123.jsonl"
        log_path.touch()

        async def mock_run_implementer(
            issue_id: str, *, flow: str = "implementer"
        ) -> IssueResult:
            return IssueResult(
                issue_id=issue_id,
                agent_id=f"{issue_id}-agent",
                success=False,  # Failure triggers followup
                summary="Quality gate failed: No commit with bd-issue-123 found",
                session_log_path=log_path,
            )

        # Direct method replacement instead of patch.object
        original_run_implementer = orchestrator.run_implementer
        orchestrator.run_implementer = mock_run_implementer  # type: ignore[method-assign]
        try:
            await orchestrator.run()
        finally:
            orchestrator.run_implementer = original_run_implementer  # type: ignore[method-assign]

        # Should have been marked as needs-followup via FakeIssueProvider
        assert len(fake_issues.followup_calls) == 1
        assert fake_issues.followup_calls[0][0] == "issue-123"

    @pytest.mark.asyncio
    async def test_success_only_when_gate_passes(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """When run_implementer returns success=True, issue should count as success."""
        fake_issues = FakeIssueProvider(
            {"issue-pass": FakeIssue(id="issue-pass", priority=1)}
        )
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            max_issues=1,
            issue_provider=fake_issues,
            runs_dir=runs_dir,
            lock_releaser=lambda _: 0,
        )

        log_path = tmp_path / "issue-pass.jsonl"
        log_path.touch()

        async def mock_run_implementer(
            issue_id: str, *, flow: str = "implementer"
        ) -> IssueResult:
            return IssueResult(
                issue_id=issue_id,
                agent_id=f"{issue_id}-agent",
                success=True,  # Success triggers closure
                summary="Completed successfully",
                session_log_path=log_path,
            )

        # Direct method replacement instead of patch.object
        original_run_implementer = orchestrator.run_implementer
        orchestrator.run_implementer = mock_run_implementer  # type: ignore[method-assign]
        try:
            success_count, _total = await orchestrator.run()
        finally:
            orchestrator.run_implementer = original_run_implementer  # type: ignore[method-assign]

        assert success_count == 1
        assert any(r.issue_id == "issue-pass" for r in orchestrator._state.completed)


class TestMissingLogFile:
    """Test run_implementer behavior when log file never appears."""

    # Use short log file wait timeout for tests (0.5s instead of 60s)
    TEST_LOG_FILE_WAIT_TIMEOUT = 0.5

    @pytest.fixture(autouse=True)
    def short_log_timeout(self) -> Generator[None, None, None]:
        """Patch log file wait timeout to avoid 60s waits in tests."""
        from src.pipeline.agent_session_runner import AgentSessionConfig

        original_init = AgentSessionConfig.__init__

        def patched_init(
            self: AgentSessionConfig, *args: object, **kwargs: object
        ) -> None:
            # Set short timeout default if not explicitly provided
            if "log_file_wait_timeout" not in kwargs:
                kwargs["log_file_wait_timeout"] = 0.5
            original_init(self, *args, **kwargs)  # type: ignore[arg-type]

        with patch.object(AgentSessionConfig, "__init__", patched_init):
            yield

    @pytest.mark.asyncio
    async def test_exits_quickly_when_log_file_missing(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """run_implementer should exit within bounded wait when log file missing."""

        orchestrator = make_orchestrator(
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
        async def mock_receive_response() -> AsyncGenerator[ResultMessage, None]:
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
            patch("claude_agent_sdk.ClaudeSDKClient", return_value=mock_client),
            patch(
                "src.orchestration.orchestrator.get_git_branch_async",
                return_value="main",
            ),
            patch(
                "src.orchestration.orchestrator.get_git_commit_async",
                return_value="abc123",
            ),
            # TracedAgentExecution removed - telemetry_provider injected via constructor
        ):
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
        max_expected = (
            self.TEST_LOG_FILE_WAIT_TIMEOUT + 5
        )  # 5s buffer for test overhead
        assert elapsed < max_expected, (
            f"run_implementer took {elapsed:.1f}s, expected < {max_expected}s"
        )

    @pytest.mark.asyncio
    async def test_summary_indicates_missing_log(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """Failure summary should clearly indicate the log file was missing."""

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            timeout_minutes=5,
        )

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.query = AsyncMock()

        async def mock_receive_response() -> AsyncGenerator[ResultMessage, None]:
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
            patch("claude_agent_sdk.ClaudeSDKClient", return_value=mock_client),
            patch(
                "src.orchestration.orchestrator.get_git_branch_async",
                return_value="main",
            ),
            patch(
                "src.orchestration.orchestrator.get_git_commit_async",
                return_value="abc123",
            ),
            # TracedAgentExecution removed - telemetry_provider injected via constructor
        ):
            result = await orchestrator.run_implementer("issue-xyz")

        # Summary should mention session log and the timeout
        assert "Session log missing after timeout:" in result.summary


class TestAgentEnvInheritance:
    """Test that agent environment inherits from os.environ.

    Verifies mala-w8w.1: Agent tool calls must see inherited env vars
    plus lock overrides (PATH/LOCK_DIR/AGENT_ID/REPO_NAMESPACE).
    """

    @pytest.fixture(autouse=True)
    def short_log_timeout(self) -> Generator[None, None, None]:
        """Patch log file wait timeout to avoid 60s waits in tests."""
        from src.pipeline.agent_session_runner import AgentSessionConfig

        original_init = AgentSessionConfig.__init__

        def patched_init(
            self: AgentSessionConfig, *args: object, **kwargs: object
        ) -> None:
            # Set short timeout default if not explicitly provided
            if "log_file_wait_timeout" not in kwargs:
                kwargs["log_file_wait_timeout"] = 0.5
            original_init(self, *args, **kwargs)  # type: ignore[arg-type]

        with patch.object(AgentSessionConfig, "__init__", patched_init):
            yield

    @pytest.mark.asyncio
    async def test_agent_env_includes_os_environ(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """Agent environment should include inherited env vars plus lock overrides."""
        orchestrator = make_orchestrator(repo_path=tmp_path, max_agents=1)

        captured_env: dict[str, str] | None = None

        # Mock ClaudeSDKClient to capture the env parameter
        type(MagicMock()).__init__

        class MockClient:
            def __init__(self, options: object) -> None:
                nonlocal captured_env
                # Capture the env from options
                captured_env = getattr(options, "env", None)

            async def __aenter__(self) -> "MockClient":
                return self

            async def __aexit__(self, *args: object) -> None:
                pass

            async def query(self, prompt: str) -> None:
                pass

            async def receive_response(self) -> AsyncGenerator[ResultMessage, None]:
                yield ResultMessage(
                    subtype="result",
                    session_id="test-session",
                    result="Done",
                    duration_ms=100,
                    duration_api_ms=80,
                    is_error=False,
                    num_turns=1,
                    total_cost_usd=0.01,
                    usage=None,
                )

        # Set a test env var that should be inherited
        test_env_var = f"TEST_INHERIT_{uuid.uuid4().hex[:8]}"
        os.environ[test_env_var] = "inherited_value"

        try:
            with (
                patch("claude_agent_sdk.ClaudeSDKClient", MockClient),
                patch(
                    "src.orchestration.orchestrator.get_git_branch_async",
                    return_value="main",
                ),
                patch(
                    "src.orchestration.orchestrator.get_git_commit_async",
                    return_value="abc123",
                ),
                # TracedAgentExecution removed - telemetry_provider injected via constructor
            ):
                await orchestrator.run_implementer("test-issue")

            # Verify the captured env includes our test var
            assert captured_env is not None, "env was not passed to ClaudeSDKClient"
            assert test_env_var in captured_env, (
                f"os.environ var {test_env_var} not inherited into agent_env"
            )
            assert captured_env[test_env_var] == "inherited_value"

            # Verify lock-related vars are also present (overrides)
            assert "PATH" in captured_env
            assert "LOCK_DIR" in captured_env
            assert "AGENT_ID" in captured_env
            assert "REPO_NAMESPACE" in captured_env
        finally:
            del os.environ[test_env_var]


class TestLockDirNestedCreation:
    """Test that LOCK_DIR creation handles nested paths.

    Verifies mala-w8w.1: Orchestrator startup must succeed when MALA_LOCK_DIR
    points to a nested, non-existent directory (mkdir uses parents=True).
    """

    @pytest.mark.asyncio
    async def test_lock_dir_created_with_parents(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """LOCK_DIR.mkdir should use parents=True for nested directories."""
        # Create a nested path that doesn't exist
        nested_lock_dir = tmp_path / "deeply" / "nested" / "lock" / "dir"
        assert not nested_lock_dir.exists()

        orchestrator = make_orchestrator(repo_path=tmp_path, max_agents=1)

        # Mock get_ready_async to return empty list (no issues to process)
        async def mock_get_ready_async(
            exclude_ids: set[str] | None = None,
            epic_id: str | None = None,
            only_ids: list[str] | None = None,
            suppress_warn_ids: set[str] | None = None,
            include_wip: bool = False,
            focus: bool = True,
            orphans_only: bool = False,
            order_preference: OrderPreference = OrderPreference.FOCUS,
        ) -> list[str]:
            return []

        # Patch LOCK_DIR to point to our nested path
        with (
            patch.object(
                orchestrator.beads, "get_ready_async", side_effect=mock_get_ready_async
            ),
            patch(
                "src.orchestration.orchestrator.get_lock_dir",
                return_value=nested_lock_dir,
            ),
            patch(
                "src.orchestration.orchestrator.get_runs_dir",
                return_value=tmp_path / "runs",
            ),
            patch("src.orchestration.orchestrator.release_run_locks"),
        ):
            # This should not raise even though parent dirs don't exist
            await orchestrator.run()

        # The nested directory should now exist
        assert nested_lock_dir.exists(), (
            "LOCK_DIR.mkdir should create nested directories with parents=True"
        )


class TestGateFlowSequencing:
    """Test Gate 1-4 sequencing with check_with_resolution.

    Uses FakeIssueProvider for behavioral assertions on issue state changes.
    """

    @pytest.mark.asyncio
    async def test_no_op_resolution_skips_per_session_validation(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """No-op resolution should skip Gate 2/3 (commit + validation evidence)."""
        fake_issues = FakeIssueProvider(
            {"issue-noop": FakeIssue(id="issue-noop", priority=1)}
        )
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            max_issues=1,
            issue_provider=fake_issues,
            runs_dir=runs_dir,
            lock_releaser=lambda _: 0,
        )

        log_path = tmp_path / "issue-noop.jsonl"
        log_path.touch()

        async def mock_run_implementer(
            issue_id: str, *, flow: str = "implementer"
        ) -> IssueResult:
            return IssueResult(
                issue_id=issue_id,
                agent_id=f"{issue_id}-agent",
                success=True,
                summary="No changes needed - already done",
                session_log_path=log_path,
            )

        # Direct method replacement
        original = orchestrator.run_implementer
        orchestrator.run_implementer = mock_run_implementer  # type: ignore[method-assign]
        try:
            success_count, _total = await orchestrator.run()
        finally:
            orchestrator.run_implementer = original  # type: ignore[method-assign]

        # Assert on FakeIssueProvider state
        assert success_count == 1
        assert "issue-noop" in fake_issues.closed

    @pytest.mark.asyncio
    async def test_obsolete_resolution_skips_per_session_validation(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """Obsolete resolution should skip Gate 2/3 (commit + validation evidence)."""
        fake_issues = FakeIssueProvider(
            {"issue-obsolete": FakeIssue(id="issue-obsolete", priority=1)}
        )
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            max_issues=1,
            issue_provider=fake_issues,
            runs_dir=runs_dir,
            lock_releaser=lambda _: 0,
        )

        log_path = tmp_path / "issue-obsolete.jsonl"
        log_path.touch()

        async def mock_run_implementer(
            issue_id: str, *, flow: str = "implementer"
        ) -> IssueResult:
            return IssueResult(
                issue_id=issue_id,
                agent_id=f"{issue_id}-agent",
                success=True,
                summary="Issue obsolete - no longer relevant",
                session_log_path=log_path,
            )

        # Direct method replacement
        original = orchestrator.run_implementer
        orchestrator.run_implementer = mock_run_implementer  # type: ignore[method-assign]
        try:
            success_count, _total = await orchestrator.run()
        finally:
            orchestrator.run_implementer = original  # type: ignore[method-assign]

        # Assert on FakeIssueProvider state
        assert success_count == 1
        assert "issue-obsolete" in fake_issues.closed


class TestRetryExhaustion:
    """Test retry exhaustion and failure handling.

    Uses FakeIssueProvider for behavioral assertions on followup state.
    """

    @pytest.mark.asyncio
    async def test_gate_retry_exhaustion_marks_failed(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """When gate retries are exhausted, issue should be marked failed."""
        fake_issues = FakeIssueProvider(
            {"issue-retry-fail": FakeIssue(id="issue-retry-fail", priority=1)}
        )
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            max_issues=1,
            max_gate_retries=2,
            issue_provider=fake_issues,
            runs_dir=runs_dir,
            lock_releaser=lambda _: 0,
        )

        log_path = tmp_path / "issue-retry-fail.jsonl"
        log_path.touch()

        async def mock_run_implementer(
            issue_id: str, *, flow: str = "implementer"
        ) -> IssueResult:
            return IssueResult(
                issue_id=issue_id,
                agent_id=f"{issue_id}-agent",
                success=False,
                summary="Quality gate failed: Missing validation commands",
                gate_attempts=2,
                session_log_path=log_path,
            )

        # Direct method replacement
        original = orchestrator.run_implementer
        orchestrator.run_implementer = mock_run_implementer  # type: ignore[method-assign]
        try:
            success_count, _total = await orchestrator.run()
        finally:
            orchestrator.run_implementer = original  # type: ignore[method-assign]

        assert success_count == 0
        # Assert on FakeIssueProvider state
        assert len(fake_issues.followup_calls) == 1
        assert fake_issues.followup_calls[0][0] == "issue-retry-fail"
        assert "issue-retry-fail" in orchestrator.failed_issues

    @pytest.mark.asyncio
    async def test_no_progress_stops_retries_early(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """When no progress is detected, retries should stop early."""
        fake_issues = FakeIssueProvider(
            {"issue-no-progress": FakeIssue(id="issue-no-progress", priority=1)}
        )
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            max_issues=1,
            max_gate_retries=5,  # Many retries available
            issue_provider=fake_issues,
            runs_dir=runs_dir,
            lock_releaser=lambda _: 0,
        )

        log_path = tmp_path / "issue-no-progress.jsonl"
        log_path.touch()

        async def mock_run_implementer(
            issue_id: str, *, flow: str = "implementer"
        ) -> IssueResult:
            return IssueResult(
                issue_id=issue_id,
                agent_id=f"{issue_id}-agent",
                success=False,
                summary="Quality gate failed: No progress",
                gate_attempts=2,  # Only 2 attempts despite 5 max
                session_log_path=log_path,
            )

        # Direct method replacement
        original = orchestrator.run_implementer
        orchestrator.run_implementer = mock_run_implementer  # type: ignore[method-assign]
        try:
            await orchestrator.run()
        finally:
            orchestrator.run_implementer = original  # type: ignore[method-assign]

        # Should have failed due to no progress
        assert "issue-no-progress" in orchestrator.failed_issues


class TestDisableValidationsRespected:
    """Test that disable-validations flags are respected."""

    def test_orchestrator_stores_disable_validations(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """Orchestrator should store disable_validations parameter."""
        disable_set = {"post-validate", "coverage"}
        orch = make_orchestrator(
            repo_path=tmp_path,
            disable_validations=disable_set,
        )
        assert orch.disable_validations == disable_set

    def test_orchestrator_default_disable_validations_is_none(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """Default disable_validations should be None."""
        orch = make_orchestrator(repo_path=tmp_path)
        assert orch.disable_validations is None


class TestGlobalValidation:
    """Test global validation after all issues complete.

    Uses FakeIssueProvider for behavioral assertions on close/followup state.
    """

    @pytest.mark.asyncio
    async def test_run_returns_non_zero_exit_on_failure(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """Run should indicate failure when issues fail."""
        fake_issues = FakeIssueProvider(
            {"failing-issue": FakeIssue(id="failing-issue", priority=1)}
        )
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            max_issues=1,
            issue_provider=fake_issues,
            runs_dir=runs_dir,
            lock_releaser=lambda _: 0,
        )

        async def mock_run_implementer(
            issue_id: str, *, flow: str = "implementer"
        ) -> IssueResult:
            return IssueResult(
                issue_id=issue_id,
                agent_id=f"{issue_id}-agent",
                success=False,
                summary="Implementation failed",
            )

        # Direct method replacement
        original = orchestrator.run_implementer
        orchestrator.run_implementer = mock_run_implementer  # type: ignore[method-assign]
        try:
            success_count, _total = await orchestrator.run()
        finally:
            orchestrator.run_implementer = original  # type: ignore[method-assign]

        assert success_count == 0
        assert _total == 1

    @pytest.mark.asyncio
    async def test_issues_closed_only_after_gate_passes(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """Issues should only be closed when quality gate passes."""
        fake_issues = FakeIssueProvider(
            {"issue-pass": FakeIssue(id="issue-pass", priority=1)}
        )
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            max_issues=1,
            issue_provider=fake_issues,
            runs_dir=runs_dir,
            lock_releaser=lambda _: 0,
        )

        log_path = tmp_path / "issue-pass.jsonl"
        log_path.touch()

        async def mock_run_implementer(
            issue_id: str, *, flow: str = "implementer"
        ) -> IssueResult:
            return IssueResult(
                issue_id=issue_id,
                agent_id=f"{issue_id}-agent",
                success=True,  # Gate passed
                summary="Completed successfully",
                session_log_path=log_path,
            )

        # Direct method replacement
        original = orchestrator.run_implementer
        orchestrator.run_implementer = mock_run_implementer  # type: ignore[method-assign]
        try:
            success_count, _total = await orchestrator.run()
        finally:
            orchestrator.run_implementer = original  # type: ignore[method-assign]

        assert success_count == 1
        # Assert on FakeIssueProvider state
        assert "issue-pass" in fake_issues.closed

    @pytest.mark.asyncio
    async def test_failed_issue_not_closed(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """Failed issues should not be closed, only marked needs-followup."""
        fake_issues = FakeIssueProvider(
            {"issue-fail": FakeIssue(id="issue-fail", priority=1)}
        )
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            max_issues=1,
            issue_provider=fake_issues,
            runs_dir=runs_dir,
            lock_releaser=lambda _: 0,
        )

        log_path = tmp_path / "issue-fail.jsonl"
        log_path.touch()

        async def mock_run_implementer(
            issue_id: str, *, flow: str = "implementer"
        ) -> IssueResult:
            return IssueResult(
                issue_id=issue_id,
                agent_id=f"{issue_id}-agent",
                success=False,  # Gate failed
                summary="Quality gate failed: Missing commit",
                session_log_path=log_path,
            )

        # Direct method replacement
        original = orchestrator.run_implementer
        orchestrator.run_implementer = mock_run_implementer  # type: ignore[method-assign]
        try:
            await orchestrator.run()
        finally:
            orchestrator.run_implementer = original  # type: ignore[method-assign]

        # Issue should NOT be closed - assert on FakeIssueProvider state
        assert "issue-fail" not in fake_issues.closed
        # But should be marked needs-followup
        assert len(fake_issues.followup_calls) == 1
        assert fake_issues.followup_calls[0][0] == "issue-fail"


class TestValidationResultMetadata:
    """Test validation metadata population from gate evidence.

    Verifies mala-3qp.8: Gate and metadata share the same validation result.
    Uses FakeIssueProvider for issue state and captures RunMetadata for assertions.
    """

    @pytest.mark.asyncio
    async def test_validation_result_populated_from_gate_evidence(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """Validation metadata should be derived from gate result's validation_evidence."""
        from src.infra.io.log_output.run_metadata import IssueRun, RunMetadata
        from src.domain.evidence_check import (
            CommandKind,
            GateResult,
            ValidationEvidence,
        )

        fake_issues = FakeIssueProvider(
            {"issue-with-validation": FakeIssue(id="issue-with-validation", priority=1)}
        )
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            max_issues=1,
            issue_provider=fake_issues,
            runs_dir=runs_dir,
            lock_releaser=lambda _: 0,
        )

        # Track recorded issues via direct method replacement
        recorded_issues: list[IssueRun] = []
        original_record = RunMetadata.record_issue

        def capture_record(self: RunMetadata, issue: IssueRun) -> None:
            recorded_issues.append(issue)
            original_record(self, issue)

        # Create gate result with validation evidence
        evidence = ValidationEvidence()
        evidence.commands_ran[CommandKind.TEST] = True
        evidence.commands_ran[CommandKind.LINT] = True
        evidence.commands_ran[CommandKind.FORMAT] = True
        evidence.commands_ran[CommandKind.TYPECHECK] = True
        evidence.failed_commands = []

        gate_result = GateResult(
            passed=True,
            failure_reasons=[],
            commit_hash="abc1234",
            validation_evidence=evidence,
        )

        async def mock_run_implementer(
            issue_id: str, *, flow: str = "implementer"
        ) -> IssueResult:
            orchestrator.async_gate_runner.last_gate_results[issue_id] = gate_result
            return IssueResult(
                issue_id=issue_id,
                agent_id=f"{issue_id}-agent",
                success=True,
                summary="Completed successfully",
            )

        # Direct method replacements
        original_run = orchestrator.run_implementer
        orchestrator.run_implementer = mock_run_implementer  # type: ignore[method-assign]
        RunMetadata.record_issue = capture_record  # type: ignore[method-assign]
        try:
            await orchestrator.run()
        finally:
            orchestrator.run_implementer = original_run  # type: ignore[method-assign]
            RunMetadata.record_issue = original_record  # type: ignore[method-assign]

        # Verify an issue was recorded
        assert len(recorded_issues) == 1
        issue_run = recorded_issues[0]
        assert issue_run.issue_id == "issue-with-validation"

        # Validation field should be populated from gate evidence
        assert issue_run.validation is not None
        assert issue_run.validation.passed is True
        assert set(issue_run.validation.commands_run) == {
            "test",
            "lint",
            "format",
            "typecheck",
        }
        assert issue_run.validation.commands_failed == []

    @pytest.mark.asyncio
    async def test_gate_and_metadata_share_same_validation_evidence(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """Gate decisions and metadata should derive from the same validation result."""
        from src.infra.io.log_output.run_metadata import IssueRun, RunMetadata
        from src.domain.evidence_check import (
            CommandKind,
            GateResult,
            ValidationEvidence,
        )

        fake_issues = FakeIssueProvider(
            {
                "issue-failed-validation": FakeIssue(
                    id="issue-failed-validation", priority=1
                )
            }
        )
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            max_issues=1,
            issue_provider=fake_issues,
            runs_dir=runs_dir,
            lock_releaser=lambda _: 0,
        )

        # Track recorded issues
        recorded_issues: list[IssueRun] = []
        original_record = RunMetadata.record_issue

        def capture_record(self: RunMetadata, issue: IssueRun) -> None:
            recorded_issues.append(issue)
            original_record(self, issue)

        # Create gate result with specific evidence
        evidence = ValidationEvidence()
        evidence.commands_ran[CommandKind.TEST] = True
        evidence.commands_ran[CommandKind.LINT] = True
        evidence.commands_ran[CommandKind.FORMAT] = True
        evidence.commands_ran[CommandKind.TYPECHECK] = True
        evidence.failed_commands = ["ruff check"]  # One command failed

        gate_result = GateResult(
            passed=False,  # Gate failed due to validation failure
            failure_reasons=["Validation commands failed (non-zero exit): ruff check"],
            commit_hash="abc1234",
            validation_evidence=evidence,
        )

        async def mock_run_implementer(
            issue_id: str, *, flow: str = "implementer"
        ) -> IssueResult:
            orchestrator.async_gate_runner.last_gate_results[issue_id] = gate_result
            return IssueResult(
                issue_id=issue_id,
                agent_id=f"{issue_id}-agent",
                success=False,  # Failed because gate failed
                summary="Quality gate failed: Validation commands failed",
            )

        # Direct method replacements
        original_run = orchestrator.run_implementer
        orchestrator.run_implementer = mock_run_implementer  # type: ignore[method-assign]
        RunMetadata.record_issue = capture_record  # type: ignore[method-assign]
        try:
            await orchestrator.run()
        finally:
            orchestrator.run_implementer = original_run  # type: ignore[method-assign]
            RunMetadata.record_issue = original_record  # type: ignore[method-assign]

        # Verify an issue was recorded
        assert len(recorded_issues) == 1
        issue_run = recorded_issues[0]
        assert issue_run.issue_id == "issue-failed-validation"

        # Quality gate result should match gate evidence
        assert issue_run.evidence_check is not None
        assert issue_run.evidence_check.passed is False
        assert "ruff check" in issue_run.evidence_check.failure_reasons[0]

        # Validation metadata should derive from same evidence
        assert issue_run.validation is not None
        assert issue_run.validation.passed is False
        assert set(issue_run.validation.commands_run) == {
            "test",
            "lint",
            "format",
            "typecheck",
        }
        assert issue_run.validation.commands_failed == ["ruff check"]


class TestResolutionRecordingInMetadata:
    """Test resolution outcome is recorded in IssueRun metadata.

    Verifies mala-yg9.1.1: When check_with_resolution returns a resolution
    (NO_CHANGE or OBSOLETE), it should be propagated through IssueResult
    to IssueRun and persisted via RunMetadata.

    Uses FakeIssueProvider for issue state and captures RunMetadata for assertions.
    """

    @pytest.mark.asyncio
    async def test_no_change_resolution_recorded_in_issue_run(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """ISSUE_NO_CHANGE resolution should be recorded in IssueRun metadata."""
        from src.infra.io.log_output.run_metadata import IssueRun, RunMetadata
        from src.domain.validation.spec import IssueResolution, ResolutionOutcome

        fake_issues = FakeIssueProvider(
            {"issue-no-change": FakeIssue(id="issue-no-change", priority=1)}
        )
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            max_issues=1,
            issue_provider=fake_issues,
            runs_dir=runs_dir,
            lock_releaser=lambda _: 0,
        )

        # Track recorded issues
        recorded_issues: list[IssueRun] = []
        original_record = RunMetadata.record_issue

        def capture_record(self: RunMetadata, issue: IssueRun) -> None:
            recorded_issues.append(issue)
            original_record(self, issue)

        # Create resolution that should be propagated
        no_change_resolution = IssueResolution(
            outcome=ResolutionOutcome.NO_CHANGE,
            rationale="Already implemented by another agent",
        )

        log_path = tmp_path / "issue-no-change.jsonl"
        log_path.touch()

        async def mock_run_implementer(
            issue_id: str, *, flow: str = "implementer"
        ) -> IssueResult:
            return IssueResult(
                issue_id=issue_id,
                agent_id=f"{issue_id}-agent",
                success=True,
                summary="No changes needed - already done",
                resolution=no_change_resolution,
                session_log_path=log_path,
            )

        # Direct method replacements
        original_run = orchestrator.run_implementer
        orchestrator.run_implementer = mock_run_implementer  # type: ignore[method-assign]
        RunMetadata.record_issue = capture_record  # type: ignore[method-assign]
        try:
            success_count, total = await orchestrator.run()
        finally:
            orchestrator.run_implementer = original_run  # type: ignore[method-assign]
            RunMetadata.record_issue = original_record  # type: ignore[method-assign]

        # Verify issue was successful
        assert success_count == 1
        assert total == 1

        # Verify resolution was recorded in IssueRun
        assert len(recorded_issues) == 1
        issue_run = recorded_issues[0]
        assert issue_run.issue_id == "issue-no-change"
        assert issue_run.resolution is not None
        assert issue_run.resolution.outcome == ResolutionOutcome.NO_CHANGE
        assert issue_run.resolution.rationale == "Already implemented by another agent"

    @pytest.mark.asyncio
    async def test_obsolete_resolution_recorded_in_issue_run(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """ISSUE_OBSOLETE resolution should be recorded in IssueRun metadata."""
        from src.infra.io.log_output.run_metadata import IssueRun, RunMetadata
        from src.domain.validation.spec import IssueResolution, ResolutionOutcome

        fake_issues = FakeIssueProvider(
            {"issue-obsolete": FakeIssue(id="issue-obsolete", priority=1)}
        )
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            max_issues=1,
            issue_provider=fake_issues,
            runs_dir=runs_dir,
            lock_releaser=lambda _: 0,
        )

        # Track recorded issues
        recorded_issues: list[IssueRun] = []
        original_record = RunMetadata.record_issue

        def capture_record(self: RunMetadata, issue: IssueRun) -> None:
            recorded_issues.append(issue)
            original_record(self, issue)

        # Create obsolete resolution
        obsolete_resolution = IssueResolution(
            outcome=ResolutionOutcome.OBSOLETE,
            rationale="Feature removed in refactoring",
        )

        log_path = tmp_path / "issue-obsolete.jsonl"
        log_path.touch()

        async def mock_run_implementer(
            issue_id: str, *, flow: str = "implementer"
        ) -> IssueResult:
            return IssueResult(
                issue_id=issue_id,
                agent_id=f"{issue_id}-agent",
                success=True,
                summary="Issue obsolete - no longer relevant",
                resolution=obsolete_resolution,
                session_log_path=log_path,
            )

        # Direct method replacements
        original_run = orchestrator.run_implementer
        orchestrator.run_implementer = mock_run_implementer  # type: ignore[method-assign]
        RunMetadata.record_issue = capture_record  # type: ignore[method-assign]
        try:
            success_count, total = await orchestrator.run()
        finally:
            orchestrator.run_implementer = original_run  # type: ignore[method-assign]
            RunMetadata.record_issue = original_record  # type: ignore[method-assign]

        # Verify issue was successful
        assert success_count == 1
        assert total == 1

        # Verify resolution was recorded in IssueRun
        assert len(recorded_issues) == 1
        issue_run = recorded_issues[0]
        assert issue_run.issue_id == "issue-obsolete"
        assert issue_run.resolution is not None
        assert issue_run.resolution.outcome == ResolutionOutcome.OBSOLETE
        assert issue_run.resolution.rationale == "Feature removed in refactoring"

    @pytest.mark.asyncio
    async def test_normal_success_has_no_resolution(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """Normal success (with commit) should have no resolution marker."""
        from src.infra.io.log_output.run_metadata import IssueRun, RunMetadata

        fake_issues = FakeIssueProvider(
            {"issue-normal": FakeIssue(id="issue-normal", priority=1)}
        )
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            max_issues=1,
            issue_provider=fake_issues,
            runs_dir=runs_dir,
            lock_releaser=lambda _: 0,
        )

        # Track recorded issues
        recorded_issues: list[IssueRun] = []
        original_record = RunMetadata.record_issue

        def capture_record(self: RunMetadata, issue: IssueRun) -> None:
            recorded_issues.append(issue)
            original_record(self, issue)

        log_path = tmp_path / "issue-normal.jsonl"
        log_path.touch()

        async def mock_run_implementer(
            issue_id: str, *, flow: str = "implementer"
        ) -> IssueResult:
            return IssueResult(
                issue_id=issue_id,
                agent_id=f"{issue_id}-agent",
                success=True,
                summary="Implemented feature successfully",
                resolution=None,  # No resolution for normal commits
                session_log_path=log_path,
            )

        # Direct method replacements
        original_run = orchestrator.run_implementer
        orchestrator.run_implementer = mock_run_implementer  # type: ignore[method-assign]
        RunMetadata.record_issue = capture_record  # type: ignore[method-assign]
        try:
            success_count, total = await orchestrator.run()
        finally:
            orchestrator.run_implementer = original_run  # type: ignore[method-assign]
            RunMetadata.record_issue = original_record  # type: ignore[method-assign]

        # Verify issue was successful
        assert success_count == 1
        assert total == 1

        # Verify NO resolution in IssueRun (normal commit flow)
        assert len(recorded_issues) == 1
        issue_run = recorded_issues[0]
        assert issue_run.issue_id == "issue-normal"
        assert issue_run.resolution is None


class TestEpicClosureAfterChildCompletion:
    """Test that epics are closed immediately after their last child completes.

    Verifies mala-0k7: Epics should close after each agent completion, not just
    at run end, so other agents can see updated epic status during the run.

    Note: These tests require epic_verifier which is only created with BeadsClient.
    Since FakeIssueProvider doesn't wire up epic_verifier, tests that need
    verify_and_close_epic must use the default orchestrator with patching.
    """

    @pytest.mark.asyncio
    async def test_epic_closure_called_after_issue_closes(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """verify_and_close_epic should be called for the affected epic after issue closes."""
        # Use default orchestrator (not FakeIssueProvider) to get epic_verifier
        orchestrator = make_orchestrator(repo_path=tmp_path, max_agents=1)

        epic_closure_calls: list[str] = []

        from src.core.models import EpicVerificationResult

        async def mock_verify_and_close_epic(
            epic_id: str,
            human_override: bool = False,
        ) -> EpicVerificationResult:
            epic_closure_calls.append(epic_id)
            return EpicVerificationResult(
                verified_count=1,
                passed_count=1,
                failed_count=0,
                verdicts={},
                remediation_issues_created=[],
            )

        async def mock_run_implementer(
            issue_id: str, *, flow: str = "implementer"
        ) -> IssueResult:
            log_path = tmp_path / f"{issue_id}.jsonl"
            log_path.touch()
            orchestrator._state.active_session_log_paths[issue_id] = log_path
            return IssueResult(
                issue_id=issue_id,
                agent_id=f"{issue_id}-agent",
                success=True,
                summary="Completed successfully",
            )

        first_call = True

        async def mock_get_ready_async(
            exclude_ids: set[str] | None = None,
            epic_id: str | None = None,
            only_ids: list[str] | None = None,
            suppress_warn_ids: set[str] | None = None,
            include_wip: bool = False,
            focus: bool = True,
            orphans_only: bool = False,
            order_preference: OrderPreference = OrderPreference.FOCUS,
        ) -> list[str]:
            nonlocal first_call
            if first_call:
                first_call = False
                return ["child-issue"]
            return []

        async def mock_claim_async(issue_id: str) -> bool:
            return True

        async def mock_get_parent_epic_async(issue_id: str) -> str | None:
            return "parent-epic" if issue_id == "child-issue" else None

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
            patch.object(orchestrator.beads, "close_async", return_value=True),
            patch.object(
                orchestrator.beads,
                "get_parent_epic_async",
                side_effect=mock_get_parent_epic_async,
            ),
            patch.object(
                orchestrator.epic_verifier,
                "verify_and_close_epic",
                side_effect=mock_verify_and_close_epic,
            ),
            patch.object(orchestrator.beads, "commit_issues_async", return_value=True),
            patch(
                "src.orchestration.orchestrator.get_lock_dir", return_value=MagicMock()
            ),
            patch("src.orchestration.orchestrator.get_runs_dir", return_value=tmp_path),
            patch("src.orchestration.orchestrator.release_run_locks"),
            patch("subprocess.run", return_value=make_subprocess_result()),
        ):
            await orchestrator.run()

        # Epic verification should have been called once for the parent epic
        assert len(epic_closure_calls) == 1
        assert epic_closure_calls[0] == "parent-epic"

    @pytest.mark.asyncio
    async def test_epic_closure_not_called_when_issue_fails(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """close_eligible_epics_async should NOT be called when issue fails."""
        # This test uses FakeIssueProvider since it doesn't need epic_verifier
        fake_issues = FakeIssueProvider(
            {"failing-child": FakeIssue(id="failing-child", priority=1)}
        )
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            max_issues=1,
            issue_provider=fake_issues,
            runs_dir=runs_dir,
            lock_releaser=lambda _: 0,
        )

        epic_closure_calls: list[str] = []

        async def mock_close_eligible_epics_async() -> bool:
            epic_closure_calls.append("called")
            return True

        log_path = tmp_path / "failing-child.jsonl"
        log_path.touch()

        async def mock_run_implementer(
            issue_id: str, *, flow: str = "implementer"
        ) -> IssueResult:
            return IssueResult(
                issue_id=issue_id,
                agent_id=f"{issue_id}-agent",
                success=False,  # Gate failed
                summary="Quality gate failed: Missing commit",
                session_log_path=log_path,
            )

        # Direct method replacements
        original_run = orchestrator.run_implementer
        original_close = orchestrator.beads.close_eligible_epics_async
        orchestrator.run_implementer = mock_run_implementer  # type: ignore[method-assign]
        orchestrator.beads.close_eligible_epics_async = mock_close_eligible_epics_async  # type: ignore[method-assign]
        try:
            await orchestrator.run()
        finally:
            orchestrator.run_implementer = original_run  # type: ignore[method-assign]
            orchestrator.beads.close_eligible_epics_async = original_close  # type: ignore[method-assign]

        # Epic closure should NOT have been called since issue failed
        assert len(epic_closure_calls) == 0

    @pytest.mark.asyncio
    async def test_multiple_issues_same_epic_no_duplicate_verification(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """Multiple issues from same epic should only trigger one verification."""
        # Use default orchestrator (not FakeIssueProvider) to get epic_verifier
        orchestrator = make_orchestrator(repo_path=tmp_path, max_agents=2)

        epic_closure_calls: list[str] = []
        issues_processed: list[str] = []

        from src.core.models import EpicVerificationResult

        async def mock_verify_and_close_epic(
            epic_id: str,
            human_override: bool = False,
        ) -> EpicVerificationResult:
            epic_closure_calls.append(epic_id)
            return EpicVerificationResult(
                verified_count=1,
                passed_count=1,
                failed_count=0,
                verdicts={},
                remediation_issues_created=[],
            )

        async def mock_run_implementer(
            issue_id: str, *, flow: str = "implementer"
        ) -> IssueResult:
            issues_processed.append(issue_id)
            log_path = tmp_path / f"{issue_id}.jsonl"
            log_path.touch()
            orchestrator._state.active_session_log_paths[issue_id] = log_path
            return IssueResult(
                issue_id=issue_id,
                agent_id=f"{issue_id}-agent",
                success=True,
                summary="Completed successfully",
            )

        call_count = 0

        async def mock_get_ready_async(
            exclude_ids: set[str] | None = None,
            epic_id: str | None = None,
            only_ids: list[str] | None = None,
            suppress_warn_ids: set[str] | None = None,
            include_wip: bool = False,
            focus: bool = True,
            orphans_only: bool = False,
            order_preference: OrderPreference = OrderPreference.FOCUS,
        ) -> list[str]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ["child-1", "child-2"]
            return []

        async def mock_claim_async(issue_id: str) -> bool:
            return True

        async def mock_get_parent_epic_async(issue_id: str) -> str | None:
            if issue_id in ("child-1", "child-2"):
                return "same-parent-epic"
            return None

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
            patch.object(orchestrator.beads, "close_async", return_value=True),
            patch.object(
                orchestrator.beads,
                "get_parent_epic_async",
                side_effect=mock_get_parent_epic_async,
            ),
            patch.object(
                orchestrator.epic_verifier,
                "verify_and_close_epic",
                side_effect=mock_verify_and_close_epic,
            ),
            patch.object(orchestrator.beads, "commit_issues_async", return_value=True),
            patch(
                "src.orchestration.orchestrator.get_lock_dir", return_value=MagicMock()
            ),
            patch("src.orchestration.orchestrator.get_runs_dir", return_value=tmp_path),
            patch("src.orchestration.orchestrator.release_run_locks"),
            patch("subprocess.run", return_value=make_subprocess_result()),
        ):
            await orchestrator.run()

        # Both issues should have been processed
        assert len(issues_processed) == 2
        # Epic verification should only be called once (duplicate prevention)
        assert len(epic_closure_calls) == 1
        assert epic_closure_calls[0] == "same-parent-epic"

    @pytest.mark.asyncio
    async def test_multiple_issues_different_epics_each_verified(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """Issues from different epics should each trigger verification for their epic."""
        # Use default orchestrator (not FakeIssueProvider) to get epic_verifier
        orchestrator = make_orchestrator(repo_path=tmp_path, max_agents=2)

        epic_closure_calls: list[str] = []
        issues_processed: list[str] = []

        from src.core.models import EpicVerificationResult

        async def mock_verify_and_close_epic(
            epic_id: str,
            human_override: bool = False,
        ) -> EpicVerificationResult:
            epic_closure_calls.append(epic_id)
            return EpicVerificationResult(
                verified_count=1,
                passed_count=1,
                failed_count=0,
                verdicts={},
                remediation_issues_created=[],
            )

        async def mock_run_implementer(
            issue_id: str, *, flow: str = "implementer"
        ) -> IssueResult:
            issues_processed.append(issue_id)
            log_path = tmp_path / f"{issue_id}.jsonl"
            log_path.touch()
            orchestrator._state.active_session_log_paths[issue_id] = log_path
            return IssueResult(
                issue_id=issue_id,
                agent_id=f"{issue_id}-agent",
                success=True,
                summary="Completed successfully",
            )

        call_count = 0

        async def mock_get_ready_async(
            exclude_ids: set[str] | None = None,
            epic_id: str | None = None,
            only_ids: list[str] | None = None,
            suppress_warn_ids: set[str] | None = None,
            include_wip: bool = False,
            focus: bool = True,
            orphans_only: bool = False,
            order_preference: OrderPreference = OrderPreference.FOCUS,
        ) -> list[str]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ["child-1", "child-2"]
            return []

        async def mock_claim_async(issue_id: str) -> bool:
            return True

        async def mock_get_parent_epic_async(issue_id: str) -> str | None:
            if issue_id == "child-1":
                return "epic-1"
            elif issue_id == "child-2":
                return "epic-2"
            return None

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
            patch.object(orchestrator.beads, "close_async", return_value=True),
            patch.object(
                orchestrator.beads,
                "get_parent_epic_async",
                side_effect=mock_get_parent_epic_async,
            ),
            patch.object(
                orchestrator.epic_verifier,
                "verify_and_close_epic",
                side_effect=mock_verify_and_close_epic,
            ),
            patch.object(orchestrator.beads, "commit_issues_async", return_value=True),
            patch(
                "src.orchestration.orchestrator.get_lock_dir", return_value=MagicMock()
            ),
            patch("src.orchestration.orchestrator.get_runs_dir", return_value=tmp_path),
            patch("src.orchestration.orchestrator.release_run_locks"),
            patch("subprocess.run", return_value=make_subprocess_result()),
        ):
            await orchestrator.run()

        # Both issues should have been processed
        assert len(issues_processed) == 2
        # Epic verification should be called for each unique parent epic
        assert len(epic_closure_calls) == 2
        assert set(epic_closure_calls) == {"epic-1", "epic-2"}


class TestEvidenceCheckAsync:
    """Test that quality gate checks are non-blocking."""

    @pytest.mark.asyncio
    async def test_run_evidence_check_uses_to_thread(
        self, orchestrator: MalaOrchestrator, tmp_path: Path
    ) -> None:
        """Quality gate should use asyncio.to_thread to avoid blocking the event loop."""
        from src.domain.evidence_check import GateResult

        # Create a mock session log
        log_path = tmp_path / "session.jsonl"
        log_path.write_text("{}\n")

        # Create a mock RetryState (now imported from lifecycle)
        from src.domain.lifecycle import RetryState

        retry_state = RetryState(baseline_timestamp=int(time.time()))

        # Create mock GateResult
        mock_gate_result = GateResult(
            passed=True,
            failure_reasons=[],
            commit_hash="abc123",
            validation_evidence=None,
        )

        # Track whether to_thread was called
        to_thread_called = False
        original_func_captured = None

        async def mock_to_thread(
            func: object, *args: object, **kwargs: object
        ) -> tuple[GateResult, int]:
            nonlocal to_thread_called, original_func_captured
            to_thread_called = True
            original_func_captured = func
            # Return mock result
            return (mock_gate_result, 0)

        with (
            patch(
                "src.pipeline.gate_runner.asyncio.to_thread",
                side_effect=mock_to_thread,
            ),
        ):
            result, offset = await orchestrator.async_gate_runner.run_gate_async(
                "test-issue", log_path, retry_state
            )

        assert to_thread_called, (
            "asyncio.to_thread should be called for non-blocking execution"
        )
        assert result.passed is True
        assert offset == 0


class TestFailedRunEvidenceCheckEvidence:
    """Test that failed runs record evidence_check evidence in IssueRun metadata.

    When a run fails (result.success=False), the orchestrator should still
    parse and record validation evidence and failure reasons in the evidence_check
    field of IssueRun. This enables troubleshooting and follow-up triage.
    """

    @pytest.mark.asyncio
    async def test_failed_run_records_evidence_check_evidence(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """Failed run should populate evidence_check with evidence and failure reasons."""
        from src.infra.io.log_output.run_metadata import IssueRun, RunMetadata

        orchestrator = make_orchestrator(repo_path=tmp_path, max_agents=1)

        # Track recorded issues
        recorded_issues: list[IssueRun] = []
        original_record = RunMetadata.record_issue

        def capture_record(self: RunMetadata, issue: IssueRun) -> None:
            recorded_issues.append(issue)
            original_record(self, issue)

        async def mock_run_implementer(
            issue_id: str, *, flow: str = "implementer"
        ) -> IssueResult:
            # Create log file with validation evidence
            log_path = tmp_path / f"{issue_id}.jsonl"
            log_content = json.dumps(
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
            log_path.write_text(log_content + "\n")
            orchestrator._state.active_session_log_paths[issue_id] = log_path
            return IssueResult(
                issue_id=issue_id,
                agent_id=f"{issue_id}-agent",
                success=False,  # Run failed
                summary="Quality gate failed: No commit with bd-issue-fail found; Missing ruff check",
                session_log_path=log_path,
            )

        first_call = True

        async def mock_get_ready_async(
            exclude_ids: set[str] | None = None,
            epic_id: str | None = None,
            only_ids: list[str] | None = None,
            suppress_warn_ids: set[str] | None = None,
            include_wip: bool = False,
            focus: bool = True,
            orphans_only: bool = False,
            order_preference: OrderPreference = OrderPreference.FOCUS,
        ) -> list[str]:
            nonlocal first_call
            if first_call:
                first_call = False
                return ["issue-fail"]
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
            patch.object(orchestrator.beads, "close_async", return_value=False),
            patch.object(orchestrator.beads, "commit_issues_async", return_value=True),
            patch.object(
                orchestrator.beads, "close_eligible_epics_async", return_value=False
            ),
            patch.object(
                orchestrator.beads, "mark_needs_followup_async", return_value=True
            ),
            patch.object(RunMetadata, "record_issue", capture_record),
            patch(
                "src.orchestration.orchestrator.get_lock_dir", return_value=MagicMock()
            ),
            patch("src.orchestration.orchestrator.get_runs_dir", return_value=tmp_path),
            patch("src.orchestration.orchestrator.release_run_locks"),
            # Mock subprocess to return no commit found
            patch("subprocess.run", return_value=make_subprocess_result()),
        ):
            await orchestrator.run()

        # Verify an issue was recorded
        assert len(recorded_issues) == 1
        issue_run = recorded_issues[0]
        assert issue_run.issue_id == "issue-fail"
        assert issue_run.status == "failed"

        # CRITICAL: evidence_check should be populated for failed runs
        assert issue_run.evidence_check is not None, (
            "evidence_check should be populated for failed runs with logs"
        )
        assert issue_run.evidence_check.passed is False
        # Evidence should be parsed from the log using spec-driven keys (CommandKind.value)
        # Note: empty evidence is valid if no validation commands were detected
        assert isinstance(issue_run.evidence_check.evidence, dict)
        # commit_found should always be present (added by orchestrator)
        assert "commit_found" in issue_run.evidence_check.evidence
        # Failure reasons should be extracted from summary
        assert len(issue_run.evidence_check.failure_reasons) > 0


def _make_mock_log_provider(log_file: Path) -> object:
    """Create a mock LogProvider that returns the given log file."""
    from collections.abc import Iterator

    from src.infra.io.session_log_parser import JsonlEntry

    class MockLogProvider:
        def get_log_path(self, repo_path: Path, session_id: str) -> Path:
            return log_file

        def iter_events(self, log_path: Path, offset: int = 0) -> Iterator[JsonlEntry]:
            return iter([])

        def get_end_offset(self, log_path: Path, start_offset: int = 0) -> int:
            return log_path.stat().st_size if log_path.exists() else start_offset

    return MockLogProvider()


class TestReviewUsesIssueCommits:
    """Tests that external review scopes to commits for the active issue.

    When multiple agents are working in parallel, we should only review
    commits created for the current issue (bd-<issue_id>: prefix), not
    unrelated commits from other agents.
    """

    @pytest.mark.asyncio
    async def test_review_scopes_to_issue_commits(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """Review should use the issue's commit list, not unrelated commits."""
        from src.infra.clients.review_output_parser import ReviewResult
        from src.domain.evidence_check import GateResult

        # Create a fake log file
        log_dir = tmp_path / ".claude" / "projects" / tmp_path.name
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "test-session.jsonl"
        log_file.write_text('{"type": "result"}\n')

        # Create mala.yaml for build_validation_spec
        (tmp_path / "mala.yaml").write_text("preset: python-uv\n")

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            timeout_minutes=1,
            log_provider=_make_mock_log_provider(log_file),  # type: ignore[arg-type]
        )

        captured_commit_lists: list[Sequence[str] | None] = []

        class MockCodeReviewer:
            def overrides_disabled_setting(self) -> bool:
                """Return True; test fakes override the disabled setting."""
                return True

            async def __call__(
                self,
                context_file: Path | None = None,
                timeout: int = 300,
                claude_session_id: str | None = None,
                author_context: str | None = None,
                *,
                commit_shas: Sequence[str],
                interrupt_event: asyncio.Event | None = None,
            ) -> ReviewResult:
                captured_commit_lists.append(commit_shas)
                return ReviewResult(
                    passed=True,
                    issues=[],
                    parse_error=None,
                    fatal_error=False,
                    review_log_path=None,
                )

        mock_reviewer = MockCodeReviewer()

        # Simulate: agent made commits for this issue, but HEAD includes other changes.
        issue_commits = ["issue_commit_abc123", "issue_commit_def456"]
        current_head = "current_head_xyz456"

        # Mock the quality gate to return a passing result with the agent's commit.
        # Gate must pass for review to be triggered (see lifecycle.py:270-284).
        mock_gate_result = GateResult(
            passed=True,  # Gate passes, triggering review
            failure_reasons=[],
            commit_hash=issue_commits[-1],
        )

        async def mock_get_git_commit_async(cwd: Path, timeout: float = 5.0) -> str:
            return current_head  # Always return current HEAD

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.query = AsyncMock()

        async def mock_receive_response() -> AsyncGenerator[ResultMessage, None]:
            yield ResultMessage(
                subtype="result",
                session_id="test-session",
                result="Commit: agent_commit_abc123\nDone implementing",
                duration_ms=1000,
                duration_api_ms=800,
                is_error=False,
                num_turns=1,
                total_cost_usd=0.01,
                usage=None,
            )

        mock_client.receive_response = mock_receive_response

        # Mock the quality gate to return our specific gate result
        mock_evidence_check = MagicMock()
        mock_evidence_check.check_with_resolution = MagicMock(
            return_value=mock_gate_result
        )
        mock_evidence_check.get_log_end_offset = MagicMock(return_value=0)
        mock_evidence_check.check_no_progress = MagicMock(return_value=False)
        mock_evidence_check.parse_validation_evidence_with_spec = MagicMock(
            return_value=MagicMock(
                pytest_ran=True,
                ruff_check_ran=True,
                ruff_format_ran=True,
                ty_check_ran=True,
            )
        )

        with (
            patch("claude_agent_sdk.ClaudeSDKClient", return_value=mock_client),
            patch(
                "src.orchestration.orchestrator.get_git_branch_async",
                return_value="main",
            ),
            patch(
                "src.orchestration.orchestrator.get_git_commit_async",
                side_effect=mock_get_git_commit_async,
            ),
            patch(
                "src.infra.git_utils.get_issue_commits_async",
                return_value=issue_commits,
            ),
            patch.object(orchestrator.review_runner, "code_reviewer", mock_reviewer),
            patch.object(orchestrator, "evidence_check", mock_evidence_check),
            patch.object(orchestrator.gate_runner, "gate_checker", mock_evidence_check),
            patch.object(
                orchestrator.beads,
                "get_issue_description_async",
                return_value="Test issue",
            ),
            patch.object(orchestrator, "_is_review_enabled", return_value=True),
            patch.object(orchestrator._session_config, "review_enabled", True),
        ):
            await orchestrator.run_implementer("test-issue")

        # The key assertion: code review should receive the issue commit list
        assert captured_commit_lists, "Code review should have been called"
        assert captured_commit_lists[0] == issue_commits


class TestRunSync:
    """Test run_sync() method for synchronous usage."""

    def test_run_sync_from_sync_context(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """run_sync() should work when called from sync code."""
        fake_issues = FakeIssueProvider()
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_issues=0,
            issue_provider=fake_issues,
            runs_dir=runs_dir,
            lock_releaser=lambda _: 0,
        )

        success_count, total = orchestrator.run_sync()

        assert success_count == 0
        assert total == 0

    def test_run_sync_from_async_context_raises(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """run_sync() should raise RuntimeError when called from async context."""
        fake_issues = FakeIssueProvider()
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            issue_provider=fake_issues,
            runs_dir=runs_dir,
            lock_releaser=lambda _: 0,
        )

        async def call_run_sync_from_async() -> None:
            # This should raise because we're already in an async context
            orchestrator.run_sync()

        with pytest.raises(RuntimeError) as exc_info:
            asyncio.run(call_run_sync_from_async())

        assert "run_sync() cannot be called from within an async context" in str(
            exc_info.value
        )
        assert "await orchestrator.run()" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_run_works_in_async_context(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """run() should work when awaited from async context."""
        fake_issues = FakeIssueProvider()
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_issues=0,
            issue_provider=fake_issues,
            runs_dir=runs_dir,
            lock_releaser=lambda _: 0,
        )

        success_count, total = await orchestrator.run()

        assert success_count == 0
        assert total == 0


class TestEventSinkIntegration:
    """Tests for event sink integration with orchestrator run lifecycle."""

    @pytest.mark.asyncio
    async def test_run_emits_none_ready_event(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """on_no_more_issues is emitted with none_ready when no issues available.

        Verifies that on_no_more_issues is called with 'none_ready' when
        get_ready returns empty and max_issues is not reached.
        """
        from src.infra.io.base_sink import NullEventSink

        # Create a tracking sink that records calls
        class TrackingSink(NullEventSink):
            def __init__(self) -> None:
                self.calls: list[tuple[str, tuple]] = []

            def on_ready_issues(self, issue_ids: list[str]) -> None:
                self.calls.append(("on_ready_issues", (issue_ids,)))

            def on_waiting_for_agents(self, count: int) -> None:
                self.calls.append(("on_waiting_for_agents", (count,)))

            def on_no_more_issues(self, reason: str) -> None:
                self.calls.append(("on_no_more_issues", (reason,)))

        tracking_sink = TrackingSink()
        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_issues=None,  # No limit, so none_ready path will be taken
            event_sink=tracking_sink,
        )

        with (
            patch.object(
                orchestrator.beads,
                "get_ready_async",
                new_callable=AsyncMock,
                return_value=[],  # No ready issues
            ),
            patch(
                "src.orchestration.orchestrator.get_lock_dir",
                return_value=tmp_path / "locks",
            ),
            patch(
                "src.orchestration.orchestrator.get_runs_dir",
                return_value=tmp_path / "runs",
            ),
            patch("src.orchestration.orchestrator.write_run_marker"),
            patch("src.orchestration.orchestrator.remove_run_marker"),
        ):
            (tmp_path / "locks").mkdir(exist_ok=True)
            (tmp_path / "runs").mkdir(exist_ok=True)

            await orchestrator.run()

        # Verify on_no_more_issues was called with "none_ready"
        no_more_calls = [c for c in tracking_sink.calls if c[0] == "on_no_more_issues"]
        assert len(no_more_calls) >= 1, "on_no_more_issues should be called"
        assert "none_ready" in no_more_calls[0][1][0]

    @pytest.mark.asyncio
    async def test_run_emits_limit_reached_event(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """on_no_more_issues is called with limit_reached when max_issues=0."""
        from src.infra.io.base_sink import NullEventSink

        class TrackingSink(NullEventSink):
            def __init__(self) -> None:
                self.no_more_reasons: list[str] = []

            def on_no_more_issues(self, reason: str) -> None:
                self.no_more_reasons.append(reason)

        tracking_sink = TrackingSink()
        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_issues=0,  # Will trigger limit_reached path immediately
            event_sink=tracking_sink,
        )

        with (
            patch.object(
                orchestrator.beads,
                "get_ready_async",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "src.orchestration.orchestrator.get_lock_dir",
                return_value=tmp_path / "locks",
            ),
            patch(
                "src.orchestration.orchestrator.get_runs_dir",
                return_value=tmp_path / "runs",
            ),
            patch("src.orchestration.orchestrator.write_run_marker"),
            patch("src.orchestration.orchestrator.remove_run_marker"),
        ):
            (tmp_path / "locks").mkdir(exist_ok=True)
            (tmp_path / "runs").mkdir(exist_ok=True)

            await orchestrator.run()

        # With max_issues=0, limit_reached is True immediately
        # When no active_tasks, on_no_more_issues should be called
        assert len(tracking_sink.no_more_reasons) == 1
        assert "limit_reached" in tracking_sink.no_more_reasons[0]

    @pytest.mark.asyncio
    async def test_event_sink_defaults_to_console_sink(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """Event sink defaults to ConsoleEventSink when not specified."""
        from src.infra.io.console_sink import ConsoleEventSink

        orchestrator = make_orchestrator(repo_path=tmp_path)

        assert isinstance(orchestrator.event_sink, ConsoleEventSink)


class TestOrchestratorFactory:
    """Tests for create_orchestrator() factory function."""

    def test_create_orchestrator_with_minimal_config(self, tmp_path: Path) -> None:
        """create_orchestrator works with just repo_path."""
        from src.orchestration.factory import create_orchestrator
        from src.orchestration.types import OrchestratorConfig

        config = OrchestratorConfig(repo_path=tmp_path)
        orchestrator = create_orchestrator(config)

        assert orchestrator.repo_path == tmp_path
        assert orchestrator.max_agents is None
        assert orchestrator.max_issues is None

    def test_create_orchestrator_invalid_config_fails(self, tmp_path: Path) -> None:
        """create_orchestrator fails fast on invalid mala.yaml."""
        from src.domain.validation.config import ConfigError
        from src.orchestration.factory import create_orchestrator
        from src.orchestration.types import OrchestratorConfig

        (tmp_path / "mala.yaml").write_text("preset: python-uv\ninvalid_field: 1\n")

        config = OrchestratorConfig(repo_path=tmp_path)
        with pytest.raises(ConfigError, match="Unknown field"):
            create_orchestrator(config)

    def test_create_orchestrator_with_full_config(self, tmp_path: Path) -> None:
        """create_orchestrator respects all config options."""
        from src.orchestration.factory import create_orchestrator
        from src.orchestration.types import OrchestratorConfig

        config = OrchestratorConfig(
            repo_path=tmp_path,
            max_agents=4,
            timeout_minutes=30,
            max_issues=10,
            epic_id="epic-123",
            only_ids=["issue-1", "issue-2"],
            max_gate_retries=5,
            max_review_retries=2,
            disable_validations={"coverage"},
            include_wip=True,
            focus=False,
        )
        orchestrator = create_orchestrator(config)

        assert orchestrator.repo_path == tmp_path
        assert orchestrator.max_agents == 4
        assert orchestrator.timeout_seconds == 30 * 60
        assert orchestrator.max_issues == 10
        assert orchestrator.epic_id == "epic-123"
        assert orchestrator.only_ids == ["issue-1", "issue-2"]
        assert orchestrator.max_gate_retries == 5
        assert orchestrator.max_review_retries == 2
        assert orchestrator.include_wip is True
        assert orchestrator.focus is False

    def test_create_orchestrator_with_custom_mala_config(self, tmp_path: Path) -> None:
        """create_orchestrator uses provided MalaConfig."""
        from dataclasses import replace

        from src.infra.io.config import MalaConfig
        from src.orchestration.factory import create_orchestrator
        from src.orchestration.types import OrchestratorConfig

        config = OrchestratorConfig(repo_path=tmp_path)
        mala_config = MalaConfig.from_env(validate=False)
        # Override a value
        mala_config = replace(mala_config, review_timeout=999)

        orchestrator = create_orchestrator(config, mala_config=mala_config)

        # Verify mala_config was used
        assert orchestrator._mala_config.review_timeout == 999

    def test_create_orchestrator_with_custom_dependencies(self, tmp_path: Path) -> None:
        """create_orchestrator uses provided dependencies."""
        from src.infra.io.base_sink import NullEventSink
        from src.orchestration.factory import create_orchestrator
        from src.orchestration.types import OrchestratorConfig, OrchestratorDependencies

        custom_sink = NullEventSink()
        deps = OrchestratorDependencies(event_sink=custom_sink)

        config = OrchestratorConfig(repo_path=tmp_path)
        orchestrator = create_orchestrator(config, deps=deps)

        assert orchestrator.event_sink is custom_sink

    def test_create_orchestrator_timeout_defaults_to_60(self, tmp_path: Path) -> None:
        """Default timeout is 60 minutes when not specified."""
        from src.orchestration.factory import create_orchestrator
        from src.orchestration.types import (
            DEFAULT_AGENT_TIMEOUT_MINUTES,
            OrchestratorConfig,
        )

        config = OrchestratorConfig(repo_path=tmp_path)
        orchestrator = create_orchestrator(config)

        assert orchestrator.timeout_seconds == DEFAULT_AGENT_TIMEOUT_MINUTES * 60

    def test_create_orchestrator_zero_timeout_uses_default(
        self, tmp_path: Path
    ) -> None:
        """Timeout of 0 is treated as falsy and uses default."""
        from src.orchestration.factory import create_orchestrator
        from src.orchestration.types import (
            DEFAULT_AGENT_TIMEOUT_MINUTES,
            OrchestratorConfig,
        )

        config = OrchestratorConfig(repo_path=tmp_path, timeout_minutes=0)
        orchestrator = create_orchestrator(config)

        # 0 is falsy, so default is used
        assert orchestrator.timeout_seconds == DEFAULT_AGENT_TIMEOUT_MINUTES * 60

    def test_orchestrator_config_defaults(self) -> None:
        """OrchestratorConfig has sensible defaults."""
        from src.orchestration.types import OrchestratorConfig

        config = OrchestratorConfig(repo_path=Path("/tmp"))

        assert config.max_agents is None
        assert config.timeout_minutes is None
        assert config.max_issues is None
        assert config.epic_id is None
        assert config.only_ids is None
        assert config.max_gate_retries == 3
        assert config.max_review_retries == 3
        assert config.disable_validations is None
        assert config.include_wip is False
        assert config.focus is True

    def test_orchestrator_dependencies_all_optional(self) -> None:
        """OrchestratorDependencies allows all fields to be None."""
        from src.orchestration.types import OrchestratorDependencies

        deps = OrchestratorDependencies()

        assert deps.issue_provider is None
        assert deps.code_reviewer is None
        assert deps.gate_checker is None
        assert deps.log_provider is None
        assert deps.telemetry_provider is None
        assert deps.event_sink is None


class TestBuildGateMetadata:
    """Tests for _build_gate_metadata pure function."""

    def test_none_gate_result_returns_empty_metadata(self) -> None:
        """When gate_result is None, returns empty GateMetadata."""
        from src.pipeline.gate_metadata import (
            build_gate_metadata as _build_gate_metadata,
        )

        result = _build_gate_metadata(None, passed=True)

        assert result.evidence_check_result is None
        assert result.validation_result is None

    def test_successful_gate_with_full_evidence(self) -> None:
        """Successful gate result with full evidence extracts all fields."""
        from src.pipeline.gate_metadata import (
            build_gate_metadata as _build_gate_metadata,
        )
        from src.domain.evidence_check import GateResult, ValidationEvidence
        from src.domain.validation.spec import CommandKind

        evidence = ValidationEvidence(
            commands_ran={CommandKind.TEST: True, CommandKind.LINT: True},
            failed_commands=[],
        )
        gate_result = GateResult(
            passed=True,
            failure_reasons=[],
            commit_hash="abc123",
            validation_evidence=evidence,
        )

        result = _build_gate_metadata(gate_result, passed=True)

        assert result.evidence_check_result is not None
        assert result.evidence_check_result.passed is True
        assert result.evidence_check_result.failure_reasons == []
        assert result.evidence_check_result.evidence["commit_found"] is True
        assert result.validation_result is not None
        assert result.validation_result.passed is True
        # Commands run uses kind.value (e.g., "test", "lint")
        assert "test" in result.validation_result.commands_run
        assert "lint" in result.validation_result.commands_run
        assert result.validation_result.commands_failed == []

    def test_failed_gate_with_partial_evidence(self) -> None:
        """Failed gate result extracts failure reasons and evidence."""
        from src.pipeline.gate_metadata import (
            build_gate_metadata as _build_gate_metadata,
        )
        from src.domain.evidence_check import GateResult, ValidationEvidence
        from src.domain.validation.spec import CommandKind

        evidence = ValidationEvidence(
            commands_ran={CommandKind.TEST: True, CommandKind.LINT: False},
            failed_commands=["pytest"],
        )
        gate_result = GateResult(
            passed=False,
            failure_reasons=["pytest failed", "no commit found"],
            commit_hash=None,
            validation_evidence=evidence,
        )

        result = _build_gate_metadata(gate_result, passed=False)

        assert result.evidence_check_result is not None
        assert result.evidence_check_result.passed is False
        assert result.evidence_check_result.failure_reasons == [
            "pytest failed",
            "no commit found",
        ]
        assert result.evidence_check_result.evidence["commit_found"] is False
        assert result.validation_result is not None
        assert result.validation_result.passed is False
        # Commands run uses kind.value (e.g., "test")
        assert "test" in result.validation_result.commands_run
        # Failed commands still uses extracted tool name from log parsing
        assert "pytest" in result.validation_result.commands_failed

    def test_empty_failure_reasons_and_missing_commit(self) -> None:
        """Gate result with empty failure reasons and missing commit."""
        from src.pipeline.gate_metadata import (
            build_gate_metadata as _build_gate_metadata,
        )
        from src.domain.evidence_check import GateResult, ValidationEvidence

        evidence = ValidationEvidence(commands_ran={}, failed_commands=[])
        gate_result = GateResult(
            passed=False,
            failure_reasons=[],
            commit_hash=None,
            validation_evidence=evidence,
        )

        result = _build_gate_metadata(gate_result, passed=False)

        assert result.evidence_check_result is not None
        assert result.evidence_check_result.failure_reasons == []
        assert result.evidence_check_result.evidence["commit_found"] is False

    def test_passed_true_overrides_gate_result_passed(self) -> None:
        """When passed=True, evidence_check_result.passed should be True."""
        from src.pipeline.gate_metadata import (
            build_gate_metadata as _build_gate_metadata,
        )
        from src.domain.evidence_check import GateResult, ValidationEvidence

        evidence = ValidationEvidence(commands_ran={}, failed_commands=[])
        gate_result = GateResult(
            passed=False,  # Gate says failed
            failure_reasons=["some reason"],
            commit_hash="abc123",
            validation_evidence=evidence,
        )

        # But overall run passed (e.g., resolution marker)
        result = _build_gate_metadata(gate_result, passed=True)

        assert result.evidence_check_result is not None
        # passed=True should override gate_result.passed
        assert result.evidence_check_result.passed is True
        # failure_reasons should be empty when passed=True
        assert result.evidence_check_result.failure_reasons == []


class TestBuildGateMetadataFromLogs:
    """Tests for _build_gate_metadata_from_logs fallback function."""

    def test_none_spec_returns_empty_metadata(
        self, tmp_path: Path, log_provider: LogProvider
    ) -> None:
        """When per_session_spec is None, returns empty GateMetadata."""
        from typing import TYPE_CHECKING, cast

        from src.pipeline.gate_metadata import (
            build_gate_metadata_from_logs as _build_gate_metadata_from_logs,
        )
        from src.domain.evidence_check import EvidenceCheck

        if TYPE_CHECKING:
            from src.core.protocols.validation import GateChecker

        log_path = tmp_path / "test.log"
        log_path.write_text("{}")
        evidence_check = cast(
            "GateChecker",
            EvidenceCheck(tmp_path, log_provider, CommandRunner(cwd=tmp_path)),
        )

        result = _build_gate_metadata_from_logs(
            log_path=log_path,
            result_summary="Quality gate failed: no tests",
            result_success=False,
            evidence_check=evidence_check,
            per_session_spec=None,
        )

        assert result.evidence_check_result is None
        assert result.validation_result is None

    def test_valid_spec_parses_evidence(
        self, tmp_path: Path, log_provider: LogProvider
    ) -> None:
        """With valid spec, parses evidence from logs."""
        import re
        from typing import TYPE_CHECKING, cast

        from src.pipeline.gate_metadata import (
            build_gate_metadata_from_logs as _build_gate_metadata_from_logs,
        )
        from src.domain.evidence_check import EvidenceCheck
        from src.domain.validation.spec import (
            CommandKind,
            ValidationCommand,
            ValidationScope,
            ValidationSpec,
        )

        if TYPE_CHECKING:
            from src.core.protocols.validation import GateChecker

        log_path = tmp_path / "test.log"
        # Write a minimal log entry
        log_path.write_text('{"type":"result"}\n')

        evidence_check = cast(
            "GateChecker",
            EvidenceCheck(tmp_path, log_provider, CommandRunner(cwd=tmp_path)),
        )
        spec = ValidationSpec(
            commands=[
                ValidationCommand(
                    name="pytest",
                    command="pytest",
                    kind=CommandKind.TEST,
                    detection_pattern=re.compile(r"pytest"),
                ),
            ],
            scope=ValidationScope.PER_SESSION,
        )

        result = _build_gate_metadata_from_logs(
            log_path=log_path,
            result_summary="Quality gate failed: pytest failed",
            result_success=False,
            evidence_check=evidence_check,
            per_session_spec=spec,
        )

        assert result.evidence_check_result is not None
        assert result.evidence_check_result.passed is False
        assert "pytest failed" in result.evidence_check_result.failure_reasons
        # validation_result is now populated with parsed evidence
        assert result.validation_result is not None
        assert result.validation_result.passed is False

    def test_result_success_determines_passed_status(
        self, tmp_path: Path, log_provider: LogProvider
    ) -> None:
        """result_success parameter determines evidence_check_result.passed."""
        from typing import TYPE_CHECKING, cast

        from src.pipeline.gate_metadata import (
            build_gate_metadata_from_logs as _build_gate_metadata_from_logs,
        )
        from src.domain.evidence_check import EvidenceCheck
        from src.domain.validation.spec import ValidationScope, ValidationSpec

        if TYPE_CHECKING:
            from src.core.protocols.validation import GateChecker

        log_path = tmp_path / "test.log"
        log_path.write_text('{"type":"result"}\n')

        evidence_check = cast(
            "GateChecker",
            EvidenceCheck(tmp_path, log_provider, CommandRunner(cwd=tmp_path)),
        )
        spec = ValidationSpec(commands=[], scope=ValidationScope.PER_SESSION)

        # Test with result_success=True
        result = _build_gate_metadata_from_logs(
            log_path=log_path,
            result_summary="Success",
            result_success=True,
            evidence_check=evidence_check,
            per_session_spec=spec,
        )

        assert result.evidence_check_result is not None
        assert result.evidence_check_result.passed is True

    def test_extracts_failure_reasons_from_summary(
        self, tmp_path: Path, log_provider: LogProvider
    ) -> None:
        """Extracts failure reasons from 'Quality gate failed:' prefix."""
        from typing import TYPE_CHECKING, cast

        from src.pipeline.gate_metadata import (
            build_gate_metadata_from_logs as _build_gate_metadata_from_logs,
        )
        from src.domain.evidence_check import EvidenceCheck
        from src.domain.validation.spec import ValidationScope, ValidationSpec

        if TYPE_CHECKING:
            from src.core.protocols.validation import GateChecker

        log_path = tmp_path / "test.log"
        log_path.write_text('{"type":"result"}\n')

        evidence_check = cast(
            "GateChecker",
            EvidenceCheck(tmp_path, log_provider, CommandRunner(cwd=tmp_path)),
        )
        spec = ValidationSpec(commands=[], scope=ValidationScope.PER_SESSION)

        result = _build_gate_metadata_from_logs(
            log_path=log_path,
            result_summary="Quality gate failed: lint failed; no commit",
            result_success=False,
            evidence_check=evidence_check,
            per_session_spec=spec,
        )

        assert result.evidence_check_result is not None
        assert result.evidence_check_result.failure_reasons == [
            "lint failed",
            "no commit",
        ]

    def test_builds_validation_result_from_evidence(
        self, tmp_path: Path, log_provider: LogProvider
    ) -> None:
        """Builds validation_result (not None) matching _build_gate_metadata behavior."""
        from typing import TYPE_CHECKING, cast

        from src.pipeline.gate_metadata import (
            build_gate_metadata_from_logs as _build_gate_metadata_from_logs,
        )
        from src.domain.evidence_check import EvidenceCheck
        from src.domain.validation.spec import ValidationScope, ValidationSpec

        if TYPE_CHECKING:
            from src.core.protocols.validation import GateChecker

        # Create log file
        log_path = tmp_path / "test.log"
        log_path.write_text('{"type":"result"}\n')

        evidence_check = cast(
            "GateChecker",
            EvidenceCheck(tmp_path, log_provider, CommandRunner(cwd=tmp_path)),
        )
        spec = ValidationSpec(commands=[], scope=ValidationScope.PER_SESSION)

        result = _build_gate_metadata_from_logs(
            log_path=log_path,
            result_summary="Success",
            result_success=True,
            evidence_check=evidence_check,
            per_session_spec=spec,
        )

        # Key assertion: validation_result is now populated (not None)
        # This matches _build_gate_metadata behavior per spec
        assert result.validation_result is not None
        assert result.validation_result.passed is True
        # commands_run and commands_failed are lists (may be empty if no commands detected)
        assert isinstance(result.validation_result.commands_run, list)
        assert isinstance(result.validation_result.commands_failed, list)


class TestSigintEscalation:
    """Tests for SIGINT three-stage escalation via LifecycleController.

    Detailed escalation state tests are in test_lifecycle_controller.py.
    These tests verify the orchestrator wiring to LifecycleController.
    """

    def test_handle_sigint_stage1_drain(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """First SIGINT sets drain mode via lifecycle controller."""
        orchestrator = make_orchestrator(repo_path=tmp_path)
        loop = MagicMock(spec=asyncio.AbstractEventLoop)

        # Ensure active_tasks exists and is empty
        orchestrator.issue_coordinator.active_tasks.clear()

        # Configure lifecycle callbacks (normally done in _reset_sigint_state)
        orchestrator._lifecycle.configure_callbacks(
            get_active_task_count=lambda: len(
                orchestrator.issue_coordinator.active_tasks
            ),
            on_drain_started=orchestrator.event_sink.on_drain_started,
            on_abort_started=orchestrator.event_sink.on_abort_started,
            on_force_abort=orchestrator.event_sink.on_force_abort,
            forward_sigint=CommandRunner.forward_sigint,
            kill_processes=CommandRunner.kill_active_process_groups,
        )

        # First SIGINT
        orchestrator._handle_sigint(loop)

        assert orchestrator._lifecycle._sigint_count == 1
        assert orchestrator._lifecycle.is_drain_mode()
        assert not orchestrator._lifecycle.is_abort_mode()

    def test_handle_sigint_stage2_abort(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """Second SIGINT sets abort mode via lifecycle controller."""
        orchestrator = make_orchestrator(repo_path=tmp_path)
        loop = MagicMock(spec=asyncio.AbstractEventLoop)
        orchestrator.issue_coordinator.active_tasks.clear()

        # Configure lifecycle callbacks
        orchestrator._lifecycle.configure_callbacks(
            get_active_task_count=lambda: len(
                orchestrator.issue_coordinator.active_tasks
            ),
            on_drain_started=orchestrator.event_sink.on_drain_started,
            on_abort_started=orchestrator.event_sink.on_abort_started,
            on_force_abort=orchestrator.event_sink.on_force_abort,
            forward_sigint=CommandRunner.forward_sigint,
            kill_processes=CommandRunner.kill_active_process_groups,
        )

        # First SIGINT (drain mode)
        orchestrator._handle_sigint(loop)

        # Second SIGINT
        orchestrator._handle_sigint(loop)

        assert orchestrator._lifecycle._sigint_count == 2
        assert orchestrator._lifecycle.is_abort_mode()
        assert orchestrator._lifecycle.abort_exit_code == 130  # No validation failure

    def test_handle_sigint_stage3_force_abort(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """Third SIGINT triggers hard abort via lifecycle controller."""
        orchestrator = make_orchestrator(repo_path=tmp_path)
        loop = MagicMock(spec=asyncio.AbstractEventLoop)
        orchestrator.issue_coordinator.active_tasks.clear()

        # Set up run task
        run_task = MagicMock(spec=asyncio.Task)
        orchestrator._lifecycle.run_task = run_task

        # Configure lifecycle callbacks
        orchestrator._lifecycle.configure_callbacks(
            get_active_task_count=lambda: len(
                orchestrator.issue_coordinator.active_tasks
            ),
            on_drain_started=orchestrator.event_sink.on_drain_started,
            on_abort_started=orchestrator.event_sink.on_abort_started,
            on_force_abort=orchestrator.event_sink.on_force_abort,
            forward_sigint=CommandRunner.forward_sigint,
            kill_processes=CommandRunner.kill_active_process_groups,
        )

        # First two SIGINTs
        orchestrator._handle_sigint(loop)
        orchestrator._handle_sigint(loop)

        # Third SIGINT
        orchestrator._handle_sigint(loop)

        assert orchestrator._lifecycle._sigint_count == 3
        assert orchestrator._lifecycle.is_shutdown_requested()

    def test_handle_sigint_exit_code_with_validation_failure(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """Stage 2 exit code is 1 when validation has failed."""
        orchestrator = make_orchestrator(repo_path=tmp_path)
        loop = MagicMock(spec=asyncio.AbstractEventLoop)
        orchestrator.issue_coordinator.active_tasks.clear()

        # Configure lifecycle callbacks
        orchestrator._lifecycle.configure_callbacks(
            get_active_task_count=lambda: len(
                orchestrator.issue_coordinator.active_tasks
            ),
            on_drain_started=orchestrator.event_sink.on_drain_started,
            on_abort_started=orchestrator.event_sink.on_abort_started,
            on_force_abort=orchestrator.event_sink.on_force_abort,
            forward_sigint=CommandRunner.forward_sigint,
            kill_processes=CommandRunner.kill_active_process_groups,
        )

        # Mark validation as failed
        orchestrator._lifecycle.validation_failed = True

        # First SIGINT
        orchestrator._handle_sigint(loop)

        # Second SIGINT - should snapshot exit code as 1
        orchestrator._handle_sigint(loop)

        assert orchestrator._lifecycle.abort_exit_code == 1

    def test_handle_sigint_escalation_window_resets_when_idle(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """Escalation window resets after 5s when not in drain/abort mode."""
        from src.orchestration.lifecycle_controller import ESCALATION_WINDOW_SECONDS

        orchestrator = make_orchestrator(repo_path=tmp_path)
        loop = MagicMock(spec=asyncio.AbstractEventLoop)
        orchestrator.issue_coordinator.active_tasks.clear()

        # Configure lifecycle callbacks
        orchestrator._lifecycle.configure_callbacks(
            get_active_task_count=lambda: len(
                orchestrator.issue_coordinator.active_tasks
            ),
            on_drain_started=orchestrator.event_sink.on_drain_started,
            on_abort_started=orchestrator.event_sink.on_abort_started,
            on_force_abort=orchestrator.event_sink.on_force_abort,
            forward_sigint=CommandRunner.forward_sigint,
            kill_processes=CommandRunner.kill_active_process_groups,
        )

        # Simulate time passage beyond escalation window
        orchestrator._lifecycle._sigint_last_at = (
            time.monotonic() - ESCALATION_WINDOW_SECONDS - 1
        )
        orchestrator._lifecycle._sigint_count = (
            1  # Leftover from previous, but not in drain mode
        )

        # This SIGINT should reset count since not in drain/abort mode
        orchestrator._handle_sigint(loop)

        # Count should be 1 (reset to 0, then incremented)
        assert orchestrator._lifecycle._sigint_count == 1
        assert orchestrator._lifecycle.is_drain_mode()

    def test_handle_sigint_escalation_window_does_not_reset_in_drain_mode(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """Escalation window does NOT reset when in drain mode (continuous escalation)."""
        from src.orchestration.lifecycle_controller import ESCALATION_WINDOW_SECONDS

        orchestrator = make_orchestrator(repo_path=tmp_path)
        loop = MagicMock(spec=asyncio.AbstractEventLoop)
        orchestrator.issue_coordinator.active_tasks.clear()

        # Configure lifecycle callbacks
        orchestrator._lifecycle.configure_callbacks(
            get_active_task_count=lambda: len(
                orchestrator.issue_coordinator.active_tasks
            ),
            on_drain_started=orchestrator.event_sink.on_drain_started,
            on_abort_started=orchestrator.event_sink.on_abort_started,
            on_force_abort=orchestrator.event_sink.on_force_abort,
            forward_sigint=CommandRunner.forward_sigint,
            kill_processes=CommandRunner.kill_active_process_groups,
        )

        # Enter drain mode
        orchestrator._handle_sigint(loop)
        assert orchestrator._lifecycle._sigint_count == 1
        assert orchestrator._lifecycle.is_drain_mode()

        # Simulate time passage beyond escalation window
        orchestrator._lifecycle._sigint_last_at = (
            time.monotonic() - ESCALATION_WINDOW_SECONDS - 1
        )

        # Second SIGINT should still escalate (not reset) since in drain mode
        orchestrator._handle_sigint(loop)

        # Should escalate to abort mode, not reset
        assert orchestrator._lifecycle._sigint_count == 2
        assert orchestrator._lifecycle.is_abort_mode()

    def test_sigint_state_reset_per_run(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """SIGINT state is reset by _reset_sigint_state()."""
        orchestrator = make_orchestrator(repo_path=tmp_path)

        # Simulate previous run's dirty state via lifecycle controller
        orchestrator._lifecycle._sigint_count = 2
        orchestrator._lifecycle._sigint_last_at = 123.456
        orchestrator._lifecycle._drain_mode_active = True
        orchestrator._lifecycle._abort_mode_active = True
        orchestrator._lifecycle._abort_exit_code = 1
        orchestrator._lifecycle._validation_failed = True
        orchestrator._lifecycle._shutdown_requested = True
        orchestrator._lifecycle._run_task = MagicMock()

        # Call the reset helper (used by run())
        orchestrator._reset_sigint_state()

        # Verify all SIGINT state is clean
        assert orchestrator._lifecycle._sigint_count == 0
        assert orchestrator._lifecycle._sigint_last_at == 0.0
        assert not orchestrator._lifecycle.is_drain_mode()
        assert not orchestrator._lifecycle.is_abort_mode()
        assert orchestrator._lifecycle.abort_exit_code == 130
        assert not orchestrator._lifecycle.validation_failed
        assert not orchestrator._lifecycle.is_shutdown_requested()
        assert orchestrator._lifecycle.run_task is None

    def test_handle_sigint_on_drain_started_receives_active_count(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """on_drain_started receives correct active task count via lifecycle."""
        orchestrator = make_orchestrator(repo_path=tmp_path)
        loop = MagicMock(spec=asyncio.AbstractEventLoop)

        # Set up mock active tasks
        mock_task = MagicMock(spec=asyncio.Task)
        orchestrator.issue_coordinator.active_tasks["issue-1"] = mock_task
        orchestrator.issue_coordinator.active_tasks["issue-2"] = mock_task

        # Mock on_drain_started to capture the count
        mock_on_drain_started = MagicMock()
        orchestrator.event_sink.on_drain_started = mock_on_drain_started  # type: ignore[method-assign]

        # Configure lifecycle callbacks (use the mocked on_drain_started)
        orchestrator._lifecycle.configure_callbacks(
            get_active_task_count=lambda: len(
                orchestrator.issue_coordinator.active_tasks
            ),
            on_drain_started=mock_on_drain_started,
            on_abort_started=orchestrator.event_sink.on_abort_started,
            on_force_abort=orchestrator.event_sink.on_force_abort,
            forward_sigint=CommandRunner.forward_sigint,
            kill_processes=CommandRunner.kill_active_process_groups,
        )

        # First SIGINT
        orchestrator._handle_sigint(loop)

        # Extract the lambda and call it to verify the count
        lambda_calls = [
            call[0][0]
            for call in loop.call_soon_threadsafe.call_args_list
            if callable(call[0][0])
        ]
        # Find the lambda (not drain_event.set)
        for fn in lambda_calls:
            if hasattr(fn, "__name__") and fn.__name__ == "<lambda>":
                fn()
                break

        mock_on_drain_started.assert_called_once_with(2)


class TestSessionResume:
    """Tests for session resume functionality when include_wip is enabled."""

    @pytest.mark.asyncio
    async def test_resume_calls_lookup_for_wip_issues(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """When include_wip=True, lookup_prior_session should be called."""
        fake_issues = FakeIssueProvider(
            {"test-issue": FakeIssue(id="test-issue", priority=1)}
        )
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        log_dir = tmp_path / ".claude" / "projects" / tmp_path.name
        log_dir.mkdir(parents=True)
        log_file = log_dir / "test-session.jsonl"
        log_file.write_text('{"type": "result"}\n')

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            issue_provider=fake_issues,
            runs_dir=runs_dir,
            lock_releaser=lambda _: 0,
            include_wip=True,
            log_provider=_make_mock_log_provider(log_file),  # type: ignore[arg-type]
        )

        # Create mock SDK client
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.query = AsyncMock()

        async def mock_receive_response() -> AsyncGenerator[ResultMessage, None]:
            yield ResultMessage(
                subtype="result",
                session_id="test-session",
                result="ISSUE_NO_CHANGE: Already implemented",
                duration_ms=1000,
                duration_api_ms=800,
                is_error=False,
                num_turns=1,
                total_cost_usd=0.01,
                usage=None,
            )

        mock_client.receive_response = mock_receive_response

        with (
            patch("claude_agent_sdk.ClaudeSDKClient", return_value=mock_client),
            patch(
                "src.orchestration.orchestrator.get_git_branch_async",
                return_value="main",
            ),
            patch(
                "src.orchestration.orchestrator.get_git_commit_async",
                return_value="abc123",
            ),
            patch.object(
                orchestrator.beads,
                "get_issue_description_async",
                return_value="Test issue",
            ),
            patch(
                "src.orchestration.orchestrator.lookup_prior_session_info",
                return_value=MagicMock(
                    session_id="prior-session-id", baseline_timestamp=1700000000
                ),
            ) as mock_lookup,
        ):
            await orchestrator.run_implementer("test-issue")

            # Verify lookup_prior_session_info was called with correct args
            mock_lookup.assert_called_once_with(tmp_path, "test-issue")

    @pytest.mark.asyncio
    async def test_strict_mode_fails_issue_when_no_session(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """When strict_resume=True and no prior session, issue should fail."""
        fake_issues = FakeIssueProvider(
            {"test-issue": FakeIssue(id="test-issue", priority=1)}
        )
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            issue_provider=fake_issues,
            runs_dir=runs_dir,
            lock_releaser=lambda _: 0,
            include_wip=True,
            strict_resume=True,
        )

        with (
            patch(
                "src.orchestration.orchestrator.get_git_branch_async",
                return_value="main",
            ),
            patch(
                "src.orchestration.orchestrator.get_git_commit_async",
                return_value="abc123",
            ),
            patch.object(
                orchestrator.beads,
                "get_issue_description_async",
                return_value="Test issue",
            ),
            patch(
                "src.orchestration.orchestrator.lookup_prior_session_info",
                return_value=None,  # No prior session
            ),
        ):
            result = await orchestrator.run_implementer("test-issue")

            assert result.success is False
            assert "No prior session found" in result.summary
            assert "strict mode" in result.summary

    @pytest.mark.asyncio
    async def test_lenient_mode_fallback_on_no_session(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """When strict_resume=False (default) and no prior session, run proceeds."""
        fake_issues = FakeIssueProvider(
            {"test-issue": FakeIssue(id="test-issue", priority=1)}
        )
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        log_dir = tmp_path / ".claude" / "projects" / tmp_path.name
        log_dir.mkdir(parents=True)
        log_file = log_dir / "test-session.jsonl"
        log_file.write_text('{"type": "result"}\n')

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            issue_provider=fake_issues,
            runs_dir=runs_dir,
            lock_releaser=lambda _: 0,
            include_wip=True,
            strict_resume=False,  # Default lenient mode
            log_provider=_make_mock_log_provider(log_file),  # type: ignore[arg-type]
        )

        # Create mock SDK client
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.query = AsyncMock()

        async def mock_receive_response() -> AsyncGenerator[ResultMessage, None]:
            yield ResultMessage(
                subtype="result",
                session_id="new-session",
                result="ISSUE_NO_CHANGE: Done",
                duration_ms=1000,
                duration_api_ms=800,
                is_error=False,
                num_turns=1,
                total_cost_usd=0.01,
                usage=None,
            )

        mock_client.receive_response = mock_receive_response

        with (
            patch("claude_agent_sdk.ClaudeSDKClient", return_value=mock_client),
            patch(
                "src.orchestration.orchestrator.get_git_branch_async",
                return_value="main",
            ),
            patch(
                "src.orchestration.orchestrator.get_git_commit_async",
                return_value="abc123",
            ),
            patch.object(
                orchestrator.beads,
                "get_issue_description_async",
                return_value="Test issue",
            ),
            patch(
                "src.orchestration.orchestrator.lookup_prior_session_info",
                return_value=None,  # No prior session
            ),
        ):
            result = await orchestrator.run_implementer("test-issue")

            # Should NOT fail early due to missing session (lenient mode)
            # The session ran (even if gate failed later for unrelated reasons)
            assert "No prior session found" not in result.summary
            # Session ran - we have a session_id from the SDK
            assert result.session_id is not None


class TestBuildResumePrompt:
    """Tests for _build_resume_prompt function."""

    def test_returns_none_when_no_issues(self, tmp_path: Path) -> None:
        """Returns None when review_issues is empty."""
        from src.domain.prompts import PromptProvider
        from src.domain.validation.config import PromptValidationCommands
        from src.orchestration.orchestrator import _build_resume_prompt

        prompts = PromptProvider(
            implementer_prompt="impl",
            review_followup_prompt="followup: {review_issues}",
            gate_followup_prompt="gate",
            fixer_prompt="fixer",
            idle_resume_prompt="idle",
            checkpoint_request_prompt="checkpoint",
            continuation_prompt="continuation",
        )
        validation_commands = PromptValidationCommands(
            lint="lint",
            format="format",
            typecheck="typecheck",
            test="test",
            custom_commands=(),
        )

        result = _build_resume_prompt(
            review_issues=[],
            prompts=prompts,
            validation_commands=validation_commands,
            issue_id="test-issue",
            max_review_retries=3,
            repo_path=tmp_path,
            prior_run_id="run-123",
        )

        assert result is None

    def test_formats_prompt_with_review_issues(self, tmp_path: Path) -> None:
        """Formats review_followup_prompt with issues and validation commands."""
        from src.domain.prompts import PromptProvider
        from src.domain.validation.config import PromptValidationCommands
        from src.orchestration.orchestrator import _build_resume_prompt

        prompts = PromptProvider(
            implementer_prompt="impl",
            review_followup_prompt=(
                "Attempt {attempt}/{max_attempts}\n"
                "Issues:\n{review_issues}\n"
                "Issue: {issue_id}\n"
                "Lint: {lint_command}\n"
                "Format: {format_command}\n"
                "Typecheck: {typecheck_command}\n"
                "Test: {test_command}\n"
                "{custom_commands_section}"
            ),
            gate_followup_prompt="gate",
            fixer_prompt="fixer",
            idle_resume_prompt="idle",
            checkpoint_request_prompt="checkpoint",
            continuation_prompt="continuation",
        )
        validation_commands = PromptValidationCommands(
            lint="uvx ruff check .",
            format="uvx ruff format .",
            typecheck="uvx ty check",
            test="uv run pytest",
            custom_commands=(),
        )
        review_issues = [
            {
                "file": str(tmp_path / "src/main.py"),
                "line_start": 10,
                "line_end": 15,
                "priority": 0,
                "title": "Fix critical bug",
                "body": "This needs immediate attention",
                "reviewer": "cerberus",
            },
        ]

        result = _build_resume_prompt(
            review_issues=review_issues,
            prompts=prompts,
            validation_commands=validation_commands,
            issue_id="test-123",
            max_review_retries=3,
            repo_path=tmp_path,
            prior_run_id="run-456",
        )

        assert result is not None
        assert "Attempt 1/3" in result
        assert "Issue: test-123" in result
        assert "Fix critical bug" in result
        assert "uvx ruff check ." in result
        assert "uvx ty check" in result

    def test_uses_attempt_1_for_resume(self, tmp_path: Path) -> None:
        """Resume always uses attempt=1 since retry counters reset."""
        from src.domain.prompts import PromptProvider
        from src.domain.validation.config import PromptValidationCommands
        from src.orchestration.orchestrator import _build_resume_prompt

        prompts = PromptProvider(
            implementer_prompt="impl",
            review_followup_prompt="Attempt {attempt}/{max_attempts}",
            gate_followup_prompt="gate",
            fixer_prompt="fixer",
            idle_resume_prompt="idle",
            checkpoint_request_prompt="checkpoint",
            continuation_prompt="continuation",
        )
        validation_commands = PromptValidationCommands(
            lint="lint",
            format="format",
            typecheck="typecheck",
            test="test",
            custom_commands=(),
        )
        review_issues = [{"file": "a.py", "title": "Issue"}]

        result = _build_resume_prompt(
            review_issues=review_issues,
            prompts=prompts,
            validation_commands=validation_commands,
            issue_id="test-issue",
            max_review_retries=5,
            repo_path=tmp_path,
            prior_run_id="run-789",
        )

        assert result == "Attempt 1/5"

    def test_includes_custom_commands_section(self, tmp_path: Path) -> None:
        """Custom commands are included in the prompt."""
        from src.domain.prompts import PromptProvider
        from src.domain.validation.config import PromptValidationCommands
        from src.orchestration.orchestrator import _build_resume_prompt

        prompts = PromptProvider(
            implementer_prompt="impl",
            review_followup_prompt="{custom_commands_section}",
            gate_followup_prompt="gate",
            fixer_prompt="fixer",
            idle_resume_prompt="idle",
            checkpoint_request_prompt="checkpoint",
            continuation_prompt="continuation",
        )
        validation_commands = PromptValidationCommands(
            lint="lint",
            format="format",
            typecheck="typecheck",
            test="test",
            custom_commands=(("security-check", "npm audit", 60, False),),
        )
        review_issues = [{"file": "a.py", "title": "Issue"}]

        result = _build_resume_prompt(
            review_issues=review_issues,
            prompts=prompts,
            validation_commands=validation_commands,
            issue_id="test-issue",
            max_review_retries=3,
            repo_path=tmp_path,
            prior_run_id="run-abc",
        )

        assert result is not None
        assert "security-check" in result or "npm audit" in result


class TestCaptureRunStartCommit:
    """Tests for _capture_run_start_commit method."""

    @pytest.mark.asyncio
    async def test_captures_head_when_run_end_configured(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """Captures HEAD commit when run_end trigger is configured."""
        from src.domain.validation.config import (
            FailureMode,
            FireOn,
            RunEndTriggerConfig,
            ValidationConfig,
            ValidationTriggersConfig,
        )

        # Create orchestrator with run_end trigger configured
        triggers = ValidationTriggersConfig(
            run_end=RunEndTriggerConfig(
                failure_mode=FailureMode.CONTINUE,
                commands=(),
                fire_on=FireOn.SUCCESS,
            )
        )
        validation_config = ValidationConfig(validation_triggers=triggers)

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            timeout_minutes=1,
            max_issues=1,
        )
        # Inject validation config after creation
        orchestrator._validation_config = validation_config

        # Mock get_git_commit_async
        with patch(
            "src.orchestration.orchestrator.get_git_commit_async",
            return_value="abc123",
        ):
            await orchestrator._capture_run_start_commit()

        assert orchestrator._state.run_start_commit == "abc123"

    @pytest.mark.asyncio
    async def test_skips_capture_when_no_trigger_needs_it(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """Skips capture when neither run_end nor epic_completion code_review is configured."""
        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            timeout_minutes=1,
            max_issues=1,
        )

        # Mock should not be called
        with patch(
            "src.orchestration.orchestrator.get_git_commit_async",
            return_value="abc123",
        ) as mock_get:
            await orchestrator._capture_run_start_commit()
            mock_get.assert_not_called()

        assert orchestrator._state.run_start_commit is None

    @pytest.mark.asyncio
    async def test_skips_capture_when_already_captured(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """Skips capture when run_start_commit already has a value."""
        from src.domain.validation.config import (
            FailureMode,
            FireOn,
            RunEndTriggerConfig,
            ValidationConfig,
            ValidationTriggersConfig,
        )

        triggers = ValidationTriggersConfig(
            run_end=RunEndTriggerConfig(
                failure_mode=FailureMode.CONTINUE,
                commands=(),
                fire_on=FireOn.SUCCESS,
            )
        )
        validation_config = ValidationConfig(validation_triggers=triggers)

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            timeout_minutes=1,
            max_issues=1,
        )
        orchestrator._validation_config = validation_config

        # Pre-set the commit (simulating resumed run)
        orchestrator._state.run_start_commit = "existing123"

        with patch(
            "src.orchestration.orchestrator.get_git_commit_async",
            return_value="new456",
        ) as mock_get:
            await orchestrator._capture_run_start_commit()
            mock_get.assert_not_called()

        # Original value preserved
        assert orchestrator._state.run_start_commit == "existing123"

    @pytest.mark.asyncio
    async def test_captures_head_when_epic_completion_code_review_enabled(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """Captures HEAD commit when epic_completion has code_review enabled."""
        from src.domain.validation.config import (
            CodeReviewConfig,
            EpicCompletionTriggerConfig,
            EpicDepth,
            FailureMode,
            FireOn,
            ValidationConfig,
            ValidationTriggersConfig,
        )

        # Create orchestrator with epic_completion + code_review (no run_end)
        code_review = CodeReviewConfig(enabled=True, baseline="since_run_start")
        triggers = ValidationTriggersConfig(
            epic_completion=EpicCompletionTriggerConfig(
                failure_mode=FailureMode.CONTINUE,
                commands=(),
                epic_depth=EpicDepth.TOP_LEVEL,
                fire_on=FireOn.SUCCESS,
                code_review=code_review,
            )
        )
        validation_config = ValidationConfig(validation_triggers=triggers)

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            timeout_minutes=1,
            max_issues=1,
        )
        orchestrator._validation_config = validation_config

        with patch(
            "src.orchestration.orchestrator.get_git_commit_async",
            return_value="epic123",
        ):
            await orchestrator._capture_run_start_commit()

        assert orchestrator._state.run_start_commit == "epic123"

    @pytest.mark.asyncio
    async def test_skips_capture_when_epic_completion_code_review_disabled(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """Skips capture when epic_completion exists but code_review is disabled."""
        from src.domain.validation.config import (
            CodeReviewConfig,
            EpicCompletionTriggerConfig,
            EpicDepth,
            FailureMode,
            FireOn,
            ValidationConfig,
            ValidationTriggersConfig,
        )

        # epic_completion with code_review.enabled=False
        code_review = CodeReviewConfig(enabled=False)
        triggers = ValidationTriggersConfig(
            epic_completion=EpicCompletionTriggerConfig(
                failure_mode=FailureMode.CONTINUE,
                commands=(),
                epic_depth=EpicDepth.TOP_LEVEL,
                fire_on=FireOn.SUCCESS,
                code_review=code_review,
            )
        )
        validation_config = ValidationConfig(validation_triggers=triggers)

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            timeout_minutes=1,
            max_issues=1,
        )
        orchestrator._validation_config = validation_config

        with patch(
            "src.orchestration.orchestrator.get_git_commit_async",
            return_value="abc123",
        ) as mock_get:
            await orchestrator._capture_run_start_commit()
            mock_get.assert_not_called()

        assert orchestrator._state.run_start_commit is None


class TestFireRunEndTrigger:
    """Tests for _fire_run_end_trigger method."""

    @pytest.mark.asyncio
    async def test_queues_trigger_on_success_when_fire_on_success(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """Queues trigger when fire_on=SUCCESS and all issues succeeded."""
        from src.domain.validation.config import (
            FailureMode,
            FireOn,
            RunEndTriggerConfig,
            ValidationConfig,
            ValidationTriggersConfig,
        )
        from src.domain.validation.config import TriggerType

        triggers = ValidationTriggersConfig(
            run_end=RunEndTriggerConfig(
                failure_mode=FailureMode.CONTINUE,
                commands=(),
                fire_on=FireOn.SUCCESS,
            )
        )
        validation_config = ValidationConfig(validation_triggers=triggers)

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            timeout_minutes=1,
            max_issues=1,
        )
        orchestrator._validation_config = validation_config

        # Mock queue_trigger_validation
        orchestrator.run_coordinator.queue_trigger_validation = MagicMock()  # type: ignore[method-assign]

        # All success: 3 success out of 3 total
        await orchestrator._fire_run_end_trigger(success_count=3, total_count=3)

        orchestrator.run_coordinator.queue_trigger_validation.assert_called_once_with(
            TriggerType.RUN_END,
            {"success_count": 3, "total_count": 3},
        )

    @pytest.mark.asyncio
    async def test_fires_trigger_on_mixed_when_fire_on_success(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """Fires trigger when fire_on=SUCCESS and some issues succeeded (mixed).

        Per spec R7: fire_on=success fires if success_count > 0.
        """
        from src.domain.validation.config import (
            FailureMode,
            FireOn,
            RunEndTriggerConfig,
            TriggerType,
            ValidationConfig,
            ValidationTriggersConfig,
        )

        triggers = ValidationTriggersConfig(
            run_end=RunEndTriggerConfig(
                failure_mode=FailureMode.CONTINUE,
                commands=(),
                fire_on=FireOn.SUCCESS,
            )
        )
        validation_config = ValidationConfig(validation_triggers=triggers)

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            timeout_minutes=1,
            max_issues=1,
        )
        orchestrator._validation_config = validation_config

        orchestrator.run_coordinator.queue_trigger_validation = MagicMock()  # type: ignore[method-assign]
        orchestrator.run_coordinator.run_trigger_validation = AsyncMock(  # type: ignore[method-assign]
            return_value=MagicMock(status="passed")
        )

        # Mixed: 2 success out of 3 total - fires per spec R7 (success_count > 0)
        await orchestrator._fire_run_end_trigger(success_count=2, total_count=3)

        orchestrator.run_coordinator.queue_trigger_validation.assert_called_once_with(
            TriggerType.RUN_END,
            {"success_count": 2, "total_count": 3},
        )

    @pytest.mark.asyncio
    async def test_queues_trigger_on_failure_when_fire_on_failure(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """Queues trigger when fire_on=FAILURE and some issues failed."""
        from src.domain.validation.config import (
            FailureMode,
            FireOn,
            RunEndTriggerConfig,
            ValidationConfig,
            ValidationTriggersConfig,
        )
        from src.domain.validation.config import TriggerType

        triggers = ValidationTriggersConfig(
            run_end=RunEndTriggerConfig(
                failure_mode=FailureMode.CONTINUE,
                commands=(),
                fire_on=FireOn.FAILURE,
            )
        )
        validation_config = ValidationConfig(validation_triggers=triggers)

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            timeout_minutes=1,
            max_issues=1,
        )
        orchestrator._validation_config = validation_config

        orchestrator.run_coordinator.queue_trigger_validation = MagicMock()  # type: ignore[method-assign]

        # Some failure: 2 success out of 3 total
        await orchestrator._fire_run_end_trigger(success_count=2, total_count=3)

        orchestrator.run_coordinator.queue_trigger_validation.assert_called_once_with(
            TriggerType.RUN_END,
            {"success_count": 2, "total_count": 3},
        )

    @pytest.mark.asyncio
    async def test_skips_trigger_on_success_when_fire_on_failure(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """Skips trigger when fire_on=FAILURE but all issues succeeded."""
        from src.domain.validation.config import (
            FailureMode,
            FireOn,
            RunEndTriggerConfig,
            ValidationConfig,
            ValidationTriggersConfig,
        )

        triggers = ValidationTriggersConfig(
            run_end=RunEndTriggerConfig(
                failure_mode=FailureMode.CONTINUE,
                commands=(),
                fire_on=FireOn.FAILURE,
            )
        )
        validation_config = ValidationConfig(validation_triggers=triggers)

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            timeout_minutes=1,
            max_issues=1,
        )
        orchestrator._validation_config = validation_config

        orchestrator.run_coordinator.queue_trigger_validation = MagicMock()  # type: ignore[method-assign]

        # All success: 3 success out of 3 total
        await orchestrator._fire_run_end_trigger(success_count=3, total_count=3)

        orchestrator.run_coordinator.queue_trigger_validation.assert_not_called()

    @pytest.mark.asyncio
    async def test_queues_trigger_always_when_fire_on_both(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """Queues trigger on both success and failure when fire_on=BOTH."""
        from src.domain.validation.config import (
            FailureMode,
            FireOn,
            RunEndTriggerConfig,
            ValidationConfig,
            ValidationTriggersConfig,
        )

        triggers = ValidationTriggersConfig(
            run_end=RunEndTriggerConfig(
                failure_mode=FailureMode.CONTINUE,
                commands=(),
                fire_on=FireOn.BOTH,
            )
        )
        validation_config = ValidationConfig(validation_triggers=triggers)

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            timeout_minutes=1,
            max_issues=1,
        )
        orchestrator._validation_config = validation_config

        orchestrator.run_coordinator.queue_trigger_validation = MagicMock()  # type: ignore[method-assign]

        # Test with all success
        await orchestrator._fire_run_end_trigger(success_count=3, total_count=3)
        orchestrator.run_coordinator.queue_trigger_validation.assert_called_once()
        orchestrator.run_coordinator.queue_trigger_validation.reset_mock()

        # Test with some failure
        await orchestrator._fire_run_end_trigger(success_count=2, total_count=3)
        orchestrator.run_coordinator.queue_trigger_validation.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_when_abort_run_set(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """Skips trigger when abort_run is True."""
        from src.domain.validation.config import (
            FailureMode,
            FireOn,
            RunEndTriggerConfig,
            ValidationConfig,
            ValidationTriggersConfig,
        )

        triggers = ValidationTriggersConfig(
            run_end=RunEndTriggerConfig(
                failure_mode=FailureMode.CONTINUE,
                commands=(),
                fire_on=FireOn.BOTH,
            )
        )
        validation_config = ValidationConfig(validation_triggers=triggers)

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            timeout_minutes=1,
            max_issues=1,
        )
        orchestrator._validation_config = validation_config

        # Set abort
        orchestrator.issue_coordinator.request_abort(reason="test")

        orchestrator.run_coordinator.queue_trigger_validation = MagicMock()  # type: ignore[method-assign]

        await orchestrator._fire_run_end_trigger(success_count=3, total_count=3)

        orchestrator.run_coordinator.queue_trigger_validation.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_when_no_issues_processed(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """Skips trigger when total_count is 0 (no issues processed)."""
        from src.domain.validation.config import (
            FailureMode,
            FireOn,
            RunEndTriggerConfig,
            ValidationConfig,
            ValidationTriggersConfig,
        )

        triggers = ValidationTriggersConfig(
            run_end=RunEndTriggerConfig(
                failure_mode=FailureMode.CONTINUE,
                commands=(),
                fire_on=FireOn.BOTH,
            )
        )
        validation_config = ValidationConfig(validation_triggers=triggers)

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            timeout_minutes=1,
            max_issues=1,
        )
        orchestrator._validation_config = validation_config

        orchestrator.run_coordinator.queue_trigger_validation = MagicMock()  # type: ignore[method-assign]

        await orchestrator._fire_run_end_trigger(success_count=0, total_count=0)

        orchestrator.run_coordinator.queue_trigger_validation.assert_not_called()

    @pytest.mark.asyncio
    async def test_fires_on_total_failure_when_fire_on_failure(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """Fires trigger when fire_on=FAILURE and all issues failed (0 success)."""
        from src.domain.validation.config import (
            FailureMode,
            FireOn,
            RunEndTriggerConfig,
            ValidationConfig,
            ValidationTriggersConfig,
        )
        from src.domain.validation.config import TriggerType

        triggers = ValidationTriggersConfig(
            run_end=RunEndTriggerConfig(
                failure_mode=FailureMode.CONTINUE,
                commands=(),
                fire_on=FireOn.FAILURE,
            )
        )
        validation_config = ValidationConfig(validation_triggers=triggers)

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            timeout_minutes=1,
            max_issues=1,
        )
        orchestrator._validation_config = validation_config

        orchestrator.run_coordinator.queue_trigger_validation = MagicMock()  # type: ignore[method-assign]

        # Total failure: 0 success out of 3 total
        await orchestrator._fire_run_end_trigger(success_count=0, total_count=3)

        orchestrator.run_coordinator.queue_trigger_validation.assert_called_once_with(
            TriggerType.RUN_END,
            {"success_count": 0, "total_count": 3},
        )

    @pytest.mark.asyncio
    async def test_skips_when_run_end_not_configured(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """Skips trigger when run_end trigger is not configured."""
        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            timeout_minutes=1,
            max_issues=1,
        )

        orchestrator.run_coordinator.queue_trigger_validation = MagicMock()  # type: ignore[method-assign]

        await orchestrator._fire_run_end_trigger(success_count=3, total_count=3)

        orchestrator.run_coordinator.queue_trigger_validation.assert_not_called()


class TestBaseShaCapture:
    """Tests for base_sha capture at issue session start (T013)."""

    @pytest.mark.asyncio
    async def test_base_sha_captured_in_issue_result(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """IssueResult.base_sha is populated after issue session starts."""
        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            timeout_minutes=5,
        )
        # Patch config after creation to avoid 60s log wait timeout
        object.__setattr__(orchestrator._session_config, "log_file_wait_timeout", 0.5)

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.query = AsyncMock()

        async def mock_receive_response() -> AsyncGenerator[ResultMessage, None]:
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

        with (
            patch("claude_agent_sdk.ClaudeSDKClient", return_value=mock_client),
            patch(
                "src.orchestration.orchestrator.get_git_branch_async",
                return_value="main",
            ),
            patch(
                "src.orchestration.orchestrator.get_git_commit_async",
                return_value="abc123def456",
            ),
        ):
            result = await orchestrator.run_implementer("test-issue")

        # base_sha should be captured from get_git_commit_async
        assert result.base_sha == "abc123def456"

    @pytest.mark.asyncio
    async def test_base_sha_matches_head_at_session_start(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """base_sha matches the HEAD commit at session start time."""
        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            timeout_minutes=5,
        )
        object.__setattr__(orchestrator._session_config, "log_file_wait_timeout", 0.5)

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.query = AsyncMock()

        async def mock_receive_response() -> AsyncGenerator[ResultMessage, None]:
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

        captured_commit = "session_start_sha_789"

        with (
            patch("claude_agent_sdk.ClaudeSDKClient", return_value=mock_client),
            patch(
                "src.orchestration.orchestrator.get_git_branch_async",
                return_value="main",
            ),
            patch(
                "src.orchestration.orchestrator.get_git_commit_async",
                return_value=captured_commit,
            ),
        ):
            result = await orchestrator.run_implementer("test-issue")

        assert result.base_sha == captured_commit

    @pytest.mark.asyncio
    async def test_base_sha_immutable_across_retries(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """base_sha does not change when issue is retried (gate retry)."""
        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            timeout_minutes=5,
        )
        object.__setattr__(orchestrator._session_config, "log_file_wait_timeout", 0.5)

        # Simulate different HEAD commits on each call
        call_count = 0
        commit_values = ["first_sha_111", "second_sha_222"]

        async def changing_commit(*args: object, **kwargs: object) -> str:
            nonlocal call_count
            result = commit_values[min(call_count, len(commit_values) - 1)]
            call_count += 1
            return result

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.query = AsyncMock()

        async def mock_receive_response() -> AsyncGenerator[ResultMessage, None]:
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

        with (
            patch("claude_agent_sdk.ClaudeSDKClient", return_value=mock_client),
            patch(
                "src.orchestration.orchestrator.get_git_branch_async",
                return_value="main",
            ),
            patch(
                "src.orchestration.orchestrator.get_git_commit_async",
                side_effect=changing_commit,
            ),
        ):
            # First run captures base_sha
            result1 = await orchestrator.run_implementer("test-issue")

            # Second run (simulating retry) should use same base_sha
            result2 = await orchestrator.run_implementer("test-issue")

        # Both should have the first captured value
        assert result1.base_sha == "first_sha_111"
        assert result2.base_sha == "first_sha_111"  # Immutable - same as first

    @pytest.mark.asyncio
    async def test_base_sha_available_via_get_base_sha_callback(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """base_sha is accessible via session_callback_factory's get_base_sha."""
        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            timeout_minutes=5,
        )
        object.__setattr__(orchestrator._session_config, "log_file_wait_timeout", 0.5)

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.query = AsyncMock()

        async def mock_receive_response() -> AsyncGenerator[ResultMessage, None]:
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

        captured_sha = "callback_accessible_sha"

        with (
            patch("claude_agent_sdk.ClaudeSDKClient", return_value=mock_client),
            patch(
                "src.orchestration.orchestrator.get_git_branch_async",
                return_value="main",
            ),
            patch(
                "src.orchestration.orchestrator.get_git_commit_async",
                return_value=captured_sha,
            ),
        ):
            await orchestrator.run_implementer("test-issue")

        # The get_base_sha callback should return the captured value
        assert orchestrator._state.issue_base_shas.get("test-issue") == captured_sha

    @pytest.mark.asyncio
    async def test_base_sha_none_when_git_fails(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """base_sha is None when git commit retrieval fails."""
        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            timeout_minutes=5,
        )
        object.__setattr__(orchestrator._session_config, "log_file_wait_timeout", 0.5)

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.query = AsyncMock()

        async def mock_receive_response() -> AsyncGenerator[ResultMessage, None]:
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

        with (
            patch("claude_agent_sdk.ClaudeSDKClient", return_value=mock_client),
            patch(
                "src.orchestration.orchestrator.get_git_branch_async",
                return_value="main",
            ),
            patch(
                "src.orchestration.orchestrator.get_git_commit_async",
                return_value="",  # Empty string simulates git failure
            ),
        ):
            result = await orchestrator.run_implementer("test-issue")

        # base_sha should be None when git commit retrieval returns empty
        assert result.base_sha is None


class TestGetTrackReviewIssues:
    """Tests for _get_track_review_issues method."""

    def test_returns_false_when_code_review_disabled_but_track_false(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """track_review_issues=false should be respected even when code_review is disabled.

        This tests the fix for mala-6xp5: users should be able to disable
        track_review_issues without enabling code_review.
        """
        from src.domain.validation.config import (
            CodeReviewConfig,
            FailureMode,
            SessionEndTriggerConfig,
            ValidationConfig,
            ValidationTriggersConfig,
        )

        # Configure session_end with code_review disabled but track_review_issues=false
        code_review = CodeReviewConfig(enabled=False, track_review_issues=False)
        session_end = SessionEndTriggerConfig(
            failure_mode=FailureMode.CONTINUE, commands=(), code_review=code_review
        )
        triggers = ValidationTriggersConfig(session_end=session_end)
        validation_config = ValidationConfig(validation_triggers=triggers)

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            timeout_minutes=1,
            max_issues=1,
        )
        orchestrator._validation_config = validation_config

        # Should return False, not fall back to env var
        result = orchestrator._get_track_review_issues()
        assert result is False

    def test_returns_true_when_code_review_disabled_and_track_true(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """track_review_issues=true should be respected when code_review is disabled."""
        from src.domain.validation.config import (
            CodeReviewConfig,
            FailureMode,
            SessionEndTriggerConfig,
            ValidationConfig,
            ValidationTriggersConfig,
        )

        code_review = CodeReviewConfig(enabled=False, track_review_issues=True)
        session_end = SessionEndTriggerConfig(
            failure_mode=FailureMode.CONTINUE, commands=(), code_review=code_review
        )
        triggers = ValidationTriggersConfig(session_end=session_end)
        validation_config = ValidationConfig(validation_triggers=triggers)

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            timeout_minutes=1,
            max_issues=1,
        )
        orchestrator._validation_config = validation_config

        result = orchestrator._get_track_review_issues()
        assert result is True

    def test_returns_config_value_when_code_review_enabled(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """track_review_issues should work normally when code_review is enabled."""
        from src.domain.validation.config import (
            CodeReviewConfig,
            FailureMode,
            SessionEndTriggerConfig,
            ValidationConfig,
            ValidationTriggersConfig,
        )

        code_review = CodeReviewConfig(enabled=True, track_review_issues=False)
        session_end = SessionEndTriggerConfig(
            failure_mode=FailureMode.CONTINUE, commands=(), code_review=code_review
        )
        triggers = ValidationTriggersConfig(session_end=session_end)
        validation_config = ValidationConfig(validation_triggers=triggers)

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            timeout_minutes=1,
            max_issues=1,
        )
        orchestrator._validation_config = validation_config

        result = orchestrator._get_track_review_issues()
        assert result is False

    def test_falls_back_to_env_when_no_code_review_config(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """Falls back to env var when session_end has no code_review config."""
        from src.domain.validation.config import (
            FailureMode,
            SessionEndTriggerConfig,
            ValidationConfig,
            ValidationTriggersConfig,
        )

        # session_end exists but code_review is None
        session_end = SessionEndTriggerConfig(
            failure_mode=FailureMode.CONTINUE, commands=()
        )
        triggers = ValidationTriggersConfig(session_end=session_end)
        validation_config = ValidationConfig(validation_triggers=triggers)

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            timeout_minutes=1,
            max_issues=1,
        )
        orchestrator._validation_config = validation_config

        # Should fall back to env var (default is True via MalaConfig)
        result = orchestrator._get_track_review_issues()
        assert result is True


class TestIsReviewEnabled:
    """Tests for _is_review_enabled wiring through factory and session_config.

    These tests create mala.yaml files with per_issue_review config to verify
    the full wiring path: mala.yaml -> ValidationConfig -> _DerivedConfig ->
    orchestrator._per_issue_review -> _is_review_enabled() -> session_config.
    """

    def test_per_issue_review_disabled_wires_through_factory(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """per_issue_review.enabled=False wires correctly through factory."""
        from textwrap import dedent

        mala_yaml = tmp_path / "mala.yaml"
        mala_yaml.write_text(
            dedent("""\
            preset: python-uv
            per_issue_review:
              enabled: false
        """)
        )

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            timeout_minutes=1,
            max_issues=1,
        )

        # Verify wiring: _per_issue_review is set from factory
        assert orchestrator._per_issue_review is not None
        assert orchestrator._per_issue_review.enabled is False
        # Verify _is_review_enabled returns False
        assert orchestrator._is_review_enabled() is False
        # Verify session_config was built with review_enabled=False
        assert orchestrator._session_config.review_enabled is False

    def test_per_issue_review_enabled_wires_through_factory(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """per_issue_review.enabled=True wires correctly through factory.

        Uses reviewer_type: agent_sdk which doesn't require external binaries.
        """
        from textwrap import dedent

        mala_yaml = tmp_path / "mala.yaml"
        mala_yaml.write_text(
            dedent("""\
            preset: python-uv
            per_issue_review:
              enabled: true
              reviewer_type: agent_sdk
        """)
        )

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            timeout_minutes=1,
            max_issues=1,
        )

        # Verify wiring
        assert orchestrator._per_issue_review is not None
        assert orchestrator._per_issue_review.enabled is True
        assert orchestrator._is_review_enabled() is True
        assert orchestrator._session_config.review_enabled is True

    def test_missing_per_issue_review_defaults_to_disabled(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """Missing per_issue_review section defaults to review disabled."""
        from textwrap import dedent

        mala_yaml = tmp_path / "mala.yaml"
        mala_yaml.write_text(
            dedent("""\
            preset: python-uv
        """)
        )

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            timeout_minutes=1,
            max_issues=1,
        )

        # Verify default disabled behavior
        assert orchestrator._per_issue_review is not None
        assert orchestrator._per_issue_review.enabled is False
        assert orchestrator._is_review_enabled() is False
        assert orchestrator._session_config.review_enabled is False

    def test_no_mala_yaml_defaults_to_disabled(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """No mala.yaml file means _per_issue_review is None and review disabled."""
        # Don't create mala.yaml - factory will catch ConfigMissingError
        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            timeout_minutes=1,
            max_issues=1,
        )

        # Verify None when config missing
        assert orchestrator._per_issue_review is None
        assert orchestrator._is_review_enabled() is False
        assert orchestrator._session_config.review_enabled is False

    def test_cli_disable_validations_overrides_per_issue_review(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """CLI --disable-validations review overrides per_issue_review.enabled=True."""
        from textwrap import dedent

        mala_yaml = tmp_path / "mala.yaml"
        mala_yaml.write_text(
            dedent("""\
            preset: python-uv
            per_issue_review:
              enabled: true
              reviewer_type: agent_sdk
        """)
        )

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            timeout_minutes=1,
            max_issues=1,
            disable_validations={"review"},
        )

        # Config says enabled, but CLI disabled it
        assert orchestrator._per_issue_review is not None
        assert orchestrator._per_issue_review.enabled is True
        # _is_review_enabled respects CLI override
        assert orchestrator._is_review_enabled() is False
        assert orchestrator._session_config.review_enabled is False


class TestFreshSessionMode:
    """Tests for fresh_session mode in run_implementer."""

    @pytest.mark.asyncio
    async def test_fresh_session_clears_resume_id_but_keeps_review_prompt(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """When fresh_session=True and prior session exists with review issues,
        resume_session_id is None but resume prompt is still built."""
        fake_issues = FakeIssueProvider(
            {"test-issue": FakeIssue(id="test-issue", priority=1)}
        )
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        log_dir = tmp_path / ".claude" / "projects" / tmp_path.name
        log_dir.mkdir(parents=True)
        log_file = log_dir / "test-session.jsonl"
        log_file.write_text('{"type": "result"}\n')

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            issue_provider=fake_issues,
            runs_dir=runs_dir,
            lock_releaser=lambda _: 0,
            include_wip=True,
            fresh_session=True,
            log_provider=_make_mock_log_provider(log_file),  # type: ignore[arg-type]
        )

        # Create mock SDK client
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.query = AsyncMock()

        async def mock_receive_response() -> AsyncGenerator[ResultMessage, None]:
            yield ResultMessage(
                subtype="result",
                session_id="new-session-id",
                result="ISSUE_NO_CHANGE: Done",
                duration_ms=1000,
                duration_api_ms=800,
                is_error=False,
                num_turns=1,
                total_cost_usd=0.01,
                usage=None,
            )

        mock_client.receive_response = mock_receive_response

        # Track ALL calls to SDK client to verify initial call has resume=None
        sdk_call_history: list[dict[str, object]] = []

        def capture_sdk_client(**kwargs: object) -> AsyncMock:
            sdk_call_history.append(dict(kwargs))
            return mock_client

        with (
            patch("claude_agent_sdk.ClaudeSDKClient", side_effect=capture_sdk_client),
            patch(
                "src.orchestration.orchestrator.get_git_branch_async",
                return_value="main",
            ),
            patch(
                "src.orchestration.orchestrator.get_git_commit_async",
                return_value="abc123",
            ),
            patch.object(
                orchestrator.beads,
                "get_issue_description_async",
                return_value="Test issue",
            ),
            patch(
                "src.orchestration.orchestrator.lookup_prior_session_info",
                return_value=MagicMock(
                    session_id="prior-session-id",
                    baseline_timestamp=1700000000,
                    # Fixture schema matches StoredReviewIssue.from_dict contract
                    # (file, line_start, line_end, priority, title, body, reviewer)
                    last_review_issues=[
                        {
                            "file": "src/test.py",
                            "line_start": 10,
                            "line_end": 12,
                            "priority": 2,
                            "title": "Fix typo",
                            "body": "Found a typo in variable name",
                            "reviewer": "test-reviewer",
                        }
                    ],
                    run_id="prior-run-id",
                ),
            ),
        ):
            result = await orchestrator.run_implementer("test-issue")

            # Verify the FIRST SDK call had resume=None (fresh session)
            # Note: Subsequent calls may have resume set after receiving session_id
            # ClaudeSDKClient is called with options=<ClaudeAgentOptions> (a dataclass
            # with a 'resume' field), not as a dict, so getattr is correct here.
            assert len(sdk_call_history) >= 1, "SDK client should have been called"
            first_call_options = sdk_call_history[0].get("options")
            assert first_call_options is not None
            assert getattr(first_call_options, "resume", "NOT_FOUND") is None

            # Session ran successfully
            assert result.session_id == "new-session-id"

    @pytest.mark.asyncio
    async def test_fresh_session_with_no_prior_session(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """When fresh_session=True and no prior session exists, behavior is safe."""
        fake_issues = FakeIssueProvider(
            {"test-issue": FakeIssue(id="test-issue", priority=1)}
        )
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        log_dir = tmp_path / ".claude" / "projects" / tmp_path.name
        log_dir.mkdir(parents=True)
        log_file = log_dir / "test-session.jsonl"
        log_file.write_text('{"type": "result"}\n')

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            issue_provider=fake_issues,
            runs_dir=runs_dir,
            lock_releaser=lambda _: 0,
            include_wip=True,
            fresh_session=True,
            log_provider=_make_mock_log_provider(log_file),  # type: ignore[arg-type]
        )

        # Create mock SDK client
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.query = AsyncMock()

        async def mock_receive_response() -> AsyncGenerator[ResultMessage, None]:
            yield ResultMessage(
                subtype="result",
                session_id="new-session-id",
                result="ISSUE_NO_CHANGE: Done",
                duration_ms=1000,
                duration_api_ms=800,
                is_error=False,
                num_turns=1,
                total_cost_usd=0.01,
                usage=None,
            )

        mock_client.receive_response = mock_receive_response

        # Track ALL calls to SDK client to verify initial call has resume=None
        sdk_call_history: list[dict[str, object]] = []

        def capture_sdk_client(**kwargs: object) -> AsyncMock:
            sdk_call_history.append(dict(kwargs))
            return mock_client

        with (
            patch("claude_agent_sdk.ClaudeSDKClient", side_effect=capture_sdk_client),
            patch(
                "src.orchestration.orchestrator.get_git_branch_async",
                return_value="main",
            ),
            patch(
                "src.orchestration.orchestrator.get_git_commit_async",
                return_value="abc123",
            ),
            patch.object(
                orchestrator.beads,
                "get_issue_description_async",
                return_value="Test issue",
            ),
            patch(
                "src.orchestration.orchestrator.lookup_prior_session_info",
                return_value=None,  # No prior session
            ),
        ):
            result = await orchestrator.run_implementer("test-issue")

            # Verify the FIRST SDK call had resume=None (no prior session)
            # Note: Subsequent calls may have resume set after receiving session_id
            # ClaudeSDKClient is called with options=<ClaudeAgentOptions>, not resume_session_id
            assert len(sdk_call_history) >= 1, "SDK client should have been called"
            first_call_options = sdk_call_history[0].get("options")
            assert first_call_options is not None
            assert getattr(first_call_options, "resume", "NOT_FOUND") is None

            # Session ran successfully - no error due to missing session
            assert result.session_id == "new-session-id"
            assert "No prior session found" not in result.summary
