"""Tests for CerberusCLI subprocess management.

Tests the cerberus_cli module including:
- Binary validation
- Environment merging
- Spawn/wait/resolve subprocess operations
- Timeout extraction
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

from src.infra.clients.cerberus_cli import (
    CerberusCLI,
    CerberusGateCLI,
    WaitResult,
)
from src.infra.tools.command_runner import CommandResult
from tests.fakes.command_runner import FakeCommandRunner


class TestValidateBinary:
    """Tests for binary validation."""

    def _make_cerberus_root(self, tmp_path: Path) -> tuple[Path, Path]:
        root = tmp_path / "cerberus-root"
        bin_dir = root / "bin"
        prompts_dir = root / "prompts" / "reviewers"
        bin_dir.mkdir(parents=True)
        prompts_dir.mkdir(parents=True)
        binary = bin_dir / "cerberus"
        binary.touch()
        binary.chmod(0o755)
        return root, bin_dir

    def test_binary_name_is_path_only_cerberus(self, tmp_path: Path) -> None:
        """Returns the v2 cerberus binary name."""
        cli = CerberusCLI(repo_path=tmp_path)
        assert cli._cerberus_bin() == "cerberus"

    def test_validates_binary_in_path_env(self, tmp_path: Path) -> None:
        """Returns None when binary is in PATH from env."""
        _, bin_dir = self._make_cerberus_root(tmp_path)

        cli = CerberusCLI(
            repo_path=tmp_path,
            env={"PATH": str(bin_dir)},
        )
        assert cli.validate_binary() is None

    def test_returns_error_when_not_in_path(self, tmp_path: Path) -> None:
        """Returns error when bare binary is not in PATH."""
        cli = CerberusCLI(
            repo_path=tmp_path,
            env={"PATH": "/nonexistent/path"},
        )
        assert cli.validate_binary() == "cerberus binary not found in PATH"

    def test_returns_error_when_root_missing_prompts(self, tmp_path: Path) -> None:
        """Returns distinct error when binary exists but root lacks prompts."""
        root = tmp_path / "cerberus-root"
        bin_dir = root / "bin"
        bin_dir.mkdir(parents=True)
        binary = bin_dir / "cerberus"
        binary.touch()
        binary.chmod(0o755)

        cli = CerberusCLI(repo_path=tmp_path, env={"PATH": str(bin_dir)})
        assert (
            cli.validate_binary() == f"cerberus root {root} missing prompts/reviewers/"
        )

    def test_compat_wrapper_validates_path_binary_when_bin_path_configured(
        self, tmp_path: Path
    ) -> None:
        """Legacy bin_path must not bypass the v2 PATH/root validation."""
        legacy_bin_path = tmp_path / "legacy-bin"
        legacy_bin_path.mkdir()

        cli = CerberusGateCLI(
            repo_path=tmp_path,
            bin_path=legacy_bin_path,
            env={"PATH": "/nonexistent/path"},
        )

        assert cli.validate_binary() == "cerberus binary not found in PATH"


class TestBuildEnv:
    """Tests for environment merging."""

    def test_builds_required_v2_env_keys(self, tmp_path: Path) -> None:
        """Builds required cerberus v2 environment keys."""
        root = tmp_path / "cerberus-root"
        (root / "prompts" / "reviewers").mkdir(parents=True)
        cli = CerberusCLI(repo_path=tmp_path, env={"CERBERUS_ROOT": str(root)})

        env = cli.build_env(
            run_key="run-1",
            state_root=tmp_path / "state",
            project_key="project-1",
            claude_session_id="claude-1",
        )

        assert env["CERBERUS_HOST"] == "generic"
        assert env["CERBERUS_ROOT"] == str(root)
        assert env["CERBERUS_RUN_KEY"] == "run-1"
        assert env["CERBERUS_STATE_ROOT"] == str(tmp_path / "state")
        assert env["CERBERUS_PROJECT_KEY"] == "project-1"
        assert env["CLAUDE_SESSION_ID"] == "claude-1"

    def test_user_project_key_overrides_default(self, tmp_path: Path) -> None:
        """User cerberus.env project key wins over mala default."""
        cli = CerberusCLI(
            repo_path=tmp_path,
            env={"CERBERUS_PROJECT_KEY": "user-project"},
        )

        env = cli.build_env(
            run_key="run-1",
            state_root=tmp_path / "state",
            project_key="mala-project",
        )

        assert env["CERBERUS_PROJECT_KEY"] == "user-project"

    def test_mala_owned_run_key_overrides_user_env(self, tmp_path: Path) -> None:
        """User cerberus.env run key cannot override mala value."""
        cli = CerberusCLI(
            repo_path=tmp_path,
            env={"CERBERUS_RUN_KEY": "user-run"},
        )

        env = cli.build_env(
            run_key="mala-run",
            state_root=tmp_path / "state",
            project_key="project",
        )

        assert env["CERBERUS_RUN_KEY"] == "mala-run"

    def test_drops_claude_session_id_when_not_passed(self, tmp_path: Path) -> None:
        """CLAUDE_SESSION_ID is kept only when caller passes one."""
        cli = CerberusCLI(
            repo_path=tmp_path,
            env={"CLAUDE_SESSION_ID": "field-session"},
        )

        env = cli.build_env(
            run_key="run-1",
            state_root=tmp_path / "state",
            project_key="project",
        )

        assert "CLAUDE_SESSION_ID" not in env

    def test_preserves_other_env_vars(self, tmp_path: Path) -> None:
        """Other env vars from self.env are preserved."""
        cli = CerberusCLI(
            repo_path=tmp_path,
            env={"FOO": "bar", "BAZ": "qux"},
        )
        env = cli.build_env(
            run_key="run-1",
            state_root=tmp_path / "state",
            project_key="project",
        )
        assert env["FOO"] == "bar"
        assert env["BAZ"] == "qux"


class TestExtractWaitTimeout:
    """Tests for timeout extraction from wait args."""

    def test_returns_none_for_empty_args(self) -> None:
        """Returns None when args is empty."""
        assert CerberusCLI.extract_wait_timeout(()) is None

    def test_returns_none_when_no_timeout_flag(self) -> None:
        """Returns None when --timeout is not present."""
        args = ("--json", "--session-key", "abc123")
        assert CerberusCLI.extract_wait_timeout(args) is None

    def test_extracts_timeout_with_equals_format(self) -> None:
        """Extracts timeout from --timeout=VALUE format."""
        args = ("--json", "--timeout=600", "--session-key", "abc123")
        assert CerberusCLI.extract_wait_timeout(args) == 600

    def test_extracts_timeout_with_space_format(self) -> None:
        """Extracts timeout from --timeout VALUE format."""
        args = ("--json", "--timeout", "300", "--session-key", "abc123")
        assert CerberusCLI.extract_wait_timeout(args) == 300

    def test_returns_none_for_non_numeric_equals_value(self) -> None:
        """Returns None when --timeout=VALUE has non-numeric value."""
        args = ("--timeout=abc",)
        assert CerberusCLI.extract_wait_timeout(args) is None

    def test_returns_none_for_non_numeric_space_value(self) -> None:
        """Returns None when --timeout VALUE has non-numeric value."""
        args = ("--timeout", "abc")
        assert CerberusCLI.extract_wait_timeout(args) is None

    def test_returns_none_for_timeout_at_end_without_value(self) -> None:
        """Returns None when --timeout is at end without value."""
        args = ("--json", "--timeout")
        assert CerberusCLI.extract_wait_timeout(args) is None


class TestSpawnCodeReview:
    """Tests for spawn-code-review subprocess."""

    @pytest.mark.asyncio
    async def test_spawns_successfully(self, tmp_path: Path) -> None:
        """Returns success when spawn exits 0."""
        cli = CerberusCLI(repo_path=tmp_path)
        runner = FakeCommandRunner(allow_unregistered=True)

        result = await cli.spawn_code_review(
            runner=runner,
            env={"CLAUDE_SESSION_ID": "test"},
            timeout=300,
            commit_shas=["commit1"],
        )

        assert result.success is True
        assert result.timed_out is False
        assert result.already_active is False

    @pytest.mark.asyncio
    async def test_returns_timeout_on_timeout(self, tmp_path: Path) -> None:
        """Returns timed_out when spawn times out."""
        cli = CerberusCLI(repo_path=tmp_path)

        # Override default response with timeout response
        class TimeoutRunner(FakeCommandRunner):
            async def run_async(
                self,
                cmd: list[str] | str,
                env: Mapping[str, str] | None = None,
                timeout: float | None = None,
                use_process_group: bool | None = None,
                shell: bool = False,
                cwd: Path | None = None,
            ) -> CommandResult:
                return CommandResult(
                    command=list(cmd) if not isinstance(cmd, str) else cmd,
                    returncode=124,
                    timed_out=True,
                )

        timeout_runner = TimeoutRunner()

        result = await cli.spawn_code_review(
            runner=timeout_runner,
            env={"CLAUDE_SESSION_ID": "test"},
            timeout=300,
            commit_shas=["commit1"],
        )

        assert result.success is False
        assert result.timed_out is True
        assert result.error_detail == "spawn timeout"

    @pytest.mark.asyncio
    async def test_detects_already_active(self, tmp_path: Path) -> None:
        """Returns already_active when spawn fails with already active error."""
        cli = CerberusCLI(repo_path=tmp_path)

        class AlreadyActiveRunner(FakeCommandRunner):
            async def run_async(
                self,
                cmd: list[str] | str,
                env: Mapping[str, str] | None = None,
                timeout: float | None = None,
                use_process_group: bool | None = None,
                shell: bool = False,
                cwd: Path | None = None,
            ) -> CommandResult:
                return CommandResult(
                    command=list(cmd) if not isinstance(cmd, str) else cmd,
                    returncode=1,
                    timed_out=False,
                    stderr="Error: session already active",
                )

        runner = AlreadyActiveRunner()

        result = await cli.spawn_code_review(
            runner=runner,
            env={"CLAUDE_SESSION_ID": "test"},
            timeout=300,
            commit_shas=["commit1"],
        )

        assert result.success is False
        assert result.already_active is True
        assert "already active" in result.error_detail

    @pytest.mark.asyncio
    async def test_returns_error_on_failure(self, tmp_path: Path) -> None:
        """Returns error detail when spawn fails."""
        cli = CerberusCLI(repo_path=tmp_path)

        class FailingRunner(FakeCommandRunner):
            async def run_async(
                self,
                cmd: list[str] | str,
                env: Mapping[str, str] | None = None,
                timeout: float | None = None,
                use_process_group: bool | None = None,
                shell: bool = False,
                cwd: Path | None = None,
            ) -> CommandResult:
                return CommandResult(
                    command=list(cmd) if not isinstance(cmd, str) else cmd,
                    returncode=1,
                    timed_out=False,
                    stderr="Some other error",
                    stdout="",
                )

        runner = FailingRunner()

        result = await cli.spawn_code_review(
            runner=runner,
            env={"CLAUDE_SESSION_ID": "test"},
            timeout=300,
            commit_shas=["commit1"],
        )

        assert result.success is False
        assert result.already_active is False
        assert "Some other error" in result.error_detail

    @pytest.mark.asyncio
    async def test_includes_context_file(self, tmp_path: Path) -> None:
        """Includes --context-file when provided."""
        cli = CerberusCLI(repo_path=tmp_path)
        runner = FakeCommandRunner(allow_unregistered=True)

        context_file = tmp_path / "context.md"
        await cli.spawn_code_review(
            runner=runner,
            env={"CLAUDE_SESSION_ID": "test"},
            timeout=300,
            context_file=context_file,
            commit_shas=["commit1"],
        )

        assert len(runner.calls) == 1
        call_args = runner.calls[0][0]
        assert "--context-file" in call_args
        assert str(context_file) in call_args

    @pytest.mark.asyncio
    async def test_includes_spawn_args(self, tmp_path: Path) -> None:
        """Includes extra spawn args when provided."""
        cli = CerberusCLI(repo_path=tmp_path)
        runner = FakeCommandRunner(allow_unregistered=True)

        await cli.spawn_code_review(
            runner=runner,
            env={"CLAUDE_SESSION_ID": "test"},
            timeout=300,
            spawn_args=("--extra", "arg"),
            commit_shas=["commit1"],
        )

        assert len(runner.calls) == 1
        call_args = runner.calls[0][0]
        assert "--extra" in call_args
        assert "arg" in call_args

    @pytest.mark.asyncio
    async def test_uses_commit_mode(self, tmp_path: Path) -> None:
        """Uses --commit mode when commit_shas provided."""
        cli = CerberusCLI(repo_path=tmp_path)
        runner = FakeCommandRunner(allow_unregistered=True)

        await cli.spawn_code_review(
            runner=runner,
            env={"CLAUDE_SESSION_ID": "test"},
            timeout=300,
            commit_shas=["abc123", "def456"],
        )

        assert len(runner.calls) == 1
        call_args = runner.calls[0][0]
        assert "--commit" in call_args
        assert "abc123" in call_args
        assert "def456" in call_args
        assert "baseline..HEAD" not in call_args

    @pytest.mark.asyncio
    async def test_uses_v2_argv_without_max_rounds(self, tmp_path: Path) -> None:
        """Uses cerberus binary and does not pass rejected v1 max-rounds flag."""
        cli = CerberusCLI(repo_path=tmp_path)
        runner = FakeCommandRunner(allow_unregistered=True)

        await cli.spawn_code_review(
            runner=runner,
            env={"CERBERUS_RUN_KEY": "run-1"},
            timeout=300,
            commit_shas=["abc123"],
        )

        call_args = runner.calls[0][0]
        assert call_args[:2] == ("cerberus", "spawn-code-review")
        assert "--max-rounds" not in call_args
        assert "0" not in call_args


class TestWaitForReview:
    """Tests for wait subprocess."""

    @pytest.mark.asyncio
    async def test_waits_successfully(self, tmp_path: Path) -> None:
        """Returns WaitResult with output on success."""
        cli = CerberusCLI(repo_path=tmp_path)

        class SuccessRunner(FakeCommandRunner):
            async def run_async(
                self,
                cmd: list[str] | str,
                env: Mapping[str, str] | None = None,
                timeout: float | None = None,
                use_process_group: bool | None = None,
                shell: bool = False,
                cwd: Path | None = None,
            ) -> CommandResult:
                return CommandResult(
                    command=list(cmd) if not isinstance(cmd, str) else cmd,
                    returncode=0,
                    stdout='{"consensus_verdict": "PASS"}',
                    stderr="",
                    timed_out=False,
                )

        runner = SuccessRunner()

        result = await cli.wait_for_review(
            run_key="test-run",
            runner=runner,
            env={"CLAUDE_SESSION_ID": "test-session"},
            cli_timeout=300,
        )

        assert result.returncode == 0
        assert "PASS" in result.stdout
        assert result.timed_out is False

    @pytest.mark.asyncio
    async def test_returns_timeout_on_timeout(self, tmp_path: Path) -> None:
        """Returns timed_out when wait times out."""
        cli = CerberusCLI(repo_path=tmp_path)

        class TimeoutRunner(FakeCommandRunner):
            async def run_async(
                self,
                cmd: list[str] | str,
                env: Mapping[str, str] | None = None,
                timeout: float | None = None,
                use_process_group: bool | None = None,
                shell: bool = False,
                cwd: Path | None = None,
            ) -> CommandResult:
                return CommandResult(
                    command=list(cmd) if not isinstance(cmd, str) else cmd,
                    returncode=124,
                    stdout="",
                    stderr="",
                    timed_out=True,
                )

        runner = TimeoutRunner()

        result = await cli.wait_for_review(
            run_key="test-run",
            runner=runner,
            env={"CLAUDE_SESSION_ID": "test-session"},
            cli_timeout=300,
        )

        assert result.returncode == 124
        assert result.stdout == ""
        assert result.timed_out is True

    @pytest.mark.asyncio
    async def test_includes_wait_args(self, tmp_path: Path) -> None:
        """Includes extra wait args when provided."""
        cli = CerberusCLI(repo_path=tmp_path)
        runner = FakeCommandRunner(allow_unregistered=True)

        await cli.wait_for_review(
            run_key="test-run",
            runner=runner,
            env={"CLAUDE_SESSION_ID": "test-session"},
            cli_timeout=300,
            wait_args=("--extra", "arg"),
        )

        assert len(runner.calls) == 1
        call_args = runner.calls[0][0]
        assert "--session-key" in call_args
        assert "test-run" in call_args
        assert "--session-id" not in call_args
        assert "--extra" in call_args
        assert "arg" in call_args

    @pytest.mark.asyncio
    async def test_uses_user_timeout_when_provided(self, tmp_path: Path) -> None:
        """Uses user-specified timeout from wait_args, does not add cli_timeout."""
        cli = CerberusCLI(repo_path=tmp_path)
        runner = FakeCommandRunner(allow_unregistered=True)

        await cli.wait_for_review(
            run_key="test-run",
            runner=runner,
            env={"CLAUDE_SESSION_ID": "test-session"},
            cli_timeout=300,
            wait_args=("--timeout", "600"),  # User timeout in wait_args
            user_timeout=600,  # Signals that wait_args already has timeout
        )

        assert len(runner.calls) == 1
        call_args = runner.calls[0][0]
        # Should include exactly one --timeout (the user's 600), not the cli_timeout (300)
        timeout_indices = [i for i, arg in enumerate(call_args) if arg == "--timeout"]
        assert len(timeout_indices) == 1
        timeout_idx = timeout_indices[0]
        assert call_args[timeout_idx + 1] == "600"  # User's timeout value

    def test_wait_result_has_exactly_raw_fields(self) -> None:
        """WaitResult has no pre-parsed v1 gate-state fields."""
        assert tuple(WaitResult.__dataclass_fields__) == (
            "returncode",
            "stdout",
            "stderr",
            "timed_out",
        )


class TestCerberusGateCLICompat:
    """Tests for the temporary adapter compatibility wrapper."""

    @pytest.mark.asyncio
    async def test_wait_uses_session_key_from_run_key(self, tmp_path: Path) -> None:
        """Legacy wait wrapper still invokes the v2 wait addressing contract."""
        cli = CerberusGateCLI(repo_path=tmp_path)
        runner = FakeCommandRunner(allow_unregistered=True)

        await cli.wait_for_review(
            session_id="claude-session",
            runner=runner,
            env={
                "CERBERUS_RUN_KEY": "mala-claude-session",
                "CERBERUS_STATE_ROOT": str(tmp_path / "state"),
                "CERBERUS_PROJECT_KEY": "project",
            },
            cli_timeout=300,
        )

        call_args = runner.calls[0][0]
        assert call_args[:2] == ("cerberus", "wait")
        assert "--session-key" in call_args
        assert "mala-claude-session" in call_args
        assert "--session-id" not in call_args


class TestSpawnEpicVerify:
    """Tests for spawn-epic-verify subprocess."""

    @pytest.mark.asyncio
    async def test_uses_v2_argv_without_max_rounds(self, tmp_path: Path) -> None:
        """Uses cerberus binary and does not pass rejected v1 max-rounds flag."""
        cli = CerberusCLI(repo_path=tmp_path)
        runner = FakeCommandRunner(allow_unregistered=True)
        epic_path = tmp_path / "EPIC.md"

        await cli.spawn_epic_verify(
            runner=runner,
            env={"CERBERUS_RUN_KEY": "run-1"},
            timeout=300,
            epic_path=epic_path,
            spawn_args=("--extra", "arg"),
        )

        call_args = runner.calls[0][0]
        assert call_args[:2] == ("cerberus", "spawn-epic-verify")
        assert "--extra" in call_args
        assert "arg" in call_args
        assert str(epic_path) in call_args
        assert "--max-rounds" not in call_args
        assert "0" not in call_args


class TestResolveGate:
    """Tests for resolve subprocess."""

    @pytest.mark.asyncio
    async def test_resolves_successfully(self, tmp_path: Path) -> None:
        """Returns success when resolve exits 0."""
        cli = CerberusCLI(repo_path=tmp_path)
        runner = FakeCommandRunner(allow_unregistered=True)

        result = await cli.resolve_gate(
            runner=runner,
            env={"CLAUDE_SESSION_ID": "test"},
        )

        assert result.success is True
        assert result.error_detail == ""

    @pytest.mark.asyncio
    async def test_returns_error_on_failure(self, tmp_path: Path) -> None:
        """Returns error detail when resolve fails."""
        cli = CerberusCLI(repo_path=tmp_path)

        class FailingRunner(FakeCommandRunner):
            async def run_async(
                self,
                cmd: list[str] | str,
                env: Mapping[str, str] | None = None,
                timeout: float | None = None,
                use_process_group: bool | None = None,
                shell: bool = False,
                cwd: Path | None = None,
            ) -> CommandResult:
                return CommandResult(
                    command=list(cmd) if not isinstance(cmd, str) else cmd,
                    returncode=1,
                    stderr="Permission denied",
                    stdout="",
                )

        runner = FailingRunner()

        result = await cli.resolve_gate(
            runner=runner,
            env={"CLAUDE_SESSION_ID": "test"},
        )

        assert result.success is False
        assert "Permission denied" in result.error_detail

    @pytest.mark.asyncio
    async def test_returns_error_on_exception(self, tmp_path: Path) -> None:
        """Returns error detail when exception is raised."""
        cli = CerberusCLI(repo_path=tmp_path)

        class ExceptionRaisingRunner(FakeCommandRunner):
            async def run_async(
                self,
                cmd: list[str] | str,
                env: Mapping[str, str] | None = None,
                timeout: float | None = None,
                use_process_group: bool | None = None,
                shell: bool = False,
                cwd: Path | None = None,
            ) -> CommandResult:
                raise Exception("Network error")

        runner = ExceptionRaisingRunner()

        result = await cli.resolve_gate(
            runner=runner,
            env={"CLAUDE_SESSION_ID": "test"},
        )

        assert result.success is False
        assert "Network error" in result.error_detail

    @pytest.mark.asyncio
    async def test_uses_custom_reason(self, tmp_path: Path) -> None:
        """Uses custom reason when provided."""
        cli = CerberusCLI(repo_path=tmp_path)
        runner = FakeCommandRunner(allow_unregistered=True)

        await cli.resolve_gate(
            runner=runner,
            env={"CLAUDE_SESSION_ID": "test"},
            reason="Custom reason",
        )

        assert len(runner.calls) == 1
        call_args = runner.calls[0][0]
        assert call_args[:2] == ("cerberus", "resolve")
        assert "--reason" in call_args
        assert "Custom reason" in call_args
