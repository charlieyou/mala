"""Tests for CerberusGateCLI subprocess management.

Tests the cerberus_gate_cli module including:
- Binary validation
- Environment merging
- Spawn/wait/resolve subprocess operations
- Timeout extraction
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

from src.infra.clients.cerberus_gate_cli import (
    CerberusGateCLI,
)
from src.infra.tools.command_runner import CommandResult
from tests.fakes.command_runner import FakeCommandRunner


class TestValidateBinary:
    """Tests for binary validation."""

    def test_validates_existing_executable_bin_path(self, tmp_path: Path) -> None:
        """Returns None when bin_path contains executable binary."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        binary = bin_dir / "review-gate"
        binary.touch()
        binary.chmod(0o755)

        cli = CerberusGateCLI(repo_path=tmp_path, bin_path=bin_dir)
        assert cli.validate_binary() is None

    def test_returns_error_for_missing_binary(self, tmp_path: Path) -> None:
        """Returns error when binary does not exist."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()

        cli = CerberusGateCLI(repo_path=tmp_path, bin_path=bin_dir)
        error = cli.validate_binary()
        assert error is not None
        assert "not found" in error

    def test_returns_error_for_non_executable_binary(self, tmp_path: Path) -> None:
        """Returns error when binary exists but is not executable."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        binary = bin_dir / "review-gate"
        binary.touch()
        binary.chmod(0o644)  # Not executable

        cli = CerberusGateCLI(repo_path=tmp_path, bin_path=bin_dir)
        error = cli.validate_binary()
        assert error is not None
        assert "not executable" in error

    def test_validates_binary_in_path_env(self, tmp_path: Path) -> None:
        """Returns None when binary is in PATH from env."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        binary = bin_dir / "review-gate"
        binary.touch()
        binary.chmod(0o755)

        cli = CerberusGateCLI(
            repo_path=tmp_path,
            bin_path=None,
            env={"PATH": str(bin_dir)},
        )
        assert cli.validate_binary() is None

    def test_returns_error_when_not_in_path(self, tmp_path: Path) -> None:
        """Returns error when bare binary is not in PATH."""
        cli = CerberusGateCLI(
            repo_path=tmp_path,
            bin_path=None,
            env={"PATH": "/nonexistent/path"},
        )
        error = cli.validate_binary()
        assert error is not None
        assert "not found in PATH" in error


class TestBuildEnv:
    """Tests for environment merging."""

    def test_sets_claude_session_id_from_parameter(self, tmp_path: Path) -> None:
        """CLAUDE_SESSION_ID from parameter takes priority."""
        cli = CerberusGateCLI(repo_path=tmp_path)
        env = cli.build_env("param-session")
        assert env["CLAUDE_SESSION_ID"] == "param-session"

    def test_sets_claude_session_id_from_env_field(self, tmp_path: Path) -> None:
        """CLAUDE_SESSION_ID from self.env is used if no parameter."""
        cli = CerberusGateCLI(
            repo_path=tmp_path,
            env={"CLAUDE_SESSION_ID": "field-session"},
        )
        env = cli.build_env(None)
        assert env["CLAUDE_SESSION_ID"] == "field-session"

    def test_sets_claude_session_id_from_os_environ(self, tmp_path: Path) -> None:
        """CLAUDE_SESSION_ID from os.environ is used as fallback."""
        cli = CerberusGateCLI(repo_path=tmp_path)
        with patch.dict(os.environ, {"CLAUDE_SESSION_ID": "environ-session"}):
            env = cli.build_env(None)
        assert env["CLAUDE_SESSION_ID"] == "environ-session"

    def test_parameter_overrides_env_field(self, tmp_path: Path) -> None:
        """Parameter takes priority over self.env."""
        cli = CerberusGateCLI(
            repo_path=tmp_path,
            env={"CLAUDE_SESSION_ID": "field-session"},
        )
        env = cli.build_env("param-session")
        assert env["CLAUDE_SESSION_ID"] == "param-session"

    def test_preserves_other_env_vars(self, tmp_path: Path) -> None:
        """Other env vars from self.env are preserved."""
        cli = CerberusGateCLI(
            repo_path=tmp_path,
            env={"FOO": "bar", "BAZ": "qux"},
        )
        env = cli.build_env("session")
        assert env["FOO"] == "bar"
        assert env["BAZ"] == "qux"


class TestExtractWaitTimeout:
    """Tests for timeout extraction from wait args."""

    def test_returns_none_for_empty_args(self) -> None:
        """Returns None when args is empty."""
        assert CerberusGateCLI.extract_wait_timeout(()) is None

    def test_returns_none_when_no_timeout_flag(self) -> None:
        """Returns None when --timeout is not present."""
        args = ("--json", "--session-key", "abc123")
        assert CerberusGateCLI.extract_wait_timeout(args) is None

    def test_extracts_timeout_with_equals_format(self) -> None:
        """Extracts timeout from --timeout=VALUE format."""
        args = ("--json", "--timeout=600", "--session-key", "abc123")
        assert CerberusGateCLI.extract_wait_timeout(args) == 600

    def test_extracts_timeout_with_space_format(self) -> None:
        """Extracts timeout from --timeout VALUE format."""
        args = ("--json", "--timeout", "300", "--session-key", "abc123")
        assert CerberusGateCLI.extract_wait_timeout(args) == 300

    def test_returns_none_for_non_numeric_equals_value(self) -> None:
        """Returns None when --timeout=VALUE has non-numeric value."""
        args = ("--timeout=abc",)
        assert CerberusGateCLI.extract_wait_timeout(args) is None

    def test_returns_none_for_non_numeric_space_value(self) -> None:
        """Returns None when --timeout VALUE has non-numeric value."""
        args = ("--timeout", "abc")
        assert CerberusGateCLI.extract_wait_timeout(args) is None

    def test_returns_none_for_timeout_at_end_without_value(self) -> None:
        """Returns None when --timeout is at end without value."""
        args = ("--json", "--timeout")
        assert CerberusGateCLI.extract_wait_timeout(args) is None


class TestSpawnCodeReview:
    """Tests for spawn-code-review subprocess."""

    @pytest.mark.asyncio
    async def test_spawns_successfully(self, tmp_path: Path) -> None:
        """Returns success when spawn exits 0."""
        cli = CerberusGateCLI(repo_path=tmp_path)
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
        cli = CerberusGateCLI(repo_path=tmp_path)

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
        cli = CerberusGateCLI(repo_path=tmp_path)

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
        cli = CerberusGateCLI(repo_path=tmp_path)

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
        cli = CerberusGateCLI(repo_path=tmp_path)
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
        cli = CerberusGateCLI(repo_path=tmp_path)
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
        cli = CerberusGateCLI(repo_path=tmp_path)
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


class TestWaitForReview:
    """Tests for wait subprocess."""

    @pytest.mark.asyncio
    async def test_waits_successfully(self, tmp_path: Path) -> None:
        """Returns WaitResult with output on success."""
        cli = CerberusGateCLI(repo_path=tmp_path)

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
            session_id="test-session",
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
        cli = CerberusGateCLI(repo_path=tmp_path)

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
            session_id="test-session",
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
        cli = CerberusGateCLI(repo_path=tmp_path)
        runner = FakeCommandRunner(allow_unregistered=True)

        await cli.wait_for_review(
            session_id="test-session",
            runner=runner,
            env={"CLAUDE_SESSION_ID": "test-session"},
            cli_timeout=300,
            wait_args=("--extra", "arg"),
        )

        assert len(runner.calls) == 1
        call_args = runner.calls[0][0]
        assert "--extra" in call_args
        assert "arg" in call_args

    @pytest.mark.asyncio
    async def test_uses_user_timeout_when_provided(self, tmp_path: Path) -> None:
        """Uses user-specified timeout from wait_args, does not add cli_timeout."""
        cli = CerberusGateCLI(repo_path=tmp_path)
        runner = FakeCommandRunner(allow_unregistered=True)

        await cli.wait_for_review(
            session_id="test-session",
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


class TestResolveGate:
    """Tests for resolve subprocess."""

    @pytest.mark.asyncio
    async def test_resolves_successfully(self, tmp_path: Path) -> None:
        """Returns success when resolve exits 0."""
        cli = CerberusGateCLI(repo_path=tmp_path)
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
        cli = CerberusGateCLI(repo_path=tmp_path)

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
        cli = CerberusGateCLI(repo_path=tmp_path)

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
        cli = CerberusGateCLI(repo_path=tmp_path)
        runner = FakeCommandRunner(allow_unregistered=True)

        await cli.resolve_gate(
            runner=runner,
            env={"CLAUDE_SESSION_ID": "test"},
            reason="Custom reason",
        )

        assert len(runner.calls) == 1
        call_args = runner.calls[0][0]
        assert "--reason" in call_args
        assert "Custom reason" in call_args
