"""Unit tests for src/validation.py - post-commit validation runner.

These tests mock subprocess and shutil.which to test validation logic
without actually running commands or creating git worktrees.
"""

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from src.validation import (
    ValidationConfig,
    ValidationResult,
    ValidationRunner,
    ValidationStepResult,
    _check_e2e_prereqs,
    _format_step_output,
    _tail,
)


class TestValidationConfig:
    """Test ValidationConfig dataclass defaults."""

    def test_default_values(self) -> None:
        config = ValidationConfig()
        assert config.run_slow_tests is True
        assert config.run_e2e is True
        assert config.coverage is True
        assert config.coverage_min == 85
        assert config.coverage_source == "."
        assert config.use_test_mutex is False  # Disabled by default to avoid deadlocks
        assert config.step_timeout_seconds is None

    def test_custom_values(self) -> None:
        config = ValidationConfig(
            run_slow_tests=False,
            run_e2e=False,
            coverage=False,
            coverage_min=90,
            coverage_source="src",
            use_test_mutex=False,
            step_timeout_seconds=60.0,
        )
        assert config.run_slow_tests is False
        assert config.run_e2e is False
        assert config.coverage is False
        assert config.coverage_min == 90
        assert config.coverage_source == "src"
        assert config.use_test_mutex is False
        assert config.step_timeout_seconds == 60.0


class TestValidationStepResult:
    """Test ValidationStepResult dataclass."""

    def test_basic_result(self) -> None:
        result = ValidationStepResult(
            name="test",
            command=["pytest"],
            ok=True,
            returncode=0,
        )
        assert result.name == "test"
        assert result.command == ["pytest"]
        assert result.ok is True
        assert result.returncode == 0
        assert result.stdout_tail == ""
        assert result.stderr_tail == ""
        assert result.duration_seconds == 0.0

    def test_failed_result_with_output(self) -> None:
        result = ValidationStepResult(
            name="ruff",
            command=["ruff", "check", "."],
            ok=False,
            returncode=1,
            stdout_tail="error.py:10:1: E501 line too long",
            stderr_tail="",
            duration_seconds=0.5,
        )
        assert result.ok is False
        assert result.returncode == 1
        assert "line too long" in result.stdout_tail


class TestValidationResult:
    """Test ValidationResult dataclass and methods."""

    def test_passed_result(self) -> None:
        result = ValidationResult(passed=True)
        assert result.passed is True
        assert result.steps == []
        assert result.failure_reasons == []
        assert result.retriable is True

    def test_failed_result(self) -> None:
        result = ValidationResult(
            passed=False,
            failure_reasons=["pytest failed (exit 1)"],
            retriable=True,
        )
        assert result.passed is False
        assert "pytest failed" in result.failure_reasons[0]

    def test_short_summary_passed(self) -> None:
        result = ValidationResult(passed=True)
        assert result.short_summary() == "passed"

    def test_short_summary_failed_no_reasons(self) -> None:
        result = ValidationResult(passed=False)
        assert result.short_summary() == "failed"

    def test_short_summary_failed_with_reasons(self) -> None:
        result = ValidationResult(
            passed=False,
            failure_reasons=["ruff check failed", "pytest failed"],
        )
        assert result.short_summary() == "ruff check failed; pytest failed"


class TestTailFunction:
    """Test the _tail helper function."""

    def test_empty_text(self) -> None:
        assert _tail("") == ""

    def test_short_text_unchanged(self) -> None:
        text = "hello world"
        assert _tail(text) == text

    def test_truncates_long_text(self) -> None:
        text = "x" * 1000
        result = _tail(text, max_chars=100)
        assert len(result) == 100
        assert result == "x" * 100

    def test_truncates_many_lines(self) -> None:
        lines = [f"line {i}" for i in range(50)]
        text = "\n".join(lines)
        result = _tail(text, max_lines=5)
        result_lines = result.splitlines()
        assert len(result_lines) == 5
        # Should get the last 5 lines
        assert result_lines[0] == "line 45"
        assert result_lines[-1] == "line 49"


class TestFormatStepOutput:
    """Test the _format_step_output helper function."""

    def test_formats_stderr(self) -> None:
        step = ValidationStepResult(
            name="test",
            command=["test"],
            ok=False,
            returncode=1,
            stderr_tail="error message",
        )
        result = _format_step_output(step)
        assert "stderr:" in result
        assert "error message" in result

    def test_formats_stdout_when_no_stderr(self) -> None:
        step = ValidationStepResult(
            name="test",
            command=["test"],
            ok=False,
            returncode=1,
            stdout_tail="stdout message",
            stderr_tail="",
        )
        result = _format_step_output(step)
        assert "stdout:" in result
        assert "stdout message" in result

    def test_prefers_stderr_over_stdout(self) -> None:
        step = ValidationStepResult(
            name="test",
            command=["test"],
            ok=False,
            returncode=1,
            stdout_tail="stdout",
            stderr_tail="stderr",
        )
        result = _format_step_output(step)
        assert "stderr" in result
        # stdout not included when stderr present
        assert "stdout:" not in result

    def test_empty_output(self) -> None:
        step = ValidationStepResult(
            name="test",
            command=["test"],
            ok=False,
            returncode=1,
        )
        result = _format_step_output(step)
        assert result == ""


class TestCheckE2EPrereqs:
    """Test the _check_e2e_prereqs helper function."""

    def test_missing_mala_cli(self) -> None:
        with patch("shutil.which", return_value=None):
            result = _check_e2e_prereqs({})
            assert result is not None
            assert "mala CLI not found" in result

    def test_missing_bd_cli(self) -> None:
        def mock_which(cmd: str) -> str | None:
            if cmd == "mala":
                return "/usr/bin/mala"
            return None

        with patch("shutil.which", side_effect=mock_which):
            result = _check_e2e_prereqs({})
            assert result is not None
            assert "bd CLI not found" in result

    def test_missing_morph_api_key(self) -> None:
        with patch("shutil.which", return_value="/usr/bin/fake"):
            result = _check_e2e_prereqs({})
            assert result is not None
            assert "MORPH_API_KEY" in result

    def test_all_prereqs_met(self) -> None:
        with patch("shutil.which", return_value="/usr/bin/fake"):
            result = _check_e2e_prereqs({"MORPH_API_KEY": "test-key"})
            assert result is None


class TestValidationRunner:
    """Test ValidationRunner class."""

    @pytest.fixture
    def runner(self, tmp_path: Path) -> ValidationRunner:
        """Create a runner with minimal config."""
        config = ValidationConfig(
            run_slow_tests=False,
            run_e2e=False,
            coverage=False,
            use_test_mutex=False,
        )
        return ValidationRunner(tmp_path, config)

    def test_init(self, tmp_path: Path) -> None:
        config = ValidationConfig()
        runner = ValidationRunner(tmp_path, config)
        assert runner.repo_path == tmp_path.resolve()
        assert runner.config is config

    def test_build_validation_commands_minimal(self, runner: ValidationRunner) -> None:
        commands = runner._build_validation_commands()
        names = [name for name, _ in commands]
        assert "uv sync" in names
        assert "ruff format" in names
        assert "ruff check" in names
        assert "ty check" in names
        assert "pytest" in names

    def test_build_validation_commands_with_coverage(self, tmp_path: Path) -> None:
        config = ValidationConfig(
            run_slow_tests=True,
            coverage=True,
            coverage_min=80,
            coverage_source="src",
            use_test_mutex=False,
        )
        runner = ValidationRunner(tmp_path, config)
        commands = runner._build_validation_commands()

        pytest_cmd = None
        for name, cmd in commands:
            if name == "pytest":
                pytest_cmd = cmd
                break

        assert pytest_cmd is not None
        assert "--cov" in pytest_cmd
        assert "src" in pytest_cmd
        assert "--cov-branch" in pytest_cmd
        assert "--cov-fail-under=80" in pytest_cmd
        assert "-m" in pytest_cmd
        assert "slow or not slow" in pytest_cmd

    def test_run_command_success(self, runner: ValidationRunner) -> None:
        mock_result = subprocess.CompletedProcess(
            args=["echo", "hello"],
            returncode=0,
            stdout="hello\n",
            stderr="",
        )
        with patch("subprocess.run", return_value=mock_result):
            result = runner._run_command("echo", ["echo", "hello"], Path("."), {})
            assert result.ok is True
            assert result.returncode == 0
            assert result.name == "echo"

    def test_run_command_failure(self, runner: ValidationRunner) -> None:
        mock_result = subprocess.CompletedProcess(
            args=["false"],
            returncode=1,
            stdout="",
            stderr="command failed",
        )
        with patch("subprocess.run", return_value=mock_result):
            result = runner._run_command("false", ["false"], Path("."), {})
            assert result.ok is False
            assert result.returncode == 1
            assert "command failed" in result.stderr_tail

    def test_run_command_timeout(self, tmp_path: Path) -> None:
        config = ValidationConfig(
            use_test_mutex=False,
            step_timeout_seconds=0.1,
        )
        runner = ValidationRunner(tmp_path, config)

        exc = subprocess.TimeoutExpired(cmd=["sleep"], timeout=0.1)
        exc.stdout = b"partial output"
        exc.stderr = b"timeout error"

        with patch("subprocess.run", side_effect=exc):
            result = runner._run_command("sleep", ["sleep", "10"], Path("."), {})
            assert result.ok is False
            assert result.returncode == 124  # Standard timeout exit code
            assert "partial output" in result.stdout_tail
            assert "timeout error" in result.stderr_tail

    def test_run_validation_steps_all_pass(self, runner: ValidationRunner) -> None:
        mock_result = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="ok",
            stderr="",
        )
        with patch("subprocess.run", return_value=mock_result):
            result = runner._run_validation_steps(Path("."), "test", include_e2e=False)
            assert result.passed is True
            assert (
                len(result.steps) == 5
            )  # uv sync, ruff format, ruff check, ty, pytest

    def test_run_validation_steps_first_fails(self, runner: ValidationRunner) -> None:
        mock_result = subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr="sync failed",
        )
        with patch("subprocess.run", return_value=mock_result):
            result = runner._run_validation_steps(Path("."), "test", include_e2e=False)
            assert result.passed is False
            assert len(result.steps) == 1  # Stops after first failure
            assert "uv sync failed" in result.failure_reasons[0]

    def test_validate_commit_worktree_add_fails(self, runner: ValidationRunner) -> None:
        mock_result = subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr="fatal: not a git repository",
        )
        with patch("subprocess.run", return_value=mock_result):
            with patch("shutil.rmtree"):
                result = runner._validate_commit_sync("abc123", "test")
                assert result.passed is False
                assert result.retriable is False
                assert "git worktree add" in result.failure_reasons[0]

    @pytest.mark.asyncio
    async def test_validate_commit_async_wrapper(
        self, runner: ValidationRunner
    ) -> None:
        """Test that validate_commit properly wraps the sync method."""
        mock_result = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="ok",
            stderr="",
        )
        with patch("subprocess.run", return_value=mock_result):
            with patch("shutil.rmtree"):
                result = await runner.validate_commit("abc123", "test")
                assert result.passed is True


class TestValidationRunnerWithMutex:
    """Test ValidationRunner with test mutex enabled."""

    def test_wrap_with_mutex(self, tmp_path: Path) -> None:
        config = ValidationConfig(use_test_mutex=True)
        runner = ValidationRunner(tmp_path, config)

        wrapped = runner._wrap_with_mutex(["pytest", "-v"])
        assert "test-mutex.sh" in wrapped[0]
        assert wrapped[1:] == ["pytest", "-v"]


class TestE2EValidation:
    """Test E2E validation (with mocked prereqs and commands)."""

    def test_run_e2e_missing_prereqs(self, tmp_path: Path) -> None:
        config = ValidationConfig(run_e2e=True, use_test_mutex=False)
        runner = ValidationRunner(tmp_path, config)

        with patch("shutil.which", return_value=None):
            result = runner._run_e2e("test", {})
            assert result.passed is False
            assert result.retriable is False
            assert "mala CLI not found" in result.failure_reasons[0]

    def test_run_e2e_success(self, tmp_path: Path) -> None:
        config = ValidationConfig(run_e2e=True, use_test_mutex=False)
        runner = ValidationRunner(tmp_path, config)

        mock_result = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="passed",
            stderr="",
        )

        with (
            patch("shutil.which", return_value="/usr/bin/fake"),
            patch("subprocess.run", return_value=mock_result),
        ):
            result = runner._run_e2e("test", {"MORPH_API_KEY": "test-key"})
            assert result.passed is True

    def test_run_e2e_command_uses_valid_flags(self, tmp_path: Path) -> None:
        """Test that E2E command only uses flags supported by the mala CLI."""
        config = ValidationConfig(run_e2e=True, use_test_mutex=False)
        runner = ValidationRunner(tmp_path, config)

        captured_cmd: list[str] = []

        def capture_run(*args, **kwargs):  # type: ignore[no-untyped-def]
            if args and isinstance(args[0], list):
                captured_cmd.extend(args[0])
            return subprocess.CompletedProcess(
                args=[], returncode=0, stdout="", stderr=""
            )

        with (
            patch("shutil.which", return_value="/usr/bin/fake"),
            patch("subprocess.run", side_effect=capture_run),
        ):
            runner._run_e2e("test", {"MORPH_API_KEY": "test-key"})

        # These flags do not exist in the mala CLI
        assert "--no-post-validate" not in captured_cmd, (
            "mala CLI has no --no-post-validate flag"
        )
        assert "--no-e2e" not in captured_cmd, "mala CLI has no --no-e2e flag"
        # These are valid flags
        assert "mala" in captured_cmd
        assert "run" in captured_cmd


class TestTimeoutHandling:
    """Test timeout handling for str/bytes output types."""

    def test_run_command_timeout_with_string_output(self, tmp_path: Path) -> None:
        """TimeoutExpired may have str stdout/stderr when text=True is used."""
        config = ValidationConfig(
            use_test_mutex=False,
            step_timeout_seconds=0.1,
        )
        runner = ValidationRunner(tmp_path, config)

        exc = subprocess.TimeoutExpired(cmd=["sleep"], timeout=0.1)
        exc.stdout = "string output"  # Already a string, not bytes
        exc.stderr = "string error"

        with patch("subprocess.run", side_effect=exc):
            result = runner._run_command("sleep", ["sleep", "10"], Path("."), {})
            assert result.ok is False
            assert result.returncode == 124
            assert "string output" in result.stdout_tail
            assert "string error" in result.stderr_tail

    def test_run_command_timeout_with_none_output(self, tmp_path: Path) -> None:
        """TimeoutExpired may have None stdout/stderr."""
        config = ValidationConfig(
            use_test_mutex=False,
            step_timeout_seconds=0.1,
        )
        runner = ValidationRunner(tmp_path, config)

        exc = subprocess.TimeoutExpired(cmd=["sleep"], timeout=0.1)
        exc.stdout = None
        exc.stderr = None

        with patch("subprocess.run", side_effect=exc):
            result = runner._run_command("sleep", ["sleep", "10"], Path("."), {})
            assert result.ok is False
            assert result.returncode == 124
            assert result.stdout_tail == ""
            assert result.stderr_tail == ""

    def test_run_command_timeout_with_bytes_output(self, tmp_path: Path) -> None:
        """TimeoutExpired may have bytes stdout/stderr when text=False."""
        config = ValidationConfig(
            use_test_mutex=False,
            step_timeout_seconds=0.1,
        )
        runner = ValidationRunner(tmp_path, config)

        exc = subprocess.TimeoutExpired(cmd=["sleep"], timeout=0.1)
        exc.stdout = b"bytes output"
        exc.stderr = b"bytes error"

        with patch("subprocess.run", side_effect=exc):
            result = runner._run_command("sleep", ["sleep", "10"], Path("."), {})
            assert result.ok is False
            assert result.returncode == 124
            assert "bytes output" in result.stdout_tail
            assert "bytes error" in result.stderr_tail
