"""Unit tests for src/validation.py - post-commit validation runner.

These tests mock subprocess and shutil.which to test validation logic
without actually running commands or creating git worktrees.
"""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.validation import (
    CommandKind,
    CoverageConfig,
    CoverageResult,
    CoverageStatus,
    E2EConfig,
    ValidationArtifacts,
    ValidationCommand,
    ValidationConfig,
    ValidationContext,
    ValidationResult,
    ValidationRunner,
    ValidationScope,
    ValidationSpec,
    ValidationStepResult,
    _check_e2e_prereqs,
    _format_step_output,
    _tail,
)


def mock_popen_success(
    stdout: str = "", stderr: str = "", returncode: int = 0
) -> MagicMock:
    """Create a mock Popen object that returns successfully."""
    mock_proc = MagicMock()
    mock_proc.communicate.return_value = (stdout, stderr)
    mock_proc.returncode = returncode
    mock_proc.pid = 12345
    return mock_proc


def mock_popen_timeout() -> MagicMock:
    """Create a mock Popen object that times out then terminates cleanly."""
    mock_proc = MagicMock()
    # First call times out, subsequent calls succeed (for termination)
    mock_proc.communicate.side_effect = [
        subprocess.TimeoutExpired(cmd=["sleep"], timeout=0.1),
        ("", ""),  # Response after SIGTERM
    ]
    mock_proc.pid = 12345
    mock_proc.returncode = None
    return mock_proc


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
        assert result.artifacts is None
        assert result.coverage_result is None

    def test_failed_result(self) -> None:
        result = ValidationResult(
            passed=False,
            failure_reasons=["pytest failed (exit 1)"],
            retriable=True,
        )
        assert result.passed is False
        assert "pytest failed" in result.failure_reasons[0]

    def test_result_with_artifacts(self, tmp_path: Path) -> None:
        artifacts = ValidationArtifacts(log_dir=tmp_path)
        result = ValidationResult(passed=True, artifacts=artifacts)
        assert result.artifacts is artifacts
        assert result.artifacts.log_dir == tmp_path

    def test_result_with_coverage(self) -> None:
        coverage = CoverageResult(
            percent=85.5,
            passed=True,
            status=CoverageStatus.PASSED,
            report_path=Path("coverage.xml"),
        )
        result = ValidationResult(passed=True, coverage_result=coverage)
        assert result.coverage_result is coverage
        assert result.coverage_result.percent == 85.5

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

    def test_missing_morph_api_key_does_not_fail(self) -> None:
        """MORPH_API_KEY is not a hard prereq - E2E should still run without it."""
        with patch("shutil.which", return_value="/usr/bin/fake"):
            result = _check_e2e_prereqs({})
            assert result is None  # Should pass even without MORPH_API_KEY

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
        mock_proc = mock_popen_success(stdout="hello\n", stderr="", returncode=0)
        with patch(
            "src.validation.command_runner.subprocess.Popen", return_value=mock_proc
        ):
            result = runner._run_command("echo", ["echo", "hello"], Path("."), {})
            assert result.ok is True
            assert result.returncode == 0
            assert result.name == "echo"

    def test_run_command_failure(self, runner: ValidationRunner) -> None:
        mock_proc = mock_popen_success(stdout="", stderr="command failed", returncode=1)
        with patch(
            "src.validation.command_runner.subprocess.Popen", return_value=mock_proc
        ):
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

        mock_proc = mock_popen_timeout()
        with patch(
            "src.validation.command_runner.subprocess.Popen", return_value=mock_proc
        ):
            result = runner._run_command("sleep", ["sleep", "10"], Path("."), {})
            assert result.ok is False
            assert result.returncode == 124  # Standard timeout exit code

    def test_run_validation_steps_all_pass(self, runner: ValidationRunner) -> None:
        mock_proc = mock_popen_success(stdout="ok", stderr="", returncode=0)
        with patch(
            "src.validation.command_runner.subprocess.Popen", return_value=mock_proc
        ):
            result = runner._run_validation_steps(Path("."), "test", include_e2e=False)
            assert result.passed is True
            assert len(result.steps) == 4  # ruff format, ruff check, ty, pytest

    def test_run_validation_steps_first_fails(self, runner: ValidationRunner) -> None:
        mock_proc = mock_popen_success(stdout="", stderr="sync failed", returncode=1)
        with patch(
            "src.validation.command_runner.subprocess.Popen", return_value=mock_proc
        ):
            result = runner._run_validation_steps(Path("."), "test", include_e2e=False)
            assert result.passed is False
            assert len(result.steps) == 1  # Stops after first failure
            assert "ruff format failed" in result.failure_reasons[0]

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
        # Mock git worktree commands (still uses subprocess.run)
        mock_git_result = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="ok",
            stderr="",
        )
        # Mock validation commands (uses Popen)
        mock_proc = mock_popen_success(stdout="ok", stderr="", returncode=0)
        with (
            patch("subprocess.run", return_value=mock_git_result),
            patch(
                "src.validation.command_runner.subprocess.Popen",
                return_value=mock_proc,
            ),
            patch("shutil.rmtree"),
        ):
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
        """Test that E2E command only uses flags supported by the mala CLI.

        Ensures --no-post-validate and --no-e2e are not used since they don't
        exist in the CLI. This covers the acceptance criteria from mala-w8w.2.
        """
        config = ValidationConfig(run_e2e=True, use_test_mutex=False)
        runner = ValidationRunner(tmp_path, config)

        captured_cmd: list[str] = []

        def capture_popen(cmd: list[str], **kwargs: object) -> MagicMock:
            captured_cmd.extend(cmd)
            mock_proc = MagicMock()
            mock_proc.communicate.return_value = ("", "")
            mock_proc.returncode = 0
            mock_proc.pid = 12345
            return mock_proc

        # Also mock subprocess.run for git init, etc. in fixture setup
        mock_run_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )

        with (
            patch("shutil.which", return_value="/usr/bin/fake"),
            patch(
                "src.validation.command_runner.subprocess.Popen",
                side_effect=capture_popen,
            ),
            patch("subprocess.run", return_value=mock_run_result),
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
    """Test timeout handling for CommandRunner.

    These tests ensure timeout is properly detected and the process
    is terminated with proper process-group killing.
    """

    def test_run_command_timeout_with_string_output(self, tmp_path: Path) -> None:
        """Timeout returns output captured during termination."""
        config = ValidationConfig(
            use_test_mutex=False,
            step_timeout_seconds=0.1,
        )
        runner = ValidationRunner(tmp_path, config)

        # Mock Popen that times out, then returns output on terminate
        mock_proc = MagicMock()
        mock_proc.communicate.side_effect = [
            subprocess.TimeoutExpired(cmd=["sleep"], timeout=0.1),
            ("terminated output", "terminated error"),  # Output from terminate
        ]
        mock_proc.pid = 12345
        mock_proc.returncode = None

        with patch(
            "src.validation.command_runner.subprocess.Popen", return_value=mock_proc
        ):
            result = runner._run_command("sleep", ["sleep", "10"], Path("."), {})
            assert result.ok is False
            assert result.returncode == 124
            assert "terminated output" in result.stdout_tail
            assert "terminated error" in result.stderr_tail

    def test_run_command_timeout_with_none_output(self, tmp_path: Path) -> None:
        """Timeout with no output captured."""
        config = ValidationConfig(
            use_test_mutex=False,
            step_timeout_seconds=0.1,
        )
        runner = ValidationRunner(tmp_path, config)

        # Mock Popen that times out with no output
        mock_proc = MagicMock()
        mock_proc.communicate.side_effect = [
            subprocess.TimeoutExpired(cmd=["sleep"], timeout=0.1),
            ("", ""),  # No output from terminate
        ]
        mock_proc.pid = 12345
        mock_proc.returncode = None

        with patch(
            "src.validation.command_runner.subprocess.Popen", return_value=mock_proc
        ):
            result = runner._run_command("sleep", ["sleep", "10"], Path("."), {})
            assert result.ok is False
            assert result.returncode == 124
            assert result.stdout_tail == ""
            assert result.stderr_tail == ""

    def test_run_command_timeout_with_bytes_output(self, tmp_path: Path) -> None:
        """Timeout - CommandRunner now uses text=True so output is always str."""
        config = ValidationConfig(
            use_test_mutex=False,
            step_timeout_seconds=0.1,
        )
        runner = ValidationRunner(tmp_path, config)

        # With text=True, Popen returns strings not bytes
        mock_proc = MagicMock()
        mock_proc.communicate.side_effect = [
            subprocess.TimeoutExpired(cmd=["sleep"], timeout=0.1),
            ("string output", "string error"),
        ]
        mock_proc.pid = 12345
        mock_proc.returncode = None

        with patch(
            "src.validation.command_runner.subprocess.Popen", return_value=mock_proc
        ):
            result = runner._run_command("sleep", ["sleep", "10"], Path("."), {})
            assert result.ok is False
            assert result.returncode == 124
            assert "string output" in result.stdout_tail
            assert "string error" in result.stderr_tail


class TestSpecBasedValidation:
    """Test ValidationRunner.run_spec() with ValidationSpec + ValidationContext."""

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

    @pytest.fixture
    def basic_spec(self) -> ValidationSpec:
        """Create a basic validation spec with a single command."""
        return ValidationSpec(
            commands=[
                ValidationCommand(
                    name="echo test",
                    command=["echo", "hello"],
                    kind=CommandKind.TEST,
                    use_test_mutex=False,
                ),
            ],
            scope=ValidationScope.PER_ISSUE,
            coverage=CoverageConfig(enabled=False),
            e2e=E2EConfig(enabled=False),
        )

    @pytest.fixture
    def context(self, tmp_path: Path) -> ValidationContext:
        """Create a basic validation context."""
        return ValidationContext(
            issue_id="test-123",
            repo_path=tmp_path,
            commit_hash="",  # Empty = validate in place
            changed_files=["src/test.py"],
            scope=ValidationScope.PER_ISSUE,
        )

    def test_run_spec_single_command_passes(
        self,
        runner: ValidationRunner,
        basic_spec: ValidationSpec,
        context: ValidationContext,
        tmp_path: Path,
    ) -> None:
        """Test run_spec with a single passing command."""
        mock_proc = mock_popen_success(stdout="hello\n", stderr="", returncode=0)
        with patch(
            "src.validation.command_runner.subprocess.Popen", return_value=mock_proc
        ):
            result = runner._run_spec_sync(basic_spec, context, log_dir=tmp_path)
            assert result.passed is True
            assert len(result.steps) == 1
            assert result.steps[0].name == "echo test"
            assert result.steps[0].ok is True
            assert result.artifacts is not None
            assert result.artifacts.log_dir == tmp_path

    def test_run_spec_single_command_fails(
        self,
        runner: ValidationRunner,
        basic_spec: ValidationSpec,
        context: ValidationContext,
        tmp_path: Path,
    ) -> None:
        """Test run_spec with a single failing command."""
        mock_proc = mock_popen_success(stdout="", stderr="error occurred", returncode=1)
        with patch(
            "src.validation.command_runner.subprocess.Popen", return_value=mock_proc
        ):
            result = runner._run_spec_sync(basic_spec, context, log_dir=tmp_path)
            assert result.passed is False
            assert len(result.steps) == 1
            assert "echo test failed" in result.failure_reasons[0]

    def test_run_spec_multiple_commands_all_pass(
        self, runner: ValidationRunner, context: ValidationContext, tmp_path: Path
    ) -> None:
        """Test run_spec with multiple passing commands."""
        spec = ValidationSpec(
            commands=[
                ValidationCommand(
                    name="step1",
                    command=["echo", "1"],
                    kind=CommandKind.FORMAT,
                ),
                ValidationCommand(
                    name="step2",
                    command=["echo", "2"],
                    kind=CommandKind.LINT,
                ),
                ValidationCommand(
                    name="step3",
                    command=["echo", "3"],
                    kind=CommandKind.TEST,
                ),
            ],
            scope=ValidationScope.PER_ISSUE,
            coverage=CoverageConfig(enabled=False),
            e2e=E2EConfig(enabled=False),
        )

        mock_proc = mock_popen_success(stdout="ok", stderr="", returncode=0)
        with patch(
            "src.validation.command_runner.subprocess.Popen", return_value=mock_proc
        ):
            result = runner._run_spec_sync(spec, context, log_dir=tmp_path)
            assert result.passed is True
            assert len(result.steps) == 3

    def test_run_spec_stops_on_first_failure(
        self, runner: ValidationRunner, context: ValidationContext, tmp_path: Path
    ) -> None:
        """Test that run_spec stops execution on first command failure."""
        spec = ValidationSpec(
            commands=[
                ValidationCommand(
                    name="pass1",
                    command=["echo", "1"],
                    kind=CommandKind.FORMAT,
                ),
                ValidationCommand(
                    name="fail2",
                    command=["false"],
                    kind=CommandKind.LINT,
                ),
                ValidationCommand(
                    name="pass3",
                    command=["echo", "3"],
                    kind=CommandKind.TEST,
                ),
            ],
            scope=ValidationScope.PER_ISSUE,
            coverage=CoverageConfig(enabled=False),
            e2e=E2EConfig(enabled=False),
        )

        call_count = 0

        def mock_popen_calls(cmd: list[str], **kwargs: object) -> MagicMock:
            nonlocal call_count
            call_count += 1
            mock_proc = MagicMock()
            mock_proc.pid = 12345
            if call_count == 2:
                mock_proc.communicate.return_value = ("", "lint error")
                mock_proc.returncode = 1
            else:
                mock_proc.communicate.return_value = ("ok", "")
                mock_proc.returncode = 0
            return mock_proc

        with patch(
            "src.validation.command_runner.subprocess.Popen",
            side_effect=mock_popen_calls,
        ):
            result = runner._run_spec_sync(spec, context, log_dir=tmp_path)
            assert result.passed is False
            assert len(result.steps) == 2  # Stopped after fail2
            assert "fail2 failed" in result.failure_reasons[0]

    def test_run_spec_allow_fail_continues(
        self, runner: ValidationRunner, context: ValidationContext, tmp_path: Path
    ) -> None:
        """Test that allow_fail=True continues execution after failure."""
        spec = ValidationSpec(
            commands=[
                ValidationCommand(
                    name="pass1",
                    command=["echo", "1"],
                    kind=CommandKind.FORMAT,
                ),
                ValidationCommand(
                    name="fail2",
                    command=["false"],
                    kind=CommandKind.LINT,
                    allow_fail=True,  # Allow this to fail
                ),
                ValidationCommand(
                    name="pass3",
                    command=["echo", "3"],
                    kind=CommandKind.TEST,
                ),
            ],
            scope=ValidationScope.PER_ISSUE,
            coverage=CoverageConfig(enabled=False),
            e2e=E2EConfig(enabled=False),
        )

        call_count = 0

        def mock_popen_calls(cmd: list[str], **kwargs: object) -> MagicMock:
            nonlocal call_count
            call_count += 1
            mock_proc = MagicMock()
            mock_proc.pid = 12345
            if call_count == 2:
                mock_proc.communicate.return_value = ("", "lint warning")
                mock_proc.returncode = 1
            else:
                mock_proc.communicate.return_value = ("ok", "")
                mock_proc.returncode = 0
            return mock_proc

        with patch(
            "src.validation.command_runner.subprocess.Popen",
            side_effect=mock_popen_calls,
        ):
            result = runner._run_spec_sync(spec, context, log_dir=tmp_path)
            assert result.passed is True  # Passed despite step 2 failing
            assert len(result.steps) == 3  # All steps ran

    def test_run_spec_uses_mutex_when_requested(
        self, runner: ValidationRunner, context: ValidationContext, tmp_path: Path
    ) -> None:
        """Test that use_test_mutex wraps command with mutex script."""
        spec = ValidationSpec(
            commands=[
                ValidationCommand(
                    name="pytest",
                    command=["pytest", "-v"],
                    kind=CommandKind.TEST,
                    use_test_mutex=True,
                ),
            ],
            scope=ValidationScope.PER_ISSUE,
            coverage=CoverageConfig(enabled=False),
            e2e=E2EConfig(enabled=False),
        )

        captured_cmd: list[str] = []

        def capture_popen(cmd: list[str], **kwargs: object) -> MagicMock:
            captured_cmd.extend(cmd)
            mock_proc = MagicMock()
            mock_proc.communicate.return_value = ("ok", "")
            mock_proc.returncode = 0
            mock_proc.pid = 12345
            return mock_proc

        with patch(
            "src.validation.command_runner.subprocess.Popen", side_effect=capture_popen
        ):
            runner._run_spec_sync(spec, context, log_dir=tmp_path)

        assert "test-mutex.sh" in captured_cmd[0]
        assert "pytest" in captured_cmd

    def test_run_spec_writes_log_files(
        self,
        runner: ValidationRunner,
        basic_spec: ValidationSpec,
        context: ValidationContext,
        tmp_path: Path,
    ) -> None:
        """Test that stdout/stderr are written to log files."""
        mock_proc = mock_popen_success(
            stdout="stdout content\n", stderr="stderr content\n", returncode=0
        )
        with patch(
            "src.validation.command_runner.subprocess.Popen", return_value=mock_proc
        ):
            result = runner._run_spec_sync(basic_spec, context, log_dir=tmp_path)
            assert result.passed is True

        # Check log files were created
        stdout_log = tmp_path / "echo_test.stdout.log"
        stderr_log = tmp_path / "echo_test.stderr.log"
        assert stdout_log.exists()
        assert stderr_log.exists()
        assert "stdout content" in stdout_log.read_text()
        assert "stderr content" in stderr_log.read_text()

    def test_run_spec_coverage_enabled_passes(
        self, runner: ValidationRunner, context: ValidationContext, tmp_path: Path
    ) -> None:
        """Test coverage validation when coverage.xml exists and passes threshold."""
        # Create a valid coverage.xml
        coverage_xml = tmp_path / "coverage.xml"
        coverage_xml.write_text(
            '<?xml version="1.0"?>\n<coverage line-rate="0.90" branch-rate="0.85" />'
        )

        spec = ValidationSpec(
            commands=[
                ValidationCommand(
                    name="pytest",
                    command=["pytest"],
                    kind=CommandKind.TEST,
                ),
            ],
            scope=ValidationScope.PER_ISSUE,
            coverage=CoverageConfig(
                enabled=True,
                min_percent=85.0,
                report_path=coverage_xml,
            ),
            e2e=E2EConfig(enabled=False),
        )

        mock_proc = mock_popen_success(stdout="ok", stderr="", returncode=0)
        with patch(
            "src.validation.command_runner.subprocess.Popen", return_value=mock_proc
        ):
            result = runner._run_spec_sync(spec, context, log_dir=tmp_path)
            assert result.passed is True
            assert result.coverage_result is not None
            assert result.coverage_result.passed is True
            assert result.coverage_result.percent == 90.0

    def test_run_spec_coverage_enabled_fails(
        self, runner: ValidationRunner, context: ValidationContext, tmp_path: Path
    ) -> None:
        """Test coverage validation when coverage.xml exists but fails threshold."""
        # Create a coverage.xml below threshold
        coverage_xml = tmp_path / "coverage.xml"
        coverage_xml.write_text(
            '<?xml version="1.0"?>\n<coverage line-rate="0.70" branch-rate="0.60" />'
        )

        spec = ValidationSpec(
            commands=[
                ValidationCommand(
                    name="pytest",
                    command=["pytest"],
                    kind=CommandKind.TEST,
                ),
            ],
            scope=ValidationScope.PER_ISSUE,
            coverage=CoverageConfig(
                enabled=True,
                min_percent=85.0,
                report_path=coverage_xml,
            ),
            e2e=E2EConfig(enabled=False),
        )

        mock_proc = mock_popen_success(stdout="ok", stderr="", returncode=0)
        with patch(
            "src.validation.command_runner.subprocess.Popen", return_value=mock_proc
        ):
            result = runner._run_spec_sync(spec, context, log_dir=tmp_path)
            assert result.passed is False
            assert result.coverage_result is not None
            assert result.coverage_result.passed is False
            assert result.coverage_result.failure_reason is not None
            assert "70.0%" in result.coverage_result.failure_reason

    def test_run_spec_coverage_missing_file(
        self, runner: ValidationRunner, context: ValidationContext, tmp_path: Path
    ) -> None:
        """Test coverage validation when coverage.xml is missing."""
        spec = ValidationSpec(
            commands=[
                ValidationCommand(
                    name="pytest",
                    command=["pytest"],
                    kind=CommandKind.TEST,
                ),
            ],
            scope=ValidationScope.PER_ISSUE,
            coverage=CoverageConfig(
                enabled=True,
                min_percent=85.0,
                report_path=tmp_path / "missing.xml",
            ),
            e2e=E2EConfig(enabled=False),
        )

        mock_proc = mock_popen_success(stdout="ok", stderr="", returncode=0)
        with patch(
            "src.validation.command_runner.subprocess.Popen", return_value=mock_proc
        ):
            result = runner._run_spec_sync(spec, context, log_dir=tmp_path)
            assert result.passed is False
            assert result.coverage_result is not None
            assert result.coverage_result.passed is False
            assert "not found" in (result.coverage_result.failure_reason or "")

    def test_run_spec_e2e_only_for_run_level(
        self, runner: ValidationRunner, context: ValidationContext, tmp_path: Path
    ) -> None:
        """Test that E2E only runs for per-issue scope."""
        # Per-issue context - E2E should not run
        per_issue_context = ValidationContext(
            issue_id="test-123",
            repo_path=tmp_path,
            commit_hash="",
            changed_files=[],
            scope=ValidationScope.PER_ISSUE,
        )

        spec = ValidationSpec(
            commands=[],
            scope=ValidationScope.PER_ISSUE,
            coverage=CoverageConfig(enabled=False),
            e2e=E2EConfig(enabled=True),
        )

        # E2E should not be called for per-issue scope
        e2e_run_called = False

        def mock_e2e_run(*args: object, **kwargs: object) -> MagicMock:
            nonlocal e2e_run_called
            e2e_run_called = True
            return MagicMock(passed=True, fixture_path=None)

        # Patch on the internal spec runner since ValidationRunner delegates to it
        with patch.object(
            runner._spec_runner, "_run_spec_e2e", side_effect=mock_e2e_run
        ):
            result = runner._run_spec_sync(spec, per_issue_context, log_dir=tmp_path)
            assert result.passed is True
            assert not e2e_run_called

    def test_run_spec_e2e_runs_for_run_level(
        self, runner: ValidationRunner, tmp_path: Path
    ) -> None:
        """Test that E2E runs for run-level scope."""
        run_level_context = ValidationContext(
            issue_id=None,
            repo_path=tmp_path,
            commit_hash="",
            changed_files=[],
            scope=ValidationScope.RUN_LEVEL,
        )

        spec = ValidationSpec(
            commands=[],
            scope=ValidationScope.RUN_LEVEL,
            coverage=CoverageConfig(enabled=False),
            e2e=E2EConfig(enabled=True),
        )

        from src.validation.e2e import E2EResult, E2EStatus

        mock_e2e_result = E2EResult(
            passed=True,
            status=E2EStatus.PASSED,
            fixture_path=tmp_path / "fixture",
        )

        # Patch on the internal spec runner since ValidationRunner delegates to it
        with patch.object(
            runner._spec_runner, "_run_spec_e2e", return_value=mock_e2e_result
        ):
            result = runner._run_spec_sync(spec, run_level_context, log_dir=tmp_path)
            assert result.passed is True

    @pytest.mark.asyncio
    async def test_run_spec_async(
        self,
        runner: ValidationRunner,
        basic_spec: ValidationSpec,
        context: ValidationContext,
        tmp_path: Path,
    ) -> None:
        """Test run_spec async wrapper."""
        mock_proc = mock_popen_success(stdout="ok", stderr="", returncode=0)
        with patch(
            "src.validation.command_runner.subprocess.Popen", return_value=mock_proc
        ):
            result = await runner.run_spec(basic_spec, context, log_dir=tmp_path)
            assert result.passed is True

    def test_run_spec_timeout_handling(
        self,
        runner: ValidationRunner,
        basic_spec: ValidationSpec,
        context: ValidationContext,
        tmp_path: Path,
    ) -> None:
        """Test that command timeout is handled correctly."""
        mock_proc = MagicMock()
        mock_proc.communicate.side_effect = [
            subprocess.TimeoutExpired(cmd=["sleep"], timeout=1.0),
            ("partial", "timeout"),  # Output after termination
        ]
        mock_proc.pid = 12345
        mock_proc.returncode = None

        with patch(
            "src.validation.command_runner.subprocess.Popen", return_value=mock_proc
        ):
            result = runner._run_spec_sync(basic_spec, context, log_dir=tmp_path)
            assert result.passed is False
            assert result.steps[0].returncode == 124
            assert "partial" in result.steps[0].stdout_tail

    def test_run_spec_with_worktree(self, tmp_path: Path) -> None:
        """Test run_spec creates worktree when commit_hash is provided."""
        # Initialize a git repo
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        # Create a mock worktree context
        from src.validation.worktree import WorktreeContext, WorktreeState

        mock_worktree_ctx = MagicMock(spec=WorktreeContext)
        mock_worktree_ctx.state = WorktreeState.CREATED
        mock_worktree_ctx.path = tmp_path / "worktree"
        mock_worktree_ctx.path.mkdir()

        config = ValidationConfig(use_test_mutex=False)
        runner = ValidationRunner(repo_path, config)

        context = ValidationContext(
            issue_id="test-123",
            repo_path=repo_path,
            commit_hash="abc123",
            changed_files=[],
            scope=ValidationScope.PER_ISSUE,
        )

        spec = ValidationSpec(
            commands=[
                ValidationCommand(
                    name="test",
                    command=["echo", "test"],
                    kind=CommandKind.TEST,
                ),
            ],
            scope=ValidationScope.PER_ISSUE,
            coverage=CoverageConfig(enabled=False),
            e2e=E2EConfig(enabled=False),
        )

        mock_proc = mock_popen_success(stdout="ok", stderr="", returncode=0)

        with (
            patch(
                "src.validation.spec_runner.create_worktree",
                return_value=mock_worktree_ctx,
            ),
            patch(
                "src.validation.spec_runner.remove_worktree",
                return_value=mock_worktree_ctx,
            ),
            patch(
                "src.validation.command_runner.subprocess.Popen",
                return_value=mock_proc,
            ),
        ):
            result = runner._run_spec_sync(spec, context, log_dir=tmp_path)
            assert result.passed is True
            assert result.artifacts is not None
            assert result.artifacts.worktree_path == mock_worktree_ctx.path

    def test_run_spec_worktree_failure(self, tmp_path: Path) -> None:
        """Test run_spec handles worktree creation failure."""
        from src.validation.worktree import WorktreeContext, WorktreeState

        mock_worktree_ctx = MagicMock(spec=WorktreeContext)
        mock_worktree_ctx.state = WorktreeState.FAILED
        mock_worktree_ctx.error = "git worktree add failed"

        config = ValidationConfig(use_test_mutex=False)
        runner = ValidationRunner(tmp_path, config)

        context = ValidationContext(
            issue_id="test-123",
            repo_path=tmp_path,
            commit_hash="abc123",
            changed_files=[],
            scope=ValidationScope.PER_ISSUE,
        )

        spec = ValidationSpec(
            commands=[],
            scope=ValidationScope.PER_ISSUE,
            coverage=CoverageConfig(enabled=False),
            e2e=E2EConfig(enabled=False),
        )

        with patch(
            "src.validation.spec_runner.create_worktree", return_value=mock_worktree_ctx
        ):
            result = runner._run_spec_sync(spec, context, log_dir=tmp_path)
            assert result.passed is False
            assert result.retriable is False
            assert "Worktree creation failed" in result.failure_reasons[0]

    def test_build_spec_env(
        self, runner: ValidationRunner, context: ValidationContext
    ) -> None:
        """Test _build_spec_env includes expected variables."""
        env = runner._build_spec_env(context, "run-123")
        assert "LOCK_DIR" in env
        assert "AGENT_ID" in env
        assert "test-123" in env["AGENT_ID"]

    def test_build_spec_env_run_level(
        self, runner: ValidationRunner, tmp_path: Path
    ) -> None:
        """Test _build_spec_env with run-level context (no issue_id)."""
        context = ValidationContext(
            issue_id=None,
            repo_path=tmp_path,
            commit_hash="",
            changed_files=[],
            scope=ValidationScope.RUN_LEVEL,
        )
        env = runner._build_spec_env(context, "run-456")
        assert "run-456" in env["AGENT_ID"]

    def test_run_spec_worktree_removed_on_success(self, tmp_path: Path) -> None:
        """Test that successful run_spec removes worktree and sets state to 'removed'."""
        from src.validation.worktree import WorktreeContext, WorktreeState

        # Create mock for worktree creation
        mock_worktree_created = MagicMock(spec=WorktreeContext)
        mock_worktree_created.state = WorktreeState.CREATED
        mock_worktree_created.path = tmp_path / "worktree"
        mock_worktree_created.path.mkdir()

        # Create mock for worktree removal (successful)
        mock_worktree_removed = MagicMock(spec=WorktreeContext)
        mock_worktree_removed.state = WorktreeState.REMOVED

        config = ValidationConfig(use_test_mutex=False)
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        runner = ValidationRunner(repo_path, config)

        context = ValidationContext(
            issue_id="test-123",
            repo_path=repo_path,
            commit_hash="abc123",
            changed_files=[],
            scope=ValidationScope.PER_ISSUE,
        )

        spec = ValidationSpec(
            commands=[
                ValidationCommand(
                    name="test",
                    command=["echo", "test"],
                    kind=CommandKind.TEST,
                ),
            ],
            scope=ValidationScope.PER_ISSUE,
            coverage=CoverageConfig(enabled=False),
            e2e=E2EConfig(enabled=False),
        )

        mock_proc = mock_popen_success(stdout="ok", stderr="", returncode=0)

        remove_worktree_called_with: list[tuple[object, bool]] = []

        def mock_remove(ctx: object, validation_passed: bool) -> MagicMock:
            remove_worktree_called_with.append((ctx, validation_passed))
            return mock_worktree_removed

        with (
            patch(
                "src.validation.spec_runner.create_worktree",
                return_value=mock_worktree_created,
            ),
            patch(
                "src.validation.spec_runner.remove_worktree", side_effect=mock_remove
            ),
            patch(
                "src.validation.command_runner.subprocess.Popen",
                return_value=mock_proc,
            ),
        ):
            result = runner._run_spec_sync(spec, context, log_dir=tmp_path)
            assert result.passed is True
            assert result.artifacts is not None
            assert result.artifacts.worktree_state == "removed"
            # Verify remove_worktree was called with validation_passed=True
            assert len(remove_worktree_called_with) == 1
            assert remove_worktree_called_with[0][1] is True

    def test_run_spec_worktree_cleanup_on_exception(self, tmp_path: Path) -> None:
        """Test that worktree is cleaned up even when execution raises an exception."""
        from src.validation.worktree import WorktreeContext, WorktreeState

        # Create mock for worktree creation
        mock_worktree_created = MagicMock(spec=WorktreeContext)
        mock_worktree_created.state = WorktreeState.CREATED
        mock_worktree_created.path = tmp_path / "worktree"
        mock_worktree_created.path.mkdir()

        # Create mock for worktree removal (kept due to failure)
        mock_worktree_kept = MagicMock(spec=WorktreeContext)
        mock_worktree_kept.state = WorktreeState.KEPT

        config = ValidationConfig(use_test_mutex=False)
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        runner = ValidationRunner(repo_path, config)

        context = ValidationContext(
            issue_id="test-123",
            repo_path=repo_path,
            commit_hash="abc123",
            changed_files=[],
            scope=ValidationScope.PER_ISSUE,
        )

        spec = ValidationSpec(
            commands=[
                ValidationCommand(
                    name="test",
                    command=["echo", "test"],
                    kind=CommandKind.TEST,
                ),
            ],
            scope=ValidationScope.PER_ISSUE,
            coverage=CoverageConfig(enabled=False),
            e2e=E2EConfig(enabled=False),
        )

        remove_worktree_called_with: list[tuple[object, bool]] = []

        def mock_remove(ctx: object, validation_passed: bool) -> MagicMock:
            remove_worktree_called_with.append((ctx, validation_passed))
            return mock_worktree_kept

        def mock_popen_raises(*args: object, **kwargs: object) -> MagicMock:
            raise RuntimeError("Command execution failed")

        with (
            patch(
                "src.validation.spec_runner.create_worktree",
                return_value=mock_worktree_created,
            ),
            patch(
                "src.validation.spec_runner.remove_worktree", side_effect=mock_remove
            ),
            patch(
                "src.validation.command_runner.subprocess.Popen",
                side_effect=mock_popen_raises,
            ),
        ):
            with pytest.raises(RuntimeError, match="Command execution failed"):
                runner._run_spec_sync(spec, context, log_dir=tmp_path)

            # Verify remove_worktree was called with validation_passed=False
            assert len(remove_worktree_called_with) == 1
            assert remove_worktree_called_with[0][1] is False


class TestSpecRunnerNoDecreaseMode:
    """Tests for the SpecValidationRunner's 'no decrease' coverage mode.

    In no-decrease mode (min_percent=None), the runner should:
    1. Capture baseline coverage before validation
    2. Auto-refresh baseline when stale or missing
    3. Compare current coverage against baseline
    """

    @pytest.fixture
    def runner(self, tmp_path: Path) -> ValidationRunner:
        """Create a runner with coverage enabled but no explicit threshold."""
        config = ValidationConfig(
            run_slow_tests=False,
            run_e2e=False,
            coverage=True,
            coverage_min=85,  # This is overridden by spec
            use_test_mutex=False,
        )
        return ValidationRunner(tmp_path, config)

    def test_no_decrease_mode_uses_baseline_when_fresh(
        self, runner: ValidationRunner, tmp_path: Path
    ) -> None:
        """When baseline is fresh, use it as threshold."""
        # Create fresh baseline at 80%
        baseline_xml = tmp_path / "coverage.xml"
        baseline_xml.write_text(
            '<?xml version="1.0"?>\n<coverage line-rate="0.80" branch-rate="0.75" />'
        )
        # Set mtime to future (fresh)
        import os

        future_time = 4102444800
        os.utime(baseline_xml, (future_time, future_time))

        # Create spec with no-decrease mode (min_percent=None)
        spec = ValidationSpec(
            commands=[
                ValidationCommand(
                    name="pytest",
                    command=["echo", "test"],
                    kind=CommandKind.TEST,
                ),
            ],
            scope=ValidationScope.PER_ISSUE,
            coverage=CoverageConfig(
                enabled=True,
                min_percent=None,  # No-decrease mode
            ),
            e2e=E2EConfig(enabled=False),
        )

        context = ValidationContext(
            issue_id="test-123",
            repo_path=tmp_path,
            commit_hash="",  # Validate in place
            changed_files=[],
            scope=ValidationScope.PER_ISSUE,
        )

        mock_proc = mock_popen_success(stdout="ok", stderr="", returncode=0)

        # Mock git commands to report clean repo with old commit
        def mock_git_run(
            args: list[str], **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            if "status" in args and "--porcelain" in args:
                return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
            elif "log" in args:
                return subprocess.CompletedProcess(
                    args, 0, stdout="1700000000\n", stderr=""
                )
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        with (
            patch(
                "src.validation.command_runner.subprocess.Popen",
                return_value=mock_proc,
            ),
            patch("src.validation.coverage.subprocess.run", side_effect=mock_git_run),
        ):
            result = runner._run_spec_sync(spec, context, log_dir=tmp_path)
            # Should pass because 80% >= 80% baseline
            assert result.passed is True
            assert result.coverage_result is not None
            assert result.coverage_result.passed is True
            assert result.coverage_result.percent == 80.0

    def test_baseline_captured_before_validation_not_during(
        self, runner: ValidationRunner, tmp_path: Path
    ) -> None:
        """Verify baseline is captured BEFORE validation, not from current run.

        This test ensures the runner uses the pre-existing baseline coverage (90%)
        for comparison, not the coverage produced by the current test run (70%).
        The test run produces lower coverage, but validation should compare against
        the baseline that existed before validation started.
        """
        import os

        # Create fresh baseline at 90% in the main repo
        baseline_xml = tmp_path / "coverage.xml"
        baseline_xml.write_text(
            '<?xml version="1.0"?>\n<coverage line-rate="0.90" branch-rate="0.85" />'
        )
        future_time = 4102444800
        os.utime(baseline_xml, (future_time, future_time))

        # Create a separate worktree directory where tests will run
        worktree_path = tmp_path / "worktree"
        worktree_path.mkdir()

        # The test run will produce 70% coverage in the worktree
        # This simulates pytest generating a new coverage.xml during the test run
        (worktree_path / "coverage.xml").write_text(
            '<?xml version="1.0"?>\n<coverage line-rate="0.70" branch-rate="0.65" />'
        )

        spec = ValidationSpec(
            commands=[
                ValidationCommand(
                    name="pytest",
                    command=["echo", "test"],
                    kind=CommandKind.TEST,
                ),
            ],
            scope=ValidationScope.PER_ISSUE,
            coverage=CoverageConfig(enabled=True, min_percent=None),
            e2e=E2EConfig(enabled=False),
        )

        context = ValidationContext(
            issue_id="test-123",
            repo_path=tmp_path,
            commit_hash="",
            changed_files=[],
            scope=ValidationScope.PER_ISSUE,
        )

        mock_proc = mock_popen_success(stdout="ok", stderr="", returncode=0)

        def mock_git_run(
            args: list[str], **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            if "status" in args and "--porcelain" in args:
                return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
            elif "log" in args:
                return subprocess.CompletedProcess(
                    args, 0, stdout="1700000000\n", stderr=""
                )
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        with (
            patch(
                "src.validation.command_runner.subprocess.Popen",
                return_value=mock_proc,
            ),
            patch("src.validation.coverage.subprocess.run", side_effect=mock_git_run),
        ):
            # Execute validation with the pre-captured baseline (90%)
            # The worktree has 70% coverage, which is below baseline
            result = runner._spec_runner._execute_spec_commands(
                spec=spec,
                context=context,
                cwd=worktree_path,  # Tests run in worktree with 70% coverage
                artifacts=ValidationArtifacts(log_dir=tmp_path),
                log_dir=tmp_path,
                run_id="test",
                baseline_percent=90.0,  # Baseline captured BEFORE validation
            )

            # Validation should FAIL because:
            # - Pre-validation baseline was 90%
            # - Current run's coverage is 70%
            # - 70% < 90% = coverage decreased
            assert result.passed is False
            assert result.coverage_result is not None
            assert result.coverage_result.passed is False
            assert result.coverage_result.percent == 70.0
            # Failure reason should mention both percentages
            assert "70.0%" in (result.coverage_result.failure_reason or "")
            assert "90.0%" in (result.coverage_result.failure_reason or "")

    def test_no_decrease_mode_fails_when_coverage_decreases(
        self, runner: ValidationRunner, tmp_path: Path
    ) -> None:
        """When current coverage is below baseline, validation fails."""
        # Create fresh baseline at 90%
        baseline_xml = tmp_path / "coverage.xml"
        baseline_xml.write_text(
            '<?xml version="1.0"?>\n<coverage line-rate="0.90" branch-rate="0.85" />'
        )
        import os

        future_time = 4102444800
        os.utime(baseline_xml, (future_time, future_time))

        # Create spec with no-decrease mode
        spec = ValidationSpec(
            commands=[
                ValidationCommand(
                    name="pytest",
                    command=["echo", "test"],
                    kind=CommandKind.TEST,
                ),
            ],
            scope=ValidationScope.PER_ISSUE,
            coverage=CoverageConfig(
                enabled=True,
                min_percent=None,  # No-decrease mode
            ),
            e2e=E2EConfig(enabled=False),
        )

        context = ValidationContext(
            issue_id="test-123",
            repo_path=tmp_path,
            commit_hash="",
            changed_files=[],
            scope=ValidationScope.PER_ISSUE,
        )

        # Create a worktree path for the test output
        worktree_path = tmp_path / "worktree"
        worktree_path.mkdir()

        # Create coverage.xml with lower coverage in worktree
        (worktree_path / "coverage.xml").write_text(
            '<?xml version="1.0"?>\n<coverage line-rate="0.70" branch-rate="0.65" />'
        )

        mock_proc = mock_popen_success(stdout="ok", stderr="", returncode=0)

        def mock_git_run(
            args: list[str], **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            if "status" in args and "--porcelain" in args:
                return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
            elif "log" in args:
                return subprocess.CompletedProcess(
                    args, 0, stdout="1700000000\n", stderr=""
                )
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        with (
            patch(
                "src.validation.command_runner.subprocess.Popen",
                return_value=mock_proc,
            ),
            patch("src.validation.coverage.subprocess.run", side_effect=mock_git_run),
        ):
            # Override context to use worktree path for coverage check
            result = runner._spec_runner._execute_spec_commands(
                spec=spec,
                context=context,
                cwd=worktree_path,
                artifacts=ValidationArtifacts(log_dir=tmp_path),
                log_dir=tmp_path,
                run_id="test",
                baseline_percent=90.0,  # Pass baseline explicitly
            )
            # Should fail because 70% < 90% baseline
            assert result.passed is False
            assert result.coverage_result is not None
            assert result.coverage_result.passed is False
            assert "70.0%" in (result.coverage_result.failure_reason or "")
            assert "90.0%" in (result.coverage_result.failure_reason or "")

    def test_no_decrease_mode_passes_when_coverage_increases(
        self, runner: ValidationRunner, tmp_path: Path
    ) -> None:
        """When current coverage exceeds baseline, validation passes."""
        worktree_path = tmp_path / "worktree"
        worktree_path.mkdir()

        # Create coverage.xml with higher coverage in worktree
        (worktree_path / "coverage.xml").write_text(
            '<?xml version="1.0"?>\n<coverage line-rate="0.95" branch-rate="0.90" />'
        )

        spec = ValidationSpec(
            commands=[
                ValidationCommand(
                    name="pytest",
                    command=["echo", "test"],
                    kind=CommandKind.TEST,
                ),
            ],
            scope=ValidationScope.PER_ISSUE,
            coverage=CoverageConfig(enabled=True, min_percent=None),
            e2e=E2EConfig(enabled=False),
        )

        context = ValidationContext(
            issue_id="test-123",
            repo_path=tmp_path,
            commit_hash="",
            changed_files=[],
            scope=ValidationScope.PER_ISSUE,
        )

        mock_proc = mock_popen_success(stdout="ok", stderr="", returncode=0)

        with patch(
            "src.validation.command_runner.subprocess.Popen",
            return_value=mock_proc,
        ):
            result = runner._spec_runner._execute_spec_commands(
                spec=spec,
                context=context,
                cwd=worktree_path,
                artifacts=ValidationArtifacts(log_dir=tmp_path),
                log_dir=tmp_path,
                run_id="test",
                baseline_percent=80.0,  # Lower baseline
            )
            # Should pass because 95% > 80% baseline
            assert result.passed is True
            assert result.coverage_result is not None
            assert result.coverage_result.passed is True
            assert result.coverage_result.percent == 95.0

    def test_explicit_threshold_overrides_no_decrease_mode(
        self, runner: ValidationRunner, tmp_path: Path
    ) -> None:
        """When min_percent is explicitly set, baseline is not used."""
        worktree_path = tmp_path / "worktree"
        worktree_path.mkdir()

        # Create coverage.xml at 75%
        (worktree_path / "coverage.xml").write_text(
            '<?xml version="1.0"?>\n<coverage line-rate="0.75" branch-rate="0.70" />'
        )

        # Explicit threshold of 70% - should pass
        spec = ValidationSpec(
            commands=[
                ValidationCommand(
                    name="pytest",
                    command=["echo", "test"],
                    kind=CommandKind.TEST,
                ),
            ],
            scope=ValidationScope.PER_ISSUE,
            coverage=CoverageConfig(
                enabled=True,
                min_percent=70.0,  # Explicit threshold
            ),
            e2e=E2EConfig(enabled=False),
        )

        context = ValidationContext(
            issue_id="test-123",
            repo_path=tmp_path,
            commit_hash="",
            changed_files=[],
            scope=ValidationScope.PER_ISSUE,
        )

        mock_proc = mock_popen_success(stdout="ok", stderr="", returncode=0)

        with patch(
            "src.validation.command_runner.subprocess.Popen",
            return_value=mock_proc,
        ):
            result = runner._spec_runner._execute_spec_commands(
                spec=spec,
                context=context,
                cwd=worktree_path,
                artifacts=ValidationArtifacts(log_dir=tmp_path),
                log_dir=tmp_path,
                run_id="test",
                baseline_percent=90.0,  # This should be ignored
            )
            # Should pass because 75% >= 70% (explicit threshold used, not baseline)
            assert result.passed is True
            assert result.coverage_result is not None
            assert result.coverage_result.passed is True


class TestSpecRunnerBaselineRefresh:
    """Tests for the SpecValidationRunner's baseline auto-refresh behavior.

    The runner should automatically refresh baseline when:
    1. Baseline file is missing
    2. Baseline file is stale (older than last commit)
    3. Repo has uncommitted changes
    """

    @pytest.fixture
    def runner(self, tmp_path: Path) -> ValidationRunner:
        """Create a runner with coverage enabled."""
        config = ValidationConfig(
            run_slow_tests=False,
            run_e2e=False,
            coverage=True,
            use_test_mutex=False,
        )
        return ValidationRunner(tmp_path, config)

    def test_baseline_refresh_when_missing(
        self, runner: ValidationRunner, tmp_path: Path
    ) -> None:
        """When baseline is missing, runner should refresh it."""
        from src.validation.spec_runner import SpecValidationRunner

        spec_runner = SpecValidationRunner(tmp_path)

        spec = ValidationSpec(
            commands=[
                ValidationCommand(
                    name="pytest",
                    command=["uv", "run", "pytest"],
                    kind=CommandKind.TEST,
                ),
            ],
            scope=ValidationScope.PER_ISSUE,
            coverage=CoverageConfig(enabled=True, min_percent=None),
            e2e=E2EConfig(enabled=False),
        )

        # Mock worktree and commands for refresh
        from src.validation.worktree import WorktreeContext, WorktreeState

        mock_worktree = MagicMock(spec=WorktreeContext)
        mock_worktree.state = WorktreeState.CREATED
        mock_worktree.path = tmp_path / "baseline-worktree"
        mock_worktree.path.mkdir()

        # Create coverage.xml in worktree
        (mock_worktree.path / "coverage.xml").write_text(
            '<?xml version="1.0"?>\n<coverage line-rate="0.85" branch-rate="0.80" />'
        )

        mock_proc = mock_popen_success(stdout="ok", stderr="", returncode=0)

        def mock_git_run(
            args: list[str], **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            if "status" in args and "--porcelain" in args:
                return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
            elif "log" in args:
                return subprocess.CompletedProcess(
                    args, 0, stdout="1700000000\n", stderr=""
                )
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        # Ensure baseline doesn't exist initially
        baseline_path = tmp_path / "coverage.xml"
        assert not baseline_path.exists()

        with (
            patch(
                "src.validation.spec_runner.create_worktree",
                return_value=mock_worktree,
            ),
            patch(
                "src.validation.spec_runner.remove_worktree",
                return_value=mock_worktree,
            ),
            patch(
                "src.validation.command_runner.subprocess.Popen",
                return_value=mock_proc,
            ),
            patch("src.validation.coverage.subprocess.run", side_effect=mock_git_run),
            patch(
                "src.validation.spec_runner.try_lock", return_value=True
            ),  # Get lock immediately
        ):
            baseline = spec_runner._refresh_baseline(spec, tmp_path)

        # Baseline should be created
        assert baseline == 85.0
        assert baseline_path.exists()

    def test_baseline_refresh_skipped_when_fresh(
        self, runner: ValidationRunner, tmp_path: Path
    ) -> None:
        """When baseline is fresh, refresh should be skipped."""
        from src.validation.spec_runner import SpecValidationRunner

        spec_runner = SpecValidationRunner(tmp_path)

        # Create fresh baseline
        baseline_path = tmp_path / "coverage.xml"
        baseline_path.write_text(
            '<?xml version="1.0"?>\n<coverage line-rate="0.88" branch-rate="0.82" />'
        )
        import os

        future_time = 4102444800
        os.utime(baseline_path, (future_time, future_time))

        spec = ValidationSpec(
            commands=[
                ValidationCommand(
                    name="pytest",
                    command=["uv", "run", "pytest"],
                    kind=CommandKind.TEST,
                ),
            ],
            scope=ValidationScope.PER_ISSUE,
            coverage=CoverageConfig(enabled=True, min_percent=None),
            e2e=E2EConfig(enabled=False),
        )

        def mock_git_run(
            args: list[str], **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            if "status" in args and "--porcelain" in args:
                return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
            elif "log" in args:
                return subprocess.CompletedProcess(
                    args, 0, stdout="1700000000\n", stderr=""
                )
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        worktree_created = False

        def mock_create_worktree(*args: object, **kwargs: object) -> MagicMock:
            nonlocal worktree_created
            worktree_created = True
            raise AssertionError("Should not create worktree when baseline is fresh")

        with (
            patch(
                "src.validation.spec_runner.create_worktree",
                side_effect=mock_create_worktree,
            ),
            patch("src.validation.coverage.subprocess.run", side_effect=mock_git_run),
        ):
            baseline = spec_runner._refresh_baseline(spec, tmp_path)

        # Should return cached baseline without creating worktree
        assert baseline == 88.0
        assert not worktree_created

    def test_baseline_refresh_with_lock_contention(
        self, runner: ValidationRunner, tmp_path: Path
    ) -> None:
        """When another agent holds the lock, wait and use their refreshed baseline."""
        from src.validation.spec_runner import SpecValidationRunner

        spec_runner = SpecValidationRunner(tmp_path)

        spec = ValidationSpec(
            commands=[
                ValidationCommand(
                    name="pytest",
                    command=["uv", "run", "pytest"],
                    kind=CommandKind.TEST,
                ),
            ],
            scope=ValidationScope.PER_ISSUE,
            coverage=CoverageConfig(enabled=True, min_percent=None),
            e2e=E2EConfig(enabled=False),
        )

        # Initially no baseline
        baseline_path = tmp_path / "coverage.xml"
        assert not baseline_path.exists()

        def mock_git_run_stale(
            args: list[str], **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            """Simulate stale baseline initially."""
            if "status" in args and "--porcelain" in args:
                return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
            elif "log" in args:
                return subprocess.CompletedProcess(
                    args, 0, stdout="1700000000\n", stderr=""
                )
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        # Simulate another agent refreshing baseline while we wait
        wait_called = False

        def mock_wait_for_lock(*args: object, **kwargs: object) -> bool:
            nonlocal wait_called
            wait_called = True
            # Simulate other agent creating the baseline
            baseline_path.write_text(
                '<?xml version="1.0"?>\n<coverage line-rate="0.92" branch-rate="0.87" />'
            )
            import os

            future_time = 4102444800
            os.utime(baseline_path, (future_time, future_time))
            return True

        def mock_git_run_fresh(
            args: list[str], **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            """After wait, baseline is fresh."""
            if "status" in args and "--porcelain" in args:
                return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
            elif "log" in args:
                return subprocess.CompletedProcess(
                    args, 0, stdout="1700000000\n", stderr=""
                )
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        with (
            patch(
                "src.validation.spec_runner.try_lock", return_value=False
            ),  # Lock held by other
            patch(
                "src.validation.spec_runner.wait_for_lock",
                side_effect=mock_wait_for_lock,
            ),
            patch(
                "src.validation.coverage.subprocess.run", side_effect=mock_git_run_fresh
            ),
        ):
            baseline = spec_runner._refresh_baseline(spec, tmp_path)

        # Should use the baseline created by the other agent
        assert baseline == 92.0
        assert wait_called
