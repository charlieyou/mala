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

        def capture_run(
            *args: object, **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
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
        exc.stdout = "string output"  # type: ignore[assignment]  # Already a string, not bytes
        exc.stderr = "string error"  # type: ignore[assignment]

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
        mock_result = subprocess.CompletedProcess(
            args=["echo", "hello"],
            returncode=0,
            stdout="hello\n",
            stderr="",
        )
        with patch("subprocess.run", return_value=mock_result):
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
        mock_result = subprocess.CompletedProcess(
            args=["echo", "hello"],
            returncode=1,
            stdout="",
            stderr="error occurred",
        )
        with patch("subprocess.run", return_value=mock_result):
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
                    kind=CommandKind.DEPS,
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

        mock_result = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="ok",
            stderr="",
        )
        with patch("subprocess.run", return_value=mock_result):
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
                    kind=CommandKind.DEPS,
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

        def mock_run(
            *args: object, **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                return subprocess.CompletedProcess(
                    args=[], returncode=1, stdout="", stderr="lint error"
                )
            return subprocess.CompletedProcess(
                args=[], returncode=0, stdout="ok", stderr=""
            )

        with patch("subprocess.run", side_effect=mock_run):
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
                    kind=CommandKind.DEPS,
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

        def mock_run(
            *args: object, **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                return subprocess.CompletedProcess(
                    args=[], returncode=1, stdout="", stderr="lint warning"
                )
            return subprocess.CompletedProcess(
                args=[], returncode=0, stdout="ok", stderr=""
            )

        with patch("subprocess.run", side_effect=mock_run):
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

        def capture_run(
            cmd: list[str], *args: object, **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            captured_cmd.extend(cmd)
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="ok", stderr=""
            )

        with patch("subprocess.run", side_effect=capture_run):
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
        mock_result = subprocess.CompletedProcess(
            args=["echo", "hello"],
            returncode=0,
            stdout="stdout content\n",
            stderr="stderr content\n",
        )
        with patch("subprocess.run", return_value=mock_result):
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

        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="ok", stderr=""
        )
        with patch("subprocess.run", return_value=mock_result):
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

        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="ok", stderr=""
        )
        with patch("subprocess.run", return_value=mock_result):
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

        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="ok", stderr=""
        )
        with patch("subprocess.run", return_value=mock_result):
            result = runner._run_spec_sync(spec, context, log_dir=tmp_path)
            assert result.passed is False
            assert "not found" in result.failure_reasons[0]

    def test_run_spec_e2e_only_for_run_level(
        self, runner: ValidationRunner, context: ValidationContext, tmp_path: Path
    ) -> None:
        """Test that E2E only runs for run-level scope."""
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

        with patch.object(runner, "_run_spec_e2e", side_effect=mock_e2e_run):
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

        with patch.object(runner, "_run_spec_e2e", return_value=mock_e2e_result):
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
        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="ok", stderr=""
        )
        with patch("subprocess.run", return_value=mock_result):
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
        exc = subprocess.TimeoutExpired(cmd=["sleep"], timeout=1.0)
        exc.stdout = b"partial"
        exc.stderr = b"timeout"

        with patch("subprocess.run", side_effect=exc):
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

        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="ok", stderr=""
        )

        with (
            patch(
                "src.validation.runner.create_worktree", return_value=mock_worktree_ctx
            ),
            patch(
                "src.validation.runner.remove_worktree", return_value=mock_worktree_ctx
            ),
            patch("subprocess.run", return_value=mock_result),
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
            "src.validation.runner.create_worktree", return_value=mock_worktree_ctx
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
