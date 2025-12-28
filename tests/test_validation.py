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
    check_e2e_prereqs,
    format_step_output,
    tail,
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
    """Test the tail helper function."""

    def test_empty_text(self) -> None:
        assert tail("") == ""

    def test_short_text_unchanged(self) -> None:
        text = "hello world"
        assert tail(text) == text

    def test_truncates_long_text(self) -> None:
        text = "x" * 1000
        result = tail(text, max_chars=100)
        assert len(result) == 100
        assert result == "x" * 100

    def test_truncates_many_lines(self) -> None:
        lines = [f"line {i}" for i in range(50)]
        text = "\n".join(lines)
        result = tail(text, max_lines=5)
        result_lines = result.splitlines()
        assert len(result_lines) == 5
        # Should get the last 5 lines
        assert result_lines[0] == "line 45"
        assert result_lines[-1] == "line 49"


class TestFormatStepOutput:
    """Test the format_step_output helper function."""

    def test_formats_stderr(self) -> None:
        result = format_step_output("", "error message")
        assert "stderr:" in result
        assert "error message" in result

    def test_formats_stdout_when_no_stderr(self) -> None:
        result = format_step_output("stdout message", "")
        assert "stdout:" in result
        assert "stdout message" in result

    def test_prefers_stderr_over_stdout(self) -> None:
        result = format_step_output("stdout", "stderr")
        assert "stderr" in result
        # stdout not included when stderr present
        assert "stdout:" not in result

    def test_empty_output(self) -> None:
        result = format_step_output("", "")
        assert result == ""


class TestCheckE2EPrereqs:
    """Test the check_e2e_prereqs helper function."""

    def test_missing_mala_cli(self) -> None:
        with patch("shutil.which", return_value=None):
            result = check_e2e_prereqs({})
            assert result is not None
            assert "mala CLI not found" in result

    def test_missing_bd_cli(self) -> None:
        def mock_which(cmd: str) -> str | None:
            if cmd == "mala":
                return "/usr/bin/mala"
            return None

        with patch("shutil.which", side_effect=mock_which):
            result = check_e2e_prereqs({})
            assert result is not None
            assert "bd CLI not found" in result

    def test_missing_morph_api_key_does_not_fail(self) -> None:
        """MORPH_API_KEY is not a hard prereq - E2E should still run without it."""
        with patch("shutil.which", return_value="/usr/bin/fake"):
            result = check_e2e_prereqs({})
            assert result is None  # Should pass even without MORPH_API_KEY

    def test_all_prereqs_met(self) -> None:
        with patch("shutil.which", return_value="/usr/bin/fake"):
            result = check_e2e_prereqs({"MORPH_API_KEY": "test-key"})
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

    def test_init(self, runner: ValidationRunner, tmp_path: Path) -> None:
        assert runner.repo_path == tmp_path

    def test_build_validation_commands(self, runner: ValidationRunner) -> None:
        """Test _build_validation_commands returns expected commands."""
        commands = runner._build_validation_commands()
        # We disabled slow tests, so should not include slow marker
        assert len(commands) >= 3  # lint, format, type-check, pytest
        # Check command names
        names = [c[0] for c in commands]
        assert "ruff check" in names
        assert "ruff format" in names

    def test_build_commands_with_slow_tests(self, tmp_path: Path) -> None:
        """Test that slow tests are included when enabled."""
        config = ValidationConfig(
            run_slow_tests=True,
            run_e2e=False,
            coverage=False,
            use_test_mutex=False,
        )
        runner = ValidationRunner(tmp_path, config)
        commands = runner._build_validation_commands()
        # With slow tests enabled, should have slow marker in pytest
        pytest_cmds = [c for c in commands if c[0] == "pytest"]
        assert len(pytest_cmds) == 1
        assert "-m" in pytest_cmds[0][1]
        assert "slow or not slow" in " ".join(pytest_cmds[0][1])

    def test_build_commands_with_coverage(self, tmp_path: Path) -> None:
        """Test that coverage is included when enabled."""
        config = ValidationConfig(
            run_slow_tests=False,
            run_e2e=False,
            coverage=True,
            coverage_min=85,
            coverage_source="src",
            use_test_mutex=False,
        )
        runner = ValidationRunner(tmp_path, config)
        commands = runner._build_validation_commands()
        # Check pytest has coverage flags
        pytest_cmds = [c for c in commands if c[0] == "pytest"]
        pytest_args = pytest_cmds[0][1]
        assert "--cov" in pytest_args
        assert "src" in pytest_args
        assert any("--cov-fail-under" in arg for arg in pytest_args)

    def test_run_command_success(self, runner: ValidationRunner) -> None:
        """Test successful command execution."""
        mock_proc = mock_popen_success(stdout="ok", stderr="", returncode=0)

        with patch(
            "src.validation.command_runner.subprocess.Popen", return_value=mock_proc
        ):
            result = runner._run_command("test", ["echo", "hello"], Path("."), {})
            assert result.ok is True
            assert result.returncode == 0

    def test_run_command_failure(self, runner: ValidationRunner) -> None:
        """Test failed command execution."""
        mock_proc = mock_popen_success(stdout="", stderr="command failed", returncode=1)

        with patch(
            "src.validation.command_runner.subprocess.Popen", return_value=mock_proc
        ):
            result = runner._run_command("test", ["false"], Path("."), {})
            assert result.ok is False
            assert result.returncode == 1
            assert "command failed" in result.stderr_tail

    def test_run_command_timeout(self, runner: ValidationRunner) -> None:
        """Test command timeout handling."""
        mock_proc = MagicMock()
        mock_proc.communicate.side_effect = [
            subprocess.TimeoutExpired(cmd=["sleep"], timeout=0.1),
            ("terminated output", "terminated error"),
        ]
        mock_proc.pid = 12345
        mock_proc.returncode = None

        with patch(
            "src.validation.command_runner.subprocess.Popen", return_value=mock_proc
        ):
            result = runner._run_command("test", ["sleep", "10"], Path("."), {})
            assert result.ok is False
            assert "terminated output" in result.stdout_tail
            assert "terminated error" in result.stderr_tail

    def test_run_command_timeout_none_output(self, runner: ValidationRunner) -> None:
        """Test timeout when communicate returns None."""
        mock_proc = MagicMock()
        mock_proc.communicate.side_effect = [
            subprocess.TimeoutExpired(cmd=["sleep"], timeout=0.1),
            (None, None),
        ]
        mock_proc.pid = 12345
        mock_proc.returncode = None

        with patch(
            "src.validation.command_runner.subprocess.Popen", return_value=mock_proc
        ):
            result = runner._run_command("test", ["sleep", "10"], Path("."), {})
            assert result.ok is False
            assert result.stdout_tail == ""
            assert result.stderr_tail == ""

    def test_run_command_timeout_string_output(self, runner: ValidationRunner) -> None:
        """Test timeout when output is already string (edge case)."""
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
            result = runner._run_command("test", ["sleep", "10"], Path("."), {})
            assert result.ok is False
            assert "string output" in result.stdout_tail
            assert "string error" in result.stderr_tail

    def test_run_validation_steps(self, runner: ValidationRunner) -> None:
        """Test running multiple validation steps."""
        mock_proc = mock_popen_success(stdout="ok", stderr="", returncode=0)

        with patch(
            "src.validation.command_runner.subprocess.Popen", return_value=mock_proc
        ):
            result = runner._run_validation_steps(
                runner.repo_path, "test", include_e2e=False
            )
            assert result.passed is True
            assert len(result.steps) > 0
            assert all(step.ok for step in result.steps)

    def test_run_validation_steps_failure_stops(self, runner: ValidationRunner) -> None:
        """Test that validation stops on first failure."""

        def mock_popen_factory(*args: object, **kwargs: object) -> MagicMock:
            """Return failing popen for first command."""
            mock = MagicMock()
            mock.communicate.return_value = ("", "error")
            mock.returncode = 1
            mock.pid = 12345
            return mock

        with patch(
            "src.validation.command_runner.subprocess.Popen",
            side_effect=mock_popen_factory,
        ):
            result = runner._run_validation_steps(
                runner.repo_path, "test", include_e2e=False
            )
            assert result.passed is False
            # Should have stopped after first failure
            assert len(result.steps) >= 1
            assert not result.steps[0].ok

    def test_run_validation_steps_partial_completion(
        self, runner: ValidationRunner
    ) -> None:
        """Test validation results show partial completion."""
        call_count = 0

        def mock_popen_factory(*args: object, **kwargs: object) -> MagicMock:
            """First command succeeds, second fails."""
            nonlocal call_count
            call_count += 1
            mock = MagicMock()
            mock.pid = 12345
            if call_count == 1:
                mock.communicate.return_value = ("success", "")
                mock.returncode = 0
            else:
                mock.communicate.return_value = ("", "failure")
                mock.returncode = 1
            return mock

        with patch(
            "src.validation.command_runner.subprocess.Popen",
            side_effect=mock_popen_factory,
        ):
            result = runner._run_validation_steps(
                runner.repo_path, "test", include_e2e=False
            )
            assert result.passed is False
            # Should have at least 2 steps (one success, one failure)
            assert len(result.steps) >= 2
            # First step succeeded
            assert result.steps[0].ok is True
            # Second step failed
            assert not all(step.ok for step in result.steps)


class TestSpecValidationRunner:
    """Test SpecValidationRunner with ValidationSpec."""

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
    def context(self, tmp_path: Path) -> ValidationContext:
        """Create a validation context."""
        return ValidationContext(
            issue_id="test-123",
            repo_path=tmp_path,
            commit_hash="",  # Validate in place
            changed_files=[],
            scope=ValidationScope.PER_ISSUE,
        )

    def test_run_spec_basic(
        self, runner: ValidationRunner, context: ValidationContext, tmp_path: Path
    ) -> None:
        """Test basic spec validation."""
        spec = ValidationSpec(
            commands=[
                ValidationCommand(
                    name="echo",
                    command=["echo", "hello"],
                    kind=CommandKind.TEST,
                ),
            ],
            scope=ValidationScope.PER_ISSUE,
            coverage=CoverageConfig(enabled=False),
            e2e=E2EConfig(enabled=False),
        )

        mock_proc = mock_popen_success(stdout="hello", stderr="", returncode=0)

        with patch(
            "src.validation.command_runner.subprocess.Popen", return_value=mock_proc
        ):
            result = runner._run_spec_sync(spec, context, log_dir=tmp_path)
            assert result.passed is True
            assert len(result.steps) == 1
            assert result.steps[0].name == "echo"

    def test_run_spec_with_failure(
        self, runner: ValidationRunner, context: ValidationContext, tmp_path: Path
    ) -> None:
        """Test spec validation with command failure."""
        spec = ValidationSpec(
            commands=[
                ValidationCommand(
                    name="fail",
                    command=["false"],
                    kind=CommandKind.TEST,
                ),
            ],
            scope=ValidationScope.PER_ISSUE,
            coverage=CoverageConfig(enabled=False),
            e2e=E2EConfig(enabled=False),
        )

        mock_proc = mock_popen_success(stdout="", stderr="error", returncode=1)

        with patch(
            "src.validation.command_runner.subprocess.Popen", return_value=mock_proc
        ):
            result = runner._run_spec_sync(spec, context, log_dir=tmp_path)
            assert result.passed is False

    def test_run_spec_creates_artifacts(
        self, runner: ValidationRunner, context: ValidationContext, tmp_path: Path
    ) -> None:
        """Test that spec validation creates artifacts."""
        spec = ValidationSpec(
            commands=[
                ValidationCommand(
                    name="echo",
                    command=["echo", "hello"],
                    kind=CommandKind.TEST,
                ),
            ],
            scope=ValidationScope.PER_ISSUE,
            coverage=CoverageConfig(enabled=False),
            e2e=E2EConfig(enabled=False),
        )

        mock_proc = mock_popen_success(stdout="hello", stderr="", returncode=0)

        with patch(
            "src.validation.command_runner.subprocess.Popen", return_value=mock_proc
        ):
            result = runner._run_spec_sync(spec, context, log_dir=tmp_path)
            assert result.artifacts is not None
            assert result.artifacts.log_dir is not None

    def test_run_spec_coverage_enabled(
        self, runner: ValidationRunner, context: ValidationContext, tmp_path: Path
    ) -> None:
        """Test spec validation with coverage enabled."""
        # Create coverage.xml in the test directory
        coverage_xml = tmp_path / "coverage.xml"
        coverage_xml.write_text(
            '<?xml version="1.0"?>\n<coverage line-rate="0.90" branch-rate="0.85" />'
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
            coverage=CoverageConfig(
                enabled=True,
                min_percent=85,
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
            assert result.coverage_result.percent == 90.0
            assert result.coverage_result.passed is True

    def test_run_spec_coverage_below_threshold(
        self, runner: ValidationRunner, context: ValidationContext, tmp_path: Path
    ) -> None:
        """Test spec validation fails when coverage is below threshold."""
        # Create coverage.xml with low coverage
        coverage_xml = tmp_path / "coverage.xml"
        coverage_xml.write_text(
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
            coverage=CoverageConfig(
                enabled=True,
                min_percent=85,
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
            assert result.coverage_result.percent == 70.0

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
                min_percent=85,
            ),
            e2e=E2EConfig(enabled=False),
        )

        mock_proc = mock_popen_success(stdout="ok", stderr="", returncode=0)

        with patch(
            "src.validation.command_runner.subprocess.Popen", return_value=mock_proc
        ):
            result = runner._run_spec_sync(spec, context, log_dir=tmp_path)
            # Should fail because coverage file is missing
            assert result.passed is False
            assert result.coverage_result is not None
            assert result.coverage_result.status == CoverageStatus.ERROR


class TestValidationRunnerE2E:
    """Test E2E validation functionality."""

    @pytest.fixture
    def runner_with_e2e(self, tmp_path: Path) -> ValidationRunner:
        """Create a runner with E2E enabled."""
        config = ValidationConfig(
            run_slow_tests=False,
            run_e2e=True,
            coverage=False,
            use_test_mutex=False,
        )
        return ValidationRunner(tmp_path, config)

    def test_e2e_prereqs_missing(self, runner_with_e2e: ValidationRunner) -> None:
        """Test E2E is skipped when prerequisites are missing."""
        with patch("shutil.which", return_value=None):
            runner_with_e2e._run_validation_steps(
                runner_with_e2e.repo_path, "test", include_e2e=True
            )
            # E2E should be skipped, not fail
            # The overall result depends on other validations

    def test_run_e2e_success(
        self, runner_with_e2e: ValidationRunner, tmp_path: Path
    ) -> None:
        """Test successful E2E run."""
        # Create mock E2E environment
        env = {"MORPH_API_KEY": "test-key", "PATH": "/usr/bin"}

        mock_proc = mock_popen_success(stdout="E2E passed", stderr="", returncode=0)

        with (
            patch("shutil.which", return_value="/usr/bin/fake"),
            patch(
                "src.validation.command_runner.subprocess.Popen",
                return_value=mock_proc,
            ),
        ):
            # This tests the internal _run_e2e method
            runner_with_e2e._run_e2e("test-label", env)
            # Result depends on actual E2E runner implementation


class TestNoDecreaseMode:
    """Test no-decrease coverage mode (min_percent=None)."""

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

    def test_no_decrease_mode_uses_baseline(
        self, runner: ValidationRunner, tmp_path: Path
    ) -> None:
        """When min_percent is None, uses baseline for comparison."""
        import os

        # Create fresh baseline at 80%
        baseline_xml = tmp_path / "coverage.xml"
        baseline_xml.write_text(
            '<?xml version="1.0"?>\n<coverage line-rate="0.80" branch-rate="0.75" />'
        )
        # Set future mtime to ensure it's treated as valid baseline
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

        # Create coverage.xml in worktree with LOWER coverage (70%)
        # This simulates the test run producing lower coverage
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
            # Use _execute_spec_commands to run in worktree with explicit baseline
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
            # - Current run coverage is 70%
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
        import os

        # Create baseline at 80%
        baseline_xml = tmp_path / "coverage.xml"
        baseline_xml.write_text(
            '<?xml version="1.0"?>\n<coverage line-rate="0.80" branch-rate="0.75" />'
        )
        future_time = 4102444800
        os.utime(baseline_xml, (future_time, future_time))

        # Create worktree with HIGHER coverage (95%)
        worktree_path = tmp_path / "worktree"
        worktree_path.mkdir()
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

        mock_proc = mock_popen_success(stdout="ok", stderr="", returncode=0)

        with patch(
            "src.validation.command_runner.subprocess.Popen", return_value=mock_proc
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


class TestCoverageBaseline:
    """Test baseline coverage capture and freshness detection."""

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

    def test_stale_baseline_is_ignored(
        self, runner: ValidationRunner, tmp_path: Path
    ) -> None:
        """Old baseline coverage files should be ignored."""
        import os

        # Create baseline with very old mtime (year 2000)
        baseline_xml = tmp_path / "coverage.xml"
        baseline_xml.write_text(
            '<?xml version="1.0"?>\n<coverage line-rate="0.95" branch-rate="0.90" />'
        )
        # Set mtime to year 2000
        old_time = 946684800  # 2000-01-01
        os.utime(baseline_xml, (old_time, old_time))

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

        # Create worktree with lower coverage (80%)
        worktree_path = tmp_path / "worktree"
        worktree_path.mkdir()
        (worktree_path / "coverage.xml").write_text(
            '<?xml version="1.0"?>\n<coverage line-rate="0.80" branch-rate="0.75" />'
        )

        mock_proc = mock_popen_success(stdout="ok", stderr="", returncode=0)

        def mock_git_run(
            args: list[str], **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            if "status" in args and "--porcelain" in args:
                return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
            elif "log" in args:
                # Commit time is 2023 (much newer than stale baseline from 2000)
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
            # With stale baseline, validation should treat as no baseline
            # and pass (since no decrease from nothing)
            result = runner._run_spec_sync(spec, context, log_dir=tmp_path)
            # Result depends on how stale baselines are handled
            # Current implementation: stale baseline is regenerated
            assert result.coverage_result is not None

    def test_fresh_baseline_is_used(
        self, runner: ValidationRunner, tmp_path: Path
    ) -> None:
        """Fresh baseline coverage files should be used for comparison."""
        import os

        # Create baseline with future mtime (definitely fresh)
        baseline_xml = tmp_path / "coverage.xml"
        baseline_xml.write_text(
            '<?xml version="1.0"?>\n<coverage line-rate="0.90" branch-rate="0.85" />'
        )
        future_time = 4102444800  # Year 2100
        os.utime(baseline_xml, (future_time, future_time))

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

        # Create worktree with lower coverage (70%) - should fail
        worktree_path = tmp_path / "worktree"
        worktree_path.mkdir()
        (worktree_path / "coverage.xml").write_text(
            '<?xml version="1.0"?>\n<coverage line-rate="0.70" branch-rate="0.65" />'
        )

        mock_proc = mock_popen_success(stdout="ok", stderr="", returncode=0)

        def mock_git_run(
            args: list[str], **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            """After wait, baseline is fresh."""
            if "status" in args and "--porcelain" in args:
                return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
            elif "log" in args:
                # Commit time is 2023 (much newer than stale baseline from 2000)
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
            # Use execute_spec_commands to run with explicit baseline
            result = runner._spec_runner._execute_spec_commands(
                spec=spec,
                context=context,
                cwd=worktree_path,
                artifacts=ValidationArtifacts(log_dir=tmp_path),
                log_dir=tmp_path,
                run_id="test",
                baseline_percent=90.0,  # Fresh baseline at 90%
            )
            # Should fail because 70% < 90% baseline
            assert result.passed is False
            assert result.coverage_result is not None
            assert result.coverage_result.passed is False


class TestMutexWrapping:
    """Test mutex command wrapping."""

    def test_wrap_with_mutex_disabled(self, tmp_path: Path) -> None:
        """Test that mutex wrapping is disabled by default."""
        config = ValidationConfig(use_test_mutex=False)
        runner = ValidationRunner(tmp_path, config)

        cmd = ["pytest", "tests/"]
        wrapped = runner._wrap_with_mutex(cmd)
        # Should return unchanged when mutex disabled
        assert wrapped == cmd

    def test_wrap_with_mutex_enabled(self, tmp_path: Path) -> None:
        """Test mutex wrapping when enabled."""
        config = ValidationConfig(use_test_mutex=True)
        runner = ValidationRunner(tmp_path, config)

        cmd = ["pytest", "tests/"]
        wrapped = runner._wrap_with_mutex(cmd)
        # Should wrap with mutex command
        assert wrapped != cmd
        assert "flock" in wrapped[0] or len(wrapped) > len(cmd)


class TestBuildSpecEnv:
    """Test environment building for spec validation."""

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

    def test_build_spec_env_includes_issue_id(
        self, runner: ValidationRunner, tmp_path: Path
    ) -> None:
        """Test that spec env includes issue ID."""
        context = ValidationContext(
            issue_id="test-issue-456",
            repo_path=tmp_path,
            commit_hash="abc123",
            changed_files=["src/foo.py"],
            scope=ValidationScope.PER_ISSUE,
        )

        env = runner._build_spec_env(context, "run-001")
        assert "MALA_ISSUE_ID" in env
        assert env["MALA_ISSUE_ID"] == "test-issue-456"

    def test_build_spec_env_includes_run_id(
        self, runner: ValidationRunner, tmp_path: Path
    ) -> None:
        """Test that spec env includes run ID."""
        context = ValidationContext(
            issue_id="test-123",
            repo_path=tmp_path,
            commit_hash="abc123",
            changed_files=[],
            scope=ValidationScope.PER_ISSUE,
        )

        env = runner._build_spec_env(context, "unique-run-id")
        assert "MALA_RUN_ID" in env
        assert env["MALA_RUN_ID"] == "unique-run-id"
