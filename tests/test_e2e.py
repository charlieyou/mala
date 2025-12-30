"""Unit tests for src/validation/e2e.py - E2E fixture runner."""

import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.tools.command_runner import CommandResult
from src.validation.e2e import (
    E2EConfig,
    E2EPrereqResult,
    E2EResult,
    E2ERunner,
    E2EStatus,
    check_e2e_prereqs,
)
from src.validation.helpers import (
    annotate_issue,
    decode_timeout_output,
    get_ready_issue_id,
    init_fixture_repo,
    tail,
    write_fixture_repo,
)


class TestE2EPrereqResult:
    """Test E2EPrereqResult dataclass."""

    def test_ok_result(self) -> None:
        result = E2EPrereqResult(ok=True)
        assert result.ok is True
        assert result.missing == []
        assert result.can_skip is False
        assert result.failure_reason() is None

    def test_failed_result_with_missing(self) -> None:
        result = E2EPrereqResult(
            ok=False,
            missing=["mala CLI not found", "MORPH_API_KEY not set"],
        )
        assert result.ok is False
        reason = result.failure_reason()
        assert reason is not None
        assert "mala CLI not found" in reason
        assert "MORPH_API_KEY not set" in reason

    def test_failed_result_empty_missing(self) -> None:
        result = E2EPrereqResult(ok=False, missing=[])
        assert result.failure_reason() == "E2E prerequisites not met"

    def test_can_skip_flag(self) -> None:
        result = E2EPrereqResult(
            ok=False, missing=["MORPH_API_KEY not set"], can_skip=True
        )
        assert result.can_skip is True


class TestE2EResult:
    """Test E2EResult dataclass and methods."""

    def test_passed_result(self) -> None:
        result = E2EResult(passed=True, status=E2EStatus.PASSED)
        assert result.passed is True
        assert result.status == E2EStatus.PASSED
        assert result.failure_reason is None
        assert result.short_summary() == "E2E passed"

    def test_failed_result(self) -> None:
        result = E2EResult(
            passed=False,
            status=E2EStatus.FAILED,
            failure_reason="mala exited 1",
        )
        assert result.passed is False
        assert result.status == E2EStatus.FAILED
        assert result.short_summary() == "E2E failed: mala exited 1"

    def test_skipped_result(self) -> None:
        result = E2EResult(
            passed=True,  # Skipped counts as "not failed"
            status=E2EStatus.SKIPPED,
            failure_reason="MORPH_API_KEY not set",
        )
        assert result.passed is True
        assert result.status == E2EStatus.SKIPPED
        assert "skipped" in result.short_summary()
        assert "MORPH_API_KEY" in result.short_summary()

    def test_failed_no_reason(self) -> None:
        result = E2EResult(passed=False, status=E2EStatus.FAILED)
        assert result.short_summary() == "E2E failed: unknown error"


class TestE2ERunnerPrereqs:
    """Test E2ERunner.check_prereqs method."""

    def test_all_prereqs_met(self) -> None:
        runner = E2ERunner()
        with patch("shutil.which", return_value="/usr/bin/fake"):
            result = runner.check_prereqs({"MORPH_API_KEY": "test-key"})
            assert result.ok is True
            assert result.missing == []

    def test_prereqs_ok_without_morph_api_key(self) -> None:
        """E2E prereqs should pass even without MORPH_API_KEY.

        The MORPH_API_KEY is only needed for Morph-specific tests,
        not for the core E2E validation to run.
        """
        runner = E2ERunner()
        with patch("shutil.which", return_value="/usr/bin/fake"):
            result = runner.check_prereqs({})
            assert result.ok is True
            assert result.missing == []

    def test_missing_mala_cli(self) -> None:
        runner = E2ERunner()
        with patch("shutil.which", return_value=None):
            result = runner.check_prereqs({})
            assert result.ok is False
            assert any("mala CLI" in m for m in result.missing)

    def test_missing_bd_cli(self) -> None:
        runner = E2ERunner()

        def mock_which(cmd: str) -> str | None:
            if cmd == "mala":
                return "/usr/bin/mala"
            return None

        with patch("shutil.which", side_effect=mock_which):
            result = runner.check_prereqs({})
            assert result.ok is False
            assert any("bd CLI" in m for m in result.missing)

    def test_uses_os_environ_when_none(self) -> None:
        runner = E2ERunner()
        with (
            patch("shutil.which", return_value="/usr/bin/fake"),
            patch.dict("os.environ", {}, clear=True),
        ):
            result = runner.check_prereqs(None)
            # Should pass even without MORPH_API_KEY
            assert result.ok is True


class TestE2ERunnerRun:
    """Test E2ERunner.run method."""

    def test_run_proceeds_without_morph_api_key(self, tmp_path: Path) -> None:
        """E2E should pass without MORPH_API_KEY - it's optional."""
        runner = E2ERunner()

        def mock_run_command(
            cmd: list[str], cwd: Path, **kwargs: object
        ) -> CommandResult:
            """Mock run_command for fixture setup commands (git, bd)."""
            return CommandResult(
                command=cmd,
                returncode=0,
                stdout='[{"id": "test-123"}]' if "ready" in cmd else "ok",
                stderr="",
            )

        with (
            patch("shutil.which", return_value="/usr/bin/fake"),
            patch("src.validation.helpers.run_command", side_effect=mock_run_command),
            patch("tempfile.mkdtemp", return_value=str(tmp_path / "fixture")),
            patch("shutil.rmtree"),
        ):
            (tmp_path / "fixture").mkdir()
            (tmp_path / "fixture" / "tests").mkdir()

            # Run WITHOUT MORPH_API_KEY - should still work
            result = runner.run(env={}, cwd=tmp_path)
            assert result.passed is True
            assert result.status == E2EStatus.PASSED

    def test_run_fails_on_missing_prereqs(self) -> None:
        runner = E2ERunner()
        with patch("shutil.which", return_value=None):
            result = runner.run(env={})
            assert result.passed is False
            assert result.status == E2EStatus.FAILED
            assert "mala CLI" in (result.failure_reason or "")

    def test_run_fails_on_fixture_setup_error(self) -> None:
        runner = E2ERunner()

        mock_run_result = CommandResult(
            command=[],
            returncode=1,
            stdout="",
            stderr="git init failed",
        )

        with (
            patch("shutil.which", return_value="/usr/bin/fake"),
            patch("src.validation.helpers.run_command", return_value=mock_run_result),
            patch("tempfile.mkdtemp", return_value="/tmp/test-fixture"),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.mkdir"),
            patch("pathlib.Path.write_text"),
            patch("shutil.rmtree"),
        ):
            result = runner.run(env={})
            assert result.passed is False
            assert "fixture setup" in (result.failure_reason or "")

    def test_run_success(self, tmp_path: Path) -> None:
        config = E2EConfig(keep_fixture=False)
        runner = E2ERunner(config)

        def mock_run_command(
            cmd: list[str], cwd: Path, **kwargs: object
        ) -> CommandResult:
            # Return success for all commands
            return CommandResult(
                command=cmd,
                returncode=0,
                stdout='[{"id": "test-123"}]' if "ready" in cmd else "ok",
                stderr="",
            )

        with (
            patch("shutil.which", return_value="/usr/bin/fake"),
            patch("src.validation.helpers.run_command", side_effect=mock_run_command),
            patch("tempfile.mkdtemp", return_value=str(tmp_path / "fixture")),
            patch("shutil.rmtree"),
        ):
            # Create the fixture path
            (tmp_path / "fixture").mkdir()
            (tmp_path / "fixture" / "tests").mkdir()

            result = runner.run(env={}, cwd=tmp_path)
            assert result.passed is True
            assert result.status == E2EStatus.PASSED

    def test_run_cleans_up_fixture(self, tmp_path: Path) -> None:
        config = E2EConfig(keep_fixture=False)
        runner = E2ERunner(config)

        cleanup_called = {"value": False}

        def mock_rmtree(*args: object, **kwargs: object) -> None:
            cleanup_called["value"] = True

        def mock_run_command(
            cmd: list[str], cwd: Path, **kwargs: object
        ) -> CommandResult:
            return CommandResult(command=cmd, returncode=0, stdout="", stderr="")

        with (
            patch("shutil.which", return_value="/usr/bin/fake"),
            patch("src.validation.helpers.run_command", side_effect=mock_run_command),
            patch("tempfile.mkdtemp", return_value=str(tmp_path / "fixture")),
            patch("shutil.rmtree", side_effect=mock_rmtree),
        ):
            (tmp_path / "fixture").mkdir()
            (tmp_path / "fixture" / "tests").mkdir()

            runner.run(env={}, cwd=tmp_path)
            assert cleanup_called["value"] is True

    def test_run_keeps_fixture_when_configured(self, tmp_path: Path) -> None:
        config = E2EConfig(keep_fixture=True)
        runner = E2ERunner(config)

        def mock_run_command(
            cmd: list[str], cwd: Path, **kwargs: object
        ) -> CommandResult:
            return CommandResult(command=cmd, returncode=0, stdout="", stderr="")

        with (
            patch("shutil.which", return_value="/usr/bin/fake"),
            patch("src.validation.helpers.run_command", side_effect=mock_run_command),
            patch("tempfile.mkdtemp", return_value=str(tmp_path / "fixture")),
            patch("shutil.rmtree") as mock_rmtree,
        ):
            (tmp_path / "fixture").mkdir()
            (tmp_path / "fixture" / "tests").mkdir()

            result = runner.run(env={}, cwd=tmp_path)
            # rmtree should not be called when keeping fixture
            mock_rmtree.assert_not_called()
            # fixture_path should be set
            assert result.fixture_path is not None

    def test_run_handles_mala_timeout(self, tmp_path: Path) -> None:
        config = E2EConfig(timeout_seconds=1.0)
        runner = E2ERunner(config)

        def mock_run_command(
            cmd: list[str], cwd: Path, **kwargs: object
        ) -> CommandResult:
            """Mock run_command for fixture setup commands (git, bd)."""
            return CommandResult(command=cmd, returncode=0, stdout="", stderr="")

        def mock_popen(*args: object, **kwargs: object) -> MagicMock:
            """Mock subprocess.Popen for CommandRunner (mala command)."""
            cmd: list[str] = list(args[0]) if args else []  # type: ignore[arg-type]
            mock_proc = MagicMock()
            mock_proc.pid = 12345
            call_count = {"value": 0}

            def communicate_timeout(timeout: float | None = None) -> tuple[str, str]:
                call_count["value"] += 1
                # Only timeout on first call (main command); subsequent calls are
                # during cleanup/termination and should return output
                if call_count["value"] == 1:
                    exc = subprocess.TimeoutExpired(cmd=cmd, timeout=1.0)
                    exc.stdout = b"partial output"
                    exc.stderr = b""
                    raise exc
                return ("partial output", "")

            mock_proc.communicate = communicate_timeout
            return mock_proc

        with (
            patch("shutil.which", return_value="/usr/bin/fake"),
            patch("src.validation.helpers.run_command", side_effect=mock_run_command),
            patch("src.tools.command_runner.subprocess.Popen", side_effect=mock_popen),
            patch("tempfile.mkdtemp", return_value=str(tmp_path / "fixture")),
            patch("shutil.rmtree"),
            patch("src.tools.command_runner.os.killpg"),  # Mock process group kill
        ):
            (tmp_path / "fixture").mkdir()
            (tmp_path / "fixture" / "tests").mkdir()

            result = runner.run(env={}, cwd=tmp_path)
            assert result.passed is False
            assert "timed out" in (result.failure_reason or "")
            assert result.returncode == 124


class TestE2ERunnerIntegration:
    """Integration coverage for real E2E runs."""

    @pytest.mark.e2e
    def test_run_real_fixture(self, tmp_path: Path) -> None:
        if shutil.which("mala") is None or shutil.which("bd") is None:
            pytest.skip("E2E requires mala and bd CLIs")

        config = E2EConfig(keep_fixture=True, timeout_seconds=600.0)
        runner = E2ERunner(config)

        result = runner.run(cwd=tmp_path)

        try:
            assert result.passed is True
            assert result.status == E2EStatus.PASSED
            assert result.fixture_path is not None
            assert (result.fixture_path / "coverage.xml").exists()
        finally:
            if result.fixture_path and result.fixture_path.exists():
                shutil.rmtree(result.fixture_path, ignore_errors=True)


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
        assert result_lines[0] == "line 45"
        assert result_lines[-1] == "line 49"


class TestDecodeTimeoutOutput:
    """Test the decode_timeout_output helper."""

    def test_none_returns_empty(self) -> None:
        assert decode_timeout_output(None) == ""

    def test_string_returned_as_is(self) -> None:
        assert decode_timeout_output("hello") == "hello"

    def test_bytes_decoded(self) -> None:
        assert decode_timeout_output(b"hello") == "hello"


class TestWriteFixtureFiles:
    """Test the write_fixture_repo helper."""

    def test_creates_expected_files(self, tmp_path: Path) -> None:
        write_fixture_repo(tmp_path)

        assert (tmp_path / "src").is_dir()
        assert (tmp_path / "src" / "app.py").exists()
        assert (tmp_path / "tests").is_dir()
        assert (tmp_path / "tests" / "test_app.py").exists()
        assert (tmp_path / "pyproject.toml").exists()

    def test_app_py_has_bug(self, tmp_path: Path) -> None:
        write_fixture_repo(tmp_path)

        content = (tmp_path / "src" / "app.py").read_text()
        assert "return a - b" in content  # The bug


class TestInitFixtureRepo:
    """Test the init_fixture_repo helper."""

    def test_returns_none_on_success(self, tmp_path: Path) -> None:
        mock_result = CommandResult(command=[], returncode=0, stdout="", stderr="")
        with patch("src.validation.helpers.run_command", return_value=mock_result):
            result = init_fixture_repo(tmp_path)
            assert result is None

    def test_returns_error_on_failure(self, tmp_path: Path) -> None:
        mock_result = CommandResult(
            command=[], returncode=1, stdout="", stderr="git init failed"
        )
        with patch("src.validation.helpers.run_command", return_value=mock_result):
            result = init_fixture_repo(tmp_path)
            assert result is not None
            assert "fixture setup failed" in result


class TestGetReadyIssueId:
    """Test the get_ready_issue_id helper."""

    def test_returns_issue_id(self, tmp_path: Path) -> None:
        mock_result = CommandResult(
            command=[],
            returncode=0,
            stdout='[{"id": "test-123", "title": "Fix bug"}]',
            stderr="",
        )
        with patch("src.validation.helpers.run_command", return_value=mock_result):
            result = get_ready_issue_id(tmp_path)
            assert result == "test-123"

    def test_returns_none_on_failure(self, tmp_path: Path) -> None:
        mock_result = CommandResult(command=[], returncode=1, stdout="", stderr="")
        with patch("src.validation.helpers.run_command", return_value=mock_result):
            result = get_ready_issue_id(tmp_path)
            assert result is None

    def test_returns_none_on_invalid_json(self, tmp_path: Path) -> None:
        mock_result = CommandResult(
            command=[], returncode=0, stdout="not json", stderr=""
        )
        with patch("src.validation.helpers.run_command", return_value=mock_result):
            result = get_ready_issue_id(tmp_path)
            assert result is None

    def test_returns_none_on_empty_list(self, tmp_path: Path) -> None:
        mock_result = CommandResult(command=[], returncode=0, stdout="[]", stderr="")
        with patch("src.validation.helpers.run_command", return_value=mock_result):
            result = get_ready_issue_id(tmp_path)
            assert result is None


class TestAnnotateIssue:
    """Test the annotate_issue helper."""

    def test_calls_bd_update(self, tmp_path: Path) -> None:
        mock_result = CommandResult(command=[], returncode=0, stdout="", stderr="")
        mock_run = MagicMock(return_value=mock_result)
        with patch("src.validation.helpers.run_command", mock_run):
            annotate_issue(tmp_path, "test-123")

            mock_run.assert_called_once()
            args = mock_run.call_args
            cmd = args[0][0]
            assert "bd" in cmd
            assert "update" in cmd
            assert "test-123" in cmd


class TestCheckE2EPrereqsLegacy:
    """Test the legacy check_e2e_prereqs function."""

    def test_returns_none_when_ok(self) -> None:
        with patch("shutil.which", return_value="/usr/bin/fake"):
            # MORPH_API_KEY is not required for prereqs to pass
            result = check_e2e_prereqs({})
            assert result is None

    def test_returns_error_when_missing_cli(self) -> None:
        with patch("shutil.which", return_value=None):
            result = check_e2e_prereqs({})
            assert result is not None
            assert "mala CLI" in result


@pytest.mark.integration
class TestE2EIntegration:
    """Integration tests for E2E runner (requires real tools)."""

    def test_real_prereq_check(self) -> None:
        """Test prereq check with real environment.

        This test just checks prerequisites - it doesn't run the full E2E.
        """
        runner = E2ERunner()
        result = runner.check_prereqs()
        # Just verify it returns a valid result
        assert isinstance(result, E2EPrereqResult)
        assert isinstance(result.ok, bool)
