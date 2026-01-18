"""Unit tests for src/validation/e2e.py - E2E fixture runner."""

import json
import shutil
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.infra.tools.command_runner import CommandResult
from tests.e2e.claude_auth import is_claude_cli_available, has_valid_oauth_credentials
from src.domain.validation.e2e import (
    E2EConfig,
    E2EPrereqResult,
    E2EResult,
    E2ERunner,
    E2EStatus,
    _commands_use_uv,
    _config_prefers_uv,
    _select_python_invoker,
    check_e2e_prereqs,
)
from src.domain.validation.helpers import (
    _generate_fixture_programmatically,
    annotate_issue,
    decode_timeout_output,
    get_ready_issue_id,
    init_fixture_repo,
    tail,
    write_fixture_repo,
)


def make_mock_env_config(tmp_path: Path | None = None) -> Mock:
    """Create a mock EnvConfigPort.

    Args:
        tmp_path: If provided, creates a real cerberus bin directory with review-gate.
                  Otherwise returns None (skips cerberus check).
    """
    mock = Mock()
    mock.scripts_dir = Path("/mock/scripts")
    mock.cache_dir = Path("/mock/cache")
    mock.lock_dir = Path("/tmp/mock-locks")

    if tmp_path is not None:
        # Create a real cerberus bin directory with review-gate
        cerberus_bin = tmp_path / "cerberus-bin"
        cerberus_bin.mkdir(exist_ok=True)
        (cerberus_bin / "review-gate").touch()
        mock.find_cerberus_bin_path.return_value = cerberus_bin
    else:
        # Return None so check_prereqs fails fast (not the file check)
        mock.find_cerberus_bin_path.return_value = None

    return mock


def make_mock_command_runner() -> Mock:
    """Create a mock CommandRunnerPort that succeeds by default."""
    mock = Mock()
    mock.run.return_value = CommandResult(
        command=["mock"],
        returncode=0,
        stdout='[{"id": "test-123"}]',
        stderr="",
    )
    return mock


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
            missing=["mala CLI not found", "br CLI not found"],
        )
        assert result.ok is False
        reason = result.failure_reason()
        assert reason is not None
        assert "mala CLI not found" in reason
        assert "br CLI not found" in reason

    def test_failed_result_empty_missing(self) -> None:
        result = E2EPrereqResult(ok=False, missing=[])
        assert result.failure_reason() == "E2E prerequisites not met"

    def test_can_skip_flag(self) -> None:
        result = E2EPrereqResult(
            ok=False, missing=["optional prereq missing"], can_skip=True
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
            failure_reason="optional prereq missing",
        )
        assert result.passed is True
        assert result.status == E2EStatus.SKIPPED
        assert "skipped" in result.short_summary()
        assert "optional prereq" in result.short_summary()

    def test_failed_no_reason(self) -> None:
        result = E2EResult(passed=False, status=E2EStatus.FAILED)
        assert result.short_summary() == "E2E failed: unknown error"


class TestE2ERunnerPrereqs:
    """Test E2ERunner.check_prereqs method."""

    def test_all_prereqs_met(self, tmp_path: Path) -> None:
        runner = E2ERunner(make_mock_env_config(tmp_path), make_mock_command_runner())
        with patch("shutil.which", return_value="/usr/bin/fake"):
            result = runner.check_prereqs({})
            assert result.ok is True
            assert result.missing == []

    def test_missing_mala_cli(self, tmp_path: Path) -> None:
        runner = E2ERunner(make_mock_env_config(tmp_path), make_mock_command_runner())
        with patch("shutil.which", return_value=None):
            result = runner.check_prereqs({})
            assert result.ok is False
            assert any("mala CLI" in m for m in result.missing)

    def test_missing_bd_cli(self, tmp_path: Path) -> None:
        runner = E2ERunner(make_mock_env_config(tmp_path), make_mock_command_runner())

        def mock_which(cmd: str) -> str | None:
            if cmd == "mala":
                return "/usr/bin/mala"
            return None

        with patch("shutil.which", side_effect=mock_which):
            result = runner.check_prereqs({})
            assert result.ok is False
            assert any("br CLI" in m for m in result.missing)

    def test_uses_os_environ_when_none(self, tmp_path: Path) -> None:
        runner = E2ERunner(make_mock_env_config(tmp_path), make_mock_command_runner())
        with (
            patch("shutil.which", return_value="/usr/bin/fake"),
            patch.dict("os.environ", {}, clear=True),
        ):
            result = runner.check_prereqs(None)
            assert result.ok is True


class TestE2ERunnerRun:
    """Test E2ERunner.run method."""

    def test_run_proceeds_without_api_keys(self, tmp_path: Path) -> None:
        """E2E should work without optional API keys."""
        runner = E2ERunner(make_mock_env_config(tmp_path), make_mock_command_runner())

        def mock_runner_run(
            cmd: list[str], cwd: Path | None = None, **kwargs: object
        ) -> CommandResult:
            """Mock CommandRunner.run for fixture setup commands (git, bd)."""
            return CommandResult(
                command=cmd,
                returncode=0,
                stdout='[{"id": "test-123"}]' if "ready" in cmd else "ok",
                stderr="",
            )

        with (
            patch("shutil.which", return_value="/usr/bin/fake"),
            patch(
                "src.infra.tools.command_runner.CommandRunner.run",
                side_effect=mock_runner_run,
            ),
            patch("tempfile.mkdtemp", return_value=str(tmp_path / "fixture")),
            patch("shutil.rmtree"),
        ):
            (tmp_path / "fixture").mkdir()
            (tmp_path / "fixture" / "tests").mkdir()

            result = runner.run(env={}, cwd=tmp_path)
            assert result.passed is True
            assert result.status == E2EStatus.PASSED

    def test_run_fails_on_missing_prereqs(self) -> None:
        runner = E2ERunner(make_mock_env_config(), make_mock_command_runner())
        with patch("shutil.which", return_value=None):
            result = runner.run(env={})
            assert result.passed is False
            assert result.status == E2EStatus.FAILED
            assert "mala CLI" in (result.failure_reason or "")

    def test_run_fails_on_fixture_setup_error(self, tmp_path: Path) -> None:
        # Create a mock command_runner that fails for git init
        mock_cmd_runner = Mock()
        mock_cmd_runner.run.return_value = CommandResult(
            command=["git", "init"],
            returncode=1,
            stdout="",
            stderr="git init failed",
        )
        runner = E2ERunner(make_mock_env_config(tmp_path), mock_cmd_runner)

        with (
            patch("shutil.which", return_value="/usr/bin/fake"),
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
        runner = E2ERunner(
            make_mock_env_config(tmp_path), make_mock_command_runner(), config
        )

        def mock_runner_run(
            cmd: list[str], cwd: Path | None = None, **kwargs: object
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
            patch(
                "src.infra.tools.command_runner.CommandRunner.run",
                side_effect=mock_runner_run,
            ),
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
        runner = E2ERunner(
            make_mock_env_config(tmp_path), make_mock_command_runner(), config
        )

        cleanup_called = {"value": False}

        def mock_rmtree(*args: object, **kwargs: object) -> None:
            cleanup_called["value"] = True

        def mock_runner_run(
            cmd: list[str], cwd: Path | None = None, **kwargs: object
        ) -> CommandResult:
            return CommandResult(command=cmd, returncode=0, stdout="", stderr="")

        with (
            patch("shutil.which", return_value="/usr/bin/fake"),
            patch(
                "src.infra.tools.command_runner.CommandRunner.run",
                side_effect=mock_runner_run,
            ),
            patch("tempfile.mkdtemp", return_value=str(tmp_path / "fixture")),
            patch("shutil.rmtree", side_effect=mock_rmtree),
        ):
            (tmp_path / "fixture").mkdir()
            (tmp_path / "fixture" / "tests").mkdir()

            runner.run(env={}, cwd=tmp_path)
            assert cleanup_called["value"] is True

    def test_run_keeps_fixture_when_configured(self, tmp_path: Path) -> None:
        config = E2EConfig(keep_fixture=True)
        runner = E2ERunner(
            make_mock_env_config(tmp_path), make_mock_command_runner(), config
        )

        def mock_runner_run(
            cmd: list[str], cwd: Path | None = None, **kwargs: object
        ) -> CommandResult:
            return CommandResult(command=cmd, returncode=0, stdout="", stderr="")

        with (
            patch("shutil.which", return_value="/usr/bin/fake"),
            patch(
                "src.infra.tools.command_runner.CommandRunner.run",
                side_effect=mock_runner_run,
            ),
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

        def mock_runner_run(
            cmd: list[str], cwd: Path | None = None, **kwargs: object
        ) -> CommandResult:
            """Mock CommandRunner.run - succeed for fixture setup, timeout for mala."""
            # Fixture setup commands (git, bd) succeed
            if any(x in cmd for x in ["git", "br"]):
                return CommandResult(command=cmd, returncode=0, stdout="", stderr="")
            # mala command times out
            return CommandResult(
                command=cmd,
                returncode=124,
                stdout="partial output",
                stderr="",
                timed_out=True,
            )

        mock_cmd_runner = Mock()
        mock_cmd_runner.run.side_effect = mock_runner_run
        runner = E2ERunner(make_mock_env_config(tmp_path), mock_cmd_runner, config)

        with (
            patch("shutil.which", return_value="/usr/bin/fake"),
            patch("tempfile.mkdtemp", return_value=str(tmp_path / "fixture")),
            patch("shutil.rmtree"),
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
    def test_run_real_fixture(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.infra.tools.command_runner import CommandRunner
        from src.infra.tools.env import EnvConfig

        if shutil.which("mala") is None or shutil.which("br") is None:
            pytest.skip("E2E requires mala and bd CLIs")
        if not is_claude_cli_available():
            pytest.skip("Claude Code CLI not installed")
        if not has_valid_oauth_credentials():
            pytest.skip(
                "Claude Code CLI not logged in or token expired - run `claude` and login"
            )

        # Use real Claude config dir to find Cerberus (conftest overrides this for isolation)
        monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)

        env_config = EnvConfig()
        config = E2EConfig(keep_fixture=True, timeout_seconds=420.0)
        runner = E2ERunner(env_config, CommandRunner(cwd=tmp_path), config)

        result = runner.run(cwd=tmp_path)

        try:
            assert result.passed is True
            assert result.status == E2EStatus.PASSED
            assert result.fixture_path is not None
            assert result.fixture_path is not None
            from src.infra.tools.env import get_repo_runs_dir

            run_dir = get_repo_runs_dir(result.fixture_path)
            assert run_dir.exists()
            run_metadata_path = max(
                run_dir.glob("*.json"), key=lambda path: path.stat().st_mtime
            )
            run_metadata = json.loads(run_metadata_path.read_text())
            run_validation = run_metadata.get("run_validation")
            # Strict assertion: run_validation must exist when run_end validation is configured.
            # A conditional check would mask regressions where validation is silently skipped.
            assert run_validation is not None, (
                "run_validation missing from run metadata"
            )
            coverage_percent = run_validation.get("coverage_percent")
            assert coverage_percent is not None
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

    def test_programmatic_fallback_creates_same_structure(self, tmp_path: Path) -> None:
        """Test that programmatic generation creates the same file structure."""
        _generate_fixture_programmatically(tmp_path)

        # Verify all expected files exist
        assert (tmp_path / "src").is_dir()
        assert (tmp_path / "src" / "app.py").exists()
        assert (tmp_path / "tests").is_dir()
        assert (tmp_path / "tests" / "test_app.py").exists()
        assert (tmp_path / "pyproject.toml").exists()
        assert (tmp_path / "mala.yaml").exists()

        # Verify the app has the expected bug
        content = (tmp_path / "src" / "app.py").read_text()
        assert "return a - b" in content

        # Verify test file has expected test
        test_content = (tmp_path / "tests" / "test_app.py").read_text()
        assert "assert add(2, 2) == 4" in test_content
        mala_yaml = (tmp_path / "mala.yaml").read_text()
        assert "validation_triggers:" in mala_yaml

    def test_fallback_used_when_template_missing(self, tmp_path: Path) -> None:
        """Test that write_fixture_repo falls back to programmatic generation.

        When the fixture template directory doesn't exist (e.g., installed packages
        where tests/ is not included), it should use programmatic generation.
        """
        # Mock Path.exists to return False for the fixture_root check
        original_exists = Path.exists

        def mock_exists(path: Path) -> bool:
            if "e2e-fixture" in str(path):
                return False
            return original_exists(path)

        with patch.object(Path, "exists", mock_exists):
            write_fixture_repo(tmp_path)

        # Verify fixture was created via programmatic fallback
        assert (tmp_path / "src" / "app.py").exists()
        assert (tmp_path / "mala.yaml").exists()
        assert (tmp_path / "tests" / "test_app.py").exists()

        # Verify content is correct
        content = (tmp_path / "src" / "app.py").read_text()
        assert "return a - b" in content
        mala_yaml = (tmp_path / "mala.yaml").read_text()
        assert "validation_triggers:" in mala_yaml

    def test_uses_repo_root_fixture_when_available(self, tmp_path: Path) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        write_fixture_repo(tmp_path, repo_root=repo_root)
        mala_yaml = (tmp_path / "mala.yaml").read_text()
        assert "validation_triggers:" in mala_yaml


class TestInitFixtureRepo:
    """Test the init_fixture_repo helper."""

    def test_returns_none_on_success(self, tmp_path: Path) -> None:
        mock_result = CommandResult(command=[], returncode=0, stdout="", stderr="")
        mock_runner = make_mock_command_runner()
        mock_runner.run.return_value = mock_result
        result = init_fixture_repo(tmp_path, mock_runner)
        assert result is None

    def test_returns_error_on_failure(self, tmp_path: Path) -> None:
        mock_result = CommandResult(
            command=[], returncode=1, stdout="", stderr="git init failed"
        )
        mock_runner = make_mock_command_runner()
        mock_runner.run.return_value = mock_result
        result = init_fixture_repo(tmp_path, mock_runner)
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
        mock_runner = make_mock_command_runner()
        mock_runner.run.return_value = mock_result
        result = get_ready_issue_id(tmp_path, mock_runner)
        assert result == "test-123"

    def test_returns_none_on_failure(self, tmp_path: Path) -> None:
        mock_result = CommandResult(command=[], returncode=1, stdout="", stderr="")
        mock_runner = make_mock_command_runner()
        mock_runner.run.return_value = mock_result
        result = get_ready_issue_id(tmp_path, mock_runner)
        assert result is None

    def test_returns_none_on_invalid_json(self, tmp_path: Path) -> None:
        mock_result = CommandResult(
            command=[], returncode=0, stdout="not json", stderr=""
        )
        mock_runner = make_mock_command_runner()
        mock_runner.run.return_value = mock_result
        result = get_ready_issue_id(tmp_path, mock_runner)
        assert result is None

    def test_returns_none_on_empty_list(self, tmp_path: Path) -> None:
        mock_result = CommandResult(command=[], returncode=0, stdout="[]", stderr="")
        mock_runner = make_mock_command_runner()
        mock_runner.run.return_value = mock_result
        result = get_ready_issue_id(tmp_path, mock_runner)
        assert result is None


class TestAnnotateIssue:
    """Test the annotate_issue helper."""

    def test_calls_bd_update(self, tmp_path: Path) -> None:
        mock_result = CommandResult(command=[], returncode=0, stdout="", stderr="")
        mock_runner = make_mock_command_runner()
        mock_runner.run.return_value = mock_result
        annotate_issue(tmp_path, "test-123", mock_runner)

        mock_runner.run.assert_called_once()
        args = mock_runner.run.call_args
        cmd = args[0][0]
        assert "br" in cmd
        assert "update" in cmd
        assert "test-123" in cmd


class TestE2EInvokerSelection:
    def test_commands_use_uv_detects_uv_commands(self) -> None:
        assert _commands_use_uv(
            [
                "uv run pytest",
                "RUFF_CACHE_DIR=/tmp/ruff uvx ruff check .",
            ]
        )

    def test_commands_use_uv_ignores_non_uv(self) -> None:
        assert not _commands_use_uv(["python -m pytest", "go test ./..."])

    def test_config_prefers_uv_for_python_uv_preset(self, tmp_path: Path) -> None:
        (tmp_path / "mala.yaml").write_text("preset: python-uv\n")
        assert _config_prefers_uv(tmp_path) is True

    def test_config_prefers_uv_false_for_go_preset(self, tmp_path: Path) -> None:
        (tmp_path / "mala.yaml").write_text("preset: go\n")
        assert _config_prefers_uv(tmp_path) is False

    def test_config_prefers_uv_false_for_node_preset(self, tmp_path: Path) -> None:
        (tmp_path / "mala.yaml").write_text("preset: node-npm\n")
        assert _config_prefers_uv(tmp_path) is False

    def test_config_prefers_uv_when_any_command_uses_uv(self, tmp_path: Path) -> None:
        (tmp_path / "mala.yaml").write_text('commands:\n  test: "uv run pytest"\n')
        assert _config_prefers_uv(tmp_path) is True

    def test_config_prefers_uv_when_uv_lock_present(self, tmp_path: Path) -> None:
        (tmp_path / "mala.yaml").write_text('commands:\n  test: "pytest"\n')
        (tmp_path / "uv.lock").write_text("# uv lock\n")
        assert _config_prefers_uv(tmp_path) is True

    def test_config_prefers_uv_false_without_uv_signals(self, tmp_path: Path) -> None:
        (tmp_path / "mala.yaml").write_text('commands:\n  test: "pytest"\n')
        assert _config_prefers_uv(tmp_path) is False

    def test_select_python_invoker_prefers_uv_when_available(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "src.domain.validation.e2e._config_prefers_uv", lambda _: True
        )
        monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/uv")

        invoker = _select_python_invoker(tmp_path)
        assert invoker[:2] == ["uv", "run"]
        assert "--directory" in invoker
        assert str(tmp_path) in invoker
        assert invoker[-1] == "python"

    def test_select_python_invoker_falls_back_to_sys_executable(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "src.domain.validation.e2e._config_prefers_uv", lambda _: True
        )
        monkeypatch.setattr(shutil, "which", lambda name: None)

        invoker = _select_python_invoker(tmp_path)
        assert invoker == [sys.executable]

    def test_select_python_invoker_ignores_uv_when_config_not_uv(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "src.domain.validation.e2e._config_prefers_uv", lambda _: False
        )
        monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/uv")

        invoker = _select_python_invoker(tmp_path)
        assert invoker == [sys.executable]


class TestCheckE2EPrereqsLegacy:
    """Test the legacy check_e2e_prereqs function."""

    def test_returns_none_when_ok(self, tmp_path: Path) -> None:
        mock_env_config = make_mock_env_config(tmp_path)
        mock_cmd_runner = make_mock_command_runner()
        with patch("shutil.which", return_value="/usr/bin/fake"):
            result = check_e2e_prereqs(mock_env_config, mock_cmd_runner, {})
            assert result is None

    def test_returns_error_when_missing_cli(self, tmp_path: Path) -> None:
        mock_env_config = make_mock_env_config(tmp_path)
        mock_cmd_runner = make_mock_command_runner()
        with patch("shutil.which", return_value=None):
            result = check_e2e_prereqs(mock_env_config, mock_cmd_runner, {})
            assert result is not None
            assert "mala CLI" in result


@pytest.mark.integration
class TestE2EIntegration:
    """Integration tests for E2E runner (requires real tools)."""

    def test_real_prereq_check(self) -> None:
        """Test prereq check with real environment.

        This test just checks prerequisites - it doesn't run the full E2E.
        """
        runner = E2ERunner(make_mock_env_config(), make_mock_command_runner())
        result = runner.check_prereqs()
        # Just verify it returns a valid result
        assert isinstance(result, E2EPrereqResult)
        assert isinstance(result.ok, bool)
