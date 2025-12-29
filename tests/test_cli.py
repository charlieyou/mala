import importlib
import sys
import types
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import patch

import pytest
import typer

import src.orchestrator
import src.beads_client
import src.tools.locking


def _reload_cli(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    monkeypatch.setenv("BRAINTRUST_API_KEY", "")
    # Reset bootstrap state before reloading
    if "src.cli" in sys.modules:
        cli_mod = sys.modules["src.cli"]
        # Reset internal state so bootstrap can run again
        cli_mod._bootstrapped = False  # type: ignore[attr-defined]
        cli_mod._braintrust_enabled = False  # type: ignore[attr-defined]
        return importlib.reload(cli_mod)
    return importlib.import_module("src.cli")


class TestImportSafety:
    """Test that importing src.cli has no side effects."""

    def test_import_does_not_load_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Importing src.cli should not call load_user_env()."""
        # Clear modules to force fresh import
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("src.cli") or mod_name.startswith("src.tools.env"):
                del sys.modules[mod_name]

        # Track if load_user_env gets called
        load_called = {"called": False}

        def mock_load_user_env() -> None:
            load_called["called"] = True

        # Patch before import
        with patch("src.tools.env.load_user_env", mock_load_user_env):
            # Force reimport of cli module
            if "src.cli" in sys.modules:
                del sys.modules["src.cli"]

            import src.cli

            # Import should NOT have triggered load_user_env
            assert not load_called["called"], (
                "load_user_env() was called at import time - should only be called via bootstrap()"
            )

            # Verify the module was imported (avoids F401 unused import warning)
            assert hasattr(src.cli, "bootstrap")

    def test_import_does_not_setup_braintrust(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Importing src.cli should not set up Braintrust."""
        # Clear modules to force fresh import
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("src.cli"):
                del sys.modules[mod_name]

        # Set a Braintrust API key that would trigger setup if called
        monkeypatch.setenv("BRAINTRUST_API_KEY", "test-key")

        setup_called = {"called": False}

        def mock_setup(*args: object, **kwargs: object) -> None:
            setup_called["called"] = True

        # Patch the Braintrust setup function
        with patch.dict(sys.modules, {"braintrust": None, "braintrust.wrappers": None}):
            if "src.cli" in sys.modules:
                del sys.modules["src.cli"]

            import src.cli

            # Import should NOT have triggered Braintrust setup
            assert not src.cli._braintrust_enabled, (
                "Braintrust was enabled at import time - should only happen via bootstrap()"
            )

    def test_bootstrap_loads_env_and_braintrust(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """bootstrap() should load env and set up Braintrust when API key is present."""
        # Clear modules
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("src.cli") or mod_name.startswith("src.tools.env"):
                del sys.modules[mod_name]

        monkeypatch.setenv("BRAINTRUST_API_KEY", "test-key")

        import src.cli

        # Reset state
        src.cli._bootstrapped = False
        src.cli._braintrust_enabled = False

        # Track calls
        load_called = {"called": False}
        original_load = src.cli.load_user_env

        def tracking_load() -> None:
            load_called["called"] = True
            original_load()

        monkeypatch.setattr(src.cli, "load_user_env", tracking_load)

        # Call bootstrap
        src.cli.bootstrap()

        assert load_called["called"], "bootstrap() should call load_user_env()"
        assert src.cli._bootstrapped, "bootstrap() should set _bootstrapped = True"

    def test_bootstrap_is_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Calling bootstrap() multiple times should only execute once."""
        cli = _reload_cli(monkeypatch)

        call_count = {"count": 0}
        original_load = cli.load_user_env

        def counting_load() -> None:
            call_count["count"] += 1
            original_load()

        monkeypatch.setattr(cli, "load_user_env", counting_load)

        # Reset state
        cli._bootstrapped = False  # type: ignore[attr-defined]
        cli._braintrust_enabled = False  # type: ignore[attr-defined]

        # Call bootstrap multiple times
        cli.bootstrap()
        cli.bootstrap()
        cli.bootstrap()

        assert call_count["count"] == 1, (
            "bootstrap() should only call load_user_env() once"
        )


class DummyOrchestrator:
    last_kwargs: dict[str, Any] | None = None

    def __init__(self, **kwargs: object) -> None:
        type(self).last_kwargs = kwargs

    async def run(self) -> tuple[int, int]:
        return (1, 1)


def test_run_without_morph_api_key_passes_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that CLI passes config to orchestrator when MORPH_API_KEY is missing."""
    cli = _reload_cli(monkeypatch)
    monkeypatch.delenv("MORPH_API_KEY", raising=False)

    config_dir = tmp_path / "config"
    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    monkeypatch.setattr(src.orchestrator, "MalaOrchestrator", DummyOrchestrator)
    monkeypatch.setattr(cli, "set_verbose", lambda _: None)

    with pytest.raises(typer.Exit) as excinfo:
        cli.run(repo_path=tmp_path)

    assert excinfo.value.exit_code == 0
    assert DummyOrchestrator.last_kwargs is not None
    # CLI now passes config instead of morph_enabled
    assert "config" in DummyOrchestrator.last_kwargs
    config = DummyOrchestrator.last_kwargs["config"]
    assert config.morph_api_key is None
    assert config.morph_enabled is False


def test_run_with_morph_api_key_passes_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that CLI passes config with morph_api_key when MORPH_API_KEY is set."""
    cli = _reload_cli(monkeypatch)
    monkeypatch.setenv("MORPH_API_KEY", "test-key")

    config_dir = tmp_path / "config"
    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    monkeypatch.setattr(src.orchestrator, "MalaOrchestrator", DummyOrchestrator)
    monkeypatch.setattr(cli, "set_verbose", lambda _: None)

    with pytest.raises(typer.Exit) as excinfo:
        cli.run(repo_path=tmp_path)

    assert excinfo.value.exit_code == 0
    assert DummyOrchestrator.last_kwargs is not None
    # CLI now passes config instead of morph_enabled
    assert "config" in DummyOrchestrator.last_kwargs
    config = DummyOrchestrator.last_kwargs["config"]
    assert config.morph_api_key == "test-key"
    assert config.morph_enabled is True


def test_run_invalid_only_exits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cli = _reload_cli(monkeypatch)
    logs: list[tuple[object, ...]] = []

    def _log(*args: object, **_kwargs: object) -> None:
        logs.append(args)

    monkeypatch.setattr(cli, "log", _log)
    monkeypatch.setattr(cli, "set_verbose", lambda _: None)

    with pytest.raises(typer.Exit) as excinfo:
        cli.run(repo_path=tmp_path, only=" , ")

    assert excinfo.value.exit_code == 1
    assert logs


def test_run_success_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cli = _reload_cli(monkeypatch)
    monkeypatch.setenv("MORPH_API_KEY", "test-key")

    verbose_calls = {"enabled": None}

    def _set_verbose(value: bool) -> None:
        verbose_calls["enabled"] = value

    config_dir = tmp_path / "config"
    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    monkeypatch.setattr(src.orchestrator, "MalaOrchestrator", DummyOrchestrator)
    monkeypatch.setattr(cli, "set_verbose", _set_verbose)

    with pytest.raises(typer.Exit) as excinfo:
        cli.run(
            repo_path=tmp_path,
            max_agents=2,
            timeout=7,
            max_issues=3,
            epic="epic-1",
            only="id-1,id-2",
            max_gate_retries=4,
            max_review_retries=5,
            verbose=True,
        )

    assert excinfo.value.exit_code == 0
    assert verbose_calls["enabled"] is True
    assert DummyOrchestrator.last_kwargs is not None
    assert DummyOrchestrator.last_kwargs["only_ids"] == {"id-1", "id-2"}
    # CLI now passes config instead of morph_enabled
    config = DummyOrchestrator.last_kwargs["config"]
    assert config.morph_enabled is True
    assert config_dir.exists()


def test_run_verbose_mode_sets_verbose_true(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that --verbose flag sets verbose to True."""
    cli = _reload_cli(monkeypatch)
    monkeypatch.setenv("MORPH_API_KEY", "test-key")

    verbose_calls = {"enabled": None}

    def _set_verbose(value: bool) -> None:
        verbose_calls["enabled"] = value

    config_dir = tmp_path / "config"
    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    monkeypatch.setattr(src.orchestrator, "MalaOrchestrator", DummyOrchestrator)
    monkeypatch.setattr(cli, "set_verbose", _set_verbose)

    with pytest.raises(typer.Exit) as excinfo:
        cli.run(repo_path=tmp_path, verbose=True)

    assert excinfo.value.exit_code == 0
    assert verbose_calls["enabled"] is True


def test_run_default_quiet_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that default is quiet (verbose=False)."""
    cli = _reload_cli(monkeypatch)
    monkeypatch.setenv("MORPH_API_KEY", "test-key")

    verbose_calls = {"enabled": None}

    def _set_verbose(value: bool) -> None:
        verbose_calls["enabled"] = value

    config_dir = tmp_path / "config"
    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    monkeypatch.setattr(src.orchestrator, "MalaOrchestrator", DummyOrchestrator)
    monkeypatch.setattr(cli, "set_verbose", _set_verbose)

    with pytest.raises(typer.Exit):
        cli.run(repo_path=tmp_path)

    assert verbose_calls["enabled"] is False


def test_run_repo_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cli = _reload_cli(monkeypatch)
    monkeypatch.setenv("MORPH_API_KEY", "test-key")

    logs: list[tuple[object, ...]] = []

    def _log(*args: object, **_kwargs: object) -> None:
        logs.append(args)

    config_dir = tmp_path / "config"
    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    monkeypatch.setattr(cli, "log", _log)
    monkeypatch.setattr(cli, "set_verbose", lambda _: None)

    missing_repo = tmp_path / "missing"

    with pytest.raises(typer.Exit) as excinfo:
        cli.run(repo_path=missing_repo)

    assert excinfo.value.exit_code == 1
    assert logs


def test_clean_removes_locks_and_logs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cli = _reload_cli(monkeypatch)
    lock_dir = tmp_path / "locks"
    run_dir = tmp_path / "runs"
    lock_dir.mkdir()
    run_dir.mkdir()

    (lock_dir / "one.lock").write_text("agent")
    (lock_dir / "two.lock").write_text("agent")
    (run_dir / "one.json").write_text("{}")
    (run_dir / "two.json").write_text("{}")

    logs: list[tuple[object, ...]] = []

    def _log(*args: object, **_kwargs: object) -> None:
        logs.append(args)

    monkeypatch.setattr(src.tools.locking, "get_lock_dir", lambda: lock_dir)
    monkeypatch.setattr(cli, "get_runs_dir", lambda: run_dir)
    monkeypatch.setattr(cli, "log", _log)
    monkeypatch.setattr(cli.typer, "confirm", lambda _msg: True)

    cli.clean()

    assert not list(lock_dir.glob("*.lock"))
    assert not list(run_dir.glob("*.json"))
    assert logs


def test_status_no_running_instance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test status when no mala instance is running in cwd."""
    cli = _reload_cli(monkeypatch)
    import src.logging.run_metadata

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / ".env").write_text("MORPH_API_KEY=test")

    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()

    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    monkeypatch.setattr(src.tools.locking, "get_lock_dir", lambda: lock_dir)
    # Mock to return no running instances
    monkeypatch.setattr(
        src.logging.run_metadata, "get_running_instances_for_dir", lambda _: []
    )

    cli.status()

    output = capsys.readouterr().out
    assert "No mala instance running in this directory" in output
    assert "Use --all to show instances in other directories" in output


def test_status_with_running_instance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test status when a mala instance is running in cwd."""
    cli = _reload_cli(monkeypatch)
    import src.logging.run_metadata
    from datetime import datetime, UTC

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / ".env").write_text("MORPH_API_KEY=test")

    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()
    for idx in range(6):
        (lock_dir / f"lock-{idx}.lock").write_text(f"agent-{idx}")

    run_dir = tmp_path / "runs"
    run_dir.mkdir()
    (run_dir / "one.json").write_text("{}")
    (run_dir / "two.json").write_text("{}")

    # Create a mock RunningInstance
    mock_instance = src.logging.run_metadata.RunningInstance(
        run_id="test-run-id",
        repo_path=tmp_path,
        started_at=datetime.now(UTC),
        pid=12345,
        max_agents=3,
    )

    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    monkeypatch.setattr(src.tools.locking, "get_lock_dir", lambda: lock_dir)
    monkeypatch.setattr(cli, "get_runs_dir", lambda: run_dir)
    monkeypatch.setattr(
        src.logging.run_metadata,
        "get_running_instances_for_dir",
        lambda _: [mock_instance],
    )

    cli.status()

    output = capsys.readouterr().out
    assert "mala running" in output
    assert str(tmp_path) in output
    assert "max-agents: 3" in output
    assert "pid: 12345" in output
    assert "active locks" in output
    assert "run metadata files" in output


def test_status_all_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test status --all flag shows all instances grouped by directory."""
    cli = _reload_cli(monkeypatch)
    import src.logging.run_metadata
    from datetime import datetime, UTC

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / ".env").write_text("MORPH_API_KEY=test")

    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()

    # Create mock instances for two different directories
    mock_instance_1 = src.logging.run_metadata.RunningInstance(
        run_id="run-1",
        repo_path=tmp_path / "repo1",
        started_at=datetime.now(UTC),
        pid=111,
        max_agents=2,
    )
    mock_instance_2 = src.logging.run_metadata.RunningInstance(
        run_id="run-2",
        repo_path=tmp_path / "repo2",
        started_at=datetime.now(UTC),
        pid=222,
        max_agents=None,
    )

    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    monkeypatch.setattr(src.tools.locking, "get_lock_dir", lambda: lock_dir)
    monkeypatch.setattr(
        src.logging.run_metadata,
        "get_running_instances",
        lambda: [mock_instance_1, mock_instance_2],
    )

    cli.status(all_instances=True)

    output = capsys.readouterr().out
    assert "2 running instance(s)" in output
    # Check directory headings (grouped by directory with colon suffix)
    assert "repo1:" in output or str(tmp_path / "repo1") + ":" in output
    assert "repo2:" in output or str(tmp_path / "repo2") + ":" in output
    # Check instance details are shown
    assert "pid: 111" in output
    assert "pid: 222" in output
    assert "max-agents: 2" in output
    assert "max-agents: unlimited" in output


def test_run_disable_validations_valid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that valid --disable-validations values are accepted and passed to orchestrator."""
    cli = _reload_cli(monkeypatch)
    monkeypatch.setenv("MORPH_API_KEY", "test-key")

    config_dir = tmp_path / "config"
    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    monkeypatch.setattr(src.orchestrator, "MalaOrchestrator", DummyOrchestrator)
    monkeypatch.setattr(cli, "set_verbose", lambda _: None)

    with pytest.raises(typer.Exit) as excinfo:
        cli.run(
            repo_path=tmp_path,
            disable_validations="coverage,integration-tests,e2e",
        )

    assert excinfo.value.exit_code == 0
    assert DummyOrchestrator.last_kwargs is not None
    assert DummyOrchestrator.last_kwargs["disable_validations"] == {
        "coverage",
        "integration-tests",
        "e2e",
    }


def test_run_disable_validations_invalid_value(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that unknown --disable-validations values produce a clear CLI error."""
    cli = _reload_cli(monkeypatch)
    monkeypatch.setenv("MORPH_API_KEY", "test-key")

    logs: list[tuple[object, ...]] = []

    def _log(*args: object, **_kwargs: object) -> None:
        logs.append(args)

    config_dir = tmp_path / "config"
    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    monkeypatch.setattr(cli, "log", _log)
    monkeypatch.setattr(cli, "set_verbose", lambda _: None)

    with pytest.raises(typer.Exit) as excinfo:
        cli.run(
            repo_path=tmp_path,
            disable_validations="coverage,invalid-value,bad-option",
        )

    assert excinfo.value.exit_code == 1
    assert logs
    # Check error message mentions the unknown values
    error_msg = str(logs[-1])
    assert "Unknown --disable-validations" in error_msg
    assert "bad-option" in error_msg
    assert "invalid-value" in error_msg


def test_run_disable_validations_empty_value(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that empty --disable-validations value produces error."""
    cli = _reload_cli(monkeypatch)
    monkeypatch.setenv("MORPH_API_KEY", "test-key")

    logs: list[tuple[object, ...]] = []

    def _log(*args: object, **_kwargs: object) -> None:
        logs.append(args)

    config_dir = tmp_path / "config"
    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    monkeypatch.setattr(cli, "log", _log)
    monkeypatch.setattr(cli, "set_verbose", lambda _: None)

    with pytest.raises(typer.Exit) as excinfo:
        cli.run(
            repo_path=tmp_path,
            disable_validations=" , , ",
        )

    assert excinfo.value.exit_code == 1


def test_run_validation_flags_passed_to_orchestrator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that validation flags are correctly passed to orchestrator."""
    cli = _reload_cli(monkeypatch)
    monkeypatch.setenv("MORPH_API_KEY", "test-key")

    config_dir = tmp_path / "config"
    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    monkeypatch.setattr(src.orchestrator, "MalaOrchestrator", DummyOrchestrator)
    monkeypatch.setattr(cli, "set_verbose", lambda _: None)

    with pytest.raises(typer.Exit) as excinfo:
        cli.run(
            repo_path=tmp_path,
            disable_validations="post-validate",
            coverage_threshold=72.5,
        )

    assert excinfo.value.exit_code == 0
    assert DummyOrchestrator.last_kwargs is not None
    assert DummyOrchestrator.last_kwargs["disable_validations"] == {"post-validate"}
    assert DummyOrchestrator.last_kwargs["coverage_threshold"] == 72.5


def test_run_validation_flags_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that validation flags have correct defaults."""
    cli = _reload_cli(monkeypatch)
    monkeypatch.setenv("MORPH_API_KEY", "test-key")

    config_dir = tmp_path / "config"
    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    monkeypatch.setattr(src.orchestrator, "MalaOrchestrator", DummyOrchestrator)
    monkeypatch.setattr(cli, "set_verbose", lambda _: None)

    with pytest.raises(typer.Exit):
        cli.run(repo_path=tmp_path)

    assert DummyOrchestrator.last_kwargs is not None
    assert DummyOrchestrator.last_kwargs["disable_validations"] is None
    assert DummyOrchestrator.last_kwargs["coverage_threshold"] is None
    assert DummyOrchestrator.last_kwargs["prioritize_wip"] is False
    assert DummyOrchestrator.last_kwargs["focus"] is True


def test_run_wip_flag_passed_to_orchestrator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that --wip flag is correctly passed to orchestrator."""
    cli = _reload_cli(monkeypatch)
    monkeypatch.setenv("MORPH_API_KEY", "test-key")

    config_dir = tmp_path / "config"
    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    monkeypatch.setattr(src.orchestrator, "MalaOrchestrator", DummyOrchestrator)
    monkeypatch.setattr(cli, "set_verbose", lambda _: None)

    with pytest.raises(typer.Exit) as excinfo:
        cli.run(repo_path=tmp_path, wip=True)

    assert excinfo.value.exit_code == 0
    assert DummyOrchestrator.last_kwargs is not None
    assert DummyOrchestrator.last_kwargs["prioritize_wip"] is True


def test_run_focus_flag_default_true(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that --focus defaults to True (epic-grouped ordering)."""
    cli = _reload_cli(monkeypatch)
    monkeypatch.setenv("MORPH_API_KEY", "test-key")

    config_dir = tmp_path / "config"
    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    monkeypatch.setattr(src.orchestrator, "MalaOrchestrator", DummyOrchestrator)
    monkeypatch.setattr(cli, "set_verbose", lambda _: None)

    with pytest.raises(typer.Exit) as excinfo:
        cli.run(repo_path=tmp_path)

    assert excinfo.value.exit_code == 0
    assert DummyOrchestrator.last_kwargs is not None
    assert DummyOrchestrator.last_kwargs["focus"] is True


def test_run_no_focus_flag_passed_to_orchestrator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that --no-focus flag sets focus=False for priority-only ordering."""
    cli = _reload_cli(monkeypatch)
    monkeypatch.setenv("MORPH_API_KEY", "test-key")

    config_dir = tmp_path / "config"
    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    monkeypatch.setattr(src.orchestrator, "MalaOrchestrator", DummyOrchestrator)
    monkeypatch.setattr(cli, "set_verbose", lambda _: None)

    with pytest.raises(typer.Exit) as excinfo:
        cli.run(repo_path=tmp_path, focus=False)

    assert excinfo.value.exit_code == 0
    assert DummyOrchestrator.last_kwargs is not None
    assert DummyOrchestrator.last_kwargs["focus"] is False


def test_run_focus_composes_with_wip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that --focus and --wip flags compose correctly."""
    cli = _reload_cli(monkeypatch)
    monkeypatch.setenv("MORPH_API_KEY", "test-key")

    config_dir = tmp_path / "config"
    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    monkeypatch.setattr(src.orchestrator, "MalaOrchestrator", DummyOrchestrator)
    monkeypatch.setattr(cli, "set_verbose", lambda _: None)

    with pytest.raises(typer.Exit) as excinfo:
        cli.run(repo_path=tmp_path, focus=True, wip=True)

    assert excinfo.value.exit_code == 0
    assert DummyOrchestrator.last_kwargs is not None
    assert DummyOrchestrator.last_kwargs["focus"] is True
    assert DummyOrchestrator.last_kwargs["prioritize_wip"] is True


def test_run_codex_review_disabled_via_disable_validations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that codex-review can be disabled via --disable-validations."""
    cli = _reload_cli(monkeypatch)
    monkeypatch.setenv("MORPH_API_KEY", "test-key")

    config_dir = tmp_path / "config"
    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    monkeypatch.setattr(src.orchestrator, "MalaOrchestrator", DummyOrchestrator)
    monkeypatch.setattr(cli, "set_verbose", lambda _: None)

    with pytest.raises(typer.Exit) as excinfo:
        cli.run(
            repo_path=tmp_path,
            disable_validations="codex-review",
        )

    assert excinfo.value.exit_code == 0
    assert DummyOrchestrator.last_kwargs is not None
    assert DummyOrchestrator.last_kwargs["disable_validations"] == {"codex-review"}


def test_run_coverage_threshold_invalid_negative(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that negative --coverage-threshold value produces error."""
    cli = _reload_cli(monkeypatch)
    monkeypatch.setenv("MORPH_API_KEY", "test-key")

    logs: list[tuple[object, ...]] = []

    def _log(*args: object, **_kwargs: object) -> None:
        logs.append(args)

    config_dir = tmp_path / "config"
    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    monkeypatch.setattr(cli, "log", _log)
    monkeypatch.setattr(cli, "set_verbose", lambda _: None)

    with pytest.raises(typer.Exit) as excinfo:
        cli.run(
            repo_path=tmp_path,
            coverage_threshold=-10.0,
        )

    assert excinfo.value.exit_code == 1
    error_msg = str(logs[-1])
    assert "Invalid --coverage-threshold" in error_msg
    assert "-10.0" in error_msg


def test_run_coverage_threshold_invalid_over_100(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that --coverage-threshold over 100 produces error."""
    cli = _reload_cli(monkeypatch)
    monkeypatch.setenv("MORPH_API_KEY", "test-key")

    logs: list[tuple[object, ...]] = []

    def _log(*args: object, **_kwargs: object) -> None:
        logs.append(args)

    config_dir = tmp_path / "config"
    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    monkeypatch.setattr(cli, "log", _log)
    monkeypatch.setattr(cli, "set_verbose", lambda _: None)

    with pytest.raises(typer.Exit) as excinfo:
        cli.run(
            repo_path=tmp_path,
            coverage_threshold=150.0,
        )

    assert excinfo.value.exit_code == 1
    error_msg = str(logs[-1])
    assert "Invalid --coverage-threshold" in error_msg
    assert "150.0" in error_msg


def test_env_overrides_runs_dir_from_dotenv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that MALA_RUNS_DIR from .env file overrides default RUNS_DIR."""
    # Clear cached modules first
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("src.tools.env") or mod_name.startswith("src.cli"):
            del sys.modules[mod_name]

    # Create a temporary .env file with custom MALA_RUNS_DIR
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    custom_runs_dir = tmp_path / "custom_runs"
    env_file = config_dir / ".env"
    env_file.write_text(f"MALA_RUNS_DIR={custom_runs_dir}\n")

    # Clear any existing env var to ensure we test .env loading
    monkeypatch.delenv("MALA_RUNS_DIR", raising=False)

    # Import env module with patched USER_CONFIG_DIR
    import src.tools.env as env_module

    monkeypatch.setattr(env_module, "USER_CONFIG_DIR", config_dir)
    env_module.load_user_env()
    result = env_module.get_runs_dir()

    assert result == custom_runs_dir


def test_env_overrides_lock_dir_from_dotenv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that MALA_LOCK_DIR from .env file overrides default LOCK_DIR."""
    # Clear cached modules first
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("src.tools.env") or mod_name.startswith("src.cli"):
            del sys.modules[mod_name]

    # Create a temporary .env file with custom MALA_LOCK_DIR
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    custom_lock_dir = tmp_path / "custom_locks"
    env_file = config_dir / ".env"
    env_file.write_text(f"MALA_LOCK_DIR={custom_lock_dir}\n")

    # Clear any existing env var to ensure we test .env loading
    monkeypatch.delenv("MALA_LOCK_DIR", raising=False)

    # Import env module with patched USER_CONFIG_DIR
    import src.tools.env as env_module

    monkeypatch.setattr(env_module, "USER_CONFIG_DIR", config_dir)
    env_module.load_user_env()
    result = env_module.get_lock_dir()

    assert result == custom_lock_dir


def test_env_defaults_when_no_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that defaults are used when env vars are not set."""
    # Clear any existing env vars
    monkeypatch.delenv("MALA_RUNS_DIR", raising=False)
    monkeypatch.delenv("MALA_LOCK_DIR", raising=False)

    # Clear cached modules
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("src.tools.env") or mod_name.startswith("src.cli"):
            del sys.modules[mod_name]

    from src.tools.env import USER_CONFIG_DIR, get_lock_dir, get_runs_dir

    runs_dir = get_runs_dir()
    lock_dir = get_lock_dir()

    assert runs_dir == USER_CONFIG_DIR / "runs"
    assert lock_dir == Path("/tmp/mala-locks")


# Tests for --dry-run functionality


class DummyBeadsClient:
    """Mock BeadsClient for testing dry-run."""

    last_kwargs: ClassVar[dict[str, object] | None] = None
    issues_to_return: ClassVar[list[dict[str, object]]] = []

    def __init__(self, repo_path: Path) -> None:
        self.repo_path = repo_path

    async def get_ready_issues_async(self, **kwargs: object) -> list[dict[str, object]]:
        type(self).last_kwargs = kwargs
        return type(self).issues_to_return


def test_dry_run_exits_without_processing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that --dry-run exits without creating orchestrator."""
    cli = _reload_cli(monkeypatch)
    monkeypatch.setenv("MORPH_API_KEY", "test-key")

    # Reset DummyOrchestrator to detect if it gets called
    DummyOrchestrator.last_kwargs = None
    DummyBeadsClient.issues_to_return = []

    config_dir = tmp_path / "config"
    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    monkeypatch.setattr(src.orchestrator, "MalaOrchestrator", DummyOrchestrator)
    monkeypatch.setattr(src.beads_client, "BeadsClient", DummyBeadsClient)
    monkeypatch.setattr(cli, "set_verbose", lambda _: None)

    with pytest.raises(typer.Exit) as excinfo:
        cli.run(repo_path=tmp_path, dry_run=True)

    assert excinfo.value.exit_code == 0
    # Orchestrator should NOT have been called
    assert DummyOrchestrator.last_kwargs is None


def test_dry_run_passes_flags_to_beads_client(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that --dry-run passes correct flags to get_ready_issues_async."""
    cli = _reload_cli(monkeypatch)
    monkeypatch.setenv("MORPH_API_KEY", "test-key")

    DummyBeadsClient.last_kwargs = None
    DummyBeadsClient.issues_to_return = []

    config_dir = tmp_path / "config"
    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    monkeypatch.setattr(src.beads_client, "BeadsClient", DummyBeadsClient)
    monkeypatch.setattr(cli, "set_verbose", lambda _: None)

    with pytest.raises(typer.Exit):
        cli.run(
            repo_path=tmp_path,
            dry_run=True,
            epic="test-epic",
            only="id-1,id-2",
            wip=True,
            focus=False,
        )

    assert DummyBeadsClient.last_kwargs is not None
    assert DummyBeadsClient.last_kwargs["epic_id"] == "test-epic"
    assert DummyBeadsClient.last_kwargs["only_ids"] == {"id-1", "id-2"}
    assert DummyBeadsClient.last_kwargs["prioritize_wip"] is True
    assert DummyBeadsClient.last_kwargs["focus"] is False


def test_dry_run_displays_empty_task_list(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test that --dry-run handles empty task list."""
    cli = _reload_cli(monkeypatch)
    monkeypatch.setenv("MORPH_API_KEY", "test-key")

    DummyBeadsClient.issues_to_return = []

    config_dir = tmp_path / "config"
    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    monkeypatch.setattr(src.beads_client, "BeadsClient", DummyBeadsClient)
    monkeypatch.setattr(cli, "set_verbose", lambda _: None)

    with pytest.raises(typer.Exit) as excinfo:
        cli.run(repo_path=tmp_path, dry_run=True)

    assert excinfo.value.exit_code == 0
    captured = capsys.readouterr()
    assert "No ready tasks found" in captured.out


def test_dry_run_displays_tasks_with_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test that --dry-run displays task ID, priority, title, and epic."""
    cli = _reload_cli(monkeypatch)
    monkeypatch.setenv("MORPH_API_KEY", "test-key")

    DummyBeadsClient.issues_to_return = [
        {
            "id": "task-1",
            "title": "Test task one",
            "priority": 1,
            "status": "open",
            "parent_epic": "epic-1",
        },
        {
            "id": "task-2",
            "title": "Test task two",
            "priority": 2,
            "status": "in_progress",
            "parent_epic": None,
        },
    ]

    config_dir = tmp_path / "config"
    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    monkeypatch.setattr(src.beads_client, "BeadsClient", DummyBeadsClient)
    monkeypatch.setattr(cli, "set_verbose", lambda _: None)

    with pytest.raises(typer.Exit) as excinfo:
        cli.run(repo_path=tmp_path, dry_run=True, focus=False)

    assert excinfo.value.exit_code == 0
    captured = capsys.readouterr()

    # Check task metadata is shown
    assert "task-1" in captured.out
    assert "Test task one" in captured.out
    assert "P1" in captured.out
    assert "epic-1" in captured.out  # Epic shown in non-focus mode

    assert "task-2" in captured.out
    assert "Test task two" in captured.out
    assert "P2" in captured.out
    assert "(WIP)" in captured.out  # WIP indicator shown


def test_dry_run_focus_mode_groups_by_epic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test that --dry-run with focus mode shows epic headers."""
    cli = _reload_cli(monkeypatch)
    monkeypatch.setenv("MORPH_API_KEY", "test-key")

    DummyBeadsClient.issues_to_return = [
        {
            "id": "task-1",
            "title": "Test task one",
            "priority": 1,
            "status": "open",
            "parent_epic": "epic-1",
        },
        {
            "id": "task-2",
            "title": "Test task two",
            "priority": 2,
            "status": "open",
            "parent_epic": "epic-1",
        },
    ]

    config_dir = tmp_path / "config"
    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    monkeypatch.setattr(src.beads_client, "BeadsClient", DummyBeadsClient)
    monkeypatch.setattr(cli, "set_verbose", lambda _: None)

    with pytest.raises(typer.Exit) as excinfo:
        cli.run(repo_path=tmp_path, dry_run=True, focus=True)

    assert excinfo.value.exit_code == 0
    captured = capsys.readouterr()

    # Check epic header is shown in focus mode
    assert "Epic: epic-1" in captured.out
    assert "task-1" in captured.out
    assert "task-2" in captured.out
    # Summary should show epic counts
    assert "By epic:" in captured.out
