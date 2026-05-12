import importlib
import sys
import types
from pathlib import Path
from collections.abc import Callable
from typing import Any, ClassVar

import pytest
import typer

import src.orchestration.orchestrator
import src.infra.clients.beads_client
import src.infra.tools.locking
import src.orchestration.cli_support
import src.orchestration.factory
import src.infra.io.log_output.run_metadata
from src.core.models import EpicVerificationResult
from src.infra.epic_verifier import DEFAULT_EPIC_VERIFY_LOCK_TIMEOUT_SECONDS


def _reload_cli(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    # Reset bootstrap state before reloading
    if "src.cli.cli" in sys.modules:
        cli_mod = sys.modules["src.cli.cli"]
        # Reset internal state so bootstrap can run again
        cli_mod._bootstrapped = False  # type: ignore[attr-defined]  # ty:ignore[unresolved-attribute]
        # Clear lazy-loaded modules cache so env changes take effect
        cli_mod._lazy_modules.clear()  # type: ignore[attr-defined]
        return importlib.reload(cli_mod)
    return importlib.import_module("src.cli.cli")


class TestImportSafety:
    """Test that importing src.cli.cli has no side effects."""

    def test_import_does_not_bootstrap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Importing src.cli.cli should not run bootstrap()."""
        # Track which modules we'll delete for cleanup
        deleted_modules = [
            mod_name
            for mod_name in list(sys.modules.keys())
            if mod_name.startswith("src.cli")
        ]
        for mod_name in deleted_modules:
            del sys.modules[mod_name]

        try:
            # Force reimport of cli module
            import src.cli.cli

            # Import should NOT have triggered bootstrap
            # Check observable state: _bootstrapped flag should still be False
            assert not src.cli.cli._bootstrapped, (
                "_bootstrapped is True after import - bootstrap() should only run when explicitly called"
            )

            # Verify the module was imported correctly
            assert hasattr(src.cli.cli, "bootstrap")
        finally:
            # Reload modules from disk to restore clean state for other tests
            for mod_name in deleted_modules:
                if mod_name in sys.modules:
                    del sys.modules[mod_name]
            for mod_name in deleted_modules:
                importlib.import_module(mod_name)


class DummyOrchestrator:
    last_orch_config: Any = None
    last_mala_config: Any = None
    last_watch_config: Any = None
    last_validation_config: Any = None

    def __init__(self, **kwargs: object) -> None:
        self._exit_code = 0

    async def run(
        self, *, watch_config: object = None, validation_config: object = None
    ) -> tuple[int, int]:
        DummyOrchestrator.last_watch_config = watch_config
        DummyOrchestrator.last_validation_config = validation_config
        return (1, 1)

    @property
    def exit_code(self) -> int:
        return self._exit_code


class DummyEpicVerifier:
    last_call: ClassVar[dict[str, object] | None] = None
    reviewer_type = "cerberus"
    lock_timeout_seconds: int | None = 120

    @property
    def lock_timeout_display(self) -> str:
        if self.lock_timeout_seconds is None:
            return f"default({DEFAULT_EPIC_VERIFY_LOCK_TIMEOUT_SECONDS}s)"
        return f"{self.lock_timeout_seconds}s"

    async def verify_epic_with_options(
        self,
        epic_id: str,
        *,
        human_override: bool = False,
        require_eligible: bool = True,
        close_epic: bool = True,
    ) -> EpicVerificationResult:
        DummyEpicVerifier.last_call = {
            "epic_id": epic_id,
            "human_override": human_override,
            "require_eligible": require_eligible,
            "close_epic": close_epic,
        }
        return EpicVerificationResult(
            verified_count=1,
            passed_count=1,
            failed_count=0,
            verdicts={},
            remediation_issues_created=[],
        )


class DummyOrchestratorWithVerifier:
    def __init__(self, verifier: DummyEpicVerifier) -> None:
        self.epic_verifier = verifier


def _make_dummy_create_orchestrator() -> Callable[[object], DummyOrchestrator]:
    """Create a dummy create_orchestrator that captures arguments."""

    def dummy_create_orchestrator(
        config: object,
        *,
        mala_config: object = None,
        deps: object = None,
    ) -> DummyOrchestrator:
        DummyOrchestrator.last_orch_config = config
        DummyOrchestrator.last_mala_config = mala_config
        return DummyOrchestrator()

    return dummy_create_orchestrator


def _make_dummy_create_orchestrator_with_verifier(
    verifier: DummyEpicVerifier,
) -> Callable[[object], DummyOrchestratorWithVerifier]:
    """Create a dummy create_orchestrator that provides an epic verifier."""

    def dummy_create_orchestrator(
        config: object,
        *,
        mala_config: object = None,
        deps: object = None,
    ) -> DummyOrchestratorWithVerifier:
        return DummyOrchestratorWithVerifier(verifier)

    return dummy_create_orchestrator


def test_run_invalid_scope_ids_exits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cli = _reload_cli(monkeypatch)
    logs: list[tuple[object, ...]] = []

    def _log(*args: object, **_kwargs: object) -> None:
        logs.append(args)

    monkeypatch.setattr(cli, "log", _log)
    monkeypatch.setattr(cli, "set_verbose", lambda _: None)

    with pytest.raises(typer.Exit) as excinfo:
        cli.run(repo_path=tmp_path, scope="ids:")

    assert excinfo.value.exit_code == 1
    assert logs


def test_run_success_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cli = _reload_cli(monkeypatch)

    verbose_calls = {"enabled": None}

    def _set_verbose(value: bool) -> None:
        verbose_calls["enabled"] = value

    config_dir = tmp_path / "config"
    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    # CLI now uses create_orchestrator from factory
    import src.orchestration.factory

    monkeypatch.setattr(
        src.orchestration.factory,
        "create_orchestrator",
        _make_dummy_create_orchestrator(),
    )
    monkeypatch.setattr(cli, "set_verbose", _set_verbose)

    with pytest.raises(typer.Exit) as excinfo:
        cli.run(
            repo_path=tmp_path,
            max_agents=2,
            timeout=7,
            max_issues=3,
            scope="ids:id-1,id-2",
            verbose=True,
        )

    assert excinfo.value.exit_code == 0
    assert verbose_calls["enabled"] is True
    # Check OrchestratorConfig passed to create_orchestrator
    orch_config = DummyOrchestrator.last_orch_config
    assert orch_config is not None
    assert orch_config.only_ids == ["id-1", "id-2"]
    assert config_dir.exists()


def test_run_verbose_mode_sets_verbose_true(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that --verbose flag sets verbose to True."""
    cli = _reload_cli(monkeypatch)

    verbose_calls = {"enabled": None}

    def _set_verbose(value: bool) -> None:
        verbose_calls["enabled"] = value

    config_dir = tmp_path / "config"
    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    import src.orchestration.factory

    monkeypatch.setattr(
        src.orchestration.factory,
        "create_orchestrator",
        _make_dummy_create_orchestrator(),
    )
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

    verbose_calls = {"enabled": None}

    def _set_verbose(value: bool) -> None:
        verbose_calls["enabled"] = value

    config_dir = tmp_path / "config"
    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    import src.orchestration.factory

    monkeypatch.setattr(
        src.orchestration.factory,
        "create_orchestrator",
        _make_dummy_create_orchestrator(),
    )
    monkeypatch.setattr(cli, "set_verbose", _set_verbose)

    with pytest.raises(typer.Exit):
        cli.run(repo_path=tmp_path)

    assert verbose_calls["enabled"] is False


def test_run_claude_settings_sources_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that --claude-settings-sources updates MalaConfig."""
    cli = _reload_cli(monkeypatch)

    config_dir = tmp_path / "config"
    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    import src.orchestration.factory

    monkeypatch.setattr(
        src.orchestration.factory,
        "create_orchestrator",
        _make_dummy_create_orchestrator(),
    )
    monkeypatch.setattr(cli, "set_verbose", lambda _: None)

    with pytest.raises(typer.Exit) as excinfo:
        cli.run(repo_path=tmp_path, claude_settings_sources="user")

    assert excinfo.value.exit_code == 0
    config = DummyOrchestrator.last_mala_config
    assert config.claude_settings_sources == ("user",)


def test_run_repo_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cli = _reload_cli(monkeypatch)

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


def test_epic_verify_invokes_verifier(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cli = _reload_cli(monkeypatch)
    verifier = DummyEpicVerifier()

    config_dir = tmp_path / "config"
    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    monkeypatch.setattr(cli, "set_verbose", lambda _: None)
    logs: list[tuple[object, ...]] = []
    monkeypatch.setattr(cli, "log", lambda *args, **_kwargs: logs.append(args))

    import src.orchestration.factory

    monkeypatch.setattr(
        src.orchestration.factory,
        "create_orchestrator",
        _make_dummy_create_orchestrator_with_verifier(verifier),
    )

    with pytest.raises(typer.Exit) as excinfo:
        cli.epic_verify(
            epic_id="epic-1",
            repo_path=tmp_path,
            force=True,
            close=False,
        )

    assert excinfo.value.exit_code == 0
    assert DummyEpicVerifier.last_call == {
        "epic_id": "epic-1",
        "human_override": False,
        "require_eligible": False,
        "close_epic": False,
    }
    log_messages = [str(args[1]) for args in logs if len(args) > 1]
    assert any("[START] Epic verification" in msg for msg in log_messages)
    assert any(f"Repository: {tmp_path}" in msg for msg in log_messages)
    assert any("Epic: epic-1" in msg for msg in log_messages)
    assert any(
        "reviewer=cerberus close=false force=true" in msg
        and "lock_timeout=120s" in msg
        for msg in log_messages
    )


def test_epic_verify_logs_default_lock_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cli = _reload_cli(monkeypatch)
    verifier = DummyEpicVerifier()
    verifier.lock_timeout_seconds = None

    config_dir = tmp_path / "config"
    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    monkeypatch.setattr(cli, "set_verbose", lambda _: None)
    logs: list[tuple[object, ...]] = []
    monkeypatch.setattr(cli, "log", lambda *args, **_kwargs: logs.append(args))

    import src.orchestration.factory

    monkeypatch.setattr(
        src.orchestration.factory,
        "create_orchestrator",
        _make_dummy_create_orchestrator_with_verifier(verifier),
    )

    with pytest.raises(typer.Exit) as excinfo:
        cli.epic_verify(epic_id="epic-1", repo_path=tmp_path, force=True)

    assert excinfo.value.exit_code == 0
    log_messages = [str(args[1]) for args in logs if len(args) > 1]
    assert any(
        f"lock_timeout=default({DEFAULT_EPIC_VERIFY_LOCK_TIMEOUT_SECONDS}s)" in msg
        for msg in log_messages
    )


def test_epic_verify_shows_ineligibility_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that epic-verify shows reason when epic is not eligible."""
    cli = _reload_cli(monkeypatch)

    class IneligibleVerifier:
        async def verify_epic_with_options(
            self,
            epic_id: str,
            **kwargs: object,
        ) -> EpicVerificationResult:
            return EpicVerificationResult(
                verified_count=0,
                passed_count=0,
                failed_count=0,
                verdicts={},
                remediation_issues_created=[],
                ineligibility_reason="3 of 5 child issues still open",
            )

    class OrchestratorWithIneligibleVerifier:
        def __init__(self, verifier: IneligibleVerifier) -> None:
            self.epic_verifier = verifier

    config_dir = tmp_path / "config"
    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    monkeypatch.setattr(cli, "set_verbose", lambda _: None)

    logs: list[tuple[object, ...]] = []

    def mock_log(*args: object, **kwargs: object) -> None:
        logs.append(args)

    monkeypatch.setattr(cli, "log", mock_log)

    import src.orchestration.factory

    def make_orch(
        config: object, **kwargs: object
    ) -> OrchestratorWithIneligibleVerifier:
        return OrchestratorWithIneligibleVerifier(IneligibleVerifier())

    monkeypatch.setattr(src.orchestration.factory, "create_orchestrator", make_orch)

    with pytest.raises(typer.Exit) as excinfo:
        cli.epic_verify(epic_id="epic-1", repo_path=tmp_path)

    assert excinfo.value.exit_code == 1
    # Check that ineligibility reason was logged
    log_messages = [str(args) for args in logs]
    assert any("3 of 5 child issues still open" in msg for msg in log_messages)


def test_clean_removes_locks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that clean removes lock files but not run metadata."""
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

    # Patch both cli_support (where cli imports from) and src.infra.tools.locking
    # (where cli_support imports from) to ensure full isolation
    monkeypatch.setattr(src.orchestration.cli_support, "get_lock_dir", lambda: lock_dir)
    monkeypatch.setattr(src.infra.tools.locking, "get_lock_dir", lambda: lock_dir)
    monkeypatch.setattr(cli, "get_runs_dir", lambda: run_dir)
    monkeypatch.setattr(cli, "log", _log)
    # Mock no running instances
    monkeypatch.setattr(
        src.orchestration.cli_support, "get_running_instances", lambda: []
    )
    monkeypatch.setattr(cli, "_lazy_modules", {"get_running_instances": lambda: []})

    cli.clean()

    # Locks should be removed
    assert not list(lock_dir.glob("*.lock"))
    # Run metadata should NOT be removed (clean no longer deletes logs)
    assert len(list(run_dir.glob("*.json"))) == 2
    assert logs


def test_clean_exits_if_instance_running(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that clean exits early if a mala instance is running."""
    cli = _reload_cli(monkeypatch)
    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()
    (lock_dir / "one.lock").write_text("agent")

    logs: list[tuple[object, ...]] = []

    def _log(*args: object, **_kwargs: object) -> None:
        logs.append(args)

    monkeypatch.setattr(src.orchestration.cli_support, "get_lock_dir", lambda: lock_dir)
    monkeypatch.setattr(src.infra.tools.locking, "get_lock_dir", lambda: lock_dir)
    monkeypatch.setattr(cli, "log", _log)

    # Mock a running instance
    from src.infra.io.log_output.run_metadata import RunningInstance
    from datetime import datetime

    running_instance = RunningInstance(
        run_id="test-run",
        repo_path=tmp_path,
        started_at=datetime.now(),
        pid=12345,
        max_agents=3,
    )
    monkeypatch.setattr(
        src.orchestration.cli_support,
        "get_running_instances",
        lambda: [running_instance],
    )
    monkeypatch.setattr(
        cli, "_lazy_modules", {"get_running_instances": lambda: [running_instance]}
    )

    with pytest.raises(typer.Exit) as exc_info:
        cli.clean()

    assert exc_info.value.exit_code == 1
    # Lock should NOT be removed (exited early)
    assert list(lock_dir.glob("*.lock"))
    # Should have logged a warning
    assert any("running" in str(log_entry).lower() for log_entry in logs)


def test_clean_force_bypasses_running_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that --force allows cleaning even when instance is running."""
    cli = _reload_cli(monkeypatch)
    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()
    (lock_dir / "one.lock").write_text("agent")

    logs: list[tuple[object, ...]] = []

    def _log(*args: object, **_kwargs: object) -> None:
        logs.append(args)

    monkeypatch.setattr(src.orchestration.cli_support, "get_lock_dir", lambda: lock_dir)
    monkeypatch.setattr(src.infra.tools.locking, "get_lock_dir", lambda: lock_dir)
    monkeypatch.setattr(cli, "log", _log)

    # Mock a running instance
    from src.infra.io.log_output.run_metadata import RunningInstance
    from datetime import datetime

    running_instance = RunningInstance(
        run_id="test-run",
        repo_path=tmp_path,
        started_at=datetime.now(),
        pid=12345,
        max_agents=3,
    )
    monkeypatch.setattr(
        src.orchestration.cli_support,
        "get_running_instances",
        lambda: [running_instance],
    )
    monkeypatch.setattr(
        cli, "_lazy_modules", {"get_running_instances": lambda: [running_instance]}
    )

    # With force=True, it should proceed
    cli.clean(force=True)

    # Lock should be removed despite running instance
    assert not list(lock_dir.glob("*.lock"))
    assert logs


def test_status_no_running_instance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test status when no mala instance is running in cwd."""
    cli = _reload_cli(monkeypatch)

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / ".env").write_text("MALA_TEST=1")

    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()

    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    # Patch both cli_support and underlying modules
    monkeypatch.setattr(src.orchestration.cli_support, "get_lock_dir", lambda: lock_dir)
    monkeypatch.setattr(src.infra.tools.locking, "get_lock_dir", lambda: lock_dir)
    # Mock to return no running instances - patch cli_support where cli imports from
    monkeypatch.setattr(
        src.orchestration.cli_support, "get_running_instances_for_dir", lambda _: []
    )
    monkeypatch.setattr(
        src.infra.io.log_output.run_metadata,
        "get_running_instances_for_dir",
        lambda _: [],
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
    from datetime import datetime, UTC

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / ".env").write_text("MALA_TEST=1")

    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()
    for idx in range(6):
        (lock_dir / f"lock-{idx}.lock").write_text(f"agent-{idx}")

    run_dir = tmp_path / "runs"
    run_dir.mkdir()
    (run_dir / "one.json").write_text("{}")
    (run_dir / "two.json").write_text("{}")

    # Create a mock RunningInstance
    mock_instance = src.infra.io.log_output.run_metadata.RunningInstance(
        run_id="test-run-id",
        repo_path=tmp_path,
        started_at=datetime.now(UTC),
        pid=12345,
        max_agents=3,
    )

    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    monkeypatch.setattr(cli, "get_runs_dir", lambda: run_dir)

    # Create mock get_all_locks that reads from our test lock_dir
    def mock_get_all_locks() -> dict[str, list[str]]:
        locks_by_agent: dict[str, list[str]] = {}
        for lock in lock_dir.glob("*.lock"):
            agent_id = lock.read_text().strip()
            if agent_id not in locks_by_agent:
                locks_by_agent[agent_id] = []
            locks_by_agent[agent_id].append(lock.stem)
        return locks_by_agent

    # Inject into lazy modules cache so cli.status() uses it
    cli._lazy_modules["get_all_locks"] = mock_get_all_locks
    # Patch get_running_instances_for_dir in cli_support where cli imports from
    monkeypatch.setattr(
        src.orchestration.cli_support,
        "get_running_instances_for_dir",
        lambda _: [mock_instance],
    )
    monkeypatch.setattr(
        src.infra.io.log_output.run_metadata,
        "get_running_instances_for_dir",
        lambda _: [mock_instance],
    )

    cli.status()

    output = capsys.readouterr().out
    assert "mala running" in output
    assert str(tmp_path) in output
    assert "max-agents: 3" in output
    assert "pid: 12345" in output
    assert "active lock" in output  # "active lock(s)"
    assert "run metadata files" in output


def test_status_all_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test status --all flag shows all instances grouped by directory."""
    cli = _reload_cli(monkeypatch)
    from datetime import datetime, UTC

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / ".env").write_text("MALA_TEST=1")

    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()

    # Create mock instances for two different directories
    mock_instance_1 = src.infra.io.log_output.run_metadata.RunningInstance(
        run_id="run-1",
        repo_path=tmp_path / "repo1",
        started_at=datetime.now(UTC),
        pid=111,
        max_agents=2,
    )
    mock_instance_2 = src.infra.io.log_output.run_metadata.RunningInstance(
        run_id="run-2",
        repo_path=tmp_path / "repo2",
        started_at=datetime.now(UTC),
        pid=222,
        max_agents=None,
    )

    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    # Patch both cli_support and underlying modules
    monkeypatch.setattr(src.orchestration.cli_support, "get_lock_dir", lambda: lock_dir)
    monkeypatch.setattr(src.infra.tools.locking, "get_lock_dir", lambda: lock_dir)
    # Patch get_running_instances in cli_support where cli imports from
    monkeypatch.setattr(
        src.orchestration.cli_support,
        "get_running_instances",
        lambda: [mock_instance_1, mock_instance_2],
    )
    monkeypatch.setattr(
        src.infra.io.log_output.run_metadata,
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


def test_run_validation_flags_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that validation flags have correct defaults."""
    cli = _reload_cli(monkeypatch)

    config_dir = tmp_path / "config"
    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    import src.orchestration.factory

    monkeypatch.setattr(
        src.orchestration.factory,
        "create_orchestrator",
        _make_dummy_create_orchestrator(),
    )
    monkeypatch.setattr(cli, "set_verbose", lambda _: None)

    with pytest.raises(typer.Exit) as excinfo:
        cli.run(repo_path=tmp_path)

    assert excinfo.value.exit_code == 0
    assert DummyOrchestrator.last_orch_config is not None
    assert DummyOrchestrator.last_orch_config.disable_validations is None
    assert DummyOrchestrator.last_orch_config.include_wip is False
    assert DummyOrchestrator.last_orch_config.focus is True


def test_run_epic_priority_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that epic-priority ordering is the default (epic-grouped ordering)."""
    cli = _reload_cli(monkeypatch)

    config_dir = tmp_path / "config"
    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    import src.orchestration.factory

    monkeypatch.setattr(
        src.orchestration.factory,
        "create_orchestrator",
        _make_dummy_create_orchestrator(),
    )
    monkeypatch.setattr(cli, "set_verbose", lambda _: None)

    with pytest.raises(typer.Exit) as excinfo:
        cli.run(repo_path=tmp_path)

    assert excinfo.value.exit_code == 0
    assert DummyOrchestrator.last_orch_config is not None
    assert DummyOrchestrator.last_orch_config.focus is True


def test_run_order_issue_priority_sets_focus_false(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that --order issue-priority sets focus=False for global priority ordering."""
    cli = _reload_cli(monkeypatch)

    config_dir = tmp_path / "config"
    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    import src.orchestration.factory

    monkeypatch.setattr(
        src.orchestration.factory,
        "create_orchestrator",
        _make_dummy_create_orchestrator(),
    )
    monkeypatch.setattr(cli, "set_verbose", lambda _: None)

    with pytest.raises(typer.Exit) as excinfo:
        cli.run(repo_path=tmp_path, order="issue-priority")

    assert excinfo.value.exit_code == 0
    assert DummyOrchestrator.last_orch_config is not None
    assert DummyOrchestrator.last_orch_config.focus is False


def test_run_order_epic_priority_composes_with_resume(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that --order epic-priority and --resume flags compose correctly."""
    cli = _reload_cli(monkeypatch)

    config_dir = tmp_path / "config"
    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    import src.orchestration.factory

    monkeypatch.setattr(
        src.orchestration.factory,
        "create_orchestrator",
        _make_dummy_create_orchestrator(),
    )
    monkeypatch.setattr(cli, "set_verbose", lambda _: None)

    with pytest.raises(typer.Exit) as excinfo:
        cli.run(repo_path=tmp_path, order="epic-priority", resume=True)

    assert excinfo.value.exit_code == 0
    assert DummyOrchestrator.last_orch_config is not None
    assert DummyOrchestrator.last_orch_config.focus is True
    assert DummyOrchestrator.last_orch_config.include_wip is True


def test_run_order_input_with_scope_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that --order input with --scope ids: sets OrderPreference.INPUT."""
    cli = _reload_cli(monkeypatch)

    config_dir = tmp_path / "config"
    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    import src.orchestration.factory
    from src.core.models import OrderPreference

    monkeypatch.setattr(
        src.orchestration.factory,
        "create_orchestrator",
        _make_dummy_create_orchestrator(),
    )
    monkeypatch.setattr(cli, "set_verbose", lambda _: None)

    with pytest.raises(typer.Exit) as excinfo:
        cli.run(repo_path=tmp_path, scope="ids:T-1,T-2,T-3", order="input")

    assert excinfo.value.exit_code == 0
    assert DummyOrchestrator.last_orch_config is not None
    assert DummyOrchestrator.last_orch_config.order_preference == OrderPreference.INPUT
    assert DummyOrchestrator.last_orch_config.only_ids == ["T-1", "T-2", "T-3"]
    assert DummyOrchestrator.last_orch_config.focus is False


def test_run_order_input_without_scope_ids_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that --order input without --scope ids: fails with clear error."""
    cli = _reload_cli(monkeypatch)

    logs: list[tuple[object, ...]] = []

    def _log(*args: object, **_kwargs: object) -> None:
        logs.append(args)

    config_dir = tmp_path / "config"
    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    monkeypatch.setattr(cli, "log", _log)
    monkeypatch.setattr(cli, "set_verbose", lambda _: None)

    with pytest.raises(typer.Exit) as excinfo:
        cli.run(repo_path=tmp_path, order="input")

    assert excinfo.value.exit_code == 1
    error_msg = str(logs[-1])
    assert "--order input" in error_msg
    assert "--scope ids:" in error_msg


def test_run_order_input_with_scope_epic_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that --order input with --scope epic: fails."""
    cli = _reload_cli(monkeypatch)

    logs: list[tuple[object, ...]] = []

    def _log(*args: object, **_kwargs: object) -> None:
        logs.append(args)

    config_dir = tmp_path / "config"
    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    monkeypatch.setattr(cli, "log", _log)
    monkeypatch.setattr(cli, "set_verbose", lambda _: None)

    with pytest.raises(typer.Exit) as excinfo:
        cli.run(repo_path=tmp_path, scope="epic:E-1", order="input")

    assert excinfo.value.exit_code == 1
    error_msg = str(logs[-1])
    assert "--order input" in error_msg
    assert "--scope ids:" in error_msg


@pytest.mark.parametrize(
    "flag",
    ["--resume", "-r"],
    ids=["resume_long", "resume_short"],
)
def test_run_resume_flags_set_include_wip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, flag: str
) -> None:
    """Test that --resume and -r set include_wip=True via CLI parsing."""
    from typer.testing import CliRunner

    cli = _reload_cli(monkeypatch)

    config_dir = tmp_path / "config"
    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    import src.orchestration.factory

    monkeypatch.setattr(
        src.orchestration.factory,
        "create_orchestrator",
        _make_dummy_create_orchestrator(),
    )
    monkeypatch.setattr(cli, "set_verbose", lambda _: None)

    runner = CliRunner()
    result = runner.invoke(cli.app, ["run", str(tmp_path), flag])

    assert result.exit_code == 0
    assert DummyOrchestrator.last_orch_config is not None
    assert DummyOrchestrator.last_orch_config.include_wip is True


def test_strict_without_resume_raises_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that --strict without --resume raises BadParameter error."""
    from typer.testing import CliRunner

    cli = _reload_cli(monkeypatch)

    config_dir = tmp_path / "config"
    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    monkeypatch.setattr(cli, "set_verbose", lambda _: None)

    runner = CliRunner()
    result = runner.invoke(cli.app, ["run", str(tmp_path), "--strict"])

    assert result.exit_code != 0
    assert "--strict requires --resume" in result.output


def test_strict_with_resume_passes_to_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that --strict with --resume passes strict_resume=True to config."""
    from typer.testing import CliRunner

    cli = _reload_cli(monkeypatch)

    config_dir = tmp_path / "config"
    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    import src.orchestration.factory

    monkeypatch.setattr(
        src.orchestration.factory,
        "create_orchestrator",
        _make_dummy_create_orchestrator(),
    )
    monkeypatch.setattr(cli, "set_verbose", lambda _: None)

    runner = CliRunner()
    result = runner.invoke(cli.app, ["run", str(tmp_path), "--resume", "--strict"])

    assert result.exit_code == 0
    assert DummyOrchestrator.last_orch_config is not None
    assert DummyOrchestrator.last_orch_config.strict_resume is True


def test_fresh_without_resume_raises_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that --fresh without --resume raises BadParameter error."""
    from typer.testing import CliRunner

    cli = _reload_cli(monkeypatch)

    config_dir = tmp_path / "config"
    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    monkeypatch.setattr(cli, "set_verbose", lambda _: None)

    runner = CliRunner()
    result = runner.invoke(cli.app, ["run", str(tmp_path), "--fresh"])

    assert result.exit_code != 0
    assert "--fresh requires --resume" in result.output


def test_fresh_with_strict_raises_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that --fresh with --strict raises BadParameter error."""
    from typer.testing import CliRunner

    cli = _reload_cli(monkeypatch)

    config_dir = tmp_path / "config"
    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    monkeypatch.setattr(cli, "set_verbose", lambda _: None)

    runner = CliRunner()
    result = runner.invoke(
        cli.app, ["run", str(tmp_path), "--resume", "--fresh", "--strict"]
    )

    assert result.exit_code != 0
    assert "--fresh and --strict are mutually exclusive" in result.output


def test_fresh_with_resume_passes_to_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that --fresh with --resume passes fresh_session=True to config."""
    from typer.testing import CliRunner

    cli = _reload_cli(monkeypatch)

    config_dir = tmp_path / "config"
    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    import src.orchestration.factory

    monkeypatch.setattr(
        src.orchestration.factory,
        "create_orchestrator",
        _make_dummy_create_orchestrator(),
    )
    monkeypatch.setattr(cli, "set_verbose", lambda _: None)

    runner = CliRunner()
    result = runner.invoke(cli.app, ["run", str(tmp_path), "--resume", "--fresh"])

    assert result.exit_code == 0
    assert DummyOrchestrator.last_orch_config is not None
    assert DummyOrchestrator.last_orch_config.fresh_session is True


def test_run_scope_epic_and_orphans_are_distinct(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that --scope epic and --scope orphans are distinct options."""
    cli = _reload_cli(monkeypatch)

    config_dir = tmp_path / "config"
    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    import src.orchestration.factory

    monkeypatch.setattr(
        src.orchestration.factory,
        "create_orchestrator",
        _make_dummy_create_orchestrator(),
    )
    monkeypatch.setattr(cli, "set_verbose", lambda _: None)

    # Test scope epic
    with pytest.raises(typer.Exit) as excinfo:
        cli.run(repo_path=tmp_path, scope="epic:epic-1")

    assert excinfo.value.exit_code == 0
    assert DummyOrchestrator.last_orch_config is not None
    assert DummyOrchestrator.last_orch_config.epic_id == "epic-1"
    assert DummyOrchestrator.last_orch_config.orphans_only is False

    # Test scope orphans
    with pytest.raises(typer.Exit) as excinfo:
        cli.run(repo_path=tmp_path, scope="orphans")

    assert excinfo.value.exit_code == 0
    assert DummyOrchestrator.last_orch_config is not None
    assert DummyOrchestrator.last_orch_config.epic_id is None
    assert DummyOrchestrator.last_orch_config.orphans_only is True


def test_env_overrides_runs_dir_from_dotenv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that MALA_RUNS_DIR from .env file overrides default RUNS_DIR."""
    # Clear cached modules first
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("src.infra.tools.env") or mod_name.startswith("src.cli"):
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
    import src.infra.tools.env as env_module

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
        if mod_name.startswith("src.infra.tools.env") or mod_name.startswith("src.cli"):
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
    import src.infra.tools.env as env_module

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
        if mod_name.startswith("src.infra.tools.env") or mod_name.startswith("src.cli"):
            del sys.modules[mod_name]

    from src.infra.tools.env import USER_CONFIG_DIR, get_lock_dir, get_runs_dir

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

    # Reset DummyOrchestrator to detect if it gets called
    DummyOrchestrator.last_orch_config = None
    DummyBeadsClient.issues_to_return = []

    config_dir = tmp_path / "config"
    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)

    monkeypatch.setattr(
        src.orchestration.factory,
        "create_orchestrator",
        _make_dummy_create_orchestrator(),
    )
    monkeypatch.setattr(
        src.orchestration.factory,
        "create_issue_provider",
        lambda repo_path, log_warning=None: DummyBeadsClient(repo_path),
    )
    monkeypatch.setattr(cli, "set_verbose", lambda _: None)

    with pytest.raises(typer.Exit) as excinfo:
        cli.run(repo_path=tmp_path, dry_run=True)

    assert excinfo.value.exit_code == 0
    # Orchestrator should NOT have been called
    assert DummyOrchestrator.last_orch_config is None


def test_dry_run_passes_flags_to_beads_client(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that --dry-run passes correct flags to get_ready_issues_async."""
    cli = _reload_cli(monkeypatch)

    DummyBeadsClient.last_kwargs = None
    DummyBeadsClient.issues_to_return = []

    config_dir = tmp_path / "config"
    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    monkeypatch.setattr(
        src.orchestration.factory,
        "create_issue_provider",
        lambda repo_path, log_warning=None: DummyBeadsClient(repo_path),
    )
    monkeypatch.setattr(cli, "set_verbose", lambda _: None)

    with pytest.raises(typer.Exit):
        cli.run(
            repo_path=tmp_path,
            dry_run=True,
            scope="ids:id-1,id-2",
            resume=True,
            order="issue-priority",
        )

    assert DummyBeadsClient.last_kwargs is not None
    assert DummyBeadsClient.last_kwargs["only_ids"] == ["id-1", "id-2"]
    assert DummyBeadsClient.last_kwargs["include_wip"] is True
    assert DummyBeadsClient.last_kwargs["focus"] is False


def test_dry_run_displays_empty_task_list(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test that --dry-run handles empty task list."""
    cli = _reload_cli(monkeypatch)

    DummyBeadsClient.issues_to_return = []

    config_dir = tmp_path / "config"
    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    monkeypatch.setattr(
        src.orchestration.factory,
        "create_issue_provider",
        lambda repo_path, log_warning=None: DummyBeadsClient(repo_path),
    )
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
    monkeypatch.setattr(
        src.orchestration.factory,
        "create_issue_provider",
        lambda repo_path, log_warning=None: DummyBeadsClient(repo_path),
    )
    monkeypatch.setattr(cli, "set_verbose", lambda _: None)

    with pytest.raises(typer.Exit) as excinfo:
        cli.run(repo_path=tmp_path, dry_run=True, order="issue-priority")

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
    monkeypatch.setattr(
        src.orchestration.factory,
        "create_issue_provider",
        lambda repo_path, log_warning=None: DummyBeadsClient(repo_path),
    )
    monkeypatch.setattr(cli, "set_verbose", lambda _: None)

    with pytest.raises(typer.Exit) as excinfo:
        cli.run(repo_path=tmp_path, dry_run=True, order="focus")

    assert excinfo.value.exit_code == 0
    captured = capsys.readouterr()

    # Check epic header is shown in focus mode
    assert "Epic: epic-1" in captured.out
    assert "task-1" in captured.out
    assert "task-2" in captured.out
    # Summary should show epic counts
    assert "By epic:" in captured.out


# ============================================================================
# Tests for _validate_run_args helper function
# ============================================================================


# ============================================================================
# Tests for _apply_config_overrides helper function
# ============================================================================


# Branch coverage for the underlying ``parse_scope`` helper lives in
# tests/unit/orchestration/test_cli_options.py alongside the helper that owns
# the parsing logic. The class below covers the thin CLI wrapper that converts
# ``ValueError`` into ``typer.Exit(1)`` and emits the dedupe warning.


class TestParseScopeCli:
    """Cover the CLI-side ``_parse_scope_cli`` wrapper."""

    def test_passes_through_valid_scope(self) -> None:
        from src.cli.cli import _parse_scope_cli

        result = _parse_scope_cli("all")
        assert result.scope_type == "all"

    def test_converts_value_error_to_exit_1(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from src.cli.cli import _parse_scope_cli

        with pytest.raises(typer.Exit) as excinfo:
            _parse_scope_cli("epic:")
        assert excinfo.value.exit_code == 1
        captured = capsys.readouterr()
        assert "epic:" in captured.out or "epic:" in captured.err

    def test_emits_warning_for_duplicate_ids(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from src.cli.cli import _parse_scope_cli

        result = _parse_scope_cli("ids:T-1,T-1,T-2")
        assert result.ids == ["T-1", "T-2"]
        captured = capsys.readouterr()
        assert "Duplicate IDs removed" in captured.out


class TestClaudeSettingsSourcesHelp:
    """Test that --claude-settings-sources appears in help output."""

    def test_run_help_shows_claude_settings_sources(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that 'run --help' includes --claude-settings-sources option."""
        from typer.testing import CliRunner

        cli = _reload_cli(monkeypatch)
        runner = CliRunner()
        result = runner.invoke(cli.app, ["run", "--help"])

        assert result.exit_code == 0
        assert "--claude-settings-sources" in result.output

    def test_epic_verify_help_shows_claude_settings_sources(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that 'epic-verify --help' includes --claude-settings-sources option."""
        from typer.testing import CliRunner

        cli = _reload_cli(monkeypatch)
        runner = CliRunner()
        result = runner.invoke(cli.app, ["epic-verify", "--help"])

        assert result.exit_code == 0
        assert "--claude-settings-sources" in result.output
