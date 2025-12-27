import importlib
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import typer


def _reload_cli(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    monkeypatch.setenv("BRAINTRUST_API_KEY", "")
    if "src.cli" in sys.modules:
        return importlib.reload(sys.modules["src.cli"])
    return importlib.import_module("src.cli")


class DummyOrchestrator:
    last_kwargs: dict[str, Any] | None = None

    def __init__(self, **kwargs: object) -> None:
        type(self).last_kwargs = kwargs

    async def run(self) -> tuple[int, int]:
        return (1, 1)


def test_validate_morph_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    cli = _reload_cli(monkeypatch)
    monkeypatch.delenv("MORPH_API_KEY", raising=False)

    with pytest.raises(SystemExit):
        cli.validate_morph_api_key()


def test_run_invalid_only_exits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cli = _reload_cli(monkeypatch)
    logs: list[tuple[object, ...]] = []

    def _log(*args: object, **_kwargs: object) -> None:
        logs.append(args)

    monkeypatch.setattr(cli, "log", _log)

    with pytest.raises(typer.Exit) as excinfo:
        cli.run(repo_path=tmp_path, only=" , ", verbose=False)

    assert excinfo.value.exit_code == 1
    assert logs


def test_run_success_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cli = _reload_cli(monkeypatch)
    monkeypatch.setenv("MORPH_API_KEY", "test-key")

    verbose_calls = {"enabled": False}

    def _set_verbose(value: bool) -> None:
        verbose_calls["enabled"] = value

    config_dir = tmp_path / "config"
    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    monkeypatch.setattr(cli, "MalaOrchestrator", DummyOrchestrator)
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
            codex_review=False,
            verbose=True,
        )

    assert excinfo.value.exit_code == 0
    assert verbose_calls["enabled"] is True
    assert DummyOrchestrator.last_kwargs["only_ids"] == {"id-1", "id-2"}
    assert config_dir.exists()


def test_run_repo_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cli = _reload_cli(monkeypatch)
    monkeypatch.setenv("MORPH_API_KEY", "test-key")

    logs: list[tuple[object, ...]] = []

    def _log(*args: object, **_kwargs: object) -> None:
        logs.append(args)

    config_dir = tmp_path / "config"
    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    monkeypatch.setattr(cli, "log", _log)

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

    monkeypatch.setattr(cli, "LOCK_DIR", lock_dir)
    monkeypatch.setattr(cli, "RUNS_DIR", run_dir)
    monkeypatch.setattr(cli, "log", _log)
    monkeypatch.setattr(cli.typer, "confirm", lambda _msg: True)

    cli.clean()

    assert not list(lock_dir.glob("*.lock"))
    assert not list(run_dir.glob("*.json"))
    assert logs


def test_status_outputs_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cli = _reload_cli(monkeypatch)
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

    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    monkeypatch.setattr(cli, "LOCK_DIR", lock_dir)
    monkeypatch.setattr(cli, "RUNS_DIR", run_dir)

    cli.status()

    output = capsys.readouterr().out
    assert "mala status" in output
    assert "active locks" in output
    assert "run metadata files" in output


def test_run_disable_validations_valid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that valid --disable-validations values are accepted and passed to orchestrator."""
    cli = _reload_cli(monkeypatch)
    monkeypatch.setenv("MORPH_API_KEY", "test-key")

    config_dir = tmp_path / "config"
    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    monkeypatch.setattr(cli, "MalaOrchestrator", DummyOrchestrator)
    monkeypatch.setattr(cli, "set_verbose", lambda _: None)

    with pytest.raises(typer.Exit) as excinfo:
        cli.run(
            repo_path=tmp_path,
            disable_validations="coverage,slow-tests,e2e",
            verbose=False,
        )

    assert excinfo.value.exit_code == 0
    assert DummyOrchestrator.last_kwargs is not None
    assert DummyOrchestrator.last_kwargs["disable_validations"] == {
        "coverage",
        "slow-tests",
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

    with pytest.raises(typer.Exit) as excinfo:
        cli.run(
            repo_path=tmp_path,
            disable_validations="coverage,invalid-value,bad-option",
            verbose=False,
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

    with pytest.raises(typer.Exit) as excinfo:
        cli.run(
            repo_path=tmp_path,
            disable_validations=" , , ",
            verbose=False,
        )

    assert excinfo.value.exit_code == 1


def test_run_validation_flags_passed_to_orchestrator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that all validation flags are correctly passed to orchestrator."""
    cli = _reload_cli(monkeypatch)
    monkeypatch.setenv("MORPH_API_KEY", "test-key")

    config_dir = tmp_path / "config"
    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    monkeypatch.setattr(cli, "MalaOrchestrator", DummyOrchestrator)
    monkeypatch.setattr(cli, "set_verbose", lambda _: None)

    with pytest.raises(typer.Exit) as excinfo:
        cli.run(
            repo_path=tmp_path,
            disable_validations="post-validate",
            coverage_threshold=72.5,
            lint_only_for_docs=True,
            skip_e2e_if_no_keys=True,
            verbose=False,
        )

    assert excinfo.value.exit_code == 0
    assert DummyOrchestrator.last_kwargs is not None
    assert DummyOrchestrator.last_kwargs["disable_validations"] == {"post-validate"}
    assert DummyOrchestrator.last_kwargs["coverage_threshold"] == 72.5
    assert DummyOrchestrator.last_kwargs["lint_only_for_docs"] is True
    assert DummyOrchestrator.last_kwargs["skip_e2e_if_no_keys"] is True


def test_run_validation_flags_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that validation flags have correct defaults."""
    cli = _reload_cli(monkeypatch)
    monkeypatch.setenv("MORPH_API_KEY", "test-key")

    config_dir = tmp_path / "config"
    monkeypatch.setattr(cli, "USER_CONFIG_DIR", config_dir)
    monkeypatch.setattr(cli, "MalaOrchestrator", DummyOrchestrator)
    monkeypatch.setattr(cli, "set_verbose", lambda _: None)

    with pytest.raises(typer.Exit):
        cli.run(repo_path=tmp_path, verbose=False)

    assert DummyOrchestrator.last_kwargs is not None
    assert DummyOrchestrator.last_kwargs["disable_validations"] is None
    assert DummyOrchestrator.last_kwargs["coverage_threshold"] == 85.0
    assert DummyOrchestrator.last_kwargs["lint_only_for_docs"] is False
    assert DummyOrchestrator.last_kwargs["skip_e2e_if_no_keys"] is False


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

    with pytest.raises(typer.Exit) as excinfo:
        cli.run(
            repo_path=tmp_path,
            coverage_threshold=-10.0,
            verbose=False,
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

    with pytest.raises(typer.Exit) as excinfo:
        cli.run(
            repo_path=tmp_path,
            coverage_threshold=150.0,
            verbose=False,
        )

    assert excinfo.value.exit_code == 1
    error_msg = str(logs[-1])
    assert "Invalid --coverage-threshold" in error_msg
    assert "150.0" in error_msg
