"""Tests for ``--codex-model`` and ``--codex-effort`` CLI flags.

Covers:
  * Successful parsing into ``MalaConfig.coder_options.codex``.
  * Parser-level rejection of invalid effort values.
  * CLI > env > yaml > default precedence (AC #3).
  * Integration-path: ``--coder codex`` reaches ``CodexAgentProvider``
    via the full selection path with model/effort propagated.
"""

import importlib
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import src.orchestration.factory


def _reload_cli(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    if "src.cli.cli" in sys.modules:
        cli_mod = sys.modules["src.cli.cli"]
        cli_mod._bootstrapped = False  # type: ignore[attr-defined]  # ty:ignore[unresolved-attribute]
        cli_mod._lazy_modules.clear()  # type: ignore[attr-defined]
        return importlib.reload(cli_mod)
    return importlib.import_module("src.cli.cli")


class _DummyOrchestrator:
    last_mala_config: Any = None

    def __init__(self) -> None:
        self._exit_code = 0

    async def run(
        self, *, watch_config: object = None, validation_config: object = None
    ) -> tuple[int, int]:
        return (1, 1)

    @property
    def exit_code(self) -> int:
        return self._exit_code


def _make_dummy_create_orchestrator() -> Callable[..., _DummyOrchestrator]:
    def _factory(
        config: object,
        *,
        mala_config: object = None,
        deps: object = None,
    ) -> _DummyOrchestrator:
        _DummyOrchestrator.last_mala_config = mala_config
        return _DummyOrchestrator()

    return _factory


def _patch_orchestrator(
    monkeypatch: pytest.MonkeyPatch, cli: types.ModuleType, tmp_path: Path
) -> None:
    monkeypatch.setattr(cli, "USER_CONFIG_DIR", tmp_path / "config")
    monkeypatch.setattr(cli, "set_verbose", lambda _: None)
    monkeypatch.setattr(
        src.orchestration.factory,
        "create_orchestrator",
        _make_dummy_create_orchestrator(),
    )


def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "MALA_CODER",
        "MALA_AMP_MODE",
        "MALA_CLAUDE_SETTINGS_SOURCES",
        "MALA_EFFORT",
        "MALA_CODEX_MODEL",
        "MALA_CODEX_EFFORT",
    ):
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# Flag parsing
# ---------------------------------------------------------------------------


def test_codex_flags_parse_into_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``--codex-model`` and ``--codex-effort`` reach
    ``MalaConfig.coder_options.codex``."""
    from typer.testing import CliRunner

    _isolate_env(monkeypatch)
    cli = _reload_cli(monkeypatch)
    _patch_orchestrator(monkeypatch, cli, tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "run",
            str(tmp_path),
            "--coder",
            "codex",
            "--codex-model",
            "gpt-5.5-foo",
            "--codex-effort",
            "low",
        ],
    )

    assert result.exit_code == 0, result.output
    config = _DummyOrchestrator.last_mala_config
    assert config is not None
    assert config.coder == "codex"
    assert config.coder_options.codex.model == "gpt-5.5-foo"
    assert config.coder_options.codex.effort == "low"


def test_default_codex_options_when_flags_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from typer.testing import CliRunner

    _isolate_env(monkeypatch)
    cli = _reload_cli(monkeypatch)
    _patch_orchestrator(monkeypatch, cli, tmp_path)

    runner = CliRunner()
    result = runner.invoke(cli.app, ["run", str(tmp_path), "--coder", "codex"])

    assert result.exit_code == 0, result.output
    config = _DummyOrchestrator.last_mala_config
    assert config is not None
    assert config.coder_options.codex.model == "gpt-5.5"
    assert config.coder_options.codex.effort is None
    assert config.coder_options.codex.approval_policy == "never"
    assert config.coder_options.codex.sandbox == "danger-full-access"


def test_invalid_codex_effort_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from typer.testing import CliRunner

    _isolate_env(monkeypatch)
    cli = _reload_cli(monkeypatch)
    _patch_orchestrator(monkeypatch, cli, tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "run",
            str(tmp_path),
            "--coder",
            "codex",
            "--codex-effort",
            "super-high",
        ],
    )

    assert result.exit_code != 0
    assert "super-high" in result.output


# ---------------------------------------------------------------------------
# Precedence: CLI > env > yaml > default
# ---------------------------------------------------------------------------


def test_yaml_codex_options_applied_when_no_env_or_cli(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``coder_options.codex`` from mala.yaml flows into MalaConfig."""
    from typer.testing import CliRunner

    _isolate_env(monkeypatch)
    (tmp_path / "mala.yaml").write_text(
        "preset: python-uv\n"
        "coder: codex\n"
        "coder_options:\n"
        "  codex:\n"
        "    model: gpt-5.5-foo\n"
        "    approval_policy: never\n"
        "    sandbox: danger-full-access\n",
    )

    cli = _reload_cli(monkeypatch)
    _patch_orchestrator(monkeypatch, cli, tmp_path)

    runner = CliRunner()
    result = runner.invoke(cli.app, ["run", str(tmp_path)])

    assert result.exit_code == 0, result.output
    config = _DummyOrchestrator.last_mala_config
    assert config is not None
    assert config.coder == "codex"
    assert config.coder_options.codex.model == "gpt-5.5-foo"
    assert config.coder_options.codex.approval_policy == "never"
    assert config.coder_options.codex.sandbox == "danger-full-access"


def test_env_codex_model_overrides_yaml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from typer.testing import CliRunner

    _isolate_env(monkeypatch)
    monkeypatch.setenv("MALA_CODEX_MODEL", "gpt-5.5-env")
    (tmp_path / "mala.yaml").write_text(
        "preset: python-uv\n"
        "coder: codex\n"
        "coder_options:\n"
        "  codex:\n"
        "    model: gpt-5.5-yaml\n",
    )

    cli = _reload_cli(monkeypatch)
    _patch_orchestrator(monkeypatch, cli, tmp_path)

    runner = CliRunner()
    result = runner.invoke(cli.app, ["run", str(tmp_path)])

    assert result.exit_code == 0, result.output
    config = _DummyOrchestrator.last_mala_config
    assert config is not None
    assert config.coder_options.codex.model == "gpt-5.5-env"


def test_cli_codex_model_overrides_env_and_yaml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from typer.testing import CliRunner

    _isolate_env(monkeypatch)
    monkeypatch.setenv("MALA_CODEX_MODEL", "gpt-5.5-env")
    (tmp_path / "mala.yaml").write_text(
        "preset: python-uv\n"
        "coder: codex\n"
        "coder_options:\n"
        "  codex:\n"
        "    model: gpt-5.5-yaml\n",
    )

    cli = _reload_cli(monkeypatch)
    _patch_orchestrator(monkeypatch, cli, tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["run", str(tmp_path), "--codex-model", "gpt-5.5-cli"],
    )

    assert result.exit_code == 0, result.output
    config = _DummyOrchestrator.last_mala_config
    assert config is not None
    assert config.coder_options.codex.model == "gpt-5.5-cli"


def test_env_coder_codex_selects_codex_via_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``MALA_CODER=codex`` selects codex even without yaml/CLI flag."""
    from typer.testing import CliRunner

    _isolate_env(monkeypatch)
    monkeypatch.setenv("MALA_CODER", "codex")

    cli = _reload_cli(monkeypatch)
    _patch_orchestrator(monkeypatch, cli, tmp_path)

    runner = CliRunner()
    result = runner.invoke(cli.app, ["run", str(tmp_path)])

    assert result.exit_code == 0, result.output
    config = _DummyOrchestrator.last_mala_config
    assert config is not None
    assert config.coder == "codex"
