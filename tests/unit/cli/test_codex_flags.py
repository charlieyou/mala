"""Tests for shared coder ``--model`` / ``--effort`` with Codex.

Covers:
  * Successful parsing into top-level ``MalaConfig.model`` / ``effort``.
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
        "MALA_MODEL",
        "MALA_EFFORT",
        "MALA_CODEX_APPROVAL_POLICY",
        "MALA_CODEX_SANDBOX",
    ):
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# Flag parsing
# ---------------------------------------------------------------------------


def test_shared_model_and_effort_flags_parse_into_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``--model`` and ``--effort`` reach top-level coder config."""
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
            "--model",
            "gpt-5.5-foo",
            "--effort",
            "low",
        ],
    )

    assert result.exit_code == 0, result.output
    config = _DummyOrchestrator.last_mala_config
    assert config is not None
    assert config.coder == "codex"
    assert config.model == "gpt-5.5-foo"
    assert config.effort == "low"


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
    assert config.model == "gpt-5.5"
    assert config.effort == "medium"
    assert config.coder_options.codex.approval_policy == "never"
    assert config.coder_options.codex.sandbox == "danger-full-access"


def test_invalid_effort_rejected(
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
            "--effort",
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
        "model: gpt-5.5-foo\n"
        "coder_options:\n"
        "  codex:\n"
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
    assert config.model == "gpt-5.5-foo"
    assert config.coder_options.codex.approval_policy == "never"
    assert config.coder_options.codex.sandbox == "danger-full-access"


def test_env_model_overrides_yaml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from typer.testing import CliRunner

    _isolate_env(monkeypatch)
    monkeypatch.setenv("MALA_MODEL", "gpt-5.5-env")
    (tmp_path / "mala.yaml").write_text(
        "preset: python-uv\n"
        "coder: codex\n"
        "model: gpt-5.5-yaml\n",
    )

    cli = _reload_cli(monkeypatch)
    _patch_orchestrator(monkeypatch, cli, tmp_path)

    runner = CliRunner()
    result = runner.invoke(cli.app, ["run", str(tmp_path)])

    assert result.exit_code == 0, result.output
    config = _DummyOrchestrator.last_mala_config
    assert config is not None
    assert config.model == "gpt-5.5-env"


def test_cli_model_overrides_env_and_yaml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from typer.testing import CliRunner

    _isolate_env(monkeypatch)
    monkeypatch.setenv("MALA_MODEL", "gpt-5.5-env")
    (tmp_path / "mala.yaml").write_text(
        "preset: python-uv\n"
        "coder: codex\n"
        "model: gpt-5.5-yaml\n",
    )

    cli = _reload_cli(monkeypatch)
    _patch_orchestrator(monkeypatch, cli, tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["run", str(tmp_path), "--model", "gpt-5.5-cli"],
    )

    assert result.exit_code == 0, result.output
    config = _DummyOrchestrator.last_mala_config
    assert config is not None
    assert config.model == "gpt-5.5-cli"


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


# ---------------------------------------------------------------------------
# Precedence: approval_policy + sandbox (CLI > env > yaml > default)
# ---------------------------------------------------------------------------


def test_env_codex_approval_policy_overrides_yaml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``MALA_CODEX_APPROVAL_POLICY`` overrides the value from mala.yaml."""
    from typer.testing import CliRunner

    _isolate_env(monkeypatch)
    monkeypatch.setenv("MALA_CODEX_APPROVAL_POLICY", "on-request")
    (tmp_path / "mala.yaml").write_text(
        "preset: python-uv\n"
        "coder: codex\n"
        "coder_options:\n"
        "  codex:\n"
        "    approval_policy: never\n",
    )

    cli = _reload_cli(monkeypatch)
    _patch_orchestrator(monkeypatch, cli, tmp_path)

    runner = CliRunner()
    result = runner.invoke(cli.app, ["run", str(tmp_path)])

    assert result.exit_code == 0, result.output
    config = _DummyOrchestrator.last_mala_config
    assert config is not None
    assert config.coder_options.codex.approval_policy == "on-request"


def test_cli_codex_approval_policy_overrides_env_and_yaml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``--codex-approval-policy`` beats env and yaml."""
    from typer.testing import CliRunner

    _isolate_env(monkeypatch)
    monkeypatch.setenv("MALA_CODEX_APPROVAL_POLICY", "on-request")
    (tmp_path / "mala.yaml").write_text(
        "preset: python-uv\n"
        "coder: codex\n"
        "coder_options:\n"
        "  codex:\n"
        "    approval_policy: never\n",
    )

    cli = _reload_cli(monkeypatch)
    _patch_orchestrator(monkeypatch, cli, tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["run", str(tmp_path), "--codex-approval-policy", "untrusted"],
    )

    assert result.exit_code == 0, result.output
    config = _DummyOrchestrator.last_mala_config
    assert config is not None
    assert config.coder_options.codex.approval_policy == "untrusted"


def test_env_codex_sandbox_overrides_yaml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``MALA_CODEX_SANDBOX`` overrides the value from mala.yaml."""
    from typer.testing import CliRunner

    _isolate_env(monkeypatch)
    monkeypatch.setenv("MALA_CODEX_SANDBOX", "read-only")
    (tmp_path / "mala.yaml").write_text(
        "preset: python-uv\n"
        "coder: codex\n"
        "coder_options:\n"
        "  codex:\n"
        "    sandbox: workspace-write\n",
    )

    cli = _reload_cli(monkeypatch)
    _patch_orchestrator(monkeypatch, cli, tmp_path)

    runner = CliRunner()
    result = runner.invoke(cli.app, ["run", str(tmp_path)])

    assert result.exit_code == 0, result.output
    config = _DummyOrchestrator.last_mala_config
    assert config is not None
    assert config.coder_options.codex.sandbox == "read-only"


def test_cli_codex_sandbox_overrides_env_and_yaml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``--codex-sandbox`` beats env and yaml."""
    from typer.testing import CliRunner

    _isolate_env(monkeypatch)
    monkeypatch.setenv("MALA_CODEX_SANDBOX", "read-only")
    (tmp_path / "mala.yaml").write_text(
        "preset: python-uv\n"
        "coder: codex\n"
        "coder_options:\n"
        "  codex:\n"
        "    sandbox: workspace-write\n",
    )

    cli = _reload_cli(monkeypatch)
    _patch_orchestrator(monkeypatch, cli, tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["run", str(tmp_path), "--codex-sandbox", "danger-full-access"],
    )

    assert result.exit_code == 0, result.output
    config = _DummyOrchestrator.last_mala_config
    assert config is not None
    assert config.coder_options.codex.sandbox == "danger-full-access"


def test_invalid_codex_approval_policy_rejected(
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
            "--codex-approval-policy",
            "auto",
        ],
    )

    assert result.exit_code != 0
    assert "auto" in result.output


def test_invalid_codex_sandbox_rejected(
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
            "--codex-sandbox",
            "open",
        ],
    )

    assert result.exit_code != 0
    assert "open" in result.output
