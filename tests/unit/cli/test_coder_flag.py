"""Tests for --coder and --amp-mode CLI flags on `mala run`.

Covers:
    - Successful parsing into MalaConfig.
    - Parser-level rejection of invalid values.
    - Independence: omitting the flag does not perturb existing config loading.
    - Cross-coder ignored-flag info logging (parity with plan L686-L690).
"""

import importlib
import logging
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import src.orchestration.factory


def _reload_cli(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Reload src.cli.cli so each test starts with a fresh lazy-module cache."""
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
    """Strip env vars that would leak the user's defaults into the CLI run."""
    for name in ("MALA_CODER", "MALA_AMP_MODE", "MALA_CLAUDE_SETTINGS_SOURCES"):
        monkeypatch.delenv(name, raising=False)


def test_coder_amp_parses_into_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`--coder amp` is accepted and lands in MalaConfig.coder."""
    from typer.testing import CliRunner

    _isolate_env(monkeypatch)
    cli = _reload_cli(monkeypatch)
    _patch_orchestrator(monkeypatch, cli, tmp_path)

    runner = CliRunner()
    result = runner.invoke(cli.app, ["run", str(tmp_path), "--coder", "amp"])

    assert result.exit_code == 0
    config = _DummyOrchestrator.last_mala_config
    assert config is not None
    assert config.coder == "amp"


def test_amp_mode_rush_parses_into_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`--amp-mode rush` is accepted and lands in coder_options.amp.mode."""
    from typer.testing import CliRunner

    _isolate_env(monkeypatch)
    cli = _reload_cli(monkeypatch)
    _patch_orchestrator(monkeypatch, cli, tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["run", str(tmp_path), "--coder", "amp", "--amp-mode", "rush"],
    )

    assert result.exit_code == 0
    config = _DummyOrchestrator.last_mala_config
    assert config is not None
    assert config.coder == "amp"
    assert config.coder_options.amp.mode == "rush"


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--coder", "gpt5"),
        ("--amp-mode", "blast"),
    ],
)
def test_invalid_value_rejected_at_parse_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, flag: str, value: str
) -> None:
    """Bogus values for --coder / --amp-mode exit non-zero from the parser."""
    from typer.testing import CliRunner

    _isolate_env(monkeypatch)
    cli = _reload_cli(monkeypatch)
    _patch_orchestrator(monkeypatch, cli, tmp_path)

    runner = CliRunner()
    result = runner.invoke(cli.app, ["run", str(tmp_path), flag, value])

    assert result.exit_code != 0
    # Parser-level error mentions the offending value (Click format).
    assert value in result.output


def test_absence_of_flags_preserves_default_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No flags → MalaConfig retains env/yaml/default coder + amp mode."""
    from typer.testing import CliRunner

    _isolate_env(monkeypatch)
    cli = _reload_cli(monkeypatch)
    _patch_orchestrator(monkeypatch, cli, tmp_path)

    runner = CliRunner()
    result = runner.invoke(cli.app, ["run", str(tmp_path)])

    assert result.exit_code == 0
    config = _DummyOrchestrator.last_mala_config
    assert config is not None
    # Defaults (no env, no yaml, no CLI override).
    assert config.coder == "claude"
    assert config.coder_options.amp.mode == "smart"


def test_amp_mode_with_coder_claude_logs_ignored(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """--amp-mode rush with --coder claude → info-level "ignored" log."""
    from typer.testing import CliRunner

    _isolate_env(monkeypatch)
    cli = _reload_cli(monkeypatch)
    _patch_orchestrator(monkeypatch, cli, tmp_path)

    runner = CliRunner()
    with caplog.at_level(logging.INFO, logger="src.infra.io.config"):
        result = runner.invoke(
            cli.app,
            [
                "run",
                str(tmp_path),
                "--coder",
                "claude",
                "--amp-mode",
                "rush",
            ],
        )

    assert result.exit_code == 0
    messages = [rec.getMessage() for rec in caplog.records]
    assert any(
        "amp mode" in msg and "ignored" in msg and "claude" in msg for msg in messages
    ), f"expected cross-coder ignore log, got: {messages}"


def test_claude_settings_sources_with_coder_amp_logs_ignored(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """--claude-settings-sources with --coder amp → info-level "ignored" log."""
    from typer.testing import CliRunner

    _isolate_env(monkeypatch)
    cli = _reload_cli(monkeypatch)
    _patch_orchestrator(monkeypatch, cli, tmp_path)

    runner = CliRunner()
    with caplog.at_level(logging.INFO, logger="src.infra.io.config"):
        result = runner.invoke(
            cli.app,
            [
                "run",
                str(tmp_path),
                "--coder",
                "amp",
                "--claude-settings-sources",
                "user",
            ],
        )

    assert result.exit_code == 0
    messages = [rec.getMessage() for rec in caplog.records]
    assert any(
        "claude_settings_sources" in msg and "ignored" in msg and "amp" in msg
        for msg in messages
    ), f"expected cross-coder ignore log, got: {messages}"
