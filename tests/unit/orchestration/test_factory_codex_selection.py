"""Integration-path tests: ``--coder codex`` reaches CodexAgentProvider stub.

Owns AC #1 (selection wiring), AC #3 (CLI > env > yaml > default for both
``coder`` and ``coder_options.codex.*``), AC #4 (configured options reach
the provider), and AC #13 (invalid yaml values rejected at config-parse
time).

Phase B does not yet drive a real Codex turn; instead, every test asserts
that the resolved ``MalaConfig`` flows through ``_create_agent_provider``
into a :class:`CodexAgentProvider` whose options match the configured
values. The downstream ``CodexNotInstalledError`` / ``CodexHookNotActiveError``
propagation is covered in :mod:`tests.unit.infra.clients.test_codex_provider`.
"""

import importlib
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import src.orchestration.factory
from src.domain.validation.config import ConfigError
from src.infra.clients.codex_provider import (
    CodexAgentProvider,
    CodexNotInstalledError,
)
from src.infra.io.config import CodexOptions, CoderOptions, MalaConfig
from src.orchestration.factory import _create_agent_provider


# ---------------------------------------------------------------------------
# Pure factory selection: MalaConfig -> CodexAgentProvider
# ---------------------------------------------------------------------------


def test_factory_returns_codex_provider_when_coder_codex() -> None:
    config = MalaConfig(
        runs_dir=Path("/tmp/runs"),
        lock_dir=Path("/tmp/locks"),
        coder="codex",
    )
    provider = _create_agent_provider(config)
    assert isinstance(provider, CodexAgentProvider)
    assert provider.name == "codex"
    assert provider.effort == "medium"


def test_factory_threads_resolved_codex_options_to_provider() -> None:
    """``MalaConfig.coder_options.codex`` flows into the provider instance.

    Integration-path evidence: the provider's ``options`` attribute is the
    same dataclass the resolver produced, so a model/effort that survived
    CLI > env > yaml > default reaches the runtime builder unchanged.
    """
    config = MalaConfig(
        runs_dir=Path("/tmp/runs"),
        lock_dir=Path("/tmp/locks"),
        coder="codex",
        coder_options=CoderOptions(
            codex=CodexOptions(
                model="gpt-5.5-foo",
                effort="medium",
                approval_policy="on-request",
                sandbox="workspace-write",
            ),
        ),
    )
    provider = _create_agent_provider(config)
    assert isinstance(provider, CodexAgentProvider)
    assert provider.model == "gpt-5.5-foo"
    assert provider.effort == "medium"
    assert provider.approval_policy == "on-request"
    assert provider.sandbox == "workspace-write"


def test_codex_provider_install_prerequisites_fails_closed_without_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end fail-closed: invoking the provider without the SDK raises.

    The integration-path test for AC #1 + AC #14: every CLI/env/yaml
    resolution that selects ``coder=codex`` reaches
    ``install_prerequisites`` (called once per run from
    ``create_orchestrator``), which raises
    :class:`CodexNotInstalledError` when ``codex_app_server`` is
    unimportable. The unit suite in
    ``tests/unit/infra/clients/test_codex_provider.py`` exercises every
    other fail-closed signature.
    """
    import importlib.util as _import_util

    monkeypatch.setattr(_import_util, "find_spec", lambda name: None)

    provider = CodexAgentProvider()

    def factory(
        agent_id: str,
        repo_path: Path,
        emit_lock_event: object,
    ) -> dict[str, object]:
        del agent_id, repo_path, emit_lock_event
        return {}

    with pytest.raises(CodexNotInstalledError, match="codex_app_server"):
        provider.install_prerequisites(Path("/tmp"), mcp_server_factory=factory)


# ---------------------------------------------------------------------------
# CLI integration path
# ---------------------------------------------------------------------------


def _reload_cli(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    if "src.cli.cli" in sys.modules:
        cli_mod = sys.modules["src.cli.cli"]
        cli_mod._bootstrapped = False  # type: ignore[attr-defined]  # ty:ignore[unresolved-attribute]
        cli_mod._lazy_modules.clear()  # type: ignore[attr-defined]
        return importlib.reload(cli_mod)
    return importlib.import_module("src.cli.cli")


class _DummyOrchestrator:
    last_mala_config: Any = None
    last_provider: Any = None

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
        mala_config: MalaConfig | None = None,
        deps: object = None,
    ) -> _DummyOrchestrator:
        _DummyOrchestrator.last_mala_config = mala_config
        # Drive the factory's selection path so we can prove the CLI input
        # reaches CodexAgentProvider with the configured options. Catch the
        # fail-closed install_prerequisites so the test asserts on the
        # provider rather than on the orchestrator construction failure.
        if mala_config is not None and mala_config.coder == "codex":
            _DummyOrchestrator.last_provider = _create_agent_provider(mala_config)
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


def test_cli_codex_selects_provider_with_default_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``mala run --coder codex`` reaches CodexAgentProvider with defaults."""
    from typer.testing import CliRunner

    _isolate_env(monkeypatch)
    cli = _reload_cli(monkeypatch)
    _patch_orchestrator(monkeypatch, cli, tmp_path)
    _DummyOrchestrator.last_provider = None

    runner = CliRunner()
    result = runner.invoke(cli.app, ["run", str(tmp_path), "--coder", "codex"])

    assert result.exit_code == 0, result.output
    provider = _DummyOrchestrator.last_provider
    assert isinstance(provider, CodexAgentProvider)
    assert provider.model == "gpt-5.5"
    assert provider.effort == "medium"
    assert provider.approval_policy == "never"
    assert provider.sandbox == "danger-full-access"


def test_yaml_codex_options_reach_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """yaml ``coder: codex`` + ``coder_options.codex.*`` reach the provider."""
    from typer.testing import CliRunner

    _isolate_env(monkeypatch)
    (tmp_path / "mala.yaml").write_text(
        "preset: python-uv\n"
        "coder: codex\n"
        "coder_options:\n"
        "  codex:\n"
        "    model: gpt-5.5-foo\n"
        "    approval_policy: on-request\n"
        "    sandbox: workspace-write\n",
    )

    cli = _reload_cli(monkeypatch)
    _patch_orchestrator(monkeypatch, cli, tmp_path)
    _DummyOrchestrator.last_provider = None

    runner = CliRunner()
    result = runner.invoke(cli.app, ["run", str(tmp_path)])

    assert result.exit_code == 0, result.output
    provider = _DummyOrchestrator.last_provider
    assert isinstance(provider, CodexAgentProvider)
    assert provider.model == "gpt-5.5-foo"
    assert provider.effort == "medium"
    assert provider.approval_policy == "on-request"
    assert provider.sandbox == "workspace-write"


def test_env_coder_codex_reaches_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from typer.testing import CliRunner

    _isolate_env(monkeypatch)
    monkeypatch.setenv("MALA_CODER", "codex")
    monkeypatch.setenv("MALA_CODEX_MODEL", "gpt-5.5-env")

    cli = _reload_cli(monkeypatch)
    _patch_orchestrator(monkeypatch, cli, tmp_path)
    _DummyOrchestrator.last_provider = None

    runner = CliRunner()
    result = runner.invoke(cli.app, ["run", str(tmp_path)])

    assert result.exit_code == 0, result.output
    provider = _DummyOrchestrator.last_provider
    assert isinstance(provider, CodexAgentProvider)
    assert provider.model == "gpt-5.5-env"


def test_invalid_codex_yaml_rejected_at_load_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Invalid ``coder_options.codex.sandbox`` fails before the orchestrator
    is constructed (AC #13 integration evidence)."""
    from typer.testing import CliRunner

    _isolate_env(monkeypatch)
    (tmp_path / "mala.yaml").write_text(
        "preset: python-uv\n"
        "coder: codex\n"
        "coder_options:\n"
        "  codex:\n"
        "    sandbox: yolo\n",
    )

    cli = _reload_cli(monkeypatch)
    _patch_orchestrator(monkeypatch, cli, tmp_path)

    runner = CliRunner()
    result = runner.invoke(cli.app, ["run", str(tmp_path)])

    assert result.exit_code != 0, result.output
    assert isinstance(result.exception, ConfigError)
    assert "sandbox" in str(result.exception)


def test_cli_overrides_yaml_codex_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """CLI > yaml: ``--codex-model`` beats yaml ``coder_options.codex.model``."""
    from typer.testing import CliRunner

    _isolate_env(monkeypatch)
    (tmp_path / "mala.yaml").write_text(
        "preset: python-uv\n"
        "coder: codex\n"
        "coder_options:\n"
        "  codex:\n"
        "    model: gpt-5.5-yaml\n",
    )

    cli = _reload_cli(monkeypatch)
    _patch_orchestrator(monkeypatch, cli, tmp_path)
    _DummyOrchestrator.last_provider = None

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["run", str(tmp_path), "--codex-model", "gpt-5.5-cli"],
    )

    assert result.exit_code == 0, result.output
    provider = _DummyOrchestrator.last_provider
    assert isinstance(provider, CodexAgentProvider)
    assert provider.model == "gpt-5.5-cli"
