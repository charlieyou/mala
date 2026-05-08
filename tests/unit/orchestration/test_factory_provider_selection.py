"""Factory-level AgentProvider selection (T007).

These tests are the unit-level evidence for AC#1: the default ``coder=amp``
constructs an :class:`AmpAgentProvider`, while ``coder=claude`` selects
:class:`ClaudeAgentProvider`. The integration evidence lives in
``tests/integration/test_amp_provider.py``.

The factory pulls ``coder`` and ``coder_options.amp.mode`` from
:class:`MalaConfig`; CLI/env precedence is verified separately in
``tests/unit/infra/io/test_coder_config.py``. These tests pin the *wiring*:
once ``mala_config.coder`` is set, the right provider lands in
``OrchestratorDependencies.agent_provider`` and ``install_prerequisites``
runs once before any session.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.infra.clients.amp_provider import AmpAgentProvider
from src.infra.clients.claude_provider import ClaudeAgentProvider
from src.infra.io.config import AmpOptions, CoderOptions, MalaConfig
from src.orchestration.factory import (
    OrchestratorConfig,
    OrchestratorDependencies,
    _create_agent_provider,
    create_orchestrator,
)
from tests.fakes.issue_provider import FakeIssueProvider


def _wire_fake_amp_self_test(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Provision a fake ``amp`` binary that satisfies the T013 self-test.

    The post-T013 ``AmpAgentProvider.install_prerequisites`` runs the
    plugin-load self-test (spawns ``amp`` and waits for the sentinel
    marker on stderr). For factory-selection unit tests we need a fake
    binary on PATH that emits the matching sentinel so the orchestrator
    can be constructed; the *behavior* of the self-test is covered in
    ``tests/unit/infra/clients/test_amp_plugin_self_test.py``.
    """
    from src.infra.clients import amp_plugin_installer
    from src.infra.clients.amp_plugin_installer import AmpPluginInstaller

    fake_home = tmp_path / "_amp_home"
    fake_home.mkdir(exist_ok=True)
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    monkeypatch.setattr(
        amp_plugin_installer,
        "DEFAULT_PLUGIN_DIR",
        fake_home / ".config" / "amp" / "plugins",
    )

    bin_dir = tmp_path / "_amp_bin"
    bin_dir.mkdir(exist_ok=True)
    plugin_hash = AmpPluginInstaller().installed_plugin_hash()
    fake_amp = bin_dir / "amp"
    fake_amp.write_text(
        "#!/usr/bin/env bash\n"
        f'echo \'{{"mala_plugin":"loaded","version":"{plugin_hash}"}}\' >&2\n'
        "sleep 5\n"
    )
    fake_amp.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")


# ---------------------------------------------------------------------------
# _create_agent_provider: pure selection given MalaConfig.coder
# ---------------------------------------------------------------------------


def test_create_agent_provider_picks_amp_by_default() -> None:
    config = MalaConfig.from_env(validate=False)
    assert config.coder == "amp"
    provider = _create_agent_provider(config)
    assert isinstance(provider, AmpAgentProvider)
    assert provider.name == "amp"


def test_create_agent_provider_picks_claude_when_configured() -> None:
    config = MalaConfig.from_env(validate=False, yaml_coder="claude")
    assert config.coder == "claude"
    provider = _create_agent_provider(config)
    assert isinstance(provider, ClaudeAgentProvider)
    assert provider.name == "claude"


def test_create_agent_provider_picks_amp_when_configured() -> None:
    config = MalaConfig(
        runs_dir=Path("/tmp/runs"),
        lock_dir=Path("/tmp/locks"),
        coder="amp",
    )
    provider = _create_agent_provider(config)
    assert isinstance(provider, AmpAgentProvider)
    assert provider.name == "amp"


def test_create_agent_provider_threads_amp_mode() -> None:
    """``coder_options.amp.mode`` reaches the AmpAgentProvider's runtime
    builder, so ``--mode <smart|rush|deep>`` lands on the spawned argv."""
    config = MalaConfig(
        runs_dir=Path("/tmp/runs"),
        lock_dir=Path("/tmp/locks"),
        coder="amp",
        coder_options=CoderOptions(amp=AmpOptions(mode="rush")),
    )
    provider = _create_agent_provider(config)
    assert isinstance(provider, AmpAgentProvider)

    def _factory(
        agent_id: str, repo_path: Path, emit_lock_event: object
    ) -> dict[str, object]:
        del agent_id, repo_path, emit_lock_event
        return {}

    builder = provider.runtime_builder(
        Path("/tmp/repo"),
        "agent-x",
        mcp_server_factory=_factory,
    )
    runtime = builder.build()  # type: ignore[attr-defined]
    assert runtime.mode == "rush"  # ty:ignore[unresolved-attribute]
    assert "--mode" in runtime.argv  # ty:ignore[unresolved-attribute]
    assert "rush" in runtime.argv  # ty:ignore[unresolved-attribute]


def test_create_agent_provider_rejects_codex_until_phase_b() -> None:
    """A8 widens the ``coder`` enum to include ``"codex"`` but does not yet
    wire the Codex provider (Phase B). Constructing :class:`MalaConfig` with
    ``coder="codex"`` succeeds; resolving an :class:`AgentProvider` for it
    raises :class:`NotImplementedError` so the gap is observable."""
    config = MalaConfig(
        runs_dir=Path("/tmp/runs"),
        lock_dir=Path("/tmp/locks"),
        coder="codex",
    )
    assert config.coder == "codex"
    with pytest.raises(NotImplementedError, match="Codex"):
        _create_agent_provider(config)


def test_create_agent_provider_threads_claude_settings_sources() -> None:
    """Claude provider receives the configured settings sources so the run's
    ``claude_settings_sources`` precedence (env > yaml > default) is honored."""
    config = MalaConfig.from_env(
        validate=False,
        yaml_coder="claude",
        yaml_claude_settings_sources=("local", "user"),
    )
    provider = _create_agent_provider(config)
    assert isinstance(provider, ClaudeAgentProvider)
    assert provider._setting_sources == ["local", "user"]


# ---------------------------------------------------------------------------
# create_orchestrator wires the chosen provider into deps + invokes
# install_prerequisites exactly once. AC#1 wiring evidence.
# ---------------------------------------------------------------------------


def test_orchestrator_uses_amp_provider_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _wire_fake_amp_self_test(tmp_path, monkeypatch)
    config = OrchestratorConfig(repo_path=tmp_path, max_agents=1)
    deps = OrchestratorDependencies(
        issue_provider=FakeIssueProvider(),
    )
    mala_config = MalaConfig.from_env(validate=False)
    orchestrator = create_orchestrator(config, mala_config=mala_config, deps=deps)

    # The provider stored on the orchestrator is the Amp one.
    assert isinstance(orchestrator._agent_provider, AmpAgentProvider)
    # And the same instance is threaded through to the run coordinator's
    # FixerService — fixers therefore follow the main coder.
    assert (
        orchestrator.run_coordinator.fixer_service._agent_provider
        is orchestrator._agent_provider
    )


def test_orchestrator_uses_amp_provider_when_coder_amp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _wire_fake_amp_self_test(tmp_path, monkeypatch)
    config = OrchestratorConfig(repo_path=tmp_path, max_agents=1)
    deps = OrchestratorDependencies(
        issue_provider=FakeIssueProvider(),
    )
    mala_config = MalaConfig(
        runs_dir=tmp_path / "runs",
        lock_dir=tmp_path / "locks",
        coder="amp",
    )
    orchestrator = create_orchestrator(config, mala_config=mala_config, deps=deps)

    assert isinstance(orchestrator._agent_provider, AmpAgentProvider)
    # FixerService spawned with the same provider — AC#5 wiring evidence.
    # The full Amp impl is pending; the assertion here is structural so the
    # symmetry is checked even before T013 turns the integration test green.
    assert (
        orchestrator.run_coordinator.fixer_service._agent_provider
        is orchestrator._agent_provider
    )


def test_install_prerequisites_invoked_once_per_run(tmp_path: Path) -> None:
    """``AgentProvider.install_prerequisites`` runs exactly once during
    ``create_orchestrator`` so the Amp self-test (T013) does not pay LLM
    cost per session, and Claude's no-op stays a no-op."""
    from tests.fakes.agent_provider import FakeAgentProvider
    from tests.fakes.sdk_client import FakeSDKClientFactory

    factory = FakeSDKClientFactory()
    provider = FakeAgentProvider(factory)
    assert provider.install_prerequisites_count == 0

    config = OrchestratorConfig(repo_path=tmp_path, max_agents=1)
    deps = OrchestratorDependencies(
        issue_provider=FakeIssueProvider(),
        agent_provider=provider,
    )
    mala_config = MalaConfig.from_env(validate=False)
    create_orchestrator(config, mala_config=mala_config, deps=deps)

    assert provider.install_prerequisites_count == 1


def test_dependencies_agent_provider_overrides_factory_selection(
    tmp_path: Path,
) -> None:
    """If a test explicitly injects an ``agent_provider`` into
    ``OrchestratorDependencies``, the factory does NOT also pick its own
    based on ``mala_config.coder`` — the injected one wins. This is the
    standard DI override pattern used by every other dep."""
    from tests.fakes.agent_provider import FakeAgentProvider
    from tests.fakes.sdk_client import FakeSDKClientFactory

    factory = FakeSDKClientFactory()
    injected = FakeAgentProvider(factory, name="claude")

    # Even with coder=amp in MalaConfig, the injected provider wins.
    mala_config = MalaConfig(
        runs_dir=tmp_path / "runs",
        lock_dir=tmp_path / "locks",
        coder="amp",
    )
    config = OrchestratorConfig(repo_path=tmp_path, max_agents=1)
    deps = OrchestratorDependencies(
        issue_provider=FakeIssueProvider(),
        agent_provider=injected,
    )
    orchestrator = create_orchestrator(config, mala_config=mala_config, deps=deps)
    assert orchestrator._agent_provider is injected


@pytest.mark.parametrize("mode", ["smart", "rush", "deep"])
def test_amp_modes_propagate_through_factory(
    tmp_path: Path, mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All three Amp modes flow through factory selection into the
    AmpAgentProvider's runtime builder."""
    _wire_fake_amp_self_test(tmp_path, monkeypatch)
    config = OrchestratorConfig(repo_path=tmp_path, max_agents=1)
    deps = OrchestratorDependencies(
        issue_provider=FakeIssueProvider(),
    )
    mala_config = MalaConfig(
        runs_dir=tmp_path / "runs",
        lock_dir=tmp_path / "locks",
        coder="amp",
        coder_options=CoderOptions(amp=AmpOptions(mode=mode)),  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
    )
    orchestrator = create_orchestrator(config, mala_config=mala_config, deps=deps)
    provider = orchestrator._agent_provider
    assert isinstance(provider, AmpAgentProvider)
    assert provider._mode == mode
