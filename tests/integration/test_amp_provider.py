"""End-to-end integration test for the Amp provider's install gate (T013).

The orchestrator's ``install_prerequisites`` hook is the safety-critical
gate documented at ``plans/2026-04-29-amp-provider-plan.md#L168-L171,
L271-L321``. This test wires :class:`AmpAgentProvider` through
``create_orchestrator`` (the same path real ``mala run --coder amp``
takes) and asserts:

  * **Pass case** — fake ``amp`` on PATH emits the sentinel marker on
    stderr; ``create_orchestrator`` returns successfully and the
    orchestrator carries an :class:`AmpAgentProvider`.
  * **Fail-closed case** — fake ``amp`` emits nothing;
    ``create_orchestrator`` raises :class:`AmpPluginNotActiveError`
    (``PLUGIN_MARKER_MISSING``) before any issue agent is spawned. This
    is the AC#17 safety contract under ``--dangerously-allow-all``.

The full per-issue lifecycle (``run_implementer`` end-to-end through the
fake amp's stream-json transcript) is deferred to a follow-up: the
:class:`AgentSessionRunner` pipeline still relies on Claude-specific
fluent runtime methods (``with_hooks``, ``with_env``, ``with_mcp``,
``with_disallowed_tools``, ``with_lint_tools``) that
:class:`AmpRuntimeBuilder` does not expose, so a coder-agnostic pipeline
refactor is required before ``run_implementer`` can drive an Amp session
end-to-end. Tracked as a follow-up; T013 ships the install_prerequisites
self-test gate (AC#16, AC#17) which is the unique safety-critical
addition over the T007 stub.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from src.infra.clients.amp_plugin_installer import AmpPluginInstaller
from src.infra.clients.amp_provider import (
    AmpAgentProvider,
    AmpPluginNotActiveError,
    AmpPluginNotActiveReason,
)
from src.infra.io.config import MalaConfig
from src.orchestration.factory import (
    OrchestratorConfig,
    OrchestratorDependencies,
    create_orchestrator,
)
from tests.fakes.issue_provider import FakeIssue, FakeIssueProvider

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fake-amp helpers
# ---------------------------------------------------------------------------


def _install_fake_amp(
    bin_dir: Path,
    *,
    body: str,
) -> Path:
    """Install a fake ``amp`` executable into ``bin_dir`` with ``body``."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    fake_amp = bin_dir / "amp"
    fake_amp.write_text(body)
    fake_amp.chmod(0o755)
    return fake_amp


def _fake_amp_emitting_sentinel(plugin_hash: str) -> str:
    """Fake amp that emits the sentinel marker and stays alive briefly.

    The orchestrator terminates ``amp`` as soon as it sees the marker
    (plan ``L309-L312``), so the long ``sleep`` is just there to mimic
    the real Amp staying alive past ``session.start``.
    """
    return (
        "#!/usr/bin/env bash\n"
        f'echo \'{{"mala_plugin":"loaded","version":"{plugin_hash}"}}\' >&2\n'
        "sleep 5\n"
    )


_FAKE_AMP_NO_PLUGIN = (
    "#!/usr/bin/env bash\n"
    "# Fake amp emitting nothing (plugin appears not to be loaded).\n"
    "exit 0\n"
)


@pytest.fixture
def amp_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    """Wire up a clean Amp environment under ``tmp_path``.

    Returns ``(bin_dir, repo_path)`` so individual tests can drop the
    fake binary in and run the orchestrator against the right repo.
    Plugin install dir is redirected to a fresh ``tmp_path`` location so
    the test never touches ``~/.config/amp/plugins/``.
    """
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr("pathlib.Path.home", lambda: fake_home)

    from src.infra.clients import amp_plugin_installer

    monkeypatch.setattr(
        amp_plugin_installer,
        "DEFAULT_PLUGIN_DIR",
        fake_home / ".config" / "amp" / "plugins",
    )

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    monkeypatch.setenv(
        "PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '/usr/bin:/bin')}"
    )
    monkeypatch.setenv("AMP_API_KEY", "fake-test-key")
    monkeypatch.delenv("MALA_DISALLOWED_TOOLS", raising=False)

    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    return bin_dir, repo_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_create_orchestrator_amp_with_active_plugin(
    amp_environment: tuple[Path, Path],
) -> None:
    """Pass case: sentinel marker → orchestrator constructed with Amp.

    Exercises the full ``create_orchestrator`` path with ``coder=amp``.
    The orchestrator's call to
    ``agent_provider.install_prerequisites(...)`` must succeed before any
    issue agent is spawned (plan ``L168``); a successful return means the
    plugin installer ran, the runtime self-test fired against the fake
    amp, the sentinel marker matched the installed plugin's hash, and
    the result was cached for the rest of the run.
    """
    bin_dir, repo_path = amp_environment
    plugin_hash = AmpPluginInstaller().installed_plugin_hash()
    _install_fake_amp(bin_dir, body=_fake_amp_emitting_sentinel(plugin_hash))

    issue = FakeIssue(
        id="bd-amp-fake.1",
        title="Fake-amp install-prerequisites smoke",
        description="Should exercise install_prerequisites end-to-end.",
    )
    issue_provider = FakeIssueProvider(issues={issue.id: issue})

    config = OrchestratorConfig(
        repo_path=repo_path,
        max_agents=1,
        max_issues=1,
        timeout_minutes=1,
    )
    mala_config = MalaConfig(
        runs_dir=repo_path / "runs",
        lock_dir=repo_path / "locks",
        coder="amp",
    )
    deps = OrchestratorDependencies(issue_provider=issue_provider)

    orchestrator = create_orchestrator(config, mala_config=mala_config, deps=deps)

    assert isinstance(orchestrator._agent_provider, AmpAgentProvider)
    assert orchestrator._agent_provider.name == "amp"


@pytest.mark.integration
def test_create_orchestrator_amp_fails_closed_when_plugin_inactive(
    amp_environment: tuple[Path, Path],
) -> None:
    """Fail-closed case: no sentinel → orchestrator construction aborts.

    AC#17 contract: under ``--dangerously-allow-all``, the orchestrator
    refuses to spawn any issue agent if the safety plugin is not active
    at runtime. Hash-only verification is explicitly insufficient
    (plan ``L171, L612-L617``); the runtime self-test is the gate.
    """
    bin_dir, repo_path = amp_environment
    _install_fake_amp(bin_dir, body=_FAKE_AMP_NO_PLUGIN)

    issue = FakeIssue(
        id="bd-amp-fake.2",
        title="Fake-amp inactive plugin",
        description="Plugin missing — must fail closed.",
    )
    issue_provider = FakeIssueProvider(issues={issue.id: issue})

    config = OrchestratorConfig(
        repo_path=repo_path,
        max_agents=1,
        max_issues=1,
        timeout_minutes=1,
    )
    mala_config = MalaConfig(
        runs_dir=repo_path / "runs",
        lock_dir=repo_path / "locks",
        coder="amp",
    )
    deps = OrchestratorDependencies(issue_provider=issue_provider)

    with pytest.raises(AmpPluginNotActiveError) as excinfo:
        create_orchestrator(config, mala_config=mala_config, deps=deps)
    assert excinfo.value.reason is AmpPluginNotActiveReason.PLUGIN_MARKER_MISSING
