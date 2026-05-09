"""End-to-end integration test for the Amp provider (T013).

Exercises three safety-critical / AC-level paths through
``create_orchestrator`` and ``run_implementer`` against a fake ``amp``
binary on PATH:

  * **Pass case (install_prerequisites)** — fake ``amp`` emits the
    sentinel marker on stderr; ``create_orchestrator`` returns
    successfully and the orchestrator carries an :class:`AmpAgentProvider`
    (plan ``L168-L171, L271-L321``).
  * **Fail-closed case (install_prerequisites)** — fake ``amp`` emits
    nothing; ``create_orchestrator`` raises
    :class:`AmpPluginNotActiveError` (``PLUGIN_MARKER_MISSING``) before
    any issue agent is spawned. AC#17 safety contract under
    ``--dangerously-allow-all``.
  * **Per-issue lifecycle (AC#6)** — fake ``amp`` emits the sentinel
    marker on stderr **and** a canned stream-json transcript on stdout;
    ``run_implementer`` drives ``AgentSessionRunner._build_session`` →
    ``client_factory.create(runtime)`` → ``query`` → message
    parsing → lifecycle → ``IssueResult`` end-to-end. Until T013 grew
    the fluent surface on :class:`AmpRuntimeBuilder` and added
    ``lint_cache`` on :class:`AmpRuntime`, this path raised
    ``AttributeError: 'AmpRuntimeBuilder' object has no attribute
    fluent surface before any Amp client could run. The test pins that
    regression: a passing assertion here means the Claude pipeline can
    consume the Amp runtime through the same fluent chain.
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


def _fake_amp_dual_role(plugin_hash: str, thread_id: str) -> str:
    """Fake amp serving both ``install_prerequisites`` and a real session.

    The same binary handles both roles because the orchestrator spawns
    the same ``amp --execute`` argv each time:

      * ``install_prerequisites`` only inspects stderr for the sentinel
        marker and terminates ``amp`` on detection (plan ``L309-L312``);
        anything we emit on stdout is ignored.
      * ``run_implementer`` (the real session) reads the canned
        stream-json from stdout to drive the message-stream processor
        through to a ``result(success)`` event.

    By emitting *both* the sentinel on stderr and a complete stream-json
    transcript on stdout, the same script is correct for both paths.
    """
    return (
        "#!/usr/bin/env bash\n"
        "# Sentinel marker — drives install_prerequisites' self-test.\n"
        f'echo \'{{"mala_plugin":"loaded","version":"{plugin_hash}"}}\' >&2\n'
        "# Drain stdin so the orchestrator's prompt write doesn't EPIPE.\n"
        "cat >/dev/null &\n"
        "# Canned stream-json transcript — drives the session pipeline.\n"
        f'echo \'{{"type":"system","subtype":"init","session_id":"{thread_id}"}}\'\n'
        'echo \'{"type":"assistant","message":{"content":[{"type":"text",'
        '"text":"Implementation complete."}]}}\'\n'
        f'echo \'{{"type":"result","subtype":"success","session_id":"{thread_id}",'
        '"result":"success"}}\'\n'
        "wait\n"
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
@pytest.mark.asyncio
async def test_run_implementer_drives_per_issue_lifecycle_through_amp_pipeline(
    amp_environment: tuple[Path, Path],
) -> None:
    """AC#6 regression: ``run_implementer`` can drive an Amp session.

    Until T013 grew :class:`AmpRuntimeBuilder` the cross-coder fluent
    surface the pipeline calls (``with_agent_timeout`` / ``with_env`` /
    ``with_mcp`` / ``with_lint_tools``) and exposed
    ``options``/``lint_cache`` on :class:`AmpRuntime`,
    ``AgentSessionRunner._build_session`` raised ``AttributeError`` on
    the very first Amp issue session — AC#6 (per-issue lifecycle works
    for both Claude and Amp) was structurally broken.

    This test exercises the full pipeline against a fake ``amp`` that
    emits a complete stream-json transcript so the lifecycle reaches an
    :class:`IssueResult`. The exact ``IssueResult.success`` value depends
    on the gate / review configuration, which is incidental here; the
    contract this test pins is:

      1. The fluent chain on :class:`AmpRuntimeBuilder` succeeds.
      2. ``client_factory.create(runtime.options)`` produces a working
         :class:`AmpClient` that consumes the Amp stream-json shape.
      3. The lifecycle returns a real :class:`IssueResult` rather than
         crashing with an ``AttributeError`` deep in the pipeline.
    """
    from src.pipeline.issue_result import IssueResult

    bin_dir, repo_path = amp_environment
    plugin_hash = AmpPluginInstaller().installed_plugin_hash()
    thread_id = "T-fake-thread-amp-001"
    _install_fake_amp(bin_dir, body=_fake_amp_dual_role(plugin_hash, thread_id))

    issue = FakeIssue(
        id="bd-amp-fake.6",
        title="Fake-amp per-issue lifecycle",
        description="Drive the full pipeline against a canned stream-json.",
    )
    issue_provider = FakeIssueProvider(issues={issue.id: issue})

    config = OrchestratorConfig(
        repo_path=repo_path,
        max_agents=1,
        max_issues=1,
        timeout_minutes=1,
        # Disable code review so the lifecycle does not need an Anthropic
        # reviewer or a configured cerberus model. The pipeline still
        # exercises the implementer path, which is what AC#6 cares about.
        disable_validations={"review"},
    )
    mala_config = MalaConfig(
        runs_dir=repo_path / "runs",
        lock_dir=repo_path / "locks",
        coder="amp",
    )
    deps = OrchestratorDependencies(issue_provider=issue_provider)
    orchestrator = create_orchestrator(config, mala_config=mala_config, deps=deps)

    result = await orchestrator.run_implementer(issue.id)
    # Regression-only assertion: the pipeline returned a real IssueResult,
    # not crashed with AttributeError on the Amp runtime. The exact
    # success/failure depends on whether the gate could find validation
    # evidence in the canned transcript, which is environmental.
    assert isinstance(result, IssueResult)
    assert result.issue_id == issue.id


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
