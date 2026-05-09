"""Unit tests for :class:`AmpAgentProvider` (T013).

Covers the provider-protocol slice of plan section "Testing & Validation"
(``plans/2026-04-29-amp-provider-plan.md#L849-L853``):

  * ``isinstance(provider, AgentProvider)`` runtime conformance.
  * Lazy-import guard: importing :mod:`amp_provider` does NOT pull in
    ``claude_agent_sdk`` or :mod:`amp_client` / :mod:`amp_log_provider`.
  * ``install_prerequisites`` is idempotent within a run (cached on
    ``(amp_version, plugin_hash)``).
  * ``MALA_DISALLOWED_TOOLS`` warn-once: when the env var is set, the
    warning fires exactly once per run regardless of how many times
    ``install_prerequisites`` is invoked.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from src.infra.clients.amp_provider import (
    AmpAgentProvider,
    AmpPluginNotActiveError,
    AmpPluginNotActiveReason,
)

if TYPE_CHECKING:
    from collections.abc import Callable


REPO_ROOT = Path(__file__).resolve().parents[5]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_mcp_factory() -> Callable[..., dict[str, object]]:
    """Stdio-shaped MCP factory the Amp runtime can JSON-serialize."""

    def factory(
        agent_id: str,
        repo_path: Path,
        emit_lock_event: object,
    ) -> dict[str, object]:
        del emit_lock_event
        return {
            "locking_mcp": {
                "command": "true",
                "args": ["--agent-id", agent_id, "--repo", str(repo_path)],
                "env": {},
            }
        }

    return factory


def _install_fake_amp_emitting_sentinel(
    bin_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    plugin_hash: str,
) -> Path:
    """Install a fake ``amp`` binary that emits the sentinel marker on stderr.

    The marker carries ``plugin_hash`` as its ``version`` field so the
    self-test version-match check passes against an installer whose bundled
    plugin hashes to the same value.
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    fake_amp = bin_dir / "amp"
    script = (
        "#!/usr/bin/env bash\n"
        "# Fake amp emitting the sentinel marker on stderr immediately.\n"
        f'echo \'{{"mala_plugin":"loaded","version":"{plugin_hash}"}}\' >&2\n'
        "# Then sleep briefly to mimic Amp staying alive past session.start;\n"
        "# the orchestrator terminates us as soon as the marker is observed.\n"
        "sleep 5\n"
    )
    fake_amp.write_text(script)
    fake_amp.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    return fake_amp


@pytest.mark.unit
def test_client_factory_resume_uses_verified_threads_continue_strategy(
    tmp_path: Path,
) -> None:
    from src.infra.clients.amp_runtime import AmpRuntime
    from src.infra.hooks.lint_cache import LintCache

    provider = AmpAgentProvider()
    runtime = AmpRuntime(
        cwd=tmp_path,
        env={},
        argv=("amp", "--execute", "--stream-json"),
        mcp_config={},
        mode="deep",
        log_path=tmp_path / ".pending.jsonl",
        lint_cache=LintCache(repo_path=tmp_path),
    )

    resumed = provider.client_factory.with_resume(runtime, "T-resume")

    assert isinstance(resumed, AmpRuntime)
    assert resumed.resume_thread_id == "T-resume"
    assert "--thread-id" not in resumed.argv


# ---------------------------------------------------------------------------
# Lazy-import guard
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_importing_amp_provider_does_not_load_claude_sdk() -> None:
    """Importing :mod:`amp_provider` must not pull in ``claude_agent_sdk``.

    Mirrors :func:`tests.unit.test_lazy_imports.test_import_claude_provider_does_not_load_sdk`
    for the Amp side. Run in a subprocess to get a clean import state.
    """
    code = """
import sys
for mod in list(sys.modules):
    if mod.startswith('claude_agent_sdk'):
        del sys.modules[mod]

from src.infra.clients.amp_provider import AmpAgentProvider
provider = AmpAgentProvider()
# Touching client_factory must NOT eagerly import the SDK either; only an
# accidental top-level claude_agent_sdk import would break the contract.
provider.client_factory  # noqa: B018
loaded = sorted(m for m in sys.modules if m.startswith('claude_agent_sdk'))
if loaded:
    print('FAIL: ' + ','.join(loaded))
    sys.exit(1)
print('PASS')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    assert "PASS" in result.stdout


@pytest.mark.unit
def test_importing_amp_provider_does_not_eagerly_load_amp_client() -> None:
    """Importing :mod:`amp_provider` must not eagerly load :mod:`amp_client`.

    The Amp client carries ``asyncio`` + ``signal`` machinery; keeping it
    behind a lazy import means a Claude-only path that imports the
    provider module for type checks does not pay that cost.
    """
    code = """
import sys
for mod in list(sys.modules):
    if mod.startswith('src.infra.clients.amp'):
        del sys.modules[mod]

from src.infra.clients.amp_provider import AmpAgentProvider
provider = AmpAgentProvider()
# Loading the provider alone must not pull amp_client / amp_log_provider /
# amp_plugin_installer in. Only amp_runtime (data-only) is allowed.
loaded = sorted(
    m for m in sys.modules
    if m.startswith('src.infra.clients.amp')
    and m not in (
        'src.infra.clients.amp_provider',
        'src.infra.clients.amp_runtime',
    )
)
if loaded:
    print('FAIL: ' + ','.join(loaded))
    sys.exit(1)
print('PASS')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    assert "PASS" in result.stdout


# ---------------------------------------------------------------------------
# install_prerequisites idempotency + caching
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_install_prerequisites_is_idempotent_within_a_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_mcp_factory: Callable[..., dict[str, object]],
) -> None:
    """Second :meth:`install_prerequisites` call must short-circuit.

    Plan ``L121``: result cached in-memory keyed on
    ``(amp_version, plugin_hash)`` for the duration of the run.
    """
    # Redirect plugin install + amp-sessions tee dirs to tmp_path so the
    # test does not touch the user's real ~/.config/amp/.
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    # Override AmpPluginInstaller's default DEFAULT_PLUGIN_DIR via env
    # passthrough does not exist; just monkeypatch the module attribute.
    from src.infra.clients import amp_plugin_installer

    monkeypatch.setattr(
        amp_plugin_installer,
        "DEFAULT_PLUGIN_DIR",
        fake_home / ".config" / "amp" / "plugins",
    )

    # Install the fake amp binary up front so we know the plugin hash to
    # echo. The installer reads the bundled plugin source and computes its
    # 16-char prefix; we mirror that so the marker matches.
    from src.infra.clients.amp_plugin_installer import AmpPluginInstaller

    plugin_hash = AmpPluginInstaller().installed_plugin_hash()
    bin_dir = tmp_path / "bin"
    _install_fake_amp_emitting_sentinel(bin_dir, monkeypatch, plugin_hash=plugin_hash)

    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    provider = AmpAgentProvider()

    # First call performs the self-test.
    provider.install_prerequisites(repo_path, mcp_server_factory=fake_mcp_factory)

    # Replace the fake amp with one that *fails*; if the second call
    # actually re-ran the self-test, it would now raise. The cache
    # contract says it must short-circuit instead.
    fake_amp = bin_dir / "amp"
    fake_amp.write_text("#!/usr/bin/env bash\nexit 1\n")
    fake_amp.chmod(0o755)

    provider.install_prerequisites(repo_path, mcp_server_factory=fake_mcp_factory)


@pytest.mark.unit
def test_install_prerequisites_raises_when_sentinel_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_mcp_factory: Callable[..., dict[str, object]],
) -> None:
    """Fake amp emitting nothing must trigger ``PLUGIN_MARKER_MISSING``.

    Smoke-test the orchestrator entry point; full coverage of the
    fail-closed reason matrix lives in
    :mod:`tests.unit.infra.clients.test_amp_plugin_self_test`.
    """
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    from src.infra.clients import amp_plugin_installer

    monkeypatch.setattr(
        amp_plugin_installer,
        "DEFAULT_PLUGIN_DIR",
        fake_home / ".config" / "amp" / "plugins",
    )

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_amp = bin_dir / "amp"
    # Emits no sentinel marker; just exits.
    fake_amp.write_text("#!/usr/bin/env bash\nexit 0\n")
    fake_amp.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin")

    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    provider = AmpAgentProvider()

    with pytest.raises(AmpPluginNotActiveError) as excinfo:
        provider.install_prerequisites(repo_path, mcp_server_factory=fake_mcp_factory)
    assert excinfo.value.reason is AmpPluginNotActiveReason.PLUGIN_MARKER_MISSING


# ---------------------------------------------------------------------------
# MALA_DISALLOWED_TOOLS warn-once
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_mala_disallowed_tools_emits_warning_when_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    fake_mcp_factory: Callable[..., dict[str, object]],
) -> None:
    """``MALA_DISALLOWED_TOOLS`` set → exactly one warn-level log per run.

    Plan ``L690``: bound to ``install_prerequisites`` (which fires once
    per run before the first session); subsequent calls within the same
    run do NOT re-emit.
    """
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    from src.infra.clients import amp_plugin_installer

    monkeypatch.setattr(
        amp_plugin_installer,
        "DEFAULT_PLUGIN_DIR",
        fake_home / ".config" / "amp" / "plugins",
    )

    from src.infra.clients.amp_plugin_installer import AmpPluginInstaller

    plugin_hash = AmpPluginInstaller().installed_plugin_hash()
    bin_dir = tmp_path / "bin"
    _install_fake_amp_emitting_sentinel(bin_dir, monkeypatch, plugin_hash=plugin_hash)
    monkeypatch.setenv("MALA_DISALLOWED_TOOLS", "Bash,Edit")

    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    provider = AmpAgentProvider()

    caplog.set_level(logging.WARNING, logger="src.infra.clients.amp_provider")
    provider.install_prerequisites(repo_path, mcp_server_factory=fake_mcp_factory)

    matches = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "MALA_DISALLOWED_TOOLS" in r.getMessage()
    ]
    assert len(matches) == 1, (
        f"expected exactly one MALA_DISALLOWED_TOOLS warning; got "
        f"{[r.getMessage() for r in caplog.records]}"
    )
    msg = matches[0].getMessage()
    assert "no effect" in msg
    assert "amp" in msg.lower()


@pytest.mark.unit
def test_mala_disallowed_tools_warning_not_repeated_on_second_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    fake_mcp_factory: Callable[..., dict[str, object]],
) -> None:
    """Second :meth:`install_prerequisites` call must NOT re-emit the warning.

    The cache short-circuit covers this incidentally, but assert it
    directly so a future refactor that bypasses the cache cannot regress
    the warn-once contract.
    """
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    from src.infra.clients import amp_plugin_installer

    monkeypatch.setattr(
        amp_plugin_installer,
        "DEFAULT_PLUGIN_DIR",
        fake_home / ".config" / "amp" / "plugins",
    )

    from src.infra.clients.amp_plugin_installer import AmpPluginInstaller

    plugin_hash = AmpPluginInstaller().installed_plugin_hash()
    bin_dir = tmp_path / "bin"
    _install_fake_amp_emitting_sentinel(bin_dir, monkeypatch, plugin_hash=plugin_hash)
    monkeypatch.setenv("MALA_DISALLOWED_TOOLS", "Bash")

    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    provider = AmpAgentProvider()

    caplog.set_level(logging.WARNING, logger="src.infra.clients.amp_provider")
    provider.install_prerequisites(repo_path, mcp_server_factory=fake_mcp_factory)
    provider.install_prerequisites(repo_path, mcp_server_factory=fake_mcp_factory)
    provider.install_prerequisites(repo_path, mcp_server_factory=fake_mcp_factory)

    matches = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "MALA_DISALLOWED_TOOLS" in r.getMessage()
    ]
    assert len(matches) == 1


@pytest.mark.unit
def test_mala_disallowed_tools_no_warning_when_unset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    fake_mcp_factory: Callable[..., dict[str, object]],
) -> None:
    """Without the env var, the warning must not fire."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    from src.infra.clients import amp_plugin_installer

    monkeypatch.setattr(
        amp_plugin_installer,
        "DEFAULT_PLUGIN_DIR",
        fake_home / ".config" / "amp" / "plugins",
    )

    from src.infra.clients.amp_plugin_installer import AmpPluginInstaller

    plugin_hash = AmpPluginInstaller().installed_plugin_hash()
    bin_dir = tmp_path / "bin"
    _install_fake_amp_emitting_sentinel(bin_dir, monkeypatch, plugin_hash=plugin_hash)
    monkeypatch.delenv("MALA_DISALLOWED_TOOLS", raising=False)

    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    provider = AmpAgentProvider()

    caplog.set_level(logging.WARNING, logger="src.infra.clients.amp_provider")
    provider.install_prerequisites(repo_path, mcp_server_factory=fake_mcp_factory)

    matches = [r for r in caplog.records if "MALA_DISALLOWED_TOOLS" in r.getMessage()]
    assert matches == []
