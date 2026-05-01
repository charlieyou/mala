"""Plugin-load self-test fail-closed coverage (T013).

Covers plan section "Testing & Validation"
(``plans/2026-04-29-amp-provider-plan.md#L813-L823``) and AC#16, AC#17:

  * Pass case: fake amp emits the sentinel marker with matching version
    hash → self-test passes; orchestrator proceeds.
  * Fail closed: plugin missing — fake amp emits no sentinel within
    timeout → ``PLUGIN_MARKER_MISSING``.
  * Fail closed: Bun unavailable — fake amp emits Bun-missing stderr
    fingerprint → ``BUN_UNAVAILABLE``.
  * Fail closed: ``PLUGINS=all`` not honored → ``PLUGINS_ALL_UNSET``.
  * Fail closed: version mismatch → ``VERSION_MISMATCH``.
  * Fail closed: amp binary missing → ``AMP_BINARY_MISSING`` before
    other checks.
  * Fail closed: npm-install fingerprint → ``NPM_INSTALL_FINGERPRINT``.
  * Self-test timeout is bounded (e.g., 10s).
  * Self-test invoked exactly once per run (idempotent within a run).
"""

from __future__ import annotations

import time
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


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_mcp_factory() -> Callable[..., dict[str, object]]:
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


@pytest.fixture
def repo_path(tmp_path: Path) -> Path:
    p = tmp_path / "repo"
    p.mkdir()
    return p


@pytest.fixture
def amp_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect plugin install dir + AMP_API_KEY for the duration of a test.

    Returns the ``bin/`` directory the test should drop a fake ``amp``
    into; PATH is updated to include it.
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
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    monkeypatch.setenv("AMP_API_KEY", "fake-key")
    monkeypatch.delenv("MALA_DISALLOWED_TOOLS", raising=False)
    return bin_dir


def _expected_plugin_hash() -> str:
    """Return the bundled plugin's 16-char version hash."""
    from src.infra.clients.amp_plugin_installer import AmpPluginInstaller

    return AmpPluginInstaller().installed_plugin_hash()


def _write_fake_amp(bin_dir: Path, body: str) -> Path:
    fake_amp = bin_dir / "amp"
    fake_amp.write_text(body)
    fake_amp.chmod(0o755)
    return fake_amp


# ---------------------------------------------------------------------------
# Pass case
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_pass_case_fake_amp_emits_matching_sentinel(
    amp_env: Path,
    repo_path: Path,
    fake_mcp_factory: Callable[..., dict[str, object]],
) -> None:
    """Fake amp emits sentinel with correct version → self-test passes."""
    plugin_hash = _expected_plugin_hash()
    _write_fake_amp(
        amp_env,
        "#!/usr/bin/env bash\n"
        f'echo \'{{"mala_plugin":"loaded","version":"{plugin_hash}"}}\' >&2\n'
        "sleep 5\n",
    )
    provider = AmpAgentProvider()
    # No exception means pass.
    provider.install_prerequisites(repo_path, mcp_server_factory=fake_mcp_factory)


# ---------------------------------------------------------------------------
# Fail-closed: plugin marker missing
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_fail_closed_plugin_marker_missing(
    amp_env: Path,
    repo_path: Path,
    fake_mcp_factory: Callable[..., dict[str, object]],
) -> None:
    """Fake amp emits no sentinel → ``PLUGIN_MARKER_MISSING``."""
    _write_fake_amp(
        amp_env,
        "#!/usr/bin/env bash\necho 'no plugin here' >&2\nexit 0\n",
    )
    provider = AmpAgentProvider()
    with pytest.raises(AmpPluginNotActiveError) as excinfo:
        provider.install_prerequisites(repo_path, mcp_server_factory=fake_mcp_factory)
    assert excinfo.value.reason is AmpPluginNotActiveReason.PLUGIN_MARKER_MISSING
    # Docs URL is appended for actionability.
    assert "ampcode.com" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Fail-closed: Bun unavailable
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_fail_closed_bun_unavailable(
    amp_env: Path,
    repo_path: Path,
    fake_mcp_factory: Callable[..., dict[str, object]],
) -> None:
    """Fake amp emits a Bun-missing fingerprint → ``BUN_UNAVAILABLE``."""
    _write_fake_amp(
        amp_env,
        "#!/usr/bin/env bash\n"
        "echo 'bun: command not found' >&2\n"
        "echo 'plugin loader could not start' >&2\n"
        "exit 1\n",
    )
    provider = AmpAgentProvider()
    with pytest.raises(AmpPluginNotActiveError) as excinfo:
        provider.install_prerequisites(repo_path, mcp_server_factory=fake_mcp_factory)
    assert excinfo.value.reason is AmpPluginNotActiveReason.BUN_UNAVAILABLE


# ---------------------------------------------------------------------------
# Fail-closed: PLUGINS=all not honored
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_fail_closed_plugins_all_unset(
    amp_env: Path,
    repo_path: Path,
    fake_mcp_factory: Callable[..., dict[str, object]],
) -> None:
    """Fake amp signals PLUGINS env not honored → ``PLUGINS_ALL_UNSET``."""
    _write_fake_amp(
        amp_env,
        "#!/usr/bin/env bash\n"
        "echo 'plugin loading is disabled (PLUGINS=all required)' >&2\n"
        "exit 1\n",
    )
    provider = AmpAgentProvider()
    with pytest.raises(AmpPluginNotActiveError) as excinfo:
        provider.install_prerequisites(repo_path, mcp_server_factory=fake_mcp_factory)
    assert excinfo.value.reason is AmpPluginNotActiveReason.PLUGINS_ALL_UNSET


# ---------------------------------------------------------------------------
# Fail-closed: version mismatch
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_fail_closed_version_mismatch(
    amp_env: Path,
    repo_path: Path,
    fake_mcp_factory: Callable[..., dict[str, object]],
) -> None:
    """Sentinel marker carries unexpected hash → ``VERSION_MISMATCH``."""
    expected = _expected_plugin_hash()
    stale = "0" * len(expected)
    assert stale != expected
    _write_fake_amp(
        amp_env,
        "#!/usr/bin/env bash\n"
        f'echo \'{{"mala_plugin":"loaded","version":"{stale}"}}\' >&2\n'
        "sleep 5\n",
    )
    provider = AmpAgentProvider()
    with pytest.raises(AmpPluginNotActiveError) as excinfo:
        provider.install_prerequisites(repo_path, mcp_server_factory=fake_mcp_factory)
    assert excinfo.value.reason is AmpPluginNotActiveReason.VERSION_MISMATCH
    msg = str(excinfo.value)
    assert expected in msg
    assert stale in msg


# ---------------------------------------------------------------------------
# Fail-closed: amp binary missing
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_fail_closed_amp_binary_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    repo_path: Path,
    fake_mcp_factory: Callable[..., dict[str, object]],
) -> None:
    """``amp`` not on PATH → ``AMP_BINARY_MISSING`` before other checks."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    from src.infra.clients import amp_plugin_installer

    monkeypatch.setattr(
        amp_plugin_installer,
        "DEFAULT_PLUGIN_DIR",
        fake_home / ".config" / "amp" / "plugins",
    )
    # PATH points at an empty directory: nothing named ``amp`` is reachable.
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    monkeypatch.setenv("AMP_API_KEY", "fake-key")

    provider = AmpAgentProvider()
    with pytest.raises(AmpPluginNotActiveError) as excinfo:
        provider.install_prerequisites(repo_path, mcp_server_factory=fake_mcp_factory)
    assert excinfo.value.reason is AmpPluginNotActiveReason.AMP_BINARY_MISSING


# ---------------------------------------------------------------------------
# Fail-closed: npm-install fingerprint
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_fail_closed_npm_install_fingerprint(
    amp_env: Path,
    repo_path: Path,
    fake_mcp_factory: Callable[..., dict[str, object]],
) -> None:
    """Sentinel absent + npm-install stderr → ``NPM_INSTALL_FINGERPRINT``."""
    _write_fake_amp(
        amp_env,
        "#!/usr/bin/env bash\n"
        "echo 'launching @sourcegraph/amp from node_modules' >&2\n"
        "echo 'plugins are not supported on the npm install path' >&2\n"
        "exit 1\n",
    )
    provider = AmpAgentProvider()
    with pytest.raises(AmpPluginNotActiveError) as excinfo:
        provider.install_prerequisites(repo_path, mcp_server_factory=fake_mcp_factory)
    assert excinfo.value.reason is AmpPluginNotActiveReason.NPM_INSTALL_FINGERPRINT
    msg = str(excinfo.value)
    assert "binary install" in msg.lower()


# ---------------------------------------------------------------------------
# Bounded timeout
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_self_test_timeout_is_bounded(
    amp_env: Path,
    repo_path: Path,
    fake_mcp_factory: Callable[..., dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fake amp that hangs forever → self-test still returns within bound.

    We tighten the bound to 1.0s for the test so a regression that turns
    the bound into an unbounded wait shows up as a wall-clock failure
    instead of a 10s+ test runtime.
    """
    from src.infra.clients import amp_provider

    monkeypatch.setattr(amp_provider, "_SELFTEST_TIMEOUT_SECONDS", 1.0)
    _write_fake_amp(
        amp_env,
        "#!/usr/bin/env bash\n# No sentinel; just sleep past the bound.\nsleep 30\n",
    )
    provider = AmpAgentProvider()

    started = time.monotonic()
    with pytest.raises(AmpPluginNotActiveError):
        provider.install_prerequisites(repo_path, mcp_server_factory=fake_mcp_factory)
    elapsed = time.monotonic() - started
    # Allow generous margin for terminate/kill grace + thread join, but
    # reject anything close to the 30s hang.
    assert elapsed < 10.0, (
        f"self-test took {elapsed:.2f}s; bound is supposed to cap it near "
        "the configured timeout"
    )


# ---------------------------------------------------------------------------
# Idempotency: invoked exactly once per run (cache short-circuit)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_self_test_invoked_exactly_once_per_run(
    amp_env: Path,
    repo_path: Path,
    fake_mcp_factory: Callable[..., dict[str, object]],
) -> None:
    """Two ``install_prerequisites`` calls must spawn ``amp`` once.

    Implemented by a counter the fake amp increments via a side-effect
    file; the cache short-circuit must keep the count at 1.
    """
    counter = amp_env.parent / "amp-invocations"
    plugin_hash = _expected_plugin_hash()
    _write_fake_amp(
        amp_env,
        "#!/usr/bin/env bash\n"
        f'printf x >> "{counter}"\n'
        f'echo \'{{"mala_plugin":"loaded","version":"{plugin_hash}"}}\' >&2\n'
        "sleep 5\n",
    )
    provider = AmpAgentProvider()

    provider.install_prerequisites(repo_path, mcp_server_factory=fake_mcp_factory)
    provider.install_prerequisites(repo_path, mcp_server_factory=fake_mcp_factory)
    provider.install_prerequisites(repo_path, mcp_server_factory=fake_mcp_factory)

    assert counter.exists(), "fake amp never ran"
    assert counter.read_text() == "x", (
        f"expected exactly one amp invocation; got {len(counter.read_text())}"
    )
