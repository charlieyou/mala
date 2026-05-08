"""Unit tests for :class:`CodexAgentProvider` (Phase C T010, Phase F T013, Phase E T015).

Covers:
  * Conformance to :class:`AgentProvider` runtime protocol.
  * ``name == "codex"`` invariant.
  * ``client_factory.create(runtime)`` returns a :class:`CodexClient`
    bound to the runtime (Phase C wiring).
  * ``runtime_builder(...).build()`` returns a real :class:`CodexRuntime`
    carrying the resolved options (integration-path evidence for AC #4).
  * ``with_resume`` populates ``runtime.resume_thread_id``.
  * ``evidence_provider`` returns a real :class:`CodexEvidenceProvider`
    that reads tee'd JSONL (Phase F / T013).
  * ``install_prerequisites`` fail-closed signatures (Phase E5 / T015):
    each :class:`CodexHookNotActiveReason` is surfaced when the matching
    precondition fails.
  * Lazy-import contract: importing ``codex_provider`` does NOT pull in
    ``codex_app_server``.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from src.core.protocols.agent_provider import AgentProvider
from src.infra.clients.codex_provider import (
    CodexAgentProvider,
    CodexHookNotActiveError,
    CodexHookNotActiveReason,
    CodexNotInstalledError,
)
from src.infra.clients.codex_runtime import CodexRuntime, CodexRuntimeBuilder

if TYPE_CHECKING:
    from collections.abc import Callable


REPO_ROOT = Path(__file__).resolve().parents[5]


@pytest.fixture
def fake_mcp_factory() -> Callable[..., dict[str, object]]:
    """No-op MCP factory satisfying the protocol shape."""

    def factory(
        agent_id: str,
        repo_path: Path,
        emit_lock_event: object,
    ) -> dict[str, object]:
        del agent_id, repo_path, emit_lock_event
        return {}

    return factory


@pytest.fixture
def fake_codex_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    """Provide a sandboxed ``CODEX_HOME`` plus a fake ``codex`` binary on PATH.

    Returns ``(codex_home, bin_dir)``. The bin_dir is empty by default so
    individual tests can drop scripts (codex / mala-codex-pre-tool-use)
    into it as needed.
    """
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    sh_path = shutil.which("sh")
    extra = "/usr/bin" if sh_path else ""
    monkeypatch.setenv("PATH", f"{bin_dir}:{extra}")
    return codex_home, bin_dir


def _make_executable(path: Path, body: str = "#!/bin/sh\nexit 0\n") -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def _install_fake_sdk(monkeypatch: pytest.MonkeyPatch, *, present: bool) -> None:
    """Stub ``importlib.util.find_spec`` so the SDK probe sees a controlled state."""
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, package: str | None = None) -> object | None:
        if name == "codex_app_server":
            if present:
                # Return any non-None object — the provider checks for
                # ``is None`` truthiness.
                return object()
            return None
        return real_find_spec(name, package=package)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)


@pytest.mark.unit
def test_provider_conforms_to_agent_provider_protocol() -> None:
    provider = CodexAgentProvider()
    assert isinstance(provider, AgentProvider)
    assert provider.name == "codex"


@pytest.mark.unit
def test_default_options_match_unattended_run_defaults() -> None:
    """Plan decisions #2, #3, #9: defaults for unattended Codex runs."""
    provider = CodexAgentProvider()
    assert provider.model == "gpt-5.5"
    assert provider.effort is None
    assert provider.approval_policy == "never"
    assert provider.sandbox == "danger-full-access"


@pytest.mark.unit
def test_client_factory_create_returns_codex_client(
    tmp_path: Path, fake_mcp_factory: Callable[..., dict[str, object]]
) -> None:
    """``client_factory.create(runtime)`` returns a real :class:`CodexClient`."""
    from src.infra.clients.codex_client import CodexClient

    provider = CodexAgentProvider()
    builder = provider.runtime_builder(
        tmp_path, "agent-x", mcp_server_factory=fake_mcp_factory
    )
    runtime = builder.build()
    assert isinstance(runtime, CodexRuntime)

    client = provider.client_factory.create(runtime)
    assert isinstance(client, CodexClient)


@pytest.mark.unit
def test_client_factory_with_resume_threads_resume_token(
    tmp_path: Path, fake_mcp_factory: Callable[..., dict[str, object]]
) -> None:
    """``client_factory.with_resume`` produces a runtime with the resume id."""
    provider = CodexAgentProvider()
    runtime = provider.runtime_builder(
        tmp_path, "agent-x", mcp_server_factory=fake_mcp_factory
    ).build()
    resumed = provider.client_factory.with_resume(runtime, "thr_abc")
    assert isinstance(resumed, CodexRuntime)
    assert resumed.resume_thread_id == "thr_abc"
    # ``with_resume(runtime, None)`` returns the runtime unchanged.
    same = provider.client_factory.with_resume(runtime, None)
    assert same is runtime


@pytest.mark.unit
def test_runtime_builder_threads_resolved_options(
    tmp_path: Path, fake_mcp_factory: Callable[..., dict[str, object]]
) -> None:
    """Configured Codex options reach the per-session runtime.

    Integration-path evidence for AC #4: the resolved values flow
    ``MalaConfig.coder_options.codex`` -> ``CodexAgentProvider`` ->
    :class:`CodexRuntimeBuilder` -> :class:`CodexRuntime`.
    """
    provider = CodexAgentProvider(
        model="gpt-5.5-foo",
        effort="medium",
        approval_policy="on-request",
        sandbox="workspace-write",
    )

    builder = provider.runtime_builder(
        tmp_path, "agent-x", mcp_server_factory=fake_mcp_factory
    )
    assert isinstance(builder, CodexRuntimeBuilder)
    runtime = builder.build()
    assert isinstance(runtime, CodexRuntime)
    assert runtime.model == "gpt-5.5-foo"
    assert runtime.effort == "medium"
    assert runtime.approval_policy == "on-request"
    assert runtime.sandbox == "workspace-write"
    assert runtime.cwd == tmp_path
    assert runtime.agent_id == "agent-x"


@pytest.mark.unit
def test_runtime_builder_records_resume_token(
    tmp_path: Path, fake_mcp_factory: Callable[..., dict[str, object]]
) -> None:
    """``with_resume`` carries the resume id onto the built runtime."""
    provider = CodexAgentProvider()
    builder = provider.runtime_builder(
        tmp_path, "agent-x", mcp_server_factory=fake_mcp_factory
    )
    runtime = builder.with_resume("thr_123").build()
    assert isinstance(runtime, CodexRuntime)
    assert runtime.resume_thread_id == "thr_123"


@pytest.mark.unit
def test_evidence_provider_is_codex_evidence_provider(tmp_path: Path) -> None:
    """``evidence_provider`` returns a real :class:`CodexEvidenceProvider`.

    Phase F (T013) replaces the prior fail-closed stub with the tee
    fallback adapter (decision #11 disconfirmed by the F1 spike — see
    ``tests/spike/test_codex_thread_read_evidence.py``). The provider
    reads ``~/.config/mala/codex-sessions/{thread_id}.jsonl`` via the
    standard :class:`EvidenceProvider` surface; assert that
    ``get_log_path`` resolves there and that the cached instance is
    returned on repeat access.
    """
    from src.core.protocols.evidence import EvidenceProvider
    from src.infra.clients.codex_evidence_provider import (
        CODEX_SESSIONS_DIR,
        CodexEvidenceProvider,
    )

    provider = CodexAgentProvider()
    evidence = provider.evidence_provider
    assert isinstance(evidence, CodexEvidenceProvider)
    assert isinstance(evidence, EvidenceProvider)
    # Cached: a second access returns the same instance.
    assert provider.evidence_provider is evidence
    log_path = evidence.get_log_path(tmp_path, "thr_x")
    assert log_path == CODEX_SESSIONS_DIR / "thr_x.jsonl"
    # ``get_end_offset`` on a non-existent file returns the start offset
    # rather than raising — parity with FileSystemLogProvider /
    # AmpLogProvider so missing tee files do not break gate replay.
    assert evidence.get_end_offset(tmp_path / "missing.jsonl") == 0


@pytest.mark.unit
def test_importing_codex_provider_does_not_pull_codex_app_server() -> None:
    """Lazy-import contract.

    ``import src.infra.clients.codex_provider`` must NOT transitively
    import ``codex_app_server``. The SDK is referenced only inside
    :class:`CodexClient.__aenter__` / :meth:`CodexClient.query` (lazy).
    Touching the lazy-instantiated factory is also part of the
    cold-path expectation; it must not load the SDK either, only
    constructing the factory class.
    """
    code = """
import sys
from src.infra.clients.codex_provider import CodexAgentProvider
provider = CodexAgentProvider()
# Touching the lazy-instantiated factory must not load codex_app_server.
provider.client_factory  # noqa: B018
provider.evidence_provider  # noqa: B018
loaded = sorted(m for m in sys.modules if m.startswith('codex_app_server'))
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
# install_prerequisites — Phase E5 fail-closed signatures (T015)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_install_prerequisites_raises_codex_not_installed_when_sdk_missing(
    monkeypatch: pytest.MonkeyPatch,
    fake_mcp_factory: Callable[..., dict[str, object]],
) -> None:
    """SDK not importable → :class:`CodexNotInstalledError` (AC #14)."""
    _install_fake_sdk(monkeypatch, present=False)

    provider = CodexAgentProvider()
    with pytest.raises(CodexNotInstalledError, match="codex_app_server"):
        provider.install_prerequisites(
            Path("/tmp"), mcp_server_factory=fake_mcp_factory
        )


@pytest.mark.unit
def test_install_prerequisites_raises_codex_binary_missing(
    monkeypatch: pytest.MonkeyPatch,
    fake_mcp_factory: Callable[..., dict[str, object]],
    tmp_path: Path,
) -> None:
    """No ``codex`` on PATH → :class:`CodexHookNotActiveError(CODEX_BINARY_MISSING)`."""
    _install_fake_sdk(monkeypatch, present=True)
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))  # no codex here

    provider = CodexAgentProvider()
    with pytest.raises(CodexHookNotActiveError) as excinfo:
        provider.install_prerequisites(
            Path("/tmp"), mcp_server_factory=fake_mcp_factory
        )
    assert excinfo.value.reason is CodexHookNotActiveReason.CODEX_BINARY_MISSING


@pytest.mark.unit
def test_install_prerequisites_raises_script_missing(
    monkeypatch: pytest.MonkeyPatch,
    fake_codex_env: tuple[Path, Path],
    fake_mcp_factory: Callable[..., dict[str, object]],
    tmp_path: Path,
) -> None:
    """``mala-codex-pre-tool-use`` not on PATH → ``SCRIPT_MISSING``."""
    codex_home, bin_dir = fake_codex_env
    del codex_home
    _install_fake_sdk(monkeypatch, present=True)
    _make_executable(bin_dir / "codex")  # codex present, hook script absent

    provider = CodexAgentProvider()
    with pytest.raises(CodexHookNotActiveError) as excinfo:
        provider.install_prerequisites(tmp_path, mcp_server_factory=fake_mcp_factory)
    assert excinfo.value.reason is CodexHookNotActiveReason.SCRIPT_MISSING


@pytest.mark.unit
def test_install_prerequisites_runs_installer_and_writes_trusted_hash(
    monkeypatch: pytest.MonkeyPatch,
    fake_codex_env: tuple[Path, Path],
    fake_mcp_factory: Callable[..., dict[str, object]],
    tmp_path: Path,
) -> None:
    """Happy path: plugin tree installed, hook-state file gets ``trusted_hash``."""
    codex_home, bin_dir = fake_codex_env
    _install_fake_sdk(monkeypatch, present=True)
    _make_executable(bin_dir / "codex")
    _make_executable(bin_dir / "mala-codex-pre-tool-use")

    provider = CodexAgentProvider()
    provider.install_prerequisites(tmp_path, mcp_server_factory=fake_mcp_factory)

    plugin_dir = codex_home / "plugins" / "mala-safety" / ".codex-plugin"
    assert (plugin_dir / "plugin.json").is_file()
    assert (plugin_dir / "hooks.json").is_file()
    assert (plugin_dir / ".mcp.json").is_file()

    hooks_toml = (codex_home / "hooks.toml").read_text(encoding="utf-8")
    assert "trusted_hash" in hooks_toml
    assert "sha256:" in hooks_toml
    # The state key is the hooks.json absolute path. Codex stores per-hook
    # trust keyed by the file location so different installs of the same
    # plugin (e.g., system + user) don't collide.
    assert str(plugin_dir / "hooks.json") in hooks_toml


@pytest.mark.unit
def test_install_prerequisites_caches_within_run(
    monkeypatch: pytest.MonkeyPatch,
    fake_codex_env: tuple[Path, Path],
    fake_mcp_factory: Callable[..., dict[str, object]],
    tmp_path: Path,
) -> None:
    """Selftest probe runs at most once per run (idempotency contract)."""
    codex_home, bin_dir = fake_codex_env
    del codex_home
    _install_fake_sdk(monkeypatch, present=True)
    _make_executable(bin_dir / "codex")
    _make_executable(bin_dir / "mala-codex-pre-tool-use")

    probe_calls: list[str] = []

    def probe(repo: Path, env: dict[str, str], plugin_hash: str) -> None:
        del repo, env
        probe_calls.append(plugin_hash)

    provider = CodexAgentProvider(selftest_probe=probe)
    provider.install_prerequisites(tmp_path, mcp_server_factory=fake_mcp_factory)
    provider.install_prerequisites(tmp_path, mcp_server_factory=fake_mcp_factory)

    assert len(probe_calls) == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    "reason",
    [
        CodexHookNotActiveReason.HOOK_MARKER_MISSING,
        CodexHookNotActiveReason.VERSION_MISMATCH,
        CodexHookNotActiveReason.PLUGIN_DISABLED,
        CodexHookNotActiveReason.TRUSTED_HASH_MISMATCH,
    ],
)
def test_install_prerequisites_propagates_probe_failures(
    reason: CodexHookNotActiveReason,
    monkeypatch: pytest.MonkeyPatch,
    fake_codex_env: tuple[Path, Path],
    fake_mcp_factory: Callable[..., dict[str, object]],
    tmp_path: Path,
) -> None:
    """Each runtime fail-closed reason from the probe surfaces with its tag.

    Drives the four runtime-only reasons (HOOK_MARKER_MISSING /
    VERSION_MISMATCH / PLUGIN_DISABLED / TRUSTED_HASH_MISMATCH) by
    injecting a probe that raises with that ``reason``. The two
    structural reasons (CODEX_BINARY_MISSING / SCRIPT_MISSING) are
    covered by their own dedicated tests above.
    """
    codex_home, bin_dir = fake_codex_env
    del codex_home
    _install_fake_sdk(monkeypatch, present=True)
    _make_executable(bin_dir / "codex")
    _make_executable(bin_dir / "mala-codex-pre-tool-use")

    def probe(repo: Path, env: dict[str, str], plugin_hash: str) -> None:
        del repo, env, plugin_hash
        raise CodexHookNotActiveError(
            f"selftest failed with {reason.value}", reason=reason
        )

    provider = CodexAgentProvider(selftest_probe=probe)
    with pytest.raises(CodexHookNotActiveError) as excinfo:
        provider.install_prerequisites(tmp_path, mcp_server_factory=fake_mcp_factory)
    assert excinfo.value.reason is reason


@pytest.mark.unit
def test_install_prerequisites_reinstall_replaces_stale_plugin(
    monkeypatch: pytest.MonkeyPatch,
    fake_codex_env: tuple[Path, Path],
    fake_mcp_factory: Callable[..., dict[str, object]],
    tmp_path: Path,
) -> None:
    """A stale on-disk plugin tree is replaced; second install is a no-op (E4)."""
    codex_home, bin_dir = fake_codex_env
    _install_fake_sdk(monkeypatch, present=True)
    _make_executable(bin_dir / "codex")
    _make_executable(bin_dir / "mala-codex-pre-tool-use")

    plugin_dir = codex_home / "plugins" / "mala-safety" / ".codex-plugin"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.json").write_text("// stale\n", encoding="utf-8")
    (plugin_dir / "hooks.json").write_text("// stale\n", encoding="utf-8")
    (plugin_dir / ".mcp.json").write_text("// stale\n", encoding="utf-8")

    provider = CodexAgentProvider()
    # First call replaces stale content.
    provider.install_prerequisites(tmp_path, mcp_server_factory=fake_mcp_factory)
    fresh_bytes = (plugin_dir / "plugin.json").read_bytes()
    assert fresh_bytes != b"// stale\n"

    # Second call short-circuits via the in-memory cache; the on-disk
    # bytes do not change.
    provider.install_prerequisites(tmp_path, mcp_server_factory=fake_mcp_factory)
    assert (plugin_dir / "plugin.json").read_bytes() == fresh_bytes
