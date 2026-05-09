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
    _build_merged_codex_plugin_mcp_json,
)
from src.infra.clients.codex_runtime import CodexRuntime, CodexRuntimeBuilder

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Self


REPO_ROOT = Path(__file__).resolve().parents[5]


@pytest.fixture(autouse=True)
def _stub_live_codex_plugin_list_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the live ``codex app-server`` plugin/list probe by default.

    Real ``_live_codex_plugin_list_probe`` spawns a Codex app-server
    subprocess (the pinned SDK's ``codex_cli_bin``) which is not
    installable in unit-test environments — most of this file's tests
    drive structural / identity / direct-spawn checks against fake
    plugin trees and stub interpreters and have no need to also pay
    for a live Codex spawn. Tests that exercise the live probe itself
    re-monkeypatch the symbol explicitly.
    """
    monkeypatch.setattr(
        "src.infra.clients.codex_provider._live_codex_plugin_list_probe",
        lambda env_overlay, repo_path, *, marketplace="local": None,
    )


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

    Also writes a stub ``$CODEX_HOME/auth.json`` so the provider's auth
    probe (T018) treats this fixture as a logged-in environment by
    default. Tests that exercise the auth-missing failure mode delete
    the file or override the env vars explicitly.
    """
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "auth.json").write_text(
        '{"OPENAI_API_KEY":"sk-test","tokens":null,"last_refresh":null}\n',
        encoding="utf-8",
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    # Clear auth env vars so the on-disk auth.json is the active source
    # for the auth probe; tests opting into the env-var path set them
    # back explicitly.
    for var in ("OPENAI_API_KEY", "CODEX_API_KEY", "CODEX_ACCESS_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    sh_path = shutil.which("sh")
    extra = "/usr/bin" if sh_path else ""
    monkeypatch.setenv("PATH", f"{bin_dir}:{extra}")
    return codex_home, bin_dir


def _make_executable(path: Path, body: str = "#!/bin/sh\nexit 0\n") -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def _install_real_hook_shim(path: Path) -> None:
    """Install a ``mala-codex-pre-tool-use`` shim that delegates to our module.

    The default selftest probe spawns the on-PATH script and compares
    its decision to the same input passed to our embedded
    ``src.infra.hooks.codex_pre_tool_use.decide``. For tests that
    exercise the happy-path install flow we want a script that agrees
    with the embedded module — a python shim importing our ``main()``
    is byte-identical, so the probe's behavioral comparison passes.
    """
    body = (
        f"#!{sys.executable}\n"
        "import sys\n"
        f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        "from src.infra.hooks.codex_pre_tool_use import main\n"
        "raise SystemExit(main())\n"
    )
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def _noop_probe(repo: Path, env: dict[str, str], plugin_hash: str) -> None:
    """Selftest probe that always succeeds. Injected by tests that
    cover :meth:`install_prerequisites` flow but don't want to pay the
    cost of the default probe's behavioral subprocess check."""
    del repo, env, plugin_hash
    return None


def _install_fake_sdk(
    monkeypatch: pytest.MonkeyPatch,
    *,
    present: bool,
    codex_cli_bin_present: bool = False,
) -> None:
    """Stub ``importlib.util.find_spec`` so the SDK probes see controlled state.

    Drives both the ``codex_app_server`` SDK probe and the
    ``codex_cli_bin`` runtime-package probe used by the SDK-bundled
    runtime path (``codex_app_server.client.resolve_codex_bin``). When
    ``codex_cli_bin_present`` is True the provider should accept the
    SDK-bundled runtime even with no external ``codex`` binary on PATH.
    """
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, package: str | None = None) -> object | None:
        if name == "codex_app_server":
            return object() if present else None
        if name == "codex_cli_bin":
            return object() if codex_cli_bin_present else None
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

    Also asserts that the bundled-plugin installer module
    (``codex_plugin_installer``) stays off the cold path — module-load
    of ``codex_provider`` must not eagerly import the installer either.
    Regression for the prior ``_DEFAULT_PLUGIN_MARKETPLACE = _default_marketplace()``
    pattern that pulled the installer at module-load time.
    """
    code = """
import sys
from src.infra.clients.codex_provider import CodexAgentProvider
provider = CodexAgentProvider()
# Touching the lazy-instantiated factory must not load codex_app_server.
provider.client_factory  # noqa: B018
provider.evidence_provider  # noqa: B018
sdk = sorted(m for m in sys.modules if m.startswith('codex_app_server'))
installer = sorted(
    m for m in sys.modules
    if m == 'src.infra.clients.codex_plugin_installer'
)
if sdk:
    print('FAIL: codex_app_server loaded: ' + ','.join(sdk))
    sys.exit(1)
if installer:
    print('FAIL: codex_plugin_installer loaded: ' + ','.join(installer))
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
    """All three runtime sources unavailable → ``CODEX_BINARY_MISSING``.

    The provider rejects only when ALL of (a) ``CODEX_BINARY`` env var,
    (b) ``codex`` on ``PATH``, and (c) the SDK-bundled ``codex_cli_bin``
    runtime package are absent (parity with
    ``codex_app_server.client.resolve_codex_bin``).
    """
    _install_fake_sdk(monkeypatch, present=True, codex_cli_bin_present=False)
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))  # no codex here
    monkeypatch.delenv("CODEX_BINARY", raising=False)

    provider = CodexAgentProvider()
    with pytest.raises(CodexHookNotActiveError) as excinfo:
        provider.install_prerequisites(
            Path("/tmp"), mcp_server_factory=fake_mcp_factory
        )
    assert excinfo.value.reason is CodexHookNotActiveReason.CODEX_BINARY_MISSING


@pytest.mark.unit
def test_install_prerequisites_raises_codex_not_installed_when_auth_missing(
    monkeypatch: pytest.MonkeyPatch,
    fake_codex_env: tuple[Path, Path],
    fake_mcp_factory: Callable[..., dict[str, object]],
    tmp_path: Path,
) -> None:
    """No auth env var and no ``$CODEX_HOME/auth.json`` → :class:`CodexNotInstalledError`.

    Codex's auth manager (``codex-rs/login/src/auth/manager.rs:465-489``)
    treats ``OPENAI_API_KEY``/``CODEX_API_KEY``/``CODEX_ACCESS_TOKEN``
    or an on-disk ``auth.json`` as the credential source. Without one
    of those, ``codex app-server`` would either fail the first turn or
    pop the interactive ``Sign in with ChatGPT`` flow. Plan I1 /
    decision #8: the provider must fail closed before the spawn with
    an actionable ``codex login`` message.
    """
    codex_home, bin_dir = fake_codex_env
    _install_fake_sdk(monkeypatch, present=True)
    _make_executable(bin_dir / "codex")
    _make_executable(bin_dir / "mala-codex-pre-tool-use")
    # Remove the fixture's seeded auth.json so the probe sees an
    # un-authenticated home.
    (codex_home / "auth.json").unlink()
    for var in ("OPENAI_API_KEY", "CODEX_API_KEY", "CODEX_ACCESS_TOKEN"):
        monkeypatch.delenv(var, raising=False)

    provider = CodexAgentProvider()
    with pytest.raises(CodexNotInstalledError) as excinfo:
        provider.install_prerequisites(tmp_path, mcp_server_factory=fake_mcp_factory)
    msg = str(excinfo.value)
    assert "codex login" in msg
    assert "auth.json" in msg


@pytest.mark.unit
@pytest.mark.parametrize(
    "auth_env_var", ["OPENAI_API_KEY", "CODEX_API_KEY", "CODEX_ACCESS_TOKEN"]
)
def test_install_prerequisites_accepts_auth_env_var(
    auth_env_var: str,
    monkeypatch: pytest.MonkeyPatch,
    fake_codex_env: tuple[Path, Path],
    fake_mcp_factory: Callable[..., dict[str, object]],
    tmp_path: Path,
) -> None:
    """Any of the three auth env vars is sufficient — no ``auth.json`` required.

    Mirrors Codex's auth manager: a non-empty
    ``OPENAI_API_KEY`` / ``CODEX_API_KEY`` / ``CODEX_ACCESS_TOKEN`` short-circuits
    the on-disk ``auth.json`` lookup. CI environments commonly inject
    ``OPENAI_API_KEY`` directly without writing the file, and the
    provider must accept that path.
    """
    codex_home, bin_dir = fake_codex_env
    # Drop the fixture's seeded auth.json so the env var is the only
    # credential source visible to the probe.
    (codex_home / "auth.json").unlink()
    monkeypatch.setenv(auth_env_var, "sk-fake-test-key")
    _install_fake_sdk(monkeypatch, present=True)
    _make_executable(bin_dir / "codex")
    _make_executable(bin_dir / "mala-codex-pre-tool-use")

    provider = CodexAgentProvider(selftest_probe=_noop_probe)
    provider.install_prerequisites(tmp_path, mcp_server_factory=fake_mcp_factory)

    plugin_dir = (
        codex_home
        / "plugins"
        / "cache"
        / "local"
        / "mala-safety"
        / "local"
        / ".codex-plugin"
    )
    assert (plugin_dir / "plugin.json").is_file()


@pytest.mark.unit
def test_install_prerequisites_accepts_keyring_credentials_store(
    monkeypatch: pytest.MonkeyPatch,
    fake_codex_env: tuple[Path, Path],
    fake_mcp_factory: Callable[..., dict[str, object]],
    tmp_path: Path,
) -> None:
    """``cli_auth_credentials_store = "keyring"`` short-circuits the file probe.

    Codex's keyring backend deletes ``auth.json`` after a successful
    save, so a user who has logged in via the keyring credential store
    has neither the file nor any auth env var. The probe must defer
    to Codex's auth manager in that case rather than fail closed —
    otherwise ``install_prerequisites`` raises :class:`CodexNotInstalledError`
    before Codex can read its keyring-backed credential.
    """
    codex_home, bin_dir = fake_codex_env
    # Drop the seeded auth.json so only the keyring opt-in remains.
    (codex_home / "auth.json").unlink()
    (codex_home / "config.toml").write_text(
        'cli_auth_credentials_store = "keyring"\n', encoding="utf-8"
    )
    _install_fake_sdk(monkeypatch, present=True)
    _make_executable(bin_dir / "codex")
    _make_executable(bin_dir / "mala-codex-pre-tool-use")

    provider = CodexAgentProvider(selftest_probe=_noop_probe)
    provider.install_prerequisites(tmp_path, mcp_server_factory=fake_mcp_factory)

    plugin_dir = (
        codex_home
        / "plugins"
        / "cache"
        / "local"
        / "mala-safety"
        / "local"
        / ".codex-plugin"
    )
    assert (plugin_dir / "plugin.json").is_file()


@pytest.mark.unit
def test_install_prerequisites_ignores_non_keyring_credentials_store(
    monkeypatch: pytest.MonkeyPatch,
    fake_codex_env: tuple[Path, Path],
    fake_mcp_factory: Callable[..., dict[str, object]],
    tmp_path: Path,
) -> None:
    """A non-keyring credentials_store value does not bypass the probe.

    Only ``cli_auth_credentials_store = "keyring"`` defers to Codex's
    auth manager. Any other value (default ``auth_json``, an unknown
    string) must still require a file or env var.
    """
    codex_home, bin_dir = fake_codex_env
    (codex_home / "auth.json").unlink()
    (codex_home / "config.toml").write_text(
        'cli_auth_credentials_store = "auth_json"\n', encoding="utf-8"
    )
    _install_fake_sdk(monkeypatch, present=True)
    _make_executable(bin_dir / "codex")
    _make_executable(bin_dir / "mala-codex-pre-tool-use")

    provider = CodexAgentProvider()
    with pytest.raises(CodexNotInstalledError, match="codex login"):
        provider.install_prerequisites(tmp_path, mcp_server_factory=fake_mcp_factory)


@pytest.mark.unit
def test_install_prerequisites_treats_blank_auth_env_var_as_missing(
    monkeypatch: pytest.MonkeyPatch,
    fake_codex_env: tuple[Path, Path],
    fake_mcp_factory: Callable[..., dict[str, object]],
    tmp_path: Path,
) -> None:
    """A whitespace-only auth env var must NOT count as a credential.

    Codex's ``read_non_empty_env_var`` (``manager.rs:484-489``) trims
    and rejects blank values. The provider mirrors that: an empty
    ``OPENAI_API_KEY`` left over from a misconfigured CI shell must
    fail closed with the same actionable message a fully-missing
    credential surfaces, not look authenticated.
    """
    codex_home, bin_dir = fake_codex_env
    (codex_home / "auth.json").unlink()
    monkeypatch.setenv("OPENAI_API_KEY", "   ")
    _install_fake_sdk(monkeypatch, present=True)
    _make_executable(bin_dir / "codex")
    _make_executable(bin_dir / "mala-codex-pre-tool-use")

    provider = CodexAgentProvider()
    with pytest.raises(CodexNotInstalledError, match="codex login"):
        provider.install_prerequisites(tmp_path, mcp_server_factory=fake_mcp_factory)


@pytest.mark.unit
def test_install_prerequisites_accepts_sdk_bundled_codex_runtime(
    monkeypatch: pytest.MonkeyPatch,
    fake_codex_env: tuple[Path, Path],
    fake_mcp_factory: Callable[..., dict[str, object]],
    tmp_path: Path,
) -> None:
    """No ``codex`` on PATH but ``codex_cli_bin`` importable → succeeds.

    Regression: ``codex_app_server.client.resolve_codex_bin`` falls back
    to importing ``codex_cli_bin`` when ``AppServerConfig.codex_bin`` is
    None — the path :class:`CodexClient.__aenter__` exercises by default.
    The provider must not reject this configuration: rejecting it would
    mean a ``codex_app_server`` + ``openai-codex-cli-bin`` install with
    no external ``codex`` binary fails install_prerequisites even though
    a real Codex session would actually run.
    """
    codex_home, bin_dir = fake_codex_env
    _install_fake_sdk(monkeypatch, present=True, codex_cli_bin_present=True)
    monkeypatch.delenv("CODEX_BINARY", raising=False)
    # mala-codex-pre-tool-use must be on PATH for the install to succeed,
    # but we deliberately do NOT install a fake ``codex`` to prove the
    # SDK-bundled runtime path is accepted.
    _make_executable(bin_dir / "mala-codex-pre-tool-use")

    # Bypass the default probe's behavioral subprocess check (covered by
    # its own dedicated tests below); the focus here is runtime
    # resolution.
    provider = CodexAgentProvider(selftest_probe=_noop_probe)
    provider.install_prerequisites(tmp_path, mcp_server_factory=fake_mcp_factory)

    # Plugin tree must be installed and config.toml must carry the
    # trusted_hash entry — the happy path completed end-to-end without
    # an external codex binary.
    plugin_dir = (
        codex_home
        / "plugins"
        / "cache"
        / "local"
        / "mala-safety"
        / "local"
        / ".codex-plugin"
    )
    assert (plugin_dir / "plugin.json").is_file()
    assert "trusted_hash" in (codex_home / "config.toml").read_text(encoding="utf-8")


@pytest.mark.unit
def test_install_prerequisites_accepts_codex_binary_env_override(
    monkeypatch: pytest.MonkeyPatch,
    fake_codex_env: tuple[Path, Path],
    fake_mcp_factory: Callable[..., dict[str, object]],
    tmp_path: Path,
) -> None:
    """``CODEX_BINARY`` env override → succeeds without PATH or SDK package.

    Mirrors the operator-override branch in
    ``codex_app_server.client.resolve_codex_bin``: if ``CODEX_BINARY``
    points anywhere, the SDK accepts and validates the path itself —
    the provider should not gate on independent PATH / package checks.
    """
    codex_home, bin_dir = fake_codex_env
    _install_fake_sdk(monkeypatch, present=True, codex_cli_bin_present=False)
    monkeypatch.setenv("CODEX_BINARY", str(tmp_path / "anywhere" / "codex"))
    _make_executable(bin_dir / "mala-codex-pre-tool-use")

    # Focus is the env-override branch; bypass the default probe.
    provider = CodexAgentProvider(selftest_probe=_noop_probe)
    provider.install_prerequisites(tmp_path, mcp_server_factory=fake_mcp_factory)

    plugin_dir = (
        codex_home
        / "plugins"
        / "cache"
        / "local"
        / "mala-safety"
        / "local"
        / ".codex-plugin"
    )
    assert (plugin_dir / "plugin.json").is_file()


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
def test_install_prerequisites_raises_trusted_hash_mismatch_on_unwritable_config(
    monkeypatch: pytest.MonkeyPatch,
    fake_codex_env: tuple[Path, Path],
    fake_mcp_factory: Callable[..., dict[str, object]],
    tmp_path: Path,
) -> None:
    """Unwritable ``$CODEX_HOME/config.toml`` → ``TRUSTED_HASH_MISMATCH``.

    Auto-trust is the bridge that lets Codex load the bundled hook;
    when we cannot persist ``trusted_hash``, Codex marks the hook
    Untrusted and silently skips it on PreToolUse
    (``codex-rs/hooks/src/engine/discovery.rs::hook_trust_status``).
    The provider must therefore fail-closed instead of caching success.

    Drives the failure by making ``config.toml`` itself a directory so
    every ``write_text`` call raises ``IsADirectoryError`` — the same
    surface a read-only ``CODEX_HOME`` would expose to ``write_text``.
    """
    codex_home, bin_dir = fake_codex_env
    _install_fake_sdk(monkeypatch, present=True)
    _make_executable(bin_dir / "codex")
    _make_executable(bin_dir / "mala-codex-pre-tool-use")
    # Place a directory at the config.toml path so write_text fails.
    (codex_home / "config.toml").mkdir()

    provider = CodexAgentProvider()
    with pytest.raises(CodexHookNotActiveError) as excinfo:
        provider.install_prerequisites(tmp_path, mcp_server_factory=fake_mcp_factory)
    assert excinfo.value.reason is CodexHookNotActiveReason.TRUSTED_HASH_MISMATCH


@pytest.mark.unit
def test_install_prerequisites_runs_installer_and_writes_trusted_hash(
    monkeypatch: pytest.MonkeyPatch,
    fake_codex_env: tuple[Path, Path],
    fake_mcp_factory: Callable[..., dict[str, object]],
    tmp_path: Path,
) -> None:
    """Happy path: plugin tree installed at Codex's cache path, both
    config.toml entries (``[plugins."<id>"]`` + ``[hooks.state."<key>"]``)
    written.
    """
    codex_home, bin_dir = fake_codex_env
    _install_fake_sdk(monkeypatch, present=True)
    _make_executable(bin_dir / "codex")
    _make_executable(bin_dir / "mala-codex-pre-tool-use")

    # Focus is the config.toml writes; bypass the default probe.
    provider = CodexAgentProvider(selftest_probe=_noop_probe)
    provider.install_prerequisites(tmp_path, mcp_server_factory=fake_mcp_factory)

    plugin_dir = (
        codex_home
        / "plugins"
        / "cache"
        / "local"
        / "mala-safety"
        / "local"
        / ".codex-plugin"
    )
    assert (plugin_dir / "plugin.json").is_file()
    assert (plugin_dir / "hooks.json").is_file()
    assert (plugin_dir / ".mcp.json").is_file()

    config_toml = (codex_home / "config.toml").read_text(encoding="utf-8")
    # Codex requires ALL FIVE preconditions for the bundled hook to fire:
    # (a) ``[features] plugins = true`` per ``codex-rs/features/src/lib.rs:951``
    #     (Feature::Plugins defaults to True but a user can opt out with
    #     ``plugins = false``; ``plugins_for_config_with_force_reload``
    #     early-returns when ``plugins_enabled`` is false);
    # (b) ``[features] plugin_hooks = true`` per ``codex-rs/features/src/lib.rs:957``
    #     (Feature::PluginHooks ships default_enabled=false; without it
    #     ``catalog_processor`` skips loading plugin-bundled hooks);
    # (c) ``[features] hooks = true`` so Codex's global hook execution
    #     gate cannot leave the loaded/trusted safety hook dormant;
    # (d) ``[plugins."<key>"] enabled = true`` per
    #     ``codex-rs/core-plugins/src/manager.rs::configured_plugins_from_stack``;
    # (e) ``[hooks.state."<key>"]`` with matching trusted_hash per
    #     ``codex-rs/hooks/src/engine/discovery.rs::hook_trust_status``.
    assert "[features]" in config_toml
    assert "plugins = true" in config_toml
    assert "plugin_hooks = true" in config_toml
    assert "hooks = true" in config_toml
    assert '[plugins."mala-safety@local"]' in config_toml
    assert "enabled = true" in config_toml
    expected_key = (
        '[hooks.state."mala-safety@local:.codex-plugin/hooks.json:pre_tool_use:0:0"]'
    )
    assert expected_key in config_toml
    assert "trusted_hash" in config_toml
    # ``current_hash`` is a 64-hex sha256 of the canonical-JSON of the
    # normalized hook identity per
    # ``codex-rs/config/src/fingerprint.rs::version_for_toml`` — not a
    # 16-char prefix of the plugin file content.
    import re as _re

    match = _re.search(r'trusted_hash = "(sha256:[0-9a-f]+)"', config_toml)
    assert match is not None, config_toml
    assert len(match.group(1)) == len("sha256:") + 64


def _provider_isolated_codex_home(provider: CodexAgentProvider) -> Path:
    """Return the per-provider isolated ``CODEX_HOME`` allocated by
    :meth:`CodexAgentProvider.install_prerequisites` when the user
    configured MCP servers. Tests use this to read the actual install
    target instead of guessing the user's real ``$CODEX_HOME``.
    """
    handle = provider._isolated_codex_home
    assert handle is not None, (
        "Provider did not allocate an isolated CODEX_HOME — expected when "
        "``coder_options.codex.mcp_servers`` is non-empty."
    )
    return Path(handle.name)


@pytest.mark.unit
def test_install_prerequisites_renders_user_mcp_servers_into_installed_mcp_json(
    monkeypatch: pytest.MonkeyPatch,
    fake_codex_env: tuple[Path, Path],
    fake_mcp_factory: Callable[..., dict[str, object]],
    tmp_path: Path,
) -> None:
    """Phase G3 / AC-3: user-supplied MCP servers from
    ``coder_options.codex.mcp_servers`` reach the installed plugin's
    ``.mcp.json`` so Codex picks them up at runtime, while a user's
    attempt to override ``mala-locking`` is silently replaced by the
    bundled launcher (bundled is mandatory, never replaced).

    The installed file lives under the provider's *isolated*
    ``CODEX_HOME`` (allocated lazily because the user has MCP servers
    configured) — that home is what
    :meth:`CodexAgentProvider.runtime_builder` plumbs into the spawned
    Codex's env, so it is the directory Codex actually reads at
    ``thread_start``.
    """
    import json as _json

    _codex_home, bin_dir = fake_codex_env
    _install_fake_sdk(monkeypatch, present=True)
    _make_executable(bin_dir / "codex")
    _make_executable(bin_dir / "mala-codex-pre-tool-use")

    user_custom_spec: dict[str, object] = {
        "command": "/usr/bin/custom-mcp",
        "args": ["--from-yaml"],
        "env": {"FOO": "bar"},
    }
    hostile_locking_spec: dict[str, object] = {
        "command": "/tmp/evil-launcher",
        "args": [],
        "env": {},
    }
    provider = CodexAgentProvider(
        mcp_servers=(
            ("custom-server", user_custom_spec),
            # User's mala-locking attempt — must be silently replaced
            # by the bundled launcher in the installed plugin file.
            ("mala-locking", hostile_locking_spec),
        ),
        selftest_probe=_noop_probe,
    )
    provider.install_prerequisites(tmp_path, mcp_server_factory=fake_mcp_factory)

    isolated_home = _provider_isolated_codex_home(provider)
    plugin_dir = (
        isolated_home
        / "plugins"
        / "cache"
        / "local"
        / "mala-safety"
        / "local"
        / ".codex-plugin"
    )
    payload = _json.loads((plugin_dir / ".mcp.json").read_text(encoding="utf-8"))
    servers = payload["mcpServers"]
    # Non-clashing user entry passes through unchanged.
    assert servers["custom-server"] == user_custom_spec
    # Bundled mala-locking is present and points at the bundled launcher,
    # NOT the user's attempted override.
    bundled = servers["mala-locking"]
    assert bundled["command"] == "mala-codex-mcp-locking"
    assert bundled["command"] != hostile_locking_spec["command"]


@pytest.mark.unit
def test_isolated_codex_home_preserves_keyring_credentials_store_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    fake_codex_env: tuple[Path, Path],
    fake_mcp_factory: Callable[..., dict[str, object]],
    tmp_path: Path,
) -> None:
    """Regression: keyring-backed users with user MCP servers configured
    must see ``cli_auth_credentials_store = "keyring"`` carried into the
    isolated ``CODEX_HOME``'s ``config.toml``.

    Codex's keyring backend deletes ``auth.json`` after a successful
    save, so a keyring user has neither the file nor any auth env var
    in their real ``$CODEX_HOME``. The pre-isolation auth probe accepts
    them via ``_codex_uses_keyring_credentials_store``, but if the
    isolated home is then seeded with no ``config.toml``, the spawned
    Codex falls back to the default ``auth_json`` credential store and
    crashes at ``thread_start``. ``_ensure_isolated_codex_home`` must
    copy the user's ``config.toml`` so the keyring opt-in survives the
    isolation.
    """
    codex_home, bin_dir = fake_codex_env
    # Keyring-backed user: no auth.json on disk, only the opt-in flag.
    (codex_home / "auth.json").unlink()
    (codex_home / "config.toml").write_text(
        'cli_auth_credentials_store = "keyring"\n', encoding="utf-8"
    )
    _install_fake_sdk(monkeypatch, present=True)
    _make_executable(bin_dir / "codex")
    _make_executable(bin_dir / "mala-codex-pre-tool-use")

    provider = CodexAgentProvider(
        mcp_servers=(
            (
                "custom-server",
                {"command": "/usr/bin/custom-mcp", "args": [], "env": {}},
            ),
        ),
        selftest_probe=_noop_probe,
    )
    provider.install_prerequisites(tmp_path, mcp_server_factory=fake_mcp_factory)

    isolated_home = _provider_isolated_codex_home(provider)
    isolated_config = (isolated_home / "config.toml").read_text(encoding="utf-8")
    # The keyring opt-in survived the seed *and* the
    # ``_write_codex_plugin_config`` rewrite that adds plugin/hook entries.
    assert 'cli_auth_credentials_store = "keyring"' in isolated_config
    # Plugin entries were also written, proving the rewrite preserved
    # the top-level keyring scalar rather than dropping it.
    assert "trusted_hash" in isolated_config


@pytest.mark.unit
def test_isolated_codex_home_seed_does_not_inherit_read_only_config_mode(
    monkeypatch: pytest.MonkeyPatch,
    fake_codex_env: tuple[Path, Path],
    fake_mcp_factory: Callable[..., dict[str, object]],
    tmp_path: Path,
) -> None:
    """Regression: a user whose real ``config.toml`` is read-only (common
    with NixOS / GNU Stow dotfile managers) must not have that mode bit
    propagated to the isolated ``CODEX_HOME``'s copy. ``shutil.copy2``
    preserved permissions; the subsequent
    ``_write_codex_plugin_config`` ``write_text`` would then crash with
    ``PermissionError``, surfacing as ``CodexHookNotActiveError`` at
    ``thread_start``. ``shutil.copyfile`` (contents only) keeps the
    isolated copy writable.
    """
    codex_home, bin_dir = fake_codex_env
    user_config = codex_home / "config.toml"
    user_config.write_text('cli_auth_credentials_store = "keyring"\n', encoding="utf-8")
    user_config.chmod(0o444)
    _install_fake_sdk(monkeypatch, present=True)
    _make_executable(bin_dir / "codex")
    _make_executable(bin_dir / "mala-codex-pre-tool-use")

    provider = CodexAgentProvider(
        mcp_servers=(
            (
                "custom-server",
                {"command": "/usr/bin/custom-mcp", "args": [], "env": {}},
            ),
        ),
        selftest_probe=_noop_probe,
    )
    try:
        provider.install_prerequisites(tmp_path, mcp_server_factory=fake_mcp_factory)
    finally:
        # Restore writable mode so pytest's tmp_path teardown can clean up.
        user_config.chmod(0o644)

    isolated_home = _provider_isolated_codex_home(provider)
    isolated_config_path = isolated_home / "config.toml"
    # Plugin/hook entries were appended without ``PermissionError``,
    # proving the seed did not propagate the read-only mode.
    assert "trusted_hash" in isolated_config_path.read_text(encoding="utf-8")
    # Owner-write must be set on the isolated copy regardless of source mode.
    assert isolated_config_path.stat().st_mode & 0o200


@pytest.mark.unit
def test_install_prerequisites_threads_isolated_codex_home_into_runtime_env(
    monkeypatch: pytest.MonkeyPatch,
    fake_codex_env: tuple[Path, Path],
    fake_mcp_factory: Callable[..., dict[str, object]],
    tmp_path: Path,
) -> None:
    """The isolated ``CODEX_HOME`` reaches the spawned Codex via the
    runtime env overlay; without that overlay, Codex would inherit the
    user's real ``CODEX_HOME`` and read the wrong (shared) plugin tree.

    Pins the runtime side of the cross-process race fix.
    """
    _codex_home, bin_dir = fake_codex_env
    _install_fake_sdk(monkeypatch, present=True)
    _make_executable(bin_dir / "codex")
    _make_executable(bin_dir / "mala-codex-pre-tool-use")

    provider = CodexAgentProvider(
        mcp_servers=(
            (
                "custom-server",
                {"command": "/usr/bin/custom-mcp", "args": [], "env": {}},
            ),
        ),
        selftest_probe=_noop_probe,
    )
    provider.install_prerequisites(tmp_path, mcp_server_factory=fake_mcp_factory)

    isolated_home = _provider_isolated_codex_home(provider)
    builder = provider.runtime_builder(
        tmp_path, "agent-z", mcp_server_factory=fake_mcp_factory
    )
    runtime = builder.build()
    assert isinstance(runtime, CodexRuntime)
    assert runtime.env["CODEX_HOME"] == str(isolated_home)


@pytest.mark.unit
def test_concurrent_install_with_different_mcp_configs_does_not_collide(
    monkeypatch: pytest.MonkeyPatch,
    fake_codex_env: tuple[Path, Path],
    fake_mcp_factory: Callable[..., dict[str, object]],
    tmp_path: Path,
) -> None:
    """Regression: two providers with *different* user MCP configs
    sharing the user's real ``CODEX_HOME`` MUST NOT trample each
    other's installed ``.mcp.json``.

    Models the cross-process race the prior shared-CODEX_HOME design
    exposed: Process A (with ``custom-server``) installs; Process B
    (with no ``coder_options.codex.mcp_servers``) installs *before* A
    spawns Codex; the prior design wrote B's bundled-only payload over
    A's merged file, so A's spawn dropped ``custom-server``. With
    per-provider isolated ``CODEX_HOME``, A and B install into
    different directories and neither can revert the other.

    Asserts:
      * Provider A's isolated home carries A's ``custom-server``.
      * Provider B's CODEX_HOME (the user's real home) carries no
        ``custom-server`` — B never saw it and never wrote it.
      * The two install targets are distinct paths.
    """
    import json as _json

    user_codex_home, bin_dir = fake_codex_env
    _install_fake_sdk(monkeypatch, present=True)
    _make_executable(bin_dir / "codex")
    _make_executable(bin_dir / "mala-codex-pre-tool-use")

    user_custom_spec: dict[str, object] = {
        "command": "/usr/bin/custom-mcp",
        "args": [],
        "env": {},
    }
    provider_a = CodexAgentProvider(
        mcp_servers=(("custom-server", user_custom_spec),),
        selftest_probe=_noop_probe,
    )
    provider_a.install_prerequisites(tmp_path, mcp_server_factory=fake_mcp_factory)

    # Process B: a fresh provider invocation with no user MCP config,
    # using the same user-real CODEX_HOME (the fixture's env-set path).
    provider_b = CodexAgentProvider(selftest_probe=_noop_probe)
    provider_b.install_prerequisites(tmp_path, mcp_server_factory=fake_mcp_factory)

    isolated_a = _provider_isolated_codex_home(provider_a)
    plugin_a_mcp = (
        isolated_a
        / "plugins"
        / "cache"
        / "local"
        / "mala-safety"
        / "local"
        / ".codex-plugin"
        / ".mcp.json"
    )
    plugin_b_mcp = (
        user_codex_home
        / "plugins"
        / "cache"
        / "local"
        / "mala-safety"
        / "local"
        / ".codex-plugin"
        / ".mcp.json"
    )
    # The two install targets are distinct paths — that is the heart
    # of the race fix.
    assert plugin_a_mcp != plugin_b_mcp
    assert provider_b._isolated_codex_home is None

    payload_a = _json.loads(plugin_a_mcp.read_text(encoding="utf-8"))
    payload_b = _json.loads(plugin_b_mcp.read_text(encoding="utf-8"))
    # A retains its custom-server (would have been wiped under the
    # prior shared-CODEX_HOME design when B's installer ran second).
    assert payload_a["mcpServers"]["custom-server"] == user_custom_spec
    # B never saw custom-server and writes its own bundled-only file.
    assert "custom-server" not in payload_b["mcpServers"]
    assert payload_b["mcpServers"]["mala-locking"]["command"] == (
        "mala-codex-mcp-locking"
    )


@pytest.mark.unit
def test_install_prerequisites_idempotent_with_same_user_mcp_servers(
    monkeypatch: pytest.MonkeyPatch,
    fake_codex_env: tuple[Path, Path],
    fake_mcp_factory: Callable[..., dict[str, object]],
    tmp_path: Path,
) -> None:
    """A second installer call with the same merged content short-circuits
    at ``action="skipped"`` rather than rewriting the tree.

    Pins the idempotency contract: routing user MCP servers through
    the installer's ``mcp_json_override`` keeps the post-install hash
    check honest, so steady-state reruns do not churn through atomic
    writes (and the cache short-circuit in
    :meth:`install_prerequisites` keeps holding).
    """
    from src.infra.clients.codex_plugin_installer import (
        PLUGIN_DIRNAME,
        CodexPluginInstaller,
        plugin_root_dir,
    )

    _codex_home, bin_dir = fake_codex_env
    _install_fake_sdk(monkeypatch, present=True)
    _make_executable(bin_dir / "codex")
    _make_executable(bin_dir / "mala-codex-pre-tool-use")

    user_servers: tuple[tuple[str, object], ...] = (
        ("custom-server", {"command": "/usr/bin/custom-mcp", "args": [], "env": {}}),
    )
    provider = CodexAgentProvider(mcp_servers=user_servers, selftest_probe=_noop_probe)
    provider.install_prerequisites(tmp_path, mcp_server_factory=fake_mcp_factory)

    # A fresh installer with the same merged bytes against the same
    # target should report ``skipped`` — proves the merged content
    # round-trips through the hash check without forcing a rewrite.
    isolated_home = _provider_isolated_codex_home(provider)
    second = CodexPluginInstaller(
        mcp_json_override=_build_merged_codex_plugin_mcp_json(user_servers)
    ).install(target_dir=plugin_root_dir(isolated_home) / PLUGIN_DIRNAME)
    assert second.action == "skipped"


@pytest.mark.unit
def test_install_prerequisites_preserves_existing_features_block(
    monkeypatch: pytest.MonkeyPatch,
    fake_codex_env: tuple[Path, Path],
    fake_mcp_factory: Callable[..., dict[str, object]],
    tmp_path: Path,
) -> None:
    """Regression: writing ``plugin_hooks = true`` must not clobber other
    feature flags the user already set.

    Codex's ``[features]`` block is a flat key=value table; replacing
    the whole section would drop unrelated flags
    (``codex-rs/features/src/lib.rs`` registers ~40 features). The
    install step must update only ``plugin_hooks`` in-place.
    """
    codex_home, bin_dir = fake_codex_env
    _install_fake_sdk(monkeypatch, present=True)
    _make_executable(bin_dir / "codex")
    _make_executable(bin_dir / "mala-codex-pre-tool-use")
    # Pre-populate config.toml with an existing features block.
    (codex_home / "config.toml").write_text(
        "[features]\nplugins = true\nremote_plugin = false\n",
        encoding="utf-8",
    )

    provider = CodexAgentProvider(selftest_probe=_noop_probe)
    provider.install_prerequisites(tmp_path, mcp_server_factory=fake_mcp_factory)

    config_toml = (codex_home / "config.toml").read_text(encoding="utf-8")
    # All four keys must coexist inside [features] after the write.
    assert "plugins = true" in config_toml
    assert "remote_plugin = false" in config_toml
    assert "plugin_hooks = true" in config_toml
    assert "hooks = true" in config_toml


@pytest.mark.unit
def test_install_prerequisites_overwrites_disabled_plugin_hooks(
    monkeypatch: pytest.MonkeyPatch,
    fake_codex_env: tuple[Path, Path],
    fake_mcp_factory: Callable[..., dict[str, object]],
    tmp_path: Path,
) -> None:
    """``plugin_hooks = false`` (the upstream Codex default) must be
    rewritten to ``plugin_hooks = true`` so the bundled hook actually
    fires."""
    codex_home, bin_dir = fake_codex_env
    _install_fake_sdk(monkeypatch, present=True)
    _make_executable(bin_dir / "codex")
    _make_executable(bin_dir / "mala-codex-pre-tool-use")
    (codex_home / "config.toml").write_text(
        "[features]\nplugin_hooks = false\n",
        encoding="utf-8",
    )

    provider = CodexAgentProvider(selftest_probe=_noop_probe)
    provider.install_prerequisites(tmp_path, mcp_server_factory=fake_mcp_factory)

    config_toml = (codex_home / "config.toml").read_text(encoding="utf-8")
    assert "plugin_hooks = true" in config_toml
    assert "plugin_hooks = false" not in config_toml


@pytest.mark.unit
def test_install_prerequisites_overwrites_user_plugins_opt_out(
    monkeypatch: pytest.MonkeyPatch,
    fake_codex_env: tuple[Path, Path],
    fake_mcp_factory: Callable[..., dict[str, object]],
    tmp_path: Path,
) -> None:
    """Regression: a user who set ``[features] plugins = false`` must have
    that flipped to ``true``.

    Codex's ``PluginsManager.plugins_for_config_with_force_reload``
    (``codex-rs/core-plugins/src/manager.rs:480``) early-returns
    ``PluginLoadOutcome::default()`` when ``plugins_enabled`` is false,
    so without flipping this flag the bundled hook never reaches the
    discovery / trust-check pipeline regardless of how
    ``plugin_hooks`` and the per-plugin entries are configured.
    """
    codex_home, bin_dir = fake_codex_env
    _install_fake_sdk(monkeypatch, present=True)
    _make_executable(bin_dir / "codex")
    _make_executable(bin_dir / "mala-codex-pre-tool-use")
    (codex_home / "config.toml").write_text(
        "[features]\nplugins = false\n",
        encoding="utf-8",
    )

    provider = CodexAgentProvider(selftest_probe=_noop_probe)
    provider.install_prerequisites(tmp_path, mcp_server_factory=fake_mcp_factory)

    config_toml = (codex_home / "config.toml").read_text(encoding="utf-8")
    assert "plugins = true" in config_toml
    assert "plugins = false" not in config_toml
    # The other hook-loading gates must also have been added inside the
    # same block.
    assert "plugin_hooks = true" in config_toml
    assert "hooks = true" in config_toml


@pytest.mark.unit
def test_install_prerequisites_overwrites_disabled_global_hooks(
    monkeypatch: pytest.MonkeyPatch,
    fake_codex_env: tuple[Path, Path],
    fake_mcp_factory: Callable[..., dict[str, object]],
    tmp_path: Path,
) -> None:
    """``hooks = false`` must be rewritten to ``hooks = true``.

    Codex gates hook execution separately from plugin discovery and
    plugin-bundled hook loading. If a user opted out globally, the
    safety hook can be loaded/trusted but never executed.
    """
    codex_home, bin_dir = fake_codex_env
    _install_fake_sdk(monkeypatch, present=True)
    _make_executable(bin_dir / "codex")
    _make_executable(bin_dir / "mala-codex-pre-tool-use")
    (codex_home / "config.toml").write_text(
        "[features]\nhooks = false\nplugins = false\nplugin_hooks = false\n",
        encoding="utf-8",
    )

    provider = CodexAgentProvider(selftest_probe=_noop_probe)
    provider.install_prerequisites(tmp_path, mcp_server_factory=fake_mcp_factory)

    config_toml = (codex_home / "config.toml").read_text(encoding="utf-8")
    assert "hooks = true" in config_toml
    assert "hooks = false" not in config_toml
    assert "plugins = true" in config_toml
    assert "plugin_hooks = true" in config_toml


@pytest.mark.unit
def test_ensure_key_in_section_preserves_well_formed_toml_without_trailing_newline() -> (
    None
):
    """Regression: ``_ensure_key_in_section`` must not concatenate the
    new key onto the previous line when the input lacks a trailing
    newline.

    A user-edited ``config.toml`` ending mid-section without a final
    ``\\n`` would previously round-trip to malformed TOML
    (``plugins = trueplugin_hooks = true``) which crashes Codex on
    startup.
    """
    from src.infra.clients.codex_provider import _ensure_key_in_section

    existing_no_trailing_newline = "[features]\nplugins = true"  # no '\n' at EOF

    rewritten = _ensure_key_in_section(
        existing_no_trailing_newline,
        section_header="[features]",
        key="plugin_hooks",
        value="true",
    )

    # The output must contain the existing key on its own line and the
    # new key on its own line — no concatenation.
    lines = rewritten.splitlines()
    assert "plugins = true" in lines
    assert "plugin_hooks = true" in lines
    # And the merged token ``trueplugin_hooks`` must NOT appear.
    assert "trueplugin_hooks" not in rewritten


@pytest.mark.unit
def test_ensure_key_in_section_handles_aligned_spaces_around_equals() -> None:
    """Regression: TOML allows arbitrary whitespace between key and ``=``
    (e.g. ``plugins    = false`` for value alignment). The matcher must
    treat such lines as the same key — otherwise the rewriter appends a
    second ``plugins = true`` line, producing a duplicate-key TOML file
    that Codex rejects on startup.
    """
    from src.infra.clients.codex_provider import _ensure_key_in_section

    aligned = "[features]\nplugins    = false\nplugin_hooks    = false\n"

    rewritten = _ensure_key_in_section(
        aligned, section_header="[features]", key="plugins", value="true"
    )

    # ``plugins = false`` must be flipped to ``plugins = true``; the
    # original aligned-spaces line is rewritten in place.
    assert "plugins = true" in rewritten
    assert "plugins    = false" not in rewritten
    # No duplicate ``plugins`` entries — the in-place rewrite did not
    # also append a fresh ``plugins = true`` line.
    plugin_lines = [
        ln
        for ln in rewritten.splitlines()
        if ln.strip().startswith("plugins")
        and not ln.strip().startswith("plugin_hooks")
    ]
    assert len(plugin_lines) == 1, plugin_lines


@pytest.mark.unit
def test_ensure_key_in_section_does_not_match_key_prefix() -> None:
    """Regression: looking for ``plugin`` must not match
    ``plugin_hooks`` — otherwise the rewriter overwrites the wrong
    line. Pins the key-bound check ``stripped[len(key):].lstrip().startswith("=")``
    behavior.
    """
    from src.infra.clients.codex_provider import _ensure_key_in_section

    # Only ``plugin_hooks`` exists; we ask for ``plugin`` (a true prefix
    # but a distinct key). The rewriter must NOT rewrite the
    # ``plugin_hooks`` line — it should leave that line alone and
    # append a fresh ``plugin = true`` instead.
    existing = "[features]\nplugin_hooks = true\n"

    rewritten = _ensure_key_in_section(
        existing, section_header="[features]", key="plugin", value="true"
    )

    assert "plugin_hooks = true" in rewritten  # untouched
    # And the new ``plugin = true`` line was added inside the section.
    plugin_lines = [
        ln for ln in rewritten.splitlines() if ln.strip() == "plugin = true"
    ]
    assert len(plugin_lines) == 1, rewritten


@pytest.mark.unit
def test_default_marketplace_matches_installer_constant() -> None:
    """Lock-step pin: the provider's hardcoded marketplace literal must
    equal the installer's ``PLUGIN_MARKETPLACE`` constant.

    The two intentionally duplicate the literal (``"local"``) so the
    provider module's lazy-import contract is preserved (importing the
    installer module at module-load time would defeat the contract).
    This regression test fails if either side drifts.
    """
    from src.infra.clients.codex_plugin_installer import PLUGIN_MARKETPLACE
    from src.infra.clients.codex_provider import _DEFAULT_PLUGIN_MARKETPLACE

    assert _DEFAULT_PLUGIN_MARKETPLACE == PLUGIN_MARKETPLACE


@pytest.mark.unit
def test_hook_identity_modules_cover_safety_critical_dependencies() -> None:
    """Pin the allowlist of modules whose source bytes form the hook's
    identity. Hashing the entry-point alone is insufficient — the hook
    imports enforcement data from sibling modules and a stale install
    with the same entry-point but different deny-pattern bytes would
    take different deny logic at runtime.

    Concretely the list must include:

      * ``src.infra.hooks.codex_pre_tool_use`` — entry-point (decide).
      * ``src.infra.hooks.dangerous_commands`` — DANGEROUS_PATTERNS /
        DESTRUCTIVE_GIT_PATTERNS / BASH_TOOL_NAMES the hook imports
        (``codex_pre_tool_use.py:32-36``).
      * ``src.infra.tool_config`` — MALA_DISALLOWED_TOOLS that
        ``dangerous_commands`` re-exports (``dangerous_commands.py:12``).
      * ``src.infra.tools.locking`` — get_lock_holder / lock-key
        canonicalization the hook uses for file-edit + shell-write
        decisions (``codex_pre_tool_use.py:332`` lazy import).
      * ``src.infra.tools.env`` — lock-dir resolution
        (``locking.py:14``).
    """
    from src.infra.clients.codex_provider import _HOOK_IDENTITY_MODULES

    expected = {
        "src.infra.hooks.codex_pre_tool_use",
        "src.infra.hooks.dangerous_commands",
        "src.infra.tool_config",
        "src.infra.tools.locking",
        "src.infra.tools.env",
    }
    assert set(_HOOK_IDENTITY_MODULES) == expected, (
        "Hook-identity allowlist drifted from the safety-critical "
        f"dependency set; got {_HOOK_IDENTITY_MODULES!r}, expected {expected!r}."
    )


@pytest.mark.unit
def test_combined_module_hash_changes_when_dangerous_commands_diverges(
    tmp_path: Path,
) -> None:
    """Regression: a divergence in ``dangerous_commands.py`` (not just
    in ``codex_pre_tool_use.py``) must change the combined hash.

    Catches the reviewer's exact concern: an attacker / stale install
    with the same entry-point but a different ``dangerous_commands.py``
    would slip past an entry-point-only hash. The combined hash must
    fold every safety-critical module's bytes in.
    """
    import hashlib
    import importlib.util

    from src.infra.clients.codex_provider import (
        _HOOK_IDENTITY_MODULES,
        _compute_combined_module_hash,
    )

    baseline = _compute_combined_module_hash(_HOOK_IDENTITY_MODULES)

    # Recompute with the same modules but inject a perturbed
    # dangerous_commands hash by swapping one module's bytes via a
    # custom helper — proves the hash function actually folds in
    # every module's bytes (a single-byte change must flip the
    # digest).
    spec_dangerous = importlib.util.find_spec("src.infra.hooks.dangerous_commands")
    assert spec_dangerous is not None and spec_dangerous.origin is not None
    real_bytes = Path(spec_dangerous.origin).read_bytes()

    # Manually compute what the combined hash would be if
    # dangerous_commands had a single byte appended. We mirror the
    # length-prefix scheme the production helper uses.
    h = hashlib.sha256()
    for name in _HOOK_IDENTITY_MODULES:
        spec = importlib.util.find_spec(name)
        assert spec is not None and spec.origin is not None
        if name == "src.infra.hooks.dangerous_commands":
            data = real_bytes + b"\x00"  # perturbed
        else:
            data = Path(spec.origin).read_bytes()
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        h.update(len(data).to_bytes(8, "big"))
        h.update(data)
    perturbed = h.hexdigest()

    assert baseline != perturbed, (
        "Combined hash did not change when dangerous_commands bytes "
        "were perturbed — entry-point-only hashing regression."
    )
    del tmp_path  # unused; helper takes no fs ops


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
def test_default_probe_raises_plugin_disabled_when_manifest_paths_wrong(
    fake_codex_env: tuple[Path, Path], tmp_path: Path
) -> None:
    """Regression: manifest paths must point at ``./.codex-plugin/...`` so
    Codex resolves them to the actual install location. If a stale or
    hand-edited plugin.json points at top-level ``./hooks.json`` (the
    pre-fix shape), the default structural probe surfaces
    ``PLUGIN_DISABLED`` rather than caching success.
    """
    import json as _json

    from src.infra.clients.codex_provider import _default_selftest_probe

    codex_home, _bin_dir = fake_codex_env
    plugin_dir = (
        codex_home
        / "plugins"
        / "cache"
        / "local"
        / "mala-safety"
        / "local"
        / ".codex-plugin"
    )
    plugin_dir.mkdir(parents=True)
    # Write a plugin.json that points at the WRONG (pre-fix) hooks path.
    (plugin_dir / "plugin.json").write_text(
        _json.dumps({"name": "mala-safety", "hooks": "./hooks.json"}),
        encoding="utf-8",
    )
    (plugin_dir / "hooks.json").write_text(
        '{"hooks": {"PreToolUse": [{"hooks": [{"type":"command","command":"mala-codex-pre-tool-use"}]}]}}',
        encoding="utf-8",
    )
    (plugin_dir / ".mcp.json").write_text("{}", encoding="utf-8")

    with pytest.raises(CodexHookNotActiveError) as excinfo:
        _default_selftest_probe(
            tmp_path,
            {"CODEX_HOME": str(codex_home)},
            "deadbeef",
        )
    assert excinfo.value.reason is CodexHookNotActiveReason.PLUGIN_DISABLED
    assert "./.codex-plugin/hooks.json" in str(excinfo.value)


@pytest.mark.unit
def test_default_probe_raises_hook_marker_missing_when_command_absent(
    fake_codex_env: tuple[Path, Path], tmp_path: Path
) -> None:
    """Default probe surfaces ``HOOK_MARKER_MISSING`` when hooks.json
    declares no command handler for ``mala-codex-pre-tool-use``.
    """
    import json as _json

    codex_home, _bin_dir = fake_codex_env
    plugin_dir = (
        codex_home
        / "plugins"
        / "cache"
        / "local"
        / "mala-safety"
        / "local"
        / ".codex-plugin"
    )
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.json").write_text(
        _json.dumps(
            {
                "name": "mala-safety",
                "hooks": "./.codex-plugin/hooks.json",
                "mcpServers": "./.codex-plugin/.mcp.json",
            }
        ),
        encoding="utf-8",
    )
    # hooks.json declares no PreToolUse handler — hook silently dormant.
    (plugin_dir / "hooks.json").write_text(_json.dumps({"hooks": {}}), encoding="utf-8")
    (plugin_dir / ".mcp.json").write_text("{}", encoding="utf-8")

    from src.infra.clients.codex_provider import _default_selftest_probe

    with pytest.raises(CodexHookNotActiveError) as excinfo:
        _default_selftest_probe(
            tmp_path,
            {"CODEX_HOME": str(codex_home)},
            "deadbeef",
        )
    assert excinfo.value.reason is CodexHookNotActiveReason.HOOK_MARKER_MISSING


def _install_valid_plugin_tree(codex_home: Path) -> Path:
    """Helper: write a valid bundled plugin tree under the cache root.

    Returns the ``.codex-plugin/`` directory the default probe reads.
    """
    import json as _json

    plugin_dir = (
        codex_home
        / "plugins"
        / "cache"
        / "local"
        / "mala-safety"
        / "local"
        / ".codex-plugin"
    )
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.json").write_text(
        _json.dumps(
            {
                "name": "mala-safety",
                "hooks": "./.codex-plugin/hooks.json",
                "mcpServers": "./.codex-plugin/.mcp.json",
            }
        ),
        encoding="utf-8",
    )
    (plugin_dir / "hooks.json").write_text(
        _json.dumps(
            {
                "hooks": {
                    "PreToolUse": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "mala-codex-pre-tool-use",
                                }
                            ]
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    (plugin_dir / ".mcp.json").write_text("{}", encoding="utf-8")
    return plugin_dir


@pytest.mark.unit
def test_default_probe_passes_when_hook_module_hash_matches(
    fake_codex_env: tuple[Path, Path], tmp_path: Path
) -> None:
    """Module-identity check: a hook script whose shebang points at
    *this* Python interpreter resolves the same
    ``src.infra.hooks.codex_pre_tool_use`` module file we have, so the
    SHA-256 matches and the probe succeeds.

    Pins the happy-path identity contract: when the on-PATH hook is
    backed by the same Python that runs mala, the module bytes are
    identical and the run continues.
    """
    from src.infra.clients.codex_provider import _default_selftest_probe

    codex_home, bin_dir = fake_codex_env
    _install_valid_plugin_tree(codex_home)
    _install_real_hook_shim(bin_dir / "mala-codex-pre-tool-use")

    # Returns None on success.
    assert (
        _default_selftest_probe(
            tmp_path,
            {"CODEX_HOME": str(codex_home)},
            "deadbeef",
        )
        is None
    )


def _install_stub_python_interpreter(path: Path, *, fake_hash: str) -> None:
    """Write a stub ``python`` interpreter that returns ``fake_hash`` on -c.

    The default probe runs ``[hook_interpreter, "-c", probe_code]`` and
    expects a hex SHA-256 on stdout. This stub ignores the probe code
    entirely and prints the requested fake hash, simulating a Python
    install whose ``src.infra.hooks.codex_pre_tool_use`` resolves to a
    different (different bytes) module file — exactly the
    stale-mala-on-PATH scenario the probe must catch.
    """
    body = f"#!/bin/sh\necho {fake_hash}\nexit 0\n"
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


@pytest.mark.unit
def test_default_probe_raises_version_mismatch_on_diverging_module_hash(
    fake_codex_env: tuple[Path, Path], tmp_path: Path
) -> None:
    """Stale install regression: when the on-PATH hook's interpreter
    resolves a *different* ``src.infra.hooks.codex_pre_tool_use``
    module (different bytes / different version of our own logic), the
    probe raises ``VERSION_MISMATCH``.

    Catches the multi-mala-install case where ``shutil.which`` finds a
    hook script whose shebang Python points at a venv whose
    site-packages contain a different version of the hook module.
    Codex would invoke that version's deny / dangerous-command /
    lock-ownership logic — even if its allow-path JSON is byte-identical
    on harmless commands.

    A single-input behavioral check on a benign command would mask this
    case (same allow shape on safe commands). The hash check exposes
    *any* logic difference because a single byte change in
    ``codex_pre_tool_use.py`` flips the SHA-256.
    """
    from src.infra.clients.codex_provider import _default_selftest_probe

    codex_home, bin_dir = fake_codex_env
    _install_valid_plugin_tree(codex_home)
    # Stub interpreter at a known path; hook shim's shebang points at it.
    stub_interpreter = bin_dir / "stub-python"
    _install_stub_python_interpreter(
        stub_interpreter,
        fake_hash="dead" * 16,  # 64 hex chars — looks like a valid sha256
    )
    # Hook shim with shebang pointing at the stub. The hook itself is
    # never executed by the probe — only its interpreter is, via the
    # shebang.
    hook_path = bin_dir / "mala-codex-pre-tool-use"
    hook_path.write_text(
        f"#!{stub_interpreter}\nexit 0\n",
        encoding="utf-8",
    )
    hook_path.chmod(0o755)

    with pytest.raises(CodexHookNotActiveError) as excinfo:
        _default_selftest_probe(
            tmp_path,
            {"CODEX_HOME": str(codex_home)},
            "deadbeef",
        )
    assert excinfo.value.reason is CodexHookNotActiveReason.VERSION_MISMATCH
    # Error message should explicitly reference the multi-module
    # identity check (not just the entry-point alone).
    msg = str(excinfo.value)
    assert "hook-identity modules" in msg
    assert "src.infra.hooks.dangerous_commands" in msg


@pytest.mark.unit
def test_default_probe_raises_hook_marker_missing_when_module_unimportable(
    fake_codex_env: tuple[Path, Path], tmp_path: Path
) -> None:
    """An interpreter that cannot import the hook module surfaces
    ``HOOK_MARKER_MISSING`` and the error message identifies the
    offending module by name.

    Mirrors the probe's actual emission shape: when ``find_spec``
    returns None for one of the hook-identity modules, the probe
    prints ``NOMODULE:<name>`` and ``sys.exit(2)``. The provider must
    surface the ``NOMODULE:<name>`` diagnostic *before* falling through
    to the generic non-zero-exit branch — otherwise the user sees a
    bare ``exited 2`` message instead of which dependency is missing
    (regression: review-20 P2).
    """
    from src.infra.clients.codex_provider import _default_selftest_probe

    codex_home, bin_dir = fake_codex_env
    _install_valid_plugin_tree(codex_home)
    stub_interpreter = bin_dir / "stub-python"
    # Stub mirrors the probe's actual emission: ``NOMODULE:<name>`` on
    # stdout AND exit 2 (the real probe's ``sys.exit(2)``).
    body = "#!/bin/sh\necho NOMODULE:src.infra.hooks.dangerous_commands\nexit 2\n"
    stub_interpreter.write_text(body, encoding="utf-8")
    stub_interpreter.chmod(0o755)

    hook_path = bin_dir / "mala-codex-pre-tool-use"
    hook_path.write_text(
        f"#!{stub_interpreter}\nexit 0\n",
        encoding="utf-8",
    )
    hook_path.chmod(0o755)

    with pytest.raises(CodexHookNotActiveError) as excinfo:
        _default_selftest_probe(
            tmp_path,
            {"CODEX_HOME": str(codex_home)},
            "deadbeef",
        )
    assert excinfo.value.reason is CodexHookNotActiveReason.HOOK_MARKER_MISSING
    # Tailored diagnostic must reach the user — not a bare
    # ``exited 2`` from the generic returncode branch.
    msg = str(excinfo.value)
    assert "NOMODULE:src.infra.hooks.dangerous_commands" in msg
    assert "hook-dependency module" in msg


@pytest.mark.unit
def test_default_probe_raises_hook_marker_missing_when_no_shebang(
    fake_codex_env: tuple[Path, Path], tmp_path: Path
) -> None:
    """A hook executable without a Python shebang surfaces
    ``HOOK_MARKER_MISSING`` — without the shebang we cannot identify
    the interpreter that would resolve the hook module."""
    from src.infra.clients.codex_provider import _default_selftest_probe

    codex_home, bin_dir = fake_codex_env
    _install_valid_plugin_tree(codex_home)
    # No shebang line — kernel won't be able to spawn this anyway.
    hook_path = bin_dir / "mala-codex-pre-tool-use"
    hook_path.write_text("(this is not a script)\n", encoding="utf-8")
    hook_path.chmod(0o755)

    with pytest.raises(CodexHookNotActiveError) as excinfo:
        _default_selftest_probe(
            tmp_path,
            {"CODEX_HOME": str(codex_home)},
            "deadbeef",
        )
    assert excinfo.value.reason is CodexHookNotActiveReason.HOOK_MARKER_MISSING


@pytest.mark.unit
def test_default_probe_raises_hook_marker_missing_when_interpreter_crashes(
    fake_codex_env: tuple[Path, Path], tmp_path: Path
) -> None:
    """A hook interpreter that crashes during the module-identity
    probe surfaces ``HOOK_MARKER_MISSING`` rather than masking the
    failure as success."""
    from src.infra.clients.codex_provider import _default_selftest_probe

    codex_home, bin_dir = fake_codex_env
    _install_valid_plugin_tree(codex_home)
    stub_interpreter = bin_dir / "stub-python"
    body = "#!/bin/sh\necho boom 1>&2\nexit 17\n"
    stub_interpreter.write_text(body, encoding="utf-8")
    stub_interpreter.chmod(0o755)

    hook_path = bin_dir / "mala-codex-pre-tool-use"
    hook_path.write_text(
        f"#!{stub_interpreter}\nexit 0\n",
        encoding="utf-8",
    )
    hook_path.chmod(0o755)

    with pytest.raises(CodexHookNotActiveError) as excinfo:
        _default_selftest_probe(
            tmp_path,
            {"CODEX_HOME": str(codex_home)},
            "deadbeef",
        )
    assert excinfo.value.reason is CodexHookNotActiveReason.HOOK_MARKER_MISSING


@pytest.mark.unit
def test_selftest_marker_env_constant_matches_hook_module() -> None:
    """The provider and hook duplicate the marker env-var literal because
    import-linter forbids ``src.infra.clients`` from importing
    ``src.infra.hooks``. Pin equality so they cannot drift.
    """
    from src.infra.clients.codex_provider import _HOOK_SELFTEST_MARKER_ENV
    from src.infra.hooks.codex_pre_tool_use import SELFTEST_MARKER_ENV

    assert _HOOK_SELFTEST_MARKER_ENV == SELFTEST_MARKER_ENV


@pytest.mark.unit
def test_selftest_identity_modules_match_provider_list() -> None:
    """The hook computes its own marker hash over the same module list
    the provider expects. Drift would surface as a runtime
    ``VERSION_MISMATCH`` from the live-hook step, but pinning the
    equality here makes the regression visible directly."""
    from src.infra.clients.codex_provider import _HOOK_IDENTITY_MODULES
    from src.infra.hooks.codex_pre_tool_use import SELFTEST_IDENTITY_MODULES

    assert SELFTEST_IDENTITY_MODULES == _HOOK_IDENTITY_MODULES


@pytest.mark.unit
def test_default_probe_raises_hook_marker_missing_when_live_hook_writes_no_marker(
    fake_codex_env: tuple[Path, Path], tmp_path: Path
) -> None:
    """Live-hook step regression: a hook script whose interpreter passes
    the module-identity probe but whose ``main()`` never emits the
    selftest marker (e.g. a stale install where the entry-point bytes
    happen to match but the marker-emit hook is missing) must surface
    ``HOOK_MARKER_MISSING``.

    The shim used here delegates the module-identity probe to our
    embedded module (so step 3 passes) but bypasses ``main()`` and
    emits valid hook output without writing the marker — exactly the
    shape a regression that drops the marker-emit call would produce.
    """
    from src.infra.clients.codex_provider import _default_selftest_probe

    codex_home, bin_dir = fake_codex_env
    _install_valid_plugin_tree(codex_home)
    hook_path = bin_dir / "mala-codex-pre-tool-use"
    body = (
        f"#!{sys.executable}\n"
        "import json, sys\n"
        "sys.stdin.read()\n"
        'sys.stdout.write(json.dumps({"hookSpecificOutput": '
        '{"hookEventName": "PreToolUse", "permissionDecision": "allow"}}))\n'
        "sys.stdout.write('\\n')\n"
        "raise SystemExit(0)\n"
    )
    hook_path.write_text(body, encoding="utf-8")
    hook_path.chmod(0o755)

    with pytest.raises(CodexHookNotActiveError) as excinfo:
        _default_selftest_probe(
            tmp_path,
            {"CODEX_HOME": str(codex_home)},
            "deadbeef",
        )
    assert excinfo.value.reason is CodexHookNotActiveReason.HOOK_MARKER_MISSING
    msg = str(excinfo.value)
    assert "live selftest marker" in msg


@pytest.mark.unit
def test_default_probe_raises_version_mismatch_when_live_marker_hash_diverges(
    fake_codex_env: tuple[Path, Path], tmp_path: Path
) -> None:
    """Live-hook step: a marker whose ``version`` field does not match
    the provider's expected hash surfaces ``VERSION_MISMATCH``.

    Catches the case where the on-PATH hook script runs end-to-end and
    emits a marker, but the marker's identity hash diverges from the
    provider's — implying the hook's identity-module list (or hash
    logic) drifted from the provider's. Belt-and-suspenders gate on
    top of step 3's static module-bytes hash.
    """
    from src.infra.clients.codex_provider import _default_selftest_probe

    codex_home, bin_dir = fake_codex_env
    _install_valid_plugin_tree(codex_home)
    hook_path = bin_dir / "mala-codex-pre-tool-use"
    # The shim delegates the module-identity probe to our embedded
    # module (so step 3 passes), then writes a marker with a
    # deliberately wrong version field on the live invocation path.
    body = (
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        'marker = os.environ.get("MALA_CODEX_HOOK_SELFTEST_MARKER")\n'
        "if marker:\n"
        "    with open(marker, 'w', encoding='utf-8') as fp:\n"
        '        json.dump({"mala_codex_hook": "loaded", "version": "deadbeef"}, fp)\n'
        "sys.stdin.read()\n"
        'sys.stdout.write(json.dumps({"hookSpecificOutput": '
        '{"hookEventName": "PreToolUse", "permissionDecision": "allow"}}))\n'
        "sys.stdout.write('\\n')\n"
        "raise SystemExit(0)\n"
    )
    hook_path.write_text(body, encoding="utf-8")
    hook_path.chmod(0o755)

    with pytest.raises(CodexHookNotActiveError) as excinfo:
        _default_selftest_probe(
            tmp_path,
            {"CODEX_HOME": str(codex_home)},
            "deadbeef",
        )
    assert excinfo.value.reason is CodexHookNotActiveReason.VERSION_MISMATCH
    msg = str(excinfo.value)
    assert "live selftest marker" in msg
    assert "deadbeef" in msg


@pytest.mark.unit
def test_default_probe_raises_hook_marker_missing_when_live_hook_exits_nonzero(
    fake_codex_env: tuple[Path, Path], tmp_path: Path
) -> None:
    """Live-hook step: a hook script that survives the module-identity
    probe but exits non-zero on the live invocation surfaces
    ``HOOK_MARKER_MISSING``.

    Codex would observe the same crash at PreToolUse and either deny
    fail-closed or skip the hook depending on its config; either way
    the safety guarantee the orchestrator depends on is gone, so the
    selftest must catch it.
    """
    from src.infra.clients.codex_provider import _default_selftest_probe

    codex_home, bin_dir = fake_codex_env
    _install_valid_plugin_tree(codex_home)
    hook_path = bin_dir / "mala-codex-pre-tool-use"
    body = (
        f"#!{sys.executable}\n"
        "import sys\n"
        "sys.stderr.write('boom\\n')\n"
        "raise SystemExit(7)\n"
    )
    hook_path.write_text(body, encoding="utf-8")
    hook_path.chmod(0o755)

    with pytest.raises(CodexHookNotActiveError) as excinfo:
        _default_selftest_probe(
            tmp_path,
            {"CODEX_HOME": str(codex_home)},
            "deadbeef",
        )
    assert excinfo.value.reason is CodexHookNotActiveReason.HOOK_MARKER_MISSING
    msg = str(excinfo.value)
    assert "exited 7" in msg


# ---------------------------------------------------------------------------
# Live Codex plugin/list probe (Phase E5 / AC-5)
# ---------------------------------------------------------------------------


class _FakePluginSummary:
    """Minimal stand-in for ``codex_app_server.PluginSummary``."""

    def __init__(self, *, plugin_id: str, installed: bool, enabled: bool) -> None:
        self.id = plugin_id
        self.installed = installed
        self.enabled = enabled


class _FakeMarketplaceEntry:
    def __init__(self, plugins: list[_FakePluginSummary]) -> None:
        self.plugins = plugins


class _FakePluginListResponse:
    def __init__(
        self,
        marketplaces: list[_FakeMarketplaceEntry],
        *,
        load_errors: list[object] | None = None,
    ) -> None:
        self.marketplaces = marketplaces
        self.marketplace_load_errors = load_errors or []


class _FakeAsyncClient:
    def __init__(self, response: _FakePluginListResponse) -> None:
        self._response = response

    async def request(
        self,
        method: str,
        params: object,
        *,
        response_model: object,
    ) -> _FakePluginListResponse:
        del method, params, response_model
        return self._response


class _FakeAsyncCodex:
    """Minimal stand-in for ``codex_app_server.AsyncCodex`` for unit tests.

    Implements the async-context-manager + ``_client.request`` surface
    the live probe exercises without spawning a real ``codex
    app-server`` subprocess. Tests inject the desired ``plugin/list``
    response shape via ``_FakePluginListResponse``.
    """

    _next_response: _FakePluginListResponse | None = None
    _next_request_error: Exception | None = None
    _next_aenter_error: Exception | None = None

    def __init__(self, *, config: object) -> None:
        del config
        self._client: object = _FakeAsyncClient(
            _FakeAsyncCodex._next_response or _FakePluginListResponse([])
        )

    async def __aenter__(self) -> Self:
        if _FakeAsyncCodex._next_aenter_error is not None:
            raise _FakeAsyncCodex._next_aenter_error
        if _FakeAsyncCodex._next_request_error is not None:
            self._client = _ErrorClient(_FakeAsyncCodex._next_request_error)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        del exc_type, exc, tb


class _ErrorClient:
    """Async client that always raises a configured exception on ``request``."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def request(
        self,
        method: str,
        params: object,
        *,
        response_model: object,
    ) -> object:
        del method, params, response_model
        raise self._exc


def _patch_async_codex(
    monkeypatch: pytest.MonkeyPatch,
    *,
    response: _FakePluginListResponse | None = None,
    request_error: Exception | None = None,
    aenter_error: Exception | None = None,
) -> None:
    """Install ``_FakeAsyncCodex`` + a fake ``PluginListResponse`` model.

    Builds dummy modules whose attributes the live probe imports, then
    swaps them into ``sys.modules`` so the lazy ``from codex_app_server
    import ...`` inside the probe picks up the fakes instead of the
    real SDK.
    """
    import sys as _sys
    import types

    _FakeAsyncCodex._next_response = response
    _FakeAsyncCodex._next_request_error = request_error
    _FakeAsyncCodex._next_aenter_error = aenter_error

    class _FakeAppServerPkg(types.ModuleType):
        AppServerConfig = staticmethod(lambda **_kw: None)
        AsyncCodex = _FakeAsyncCodex

    class _FakeV2(types.ModuleType):
        PluginListResponse = object

    fake_pkg = _FakeAppServerPkg("codex_app_server")
    fake_generated = types.ModuleType("codex_app_server.generated")
    fake_v2 = _FakeV2("codex_app_server.generated.v2_all")
    monkeypatch.setitem(_sys.modules, "codex_app_server", fake_pkg)
    monkeypatch.setitem(_sys.modules, "codex_app_server.generated", fake_generated)
    monkeypatch.setitem(_sys.modules, "codex_app_server.generated.v2_all", fake_v2)


@pytest.mark.unit
def test_live_codex_plugin_list_probe_passes_when_plugin_installed_and_enabled(
    _stub_live_codex_plugin_list_probe: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Happy path: Codex reports our plugin installed+enabled → probe succeeds."""
    # Override the autouse stub so we drive the real implementation.
    monkeypatch.undo()
    from src.infra.clients.codex_provider import _live_codex_plugin_list_probe

    _patch_async_codex(
        monkeypatch,
        response=_FakePluginListResponse(
            marketplaces=[
                _FakeMarketplaceEntry(
                    plugins=[
                        _FakePluginSummary(
                            plugin_id="mala-safety@local",
                            installed=True,
                            enabled=True,
                        )
                    ]
                )
            ]
        ),
    )

    # No raise on success.
    _live_codex_plugin_list_probe({"CODEX_HOME": str(tmp_path)}, tmp_path)


@pytest.mark.unit
def test_live_codex_plugin_list_probe_raises_plugin_disabled_when_plugin_missing(
    _stub_live_codex_plugin_list_probe: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Codex's plugin/list does not list the bundled plugin → PLUGIN_DISABLED.

    Catches the case where the on-disk install is shaped right per our
    structural checks but Codex's plugin discovery did not see it
    (different cache root, manifest field rename, etc.).
    """
    monkeypatch.undo()
    from src.infra.clients.codex_provider import _live_codex_plugin_list_probe

    _patch_async_codex(
        monkeypatch,
        response=_FakePluginListResponse(marketplaces=[]),
    )

    with pytest.raises(CodexHookNotActiveError) as excinfo:
        _live_codex_plugin_list_probe({"CODEX_HOME": str(tmp_path)}, tmp_path)
    assert excinfo.value.reason is CodexHookNotActiveReason.PLUGIN_DISABLED
    assert "does not include" in str(excinfo.value)


@pytest.mark.unit
def test_live_codex_plugin_list_probe_raises_plugin_disabled_when_not_installed(
    _stub_live_codex_plugin_list_probe: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Codex sees the plugin but reports installed=False → PLUGIN_DISABLED."""
    monkeypatch.undo()
    from src.infra.clients.codex_provider import _live_codex_plugin_list_probe

    _patch_async_codex(
        monkeypatch,
        response=_FakePluginListResponse(
            marketplaces=[
                _FakeMarketplaceEntry(
                    plugins=[
                        _FakePluginSummary(
                            plugin_id="mala-safety@local",
                            installed=False,
                            enabled=True,
                        )
                    ]
                )
            ]
        ),
    )

    with pytest.raises(CodexHookNotActiveError) as excinfo:
        _live_codex_plugin_list_probe({"CODEX_HOME": str(tmp_path)}, tmp_path)
    assert excinfo.value.reason is CodexHookNotActiveReason.PLUGIN_DISABLED
    assert "NOT installed" in str(excinfo.value)


@pytest.mark.unit
def test_live_codex_plugin_list_probe_raises_plugin_disabled_when_not_enabled(
    _stub_live_codex_plugin_list_probe: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Codex sees the plugin installed but enabled=False → PLUGIN_DISABLED.

    This is the failure mode the reviewer specifically called out: the
    on-disk ``[plugins."<id>"] enabled = true`` write looked successful
    but Codex's evaluation reports the plugin disabled, meaning real
    PreToolUse turns would skip the safety hook. The live probe is the
    only signal that catches this discrepancy.
    """
    monkeypatch.undo()
    from src.infra.clients.codex_provider import _live_codex_plugin_list_probe

    _patch_async_codex(
        monkeypatch,
        response=_FakePluginListResponse(
            marketplaces=[
                _FakeMarketplaceEntry(
                    plugins=[
                        _FakePluginSummary(
                            plugin_id="mala-safety@local",
                            installed=True,
                            enabled=False,
                        )
                    ]
                )
            ]
        ),
    )

    with pytest.raises(CodexHookNotActiveError) as excinfo:
        _live_codex_plugin_list_probe({"CODEX_HOME": str(tmp_path)}, tmp_path)
    assert excinfo.value.reason is CodexHookNotActiveReason.PLUGIN_DISABLED
    assert "NOT enabled" in str(excinfo.value)


@pytest.mark.unit
def test_live_codex_plugin_list_probe_raises_plugin_disabled_on_marketplace_load_errors(
    _stub_live_codex_plugin_list_probe: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Codex returns marketplace_load_errors → PLUGIN_DISABLED.

    Surfaces the case where Codex's loader could not parse the
    on-disk plugin tree even though our structural JSON parse
    succeeded (e.g. unexpected ``plugin.json`` field, permission
    error, missing referenced file).
    """
    monkeypatch.undo()
    from src.infra.clients.codex_provider import _live_codex_plugin_list_probe

    _patch_async_codex(
        monkeypatch,
        response=_FakePluginListResponse(
            marketplaces=[],
            load_errors=["unexpected manifest field"],
        ),
    )

    with pytest.raises(CodexHookNotActiveError) as excinfo:
        _live_codex_plugin_list_probe({"CODEX_HOME": str(tmp_path)}, tmp_path)
    assert excinfo.value.reason is CodexHookNotActiveReason.PLUGIN_DISABLED
    assert "marketplace_load_errors" in str(excinfo.value)


@pytest.mark.unit
def test_live_codex_plugin_list_probe_raises_codex_binary_missing_when_spawn_fails(
    _stub_live_codex_plugin_list_probe: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """SDK fails to spawn the codex app-server → CODEX_BINARY_MISSING.

    Real failure shape from the SDK's ``_resolve_codex_bin`` when the
    runtime probe upstream succeeds but the actual spawn cannot find
    a binary (a packaging regression in ``codex_cli_bin``, a stale
    ``CODEX_BINARY`` env var pointing at a deleted path, etc.).
    """
    monkeypatch.undo()
    from src.infra.clients.codex_provider import _live_codex_plugin_list_probe

    _patch_async_codex(
        monkeypatch,
        aenter_error=FileNotFoundError("Unable to locate the pinned Codex runtime."),
    )

    with pytest.raises(CodexHookNotActiveError) as excinfo:
        _live_codex_plugin_list_probe({"CODEX_HOME": str(tmp_path)}, tmp_path)
    assert excinfo.value.reason is CodexHookNotActiveReason.CODEX_BINARY_MISSING


@pytest.mark.unit
def test_default_probe_invokes_live_codex_probe_after_other_checks(
    fake_codex_env: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """End-to-end pin: ``_default_selftest_probe`` calls the live probe
    after the structural / identity / direct-spawn checks all pass.

    Catches the regression where a refactor accidentally drops the
    live-Codex step — without it the selftest would silently lose the
    only Codex-side validation we have.
    """
    from src.infra.clients.codex_provider import _default_selftest_probe

    codex_home, bin_dir = fake_codex_env
    _install_valid_plugin_tree(codex_home)
    _install_real_hook_shim(bin_dir / "mala-codex-pre-tool-use")

    calls: list[tuple[dict[str, str], Path]] = []

    def _record_call(
        env: dict[str, str], repo: Path, *, marketplace: str = "local"
    ) -> None:
        del marketplace
        calls.append((dict(env), repo))

    monkeypatch.setattr(
        "src.infra.clients.codex_provider._live_codex_plugin_list_probe",
        _record_call,
    )

    _default_selftest_probe(
        tmp_path,
        {"CODEX_HOME": str(codex_home)},
        "deadbeef",
    )

    assert len(calls) == 1
    assert calls[0][0] == {"CODEX_HOME": str(codex_home)}
    assert calls[0][1] == tmp_path


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

    plugin_dir = (
        codex_home
        / "plugins"
        / "cache"
        / "local"
        / "mala-safety"
        / "local"
        / ".codex-plugin"
    )
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.json").write_text("// stale\n", encoding="utf-8")
    (plugin_dir / "hooks.json").write_text("// stale\n", encoding="utf-8")
    (plugin_dir / ".mcp.json").write_text("// stale\n", encoding="utf-8")

    # Focus is the installer's stale-replace behavior; bypass the probe.
    provider = CodexAgentProvider(selftest_probe=_noop_probe)
    # First call replaces stale content.
    provider.install_prerequisites(tmp_path, mcp_server_factory=fake_mcp_factory)
    fresh_bytes = (plugin_dir / "plugin.json").read_bytes()
    assert fresh_bytes != b"// stale\n"

    # Second call short-circuits via the in-memory cache; the on-disk
    # bytes do not change.
    provider.install_prerequisites(tmp_path, mcp_server_factory=fake_mcp_factory)
    assert (plugin_dir / "plugin.json").read_bytes() == fresh_bytes
