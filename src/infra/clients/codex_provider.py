"""Codex :class:`AgentProvider` (Phase C T010, Phase F T013, Phase E T015).

Wires :class:`CodexAgentProvider` to the real :class:`CodexClient` /
:class:`CodexRuntime` / :class:`CodexRuntimeBuilder` shipped in Phase C
(T010), to the real :class:`CodexEvidenceProvider` shipped in Phase F
(T013), and to the bundled ``mala-safety`` plugin install + selftest
shipped in Phase E (T015). ``install_prerequisites`` now performs the
full safety-critical fail-closed gate — SDK / binary checks, plugin
install, trusted-hash auto-write, hook self-test — paralleling the
Amp provider's posture.

Lazy-import contract (plan ``L733``): importing this module does NOT
transitively import ``codex_app_server``. The SDK is referenced only
inside runtime/probe paths such as :class:`CodexClient.__aenter__`,
:meth:`CodexClient.query`, and the live Codex selftests;
:class:`CodexClient` itself is imported lazily by
:meth:`_CodexClientFactory.create` so module-load remains SDK-free.
:class:`CodexEvidenceProvider` does not pull the SDK either — it reads
tee'd JSONL only — so eager construction is safe. The plugin installer
imports stay lazy too so a Claude/Amp-only run does not pay for the
Codex install path on import.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import tempfile
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from src.core.constants import (
    DEFAULT_CODEX_APPROVAL_POLICY,
    DEFAULT_CODEX_EFFORT,
    DEFAULT_CODEX_MODEL,
    DEFAULT_CODEX_SANDBOX,
)
from src.infra.clients._plugin_provider_base import (
    PluginInstallError,
    SelftestCache,
    install_with_selftest,
)
from src.infra.clients.codex_evidence_provider import CodexEvidenceProvider
from src.infra.clients.codex_mcp_factory import (
    _build_merged_codex_plugin_mcp_json,
    _create_codex_mcp_server_factory,
)
from src.infra.clients.codex_sdk_compat import (
    codex_runtime_resolvable,
)
from src.infra.clients.codex_selftest import _default_selftest_probe

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.core.protocols.agent_provider import CoderRuntimeBuilder
    from src.core.protocols.evidence import EvidenceProvider
    from src.core.protocols.lifecycle import DeadlockMonitorProtocol
    from src.core.protocols.sdk import (
        McpServerFactory,
        SDKClientFactoryProtocol,
        SDKClientProtocol,
    )


# ---------------------------------------------------------------------------
# Public exception + reason enum (Phase E5, T015)
# ---------------------------------------------------------------------------


class CodexNotInstalledError(RuntimeError):
    """Raised when Codex itself is not usable — SDK, runtime, or auth missing.

    Plan AC #14 (decision #8): ``coder=codex`` runs fail closed with an
    actionable message before any Codex turn is attempted. Mirrors
    :class:`AmpPluginNotActiveError`'s posture without conflating
    "Codex not installed" with "plugin/hook not loaded" (the latter is
    :class:`CodexHookNotActiveError`).
    """


class CodexHookNotActiveReason(StrEnum):
    """Why the Codex safety-hook selftest failed (structured form for callers).

    The string values are stable identifiers the orchestrator / telemetry
    can branch on without parsing prose error messages. Mirrors
    :class:`AmpPluginNotActiveReason` shape for cross-coder dashboards.
    """

    HOOK_MARKER_MISSING = "hook_marker_missing"
    VERSION_MISMATCH = "version_mismatch"
    SCRIPT_MISSING = "script_missing"
    PLUGIN_DISABLED = "plugin_disabled"
    TRUSTED_HASH_MISMATCH = "trusted_hash_mismatch"
    CODEX_BINARY_MISSING = "codex_binary_missing"


_HOOK_DOCS_URL = (
    "https://github.com/charlieyou/mala/blob/main/docs/cli-reference.md"
    "#codex-prerequisites"
)


class CodexHookNotActiveError(PluginInstallError):
    """Fail-closed error raised when the Codex safety hook is not active.

    Carries a structured :attr:`reason` so the orchestrator can branch on
    failure type and present a tailored remediation. The message always
    points at the docs URL so users have a single landing page for the
    fix; the docs subsection enumerates each reason and the matching
    one-time prerequisite step (relevant when auto-trust is unavailable —
    plan E6 fallback).
    """

    def __init__(
        self,
        message: str,
        *,
        reason: CodexHookNotActiveReason,
        docs_url: str = _HOOK_DOCS_URL,
    ) -> None:
        super().__init__(f"{message} See {docs_url} for the prerequisite steps.")
        self.reason = reason
        self.docs_url = docs_url


# Selftest probe contract: a callable that exercises a one-shot Codex turn
# and verifies the bundled hook ran with the expected version hash. Returns
# ``None`` on success; raises :class:`CodexHookNotActiveError` on any
# fail-closed signature. The provider's default probe stays minimal in
# this issue (T015) — Phase I/T020 layers SDK / runtime / auth checks on
# top, and the real-Codex e2e gate (I3) drives the spawn path. Tests
# inject custom probes to drive each :class:`CodexHookNotActiveReason`.


# ---------------------------------------------------------------------------
# Real client factory (Phase C)
# ---------------------------------------------------------------------------


class _CodexClientFactory:
    """:class:`SDKClientFactoryProtocol` for the Codex coder.

    ``create(runtime)`` constructs a :class:`CodexClient` bound to the
    provided :class:`CodexRuntime` (lazy import preserves the SDK
    isolation contract). ``with_resume(runtime, resume)`` returns a
    sibling runtime with ``resume_thread_id`` populated so the next
    :meth:`CodexClient.query` picks ``AsyncCodex.thread_resume`` over
    ``thread_start``.
    """

    def create(self, runtime: object) -> SDKClientProtocol:
        # Lazy import keeps ``codex_provider`` free of the
        # ``codex_client`` module's module-load-time cost; under
        # Claude/Amp-only runs the factory is constructed but
        # ``create`` is never reached.
        from src.infra.clients.codex_client import CodexClient
        from src.infra.clients.codex_runtime import CodexRuntime

        if not isinstance(runtime, CodexRuntime):
            raise TypeError(
                "CodexAgentProvider.client_factory.create(runtime) requires "
                "a CodexRuntime; got "
                f"{type(runtime).__name__}."
            )
        return cast("SDKClientProtocol", CodexClient(runtime))

    def with_resume(self, runtime: object, resume: str | None) -> object:
        from dataclasses import replace

        from src.infra.clients.codex_runtime import CodexRuntime

        if not isinstance(runtime, CodexRuntime):
            raise TypeError(
                "CodexAgentProvider.client_factory.with_resume(runtime, resume) "
                "requires a CodexRuntime; got "
                f"{type(runtime).__name__}."
            )
        if resume is None:
            return runtime
        return replace(runtime, resume_thread_id=resume)


# ---------------------------------------------------------------------------
# Codex runtime resolution (Phase E5)
# ---------------------------------------------------------------------------


def _codex_runtime_resolvable() -> bool:
    """Return True iff Codex can find a runtime at session-start.

    Kept as a provider-local wrapper because tests and the e2e gate import
    it directly. The underlying helper is also used by the SDK client/probes,
    so the gate and the real ``AppServerConfig`` code path stay aligned.
    """
    return codex_runtime_resolvable()


_CODEX_AUTH_ENV_VARS: tuple[str, ...] = (
    "OPENAI_API_KEY",
    "CODEX_API_KEY",
    "CODEX_ACCESS_TOKEN",
)
"""Env vars Codex's auth manager treats as a valid credential source
(``codex-rs/login/src/auth/manager.rs:465-489``). A non-empty value in
any of these short-circuits the on-disk ``auth.json`` lookup, so an
auth probe that ignores them would falsely fail when a CI environment
injects ``OPENAI_API_KEY`` directly."""

_CODEX_CONFIG_AUTH_SEED_KEYS: tuple[str, ...] = (
    "chatgpt_base_url",
    "cli_auth_credentials_store",
    "forced_chatgpt_workspace_id",
    "forced_login_method",
)
"""User ``config.toml`` top-level keys allowed into Mala's isolated home.

Mala-launched Codex workers need enough user config for authentication
to keep working (notably keyring-backed ChatGPT credentials), but they
must not inherit user plugin/hook/MCP/tooling config. Inheriting those
tables lets interactive local Codex extensions (for example stop hooks
that wait for a review gate) run inside unattended Mala workers and can
block completion after the model has already emitted its final marker.

Keep this allowlist intentionally small and auth-scoped. Mala supplies
its own per-worker model/sandbox/approval/cwd values through the SDK and
writes its own plugin/hook trust state into the isolated config later.
"""


def _codex_auth_present() -> bool:
    """Return True iff Codex has a usable credential at session-start.

    Mirrors the credential discovery order Codex itself walks
    (``codex-rs/login/src/auth/manager.rs``): a non-empty
    ``OPENAI_API_KEY`` / ``CODEX_API_KEY`` / ``CODEX_ACCESS_TOKEN`` env
    var is accepted directly, otherwise the on-disk
    ``$CODEX_HOME/auth.json`` written by ``codex login`` is the source
    of truth (``codex-rs/login/src/auth/auth_tests.rs:57-114``).
    When ``cli_auth_credentials_store = "keyring"`` is configured in
    ``$CODEX_HOME/config.toml``, Codex's keyring backend deletes
    ``auth.json`` after a successful save, so neither env vars nor the
    file may be present even though Codex itself has a valid stored
    credential — the probe defers to Codex's auth manager in that case
    by treating the keyring opt-in as authenticated.
    Returning False fails closed via
    :class:`CodexNotInstalledError` so unattended ``coder=codex`` runs
    do not silently spawn ``codex app-server`` only to crash on the
    first turn.

    The probe is non-invasive: it does not import the Codex SDK,
    spawn a subprocess, or trigger an interactive ``Sign in with
    ChatGPT`` flow. That matters because ``install_prerequisites`` is
    called on the orchestrator's hot path before any thread starts;
    side effects there would race concurrent agents.
    """
    from src.infra.clients.codex_plugin_installer import _resolve_codex_home

    for var in _CODEX_AUTH_ENV_VARS:
        if (os.environ.get(var) or "").strip():
            return True
    codex_home = _resolve_codex_home()
    if (codex_home / "auth.json").is_file():
        return True
    return _codex_uses_keyring_credentials_store(codex_home)


def _codex_uses_keyring_credentials_store(codex_home: Path) -> bool:
    """Return True iff ``config.toml`` opts into the keyring backend.

    Codex's ``cli_auth_credentials_store = "keyring"`` knob switches
    credential storage from ``auth.json`` to the OS keyring; on
    successful keyring save the Rust auth manager removes the on-disk
    file, so a fully authenticated user can have neither the file nor
    any auth env var. Detecting the opt-in here lets the probe defer
    the actual credential read to Codex itself (which the probe cannot
    perform without importing the SDK or invoking the CLI) instead of
    incorrectly raising :class:`CodexNotInstalledError`.

    Failures to read or parse ``config.toml`` are swallowed and treated
    as "no keyring opt-in" so a malformed config file falls back to the
    file/env probe rather than masking a genuinely missing credential.
    """
    from src.infra.clients.codex_plugin_installer import _HOOK_CONFIG_FILENAME

    config_path = codex_home / _HOOK_CONFIG_FILENAME
    if not config_path.is_file():
        return False
    import tomllib

    try:
        raw = config_path.read_bytes()
        parsed = tomllib.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return False
    value = parsed.get("cli_auth_credentials_store")
    return isinstance(value, str) and value.strip().lower() == "keyring"


def _render_toml_seed_scalar(value: object) -> str | None:
    """Render a small TOML scalar for the isolated Codex config seed.

    The auth seed allowlist currently contains string-valued Codex
    settings, but accepting primitive scalars keeps the helper tolerant
    if Codex adds a boolean/int auth knob later. Complex tables and
    arrays are deliberately skipped: copying tables is exactly how user
    ``[plugins]`` / ``[hooks]`` / ``[mcp_servers]`` config leaked into
    unattended workers.
    """
    if isinstance(value, str):
        import json as _json

        return _json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    return None


def _build_isolated_codex_config_seed(user_config: Path) -> str:
    """Return the sanitized ``config.toml`` seed for Mala's ``CODEX_HOME``.

    The prior implementation copied the user's whole ``config.toml`` into
    the temporary home and then appended Mala's plugin/trust entries. That
    preserved unrelated user plugins and hook trust state; when a local
    plugin registered a long-running ``Stop`` hook, Mala's noninteractive
    Codex worker could hang after emitting a final result because Codex
    waited for that user hook to finish. This seed copies only the
    top-level auth-scoped scalars in
    :data:`_CODEX_CONFIG_AUTH_SEED_KEYS`; everything else is recreated by
    Mala or left to Codex defaults.

    Read/parse failures are best-effort and return an empty seed, matching
    the existing isolation behavior where auth was probed against the
    user's real home before the isolated home was allocated.
    """
    import tomllib

    try:
        parsed = tomllib.loads(user_config.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return ""

    lines: list[str] = []
    for key in _CODEX_CONFIG_AUTH_SEED_KEYS:
        rendered = _render_toml_seed_scalar(parsed.get(key))
        if rendered is not None:
            lines.append(f"{key} = {rendered}\n")
    if not lines:
        return ""
    return (
        "# Generated by mala for isolated Codex workers. Only auth-related "
        "user config is copied.\n" + "".join(lines)
    )


# ---------------------------------------------------------------------------
# Trusted-hash auto-trust helpers (Phase E6, decision #16)
# ---------------------------------------------------------------------------
#
# The TOML rewriting helpers (``_resolve_codex_home``, ``_plugin_id``,
# ``_hook_state_key``, ``_write_codex_plugin_config``,
# ``_ensure_key_in_section``, ``_rewrite_toml_block``) and the supporting
# constants live in :mod:`src.infra.clients.codex_plugin_installer`. The
# provider delegates to that module via lazy imports inside the call
# sites so importing :mod:`codex_provider` does not pull the installer
# on the cold path (see
# ``test_importing_codex_provider_does_not_pull_codex_app_server``).
#
# The default selftest probe, live ``plugin/list`` + hook-dispatch
# probes, hook-identity hashing, hooks.json declarators, and the
# probe-time interpreter resolver live in
# :mod:`src.infra.clients.codex_selftest`. The provider remains the
# single source of truth for :class:`CodexHookNotActiveError` /
# :class:`CodexHookNotActiveReason` so callers that import the error
# types by name keep working unchanged.


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class CodexAgentProvider:
    """:class:`AgentProvider` for the Codex coder backend (T010, T013, T015).

    Conforms structurally to
    :class:`src.core.protocols.agent_provider.AgentProvider`. After T010
    ``client_factory`` and ``runtime_builder`` are real; after T013
    ``evidence_provider`` is real; after T015 (this issue)
    ``install_prerequisites`` runs the bundled-plugin install + hook
    self-test and is fail-closed on every documented signature.

    Attributes:
        name: Provider identifier (always ``"codex"``).
        client_factory: :class:`SDKClientFactoryProtocol` whose
            ``create`` returns a :class:`CodexClient` bound to the
            given :class:`CodexRuntime`.
        evidence_provider: :class:`CodexEvidenceProvider` reading the
            per-thread tee'd notification stream.
    """

    name: Literal["codex"] = "codex"

    def __init__(
        self,
        *,
        model: str = DEFAULT_CODEX_MODEL,
        effort: str | None = DEFAULT_CODEX_EFFORT,
        approval_policy: Literal[
            "never", "on-request", "on-failure", "untrusted"
        ] = DEFAULT_CODEX_APPROVAL_POLICY,
        sandbox: Literal[
            "read-only", "workspace-write", "danger-full-access"
        ] = DEFAULT_CODEX_SANDBOX,
        mcp_servers: tuple[tuple[str, object], ...] = (),
        selftest_probe: Callable[[Path, dict[str, str], str], None] | None = None,
    ) -> None:
        """Initialize the provider.

        Individual ``model`` / ``effort`` / ``approval_policy`` / ``sandbox``
        kwargs let callers construct the provider without importing
        ``src.infra.io.config``, preserving the import-linter contract that
        ``src.infra.clients`` does not depend on ``src.infra.io``. The
        :func:`src.orchestration.factory._create_agent_provider` branch
        unpacks ``MalaConfig.coder_options.codex`` into these kwargs so
        the resolved options reach the provider unchanged.

        ``selftest_probe`` is the runtime hook used by
        :meth:`install_prerequisites` to verify the bundled safety hook
        is loaded. The default probe statically validates the on-disk
        plugin tree (manifest path links + hooks.json declarations);
        tests inject custom probes to drive each
        :class:`CodexHookNotActiveReason` from
        :class:`CodexHookNotActiveError`, including the runtime-only
        reasons (HOOK_MARKER_MISSING / VERSION_MISMATCH /
        PLUGIN_DISABLED / TRUSTED_HASH_MISMATCH).
        """
        self._model: str = model
        self._effort: str | None = effort
        self._approval_policy: Literal[
            "never", "on-request", "on-failure", "untrusted"
        ] = approval_policy
        self._sandbox: Literal["read-only", "workspace-write", "danger-full-access"] = (
            sandbox
        )
        self._mcp_servers: tuple[tuple[str, object], ...] = mcp_servers
        self._selftest_probe: Callable[[Path, dict[str, str], str], None] = (
            selftest_probe if selftest_probe is not None else _default_selftest_probe
        )
        self._client_factory_cached: _CodexClientFactory | None = None
        self._evidence_provider_cached: CodexEvidenceProvider | None = None
        # In-memory selftest cache keyed on the installed plugin hash. Plan
        # E5 invokes selftest at most once per run; the key matches the
        # Amp provider's ``(coder_version, plugin_hash)`` pattern keyed
        # with a placeholder coder version. The cache holder comes from
        # the shared :mod:`_plugin_provider_base` driver so the cache
        # mutation rules stay in lockstep with the Amp provider's
        # post-T_B6 migration.
        self._selftest_cache: SelftestCache = SelftestCache()
        # Per-provider isolated ``CODEX_HOME`` populated by
        # :meth:`install_prerequisites`. Codex's plugin tree and hook
        # trust state are global to a ``CODEX_HOME``; writing the
        # bundled ``mala-safety`` plugin into the user's real
        # ``~/.codex`` makes normal, non-mala Codex sessions discover
        # the safety hook too. Routing every mala Codex invocation
        # through a provider-private CODEX_HOME keeps the plugin scoped
        # to the Codex app-server subprocesses mala launches, while
        # also preserving the previous user-MCP race fix (each
        # invocation owns its merged ``.mcp.json`` and ``config.toml``).
        # ``None`` until :meth:`install_prerequisites` allocates it.
        self._isolated_codex_home: tempfile.TemporaryDirectory[str] | None = None

    # ------------------------------------------------------------------
    # AgentProvider protocol surface
    # ------------------------------------------------------------------

    @property
    def client_factory(self) -> SDKClientFactoryProtocol:
        if self._client_factory_cached is None:
            self._client_factory_cached = _CodexClientFactory()
        return cast("SDKClientFactoryProtocol", self._client_factory_cached)

    @property
    def evidence_provider(self) -> EvidenceProvider:
        if self._evidence_provider_cached is None:
            self._evidence_provider_cached = CodexEvidenceProvider()
        return cast("EvidenceProvider", self._evidence_provider_cached)

    @property
    def model(self) -> str:
        """Resolved Codex model carried by this provider."""
        return self._model

    @property
    def effort(self) -> str | None:
        """Resolved Codex reasoning effort."""
        return self._effort

    @property
    def approval_policy(
        self,
    ) -> Literal["never", "on-request", "on-failure", "untrusted"]:
        """Resolved Codex approval policy."""
        return self._approval_policy

    @property
    def sandbox(
        self,
    ) -> Literal["read-only", "workspace-write", "danger-full-access"]:
        """Resolved Codex sandbox mode."""
        return self._sandbox

    def runtime_builder(
        self,
        repo_path: Path,
        agent_id: str,
        *,
        mcp_server_factory: McpServerFactory,
        deadlock_monitor: DeadlockMonitorProtocol | None = None,
    ) -> CoderRuntimeBuilder:
        """Construct a per-session :class:`CodexRuntimeBuilder`.

        Threads the resolved Codex options
        (``MalaConfig.coder_options.codex.*``) and the injected
        ``mcp_server_factory`` into the builder. ``deadlock_monitor``
        is accepted for protocol parity (plan A6); Phase C does not
        wire it because Codex's lock-event surface ships with the
        Phase E hook + Phase G MCP, which arrive after T010.
        """
        del deadlock_monitor
        # Lazy import so module-load of ``codex_provider`` does not
        # transitively reach ``src.infra.hooks`` via the runtime's
        # ``LintCache`` carrier (the import-linter contract is OK with
        # the codex_runtime exception, but keeping this lazy mirrors
        # the Amp path's posture and keeps the ``coder=claude`` /
        # ``coder=amp`` cold-path unchanged unless Codex is selected.
        from src.infra.clients.codex_runtime import CodexRuntimeBuilder

        builder = CodexRuntimeBuilder(
            repo_path,
            agent_id,
            mcp_server_factory,
            model=self._model,
            effort=self._effort,
            approval_policy=self._approval_policy,
            sandbox=self._sandbox,
        )
        # Thread the per-provider isolated ``CODEX_HOME`` into the
        # runtime's per-process env so the spawned ``codex app-server``
        # reads the mala-scoped plugin tree from the isolated location
        # instead of the user's normal Codex home. Without this overlay
        # the installation would be isolated on disk but Codex would
        # still inherit the user's ``CODEX_HOME`` at runtime.
        if self._isolated_codex_home is not None:
            builder.with_env(
                extra={"CODEX_HOME": str(Path(self._isolated_codex_home.name))}
            )
        return cast("CoderRuntimeBuilder", builder)

    def mcp_server_factory(self) -> McpServerFactory:
        """Return the Codex-shaped MCP server factory (Phase G3, T016).

        The factory closes over ``self._mcp_servers`` (user-supplied
        ``coder_options.codex.mcp_servers``) and merges them with the
        bundled ``mala-locking`` launch spec; the bundled key is always
        last-wins so user config cannot override it (plan G3).
        """
        return _create_codex_mcp_server_factory(self._mcp_servers)

    def _ensure_isolated_codex_home(self, user_codex_home: Path) -> Path:
        """Lazily allocate this provider's private ``CODEX_HOME`` and
        seed it with the user's auth credential.

        The plugin tree + ``config.toml`` Codex reads at
        ``thread_start`` live under ``$CODEX_HOME``. Installing Mala's
        safety plugin into the user's real Codex home makes it visible
        to ordinary Codex CLI sessions, where ``MALA_AGENT_ID`` /
        ``MALA_LOCK_DIR`` are absent and the hook fails closed on
        edits. Allocating a per-provider ``TemporaryDirectory`` whose
        path is fed to the spawned Codex via :meth:`runtime_builder`
        scopes the plugin and trust entries to this mala run only.

        The same isolation also removes the user-MCP race the prior
        design fixed only for ``coder_options.codex.mcp_servers``: a
        concurrent mala invocation with a different MCP config cannot
        overwrite this invocation's merged ``.mcp.json`` because no
        other process can name this directory.

        Auth.json is symlinked from the user's real ``$CODEX_HOME`` so
        Codex's auth manager finds the credential at the same logical
        path it would otherwise read directly. When ``OPENAI_API_KEY`` /
        ``CODEX_API_KEY`` / ``CODEX_ACCESS_TOKEN`` is set the env-var
        path covers auth and the symlink is unnecessary, but we
        attempt it regardless so a refresh-token rotation by Codex
        (which writes back to ``auth.json``) lands in the user's real
        file rather than the temp dir.

        ``config.toml`` is **sanitized** (not symlinked or copied
        wholesale) from the user's real ``$CODEX_HOME``. Only
        auth-scoped top-level scalars such as
        ``cli_auth_credentials_store = "keyring"`` carry into the
        isolated home. Without this seed, a user who logged in via
        Codex's keyring backend has
        neither ``auth.json`` (the keyring backend deletes it after a
        successful save) nor the keyring opt-in inside the isolated
        home — the spawned Codex falls back to the default
        ``auth_json`` credential store, finds no file, and crashes at
        ``thread_start`` instead of using the stored keyring
        credential. Sanitizing (rather than symlinking or copying) is
        required because :func:`_write_codex_plugin_config` writes to
        this file later and because third-party user plugin/hook tables
        must not enter Mala's unattended worker config.

        The :class:`tempfile.TemporaryDirectory` handle is held on the
        provider so the directory survives for the orchestrator's
        lifetime; cleanup happens when the provider is garbage-collected
        or the process exits, mirroring the lifecycle of a single mala
        invocation.
        """
        from src.infra.clients.codex_plugin_installer import _HOOK_CONFIG_FILENAME

        if self._isolated_codex_home is None:
            self._isolated_codex_home = tempfile.TemporaryDirectory(
                prefix="mala-codex-home-"
            )
            isolated = Path(self._isolated_codex_home.name)
            user_auth = user_codex_home / "auth.json"
            target_auth = isolated / "auth.json"
            if user_auth.is_file():
                try:
                    target_auth.symlink_to(user_auth)
                except OSError:
                    # Filesystems without symlink support (rare on POSIX,
                    # possible on Windows-mounted volumes) fall back to
                    # a copy. The copy will diverge if Codex refreshes
                    # the user's auth.json mid-run; that is the trade-off
                    # for those filesystems and matches Amp's posture.
                    try:
                        shutil.copy2(user_auth, target_auth)
                    except OSError:
                        # Best-effort: if even copy fails, leave the
                        # isolated home auth-less and let the spawned
                        # Codex surface the auth error itself. The
                        # provider's auth probe ran before this point
                        # against the user's real home, so the failure
                        # here would only manifest at thread_start.
                        pass
            user_config = user_codex_home / _HOOK_CONFIG_FILENAME
            if user_config.is_file():
                try:
                    # Write a sanitized config seed rather than copying
                    # the user's full config.toml. That preserves the
                    # auth knobs Codex needs (notably keyring-backed
                    # credentials) while stripping user plugins, hooks,
                    # MCP servers, shell env policies, and other
                    # interactive local extensions that can hang Mala's
                    # noninteractive workers. ``write_text`` also keeps
                    # the isolated copy writable even when the user's
                    # real config is managed read-only by dotfile tools.
                    config_seed = _build_isolated_codex_config_seed(user_config)
                    if config_seed:
                        (isolated / _HOOK_CONFIG_FILENAME).write_text(
                            config_seed, encoding="utf-8"
                        )
                except OSError:
                    # Best-effort: a missing seed here means
                    # ``_write_codex_plugin_config`` writes a
                    # plugin-only file and a keyring user loses the
                    # opt-in. The auth probe ran against the user's
                    # real home before this point so the failure mode
                    # surfaces only at thread_start; matches the
                    # auth.json copy fallback above.
                    pass
        return Path(self._isolated_codex_home.name)

    def install_prerequisites(
        self,
        repo_path: Path,
        *,
        mcp_server_factory: McpServerFactory,
    ) -> None:
        """Install the bundled Codex plugin and run the hook self-test (T015).

        Idempotent within a run: the result is cached on the installed
        plugin hash; subsequent calls within the same run short-circuit
        without re-running the install or the selftest.

        Steps (plan ``L778-L924``):

          1. Verify ``codex_app_server`` SDK is importable (raise
             :class:`CodexNotInstalledError` if absent).
          2. Verify the ``codex`` binary is on ``PATH`` (raise
             :class:`CodexHookNotActiveError(CODEX_BINARY_MISSING)`).
          3. Verify Codex auth state is present — either an
             ``OPENAI_API_KEY`` / ``CODEX_API_KEY`` /
             ``CODEX_ACCESS_TOKEN`` env var is set, or
             ``$CODEX_HOME/auth.json`` exists (raise
             :class:`CodexNotInstalledError` pointing at ``codex login``
             otherwise). Plan I1 / decision #8: missing auth fails
             closed before any ``codex app-server`` spawn so unattended
             runs never trip an interactive ``Sign in with ChatGPT``
             prompt.
          4. Run :class:`CodexPluginInstaller` to copy the bundled plugin
             tree into a per-run isolated Codex home at
             ``<isolated CODEX_HOME>/plugins/cache/<marketplace>/<plugin>/<version>/.codex-plugin/``
             (per ``codex-rs/core-plugins/src/store.rs::PluginStore.plugin_root``).
          5. Verify the ``mala-codex-pre-tool-use`` console script is on
             ``PATH`` (raise
             :class:`CodexHookNotActiveError(SCRIPT_MISSING)`).
          6. Auto-write the five Codex-config preconditions to the
             isolated ``CODEX_HOME/config.toml`` (decision #16):
             ``[features] plugins = true`` to override an opt-out
             setting that would short-circuit plugin loading entirely
             (``plugins_for_config_with_force_reload`` early-returns
             when ``plugins_enabled`` is false);
             ``[features] plugin_hooks = true`` so Codex's
             ``catalog_processor`` actually loads plugin-bundled hooks
             (this feature ships ``default_enabled = false``);
             ``[features] hooks = true`` so Codex's global hook
             execution gate cannot leave the loaded/trusted safety hook
             dormant;
             ``[plugins."<plugin_id>"] enabled = true`` so Codex's
             ``configured_plugins_from_stack`` enumerates the plugin;
             and
             ``[hooks.state."<plugin_id>:.codex-plugin/hooks.json:pre_tool_use:0:0"]``
             with ``enabled = true`` + ``trusted_hash`` so the hook is
             marked Trusted. I/O failures raise
             :class:`CodexHookNotActiveError(TRUSTED_HASH_MISMATCH)`.
          7. Run the selftest probe (default: structural verification
             of the installed plugin tree at the install target); it
             raises :class:`CodexHookNotActiveError` on
             ``HOOK_MARKER_MISSING`` / ``VERSION_MISMATCH`` /
             ``PLUGIN_DISABLED`` / ``TRUSTED_HASH_MISMATCH``. Tests
             inject custom probes to drive the runtime-only reasons.

        Args:
            repo_path: Working directory the selftest probe should use
                (mirrors the runtime_builder shape).
            mcp_server_factory: Codex-shaped MCP factory threaded
                through the selftest so the probe can build a runtime
                that matches a real session's MCP wiring.

        Raises:
            CodexNotInstalledError: When the Codex SDK is not
                importable or when no Codex credential is detectable.
            CodexHookNotActiveError: When any structural or runtime
                fail-closed signature fires.
        """
        del (
            mcp_server_factory
        )  # threaded for protocol parity; selftest probe doesn't need it directly

        # 1. SDK importability — module-spec lookup avoids actually
        # importing ``codex_app_server`` so the lazy-import contract
        # (plan L733) is preserved on the success path. ``find_spec``
        # itself can import parent packages but does not load the
        # module body.
        try:
            sdk_spec = importlib.util.find_spec("codex_app_server")
        except (ImportError, ValueError):
            sdk_spec = None
        if sdk_spec is None:
            raise CodexNotInstalledError(
                "codex_app_server SDK is not importable. Install it with "
                "`uv add openai-codex-app-server-sdk` and rerun mala. "
                "See the docs for the full Codex prerequisites."
            )

        # 2. Codex runtime resolvable. Either an external ``codex`` binary
        # on PATH (passed explicitly as ``AppServerConfig.codex_bin``), or
        # the SDK-bundled ``codex_cli_bin`` runtime package (which
        # ``codex_app_server.client.resolve_codex_bin`` falls back to when
        # ``AppServerConfig.codex_bin`` is None). Honors ``CODEX_BINARY`` so
        # an explicit operator override skips both lookups.
        if not _codex_runtime_resolvable():
            raise CodexHookNotActiveError(
                "Codex runtime not resolvable: no `codex` binary on PATH, "
                "no `CODEX_BINARY` override, and the bundled "
                "`codex_cli_bin` runtime package is not importable. "
                "Install `openai-codex-cli-bin` (typically as a dependency "
                "of `openai-codex-app-server-sdk`) or place a `codex` "
                "binary on PATH.",
                reason=CodexHookNotActiveReason.CODEX_BINARY_MISSING,
            )

        # 3. Codex auth state. Without a credential the spawned
        # ``codex app-server`` would either fail the first turn with an
        # opaque RPC error or — worse, on a fresh machine — pop the
        # interactive ``Sign in with ChatGPT`` flow. Both are
        # unacceptable on the unattended orchestrator path; fail closed
        # here with the install-and-login command per decision #8 / AC
        # #14.
        if not _codex_auth_present():
            from src.infra.clients.codex_plugin_installer import (
                _resolve_codex_home as _resolve_codex_home_for_msg,
            )

            raise CodexNotInstalledError(
                "Codex auth missing: none of `OPENAI_API_KEY`, "
                "`CODEX_API_KEY`, or `CODEX_ACCESS_TOKEN` is set, "
                f"`{_resolve_codex_home_for_msg() / 'auth.json'}` does not exist, "
                'and `cli_auth_credentials_store = "keyring"` is not '
                "configured in `config.toml`. "
                "Run `codex login` (Sign in with ChatGPT) or set "
                "`OPENAI_API_KEY` to an OpenAI API key, then rerun mala. "
                "See https://developers.openai.com/codex/auth for the "
                "full auth setup."
            )

        # 4-6. Plugin install + structural script-on-PATH check + config
        # writes are bundled into a single ``install`` closure handed to
        # the shared :func:`install_with_selftest` driver. The closure
        # returns the plugin hash that keys both the in-memory cache and
        # the selftest's expected version marker; the driver
        # short-circuits on a matching ``(coder_version, plugin_hash)``
        # key before invoking the probe so a long-running orchestrator
        # pays the install + selftest at most once per run.
        from src.infra.clients.codex_plugin_installer import (
            PLUGIN_DIRNAME,
            CodexPluginInstallError,
            CodexPluginInstaller,
            _resolve_codex_home,
            _write_codex_plugin_config,
            plugin_root_dir,
        )

        # Choose the active ``CODEX_HOME`` for the rest of install.
        # Always allocate a provider-private isolated home so the
        # bundled plugin + hook trust entries are visible only to Codex
        # app-server subprocesses launched by this mala provider, never
        # to the user's ordinary Codex CLI sessions. The isolated path
        # is also threaded into runtime env via :meth:`runtime_builder`.
        user_codex_home = _resolve_codex_home()
        codex_home = self._ensure_isolated_codex_home(user_codex_home)
        install_target = plugin_root_dir(codex_home) / PLUGIN_DIRNAME

        def _install() -> str:
            # Build the merged ``.mcp.json`` payload (Phase G3 / AC-3)
            # and hand it to the installer as ``mcp_json_override`` so
            # the installer is the sole writer of the plugin tree.
            # Routing the merge through the installer keeps idempotency
            # intact (a rerun with the same merged bytes short-circuits
            # at ``action="skipped"``) and — combined with the
            # per-provider isolated ``CODEX_HOME`` above — closes the
            # cross-process race the prior post-install rewrite (and
            # the prior single-shared-CODEX_HOME design) opened:
            # without isolation, Process B's installer would silently
            # revert Process A's merged file to a bundled-only payload
            # because B's own user config builds a different override;
            # with isolation, B writes to its own ``CODEX_HOME`` and
            # A's plugin tree is untouched. The same isolation now
            # applies to the default bundled-only path, so a mala run
            # never mutates the user's normal Codex plugin state (plan
            # ``L954``: bundled is mandatory, never replaced;
            # non-conflicting user keys pass through).
            mcp_json_override = _build_merged_codex_plugin_mcp_json(self._mcp_servers)
            try:
                install_result = CodexPluginInstaller(
                    mcp_json_override=mcp_json_override
                ).install(target_dir=install_target)
            except CodexPluginInstallError as exc:
                raise CodexHookNotActiveError(
                    f"Codex plugin install failed: {exc}",
                    reason=CodexHookNotActiveReason.PLUGIN_DISABLED,
                ) from exc

            # Hook script must be on PATH so Codex can launch it via
            # ``"command": "mala-codex-pre-tool-use"`` from hooks.json.
            if shutil.which("mala-codex-pre-tool-use") is None:
                raise CodexHookNotActiveError(
                    "mala-codex-pre-tool-use console script not on PATH. "
                    "Ensure mala is installed via `uv tool install mala-agent` "
                    "or `uv sync` so the project.scripts entry point is exposed.",
                    reason=CodexHookNotActiveReason.SCRIPT_MISSING,
                )

            # Auto-write the five config-side preconditions to
            # ``<isolated codex_home>/config.toml`` (decision #16). Codex
            # requires ALL FIVE for the hook to fire:
            # ``[features] plugins = true`` to override an opt-out user
            # config that early-returns from
            # ``plugins_for_config_with_force_reload``;
            # ``[features] plugin_hooks = true`` to enable plugin-bundled
            # hook loading (default-off in upstream Codex);
            # ``[features] hooks = true`` to enable global hook
            # execution; ``[plugins."<id>"] enabled = true`` to surface
            # the plugin via ``configured_plugins_from_stack``; and
            # ``[hooks.state."<id>:..."]`` with the matching
            # ``trusted_hash`` to mark the hook Trusted. ``codex_home``
            # here is the *active* home (isolated when user MCP is
            # configured, user's real home otherwise) so the writes
            # land in the same directory the spawned Codex will read.
            _write_codex_plugin_config(codex_home=codex_home)

            return install_result.plugin_hash

        def _selftest(plugin_hash: str) -> None:
            # The probe contract raises CodexHookNotActiveError on any
            # fail-closed signature; default probe is a no-op. Tests
            # inject custom probes that simulate each Reason.
            env_overlay: dict[str, str] = {
                "CODEX_HOME": str(codex_home),
            }
            self._selftest_probe(repo_path, env_overlay, plugin_hash)

        # ``coder_version`` placeholder "unknown" matches the Amp
        # provider's posture until a reliable ``codex --version`` parse
        # is added (plan ``L121`` / ``L823`` shared cache shape).
        install_with_selftest(
            install=_install,
            selftest=_selftest,
            cache=self._selftest_cache,
        )
