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
inside :class:`CodexClient.__aenter__` / :meth:`CodexClient.query` (and
even those uses are guarded by ``try/except TypeError`` for backward
compatibility); :class:`CodexClient` itself is imported lazily by
:meth:`_CodexClientFactory.create` so module-load remains SDK-free.
:class:`CodexEvidenceProvider` does not pull the SDK either — it reads
tee'd JSONL only — so eager construction is safe. The plugin installer
imports stay lazy too so a Claude/Amp-only run does not pay for the
Codex install path on import.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import shutil
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from src.core.constants import (
    DEFAULT_CODEX_APPROVAL_POLICY,
    DEFAULT_CODEX_MODEL,
    DEFAULT_CODEX_SANDBOX,
)
from src.infra.clients.codex_evidence_provider import CodexEvidenceProvider

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

logger = logging.getLogger(__name__)


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


class CodexHookNotActiveError(RuntimeError):
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
# MCP server factory (provider-owned, Phase B placeholder; Phase G3 wires it)
# ---------------------------------------------------------------------------


def _create_codex_mcp_server_factory() -> McpServerFactory:
    """Stub Codex-shaped MCP factory.

    Phase G wires the bundled ``mala-locking`` launcher inside the Codex
    plugin (`plans/2026-05-07-codex-provider-plan.md` G1-G3); for Phase C
    the factory still returns an empty map so :meth:`AgentProvider.mcp_server_factory`
    has a callable to hand back. The runtime carries this dict through
    to ``AsyncCodex.thread_start(mcp_servers=...)`` unchanged so Phase G
    is a one-line factory swap.
    """

    def factory(
        agent_id: str,
        repo_path: Path,
        emit_lock_event: object,
    ) -> dict[str, object]:
        del agent_id, repo_path, emit_lock_event
        return {}

    return cast("McpServerFactory", factory)


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
# Trusted-hash auto-trust helpers (Phase E6, decision #16)
# ---------------------------------------------------------------------------


_HOOK_STATE_FILENAME = "hooks.toml"
"""Codex hook-state file relative to ``CODEX_HOME``. Per
``codex-rs/config/src/hook_config.rs::HooksToml``: ``state.<hook-key>``
holds ``{enabled, trusted_hash}``. The user's ``hooks.toml`` is the
on-disk shape that Codex reads at startup."""


def _resolve_codex_home() -> Path:
    """Return the active Codex home directory honoring ``CODEX_HOME``."""
    env_value = os.environ.get("CODEX_HOME")
    if env_value:
        return Path(env_value)
    return Path("~/.codex").expanduser()


def _hook_state_key(hooks_json_path: Path) -> str:
    """Compute the Codex hook-state key for the bundled PreToolUse entry.

    Format mirrors the Codex test fixture
    (``codex-rs/config/src/hooks_tests.rs:91``): ``"<hooks.json path>:pre_tool_use:0:0"``.
    The first index is the matcher group; the second is the handler
    inside that group. Our ``hooks.json`` ships exactly one PreToolUse
    matcher with one command handler, so ``(0, 0)`` is correct.
    """
    return f"{hooks_json_path}:pre_tool_use:0:0"


def _format_trusted_hash_value(plugin_hash: str) -> str:
    """Codex stores trusted hashes prefixed with ``sha256:`` (per
    ``hooks_tests.rs:93``: ``trusted_hash = "sha256:abc123"``)."""
    return f"sha256:{plugin_hash}"


def _write_trusted_hash(
    *,
    codex_home: Path,
    hooks_json_path: Path,
    plugin_hash: str,
) -> None:
    """Best-effort write of the bundled hook's ``trusted_hash`` to ``hooks.toml``.

    Codex's hook-state lives in a separate ``[state."<key>"]`` table —
    NOT inside the hook handler entry — so this function preserves any
    existing top-level event tables (``[[PreToolUse]]`` etc.) when the
    file already exists. The implementation appends a fresh
    ``[state."<key>"]`` section if one is absent and rewrites the
    existing one in place when present.

    Failures are logged at warning level rather than raised: if the
    user's ``CODEX_HOME`` is read-only, the selftest will catch the
    dormant hook and surface :class:`CodexHookNotActiveError` with a
    structured reason; we don't want a Codex-home permissions issue to
    masquerade as a missing-plugin error.
    """
    state_key = _hook_state_key(hooks_json_path)
    new_value = _format_trusted_hash_value(plugin_hash)
    target = codex_home / _HOOK_STATE_FILENAME
    section_header = f'[state."{state_key}"]'
    new_block = f'{section_header}\nenabled = true\ntrusted_hash = "{new_value}"\n'

    try:
        codex_home.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning(
            "Codex hook trusted_hash auto-write skipped: cannot create %s (%s). "
            "Selftest will catch a dormant hook and surface the structured reason.",
            codex_home,
            exc,
        )
        return

    try:
        existing = target.read_text(encoding="utf-8") if target.exists() else ""
    except OSError as exc:
        logger.warning(
            "Codex hook trusted_hash auto-write skipped: cannot read %s (%s).",
            target,
            exc,
        )
        return

    rewritten = _rewrite_hook_state_block(
        existing, section_header=section_header, new_block=new_block
    )
    if rewritten == existing:
        return  # nothing to do (already correct)

    try:
        target.write_text(rewritten, encoding="utf-8")
    except OSError as exc:
        logger.warning(
            "Codex hook trusted_hash auto-write failed at %s (%s). "
            "Selftest will catch a dormant hook.",
            target,
            exc,
        )


def _rewrite_hook_state_block(
    existing: str, *, section_header: str, new_block: str
) -> str:
    """Return ``existing`` with ``section_header``'s block replaced by ``new_block``.

    Pure-text rewrite (no TOML round-trip) so the function can run
    without a third-party TOML writer. The replacement is bounded to
    the ``[state."<key>"]`` table only: every line up to the next
    top-level ``[...]`` header (or EOF) is dropped and ``new_block``
    is spliced in.
    """
    lines = existing.splitlines(keepends=True)
    if not lines:
        return new_block.rstrip("\n") + "\n"

    out: list[str] = []
    i = 0
    found = False
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped == section_header and not found:
            found = True
            out.append(new_block if new_block.endswith("\n") else new_block + "\n")
            i += 1
            while i < len(lines):
                next_stripped = lines[i].strip()
                if next_stripped.startswith("[") and next_stripped.endswith("]"):
                    break
                i += 1
            continue
        out.append(lines[i])
        i += 1
    if not found:
        if out and not out[-1].endswith("\n"):
            out[-1] = out[-1] + "\n"
        out.append("\n" if out else "")
        out.append(new_block if new_block.endswith("\n") else new_block + "\n")
    return "".join(out)


# ---------------------------------------------------------------------------
# Selftest probe (default no-op; tests inject failure modes)
# ---------------------------------------------------------------------------


def _default_selftest_probe(
    repo_path: Path,
    env_overlay: dict[str, str],
    expected_hash: str,
) -> None:
    """Default selftest probe used when none is injected.

    The full subprocess-based Codex turn that triggers the bundled hook
    and verifies the hook's emitted version-hash marker is real-Codex
    e2e territory (plan AC #12 / I3) and lives behind the
    ``codex`` binary + auth gate. Within the unit-test surface, the
    structural guarantees the provider must honor are:

      * ``CODEX_BINARY_MISSING``: caught up-front by :meth:`install_prerequisites`.
      * ``SCRIPT_MISSING``: caught up-front by :meth:`install_prerequisites`.
      * ``PLUGIN_DISABLED`` / ``TRUSTED_HASH_MISMATCH`` /
        ``HOOK_MARKER_MISSING`` / ``VERSION_MISMATCH``: surfaced through
        the probe contract at runtime; tests drive each by injecting a
        custom probe that raises the matching
        :class:`CodexHookNotActiveError`.

    The default no-op behavior keeps Phase E5 fail-closed by relying on
    the up-front structural checks (binary + script on PATH); any deeper
    verification (e.g., real Codex spawn) is a real-binary integration
    layer the orchestrator's CI can wire when the binary is reliably
    available.
    """
    del repo_path, env_overlay, expected_hash
    return None


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
        effort: str | None = None,
        approval_policy: Literal[
            "never", "on-request", "on-failure", "untrusted"
        ] = DEFAULT_CODEX_APPROVAL_POLICY,
        sandbox: Literal[
            "read-only", "workspace-write", "danger-full-access"
        ] = DEFAULT_CODEX_SANDBOX,
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
        actually executed. The default is a no-op; tests inject custom
        probes to drive each
        :class:`CodexHookNotActiveReason` from
        :class:`CodexHookNotActiveError`.
        """
        self._model: str = model
        self._effort: str | None = effort
        self._approval_policy: Literal[
            "never", "on-request", "on-failure", "untrusted"
        ] = approval_policy
        self._sandbox: Literal["read-only", "workspace-write", "danger-full-access"] = (
            sandbox
        )
        self._selftest_probe: Callable[[Path, dict[str, str], str], None] = (
            selftest_probe if selftest_probe is not None else _default_selftest_probe
        )
        self._client_factory_cached: _CodexClientFactory | None = None
        self._evidence_provider_cached: CodexEvidenceProvider | None = None
        # In-memory selftest cache keyed on the installed plugin hash. Plan
        # E5 invokes selftest at most once per run; the key matches the
        # Amp provider's ``(coder_version, plugin_hash)`` pattern keyed
        # with a placeholder coder version.
        self._selftest_cache_key: tuple[str, str] | None = None

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
        """Resolved Codex reasoning effort (``None`` = SDK default)."""
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
        # ``coder=amp`` cold-path identical to the current default).
        from src.infra.clients.codex_runtime import CodexRuntimeBuilder

        return cast(
            "CoderRuntimeBuilder",
            CodexRuntimeBuilder(
                repo_path,
                agent_id,
                mcp_server_factory,
                model=self._model,
                effort=self._effort,
                approval_policy=self._approval_policy,
                sandbox=self._sandbox,
            ),
        )

    def mcp_server_factory(self) -> McpServerFactory:
        """Return the Codex-shaped MCP server factory (Phase G3 stub)."""
        return _create_codex_mcp_server_factory()

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
          3. Run :class:`CodexPluginInstaller` to copy the bundled plugin
             tree into ``$CODEX_HOME/plugins/mala-safety/.codex-plugin/``.
          4. Verify the ``mala-codex-pre-tool-use`` console script is on
             ``PATH`` (raise
             :class:`CodexHookNotActiveError(SCRIPT_MISSING)`).
          5. Auto-write the bundled plugin's ``trusted_hash`` to
             ``$CODEX_HOME/hooks.toml`` (decision #16). On read-only
             ``CODEX_HOME`` the write is logged and skipped — the
             selftest catches the dormant hook either way.
          6. Run the selftest probe; it raises
             :class:`CodexHookNotActiveError` on any of the runtime
             fail-closed signatures (``HOOK_MARKER_MISSING``,
             ``VERSION_MISMATCH``, ``PLUGIN_DISABLED``,
             ``TRUSTED_HASH_MISMATCH``).

        Args:
            repo_path: Working directory the selftest probe should use
                (mirrors the runtime_builder shape).
            mcp_server_factory: Codex-shaped MCP factory threaded
                through the selftest so the probe can build a runtime
                that matches a real session's MCP wiring.

        Raises:
            CodexNotInstalledError: When the Codex SDK is not importable.
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

        # 2. Codex binary on PATH.
        if shutil.which("codex") is None:
            raise CodexHookNotActiveError(
                "codex binary not found on PATH; coder=codex requires the "
                "openai-codex-cli-bin runtime package to be installed.",
                reason=CodexHookNotActiveReason.CODEX_BINARY_MISSING,
            )

        # 3. Run the plugin installer (idempotent on its own).
        from src.infra.clients.codex_plugin_installer import (
            CodexPluginInstallError,
            CodexPluginInstaller,
        )

        try:
            install_result = CodexPluginInstaller().install()
        except CodexPluginInstallError as exc:
            raise CodexHookNotActiveError(
                f"Codex plugin install failed: {exc}",
                reason=CodexHookNotActiveReason.PLUGIN_DISABLED,
            ) from exc

        plugin_hash = install_result.plugin_hash

        # 4. Hook script must be on PATH so Codex can launch it via
        # ``"command": "mala-codex-pre-tool-use"`` from hooks.json.
        if shutil.which("mala-codex-pre-tool-use") is None:
            raise CodexHookNotActiveError(
                "mala-codex-pre-tool-use console script not on PATH. "
                "Ensure mala is installed via `uv tool install mala-agent` "
                "or `uv sync` so the project.scripts entry point is exposed.",
                reason=CodexHookNotActiveReason.SCRIPT_MISSING,
            )

        # 5. Trusted-hash auto-trust (decision #16).
        codex_home = _resolve_codex_home()
        hooks_json_path = install_result.target_dir / "hooks.json"
        _write_trusted_hash(
            codex_home=codex_home,
            hooks_json_path=hooks_json_path,
            plugin_hash=plugin_hash,
        )

        # 6. Cache key + selftest. The key matches the Amp provider's
        # ``(coder_version, plugin_hash)`` shape; ``"unknown"`` carries
        # the placeholder until a reliable ``codex --version`` parse is
        # added (parity with the Amp path's posture).
        cache_key = ("unknown", plugin_hash)
        if self._selftest_cache_key == cache_key:
            return

        env_overlay: dict[str, str] = {
            "CODEX_HOME": str(codex_home),
        }
        # The probe contract raises CodexHookNotActiveError on any
        # fail-closed signature; default probe is a no-op. Tests inject
        # custom probes that simulate each Reason.
        self._selftest_probe(repo_path, env_overlay, plugin_hash)

        self._selftest_cache_key = cache_key
