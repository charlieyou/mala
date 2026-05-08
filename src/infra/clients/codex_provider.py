"""Codex :class:`AgentProvider` (Phase C, T010).

Wires :class:`CodexAgentProvider` to the real :class:`CodexClient` /
:class:`CodexRuntime` / :class:`CodexRuntimeBuilder` shipped in Phase C
(T010). Replaces the Phase B stub's ``client_factory`` / ``runtime_builder``
implementations with concrete ones; ``evidence_provider`` and
``install_prerequisites`` remain fail-closed stubs until Phase F (T013)
and Phase E + I (T014, T020) respectively.

Lazy-import contract (plan ``L733``): importing this module does NOT
transitively import ``codex_app_server``. The SDK is referenced only
inside :class:`CodexClient.__aenter__` / :meth:`CodexClient.query` (and
even those uses are guarded by ``try/except TypeError`` for backward
compatibility); :class:`CodexClient` itself is imported lazily by
:meth:`_CodexClientFactory.create` so module-load remains SDK-free.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

from src.core.constants import (
    DEFAULT_CODEX_APPROVAL_POLICY,
    DEFAULT_CODEX_MODEL,
    DEFAULT_CODEX_SANDBOX,
)

if TYPE_CHECKING:
    from pathlib import Path

    from src.core.protocols.agent_provider import CoderRuntimeBuilder
    from src.core.protocols.evidence import EvidenceProvider, JsonlEntryProtocol
    from src.core.protocols.lifecycle import DeadlockMonitorProtocol
    from src.core.protocols.sdk import (
        McpServerFactory,
        SDKClientFactoryProtocol,
        SDKClientProtocol,
    )


_STUB_MESSAGE = (
    "CodexAgentProvider stub: full implementation lands in Phases C-I. "
    "Phase C (T010) wires client_factory + runtime_builder; this entry "
    "point is still a stub and lands in a later phase."
)


class CodexNotImplementedError(NotImplementedError):
    """Raised when an unfinished Codex provider entry point is exercised.

    Phase C (T010) wires ``client_factory`` and ``runtime_builder`` to
    real implementations; ``evidence_provider`` (Phase F / T013) and
    ``install_prerequisites`` (Phase E + I / T014, T020) still raise
    this so a Codex run aborts cleanly with an actionable error
    instead of routing to a half-built pipeline.
    """


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
# Stub evidence provider (Phase F replaces)
# ---------------------------------------------------------------------------


class _CodexStubEvidenceProvider:
    """:class:`EvidenceProvider` stub for the Codex coder.

    Every call raises :class:`CodexNotImplementedError`; Phase F (T013)
    replaces this with the native ``Thread.read(include_turns=True)``
    adapter (or the tee fallback if the F1 spike disconfirms native
    viability).
    """

    def get_log_path(self, repo_path: Path, session_id: str) -> Path:
        del repo_path, session_id
        raise CodexNotImplementedError(_STUB_MESSAGE)

    def iter_session_events(self, log_path: Path, offset: int = 0) -> object:
        del log_path, offset
        raise CodexNotImplementedError(_STUB_MESSAGE)

    def iter_thread_evidence(self, log_path: Path, offset: int = 0) -> object:
        del log_path, offset
        raise CodexNotImplementedError(_STUB_MESSAGE)

    async def wait_for_session_ready(
        self,
        repo_path: Path,
        session_id: str,
        *,
        timeout: float,
        poll_interval: float = 0.5,
    ) -> None:
        del repo_path, session_id, timeout, poll_interval
        raise CodexNotImplementedError(_STUB_MESSAGE)

    def get_end_offset(self, log_path: Path, start_offset: int = 0) -> int:
        del log_path, start_offset
        raise CodexNotImplementedError(_STUB_MESSAGE)

    def extract_bash_commands(self, entry: JsonlEntryProtocol) -> list[tuple[str, str]]:
        del entry
        raise CodexNotImplementedError(_STUB_MESSAGE)

    def extract_tool_results(self, entry: JsonlEntryProtocol) -> list[tuple[str, bool]]:
        del entry
        raise CodexNotImplementedError(_STUB_MESSAGE)

    def extract_assistant_text_blocks(self, entry: JsonlEntryProtocol) -> list[str]:
        del entry
        raise CodexNotImplementedError(_STUB_MESSAGE)

    def extract_tool_result_content(
        self, entry: JsonlEntryProtocol
    ) -> list[tuple[str, str]]:
        del entry
        raise CodexNotImplementedError(_STUB_MESSAGE)


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class CodexAgentProvider:
    """Phase C :class:`AgentProvider` for the Codex coder.

    Conforms structurally to
    :class:`src.core.protocols.agent_provider.AgentProvider`. After T010
    (this issue) ``client_factory`` and ``runtime_builder`` are real;
    ``evidence_provider`` (Phase F / T013) and ``install_prerequisites``
    (Phase E + I / T014, T020) are still fail-closed stubs.

    Attributes:
        name: Provider identifier (always ``"codex"``).
        client_factory: :class:`SDKClientFactoryProtocol` whose
            ``create`` returns a :class:`CodexClient` bound to the
            given :class:`CodexRuntime`.
        evidence_provider: Stub :class:`EvidenceProvider` whose methods
            all raise :class:`CodexNotImplementedError`.
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
    ) -> None:
        """Initialize the provider.

        Individual ``model`` / ``effort`` / ``approval_policy`` / ``sandbox``
        kwargs let callers construct the provider without importing
        ``src.infra.io.config``, preserving the import-linter contract that
        ``src.infra.clients`` does not depend on ``src.infra.io``. The
        :func:`src.orchestration.factory._create_agent_provider` branch
        unpacks ``MalaConfig.coder_options.codex`` into these kwargs so
        the resolved options reach the provider unchanged.
        """
        self._model: str = model
        self._effort: str | None = effort
        self._approval_policy: Literal[
            "never", "on-request", "on-failure", "untrusted"
        ] = approval_policy
        self._sandbox: Literal["read-only", "workspace-write", "danger-full-access"] = (
            sandbox
        )
        self._client_factory_cached: _CodexClientFactory | None = None
        self._evidence_provider_cached: _CodexStubEvidenceProvider | None = None

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
            self._evidence_provider_cached = _CodexStubEvidenceProvider()
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
        """Phase E/I fail-closed prerequisite hook (still stubbed in C).

        Phase E (T014) installs the bundled Codex plugin and runs the
        hook self-test; Phase I (T020) layers the SDK-importability /
        binary / auth checks on top. Until those land, every Codex run
        aborts cleanly here so the orchestrator does not silently route
        to a half-built session pipeline.
        """
        del repo_path, mcp_server_factory
        raise CodexNotImplementedError(_STUB_MESSAGE)
