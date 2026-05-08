"""Codex :class:`AgentProvider` stub (Phase B).

Wires ``coder=codex`` selection through to a fail-closed stub that conforms
structurally to :class:`src.core.protocols.agent_provider.AgentProvider`.
The stub raises :class:`CodexNotImplementedError` from every method that
would normally drive a Codex turn so downstream tests prove the selection
path reaches the provider; full implementations land in Phases C-I per
``plans/2026-05-07-codex-provider-plan.md``.

Lazy-import contract: importing this module must NOT pull in
``codex_app_server`` (Phase B test guards this; the SDK lands in Phase C).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

from src.core.constants import (
    DEFAULT_CODEX_APPROVAL_POLICY,
    DEFAULT_CODEX_MODEL,
    DEFAULT_CODEX_SANDBOX,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Set as AbstractSet
    from pathlib import Path

    from src.core.protocols.agent_provider import CoderRuntimeBuilder
    from src.core.protocols.evidence import EvidenceProvider, JsonlEntryProtocol
    from src.core.protocols.lifecycle import DeadlockMonitorProtocol
    from src.core.protocols.sdk import (
        McpServerFactory,
        SDKClientFactoryProtocol,
        SDKClientProtocol,
    )


_STUB_MESSAGE = "CodexAgentProvider stub: full implementation lands in Phases C-I"


class CodexNotImplementedError(NotImplementedError):
    """Raised when the Phase B Codex stub is exercised.

    The Codex provider is selectable end-to-end after Phase B (CLI / env /
    yaml all reach this stub) but cannot drive a real turn until Phases
    C-I land. Every entry point that would touch ``codex_app_server`` or
    spawn a Codex session raises this exception so the gap is observable
    without leaking subprocess errors.
    """


# ---------------------------------------------------------------------------
# MCP server factory (provider-owned, Phase B placeholder)
# ---------------------------------------------------------------------------


def _create_codex_mcp_server_factory() -> McpServerFactory:
    """Stub Codex-shaped MCP factory.

    Phase G wires the bundled ``mala-locking`` launcher inside the Codex
    plugin (`plans/2026-05-07-codex-provider-plan.md` G1-G3); for Phase B
    the factory returns an empty map so :meth:`AgentProvider.mcp_server_factory`
    has a callable to hand back. Real launches go through the matching
    ``runtime_builder``, which raises :class:`CodexNotImplementedError`
    before the factory is consulted.
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
# Stub client factory + evidence provider
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CodexStubRuntime:
    """Phase B placeholder for the per-session Codex runtime.

    The real :class:`CodexRuntime` (Phase C) carries cwd, agent id, model,
    effort, approval_policy, sandbox, mcp_servers, env, resume_thread_id,
    and the lint cache (per plan ``L412-L432``). Phase B captures only the
    fields the integration-path test asserts reach the provider; the
    pipeline never inspects them because every path that consumes the
    runtime raises :class:`CodexNotImplementedError` first.
    """

    cwd: Path
    agent_id: str
    model: str
    effort: str | None
    approval_policy: Literal["never", "on-request", "on-failure", "untrusted"]
    sandbox: Literal["read-only", "workspace-write", "danger-full-access"]
    resume_thread_id: str | None = None


class _CodexStubClientFactory:
    """:class:`SDKClientFactoryProtocol` stub for the Codex coder.

    ``create()`` raises :class:`CodexNotImplementedError`; ``with_resume()``
    threads a ``resume_thread_id`` onto the runtime so tests can assert
    the resume token reaches the provider before Phase C lands the real
    ``AsyncCodex.thread_resume`` wiring.
    """

    def create(self, runtime: object) -> SDKClientProtocol:
        del runtime
        raise CodexNotImplementedError(_STUB_MESSAGE)

    def with_resume(self, runtime: object, resume: str | None) -> object:
        if not isinstance(runtime, _CodexStubRuntime):
            raise TypeError(
                "CodexAgentProvider.client_factory.with_resume(runtime, resume) "
                "requires the Codex stub runtime; got "
                f"{type(runtime).__name__}."
            )
        if resume is None:
            return runtime
        from dataclasses import replace

        return replace(runtime, resume_thread_id=resume)


class _CodexStubEvidenceProvider:
    """:class:`EvidenceProvider` stub for the Codex coder.

    Every call raises :class:`CodexNotImplementedError`; Phase F replaces
    this with the native ``Thread.read(include_turns=True)`` adapter (or
    the tee fallback if the F1 spike disconfirms native viability).
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
# Stub runtime builder
# ---------------------------------------------------------------------------


class _CodexStubRuntimeBuilder:
    """:class:`CoderRuntimeBuilder` stub for Codex (Phase B).

    Captures the configured options on construction so
    :meth:`AgentProvider.runtime_builder` can produce a builder whose
    :meth:`build` returns a runtime that *carries* the configured
    ``model``, ``effort``, ``approval_policy``, and ``sandbox``. The
    pipeline never inspects this runtime in Phase B (every consumer
    raises :class:`CodexNotImplementedError`); the integration-path test
    builds it explicitly to prove the resolution chain reaches the
    provider with the configured values.
    """

    def __init__(
        self,
        repo_path: Path,
        agent_id: str,
        *,
        model: str,
        effort: str | None,
        approval_policy: Literal["never", "on-request", "on-failure", "untrusted"],
        sandbox: Literal["read-only", "workspace-write", "danger-full-access"],
    ) -> None:
        self._repo_path = repo_path
        self._agent_id = agent_id
        self._model = model
        self._effort = effort
        self._approval_policy = approval_policy
        self._sandbox = sandbox
        self._resume_id: str | None = None

    def with_resume(self, resume_id: str | None) -> _CodexStubRuntimeBuilder:
        self._resume_id = resume_id
        return self

    def with_agent_timeout(
        self, timeout_seconds: float | None
    ) -> _CodexStubRuntimeBuilder:
        del timeout_seconds
        return self

    def with_env(
        self, extra: Mapping[str, str] | None = None
    ) -> _CodexStubRuntimeBuilder:
        del extra
        return self

    def with_lint_tools(
        self, lint_tools: AbstractSet[str] | None = None
    ) -> _CodexStubRuntimeBuilder:
        del lint_tools
        return self

    def with_mcp(
        self, servers: Mapping[str, object] | None = None
    ) -> _CodexStubRuntimeBuilder:
        del servers
        return self

    def build(self) -> _CodexStubRuntime:
        return _CodexStubRuntime(
            cwd=self._repo_path,
            agent_id=self._agent_id,
            model=self._model,
            effort=self._effort,
            approval_policy=self._approval_policy,
            sandbox=self._sandbox,
            resume_thread_id=self._resume_id,
        )


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class CodexAgentProvider:
    """Phase B :class:`AgentProvider` stub for the Codex coder.

    Conforms structurally to
    :class:`src.core.protocols.agent_provider.AgentProvider`. Selection
    wiring works end-to-end (CLI ``--coder codex`` / ``MALA_CODER=codex``
    / yaml ``coder: codex`` all reach this stub) but every method that
    would touch ``codex_app_server`` raises
    :class:`CodexNotImplementedError`.

    Attributes:
        name: Provider identifier (always ``"codex"``).
        client_factory: Stub :class:`SDKClientFactoryProtocol` whose
            ``create`` raises :class:`CodexNotImplementedError`.
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
        self._client_factory_cached: _CodexStubClientFactory | None = None
        self._evidence_provider_cached: _CodexStubEvidenceProvider | None = None

    # ------------------------------------------------------------------
    # AgentProvider protocol surface
    # ------------------------------------------------------------------

    @property
    def client_factory(self) -> SDKClientFactoryProtocol:
        if self._client_factory_cached is None:
            self._client_factory_cached = _CodexStubClientFactory()
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
        """Construct a per-session Codex runtime builder.

        Phase B builds a stub runtime carrying the configured
        ``model``, ``effort``, ``approval_policy``, and ``sandbox`` so
        the integration-path test can verify those values reach the
        provider via ``MalaConfig.coder_options.codex``. Phase C
        replaces the stub with a real ``CodexRuntimeBuilder`` that
        consumes ``mcp_server_factory`` to wire the bundled
        ``mala-locking`` server.
        """
        del mcp_server_factory, deadlock_monitor
        return cast(
            "CoderRuntimeBuilder",
            _CodexStubRuntimeBuilder(
                repo_path,
                agent_id,
                model=self._model,
                effort=self._effort,
                approval_policy=self._approval_policy,
                sandbox=self._sandbox,
            ),
        )

    def mcp_server_factory(self) -> McpServerFactory:
        """Return the Codex-shaped MCP server factory (Phase G stub)."""
        return _create_codex_mcp_server_factory()

    def install_prerequisites(
        self,
        repo_path: Path,
        *,
        mcp_server_factory: McpServerFactory,
    ) -> None:
        """Phase B fail-closed prerequisite hook.

        Phases E + I fold in the real prerequisite checks (plugin install,
        SDK importability, auth probe, hook self-test, fail-closed
        actionable errors). Phase B already needs to refuse to spawn any
        issue agent when ``coder=codex``: the orchestrator calls
        :meth:`install_prerequisites` once before the first session, and
        every other entry point on this provider raises, so the run
        terminates cleanly with a clear "not yet implemented" message
        instead of silently routing to a half-built session pipeline.
        """
        del repo_path, mcp_server_factory
        raise CodexNotImplementedError(_STUB_MESSAGE)
