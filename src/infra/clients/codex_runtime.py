"""Codex coder runtime + builder (Phase C, T010).

A :class:`CodexRuntime` is the per-session description consumed by
:class:`src.infra.clients.codex_client.CodexClient`. It bundles every value
the client needs to call ``AsyncCodex.thread_start`` /
``AsyncCodex.thread_resume`` for one Codex session: model, effort,
approval policy, sandbox, MCP server map, the per-process env dict (the
parent ``os.environ`` plus mala-specific overlays), the agent id, and the
working directory.

Per-process env isolation contract (plan ``L735``, ``L815``): the runtime's
``env`` is composed as ``{**os.environ, ...overlays}`` at :meth:`build`
time. Mala MUST NOT mutate the parent process's ``os.environ`` to pass
``MALA_*`` lock-ownership vars; under ``--max-agents > 1`` concurrent
agents would clobber each other's ``MALA_AGENT_ID``. Mirrors the Amp
path's posture in :class:`AmpRuntimeBuilder.build`
(``src/infra/clients/amp_runtime.py:283-295``).

Lazy-import contract (plan ``L733``): this module does NOT import
``codex_app_server``. The :attr:`approval_policy` and :attr:`sandbox`
fields are typed as plain ``Literal``s so the runtime can be instantiated
without the SDK installed; :class:`CodexClient` translates them to the
SDK's ``AskForApproval`` / ``SandboxMode`` enums when it constructs the
``AsyncCodex`` thread (Phase C, lazy).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping, Set as AbstractSet
    from pathlib import Path

    from src.core.protocols.sdk import McpServerFactory
    from src.infra.hooks.lint_cache import LintCache


CodexApprovalPolicy = Literal["never", "on-request", "on-failure", "untrusted"]
"""String form of ``codex_app_server.AskForApproval``.

Kept as a string Literal here so ``CodexRuntime`` can be constructed
without importing the SDK; the client maps the value to the SDK enum
inside its lazy-imported code path.
"""

CodexSandbox = Literal["read-only", "workspace-write", "danger-full-access"]
"""String form of ``codex_app_server.SandboxMode``."""


@dataclass(frozen=True)
class CodexRuntime:
    """Complete description for one ``AsyncCodex`` session.

    Returned by :meth:`CodexRuntimeBuilder.build`. Consumed privately by
    :class:`CodexAgentProvider.client_factory` (T010), which unpacks
    these fields into ``AsyncCodex.thread_start`` arguments at
    ``create(runtime)`` time. The pipeline forwards the runtime opaquely;
    only :class:`CodexClient` knows the runtime's shape.

    ``mcp_servers`` is populated with the bundled ``mala-locking`` launch
    spec by Phase G (T016); Phase C leaves it as the dict the injected
    MCP factory returns (empty in the Phase B placeholder factory). The
    field is plumbed end-to-end now so Phase G is a one-line factory
    swap rather than a runtime-shape change.
    """

    cwd: Path
    agent_id: str
    model: str
    effort: str | None
    approval_policy: CodexApprovalPolicy | None
    sandbox: CodexSandbox | None
    base_instructions: str | None
    mcp_servers: dict[str, object]
    env: Mapping[str, str]
    lint_cache: LintCache
    """In-memory :class:`LintCache` carried for protocol parity with the
    Claude path (``AgentRuntime.lint_cache``). The Codex client does not
    consume the cache itself in Phase C; ``AgentSessionRunner`` reads
    ``runtime.lint_cache`` unconditionally so a real instance must be
    present even before Phase D wires Codex's bash events into it."""
    resume_thread_id: str | None = None


class CodexRuntimeBuilder:
    """Build :class:`CodexRuntime` for one Codex session.

    Conforms structurally to
    :class:`src.core.protocols.agent_provider.CoderRuntimeBuilder`.
    Exposes only the cross-coder narrowed fluent surface (plan A6 /
    plan ``L501-L508``); Codex-specific knobs (model, effort, approval
    policy, sandbox) are injected via the constructor by
    :class:`CodexAgentProvider.runtime_builder`, which threads them down
    from ``MalaConfig.coder_options.codex``.
    """

    def __init__(
        self,
        repo_path: Path,
        agent_id: str,
        mcp_server_factory: McpServerFactory,
        *,
        model: str,
        effort: str | None,
        approval_policy: CodexApprovalPolicy,
        sandbox: CodexSandbox,
    ) -> None:
        """Initialize the builder.

        Args:
            repo_path: Repository root; becomes the runtime's ``cwd`` and
                ``MALA_REPO_NAMESPACE``.
            agent_id: Per-issue agent identifier; becomes
                ``MALA_AGENT_ID`` in the runtime's per-process env.
            mcp_server_factory: Produces the MCP server map the Codex
                ``thread_start`` consumes (Phase G3 wires the bundled
                ``mala-locking`` launcher here).
            model: Codex model id; threaded from
                ``MalaConfig.coder_options.codex.model``.
            effort: Optional reasoning effort; threaded from
                ``MalaConfig.coder_options.codex.effort``.
            approval_policy: Codex approval policy. Defaults are resolved
                upstream by :class:`CodexAgentProvider`.
            sandbox: Codex sandbox mode. Defaults are resolved upstream
                by :class:`CodexAgentProvider`.
        """
        self._repo_path = repo_path
        self._agent_id = agent_id
        self._mcp_server_factory = mcp_server_factory
        self._model = model
        self._effort = effort
        self._approval_policy: CodexApprovalPolicy = approval_policy
        self._sandbox: CodexSandbox = sandbox
        self._resume_thread_id: str | None = None
        self._env_extra: dict[str, str] = {}
        self._mcp_servers_override: Mapping[str, object] | None = None
        self._lint_tools: AbstractSet[str] | None = None
        self._agent_timeout_seconds: float | None = None
        self._base_instructions: str | None = None

    def with_resume(self, resume_id: str | None) -> CodexRuntimeBuilder:
        """Configure the next ``build()`` to resume an existing Codex thread.

        ``None`` means no resumption (a fresh thread). The resume id is
        stored on :attr:`CodexRuntime.resume_thread_id` and consumed by
        :meth:`CodexClient.query` (it picks ``thread_resume`` over
        ``thread_start`` when the id is present).
        """
        self._resume_thread_id = resume_id
        return self

    def with_agent_timeout(self, timeout_seconds: float | None) -> CodexRuntimeBuilder:
        """Record the per-agent timeout for parity with the protocol.

        Codex does not currently surface a timeout knob through
        ``thread_start``; the per-agent budget is enforced by the
        pipeline's :class:`IdleTimeoutRetryPolicy` wrapping
        :meth:`CodexClient.receive_response`. Stored verbatim so a future
        SDK version that does honor a timeout can wire it without
        changing the protocol surface.
        """
        self._agent_timeout_seconds = timeout_seconds
        return self

    def with_env(self, extra: Mapping[str, str] | None = None) -> CodexRuntimeBuilder:
        """Overlay extra env vars on top of the env composed in :meth:`build`.

        Mirrors :meth:`AmpRuntimeBuilder.with_env`'s ``extra=`` shape.
        The overlays are layered on top of ``os.environ`` and the
        mandatory ``MALA_*`` overlays.
        """
        if extra:
            self._env_extra.update(extra)
        return self

    def with_lint_tools(
        self, lint_tools: AbstractSet[str] | None = None
    ) -> CodexRuntimeBuilder:
        """Record the lint-tool set used to construct :attr:`CodexRuntime.lint_cache`."""
        self._lint_tools = lint_tools
        return self

    def with_mcp(
        self, servers: Mapping[str, object] | None = None
    ) -> CodexRuntimeBuilder:
        """Configure MCP servers; default uses the injected factory.

        ``servers`` is provided for parity with the Claude / Amp builders'
        explicit override (used by :class:`FixerService` to short-circuit
        MCP wiring when no factory is available). When omitted, the
        runtime's ``mcp_servers`` is built from the injected
        ``mcp_server_factory`` in :meth:`build` exactly as before.
        """
        if servers is not None:
            self._mcp_servers_override = servers
        return self

    def build(self) -> CodexRuntime:
        """Materialize the runtime.

        Composes the per-process env dict from ``os.environ`` plus the
        full ``MALA_*`` overlay the bundled ``mala-codex-pre-tool-use``
        hook needs to evaluate lock-gated writes (``MALA_AGENT_ID``,
        ``MALA_LOCK_DIR``, ``MALA_REPO_NAMESPACE``). ``MALA_LOCK_DIR``
        is resolved through :func:`src.infra.tools.env.get_lock_dir`
        rather than relying on the parent ``os.environ``: the hook
        reads its env strictly from what the Codex subprocess inherits,
        and an unset ``MALA_LOCK_DIR`` in the orchestrator's env would
        cause the hook to deny every lock-gated shell/apply_patch write
        fail-closed. Mirrors the Amp path's posture at
        ``src/infra/clients/amp_runtime.py:288``.

        Other parent env keys are inherited unchanged. Constructs a
        :class:`LintCache` instance for the runtime so the pipeline can
        read ``runtime.lint_cache`` without branching on coder.
        """
        if self._mcp_servers_override is not None:
            mcp_servers: dict[str, object] = dict(self._mcp_servers_override)
        else:
            mcp_servers = dict(
                self._mcp_server_factory(self._agent_id, self._repo_path, None)
            )

        # Lazy import to keep ``codex_runtime`` free of a hard
        # ``src.infra.tools.env`` dep at module-load time; the env
        # resolver is only needed when actually building a runtime.
        from src.infra.tools.env import get_lock_dir

        env: dict[str, str] = {
            **os.environ,
            "MALA_AGENT_ID": self._agent_id,
            "MALA_LOCK_DIR": str(get_lock_dir()),
            "MALA_REPO_NAMESPACE": str(self._repo_path),
            **self._env_extra,
        }

        # Lazy import to keep ``codex_runtime`` free of a hard
        # ``src.infra.hooks`` dep at module-load time; the import-linter
        # exception (``src.infra.clients.codex_runtime ->
        # src.infra.hooks.lint_cache``) acknowledges the runtime carries
        # a ``LintCache`` for protocol parity with the Claude/Amp paths.
        from src.infra.hooks.lint_cache import LintCache

        lint_cache = LintCache(
            repo_path=self._repo_path,
            lint_tools=self._lint_tools,
        )

        return CodexRuntime(
            cwd=self._repo_path,
            agent_id=self._agent_id,
            model=self._model,
            effort=self._effort,
            approval_policy=self._approval_policy,
            sandbox=self._sandbox,
            base_instructions=self._base_instructions,
            mcp_servers=mcp_servers,
            env=env,
            lint_cache=lint_cache,
            resume_thread_id=self._resume_thread_id,
        )
