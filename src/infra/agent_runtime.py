"""Agent runtime configuration builder.

This module provides ClaudeAgentRuntimeBuilder for centralized agent runtime setup.
It consolidates duplicated configuration logic from AgentSessionRunner and
RunCoordinator into a single, testable builder.

Design principles:
- Builder pattern with fluent API for configuration
- All SDK imports local to build() method for lazy-import guarantees
- Returns AgentRuntime dataclass with all components needed for session
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from src.infra.tools.env import SCRIPTS_DIR, format_mcp_timeout_ms, get_lock_dir

logger = logging.getLogger(__name__)


class _Unset(Enum):
    """Sentinel for distinguishing 'not provided' from 'explicitly None'."""

    TOKEN = auto()


_UNSET = _Unset.TOKEN

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Set as AbstractSet
    from pathlib import Path

    from src.core.models import LockEvent
    from src.core.protocols.lifecycle import DeadlockMonitorProtocol
    from src.core.protocols.sdk import McpServerFactory, SDKClientProtocol
    from src.infra.hooks import FileReadCache, LintCache


@runtime_checkable
class ClaudeSDKClientFactoryProtocol(Protocol):
    """Claude-private factory protocol.

    Exposes the slim cross-coder surface (``create`` / ``with_resume``)
    plus the Claude-only knobs (``create_options`` /
    ``create_hook_matcher``) consumed by Claude-specific wiring code
    (``ClaudeAgentRuntimeBuilder``, ``AgentSDKReviewer``). Declared next to
    its Claude-side consumers — and pointedly NOT alongside the
    cross-coder ``SDKClientFactoryProtocol`` in ``core/protocols/sdk.py``
    — so the cross-coder protocol stays free of Claude vocabulary
    (plan AC#16).

    Lives here rather than in ``src.infra.sdk_adapter`` so importing
    this protocol does not transitively pull in ``claude_agent_sdk``
    (the ``SDK confined to infra`` import-linter contract forbids that
    chain from ``agent_runtime``).
    """

    def create(self, runtime: object) -> SDKClientProtocol: ...

    def with_resume(self, runtime: object, resume: str | None) -> object: ...

    def create_options(
        self,
        *,
        cwd: str,
        permission_mode: str = "bypassPermissions",
        model: str = "opus",
        system_prompt: dict[str, str] | None = None,
        output_format: object | None = None,
        settings: str | None = None,
        setting_sources: list[str] | None = None,
        mcp_servers: object | None = None,
        disallowed_tools: list[str] | None = None,
        env: dict[str, str] | None = None,
        hooks: dict[str, list[object]] | None = None,
        resume: str | None = None,
        effort: str | None = None,
    ) -> object: ...

    def create_hook_matcher(
        self,
        matcher: object | None,
        hooks: list[object],
    ) -> object: ...


@dataclass
class AgentRuntime:
    """Runtime configuration bundle for agent sessions.

    Contains all components needed to run an agent session:
    - SDK options for client creation
    - Caches for file read and lint deduplication
    - Environment variables for agent execution
    - Hook lists for PreToolUse, PostToolUse, and Stop events

    Attributes:
        options: SDK ClaudeAgentOptions for session creation.
        file_read_cache: Cache for blocking redundant file reads.
        lint_cache: Cache for blocking redundant lint commands.
        env: Environment variables for agent execution.
        pre_tool_hooks: List of PreToolUse hook callables.
        post_tool_hooks: List of PostToolUse hook callables.
        stop_hooks: List of Stop hook callables.
    """

    options: object
    file_read_cache: FileReadCache
    lint_cache: LintCache
    env: dict[str, str]
    pre_tool_hooks: list[object] = field(default_factory=list)
    post_tool_hooks: list[object] = field(default_factory=list)
    stop_hooks: list[object] = field(default_factory=list)


@dataclass(frozen=True)
class RuntimeComponents:
    """Pre-constructed hooks and caches injected into ClaudeAgentRuntimeBuilder.

    Decouples hook/cache instantiation from SDK struct formatting so the
    builder body (:meth:`ClaudeAgentRuntimeBuilder.build`) only formats
    these into the SDK ``hooks`` dict + ``AgentRuntime``. Construct via
    :func:`build_default_runtime_components` for the standard mala
    composition, or assemble manually for advanced injection.

    Attributes:
        file_read_cache: Cache for blocking redundant file reads.
        lint_cache: Cache for blocking redundant lint commands.
        pre_tool_hooks: Ordered list of PreToolUse hook callables.
        post_tool_hooks: List of PostToolUse hook callables.
        stop_hooks: List of Stop hook callables.
        precompact_hook: PreCompact hook callable wrapped into the SDK
            ``PreCompact`` matcher by the builder.
    """

    file_read_cache: FileReadCache
    lint_cache: LintCache
    pre_tool_hooks: list[object]
    post_tool_hooks: list[object]
    stop_hooks: list[object]
    precompact_hook: object


def build_default_runtime_components(
    repo_path: Path,
    agent_id: str,
    *,
    lint_tools: AbstractSet[str] | None = None,
    include_stop_hook: bool = True,
    include_mala_disallowed_tools_hook: bool = True,
    include_lock_enforcement_hook: bool = True,
    deadlock_monitor: DeadlockMonitorProtocol | None = None,
) -> RuntimeComponents:
    """Build the standard mala hook/cache composition.

    Performs all hook/cache instantiation that :class:`ClaudeAgentRuntimeBuilder`
    previously did inline; the result is injected into the builder so
    ``build()`` only formats data into the SDK struct (Finding #13). Callers
    that need custom composition (fixer hook flags, custom lint tools)
    invoke this helper directly and pass the result to the builder via
    the ``runtime_components`` kwarg.

    Args:
        repo_path: Repository root.
        agent_id: Per-session agent identifier (used for lock ownership and
            stop/commit-guard hook plumbing).
        lint_tools: Optional set of lint tool names for the
            :class:`LintCache`. ``None`` uses cache defaults.
        include_stop_hook: When True, registers a Stop hook that cleans up
            the agent's locks on session end.
        include_mala_disallowed_tools_hook: When True, prepends the
            ``block_mala_disallowed_tools`` PreToolUse hook. Set False for
            fixer agents.
        include_lock_enforcement_hook: When True, prepends the lock
            enforcement PreToolUse hook. Set False when the runtime does
            not bundle the mala locking MCP tools.
        deadlock_monitor: Optional monitor whose ``handle_event`` is wired
            into a PostToolUse hook for ACQUIRED/RELEASED events.

    Returns:
        :class:`RuntimeComponents` with all hooks and caches pre-built.
    """
    from src.infra.hooks import (
        FileReadCache,
        LintCache,
        block_dangerous_commands,
        block_mala_disallowed_tools,
        make_commit_guard_hook,
        make_file_read_cache_hook,
        make_lint_cache_hook,
        make_lock_enforcement_hook,
        make_lock_event_hook,
        make_precompact_hook,
        make_stop_hook,
    )

    file_read_cache = FileReadCache()
    lint_cache = LintCache(
        repo_path=repo_path,
        lint_tools=lint_tools,
    )

    pre_tool_hooks: list[object] = [
        block_dangerous_commands,
        make_commit_guard_hook(agent_id, str(repo_path)),
        make_file_read_cache_hook(file_read_cache),
        make_lint_cache_hook(lint_cache),
    ]

    if include_lock_enforcement_hook:
        pre_tool_hooks.insert(1, make_lock_enforcement_hook(agent_id, str(repo_path)))

    if include_mala_disallowed_tools_hook:
        pre_tool_hooks.insert(1, block_mala_disallowed_tools)

    post_tool_hooks: list[object] = []
    stop_hooks: list[object] = []

    if include_stop_hook:
        stop_hooks.append(make_stop_hook(agent_id))

    if deadlock_monitor is not None:
        logger.info("Wiring deadlock monitor hooks: agent_id=%s", agent_id)
        from src.core.models import LockEvent, LockEventType

        post_tool_hooks.append(
            make_lock_event_hook(
                agent_id=agent_id,
                emit_event=deadlock_monitor.handle_event,
                repo_namespace=str(repo_path),
                lock_event_class=LockEvent,
                lock_event_type_enum=LockEventType,
            )
        )
    else:
        logger.info(
            "No deadlock monitor configured; locking MCP tools available but events not tracked"
        )

    precompact_hook = make_precompact_hook(repo_path)

    return RuntimeComponents(
        file_read_cache=file_read_cache,
        lint_cache=lint_cache,
        pre_tool_hooks=pre_tool_hooks,
        post_tool_hooks=post_tool_hooks,
        stop_hooks=stop_hooks,
        precompact_hook=precompact_hook,
    )


class ClaudeAgentRuntimeBuilder:
    """Claude-private builder for agent runtime configuration.

    Conforms structurally to
    :class:`src.core.protocols.agent_provider.CoderRuntimeBuilder` (the
    cross-coder fluent surface) and additionally exposes Claude-only
    configuration (``with_hooks`` for SDK PreToolUse / Stop / PostToolUse
    hook gating; ``with_disallowed_tools`` for MALA_DISALLOWED_TOOLS).
    Cross-coder pipeline code interacts only with the protocol surface;
    Claude-private callers (e.g. :class:`FixerService` for fixer-specific
    hook flags) downcast to this class via ``isinstance``.

    Each ``with_*`` method returns self for chaining. Call ``.build()`` to
    create the :class:`AgentRuntime` with all configured components.

    Attributes:
        repo_path: Path to the repository root.
        agent_id: Unique agent identifier for lock management.
        sdk_client_factory: Factory for creating SDK options and matchers.
    """

    def __init__(
        self,
        repo_path: Path,
        agent_id: str,
        sdk_client_factory: ClaudeSDKClientFactoryProtocol,
        mcp_server_factory: McpServerFactory | None = None,
        setting_sources: list[str] | None = None,
        model: str | None = None,
        effort: str | None = None,
        deadlock_monitor: DeadlockMonitorProtocol | None = None,
        *,
        runtime_components: RuntimeComponents | None = None,
    ) -> None:
        """Initialize the builder.

        Args:
            repo_path: Path to the repository root.
            agent_id: Unique agent identifier for lock management.
            sdk_client_factory: Factory for creating SDK options and matchers.
            mcp_server_factory: Optional factory for creating MCP server configs.
                Required unless MCP servers are explicitly provided via with_mcp().
            setting_sources: Optional list of Claude settings sources to use.
                E.g., ["local", "project"]. If None, SDK defaults are used.
            model: Optional Mala-level model forwarded to
                ``ClaudeAgentOptions.model`` for the coder session. ``None``
                uses the historical Claude default ``opus[1m]``.
            effort: Optional Mala-level reasoning effort forwarded to
                ``ClaudeAgentOptions.effort`` for the coder session. ``None``
                leaves the SDK default in place; reviewer / epic-verifier
                option construction is intentionally not routed through this
                builder.
            deadlock_monitor: Optional deadlock monitor whose ``handle_event``
                is wired into a Claude SDK PostToolUse hook. Threaded through
                :meth:`AgentProvider.runtime_builder` (plan A6) so the
                cross-coder pipeline does not need to call ``with_hooks``.
            runtime_components: Optional pre-constructed hooks and caches
                (Finding #13). When provided, :meth:`build` uses these
                directly without calling :func:`build_default_runtime_components`,
                so the builder body only formats them into the SDK struct.
                When ``None``, the builder calls the helper at :meth:`build`
                time using the recorded config (``with_hooks``,
                ``with_lint_tools``, constructor ``deadlock_monitor``).
                Calls to :meth:`with_hooks` or :meth:`with_lint_tools` after
                injection clear the components so the next :meth:`build`
                rebuilds with the updated config.
        """
        self._repo_path = repo_path
        self._agent_id = agent_id
        self._sdk_client_factory = sdk_client_factory
        self._mcp_server_factory = mcp_server_factory
        # Normalize to list (e.g., tuple from config -> list). Preserve empty list.
        self._setting_sources = (
            None if setting_sources is None else list(setting_sources)
        )
        self._model: str = model or "opus[1m]"
        self._effort: str | None = effort

        # Lint tools configuration (used by the default helper fallback path)
        self._lint_tools: AbstractSet[str] | None = None

        # Hook configuration (used by the default helper fallback path)
        self._deadlock_monitor: DeadlockMonitorProtocol | None = deadlock_monitor
        self._include_stop_hook: bool = True
        self._include_mala_disallowed_tools_hook: bool = True
        self._include_lock_enforcement_hook: bool = True

        # Pre-constructed components (Finding #13). ``build()`` never
        # invokes :func:`build_default_runtime_components`; the helper is
        # called eagerly here (when no injection is supplied) and from
        # :meth:`with_hooks` / :meth:`with_lint_tools` on config change, so
        # ``_runtime_components`` is always populated before ``build()``
        # runs. ``build()``'s body then only formats the components into
        # the SDK struct.
        if runtime_components is None:
            self._runtime_components: RuntimeComponents = (
                self._compose_components_from_config()
            )
        else:
            self._runtime_components = runtime_components

        # Environment and options. ``_disallowed_tools`` defaults to the
        # standard MALA_DISALLOWED_TOOLS list so coder-agnostic pipeline
        # callers (which no longer call ``with_disallowed_tools()``) get
        # the expected restrictions.
        from src.infra.tool_config import MALA_DISALLOWED_TOOLS

        self._env: dict[str, str] | None = None
        self._mcp_servers: object | None = None
        self._disallowed_tools: list[str] | None = list(MALA_DISALLOWED_TOOLS)
        self._agent_timeout_seconds: float | None = None
        self._resume_id: str | None = None

    def _compose_components_from_config(self) -> RuntimeComponents:
        """Build :class:`RuntimeComponents` via the external helper.

        Single dispatch point to :func:`build_default_runtime_components`
        used by :meth:`__init__` (no-injection fallback) and the eager
        rebuild paths in :meth:`with_hooks` / :meth:`with_lint_tools`.
        Keeping the helper call out of :meth:`build` ensures the build
        body does no hook/cache instantiation (Finding #13).
        """
        return build_default_runtime_components(
            self._repo_path,
            self._agent_id,
            lint_tools=self._lint_tools,
            include_stop_hook=self._include_stop_hook,
            include_mala_disallowed_tools_hook=self._include_mala_disallowed_tools_hook,
            include_lock_enforcement_hook=self._include_lock_enforcement_hook,
            deadlock_monitor=self._deadlock_monitor,
        )

    def with_hooks(
        self,
        *,
        include_stop_hook: bool | _Unset = _UNSET,
        include_mala_disallowed_tools_hook: bool | _Unset = _UNSET,
        include_lock_enforcement_hook: bool | _Unset = _UNSET,
    ) -> ClaudeAgentRuntimeBuilder:
        """Configure Claude-only hook behavior (Claude-private).

        Only parameters that are explicitly provided will be updated;
        omitted parameters preserve their current state. Not part of the
        cross-coder :class:`CoderRuntimeBuilder` protocol — Claude-private
        callers (e.g. :class:`FixerService` for fixer agents) call this
        after downcasting via ``isinstance``.

        Args:
            include_stop_hook: Whether to include stop hook. Omit to preserve
                current state (initially True).
            include_mala_disallowed_tools_hook: Whether to include the
                block_mala_disallowed_tools hook. Omit to preserve current
                state (initially True). Set False for fixer agents which
                don't need this restriction.
            include_lock_enforcement_hook: Whether to include the lock
                enforcement hook. Omit to preserve current state (initially
                True). Set False when MCP servers do not include locking
                tools (e.g., custom with_mcp(servers=...)).

        Returns:
            Self for chaining.
        """
        changed = False
        if (
            include_stop_hook is not _UNSET
            and include_stop_hook != self._include_stop_hook
        ):
            self._include_stop_hook = include_stop_hook
            changed = True
        if (
            include_mala_disallowed_tools_hook is not _UNSET
            and include_mala_disallowed_tools_hook
            != self._include_mala_disallowed_tools_hook
        ):
            self._include_mala_disallowed_tools_hook = (
                include_mala_disallowed_tools_hook
            )
            changed = True
        if (
            include_lock_enforcement_hook is not _UNSET
            and include_lock_enforcement_hook != self._include_lock_enforcement_hook
        ):
            self._include_lock_enforcement_hook = include_lock_enforcement_hook
            changed = True
        # Only rebuild components when a flag actually changed; no-op
        # calls preserve provider-injected components. The rebuild
        # happens eagerly via the external helper so ``build()``'s body
        # never instantiates hooks/caches itself (Finding #13).
        if changed:
            self._runtime_components = self._compose_components_from_config()
        return self

    def with_resume(self, resume_id: str | None) -> ClaudeAgentRuntimeBuilder:
        """Configure session resumption for the next :meth:`build` call.

        ``None`` (the default) means no resumption. The resume id is
        forwarded to ``create_options(resume=...)``. Pipeline code may
        also apply resumption via
        :meth:`SDKClientFactory.with_resume(runtime, resume)` after
        ``build()``; both pathways converge on the same SDK option.
        """
        self._resume_id = resume_id
        return self

    def with_agent_timeout(
        self, timeout_seconds: float | None
    ) -> ClaudeAgentRuntimeBuilder:
        """Configure the per-agent timeout used for MCP tool calls.

        Mala's session timeout is in seconds; SDK MCP timeouts are configured
        through the ``MCP_TIMEOUT`` environment variable in milliseconds.
        """
        self._agent_timeout_seconds = timeout_seconds
        if self._env is not None:
            self._env["MCP_TIMEOUT"] = format_mcp_timeout_ms(timeout_seconds)
        return self

    def with_env(
        self, extra: Mapping[str, str] | None = None
    ) -> ClaudeAgentRuntimeBuilder:
        """Configure environment variables.

        Builds standard environment with PATH, LOCK_DIR, AGENT_ID, etc.
        Merges with os.environ and any extra variables provided.

        Args:
            extra: Additional environment variables to include.

        Returns:
            Self for chaining.
        """
        self._env = {
            **os.environ,
            "PATH": f"{SCRIPTS_DIR}:{os.environ.get('PATH', '')}",
            "LOCK_DIR": str(get_lock_dir()),
            "AGENT_ID": self._agent_id,
            "REPO_NAMESPACE": str(self._repo_path),
            "MCP_TIMEOUT": format_mcp_timeout_ms(self._agent_timeout_seconds),
        }
        if extra:
            self._env.update(extra)
        self._env["MCP_TIMEOUT"] = format_mcp_timeout_ms(self._agent_timeout_seconds)
        return self

    def with_mcp(
        self,
        servers: Mapping[str, object] | None = None,
        *,
        emit_lock_event: Callable[[LockEvent], object] | _Unset | None = _UNSET,
    ) -> ClaudeAgentRuntimeBuilder:
        """Configure MCP servers.

        Args:
            servers: MCP server configuration. If None, defers to build() for
                late-binding with deadlock monitor (if configured).
            emit_lock_event: Callback for lock events. When _UNSET (default),
                defers to build() for late-binding. Pass None to use a no-op
                handler (locking tools work but events aren't tracked for
                deadlock detection), or a callback to enable event tracking.

        Returns:
            Self for chaining.
        """
        if servers is not None:
            self._mcp_servers = servers
        elif emit_lock_event is not _UNSET:
            # Explicit emit_lock_event provided (including None) - configure now
            self._mcp_servers = self._build_mcp_servers(emit_lock_event)
        # else: emit_lock_event is _UNSET, defer to build() for late-binding
        return self

    def with_disallowed_tools(
        self, tools: list[str] | None = None
    ) -> ClaudeAgentRuntimeBuilder:
        """Configure disallowed tools (Claude-private).

        Not part of the cross-coder :class:`CoderRuntimeBuilder` protocol;
        Amp does not honor SDK-level ``disallowed_tools``. The constructor
        already defaults this to ``MALA_DISALLOWED_TOOLS``; call this
        method only to override (e.g., disable for tests).

        Args:
            tools: List of disallowed tool names. If None, uses MALA_DISALLOWED_TOOLS.

        Returns:
            Self for chaining.
        """
        if tools is not None:
            self._disallowed_tools = tools
        else:
            from src.infra.tool_config import MALA_DISALLOWED_TOOLS

            self._disallowed_tools = list(MALA_DISALLOWED_TOOLS)
        return self

    def with_lint_tools(
        self, lint_tools: AbstractSet[str] | None = None
    ) -> ClaudeAgentRuntimeBuilder:
        """Configure lint tools for cache.

        Args:
            lint_tools: Set of lint tool names. If None, uses defaults.

        Returns:
            Self for chaining.
        """
        # Only rebuild components when the lint set actually changes.
        # AgentSessionRunner unconditionally forwards
        # ``self.config.lint_tools``; the no-op case preserves the
        # provider's injected components. When it changes, the rebuild
        # uses the external :func:`build_default_runtime_components`
        # helper, so ``build()``'s body never instantiates hooks/caches
        # (Finding #13).
        if lint_tools != self._lint_tools:
            self._lint_tools = lint_tools
            self._runtime_components = self._compose_components_from_config()
        return self

    def _build_mcp_servers(
        self, emit_lock_event: Callable[[LockEvent], object] | None
    ) -> dict[str, object]:
        """Build MCP servers configuration.

        Args:
            emit_lock_event: Optional callback to emit lock events. If None,
                a no-op handler is used (locking tools work but events
                aren't tracked for deadlock detection).

        Returns:
            Dictionary of MCP server configurations.
        """
        if self._mcp_server_factory is None:
            msg = (
                "MCP server factory is required. Either provide mcp_server_factory "
                "or explicitly provide servers via with_mcp(servers={...}). "
                "If your custom servers don't include locking tools, also call "
                "with_hooks(include_lock_enforcement_hook=False) on the "
                "Claude-private builder."
            )
            raise ValueError(msg)

        return self._mcp_server_factory(
            self._agent_id,
            self._repo_path,
            emit_lock_event,
        )

    def build(self) -> AgentRuntime:
        """Build the agent runtime configuration.

        Formats the pre-populated :class:`RuntimeComponents` into the SDK
        ``hooks`` dict and :class:`ClaudeAgentOptions` — no hook or cache
        instantiation happens in this body (Finding #13). Components are
        always populated before ``build()`` runs: by injection at
        construction, by :func:`build_default_runtime_components` called
        eagerly in :meth:`__init__` (no-injection case), or by the
        eager rebuild path in :meth:`with_hooks` / :meth:`with_lint_tools`.

        Returns:
            AgentRuntime with all configured components.

        Raises:
            RuntimeError: If required configuration is missing.
        """
        components = self._runtime_components

        # Build environment if not explicitly set
        if self._env is None:
            self.with_env()
        env = self._env or {}

        # Build MCP servers if not explicitly set
        if self._mcp_servers is None:
            monitor = self._deadlock_monitor
            emit_lock_event = monitor.handle_event if monitor is not None else None
            self.with_mcp(emit_lock_event=emit_lock_event)

        # Log and validate setting sources BEFORE any SDK initialization
        # (create_hook_matcher imports SDK types, so this must come first)
        resolved_sources = (
            ["local", "project"]
            if self._setting_sources is None
            else self._setting_sources
        )
        if resolved_sources:
            logger.info("Claude settings sources: %s", ", ".join(resolved_sources))
        else:
            logger.info("Claude settings sources: (none)")
        if "local" in resolved_sources:
            local_settings_path = self._repo_path / ".claude/settings.local.json"
            if not local_settings_path.exists():
                logger.warning(
                    "Claude settings file .claude/settings.local.json not found "
                    "(will be skipped)"
                )

        # Format pre-built hooks into the SDK hooks dict
        make_matcher = self._sdk_client_factory.create_hook_matcher
        hooks_dict: dict[str, list[object]] = {
            "PreToolUse": [make_matcher(None, components.pre_tool_hooks)],
            "PreCompact": [make_matcher(None, [components.precompact_hook])],
        }
        if components.stop_hooks:
            hooks_dict["Stop"] = [make_matcher(None, components.stop_hooks)]
        if components.post_tool_hooks:
            hooks_dict["PostToolUse"] = [make_matcher(None, components.post_tool_hooks)]

        logger.debug(
            "Built hooks: PreToolUse=%d PostToolUse=%d Stop=%d PreCompact=1",
            len(components.pre_tool_hooks),
            len(components.post_tool_hooks),
            len(components.stop_hooks),
        )

        # Build SDK options
        options = self._sdk_client_factory.create_options(
            cwd=str(self._repo_path),
            model=self._model,
            mcp_servers=self._mcp_servers,
            disallowed_tools=self._disallowed_tools,
            env=env,
            hooks=hooks_dict,
            setting_sources=self._setting_sources,
            settings='{"autoCompactEnabled": true}',
            effort=self._effort,
            resume=self._resume_id,
        )

        return AgentRuntime(
            options=options,
            file_read_cache=components.file_read_cache,
            lint_cache=components.lint_cache,
            env=env,
            pre_tool_hooks=components.pre_tool_hooks,
            post_tool_hooks=components.post_tool_hooks,
            stop_hooks=components.stop_hooks,
        )
