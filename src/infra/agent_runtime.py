"""Agent runtime configuration builder.

This module provides AgentRuntimeBuilder for centralized agent runtime setup.
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
from typing import TYPE_CHECKING

from src.infra.tools.env import SCRIPTS_DIR, get_lock_dir

logger = logging.getLogger(__name__)


class _Unset(Enum):
    """Sentinel for distinguishing 'not provided' from 'explicitly None'."""

    TOKEN = auto()


_UNSET = _Unset.TOKEN

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from src.core.models import LockEvent
    from src.core.protocols.lifecycle import DeadlockMonitorProtocol
    from src.core.protocols.sdk import McpServerFactory, SDKClientFactoryProtocol
    from src.infra.hooks import FileReadCache, LintCache


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


class AgentRuntimeBuilder:
    """Builder for agent runtime configuration.

    Provides a fluent API for configuring agent sessions. Each .with_*()
    method returns self for chaining. Call .build() to create the
    AgentRuntime with all configured components.

    Example:
        runtime = (
            AgentRuntimeBuilder(repo_path, agent_id, factory)
            .with_hooks(deadlock_monitor=monitor)
            .with_env()
            .with_mcp()
            .with_disallowed_tools()
            .build()
        )
        # Use runtime.options to create SDK client

    Attributes:
        repo_path: Path to the repository root.
        agent_id: Unique agent identifier for lock management.
        sdk_client_factory: Factory for creating SDK options and matchers.
    """

    def __init__(
        self,
        repo_path: Path,
        agent_id: str,
        sdk_client_factory: SDKClientFactoryProtocol,
        mcp_server_factory: McpServerFactory | None = None,
        setting_sources: list[str] | None = None,
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
        """
        self._repo_path = repo_path
        self._agent_id = agent_id
        self._sdk_client_factory = sdk_client_factory
        self._mcp_server_factory = mcp_server_factory
        # Normalize to list (e.g., tuple from config -> list). Preserve empty list.
        self._setting_sources = (
            None if setting_sources is None else list(setting_sources)
        )

        # Lint tools configuration
        self._lint_tools: set[str] | frozenset[str] | None = None

        # Hook configuration
        self._deadlock_monitor: DeadlockMonitorProtocol | None = None
        self._include_stop_hook: bool = True
        self._include_mala_disallowed_tools_hook: bool = True
        self._include_lock_enforcement_hook: bool = True

        # Environment and options
        self._env: dict[str, str] | None = None
        self._mcp_servers: object | None = None
        self._disallowed_tools: list[str] | None = None

    def with_hooks(
        self,
        *,
        deadlock_monitor: DeadlockMonitorProtocol | None | _Unset = _UNSET,
        include_stop_hook: bool | _Unset = _UNSET,
        include_mala_disallowed_tools_hook: bool | _Unset = _UNSET,
        include_lock_enforcement_hook: bool | _Unset = _UNSET,
    ) -> AgentRuntimeBuilder:
        """Configure hook behavior.

        Only parameters that are explicitly provided will be updated;
        omitted parameters preserve their current state.

        Args:
            deadlock_monitor: Optional DeadlockMonitor for lock event hooks.
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
        if deadlock_monitor is not _UNSET:
            self._deadlock_monitor = deadlock_monitor
        if include_stop_hook is not _UNSET:
            self._include_stop_hook = include_stop_hook
        if include_mala_disallowed_tools_hook is not _UNSET:
            self._include_mala_disallowed_tools_hook = (
                include_mala_disallowed_tools_hook
            )
        if include_lock_enforcement_hook is not _UNSET:
            self._include_lock_enforcement_hook = include_lock_enforcement_hook
        return self

    def with_env(self, extra: dict[str, str] | None = None) -> AgentRuntimeBuilder:
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
            "MCP_TIMEOUT": "300000",
        }
        if extra:
            self._env.update(extra)
        return self

    def with_mcp(
        self,
        servers: object | None = None,
        *,
        emit_lock_event: Callable[[LockEvent], object] | _Unset | None = _UNSET,
    ) -> AgentRuntimeBuilder:
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
    ) -> AgentRuntimeBuilder:
        """Configure disallowed tools.

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
        self, lint_tools: set[str] | frozenset[str] | None = None
    ) -> AgentRuntimeBuilder:
        """Configure lint tools for cache.

        Args:
            lint_tools: Set of lint tool names. If None, uses defaults.

        Returns:
            Self for chaining.
        """
        self._lint_tools = lint_tools
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
                "with_hooks(include_lock_enforcement_hook=False)."
            )
            raise ValueError(msg)

        return self._mcp_server_factory(
            self._agent_id,
            self._repo_path,
            emit_lock_event,
        )

    def build(self) -> AgentRuntime:
        """Build the agent runtime configuration.

        Creates all caches, hooks, environment, and SDK options. All SDK
        imports happen here to preserve lazy-import guarantees.

        Returns:
            AgentRuntime with all configured components.

        Raises:
            RuntimeError: If required configuration is missing.
        """
        # Import hooks locally for lazy-import guarantees
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

        # Create caches
        file_read_cache = FileReadCache()
        lint_cache = LintCache(
            repo_path=self._repo_path,
            lint_tools=self._lint_tools,
        )

        # Build pre-tool hooks (order matters)
        pre_tool_hooks: list[object] = [
            block_dangerous_commands,
            make_commit_guard_hook(self._agent_id, str(self._repo_path)),
            make_file_read_cache_hook(file_read_cache),
            make_lint_cache_hook(lint_cache),
        ]

        # Conditionally add lock enforcement hook (requires locking MCP tools)
        if self._include_lock_enforcement_hook:
            pre_tool_hooks.insert(
                1, make_lock_enforcement_hook(self._agent_id, str(self._repo_path))
            )

        # Conditionally add mala disallowed tools hook (not needed for fixer agents)
        if self._include_mala_disallowed_tools_hook:
            pre_tool_hooks.insert(1, block_mala_disallowed_tools)

        post_tool_hooks: list[object] = []
        stop_hooks: list[object] = []

        if self._include_stop_hook:
            stop_hooks.append(make_stop_hook(self._agent_id))

        # Add deadlock monitor hooks if configured
        monitor = self._deadlock_monitor
        if monitor is not None:
            logger.info("Wiring deadlock monitor hooks: agent_id=%s", self._agent_id)
            # Import LockEvent types here to inject into hooks
            from src.core.models import LockEvent, LockEventType

            # PostToolUse hook for ACQUIRED/RELEASED events from MCP locking tools
            # (WAITING events are emitted by the MCP tool handlers directly)
            post_tool_hooks.append(
                make_lock_event_hook(
                    agent_id=self._agent_id,
                    emit_event=monitor.handle_event,
                    repo_namespace=str(self._repo_path),
                    lock_event_class=LockEvent,
                    lock_event_type_enum=LockEventType,
                )
            )
        else:
            logger.info(
                "No deadlock monitor configured; locking MCP tools available but events not tracked"
            )

        # Build environment if not explicitly set
        if self._env is None:
            self.with_env()
        env = self._env or {}

        # Build MCP servers if not explicitly set
        if self._mcp_servers is None:
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

        # Build hooks dict using factory
        make_matcher = self._sdk_client_factory.create_hook_matcher
        precompact_hook = make_precompact_hook(self._repo_path)
        hooks_dict: dict[str, list[object]] = {
            "PreToolUse": [make_matcher(None, pre_tool_hooks)],
            "PreCompact": [make_matcher(None, [precompact_hook])],
        }
        if stop_hooks:
            hooks_dict["Stop"] = [make_matcher(None, stop_hooks)]
        if post_tool_hooks:
            hooks_dict["PostToolUse"] = [make_matcher(None, post_tool_hooks)]

        logger.debug(
            "Built hooks: PreToolUse=%d PostToolUse=%d Stop=%d PreCompact=1",
            len(pre_tool_hooks),
            len(post_tool_hooks),
            len(stop_hooks),
        )

        # Build SDK options
        options = self._sdk_client_factory.create_options(
            cwd=str(self._repo_path),
            mcp_servers=self._mcp_servers,
            disallowed_tools=self._disallowed_tools,
            env=env,
            hooks=hooks_dict,
            setting_sources=self._setting_sources,
            settings='{"autoCompactEnabled": true}',
        )

        return AgentRuntime(
            options=options,
            file_read_cache=file_read_cache,
            lint_cache=lint_cache,
            env=env,
            pre_tool_hooks=pre_tool_hooks,
            post_tool_hooks=post_tool_hooks,
            stop_hooks=stop_hooks,
        )
