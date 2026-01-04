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
from typing import TYPE_CHECKING

from src.infra.tools.env import SCRIPTS_DIR, get_lock_dir

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pathlib import Path

    from src.core.protocols import DeadlockMonitorProtocol, SDKClientFactoryProtocol
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
    ) -> None:
        """Initialize the builder.

        Args:
            repo_path: Path to the repository root.
            agent_id: Unique agent identifier for lock management.
            sdk_client_factory: Factory for creating SDK options and matchers.
        """
        self._repo_path = repo_path
        self._agent_id = agent_id
        self._sdk_client_factory = sdk_client_factory

        # Lint tools configuration
        self._lint_tools: set[str] | frozenset[str] | None = None

        # Hook configuration
        self._deadlock_monitor: DeadlockMonitorProtocol | None = None
        self._include_stop_hook: bool = True
        self._include_mala_disallowed_tools_hook: bool = True

        # Environment and options
        self._env: dict[str, str] | None = None
        self._mcp_servers: object | None = None
        self._disallowed_tools: list[str] | None = None

    def with_hooks(
        self,
        *,
        deadlock_monitor: DeadlockMonitorProtocol | None = None,
        include_stop_hook: bool = True,
        include_mala_disallowed_tools_hook: bool = True,
    ) -> AgentRuntimeBuilder:
        """Configure hook behavior.

        Args:
            deadlock_monitor: Optional DeadlockMonitor for lock event hooks.
            include_stop_hook: Whether to include stop hook (default True).
            include_mala_disallowed_tools_hook: Whether to include the
                block_mala_disallowed_tools hook (default True). Set False
                for fixer agents which don't need this restriction.

        Returns:
            Self for chaining.
        """
        self._deadlock_monitor = deadlock_monitor
        self._include_stop_hook = include_stop_hook
        self._include_mala_disallowed_tools_hook = include_mala_disallowed_tools_hook
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

    def with_mcp(self, servers: object | None = None) -> AgentRuntimeBuilder:
        """Configure MCP servers.

        Args:
            servers: MCP server configuration. If None, uses get_mcp_servers().

        Returns:
            Self for chaining.
        """
        if servers is not None:
            self._mcp_servers = servers
        else:
            from src.infra.mcp import get_mcp_servers

            self._mcp_servers = get_mcp_servers(self._repo_path)
        return self

    def with_disallowed_tools(
        self, tools: list[str] | None = None
    ) -> AgentRuntimeBuilder:
        """Configure disallowed tools.

        Args:
            tools: List of disallowed tool names. If None, uses get_disallowed_tools().

        Returns:
            Self for chaining.
        """
        if tools is not None:
            self._disallowed_tools = tools
        else:
            from src.infra.mcp import get_disallowed_tools

            self._disallowed_tools = get_disallowed_tools()
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
            make_file_read_cache_hook,
            make_lint_cache_hook,
            make_lock_enforcement_hook,
            make_lock_event_hook,
            make_lock_wait_hook,
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
            make_lock_enforcement_hook(self._agent_id, str(self._repo_path)),
            make_file_read_cache_hook(file_read_cache),
            make_lint_cache_hook(lint_cache),
        ]

        # Conditionally add mala disallowed tools hook (not needed for fixer agents)
        if self._include_mala_disallowed_tools_hook:
            pre_tool_hooks.insert(1, block_mala_disallowed_tools)

        post_tool_hooks: list[object] = []
        stop_hooks: list[object] = []

        if self._include_stop_hook:
            stop_hooks.append(make_stop_hook(self._agent_id))

        # Add deadlock monitor hooks if configured
        if self._deadlock_monitor is not None:
            logger.info("Wiring deadlock monitor hooks: agent_id=%s", self._agent_id)
            # Import LockEvent types here to inject into hooks
            from src.core.models import LockEvent, LockEventType

            monitor = self._deadlock_monitor
            # PreToolUse hook for real-time WAITING detection on lock-wait.sh
            pre_tool_hooks.append(
                make_lock_wait_hook(
                    agent_id=self._agent_id,
                    emit_event=monitor.handle_event,
                    repo_namespace=str(self._repo_path),
                    lock_event_class=LockEvent,
                    lock_event_type_enum=LockEventType,
                )
            )
            # PostToolUse hook for ACQUIRED/RELEASED events
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
            logger.info("No deadlock monitor configured; skipping lock event hooks")

        # Build environment if not explicitly set
        if self._env is None:
            self.with_env()
        env = self._env or {}

        # Build hooks dict using factory
        make_matcher = self._sdk_client_factory.create_hook_matcher
        hooks_dict: dict[str, list[object]] = {
            "PreToolUse": [make_matcher(None, pre_tool_hooks)],
        }
        if stop_hooks:
            hooks_dict["Stop"] = [make_matcher(None, stop_hooks)]
        if post_tool_hooks:
            hooks_dict["PostToolUse"] = [make_matcher(None, post_tool_hooks)]

        logger.debug(
            "Built hooks: PreToolUse=%d PostToolUse=%d Stop=%d",
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
