"""SDK client protocols for Claude agent abstraction.

This module defines protocols for abstracting Claude SDK client operations,
enabling the pipeline layer to use SDK clients without importing
claude_agent_sdk directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from collections.abc import Callable

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from types import TracebackType
    from typing import Self


# Factory function for creating MCP servers configuration.
# Parameters: agent_id, repo_path, optional emit_lock_event callback
# Returns: Dict mapping server names to server configurations
McpServerFactory = Callable[[str, Path, Callable | None], dict[str, object]]


@runtime_checkable
class SDKClientProtocol(Protocol):
    """Protocol for Claude SDK client abstraction.

    Enables the pipeline layer to use SDK clients without importing
    claude_agent_sdk directly. The canonical implementation is
    ClaudeSDKClient, wrapped by SDKClientFactory in infra.

    This protocol captures the async context manager and streaming
    interface used by AgentSessionRunner.
    """

    async def __aenter__(self) -> Self:
        """Enter async context."""
        ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit async context."""
        ...

    async def query(self, prompt: str, session_id: str | None = None) -> None:
        """Send a query to the agent.

        Args:
            prompt: The prompt text to send.
            session_id: Optional session ID for continuation.
        """
        ...

    def receive_response(self) -> AsyncIterator[object]:
        """Get an async iterator of response messages.

        Returns:
            AsyncIterator yielding AssistantMessage, ResultMessage, etc.
        """
        ...

    async def disconnect(self) -> None:
        """Disconnect the client."""
        ...


@runtime_checkable
class SDKClientFactoryProtocol(Protocol):
    """Protocol for SDK client factory.

    Enables dependency injection of the factory into pipeline components,
    allowing tests to provide mock factories.
    """

    def create(self, options: object) -> SDKClientProtocol:
        """Create a new SDK client with the given options.

        Args:
            options: ClaudeAgentOptions for the client.

        Returns:
            SDKClientProtocol instance.
        """
        ...

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
    ) -> object:
        """Create SDK options without requiring SDK import in caller.

        Args:
            cwd: Working directory for the agent.
            permission_mode: Permission mode.
            model: Model to use.
            system_prompt: System prompt configuration.
            output_format: Optional structured output format configuration.
            settings: JSON settings string passed to ClaudeAgentOptions.
            setting_sources: List of setting sources.
            mcp_servers: List of MCP server configurations.
            disallowed_tools: List of tools to disallow.
            env: Environment variables for the agent.
            hooks: Hook configurations keyed by event type.
            resume: Session ID to resume from. When set, the SDK loads
                the prior conversation context before processing the query.

        Returns:
            ClaudeAgentOptions instance.
        """
        ...

    def create_hook_matcher(
        self,
        matcher: object | None,
        hooks: list[object],
    ) -> object:
        """Create a HookMatcher for SDK hook registration.

        Args:
            matcher: Optional matcher configuration.
            hooks: List of hook callables.

        Returns:
            HookMatcher instance.
        """
        ...

    def with_resume(self, options: object, resume: str | None) -> object:
        """Create a copy of options with a different resume session ID.

        This is used to resume a prior session when retrying after idle timeout
        or review failures. The SDK's resume feature loads the prior conversation
        context before processing the next query.

        Args:
            options: Existing ClaudeAgentOptions to clone.
            resume: Session ID to resume from, or None to start fresh.

        Returns:
            New ClaudeAgentOptions with the resume field set.
        """
        ...
