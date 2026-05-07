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
    """Cross-coder factory protocol consumed by the pipeline.

    Slimmed to the surface every coder backend (Claude, Amp, Codex) must
    expose: spawn a client and amend a runtime with a resume token.
    Backend-specific knobs live on the concrete provider factory classes
    and never appear here.

    The ``runtime`` argument is opaque: each provider's factory privately
    unpacks the runtime shape it produced from its own
    :class:`CoderRuntimeBuilder`.
    """

    def create(self, runtime: object) -> SDKClientProtocol:
        """Create a new SDK client for the given coder-shaped runtime.

        Args:
            runtime: Opaque coder-shaped runtime produced by the matching
                provider's ``runtime_builder().build()``. The factory
                privately unpacks it (Claude → ``ClaudeAgentOptions``,
                Amp → ``AmpClientOptions``, etc.).

        Returns:
            SDKClientProtocol instance.
        """
        ...

    def with_resume(self, runtime: object, resume: str | None) -> object:
        """Return a runtime continuing from ``resume``.

        Used when retrying after idle timeout or review failures. Each
        provider maps ``resume`` onto its native resume primitive (Claude
        → SDK ``resume=`` field, Amp → thread continue id, etc.).

        Args:
            runtime: Existing coder-shaped runtime to clone.
            resume: Session/thread id to resume from, or None to start fresh.

        Returns:
            New coder-shaped runtime carrying the resume token.
        """
        ...
