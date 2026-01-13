"""Agent lifecycle event protocols.

This module defines protocols for agent-level events and SDK message streaming.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class AgentLifecycleEvents(Protocol):
    """Protocol for agent lifecycle events.

    These events track individual agent spawning, completion, and messaging.
    """

    # -------------------------------------------------------------------------
    # Agent lifecycle
    # -------------------------------------------------------------------------

    def on_agent_started(self, agent_id: str, issue_id: str) -> None:
        """Called when an agent is spawned for an issue.

        Args:
            agent_id: Unique agent identifier.
            issue_id: Issue being worked on.
        """
        ...

    def on_agent_completed(
        self,
        agent_id: str,
        issue_id: str,
        success: bool,
        duration_seconds: float,
        summary: str,
    ) -> None:
        """Called when an agent completes (success or failure).

        Args:
            agent_id: Unique agent identifier.
            issue_id: Issue that was worked on.
            success: Whether the agent succeeded.
            duration_seconds: Total execution time.
            summary: Result summary or error message.
        """
        ...

    def on_claim_failed(self, agent_id: str, issue_id: str) -> None:
        """Called when claiming an issue fails.

        Args:
            agent_id: Agent that attempted the claim.
            issue_id: Issue that could not be claimed.
        """
        ...

    # -------------------------------------------------------------------------
    # SDK message streaming
    # -------------------------------------------------------------------------

    def on_tool_use(
        self,
        agent_id: str,
        tool_name: str,
        description: str = "",
        arguments: dict[str, Any] | None = None,
    ) -> None:
        """Called when an agent invokes a tool.

        Args:
            agent_id: Agent invoking the tool.
            tool_name: Name of the tool being called.
            description: Brief description of the action.
            arguments: Tool arguments (may be truncated for display).
        """
        ...

    def on_agent_text(self, agent_id: str, text: str) -> None:
        """Called when an agent emits text output.

        Args:
            agent_id: Agent emitting text.
            text: Text content (may be truncated for display).
        """
        ...
