"""Agent lifecycle event protocols.

This module defines protocols for agent-level events and SDK message streaming.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable


@runtime_checkable
class AgentLifecycleEvents(Protocol):
    """Protocol for agent lifecycle events.

    These events track individual agent spawning, completion, and messaging.
    """

    # -------------------------------------------------------------------------
    # Agent lifecycle
    # -------------------------------------------------------------------------

    def on_agent_started(
        self,
        agent_id: str,
        issue_id: str,
        *,
        coder: Literal["claude", "amp", "codex"] | None = None,
    ) -> None:
        """Called when an agent is spawned for an issue.

        Args:
            agent_id: Unique agent identifier.
            issue_id: Issue being worked on.
            coder: Active coder backend (``claude``, ``amp``, or
                ``codex``); enables per-coder dashboard breakdowns.
                Optional for back-compat.
        """
        ...

    def on_agent_completed(
        self,
        agent_id: str,
        issue_id: str,
        success: bool,
        duration_seconds: float,
        summary: str,
        *,
        coder: Literal["claude", "amp", "codex"] | None = None,
    ) -> None:
        """Called when an agent completes (success or failure).

        Args:
            agent_id: Unique agent identifier.
            issue_id: Issue that was worked on.
            success: Whether the agent succeeded.
            duration_seconds: Total execution time.
            summary: Result summary or error message.
            coder: Active coder backend (``claude``, ``amp``, or ``codex``).
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

    def on_background_wait(
        self,
        agent_id: str,
        tool_use_id: str,
        elapsed_seconds: float,
        budget_seconds: float,
    ) -> None:
        """Called periodically while waiting on a backgrounded task to finish.

        Surfaces the otherwise-silent between-turn wait that keeps the SDK client
        connected after an agent backgrounds work and yields, so the run does not
        look hung.

        Args:
            agent_id: Agent whose backgrounded task is being awaited.
            tool_use_id: tool_use id of the background launch being awaited.
            elapsed_seconds: Seconds spent waiting so far.
            budget_seconds: Maximum seconds the wait will run before giving up.
        """
        ...
