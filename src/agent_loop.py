"""
Reusable Claude SDK agent receive loop.

Extracted from MalaOrchestrator.run_implementer for modularity.

This module provides the core message receive loop that:
- Sends a prompt to the Claude SDK client
- Processes AssistantMessage and ResultMessage responses
- Logs to JSONL, Braintrust, and console
- Returns the final result text

Usage:
    from .agent_loop import run_agent_loop, AgentLoopResult

    result = await run_agent_loop(
        prompt=prompt,
        options=options,
        jsonl_logger=jsonl_logger,
        tracer=tracer,
        console_callback=my_console_callback,
    )
"""

from dataclasses import dataclass
from typing import Any, Callable

from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
)

from .braintrust_integration import TracedAgentExecution


# Type alias for console callback
# Callback receives the message and should handle console output
ConsoleCallback = Callable[[AssistantMessage | ResultMessage], None]


@dataclass
class AgentLoopResult:
    """Result from running the agent loop."""

    result_text: str
    """The final result text from the agent."""

    message_count: int
    """Number of messages received."""


class JSONLLoggerProtocol:
    """Protocol for JSONL logger (duck typing for JSONLLogger in main.py)."""

    def log_user_prompt(self, prompt: str) -> None:
        """Log the initial user prompt."""
        ...

    def log_message(self, message: Any) -> None:
        """Log a message from the SDK."""
        ...


async def run_agent_loop(
    prompt: str,
    options: ClaudeAgentOptions,
    jsonl_logger: JSONLLoggerProtocol | None = None,
    tracer: TracedAgentExecution | None = None,
    console_callback: ConsoleCallback | None = None,
) -> AgentLoopResult:
    """
    Run the Claude SDK agent loop.

    This is the core receive loop extracted from MalaOrchestrator.run_implementer.
    It handles:
    - Sending the prompt to the Claude SDK client
    - Receiving and processing messages
    - Logging to JSONL (if jsonl_logger provided)
    - Logging to Braintrust (if tracer provided)
    - Console output (if console_callback provided)

    Args:
        prompt: The prompt to send to the agent.
        options: ClaudeAgentOptions for configuring the client.
        jsonl_logger: Optional JSONL logger for full message logging.
        tracer: Optional TracedAgentExecution for Braintrust logging.
        console_callback: Optional callback for console output.
            Receives AssistantMessage or ResultMessage.

    Returns:
        AgentLoopResult with the final result text and message count.
    """
    final_result = ""
    message_count = 0

    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)

        # Log the initial prompt
        if jsonl_logger is not None:
            jsonl_logger.log_user_prompt(prompt)

        async for message in client.receive_response():
            message_count += 1

            # JSONL logging
            if jsonl_logger is not None:
                jsonl_logger.log_message(message)

            # Braintrust logging
            if tracer is not None:
                tracer.log_message(message)

            # Console output via callback (only for AssistantMessage/ResultMessage)
            if console_callback is not None and isinstance(
                message, (AssistantMessage, ResultMessage)
            ):
                console_callback(message)

            # Extract final result
            if isinstance(message, ResultMessage):
                final_result = message.result or ""

    return AgentLoopResult(
        result_text=final_result,
        message_count=message_count,
    )


def make_default_console_callback(
    issue_id: str,
) -> ConsoleCallback:
    """
    Create a default console callback that logs messages in Claude Code style.

    This mirrors the console output behavior from MalaOrchestrator.run_implementer.

    Args:
        issue_id: The issue ID for agent color coding.

    Returns:
        A callback function for console output.
    """
    from .logging.console import log_tool, log_agent_text

    def callback(message: AssistantMessage | ResultMessage) -> None:
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    log_agent_text(block.text, issue_id)
                elif isinstance(block, ToolUseBlock):
                    log_tool(
                        block.name,
                        str(block.input),
                        agent_id=issue_id,
                    )

    return callback
