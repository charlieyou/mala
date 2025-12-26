"""
Braintrust integration for Claude Agent SDK.

LLM spans are automatically traced by the braintrust.wrappers.claude_agent_sdk wrapper
which is set up in cli.py BEFORE importing claude_agent_sdk.

This module provides:
- TracedAgentExecution: Context manager for creating parent spans with issue metadata
- flush_braintrust: Ensure all traces are sent before process exits

Usage:
    # In cli.py (BEFORE importing claude_agent_sdk):
    from braintrust.wrappers.claude_agent_sdk import setup_claude_agent_sdk
    setup_claude_agent_sdk(project="mala")

    # Then in agent code:
    from .braintrust_integration import TracedAgentExecution, flush_braintrust

    with TracedAgentExecution(issue_id, agent_id) as tracer:
        tracer.log_input(prompt)
        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt)
            # LLM calls are auto-traced by the wrapper
            async for message in client.receive_response():
                tracer.log_message(message)  # For output/tool tracking
        tracer.set_success(True)
"""

import os
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
)

# Braintrust imports - gracefully handle if not configured
try:
    import braintrust
    from braintrust import start_span

    BRAINTRUST_AVAILABLE = True
except ImportError:
    BRAINTRUST_AVAILABLE = False
    braintrust = None
    start_span = None


def is_braintrust_enabled() -> bool:
    """Check if Braintrust is available and configured."""
    return BRAINTRUST_AVAILABLE and os.environ.get("BRAINTRUST_API_KEY") is not None


def flush_braintrust() -> None:
    """Flush pending logs to Braintrust."""
    if BRAINTRUST_AVAILABLE and braintrust is not None:
        try:
            braintrust.flush()
        except Exception:
            pass


class TracedAgentExecution:
    """
    Context manager for tracing a single agent execution.

    Creates a parent span for the agent with issue-level metadata.
    LLM calls within this context are automatically traced by the wrapper.

    Captures:
    - Initial prompt (input)
    - Tool call counts
    - Final result (output)
    - Success/failure status
    """

    def __init__(
        self, issue_id: str, agent_id: str, metadata: dict[str, Any] | None = None
    ):
        self.issue_id = issue_id
        self.agent_id = agent_id
        self.metadata = metadata or {}
        self.span = None
        self.input_prompt: str | None = None
        self.output_text: str = ""
        self.tool_calls: list[dict[str, Any]] = []
        self.success: bool = False
        self.error: str | None = None

    def __enter__(self):
        if not is_braintrust_enabled():
            return self

        # Start the root span for this agent execution
        try:
            self.span = start_span(
                name=f"agent:{self.issue_id}",
                type="task",
                metadata={
                    "issue_id": self.issue_id,
                    "agent_id": self.agent_id,
                    **self.metadata,
                },
            )
            self.span.__enter__()
        except Exception as e:
            import sys

            print(f"[braintrust] Failed to start span: {e}", file=sys.stderr)
            self.span = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.span is None:
            return

        try:
            # Record the final output and metrics
            if exc_type is not None:
                self.error = str(exc_val)
                self.success = False

            self.span.log(
                input=self.input_prompt,
                output=self.output_text,
                metadata={
                    "success": self.success,
                    "error": self.error,
                    "tool_calls_count": len(self.tool_calls),
                },
                scores={"success": 1.0 if self.success else 0.0},
            )
            self.span.__exit__(exc_type, exc_val, exc_tb)
            # Flush to ensure data is sent before process exits
            flush_braintrust()
        except Exception as e:
            import sys

            print(f"[braintrust] Failed to close span: {e}", file=sys.stderr)
            # Suppress the exception - tracing is best-effort

    def log_input(self, prompt: str):
        """Log the initial user prompt."""
        self.input_prompt = prompt

    def log_message(self, message: Any):
        """Log a message from the Claude Agent SDK (for output/tool tracking)."""
        try:
            if isinstance(message, AssistantMessage):
                self._handle_assistant_message(message)
            elif isinstance(message, ResultMessage):
                self._handle_result_message(message)
        except Exception:
            pass  # Best-effort tracking

    def _handle_assistant_message(self, message: AssistantMessage):
        """Process an assistant message, tracking text and tool calls."""
        for block in message.content:
            if isinstance(block, TextBlock):
                self.output_text += block.text + "\n"

            elif isinstance(block, ToolUseBlock):
                # Track tool calls for metadata (LLM spans are auto-traced by wrapper)
                tool_use_id = getattr(block, "id", f"tool_{len(self.tool_calls)}")
                tool_name = block.name
                tool_input = block.input

                self.tool_calls.append(
                    {
                        "id": tool_use_id,
                        "name": tool_name,
                        "input": tool_input,
                    }
                )

            elif isinstance(block, ToolResultBlock):
                pass  # Tool results tracked in wrapper

    def _handle_result_message(self, message: ResultMessage):
        """Process the final result message."""
        self.output_text = message.result or self.output_text

    def set_success(self, success: bool):
        """Mark the execution as successful or failed."""
        self.success = success

    def set_error(self, error: str):
        """Record an error message."""
        self.error = error
        self.success = False
