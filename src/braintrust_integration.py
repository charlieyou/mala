"""
Braintrust integration for Claude Agent SDK.

Provides tracing and observability for agent executions using Braintrust's
logging infrastructure. This integration captures:
- Agent queries and responses
- Tool invocations with inputs/outputs
- Execution timing and success/failure states

Usage:
    from .braintrust_integration import init_braintrust, TracedAgentExecution

    # Initialize at startup
    init_braintrust(project_name="my-project")

    # Use TracedAgentExecution context manager for agent runs
    with TracedAgentExecution(issue_id, agent_id) as tracer:
        tracer.log_input(prompt)
        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt)
            async for message in client.receive_response():
                tracer.log_message(message)
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
    from braintrust import init_logger, start_span

    BRAINTRUST_AVAILABLE = True
except ImportError:
    BRAINTRUST_AVAILABLE = False
    braintrust = None
    init_logger = None
    start_span = None


# Global logger reference
_logger = None
# Disable tracing if a failure occurs (best-effort)
_tracing_disabled = False


def init_braintrust(project_name: str = "mala") -> bool:
    """
    Initialize Braintrust logging.

    Requires BRAINTRUST_API_KEY environment variable to be set.
    Returns True if initialization was successful, False otherwise.
    """
    global _logger

    if not BRAINTRUST_AVAILABLE:
        return False

    api_key = os.environ.get("BRAINTRUST_API_KEY")
    if not api_key:
        return False

    try:
        _logger = init_logger(project=project_name)
        return True
    except Exception:
        return False


def is_braintrust_enabled() -> bool:
    """Check if Braintrust is initialized and ready."""
    return _logger is not None and not _tracing_disabled


def flush_braintrust() -> None:
    """Flush pending logs to Braintrust."""
    if _logger is not None:
        try:
            _logger.flush()
        except Exception:
            pass


def _disable_tracing(error: Exception) -> None:
    """Disable tracing after a failure (best-effort tracing)."""
    global _tracing_disabled
    _tracing_disabled = True
    import sys

    print(f"[braintrust] Tracing disabled: {error}", file=sys.stderr)


class TracedAgentExecution:
    """
    Context manager for tracing a single agent execution.

    Captures the full agent lifecycle including:
    - Initial prompt (input)
    - Tool call counts (nested tool spans disabled for SDK compatibility)
    - Final result (output)
    - Success/failure status
    - Duration timing
    """

    def __init__(
        self, issue_id: str, agent_id: str, metadata: dict[str, Any] | None = None
    ):
        self.issue_id = issue_id
        self.agent_id = agent_id
        self.metadata = metadata or {}
        self.span = None
        self.tool_spans: dict[str, Any] = {}  # Track active tool spans by tool_use_id
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
            _disable_tracing(e)
            self.span = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not is_braintrust_enabled() or self.span is None:
            return

        try:
            # Close any open tool spans
            for tool_span in self.tool_spans.values():
                try:
                    tool_span.__exit__(None, None, None)
                except Exception:
                    pass
            self.tool_spans.clear()

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
            _disable_tracing(e)
            # Suppress the exception - tracing is best-effort

    def log_input(self, prompt: str):
        """Log the initial user prompt."""
        self.input_prompt = prompt

    def log_message(self, message: Any):
        """Log a message from the Claude Agent SDK."""
        if not is_braintrust_enabled():
            return

        try:
            if isinstance(message, AssistantMessage):
                self._handle_assistant_message(message)
            elif isinstance(message, ResultMessage):
                self._handle_result_message(message)
        except Exception as e:
            _disable_tracing(e)
            # Suppress the exception - tracing is best-effort

    def _handle_assistant_message(self, message: AssistantMessage):
        """Process an assistant message, logging tools and text."""
        for block in message.content:
            if isinstance(block, TextBlock):
                self.output_text += block.text + "\n"

            elif isinstance(block, ToolUseBlock):
                # Track tool calls for metadata
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
                # NOTE: Nested tool spans disabled - parent span reference causes
                # SpanComponents encoding errors with some SDK versions

            elif isinstance(block, ToolResultBlock):
                # Tool spans disabled - nothing to close
                pass

    def _handle_result_message(self, message: ResultMessage):
        """Process the final result message."""
        self.output_text = message.result or self.output_text
        # Success is determined externally based on issue status

    def set_success(self, success: bool):
        """Mark the execution as successful or failed."""
        self.success = success

    def set_error(self, error: str):
        """Record an error message."""
        self.error = error
        self.success = False
