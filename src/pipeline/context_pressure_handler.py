"""ContextPressureHandler: Context pressure checkpoint and restart handling.

Extracted from AgentSessionRunner to separate checkpoint/restart logic from
session lifecycle management. This module handles:
- Fetching checkpoints from agents via SDK client
- Building continuation prompts with checkpoint context
- Managing restart loop state (continuation count, prompts)

Design principles:
- Protocol-based SDK client for testability
- Explicit error type for pressure threshold detection
- Timeout handling for checkpoint fetch operations
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)

from src.domain.prompts import (
    build_continuation_prompt,
    extract_checkpoint,
)
from src.pipeline.message_stream_processor import ContextPressureError

if TYPE_CHECKING:
    from src.core.protocols.sdk import SDKClientFactoryProtocol


logger = logging.getLogger(__name__)

# Default timeout for checkpoint fetch operations (30 seconds)
DEFAULT_CHECKPOINT_TIMEOUT_SECONDS = 30


@dataclass
class ContextPressureConfig:
    """Configuration for context pressure handling.

    Attributes:
        checkpoint_request_prompt: Prompt template to request checkpoint from agent.
        continuation_template: Template for continuation prompt with checkpoint.
        checkpoint_timeout_seconds: Timeout for checkpoint fetch operations.
    """

    checkpoint_request_prompt: str
    continuation_template: str
    checkpoint_timeout_seconds: float = DEFAULT_CHECKPOINT_TIMEOUT_SECONDS


@dataclass
class CheckpointResult:
    """Result of fetching a checkpoint from an agent.

    Attributes:
        checkpoint: Extracted checkpoint text (may be empty).
        timed_out: Whether the fetch timed out.
    """

    checkpoint: str
    timed_out: bool = False


class ContextPressureHandler:
    """Handles context pressure detection and checkpoint/restart logic.

    This handler encapsulates:
    - Fetching checkpoints from agents before restart
    - Building continuation prompts with checkpoint context
    - Managing restart loop state

    The handler is stateless per-call; restart state is managed by the caller.
    """

    def __init__(
        self,
        config: ContextPressureConfig,
        sdk_client_factory: SDKClientFactoryProtocol,
    ) -> None:
        """Initialize the handler.

        Args:
            config: Context pressure configuration.
            sdk_client_factory: Factory for creating SDK clients.
        """
        self.config = config
        self.sdk_client_factory = sdk_client_factory

    async def fetch_checkpoint(
        self,
        session_id: str,
        issue_id: str,
        options: object,
        timeout_seconds: float | None = None,
    ) -> CheckpointResult:
        """Fetch checkpoint from agent before context restart.

        Sends checkpoint_request_prompt to the current session and extracts
        the checkpoint block from the response.

        Args:
            session_id: SDK session ID from ContextPressureError.
            issue_id: Issue ID for logging.
            options: SDK client options.
            timeout_seconds: Optional override for checkpoint timeout.

        Returns:
            CheckpointResult with extracted checkpoint text.
        """
        effective_timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else self.config.checkpoint_timeout_seconds
        )

        logger.info(
            "Session %s: requesting checkpoint from session %s...",
            issue_id,
            session_id[:8] if session_id else "unknown",
        )

        checkpoint_prompt = self.config.checkpoint_request_prompt
        if not checkpoint_prompt:
            logger.warning(
                "Session %s: no checkpoint_request prompt configured, using empty checkpoint",
                issue_id,
            )
            return CheckpointResult(checkpoint="", timed_out=False)

        # Create client with resume to load prior context for checkpoint extraction
        resumed_options = self.sdk_client_factory.with_resume(options, session_id)
        client = self.sdk_client_factory.create(resumed_options)

        response_text = ""
        try:
            async with asyncio.timeout(effective_timeout):
                async with client:
                    await client.query(checkpoint_prompt)
                    async for message in client.receive_response():
                        # Extract text from AssistantMessage
                        content = getattr(message, "content", None)
                        if content is not None:
                            for block in cast("list[Any]", content):
                                text = getattr(block, "text", None)
                                if text is not None:
                                    response_text += text
        except TimeoutError:
            logger.warning(
                "Session %s: checkpoint fetch timed out after %.1fs, using empty checkpoint",
                issue_id,
                effective_timeout,
            )
            return CheckpointResult(checkpoint="", timed_out=True)
        except Exception as e:
            logger.warning(
                "Session %s: checkpoint query failed: %s, using empty checkpoint",
                issue_id,
                e,
            )
            return CheckpointResult(checkpoint="", timed_out=False)

        # Extract checkpoint from response
        checkpoint = extract_checkpoint(response_text)
        logger.debug(
            "Session %s: extracted checkpoint (%d chars)",
            issue_id,
            len(checkpoint),
        )
        return CheckpointResult(checkpoint=checkpoint, timed_out=False)

    def build_continuation_prompt(self, checkpoint: str) -> str:
        """Build continuation prompt with checkpoint context.

        Args:
            checkpoint: Extracted checkpoint text from previous session.

        Returns:
            Formatted continuation prompt for restart.
        """
        continuation_template = self.config.continuation_template
        if continuation_template:
            return build_continuation_prompt(continuation_template, checkpoint)
        # Fallback: just use checkpoint as prompt
        return f"Continue from checkpoint:\n\n{checkpoint}"

    async def handle_pressure_error(
        self,
        error: ContextPressureError,
        issue_id: str,
        options: object,
        continuation_count: int,
        remaining_time: float,
    ) -> tuple[str, int]:
        """Handle a ContextPressureError by fetching checkpoint and building continuation.

        This is the main entry point for handling context pressure. It:
        1. Calculates effective timeout based on remaining session time
        2. Fetches checkpoint from the current session
        3. Builds the continuation prompt
        4. Logs restart information

        Args:
            error: The ContextPressureError that triggered the restart.
            issue_id: Issue ID for logging.
            options: SDK client options.
            continuation_count: Current restart count (will be incremented).
            remaining_time: Remaining session time in seconds.

        Returns:
            Tuple of (continuation_prompt, new_continuation_count).
        """
        # Calculate effective timeout bounded by remaining session time
        effective_timeout = max(
            0, min(remaining_time, self.config.checkpoint_timeout_seconds)
        )

        # Fetch checkpoint from current session
        result = await self.fetch_checkpoint(
            session_id=error.session_id,
            issue_id=issue_id,
            options=options,
            timeout_seconds=effective_timeout,
        )

        new_count = continuation_count + 1
        logger.info(
            "Session %s: context restart #%d at %.1f%%",
            issue_id,
            new_count,
            error.pressure_ratio * 100,
        )

        # Build continuation prompt
        continuation_prompt = self.build_continuation_prompt(result.checkpoint)

        return continuation_prompt, new_count


# Re-export ContextPressureError for convenience
__all__ = [
    "DEFAULT_CHECKPOINT_TIMEOUT_SECONDS",
    "CheckpointResult",
    "ContextPressureConfig",
    "ContextPressureError",
    "ContextPressureHandler",
]
