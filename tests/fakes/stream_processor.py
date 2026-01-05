"""Fake stream processor for testing.

Provides FakeStreamProcessor implementing the MessageStreamProcessor interface
for isolated testing of idle retry policy and other stream consumers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.pipeline.message_stream_processor import (
    IdleTimeoutError,
    MessageIterationResult,
)

if TYPE_CHECKING:
    from src.domain.lifecycle import LifecycleContext
    from src.infra.telemetry import TelemetrySpan
    from src.pipeline.message_stream_processor import (
        IdleTimeoutStream,
        LintCacheProtocol,
        MessageIterationState,
    )


@dataclass
class FakeStreamProcessor:
    """In-memory stream processor for testing.

    Configurable to return success, raise timeout, or use custom behavior.

    Usage:
        # Simple success case
        processor = FakeStreamProcessor(
            result=MessageIterationResult(success=True, session_id="test-123")
        )

        # Timeout on first N calls, then succeed
        processor = FakeStreamProcessor(
            timeout_until=2,
            result=MessageIterationResult(success=True, session_id="resumed")
        )

        # Custom side effect
        processor = FakeStreamProcessor(side_effect=CustomException("error"))

    Attributes:
        result: Pre-configured result to return on success.
        timeout_until: Number of calls that should timeout before success.
        side_effect: Exception to raise instead of returning result.
        calls: List of captured call arguments for assertions.
    """

    result: MessageIterationResult = field(
        default_factory=lambda: MessageIterationResult(success=True, session_id="fake")
    )
    timeout_until: int = 0
    side_effect: Exception | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)
    _call_count: int = field(default=0, init=False)

    async def process_stream(
        self,
        stream: IdleTimeoutStream,
        issue_id: str,
        state: MessageIterationState,
        lifecycle_ctx: LifecycleContext,
        lint_cache: LintCacheProtocol,
        query_start: float,
        tracer: TelemetrySpan | None,
    ) -> MessageIterationResult:
        """Process stream and return configured result.

        Records all calls for test assertions.
        """
        self._call_count += 1
        self.calls.append(
            {
                "stream": stream,
                "issue_id": issue_id,
                "state": state,
                "lifecycle_ctx": lifecycle_ctx,
                "lint_cache": lint_cache,
                "query_start": query_start,
                "tracer": tracer,
            }
        )

        if self.side_effect is not None:
            raise self.side_effect

        if self._call_count <= self.timeout_until:
            raise IdleTimeoutError("idle timeout")

        return self.result

    @property
    def call_count(self) -> int:
        """Number of times process_stream was called."""
        return self._call_count
