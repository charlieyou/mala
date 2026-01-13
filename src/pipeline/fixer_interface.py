"""FixerInterface protocol for session_end remediation.

Provides a narrow interface for invoking fixer agents, avoiding tight coupling
between the session_end callback and RunCoordinator implementation details.

Design principles:
- Protocol-based for testability and flexibility
- Simple signature suitable for session_end callback usage
- Adapter pattern to wrap FixerService
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from src.pipeline.fixer_service import FixerService


@dataclass
class FixerResult:
    """Result from a fixer agent session.

    Attributes:
        success: Whether the fixer completed successfully. None if not evaluated.
        interrupted: Whether the fixer was interrupted by SIGINT.
        log_path: Path to the fixer session log (if captured).
    """

    success: bool | None
    interrupted: bool = False
    log_path: str | None = None


class FixerInterface(Protocol):
    """Protocol for invoking fixer agents.

    This interface abstracts fixer agent invocation for session_end remediation,
    allowing injection of a narrow interface rather than the full RunCoordinator.
    """

    async def run_fixer(self, failure_output: str, issue_id: str) -> FixerResult:
        """Run a fixer agent to address validation failures.

        Args:
            failure_output: Human-readable description of what failed.
            issue_id: The issue ID for context (used in logging/agent naming).

        Returns:
            FixerResult indicating success/failure/interrupted status.
        """
        ...


class FixerServiceAdapter:
    """Adapter that wraps FixerService to satisfy FixerInterface.

    This adapter provides a simplified interface over FixerService.run_fixer(),
    making it suitable for injection into the session_end callback.
    """

    def __init__(self, fixer_service: FixerService, max_attempts: int = 3) -> None:
        """Initialize adapter with a FixerService instance.

        Args:
            fixer_service: The FixerService to delegate fixer calls to.
            max_attempts: Maximum number of fixer attempts (for prompt context).
        """
        self._fixer_service = fixer_service
        self._max_attempts = max_attempts
        self._attempt_counter: dict[str, int] = {}

    async def run_fixer(self, failure_output: str, issue_id: str) -> FixerResult:
        """Run a fixer agent via the wrapped FixerService.

        Args:
            failure_output: Human-readable description of what failed.
            issue_id: The issue ID for context.

        Returns:
            FixerResult from the underlying service.
        """
        from src.pipeline.fixer_service import FailureContext

        # Track attempts per issue for multi-call scenarios
        attempt = self._attempt_counter.get(issue_id, 0) + 1
        self._attempt_counter[issue_id] = attempt

        # Build context and delegate to FixerService
        context = FailureContext(
            failure_output=failure_output,
            attempt=attempt,
            max_attempts=self._max_attempts,
            failed_command="session_end validation",
        )
        result = await self._fixer_service.run_fixer(context, interrupt_event=None)

        # Convert to our FixerResult (same structure, but our type)
        return FixerResult(
            success=result.success,
            interrupted=result.interrupted,
            log_path=result.log_path,
        )


# Keep backward compatibility alias
RunCoordinatorFixerAdapter = FixerServiceAdapter
