"""FixerInterface protocol for session_end remediation.

Provides a narrow interface for invoking fixer agents, avoiding tight coupling
between the session_end callback and RunCoordinator implementation details.

Design principles:
- Protocol-based for testability and flexibility
- Simple signature suitable for session_end callback usage
- Adapter pattern to wrap existing RunCoordinator._run_fixer_agent()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from src.pipeline.run_coordinator import RunCoordinator


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


class RunCoordinatorFixerAdapter:
    """Adapter that wraps RunCoordinator to satisfy FixerInterface.

    This adapter provides a simplified interface over RunCoordinator._run_fixer_agent(),
    making it suitable for injection into the session_end callback.
    """

    def __init__(self, coordinator: RunCoordinator) -> None:
        """Initialize adapter with a RunCoordinator instance.

        Args:
            coordinator: The RunCoordinator to delegate fixer calls to.
        """
        self._coordinator = coordinator
        self._attempt_counter: dict[str, int] = {}

    async def run_fixer(self, failure_output: str, issue_id: str) -> FixerResult:
        """Run a fixer agent via the wrapped RunCoordinator.

        Args:
            failure_output: Human-readable description of what failed.
            issue_id: The issue ID for context.

        Returns:
            FixerResult from the underlying coordinator.
        """
        # Track attempts per issue for multi-call scenarios
        attempt = self._attempt_counter.get(issue_id, 0) + 1
        self._attempt_counter[issue_id] = attempt

        # Call the coordinator's internal fixer method
        result = await self._coordinator._run_fixer_agent(
            failure_output=failure_output,
            attempt=attempt,
            spec=None,  # session_end doesn't have spec context
            interrupt_event=None,  # session_end handles its own interrupts
            failed_command="session_end validation",
        )

        # Convert to our FixerResult (same structure, but our type)
        # Access attributes directly to avoid import issues
        return FixerResult(
            success=result.success,
            interrupted=result.interrupted,
            log_path=result.log_path,
        )
