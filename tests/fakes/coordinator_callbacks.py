"""Fake coordinator callbacks for testing.

Provides fake implementations of SpawnCallback, FinalizeCallback, and AbortCallback
protocols for testing IssueExecutionCoordinator without real agent spawning.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from src.pipeline.issue_execution_coordinator import AbortResult


@dataclass
class FakeSpawnCallback:
    """Fake spawn callback that creates no-op tasks.

    Configurable behavior:
    - spawn_delay: How long spawn takes (for timing tests)
    - fail_issues: Set of issue IDs that should fail to spawn
    - task_duration: How long spawned tasks run before completing

    Observable state:
    - spawned_issues: List of issue IDs spawn was called with
    - spawned_tasks: Dict mapping issue_id to created task
    """

    spawn_delay: float = 0.0
    fail_issues: set[str] = field(default_factory=set)
    task_duration: float = 0.0
    spawned_issues: list[str] = field(default_factory=list)
    spawned_tasks: dict[str, asyncio.Task[Any]] = field(default_factory=dict)

    async def __call__(self, issue_id: str) -> asyncio.Task[Any] | None:
        """Spawn a fake task for the issue."""
        self.spawned_issues.append(issue_id)

        if self.spawn_delay > 0:
            await asyncio.sleep(self.spawn_delay)

        if issue_id in self.fail_issues:
            return None

        async def fake_agent_work() -> None:
            if self.task_duration > 0:
                await asyncio.sleep(self.task_duration)

        task = asyncio.create_task(fake_agent_work())
        self.spawned_tasks[issue_id] = task
        return task


@dataclass
class FakeFinalizeCallback:
    """Fake finalize callback that records completions.

    Observable state:
    - finalized_issues: List of (issue_id, task) tuples
    """

    finalized_issues: list[tuple[str, asyncio.Task[Any]]] = field(default_factory=list)
    coordinator: Any = None  # Optional coordinator reference for mark_completed

    async def __call__(self, issue_id: str, task: asyncio.Task[Any]) -> None:
        """Record the finalization."""
        self.finalized_issues.append((issue_id, task))
        if self.coordinator is not None:
            self.coordinator.mark_completed(issue_id)


@dataclass
class FakeAbortCallback:
    """Fake abort callback that cancels tasks.

    Observable state:
    - abort_calls: List of abort call kwargs
    - aborted_count: Total tasks aborted across all calls
    """

    abort_calls: list[dict[str, Any]] = field(default_factory=list)
    aborted_count: int = 0
    coordinator: Any = None  # Optional coordinator reference for task cleanup

    async def __call__(self, *, is_interrupt: bool = False) -> AbortResult:
        """Cancel all active tasks and return AbortResult."""
        self.abort_calls.append({"is_interrupt": is_interrupt})

        if self.coordinator is None:
            return AbortResult(aborted_count=0)

        tasks = list(self.coordinator.active_tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.aborted_count += len(tasks)
        return AbortResult(aborted_count=len(tasks))
