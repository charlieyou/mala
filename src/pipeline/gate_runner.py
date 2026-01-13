"""GateRunner: Quality gate checking pipeline stage.

Extracted from MalaOrchestrator to separate gate/fixer policy from orchestration.
This module handles:
- Per-session quality gate checks with retry state management
- No-progress detection for retry termination
- Async gate running for non-blocking orchestration

The GateRunner receives explicit inputs and returns explicit outputs,
making it testable without SDK or subprocess dependencies.

Design principles:
- Pure functions where possible (gate checking is stateless)
- Protocol-based dependencies for testability
- Explicit input/output types for clarity
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from src.domain.evidence_check import GateResult
from src.domain.validation.spec import (
    ValidationScope,
    build_validation_spec,
)
from src.infra.sigint_guard import InterruptGuard

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pathlib import Path

    from src.domain.lifecycle import RetryState
    from src.core.protocols.validation import (
        GateChecker,
        GateResultProtocol,
        ValidationSpecProtocol,
    )
    from src.domain.validation.config import ValidationConfig
    from src.domain.validation.spec import ValidationSpec


@dataclass
class GateRunnerConfig:
    """Configuration for GateRunner behavior.

    Attributes:
        max_gate_retries: Maximum number of gate retry attempts.
        disable_validations: Set of validation names to disable.
    """

    max_gate_retries: int = 3
    disable_validations: set[str] | None = None
    validation_config: ValidationConfig | None = None
    validation_config_missing: bool = False


@dataclass
class PerSessionGateInput:
    """Input for per-session quality gate check.

    Bundles all the data needed to run a single gate check.

    Attributes:
        issue_id: The issue being checked.
        log_path: Path to the JSONL session log file.
        retry_state: Current retry state (gate attempt, log offset, etc.).
        spec: ValidationSpec defining what to check. If None, will be built.
    """

    issue_id: str
    log_path: Path
    retry_state: RetryState
    spec: ValidationSpec | None = None


@dataclass
class PerSessionGateOutput:
    """Output from per-session quality gate check.

    Attributes:
        gate_result: The GateResult from the quality gate.
        new_log_offset: Updated log offset for next retry attempt.
        interrupted: Whether the gate check was interrupted by SIGINT.
    """

    gate_result: GateResultProtocol
    new_log_offset: int
    interrupted: bool = False


@dataclass
class GateRunner:
    """Quality gate runner for per-session validation.

    This class encapsulates the gate checking logic that was previously
    inline in MalaOrchestrator._run_evidence_check_sync. It receives a
    GateChecker (protocol) for actual validation execution.

    The GateRunner is responsible for:
    - Building/using ValidationSpec for scope-aware checks
    - Running gate checks via the injected GateChecker
    - Detecting no-progress conditions for retry termination
    - Returning updated log offsets for retry scoping

    Usage:
        runner = GateRunner(
            gate_checker=evidence_check,
            repo_path=repo_path,
            config=GateRunnerConfig(max_gate_retries=3),
        )
        output = runner.run_per_session_gate(input)

    Attributes:
        gate_checker: GateChecker implementation for running checks.
        repo_path: Path to the repository.
        config: Configuration for gate behavior.
        per_session_spec: Cached per-session ValidationSpec (built lazily).
    """

    gate_checker: GateChecker
    repo_path: Path
    config: GateRunnerConfig = field(default_factory=GateRunnerConfig)
    per_session_spec: ValidationSpec | None = field(default=None, init=False)

    def _get_or_build_spec(
        self, provided_spec: ValidationSpec | None
    ) -> ValidationSpec:
        """Get provided spec or build/cache a per-session spec.

        Args:
            provided_spec: Spec provided in input, or None.

        Returns:
            ValidationSpec to use for the gate check.
        """
        if provided_spec is not None:
            return provided_spec

        # Build and cache per-session spec if not already cached
        if self.per_session_spec is None:
            self.per_session_spec = build_validation_spec(
                self.repo_path,
                scope=ValidationScope.PER_SESSION,
                disable_validations=self.config.disable_validations,
                validation_config=self.config.validation_config,
                config_missing=self.config.validation_config_missing,
            )
        return self.per_session_spec

    def run_per_session_gate(
        self,
        input: PerSessionGateInput,
        interrupt_event: asyncio.Event | None = None,
    ) -> PerSessionGateOutput:
        """Run quality gate check for a single issue.

        This is a synchronous method that performs blocking I/O.
        The orchestrator should call this via asyncio.to_thread().

        Args:
            input: PerSessionGateInput with issue_id, log_path, retry_state.
            interrupt_event: Optional event to check for SIGINT interrupts.

        Returns:
            PerSessionGateOutput with gate_result and new_log_offset.
            If interrupted, returns with interrupted=True.
        """
        guard = InterruptGuard(interrupt_event)

        # Check for interrupt before gate execution
        if guard.is_interrupted():
            # Return an empty gate result with interrupted=True
            return PerSessionGateOutput(
                gate_result=GateResult(
                    passed=False,
                    failure_reasons=["Gate check interrupted"],
                    commit_hash=None,
                ),
                new_log_offset=input.retry_state.log_offset,
                interrupted=True,
            )

        spec = self._get_or_build_spec(input.spec)
        logger.debug(
            "Gate check: issue_id=%s attempt=%d",
            input.issue_id,
            input.retry_state.gate_attempt,
        )

        # Run gate check via injected checker
        gate_result = self.gate_checker.check_with_resolution(
            issue_id=input.issue_id,
            log_path=input.log_path,
            baseline_timestamp=input.retry_state.baseline_timestamp,
            log_offset=input.retry_state.log_offset,
            spec=cast("ValidationSpecProtocol | None", spec),
        )

        # Calculate new offset for next attempt
        new_offset = self.gate_checker.get_log_end_offset(
            input.log_path, start_offset=input.retry_state.log_offset
        )

        # Check for no-progress condition on retries
        if input.retry_state.gate_attempt > 1 and not gate_result.passed:
            no_progress = self.gate_checker.check_no_progress(
                input.log_path,
                input.retry_state.log_offset,
                input.retry_state.previous_commit_hash,
                gate_result.commit_hash,
                spec=cast("ValidationSpecProtocol | None", spec),
            )
            if no_progress:
                logger.warning("No progress detected: issue_id=%s", input.issue_id)
                # Add no-progress to failure reasons
                updated_reasons = list(gate_result.failure_reasons)
                updated_reasons.append(
                    "No progress: commit unchanged and no new validation evidence"
                )
                gate_result = GateResult(
                    passed=False,
                    failure_reasons=updated_reasons,
                    commit_hash=gate_result.commit_hash,
                    validation_evidence=gate_result.validation_evidence,
                    no_progress=True,
                    resolution=gate_result.resolution,
                )

        return PerSessionGateOutput(
            gate_result=gate_result,
            new_log_offset=new_offset,
        )

    def get_cached_spec(self) -> ValidationSpec | None:
        """Get the cached per-session spec, if any.

        This allows the orchestrator to access the spec for other purposes
        (e.g., evidence parsing) without rebuilding it.

        Returns:
            The cached ValidationSpec, or None if not yet built.
        """
        return self.per_session_spec

    def set_cached_spec(self, spec: ValidationSpec) -> None:
        """Set the cached per-session spec.

        Allows the orchestrator to pre-populate the cache with a spec
        built at run start.

        Args:
            spec: ValidationSpec to cache.
        """
        self.per_session_spec = spec
        logger.debug("Validation spec cached: issue_id=*")


@dataclass
class AsyncGateRunner:
    """Async wrapper for GateRunner that runs gate checks in a thread pool.

    This class implements the GateAsyncRunner protocol used by SessionCallbackFactory.
    It wraps a GateRunner and delegates to it via asyncio.to_thread for non-blocking
    execution.

    The AsyncGateRunner maintains its own state for:
    - per_session_spec: Cached validation spec (synced with underlying GateRunner)
    - last_gate_results: Most recent gate results per issue

    Usage:
        gate_runner = GateRunner(gate_checker=..., repo_path=..., config=...)
        async_runner = AsyncGateRunner(gate_runner=gate_runner)
        result, offset = await async_runner.run_gate_async(issue_id, log_path, retry_state)
    """

    gate_runner: GateRunner
    per_session_spec: ValidationSpec | None = field(default=None)
    last_gate_results: dict[str, GateResult | GateResultProtocol] = field(
        default_factory=dict
    )

    def _run_gate_sync(
        self,
        issue_id: str,
        log_path: Path,
        retry_state: RetryState,
        interrupt_event: asyncio.Event | None = None,
    ) -> tuple[GateResult | GateResultProtocol, int]:
        """Synchronous gate check (blocking I/O).

        Delegates to GateRunner for actual gate checking logic.

        Args:
            issue_id: The issue being checked.
            log_path: Path to the session log file.
            retry_state: Current retry state for this issue.
            interrupt_event: Optional event to check for SIGINT interrupts.
        """
        # Sync per_session_spec with gate_runner
        if self.per_session_spec is not None:
            self.gate_runner.set_cached_spec(self.per_session_spec)

        gate_input = PerSessionGateInput(
            issue_id=issue_id,
            log_path=log_path,
            retry_state=retry_state,
            spec=self.per_session_spec,
        )
        output = self.gate_runner.run_per_session_gate(gate_input, interrupt_event)

        # Sync cached spec back (gate_runner may have built it)
        if self.per_session_spec is None:
            self.per_session_spec = self.gate_runner.get_cached_spec()

        # Store gate result for later retrieval
        self.last_gate_results[issue_id] = output.gate_result

        return (output.gate_result, output.new_log_offset)

    async def run_gate_async(
        self,
        issue_id: str,
        log_path: Path,
        retry_state: RetryState,
        interrupt_event: asyncio.Event | None = None,
    ) -> tuple[GateResult | GateResultProtocol, int]:
        """Run quality gate check asynchronously (GateAsyncRunner protocol).

        Wraps the blocking gate check to avoid stalling the event loop.
        This allows the orchestrator to service other agents while a gate runs.

        Args:
            issue_id: The issue being checked.
            log_path: Path to the session log file.
            retry_state: Current retry state for this issue.
            interrupt_event: Optional event to check for SIGINT interrupts.

        Returns:
            Tuple of (GateResult, new_log_offset).
        """
        return await asyncio.to_thread(
            self._run_gate_sync, issue_id, log_path, retry_state, interrupt_event
        )

    def get_last_gate_result(
        self, issue_id: str
    ) -> GateResult | GateResultProtocol | None:
        """Get the last gate result for an issue.

        Args:
            issue_id: The issue ID.

        Returns:
            The last gate result, or None if not available.
        """
        return self.last_gate_results.get(issue_id)

    def clear_gate_result(self, issue_id: str) -> None:
        """Clear the stored gate result for an issue.

        Args:
            issue_id: The issue ID.
        """
        self.last_gate_results.pop(issue_id, None)
