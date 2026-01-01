"""GateRunner: Quality gate checking pipeline stage.

Extracted from MalaOrchestrator to separate gate/fixer policy from orchestration.
This module handles:
- Per-issue quality gate checks with retry state management
- No-progress detection for retry termination

The GateRunner receives explicit inputs and returns explicit outputs,
making it testable without SDK or subprocess dependencies.

Design principles:
- Pure functions where possible (gate checking is stateless)
- Protocol-based dependencies for testability
- Explicit input/output types for clarity
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from src.domain.quality_gate import GateResult
from src.domain.validation.spec import (
    ValidationScope,
    build_validation_spec,
)

if TYPE_CHECKING:
    from pathlib import Path

    from src.domain.lifecycle import RetryState
    from src.protocols import GateChecker, GateResultProtocol, ValidationSpecProtocol
    from src.domain.validation.spec import ValidationSpec


@dataclass
class GateRunnerConfig:
    """Configuration for GateRunner behavior.

    Attributes:
        max_gate_retries: Maximum number of gate retry attempts.
        disable_validations: Set of validation names to disable.
        coverage_threshold: Optional coverage threshold override.
    """

    max_gate_retries: int = 3
    disable_validations: set[str] | None = None
    coverage_threshold: float | None = None


@dataclass
class PerIssueGateInput:
    """Input for per-issue quality gate check.

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
class PerIssueGateOutput:
    """Output from per-issue quality gate check.

    Attributes:
        gate_result: The GateResult from the quality gate.
        new_log_offset: Updated log offset for next retry attempt.
    """

    gate_result: GateResultProtocol
    new_log_offset: int


@dataclass
class GateRunner:
    """Quality gate runner for per-issue validation.

    This class encapsulates the gate checking logic that was previously
    inline in MalaOrchestrator._run_quality_gate_sync. It receives a
    GateChecker (protocol) for actual validation execution.

    The GateRunner is responsible for:
    - Building/using ValidationSpec for scope-aware checks
    - Running gate checks via the injected GateChecker
    - Detecting no-progress conditions for retry termination
    - Returning updated log offsets for retry scoping

    Usage:
        runner = GateRunner(
            gate_checker=quality_gate,
            repo_path=repo_path,
            config=GateRunnerConfig(max_gate_retries=3),
        )
        output = runner.run_per_issue_gate(input)

    Attributes:
        gate_checker: GateChecker implementation for running checks.
        repo_path: Path to the repository.
        config: Configuration for gate behavior.
        per_issue_spec: Cached per-issue ValidationSpec (built lazily).
    """

    gate_checker: GateChecker
    repo_path: Path
    config: GateRunnerConfig = field(default_factory=GateRunnerConfig)
    per_issue_spec: ValidationSpec | None = field(default=None, init=False)

    def _get_or_build_spec(
        self, provided_spec: ValidationSpec | None
    ) -> ValidationSpec:
        """Get provided spec or build/cache a per-issue spec.

        Args:
            provided_spec: Spec provided in input, or None.

        Returns:
            ValidationSpec to use for the gate check.
        """
        if provided_spec is not None:
            return provided_spec

        # Build and cache per-issue spec if not already cached
        if self.per_issue_spec is None:
            self.per_issue_spec = build_validation_spec(
                scope=ValidationScope.PER_ISSUE,
                disable_validations=self.config.disable_validations,
                coverage_threshold=self.config.coverage_threshold,
                repo_path=self.repo_path,
            )
        return self.per_issue_spec

    def run_per_issue_gate(self, input: PerIssueGateInput) -> PerIssueGateOutput:
        """Run quality gate check for a single issue.

        This is a synchronous method that performs blocking I/O.
        The orchestrator should call this via asyncio.to_thread().

        Args:
            input: PerIssueGateInput with issue_id, log_path, retry_state.

        Returns:
            PerIssueGateOutput with gate_result and new_log_offset.
        """
        spec = self._get_or_build_spec(input.spec)

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

        return PerIssueGateOutput(
            gate_result=gate_result,
            new_log_offset=new_offset,
        )

    def get_cached_spec(self) -> ValidationSpec | None:
        """Get the cached per-issue spec, if any.

        This allows the orchestrator to access the spec for other purposes
        (e.g., evidence parsing) without rebuilding it.

        Returns:
            The cached ValidationSpec, or None if not yet built.
        """
        return self.per_issue_spec

    def set_cached_spec(self, spec: ValidationSpec) -> None:
        """Set the cached per-issue spec.

        Allows the orchestrator to pre-populate the cache with a spec
        built at run start.

        Args:
            spec: ValidationSpec to cache.
        """
        self.per_issue_spec = spec
