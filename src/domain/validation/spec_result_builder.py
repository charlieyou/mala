"""Result building stage for mala validation.

This module provides SpecResultBuilder which handles the post-command-execution
stages of validation:
- Coverage checks (threshold or no-decrease mode)
- E2E execution (global only)
- ValidationResult assembly

The builder is called after commands have been executed and produces the final
ValidationResult.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .coverage import (
    CoverageResult,
    CoverageStatus,
    check_coverage_from_config,
    parse_and_check_coverage,
)
from .e2e import E2EConfig as E2ERunnerConfig
from .e2e import E2ERunner, E2EStatus
from .result import ValidationResult

from pathlib import Path

if TYPE_CHECKING:
    from collections.abc import Mapping

    from src.core.protocols.events import MalaEventSink
    from src.core.protocols.infra import CommandRunnerPort, EnvConfigPort

    from .config import YamlCoverageConfig
    from .e2e import E2EResult
    from .result import ValidationStepResult
    from .spec import (
        CoverageConfig,
        ValidationArtifacts,
        ValidationContext,
        ValidationSpec,
    )


@dataclass
class ResultBuilderInput:
    """Input for the result builder stage.

    Attributes:
        spec: The validation spec being executed.
        context: The validation context.
        steps: Steps from command execution.
        artifacts: Validation artifacts to update.
        cwd: Working directory where coverage.xml should be.
        log_dir: Directory for logs.
        env: Environment variables for E2E.
        baseline_percent: Baseline coverage for "no decrease" mode.
        env_config: Environment configuration for paths.
        command_runner: Command runner for executing commands.
        yaml_coverage_config: Coverage configuration from mala.yaml, or None to
            use spec-based coverage checking (legacy mode).
    """

    spec: ValidationSpec
    context: ValidationContext
    steps: list[ValidationStepResult]
    artifacts: ValidationArtifacts
    cwd: Path
    log_dir: Path
    env: Mapping[str, str]
    baseline_percent: float | None
    env_config: EnvConfigPort
    command_runner: CommandRunnerPort
    yaml_coverage_config: YamlCoverageConfig | None = None
    event_sink: MalaEventSink | None = None


class SpecResultBuilder:
    """Builds ValidationResult from command execution output.

    This builder handles the post-processing stages after commands have been
    executed:
    1. Coverage check (if enabled)
    2. E2E execution (if enabled and scope is global)
    3. Final ValidationResult assembly

    The builder owns the coverage and E2E checking logic, keeping the runner
    focused on workspace setup and command execution.
    """

    def build(self, input: ResultBuilderInput) -> ValidationResult:
        """Build the final ValidationResult.

        Args:
            input: The builder input containing steps, artifacts, and config.

        Returns:
            Complete ValidationResult with coverage/E2E results.
        """
        # Step 1: Check coverage
        cov = self._check_coverage_if_enabled(
            spec=input.spec,
            cwd=input.cwd,
            log_dir=input.log_dir,
            artifacts=input.artifacts,
            baseline_percent=input.baseline_percent,
            env=input.env,
            yaml_coverage_config=input.yaml_coverage_config,
            command_runner=input.command_runner,
        )
        if cov is not None and not cov.passed:
            reason = cov.failure_reason or "Coverage check failed"
            return self._build_failure_result(
                steps=input.steps,
                reason=reason,
                artifacts=input.artifacts,
                coverage_result=cov,
            )

        # Step 2: Run E2E
        e2e = self._run_e2e_if_enabled(
            spec=input.spec,
            env=input.env,
            cwd=input.cwd,
            log_dir=input.log_dir,
            artifacts=input.artifacts,
            env_config=input.env_config,
            command_runner=input.command_runner,
            event_sink=input.event_sink,
        )
        if e2e is not None and not e2e.passed and e2e.status != E2EStatus.SKIPPED:
            reason = e2e.failure_reason or "E2E failed"
            return self._build_failure_result(
                steps=input.steps,
                reason=reason,
                artifacts=input.artifacts,
                coverage_result=cov,
                e2e_result=e2e,
            )

        # Step 3: Success
        return ValidationResult(
            passed=True,
            steps=input.steps,
            artifacts=input.artifacts,
            coverage_result=cov,
            e2e_result=e2e,
        )

    def _check_coverage_if_enabled(
        self,
        spec: ValidationSpec,
        cwd: Path,
        log_dir: Path,
        artifacts: ValidationArtifacts,
        baseline_percent: float | None,
        env: Mapping[str, str],
        command_runner: CommandRunnerPort,
        yaml_coverage_config: YamlCoverageConfig | None = None,
    ) -> CoverageResult | None:
        """Run coverage check if enabled.

        Args:
            spec: Validation spec with coverage config.
            cwd: Working directory where coverage.xml should be.
            log_dir: Directory for logs.
            artifacts: Artifacts to update with coverage report path.
            baseline_percent: Baseline coverage for "no decrease" mode.
            env: Environment variables for command execution.
            command_runner: Command runner for executing coverage command.
            yaml_coverage_config: Coverage configuration from mala.yaml, or None
                to use spec-based coverage checking (legacy mode).

        Returns:
            CoverageResult if coverage is enabled, None otherwise.
        """
        if not spec.coverage.enabled:
            return None

        # If yaml_coverage_config is provided, use config-driven coverage checking
        if yaml_coverage_config is not None:
            coverage_command_result = self._run_coverage_command_if_configured(
                yaml_coverage_config, cwd, env, command_runner
            )
            if coverage_command_result is not None:
                return coverage_command_result
            coverage_result = check_coverage_from_config(yaml_coverage_config, cwd)
            if coverage_result is not None and coverage_result.report_path:
                artifacts.coverage_report = coverage_result.report_path
            return coverage_result

        # Legacy mode: use spec.coverage
        coverage_result = self._check_coverage(
            spec.coverage, cwd, log_dir, baseline_percent
        )
        if coverage_result.report_path:
            artifacts.coverage_report = coverage_result.report_path

        return coverage_result

    def _run_coverage_command_if_configured(
        self,
        coverage_config: YamlCoverageConfig,
        cwd: Path,
        env: Mapping[str, str],
        command_runner: CommandRunnerPort,
    ) -> CoverageResult | None:
        """Run coverage command if configured.

        Args:
            coverage_config: Coverage configuration from mala.yaml.
            cwd: Working directory for the command.
            env: Environment variables for the command.
            command_runner: Command runner for executing the command.

        Returns:
            CoverageResult on failure, None if command not configured or succeeded.
        """
        if coverage_config.command is None:
            return None

        timeout_seconds = coverage_config.timeout or 120
        result = command_runner.run(
            coverage_config.command,
            env=env,
            shell=True,
            cwd=cwd,
            timeout=timeout_seconds,
        )

        if result.ok:
            return None

        details: list[str] = []
        if result.timed_out:
            details.append("coverage command timed out")
        else:
            details.append(f"coverage command exited {result.returncode}")

        tail = result.stderr_tail() or result.stdout_tail()
        if tail:
            details.append(tail)

        report_path = Path(coverage_config.file)
        if not report_path.is_absolute():
            report_path = cwd / report_path

        return CoverageResult(
            percent=None,
            passed=False,
            status=CoverageStatus.ERROR,
            report_path=report_path,
            failure_reason="Coverage command failed: " + "; ".join(details),
        )

    def _check_coverage(
        self,
        config: CoverageConfig,
        cwd: Path,
        log_dir: Path,
        baseline_percent: float | None = None,
    ) -> CoverageResult:
        """Check coverage against threshold.

        Args:
            config: Coverage configuration.
            cwd: Working directory where coverage.xml should be.
            log_dir: Directory for logs.
            baseline_percent: Baseline coverage percentage for "no decrease" mode.
                Used when config.min_percent is None.

        Returns:
            CoverageResult with pass/fail status.
        """
        # Look for coverage.xml in cwd
        # Resolve relative paths against cwd to ensure correct file lookup
        if config.report_path is not None:
            report_path = config.report_path
            if not report_path.is_absolute():
                report_path = cwd / report_path
        else:
            report_path = cwd / "coverage.xml"

        if not report_path.exists():
            # Coverage report not found - this is only an error if coverage was expected
            return CoverageResult(
                percent=None,
                passed=False,
                status=CoverageStatus.ERROR,
                report_path=report_path,
                failure_reason=f"Coverage report not found: {report_path}",
            )

        # Determine threshold: use config.min_percent if set, else baseline_percent
        threshold = (
            config.min_percent if config.min_percent is not None else baseline_percent
        )

        return parse_and_check_coverage(report_path, threshold)

    def _run_e2e_if_enabled(
        self,
        spec: ValidationSpec,
        env: Mapping[str, str],
        cwd: Path,
        log_dir: Path,
        artifacts: ValidationArtifacts,
        env_config: EnvConfigPort,
        command_runner: CommandRunnerPort,
        event_sink: MalaEventSink | None = None,
    ) -> E2EResult | None:
        """Run E2E validation if enabled (only for global scope).

        Args:
            spec: Validation spec with E2E config.
            env: Environment variables.
            cwd: Working directory.
            log_dir: Directory for logs.
            artifacts: Artifacts to update with fixture path.
            env_config: Environment configuration for paths.
            command_runner: Command runner for executing commands.
            event_sink: Event sink for emitting validation step events.

        Returns:
            E2EResult if E2E is enabled and scope is global, None otherwise.
        """
        from .spec import ValidationScope

        if not spec.e2e.enabled or spec.scope != ValidationScope.GLOBAL:
            return None

        # Check prereqs before emitting "running" to match cached command behavior
        # (skipped steps only emit "skipped", not "running" then "skipped")
        runner_config = E2ERunnerConfig(
            enabled=spec.e2e.enabled,
            skip_if_no_keys=True,
            keep_fixture=True,
            timeout_seconds=1200.0,
        )
        runner = E2ERunner(env_config, command_runner, runner_config)
        prereq = runner.check_prereqs(env)

        if not prereq.ok:
            reason = prereq.failure_reason() or "prerequisites not met"
            if event_sink is not None:
                event_sink.on_validation_step_skipped("e2e", reason)
            from .e2e import E2EResult

            return E2EResult(
                passed=False,
                status=E2EStatus.SKIPPED if prereq.can_skip else E2EStatus.FAILED,
                failure_reason=reason,
                returncode=1,
            )

        if event_sink is not None:
            event_sink.on_validation_step_running("e2e")

        e2e_result = runner.run(env=dict(env), cwd=cwd)
        if e2e_result.fixture_path:
            artifacts.e2e_fixture_path = e2e_result.fixture_path

        if event_sink is not None:
            if e2e_result.passed:
                event_sink.on_validation_step_passed("e2e", e2e_result.duration_seconds)
            else:
                event_sink.on_validation_step_failed("e2e", e2e_result.returncode or 1)

        return e2e_result

    def _build_failure_result(
        self,
        steps: list[ValidationStepResult],
        reason: str,
        artifacts: ValidationArtifacts,
        coverage_result: CoverageResult | None = None,
        e2e_result: E2EResult | None = None,
    ) -> ValidationResult:
        """Build a failed ValidationResult.

        Args:
            steps: Command execution steps.
            reason: Failure reason.
            artifacts: Validation artifacts.
            coverage_result: Optional coverage result.
            e2e_result: Optional E2E result.

        Returns:
            ValidationResult with passed=False.
        """
        return ValidationResult(
            passed=False,
            steps=steps,
            failure_reasons=[reason],
            artifacts=artifacts,
            coverage_result=coverage_result,
            e2e_result=e2e_result,
        )
