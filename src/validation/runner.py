"""Post-commit validation runner for mala.

Runs clean-room validations in a temporary git worktree and (optionally) a
true end-to-end `mala run` smoke test using a fixture repo.

This module provides two APIs:
1. Legacy: ValidationRunner.validate_commit() - uses ValidationConfig
2. New: ValidationRunner.run_spec() - uses ValidationSpec + ValidationContext
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ..tools.env import LOCK_DIR, SCRIPTS_DIR
from .coverage import CoverageResult, CoverageStatus, parse_and_check_coverage
from .e2e import E2EConfig as E2ERunnerConfig
from .e2e import E2ERunner, E2EStatus
from .spec import ValidationArtifacts, ValidationScope
from .worktree import (
    WorktreeConfig,
    WorktreeState,
    create_worktree,
    remove_worktree,
)

if TYPE_CHECKING:
    from .e2e import E2EResult
    from .spec import (
        CoverageConfig,
        E2EConfig,
        ValidationCommand,
        ValidationContext,
        ValidationSpec,
    )
    from .worktree import WorktreeContext


@dataclass
class ValidationConfig:
    """Configuration for post-commit validation."""

    run_slow_tests: bool = True
    run_e2e: bool = True
    coverage: bool = True
    coverage_min: int = 85
    coverage_source: str = "."
    use_test_mutex: bool = (
        False  # Disabled to avoid deadlocks; isolated caches prevent conflicts
    )
    step_timeout_seconds: float | None = None


@dataclass
class ValidationStepResult:
    """Result of a single validation step."""

    name: str
    command: list[str]
    ok: bool
    returncode: int
    stdout_tail: str = ""
    stderr_tail: str = ""
    duration_seconds: float = 0.0


@dataclass
class ValidationResult:
    """Result of a validation run."""

    passed: bool
    steps: list[ValidationStepResult] = field(default_factory=list)
    failure_reasons: list[str] = field(default_factory=list)
    retriable: bool = True
    artifacts: ValidationArtifacts | None = None
    coverage_result: CoverageResult | None = None

    def short_summary(self) -> str:
        """One-line summary for logs/prompts."""
        if self.passed:
            return "passed"
        if not self.failure_reasons:
            return "failed"
        return "; ".join(self.failure_reasons)


class ValidationRunner:
    """Runs post-commit validation in a clean git worktree."""

    def __init__(self, repo_path: Path, config: ValidationConfig):
        self.repo_path = repo_path.resolve()
        self.config = config

    async def validate_commit(self, commit_sha: str, label: str) -> ValidationResult:
        """Validate a commit in a clean worktree (async wrapper)."""
        return await asyncio.to_thread(self._validate_commit_sync, commit_sha, label)

    def _validate_commit_sync(self, commit_sha: str, label: str) -> ValidationResult:
        """Run validation in a temporary worktree (sync)."""
        worktree_dir = Path(
            tempfile.mkdtemp(prefix=f"mala-validate-{label}-")
        ).resolve()
        try:
            add_result = subprocess.run(
                ["git", "worktree", "add", "--detach", str(worktree_dir), commit_sha],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
            )
            if add_result.returncode != 0:
                reason = f"Post-validation failed: git worktree add exited {add_result.returncode}"
                if add_result.stderr.strip():
                    reason += f" ({_tail(add_result.stderr.strip())})"
                return ValidationResult(
                    passed=False,
                    failure_reasons=[reason],
                    retriable=False,
                )

            return self._run_validation_steps(
                worktree_dir, label=label, include_e2e=self.config.run_e2e
            )
        finally:
            try:
                subprocess.run(
                    ["git", "worktree", "remove", "--force", str(worktree_dir)],
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True,
                )
            finally:
                shutil.rmtree(worktree_dir, ignore_errors=True)

    def _run_validation_steps(
        self, cwd: Path, label: str, include_e2e: bool
    ) -> ValidationResult:
        env = {
            **os.environ,
            "LOCK_DIR": str(LOCK_DIR),
            "AGENT_ID": f"validator-{label}",
        }

        steps: list[ValidationStepResult] = []

        for name, cmd in self._build_validation_commands():
            step = self._run_command(name, cmd, cwd, env)
            steps.append(step)
            if not step.ok:
                reason = f"{name} failed (exit {step.returncode})"
                details = _format_step_output(step)
                if details:
                    reason = f"{reason}: {details}"
                return ValidationResult(
                    passed=False,
                    steps=steps,
                    failure_reasons=[reason],
                )

        if include_e2e:
            e2e_result = self._run_e2e(label=label, env=env)
            steps.extend(e2e_result.steps)
            if not e2e_result.passed:
                return e2e_result

        return ValidationResult(passed=True, steps=steps)

    def _build_validation_commands(self) -> list[tuple[str, list[str]]]:
        commands: list[tuple[str, list[str]]] = [
            ("uv sync", ["uv", "sync", "--all-extras"]),
            ("ruff format", ["uvx", "ruff", "format", "--check", "."]),
            ("ruff check", ["uvx", "ruff", "check", "."]),
            ("ty check", ["uvx", "ty", "check"]),
        ]

        pytest_cmd = ["uv", "run", "pytest"]
        if self.config.run_slow_tests:
            pytest_cmd += ["-m", "slow or not slow"]
        if self.config.coverage:
            pytest_cmd += [
                "--cov",
                self.config.coverage_source,
                "--cov-branch",
                "--cov-report=term-missing",
                f"--cov-fail-under={self.config.coverage_min}",
            ]
        commands.append(("pytest", pytest_cmd))
        return commands

    def _run_command(
        self, name: str, cmd: list[str], cwd: Path, env: dict[str, str]
    ) -> ValidationStepResult:
        full_cmd = self._wrap_with_mutex(cmd) if self.config.use_test_mutex else cmd
        start = time.monotonic()
        try:
            result = subprocess.run(
                full_cmd,
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                timeout=self.config.step_timeout_seconds,
            )
            duration = time.monotonic() - start
            stdout_tail = _tail(result.stdout)
            stderr_tail = _tail(result.stderr)
            return ValidationStepResult(
                name=name,
                command=cmd,
                ok=result.returncode == 0,
                returncode=result.returncode,
                stdout_tail=stdout_tail,
                stderr_tail=stderr_tail,
                duration_seconds=duration,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.monotonic() - start
            stdout_tail = _decode_timeout_output(exc.stdout)
            stderr_tail = _decode_timeout_output(exc.stderr)
            return ValidationStepResult(
                name=name,
                command=cmd,
                ok=False,
                returncode=124,
                stdout_tail=stdout_tail,
                stderr_tail=stderr_tail,
                duration_seconds=duration,
            )

    def _wrap_with_mutex(self, cmd: list[str]) -> list[str]:
        return [str(SCRIPTS_DIR / "test-mutex.sh"), *cmd]

    def _run_e2e(self, label: str, env: dict[str, str]) -> ValidationResult:
        prereq_failure = _check_e2e_prereqs(env)
        if prereq_failure:
            return ValidationResult(
                passed=False,
                failure_reasons=[prereq_failure],
                retriable=False,
            )

        with tempfile.TemporaryDirectory(prefix=f"mala-e2e-{label}-") as tmp_dir:
            repo_path = Path(tmp_dir).resolve()
            _write_fixture_repo(repo_path)
            init_result = _init_fixture_repo(repo_path)
            if init_result:
                return ValidationResult(
                    passed=False,
                    failure_reasons=[init_result],
                    retriable=False,
                )

            issue_id = _get_ready_issue_id(repo_path)
            if issue_id:
                _annotate_issue(repo_path, issue_id)

            cmd = [
                "mala",
                "run",
                str(repo_path),
                "--max-agents",
                "1",
                "--max-issues",
                "1",
                "--timeout",
                "30",
            ]
            # Keep coverage/slow-tests/codex-review defaults in the nested run.
            step = self._run_command("mala e2e", cmd, self.repo_path, env)
            if step.ok:
                return ValidationResult(passed=True, steps=[step])

            reason = f"mala e2e failed (exit {step.returncode})"
            details = _format_step_output(step)
            if details:
                reason = f"{reason}: {details}"
            return ValidationResult(
                passed=False,
                steps=[step],
                failure_reasons=[reason],
                retriable=False,
            )

    async def run_spec(
        self,
        spec: ValidationSpec,
        context: ValidationContext,
        log_dir: Path | None = None,
    ) -> ValidationResult:
        """Run validation according to a ValidationSpec.

        This is the new API that uses ValidationSpec + ValidationContext instead
        of the legacy ValidationConfig. It supports:
        - Scope-aware validation (per-issue vs run-level)
        - Per-command mutex settings
        - Integrated worktree, coverage, and E2E handling
        - Artifact tracking

        Args:
            spec: What validations to run.
            context: Immutable context for the validation run.
            log_dir: Directory for logs/artifacts. Uses temp dir if None.

        Returns:
            ValidationResult with steps, artifacts, and coverage info.
        """
        return await asyncio.to_thread(self._run_spec_sync, spec, context, log_dir)

    def _run_spec_sync(
        self,
        spec: ValidationSpec,
        context: ValidationContext,
        log_dir: Path | None = None,
    ) -> ValidationResult:
        """Synchronous implementation of run_spec."""
        # Set up log directory
        if log_dir is None:
            log_dir = Path(tempfile.mkdtemp(prefix="mala-validation-logs-"))

        log_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique run ID
        run_id = f"run-{uuid.uuid4().hex[:8]}"
        issue_id = context.issue_id or "run-level"

        # Initialize artifacts
        artifacts = ValidationArtifacts(log_dir=log_dir)

        # Set up worktree if we have a commit to validate
        worktree_ctx: WorktreeContext | None = None
        validation_cwd: Path

        if context.commit_hash:
            worktree_config = WorktreeConfig(
                base_dir=log_dir / "worktrees",
                keep_on_failure=True,  # Keep for debugging
            )
            worktree_ctx = create_worktree(
                repo_path=context.repo_path,
                commit_sha=context.commit_hash,
                config=worktree_config,
                run_id=run_id,
                issue_id=issue_id,
                attempt=1,
            )

            if worktree_ctx.state == WorktreeState.FAILED:
                return ValidationResult(
                    passed=False,
                    failure_reasons=[f"Worktree creation failed: {worktree_ctx.error}"],
                    retriable=False,
                    artifacts=artifacts,
                )

            validation_cwd = worktree_ctx.path
            artifacts.worktree_path = worktree_ctx.path
        else:
            # No commit specified, validate in place
            validation_cwd = context.repo_path

        # Execute commands and capture result
        result = self._execute_spec_commands(
            spec=spec,
            context=context,
            cwd=validation_cwd,
            artifacts=artifacts,
            log_dir=log_dir,
            run_id=run_id,
        )

        # Clean up worktree with correct pass/fail status
        if worktree_ctx is not None:
            worktree_ctx = remove_worktree(
                worktree_ctx, validation_passed=result.passed
            )
            if worktree_ctx.state == WorktreeState.KEPT:
                artifacts.worktree_state = "kept"
            elif worktree_ctx.state == WorktreeState.REMOVED:
                artifacts.worktree_state = "removed"

        return result

    def _execute_spec_commands(
        self,
        spec: ValidationSpec,
        context: ValidationContext,
        cwd: Path,
        artifacts: ValidationArtifacts,
        log_dir: Path,
        run_id: str,
    ) -> ValidationResult:
        """Execute all commands in the spec.

        Args:
            spec: Validation spec with commands.
            context: Validation context.
            cwd: Working directory for commands.
            artifacts: Artifacts to update.
            log_dir: Directory for logs.
            run_id: Unique run identifier.

        Returns:
            ValidationResult with all step results.
        """
        # Build environment
        env = self._build_spec_env(context, run_id)

        steps: list[ValidationStepResult] = []
        coverage_result: CoverageResult | None = None

        # Execute each command in order
        for cmd in spec.commands:
            step = self._run_spec_command(cmd, cwd, env, log_dir)
            steps.append(step)

            # Log output to files
            self._write_step_logs(step, log_dir)

            if not step.ok and not cmd.allow_fail:
                reason = f"{cmd.name} failed (exit {step.returncode})"
                details = _format_step_output(step)
                if details:
                    reason = f"{reason}: {details}"
                return ValidationResult(
                    passed=False,
                    steps=steps,
                    failure_reasons=[reason],
                    artifacts=artifacts,
                )

        # Handle coverage if enabled
        if spec.coverage.enabled:
            coverage_result = self._check_coverage(spec.coverage, cwd, log_dir)
            if coverage_result.report_path:
                artifacts.coverage_report = coverage_result.report_path

            # Fail on coverage error or threshold not met
            if not coverage_result.passed:
                return ValidationResult(
                    passed=False,
                    steps=steps,
                    failure_reasons=[
                        coverage_result.failure_reason or "Coverage check failed"
                    ],
                    artifacts=artifacts,
                    coverage_result=coverage_result,
                )

        # Handle E2E if enabled (only for run-level scope)
        if spec.e2e.enabled and spec.scope == ValidationScope.RUN_LEVEL:
            e2e_result = self._run_spec_e2e(spec.e2e, env, cwd, log_dir)
            if e2e_result.fixture_path:
                artifacts.e2e_fixture_path = e2e_result.fixture_path

            if not e2e_result.passed and e2e_result.status != E2EStatus.SKIPPED:
                return ValidationResult(
                    passed=False,
                    steps=steps,
                    failure_reasons=[e2e_result.failure_reason or "E2E failed"],
                    artifacts=artifacts,
                    coverage_result=coverage_result,
                )

        return ValidationResult(
            passed=True,
            steps=steps,
            artifacts=artifacts,
            coverage_result=coverage_result,
        )

    def _build_spec_env(
        self,
        context: ValidationContext,
        run_id: str,
    ) -> dict[str, str]:
        """Build environment for spec-based validation."""
        return {
            **os.environ,
            "LOCK_DIR": str(LOCK_DIR),
            "AGENT_ID": f"validator-{context.issue_id or run_id}",
        }

    def _run_spec_command(
        self,
        cmd: ValidationCommand,
        cwd: Path,
        env: dict[str, str],
        log_dir: Path,
    ) -> ValidationStepResult:
        """Run a single ValidationCommand.

        Args:
            cmd: The command to run.
            cwd: Working directory.
            env: Environment variables.
            log_dir: Directory for logs.

        Returns:
            ValidationStepResult with execution details.
        """
        # Wrap with mutex if requested
        full_cmd = (
            self._wrap_with_mutex(cmd.command) if cmd.use_test_mutex else cmd.command
        )

        start = time.monotonic()
        try:
            result = subprocess.run(
                full_cmd,
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                timeout=self.config.step_timeout_seconds,
            )
            duration = time.monotonic() - start
            return ValidationStepResult(
                name=cmd.name,
                command=cmd.command,
                ok=result.returncode == 0,
                returncode=result.returncode,
                stdout_tail=_tail(result.stdout),
                stderr_tail=_tail(result.stderr),
                duration_seconds=duration,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.monotonic() - start
            return ValidationStepResult(
                name=cmd.name,
                command=cmd.command,
                ok=False,
                returncode=124,
                stdout_tail=_decode_timeout_output(exc.stdout),
                stderr_tail=_decode_timeout_output(exc.stderr),
                duration_seconds=duration,
            )

    def _write_step_logs(self, step: ValidationStepResult, log_dir: Path) -> None:
        """Write step stdout/stderr to log files.

        Args:
            step: The step result to log.
            log_dir: Directory to write logs to.
        """
        # Sanitize step name for filename
        safe_name = step.name.replace(" ", "_").replace("/", "_")

        if step.stdout_tail:
            stdout_path = log_dir / f"{safe_name}.stdout.log"
            stdout_path.write_text(step.stdout_tail)

        if step.stderr_tail:
            stderr_path = log_dir / f"{safe_name}.stderr.log"
            stderr_path.write_text(step.stderr_tail)

    def _check_coverage(
        self,
        config: CoverageConfig,
        cwd: Path,
        log_dir: Path,
    ) -> CoverageResult:
        """Check coverage against threshold.

        Args:
            config: Coverage configuration.
            cwd: Working directory where coverage.xml should be.
            log_dir: Directory for logs.

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

        return parse_and_check_coverage(report_path, config.min_percent)

    def _run_spec_e2e(
        self,
        config: E2EConfig,
        env: dict[str, str],
        cwd: Path,
        log_dir: Path,
    ) -> E2EResult:
        """Run E2E validation using the E2ERunner.

        Args:
            config: E2E configuration from spec.
            env: Environment variables.
            cwd: Working directory.
            log_dir: Directory for logs.

        Returns:
            E2EResult with pass/fail status.
        """
        runner_config = E2ERunnerConfig(
            enabled=config.enabled,
            skip_if_no_keys=True,  # Don't fail if API key missing
            keep_fixture=True,  # Keep for debugging
        )
        runner = E2ERunner(runner_config)
        return runner.run(env=env, cwd=cwd)


def _tail(text: str, max_chars: int = 800, max_lines: int = 20) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    clipped = "\n".join(lines)
    if len(clipped) > max_chars:
        return clipped[-max_chars:]
    return clipped


def _decode_timeout_output(data: bytes | str | None) -> str:
    """Decode TimeoutExpired stdout/stderr which may be bytes, str, or None."""
    if data is None:
        return ""
    if isinstance(data, str):
        return _tail(data)
    return _tail(data.decode())


def _format_step_output(step: ValidationStepResult) -> str:
    parts = []
    if step.stderr_tail:
        parts.append(f"stderr: {_tail(step.stderr_tail, max_chars=300, max_lines=6)}")
    if step.stdout_tail and not step.stderr_tail:
        parts.append(f"stdout: {_tail(step.stdout_tail, max_chars=300, max_lines=6)}")
    return " | ".join(parts)


def _check_e2e_prereqs(env: dict[str, str]) -> str | None:
    if not shutil.which("mala"):
        return "E2E prereq missing: mala CLI not found in PATH"
    if not shutil.which("bd"):
        return "E2E prereq missing: bd CLI not found in PATH"
    if not env.get("MORPH_API_KEY"):
        return "E2E prereq missing: MORPH_API_KEY not set"
    return None


def _write_fixture_repo(repo_path: Path) -> None:
    (repo_path / "tests").mkdir(parents=True, exist_ok=True)
    (repo_path / "app.py").write_text(
        "def add(a: int, b: int) -> int:\n    return a - b\n"
    )
    (repo_path / "tests" / "test_app.py").write_text(
        "from app import add\n\n\ndef test_add():\n    assert add(2, 2) == 4\n"
    )
    (repo_path / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                'name = "mala-e2e-fixture"',
                'version = "0.0.0"',
                'description = "Fixture repo for mala e2e validation"',
                'requires-python = ">=3.11"',
                "dependencies = []",
                "",
                "[project.optional-dependencies]",
                'dev = ["pytest>=8.0.0", "pytest-cov>=4.1.0"]',
            ]
        )
        + "\n"
    )


def _init_fixture_repo(repo_path: Path) -> str | None:
    for cmd in (
        ["git", "init"],
        ["git", "config", "user.email", "mala-e2e@example.com"],
        ["git", "config", "user.name", "Mala E2E"],
        ["git", "add", "."],
        ["git", "commit", "-m", "initial"],
        ["bd", "init"],
        ["bd", "create", "Fix failing add() test", "-p", "1"],
    ):
        result = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            reason = (
                f"E2E fixture setup failed: {' '.join(cmd)} (exit {result.returncode})"
            )
            if stderr:
                reason = f"{reason}: {_tail(stderr)}"
            return reason
    return None


def _get_ready_issue_id(repo_path: Path) -> str | None:
    result = subprocess.run(
        ["bd", "ready", "--json"],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    try:
        issues = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    for issue in issues:
        issue_id = issue.get("id")
        if isinstance(issue_id, str):
            return issue_id
    return None


def _annotate_issue(repo_path: Path, issue_id: str) -> None:
    notes = "\n".join(
        [
            "Context:",
            "- Tests are failing for add() in app.py",
            "",
            "Acceptance Criteria:",
            "- Fix add() so tests pass",
            "- Run full validation suite",
            "",
            "Test Plan:",
            "- uv run pytest",
        ]
    )
    subprocess.run(
        ["bd", "update", issue_id, "--notes", notes],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )
