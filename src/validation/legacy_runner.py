"""Legacy validation runner for mala.

This module provides LegacyValidationRunner which uses ValidationConfig
to run validation. This is the older API preserved for backwards compatibility.

For new code, prefer SpecValidationRunner with ValidationSpec.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from ..tools.env import LOCK_DIR, SCRIPTS_DIR
from .helpers import (
    annotate_issue,
    check_e2e_prereqs,
    decode_timeout_output,
    format_step_output,
    get_ready_issue_id,
    init_fixture_repo,
    tail,
    write_fixture_repo,
)
from .result import ValidationResult, ValidationStepResult


@dataclass
class ValidationConfig:
    """Configuration for post-commit validation (legacy API).

    For new code, prefer ValidationSpec with SpecValidationRunner.
    """

    run_slow_tests: bool = True
    run_e2e: bool = True
    coverage: bool = True
    coverage_min: int = 85
    coverage_source: str = "."
    use_test_mutex: bool = (
        False  # Disabled to avoid deadlocks; isolated caches prevent conflicts
    )
    step_timeout_seconds: float | None = None


class LegacyValidationRunner:
    """Runs post-commit validation in a clean git worktree (legacy API).

    This runner uses ValidationConfig and provides the validate_commit() method.
    For new code, prefer SpecValidationRunner with ValidationSpec.
    """

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
                    reason += f" ({tail(add_result.stderr.strip())})"
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
                details = format_step_output(step.stdout_tail, step.stderr_tail)
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
            stdout_tail = tail(result.stdout)
            stderr_tail = tail(result.stderr)
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
            stdout_tail = decode_timeout_output(exc.stdout)
            stderr_tail = decode_timeout_output(exc.stderr)
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
        prereq_failure = check_e2e_prereqs(env)
        if prereq_failure:
            return ValidationResult(
                passed=False,
                failure_reasons=[prereq_failure],
                retriable=False,
            )

        with tempfile.TemporaryDirectory(prefix=f"mala-e2e-{label}-") as tmp_dir:
            repo_path = Path(tmp_dir).resolve()
            write_fixture_repo(repo_path)
            init_result = init_fixture_repo(repo_path)
            if init_result:
                return ValidationResult(
                    passed=False,
                    failure_reasons=[init_result],
                    retriable=False,
                )

            issue_id = get_ready_issue_id(repo_path)
            if issue_id:
                annotate_issue(repo_path, issue_id)

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
            details = format_step_output(step.stdout_tail, step.stderr_tail)
            if details:
                reason = f"{reason}: {details}"
            return ValidationResult(
                passed=False,
                steps=[step],
                failure_reasons=[reason],
                retriable=False,
            )
