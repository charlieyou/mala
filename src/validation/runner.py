"""Post-commit validation runner for mala.

Runs clean-room validations in a temporary git worktree and (optionally) a
true end-to-end `mala run` smoke test using a fixture repo.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..tools.env import LOCK_DIR, SCRIPTS_DIR


@dataclass
class ValidationConfig:
    """Configuration for post-commit validation."""

    run_slow_tests: bool = True
    run_e2e: bool = True
    coverage: bool = True
    coverage_min: int = 85
    coverage_source: str = "."
    use_test_mutex: bool = True
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
