"""E2E fixture runner for mala validation.

This module provides end-to-end validation using a fixture repository.
It creates a temporary repo with a known bug, runs mala to fix it,
and validates the result.

Key types:
- E2EResult: Result of an E2E validation run
- E2EConfig: Configuration for E2E validation
- E2ERunner: Orchestrates the E2E validation flow
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from .helpers import (
    annotate_issue,
    get_ready_issue_id,
    init_fixture_repo,
    write_fixture_repo,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from src.core.protocols.infra import CommandRunnerPort, EnvConfigPort


class E2EStatus(Enum):
    """Status of E2E validation."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"  # Skipped due to missing prerequisites


@dataclass
class E2EPrereqResult:
    """Result of prerequisite check.

    Attributes:
        ok: Whether all prerequisites are met.
        missing: List of missing prerequisites.
        can_skip: Whether E2E can be skipped rather than failed.
    """

    ok: bool
    missing: list[str] = field(default_factory=list)
    can_skip: bool = False

    def failure_reason(self) -> str | None:
        """Return failure reason string, or None if ok."""
        if self.ok:
            return None
        if not self.missing:
            return "E2E prerequisites not met"
        return f"E2E prereq missing: {', '.join(self.missing)}"


@dataclass
class E2EResult:
    """Result of an E2E validation run.

    Attributes:
        passed: Whether E2E validation passed.
        status: Status code for the E2E run.
        failure_reason: Explanation for failure (None if passed).
        fixture_path: Path to the fixture repo (if created).
        duration_seconds: How long the E2E run took.
        command_output: Output from the mala command (truncated).
        returncode: Exit code from the mala command.
    """

    passed: bool
    status: E2EStatus
    failure_reason: str | None = None
    fixture_path: Path | None = None
    duration_seconds: float = 0.0
    command_output: str = ""
    returncode: int = 0

    def short_summary(self) -> str:
        """One-line summary for logs/prompts."""
        if self.status == E2EStatus.SKIPPED:
            return f"E2E skipped: {self.failure_reason or 'prerequisites not met'}"
        if self.passed:
            return "E2E passed"
        return f"E2E failed: {self.failure_reason or 'unknown error'}"


@dataclass
class E2EConfig:
    """Configuration for E2E validation.

    Attributes:
        enabled: Whether E2E is enabled.
        skip_if_no_keys: Deprecated, kept for backward compatibility.
        keep_fixture: Keep fixture repo after completion (for debugging).
        timeout_seconds: Timeout for the mala run command (default 300s/5min).
        max_agents: Maximum agents for the mala run.
        max_issues: Maximum issues to process in the mala run.
        cerberus_mode: Cerberus review mode (fast/smart/max). Default "fast" for E2E.
    """

    enabled: bool = True
    skip_if_no_keys: bool = False
    keep_fixture: bool = False
    timeout_seconds: float = 300.0
    max_agents: int = 1
    max_issues: int = 1
    cerberus_mode: str = "fast"


class E2ERunner:
    """Orchestrates E2E validation using a fixture repository."""

    def __init__(
        self,
        env_config: EnvConfigPort,
        command_runner: CommandRunnerPort,
        config: E2EConfig | None = None,
    ):
        """Initialize the E2E runner.

        Args:
            env_config: Environment configuration for finding cerberus bin path.
            command_runner: CommandRunnerPort for running mala commands.
            config: E2E configuration. Uses defaults if None.
        """
        self.config = config or E2EConfig()
        self.env_config = env_config
        self._command_runner = command_runner

    def check_prereqs(self, env: Mapping[str, str] | None = None) -> E2EPrereqResult:
        """Check if all E2E prerequisites are met.

        Args:
            env: Environment variables to check. Uses os.environ if None.

        Returns:
            E2EPrereqResult with details about missing prerequisites.
        """
        import os

        if env is None:
            env = os.environ

        missing: list[str] = []

        # Check for mala CLI
        if not shutil.which("mala"):
            missing.append("mala CLI not found in PATH")

        # Check for br CLI
        if not shutil.which("br"):
            missing.append("br CLI not found in PATH")

        # Check for Cerberus review-gate (required for E2E to test review flow)
        cerberus_bin = self.env_config.find_cerberus_bin_path()
        if cerberus_bin is None:
            missing.append(
                "Cerberus review-gate not installed (check ~/.claude/plugins)"
            )
        elif not (cerberus_bin / "review-gate").exists():
            missing.append(f"review-gate binary not found at {cerberus_bin}")

        if missing:
            return E2EPrereqResult(ok=False, missing=missing, can_skip=False)

        return E2EPrereqResult(ok=True)

    def run(
        self, env: Mapping[str, str] | None = None, cwd: Path | None = None
    ) -> E2EResult:
        """Run E2E validation.

        Creates a fixture repo, runs mala on it, and validates the result.
        Cleans up the fixture repo unless keep_fixture is True.

        Args:
            env: Environment variables for subprocess. Uses os.environ if None.
            cwd: Working directory for mala command. Uses current directory if None.

        Returns:
            E2EResult with details about the validation.
        """
        import os

        if env is None:
            env = dict(os.environ)
        else:
            env = dict(env)

        if cwd is None:
            cwd = Path.cwd()

        # Check prerequisites
        prereq = self.check_prereqs(env)
        if not prereq.ok:
            if prereq.can_skip:
                return E2EResult(
                    passed=True,  # Skipped is considered "not failed"
                    status=E2EStatus.SKIPPED,
                    failure_reason=prereq.failure_reason(),
                )
            return E2EResult(
                passed=False,
                status=E2EStatus.FAILED,
                failure_reason=prereq.failure_reason(),
            )

        repo_root = _resolve_repo_root(cwd)

        # Create fixture repo
        fixture_path = Path(tempfile.mkdtemp(prefix="mala-e2e-fixture-"))
        start_time = time.monotonic()

        try:
            # Write fixture files
            setup_error = self._setup_fixture(fixture_path, repo_root)
            if setup_error:
                duration = time.monotonic() - start_time
                return E2EResult(
                    passed=False,
                    status=E2EStatus.FAILED,
                    failure_reason=setup_error,
                    fixture_path=fixture_path if self.config.keep_fixture else None,
                    duration_seconds=duration,
                )

            # Run mala
            result = self._run_mala(fixture_path, env, cwd, repo_root)
            result.fixture_path = fixture_path if self.config.keep_fixture else None

            return result

        finally:
            duration = time.monotonic() - start_time
            # Cleanup fixture unless keeping it
            if not self.config.keep_fixture and fixture_path.exists():
                shutil.rmtree(fixture_path, ignore_errors=True)

    def _setup_fixture(self, repo_path: Path, repo_root: Path) -> str | None:
        """Set up the fixture repository.

        Args:
            repo_path: Path to create the fixture repo in.

        Returns:
            Error message if setup failed, None on success.
        """
        # Write fixture files using shared helper
        write_fixture_repo(repo_path, repo_root=repo_root)
        _disable_fixture_e2e_config(repo_path)

        # Initialize git and beads using shared helper
        return init_fixture_repo(repo_path, self._command_runner)

    def _run_mala(
        self,
        fixture_path: Path,
        env: Mapping[str, str],
        cwd: Path,
        repo_root: Path,
    ) -> E2EResult:
        """Run mala on the fixture repo.

        Args:
            fixture_path: Path to the fixture repository.
            env: Environment variables for subprocess.
            cwd: Working directory for the mala command.

        Returns:
            E2EResult with command execution details.
        """
        # Annotate the issue with context using shared helper
        issue_id = get_ready_issue_id(fixture_path, self._command_runner)
        if issue_id:
            annotate_issue(fixture_path, issue_id, self._command_runner)

        # Override CLAUDE_SESSION_ID to avoid conflicts with parent session's review gate.
        # The Cerberus review-gate tracks pending reviews per session, so running e2e
        # inside an existing mala session (which already has a review gate active) would
        # fail with "Review gate already active" unless we use a distinct session ID.
        child_env = dict(env)
        child_env["CLAUDE_SESSION_ID"] = f"e2e-{uuid.uuid4()}"

        # Inject cerberus mode into fixture's mala.yaml
        _inject_cerberus_mode(fixture_path, self.config.cerberus_mode)

        # Convert timeout from seconds to minutes for CLI (which expects minutes)
        timeout_minutes = max(1, int(self.config.timeout_seconds // 60))

        # Prefer invoking the local module to avoid mismatched global installs.
        # Ensure the repo root is on PYTHONPATH so src.cli.main is importable.
        pythonpath = child_env.get("PYTHONPATH", "")
        if str(repo_root) not in pythonpath.split(os.pathsep):
            child_env["PYTHONPATH"] = (
                f"{repo_root}{os.pathsep}{pythonpath}" if pythonpath else str(repo_root)
            )

        cmd = [
            *_select_python_invoker(repo_root),
            "-m",
            "src.cli.main",
            "run",
            str(fixture_path),
            "--max-agents",
            str(self.config.max_agents),
            "--max-issues",
            str(self.config.max_issues),
            "--timeout",
            str(timeout_minutes),
        ]

        runner = self._command_runner
        result = runner.run(cmd, env=child_env, cwd=cwd)

        if result.ok:
            return E2EResult(
                passed=True,
                status=E2EStatus.PASSED,
                duration_seconds=result.duration_seconds,
                command_output=result.stdout_tail(),
                returncode=0,
            )

        if result.timed_out:
            return E2EResult(
                passed=False,
                status=E2EStatus.FAILED,
                failure_reason=f"mala timed out after {self.config.timeout_seconds}s",
                duration_seconds=result.duration_seconds,
                command_output=result.stderr_tail() or result.stdout_tail(),
                returncode=124,
            )

        output = result.stderr_tail() or result.stdout_tail()
        return E2EResult(
            passed=False,
            status=E2EStatus.FAILED,
            failure_reason=f"mala exited {result.returncode}: {output}",
            duration_seconds=result.duration_seconds,
            command_output=output,
            returncode=result.returncode,
        )


def check_e2e_prereqs(
    env_config: EnvConfigPort,
    command_runner: CommandRunnerPort,
    env: Mapping[str, str],
) -> str | None:
    """Check E2E prerequisites.

    Args:
        env_config: Environment configuration for paths.
        command_runner: Command runner for executing commands.
        env: Environment variables to check.

    Returns:
        Error message if prerequisites not met, None if all ok.
    """
    runner = E2ERunner(env_config, command_runner)
    result = runner.check_prereqs(env)
    return result.failure_reason()


def _disable_fixture_e2e_config(repo_path: Path) -> None:
    """Disable E2E in the fixture config to avoid recursive runs."""
    config_path = repo_path / "mala.yaml"
    if not config_path.exists():
        return

    lines = config_path.read_text().splitlines()
    output: list[str] = []
    in_commands = False
    e2e_set = False
    indent = 0

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("commands:") and not line.startswith(" "):
            in_commands = True
            indent = len(line) - len(line.lstrip())
            output.append(line)
            continue

        if in_commands:
            if stripped and (len(line) - len(line.lstrip())) <= indent:
                if not e2e_set:
                    output.append(" " * (indent + 2) + "e2e: null")
                    e2e_set = True
                in_commands = False
            elif stripped.startswith("e2e:"):
                output.append(" " * (len(line) - len(line.lstrip())) + "e2e: null")
                e2e_set = True
                continue

        output.append(line)

    if in_commands and not e2e_set:
        output.append(" " * (indent + 2) + "e2e: null")
        e2e_set = True

    if not any(
        line.strip().startswith("commands:") and not line.startswith(" ")
        for line in lines
    ):
        output.append("")
        output.append("commands:")
        output.append("  e2e: null")

    config_path.write_text("\n".join(output) + "\n")


def _inject_cerberus_mode(repo_path: Path, cerberus_mode: str) -> None:
    """Inject cerberus mode into the fixture's mala.yaml.

    Merges session_end.code_review.cerberus.spawn_args into the existing
    validation_triggers config using proper YAML parsing to avoid duplicate keys.

    Args:
        repo_path: Path to the fixture repository.
        cerberus_mode: Cerberus mode to set (e.g., "fast", "smart", "max").
    """
    import yaml

    config_path = repo_path / "mala.yaml"
    if not config_path.exists():
        return

    content = config_path.read_text()
    loaded = yaml.safe_load(content)
    # Guard against non-dict YAML roots (list, scalar, null)
    config = loaded if isinstance(loaded, dict) else {}

    # Ensure nested structure exists
    if "validation_triggers" not in config:
        config["validation_triggers"] = {}
    if "session_end" not in config["validation_triggers"]:
        # Include required failure_mode when creating session_end
        config["validation_triggers"]["session_end"] = {"failure_mode": "continue"}
    if "code_review" not in config["validation_triggers"]["session_end"]:
        config["validation_triggers"]["session_end"]["code_review"] = {}
    if "cerberus" not in config["validation_triggers"]["session_end"]["code_review"]:
        config["validation_triggers"]["session_end"]["code_review"]["cerberus"] = {}

    # Set the spawn_args
    config["validation_triggers"]["session_end"]["code_review"]["cerberus"][
        "spawn_args"
    ] = [f"--mode={cerberus_mode}"]

    # Write back with yaml.dump
    config_path.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))


def _select_python_invoker(repo_root: Path) -> list[str]:
    """Select a python invoker for running mala in E2E.

    Prefers uv when the repo's validation config uses uv-based commands
    and uv is available; otherwise falls back to the current interpreter.
    """
    if _config_prefers_uv(repo_root) and shutil.which("uv"):
        return ["uv", "run", "--directory", str(repo_root), "python"]
    return [sys.executable]


def _config_prefers_uv(repo_root: Path) -> bool:
    """Return True if validation commands indicate uv usage."""
    try:
        from src.domain.validation.spec import ValidationScope, build_validation_spec

        spec = build_validation_spec(repo_root, scope=ValidationScope.GLOBAL)
    except Exception:
        return False

    if _commands_use_uv(cmd.command for cmd in spec.commands):
        return True

    return (repo_root / "uv.lock").exists()


def _resolve_repo_root(cwd: Path) -> Path:
    """Resolve the repo root for running E2E.

    The E2E runner expects to be invoked with cwd set to the repo/worktree root,
    so treat the working directory as the repo root.
    """
    return cwd


def _commands_use_uv(commands: Iterable[str]) -> bool:
    for command in commands:
        tokens = _parse_command_tokens(command)
        if not tokens:
            continue
        first = tokens[0].lower()
        if first in {"uv", "uvx"}:
            return True
    return False


def _parse_command_tokens(command: str) -> list[str]:
    """Parse command string and drop leading env assignments."""
    import shlex

    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()

    idx = 0
    while idx < len(tokens) and _is_env_assignment(tokens[idx]):
        idx += 1
    return tokens[idx:]


def _is_env_assignment(token: str) -> bool:
    if "=" not in token:
        return False
    name, _, _ = token.partition("=")
    if not name or name[0].isdigit():
        return False
    return all(c.isalnum() or c == "_" for c in name)
