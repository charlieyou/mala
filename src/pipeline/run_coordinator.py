"""RunCoordinator: Main run orchestration pipeline stage.

Extracted from MalaOrchestrator to separate run-level coordination from
the main class. This module handles:
- Main run loop (spawning agents, waiting for completion)
- Run-level validation (Gate 4)
- Fixer agent spawning for validation failures

The RunCoordinator receives explicit inputs and returns explicit outputs,
making it testable without SDK or subprocess dependencies.

Design principles:
- Protocol-based dependencies for testability
- Explicit input/output types for clarity
- Pure functions where possible
"""

from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.infra.hooks import (
    FileReadCache,
    LintCache,
    block_dangerous_commands,
    make_file_read_cache_hook,
    make_lint_cache_hook,
    make_lock_enforcement_hook,
    make_stop_hook,
)
from src.infra.mcp import get_disallowed_tools, get_mcp_servers
from src.infra.tools.env import SCRIPTS_DIR, get_lock_dir
from src.infra.tools.locking import cleanup_agent_locks
from src.domain.validation.e2e import E2EStatus
from src.domain.validation.spec import (
    ValidationContext,
    ValidationScope,
    build_validation_spec,
)
from src.infra.tool_name_extractor import extract_lint_tools_from_spec
from src.domain.validation.spec_runner import SpecValidationRunner

if TYPE_CHECKING:
    from src.infra.io.event_protocol import MalaEventSink
    from src.infra.io.log_output.run_metadata import (
        RunMetadata,
        ValidationResult as MetaValidationResult,
    )
    from src.core.protocols import (
        CommandRunnerPort,
        EnvConfigPort,
        GateChecker,
        LockManagerPort,
    )
    from src.domain.validation.result import ValidationResult
    from src.domain.validation.spec import ValidationSpec
    from pathlib import Path


class _FixerPromptNotSet:
    """Sentinel indicating fixer_prompt was not set.

    Raises RuntimeError if used, preventing silent failures from empty prompts.
    """

    def format(self, **kwargs: object) -> str:
        raise RuntimeError(
            "fixer_prompt not configured. "
            "Pass fixer_prompt to RunCoordinatorConfig or use build_run_coordinator()."
        )


_FIXER_PROMPT_NOT_SET = _FixerPromptNotSet()


@dataclass
class RunCoordinatorConfig:
    """Configuration for RunCoordinator.

    Attributes:
        repo_path: Path to the repository.
        timeout_seconds: Session timeout in seconds.
        max_gate_retries: Maximum gate retry attempts.
        disable_validations: Set of validation names to disable.
        coverage_threshold: Optional coverage threshold override.
        fixer_prompt: Template for the fixer agent prompt.
    """

    repo_path: Path
    timeout_seconds: int
    max_gate_retries: int = 3
    disable_validations: set[str] | None = None
    coverage_threshold: float | None = None
    fixer_prompt: str | _FixerPromptNotSet = _FIXER_PROMPT_NOT_SET


@dataclass
class RunLevelValidationInput:
    """Input for run-level validation.

    Attributes:
        run_metadata: Run metadata tracker.
    """

    run_metadata: RunMetadata


@dataclass
class RunLevelValidationOutput:
    """Output from run-level validation.

    Attributes:
        passed: Whether validation passed.
    """

    passed: bool


@dataclass
class SpecResultBuilder:
    """Builds ValidationResult to MetaValidationResult conversions.

    Encapsulates the logic for deriving e2e_passed status and building
    metadata results from validation results.
    """

    @staticmethod
    def derive_e2e_passed(result: ValidationResult) -> bool | None:
        """Derive e2e_passed from E2E execution result.

        Args:
            result: ValidationResult to check.

        Returns:
            None if E2E was not executed (disabled or skipped)
            True if E2E was executed and passed
            False if E2E was executed and failed
        """
        if result.e2e_result is None:
            return None
        if result.e2e_result.status == E2EStatus.SKIPPED:
            return None
        return result.e2e_result.passed

    @staticmethod
    def build_meta_result(
        result: ValidationResult,
        passed: bool,
    ) -> MetaValidationResult:
        """Build a MetaValidationResult from a ValidationResult.

        Args:
            result: The validation result.
            passed: Whether validation passed overall.

        Returns:
            MetaValidationResult for run metadata.
        """
        from src.infra.io.log_output.run_metadata import (
            ValidationResult as MetaValidationResult,
        )

        e2e_passed = SpecResultBuilder.derive_e2e_passed(result)
        failed_commands = (
            [s.name for s in result.steps if not s.ok] if not passed else []
        )

        return MetaValidationResult(
            passed=passed,
            commands_run=[s.name for s in result.steps],
            commands_failed=failed_commands,
            artifacts=result.artifacts,
            coverage_percent=result.coverage_result.percent
            if result.coverage_result
            else None,
            e2e_passed=e2e_passed,
        )


@dataclass
class RunCoordinator:
    """Run-level coordination for MalaOrchestrator.

    This class encapsulates the run-level orchestration logic that was
    previously inline in MalaOrchestrator. It handles:
    - Run-level validation (Gate 4) with fixer retries
    - Fixer agent spawning

    The orchestrator delegates to this class for run-level operations
    while retaining per-issue coordination.

    Attributes:
        config: Configuration for run behavior.
        gate_checker: GateChecker for run-level validation.
        command_runner: CommandRunner for executing validation commands.
        env_config: Environment configuration for paths.
        lock_manager: Lock manager for file locking.
        event_sink: Optional event sink for structured logging.
    """

    config: RunCoordinatorConfig
    gate_checker: GateChecker
    command_runner: CommandRunnerPort
    env_config: EnvConfigPort
    lock_manager: LockManagerPort
    event_sink: MalaEventSink | None = None
    _active_fixer_ids: list[str] = field(default_factory=list, init=False)

    async def run_validation(
        self, input: RunLevelValidationInput
    ) -> RunLevelValidationOutput:
        """Run Gate 4 validation after all issues complete.

        This runs validation with RUN_LEVEL scope, which includes E2E tests.
        On failure, spawns a fixer agent and retries up to max_gate_retries.

        Args:
            input: RunLevelValidationInput with run metadata.

        Returns:
            RunLevelValidationOutput indicating pass/fail.
        """
        from src.infra.git_utils import get_git_commit_async

        # Check if run-level validation is disabled
        if "run-level-validate" in (self.config.disable_validations or set()):
            if self.event_sink is not None:
                self.event_sink.on_run_level_validation_disabled()
            return RunLevelValidationOutput(passed=True)

        # Get current HEAD commit
        commit_hash = await get_git_commit_async(self.config.repo_path)
        if not commit_hash:
            if self.event_sink is not None:
                self.event_sink.on_warning(
                    "Could not get HEAD commit for run-level validation"
                )
            return RunLevelValidationOutput(passed=True)

        # Build run-level validation spec
        spec = build_validation_spec(
            self.config.repo_path,
            scope=ValidationScope.RUN_LEVEL,
            disable_validations=self.config.disable_validations,
        )

        # Build validation context
        context = ValidationContext(
            issue_id=None,
            repo_path=self.config.repo_path,
            commit_hash=commit_hash,
            changed_files=[],
            scope=ValidationScope.RUN_LEVEL,
        )

        # Create validation runner
        runner = SpecValidationRunner(
            self.config.repo_path,
            event_sink=self.event_sink,
            command_runner=self.command_runner,
            env_config=self.env_config,
            lock_manager=self.lock_manager,
        )

        # Retry loop with fixer agent
        for attempt in range(1, self.config.max_gate_retries + 1):
            if self.event_sink is not None:
                self.event_sink.on_gate_started(
                    None, attempt, self.config.max_gate_retries
                )

            # Run validation
            try:
                result = await runner.run_spec(spec, context)
            except Exception as e:
                if self.event_sink is not None:
                    self.event_sink.on_warning(f"Validation runner error: {e}")
                result = None

            if result and result.passed:
                if self.event_sink is not None:
                    self.event_sink.on_gate_passed(None)
                meta_result = SpecResultBuilder.build_meta_result(result, passed=True)
                input.run_metadata.record_run_validation(meta_result)
                return RunLevelValidationOutput(passed=True)

            # Validation failed - build failure output for fixer
            failure_output = self._build_validation_failure_output(result)

            # Record failure in metadata
            if result:
                meta_result = SpecResultBuilder.build_meta_result(result, passed=False)
                input.run_metadata.record_run_validation(meta_result)

            # Check if we have retries left
            if attempt >= self.config.max_gate_retries:
                if self.event_sink is not None:
                    self.event_sink.on_gate_failed(
                        None, attempt, self.config.max_gate_retries
                    )
                    failure_reasons = (
                        result.failure_reasons
                        if result and result.failure_reasons
                        else []
                    )
                    self.event_sink.on_gate_result(
                        None, passed=False, failure_reasons=failure_reasons
                    )
                return RunLevelValidationOutput(passed=False)

            # Spawn fixer agent
            if self.event_sink is not None:
                self.event_sink.on_fixer_started(attempt, self.config.max_gate_retries)

            fixer_success = await self._run_fixer_agent(
                failure_output=failure_output,
                attempt=attempt,
                spec=spec,
            )

            if not fixer_success:
                # Fixer failure is logged via on_fixer_failed in _run_fixer_agent
                pass  # Continue to retry validation anyway

            # Update commit hash for next validation attempt
            new_commit = await get_git_commit_async(self.config.repo_path)
            if new_commit and new_commit != commit_hash:
                commit_hash = new_commit
                context = ValidationContext(
                    issue_id=None,
                    repo_path=self.config.repo_path,
                    commit_hash=commit_hash,
                    changed_files=[],
                    scope=ValidationScope.RUN_LEVEL,
                )

        return RunLevelValidationOutput(passed=False)

    def _build_validation_failure_output(self, result: ValidationResult | None) -> str:
        """Build failure output string for fixer agent prompt.

        Args:
            result: Validation result, or None if validation crashed.

        Returns:
            Human-readable failure output.
        """
        if result is None:
            return "Validation crashed - check logs for details."

        lines: list[str] = []
        if result.failure_reasons:
            lines.append("**Failure reasons:**")
            for reason in result.failure_reasons:
                lines.append(f"- {reason}")
            lines.append("")

        # Add step details for failed steps
        failed_steps = [s for s in result.steps if not s.ok]
        if failed_steps:
            lines.append("**Failed validation steps:**")
            for step in failed_steps:
                lines.append(f"\n### {step.name} (exit {step.returncode})")
                if step.stderr_tail:
                    lines.append("```")
                    lines.append(step.stderr_tail[:2000])
                    lines.append("```")
                elif step.stdout_tail:
                    lines.append("```")
                    lines.append(step.stdout_tail[:2000])
                    lines.append("```")

        return (
            "\n".join(lines) if lines else "Validation failed (no details available)."
        )

    async def _run_fixer_agent(
        self,
        failure_output: str,
        attempt: int,
        spec: ValidationSpec | None = None,
    ) -> bool:
        """Spawn a fixer agent to address run-level validation failures.

        Args:
            failure_output: Human-readable description of what failed.
            attempt: Current attempt number.
            spec: Optional ValidationSpec for extracting lint tool names.

        Returns:
            True if fixer agent completed successfully, False otherwise.
        """
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ClaudeSDKClient,
            ResultMessage,
            TextBlock,
            ToolResultBlock,
            ToolUseBlock,
        )
        from claude_agent_sdk.types import HookMatcher

        agent_id = f"fixer-{uuid.uuid4().hex[:8]}"
        self._active_fixer_ids.append(agent_id)

        prompt = self.config.fixer_prompt.format(
            attempt=attempt,
            max_attempts=self.config.max_gate_retries,
            failure_output=failure_output,
        )

        fixer_cwd = self.config.repo_path

        # Set up environment
        agent_env = {
            **os.environ,
            "PATH": f"{SCRIPTS_DIR}:{os.environ.get('PATH', '')}",
            "LOCK_DIR": str(get_lock_dir()),
            "AGENT_ID": agent_id,
            "REPO_NAMESPACE": str(self.config.repo_path),
            "MCP_TIMEOUT": "300000",
        }

        # Build hooks
        file_read_cache = FileReadCache()
        lint_tools = extract_lint_tools_from_spec(spec)
        lint_cache = LintCache(repo_path=fixer_cwd, lint_tools=lint_tools)
        pre_tool_hooks: list = [
            block_dangerous_commands,
            make_lock_enforcement_hook(agent_id, str(self.config.repo_path)),
            make_file_read_cache_hook(file_read_cache),
            make_lint_cache_hook(lint_cache),
        ]

        options = ClaudeAgentOptions(
            cwd=str(fixer_cwd),
            permission_mode="bypassPermissions",
            model="opus",
            system_prompt={"type": "preset", "preset": "claude_code"},
            setting_sources=["project", "user"],
            mcp_servers=get_mcp_servers(
                fixer_cwd,
            ),
            disallowed_tools=get_disallowed_tools(),
            env=agent_env,
            hooks={
                "PreToolUse": [
                    HookMatcher(
                        matcher=None,
                        hooks=pre_tool_hooks,  # type: ignore[arg-type]
                    )
                ],
                "Stop": [
                    HookMatcher(matcher=None, hooks=[make_stop_hook(agent_id)])  # type: ignore[arg-type]
                ],
            },
        )

        pending_lint_commands: dict[str, tuple[str, str]] = {}

        try:
            async with asyncio.timeout(self.config.timeout_seconds):
                async with ClaudeSDKClient(options=options) as client:
                    await client.query(prompt)

                    async for message in client.receive_response():
                        if isinstance(message, AssistantMessage):
                            for block in message.content:
                                if isinstance(block, TextBlock):
                                    if self.event_sink is not None:
                                        self.event_sink.on_fixer_text(
                                            attempt, block.text
                                        )
                                elif isinstance(block, ToolUseBlock):
                                    if self.event_sink is not None:
                                        self.event_sink.on_fixer_tool_use(
                                            attempt, block.name, block.input
                                        )
                                    if block.name.lower() == "bash":
                                        cmd = block.input.get("command", "")
                                        lint_type = lint_cache.detect_lint_command(cmd)
                                        if lint_type:
                                            pending_lint_commands[block.id] = (
                                                lint_type,
                                                cmd,
                                            )
                                elif isinstance(block, ToolResultBlock):
                                    tool_use_id = getattr(block, "tool_use_id", None)
                                    if tool_use_id in pending_lint_commands:
                                        lint_type, cmd = pending_lint_commands.pop(
                                            tool_use_id
                                        )
                                        if not getattr(block, "is_error", False):
                                            lint_cache.mark_success(lint_type, cmd)
                        elif isinstance(message, ResultMessage):
                            if self.event_sink is not None:
                                self.event_sink.on_fixer_completed(message.result or "")

            return True

        except TimeoutError:
            if self.event_sink is not None:
                self.event_sink.on_fixer_failed("timeout")
            return False
        except Exception as e:
            if self.event_sink is not None:
                self.event_sink.on_fixer_failed(str(e))
            return False
        finally:
            self._active_fixer_ids.remove(agent_id)
            cleanup_agent_locks(agent_id)

    def cleanup_fixer_locks(self) -> None:
        """Clean up any remaining fixer agent locks."""
        for agent_id in self._active_fixer_ids:
            cleanup_agent_locks(agent_id)
        self._active_fixer_ids.clear()
