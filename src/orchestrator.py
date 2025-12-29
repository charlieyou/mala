"""MalaOrchestrator: Orchestrates parallel issue processing using Claude Agent SDK."""

import asyncio
import functools
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
)
from claude_agent_sdk.types import HookMatcher

from .beads_client import BeadsClient
from .config import MalaConfig
from .telemetry import BraintrustProvider, NullTelemetryProvider, TelemetryProvider
from .codex_review import run_codex_review, format_review_issues, CodexReviewResult
from .git_utils import (
    get_git_commit_async,
    get_git_branch_async,
    get_baseline_for_issue,
)
from .hooks import (
    MORPH_DISALLOWED_TOOLS,
    block_dangerous_commands,
    block_morph_replaced_tools,
    make_lock_enforcement_hook,
    make_stop_hook,
)
from .lifecycle import (
    Effect,
    ImplementerLifecycle,
    LifecycleConfig,
    LifecycleContext,
    LifecycleState,
    RetryState,
)
from .logging.console import (
    Colors,
    log,
    log_tool,
    log_agent_text,
    log_verbose,
    truncate_text,
    is_verbose_enabled,
)
from .logging.run_metadata import (
    RunMetadata,
    RunConfig,
    IssueRun,
    QualityGateResult,
    ValidationResult as MetaValidationResult,
    remove_run_marker,
    write_run_marker,
)
from .protocols import CodeReviewer, GateChecker, IssueProvider
from .quality_gate import QualityGate, GateResult
from .tools.env import (
    USER_CONFIG_DIR,
    SCRIPTS_DIR,
    get_claude_log_path,
    get_lock_dir,
    get_runs_dir,
)
from .tools.locking import (
    release_run_locks,
    cleanup_agent_locks,
)
from .validation.spec import (
    IssueResolution,
    ValidationContext,
    ValidationScope,
    build_validation_spec,
)
from .validation.spec_runner import SpecValidationRunner
from .validation.e2e import E2EStatus

if TYPE_CHECKING:
    from .validation.result import ValidationResult
    from .validation.spec import ValidationSpec


# Version (from package metadata)
from importlib.metadata import version as pkg_version

__version__ = pkg_version("mala")

# Prompt file paths (actual file reads deferred to first use)
_PROMPT_DIR = Path(__file__).parent / "prompts"
PROMPT_FILE = _PROMPT_DIR / "implementer_prompt.md"
GATE_FOLLOWUP_FILE = _PROMPT_DIR / "gate_followup.md"
REVIEW_FOLLOWUP_FILE = _PROMPT_DIR / "review_followup.md"
FIXER_PROMPT_FILE = _PROMPT_DIR / "fixer.md"

# Default timeout for agent execution (protects against hung MCP server subprocesses)
# This ensures stuck subprocesses (e.g., ripgrep searching from wrong directory)
# don't hang indefinitely even when cwd is misconfigured.
DEFAULT_AGENT_TIMEOUT_MINUTES = 30


@functools.cache
def _get_implementer_prompt() -> str:
    """Load implementer prompt template (cached on first use)."""
    return PROMPT_FILE.read_text()


@functools.cache
def _get_gate_followup_prompt() -> str:
    """Load gate follow-up prompt (cached on first use)."""
    return GATE_FOLLOWUP_FILE.read_text()


@functools.cache
def _get_review_followup_prompt() -> str:
    """Load review follow-up prompt (cached on first use)."""
    return REVIEW_FOLLOWUP_FILE.read_text()


@functools.cache
def _get_fixer_prompt() -> str:
    """Load fixer prompt (cached on first use)."""
    return FIXER_PROMPT_FILE.read_text()


# Bounded wait for log file (seconds)
LOG_FILE_WAIT_TIMEOUT = 30
LOG_FILE_POLL_INTERVAL = 0.5


class DefaultCodeReviewer:
    """Default CodeReviewer implementation that wraps run_codex_review.

    This class conforms to the CodeReviewer protocol and provides the default
    behavior for the orchestrator. Tests can inject alternative implementations.
    """

    async def __call__(
        self,
        repo_path: Path,
        commit_sha: str,
        max_retries: int = 2,
        issue_description: str | None = None,
        baseline_commit: str | None = None,
        capture_session_log: bool = False,
        thinking_mode: str | None = None,
    ) -> CodexReviewResult:
        """Delegate to run_codex_review function."""
        return await run_codex_review(
            repo_path,
            commit_sha,
            max_retries=max_retries,
            issue_description=issue_description,
            baseline_commit=baseline_commit,
            capture_session_log=capture_session_log,
            thinking_mode=thinking_mode,
        )


def get_mcp_servers(
    repo_path: Path, morph_api_key: str | None = None, morph_enabled: bool = True
) -> dict:
    """Get MCP servers configuration for agents.

    Args:
        repo_path: Path to the repository.
        morph_api_key: Morph API key. Required when morph_enabled=True.
        morph_enabled: Whether to enable Morph MCP server. Defaults to True.

    Returns:
        Dictionary of MCP server configurations. Empty if morph_enabled=False.

    Raises:
        ValueError: If morph_enabled=True but morph_api_key is not provided.
    """
    if not morph_enabled:
        return {}
    if not morph_api_key:
        raise ValueError(
            "morph_api_key is required when morph_enabled=True. "
            "Pass morph_api_key from MalaConfig or set morph_enabled=False."
        )
    return {
        "morphllm": {
            "command": "npx",
            "args": ["-y", "@morphllm/morphmcp"],
            "cwd": str(repo_path),
            "env": {
                "MORPH_API_KEY": morph_api_key,
                "ENABLED_TOOLS": "all",
                "WORKSPACE_MODE": "true",
                "WORKSPACE_PATH": str(repo_path),
            },
        }
    }


@dataclass
class IssueResult:
    """Result from a single issue implementation."""

    issue_id: str
    agent_id: str
    success: bool
    summary: str
    duration_seconds: float = 0.0
    session_id: str | None = None  # Claude SDK session ID
    gate_attempts: int = 1  # Number of gate retry attempts
    review_attempts: int = 0  # Number of Codex review attempts
    resolution: IssueResolution | None = None  # Resolution outcome if using markers


class MalaOrchestrator:
    """Orchestrates parallel issue processing using Claude Agent SDK."""

    def __init__(
        self,
        repo_path: Path,
        max_agents: int | None = None,
        timeout_minutes: int | None = None,
        max_issues: int | None = None,
        epic_id: str | None = None,
        only_ids: set[str] | None = None,
        braintrust_enabled: bool | None = None,
        max_gate_retries: int = 3,
        max_review_retries: int = 2,
        disable_validations: set[str] | None = None,
        coverage_threshold: float | None = None,
        morph_enabled: bool | None = None,
        prioritize_wip: bool = False,
        focus: bool = True,
        cli_args: dict[str, object] | None = None,
        codex_thinking_mode: str | None = None,
        # Protocol-based dependency injection for testability
        issue_provider: IssueProvider | None = None,
        code_reviewer: CodeReviewer | None = None,
        gate_checker: GateChecker | None = None,
        telemetry_provider: TelemetryProvider | None = None,
        # Centralized configuration (optional, uses from_env() if not provided)
        config: MalaConfig | None = None,
    ):
        self.repo_path = repo_path.resolve()
        self.max_agents = max_agents
        # Use default timeout if not specified to protect against hung MCP subprocesses
        effective_timeout = (
            timeout_minutes if timeout_minutes else DEFAULT_AGENT_TIMEOUT_MINUTES
        )
        self.timeout_seconds = effective_timeout * 60
        self.max_issues = max_issues
        self.epic_id = epic_id
        self.only_ids = only_ids

        # Use config if provided, otherwise load from environment
        # Store config for later use (e.g., passing API keys to get_mcp_servers)
        self._config = (
            config if config is not None else MalaConfig.from_env(validate=False)
        )

        # Derive feature flags from config if not explicitly provided
        # This allows explicit parameters to override config values
        if braintrust_enabled is not None:
            self.braintrust_enabled = braintrust_enabled
        else:
            self.braintrust_enabled = self._config.braintrust_enabled

        if morph_enabled is not None:
            self.morph_enabled = morph_enabled
        else:
            self.morph_enabled = self._config.morph_enabled

        self.max_gate_retries = max_gate_retries
        self.max_review_retries = max_review_retries
        self.disable_validations = disable_validations
        self.coverage_threshold = coverage_threshold
        self.prioritize_wip = prioritize_wip
        self.focus = focus
        self.cli_args = cli_args
        self.codex_thinking_mode = codex_thinking_mode

        self.active_tasks: dict[str, asyncio.Task] = {}
        self.agent_ids: dict[str, str] = {}
        self.completed: list[IssueResult] = []
        self.failed_issues: set[str] = set()

        # Track session log paths for quality gate (issue_id -> log_path)
        self.session_log_paths: dict[str, Path] = {}

        # Track codex review session log paths (issue_id -> log_path)
        self.codex_review_log_paths: dict[str, str] = {}

        # Initialize pipeline stage implementations (use defaults if not provided)
        # GateChecker: QualityGate for post-run validation
        self.quality_gate: GateChecker = (
            QualityGate(self.repo_path) if gate_checker is None else gate_checker
        )

        # IssueProvider: BeadsClient for issue tracking
        def log_warning(msg: str) -> None:
            log("⚠", msg, Colors.YELLOW)

        self.beads: IssueProvider = (
            BeadsClient(self.repo_path, log_warning=log_warning)
            if issue_provider is None
            else issue_provider
        )

        # CodeReviewer: DefaultCodeReviewer wrapping run_codex_review
        self.code_reviewer: CodeReviewer = (
            DefaultCodeReviewer() if code_reviewer is None else code_reviewer
        )

        # TelemetryProvider: BraintrustProvider when enabled, else NullTelemetryProvider
        if telemetry_provider is not None:
            self.telemetry_provider: TelemetryProvider = telemetry_provider
        elif self.braintrust_enabled:
            self.telemetry_provider = BraintrustProvider()
        else:
            self.telemetry_provider = NullTelemetryProvider()

        # Cached per-issue validation spec (built once at run start)
        self.per_issue_spec: ValidationSpec | None = None

    def _cleanup_agent_locks(self, agent_id: str) -> None:
        """Remove locks held by a specific agent (crash/timeout cleanup)."""
        cleaned = cleanup_agent_locks(agent_id)
        if cleaned:
            log(
                "🧹",
                f"Cleaned {cleaned} locks for {agent_id[:8]}",
                Colors.MUTED,
            )

    def _run_quality_gate_sync(
        self,
        issue_id: str,
        log_path: Path,
        retry_state: RetryState,
    ) -> tuple[GateResult, int]:
        """Synchronous quality gate check (blocking I/O).

        This is the blocking implementation that gets run via asyncio.to_thread.
        """
        # Use cached per-issue validation spec (or build if not set)
        if self.per_issue_spec is None:
            self.per_issue_spec = build_validation_spec(
                scope=ValidationScope.PER_ISSUE,
                disable_validations=self.disable_validations,
                coverage_threshold=self.coverage_threshold,
                repo_path=self.repo_path,
            )
        spec = self.per_issue_spec

        # Use check_with_resolution which handles no-op/obsolete cases
        gate_result = self.quality_gate.check_with_resolution(
            issue_id=issue_id,
            log_path=log_path,
            baseline_timestamp=retry_state.baseline_timestamp,
            log_offset=retry_state.log_offset,
            spec=spec,
        )

        # Calculate new offset for next attempt (lightweight - just get file end position)
        new_offset = self.quality_gate.get_log_end_offset(
            log_path, start_offset=retry_state.log_offset
        )

        # Check for no-progress condition (same commit, no new evidence)
        # Only check if this is a retry and the gate didn't already pass
        if retry_state.gate_attempt > 1 and not gate_result.passed:
            no_progress = self.quality_gate.check_no_progress(
                log_path,
                retry_state.log_offset,
                retry_state.previous_commit_hash,
                gate_result.commit_hash,
                spec=spec,  # Pass spec for spec-driven evidence detection
            )
            if no_progress:
                # Add no-progress to failure reasons and set flag
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

        return (gate_result, new_offset)

    async def _run_quality_gate_async(
        self,
        issue_id: str,
        log_path: Path,
        retry_state: RetryState,
    ) -> tuple[GateResult, int]:
        """Run quality gate check asynchronously via to_thread.

        Wraps the blocking _run_quality_gate_sync to avoid stalling the event loop.
        This allows the orchestrator to service other agents while a gate runs.

        Args:
            issue_id: The issue being checked.
            log_path: Path to the session log file.
            retry_state: Current retry state for this issue.

        Returns:
            Tuple of (GateResult, new_log_offset).
        """
        return await asyncio.to_thread(
            self._run_quality_gate_sync, issue_id, log_path, retry_state
        )

    async def _run_run_level_validation(
        self,
        run_metadata: RunMetadata,
    ) -> bool:
        """Run Gate 4 validation after all issues complete.

        This runs validation with RUN_LEVEL scope, which includes E2E tests.
        On failure, spawns a fixer agent and retries up to max_gate_retries times.

        Args:
            run_metadata: Run metadata tracker to record validation results.

        Returns:
            True if validation passed (or was skipped), False if failed after retries.
        """
        # Check if run-level validation is disabled
        if "run-level-validate" in (self.disable_validations or set()):
            log("◐", "Run-level validation disabled", Colors.MUTED)
            return True

        # Get current HEAD commit
        commit_hash = await get_git_commit_async(self.repo_path)
        if not commit_hash:
            log(
                "⚠", "Could not get HEAD commit for run-level validation", Colors.YELLOW
            )
            return True  # Don't fail the run if we can't get the commit

        log("◐", "Running Gate 4 (run-level validation)...", Colors.CYAN)

        # Build run-level validation spec
        spec = build_validation_spec(
            scope=ValidationScope.RUN_LEVEL,
            disable_validations=self.disable_validations,
            coverage_threshold=self.coverage_threshold,
            repo_path=self.repo_path,
        )

        # Build validation context
        context = ValidationContext(
            issue_id=None,  # Run-level, no specific issue
            repo_path=self.repo_path,
            commit_hash=commit_hash,
            changed_files=[],  # Not needed for run-level
            scope=ValidationScope.RUN_LEVEL,
        )

        # Create validation runner (uses SpecValidationRunner directly)
        runner = SpecValidationRunner(self.repo_path)

        # Retry loop with fixer agent
        for attempt in range(1, self.max_gate_retries + 1):
            log(
                "◐",
                f"Gate 4 attempt {attempt}/{self.max_gate_retries}",
                Colors.CYAN,
            )

            # Run validation
            try:
                result = await runner.run_spec(spec, context)
            except Exception as e:
                log("⚠", f"Validation runner error: {e}", Colors.YELLOW)
                result = None

            if result and result.passed:
                log("✓", "Gate 4 passed", Colors.GREEN)
                # Record in metadata
                # Derive e2e_passed from actual E2E execution result:
                # - None if E2E was not executed (disabled or skipped)
                # - True if E2E was executed and passed
                # - False if E2E was executed and failed
                e2e_passed: bool | None = None
                if result.e2e_result is not None:
                    if result.e2e_result.status == E2EStatus.SKIPPED:
                        e2e_passed = None
                    else:
                        e2e_passed = result.e2e_result.passed
                meta_result = MetaValidationResult(
                    passed=True,
                    commands_run=[s.name for s in result.steps],
                    commands_failed=[],
                    artifacts=result.artifacts,
                    coverage_percent=result.coverage_result.percent
                    if result.coverage_result
                    else None,
                    e2e_passed=e2e_passed,
                )
                run_metadata.record_run_validation(meta_result)
                return True

            # Validation failed - build failure output for fixer
            failure_output = self._build_validation_failure_output(result)

            # Record failure in metadata (will be overwritten on success)
            if result:
                # Derive e2e_passed from actual E2E execution result
                e2e_passed_fail: bool | None = None
                if result.e2e_result is not None:
                    if result.e2e_result.status == E2EStatus.SKIPPED:
                        e2e_passed_fail = None
                    else:
                        e2e_passed_fail = result.e2e_result.passed
                meta_result = MetaValidationResult(
                    passed=False,
                    commands_run=[s.name for s in result.steps],
                    commands_failed=[s.name for s in result.steps if not s.ok],
                    artifacts=result.artifacts,
                    coverage_percent=result.coverage_result.percent
                    if result.coverage_result
                    else None,
                    e2e_passed=e2e_passed_fail,
                )
                run_metadata.record_run_validation(meta_result)

            # Check if we have retries left
            if attempt >= self.max_gate_retries:
                log(
                    "✗",
                    f"Gate 4 failed after {attempt} attempts",
                    Colors.RED,
                )
                return False

            # Spawn fixer agent
            log(
                "↻",
                f"Spawning fixer agent (attempt {attempt}/{self.max_gate_retries})",
                Colors.YELLOW,
            )

            fixer_success = await self._run_fixer_agent(
                failure_output=failure_output,
                attempt=attempt,
            )

            if not fixer_success:
                log("⚠", "Fixer agent did not complete successfully", Colors.YELLOW)
                # Continue to retry validation anyway - fixer may have made progress

            # Update commit hash for next validation attempt
            new_commit = await get_git_commit_async(self.repo_path)
            if new_commit and new_commit != commit_hash:
                commit_hash = new_commit
                context = ValidationContext(
                    issue_id=None,
                    repo_path=self.repo_path,
                    commit_hash=commit_hash,
                    changed_files=[],
                    scope=ValidationScope.RUN_LEVEL,
                )

        return False

    def _build_validation_failure_output(
        self, result: "ValidationResult | None"
    ) -> str:
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
                    lines.append(step.stderr_tail[:2000])  # Truncate
                    lines.append("```")
                elif step.stdout_tail:
                    lines.append("```")
                    lines.append(step.stdout_tail[:2000])  # Truncate
                    lines.append("```")

        return (
            "\n".join(lines) if lines else "Validation failed (no details available)."
        )

    async def _run_fixer_agent(
        self,
        failure_output: str,
        attempt: int,
    ) -> bool:
        """Spawn a fixer agent to address run-level validation failures.

        Args:
            failure_output: Human-readable description of what failed.
            attempt: Current attempt number.

        Returns:
            True if fixer agent completed successfully, False otherwise.
        """
        agent_id = f"fixer-{uuid.uuid4().hex[:8]}"

        prompt = _get_fixer_prompt().format(
            attempt=attempt,
            max_attempts=self.max_gate_retries,
            failure_output=failure_output,
        )

        # Set up environment
        agent_env = {
            **os.environ,
            "PATH": f"{SCRIPTS_DIR}:{os.environ.get('PATH', '')}",
            "LOCK_DIR": str(get_lock_dir()),
            "AGENT_ID": agent_id,
            "REPO_NAMESPACE": str(self.repo_path),
            # MCP server startup timeout (5 minutes) to prevent hung subprocesses
            "MCP_TIMEOUT": "300000",
        }

        # Build hooks - similar to implementer but without lock enforcement
        pre_tool_hooks: list = [block_dangerous_commands]
        if self.morph_enabled:
            pre_tool_hooks.append(block_morph_replaced_tools)

        options = ClaudeAgentOptions(
            cwd=str(self.repo_path),
            permission_mode="bypassPermissions",
            model="opus",
            system_prompt={"type": "preset", "preset": "claude_code"},
            setting_sources=["project", "user"],
            mcp_servers=get_mcp_servers(
                self.repo_path,
                morph_api_key=self._config.morph_api_key,
                morph_enabled=self.morph_enabled,
            ),
            disallowed_tools=MORPH_DISALLOWED_TOOLS if self.morph_enabled else ["Edit"],
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

        try:
            async with asyncio.timeout(self.timeout_seconds):
                async with ClaudeSDKClient(options=options) as client:
                    await client.query(prompt)

                    async for message in client.receive_response():
                        # Log progress
                        if isinstance(message, AssistantMessage):
                            for block in message.content:
                                if isinstance(block, TextBlock):
                                    log_agent_text(block.text, f"fixer-{attempt}")
                                elif isinstance(block, ToolUseBlock):
                                    log_tool(
                                        block.name,
                                        agent_id=f"fixer-{attempt}",
                                        arguments=block.input,
                                    )
                        elif isinstance(message, ResultMessage):
                            log(
                                "◐",
                                f"Fixer completed: {truncate_text(message.result or '', 50)}",
                                Colors.MUTED,
                            )

            return True

        except TimeoutError:
            log("⚠", "Fixer agent timed out", Colors.YELLOW)
            return False
        except Exception as e:
            log("⚠", f"Fixer agent error: {e}", Colors.YELLOW)
            return False
        finally:
            self._cleanup_agent_locks(agent_id)

    async def run_implementer(self, issue_id: str) -> IssueResult:
        """Run implementer agent for a single issue with gate retry support.

        Uses ImplementerLifecycle to drive policy decisions. The orchestrator
        handles I/O (SDK calls, logs, hooks) based on Effect returns.
        """
        agent_id = f"{issue_id}-{uuid.uuid4().hex[:8]}"
        self.agent_ids[issue_id] = agent_id

        # Fetch issue description for scope verification in codex review
        issue_description = await self.beads.get_issue_description_async(issue_id)

        # Derive baseline commit for cumulative diff review:
        # 1. If git history has commits for this issue → use parent of first commit
        # 2. If no commits yet → capture current HEAD (fresh issue)
        baseline_commit = await get_baseline_for_issue(self.repo_path, issue_id)
        if baseline_commit is None:
            # Fresh issue - no prior commits, use current HEAD as baseline
            baseline_commit = await get_git_commit_async(self.repo_path)

        # Initialize lifecycle with config derived from orchestrator settings
        codex_review_enabled = "codex-review" not in (self.disable_validations or set())
        lifecycle_config = LifecycleConfig(
            max_gate_retries=self.max_gate_retries,
            max_review_retries=self.max_review_retries,
            codex_review_enabled=codex_review_enabled,
        )
        lifecycle = ImplementerLifecycle(lifecycle_config)
        lifecycle_ctx = LifecycleContext()

        # Set baseline timestamp in lifecycle context
        lifecycle_ctx.retry_state.baseline_timestamp = int(time.time())

        # Track baseline_commit_hash separately (not in lifecycle's RetryState)
        # This is used for cumulative diff review in Codex
        baseline_commit_hash = baseline_commit

        # Claude session ID will be captured from ResultMessage
        claude_session_id: str | None = None

        prompt = _get_implementer_prompt().format(
            issue_id=issue_id,
            repo_path=self.repo_path,
            lock_dir=get_lock_dir(),
            scripts_dir=SCRIPTS_DIR,
            agent_id=agent_id,
        )

        final_result = ""
        start_time = asyncio.get_event_loop().time()
        success = False
        gate_result: GateResult | None = None

        # Set up environment: inherit os.environ, then override with lock scripts/vars
        agent_env = {
            **os.environ,
            "PATH": f"{SCRIPTS_DIR}:{os.environ.get('PATH', '')}",
            "LOCK_DIR": str(get_lock_dir()),
            "AGENT_ID": agent_id,
            "REPO_NAMESPACE": str(self.repo_path),
            # MCP server startup timeout (5 minutes) to prevent hung subprocesses
            "MCP_TIMEOUT": "300000",
        }

        # Build hooks list - always include dangerous commands and lock enforcement
        pre_tool_hooks: list = [
            block_dangerous_commands,
            make_lock_enforcement_hook(agent_id, str(self.repo_path)),
        ]
        # Add Morph-related hook only when Morph is enabled
        if self.morph_enabled:
            pre_tool_hooks.insert(1, block_morph_replaced_tools)

        options = ClaudeAgentOptions(
            cwd=str(self.repo_path),
            permission_mode="bypassPermissions",
            model="opus",
            system_prompt={"type": "preset", "preset": "claude_code"},
            setting_sources=["project", "user"],
            mcp_servers=get_mcp_servers(
                self.repo_path,
                morph_api_key=self._config.morph_api_key,
                morph_enabled=self.morph_enabled,
            ),
            # Always block Edit since lock enforcement doesn't cover it.
            # When morph enabled, also block Grep (replaced by warpgrep).
            disallowed_tools=MORPH_DISALLOWED_TOOLS if self.morph_enabled else ["Edit"],
            env=agent_env,
            hooks={
                "PreToolUse": [
                    HookMatcher(
                        matcher=None,
                        hooks=pre_tool_hooks,  # type: ignore[arg-type]
                    )
                ],
                "Stop": [HookMatcher(matcher=None, hooks=[make_stop_hook(agent_id)])],  # type: ignore[arg-type]
            },
        )

        # Telemetry tracing context (uses injected provider)
        tracer = self.telemetry_provider.create_span(
            issue_id,
            metadata={
                "agent_id": agent_id,
                "mala_version": __version__,
                "project_dir": self.repo_path.name,
                "git_branch": await get_git_branch_async(self.repo_path),
                "git_commit": await get_git_commit_async(self.repo_path),
                "timeout_seconds": self.timeout_seconds,
            },
        )

        try:
            with tracer:
                tracer.log_input(prompt)

                try:
                    async with asyncio.timeout(self.timeout_seconds):
                        async with ClaudeSDKClient(options=options) as client:
                            # Start lifecycle
                            lifecycle.start()
                            log_verbose(
                                "→",
                                f"State: {lifecycle.state.name}",
                                agent_id=issue_id,
                            )

                            # Initial query
                            await client.query(prompt)

                            # Main loop - driven by lifecycle transitions
                            while not lifecycle.is_terminal:
                                # Process messages from SDK
                                async for message in client.receive_response():
                                    # Log to Braintrust
                                    tracer.log_message(message)

                                    # Console output
                                    if isinstance(message, AssistantMessage):
                                        for block in message.content:
                                            if isinstance(block, TextBlock):
                                                log_agent_text(block.text, issue_id)
                                            elif isinstance(block, ToolUseBlock):
                                                log_tool(
                                                    block.name,
                                                    agent_id=issue_id,
                                                    arguments=block.input,
                                                )

                                    elif isinstance(message, ResultMessage):
                                        claude_session_id = message.session_id
                                        lifecycle_ctx.session_id = claude_session_id
                                        lifecycle_ctx.final_result = (
                                            message.result or ""
                                        )

                                # Messages complete - transition lifecycle
                                result = lifecycle.on_messages_complete(
                                    lifecycle_ctx,
                                    has_session_id=bool(claude_session_id),
                                )
                                log_verbose(
                                    "→",
                                    f"State: {lifecycle.state.name}",
                                    agent_id=issue_id,
                                )

                                if result.effect == Effect.COMPLETE_FAILURE:
                                    final_result = lifecycle_ctx.final_result
                                    break

                                # Handle WAIT_FOR_LOG effect
                                if result.effect == Effect.WAIT_FOR_LOG:
                                    log_path = get_claude_log_path(
                                        self.repo_path,
                                        claude_session_id,  # type: ignore[arg-type]
                                    )
                                    self.session_log_paths[issue_id] = log_path

                                    log(
                                        "◌",
                                        "Waiting for session log...",
                                        Colors.MUTED,
                                        agent_id=issue_id,
                                    )

                                    # Wait for log file with bounded timeout
                                    wait_elapsed = 0.0
                                    while not log_path.exists():
                                        if wait_elapsed >= LOG_FILE_WAIT_TIMEOUT:
                                            result = lifecycle.on_log_timeout(
                                                lifecycle_ctx, str(log_path)
                                            )
                                            log(
                                                "⚠",
                                                f"Log file not found: {log_path.name}",
                                                Colors.YELLOW,
                                                agent_id=issue_id,
                                            )
                                            break
                                        await asyncio.sleep(LOG_FILE_POLL_INTERVAL)
                                        wait_elapsed += LOG_FILE_POLL_INTERVAL

                                    if lifecycle.state == LifecycleState.FAILED:
                                        final_result = lifecycle_ctx.final_result
                                        break

                                    # Log file found - transition to running gate
                                    if log_path.exists():
                                        log(
                                            "✓",
                                            "Log file ready",
                                            Colors.GREEN,
                                            agent_id=issue_id,
                                        )
                                    result = lifecycle.on_log_ready(lifecycle_ctx)

                                # Handle RUN_GATE effect
                                if result.effect == Effect.RUN_GATE:
                                    log(
                                        "◐",
                                        "Running quality gate...",
                                        Colors.CYAN,
                                        agent_id=issue_id,
                                    )

                                    log_path = self.session_log_paths[issue_id]
                                    (
                                        gate_result,
                                        new_offset,
                                    ) = await self._run_quality_gate_async(
                                        issue_id, log_path, lifecycle_ctx.retry_state
                                    )

                                    # Transition based on gate result
                                    result = lifecycle.on_gate_result(
                                        lifecycle_ctx, gate_result, new_offset
                                    )

                                    if result.effect == Effect.COMPLETE_SUCCESS:
                                        log(
                                            "✓",
                                            "Quality gate passed",
                                            Colors.GREEN,
                                            agent_id=issue_id,
                                        )
                                        success = True
                                        final_result = lifecycle_ctx.final_result
                                        break

                                    if result.effect == Effect.COMPLETE_FAILURE:
                                        reasons = "; ".join(gate_result.failure_reasons)
                                        log(
                                            "✗",
                                            f"Quality gate failed: {reasons}",
                                            Colors.RED,
                                            agent_id=issue_id,
                                        )
                                        final_result = lifecycle_ctx.final_result
                                        break

                                    if result.effect == Effect.SEND_GATE_RETRY:
                                        # Log retry attempt
                                        log(
                                            "↻",
                                            f"Gate retry {lifecycle_ctx.retry_state.gate_attempt}/{self.max_gate_retries}",
                                            Colors.YELLOW,
                                            agent_id=issue_id,
                                        )

                                        # Build follow-up prompt with failure context
                                        failure_text = "\n".join(
                                            f"- {r}"
                                            for r in gate_result.failure_reasons
                                        )
                                        followup = _get_gate_followup_prompt().format(
                                            attempt=lifecycle_ctx.retry_state.gate_attempt,
                                            max_attempts=self.max_gate_retries,
                                            failure_reasons=failure_text,
                                            issue_id=issue_id,
                                        )

                                        # Resume session with follow-up prompt
                                        assert claude_session_id is not None
                                        await client.query(
                                            followup,
                                            session_id=claude_session_id,
                                        )
                                        continue

                                # Handle RUN_REVIEW effect
                                if result.effect == Effect.RUN_REVIEW:
                                    # Check for no-progress on review retries BEFORE running review
                                    # For review retries, only check commit and working tree changes
                                    # (not validation evidence - that's for gate retries)
                                    # This prevents burning a review attempt when agent made no progress
                                    no_progress = False
                                    if lifecycle_ctx.retry_state.review_attempt > 1:
                                        log_path = self.session_log_paths[issue_id]
                                        current_commit = (
                                            lifecycle_ctx.last_gate_result.commit_hash
                                            if lifecycle_ctx.last_gate_result
                                            else None
                                        )
                                        no_progress = self.quality_gate.check_no_progress(
                                            log_path,
                                            lifecycle_ctx.retry_state.log_offset,
                                            lifecycle_ctx.retry_state.previous_commit_hash,
                                            current_commit,
                                            spec=self.per_issue_spec,
                                            check_validation_evidence=False,
                                        )
                                        if no_progress:
                                            # Fail fast without running review
                                            log(
                                                "✗",
                                                "Review skipped: No progress (commit unchanged, no working tree changes)",
                                                Colors.RED,
                                                agent_id=issue_id,
                                            )
                                            # Create a synthetic failed review result
                                            synthetic_result = CodexReviewResult(
                                                passed=False,
                                                issues=[],
                                                parse_error=None,
                                            )
                                            new_offset = self.quality_gate.get_log_end_offset(
                                                log_path,
                                                start_offset=lifecycle_ctx.retry_state.log_offset,
                                            )
                                            result = lifecycle.on_review_result(
                                                lifecycle_ctx,
                                                synthetic_result,
                                                new_offset,
                                                no_progress=True,
                                            )
                                            final_result = lifecycle_ctx.final_result
                                            break

                                    log(
                                        "◐",
                                        f"Running Codex review (attempt {lifecycle_ctx.retry_state.review_attempt}/{self.max_review_retries})",
                                        Colors.CYAN,
                                        agent_id=issue_id,
                                    )

                                    # Use current HEAD for review instead of the agent's specific commit.
                                    # This ensures that if another agent made a subsequent commit that
                                    # fixes issues from this agent's commit, Codex reviews the clean
                                    # cumulative diff (baseline..HEAD) rather than the stale diff
                                    # (baseline..agent_commit) which would still show the old issues.
                                    current_head = await get_git_commit_async(
                                        self.repo_path
                                    )
                                    review_result = await self.code_reviewer(
                                        self.repo_path,
                                        current_head,
                                        max_retries=2,  # JSON parse retries
                                        issue_description=issue_description,
                                        baseline_commit=baseline_commit_hash,
                                        capture_session_log=is_verbose_enabled(),
                                        thinking_mode=self.codex_thinking_mode,
                                    )

                                    # Store codex review log path if captured
                                    if review_result.session_log_path:
                                        self.codex_review_log_paths[issue_id] = (
                                            review_result.session_log_path
                                        )

                                    log_path = self.session_log_paths[issue_id]
                                    new_offset = self.quality_gate.get_log_end_offset(
                                        log_path,
                                        start_offset=lifecycle_ctx.retry_state.log_offset,
                                    )

                                    # Transition based on review result
                                    result = lifecycle.on_review_result(
                                        lifecycle_ctx,
                                        review_result,
                                        new_offset,
                                    )

                                    if result.effect == Effect.COMPLETE_SUCCESS:
                                        log(
                                            "✓",
                                            "Codex review passed",
                                            Colors.GREEN,
                                            agent_id=issue_id,
                                        )
                                        success = True
                                        final_result = lifecycle_ctx.final_result
                                        break

                                    if result.effect == Effect.COMPLETE_FAILURE:
                                        final_result = lifecycle_ctx.final_result
                                        break

                                    if result.effect == Effect.SEND_REVIEW_RETRY:
                                        # Log review failure
                                        if review_result.parse_error:
                                            log(
                                                "⚠",
                                                f"Codex review parse error: {review_result.parse_error}",
                                                Colors.YELLOW,
                                                agent_id=issue_id,
                                            )
                                        else:
                                            error_count = sum(
                                                1
                                                for i in review_result.issues
                                                if i.severity == "error"
                                            )
                                            log(
                                                "↻",
                                                f"Review retry {lifecycle_ctx.retry_state.review_attempt}/{self.max_review_retries} ({error_count} errors)",
                                                Colors.YELLOW,
                                                agent_id=issue_id,
                                            )

                                        # Build follow-up prompt with review issues
                                        review_issues_text = (
                                            format_review_issues(review_result.issues)
                                            if review_result.issues
                                            else review_result.parse_error
                                            or "Unknown review failure"
                                        )
                                        followup = _get_review_followup_prompt().format(
                                            attempt=lifecycle_ctx.retry_state.review_attempt,
                                            max_attempts=self.max_review_retries,
                                            review_issues=review_issues_text,
                                            issue_id=issue_id,
                                        )

                                        # Resume session with follow-up prompt
                                        assert claude_session_id is not None
                                        await client.query(
                                            followup,
                                            session_id=claude_session_id,
                                        )
                                        continue

                    tracer.set_success(success)

                except TimeoutError:
                    timeout_mins = (
                        self.timeout_seconds // 60 if self.timeout_seconds else 0
                    )
                    lifecycle.on_timeout(lifecycle_ctx, timeout_mins)
                    final_result = lifecycle_ctx.final_result
                    tracer.set_error(final_result)
                    self._cleanup_agent_locks(agent_id)

                except Exception as e:
                    lifecycle.on_error(lifecycle_ctx, e)
                    final_result = lifecycle_ctx.final_result
                    tracer.set_error(final_result)
                    self._cleanup_agent_locks(agent_id)

        finally:
            duration = asyncio.get_event_loop().time() - start_time

            # Ensure locks are cleaned
            self._cleanup_agent_locks(agent_id)
            self.agent_ids.pop(issue_id, None)

        return IssueResult(
            issue_id=issue_id,
            agent_id=agent_id,
            success=lifecycle_ctx.success,
            summary=final_result,
            duration_seconds=duration,
            session_id=claude_session_id,
            gate_attempts=lifecycle_ctx.retry_state.gate_attempt,
            review_attempts=lifecycle_ctx.retry_state.review_attempt,
            resolution=lifecycle_ctx.resolution,
        )

    async def spawn_agent(self, issue_id: str) -> bool:
        """Spawn a new agent task for an issue. Returns True if spawned."""
        if not await self.beads.claim_async(issue_id):
            self.failed_issues.add(issue_id)
            log("⚠", f"Failed to claim {issue_id}", Colors.YELLOW, agent_id=issue_id)
            return False

        task = asyncio.create_task(self.run_implementer(issue_id))
        self.active_tasks[issue_id] = task
        log("▶", "Agent started", Colors.BLUE, agent_id=issue_id)
        return True

    async def run(self) -> tuple[int, int]:
        """Main orchestration loop. Returns (success_count, total_count).

        This is the async entry point - use this when you're already in an async context
        (e.g., inside an async function, using asyncio.run(), or in a framework that
        manages the event loop).

        For sync usage, see run_sync().
        """
        print()
        log("●", "mala orchestrator", Colors.MAGENTA)
        log("◐", f"repo: {self.repo_path}", Colors.MUTED)
        limit_str = str(self.max_issues) if self.max_issues is not None else "unlimited"
        agents_str = (
            str(self.max_agents) if self.max_agents is not None else "unlimited"
        )
        timeout_str = (
            f"{self.timeout_seconds // 60}m" if self.timeout_seconds else "none"
        )
        log(
            "◐",
            f"max-agents: {agents_str}, timeout: {timeout_str}, max-issues: {limit_str}",
            Colors.MUTED,
        )
        log(
            "◐",
            f"gate-retries: {self.max_gate_retries}",
            Colors.MUTED,
        )
        codex_review_enabled = "codex-review" not in (self.disable_validations or set())
        if codex_review_enabled:
            thinking_mode_str = (
                f", thinking: {self.codex_thinking_mode}"
                if self.codex_thinking_mode
                else ""
            )
            log(
                "◐",
                f"codex-review: enabled (max-retries: {self.max_review_retries}{thinking_mode_str})",
                Colors.CYAN,
            )

        # Report epic filter
        if self.epic_id:
            log("◐", f"epic filter: {self.epic_id}", Colors.CYAN)

        # Report only_ids filter
        if self.only_ids:
            log(
                "◐",
                f"only processing: {', '.join(sorted(self.only_ids))}",
                Colors.CYAN,
            )

        # Report WIP prioritization
        if self.prioritize_wip:
            log(
                "◐",
                "wip: prioritizing in_progress issues",
                Colors.CYAN,
            )

        # Report Braintrust status
        if self.braintrust_enabled:
            log(
                "◐",
                "braintrust: enabled (LLM spans auto-traced)",
                Colors.CYAN,
            )
        else:
            log(
                "◐",
                f"braintrust: disabled (add BRAINTRUST_API_KEY to {USER_CONFIG_DIR}/.env)",
                Colors.MUTED,
            )

        # Report Morph MCP status
        if self.morph_enabled:
            log(
                "◐",
                "morph: enabled (edit_file, warpgrep_codebase_search)",
                Colors.CYAN,
            )
            log(
                "◐",
                f"morph: blocked tools: {', '.join(MORPH_DISALLOWED_TOOLS)}",
                Colors.MUTED,
            )
        else:
            log(
                "◐",
                "morph: disabled (MORPH_API_KEY not set)",
                Colors.MUTED,
            )

        # Log CLI args that affect run behavior
        if self.cli_args:
            cli_parts = []
            if self.cli_args.get("disable_validations"):
                cli_parts.append(
                    f"--disable-validations={self.cli_args['disable_validations']}"
                )
            if self.cli_args.get("coverage_threshold") is not None:
                cli_parts.append(
                    f"--coverage-threshold={self.cli_args['coverage_threshold']}"
                )
            if self.cli_args.get("wip"):
                cli_parts.append("--wip")
            if self.cli_args.get("max_issues") is not None:
                cli_parts.append(f"--max-issues={self.cli_args['max_issues']}")
            if self.cli_args.get("max_gate_retries") is not None:
                cli_parts.append(
                    f"--max-gate-retries={self.cli_args['max_gate_retries']}"
                )
            if self.cli_args.get("max_review_retries") is not None:
                cli_parts.append(
                    f"--max-review-retries={self.cli_args['max_review_retries']}"
                )
            if self.cli_args.get("braintrust"):
                cli_parts.append("--braintrust")
            if cli_parts:
                log("◐", f"cli: {' '.join(cli_parts)}", Colors.MUTED)

        print()

        # Setup directories
        get_lock_dir().mkdir(parents=True, exist_ok=True)
        get_runs_dir().mkdir(parents=True, exist_ok=True)

        # Create run metadata tracker
        run_config = RunConfig(
            max_agents=self.max_agents,
            timeout_minutes=self.timeout_seconds // 60
            if self.timeout_seconds
            else None,
            max_issues=self.max_issues,
            epic_id=self.epic_id,
            only_ids=list(self.only_ids) if self.only_ids else None,
            braintrust_enabled=self.braintrust_enabled,
            max_gate_retries=self.max_gate_retries,
            max_review_retries=self.max_review_retries,
            codex_review="codex-review" not in (self.disable_validations or set()),
            cli_args=self.cli_args,
        )
        run_metadata = RunMetadata(self.repo_path, run_config, __version__)

        # Build per-issue validation spec once (reused throughout the run)
        self.per_issue_spec = build_validation_spec(
            scope=ValidationScope.PER_ISSUE,
            disable_validations=self.disable_validations,
            coverage_threshold=self.coverage_threshold,
            repo_path=self.repo_path,
        )

        # Write run marker to indicate this instance is running
        # This is used by `mala status` to detect running instances
        write_run_marker(
            run_id=run_metadata.run_id,
            repo_path=self.repo_path,
            max_agents=self.max_agents,
        )

        issues_spawned = 0

        try:
            while True:
                # Check if we've reached the issue limit
                limit_reached = (
                    self.max_issues is not None and issues_spawned >= self.max_issues
                )

                # Fill up to max_agents (unless limit reached)
                suppress_warn_ids = None
                if self.only_ids:
                    suppress_warn_ids = (
                        self.failed_issues
                        | set(self.active_tasks.keys())
                        | {result.issue_id for result in self.completed}
                    )
                ready = (
                    await self.beads.get_ready_async(
                        self.failed_issues,
                        epic_id=self.epic_id,
                        only_ids=self.only_ids,
                        suppress_warn_ids=suppress_warn_ids,
                        prioritize_wip=self.prioritize_wip,
                        focus=self.focus,
                    )
                    if not limit_reached
                    else []
                )

                if ready:
                    log("◌", f"Ready issues: {', '.join(ready)}", Colors.MUTED)

                while (
                    self.max_agents is None or len(self.active_tasks) < self.max_agents
                ) and ready:
                    issue_id = ready.pop(0)
                    if issue_id not in self.active_tasks:
                        if await self.spawn_agent(issue_id):
                            issues_spawned += 1
                            # Check limit after each spawn
                            if (
                                self.max_issues is not None
                                and issues_spawned >= self.max_issues
                            ):
                                break

                if not self.active_tasks:
                    if limit_reached:
                        log(
                            "○", f"Issue limit reached ({self.max_issues})", Colors.GRAY
                        )
                    elif not ready:
                        log("○", "No more issues to process", Colors.GRAY)
                    break

                # Wait for ANY task to complete
                log(
                    "◌",
                    f"Waiting for {len(self.active_tasks)} agent(s)...",
                    Colors.MUTED,
                )

                done, _ = await asyncio.wait(
                    self.active_tasks.values(),
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # Handle completed tasks
                for task in done:
                    for issue_id, t in list(self.active_tasks.items()):
                        if t is task:
                            try:
                                result = task.result()
                            except Exception as e:
                                result = IssueResult(
                                    issue_id=issue_id,
                                    agent_id=self.agent_ids.get(issue_id, "unknown"),
                                    success=False,
                                    summary=str(e),
                                )

                            # Quality gate already run in run_implementer for retry support
                            # Just handle the final result here
                            quality_gate_result: QualityGateResult | None = None
                            validation_result: MetaValidationResult | None = None
                            log_path = self.session_log_paths.get(issue_id)

                            if result.success and log_path and log_path.exists():
                                # Use cached per-issue validation spec
                                assert self.per_issue_spec is not None
                                spec = self.per_issue_spec
                                # Parse final evidence for metadata (full session) using spec-driven parsing
                                evidence = self.quality_gate.parse_validation_evidence_with_spec(
                                    log_path, spec
                                )
                                commit_result = self.quality_gate.check_commit_exists(
                                    issue_id
                                )
                                quality_gate_result = QualityGateResult(
                                    passed=True,
                                    evidence={
                                        "pytest_ran": evidence.pytest_ran,
                                        "ruff_check_ran": evidence.ruff_check_ran,
                                        "ruff_format_ran": evidence.ruff_format_ran,
                                        "ty_check_ran": evidence.ty_check_ran,
                                        "commit_found": commit_result.exists,
                                    },
                                    failure_reasons=[],
                                )

                                # Run SpecValidationRunner.run_spec after commit detection
                                if commit_result.exists and commit_result.commit_hash:
                                    try:
                                        # Reuse spec built above
                                        context = ValidationContext(
                                            issue_id=issue_id,
                                            repo_path=self.repo_path,
                                            commit_hash=commit_result.commit_hash,
                                            changed_files=[],  # Not needed for per-issue
                                            scope=ValidationScope.PER_ISSUE,
                                            log_path=log_path,
                                        )
                                        runner = SpecValidationRunner(self.repo_path)
                                        runner_result = await runner.run_spec(
                                            spec, context
                                        )
                                        # Convert runner result to metadata result
                                        validation_result = MetaValidationResult(
                                            passed=runner_result.passed,
                                            commands_run=[
                                                s.name for s in runner_result.steps
                                            ],
                                            commands_failed=[
                                                s.name
                                                for s in runner_result.steps
                                                if not s.ok
                                            ],
                                            artifacts=runner_result.artifacts,
                                            coverage_percent=runner_result.coverage_result.percent
                                            if runner_result.coverage_result
                                            else None,
                                        )
                                        log(
                                            "◐",
                                            f"Validation: {validation_result.passed}",
                                            Colors.CYAN
                                            if validation_result.passed
                                            else Colors.YELLOW,
                                            agent_id=issue_id,
                                        )
                                    except Exception as e:
                                        log(
                                            "⚠",
                                            f"Validation runner error: {e}",
                                            Colors.YELLOW,
                                            agent_id=issue_id,
                                        )
                                        # Still populate a failed validation result
                                        validation_result = MetaValidationResult(
                                            passed=False,
                                            commands_run=[],
                                            commands_failed=[],
                                        )

                                # Gate passed - close the issue
                                if await self.beads.close_async(issue_id):
                                    log(
                                        "◐",
                                        "Closed issue",
                                        Colors.MUTED,
                                        agent_id=issue_id,
                                    )
                                    # Check if this closes any parent epics
                                    if await self.beads.close_eligible_epics_async():
                                        log(
                                            "◐",
                                            "Auto-closed completed epic(s)",
                                            Colors.MUTED,
                                            agent_id=issue_id,
                                        )

                            elif not result.success and log_path and log_path.exists():
                                # Failed run - record evidence for metadata using cached spec
                                assert self.per_issue_spec is not None
                                spec = self.per_issue_spec
                                evidence = self.quality_gate.parse_validation_evidence_with_spec(
                                    log_path, spec
                                )
                                commit_result = self.quality_gate.check_commit_exists(
                                    issue_id
                                )
                                # Extract failure reasons from result summary
                                failure_reasons = []
                                if "Quality gate failed:" in result.summary:
                                    reasons_part = result.summary.replace(
                                        "Quality gate failed: ", ""
                                    )
                                    failure_reasons = [
                                        r.strip() for r in reasons_part.split(";")
                                    ]
                                quality_gate_result = QualityGateResult(
                                    passed=False,
                                    evidence={
                                        "pytest_ran": evidence.pytest_ran,
                                        "ruff_check_ran": evidence.ruff_check_ran,
                                        "ruff_format_ran": evidence.ruff_format_ran,
                                        "ty_check_ran": evidence.ty_check_ran,
                                        "commit_found": commit_result.exists,
                                    },
                                    failure_reasons=failure_reasons,
                                )

                            self.completed.append(result)
                            del self.active_tasks[issue_id]

                            # Record to run metadata
                            issue_run = IssueRun(
                                issue_id=result.issue_id,
                                agent_id=result.agent_id,
                                status="success" if result.success else "failed",
                                duration_seconds=result.duration_seconds,
                                session_id=result.session_id,
                                log_path=str(log_path) if log_path else None,
                                quality_gate=quality_gate_result,
                                error=result.summary if not result.success else None,
                                gate_attempts=result.gate_attempts,
                                review_attempts=result.review_attempts,
                                validation=validation_result,
                                resolution=result.resolution,
                                codex_review_log_path=self.codex_review_log_paths.get(
                                    issue_id
                                ),
                            )
                            run_metadata.record_issue(issue_run)

                            # Pop log paths after recording
                            self.session_log_paths.pop(issue_id, None)
                            self.codex_review_log_paths.pop(issue_id, None)

                            duration_str = f"{result.duration_seconds:.0f}s"
                            if result.success:
                                summary = truncate_text(result.summary, 50)
                                log(
                                    "✓",
                                    f"({duration_str}): {summary}",
                                    Colors.GREEN,
                                    agent_id=issue_id,
                                )
                            else:
                                log(
                                    "✗",
                                    f"({duration_str}): {result.summary}",
                                    Colors.RED,
                                    agent_id=issue_id,
                                )
                                self.failed_issues.add(issue_id)
                                # Mark needs-followup with log path
                                await self.beads.mark_needs_followup_async(
                                    issue_id, result.summary, log_path=log_path
                                )
                            break

        finally:
            # Final cleanup - only release locks owned by this run's agents
            released = release_run_locks(list(self.agent_ids.values()))
            if released:
                log("🧹", f"Released {released} remaining locks", Colors.MUTED)

            # Remove run marker to indicate this instance is no longer running
            remove_run_marker(run_metadata.run_id)

        # Calculate success_count before Gate 4
        success_count = sum(1 for r in self.completed if r.success)
        total = len(self.completed)

        # Run Gate 4 (run-level validation) after all issues complete
        # Only run if there were successful issues (something to validate)
        run_validation_passed = True
        if success_count > 0:
            run_validation_passed = await self._run_run_level_validation(run_metadata)

        # Summary
        print()

        if success_count == total and total > 0 and run_validation_passed:
            log("●", f"Completed: {success_count}/{total} issues", Colors.GREEN)
        elif success_count > 0:
            if not run_validation_passed:
                log(
                    "◐",
                    f"Completed: {success_count}/{total} issues (Gate 4 failed)",
                    Colors.YELLOW,
                )
            else:
                log("◐", f"Completed: {success_count}/{total} issues", Colors.YELLOW)
        elif total == 0:
            log("○", "No issues to process", Colors.GRAY)
        else:
            log("○", f"Completed: {success_count}/{total} issues", Colors.RED)

        # Commit .beads/issues.jsonl if there were successes
        if success_count > 0:
            if await self.beads.commit_issues_async():
                log("◐", "Committed .beads/issues.jsonl", Colors.MUTED)

        # Save run metadata
        if total > 0:
            metadata_path = run_metadata.save()
            log("◐", f"Run metadata: {metadata_path}", Colors.MUTED)

        print()
        # Return success count for CLI exit code logic
        # Run-level validation failure should cause non-zero exit
        # Also indicate if no issues were processed (total == 0 is a no-op, not failure)
        if not run_validation_passed:
            # Gate 4 failed - return 0 successes to trigger non-zero exit
            return (0, total)
        return (success_count, total)

    def run_sync(self) -> tuple[int, int]:
        """Synchronous wrapper for run(). Returns (success_count, total_count).

        Use this method when calling from synchronous code. It handles event loop
        creation and cleanup automatically.

        Raises:
            RuntimeError: If called from within an async context (e.g., inside an
                async function or when an event loop is already running). In that
                case, use `await orchestrator.run()` instead.

        Example usage::

            # From sync code (scripts, CLI, tests)
            orchestrator = MalaOrchestrator(repo_path=Path("."))
            success_count, total = orchestrator.run_sync()

            # From async code
            async def main():
                orchestrator = MalaOrchestrator(repo_path=Path("."))
                success_count, total = await orchestrator.run()

        Returns:
            Tuple of (success_count, total_count).
        """
        # Check if we're already in an async context
        try:
            asyncio.get_running_loop()
            # If we get here, there's a running event loop - can't use run_sync()
            raise RuntimeError(
                "run_sync() cannot be called from within an async context. "
                "Use 'await orchestrator.run()' instead."
            )
        except RuntimeError as e:
            # Check if it's our error or the "no running event loop" error
            if "run_sync() cannot be called" in str(e):
                raise
            # No running event loop - this is the expected case for sync usage

        # Create a new event loop and run
        return asyncio.run(self.run())
