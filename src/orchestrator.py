"""MalaOrchestrator: Orchestrates parallel issue processing using Claude Agent SDK."""

import asyncio
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

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
from .braintrust_integration import TracedAgentExecution
from .codex_review import run_codex_review, format_review_issues
from .git_utils import get_git_commit_async, get_git_branch_async
from .hooks import (
    MORPH_DISALLOWED_TOOLS,
    block_dangerous_commands,
    block_morph_replaced_tools,
    make_lock_enforcement_hook,
)
from .logging.console import Colors, log, log_tool, log_agent_text, truncate_text
from .logging.run_metadata import RunMetadata, RunConfig, IssueRun, QualityGateResult
from .quality_gate import QualityGate, GateResult
from .tools.env import USER_CONFIG_DIR, RUNS_DIR, SCRIPTS_DIR, get_claude_log_path
from .tools.locking import (
    LOCK_DIR,
    release_run_locks,
    cleanup_agent_locks,
    make_stop_hook,
)


# Version (from package metadata)
from importlib.metadata import version as pkg_version

__version__ = pkg_version("mala")

# Load implementer prompt from file
PROMPT_FILE = Path(__file__).parent / "prompts" / "implementer_prompt.md"
IMPLEMENTER_PROMPT_TEMPLATE = PROMPT_FILE.read_text()

# Follow-up prompt for gate failures
GATE_FOLLOWUP_PROMPT = """## Quality Gate Failed (Attempt {attempt}/{max_attempts})

The quality gate check failed with the following issues:
{failure_reasons}

**Required actions:**
1. Fix the issues listed above
2. Re-run the full validation suite:
   - `uv sync`
   - `uv run pytest`
   - `uvx ruff check .`
   - `uvx ruff format .`
   - `uvx ty check`
3. Commit your changes with message: `bd-{issue_id}: <description>`

Note: The orchestrator requires NEW validation evidence - re-run all validations even if you ran them before.
"""

# Follow-up prompt for Codex review failures
CODEX_REVIEW_FOLLOWUP_PROMPT = """## Codex Review Failed (Attempt {attempt}/{max_attempts})

The automated code review found the following issues:

{review_issues}

**Required actions:**
1. Fix ALL issues marked as ERROR above
2. Optionally address warnings if appropriate
3. Re-run the full validation suite:
   - `uv sync`
   - `uv run pytest`
   - `uvx ruff check .`
   - `uvx ruff format .`
   - `uvx ty check`
4. Commit your changes with message: `bd-{issue_id}: <description>`

Note: The orchestrator will re-run both the quality gate and Codex review after your fixes.
"""

# Bounded wait for log file (seconds)
LOG_FILE_WAIT_TIMEOUT = 30
LOG_FILE_POLL_INTERVAL = 0.5


def get_mcp_servers(repo_path: Path) -> dict:
    """Get MCP servers configuration for agents."""
    return {
        "morphllm": {
            "command": "npx",
            "args": ["-y", "@morphllm/morphmcp"],
            "env": {
                "MORPH_API_KEY": os.environ["MORPH_API_KEY"],
                "ENABLED_TOOLS": "all",
                "WORKSPACE_MODE": "true",
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


@dataclass
class RetryState:
    """Per-issue state for tracking gate and review retry attempts."""

    attempt: int = 1  # Gate attempt number
    review_attempt: int = 0  # Review attempt number (0 = not started)
    log_offset: int = 0  # Byte offset for scoped evidence parsing
    previous_commit_hash: str | None = None


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
        braintrust_enabled: bool = False,
        max_gate_retries: int = 3,
        max_review_retries: int = 2,
        codex_review: bool = False,
    ):
        self.repo_path = repo_path.resolve()
        self.max_agents = max_agents
        self.timeout_seconds = timeout_minutes * 60 if timeout_minutes else None
        self.max_issues = max_issues
        self.epic_id = epic_id
        self.only_ids = only_ids
        self.braintrust_enabled = braintrust_enabled
        self.max_gate_retries = max_gate_retries
        self.max_review_retries = max_review_retries
        self.codex_review = codex_review

        self.active_tasks: dict[str, asyncio.Task] = {}
        self.agent_ids: dict[str, str] = {}
        self.completed: list[IssueResult] = []
        self.failed_issues: set[str] = set()

        # Track session log paths for quality gate (issue_id -> log_path)
        self.session_log_paths: dict[str, Path] = {}

        # Quality gate for post-run validation
        self.quality_gate = QualityGate(self.repo_path)

        # Initialize BeadsClient with warning logger
        def log_warning(msg: str) -> None:
            log("⚠", msg, Colors.YELLOW)

        self.beads = BeadsClient(self.repo_path, log_warning=log_warning)

    def _cleanup_agent_locks(self, agent_id: str):
        """Remove locks held by a specific agent (crash/timeout cleanup)."""
        cleaned = cleanup_agent_locks(agent_id)
        if cleaned:
            log(
                "🧹",
                f"Cleaned {cleaned} locks for {agent_id[:8]}",
                Colors.GRAY,
                dim=True,
            )

    def _run_quality_gate(
        self,
        issue_id: str,
        log_path: Path,
        retry_state: RetryState,
    ) -> tuple[GateResult, int]:
        """Run quality gate check with attempt-scoped evidence.

        Args:
            issue_id: The issue being checked.
            log_path: Path to the session log file.
            retry_state: Current retry state for this issue.

        Returns:
            Tuple of (GateResult, new_log_offset).
        """
        # Check for matching commit
        commit_result = self.quality_gate.check_commit_exists(issue_id)

        # Parse validation evidence from the current attempt's log offset
        evidence, new_offset = self.quality_gate.parse_validation_evidence_from_offset(
            log_path, offset=retry_state.log_offset
        )

        failure_reasons: list[str] = []

        if not commit_result.exists:
            failure_reasons.append(
                f"No commit with bd-{issue_id} found in last 30 days"
            )

        if not evidence.has_minimum_validation():
            missing = evidence.missing_commands()
            failure_reasons.append(f"Missing validation commands: {', '.join(missing)}")

        # Check for no-progress condition (same commit, no new evidence)
        no_progress = False
        if retry_state.attempt > 1:
            no_progress = self.quality_gate.check_no_progress(
                log_path,
                retry_state.log_offset,
                retry_state.previous_commit_hash,
                commit_result.commit_hash,
            )
            if no_progress:
                failure_reasons.append(
                    "No progress: commit unchanged and no new validation evidence"
                )

        return (
            GateResult(
                passed=len(failure_reasons) == 0,
                failure_reasons=failure_reasons,
                commit_hash=commit_result.commit_hash,
                validation_evidence=evidence,
                no_progress=no_progress,
            ),
            new_offset,
        )

    async def run_implementer(self, issue_id: str) -> IssueResult:
        """Run implementer agent for a single issue with gate retry support."""
        agent_id = f"{issue_id}-{uuid.uuid4().hex[:8]}"
        self.agent_ids[issue_id] = agent_id

        # Initialize retry state for this issue
        retry_state = RetryState()

        # Claude session ID will be captured from ResultMessage
        claude_session_id: str | None = None

        prompt = IMPLEMENTER_PROMPT_TEMPLATE.format(
            issue_id=issue_id,
            repo_path=self.repo_path,
            lock_dir=LOCK_DIR,
            scripts_dir=SCRIPTS_DIR,
            agent_id=agent_id,
        )

        final_result = ""
        start_time = asyncio.get_event_loop().time()
        success = False
        gate_result: GateResult | None = None

        # Set up environment with lock scripts in PATH
        agent_env = {
            "PATH": f"{SCRIPTS_DIR}:{os.environ.get('PATH', '')}",
            "LOCK_DIR": str(LOCK_DIR),
            "AGENT_ID": agent_id,
            "REPO_NAMESPACE": str(self.repo_path),
        }

        options = ClaudeAgentOptions(
            cwd=str(self.repo_path),
            permission_mode="bypassPermissions",
            model="opus",
            system_prompt={"type": "preset", "preset": "claude_code"},
            setting_sources=["project", "user"],
            mcp_servers=get_mcp_servers(self.repo_path),
            disallowed_tools=MORPH_DISALLOWED_TOOLS,
            env=agent_env,
            hooks={
                "PreToolUse": [
                    HookMatcher(
                        matcher=None,
                        hooks=[
                            block_dangerous_commands,
                            block_morph_replaced_tools,
                            make_lock_enforcement_hook(agent_id, str(self.repo_path)),
                        ],  # type: ignore[arg-type]
                    )
                ],
                "Stop": [HookMatcher(matcher=None, hooks=[make_stop_hook(agent_id)])],
            },
        )

        # Braintrust tracing context
        tracer = TracedAgentExecution(
            issue_id=issue_id,
            agent_id=agent_id,
            metadata={
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
                            # Initial query
                            await client.query(prompt)

                            # Gate retry loop
                            while True:
                                # Process messages from this attempt
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
                                        final_result = message.result or ""

                                # Agent completed this attempt - run quality gate
                                if claude_session_id:
                                    log_path = get_claude_log_path(
                                        self.repo_path, claude_session_id
                                    )
                                    self.session_log_paths[issue_id] = log_path

                                    # Wait for log file with bounded timeout
                                    wait_elapsed = 0.0
                                    while not log_path.exists():
                                        if wait_elapsed >= LOG_FILE_WAIT_TIMEOUT:
                                            final_result = (
                                                f"Session log missing after "
                                                f"{LOG_FILE_WAIT_TIMEOUT}s: {log_path}"
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

                                    # Exit retry loop if log file never appeared
                                    if not log_path.exists():
                                        break

                                    if log_path.exists():
                                        gate_result, new_offset = (
                                            self._run_quality_gate(
                                                issue_id, log_path, retry_state
                                            )
                                        )

                                        if gate_result.passed:
                                            # Gate passed - run Codex review if enabled
                                            if (
                                                self.codex_review
                                                and gate_result.commit_hash
                                            ):
                                                retry_state.review_attempt += 1
                                                log(
                                                    "◐",
                                                    f"Running Codex review (attempt {retry_state.review_attempt}/{self.max_review_retries})",
                                                    Colors.CYAN,
                                                    agent_id=issue_id,
                                                )

                                                review_result = await run_codex_review(
                                                    self.repo_path,
                                                    gate_result.commit_hash,
                                                    max_retries=2,  # JSON parse retries
                                                )

                                                if review_result.passed:
                                                    # Review passed - we're done
                                                    log(
                                                        "✓",
                                                        "Codex review passed",
                                                        Colors.GREEN,
                                                        agent_id=issue_id,
                                                    )
                                                    success = True
                                                    break

                                                # Review failed - check if we can retry
                                                if (
                                                    retry_state.review_attempt
                                                    < self.max_review_retries
                                                ):
                                                    # Update log offset for next gate check
                                                    retry_state.log_offset = new_offset
                                                    retry_state.previous_commit_hash = (
                                                        gate_result.commit_hash
                                                    )

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
                                                            f"Review retry {retry_state.review_attempt}/{self.max_review_retries} ({error_count} errors)",
                                                            Colors.YELLOW,
                                                            agent_id=issue_id,
                                                        )

                                                    # Build follow-up prompt with review issues
                                                    review_issues_text = (
                                                        format_review_issues(
                                                            review_result.issues
                                                        )
                                                        if review_result.issues
                                                        else review_result.parse_error
                                                        or "Unknown review failure"
                                                    )
                                                    followup = CODEX_REVIEW_FOLLOWUP_PROMPT.format(
                                                        attempt=retry_state.review_attempt,
                                                        max_attempts=self.max_review_retries,
                                                        review_issues=review_issues_text,
                                                        issue_id=issue_id,
                                                    )

                                                    # Resume session with follow-up prompt
                                                    await client.query(
                                                        followup,
                                                        session_id=claude_session_id,
                                                    )
                                                    # Continue the retry loop (will re-run gate + review)
                                                    continue

                                                # No review retries left - fail
                                                if review_result.parse_error:
                                                    final_result = f"Codex review failed: {review_result.parse_error}"
                                                else:
                                                    error_msgs = [
                                                        f"{i.file}:{i.line or 0}: {i.message}"
                                                        for i in review_result.issues
                                                        if i.severity == "error"
                                                    ]
                                                    final_result = f"Codex review failed: {'; '.join(error_msgs[:3])}"
                                                break

                                            # No Codex review or no commit - we're done
                                            success = True
                                            break

                                        # Gate failed - check if we can retry
                                        if (
                                            retry_state.attempt < self.max_gate_retries
                                            and not gate_result.no_progress
                                        ):
                                            # Prepare for retry
                                            retry_state.attempt += 1
                                            retry_state.log_offset = new_offset
                                            retry_state.previous_commit_hash = (
                                                gate_result.commit_hash
                                            )

                                            # Log retry attempt
                                            log(
                                                "↻",
                                                f"Gate retry {retry_state.attempt}/{self.max_gate_retries}",
                                                Colors.YELLOW,
                                                agent_id=issue_id,
                                            )

                                            # Build follow-up prompt with failure context
                                            failure_text = "\n".join(
                                                f"- {r}"
                                                for r in gate_result.failure_reasons
                                            )
                                            followup = GATE_FOLLOWUP_PROMPT.format(
                                                attempt=retry_state.attempt,
                                                max_attempts=self.max_gate_retries,
                                                failure_reasons=failure_text,
                                                issue_id=issue_id,
                                            )

                                            # Resume session with follow-up prompt
                                            await client.query(
                                                followup, session_id=claude_session_id
                                            )
                                            # Continue the retry loop
                                            continue

                                        # No retries left or no progress - fail
                                        final_result = (
                                            f"Quality gate failed: "
                                            f"{'; '.join(gate_result.failure_reasons)}"
                                        )
                                        break
                                else:
                                    # No session ID - can't run gate
                                    break

                    tracer.set_success(success)

                except TimeoutError:
                    timeout_mins = (
                        self.timeout_seconds // 60 if self.timeout_seconds else 0
                    )
                    final_result = f"Timeout after {timeout_mins} minutes"
                    tracer.set_error(final_result)
                    self._cleanup_agent_locks(agent_id)

                except Exception as e:
                    final_result = str(e)
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
            success=success,
            summary=final_result,
            duration_seconds=duration,
            session_id=claude_session_id,
            gate_attempts=retry_state.attempt,
            review_attempts=retry_state.review_attempt,
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
        """Main orchestration loop. Returns (success_count, total_count)."""
        print()
        log("●", "mala orchestrator", Colors.MAGENTA)
        log("◐", f"repo: {self.repo_path}", Colors.GRAY, dim=True)
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
            Colors.GRAY,
            dim=True,
        )
        log(
            "◐",
            f"gate-retries: {self.max_gate_retries}",
            Colors.GRAY,
            dim=True,
        )
        if self.codex_review:
            log(
                "◐",
                f"codex-review: enabled (max-retries: {self.max_review_retries})",
                Colors.CYAN,
                dim=True,
            )

        # Report epic filter
        if self.epic_id:
            log("◐", f"epic filter: {self.epic_id}", Colors.CYAN, dim=True)

        # Report only_ids filter
        if self.only_ids:
            log(
                "◐",
                f"only processing: {', '.join(sorted(self.only_ids))}",
                Colors.CYAN,
                dim=True,
            )

        # Report Braintrust status
        if self.braintrust_enabled:
            log(
                "◐",
                "braintrust: enabled (LLM spans auto-traced)",
                Colors.CYAN,
                dim=True,
            )
        else:
            log(
                "◐",
                f"braintrust: disabled (add BRAINTRUST_API_KEY to {USER_CONFIG_DIR}/.env)",
                Colors.GRAY,
                dim=True,
            )

        # Report Morph MCP status
        log(
            "◐",
            "morph: enabled (edit_file, warpgrep_codebase_search)",
            Colors.CYAN,
            dim=True,
        )
        log(
            "◐",
            f"morph: blocked tools: {', '.join(MORPH_DISALLOWED_TOOLS)}",
            Colors.GRAY,
            dim=True,
        )
        print()

        # Setup directories
        LOCK_DIR.mkdir(exist_ok=True)
        RUNS_DIR.mkdir(parents=True, exist_ok=True)

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
            codex_review=self.codex_review,
        )
        run_metadata = RunMetadata(self.repo_path, run_config, __version__)

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
                    )
                    if not limit_reached
                    else []
                )

                if ready:
                    log("◌", f"Ready issues: {', '.join(ready)}", Colors.GRAY, dim=True)

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
                    Colors.GRAY,
                    dim=True,
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
                            log_path = self.session_log_paths.get(issue_id)

                            if result.success and log_path and log_path.exists():
                                # Parse final evidence for metadata (full session)
                                evidence = self.quality_gate.parse_validation_evidence(
                                    log_path
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
                                        "uv_sync_ran": evidence.uv_sync_ran,
                                        "ty_check_ran": evidence.ty_check_ran,
                                        "commit_found": commit_result.exists,
                                    },
                                    failure_reasons=[],
                                )
                                # Gate passed - close the issue
                                if await self.beads.close_async(issue_id):
                                    log(
                                        "◐",
                                        "Closed issue",
                                        Colors.GRAY,
                                        dim=True,
                                        agent_id=issue_id,
                                    )
                            elif not result.success and log_path and log_path.exists():
                                # Gate failed - record evidence for metadata
                                evidence = self.quality_gate.parse_validation_evidence(
                                    log_path
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
                                        "uv_sync_ran": evidence.uv_sync_ran,
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
                            )
                            run_metadata.record_issue(issue_run)

                            # Pop log path after recording
                            self.session_log_paths.pop(issue_id, None)

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
                log("🧹", f"Released {released} remaining locks", Colors.GRAY, dim=True)

        # Summary
        print()
        success_count = sum(1 for r in self.completed if r.success)
        total = len(self.completed)

        if success_count == total and total > 0:
            log("●", f"Completed: {success_count}/{total} issues", Colors.GREEN)
        elif success_count > 0:
            log("◐", f"Completed: {success_count}/{total} issues", Colors.YELLOW)
        elif total == 0:
            log("○", "No issues to process", Colors.GRAY)
        else:
            log("○", f"Completed: {success_count}/{total} issues", Colors.RED)

        # Commit .beads/issues.jsonl if there were successes
        if success_count > 0:
            if await self.beads.commit_issues_async():
                log("◐", "Committed .beads/issues.jsonl", Colors.GRAY, dim=True)

            # Auto-close epics where all children are complete
            if await self.beads.close_eligible_epics_async():
                log("◐", "Auto-closed completed epics", Colors.GRAY, dim=True)

        # Save run metadata
        if total > 0:
            metadata_path = run_metadata.save()
            log("◐", f"Run metadata: {metadata_path}", Colors.GRAY, dim=True)

        print()
        # Return success count for CLI exit code logic
        # Also indicate if no issues were processed (total == 0 is a no-op, not failure)
        return (success_count, total)
