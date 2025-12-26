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
from .git_utils import get_git_commit, get_git_branch
from .hooks import (
    MORPH_DISALLOWED_TOOLS,
    block_dangerous_commands,
    block_morph_replaced_tools,
)
from .logging.console import Colors, get_agent_color, log, log_tool
from .logging.jsonl import JSONLLogger
from .tools.env import USER_CONFIG_DIR, JSONL_LOG_DIR, SCRIPTS_DIR
from .tools.locking import (
    LOCK_DIR,
    release_all_locks,
    cleanup_agent_locks,
    make_stop_hook,
)


# Version (from package metadata)
from importlib.metadata import version as pkg_version

__version__ = pkg_version("mala")

# Load implementer prompt from file
PROMPT_FILE = Path(__file__).parent / "prompts" / "implementer_prompt.md"
IMPLEMENTER_PROMPT_TEMPLATE = PROMPT_FILE.read_text()


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


class MalaOrchestrator:
    """Orchestrates parallel issue processing using Claude Agent SDK."""

    def __init__(
        self,
        repo_path: Path,
        max_agents: int | None = None,
        timeout_minutes: int = 30,
        max_issues: int | None = None,
        epic_id: str | None = None,
        only_ids: set[str] | None = None,
        braintrust_enabled: bool = False,
    ):
        self.repo_path = repo_path.resolve()
        self.max_agents = max_agents
        self.timeout_seconds = timeout_minutes * 60
        self.max_issues = max_issues
        self.epic_id = epic_id
        self.only_ids = only_ids
        self.braintrust_enabled = braintrust_enabled

        self.active_tasks: dict[str, asyncio.Task] = {}
        self.agent_ids: dict[str, str] = {}
        self.completed: list[IssueResult] = []
        self.failed_issues: set[str] = set()

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

    async def run_implementer(self, issue_id: str) -> IssueResult:
        """Run implementer agent for a single issue."""
        session_id = str(uuid.uuid4())
        agent_id = f"{issue_id}-{session_id[:8]}"
        self.agent_ids[issue_id] = agent_id

        # JSONL logger for full message logging
        jsonl_logger = JSONLLogger(session_id, self.repo_path)

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

        options = ClaudeAgentOptions(
            cwd=str(self.repo_path),
            permission_mode="bypassPermissions",
            model="opus",
            system_prompt={"type": "preset", "preset": "claude_code"},
            setting_sources=["project", "user"],
            mcp_servers=get_mcp_servers(self.repo_path),
            disallowed_tools=MORPH_DISALLOWED_TOOLS,
            hooks={
                "PreToolUse": [
                    HookMatcher(
                        matcher=None,
                        hooks=[block_dangerous_commands, block_morph_replaced_tools],  # type: ignore[arg-type]
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
                "git_branch": get_git_branch(self.repo_path),
                "git_commit": get_git_commit(self.repo_path),
                "session_id": session_id,
                "timeout_seconds": self.timeout_seconds,
            },
        )

        try:
            with tracer:
                tracer.log_input(prompt)

                try:
                    async with asyncio.timeout(self.timeout_seconds):
                        async with ClaudeSDKClient(options=options) as client:
                            await client.query(prompt)
                            jsonl_logger.log_user_prompt(prompt)

                            async for message in client.receive_response():
                                # Log full message to JSONL
                                jsonl_logger.log_message(message)

                                # Log to Braintrust
                                tracer.log_message(message)

                                # Console output
                                if isinstance(message, AssistantMessage):
                                    for block in message.content:
                                        if isinstance(block, TextBlock):
                                            text = (
                                                block.text[:100] + "..."
                                                if len(block.text) > 100
                                                else block.text
                                            )
                                            agent_color = get_agent_color(issue_id)
                                            print(
                                                f"    {agent_color}[{issue_id}]{Colors.RESET} {Colors.DIM}{text}{Colors.RESET}"
                                            )
                                        elif isinstance(block, ToolUseBlock):
                                            log_tool(
                                                block.name,
                                                str(block.input)[:50],
                                                agent_id=issue_id,
                                            )

                                elif isinstance(message, ResultMessage):
                                    final_result = message.result or ""

                    # Check if issue was closed (inside tracer context)
                    status = self.beads.get_issue_status(issue_id)
                    if status == "closed":
                        success = True

                    tracer.set_success(success)

                except TimeoutError:
                    final_result = f"Timeout after {self.timeout_seconds // 60} minutes"
                    tracer.set_error(final_result)
                    self._cleanup_agent_locks(agent_id)

                except Exception as e:
                    final_result = str(e)
                    tracer.set_error(final_result)
                    self._cleanup_agent_locks(agent_id)

        finally:
            duration = asyncio.get_event_loop().time() - start_time
            jsonl_logger.close()

            # Ensure locks are cleaned
            self._cleanup_agent_locks(agent_id)
            self.agent_ids.pop(issue_id, None)

        return IssueResult(
            issue_id=issue_id,
            agent_id=agent_id,
            success=success,
            summary=final_result,
            duration_seconds=duration,
        )

    async def spawn_agent(self, issue_id: str) -> bool:
        """Spawn a new agent task for an issue. Returns True if spawned."""
        if not self.beads.claim(issue_id):
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
        log(
            "◐",
            f"max-agents: {agents_str}, timeout: {self.timeout_seconds // 60}m, max-issues: {limit_str}",
            Colors.GRAY,
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
        JSONL_LOG_DIR.mkdir(parents=True, exist_ok=True)

        issues_spawned = 0

        try:
            while True:
                # Check if we've reached the issue limit
                limit_reached = (
                    self.max_issues is not None and issues_spawned >= self.max_issues
                )

                # Fill up to max_agents (unless limit reached)
                ready = (
                    self.beads.get_ready(
                        self.failed_issues,
                        epic_id=self.epic_id,
                        only_ids=self.only_ids,
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

                            self.completed.append(result)
                            del self.active_tasks[issue_id]

                            duration_str = f"{result.duration_seconds:.0f}s"
                            if result.success:
                                summary = (
                                    result.summary[:50] + "..."
                                    if len(result.summary) > 50
                                    else result.summary
                                )
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
                                self.beads.reset(issue_id)
                            break

        finally:
            # Final cleanup
            release_all_locks()
            log("🧹", "Released all remaining locks", Colors.GRAY, dim=True)

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
            if self.beads.commit_issues():
                log("◐", "Committed .beads/issues.jsonl", Colors.GRAY, dim=True)

            # Auto-close epics where all children are complete
            if self.beads.close_eligible_epics():
                log("◐", "Auto-closed completed epics", Colors.GRAY, dim=True)

        print()
        # Return success count for CLI exit code logic
        # Also indicate if no issues were processed (total == 0 is a no-op, not failure)
        return (success_count, total)
