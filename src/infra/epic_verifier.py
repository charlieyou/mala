"""Epic verification for ensuring epic acceptance criteria are met.

This module provides the EpicVerifier class which verifies that all code changes
made for issues under an epic collectively meet the epic's acceptance criteria
before allowing the epic to close.

Key components:
- EpicVerifier: Main verification orchestrator
- ClaudeEpicVerificationModel: Claude-based implementation of EpicVerificationModel protocol

Design principles:
- Scoped commits: Only include commits linked to child issues (by bd-<id>: prefix)
- Agent-driven exploration: Verification agent explores repo using commit list
- Remediation issues: Auto-create with deduplication for unmet criteria
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from src.core.models import (
    EpicVerdict,
    EpicVerificationResult,
    RetryConfig,
    UnmetCriterion,
)
from src.infra.epic_scope import EpicScopeAnalyzer

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from src.core.protocols.events import MalaEventSink
    from src.core.protocols.infra import CommandRunnerPort, LockManagerPort
    from src.core.protocols.validation import EpicVerificationModel
    from src.infra.clients.beads_client import BeadsClient


class VerificationRetryPolicyProtocol(Protocol):
    """Protocol for retry policy to avoid importing from domain layer.

    Duck typing interface matching VerificationRetryPolicy from domain.validation.config.
    """

    timeout_retries: int
    execution_retries: int
    parse_retries: int


logger = logging.getLogger(__name__)

# Default lock timeout for epic verification (5 minutes)
DEFAULT_EPIC_VERIFY_LOCK_TIMEOUT_SECONDS = 300


@asynccontextmanager
async def epic_verify_lock(
    epic_id: str,
    repo_path: Path,
    lock_manager: LockManagerPort | None,
    timeout_seconds: int | None = None,
) -> AsyncIterator[bool]:
    """Acquire per-epic verification lock with automatic cleanup.

    Args:
        epic_id: Epic identifier for lock key.
        repo_path: Repository path for lock context.
        lock_manager: Lock manager instance or None (no locking).
        timeout_seconds: Lock acquisition timeout. None uses default (300s).

    Yields True if lock acquired (or locking unavailable), False if timed out.
    Caller decides how to handle False (skip, return empty result, etc.)
    """
    if lock_manager is None:
        yield True
        return

    lock_key = f"epic_verify:{epic_id}"
    lock_agent_id = f"epic_verifier_{os.getpid()}"
    effective_timeout = (
        timeout_seconds
        if timeout_seconds is not None
        else DEFAULT_EPIC_VERIFY_LOCK_TIMEOUT_SECONDS
    )

    acquired = await asyncio.to_thread(
        lock_manager.wait_for_lock,
        lock_key,
        lock_agent_id,
        str(repo_path),
        effective_timeout,
    )

    try:
        yield acquired
    finally:
        if acquired:
            lock_manager.release_lock(lock_key, lock_agent_id, str(repo_path))


def _compute_criterion_hash(criterion: str) -> str:
    """Compute SHA256 hash of criterion text for deduplication."""
    return hashlib.sha256(criterion.encode()).hexdigest()


def _load_prompt_template() -> str:
    """Load the epic verification prompt template."""
    # Navigate from src/infra/ to src/prompts/
    prompt_path = Path(__file__).parent.parent / "prompts" / "epic_verification.md"
    template = prompt_path.read_text()
    escaped = template.replace("{", "{{").replace("}", "}}")
    # Replace ${EPIC_CONTEXT} placeholder (cerberus style)
    escaped = escaped.replace("${{EPIC_CONTEXT}}", "{epic_context}")
    return escaped


@dataclass
class EpicVerificationContext:
    """Context captured during verification for richer remediation issues."""

    epic_description: str
    child_ids: set[str]
    blocker_ids: set[str]
    commit_shas: list[str]
    commit_range: str | None
    commit_summary: str


def _extract_json_from_code_blocks(text: str) -> str | None:
    """Extract JSON content from markdown code blocks.

    This function properly handles responses with multiple code blocks by
    parsing fences line-by-line and using json.JSONDecoder().raw_decode()
    to extract balanced JSON objects. This approach correctly handles
    nested code blocks inside JSON string values (e.g., code examples
    embedded in a "body" field).

    Args:
        text: The raw model response text that may contain markdown code blocks.

    Returns:
        The JSON string extracted from the first JSON code block, or None if
        no valid JSON code block is found.
    """
    decoder = json.JSONDecoder()

    def _extract_first_json_object(block: str) -> str | None:
        """Extract first balanced JSON object from a code block."""
        s = block.strip()
        search_from = 0
        while True:
            start = s.find("{", search_from)
            if start == -1:
                return None
            try:
                obj, end = decoder.raw_decode(s[start:])
            except json.JSONDecodeError:
                search_from = start + 1
                continue
            if isinstance(obj, dict):
                return s[start : start + end]
            search_from = start + 1

    in_fence = False
    buf: list[str] = []

    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("```"):
            if not in_fence:
                in_fence = True
                buf = []
                continue
            if stripped.strip() == "```":
                in_fence = False
                candidate = _extract_first_json_object("".join(buf))
                if candidate is not None:
                    return candidate
                buf = []
                continue

        if in_fence:
            buf.append(line)

    if in_fence and buf:
        return _extract_first_json_object("".join(buf))

    return None


class ClaudeEpicVerificationModel:
    """Claude-based implementation of EpicVerificationModel protocol.

    Uses the Claude Agent SDK to verify epic acceptance criteria
    against scoped commits, matching the main agent execution path.
    """

    # Default model for epic verification
    DEFAULT_MODEL = "claude-sonnet-4-20250514"

    def __init__(
        self,
        model: str | None = None,
        timeout_ms: int = 300000,
        repo_path: Path | None = None,
        retry_config: RetryConfig | None = None,
    ):
        """Initialize ClaudeEpicVerificationModel.

        Args:
            model: The Claude model to use for verification. Defaults to
                DEFAULT_MODEL if not specified.
            timeout_ms: Timeout for model calls in milliseconds. Default is
                5 minutes (300000ms) to allow sufficient time for agent-driven
                repository exploration on non-trivial epics.
            repo_path: Repository path for agent execution context.
            retry_config: Configuration retained for compatibility with prior
                retry behavior. Currently unused but stored for future use.
        """
        self.model = model or self.DEFAULT_MODEL
        self.timeout_ms = timeout_ms
        self.repo_path = repo_path or Path.cwd()
        self.retry_config = retry_config or RetryConfig()
        self._prompt_template = _load_prompt_template()

    async def verify(
        self,
        epic_context: str,
    ) -> EpicVerdict:
        """Verify if the epic's acceptance criteria are met.

        The agent explores the repository using its tools to find and verify
        the implementation of each acceptance criterion.

        Args:
            epic_context: Combined epic content including description, plan,
                and spec file content. This is injected into the prompt's
                EPIC_CONTEXT section.

        Returns:
            Structured verdict with pass/fail and unmet criteria details.

        """
        prompt = self._prompt_template.format(
            epic_context=epic_context,
        )

        response_text = await self._verify_with_agent_sdk(prompt)
        return self._parse_verdict(response_text)

    async def _verify_with_agent_sdk(self, prompt: str) -> str:
        """Verify using Claude Agent SDK."""
        try:
            from claude_agent_sdk import (
                AssistantMessage,
                ClaudeAgentOptions,
                ClaudeSDKClient,
                TextBlock,
            )
        except ImportError as exc:
            return json.dumps(
                {
                    "findings": [],
                    "verdict": "FAIL",
                    "summary": (
                        "claude_agent_sdk is not installed; epic verification "
                        f"requires the agent SDK ({exc})"
                    ),
                }
            )

        options = ClaudeAgentOptions(
            cwd=str(self.repo_path),
            permission_mode="bypassPermissions",
            model=self.model,
            system_prompt={"type": "preset", "preset": "claude_code"},
            setting_sources=["project", "user"],
            settings='{"autoCompactEnabled": true}',
            mcp_servers={},
            allowed_tools=[
                # Only these tools are permitted; all others (Edit, Write, etc.)
                # are blocked by omission. Bash is needed for git commands.
                "Bash",
                "Glob",
                "Grep",
                "Read",
                "Task",
            ],
            env=dict(os.environ),
        )

        response_chunks: list[str] = []
        async with ClaudeSDKClient(options=options) as client:
            async with asyncio.timeout(self.timeout_ms / 1000):
                await client.query(prompt)
                async for message in client.receive_response():
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                response_chunks.append(block.text)
                    # Note: ResultMessage.result (tool outputs) are intentionally
                    # excluded to avoid polluting the response with JSON that may
                    # appear in tool outputs (e.g., file contents). The final
                    # verdict JSON should come from the assistant's text response.

        return "".join(response_chunks).strip()

    def _parse_verdict(self, response_text: str) -> EpicVerdict:
        """Parse model response into EpicVerdict.

        Uses a robust code block parser that handles responses with multiple
        markdown code blocks by extracting each block individually, avoiding
        the issue where a greedy regex could span multiple blocks.

        Args:
            response_text: The raw model response text.

        Returns:
            Parsed EpicVerdict.
        """
        # Extract JSON from code blocks using the robust non-greedy parser
        json_str = _extract_json_from_code_blocks(response_text)

        if json_str is None:
            # Try to find raw JSON object (not in code block)
            json_match = re.search(
                r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", response_text, re.DOTALL
            )
            if json_match:
                json_str = json_match.group(0)
            else:
                # Fallback: assume entire response is JSON
                json_str = response_text

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return EpicVerdict(
                passed=False,
                unmet_criteria=[],
                reasoning=f"Failed to parse model response: {response_text[:500]}",
            )

        unmet_criteria = []
        for item in data.get("findings", []):
            title = str(item.get("title", "")).strip()
            body = str(item.get("body", "")).strip()
            criterion = title or body or "Unspecified criterion"
            priority = item.get("priority", 1)
            # Accept int, float, or numeric string; cast to int, clamp to valid range 0-3
            if isinstance(priority, (int, float)):
                priority = max(0, min(3, int(priority)))
            elif isinstance(priority, str) and priority.isdigit():
                priority = max(0, min(3, int(priority)))
            else:
                priority = 1
            unmet_criteria.append(
                UnmetCriterion(
                    criterion=criterion,
                    evidence=body,
                    priority=priority,
                    criterion_hash=_compute_criterion_hash(criterion),
                )
            )

        verdict = data.get("verdict", "FAIL")
        passed = verdict == "PASS"

        return EpicVerdict(
            passed=passed,
            unmet_criteria=unmet_criteria,
            reasoning=data.get("summary", ""),
        )


class EpicVerifier:
    """Main verification orchestrator for epic acceptance criteria.

    Gathers epic data, computes scoped commits from child issue commits,
    invokes verification model, and creates remediation issues for unmet criteria.
    """

    def __init__(
        self,
        beads: BeadsClient,
        model: EpicVerificationModel,
        repo_path: Path,
        command_runner: CommandRunnerPort,
        retry_config: RetryConfig | None = None,
        lock_manager: LockManagerPort | None = None,
        event_sink: MalaEventSink | None = None,
        scope_analyzer: EpicScopeAnalyzer | None = None,
        max_diff_size_kb: int | None = None,
        lock_timeout_seconds: int | None = None,
        reviewer_type: str = "agent_sdk",
        retry_policy: VerificationRetryPolicyProtocol | None = None,
    ):
        """Initialize EpicVerifier.

        Args:
            beads: BeadsClient for issue operations.
            model: EpicVerificationModel for verification.
            repo_path: Path to the repository.
            command_runner: CommandRunnerPort for executing commands.
            retry_config: Configuration for retry behavior.
            lock_manager: Optional lock manager for sequential processing.
            event_sink: Optional event sink for emitting verification lifecycle events.
            scope_analyzer: Optional EpicScopeAnalyzer for computing scoped commits.
                If not provided, a default instance is created.
            max_diff_size_kb: Maximum diff size in KB. If set and diff exceeds limit,
                verification is skipped and returns a verdict requiring human review.
            lock_timeout_seconds: Timeout in seconds for acquiring epic verification
                lock. None uses the default (300 seconds).
            reviewer_type: Type of reviewer ('agent_sdk' or 'cerberus'). Used for
                telemetry and event reporting.
            retry_policy: Per-category retry limits for verification failures.
                If None, a default policy is used.
        """
        self.beads = beads
        self.model = model
        self.repo_path = repo_path
        self.retry_config = retry_config or RetryConfig()
        self.lock_manager = lock_manager
        self.event_sink = event_sink
        self._runner = command_runner
        self.scope_analyzer = scope_analyzer or EpicScopeAnalyzer(
            repo_path, self._runner
        )
        self.max_diff_size_kb = max_diff_size_kb
        self.lock_timeout_seconds = lock_timeout_seconds
        self.reviewer_type = reviewer_type
        self.retry_policy = retry_policy

    async def _get_diff_size_kb(self, commit_range: str) -> int | None:
        """Get approximate diff size in KB for a commit range.

        Uses git diff --stat to estimate the size based on line changes.
        Approximates 80 bytes per line for size estimation.

        Args:
            commit_range: Git commit range (e.g., "abc123^..def456") or single SHA.

        Returns:
            Estimated diff size in KB, or None if unable to compute.
        """
        # Parse commit range to get from/to commits
        # Format: "abc123^..def456" -> from="abc123^", to="def456"
        # Single commit: "abc123" -> from="abc123^", to="abc123"
        if ".." in commit_range:
            parts = commit_range.split("..", 1)
            if len(parts) != 2:
                return None
            from_commit, to_commit = parts[0], parts[1]
            if not from_commit or not to_commit:
                return None
        else:
            # Single commit - diff against its parent
            if not commit_range:
                return None
            from_commit = f"{commit_range}^"
            to_commit = commit_range

        result = await self._runner.run_async(
            ["git", "diff", "--numstat", from_commit, to_commit],
            cwd=self.repo_path,
        )
        if not result.ok:
            return None

        total_lines = 0
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                added, removed = parts[0], parts[1]
                # Skip binary files (marked with "-")
                if added != "-" and removed != "-":
                    try:
                        total_lines += int(added) + int(removed)
                    except ValueError:
                        pass

        # Approximate 80 bytes per line, convert to KB
        return (total_lines * 80) // 1024

    async def verify_and_close_eligible(
        self,
        human_override_epic_ids: set[str] | None = None,
    ) -> EpicVerificationResult:
        """Check for epics eligible to close, verify each, and close those that pass.

        This method discovers eligible epics BEFORE attempting to close them,
        ensuring verification happens first.

        Args:
            human_override_epic_ids: Epic IDs to close without verification
                                     (explicit human override).

        Returns:
            Summary of verification results.
        """
        human_override_epic_ids = human_override_epic_ids or set()

        # Get eligible epics (those with all children closed)
        eligible_epics = await self._get_eligible_epics()

        # Aggregate results from individual verifications
        all_verdicts: dict[str, EpicVerdict] = {}
        all_remediation_issues: list[str] = []
        verified_count = 0
        passed_count = 0
        failed_count = 0

        for epic_id in eligible_epics:
            is_override = epic_id in human_override_epic_ids
            result = await self.verify_epic_with_options(
                epic_id,
                human_override=is_override,
                require_eligible=False,  # Already filtered by _get_eligible_epics
                close_epic=True,
            )
            # Aggregate result counts and verdicts
            all_verdicts.update(result.verdicts)
            all_remediation_issues.extend(result.remediation_issues_created)
            verified_count += result.verified_count
            passed_count += result.passed_count
            failed_count += result.failed_count

        return EpicVerificationResult(
            verified_count=verified_count,
            passed_count=passed_count,
            failed_count=failed_count,
            verdicts=all_verdicts,
            remediation_issues_created=all_remediation_issues,
        )

    async def verify_epic(self, epic_id: str) -> EpicVerdict:
        """Verify a single epic against its acceptance criteria.

        Args:
            epic_id: The epic ID to verify.

        Returns:
            EpicVerdict with verification result.
        """
        verdict, _ = await self._verify_epic_with_context(epic_id)
        return verdict

    async def _verify_epic_with_context(
        self, epic_id: str
    ) -> tuple[EpicVerdict, EpicVerificationContext | None]:
        """Verify a single epic and return verdict plus context."""
        # Get epic description (contains acceptance criteria)
        epic_description = await self.beads.get_issue_description_async(epic_id)
        if not epic_description:
            return (
                EpicVerdict(
                    passed=False,
                    unmet_criteria=[],
                    reasoning="No acceptance criteria found for epic",
                ),
                None,
            )

        # Get child issue IDs
        child_ids = await self.beads.get_epic_children_async(epic_id)
        if not child_ids:
            return (
                EpicVerdict(
                    passed=False,
                    unmet_criteria=[],
                    reasoning="No child issues found for epic",
                ),
                None,
            )

        # Get blocker issue IDs (remediation issues from previous verification runs)
        blocker_ids = await self.beads.get_epic_blockers_async(epic_id) or set()

        # Compute scoped commits using EpicScopeAnalyzer
        scoped = await self.scope_analyzer.compute_scoped_commits(
            child_ids, blocker_ids
        )
        if not scoped.commit_shas:
            return (
                EpicVerdict(
                    passed=False,
                    unmet_criteria=[],
                    reasoning="No commits found for child issues",
                ),
                None,
            )

        # Check diff size limit if configured
        if self.max_diff_size_kb is not None and scoped.commit_range:
            diff_size_kb = await self._get_diff_size_kb(scoped.commit_range)
            if diff_size_kb is not None and diff_size_kb > self.max_diff_size_kb:
                criterion = (
                    f"Diff size must not exceed {self.max_diff_size_kb} KB "
                    "for automated verification"
                )
                return (
                    EpicVerdict(
                        passed=False,
                        unmet_criteria=[
                            UnmetCriterion(
                                criterion=criterion,
                                evidence=(
                                    f"Diff size ({diff_size_kb} KB) exceeds limit "
                                    f"({self.max_diff_size_kb} KB). "
                                    "Requires human review."
                                ),
                                priority=1,  # P1 blocking
                                criterion_hash=_compute_criterion_hash(criterion),
                            )
                        ],
                        reasoning=(
                            f"Diff size ({diff_size_kb} KB) exceeds limit "
                            f"({self.max_diff_size_kb} KB). Requires human review."
                        ),
                    ),
                    None,
                )

        context = EpicVerificationContext(
            epic_description=epic_description,
            child_ids=child_ids,
            blocker_ids=blocker_ids,
            commit_shas=scoped.commit_shas,
            commit_range=scoped.commit_range,
            commit_summary=scoped.commit_summary,
        )

        # Invoke verification model with per-category retry policy (R6)
        # Per spec: timeouts/errors should trigger human review, not abort
        verdict = await self._verify_with_category_retries(epic_description)

        return verdict, context

    async def _verify_with_category_retries(
        self,
        epic_context: str,
    ) -> EpicVerdict:
        """Verify with per-category retry limits (R6 implementation).

        Different failure categories have different retry characteristics:
        - Timeout errors are often transient and worth aggressive retrying
        - Execution errors may be environmental issues worth moderate retrying
        - Parse errors are often deterministic and unlikely to succeed on retry

        Note: Per-category retries apply to Cerberus-specific exceptions. For
        Agent SDK, asyncio.TimeoutError is treated as a timeout error. Other
        SDK failures fall through to generic exception handling (no retry).

        Args:
            epic_context: Combined epic content for verification.

        Returns:
            EpicVerdict with verification result.
        """
        # Import error types from cerberus verifier
        from src.infra.clients.cerberus_epic_verifier import (
            VerificationExecutionError,
            VerificationParseError,
            VerificationTimeoutError,
        )

        # Get retry limits from policy or use defaults, with validation
        # Clamp to non-negative integers to prevent infinite loops from invalid input
        def _safe_int(val: object, default: int) -> int:
            """Safely convert to non-negative int, clamping invalid values."""
            try:
                n = int(val)  # type: ignore[arg-type]
                return max(0, n)
            except (TypeError, ValueError):
                return default

        if self.retry_policy is not None:
            timeout_retries = _safe_int(self.retry_policy.timeout_retries, 3)
            execution_retries = _safe_int(self.retry_policy.execution_retries, 2)
            parse_retries = _safe_int(self.retry_policy.parse_retries, 1)
        else:
            # Default policy: aggressive for timeout, moderate for execution, minimal for parse
            timeout_retries = 3
            execution_retries = 2
            parse_retries = 1

        # Track attempts per category
        timeout_attempts = 0
        execution_attempts = 0
        parse_attempts = 0
        last_error: Exception | None = None

        while True:
            try:
                result = await self.model.verify(epic_context)
                return EpicVerdict(
                    passed=result.passed,
                    unmet_criteria=[
                        UnmetCriterion(
                            criterion=c.criterion,
                            evidence=c.evidence,
                            priority=c.priority,
                            criterion_hash=c.criterion_hash,
                        )
                        for c in result.unmet_criteria
                    ],
                    reasoning=result.reasoning,
                )
            except VerificationTimeoutError as e:
                timeout_attempts += 1
                last_error = e
                if timeout_attempts <= timeout_retries:
                    logger.warning(
                        "Verification timeout (attempt %d/%d): %s",
                        timeout_attempts,
                        timeout_retries + 1,
                        e,
                    )
                    continue
                # Exhausted retries for this category
                break
            except TimeoutError as e:
                # Agent SDK may raise asyncio.TimeoutError - treat as timeout category.
                # Note: Since Python 3.11, asyncio.TimeoutError IS TimeoutError (same class).
                # In Python 3.8-3.10, it was a subclass. Either way, catching TimeoutError
                # catches asyncio.TimeoutError.
                timeout_attempts += 1
                last_error = e
                if timeout_attempts <= timeout_retries:
                    logger.warning(
                        "Verification timeout (attempt %d/%d): %s",
                        timeout_attempts,
                        timeout_retries + 1,
                        e,
                    )
                    continue
                break
            except VerificationExecutionError as e:
                execution_attempts += 1
                last_error = e
                if execution_attempts <= execution_retries:
                    logger.warning(
                        "Verification execution error (attempt %d/%d): %s",
                        execution_attempts,
                        execution_retries + 1,
                        e,
                    )
                    continue
                break
            except VerificationParseError as e:
                parse_attempts += 1
                last_error = e
                if parse_attempts <= parse_retries:
                    logger.warning(
                        "Verification parse error (attempt %d/%d): %s",
                        parse_attempts,
                        parse_retries + 1,
                        e,
                    )
                    continue
                break
            except Exception as e:
                # Unknown error - don't retry, fail immediately
                last_error = e
                break

        # All retries exhausted or unknown error - return failure verdict
        error_msg = str(last_error) if last_error else "Unknown error"
        return EpicVerdict(
            passed=False,
            unmet_criteria=[],
            reasoning=f"Model verification failed: {error_msg}",
        )

    async def _get_eligible_epics(self) -> list[str]:
        """Get list of epics eligible for closure.

        Returns:
            List of epic IDs that have all children closed.
        """
        # Prefer bd epic status --eligible-only when supported; fall back to
        # full status and filter eligible_for_close.
        result = await self._runner.run_async(
            ["br", "epic", "status", "--eligible-only", "--json"],
            cwd=self.repo_path,
        )
        if not result.ok:
            result = await self._runner.run_async(
                ["br", "epic", "status", "--json"], cwd=self.repo_path
            )
        if not result.ok:
            return []
        try:
            rows = json.loads(result.stdout)
            eligible: list[str] = []
            if isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    if row.get("eligible_for_close"):
                        epic = row.get("epic") or {}
                        if isinstance(epic, dict):
                            epic_id = epic.get("id")
                            if epic_id:
                                eligible.append(str(epic_id))
            return eligible
        except json.JSONDecodeError:
            return []

    async def _is_epic_eligible(self, epic_id: str) -> bool:
        """Check if a specific epic is eligible for closure.

        An epic is eligible if all its children are closed.

        Args:
            epic_id: The epic ID to check.

        Returns:
            True if the epic is eligible for closure, False otherwise.
        """
        eligible_epics = await self._get_eligible_epics()
        return epic_id in eligible_epics

    async def _get_ineligibility_reason(self, epic_id: str) -> str | None:
        """Get reason why an epic is not eligible for closure.

        Args:
            epic_id: The epic ID to check.

        Returns:
            Human-readable reason if ineligible, None if eligible or status unavailable.
        """
        result = await self._runner.run_async(
            ["br", "epic", "status", "--json"], cwd=self.repo_path
        )
        if not result.ok:
            return None
        try:
            rows = json.loads(result.stdout)
            if not isinstance(rows, list):
                return None
            for row in rows:
                if not isinstance(row, dict):
                    continue
                epic = row.get("epic") or {}
                if isinstance(epic, dict) and epic.get("id") == epic_id:
                    if row.get("eligible_for_close") is True:
                        return None  # Actually eligible
                    try:
                        total = int(row.get("total_children") or 0)
                        closed = int(row.get("closed_children") or 0)
                        open_count = max(0, total - closed)
                    except (TypeError, ValueError):
                        return "Epic is not eligible for closure"
                    if open_count > 0:
                        return f"{open_count} of {total} child issues still open"
                    # Epic found but not eligible for unknown reason
                    return "Epic is not eligible for closure"
            # Epic not found in status output
            return f"Epic {epic_id} not found"
        except json.JSONDecodeError:
            return None

    async def verify_and_close_epic(
        self,
        epic_id: str,
        human_override: bool = False,
    ) -> EpicVerificationResult:
        """Verify and close a single specific epic if eligible.

        This method checks if the specified epic is eligible (all children closed),
        then verifies it if eligible, and closes it if verification passes.

        Args:
            epic_id: The epic ID to verify and close.
            human_override: If True, bypass verification and close directly.

        Returns:
            Summary of verification result for this epic.
        """
        return await self.verify_epic_with_options(
            epic_id,
            human_override=human_override,
            require_eligible=True,
            close_epic=True,
        )

    async def verify_epic_with_options(
        self,
        epic_id: str,
        *,
        human_override: bool = False,
        require_eligible: bool = True,
        close_epic: bool = True,
    ) -> EpicVerificationResult:
        """Verify a specific epic with optional eligibility and closing behavior.

        Args:
            epic_id: The epic ID to verify.
            human_override: If True, bypass verification and (optionally) close.
            require_eligible: If True, only run when all children are closed.
            close_epic: If True, close the epic after a passing verification.

        Returns:
            Summary of verification result for this epic.
        """
        verdicts: dict[str, EpicVerdict] = {}
        remediation_issues: list[str] = []
        passed_count = 0
        failed_count = 0
        verified_count = 0

        if require_eligible and not await self._is_epic_eligible(epic_id):
            reason = await self._get_ineligibility_reason(epic_id)
            return EpicVerificationResult(
                verified_count=0,
                passed_count=0,
                failed_count=0,
                verdicts={},
                remediation_issues_created=[],
                ineligibility_reason=reason,
            )

        if human_override:
            verified_count = 1
            if close_epic:
                closed = await self.beads.close_async(epic_id)
                if closed:
                    passed_count = 1
                    verdicts[epic_id] = EpicVerdict(
                        passed=True,
                        unmet_criteria=[],
                        reasoning="Human override - bypassed verification",
                    )
                else:
                    failed_count = 1
                    verdicts[epic_id] = EpicVerdict(
                        passed=False,
                        unmet_criteria=[],
                        reasoning="Human override close failed - epic could not be closed",
                    )
                    if self.event_sink is not None:
                        self.event_sink.on_epic_verification_failed(
                            epic_id,
                            0,
                            [],
                            reason="Epic could not be closed",
                            reviewer_type=self.reviewer_type,
                        )
            else:
                verdicts[epic_id] = EpicVerdict(
                    passed=True,
                    unmet_criteria=[],
                    reasoning="Human override - bypassed verification (no close)",
                )
                passed_count = 1

            return EpicVerificationResult(
                verified_count=verified_count,
                passed_count=passed_count,
                failed_count=failed_count,
                verdicts=verdicts,
                remediation_issues_created=remediation_issues,
            )

        async with epic_verify_lock(
            epic_id, self.repo_path, self.lock_manager, self.lock_timeout_seconds
        ) as acquired:
            if not acquired:
                return EpicVerificationResult(
                    verified_count=0,
                    passed_count=0,
                    failed_count=0,
                    verdicts={},
                    remediation_issues_created=[],
                )

            if self.event_sink is not None:
                self.event_sink.on_epic_verification_started(
                    epic_id, reviewer_type=self.reviewer_type
                )

            verdict, context = await self._verify_epic_with_context(epic_id)
            verdicts[epic_id] = verdict
            verified_count = 1

            # Create remediation/advisory issues for any unmet criteria
            blocking_ids: list[str] = []
            informational_ids: list[str] = []
            if verdict.unmet_criteria:
                (
                    blocking_ids,
                    informational_ids,
                ) = await self.create_remediation_issues(epic_id, verdict, context)
                remediation_issues.extend(blocking_ids)
                remediation_issues.extend(informational_ids)

            # Epic fails if: has P0/P1 blocking issues OR model explicitly said passed=false
            if blocking_ids or not verdict.passed:
                if blocking_ids:
                    await self.add_epic_blockers(epic_id, blocking_ids)
                failed_count = 1
                if self.event_sink is not None:
                    # If no blocking issues but still failed, include reasoning in log
                    if not blocking_ids:
                        self.event_sink.on_epic_verification_failed(
                            epic_id,
                            0,
                            [],
                            reason=verdict.reasoning or "Verification failed",
                            reviewer_type=self.reviewer_type,
                        )
                    else:
                        self.event_sink.on_epic_verification_failed(
                            epic_id,
                            len(blocking_ids),
                            blocking_ids,
                            reviewer_type=self.reviewer_type,
                        )
            else:
                # No blocking issues and verdict.passed=True - close epic if requested (may have P2/P3 advisories)
                if close_epic:
                    closed = await self.beads.close_async(epic_id)
                    if closed:
                        passed_count = 1
                        if self.event_sink is not None:
                            self.event_sink.on_epic_verification_passed(
                                epic_id,
                                reviewer_type=self.reviewer_type,
                            )
                    else:
                        # Close failed - treat as verification failure
                        failed_count = 1
                        if self.event_sink is not None:
                            self.event_sink.on_epic_verification_failed(
                                epic_id,
                                0,
                                [],
                                reason="Epic could not be closed",
                                reviewer_type=self.reviewer_type,
                            )
                else:
                    passed_count = 1
                    if self.event_sink is not None:
                        self.event_sink.on_epic_verification_passed(
                            epic_id,
                            reviewer_type=self.reviewer_type,
                        )
        return EpicVerificationResult(
            verified_count=verified_count,
            passed_count=passed_count,
            failed_count=failed_count,
            verdicts=verdicts,
            remediation_issues_created=remediation_issues,
        )

    def _truncate_text(self, text: str, max_chars: int = 4000) -> str:
        """Truncate text for issue descriptions to keep context manageable."""
        if len(text) <= max_chars:
            return text
        return f"{text[:max_chars]}\n\n[truncated]"

    def _format_remediation_context(
        self, context: EpicVerificationContext | None
    ) -> str:
        """Build a rich context block for remediation issue descriptions."""
        if context is None:
            return ""

        sections: list[str] = []
        sections.append(
            "## Epic Description / Acceptance Criteria\n"
            + self._truncate_text(context.epic_description)
        )

        commit_range = context.commit_range or "Unavailable"
        commit_list = context.commit_summary or "No commits found."
        sections.append(
            f"## Commit Scope\n- Range hint: {commit_range}\n- Commits:\n{commit_list}"
        )

        if context.child_ids:
            child_list = "\n".join(f"- {cid}" for cid in sorted(context.child_ids))
            sections.append(f"## Child Issues\n{child_list}")

        if context.blocker_ids:
            blocker_list = "\n".join(f"- {bid}" for bid in sorted(context.blocker_ids))
            sections.append(f"## Existing Blockers\n{blocker_list}")

        return "\n\n".join(sections)

    async def create_remediation_issues(
        self,
        epic_id: str,
        verdict: EpicVerdict,
        context: EpicVerificationContext | None = None,
    ) -> tuple[list[str], list[str]]:
        """Create issues for unmet criteria, return blocking and informational IDs.

        Deduplication: Checks for existing issues with matching
        epic_remediation:<epic_id>:<criterion_hash> tag before creating.

        P0/P1 issues are blocking (parented to epic, block closure).
        P2/P3 issues are informational (standalone, don't block closure).

        Args:
            epic_id: The epic ID the issues are for.
            verdict: The verification verdict with unmet criteria.
            context: Optional verification context for richer issue descriptions.

        Returns:
            Tuple of (blocking_issue_ids, informational_issue_ids).
        """
        blocking_ids: list[str] = []
        informational_ids: list[str] = []
        context_block = self._format_remediation_context(context)
        prev_issue_id: str | None = None

        for criterion in verdict.unmet_criteria:
            is_blocking = criterion.priority <= 1  # P0/P1 are blocking

            # Build dedup tag (truncate to fit 50-char label limit)
            # Format: "er:{epic_id_prefix}:{hash_prefix}" = 3 + 16 + 1 + 16 = 36 chars max
            dedup_tag = f"er:{epic_id[:16]}:{criterion.criterion_hash[:16]}"

            # Check for existing issue with this tag
            existing_id = await self.beads.find_issue_by_tag_async(dedup_tag)
            if existing_id:
                if is_blocking:
                    blocking_ids.append(existing_id)
                else:
                    informational_ids.append(existing_id)
                continue

            # Create new remediation issue
            # Sanitize criterion text for title: remove newlines, collapse whitespace
            sanitized_criterion = re.sub(r"\s+", " ", criterion.criterion.strip())
            prefix = "[Remediation]" if is_blocking else "[Advisory]"
            title = f"{prefix} {sanitized_criterion[:60]}"
            if len(sanitized_criterion) > 60:
                title = title + "..."

            priority_label = f"P{criterion.priority}"
            blocking_note = (
                "Address this criterion to unblock epic closure."
                if is_blocking
                else "This is advisory (P2/P3) and does not block epic closure."
            )

            description = f"""## Context
This issue was auto-created by epic verification for epic `{epic_id}`.

{context_block}

## Unmet Criterion
{criterion.criterion}

## Evidence
{criterion.evidence}

## Priority
{priority_label} ({"blocking" if is_blocking else "informational"})

## Resolution
{blocking_note} When complete, close this issue.
"""

            priority_str = f"P{criterion.priority}"

            # Blocking issues are parented to epic; informational are standalone
            parent_id = epic_id if is_blocking else None

            issue_id = await self.beads.create_issue_async(
                title=title,
                description=description,
                priority=priority_str,
                tags=[dedup_tag, "auto_generated"],
                parent_id=parent_id,
            )
            if issue_id:
                # Add sequential dependency on previous issue
                if prev_issue_id is not None:
                    await self.beads.add_dependency_async(issue_id, prev_issue_id)
                prev_issue_id = issue_id
                if is_blocking:
                    blocking_ids.append(issue_id)
                else:
                    informational_ids.append(issue_id)
                # Emit remediation created event
                if self.event_sink is not None:
                    self.event_sink.on_epic_remediation_created(
                        epic_id, issue_id, criterion.criterion
                    )

        return blocking_ids, informational_ids

    async def add_epic_blockers(
        self, epic_id: str, blocker_issue_ids: list[str]
    ) -> None:
        """Add issues as blockers of the epic via br dep add.

        Args:
            epic_id: The epic to block.
            blocker_issue_ids: Issues that must be resolved before epic closes.
        """
        for blocker_id in blocker_issue_ids:
            await self._runner.run_async(
                ["br", "dep", "add", epic_id, blocker_id],
                cwd=self.repo_path,
            )
