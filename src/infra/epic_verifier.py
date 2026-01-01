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
- Human review: Flag low confidence, missing criteria, or timeout cases
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.core.models import (
    EpicVerdict,
    EpicVerificationResult,
    RetryConfig,
    UnmetCriterion,
)
from src.infra.tools.command_runner import CommandRunner

if TYPE_CHECKING:
    from src.infra.clients.beads_client import BeadsClient
    from src.infra.io.event_sink import MalaEventSink
    from src.core.protocols import EpicVerificationModel

# Spec path patterns from docs (case-insensitive)
SPEC_PATH_PATTERNS = [
    r"[Ss]ee\s+(specs/[\w/-]+\.(?:md|MD))",  # "See specs/foo/bar.md"
    r"[Ss]pec:\s*(specs/[\w/-]+\.(?:md|MD))",  # "Spec: specs/foo.md"
    r"\[(specs/[\w/-]+\.(?:md|MD))\]",  # "[specs/foo.md]"
    r"(?:^|[\s(])(specs/[\w/-]+\.(?:md|MD))(?:\s|[.,;:!?)]|$)",  # Bare "specs/foo.md" or "(specs/foo.md)"
]


# Low confidence threshold for human review
LOW_CONFIDENCE_THRESHOLD = 0.5

# Lock timeout for epic verification (5 minutes)
EPIC_VERIFY_LOCK_TIMEOUT_SECONDS = 300


def _compute_criterion_hash(criterion: str) -> str:
    """Compute SHA256 hash of criterion text for deduplication."""
    return hashlib.sha256(criterion.encode()).hexdigest()


def extract_spec_paths(text: str) -> list[str]:
    """Extract spec file paths from text using documented patterns.

    Args:
        text: The text to search for spec paths.

    Returns:
        List of unique spec paths found.
    """
    paths: list[str] = []
    for pattern in SPEC_PATH_PATTERNS:
        matches = re.findall(pattern, text, re.MULTILINE)
        paths.extend(matches)
    # Deduplicate while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for path in paths:
        if path not in seen:
            seen.add(path)
            result.append(path)
    return result


def _load_prompt_template() -> str:
    """Load the epic verification prompt template."""
    # Navigate from src/infra/ to src/prompts/
    prompt_path = Path(__file__).parent.parent / "prompts" / "epic_verification.md"
    template = prompt_path.read_text()
    escaped = template.replace("{", "{{").replace("}", "}}")
    for key in ("epic_criteria", "spec_content", "commit_range", "commit_list"):
        escaped = escaped.replace(f"{{{{{key}}}}}", f"{{{key}}}")
    return escaped


@dataclass
class EpicVerificationContext:
    """Context captured during verification for richer remediation issues."""

    epic_description: str
    spec_content: str | None
    child_ids: set[str]
    blocker_ids: set[str]
    commit_shas: list[str]
    commit_range: str | None
    commit_summary: str


def _extract_json_from_code_blocks(text: str) -> str | None:
    """Extract JSON content from markdown code blocks.

    This function properly handles responses with multiple code blocks by
    extracting each code block individually and returning the first one
    that contains valid JSON (i.e., starts with '{').

    Args:
        text: The raw model response text that may contain markdown code blocks.

    Returns:
        The JSON string extracted from the first JSON code block, or None if
        no valid JSON code block is found.
    """
    # Pattern to match individual code blocks non-greedily
    # Matches: ```json ... ``` or ``` ... ```
    # The .*? is non-greedy and DOTALL is NOT used, so we need to handle newlines
    # Use a pattern that matches until the closing ```
    code_block_pattern = r"```(?:json)?\s*\n?([\s\S]*?)```"

    matches = re.finditer(code_block_pattern, text)

    for match in matches:
        content = match.group(1).strip()
        # Check if this looks like JSON (starts with '{')
        if content.startswith("{"):
            return content

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
        epic_criteria: str,
        commit_range: str,
        commit_list: str,
        spec_content: str | None,
    ) -> EpicVerdict:
        """Verify if the commit scope satisfies the epic's acceptance criteria.

        Args:
            epic_criteria: The epic's acceptance criteria text.
            commit_range: Commit range hint covering child issue commits.
            commit_list: Authoritative list of commit SHAs to inspect.
            spec_content: Optional content of linked spec file.

        Returns:
            Structured verdict with pass/fail and unmet criteria details.

        """
        prompt = self._prompt_template.format(
            epic_criteria=epic_criteria,
            spec_content=spec_content or "No spec file available.",
            commit_range=commit_range or "Unavailable.",
            commit_list=commit_list or "No commits found.",
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
                    "passed": False,
                    "confidence": 0.0,
                    "reasoning": (
                        "claude_agent_sdk is not installed; epic verification "
                        f"requires the agent SDK ({exc})"
                    ),
                    "unmet_criteria": [],
                }
            )

        options = ClaudeAgentOptions(
            cwd=str(self.repo_path),
            permission_mode="bypassPermissions",
            model=self.model,
            system_prompt={"type": "preset", "preset": "claude_code"},
            setting_sources=["project", "user"],
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
            # If parsing fails, return a low-confidence failure
            return EpicVerdict(
                passed=False,
                unmet_criteria=[],
                confidence=0.0,
                reasoning=f"Failed to parse model response: {response_text[:500]}",
            )

        unmet_criteria = []
        for item in data.get("unmet_criteria", []):
            criterion = item.get("criterion", "")
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
                    evidence=item.get("evidence", ""),
                    priority=priority,
                    criterion_hash=_compute_criterion_hash(criterion),
                )
            )

        return EpicVerdict(
            passed=data.get("passed", False),
            unmet_criteria=unmet_criteria,
            confidence=data.get("confidence", 0.5),
            reasoning=data.get("reasoning", ""),
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
        retry_config: RetryConfig | None = None,
        lock_manager: object | None = None,
        event_sink: MalaEventSink | None = None,
    ):
        """Initialize EpicVerifier.

        Args:
            beads: BeadsClient for issue operations.
            model: EpicVerificationModel for verification.
            repo_path: Path to the repository.
            retry_config: Configuration for retry behavior.
            lock_manager: Optional lock manager for sequential processing.
            event_sink: Optional event sink for emitting verification lifecycle events.
        """
        self.beads = beads
        self.model = model
        self.repo_path = repo_path
        self.retry_config = retry_config or RetryConfig()
        self.lock_manager = lock_manager
        self.event_sink = event_sink
        self._runner = CommandRunner(cwd=repo_path)

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

        verdicts: dict[str, EpicVerdict] = {}
        remediation_issues: list[str] = []
        passed_count = 0
        failed_count = 0
        human_review_count = 0
        verified_count = 0  # Track actually verified epics (not skipped due to lock)

        for epic_id in eligible_epics:
            # Human override bypasses verification
            if epic_id in human_override_epic_ids:
                closed = await self.beads.close_async(epic_id)
                verified_count += 1  # Human override counts as verified
                if closed:
                    passed_count += 1
                    verdicts[epic_id] = EpicVerdict(
                        passed=True,
                        unmet_criteria=[],
                        confidence=1.0,
                        reasoning="Human override - bypassed verification",
                    )
                else:
                    # Close failed - flag for human review
                    human_review_count += 1
                    reason = "Human override close failed - epic could not be closed"
                    review_id = await self.request_human_review(
                        epic_id,
                        reason,
                        None,
                    )
                    remediation_issues.append(review_id)
                    verdicts[epic_id] = EpicVerdict(
                        passed=False,
                        unmet_criteria=[],
                        confidence=0.0,
                        reasoning=reason,
                    )
                    if self.event_sink is not None:
                        self.event_sink.on_epic_verification_human_review(
                            epic_id, reason, review_id
                        )
                continue

            # Acquire per-epic lock for sequential processing
            lock_key: str | None = None
            lock_agent_id: str | None = None
            if self.lock_manager is not None:
                # Use lock context manager if available
                lock_key = f"epic_verify:{epic_id}"
                # Use unique agent_id per process to ensure mutual exclusion
                lock_agent_id = f"epic_verifier_{os.getpid()}"
                try:
                    # Try to acquire lock with timeout
                    from src.infra.tools.locking import wait_for_lock

                    acquired = await asyncio.to_thread(
                        wait_for_lock,
                        lock_key,
                        lock_agent_id,
                        str(self.repo_path),  # repo_namespace for consistent lock keys
                        EPIC_VERIFY_LOCK_TIMEOUT_SECONDS,
                    )
                    if not acquired:
                        # Could not acquire lock, skip this epic
                        continue
                except ImportError:
                    pass  # No locking available

            try:
                # Emit verification started event
                if self.event_sink is not None:
                    self.event_sink.on_epic_verification_started(epic_id)

                verdict, context = await self._verify_epic_with_context(epic_id)
                verdicts[epic_id] = verdict
                verified_count += 1  # Epic was actually verified (not skipped)

                if verdict.confidence < LOW_CONFIDENCE_THRESHOLD:
                    # Low confidence, flag for human review
                    reason = f"Low confidence ({verdict.confidence:.2f})"
                    review_id = await self.request_human_review(
                        epic_id,
                        reason,
                        verdict,
                    )
                    remediation_issues.append(review_id)
                    human_review_count += 1
                    if self.event_sink is not None:
                        self.event_sink.on_epic_verification_human_review(
                            epic_id, reason, review_id
                        )
                    continue

                # Create remediation/advisory issues for any unmet criteria
                blocking_ids: list[str] = []
                informational_ids: list[str] = []
                if verdict.unmet_criteria:
                    blocking_ids, informational_ids = await self.create_remediation_issues(
                        epic_id, verdict, context
                    )
                    remediation_issues.extend(blocking_ids)
                    remediation_issues.extend(informational_ids)

                # Epic fails if: has P0/P1 blocking issues OR model explicitly said passed=false
                if blocking_ids or not verdict.passed:
                    if blocking_ids:
                        await self.add_epic_blockers(epic_id, blocking_ids)
                    failed_count += 1
                    if self.event_sink is not None:
                        self.event_sink.on_epic_verification_failed(
                            epic_id, len(blocking_ids), blocking_ids
                        )
                else:
                    # No blocking issues and verdict.passed=True - close epic (may have P2/P3 advisories)
                    closed = await self.beads.close_async(epic_id)
                    if closed:
                        passed_count += 1
                        if self.event_sink is not None:
                            self.event_sink.on_epic_verification_passed(
                                epic_id, verdict.confidence
                            )
                    else:
                        # Close failed - flag for human review
                        human_review_count += 1
                        reason = "Verification passed but epic close failed"
                        review_id = await self.request_human_review(
                            epic_id, reason, verdict
                        )
                        remediation_issues.append(review_id)
                        if self.event_sink is not None:
                            self.event_sink.on_epic_verification_human_review(
                                epic_id, reason, review_id
                            )
            finally:
                # Release lock if we acquired one
                if self.lock_manager is not None and lock_key is not None:
                    try:
                        from src.infra.tools.locking import get_lock_holder, lock_path

                        repo_ns = str(self.repo_path)
                        lp = lock_path(lock_key, repo_ns)
                        if (
                            lp.exists()
                            and get_lock_holder(lock_key, repo_ns) == lock_agent_id
                        ):
                            lp.unlink(missing_ok=True)
                    except ImportError:
                        pass

        return EpicVerificationResult(
            verified_count=verified_count,
            passed_count=passed_count,
            failed_count=failed_count,
            human_review_count=human_review_count,
            verdicts=verdicts,
            remediation_issues_created=remediation_issues,
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
                    confidence=0.0,
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
                    confidence=0.0,
                    reasoning="No child issues found for epic",
                ),
                None,
            )

        # Get blocker issue IDs (remediation issues from previous verification runs)
        blocker_ids = await self.beads.get_epic_blockers_async(epic_id) or set()

        # Compute scoped commit list from child and blocker commits
        commit_shas = await self._compute_scoped_commits(child_ids, blocker_ids)
        if not commit_shas:
            return (
                EpicVerdict(
                    passed=False,
                    unmet_criteria=[],
                    confidence=0.0,
                    reasoning="No commits found for child issues",
                ),
                None,
            )

        commit_range = await self._summarize_commit_range(commit_shas)
        commit_summary = await self._format_commit_summary(commit_shas)

        # Extract and load spec content if referenced
        spec_paths = extract_spec_paths(epic_description)
        spec_content = None
        for spec_path in spec_paths:
            full_path = self.repo_path / spec_path
            if full_path.exists():
                try:
                    spec_content = await asyncio.to_thread(full_path.read_text)
                    break  # Use first found spec
                except OSError:
                    pass

        context = EpicVerificationContext(
            epic_description=epic_description,
            spec_content=spec_content,
            child_ids=child_ids,
            blocker_ids=blocker_ids,
            commit_shas=commit_shas,
            commit_range=commit_range,
            commit_summary=commit_summary,
        )

        # Invoke verification model with error handling
        # Per spec: timeouts/errors should trigger human review, not abort
        try:
            result = await self.model.verify(
                epic_description, commit_range or "", commit_summary, spec_content
            )
            verdict = EpicVerdict(
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
                confidence=result.confidence,
                reasoning=result.reasoning,
            )
        except Exception as e:
            verdict = EpicVerdict(
                passed=False,
                unmet_criteria=[],
                confidence=0.0,
                reasoning=f"Model verification failed: {e}",
            )

        return verdict, context

    async def _get_eligible_epics(self) -> list[str]:
        """Get list of epics eligible for closure.

        Returns:
            List of epic IDs that have all children closed.
        """
        # Prefer bd epic status --eligible-only when supported; fall back to
        # full status and filter eligible_for_close.
        result = await self._runner.run_async(
            ["bd", "epic", "status", "--eligible-only", "--json"]
        )
        if not result.ok:
            result = await self._runner.run_async(["bd", "epic", "status", "--json"])
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
        human_review_count = 0
        verified_count = 0

        if require_eligible and not await self._is_epic_eligible(epic_id):
            return EpicVerificationResult(
                verified_count=0,
                passed_count=0,
                failed_count=0,
                human_review_count=0,
                verdicts={},
                remediation_issues_created=[],
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
                        confidence=1.0,
                        reasoning="Human override - bypassed verification",
                    )
                else:
                    human_review_count = 1
                    reason = "Human override close failed - epic could not be closed"
                    review_id = await self.request_human_review(epic_id, reason, None)
                    remediation_issues.append(review_id)
                    verdicts[epic_id] = EpicVerdict(
                        passed=False,
                        unmet_criteria=[],
                        confidence=0.0,
                        reasoning=reason,
                    )
                    if self.event_sink is not None:
                        self.event_sink.on_epic_verification_human_review(
                            epic_id, reason, review_id
                        )
            else:
                verdicts[epic_id] = EpicVerdict(
                    passed=True,
                    unmet_criteria=[],
                    confidence=1.0,
                    reasoning="Human override - bypassed verification (no close)",
                )
                passed_count = 1

            return EpicVerificationResult(
                verified_count=verified_count,
                passed_count=passed_count,
                failed_count=failed_count,
                human_review_count=human_review_count,
                verdicts=verdicts,
                remediation_issues_created=remediation_issues,
            )

        lock_key: str | None = None
        lock_agent_id: str | None = None
        if self.lock_manager is not None:
            lock_key = f"epic_verify:{epic_id}"
            # Use unique agent_id per process to ensure mutual exclusion
            lock_agent_id = f"epic_verifier_{os.getpid()}"
            try:
                from src.infra.tools.locking import wait_for_lock

                acquired = await asyncio.to_thread(
                    wait_for_lock,
                    lock_key,
                    lock_agent_id,
                    str(self.repo_path),  # repo_namespace for consistent lock keys
                    EPIC_VERIFY_LOCK_TIMEOUT_SECONDS,
                )
                if not acquired:
                    return EpicVerificationResult(
                        verified_count=0,
                        passed_count=0,
                        failed_count=0,
                        human_review_count=0,
                        verdicts={},
                        remediation_issues_created=[],
                    )
            except ImportError:
                pass

        try:
            if self.event_sink is not None:
                self.event_sink.on_epic_verification_started(epic_id)

            verdict, context = await self._verify_epic_with_context(epic_id)
            verdicts[epic_id] = verdict
            verified_count = 1

            if verdict.confidence < LOW_CONFIDENCE_THRESHOLD:
                reason = f"Low confidence ({verdict.confidence:.2f})"
                review_id = await self.request_human_review(epic_id, reason, verdict)
                remediation_issues.append(review_id)
                human_review_count = 1
                if self.event_sink is not None:
                    self.event_sink.on_epic_verification_human_review(
                        epic_id, reason, review_id
                    )
            else:
                # Create remediation/advisory issues for any unmet criteria
                blocking_ids: list[str] = []
                informational_ids: list[str] = []
                if verdict.unmet_criteria:
                    blocking_ids, informational_ids = await self.create_remediation_issues(
                        epic_id, verdict, context
                    )
                    remediation_issues.extend(blocking_ids)
                    remediation_issues.extend(informational_ids)

                # Epic fails if: has P0/P1 blocking issues OR model explicitly said passed=false
                if blocking_ids or not verdict.passed:
                    if blocking_ids:
                        await self.add_epic_blockers(epic_id, blocking_ids)
                    failed_count = 1
                    if self.event_sink is not None:
                        self.event_sink.on_epic_verification_failed(
                            epic_id, len(blocking_ids), blocking_ids
                        )
                else:
                    # No blocking issues and verdict.passed=True - close epic if requested (may have P2/P3 advisories)
                    if close_epic:
                        closed = await self.beads.close_async(epic_id)
                        if closed:
                            passed_count = 1
                            if self.event_sink is not None:
                                self.event_sink.on_epic_verification_passed(
                                    epic_id, verdict.confidence
                                )
                        else:
                            human_review_count = 1
                            reason = "Verification passed but epic close failed"
                            review_id = await self.request_human_review(
                                epic_id, reason, verdict
                            )
                            remediation_issues.append(review_id)
                            if self.event_sink is not None:
                                self.event_sink.on_epic_verification_human_review(
                                    epic_id, reason, review_id
                                )
                    else:
                        passed_count = 1
                        if self.event_sink is not None:
                            self.event_sink.on_epic_verification_passed(
                                epic_id, verdict.confidence
                            )
        finally:
            if self.lock_manager is not None and lock_key is not None:
                try:
                    from src.infra.tools.locking import get_lock_holder, lock_path

                    repo_ns = str(self.repo_path)
                    lp = lock_path(lock_key, repo_ns)
                    if (
                        lp.exists()
                        and get_lock_holder(lock_key, repo_ns) == lock_agent_id
                    ):
                        lp.unlink(missing_ok=True)
                except ImportError:
                    pass

        return EpicVerificationResult(
            verified_count=verified_count,
            passed_count=passed_count,
            failed_count=failed_count,
            human_review_count=human_review_count,
            verdicts=verdicts,
            remediation_issues_created=remediation_issues,
        )

    async def _compute_scoped_commits(
        self, child_ids: set[str], blocker_ids: set[str] | None = None
    ) -> list[str]:
        """Compute scoped commit list from child and blocker issue commits.

        Collects all commits matching bd-<issue_id>: prefix for each child
        issue and blocker issue (remediation issues), skips merge commits,
        and returns a deduplicated list of commit SHAs.

        Args:
            child_ids: Set of child issue IDs.
            blocker_ids: Optional set of blocker issue IDs (e.g., remediation
                issues). Commits from these issues are also included in the
                scope to capture work done to address epic verification failures.

        Returns:
            List of commit SHAs, or empty list if no commits found.
        """
        # Combine child IDs and blocker IDs
        all_issue_ids = child_ids.copy()
        if blocker_ids:
            all_issue_ids.update(blocker_ids)

        all_commits: list[str] = []

        for issue_id in all_issue_ids:
            # Find commits matching bd-<issue_id>: prefix
            # Note: Intentionally searches only HEAD branch - child issue commits
            # should be merged before epic verification runs
            result = await self._runner.run_async(
                [
                    "git",
                    "log",
                    "--oneline",
                    "--no-merges",  # Skip merge commits
                    f"--grep=bd-{issue_id}:",  # Substring match, not start-of-line
                    "--format=%H",
                ]
            )
            if result.ok and result.stdout.strip():
                commits = result.stdout.strip().split("\n")
                all_commits.extend(commits)

        if not all_commits:
            return []

        # Deduplicate commits while preserving order
        # A single commit may fix multiple child issues under the same epic
        unique_commits = list(dict.fromkeys(all_commits))

        return unique_commits

    async def _summarize_commit_range(self, commits: list[str]) -> str | None:
        """Summarize commit range from a list of commit SHAs.

        Returns a git range hint covering all commits, or None if timestamps
        cannot be retrieved. The agent still receives the authoritative commit
        list even when this returns None.

        Note: For non-linear histories, the range may include unrelated commits.
        The authoritative commit list should be used for precise scoping.
        """
        if not commits:
            return None
        timestamps: list[tuple[int, str]] = []
        for commit in commits:
            result = await self._runner.run_async(
                ["git", "show", "-s", "--format=%ct", commit]
            )
            if result.ok and result.stdout.strip().isdigit():
                timestamps.append((int(result.stdout.strip()), commit))
        # Only provide a range hint if we have timestamps for all commits.
        # The commits list is aggregated from multiple git log calls over an
        # unordered set of issue IDs, so we cannot assume any ordering without
        # timestamps. When timestamps are unavailable, return None and rely on
        # the authoritative commit list instead.
        if len(timestamps) < len(commits):
            return None
        timestamps.sort(key=lambda item: item[0])
        base = timestamps[0][1]
        tip = timestamps[-1][1]
        if base == tip:
            return base
        # Check if base has a parent (not a root commit) before using base^
        parent_check = await self._runner.run_async(
            ["git", "rev-parse", "--verify", f"{base}^", "--"]
        )
        if parent_check.ok:
            # base has a parent, use base^..tip for inclusive range
            return f"{base}^..{tip}"
        else:
            # base is a root commit; return valid range syntax only.
            # The agent uses the authoritative commit list for precise scoping.
            # Note: base..tip excludes base, so agent should inspect base separately.
            return f"{base}..{tip}"

    async def _format_commit_summary(
        self, commits: list[str], max_commits: int = 50
    ) -> str:
        """Format commit list with SHA and subject for prompts/issues.

        Args:
            commits: List of commit SHAs to format.
            max_commits: Maximum number of commits to include (default 50).
                Prevents excessively large prompts/issue bodies.

        Returns:
            Formatted commit summary string.
        """
        if not commits:
            return "No commits found."

        truncated = len(commits) > max_commits
        display_commits = commits[:max_commits] if truncated else commits

        lines: list[str] = []
        for commit in display_commits:
            result = await self._runner.run_async(
                ["git", "show", "-s", "--format=%H %s", commit]
            )
            summary = result.stdout.strip() if result.ok else ""
            if summary:
                lines.append(f"- {summary}")
            else:
                lines.append(f"- {commit}")

        if truncated:
            lines.append(f"\n[... {len(commits) - max_commits} more commits omitted]")

        return "\n".join(lines)

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

        if context.spec_content:
            sections.append(
                "## Spec Content\n" + self._truncate_text(context.spec_content)
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

        for criterion in verdict.unmet_criteria:
            is_blocking = criterion.priority <= 1  # P0/P1 are blocking

            # Build dedup tag
            dedup_tag = f"epic_remediation:{epic_id}:{criterion.criterion_hash}"

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
{priority_label} ({'blocking' if is_blocking else 'informational'})

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
        """Add issues as blockers of the epic via bd dep add.

        Args:
            epic_id: The epic to block.
            blocker_issue_ids: Issues that must be resolved before epic closes.
        """
        for blocker_id in blocker_issue_ids:
            await self._runner.run_async(
                ["bd", "dep", "add", epic_id, "--blocked-by", blocker_id]
            )

    async def request_human_review(
        self, epic_id: str, reason: str, verdict: EpicVerdict | None = None
    ) -> str:
        """Create a human review issue that blocks the epic.

        Args:
            epic_id: The epic requiring review.
            reason: Why human review is needed.
            verdict: Optional verdict details.

        Returns:
            Issue ID of the created review request.
        """
        title = f"[Human Review] Epic {epic_id} requires manual verification"

        description = f"""## Context
This epic has been flagged for human review.

## Reason
{reason}

## Epic ID
{epic_id}
"""
        if verdict:
            description += f"""
## Verification Details
- Confidence: {verdict.confidence:.2f}
- Reasoning: {verdict.reasoning}
"""

        description += f"""
## Resolution
Review the epic and its child issues manually. When satisfied, close this issue
to unblock the epic, or use `--epic-override {epic_id}` to force closure.
"""

        issue_id = await self.beads.create_issue_async(
            title=title,
            description=description,
            priority="P1",
            tags=["epic_human_review", f"epic_id:{epic_id}"],
            parent_id=epic_id,
        )

        if issue_id:
            await self.add_epic_blockers(epic_id, [issue_id])

        return issue_id or ""
