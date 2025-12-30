"""Epic verification for ensuring epic acceptance criteria are met.

This module provides the EpicVerifier class which verifies that all code changes
made for issues under an epic collectively meet the epic's acceptance criteria
before allowing the epic to close.

Key components:
- EpicVerifier: Main verification orchestrator
- ClaudeEpicVerificationModel: Claude-based implementation of EpicVerificationModel protocol

Design principles:
- Scoped diffs: Only include commits linked to child issues (by bd-<id>: prefix)
- Large diff handling: Tiered approach (100KB default, file-summary, file-list)
- Remediation issues: Auto-create with deduplication for unmet criteria
- Human review: Flag low confidence, missing criteria, or timeout cases
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from src.models import (
    EpicVerdict,
    EpicVerificationResult,
    RetryConfig,
    UnmetCriterion,
)
from src.tools.command_runner import CommandRunner

if TYPE_CHECKING:
    from src.beads_client import BeadsClient
    from src.event_sink import MalaEventSink
    from src.protocols import EpicVerificationModel

# Spec path patterns from docs (case-insensitive)
SPEC_PATH_PATTERNS = [
    r"[Ss]ee\s+(specs/[\w/-]+\.(?:md|MD))",  # "See specs/foo/bar.md"
    r"[Ss]pec:\s*(specs/[\w/-]+\.(?:md|MD))",  # "Spec: specs/foo.md"
    r"\[(specs/[\w/-]+\.(?:md|MD))\]",  # "[specs/foo.md]"
    r"(?:^|[\s(])(specs/[\w/-]+\.(?:md|MD))(?:\s|[.,;:!?)]|$)",  # Bare "specs/foo.md" or "(specs/foo.md)"
]


async def _get_file_at_commit(
    file_path: str, repo_path: Path, commit: str = "HEAD"
) -> str | None:
    """Get file content at a specific commit using git show.

    This reads the file content from the git repository at the specified
    commit, avoiding potential discrepancies with uncommitted working tree
    changes.

    Args:
        file_path: The path to the file (relative to repo root).
        repo_path: Path to the repository.
        commit: The commit to read from (default: HEAD).

    Returns:
        The file content as a string, or None if the file doesn't exist
        at that commit or cannot be read.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "git",
            "show",
            f"{commit}:{file_path}",
            cwd=repo_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode != 0:
            return None
        return stdout.decode()
    except UnicodeDecodeError:
        # Binary file
        return None


# Default diff size limit (100KB)
DEFAULT_MAX_DIFF_SIZE_KB = 100

# File-summary mode diff limit (500KB)
FILE_SUMMARY_MAX_SIZE_KB = 500

# Maximum file size for inclusion in file-list mode (5KB)
FILE_LIST_MAX_FILE_SIZE_KB = 5

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
    prompt_path = Path(__file__).parent / "prompts" / "epic_verification.md"
    return prompt_path.read_text()


class ClaudeEpicVerificationModel:
    """Claude-based implementation of EpicVerificationModel protocol.

    Uses the shared Anthropic client factory to verify epic acceptance criteria
    against code diffs. This ensures consistent configuration, observability,
    and routing behavior across all LLM calls in the orchestrator.

    LLM Infrastructure:
        Uses create_anthropic_client() from src.anthropic_client, which provides:
        - Consistent configuration from MalaConfig (api_key, base_url)
        - Automatic Braintrust tracing when BRAINTRUST_API_KEY is set
        - MorphLLM routing when base_url is configured
    """

    # Default model for epic verification
    DEFAULT_MODEL = "claude-sonnet-4-20250514"

    def __init__(
        self,
        model: str | None = None,
        timeout_ms: int = 120000,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        """Initialize ClaudeEpicVerificationModel.

        Args:
            model: The Claude model to use for verification. Defaults to
                DEFAULT_MODEL if not specified.
            timeout_ms: Timeout for model calls in milliseconds.
            api_key: Anthropic API key. If not provided, the Anthropic client
                will use the ANTHROPIC_API_KEY environment variable.
            base_url: Optional base URL for API requests. Useful for routing
                through MorphLLM or other proxies.
        """
        self.model = model or self.DEFAULT_MODEL
        self.timeout_ms = timeout_ms
        self.api_key = api_key
        self.base_url = base_url
        self._prompt_template = _load_prompt_template()

    async def verify(
        self,
        epic_criteria: str,
        diff: str,
        spec_content: str | None,
    ) -> EpicVerdict:
        """Verify if the diff satisfies the epic's acceptance criteria.

        Args:
            epic_criteria: The epic's acceptance criteria text.
            diff: Scoped git diff of child issue commits only.
            spec_content: Optional content of linked spec file.

        Returns:
            Structured verdict with pass/fail and unmet criteria details.
        """
        from src.anthropic_client import create_anthropic_client

        prompt = self._prompt_template.format(
            epic_criteria=epic_criteria,
            spec_content=spec_content or "No spec file available.",
            diff_content=diff,
        )

        # Use the shared client factory for consistent configuration and tracing
        client = create_anthropic_client(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout_ms / 1000,
        )

        # Use asyncio.to_thread to avoid blocking the event loop during API calls
        response = await asyncio.to_thread(
            client.messages.create,
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )

        # Parse the response
        response_text = response.content[0].text
        return self._parse_verdict(response_text)

    def _parse_verdict(self, response_text: str) -> EpicVerdict:
        """Parse model response into EpicVerdict.

        Args:
            response_text: The raw model response text.

        Returns:
            Parsed EpicVerdict.
        """
        # Extract JSON from response (may be wrapped in markdown code blocks)
        json_match = re.search(
            r"```(?:json)?\s*(\{.*\})\s*```", response_text, re.DOTALL
        )
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON object
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
            unmet_criteria.append(
                UnmetCriterion(
                    criterion=criterion,
                    evidence=item.get("evidence", ""),
                    severity=item.get("severity", "major"),
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

    Gathers epic data, computes scoped diff from child issue commits,
    invokes verification model, and creates remediation issues for unmet criteria.
    """

    def __init__(
        self,
        beads: BeadsClient,
        model: EpicVerificationModel,
        repo_path: Path,
        max_diff_size_kb: int = DEFAULT_MAX_DIFF_SIZE_KB,
        retry_config: RetryConfig | None = None,
        lock_manager: object | None = None,
        event_sink: MalaEventSink | None = None,
    ):
        """Initialize EpicVerifier.

        Args:
            beads: BeadsClient for issue operations.
            model: EpicVerificationModel for verification.
            repo_path: Path to the repository.
            max_diff_size_kb: Maximum diff size in KB before truncation.
            retry_config: Configuration for retry behavior.
            lock_manager: Optional lock manager for sequential processing.
            event_sink: Optional event sink for emitting verification lifecycle events.
        """
        self.beads = beads
        self.model = model
        self.repo_path = repo_path
        self.max_diff_size_kb = max_diff_size_kb
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
            if self.lock_manager is not None:
                # Use lock context manager if available
                lock_key = f"epic_verify:{epic_id}"
                try:
                    # Try to acquire lock with timeout
                    from src.tools.locking import wait_for_lock

                    acquired = await asyncio.to_thread(
                        wait_for_lock,
                        lock_key,
                        "epic_verifier",
                        timeout_seconds=EPIC_VERIFY_LOCK_TIMEOUT_SECONDS,
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

                verdict = await self.verify_epic(epic_id)
                verdicts[epic_id] = verdict
                verified_count += 1  # Epic was actually verified (not skipped)

                if verdict.passed and verdict.confidence >= LOW_CONFIDENCE_THRESHOLD:
                    # Verification passed, close the epic
                    closed = await self.beads.close_async(epic_id)
                    if closed:
                        passed_count += 1
                        # Emit passed event
                        if self.event_sink is not None:
                            self.event_sink.on_epic_verification_passed(
                                epic_id, verdict.confidence
                            )
                    else:
                        # Close failed despite verification passing - flag for human review
                        human_review_count += 1
                        reason = "Verification passed but epic close failed"
                        review_id = await self.request_human_review(
                            epic_id,
                            reason,
                            verdict,
                        )
                        remediation_issues.append(review_id)
                        if self.event_sink is not None:
                            self.event_sink.on_epic_verification_human_review(
                                epic_id, reason, review_id
                            )
                elif verdict.confidence < LOW_CONFIDENCE_THRESHOLD:
                    # Low confidence, flag for human review
                    reason = f"Low confidence ({verdict.confidence:.2f})"
                    review_id = await self.request_human_review(
                        epic_id,
                        reason,
                        verdict,
                    )
                    remediation_issues.append(review_id)
                    human_review_count += 1
                    # Emit human review event
                    if self.event_sink is not None:
                        self.event_sink.on_epic_verification_human_review(
                            epic_id, reason, review_id
                        )
                else:
                    # Verification failed, create remediation issues
                    issue_ids = await self.create_remediation_issues(epic_id, verdict)
                    remediation_issues.extend(issue_ids)
                    await self.add_epic_blockers(epic_id, issue_ids)
                    failed_count += 1
                    # Emit failed event
                    if self.event_sink is not None:
                        self.event_sink.on_epic_verification_failed(
                            epic_id, len(verdict.unmet_criteria), issue_ids
                        )
            finally:
                # Release lock if we acquired one
                if self.lock_manager is not None and lock_key is not None:
                    try:
                        from src.tools.locking import get_lock_holder, lock_path

                        lp = lock_path(lock_key)
                        if lp.exists() and get_lock_holder(lock_key) == "epic_verifier":
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
        # Get epic description (contains acceptance criteria)
        epic_description = await self.beads.get_issue_description_async(epic_id)
        if not epic_description:
            # No acceptance criteria - flag for human review
            return EpicVerdict(
                passed=False,
                unmet_criteria=[],
                confidence=0.0,
                reasoning="No acceptance criteria found for epic",
            )

        # Get child issue IDs
        child_ids = await self.beads.get_epic_children_async(epic_id)
        if not child_ids:
            # No children - flag for human review
            return EpicVerdict(
                passed=False,
                unmet_criteria=[],
                confidence=0.0,
                reasoning="No child issues found for epic",
            )

        # Get blocker issue IDs (remediation issues from previous verification runs)
        blocker_ids = await self.beads.get_epic_blockers_async(epic_id)

        # Compute scoped diff from child and blocker commits
        diff = await self._compute_scoped_diff(child_ids, blocker_ids)
        if not diff:
            # No commits found - flag for human review
            return EpicVerdict(
                passed=False,
                unmet_criteria=[],
                confidence=0.0,
                reasoning="No commits found for child issues",
            )

        # Handle large diffs
        diff = await self._handle_large_diff(diff)

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

        # Invoke verification model with error handling
        # Per spec: timeouts/errors should trigger human review, not abort
        try:
            return await self.model.verify(epic_description, diff, spec_content)
        except Exception as e:
            # Model timeout or error - return low-confidence verdict for human review
            return EpicVerdict(
                passed=False,
                unmet_criteria=[],
                confidence=0.0,
                reasoning=f"Model verification failed: {e}",
            )

    async def _get_eligible_epics(self) -> list[str]:
        """Get list of epics eligible for closure.

        Returns:
            List of epic IDs that have all children closed.
        """
        # Use bd epic list --eligible to get epics ready for closure
        result = await self._runner.run_async(
            ["bd", "epic", "list", "--eligible", "--json"]
        )
        if not result.ok:
            return []
        try:
            epics = json.loads(result.stdout)
            if isinstance(epics, list):
                return [str(e.get("id", "")) for e in epics if e.get("id")]
            return []
        except json.JSONDecodeError:
            return []

    async def _compute_scoped_diff(
        self, child_ids: set[str], blocker_ids: set[str] | None = None
    ) -> str:
        """Compute scoped diff from child issue and blocker issue commits.

        Collects all commits matching bd-<issue_id>: prefix for each child
        issue and blocker issue (remediation issues), skips merge commits,
        and generates a combined diff.

        Args:
            child_ids: Set of child issue IDs.
            blocker_ids: Optional set of blocker issue IDs (e.g., remediation
                issues). Commits from these issues are also included in the
                diff to capture work done to address epic verification failures.

        Returns:
            Combined diff string, or empty string if no commits found.
        """
        # Combine child IDs and blocker IDs
        all_issue_ids = child_ids.copy()
        if blocker_ids:
            all_issue_ids.update(blocker_ids)

        all_commits: list[str] = []

        for issue_id in all_issue_ids:
            # Find commits matching bd-<issue_id>: prefix
            result = await self._runner.run_async(
                [
                    "git",
                    "log",
                    "--oneline",
                    "--no-merges",  # Skip merge commits
                    f"--grep=^bd-{issue_id}:",
                    "--format=%H",
                ]
            )
            if result.ok and result.stdout.strip():
                commits = result.stdout.strip().split("\n")
                all_commits.extend(commits)

        if not all_commits:
            return ""

        # Deduplicate commits while preserving order
        # A single commit may fix multiple child issues under the same epic
        unique_commits = list(dict.fromkeys(all_commits))

        # Generate combined diff for all commits
        # Use git show to get diff for each commit and concatenate
        diffs: list[str] = []
        for commit in unique_commits:
            result = await self._runner.run_async(["git", "show", "--format=", commit])
            if result.ok and result.stdout:
                diffs.append(result.stdout)

        return "\n".join(diffs)

    async def _handle_large_diff(self, diff: str) -> str:
        """Handle large diffs using tiered truncation.

        Tier 1: If under max_diff_size_kb, return as-is
        Tier 2: If under FILE_SUMMARY_MAX_SIZE_KB, use file-summary mode
        Tier 3: Otherwise, use file-list mode

        Args:
            diff: The original diff content.

        Returns:
            Possibly truncated diff content.
        """
        diff_size_kb = len(diff.encode()) / 1024

        if diff_size_kb <= self.max_diff_size_kb:
            return diff

        if diff_size_kb <= FILE_SUMMARY_MAX_SIZE_KB:
            return self._to_file_summary_mode(diff)

        return await self._to_file_list_mode(diff)

    def _to_file_summary_mode(self, diff: str) -> str:
        """Convert diff to file-summary mode.

        Lists files changed with truncated diffs (first 50 lines per file).

        Args:
            diff: The original diff content.

        Returns:
            File-summary mode diff.
        """
        lines = diff.split("\n")
        result_lines: list[str] = ["# Diff truncated - file-summary mode", ""]

        current_file: str | None = None
        file_lines: list[str] = []

        for line in lines:
            if line.startswith("diff --git"):
                # Save previous file
                if current_file and file_lines:
                    result_lines.append(f"## {current_file}")
                    result_lines.extend(file_lines[:50])
                    if len(file_lines) > 50:
                        result_lines.append(f"... ({len(file_lines) - 50} more lines)")
                    result_lines.append("")

                # Start new file
                match = re.search(r"diff --git a/(.+?) b/", line)
                current_file = match.group(1) if match else "unknown"
                file_lines = [line]
            else:
                file_lines.append(line)

        # Save last file
        if current_file and file_lines:
            result_lines.append(f"## {current_file}")
            result_lines.extend(file_lines[:50])
            if len(file_lines) > 50:
                result_lines.append(f"... ({len(file_lines) - 50} more lines)")

        return "\n".join(result_lines)

    async def _to_file_list_mode(self, diff: str) -> str:
        """Convert diff to file-list mode.

        Lists changed files only, with contents for files under 5KB.
        File contents are read from HEAD commit to ensure consistency
        with the scoped diff.

        Args:
            diff: The original diff content.

        Returns:
            File-list mode summary.
        """
        lines = diff.split("\n")
        files_changed: list[str] = []

        for line in lines:
            if line.startswith("diff --git"):
                match = re.search(r"diff --git a/(.+?) b/", line)
                if match:
                    files_changed.append(match.group(1))

        result_lines = [
            "# Diff too large - file-list mode",
            "",
            f"Changed files ({len(files_changed)}):",
        ]
        for f in files_changed:
            result_lines.append(f"  - {f}")

        result_lines.append("")
        result_lines.append("Small file contents (under 5KB):")
        result_lines.append("")

        for f in files_changed:
            # Read file content from HEAD commit to ensure consistency with
            # the scoped diff (avoids issues with uncommitted working tree changes)
            file_content = await _get_file_at_commit(f, self.repo_path)
            if file_content is not None:
                if len(file_content.encode()) <= FILE_LIST_MAX_FILE_SIZE_KB * 1024:
                    result_lines.append(f"### {f}")
                    result_lines.append("```")
                    result_lines.append(file_content)
                    result_lines.append("```")
                    result_lines.append("")

        return "\n".join(result_lines)

    async def create_remediation_issues(
        self, epic_id: str, verdict: EpicVerdict
    ) -> list[str]:
        """Create issues for unmet criteria, return their IDs.

        Deduplication: Checks for existing issues with matching
        epic_remediation:<epic_id>:<criterion_hash> tag before creating.

        Args:
            epic_id: The epic ID the issues are for.
            verdict: The verification verdict with unmet criteria.

        Returns:
            List of created (or existing) issue IDs.
        """
        issue_ids: list[str] = []

        for criterion in verdict.unmet_criteria:
            # Build dedup tag
            dedup_tag = f"epic_remediation:{epic_id}:{criterion.criterion_hash}"

            # Check for existing issue with this tag
            existing_id = await self._find_issue_by_tag(dedup_tag)
            if existing_id:
                issue_ids.append(existing_id)
                continue

            # Create new remediation issue
            # Sanitize criterion text for title: remove newlines, collapse whitespace
            sanitized_criterion = re.sub(r"\s+", " ", criterion.criterion.strip())
            title = f"[Remediation] {sanitized_criterion[:60]}"
            if len(sanitized_criterion) > 60:
                title = title + "..."

            description = f"""## Context
This issue was auto-created by epic verification for epic `{epic_id}`.

## Unmet Criterion
{criterion.criterion}

## Evidence
{criterion.evidence}

## Severity
{criterion.severity}

## Resolution
Address this criterion to unblock epic closure. When complete, close this issue.
"""

            # Determine priority based on severity
            priority = await self._get_adjusted_priority(epic_id, criterion.severity)

            issue_id = await self._create_issue(
                title=title,
                description=description,
                priority=priority,
                tags=[dedup_tag, "auto_generated"],
            )
            if issue_id:
                issue_ids.append(issue_id)
                # Emit remediation created event
                if self.event_sink is not None:
                    self.event_sink.on_epic_remediation_created(
                        epic_id, issue_id, criterion.criterion
                    )

        return issue_ids

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

        issue_id = await self._create_issue(
            title=title,
            description=description,
            priority="P1",
            tags=["epic_human_review", f"epic_id:{epic_id}"],
        )

        if issue_id:
            await self.add_epic_blockers(epic_id, [issue_id])

        return issue_id or ""

    async def _find_issue_by_tag(self, tag: str) -> str | None:
        """Find an existing issue with the given tag.

        Args:
            tag: The tag to search for.

        Returns:
            Issue ID if found, None otherwise.
        """
        result = await self._runner.run_async(["bd", "list", "--tag", tag, "--json"])
        if not result.ok:
            return None
        try:
            issues = json.loads(result.stdout)
            if isinstance(issues, list) and issues:
                # Return first matching issue (should be only one due to dedup)
                return str(issues[0].get("id", ""))
            return None
        except json.JSONDecodeError:
            return None

    async def _get_adjusted_priority(self, epic_id: str, severity: str) -> str:
        """Get priority for remediation issue based on epic priority and severity.

        Args:
            epic_id: The epic ID to get base priority from.
            severity: The severity of the unmet criterion.

        Returns:
            Priority string (P1, P2, P3, etc.)
        """
        # Get epic priority
        result = await self._runner.run_async(["bd", "show", epic_id, "--json"])
        base_priority = 2  # Default P2

        if result.ok:
            try:
                data = json.loads(result.stdout)
                if isinstance(data, dict):
                    priority_str = data.get("priority", "P2")
                    if isinstance(priority_str, str) and priority_str.startswith("P"):
                        base_priority = int(priority_str[1:])
            except (json.JSONDecodeError, ValueError):
                pass

        # Adjust based on severity
        if severity == "critical":
            adjusted = base_priority
        elif severity == "major":
            adjusted = base_priority + 1
        else:  # minor
            adjusted = base_priority + 2

        # Cap at P5
        adjusted = min(adjusted, 5)
        return f"P{adjusted}"

    async def _create_issue(
        self,
        title: str,
        description: str,
        priority: str,
        tags: list[str],
    ) -> str | None:
        """Create a new issue via bd CLI.

        Args:
            title: Issue title.
            description: Issue description.
            priority: Priority string (P1, P2, etc.)
            tags: List of tags to apply.

        Returns:
            Created issue ID, or None on failure.
        """
        cmd = [
            "bd",
            "issue",
            "create",
            "--title",
            title,
            "--description",
            description,
            "--priority",
            priority,
        ]
        for tag in tags:
            cmd.extend(["--tag", tag])

        result = await self._runner.run_async(cmd)
        if not result.ok:
            return None

        # Parse issue ID from output (typically "Created issue: <id>")
        match = re.search(r"Created issue:\s*(\S+)", result.stdout)
        if match:
            return match.group(1)

        # Try parsing as JSON if the CLI returns JSON
        try:
            data = json.loads(result.stdout)
            if isinstance(data, dict):
                return str(data.get("id", ""))
        except json.JSONDecodeError:
            pass

        return None
