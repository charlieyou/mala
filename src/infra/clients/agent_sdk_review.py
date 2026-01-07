"""Agent SDK-based code reviewer for mala orchestrator.

This module provides an Agent SDK implementation of the CodeReviewer protocol.
Unlike the Cerberus DefaultReviewer that spawns external CLI processes, this
reviewer uses the Claude Agent SDK to run code reviews in-process with tool
access for interactive codebase exploration.

Key features:
- Uses agent sessions (not simple API calls) for interactive review
- Agent has access to git and file reading tools
- Configurable via ValidationConfig (reviewer_type: agent_sdk)
- Returns structured ReviewResult compatible with orchestrator
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.infra.clients.review_output_parser import ReviewIssue, ReviewResult
from src.infra.tools.command_runner import CommandRunner

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from src.core.protocols import MalaEventSink, SDKClientFactoryProtocol


@dataclass
class AgentSDKReviewer:
    """Code reviewer using Claude Agent SDK with tool access.

    Unlike Messages API reviewers, this uses agent sessions where the agent
    can interactively explore the codebase using git and file reading tools.

    This class conforms to the CodeReviewer protocol and provides an alternative
    to the Cerberus-based DefaultReviewer. It requires the claude_agent_sdk package.

    Attributes:
        repo_path: Path to the repository to review.
        review_agent_prompt: Agent-friendly prompt with tool usage guidance.
        sdk_client_factory: Factory for creating SDK clients (injected for testability).
        event_sink: Optional event sink for telemetry and warnings.
        model: Model short name (sonnet, opus, haiku). Default: sonnet.
        default_timeout: Default timeout in seconds. Default: 600.
    """

    repo_path: Path
    review_agent_prompt: str
    sdk_client_factory: SDKClientFactoryProtocol
    event_sink: MalaEventSink | None = None
    model: str = "sonnet"
    default_timeout: int = 600

    def overrides_disabled_setting(self) -> bool:
        """Return False; AgentSDKReviewer respects the disabled setting."""
        return False

    async def __call__(
        self,
        diff_range: str,
        context_file: Path | None = None,
        timeout: int = 600,
        claude_session_id: str | None = None,
        *,
        commit_shas: Sequence[str] | None = None,
    ) -> ReviewResult:
        """Run code review using Agent SDK.

        Creates an agent session with access to git and file reading tools,
        provides review instructions, and collects JSON output.

        Args:
            diff_range: Git diff range (e.g., "HEAD~1..HEAD").
            context_file: Optional file with implementation context.
            timeout: Maximum time for agent session (seconds).
            claude_session_id: Optional session ID for telemetry.
            commit_shas: Specific commit SHAs being reviewed.

        Returns:
            ReviewResult with verdict and findings.
        """
        # Check for empty diff (short-circuit without running agent)
        # For commit_shas, check each commit individually
        if commit_shas is not None:
            # Empty list means no commits to review
            if len(commit_shas) == 0:
                return ReviewResult(
                    passed=True,
                    issues=[],
                    parse_error=None,
                    fatal_error=False,
                    review_log_path=None,
                )
            all_empty = True
            for sha in commit_shas:
                if not await self._check_diff_empty(f"{sha}^..{sha}"):
                    all_empty = False
                    break
            if all_empty:
                return ReviewResult(
                    passed=True,
                    issues=[],
                    parse_error=None,
                    fatal_error=False,
                    review_log_path=None,
                )
        elif await self._check_diff_empty(diff_range):
            return ReviewResult(
                passed=True,
                issues=[],
                parse_error=None,
                fatal_error=False,
                review_log_path=None,
            )

        # Load context file if provided
        context = await self._load_context(context_file)

        # Create review query
        query = self._create_review_query(diff_range, context, commit_shas)

        # Run agent session
        try:
            response_text, log_path = await self._run_agent_session(
                query, timeout, claude_session_id
            )
        except TimeoutError as e:
            if self.event_sink is not None:
                self.event_sink.on_review_warning(f"Agent session timed out: {e}")
            return ReviewResult(
                passed=False,
                issues=[],
                parse_error=f"Agent session timed out: {e}",
                fatal_error=False,
                review_log_path=None,
            )
        except Exception as e:
            error_msg = f"SDK error: {e}"
            if self.event_sink is not None:
                self.event_sink.on_review_warning(error_msg)
            return ReviewResult(
                passed=False,
                issues=[],
                parse_error=error_msg,
                fatal_error=False,
                review_log_path=None,
            )

        # Parse response
        result = self._parse_response(response_text)
        # Set review_log_path from agent session
        return ReviewResult(
            passed=result.passed,
            issues=result.issues,
            parse_error=result.parse_error,
            fatal_error=result.fatal_error,
            review_log_path=log_path,
        )

    async def _check_diff_empty(self, diff_range: str) -> bool:
        """Check if diff range is empty using git diff --stat.

        Args:
            diff_range: Git diff range to check.

        Returns:
            True if the diff is empty, False otherwise.
        """
        runner = CommandRunner(cwd=self.repo_path)
        result = await runner.run_async(
            ["git", "diff", "--stat", diff_range],
            timeout=30,
        )
        # Empty diff has no output
        return result.returncode == 0 and not result.stdout.strip()

    async def _load_context(self, context_file: Path | None) -> str:
        """Load context file asynchronously.

        Args:
            context_file: Path to context file, or None.

        Returns:
            Context file content as string, or empty string if no file.
        """
        if context_file is None or not context_file.exists():
            return ""

        # Use asyncio.to_thread for non-blocking file I/O
        return await asyncio.to_thread(context_file.read_text)

    def _create_review_query(
        self,
        diff_range: str,
        context: str,
        commit_shas: Sequence[str] | None,
    ) -> str:
        """Construct review query for agent.

        Combines review instructions, diff range, context, and tool usage guidance.

        Args:
            diff_range: Git diff range to review.
            context: Context from context file.
            commit_shas: Optional specific commit SHAs.

        Returns:
            Formatted query string for the agent.
        """
        parts = [self.review_agent_prompt]

        parts.append(f"\n\n## Diff Range\n{diff_range}")

        if commit_shas:
            parts.append(f"\n\n## Specific Commits\n{', '.join(commit_shas)}")

        if context:
            parts.append(f"\n\n## Implementation Context\n{context}")

        parts.append(
            "\n\n## Output Format\n"
            "Return your review as a JSON object with the following structure:\n"
            "```json\n"
            "{\n"
            '  "consensus_verdict": "PASS" | "FAIL" | "NEEDS_WORK",\n'
            '  "aggregated_findings": [\n'
            "    {\n"
            '      "file_path": "path/to/file.py",\n'
            '      "line_start": 10,\n'
            '      "line_end": 15,\n'
            '      "priority": "P1" | "P2" | "P3" | 1 | 2 | 3,\n'
            '      "title": "Issue title",\n'
            '      "body": "Detailed description",\n'
            '      "reviewer": "agent_sdk"\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "```"
        )

        return "".join(parts)

    async def _run_agent_session(
        self,
        query: str,
        timeout: int,
        session_id: str | None,
    ) -> tuple[str, Path | None]:
        """Run agent session and collect final output.

        Creates SDK client, opens session, sends query, streams responses,
        and extracts final JSON output.

        Args:
            query: Review query to send to the agent.
            timeout: Timeout in seconds.
            session_id: Optional session ID for telemetry.

        Returns:
            Tuple of (raw response text, session log path).

        Raises:
            TimeoutError: If agent session times out.
            Exception: If SDK client creation or query fails.
        """
        # Create SDK options
        options = self.sdk_client_factory.create_options(
            cwd=str(self.repo_path),
            model=self.model,
            permission_mode="bypassPermissions",
        )

        # Create SDK client
        client = self.sdk_client_factory.create(options)

        # Wrap the session in a timeout
        return await asyncio.wait_for(
            self._execute_agent_session(client, query, session_id),
            timeout=timeout,
        )

    async def _execute_agent_session(
        self,
        client: object,
        query: str,
        session_id: str | None,
    ) -> tuple[str, Path | None]:
        """Execute the agent session and collect output.

        Args:
            client: SDK client instance.
            query: Review query to send.
            session_id: Optional session ID for telemetry.

        Returns:
            Tuple of (response text, session log path).
        """
        from pathlib import Path as PathLib  # noqa: TC003 (runtime import for log_path)

        response_text = ""
        log_path: PathLib | None = None
        result_session_id: str | None = None

        async with client:  # type: ignore[union-attr]
            await client.query(query, session_id=session_id)  # type: ignore[union-attr]

            async for msg in client.receive_response():  # type: ignore[union-attr]
                # Extract text from assistant messages
                if hasattr(msg, "subtype") and msg.subtype == "assistant":
                    content = getattr(msg, "content", None)
                    if content is not None and isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                response_text += block.get("text", "")

                # Extract session info from result message
                if hasattr(msg, "subtype") and msg.subtype == "result":
                    result_session_id = getattr(msg, "session_id", None)

        # Construct log path from session ID if available
        if result_session_id:
            log_path = self._get_session_log_path(result_session_id)

        return response_text, log_path

    def _get_session_log_path(self, session_id: str) -> Path | None:
        """Get the log path for a session ID.

        SDK logs are at ~/.claude/projects/{encoded-path}/{session_id}.jsonl

        Args:
            session_id: The session ID from ResultMessage.

        Returns:
            Path to the session log, or None if it can't be computed.
        """
        from pathlib import Path as PathLib
        import base64

        try:
            # Encode repo path like the SDK does
            repo_str = str(self.repo_path.resolve())
            encoded_path = base64.urlsafe_b64encode(repo_str.encode()).decode()

            # Construct log path
            home = PathLib.home()
            log_path = (
                home / ".claude" / "projects" / encoded_path / f"{session_id}.jsonl"
            )
            return log_path
        except Exception:
            return None

    def _parse_response(self, response_text: str) -> ReviewResult:
        """Parse JSON response into ReviewResult.

        Handles:
        - JSON extraction from markdown code blocks
        - Fallback: find first { and last }
        - Priority conversion (string "P1" -> int 1)
        - Verdict normalization (PASS/FAIL/NEEDS_WORK)

        Args:
            response_text: Raw response text from agent.

        Returns:
            Parsed ReviewResult.
        """
        if not response_text.strip():
            if self.event_sink is not None:
                self.event_sink.on_review_warning("Empty response from agent")
            return ReviewResult(
                passed=False,
                issues=[],
                parse_error="Empty response from agent",
                fatal_error=False,
                review_log_path=None,
            )

        # Try to extract JSON from markdown code blocks
        json_str = self._extract_json(response_text)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            if self.event_sink is not None:
                self.event_sink.on_review_warning(f"JSON parse error: {e}")
            return ReviewResult(
                passed=False,
                issues=[],
                parse_error=f"JSON parse error: {e}",
                fatal_error=False,
                review_log_path=None,
            )

        if not isinstance(data, dict):
            if self.event_sink is not None:
                self.event_sink.on_review_warning("Response is not a JSON object")
            return ReviewResult(
                passed=False,
                issues=[],
                parse_error="Response is not a JSON object",
                fatal_error=False,
                review_log_path=None,
            )

        # Extract verdict
        verdict = data.get("consensus_verdict", "")
        if verdict not in ("PASS", "FAIL", "NEEDS_WORK"):
            if self.event_sink is not None:
                self.event_sink.on_review_warning(f"Invalid verdict: {verdict}")
            return ReviewResult(
                passed=False,
                issues=[],
                parse_error=f"Invalid verdict: {verdict}",
                fatal_error=False,
                review_log_path=None,
            )

        passed = verdict == "PASS"

        # Parse issues
        raw_issues = data.get("aggregated_findings", [])
        if not isinstance(raw_issues, list):
            if self.event_sink is not None:
                self.event_sink.on_review_warning("aggregated_findings is not a list")
            return ReviewResult(
                passed=False,
                issues=[],
                parse_error="aggregated_findings is not a list",
                fatal_error=False,
                review_log_path=None,
            )

        issues: list[ReviewIssue] = []
        for item in raw_issues:
            if not isinstance(item, dict):
                continue

            # Extract and convert priority
            priority_raw = item.get("priority")
            priority: int | None = None
            if priority_raw is not None:
                priority = self._convert_priority(priority_raw)

            issues.append(
                ReviewIssue(
                    file=item.get("file_path", ""),
                    line_start=item.get("line_start", 0),
                    line_end=item.get("line_end", 0),
                    priority=priority,
                    title=item.get("title", ""),
                    body=item.get("body", ""),
                    reviewer=item.get("reviewer", "agent_sdk"),
                )
            )

        return ReviewResult(
            passed=passed,
            issues=issues,
            parse_error=None,
            fatal_error=False,
            review_log_path=None,
        )

    def _extract_json(self, text: str) -> str:
        """Extract JSON from text, handling markdown code blocks.

        Args:
            text: Raw text that may contain JSON in code blocks.

        Returns:
            Extracted JSON string.
        """
        # Try markdown code block first (flexible whitespace handling)
        code_block_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if code_block_match:
            return code_block_match.group(1).strip()

        # Fallback: find first { and last }
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            return text[first_brace : last_brace + 1]

        # Return original text if no JSON found
        return text

    def _convert_priority(self, priority_raw: str | int) -> int | None:
        """Convert priority string to int.

        Args:
            priority_raw: Priority as string ("P1", "P2") or int.

        Returns:
            Priority as int, or None if conversion fails.
        """
        if isinstance(priority_raw, int):
            return priority_raw

        if isinstance(priority_raw, str):
            # Handle "P1", "P2", etc.
            if priority_raw.startswith("P") and len(priority_raw) >= 2:
                try:
                    return int(priority_raw[1:])
                except ValueError:
                    return None
            # Try direct int conversion
            try:
                return int(priority_raw)
            except ValueError:
                return None

        return None
