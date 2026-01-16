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

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from src.core.protocols.events import MalaEventSink
    from src.core.protocols.sdk import SDKClientFactoryProtocol


class StructuredOutputError(RuntimeError):
    """Raised when the SDK reports a structured output failure."""

    def __init__(self, subtype: str) -> None:
        super().__init__(f"Structured output error: {subtype}")
        self.subtype = subtype


# Structured output schema for review results (matches Cerberus aggregate shape).
_REVIEW_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "consensus_verdict": {
            "type": "string",
            "enum": ["PASS", "FAIL", "NEEDS_WORK"],
        },
        "aggregated_findings": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                    "priority": {"type": "integer", "enum": [0, 1, 2, 3]},
                    "file_path": {"type": ["string", "null"]},
                    "line_start": {"type": ["integer", "null"]},
                    "line_end": {"type": ["integer", "null"]},
                    "reviewer": {"type": "string"},
                },
                "required": [
                    "title",
                    "body",
                    "priority",
                    "file_path",
                    "line_start",
                    "line_end",
                    "reviewer",
                ],
            },
        },
    },
    "required": ["consensus_verdict", "aggregated_findings"],
}

_REVIEW_OUTPUT_FORMAT: dict[str, object] = {
    "type": "json_schema",
    "schema": _REVIEW_OUTPUT_SCHEMA,
}


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
        context_file: Path | None = None,
        timeout: int | None = None,
        claude_session_id: str | None = None,
        author_context: str | None = None,
        *,
        commit_shas: Sequence[str],
        interrupt_event: asyncio.Event | None = None,
    ) -> ReviewResult:
        """Run code review using Agent SDK.

        Creates an agent session with access to git and file reading tools,
        provides review instructions, and collects JSON output.

        Args:
            context_file: Optional file with implementation context.
            timeout: Maximum time for agent session (seconds). Uses default_timeout if None.
            claude_session_id: Optional session ID for telemetry.
            commit_shas: Specific commit SHAs being reviewed.
            interrupt_event: Optional event to check for SIGINT interruption.

        Returns:
            ReviewResult with verdict and findings.
        """
        from src.infra.sigint_guard import InterruptGuard

        guard = InterruptGuard(interrupt_event)

        # Check for early interrupt before starting
        if guard.is_interrupted():
            logger.info("Review interrupted before starting")
            return ReviewResult(
                passed=False,
                issues=[],
                parse_error=None,
                fatal_error=False,
                review_log_path=None,
                interrupted=True,
            )

        # Use instance default_timeout if not specified
        effective_timeout = timeout if timeout is not None else self.default_timeout

        if not commit_shas:
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
        # author_context is already included in context (from context_file)
        # with prominent formatting by review_runner.py
        _ = author_context
        query = self._create_review_query(context, commit_shas)

        # Run agent session
        try:
            response_text, log_path, was_interrupted = await self._run_agent_session(
                query, effective_timeout, claude_session_id, guard
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
        except StructuredOutputError as e:
            if self.event_sink is not None:
                self.event_sink.on_review_warning(str(e))
            return ReviewResult(
                passed=False,
                issues=[],
                parse_error=str(e),
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

        # If interrupted during session, return with interrupted flag
        if was_interrupted:
            return ReviewResult(
                passed=False,
                issues=[],
                parse_error=None,
                fatal_error=False,
                review_log_path=log_path,
                interrupted=True,
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
        context: str,
        commit_shas: Sequence[str] | None,
    ) -> str:
        """Construct review query for agent.

        Combines review instructions, commit list, and context.
        Note: author_context is already included in context by review_runner.py.

        Args:
            context: Context from context file (includes issue description and
                author's response to previous review findings).
            commit_shas: Optional specific commit SHAs.

        Returns:
            Formatted query string for the agent.
        """
        prompt = self.review_agent_prompt
        has_commit_placeholder = "{commit_list}" in prompt
        has_context_placeholder = "{context_section}" in prompt
        context_section = f"## Implementation Context\n{context}" if context else ""

        if has_commit_placeholder:
            prompt = prompt.replace("{commit_list}", ", ".join(commit_shas or []))
        if has_context_placeholder:
            prompt = prompt.replace("{context_section}", context_section)

        parts = [prompt]

        if commit_shas:
            parts.append(f"\n\n## Specific Commits\n{', '.join(commit_shas)}")

        if context and not has_context_placeholder:
            # Context file already contains issue description and author's response
            # to previous review findings (formatted by review_runner.py)
            parts.append(f"\n\n## Implementation Context\n{context}")

        # Output Format section is already in the prompt file (review_agent.md)
        # - Do not duplicate it here to avoid conflicting instructions

        return "".join(parts)

    async def _run_agent_session(
        self,
        query: str,
        timeout: int,
        session_id: str | None,
        guard: object,
    ) -> tuple[str, Path | None, bool]:
        """Run agent session and collect final output.

        Creates SDK client, opens session, sends query, streams responses,
        and extracts final JSON output.

        Args:
            query: Review query to send to the agent.
            timeout: Timeout in seconds.
            session_id: Optional session ID for telemetry.
            guard: InterruptGuard for checking interrupt status.

        Returns:
            Tuple of (raw response text, session log path, was_interrupted).

        Raises:
            TimeoutError: If agent session times out.
            Exception: If SDK client creation or query fails.
        """
        # Create SDK options
        options = self.sdk_client_factory.create_options(
            cwd=str(self.repo_path),
            model=self.model,
            permission_mode="bypassPermissions",
            output_format=_REVIEW_OUTPUT_FORMAT,
            settings='{"autoCompactEnabled": true}',
            env={"MALA_SDK_FLOW": "reviewer"},
        )

        # Create SDK client
        client = self.sdk_client_factory.create(options)

        # Wrap the session in a timeout
        return await asyncio.wait_for(
            self._execute_agent_session(client, query, session_id, guard),
            timeout=timeout,
        )

    async def _execute_agent_session(
        self,
        client: object,
        query: str,
        session_id: str | None,
        guard: object,
    ) -> tuple[str, Path | None, bool]:
        """Execute the agent session and collect output.

        Args:
            client: SDK client instance.
            query: Review query to send.
            session_id: Optional session ID for telemetry.
            guard: InterruptGuard for checking interrupt status.

        Returns:
            Tuple of (response text, session log path, was_interrupted).
        """
        from pathlib import Path as PathLib  # noqa: TC003 (runtime import for log_path)

        response_text = ""
        log_path: PathLib | None = None
        result_session_id: str | None = None
        structured_output: object | None = None
        result_subtype: str | None = None
        was_interrupted = False

        response_iter = None
        async with client:  # type: ignore[union-attr]
            await client.query(query, session_id=session_id)  # type: ignore[union-attr]

            response_iter = client.receive_response()  # type: ignore[union-attr]
            try:
                async for msg in response_iter:
                    # Check for interrupt during iteration (Finding 4)
                    if guard.is_interrupted():  # type: ignore[union-attr]
                        logger.info("Review interrupted during SDK session iteration")
                        was_interrupted = True
                        break

                    msg_type = getattr(msg, "type", None)
                    msg_subtype = getattr(msg, "subtype", None)
                    msg_class = type(msg).__name__

                    # Extract text from assistant messages (SDK dataclass or dict-like)
                    if (
                        msg_subtype == "assistant"
                        or msg_type == "assistant"
                        or msg_class == "AssistantMessage"
                    ):
                        content = getattr(msg, "content", None)
                        if content is not None and isinstance(content, list):
                            for block in content:
                                # Handle both SDK TextBlock objects and dict format
                                if type(block).__name__ == "TextBlock":
                                    response_text += getattr(block, "text", "")
                                elif (
                                    isinstance(block, dict)
                                    and block.get("type") == "text"
                                ):
                                    response_text += block.get("text", "")

                    # Extract session info from result message
                    if (
                        msg_type == "result"
                        or msg_subtype == "result"
                        or msg_class == "ResultMessage"
                    ):
                        result_subtype = msg_subtype
                        result_session_id = getattr(msg, "session_id", None)
                        structured_output = getattr(msg, "structured_output", None)
                        result_text = getattr(msg, "result", None)
                        if structured_output is None and isinstance(result_text, dict):
                            structured_output = result_text
                        if isinstance(result_text, str) and not response_text.strip():
                            response_text = result_text
            finally:
                if response_iter is not None:
                    aclose = getattr(response_iter, "aclose", None)
                    if callable(aclose):
                        try:
                            await aclose()
                        except Exception as e:  # pragma: no cover - best effort cleanup
                            logger.debug(f"Failed to close SDK response iterator: {e}")

                query_handle = getattr(client, "_query", None)
                if query_handle is not None:
                    for stream_name in ("_message_send", "_message_receive"):
                        stream = getattr(query_handle, stream_name, None)
                        aclose = getattr(stream, "aclose", None)
                        if callable(aclose):
                            try:
                                await aclose()
                            except (
                                Exception
                            ) as e:  # pragma: no cover - best effort cleanup
                                logger.debug(
                                    "Failed to close SDK query stream %s: %s",
                                    stream_name,
                                    e,
                                )

        try:
            await client.disconnect()  # type: ignore[union-attr]
        except Exception as e:  # pragma: no cover - best effort cleanup
            logger.debug(f"Failed to disconnect SDK client: {e}")

        # If interrupted, return early without checking structured output errors
        if was_interrupted:
            if result_session_id:
                from src.infra.tools.env import get_claude_log_path

                log_path = get_claude_log_path(self.repo_path, result_session_id)
            return response_text, log_path, True

        if result_subtype == "error_max_structured_output_retries":
            raise StructuredOutputError(result_subtype)

        if structured_output is not None:
            if isinstance(structured_output, str):
                response_text = structured_output
            else:
                response_text = json.dumps(structured_output)

        # Construct log path from session ID if available using canonical helper
        if result_session_id:
            from src.infra.tools.env import get_claude_log_path

            log_path = get_claude_log_path(self.repo_path, result_session_id)

        return response_text, log_path, False

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

            # Coerce nullable fields to expected types (None -> default)
            file_path = item.get("file_path")
            line_start = item.get("line_start")
            line_end = item.get("line_end")
            title = item.get("title")
            body = item.get("body")
            reviewer = item.get("reviewer")

            issues.append(
                ReviewIssue(
                    file=file_path if isinstance(file_path, str) else "",
                    line_start=line_start if isinstance(line_start, int) else 0,
                    line_end=line_end if isinstance(line_end, int) else 0,
                    priority=priority,
                    title=title if isinstance(title, str) else "",
                    body=body if isinstance(body, str) else "",
                    reviewer=reviewer if isinstance(reviewer, str) else "agent_sdk",
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

        Tries each code block and returns the first one that parses as valid JSON.
        Falls back to brace-scanning if no valid JSON found in code blocks.

        Args:
            text: Raw text that may contain JSON in code blocks.

        Returns:
            Extracted JSON string.
        """
        # Find all code blocks and try each one for valid JSON
        code_blocks = re.findall(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        for block_content in code_blocks:
            candidate = block_content.strip()
            if candidate:
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    continue

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
