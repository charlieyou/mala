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

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.infra.clients.review_output_parser import ReviewResult  # noqa: TC001 (runtime)

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

        Raises:
            NotImplementedError: Skeleton implementation - T004 will add real logic.
        """
        # Skeleton implementation - T004 will implement real logic
        raise NotImplementedError(
            "AgentSDKReviewer.__call__ not implemented yet. See T004."
        )

    async def _check_diff_empty(self, diff_range: str) -> bool:
        """Check if diff range is empty using git diff --stat.

        Args:
            diff_range: Git diff range to check.

        Returns:
            True if the diff is empty, False otherwise.

        Raises:
            NotImplementedError: Skeleton implementation.
        """
        raise NotImplementedError("_check_diff_empty not implemented yet. See T004.")

    async def _load_context(self, context_file: Path | None) -> str:
        """Load context file asynchronously.

        Args:
            context_file: Path to context file, or None.

        Returns:
            Context file content as string, or empty string if no file.

        Raises:
            NotImplementedError: Skeleton implementation.
        """
        raise NotImplementedError("_load_context not implemented yet. See T004.")

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

        Raises:
            NotImplementedError: Skeleton implementation.
        """
        raise NotImplementedError("_create_review_query not implemented yet. See T004.")

    async def _run_agent_session(
        self,
        query: str,
        timeout: int,
        session_id: str | None,
    ) -> str:
        """Run agent session and collect final output.

        Creates SDK client, opens session, sends query, streams responses,
        and extracts final JSON output.

        Args:
            query: Review query to send to the agent.
            timeout: Timeout in seconds.
            session_id: Optional session ID for telemetry.

        Returns:
            Raw response text from the agent.

        Raises:
            NotImplementedError: Skeleton implementation.
        """
        raise NotImplementedError("_run_agent_session not implemented yet. See T004.")

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

        Raises:
            NotImplementedError: Skeleton implementation.
        """
        raise NotImplementedError("_parse_response not implemented yet. See T004.")
