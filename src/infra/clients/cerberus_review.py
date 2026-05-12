"""Cerberus review adapter for mala orchestrator.

This module provides a thin adapter for the Cerberus CLI.
It handles:
- CLI orchestration (spawn, wait, resolve)
- Issue formatting for follow-up prompts

JSON parsing and exit-code mapping are delegated to ReviewOutputParser.
Subprocess management is delegated to CerberusCLI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from src.core.constants import DEFAULT_CERBERUS_REVIEW_TIMEOUT_SECONDS
from src.infra.clients.cerberus_cli import CerberusCLI
from src.infra.clients.cerberus_output_parser import (
    ReviewIssue,  # noqa: TC001 (used at runtime in format_review_issues)
    ReviewResult,
    map_exit_code_to_result,
)
from src.infra.tools.command_runner import CommandRunner

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import asyncio
    from collections.abc import Sequence

    from src.core.protocols.events import MalaEventSink


@dataclass
class DefaultReviewer:
    """Default CodeReviewer implementation using the Cerberus CLI.

    This class conforms to the CodeReviewer protocol and provides the default
    behavior for the orchestrator. Tests can inject alternative implementations.

    The reviewer spawns Cerberus CLI processes and parses their output.
    Extra args/env can be provided to customize Cerberus behavior.
    For initial review with an empty diff, short-circuits to PASS without spawning.

    Subprocess management is delegated to CerberusCLI; this class handles
    orchestration logic and result mapping.
    """

    repo_path: Path
    state_root: Path | None = None
    project_key: str | None = None
    spawn_args: tuple[str, ...] = field(default_factory=tuple)
    wait_args: tuple[str, ...] = field(default_factory=tuple)
    env: dict[str, str] = field(default_factory=dict)
    event_sink: MalaEventSink | None = None

    def _get_cli(self) -> CerberusCLI:
        """Get the CerberusCLI instance for subprocess operations."""
        return CerberusCLI(
            repo_path=self.repo_path,
            env=self.env,
        )

    def overrides_disabled_setting(self) -> bool:
        """Return False; DefaultReviewer respects the disabled setting."""
        return False

    def _project_key(self) -> str:
        """Return the non-empty Cerberus project key for this review run."""
        return (
            self.project_key
            or self.env.get("CERBERUS_PROJECT_KEY")
            or self.repo_path.name
            or "default"
        )

    @staticmethod
    def _is_no_changes_error(error_detail: str) -> bool:
        """Return True when Cerberus skipped spawn because the diff is empty."""
        normalized = error_detail.lower()
        return "no changes found" in normalized and "diff mode" in normalized

    async def __call__(
        self,
        context_file: Path | None = None,
        timeout: int = DEFAULT_CERBERUS_REVIEW_TIMEOUT_SECONDS,
        claude_session_id: str | None = None,
        author_context: str | None = None,
        *,
        commit_shas: Sequence[str],
        interrupt_event: asyncio.Event | None = None,
    ) -> ReviewResult:
        # author_context is already included in context_file by review_runner.py
        # with prominent formatting to highlight implementer's response to previous findings
        _ = author_context
        cli = self._get_cli()

        # Validate the Cerberus binary exists and is executable before proceeding.
        validation_error = cli.validate_binary()
        if validation_error is not None:
            return ReviewResult(
                passed=False,
                issues=[],
                parse_error=validation_error,
                fatal_error=True,
                review_log_path=None,
            )

        runner = CommandRunner(cwd=self.repo_path)
        run_key = f"mala-{claude_session_id}" if claude_session_id else "mala-unknown"
        state_root = self.state_root or self.repo_path / ".mala" / "cerberus"
        project_key = self._project_key()
        env = cli.build_env(
            run_key=run_key,
            state_root=state_root,
            project_key=project_key,
            claude_session_id=claude_session_id,
        )
        env["CERBERUS_PROJECT_KEY"] = project_key
        if "CLAUDE_SESSION_ID" not in env:
            return ReviewResult(
                passed=False,
                issues=[],
                parse_error="CLAUDE_SESSION_ID missing; must be provided by agent session id",
                fatal_error=True,
                review_log_path=None,
            )

        if not commit_shas:
            return ReviewResult(
                passed=True,
                issues=[],
                parse_error=None,
                fatal_error=False,
                review_log_path=None,
            )

        # Spawn code review
        spawn_result = await cli.spawn_code_review(
            runner=runner,
            env=env,
            timeout=timeout,
            spawn_args=self.spawn_args,
            context_file=context_file,
            commit_shas=commit_shas,
        )

        if spawn_result.timed_out:
            return ReviewResult(
                passed=False,
                issues=[],
                parse_error="spawn timeout",
                fatal_error=False,
            )

        if not spawn_result.success:
            if self._is_no_changes_error(spawn_result.error_detail):
                logger.info("Review skipped because Cerberus found no changes in diff")
                return ReviewResult(
                    passed=True,
                    issues=[],
                    parse_error=None,
                    fatal_error=False,
                )
            return ReviewResult(
                passed=False,
                issues=[],
                parse_error=f"spawn failed: {spawn_result.error_detail}",
                fatal_error=False,
            )

        # Check if wait_args already specifies --timeout to avoid duplicates
        user_timeout = CerberusCLI.extract_wait_timeout(self.wait_args)

        # Wait for review completion
        wait_result = await cli.wait_for_review(
            run_key=run_key,
            runner=runner,
            env=env,
            cli_timeout=timeout,
            wait_args=self.wait_args,
            user_timeout=user_timeout,
        )

        if wait_result.timed_out:
            logger.warning("Review timeout after=%ds", timeout)
            return ReviewResult(
                passed=False,
                issues=[],
                parse_error="timeout",
                fatal_error=False,
            )

        result = map_exit_code_to_result(
            wait_result.returncode,
            wait_result.stdout,
            wait_result.stderr,
            event_sink=self.event_sink,
            state_root=state_root,
            project_key=env["CERBERUS_PROJECT_KEY"],
            run_key=run_key,
        )
        logger.info(
            "Review completed: passed=%s issues=%d",
            result.passed,
            len(result.issues),
        )
        return result


def _to_relative_path(file_path: str, base_path: Path | None = None) -> str:
    """Convert an absolute file path to a relative path for display.

    Strips the base path prefix to show paths relative to the repository root.

    Args:
        file_path: Absolute or relative file path.
        base_path: Base path (typically repository root) for relativization.
            If None, uses Path.cwd() as fallback.

    Returns:
        Relative path suitable for display. If relativization fails,
        returns the original path to preserve directory context.
    """
    # If already relative, return as-is
    if not file_path.startswith("/"):
        return file_path

    # Use provided base_path or fall back to cwd. Resolve both sides
    # consistently so symlink/firmlink-redirected paths still match.
    try:
        base = base_path.resolve() if base_path else Path.cwd()
        abs_path = Path(file_path)
        if abs_path.is_relative_to(base):
            return str(abs_path.relative_to(base))
        abs_resolved = abs_path.resolve()
        if abs_resolved.is_relative_to(base):
            return str(abs_resolved.relative_to(base))
    except (ValueError, OSError):
        pass

    # Preserve original path if relativization fails (don't strip to filename)
    return file_path


def format_review_issues(
    issues: list[ReviewIssue], base_path: Path | None = None
) -> str:
    """Format review issues as a human-readable string for follow-up prompts.

    Args:
        issues: List of ReviewIssue objects to format.
        base_path: Base path (typically repository root) for path relativization.
            If None, uses Path.cwd() as fallback.

    Returns:
        Formatted string with issues grouped by file.
    """
    if not issues:
        return "No specific issues found."

    lines = []
    current_file = None

    # Sort by file, then by line, then by priority (lower = more important)
    sorted_issues = sorted(
        issues,
        key=lambda x: (
            x.file,
            x.line_start,
            x.priority if x.priority is not None else 4,
        ),
    )

    for issue in sorted_issues:
        # Convert absolute paths to relative for cleaner display
        display_file = _to_relative_path(issue.file, base_path)
        if display_file != current_file:
            if current_file is not None:
                lines.append("")  # Blank line between files
            current_file = display_file
            lines.append(f"File: {display_file}")

        loc = (
            f"L{issue.line_start}-{issue.line_end}"
            if issue.line_start != issue.line_end
            else f"L{issue.line_start}"
        )
        # Include reviewer attribution
        reviewer_tag = f"[{issue.reviewer}]" if issue.reviewer else ""
        lines.append(f"  {loc}: {reviewer_tag} {issue.title}")
        if issue.body:
            lines.append(f"    {issue.body}")

    return "\n".join(lines)
