"""Cerberus review-gate adapter for mala orchestrator.

This module provides a thin adapter for the Cerberus review-gate CLI.
It handles:
- CLI orchestration (spawn, wait, resolve)
- Issue formatting for follow-up prompts

JSON parsing and exit-code mapping are delegated to ReviewOutputParser.
Subprocess management is delegated to CerberusGateCLI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from src.infra.clients.cerberus_gate_cli import CerberusGateCLI
from src.infra.clients.review_output_parser import (
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
    """Default CodeReviewer implementation using Cerberus review-gate CLI.

    This class conforms to the CodeReviewer protocol and provides the default
    behavior for the orchestrator. Tests can inject alternative implementations.

    The reviewer spawns review-gate CLI processes and parses their output.
    Extra args/env can be provided to customize review-gate behavior.
    For initial review with an empty diff, short-circuits to PASS without spawning.

    Subprocess management is delegated to CerberusGateCLI; this class handles
    orchestration logic, stale gate recovery, and result mapping.
    """

    repo_path: Path
    bin_path: Path | None = None
    spawn_args: tuple[str, ...] = field(default_factory=tuple)
    wait_args: tuple[str, ...] = field(default_factory=tuple)
    env: dict[str, str] = field(default_factory=dict)
    event_sink: MalaEventSink | None = None

    def _get_cli(self) -> CerberusGateCLI:
        """Get the CerberusGateCLI instance for subprocess operations."""
        return CerberusGateCLI(
            repo_path=self.repo_path,
            bin_path=self.bin_path,
            env=self.env,
        )

    def _validate_review_gate_bin(self) -> str | None:
        """Validate that the review-gate binary exists and is executable.

        Delegates to CerberusGateCLI.validate_binary().

        Returns:
            None if the binary is valid, or an error message if not.
        """
        return self._get_cli().validate_binary()

    @staticmethod
    def _extract_wait_timeout(args: tuple[str, ...]) -> int | None:
        """Extract --timeout value from wait args if provided.

        Delegates to CerberusGateCLI.extract_wait_timeout().

        Args:
            args: Tuple of command-line arguments to search.

        Returns:
            The timeout value as int if found and valid, None otherwise.
        """
        return CerberusGateCLI.extract_wait_timeout(args)

    def overrides_disabled_setting(self) -> bool:
        """Return False; DefaultReviewer respects the disabled setting."""
        return False

    async def __call__(
        self,
        context_file: Path | None = None,
        timeout: int = 300,
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

        # Validate review-gate binary exists and is executable before proceeding
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
        env = cli.build_env(claude_session_id)
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
            # Check for stale gate from a previous attempt in this session.
            if spawn_result.already_active:
                if self.event_sink is not None:
                    self.event_sink.on_review_warning(
                        "Resolving stale gate from previous attempt"
                    )
                resolve_result = await cli.resolve_gate(runner, env)
                if resolve_result.success:
                    logger.info("Stale gate resolved")
                    # Retry spawn after resolving the stale gate
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
                        # If still "already active", another session owns the gate
                        if spawn_result.already_active:
                            return ReviewResult(
                                passed=False,
                                issues=[],
                                parse_error=(
                                    "Another review gate is active (not from this session). "
                                    "Wait for it to finish or resolve manually with "
                                    "`review-gate resolve`."
                                ),
                                fatal_error=True,
                            )
                        return ReviewResult(
                            passed=False,
                            issues=[],
                            parse_error=f"spawn failed: {spawn_result.error_detail}",
                            fatal_error=False,
                        )
                    # Successfully spawned after clearing gate - continue to wait phase
                else:
                    return ReviewResult(
                        passed=False,
                        issues=[],
                        parse_error=f"spawn failed: {spawn_result.error_detail} (auto-resolve failed: {resolve_result.error_detail})",
                        fatal_error=False,
                    )
            else:
                return ReviewResult(
                    passed=False,
                    issues=[],
                    parse_error=f"spawn failed: {spawn_result.error_detail}",
                    fatal_error=False,
                )

        # spawn-code-review doesn't output JSON to stdout - it just spawns reviewers
        # and writes state to disk. We use --session-id with the Claude session ID
        # to let the wait command find the state file.
        session_id = env["CLAUDE_SESSION_ID"]

        # Check if wait_args already specifies --timeout to avoid duplicates
        user_timeout = CerberusGateCLI.extract_wait_timeout(self.wait_args)

        # Wait for review completion
        wait_result = await cli.wait_for_review(
            session_id=session_id,
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
            review_log_path=wait_result.session_dir,
            event_sink=self.event_sink,
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

    # Use provided base_path or fall back to cwd
    base = base_path.resolve() if base_path else Path.cwd()
    try:
        abs_path = Path(file_path)
        if abs_path.is_relative_to(base):
            return str(abs_path.relative_to(base))
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
