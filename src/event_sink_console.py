"""Console event sink for MalaOrchestrator.

Implements MalaEventSink using existing console logging helpers.
Produces human-readable output for terminal use.
"""

from typing import Any

from .event_sink import EventRunConfig, MalaEventSink
from .logging.console import (
    Colors,
    log,
    log_agent_text,
    log_tool,
    log_verbose,
)


class ConsoleEventSink:
    """Event sink that outputs to the console using existing log helpers.

    Implements all MalaEventSink methods, delegating to log(), log_tool(),
    and log_agent_text() from src/logging/console.py.

    Example:
        sink = ConsoleEventSink()
        orchestrator = MalaOrchestrator(..., event_sink=sink)
        await orchestrator.run()  # Produces console output
    """

    # -------------------------------------------------------------------------
    # Run lifecycle
    # -------------------------------------------------------------------------

    def on_run_started(self, config: EventRunConfig) -> None:
        """Log run configuration at startup."""
        log("▶", f"Starting run: {config.repo_path}", Colors.GREEN)
        details = []
        if config.max_agents is not None:
            details.append(f"agents={config.max_agents}")
        if config.max_issues is not None:
            details.append(f"max_issues={config.max_issues}")
        if config.timeout_minutes is not None:
            details.append(f"timeout={config.timeout_minutes}m")
        if config.epic_id:
            details.append(f"epic={config.epic_id}")
        if details:
            log_verbose("◦", ", ".join(details), Colors.MUTED)

    def on_run_completed(
        self,
        success_count: int,
        total_count: int,
        run_validation_passed: bool,
    ) -> None:
        """Log run completion summary."""
        status = "✓" if run_validation_passed else "✗"
        color = Colors.GREEN if run_validation_passed else Colors.RED
        log(
            status,
            f"Run complete: {success_count}/{total_count} issues, "
            f"validation={'passed' if run_validation_passed else 'failed'}",
            color,
        )

    def on_ready_issues(self, issue_ids: list[str]) -> None:
        """Log list of ready issues."""
        count = len(issue_ids)
        if count == 0:
            log_verbose("◦", "No ready issues found", Colors.MUTED)
        else:
            log("→", f"{count} issue(s) ready: {', '.join(issue_ids[:5])}", Colors.CYAN)
            if count > 5:
                log_verbose("◦", f"...and {count - 5} more", Colors.MUTED)

    def on_waiting_for_agents(self, count: int) -> None:
        """Log when waiting for agents to complete."""
        log_verbose("◦", f"Waiting for {count} agent(s) to complete...", Colors.MUTED)

    def on_no_more_issues(self, reason: str) -> None:
        """Log when no more issues are available."""
        log("◦", f"No more issues: {reason}", Colors.MUTED)

    # -------------------------------------------------------------------------
    # Agent lifecycle
    # -------------------------------------------------------------------------

    def on_agent_started(self, agent_id: str, issue_id: str) -> None:
        """Log agent spawn."""
        log("▶", f"Spawning agent for {issue_id}", Colors.GREEN, agent_id=agent_id)

    def on_agent_completed(
        self,
        agent_id: str,
        issue_id: str,
        success: bool,
        duration_seconds: float,
        summary: str,
    ) -> None:
        """Log agent completion."""
        status = "✓" if success else "✗"
        color = Colors.GREEN if success else Colors.RED
        duration_str = f"{duration_seconds:.1f}s"
        log(
            status,
            f"Agent completed ({duration_str}): {summary}",
            color,
            agent_id=agent_id,
        )

    def on_claim_failed(self, agent_id: str, issue_id: str) -> None:
        """Log when claiming an issue fails."""
        log("!", f"Failed to claim {issue_id}", Colors.YELLOW, agent_id=agent_id)

    # -------------------------------------------------------------------------
    # SDK message streaming
    # -------------------------------------------------------------------------

    def on_tool_use(
        self,
        agent_id: str,
        tool_name: str,
        description: str = "",
        arguments: dict[str, Any] | None = None,
    ) -> None:
        """Log tool usage."""
        log_tool(tool_name, description, agent_id=agent_id, arguments=arguments)

    def on_agent_text(self, agent_id: str, text: str) -> None:
        """Log agent text output."""
        log_agent_text(text, agent_id)

    # -------------------------------------------------------------------------
    # Quality gate events
    # -------------------------------------------------------------------------

    def on_gate_started(
        self,
        agent_id: str | None,
        attempt: int,
        max_attempts: int,
    ) -> None:
        """Log quality gate check start."""
        scope = "run-level" if agent_id is None else ""
        log_verbose(
            "→",
            f"Quality gate {scope}check (attempt {attempt}/{max_attempts})",
            Colors.MUTED,
            agent_id=agent_id,
        )

    def on_gate_passed(self, agent_id: str | None) -> None:
        """Log quality gate passed."""
        scope = "Run-level gate" if agent_id is None else "Gate"
        log("✓", f"{scope} passed", Colors.GREEN, agent_id=agent_id)

    def on_gate_failed(
        self,
        agent_id: str | None,
        attempt: int,
        max_attempts: int,
    ) -> None:
        """Log quality gate failed after all retries."""
        scope = "Run-level gate" if agent_id is None else "Gate"
        log(
            "✗",
            f"{scope} failed after {attempt}/{max_attempts} attempts",
            Colors.RED,
            agent_id=agent_id,
        )

    def on_gate_retry(
        self,
        agent_id: str,
        attempt: int,
        max_attempts: int,
    ) -> None:
        """Log quality gate retry."""
        log(
            "↻",
            f"Retrying gate (attempt {attempt}/{max_attempts})",
            Colors.YELLOW,
            agent_id=agent_id,
        )

    def on_gate_result(
        self,
        agent_id: str | None,
        passed: bool,
        failure_reasons: list[str] | None = None,
    ) -> None:
        """Log detailed gate result."""
        if passed:
            return  # on_gate_passed already handles success
        if failure_reasons:
            for reason in failure_reasons:
                log_verbose("  •", reason, Colors.RED, agent_id=agent_id)

    # -------------------------------------------------------------------------
    # Codex review events
    # -------------------------------------------------------------------------

    def on_review_started(
        self,
        agent_id: str,
        attempt: int,
        max_attempts: int,
    ) -> None:
        """Log Codex review start."""
        log_verbose(
            "→",
            f"Codex review (attempt {attempt}/{max_attempts})",
            Colors.MUTED,
            agent_id=agent_id,
        )

    def on_review_passed(self, agent_id: str) -> None:
        """Log Codex review passed."""
        log("✓", "Codex review passed", Colors.GREEN, agent_id=agent_id)

    def on_review_retry(
        self,
        agent_id: str,
        attempt: int,
        max_attempts: int,
        error_count: int | None = None,
        parse_error: str | None = None,
    ) -> None:
        """Log Codex review retry."""
        if parse_error:
            detail = f"parse error: {parse_error}"
        elif error_count is not None:
            detail = f"{error_count} errors found"
        else:
            detail = "issues found"
        log(
            "↻",
            f"Retrying review ({detail}, attempt {attempt}/{max_attempts})",
            Colors.YELLOW,
            agent_id=agent_id,
        )

    # -------------------------------------------------------------------------
    # Fixer agent events
    # -------------------------------------------------------------------------

    def on_fixer_started(
        self,
        attempt: int,
        max_attempts: int,
    ) -> None:
        """Log fixer agent spawn."""
        log(
            "▶", f"Spawning fixer agent (attempt {attempt}/{max_attempts})", Colors.CYAN
        )

    def on_fixer_completed(self, result: str) -> None:
        """Log fixer agent completion."""
        log("✓", f"Fixer completed: {result}", Colors.GREEN)

    def on_fixer_failed(self, reason: str) -> None:
        """Log fixer agent failure."""
        log("✗", f"Fixer failed: {reason}", Colors.RED)

    # -------------------------------------------------------------------------
    # Issue lifecycle
    # -------------------------------------------------------------------------

    def on_issue_closed(self, agent_id: str, issue_id: str) -> None:
        """Log issue closure."""
        log("✓", f"Closed {issue_id}", Colors.GREEN, agent_id=agent_id)

    def on_issue_completed(
        self,
        agent_id: str,
        issue_id: str,
        success: bool,
        duration_seconds: float,
        summary: str,
    ) -> None:
        """Log issue completion."""
        status = "✓" if success else "✗"
        color = Colors.GREEN if success else Colors.RED
        duration_str = f"{duration_seconds:.1f}s"
        log(
            status,
            f"Issue {issue_id} completed ({duration_str}): {summary}",
            color,
            agent_id=agent_id,
        )

    def on_epic_closed(self, agent_id: str) -> None:
        """Log parent epic auto-closure."""
        log("✓", "Parent epic auto-closed", Colors.GREEN, agent_id=agent_id)

    def on_validation_result(
        self,
        agent_id: str,
        passed: bool,
    ) -> None:
        """Log per-issue validation result."""
        if passed:
            log_verbose("✓", "Validation passed", Colors.GREEN, agent_id=agent_id)
        else:
            log("✗", "Validation failed", Colors.RED, agent_id=agent_id)

    # -------------------------------------------------------------------------
    # Warnings and diagnostics
    # -------------------------------------------------------------------------

    def on_warning(self, message: str, agent_id: str | None = None) -> None:
        """Log warning condition."""
        log("!", message, Colors.YELLOW, agent_id=agent_id)

    def on_log_timeout(self, agent_id: str, log_path: str) -> None:
        """Log when waiting for a log file times out."""
        log("!", f"Log file not found: {log_path}", Colors.YELLOW, agent_id=agent_id)

    def on_locks_cleaned(self, agent_id: str, count: int) -> None:
        """Log stale lock cleanup."""
        log_verbose(
            "◦", f"Cleaned {count} stale lock(s)", Colors.MUTED, agent_id=agent_id
        )

    def on_locks_released(self, count: int) -> None:
        """Log remaining locks released at run end."""
        if count > 0:
            log_verbose("◦", f"Released {count} remaining lock(s)", Colors.MUTED)

    def on_issues_committed(self) -> None:
        """Log issues.jsonl commit."""
        log_verbose("◦", "Committed .beads/issues.jsonl", Colors.MUTED)

    def on_run_metadata_saved(self, path: str) -> None:
        """Log run metadata save."""
        log_verbose("◦", f"Saved run metadata: {path}", Colors.MUTED)

    def on_run_level_validation_disabled(self) -> None:
        """Log when run-level validation is disabled."""
        log_verbose("◦", "Run-level validation disabled", Colors.MUTED)


# Verify ConsoleEventSink implements MalaEventSink at import time
assert isinstance(ConsoleEventSink(), MalaEventSink)
