"""Console event sink implementation for MalaOrchestrator.

Provides ConsoleEventSink which outputs orchestrator events to the console
using the log helpers from log_output/console.py.
"""

import math
import re
from typing import Any

from src.core.protocols.events import EventRunConfig, MalaEventSink
from src.core.protocols.lifecycle import DeadlockInfoProtocol
from src.infra.io.base_sink import BaseEventSink
from src.infra.io.log_output.console import (
    Colors,
    log,
    log_agent_text,
    log_tool,
    log_verbose,
    truncate_text,
)


class ConsoleEventSink(BaseEventSink):
    """Event sink that outputs to the console using existing log helpers.

    Implements all MalaEventSink methods, delegating to log(), log_tool(),
    and log_agent_text() from src/infra/io/log_output/console.py.

    Example:
        from src.orchestration.factory import create_orchestrator, OrchestratorDependencies

        sink = ConsoleEventSink()
        deps = OrchestratorDependencies(event_sink=sink)
        orchestrator = create_orchestrator(config, deps=deps)
        await orchestrator.run()  # Produces console output
    """

    # -------------------------------------------------------------------------
    # Run lifecycle
    # -------------------------------------------------------------------------

    def on_run_started(self, config: EventRunConfig) -> None:
        log("→", "[START] Run started", agent_id="run")
        log("◦", f"Repository: {config.repo_path}", agent_id="run")
        if config.epic_id:
            log("◦", f"Epic: {config.epic_id}", agent_id="run")
        if config.only_ids:
            log("◦", f"Issues: {', '.join(config.only_ids)}", agent_id="run")
        if config.include_wip:
            log("◦", "Mode: include WIP", agent_id="run")
        if config.orphans_only:
            log("◦", "Mode: orphans only", agent_id="run")
        self._log_config(config)
        self._log_cli_args(config)

    def _log_cli_args(self, config: EventRunConfig) -> None:
        # Log CLI arguments if available
        if config.cli_args:
            # Filter out sensitive or empty arguments
            safe_args = {
                k: v
                for k, v in config.cli_args.items()
                if v is not None and k not in ("api_key",)
            }
            if safe_args:
                log_verbose("◦", f"CLI args: {safe_args}", agent_id="run")

    def _log_config(self, config: EventRunConfig) -> None:
        """Log all configuration values at run start.

        Displays a comprehensive summary of effective configuration,
        making it easy to understand exactly how the run is configured.
        """
        # Build config items as key=value pairs
        items: list[str] = []

        # Parallelism
        agents = config.max_agents if config.max_agents is not None else "unlimited"
        items.append(f"agents={agents}")

        # Timeout
        if config.timeout_minutes is not None:
            items.append(f"timeout={config.timeout_minutes}m")

        # Max issues
        if config.max_issues is not None:
            items.append(f"max_issues={config.max_issues}")

        # Gate and review retries
        items.append(f"gate_retries={config.max_gate_retries}")
        items.append(f"review_retries={config.max_review_retries}")

        # Review status
        if config.review_enabled:
            items.append("review=enabled")
        else:
            reason = config.review_disabled_reason or "disabled"
            # Normalize whitespace to underscores to keep key=value as single token
            reason = re.sub(r"\s+", "_", reason)
            items.append(f"review={reason}")

        # CLI args extras (watch)
        if config.cli_args:
            if config.cli_args.get("watch"):
                items.append("watch=true")

        # Format: show all items on one line if short, otherwise multi-line
        config_str = " ".join(items)
        if len(config_str) <= 80:
            log("◐", f"Config: {config_str}", agent_id="run")
        else:
            # Multi-line for long configs
            log("◐", "Config:", agent_id="run")
            for item in items:
                log("◦", f"  {item}", agent_id="run")

        # Log validation triggers summary
        self._log_triggers(config)

    def _log_triggers(self, config: EventRunConfig) -> None:
        """Log validation trigger configuration summary.

        Shows which triggers are enabled, their failure modes, and command counts.
        CLI output shows a compact summary; debug/verbose log shows full details.
        """
        triggers = config.validation_triggers
        if triggers is None or not triggers.has_any_enabled():
            log_verbose("◦", "Triggers: none configured", agent_id="run")
            return

        # Build trigger summary items for CLI output
        trigger_items: list[str] = []

        def format_trigger(name: str, t: object) -> str:
            """Format a trigger for CLI display, omitting mode if not set."""
            mode = getattr(t, "failure_mode", None)
            cmd_count = getattr(t, "command_count", 0)
            if mode is not None:
                return f"{name}({mode},{cmd_count}cmd)"
            return f"{name}({cmd_count}cmd)"

        if triggers.epic_completion and triggers.epic_completion.enabled:
            trigger_items.append(
                format_trigger("epic_completion", triggers.epic_completion)
            )

        if triggers.session_end and triggers.session_end.enabled:
            trigger_items.append(format_trigger("session_end", triggers.session_end))

        if triggers.periodic and triggers.periodic.enabled:
            trigger_items.append(format_trigger("periodic", triggers.periodic))

        if trigger_items:
            triggers_str = " ".join(trigger_items)
            log("◐", f"Triggers: {triggers_str}", agent_id="run")

        # Verbose/debug output with more detail per trigger
        self._log_triggers_verbose(triggers)

    def _log_triggers_verbose(self, triggers: object) -> None:
        """Log detailed trigger configuration for debugging.

        Args:
            triggers: ValidationTriggersSummary object with trigger details.
        """
        # Get individual trigger summaries
        epic = getattr(triggers, "epic_completion", None)
        session = getattr(triggers, "session_end", None)
        periodic = getattr(triggers, "periodic", None)
        run_end = getattr(triggers, "run_end", None)

        def log_trigger_detail(name: str, trigger: object) -> None:
            """Log a single trigger's detailed configuration."""
            mode = getattr(trigger, "failure_mode", None)
            cmd_names = getattr(trigger, "command_names", ())
            mode_str = mode if mode is not None else "unset"
            commands_str = ", ".join(cmd_names) if cmd_names else "none"
            log_verbose(
                "◦",
                f"  {name}: mode={mode_str}, commands=[{commands_str}]",
                agent_id="run",
            )

        if epic and getattr(epic, "enabled", False):
            log_trigger_detail("epic_completion", epic)

        if session and getattr(session, "enabled", False):
            log_trigger_detail("session_end", session)

        if periodic and getattr(periodic, "enabled", False):
            log_trigger_detail("periodic", periodic)

        if run_end and getattr(run_end, "enabled", False):
            log_trigger_detail("run_end", run_end)

    def on_run_completed(
        self,
        success_count: int,
        total_count: int,
        run_validation_passed: bool | None,
        abort_reason: str | None = None,
        *,
        validation_ran: bool = True,
    ) -> None:
        status_icon = "✓" if success_count == total_count else "✗"
        status = f"DONE {status_icon} {success_count}/{total_count} issues completed"
        if abort_reason:
            status += f" (aborted: {abort_reason})"
        log("→", status, agent_id="run")
        if run_validation_passed is True:
            log("✓", "RUN VALIDATION passed", agent_id="run")
        elif run_validation_passed is False:
            log(
                "✗",
                f"RUN VALIDATION {Colors.RED}failed{Colors.RESET}",
                agent_id="run",
            )
        elif not validation_ran:
            log("◦", "RUN VALIDATION skipped (no issues to validate)", agent_id="run")
        else:
            log("◦", "RUN VALIDATION skipped (interrupted)", agent_id="run")

    def on_ready_issues(self, issue_ids: list[str]) -> None:
        log("→", f"Ready issues ({len(issue_ids)}): {issue_ids}", agent_id="run")

    def on_waiting_for_agents(self, count: int) -> None:
        log_verbose("◦", f"Waiting for {count} agents to complete...", agent_id="run")

    def on_no_more_issues(self, reason: str) -> None:
        log("→", f"No more issues: {reason}", agent_id="run")

    # -------------------------------------------------------------------------
    # Agent lifecycle
    # -------------------------------------------------------------------------

    def on_agent_started(self, agent_id: str, issue_id: str) -> None:
        log("▶", f"Claimed {issue_id}", agent_id=agent_id)

    def on_agent_completed(
        self,
        agent_id: str,
        issue_id: str,
        success: bool,
        duration_seconds: float,
        summary: str,
    ) -> None:
        # Use verbose logging since on_issue_completed provides similar info
        status_icon = "✓" if success else "✗"
        log_verbose(
            status_icon,
            f"Agent {agent_id} completed in {duration_seconds:.1f}s: {summary}",
            agent_id=agent_id,
        )

    def on_claim_failed(self, agent_id: str, issue_id: str) -> None:
        log("○", f"SKIP {issue_id} already claimed", agent_id=agent_id)

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
        log_tool(tool_name, description, agent_id=agent_id, arguments=arguments)

    def on_agent_text(self, agent_id: str, text: str) -> None:
        log_agent_text(text, agent_id)

    # -------------------------------------------------------------------------
    # Quality gate events
    # -------------------------------------------------------------------------

    def on_gate_started(
        self,
        agent_id: str | None,
        attempt: int,
        max_attempts: int,
        issue_id: str | None = None,
    ) -> None:
        log(
            "→",
            f"GATE Attempt {attempt}/{max_attempts}",
            agent_id=agent_id or "run",
            issue_id=issue_id,
        )

    def on_gate_passed(
        self,
        agent_id: str | None,
        issue_id: str | None = None,
    ) -> None:
        log("✓", "GATE passed", agent_id=agent_id or "run", issue_id=issue_id)

    def on_gate_failed(
        self,
        agent_id: str | None,
        attempt: int,
        max_attempts: int,
        issue_id: str | None = None,
    ) -> None:
        log(
            "✗",
            f"GATE {Colors.RED}failed{Colors.RESET} ({attempt}/{max_attempts})",
            agent_id=agent_id or "run",
            issue_id=issue_id,
        )

    def on_gate_retry(
        self,
        agent_id: str,
        attempt: int,
        max_attempts: int,
        issue_id: str | None = None,
    ) -> None:
        log(
            "→",
            f"GATE Retry {attempt}/{max_attempts}",
            agent_id=agent_id,
            issue_id=issue_id,
        )

    def on_gate_result(
        self,
        agent_id: str | None,
        passed: bool,
        failure_reasons: list[str] | None = None,
        issue_id: str | None = None,
    ) -> None:
        if passed:
            log(
                "✓",
                "GATE all checks passed",
                agent_id=agent_id or "run",
                issue_id=issue_id,
            )
        elif failure_reasons:
            log(
                "✗",
                f"GATE {Colors.RED}{len(failure_reasons)} checks failed{Colors.RESET}",
                agent_id=agent_id or "run",
                issue_id=issue_id,
            )
            for reason in failure_reasons:
                log("→", f"  - {reason}", agent_id=agent_id or "run", issue_id=issue_id)

    # -------------------------------------------------------------------------
    # Codex review events
    # -------------------------------------------------------------------------

    def on_review_started(
        self,
        agent_id: str,
        attempt: int,
        max_attempts: int,
        issue_id: str | None = None,
    ) -> None:
        log(
            "→",
            f"REVIEW Attempt {attempt}/{max_attempts}",
            agent_id=agent_id,
            issue_id=issue_id,
        )

    def on_review_passed(
        self,
        agent_id: str,
        issue_id: str | None = None,
    ) -> None:
        log("✓", "REVIEW approved", agent_id=agent_id, issue_id=issue_id)

    def on_review_retry(
        self,
        agent_id: str,
        attempt: int,
        max_attempts: int,
        error_count: int | None = None,
        parse_error: str | None = None,
        issue_id: str | None = None,
    ) -> None:
        details = ""
        if error_count is not None:
            details = f" ({error_count} errors)"
        elif parse_error:
            details = f" (parse error: {parse_error})"
        log(
            "→",
            f"REVIEW Retry {attempt}/{max_attempts}{details}",
            agent_id=agent_id,
            issue_id=issue_id,
        )

    def on_review_warning(
        self,
        message: str,
        agent_id: str | None = None,
        issue_id: str | None = None,
    ) -> None:
        log(
            "⚠",
            f"REVIEW {Colors.YELLOW}{message}{Colors.RESET}",
            agent_id=agent_id or "run",
            issue_id=issue_id,
        )

    # -------------------------------------------------------------------------
    # Fixer agent events
    # -------------------------------------------------------------------------

    def on_fixer_started(
        self,
        attempt: int,
        max_attempts: int,
    ) -> None:
        log("→", f"FIXER Attempt {attempt}/{max_attempts}", agent_id="fixer")

    def on_fixer_completed(self, result: str) -> None:
        log("✓", f"FIXER {result}", agent_id="fixer")

    def on_fixer_failed(self, reason: str) -> None:
        log("✗", f"FIXER {Colors.RED}{reason}{Colors.RESET}", agent_id="fixer")

    # -------------------------------------------------------------------------
    # Issue lifecycle
    # -------------------------------------------------------------------------

    def on_issue_closed(self, agent_id: str, issue_id: str) -> None:
        log("→", f"CLOSE {issue_id}", agent_id=agent_id)

    def on_issue_completed(
        self,
        agent_id: str,
        issue_id: str,
        success: bool,
        duration_seconds: float,
        summary: str,
    ) -> None:
        status_icon = "✓" if success else "✗"
        color = Colors.GREEN if success else Colors.RED
        log(
            status_icon,
            f"{color}{issue_id} completed in {duration_seconds:.1f}s: {summary}{Colors.RESET}",
            agent_id=agent_id,
        )

    def on_epic_closed(self, agent_id: str) -> None:
        log("→", "EPIC Closed", agent_id=agent_id)

    def on_validation_started(
        self,
        agent_id: str,
        issue_id: str | None = None,
    ) -> None:
        log("→", "VALIDATE Starting validation", agent_id=agent_id, issue_id=issue_id)

    def on_validation_result(
        self,
        agent_id: str,
        passed: bool,
        issue_id: str | None = None,
    ) -> None:
        status_icon = "✓" if passed else "✗"
        log(status_icon, "VALIDATE", agent_id=agent_id, issue_id=issue_id)

    def on_validation_step_running(
        self,
        step_name: str,
        agent_id: str | None = None,
    ) -> None:
        log("▸", f"  {step_name} running...", agent_id=agent_id or "run")

    def on_validation_step_skipped(
        self,
        step_name: str,
        reason: str,
        agent_id: str | None = None,
    ) -> None:
        log(
            "○",
            f"  {step_name} {Colors.YELLOW}skipped: {reason}{Colors.RESET}",
            agent_id=agent_id or "run",
        )

    def on_validation_step_passed(
        self,
        step_name: str,
        duration_seconds: float,
        agent_id: str | None = None,
    ) -> None:
        log(
            "✓",
            f"  {step_name} ({duration_seconds:.1f}s)",
            agent_id=agent_id or "run",
        )

    def on_validation_step_failed(
        self,
        step_name: str,
        exit_code: int,
        agent_id: str | None = None,
    ) -> None:
        log(
            "✗",
            f"  {step_name} {Colors.RED}exit {exit_code}{Colors.RESET}",
            agent_id=agent_id or "run",
        )

    # -------------------------------------------------------------------------
    # Warnings and diagnostics
    # -------------------------------------------------------------------------

    def on_warning(self, message: str, agent_id: str | None = None) -> None:
        log("⚠", f"{Colors.YELLOW}{message}{Colors.RESET}", agent_id=agent_id or "run")

    def on_log_timeout(self, agent_id: str, log_path: str) -> None:
        log(
            "⚠",
            f"{Colors.YELLOW}Log timeout. Check: {log_path}{Colors.RESET}",
            agent_id=agent_id,
        )

    def on_locks_cleaned(self, agent_id: str, count: int) -> None:
        log("→", f"Cleaned {count} stale locks", agent_id=agent_id)

    def on_locks_released(self, count: int) -> None:
        log("→", f"Released {count} locks", agent_id="run")

    def on_issues_committed(self) -> None:
        log("→", "COMMIT Issues committed", agent_id="run")

    def on_run_metadata_saved(self, path: str) -> None:
        log("◦", f"Run metadata saved to {path}", agent_id="run")

    def on_global_validation_disabled(self) -> None:
        log_verbose("◦", "Global validation disabled", agent_id="run")

    def on_abort_requested(self, reason: str) -> None:
        log("⚠", f"{Colors.YELLOW}ABORT {reason}{Colors.RESET}", agent_id="run")

    def on_tasks_aborting(self, count: int, reason: str) -> None:
        log("→", f"ABORT Cancelling {count} tasks: {reason}", agent_id="run")

    # -------------------------------------------------------------------------
    # SIGINT escalation lifecycle
    # -------------------------------------------------------------------------

    def on_drain_started(self, active_task_count: int) -> None:
        log(
            "→",
            f"Ctrl-C: draining {active_task_count} active task(s)...",
            agent_id="run",
        )

    def on_abort_started(self) -> None:
        log("→", "Ctrl-C: aborting...", agent_id="run")

    def on_force_abort(self) -> None:
        log("→", "Ctrl-C: force killing...", agent_id="run")

    # -------------------------------------------------------------------------
    # Epic verification lifecycle
    # -------------------------------------------------------------------------

    def on_epic_verification_started(
        self, epic_id: str, *, reviewer_type: str = "agent_sdk"
    ) -> None:
        log("→", f"VERIFY Starting verification for {epic_id}", agent_id="epic")

    def on_epic_verification_passed(
        self, epic_id: str, *, reviewer_type: str = "agent_sdk"
    ) -> None:
        log(
            "✓",
            f"VERIFY {epic_id} passed",
            agent_id="epic",
        )

    def on_epic_verification_failed(
        self,
        epic_id: str,
        unmet_count: int,
        remediation_ids: list[str],
        *,
        reason: str | None = None,
        reviewer_type: str = "agent_sdk",
    ) -> None:
        if reason:
            log(
                "✗",
                f"VERIFY {Colors.RED}{epic_id}: {reason}{Colors.RESET}",
                agent_id="epic",
            )
        else:
            log(
                "✗",
                f"VERIFY {Colors.RED}{epic_id}: {unmet_count} criteria unmet{Colors.RESET}",
                agent_id="epic",
            )
        for issue_id in remediation_ids:
            log("→", f"  → Remediation: {issue_id}", agent_id="epic")

    def on_epic_remediation_created(
        self,
        epic_id: str,
        issue_id: str,
        criterion: str,
    ) -> None:
        truncated = truncate_text(criterion, 80)
        log("→", f"REMEDIATE {epic_id} → {issue_id}: {truncated}", agent_id="epic")

    # -------------------------------------------------------------------------
    # Pipeline module events
    # -------------------------------------------------------------------------

    def on_lifecycle_state(self, agent_id: str, state: str) -> None:
        log_verbose("◦", f"LIFECYCLE {state}", agent_id=agent_id)

    def on_log_waiting(self, agent_id: str) -> None:
        log_verbose("◦", "LOG Waiting for session log...", agent_id=agent_id)

    def on_log_ready(self, agent_id: str) -> None:
        log_verbose("◦", "LOG Session log ready", agent_id=agent_id)

    def on_review_skipped_no_progress(self, agent_id: str) -> None:
        log(
            "⚠",
            f"REVIEW {Colors.YELLOW}Skipped (no code changes){Colors.RESET}",
            agent_id=agent_id,
        )

    def on_fixer_text(self, attempt: int, text: str) -> None:
        # Strip ANSI codes for cleaner output
        clean_text = re.sub(r"\x1b\[[0-9;]*m", "", text)
        # log_agent_text handles truncation; agent_id already contains attempt number
        log_agent_text(clean_text, f"fixer-{attempt}")

    def on_fixer_tool_use(
        self,
        attempt: int,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> None:
        log_tool(tool_name, "", agent_id=f"fixer-{attempt}", arguments=arguments)

    def on_deadlock_detected(self, info: DeadlockInfoProtocol) -> None:
        cycle_str = " → ".join(info.cycle)
        victim_issue = info.victim_issue_id or "unknown"
        blocker_issue = info.blocker_issue_id or "unknown"
        log("⚠", f"{Colors.YELLOW}Deadlock detected{Colors.RESET}")
        log("◦", f"  cycle: {cycle_str}")
        log("◦", f"  victim: {info.victim_id} ({victim_issue})")
        log("◦", f"  blocked_on: {info.blocked_on}")
        log("◦", f"  blocker: {info.blocker_id} ({blocker_issue})")

    def on_watch_idle(self, wait_seconds: float, issues_blocked: int | None) -> None:
        poll_s = (
            math.ceil(wait_seconds) if math.isfinite(wait_seconds) else wait_seconds
        )
        if issues_blocked is None or issues_blocked == 0:
            log(
                "◦",
                f"{Colors.MUTED}Idle: no ready issues. Polling in {poll_s}s...{Colors.RESET}",
                agent_id="run",
            )
        else:
            log(
                "◦",
                f"{Colors.MUTED}Idle: {issues_blocked} issues exist but none ready. "
                f"Polling in {poll_s}s...{Colors.RESET}",
                agent_id="run",
            )

    # -------------------------------------------------------------------------
    # Trigger validation lifecycle
    # -------------------------------------------------------------------------

    def on_trigger_validation_queued(
        self, trigger_type: str, trigger_context: str
    ) -> None:
        log("◦", f"[{trigger_type}] queued: {trigger_context}", agent_id="trigger")

    def on_trigger_validation_started(
        self, trigger_type: str, commands: list[str]
    ) -> None:
        cmds_str = ", ".join(commands) if commands else "(none)"
        log("→", f"[{trigger_type}] VALIDATE {cmds_str}", agent_id="trigger")

    def on_trigger_command_started(
        self, trigger_type: str, command_ref: str, index: int, total_commands: int
    ) -> None:
        log(
            "◦",
            f"[{trigger_type}] ({index + 1}/{total_commands}) {command_ref.upper()} ...",
            agent_id="trigger",
        )

    def on_trigger_command_completed(
        self,
        trigger_type: str,
        command_ref: str,
        index: int,
        total_commands: int,
        passed: bool,
        duration_seconds: float,
    ) -> None:
        status = (
            f"{Colors.GREEN}✓{Colors.RESET}"
            if passed
            else f"{Colors.RED}✗{Colors.RESET}"
        )
        log(
            "◦",
            f"[{trigger_type}] ({index + 1}/{total_commands}) {command_ref.upper()} {status} ({duration_seconds:.1f}s)",
            agent_id="trigger",
        )

    def on_trigger_validation_passed(
        self, trigger_type: str, duration_seconds: float
    ) -> None:
        log(
            "✓",
            f"{Colors.GREEN}[{trigger_type}] validation_passed{Colors.RESET} ({duration_seconds:.1f}s)",
            agent_id="trigger",
        )

    def on_trigger_validation_failed(
        self, trigger_type: str, failed_command: str, failure_mode: str
    ) -> None:
        log(
            "✗",
            f"{Colors.RED}[{trigger_type}] validation_failed{Colors.RESET}: {failed_command} (mode={failure_mode})",
            agent_id="trigger",
        )

    def on_trigger_validation_skipped(self, trigger_type: str, reason: str) -> None:
        log(
            "◦",
            f"{Colors.MUTED}[{trigger_type}] validation_skipped: {reason}{Colors.RESET}",
            agent_id="trigger",
        )

    def on_trigger_remediation_started(
        self, trigger_type: str, attempt: int, max_retries: int
    ) -> None:
        log(
            "→",
            f"[{trigger_type}] remediation_started: attempt {attempt}/{max_retries}",
            agent_id="trigger",
        )

    def on_trigger_remediation_succeeded(self, trigger_type: str, attempt: int) -> None:
        log(
            "✓",
            f"{Colors.GREEN}[{trigger_type}] remediation_succeeded{Colors.RESET} on attempt {attempt}",
            agent_id="trigger",
        )

    def on_trigger_remediation_exhausted(
        self, trigger_type: str, attempts: int
    ) -> None:
        log(
            "✗",
            f"{Colors.RED}[{trigger_type}] remediation_exhausted{Colors.RESET}: {attempts} attempts",
            agent_id="trigger",
        )

    # -------------------------------------------------------------------------
    # Trigger code review lifecycle
    # -------------------------------------------------------------------------

    def on_trigger_code_review_started(self, trigger_type: str) -> None:
        log("→", f"[{trigger_type}] code_review_started", agent_id="trigger")

    def on_trigger_code_review_skipped(self, trigger_type: str, reason: str) -> None:
        log(
            "◦",
            f"{Colors.MUTED}[{trigger_type}] code_review_skipped: {reason}{Colors.RESET}",
            agent_id="trigger",
        )

    def on_trigger_code_review_passed(self, trigger_type: str) -> None:
        log(
            "✓",
            f"{Colors.GREEN}[{trigger_type}] code_review_passed{Colors.RESET}",
            agent_id="trigger",
        )

    def on_trigger_code_review_failed(
        self, trigger_type: str, blocking_count: int
    ) -> None:
        log(
            "✗",
            f"{Colors.RED}[{trigger_type}] code_review_failed ({blocking_count} blocking){Colors.RESET}",
            agent_id="trigger",
        )

    def on_trigger_code_review_error(self, trigger_type: str, error: str) -> None:
        log(
            "✗",
            f"{Colors.RED}[{trigger_type}] code_review_error: {error}{Colors.RESET}",
            agent_id="trigger",
        )

    # -------------------------------------------------------------------------
    # Session end lifecycle
    # -------------------------------------------------------------------------

    def on_session_end_started(self, issue_id: str) -> None:
        log(
            "→",
            "[session_end] started",
            issue_id=issue_id,
        )

    def on_session_end_completed(self, issue_id: str, result: str) -> None:
        log(
            "✓",
            f"[session_end] completed: result={Colors.GREEN}{result}{Colors.RESET}",
            issue_id=issue_id,
        )

    def on_session_end_skipped(self, issue_id: str, reason: str) -> None:
        log(
            "○",
            f"[session_end] skipped: reason={reason}",
            issue_id=issue_id,
        )


# Protocol assertion to verify implementation compliance
assert isinstance(ConsoleEventSink(), MalaEventSink)
