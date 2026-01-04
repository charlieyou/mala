"""Console event sink implementation for MalaOrchestrator.

Provides ConsoleEventSink which outputs orchestrator events to the console
using the log helpers from log_output/console.py.
"""

import re
from typing import Any

from src.core.protocols import DeadlockInfoProtocol, EventRunConfig, MalaEventSink
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
        if config.prioritize_wip:
            log("◦", "Mode: prioritize WIP", agent_id="run")
        if config.orphans_only:
            log("◦", "Mode: orphans only", agent_id="run")
        log_verbose("◦", f"Parallelism: {config.max_agents}", agent_id="run")
        self._log_limits(config)
        self._log_review_config(config)
        self._log_braintrust_config(config)
        self._log_cli_args(config)

    def _log_limits(self, config: EventRunConfig) -> None:
        limits_parts: list[str] = []
        if config.timeout_minutes is not None:
            limits_parts.append(f"Time: {config.timeout_minutes} min")
        if config.max_issues is not None:
            limits_parts.append(f"Issues: {config.max_issues}")
        if limits_parts:
            log_verbose("◦", f"Limits: {', '.join(limits_parts)}", agent_id="run")

    def _log_review_config(self, config: EventRunConfig) -> None:
        review_type = "LLM review" if config.review_enabled else "disabled"
        log_verbose("◦", f"Review: {review_type}", agent_id="run")
        log_verbose(
            "◦",
            f"Review retries: {config.max_review_retries} "
            f"(gate retries: {config.max_gate_retries})",
            agent_id="run",
        )

    def _log_braintrust_config(self, config: EventRunConfig) -> None:
        braintrust_mode = "enabled" if config.braintrust_enabled else "disabled"
        log_verbose("◦", f"Braintrust: {braintrust_mode}", agent_id="run")

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

    def on_run_completed(
        self,
        success_count: int,
        total_count: int,
        run_validation_passed: bool,
        abort_reason: str | None = None,
    ) -> None:
        status_icon = "✓" if success_count == total_count else "✗"
        status = f"DONE {status_icon} {success_count}/{total_count} issues completed"
        if abort_reason:
            status += f" (aborted: {abort_reason})"
        log("→", status, agent_id="run")
        if run_validation_passed:
            log("✓", "RUN VALIDATION passed", agent_id="run")
        else:
            log(
                "✗",
                f"RUN VALIDATION {Colors.RED}failed{Colors.RESET}",
                agent_id="run",
            )

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
        log(
            status_icon,
            f"{issue_id} completed in {duration_seconds:.1f}s: {summary}",
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

    def on_run_level_validation_disabled(self) -> None:
        log_verbose("◦", "Run-level validation disabled", agent_id="run")

    def on_abort_requested(self, reason: str) -> None:
        log("⚠", f"{Colors.YELLOW}ABORT {reason}{Colors.RESET}", agent_id="run")

    def on_tasks_aborting(self, count: int, reason: str) -> None:
        log("→", f"ABORT Cancelling {count} tasks: {reason}", agent_id="run")

    # -------------------------------------------------------------------------
    # Epic verification lifecycle
    # -------------------------------------------------------------------------

    def on_epic_verification_started(self, epic_id: str) -> None:
        log("→", f"VERIFY Starting verification for {epic_id}", agent_id="epic")

    def on_epic_verification_passed(self, epic_id: str, confidence: float) -> None:
        log(
            "✓",
            f"VERIFY {epic_id} passed (confidence: {confidence:.0%})",
            agent_id="epic",
        )

    def on_epic_verification_failed(
        self,
        epic_id: str,
        unmet_count: int,
        remediation_ids: list[str],
    ) -> None:
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
        log("→", f"[{attempt}] {clean_text}", agent_id="fixer")

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
        log(
            "⚠",
            f"{Colors.YELLOW}Deadlock detected:{Colors.RESET} cycle=[{cycle_str}] "
            f"victim={info.victim_id}({victim_issue}) blocked_on={info.blocked_on} "
            f"blocker={info.blocker_id}({blocker_issue})",
        )


# Protocol assertion to verify implementation compliance
assert isinstance(ConsoleEventSink(), MalaEventSink)
