"""Event sink implementations for MalaOrchestrator.

Provides concrete implementations of the MalaEventSink protocol:
- BaseEventSink: Base class with no-op implementations
- NullEventSink: Silent sink for testing
- ConsoleEventSink: Full console output implementation
"""

import re
from typing import Any

from .base_sink import BaseEventSink, NullEventSink
from .event_protocol import EventRunConfig, MalaEventSink
from .log_output.console import (
    Colors,
    log,
    log_agent_text,
    log_tool,
    log_verbose,
    truncate_text,
)

__all__ = [
    "BaseEventSink",
    "ConsoleEventSink",
    "NullEventSink",
]


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
        log("run", f"[START] {config.run_id}")
        log_verbose("run", f"Parallelism: {config.max_agents}")
        log_verbose("run", f"Dry-run: {config.dry_run}")
        self._log_limits(config)
        self._log_review_config(config)
        self._log_filters(config)
        self._log_braintrust_config(config)
        self._log_cli_args(config)

    def _log_limits(self, config: EventRunConfig) -> None:
        limits_parts: list[str] = []
        if config.time_limit_minutes is not None:
            limits_parts.append(f"Time: {config.time_limit_minutes} min")
        if config.issue_limit is not None:
            limits_parts.append(f"Issues: {config.issue_limit}")
        if limits_parts:
            log_verbose("run", f"Limits: {', '.join(limits_parts)}")

    def _log_review_config(self, config: EventRunConfig) -> None:
        review_type = "LLM review"
        log_verbose("run", f"Review: {review_type}")
        log_verbose(
            "run",
            f"Review retries: {config.review_max_retries} "
            f"(gate retries: {config.gate_max_retries})",
        )

    def _log_filters(self, config: EventRunConfig) -> None:
        if config.issue_filter:
            log_verbose(
                "run",
                f"Include filter: {config.issue_filter}",
            )
        if config.exclude_filter:
            log_verbose(
                "run",
                f"Exclude filter: {config.exclude_filter}",
            )
        if config.type_filter:
            log_verbose(
                "run",
                f"Type filter: {', '.join(config.type_filter)}",
            )

    def _log_braintrust_config(self, config: EventRunConfig) -> None:
        braintrust_mode = "enabled" if config.braintrust_project else "disabled"
        if config.braintrust_project:
            log_verbose(
                "run",
                f"Braintrust: {braintrust_mode} (project={config.braintrust_project})",
            )
        else:
            log_verbose("run", f"Braintrust: {braintrust_mode}")

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
                log_verbose("run", f"CLI args: {safe_args}")

    def on_run_completed(
        self,
        success_count: int,
        total_count: int,
        run_validation_passed: bool,
        abort_reason: str | None = None,
    ) -> None:
        status_icon = "✓" if success_count == total_count else "✗"
        status = f"[DONE] {status_icon} {success_count}/{total_count} issues completed"
        if abort_reason:
            status += f" (aborted: {abort_reason})"
        log("run", status)
        if run_validation_passed:
            log("run", "[RUN VALIDATION] ✓ passed")
        else:
            log("run", f"[RUN VALIDATION] {Colors.RED}✗ failed{Colors.RESET}")

    def on_ready_issues(self, issue_ids: list[str]) -> None:
        log("run", f"Ready issues ({len(issue_ids)}): {issue_ids}")

    def on_waiting_for_agents(self, count: int) -> None:
        log_verbose("run", f"Waiting for {count} agents to complete...")

    def on_no_more_issues(self, reason: str) -> None:
        log("run", f"No more issues: {reason}")

    # -------------------------------------------------------------------------
    # Agent lifecycle
    # -------------------------------------------------------------------------

    def on_agent_started(self, agent_id: str, issue_id: str) -> None:
        log(agent_id, f"[→] Claimed {issue_id}")

    def on_agent_completed(
        self,
        agent_id: str,
        issue_id: str,
        success: bool,
        duration_seconds: float,
        summary: str,
    ) -> None:
        status = "✓" if success else "✗"
        log(
            agent_id,
            f"[{status}] Completed {issue_id} in {duration_seconds:.1f}s: {summary}",
        )

    def on_claim_failed(self, agent_id: str, issue_id: str) -> None:
        log(agent_id, f"[SKIP] {issue_id} already claimed")

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
        log_tool(agent_id, tool_name, description, arguments)

    def on_agent_text(self, agent_id: str, text: str) -> None:
        log_agent_text(agent_id, text)

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
        scope = f" ({issue_id})" if issue_id else ""
        tag = agent_id or "run"
        log(tag, f"[GATE{scope}] Attempt {attempt}/{max_attempts}")

    def on_gate_passed(
        self,
        agent_id: str | None,
        issue_id: str | None = None,
    ) -> None:
        scope = f" ({issue_id})" if issue_id else ""
        tag = agent_id or "run"
        log(tag, f"[GATE{scope}] ✓ passed")

    def on_gate_failed(
        self,
        agent_id: str | None,
        attempt: int,
        max_attempts: int,
        issue_id: str | None = None,
    ) -> None:
        scope = f" ({issue_id})" if issue_id else ""
        tag = agent_id or "run"
        log(
            tag,
            f"[GATE{scope}] {Colors.RED}✗ failed{Colors.RESET} ({attempt}/{max_attempts})",
        )

    def on_gate_retry(
        self,
        agent_id: str,
        attempt: int,
        max_attempts: int,
        issue_id: str | None = None,
    ) -> None:
        scope = f" ({issue_id})" if issue_id else ""
        log(agent_id, f"[GATE{scope}] Retry {attempt}/{max_attempts}")

    def on_gate_result(
        self,
        agent_id: str | None,
        passed: bool,
        failure_reasons: list[str] | None = None,
        issue_id: str | None = None,
    ) -> None:
        scope = f" ({issue_id})" if issue_id else ""
        tag = agent_id or "run"
        if passed:
            log(tag, f"[GATE{scope}] ✓ all checks passed")
        elif failure_reasons:
            log(
                tag,
                f"[GATE{scope}] {Colors.RED}✗ {len(failure_reasons)} checks failed{Colors.RESET}",
            )
            for reason in failure_reasons:
                log(tag, f"  - {reason}")

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
        scope = f" ({issue_id})" if issue_id else ""
        log(agent_id, f"[REVIEW{scope}] Attempt {attempt}/{max_attempts}")

    def on_review_passed(
        self,
        agent_id: str,
        issue_id: str | None = None,
    ) -> None:
        scope = f" ({issue_id})" if issue_id else ""
        log(agent_id, f"[REVIEW{scope}] ✓ approved")

    def on_review_retry(
        self,
        agent_id: str,
        attempt: int,
        max_attempts: int,
        error_count: int | None = None,
        parse_error: str | None = None,
        issue_id: str | None = None,
    ) -> None:
        scope = f" ({issue_id})" if issue_id else ""
        details = ""
        if error_count is not None:
            details = f" ({error_count} errors)"
        elif parse_error:
            details = f" (parse error: {parse_error})"
        log(agent_id, f"[REVIEW{scope}] Retry {attempt}/{max_attempts}{details}")

    def on_review_warning(
        self,
        message: str,
        agent_id: str | None = None,
        issue_id: str | None = None,
    ) -> None:
        scope = f" ({issue_id})" if issue_id else ""
        tag = agent_id or "run"
        log(tag, f"[REVIEW{scope}] {Colors.YELLOW}⚠ {message}{Colors.RESET}")

    # -------------------------------------------------------------------------
    # Fixer agent events
    # -------------------------------------------------------------------------

    def on_fixer_started(
        self,
        attempt: int,
        max_attempts: int,
    ) -> None:
        log("fixer", f"[FIXER] Attempt {attempt}/{max_attempts}")

    def on_fixer_completed(self, result: str) -> None:
        log("fixer", f"[FIXER] ✓ {result}")

    def on_fixer_failed(self, reason: str) -> None:
        log("fixer", f"[FIXER] {Colors.RED}✗ {reason}{Colors.RESET}")

    # -------------------------------------------------------------------------
    # Issue lifecycle
    # -------------------------------------------------------------------------

    def on_issue_closed(self, agent_id: str, issue_id: str) -> None:
        log(agent_id, f"[CLOSE] {issue_id}")

    def on_issue_completed(
        self,
        agent_id: str,
        issue_id: str,
        success: bool,
        duration_seconds: float,
        summary: str,
    ) -> None:
        status = "✓" if success else "✗"
        log(
            agent_id,
            f"[{status}] {issue_id} completed in {duration_seconds:.1f}s: {summary}",
        )

    def on_epic_closed(self, agent_id: str) -> None:
        log(agent_id, "[EPIC] Closed")

    def on_validation_started(
        self,
        agent_id: str,
        issue_id: str | None = None,
    ) -> None:
        scope = f" ({issue_id})" if issue_id else ""
        log(agent_id, f"[VALIDATE{scope}] Starting validation")

    def on_validation_result(
        self,
        agent_id: str,
        passed: bool,
        issue_id: str | None = None,
    ) -> None:
        scope = f" ({issue_id})" if issue_id else ""
        status = "✓" if passed else "✗"
        log(agent_id, f"[VALIDATE{scope}] {status}")

    def on_validation_step_running(
        self,
        step_name: str,
        agent_id: str | None = None,
    ) -> None:
        tag = agent_id or "run"
        log(tag, f"  [{step_name}] running...")

    def on_validation_step_skipped(
        self,
        step_name: str,
        reason: str,
        agent_id: str | None = None,
    ) -> None:
        tag = agent_id or "run"
        log(tag, f"  [{step_name}] {Colors.YELLOW}skipped: {reason}{Colors.RESET}")

    def on_validation_step_passed(
        self,
        step_name: str,
        duration_seconds: float,
        agent_id: str | None = None,
    ) -> None:
        tag = agent_id or "run"
        log(tag, f"  [{step_name}] ✓ ({duration_seconds:.1f}s)")

    def on_validation_step_failed(
        self,
        step_name: str,
        exit_code: int,
        agent_id: str | None = None,
    ) -> None:
        tag = agent_id or "run"
        log(tag, f"  [{step_name}] {Colors.RED}✗ exit {exit_code}{Colors.RESET}")

    # -------------------------------------------------------------------------
    # Warnings and diagnostics
    # -------------------------------------------------------------------------

    def on_warning(self, message: str, agent_id: str | None = None) -> None:
        tag = agent_id or "run"
        log(tag, f"{Colors.YELLOW}⚠ {message}{Colors.RESET}")

    def on_log_timeout(self, agent_id: str, log_path: str) -> None:
        log(agent_id, f"{Colors.YELLOW}⚠ Log timeout. Check: {log_path}{Colors.RESET}")

    def on_locks_cleaned(self, agent_id: str, count: int) -> None:
        log(agent_id, f"Cleaned {count} stale locks")

    def on_locks_released(self, count: int) -> None:
        log("run", f"Released {count} locks")

    def on_issues_committed(self) -> None:
        log("run", "[COMMIT] Issues committed")

    def on_run_metadata_saved(self, path: str) -> None:
        log_verbose("run", f"Run metadata saved to {path}")

    def on_run_level_validation_disabled(self) -> None:
        log_verbose("run", "Run-level validation disabled")

    def on_abort_requested(self, reason: str) -> None:
        log("run", f"{Colors.YELLOW}[ABORT] {reason}{Colors.RESET}")

    def on_tasks_aborting(self, count: int, reason: str) -> None:
        log("run", f"[ABORT] Cancelling {count} tasks: {reason}")

    # -------------------------------------------------------------------------
    # Epic verification lifecycle
    # -------------------------------------------------------------------------

    def on_epic_verification_started(self, epic_id: str) -> None:
        log("epic", f"[VERIFY] Starting verification for {epic_id}")

    def on_epic_verification_passed(self, epic_id: str, confidence: float) -> None:
        log("epic", f"[VERIFY] ✓ {epic_id} passed (confidence: {confidence:.0%})")

    def on_epic_verification_failed(
        self,
        epic_id: str,
        unmet_count: int,
        remediation_ids: list[str],
    ) -> None:
        log(
            "epic",
            f"[VERIFY] {Colors.RED}✗ {epic_id}: {unmet_count} criteria unmet{Colors.RESET}",
        )
        for issue_id in remediation_ids:
            log("epic", f"  → Remediation: {issue_id}")

    def on_epic_verification_human_review(
        self,
        epic_id: str,
        reason: str,
        review_issue_id: str,
    ) -> None:
        log(
            "epic",
            f"[VERIFY] {Colors.YELLOW}? {epic_id} needs human review: {reason}{Colors.RESET}",
        )
        log("epic", f"  → Review issue: {review_issue_id}")

    def on_epic_remediation_created(
        self,
        epic_id: str,
        issue_id: str,
        criterion: str,
    ) -> None:
        truncated = truncate_text(criterion, 80)
        log("epic", f"[REMEDIATE] {epic_id} → {issue_id}: {truncated}")

    # -------------------------------------------------------------------------
    # Pipeline module events
    # -------------------------------------------------------------------------

    def on_lifecycle_state(self, agent_id: str, state: str) -> None:
        log_verbose(agent_id, f"[LIFECYCLE] {state}")

    def on_log_waiting(self, agent_id: str) -> None:
        log_verbose(agent_id, "[LOG] Waiting for session log...")

    def on_log_ready(self, agent_id: str) -> None:
        log_verbose(agent_id, "[LOG] Session log ready")

    def on_review_skipped_no_progress(self, agent_id: str) -> None:
        log(
            agent_id,
            f"[REVIEW] {Colors.YELLOW}⚠ Skipped (no code changes){Colors.RESET}",
        )

    def on_fixer_text(self, attempt: int, text: str) -> None:
        # Strip ANSI codes for cleaner output
        clean_text = re.sub(r"\x1b\[[0-9;]*m", "", text)
        log("fixer", f"[{attempt}] {clean_text}")

    def on_fixer_tool_use(
        self,
        attempt: int,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> None:
        log_tool(f"fixer-{attempt}", tool_name, "", arguments)


# Protocol assertion to verify implementation compliance
assert isinstance(ConsoleEventSink(), MalaEventSink)
