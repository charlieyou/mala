#!/usr/bin/env python3
# ruff: noqa: E402
"""
mala CLI: Agent SDK orchestrator for parallel issue processing.

Usage:
    mala run [OPTIONS] [REPO_PATH]
    mala epic-verify [OPTIONS] EPIC_ID [REPO_PATH]
    mala clean
    mala status
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..orchestration.cli_support import USER_CONFIG_DIR, get_runs_dir, load_user_env

if TYPE_CHECKING:
    from src.infra.io.config import MalaConfig, ResolvedConfig

# Bootstrap state: tracks whether bootstrap() has been called
# These are populated on first access via __getattr__
_bootstrapped = False
_braintrust_enabled = False

# Lazy-loaded module cache for SDK-dependent imports
# These are populated on first access via __getattr__
_lazy_modules: dict[str, Any] = {}

# Names that are lazily loaded (for __getattr__ to handle)
_LAZY_NAMES = frozenset(
    {
        "BeadsClient",
        "MalaConfig",
        "MalaOrchestrator",
        "OrchestratorConfig",
        "create_orchestrator",
        "get_lock_dir",
        "get_running_instances",
        "get_running_instances_for_dir",
    }
)


def _lazy(name: str) -> Any:  # noqa: ANN401
    """Access a lazy-loaded module attribute.

    This function provides access to SDK-dependent imports that are
    lazy-loaded via __getattr__. Using this avoids runtime imports
    inside command functions while still deferring SDK loading.
    """
    return getattr(sys.modules[__name__], name)


def bootstrap() -> None:
    """Initialize environment and Braintrust tracing.

    Must be called before using any CLI commands or importing claude_agent_sdk.
    This function is idempotent - calling it multiple times has no additional effect.

    Side effects:
        - Loads environment variables from ~/.config/mala/.env
        - Sets up Braintrust SDK patching if BRAINTRUST_API_KEY is set
    """
    global _bootstrapped, _braintrust_enabled

    if _bootstrapped:
        return

    # Load environment variables early (needed for config)
    load_user_env()

    # Setup Braintrust tracing BEFORE importing claude_agent_sdk anywhere
    # This patches the SDK before any imports can use the unpatched version
    # Note: We use MalaConfig here to read BRAINTRUST_API_KEY consistently
    # from the environment (which now includes user .env after load_user_env)
    from src.infra.io.config import MalaConfig

    config = MalaConfig.from_env(validate=False)
    if config.braintrust_api_key:
        try:
            from braintrust.wrappers.claude_agent_sdk import setup_claude_agent_sdk

            setup_claude_agent_sdk(project="mala")
            _braintrust_enabled = True
        except ImportError:
            pass  # braintrust not installed

    _bootstrapped = True


def is_braintrust_enabled() -> bool:
    """Return whether Braintrust tracing was successfully enabled.

    Must be called after bootstrap().
    """
    return _braintrust_enabled


# Import modules that do NOT depend on claude_agent_sdk at module level
import asyncio
from datetime import datetime
from typing import Annotated, Never

import typer

from ..orchestration.cli_support import Colors, log, set_verbose

# SDK-dependent imports (BeadsClient, MalaOrchestrator, get_lock_dir, run_metadata)
# are lazy-loaded via __getattr__ to ensure bootstrap() runs before claude_agent_sdk
# is imported. Access them as module attributes: BeadsClient, MalaOrchestrator, etc.


def display_dry_run_tasks(
    issues: list[dict[str, object]],
    focus: bool,
) -> None:
    """Display task order for dry-run preview.

    Shows tasks in order with epic grouping when focus mode is enabled.

    Args:
        issues: List of issue dicts with id, title, priority, status, parent_epic.
        focus: If True, display with epic headers for grouped output.
    """
    if not issues:
        log("○", "No ready tasks found", Colors.GRAY)
        return

    print()
    log("◐", f"Dry run: {len(issues)} task(s) would be processed", Colors.CYAN)
    print()

    if focus:
        # Group by epic for display
        current_epic: str | None = None
        epic_count = 0
        task_in_epic = 0
        for issue in issues:
            epic = issue.get("parent_epic")
            # Ensure epic is a string or None
            epic_str: str | None = str(epic) if epic is not None else None
            if epic_str != current_epic:
                if current_epic is not None or (
                    current_epic is None and task_in_epic > 0
                ):
                    print()  # Add spacing between epics
                current_epic = epic_str
                epic_count += 1
                task_in_epic = 0
                if epic_str:
                    print(f"  {Colors.MAGENTA}▸ Epic: {epic_str}{Colors.RESET}")
                else:
                    print(f"  {Colors.GRAY}▸ (Orphan tasks){Colors.RESET}")

            task_in_epic += 1
            _print_task_line(issue, indent="    ")
    else:
        # Simple flat list - show epic per task since there are no group headers
        for issue in issues:
            _print_task_line(issue, indent="  ", show_epic=True)

    print()
    # Summary: count tasks per epic
    if focus:
        epic_counts: dict[str | None, int] = {}
        for issue in issues:
            epic = issue.get("parent_epic")
            epic_key: str | None = str(epic) if epic is not None else None
            epic_counts[epic_key] = epic_counts.get(epic_key, 0) + 1

        epic_summary = ", ".join(
            f"{epic or '(orphan)'}: {count}"
            for epic, count in sorted(
                epic_counts.items(), key=lambda x: (x[0] is None, x[0] or "")
            )
        )
        log("◐", f"By epic: {epic_summary}", Colors.MUTED)


def _print_task_line(
    issue: dict[str, object], indent: str = "  ", show_epic: bool = False
) -> None:
    """Print a single task line with ID, priority, title, and optionally epic."""
    issue_id = issue.get("id", "?")
    title = issue.get("title", "")
    priority = issue.get("priority")
    status = issue.get("status", "")
    parent_epic = issue.get("parent_epic")

    # Format priority badge
    prio_str = f"P{priority}" if priority is not None else "P?"

    # Status indicator
    status_indicator = ""
    if status == "in_progress":
        status_indicator = f" {Colors.YELLOW}(WIP){Colors.RESET}"

    # Epic indicator (shown in non-focus mode)
    epic_indicator = ""
    if show_epic and parent_epic:
        epic_indicator = f" {Colors.MUTED}({parent_epic}){Colors.RESET}"

    print(
        f"{indent}{Colors.CYAN}{issue_id}{Colors.RESET} "
        f"[{Colors.MUTED}{prio_str}{Colors.RESET}] "
        f"{title}{epic_indicator}{status_indicator}"
    )


# Valid values for --disable-validations flag
# Each value controls a specific validation phase:
#   post-validate: Skip all validation commands (pytest, ruff, ty) after agent commits
#   run-level-validate: Skip run-level validation at end of batch (reserved for future use)
#   integration-tests: Exclude @pytest.mark.integration tests from pytest runs
#   coverage: Disable code coverage enforcement (threshold check)
#   e2e: Disable end-to-end fixture repo tests (run-level only)
#   review: Disable automated LLM code review after quality gate passes
#   followup-on-run-validate-fail: Disable auto-retry on run validation failures (reserved)
VALID_DISABLE_VALUES = frozenset(
    {
        "post-validate",
        "run-level-validate",
        "integration-tests",
        "coverage",
        "e2e",
        "review",
        "followup-on-run-validate-fail",
    }
)


@dataclass(frozen=True)
class ValidatedRunArgs:
    """Validated and parsed CLI arguments for the run command."""

    only_ids: set[str] | None
    disable_set: set[str] | None
    epic_override_ids: set[str] | None


@dataclass(frozen=True)
class ConfigOverrideResult:
    """Result of applying CLI overrides to config."""

    resolved: ResolvedConfig
    updated_config: MalaConfig


def _build_cli_args_metadata(
    *,
    disable_validations: str | None,
    coverage_threshold: float | None,
    wip: bool,
    max_issues: int | None,
    max_gate_retries: int,
    max_review_retries: int,
    no_morph: bool,
    epic_override: str | None,
    resolved: ResolvedConfig,
    braintrust_enabled: bool,
) -> dict[str, object]:
    """Build the cli_args metadata dictionary for logging and OrchestratorConfig.

    Args:
        disable_validations: Raw disable validations string from CLI.
        coverage_threshold: Coverage threshold from CLI.
        wip: Whether WIP prioritization is enabled.
        max_issues: Maximum issues to process.
        max_gate_retries: Maximum gate retry attempts.
        max_review_retries: Maximum review retry attempts.
        no_morph: Whether morph is disabled.
        epic_override: Raw epic override string from CLI.
        resolved: Resolved config with effective values.
        braintrust_enabled: Whether braintrust actually initialized successfully.

    Returns:
        Dictionary of CLI arguments for logging/metadata.
    """
    return {
        "disable_validations": disable_validations,
        "coverage_threshold": coverage_threshold,
        "wip": wip,
        "max_issues": max_issues,
        "max_gate_retries": max_gate_retries,
        "max_review_retries": max_review_retries,
        "braintrust": braintrust_enabled,
        "no_morph": no_morph,
        "review_timeout": resolved.review_timeout,
        "cerberus_spawn_args": list(resolved.cerberus_spawn_args),
        "cerberus_wait_args": list(resolved.cerberus_wait_args),
        "cerberus_env": dict(resolved.cerberus_env),
        "epic_override": epic_override,
        "max_epic_verification_retries": resolved.max_epic_verification_retries,
    }


def _apply_config_overrides(
    config: MalaConfig,
    review_timeout: int | None,
    cerberus_spawn_args: str | None,
    cerberus_wait_args: str | None,
    cerberus_env: str | None,
    max_epic_verification_retries: int | None,
    no_morph: bool,
    braintrust_enabled: bool,
    disable_review: bool,
) -> ConfigOverrideResult:
    """Apply CLI overrides to MalaConfig, raising typer.Exit(1) on parse errors.

    Args:
        config: Base MalaConfig from environment.
        review_timeout: Optional review timeout override.
        cerberus_spawn_args: Raw string of extra args for spawn.
        cerberus_wait_args: Raw string of extra args for wait.
        cerberus_env: Raw string of extra env vars (key=value pairs).
        max_epic_verification_retries: Optional max retries override.
        no_morph: Whether to disable morph.
        braintrust_enabled: Whether braintrust is enabled.
        disable_review: Whether review is disabled.

    Returns:
        ConfigOverrideResult with resolved config and updated MalaConfig.

    Raises:
        typer.Exit: If any override parsing fails.
    """
    from src.infra.io.config import CLIOverrides, build_resolved_config

    cli_overrides = CLIOverrides(
        cerberus_spawn_args=cerberus_spawn_args,
        cerberus_wait_args=cerberus_wait_args,
        cerberus_env=cerberus_env,
        review_timeout=review_timeout,
        max_epic_verification_retries=max_epic_verification_retries,
        no_morph=no_morph,
        no_braintrust=not braintrust_enabled,
        disable_review=disable_review,
    )

    try:
        resolved = build_resolved_config(config, cli_overrides)
    except ValueError as exc:
        log("✗", str(exc), Colors.RED)
        raise typer.Exit(1)

    updated_config = replace(
        config,
        review_timeout=resolved.review_timeout,
        cerberus_spawn_args=resolved.cerberus_spawn_args,
        cerberus_wait_args=resolved.cerberus_wait_args,
        cerberus_env=resolved.cerberus_env,
        max_epic_verification_retries=resolved.max_epic_verification_retries,
    )

    return ConfigOverrideResult(resolved=resolved, updated_config=updated_config)


def _handle_dry_run(
    repo_path: Path,
    epic: str | None,
    only_ids: set[str] | None,
    wip: bool,
    focus: bool,
    orphans_only: bool,
) -> Never:
    """Execute dry-run mode: display task order and exit.

    Args:
        repo_path: Path to the repository.
        epic: Epic ID to filter by.
        only_ids: Set of specific issue IDs to process.
        wip: Whether to prioritize WIP issues.
        focus: Whether focus mode is enabled.
        orphans_only: Whether to only process orphan issues.

    Raises:
        typer.Exit: Always exits with code 0 after displaying.
    """

    async def _dry_run() -> None:
        beads = _lazy("BeadsClient")(repo_path)
        issues = await beads.get_ready_issues_async(
            epic_id=epic,
            only_ids=only_ids,
            prioritize_wip=wip,
            focus=focus,
            orphans_only=orphans_only,
        )
        display_dry_run_tasks(issues, focus=focus)

    asyncio.run(_dry_run())
    raise typer.Exit(0)


def _validate_run_args(
    only: str | None,
    disable_validations: str | None,
    coverage_threshold: float | None,
    epic: str | None,
    orphans_only: bool,
    epic_override: str | None,
    repo_path: Path,
) -> ValidatedRunArgs:
    """Validate and parse CLI arguments, raising typer.Exit(1) on errors.

    Args:
        only: Comma-separated issue IDs to process (--only flag)
        disable_validations: Comma-separated validation names to disable
        coverage_threshold: Coverage threshold percentage (0-100)
        epic: Epic ID to filter by
        orphans_only: Whether to only process orphan issues
        epic_override: Comma-separated epic IDs to override
        repo_path: Path to the repository (must exist)

    Returns:
        ValidatedRunArgs with parsed values

    Raises:
        typer.Exit: If any validation fails
    """
    # Parse --only flag into a set of issue IDs
    only_ids: set[str] | None = None
    if only:
        only_ids = {
            issue_id.strip() for issue_id in only.split(",") if issue_id.strip()
        }
        if not only_ids:
            log("✗", "Invalid --only value: no valid issue IDs found", Colors.RED)
            raise typer.Exit(1)

    # Parse --disable-validations flag into a set
    disable_set: set[str] | None = None
    if disable_validations:
        disable_set = {
            val.strip() for val in disable_validations.split(",") if val.strip()
        }
        if not disable_set:
            log(
                "✗",
                "Invalid --disable-validations value: no valid values found",
                Colors.RED,
            )
            raise typer.Exit(1)
        # Validate against known values
        unknown = disable_set - VALID_DISABLE_VALUES
        if unknown:
            log(
                "✗",
                f"Unknown --disable-validations value(s): {', '.join(sorted(unknown))}. "
                f"Valid values: {', '.join(sorted(VALID_DISABLE_VALUES))}",
                Colors.RED,
            )
            raise typer.Exit(1)

    # Validate coverage threshold range
    if coverage_threshold is not None and not 0 <= coverage_threshold <= 100:
        log(
            "✗",
            f"Invalid --coverage-threshold value: {coverage_threshold}. Must be between 0 and 100.",
            Colors.RED,
        )
        raise typer.Exit(1)

    # Validate --epic and --orphans-only are mutually exclusive
    if epic and orphans_only:
        log(
            "✗",
            "--epic and --orphans-only are mutually exclusive. "
            "--epic filters to children of an epic, while --orphans-only filters to issues without a parent epic.",
            Colors.RED,
        )
        raise typer.Exit(1)

    # Parse --epic-override flag into a set of epic IDs
    epic_override_ids: set[str] | None = None
    if epic_override:
        epic_override_ids = {
            eid.strip() for eid in epic_override.split(",") if eid.strip()
        }
        if not epic_override_ids:
            log(
                "✗",
                "Invalid --epic-override value: no valid epic IDs found",
                Colors.RED,
            )
            raise typer.Exit(1)

    # Validate repo_path exists
    if not repo_path.exists():
        log("✗", f"Repository not found: {repo_path}", Colors.RED)
        raise typer.Exit(1)

    return ValidatedRunArgs(
        only_ids=only_ids,
        disable_set=disable_set,
        epic_override_ids=epic_override_ids,
    )


app = typer.Typer(
    name="mala",
    help="Parallel issue processing with Claude Agent SDK",
    add_completion=False,
)


@app.command()
def run(
    repo_path: Annotated[
        Path,
        typer.Argument(
            help="Path to repository with beads issues",
        ),
    ] = Path("."),
    max_agents: Annotated[
        int | None,
        typer.Option(
            "--max-agents", "-n", help="Maximum concurrent agents (default: unlimited)"
        ),
    ] = None,
    timeout: Annotated[
        int | None,
        typer.Option(
            "--timeout", "-t", help="Timeout per agent in minutes (default: 60)"
        ),
    ] = None,
    max_issues: Annotated[
        int | None,
        typer.Option(
            "--max-issues", "-i", help="Maximum issues to process (default: unlimited)"
        ),
    ] = None,
    epic: Annotated[
        str | None,
        typer.Option(
            "--epic", "-e", help="Only process tasks that are children of this epic"
        ),
    ] = None,
    only: Annotated[
        str | None,
        typer.Option(
            "--only",
            "-o",
            help="Comma-separated list of issue IDs to process exclusively",
        ),
    ] = None,
    max_gate_retries: Annotated[
        int,
        typer.Option(
            "--max-gate-retries",
            help="Maximum quality gate retry attempts per issue (default: 3)",
        ),
    ] = 3,
    max_review_retries: Annotated[
        int,
        typer.Option(
            "--max-review-retries",
            help="Maximum codex review retry attempts per issue (default: 3)",
        ),
    ] = 3,
    disable_validations: Annotated[
        str | None,
        typer.Option(
            "--disable-validations",
            help=(
                "Comma-separated validations to skip. Options: "
                "post-validate (skip pytest/ruff/ty after commits), "
                "integration-tests (exclude @pytest.mark.integration tests), "
                "coverage (disable coverage threshold check), "
                "e2e (skip end-to-end fixture tests), "
                "review (skip LLM code review)"
            ),
        ),
    ] = None,
    coverage_threshold: Annotated[
        float | None,
        typer.Option(
            "--coverage-threshold",
            help="Minimum coverage percentage (0-100). If not set, uses 'no decrease' mode which requires coverage >= previous baseline.",
        ),
    ] = None,
    wip: Annotated[
        bool,
        typer.Option(
            "--wip",
            help="Prioritize in_progress issues before open issues",
        ),
    ] = False,
    focus: Annotated[
        bool,
        typer.Option(
            "--focus/--no-focus",
            help="Group tasks by epic for focused work (default: on); --no-focus uses priority-only ordering",
        ),
    ] = True,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Preview task order without processing; shows what would be run",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable verbose output; shows full tool arguments instead of single line per tool call",
        ),
    ] = False,
    review_timeout: Annotated[
        int | None,
        typer.Option(
            "--review-timeout",
            help="Timeout in seconds for review operations (default: 1200)",
        ),
    ] = None,
    cerberus_spawn_args: Annotated[
        str | None,
        typer.Option(
            "--cerberus-spawn-args",
            help="Extra args for `review-gate spawn-code-review` (shlex-style string)",
        ),
    ] = None,
    cerberus_wait_args: Annotated[
        str | None,
        typer.Option(
            "--cerberus-wait-args",
            help="Extra args for `review-gate wait` (shlex-style string)",
        ),
    ] = None,
    cerberus_env: Annotated[
        str | None,
        typer.Option(
            "--cerberus-env",
            help="Extra env for review-gate (JSON object or comma KEY=VALUE list)",
        ),
    ] = None,
    no_morph: Annotated[
        bool,
        typer.Option(
            "--no-morph",
            help="Disable MorphLLM routing; use models directly for debugging or when MorphLLM is unavailable",
        ),
    ] = False,
    epic_override: Annotated[
        str | None,
        typer.Option(
            "--epic-override",
            help="Comma-separated epic IDs to close without verification (explicit human bypass)",
        ),
    ] = None,
    orphans_only: Annotated[
        bool,
        typer.Option(
            "--orphans-only",
            help="Only process issues with no parent epic (standalone/orphan issues)",
        ),
    ] = False,
    max_epic_verification_retries: Annotated[
        int | None,
        typer.Option(
            "--max-epic-verification-retries",
            help="Maximum retries for epic verification loop (default: 3)",
        ),
    ] = None,
) -> Never:
    """Run parallel issue processing."""
    # Apply verbose setting
    set_verbose(verbose)

    repo_path = repo_path.resolve()

    # Validate and parse CLI arguments
    validated = _validate_run_args(
        only=only,
        disable_validations=disable_validations,
        coverage_threshold=coverage_threshold,
        epic=epic,
        orphans_only=orphans_only,
        epic_override=epic_override,
        repo_path=repo_path,
    )
    only_ids = validated.only_ids
    disable_set = validated.disable_set
    epic_override_ids = validated.epic_override_ids

    # Ensure user config directory exists
    USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # Handle dry-run mode: display task order and exit
    if dry_run:
        _handle_dry_run(
            repo_path=repo_path,
            epic=epic,
            only_ids=only_ids,
            wip=wip,
            focus=focus,
            orphans_only=orphans_only,
        )

    # Build and configure MalaConfig from environment
    config = _lazy("MalaConfig").from_env(validate=False)

    # Apply CLI overrides to config
    override_result = _apply_config_overrides(
        config=config,
        review_timeout=review_timeout,
        cerberus_spawn_args=cerberus_spawn_args,
        cerberus_wait_args=cerberus_wait_args,
        cerberus_env=cerberus_env,
        max_epic_verification_retries=max_epic_verification_retries,
        no_morph=no_morph,
        braintrust_enabled=_braintrust_enabled,
        disable_review="review" in (disable_set or set()),
    )

    # Build cli_args metadata for logging
    cli_args = _build_cli_args_metadata(
        disable_validations=disable_validations,
        coverage_threshold=coverage_threshold,
        wip=wip,
        max_issues=max_issues,
        max_gate_retries=max_gate_retries,
        max_review_retries=max_review_retries,
        no_morph=no_morph,
        epic_override=epic_override,
        resolved=override_result.resolved,
        braintrust_enabled=_braintrust_enabled,
    )

    # Build OrchestratorConfig and run
    orch_config = _lazy("OrchestratorConfig")(
        repo_path=repo_path,
        max_agents=max_agents,
        timeout_minutes=timeout,
        max_issues=max_issues,
        epic_id=epic,
        only_ids=only_ids,
        braintrust_enabled=override_result.resolved.braintrust_enabled,
        max_gate_retries=max_gate_retries,
        max_review_retries=max_review_retries,
        disable_validations=disable_set,
        coverage_threshold=coverage_threshold,
        morph_enabled=override_result.resolved.morph_enabled,
        prioritize_wip=wip,
        focus=focus,
        cli_args=cli_args,
        epic_override_ids=epic_override_ids,
        orphans_only=orphans_only,
    )

    orchestrator = _lazy("create_orchestrator")(
        orch_config, mala_config=override_result.updated_config
    )

    success_count, total = asyncio.run(orchestrator.run())
    # Exit 0 if: no issues to process (no-op) OR at least one succeeded
    # Exit 1 only if: issues were processed but all failed
    raise typer.Exit(0 if success_count > 0 or total == 0 else 1)


@app.command("epic-verify")
def epic_verify(
    epic_id: Annotated[
        str,
        typer.Argument(
            help="Epic ID to verify and optionally close",
        ),
    ],
    repo_path: Annotated[
        Path,
        typer.Argument(
            help="Path to repository with beads issues",
        ),
    ] = Path("."),
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Run verification even if epic is not eligible or already closed",
        ),
    ] = False,
    close: Annotated[
        bool,
        typer.Option(
            "--close/--no-close",
            help="Close the epic after passing verification (default: close)",
        ),
    ] = True,
    human_override: Annotated[
        bool,
        typer.Option(
            "--human-override",
            help="Bypass verification and close directly (requires --close)",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable verbose output",
        ),
    ] = False,
) -> None:
    """Verify a single epic and optionally close it without running tasks."""
    set_verbose(verbose)

    if human_override and not close:
        log("✗", "--human-override requires --close", Colors.RED)
        raise typer.Exit(1)

    repo_path = repo_path.resolve()
    if not repo_path.exists():
        log("✗", f"Repository not found: {repo_path}", Colors.RED)
        raise typer.Exit(1)

    USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    config = _lazy("MalaConfig").from_env(validate=False)
    orch_config = _lazy("OrchestratorConfig")(repo_path=repo_path)
    orchestrator = _lazy("create_orchestrator")(orch_config, mala_config=config)
    verifier = getattr(orchestrator, "epic_verifier", None)

    if verifier is None:
        log("✗", "Epic verifier unavailable for this configuration", Colors.RED)
        raise typer.Exit(1)

    result = asyncio.run(
        verifier.verify_epic_with_options(
            epic_id,
            human_override=human_override,
            require_eligible=not force,
            close_epic=close,
        )
    )

    if result.verified_count == 0:
        log("○", f"No verification run for epic {epic_id}", Colors.GRAY)
        if not force:
            log("◐", "Use --force to bypass eligibility checks", Colors.MUTED)
        raise typer.Exit(1)

    if result.remediation_issues_created:
        log(
            "◐",
            f"Remediation issues: {', '.join(result.remediation_issues_created)}",
            Colors.MUTED,
        )

    if result.failed_count > 0:
        log("✗", f"Epic {epic_id} failed verification", Colors.RED)
        raise typer.Exit(1)

    if result.human_review_count > 0:
        log("⚠", f"Epic {epic_id} requires human review", Colors.YELLOW)
        raise typer.Exit(1)

    if close:
        log("✓", f"Epic {epic_id} verified and closed", Colors.GREEN)
    else:
        log("✓", f"Epic {epic_id} verified (no close)", Colors.GREEN)
    raise typer.Exit(0)


@app.command()
def clean() -> None:
    """Clean up locks and JSONL logs."""
    cleaned_locks = 0
    cleaned_logs = 0

    lock_dir = _lazy("get_lock_dir")()
    if lock_dir.exists():
        for lock in lock_dir.glob("*.lock"):
            lock.unlink()
            cleaned_locks += 1

    if cleaned_locks:
        log("🧹", f"Removed {cleaned_locks} lock files", Colors.GREEN)

    if get_runs_dir().exists():
        # Use rglob to find run files in both legacy flat structure
        # and new repo-segmented subdirectories
        run_files = list(get_runs_dir().rglob("*.json"))
        run_count = len(run_files)
        if run_count > 0:
            if typer.confirm(f"Remove {run_count} run metadata files?"):
                for run_file in run_files:
                    run_file.unlink()
                    cleaned_logs += 1
                # Also remove empty repo subdirectories
                for subdir in get_runs_dir().iterdir():
                    if subdir.is_dir() and not any(subdir.iterdir()):
                        subdir.rmdir()
                log("🧹", f"Removed {cleaned_logs} run metadata files", Colors.GREEN)


@app.command()
def status(
    all_instances: Annotated[
        bool,
        typer.Option(
            "--all",
            help="Show all running instances across all directories",
        ),
    ] = False,
) -> None:
    """Show status of mala instance(s).

    By default, shows status only for mala running in the current directory.
    Use --all to show all running instances across all directories.
    """
    print()
    cwd = Path.cwd().resolve()

    # Get running instances (filtered by cwd unless --all)
    if all_instances:
        instances = _lazy("get_running_instances")()
    else:
        instances = _lazy("get_running_instances_for_dir")(cwd)

    if not instances:
        if all_instances:
            log("○", "No mala instances running", Colors.GRAY)
        else:
            log("○", "No mala instance running in this directory", Colors.GRAY)
            log("◐", f"cwd: {cwd}", Colors.MUTED)
            log("◐", "Use --all to show instances in other directories", Colors.MUTED)
        print()
        return

    # Display running instances
    if all_instances:
        log("●", f"{len(instances)} running instance(s)", Colors.MAGENTA)
        print()

        # Group by repo path for display with directory headings
        instances_by_repo: dict[Path, list[object]] = {}
        for instance in instances:
            repo = getattr(instance, "repo_path", Path("."))
            if repo not in instances_by_repo:
                instances_by_repo[repo] = []
            instances_by_repo[repo].append(instance)

        for repo_path, repo_instances in instances_by_repo.items():
            # Directory heading
            print(f"{Colors.CYAN}{repo_path}:{Colors.RESET}")
            for instance in repo_instances:
                _display_instance(instance, indent=True)
            print()
    else:
        # Single instance for this directory
        log("●", "mala running", Colors.MAGENTA)
        print()
        for instance in instances:
            _display_instance(instance)
        print()

    # Show config directory
    config_env = USER_CONFIG_DIR / ".env"
    if config_env.exists():
        log("◐", f"config: {config_env}", Colors.MUTED)
    else:
        log("○", f"config: {config_env} (not found)", Colors.MUTED)
    print()

    # Check locks (show all locks, not filtered by directory)
    lock_dir = _lazy("get_lock_dir")()
    if lock_dir.exists():
        locks = list(lock_dir.glob("*.lock"))
        if locks:
            log("⚠", f"{len(locks)} active locks", Colors.YELLOW)
            for lock in locks[:5]:
                try:
                    holder = lock.read_text().strip() if lock.is_file() else "unknown"
                except OSError:
                    holder = "unknown"
                print(f"    {Colors.MUTED}{lock.stem} → {holder}{Colors.RESET}")
            if len(locks) > 5:
                print(f"    {Colors.MUTED}... and {len(locks) - 5} more{Colors.RESET}")
        else:
            log("○", "No active locks", Colors.GRAY)
    else:
        log("○", "No active locks", Colors.GRAY)

    # Check run metadata (completed runs)
    if get_runs_dir().exists():
        # Use rglob to find run files in both legacy flat structure
        # and new repo-segmented subdirectories
        run_files = list(get_runs_dir().rglob("*.json"))
        if run_files:
            log(
                "◐",
                f"{len(run_files)} run metadata files in {get_runs_dir()}",
                Colors.MUTED,
            )
            recent = sorted(run_files, key=lambda p: p.stat().st_mtime, reverse=True)[
                :3
            ]
            for run_file in recent:
                mtime = datetime.fromtimestamp(run_file.stat().st_mtime).strftime(
                    "%H:%M:%S"
                )
                # Show relative path from runs_dir for clarity
                rel_path = run_file.relative_to(get_runs_dir())
                print(f"    {Colors.MUTED}{mtime} {rel_path}{Colors.RESET}")
    print()


def _display_instance(instance: object, indent: bool = False) -> None:
    """Display a single running instance.

    Args:
        instance: RunningInstance object (typed as object to avoid import issues)
        indent: If True, indent output (for grouped display under directory heading)
    """
    # Access attributes directly (duck typing)
    repo_path = getattr(instance, "repo_path", None)
    started_at = getattr(instance, "started_at", None)
    max_agents = getattr(instance, "max_agents", None)
    pid = getattr(instance, "pid", None)

    prefix = "  " if indent else ""

    # Only show repo path when not indented (when indented, directory is already shown as heading)
    if not indent:
        log("◐", f"repo: {repo_path}", Colors.CYAN)

    if started_at:
        # Calculate duration
        from datetime import datetime, UTC

        now = datetime.now(UTC)
        duration = now - started_at
        minutes = int(duration.total_seconds() // 60)
        if minutes >= 60:
            hours = minutes // 60
            mins = minutes % 60
            duration_str = f"{hours}h {mins}m"
        elif minutes > 0:
            duration_str = f"{minutes} minute(s)"
        else:
            seconds = int(duration.total_seconds())
            duration_str = f"{seconds} second(s)"
        if indent:
            print(f"{prefix}{Colors.MUTED}started: {duration_str} ago{Colors.RESET}")
        else:
            log("◐", f"started: {duration_str} ago", Colors.MUTED)

    if max_agents is not None:
        if indent:
            print(f"{prefix}{Colors.MUTED}max-agents: {max_agents}{Colors.RESET}")
        else:
            log("◐", f"max-agents: {max_agents}", Colors.MUTED)
    else:
        if indent:
            print(f"{prefix}{Colors.MUTED}max-agents: unlimited{Colors.RESET}")
        else:
            log("◐", "max-agents: unlimited", Colors.MUTED)

    if pid:
        if indent:
            print(f"{prefix}{Colors.MUTED}pid: {pid}{Colors.RESET}")
        else:
            log("◐", f"pid: {pid}", Colors.MUTED)


def __getattr__(name: str) -> Any:  # noqa: ANN401
    """Lazy-load SDK-dependent modules on first access."""
    if name not in _LAZY_NAMES:
        raise AttributeError(f"module {__name__} has no attribute {name}")

    if name in _lazy_modules:
        return _lazy_modules[name]

    if name == "BeadsClient":
        from ..orchestration.cli_support import BeadsClient

        _lazy_modules[name] = BeadsClient
    elif name == "MalaConfig":
        from src.infra.io.config import MalaConfig

        _lazy_modules[name] = MalaConfig
    elif name == "MalaOrchestrator":
        from ..orchestration.orchestrator import MalaOrchestrator

        _lazy_modules[name] = MalaOrchestrator
    elif name == "OrchestratorConfig":
        from ..orchestration.types import OrchestratorConfig

        _lazy_modules[name] = OrchestratorConfig
    elif name == "create_orchestrator":
        from ..orchestration.factory import create_orchestrator

        _lazy_modules[name] = create_orchestrator
    elif name == "get_lock_dir":
        from ..orchestration.cli_support import get_lock_dir

        _lazy_modules[name] = get_lock_dir
    elif name == "get_running_instances":
        from ..orchestration.cli_support import get_running_instances

        _lazy_modules[name] = get_running_instances
    elif name == "get_running_instances_for_dir":
        from ..orchestration.cli_support import get_running_instances_for_dir

        _lazy_modules[name] = get_running_instances_for_dir

    return _lazy_modules[name]
