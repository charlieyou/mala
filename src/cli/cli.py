#!/usr/bin/env python3
# ruff: noqa: E402
"""
mala CLI: Agent SDK orchestrator for parallel issue processing.

Usage:
    mala run [OPTIONS] [REPO_PATH]
    mala epic-verify [OPTIONS] EPIC_ID [REPO_PATH]
    mala logs [list|sessions|show] [OPTIONS]
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
    from src.core.models import OrderPreference
    from src.infra.io.config import MalaConfig, ResolvedConfig

# Bootstrap state: tracks whether bootstrap() has been called
_bootstrapped = False

# Lazy-loaded module cache for SDK-dependent imports
# These are populated on first access via __getattr__
_lazy_modules: dict[str, Any] = {}

# Names that are lazily loaded (for __getattr__ to handle)
_LAZY_NAMES = frozenset(
    {
        "MalaConfig",
        "MalaOrchestrator",
        "OrderPreference",
        "OrchestratorConfig",
        "PeriodicValidationConfig",
        "WatchConfig",
        "create_issue_provider",
        "create_orchestrator",
        "get_all_locks",
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
    """Initialize environment.

    Must be called before using any CLI commands or importing claude_agent_sdk.
    This function is idempotent - calling it multiple times has no additional effect.

    Side effects:
        - Loads environment variables from ~/.config/mala/.env
    """
    global _bootstrapped

    if _bootstrapped:
        return

    # Load environment variables early (needed for config)
    load_user_env()

    _bootstrapped = True


# Import modules that do NOT depend on claude_agent_sdk at module level
import asyncio
from datetime import datetime
from typing import Annotated, Never

import typer

from ..orchestration.cli_support import Colors, log, set_verbose
from .logs import logs_app

# SDK-dependent imports (MalaOrchestrator, get_lock_dir, run_metadata, etc.)
# are lazy-loaded via __getattr__ to ensure bootstrap() runs before claude_agent_sdk
# is imported. Access them as module attributes: MalaOrchestrator, create_issue_provider, etc.


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


# Valid values for --disable flag
# Each value controls a specific validation phase:
#   post-validate: Skip all validation commands (pytest, ruff, ty) after agent commits
#   global-validate: Skip global validation at end of batch (reserved for future use)
#   integration-tests: Exclude @pytest.mark.integration tests from pytest runs
#   coverage: Disable code coverage enforcement (threshold check)
#   e2e: Disable end-to-end fixture repo tests (global only)
#   review: Disable automated LLM code review after quality gate passes
#   followup-on-run-validate-fail: Disable auto-retry on run validation failures (reserved)
VALID_DISABLE_VALUES = frozenset(
    {
        "post-validate",
        "global-validate",
        "integration-tests",
        "coverage",
        "e2e",
        "review",
        "followup-on-run-validate-fail",
    }
)


@dataclass(frozen=True)
class ScopeConfig:
    """Parsed scope configuration from --scope option.

    Attributes:
        scope_type: The type of scope ('all', 'epic', 'orphans', 'ids').
        ids: List of issue IDs to process (from 'ids:' prefix). Order-preserving.
        epic_id: Epic ID to filter by (from 'epic:' prefix).
    """

    scope_type: str = "all"
    ids: list[str] | None = None
    epic_id: str | None = None


def parse_scope(scope: str) -> ScopeConfig:
    """Parse the --scope option value into a ScopeConfig.

    Supported formats:
        - 'all' (default): Process all issues
        - 'epic:<id>': Filter to issues under the specified epic
        - 'orphans': Process only orphan issues (no parent epic)
        - 'ids:<id1>,<id2>,...': Process specific IDs in order (deduplicates with warning)

    Args:
        scope: The scope string (e.g., 'ids:T-1,T-2').

    Returns:
        ScopeConfig with parsed values.

    Raises:
        typer.Exit: If scope format is invalid.
    """
    import typer

    scope = scope.strip()

    # Handle 'all' scope
    if scope == "all":
        return ScopeConfig(scope_type="all")

    # Handle 'orphans' scope
    if scope == "orphans":
        return ScopeConfig(scope_type="orphans")

    # Handle 'epic:<id>' scope
    if scope.startswith("epic:"):
        epic_id = scope[5:].strip()
        if not epic_id:
            log(
                "✗",
                "Invalid --scope: 'epic:' requires an epic ID. "
                "Valid formats: all, epic:<id>, orphans, ids:<id,...>",
                Colors.RED,
            )
            raise typer.Exit(1)
        return ScopeConfig(scope_type="epic", epic_id=epic_id)

    # Handle 'ids:<id,...>' scope
    if scope.startswith("ids:"):
        ids_str = scope[4:].strip()
        if not ids_str:
            log(
                "✗",
                "Invalid --scope: 'ids:' requires at least one ID. "
                "Valid formats: all, epic:<id>, orphans, ids:<id,...>",
                Colors.RED,
            )
            raise typer.Exit(1)

        # Parse comma-separated IDs, preserving order
        raw_ids = [id_.strip() for id_ in ids_str.split(",") if id_.strip()]
        if not raw_ids:
            log(
                "✗",
                "Invalid --scope: 'ids:' requires at least one ID. "
                "Valid formats: all, epic:<id>, orphans, ids:<id,...>",
                Colors.RED,
            )
            raise typer.Exit(1)

        # Deduplicate while preserving order, warn if duplicates found
        seen: set[str] = set()
        unique_ids: list[str] = []
        duplicates: list[str] = []
        for id_ in raw_ids:
            if id_ in seen:
                duplicates.append(id_)
            else:
                seen.add(id_)
                unique_ids.append(id_)

        if duplicates:
            log(
                "⚠",
                f"Duplicate IDs removed from --scope: {', '.join(duplicates)}",
                Colors.YELLOW,
            )

        return ScopeConfig(scope_type="ids", ids=unique_ids)

    # Unknown scope format
    log(
        "✗",
        f"Invalid --scope value: '{scope}'. "
        "Valid formats: all, epic:<id>, orphans, ids:<id,...>",
        Colors.RED,
    )
    raise typer.Exit(1)


@dataclass(frozen=True)
class ValidatedRunArgs:
    """Validated and parsed CLI arguments for the run command."""

    only_ids: list[str] | None
    disable_set: set[str] | None
    epic_override_ids: set[str]


@dataclass(frozen=True)
class ConfigOverrideResult:
    """Result of applying CLI overrides to config.

    On success: resolved and updated_config are set, error is None.
    On failure: error is set, resolved and updated_config are None.
    """

    resolved: ResolvedConfig | None = None
    updated_config: MalaConfig | None = None
    error: str | None = None

    @property
    def is_error(self) -> bool:
        """Return True if this result represents an error."""
        return self.error is not None


def _warn_stderr(msg: str) -> None:
    """Emit a warning message to stderr with yellow color and warning icon."""
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.RESET}", file=sys.stderr)


def _build_cli_args_metadata(
    *,
    disable: list[str] | None,
    resume: bool,
    max_issues: int | None,
    max_gate_retries: int,
    max_review_retries: int,
    epic_override: list[str] | None,
    resolved: ResolvedConfig,
    watch: bool,
    validate_every: int | None,
) -> dict[str, object]:
    """Build the cli_args metadata dictionary for logging and OrchestratorConfig.

    Args:
        disable: List of validation names to disable from CLI.
        resume: Whether WIP prioritization is enabled (--resume flag).
        max_issues: Maximum issues to process.
        max_gate_retries: Maximum gate retry attempts.
        max_review_retries: Maximum review retry attempts.
        epic_override: List of epic IDs to override from CLI.
        resolved: Resolved config with effective values.
        watch: Whether watch mode is enabled.
        validate_every: Run validation after every N issues.

    Returns:
        Dictionary of CLI arguments for logging/metadata.
    """
    normalized_disable = sorted(set(_normalize_repeatable_option(disable)))
    normalized_epic = sorted(set(_normalize_repeatable_option(epic_override)))
    return {
        "disable_validations": normalized_disable if normalized_disable else None,
        "wip": resume,
        "max_issues": max_issues,
        "max_gate_retries": max_gate_retries,
        "max_review_retries": max_review_retries,
        "review_timeout": resolved.review_timeout,
        "review_spawn_args": list(resolved.cerberus_spawn_args),
        "review_wait_args": list(resolved.cerberus_wait_args),
        "review_env": dict(resolved.cerberus_env),
        "epic_override": normalized_epic if normalized_epic else None,
        "max_epic_verification_retries": resolved.max_epic_verification_retries,
        "watch": watch,
        "validate_every": validate_every,
    }


def _apply_config_overrides(
    config: MalaConfig,
    review_timeout: int | None,
    cerberus_spawn_args: str | None,
    cerberus_wait_args: str | None,
    cerberus_env: str | None,
    max_epic_verification_retries: int | None,
    disable_review: bool,
    claude_settings_sources: str | None,
) -> ConfigOverrideResult:
    """Apply CLI overrides to MalaConfig, returning error on parse failures.

    Args:
        config: Base MalaConfig from environment.
        review_timeout: Optional review timeout override.
        cerberus_spawn_args: Raw string of extra args for spawn.
        cerberus_wait_args: Raw string of extra args for wait.
        cerberus_env: Raw string of extra env vars (key=value pairs).
        max_epic_verification_retries: Optional max retries override.
        disable_review: Whether review is disabled.
        claude_settings_sources: Raw comma-separated list of settings sources.

    Returns:
        ConfigOverrideResult with resolved config and updated MalaConfig on success,
        or with error message on failure.
    """
    from src.infra.io.config import CLIOverrides, build_resolved_config

    cli_overrides = CLIOverrides(
        cerberus_spawn_args=cerberus_spawn_args,
        cerberus_wait_args=cerberus_wait_args,
        cerberus_env=cerberus_env,
        review_timeout=review_timeout,
        max_epic_verification_retries=max_epic_verification_retries,
        disable_review=disable_review,
        claude_settings_sources=claude_settings_sources,
    )

    try:
        resolved = build_resolved_config(config, cli_overrides)
    except ValueError as exc:
        return ConfigOverrideResult(error=str(exc))

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
    scope_config: ScopeConfig | None,
    resume: bool,
    order_preference: OrderPreference,
) -> Never:
    """Execute dry-run mode: display task order and exit.

    Args:
        repo_path: Path to the repository.
        scope_config: Parsed scope configuration (epic, ids, orphans, or all).
        resume: Whether to include WIP issues (--resume flag).
        order_preference: Issue ordering preference (OrderPreference enum).

    Raises:
        typer.Exit: Always exits with code 0.
    """
    epic_id = scope_config.epic_id if scope_config else None
    only_ids = scope_config.ids if scope_config else None
    orphans_only = scope_config.scope_type == "orphans" if scope_config else False
    focus = order_preference in (
        _lazy("OrderPreference").FOCUS,
        _lazy("OrderPreference").EPIC_PRIORITY,
    )

    async def _dry_run() -> None:
        beads = _lazy("create_issue_provider")(repo_path)
        issues = await beads.get_ready_issues_async(
            epic_id=epic_id,
            only_ids=only_ids,
            include_wip=resume,
            focus=focus,
            orphans_only=orphans_only,
            order_preference=order_preference,
        )
        display_dry_run_tasks(issues, focus=focus)

    asyncio.run(_dry_run())
    raise typer.Exit(0)


def _normalize_repeatable_option(values: list[str] | None) -> list[str]:
    """Normalize a repeatable option that may contain comma-separated values.

    Supports both:
    - Repeated: --disable coverage --disable review
    - Comma-separated: --disable "coverage,review"
    - Mixed: --disable coverage --disable "review,e2e"

    Returns:
        Flattened list of stripped, non-empty values.
    """
    if not values:
        return []
    result: list[str] = []
    for val in values:
        # Split by comma for backward compat
        for part in val.split(","):
            stripped = part.strip()
            if stripped:
                result.append(stripped)
    return result


def _validate_run_args(
    disable: list[str] | None,
    epic_override: list[str] | None,
    repo_path: Path,
) -> ValidatedRunArgs:
    """Validate and parse CLI arguments, raising typer.Exit(1) on errors.

    Args:
        disable: List of validation names to disable (repeatable, comma-separated supported)
        epic_override: List of epic IDs to override (repeatable, comma-separated supported)
        repo_path: Path to the repository (must exist)

    Returns:
        ValidatedRunArgs with parsed values

    Raises:
        typer.Exit: If any validation fails
    """
    # Parse --disable flag into a set (supports both repeatable and comma-separated)
    disable_set: set[str] | None = None
    normalized_disable = _normalize_repeatable_option(disable)
    if normalized_disable:
        disable_set = set(normalized_disable)
        unknown = disable_set - VALID_DISABLE_VALUES
        if unknown:
            log(
                "✗",
                f"Unknown --disable value(s): {', '.join(sorted(unknown))}. "
                f"Valid values: {', '.join(sorted(VALID_DISABLE_VALUES))}",
                Colors.RED,
            )
            raise typer.Exit(1)

    # Parse --epic-override flag into a set of epic IDs (supports both repeatable and comma-separated)
    epic_override_ids: set[str] = set()
    normalized_epic_override = _normalize_repeatable_option(epic_override)
    if normalized_epic_override:
        epic_override_ids = set(normalized_epic_override)

    # Validate repo_path exists
    if not repo_path.exists():
        log("✗", f"Repository not found: {repo_path}", Colors.RED)
        raise typer.Exit(1)

    return ValidatedRunArgs(
        only_ids=None,
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
            "--max-agents",
            "-n",
            help="Maximum concurrent agents (default: unlimited)",
            rich_help_panel="Execution Limits",
        ),
    ] = None,
    timeout: Annotated[
        int | None,
        typer.Option(
            "--timeout",
            "-t",
            help="Timeout per agent in minutes (default: 60)",
            rich_help_panel="Execution Limits",
        ),
    ] = None,
    max_issues: Annotated[
        int | None,
        typer.Option(
            "--max-issues",
            "-i",
            help="Maximum issues to process (default: unlimited)",
            rich_help_panel="Execution Limits",
        ),
    ] = None,
    scope: Annotated[
        str | None,
        typer.Option(
            "--scope",
            "-s",
            help="Scope filter (e.g., 'ids:T-1,T-2'). Replaces --only.",
            rich_help_panel="Scope & Ordering",
        ),
    ] = None,
    max_gate_retries: Annotated[
        int,
        typer.Option(
            "--max-gate-retries",
            help="Maximum quality gate retry attempts per issue (default: 3)",
            rich_help_panel="Quality Gates",
        ),
    ] = 3,
    max_review_retries: Annotated[
        int,
        typer.Option(
            "--max-review-retries",
            help="Maximum codex review retry attempts per issue (default: 3)",
            rich_help_panel="Quality Gates",
        ),
    ] = 3,
    disable: Annotated[
        list[str] | None,
        typer.Option(
            "--disable",
            help=(
                "Validations to skip (repeatable). Options: "
                "post-validate (skip pytest/ruff/ty after commits), "
                "integration-tests (exclude @pytest.mark.integration tests), "
                "coverage (disable coverage threshold check), "
                "e2e (skip end-to-end fixture tests), "
                "review (skip LLM code review)"
            ),
            rich_help_panel="Quality Gates",
        ),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option(
            "--resume/--no-resume",
            "-r",
            help="Include in_progress issues AND attempt to resume their prior Claude sessions",
            rich_help_panel="Scope & Ordering",
        ),
    ] = False,
    strict: Annotated[
        bool,
        typer.Option(
            "--strict/--no-strict",
            help="Fail issues if no prior session found (requires --resume)",
            rich_help_panel="Scope & Ordering",
        ),
    ] = False,
    order: Annotated[
        str | None,
        typer.Option(
            "--order",
            help="Issue ordering: 'epic-priority' (default, epic-grouped), 'issue-priority' (global priority), 'focus' (single-epic), 'input' (preserve --scope ids: order)",
            rich_help_panel="Scope & Ordering",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            "-d",
            help="Preview task order without processing; shows what would be run",
            rich_help_panel="Debugging",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable verbose output; shows full tool arguments instead of single line per tool call",
            rich_help_panel="Debugging",
        ),
    ] = False,
    review_timeout: Annotated[
        int | None,
        typer.Option(
            "--review-timeout",
            help="Timeout in seconds for review operations (default: 1200)",
            rich_help_panel="Review Backend",
        ),
    ] = None,
    review_spawn_args: Annotated[
        str | None,
        typer.Option(
            "--review-spawn-args",
            help="Extra args for `review-gate spawn-code-review` (shlex-style string)",
            rich_help_panel="Review Backend",
        ),
    ] = None,
    review_wait_args: Annotated[
        str | None,
        typer.Option(
            "--review-wait-args",
            help="Extra args for `review-gate wait` (shlex-style string)",
            rich_help_panel="Review Backend",
        ),
    ] = None,
    review_env: Annotated[
        str | None,
        typer.Option(
            "--review-env",
            help="Extra env for review-gate (JSON object or comma KEY=VALUE list)",
            rich_help_panel="Review Backend",
        ),
    ] = None,
    epic_override: Annotated[
        list[str] | None,
        typer.Option(
            "--epic-override",
            help="Epic IDs to close without verification (repeatable)",
            rich_help_panel="Epic Verification",
        ),
    ] = None,
    max_epic_verification_retries: Annotated[
        int | None,
        typer.Option(
            "--max-epic-verification-retries",
            help="Maximum retries for epic verification loop (default: 3)",
            rich_help_panel="Epic Verification",
        ),
    ] = None,
    watch: Annotated[
        bool,
        typer.Option(
            "--watch",
            help="Keep running and poll for new issues instead of exiting when idle",
        ),
    ] = False,
    validate_every: Annotated[
        int | None,
        typer.Option(
            "--validate-every",
            min=1,
            help="Run validation after every N issues complete (default: 10 in watch mode)",
        ),
    ] = None,
    claude_settings_sources: Annotated[
        str | None,
        typer.Option(
            "--claude-settings-sources",
            help=(
                "Comma-separated list of Claude settings sources "
                "(local, project, user). Default: local,project"
            ),
            rich_help_panel="Claude Settings",
        ),
    ] = None,
) -> Never:
    """Run parallel issue processing."""
    # Apply verbose setting
    set_verbose(verbose)

    # Validate max_issues (Typer min= doesn't work well with Optional[int])
    if max_issues is not None and max_issues < 1:
        log("✗", "Error: --max-issues must be at least 1", Colors.RED)
        raise typer.Exit(2)

    # Validate --strict requires --resume
    if strict and not resume:
        raise typer.BadParameter("--strict requires --resume flag")

    repo_path = repo_path.resolve()

    # Parse --scope option
    scope_config: ScopeConfig | None = None
    if scope is not None:
        scope_config = parse_scope(scope)

    # Parse and validate --order option
    order_preference = _lazy("OrderPreference").EPIC_PRIORITY  # default
    if order is not None:
        order_lower = order.lower()
        if order_lower == "epic-priority":
            order_preference = _lazy("OrderPreference").EPIC_PRIORITY
        elif order_lower == "issue-priority":
            order_preference = _lazy("OrderPreference").ISSUE_PRIORITY
        elif order_lower == "focus":
            order_preference = _lazy("OrderPreference").FOCUS
        elif order_lower == "input":
            order_preference = _lazy("OrderPreference").INPUT
            # --order input requires --scope ids:
            if scope_config is None or scope_config.scope_type != "ids":
                log(
                    "✗",
                    "--order input requires --scope ids:<id,...>",
                    Colors.RED,
                )
                raise typer.Exit(1)
        else:
            log(
                "✗",
                f"Invalid --order value: '{order}'. Valid values: focus, epic-priority, issue-priority, input",
                Colors.RED,
            )
            raise typer.Exit(1)

    # Derive scope values from scope_config
    epic_id = scope_config.epic_id if scope_config else None
    only_ids = scope_config.ids if scope_config else None
    orphans_only = scope_config.scope_type == "orphans" if scope_config else False
    focus = order_preference in (
        _lazy("OrderPreference").FOCUS,
        _lazy("OrderPreference").EPIC_PRIORITY,
    )

    # Validate and parse CLI arguments
    validated = _validate_run_args(
        disable=disable,
        epic_override=epic_override,
        repo_path=repo_path,
    )
    disable_set = validated.disable_set
    epic_override_ids = validated.epic_override_ids

    # Ensure user config directory exists
    USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # Handle dry-run mode: display task order and exit
    if dry_run:
        _handle_dry_run(
            repo_path=repo_path,
            scope_config=scope_config,
            resume=resume,
            order_preference=order_preference,
        )

    # Build and configure MalaConfig from environment
    config = _lazy("MalaConfig").from_env(validate=False)

    # Apply CLI overrides to config
    override_result = _apply_config_overrides(
        config=config,
        review_timeout=review_timeout,
        cerberus_spawn_args=review_spawn_args,
        cerberus_wait_args=review_wait_args,
        cerberus_env=review_env,
        max_epic_verification_retries=max_epic_verification_retries,
        disable_review="review" in (disable_set or set()),
        claude_settings_sources=claude_settings_sources,
    )
    if override_result.is_error:
        assert override_result.error is not None  # for type narrowing
        log("✗", override_result.error, Colors.RED)
        raise typer.Exit(1)

    # After error check, resolved and updated_config are guaranteed non-None
    assert override_result.resolved is not None
    assert override_result.updated_config is not None

    # Derive effective validate_every: explicit > watch default (10) > disabled (None)
    effective_validate_every: int | None
    if validate_every is not None:
        # User explicitly specified --validate-every N
        effective_validate_every = validate_every
    elif watch:
        # Watch mode enabled, use default 10
        effective_validate_every = 10
    else:
        # No watch, no explicit --validate-every → disabled
        effective_validate_every = None

    # Build cli_args metadata for logging
    cli_args = _build_cli_args_metadata(
        disable=disable,
        resume=resume,
        max_issues=max_issues,
        max_gate_retries=max_gate_retries,
        max_review_retries=max_review_retries,
        epic_override=epic_override,
        resolved=override_result.resolved,
        watch=watch,
        validate_every=effective_validate_every,
    )

    # Build OrchestratorConfig and run
    orch_config = _lazy("OrchestratorConfig")(
        repo_path=repo_path,
        max_agents=max_agents,
        timeout_minutes=timeout,
        max_issues=max_issues,
        epic_id=epic_id,
        only_ids=only_ids,
        max_gate_retries=max_gate_retries,
        max_review_retries=max_review_retries,
        disable_validations=disable_set,
        include_wip=resume,
        strict_resume=strict,
        focus=focus,
        order_preference=order_preference,
        cli_args=cli_args,
        epic_override_ids=epic_override_ids,
        orphans_only=orphans_only,
    )

    orchestrator = _lazy("create_orchestrator")(
        orch_config,
        mala_config=override_result.updated_config,
    )

    # Build WatchConfig and PeriodicValidationConfig, then run orchestrator
    watch_config = _lazy("WatchConfig")(enabled=watch)
    validation_config = (
        _lazy("PeriodicValidationConfig")(validate_every=effective_validate_every)
        if effective_validate_every is not None
        else None
    )
    success_count, total = asyncio.run(
        orchestrator.run(watch_config=watch_config, validation_config=validation_config)
    )

    # Determine exit code:
    # - Use orchestrator.exit_code for interrupt (130), validation failure (1), abort (3)
    # - Otherwise: exit 0 if no issues to process or at least one succeeded
    # - Exit 1 only if issues were processed but all failed
    orch_exit = orchestrator.exit_code
    if orch_exit != 0:
        raise typer.Exit(orch_exit)
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
    claude_settings_sources: Annotated[
        str | None,
        typer.Option(
            "--claude-settings-sources",
            help=(
                "Comma-separated list of Claude settings sources "
                "(local, project, user). Default: local,project"
            ),
            rich_help_panel="Claude Settings",
        ),
    ] = None,
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

    # Apply CLI overrides to config
    override_result = _apply_config_overrides(
        config=config,
        review_timeout=None,
        cerberus_spawn_args=None,
        cerberus_wait_args=None,
        cerberus_env=None,
        max_epic_verification_retries=None,
        disable_review=False,
        claude_settings_sources=claude_settings_sources,
    )
    if override_result.is_error:
        assert override_result.error is not None
        log("✗", override_result.error, Colors.RED)
        raise typer.Exit(1)
    assert override_result.updated_config is not None
    config = override_result.updated_config

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
        if result.ineligibility_reason:
            log("◐", f"Reason: {result.ineligibility_reason}", Colors.MUTED)
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

    if close:
        log("✓", f"Epic {epic_id} verified and closed", Colors.GREEN)
    else:
        log("✓", f"Epic {epic_id} verified (no close)", Colors.GREEN)
    raise typer.Exit(0)


@app.command()
def clean(
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Force cleanup even if a mala instance is running",
        ),
    ] = False,
) -> None:
    """Clean up lock files.

    Removes orphaned lock files from the lock directory. Use this when
    a previous run left stale locks behind (e.g., after a crash).

    If a mala instance is currently running, the command will exit early
    unless --force is specified.
    """
    # Check for running instances
    running = _lazy("get_running_instances")()
    if running and not force:
        log(
            "⚠",
            f"A mala instance is running (PID {running[0].pid}). "
            "Use --force to clean anyway.",
            Colors.YELLOW,
        )
        raise typer.Exit(1)

    cleaned_locks = 0

    lock_dir = _lazy("get_lock_dir")()
    if lock_dir.exists():
        for lock in lock_dir.glob("*.lock"):
            lock.unlink()
            cleaned_locks += 1

    if cleaned_locks:
        log("🧹", f"Removed {cleaned_locks} lock files", Colors.GREEN)
    else:
        log("○", "No lock files to clean", Colors.GRAY)


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

    # Check locks (show all locks grouped by agent)
    locks_by_agent = _lazy("get_all_locks")()
    if locks_by_agent:
        total_locks = sum(len(files) for files in locks_by_agent.values())
        log(
            "⚠",
            f"{total_locks} active lock(s) held by {len(locks_by_agent)} agent(s)",
            Colors.YELLOW,
        )
        for agent_id, files in sorted(locks_by_agent.items()):
            print(f"    {Colors.CYAN}{agent_id}{Colors.RESET}")
            for filepath in sorted(files):
                print(f"      {Colors.MUTED}{filepath}{Colors.RESET}")
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


# Register subcommand apps (after main commands for better --help ordering)
app.add_typer(logs_app, name="logs")


def __getattr__(name: str) -> Any:  # noqa: ANN401
    """Lazy-load SDK-dependent modules on first access."""
    if name not in _LAZY_NAMES:
        raise AttributeError(f"module {__name__} has no attribute {name}")

    if name in _lazy_modules:
        return _lazy_modules[name]

    if name == "MalaConfig":
        from src.infra.io.config import MalaConfig

        _lazy_modules[name] = MalaConfig
    elif name == "MalaOrchestrator":
        from ..orchestration.orchestrator import MalaOrchestrator

        _lazy_modules[name] = MalaOrchestrator
    elif name == "OrderPreference":
        from ..core.models import OrderPreference

        _lazy_modules[name] = OrderPreference
    elif name == "OrchestratorConfig":
        from ..orchestration.types import OrchestratorConfig

        _lazy_modules[name] = OrchestratorConfig
    elif name == "PeriodicValidationConfig":
        from ..core.models import PeriodicValidationConfig

        _lazy_modules[name] = PeriodicValidationConfig
    elif name == "WatchConfig":
        from ..core.models import WatchConfig

        _lazy_modules[name] = WatchConfig
    elif name == "create_issue_provider":
        from ..orchestration.factory import create_issue_provider

        _lazy_modules[name] = create_issue_provider
    elif name == "create_orchestrator":
        from ..orchestration.factory import create_orchestrator

        _lazy_modules[name] = create_orchestrator
    elif name == "get_all_locks":
        from ..infra.tools.locking import get_all_locks

        _lazy_modules[name] = get_all_locks
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
