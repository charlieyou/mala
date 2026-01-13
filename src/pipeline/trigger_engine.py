"""TriggerEngine for stateless trigger policy evaluation.

Provides event-based interface for trigger decisions, returning explicit
TriggerActions instead of queue-based approach.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.domain.validation.config import (
    ConfigError,
    EpicDepth,
    FireOn,
    TriggerType,
)

if TYPE_CHECKING:
    from src.domain.validation.config import (
        BaseTriggerConfig,
        FailureMode,
        ValidationConfig,
        ValidationTriggersConfig,
    )
    from src.pipeline.issue_result import IssueResult


@dataclass(frozen=True)
class ResolvedCommand:
    """A trigger command resolved from the base pool with overrides applied.

    Attributes:
        ref: The command reference name (e.g., "test", "lint").
        effective_command: The resolved command string to execute.
        effective_timeout: The resolved timeout in seconds (None for system default).
    """

    ref: str
    effective_command: str
    effective_timeout: int | None


@dataclass(frozen=True)
class TriggerActions:
    """Result from trigger policy evaluation.

    Attributes:
        should_run: Whether the trigger should fire.
        commands: Resolved commands to execute if should_run is True.
        failure_mode: How to handle validation failures.
        trigger_type: The type of trigger that fired.
        trigger_config: The trigger configuration (for code_review, max_retries, etc.).
    """

    should_run: bool
    commands: list[ResolvedCommand]
    failure_mode: FailureMode
    trigger_type: TriggerType | None = None
    trigger_config: BaseTriggerConfig | None = None


class TriggerEngine:
    """Stateless service for trigger policy evaluation.

    Evaluates trigger conditions and returns explicit TriggerActions
    containing commands to run. Does not maintain queue state.

    Constructed per-run with validation configuration.
    """

    def __init__(self, validation_config: ValidationConfig | None) -> None:
        """Initialize TriggerEngine.

        Args:
            validation_config: The validation configuration (already merged with preset).
                If None, all trigger methods return should_run=False.
        """
        self._validation_config = validation_config
        self._triggers_config: ValidationTriggersConfig | None = None
        self._base_pool: dict[str, tuple[str, int | None]] = {}

        if validation_config is not None:
            self._triggers_config = validation_config.validation_triggers
            self._base_pool = self._build_base_pool(validation_config)

    def on_issue_completed(
        self, issue_id: str, result: IssueResult
    ) -> TriggerActions | None:
        """Evaluate triggers for issue completion (session_end).

        Called when a single issue implementation completes.

        Args:
            issue_id: The issue ID that completed.
            result: The result of the issue implementation.

        Returns:
            TriggerActions if session_end trigger is configured and should fire,
            None otherwise.
        """
        if self._triggers_config is None:
            return None

        trigger_config = self._triggers_config.session_end
        if trigger_config is None:
            return None

        # session_end fires on every issue completion (no fire_on filter)
        resolved_commands = self._resolve_commands(
            TriggerType.SESSION_END, trigger_config
        )

        return TriggerActions(
            should_run=True,
            commands=resolved_commands,
            failure_mode=trigger_config.failure_mode,
            trigger_type=TriggerType.SESSION_END,
            trigger_config=trigger_config,
        )

    def on_epic_closed(
        self, epic_id: str, *, is_top_level: bool, success: bool
    ) -> TriggerActions | None:
        """Evaluate triggers for epic completion.

        Called when an epic closes.

        Args:
            epic_id: The epic ID that closed.
            is_top_level: True if this is a top-level (root) epic.
            success: True if the epic completed successfully.

        Returns:
            TriggerActions if epic_completion trigger is configured and should fire,
            None otherwise.
        """
        if self._triggers_config is None:
            return None

        trigger_config = self._triggers_config.epic_completion
        if trigger_config is None:
            return None

        # Check epic_depth filter
        if trigger_config.epic_depth == EpicDepth.TOP_LEVEL and not is_top_level:
            return None

        # Check fire_on filter
        if not self._should_fire(trigger_config.fire_on, success):
            return None

        resolved_commands = self._resolve_commands(
            TriggerType.EPIC_COMPLETION, trigger_config
        )

        return TriggerActions(
            should_run=True,
            commands=resolved_commands,
            failure_mode=trigger_config.failure_mode,
            trigger_type=TriggerType.EPIC_COMPLETION,
            trigger_config=trigger_config,
        )

    def on_run_end(self, success_count: int, total_count: int) -> TriggerActions | None:
        """Evaluate triggers for run completion.

        Called when mala run completes.

        Args:
            success_count: Number of successfully completed issues.
            total_count: Total number of issues attempted.

        Returns:
            TriggerActions if run_end trigger is configured and should fire,
            None otherwise.
        """
        if self._triggers_config is None:
            return None

        trigger_config = self._triggers_config.run_end
        if trigger_config is None:
            return None

        # Determine success based on counts
        success = success_count == total_count and total_count > 0

        # Check fire_on filter
        if not self._should_fire(trigger_config.fire_on, success):
            return None

        resolved_commands = self._resolve_commands(TriggerType.RUN_END, trigger_config)

        return TriggerActions(
            should_run=True,
            commands=resolved_commands,
            failure_mode=trigger_config.failure_mode,
            trigger_type=TriggerType.RUN_END,
            trigger_config=trigger_config,
        )

    def resolve_commands(
        self, trigger_config: BaseTriggerConfig, trigger_type: TriggerType
    ) -> list[ResolvedCommand]:
        """Resolve commands for a trigger configuration.

        Public interface for command resolution. Used by callers that need
        to resolve commands without evaluating trigger conditions.

        Args:
            trigger_config: The trigger configuration with command refs.
            trigger_type: The trigger type (for error messages).

        Returns:
            List of resolved commands.

        Raises:
            ConfigError: If a command ref is not found in the base pool.
        """
        return self._resolve_commands(trigger_type, trigger_config)

    def _should_fire(self, fire_on: FireOn, success: bool) -> bool:
        """Check if trigger should fire based on fire_on setting.

        Args:
            fire_on: The fire_on setting from trigger config.
            success: Whether the operation succeeded.

        Returns:
            True if trigger should fire, False otherwise.
        """
        if fire_on == FireOn.BOTH:
            return True
        if fire_on == FireOn.SUCCESS:
            return success
        if fire_on == FireOn.FAILURE:
            return not success
        return False

    def _build_base_pool(
        self, validation_config: ValidationConfig
    ) -> dict[str, tuple[str, int | None]]:
        """Build the base command pool from validation config.

        The base pool maps command names to (command_string, timeout) tuples.

        Args:
            validation_config: The validation configuration.

        Returns:
            Dict mapping command ref names to (command, timeout) tuples.
        """
        pool: dict[str, tuple[str, int | None]] = {}
        base_cmds = validation_config.commands

        # Add built-in commands
        for cmd_name in (
            "test",
            "lint",
            "format",
            "typecheck",
            "e2e",
            "setup",
            "build",
        ):
            cmd = getattr(base_cmds, cmd_name, None)
            if cmd is not None:
                pool[cmd_name] = (cmd.command, cmd.timeout)

        # Add custom commands
        for name, custom_cmd in base_cmds.custom_commands.items():
            pool[name] = (custom_cmd.command, custom_cmd.timeout)

        return pool

    def _resolve_commands(
        self, trigger_type: TriggerType, trigger_config: BaseTriggerConfig
    ) -> list[ResolvedCommand]:
        """Resolve trigger command refs to executable commands.

        For each TriggerCommandRef in the trigger config, looks up the command
        in the base pool and applies any overrides (command string, timeout).

        Args:
            trigger_type: The type of trigger (for error messages).
            trigger_config: The trigger configuration with command refs.

        Returns:
            List of ResolvedCommand with effective command and timeout.

        Raises:
            ConfigError: If a ref is not found in the base pool.
        """
        resolved: list[ResolvedCommand] = []

        for cmd_ref in trigger_config.commands:
            if cmd_ref.ref not in self._base_pool:
                available = ", ".join(sorted(self._base_pool.keys()))
                raise ConfigError(
                    f"trigger {trigger_type.value} references unknown command "
                    f"'{cmd_ref.ref}'. Available: {available}"
                )

            base_cmd, base_timeout = self._base_pool[cmd_ref.ref]

            # Apply overrides - use `is not None` to allow falsy values like timeout=0
            effective_command = (
                cmd_ref.command if cmd_ref.command is not None else base_cmd
            )
            effective_timeout = (
                cmd_ref.timeout if cmd_ref.timeout is not None else base_timeout
            )

            resolved.append(
                ResolvedCommand(
                    ref=cmd_ref.ref,
                    effective_command=effective_command,
                    effective_timeout=effective_timeout,
                )
            )

        return resolved
