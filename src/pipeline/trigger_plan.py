"""Shared trigger command resolution.

This module keeps trigger command reference resolution independent from
execution so run-level and session-end trigger paths use the same rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.domain.validation.config import ConfigError

if TYPE_CHECKING:
    from src.domain.validation.config import (
        BaseTriggerConfig,
        TriggerCommandRef,
        TriggerType,
        ValidationConfig,
    )


_BUILTIN_COMMANDS = ("test", "lint", "format", "typecheck", "e2e", "setup", "build")


@dataclass(frozen=True)
class TriggerPlan:
    """A trigger command resolved from the base pool with overrides applied."""

    ref: str
    effective_command: str
    effective_timeout: int | None


@dataclass(frozen=True)
class TriggerOutcome:
    """Resolved command plan for a trigger."""

    trigger_type: TriggerType
    commands: tuple[TriggerPlan, ...]


def resolve_trigger(
    validation_config: ValidationConfig,
    trigger_config: BaseTriggerConfig,
    trigger_type: TriggerType,
    *,
    command_ref: TriggerCommandRef | None = None,
) -> TriggerOutcome:
    """Resolve trigger command refs to executable command plans.

    Args:
        validation_config: Validation config containing the base command pool.
        trigger_config: Trigger config containing command refs.
        trigger_type: Trigger type for diagnostic messages.
        command_ref: Optional single command ref to resolve using the same rules.

    Returns:
        TriggerOutcome with all requested commands resolved.

    Raises:
        ConfigError: If a command ref is not found in the base pool.
    """
    base_pool = _build_base_pool(validation_config)
    command_refs = (
        (command_ref,) if command_ref is not None else trigger_config.commands
    )

    return TriggerOutcome(
        trigger_type=trigger_type,
        commands=tuple(
            _resolve_command_ref(base_pool, trigger_type, ref) for ref in command_refs
        ),
    )


def _build_base_pool(
    validation_config: ValidationConfig,
) -> dict[str, tuple[str, int | None]]:
    pool: dict[str, tuple[str, int | None]] = {}
    base_cmds = validation_config.commands

    for name in _BUILTIN_COMMANDS:
        command_config = getattr(base_cmds, name, None)
        if command_config is not None:
            pool[name] = (command_config.command, command_config.timeout)

    for name, custom_command in base_cmds.custom_commands.items():
        pool[name] = (custom_command.command, custom_command.timeout)

    return pool


def _resolve_command_ref(
    base_pool: dict[str, tuple[str, int | None]],
    trigger_type: TriggerType,
    command_ref: TriggerCommandRef,
) -> TriggerPlan:
    if command_ref.ref not in base_pool:
        available = ", ".join(sorted(base_pool.keys()))
        raise ConfigError(
            f"trigger {trigger_type.value} references unknown command "
            f"'{command_ref.ref}'. Available: {available}"
        )

    base_command, base_timeout = base_pool[command_ref.ref]
    effective_command = (
        command_ref.command if command_ref.command is not None else base_command
    )
    effective_timeout = (
        command_ref.timeout if command_ref.timeout is not None else base_timeout
    )

    return TriggerPlan(
        ref=command_ref.ref,
        effective_command=effective_command,
        effective_timeout=effective_timeout,
    )
