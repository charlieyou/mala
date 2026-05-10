"""Tests for shared trigger command resolution."""

from __future__ import annotations

import pytest

from src.domain.validation.config import (
    CommandConfig,
    CommandsConfig,
    ConfigError,
    CustomCommandConfig,
    FailureMode,
    SessionEndTriggerConfig,
    TriggerCommandRef,
    TriggerType,
    ValidationConfig,
)
from src.pipeline.trigger_plan import TriggerPlan, resolve_trigger


def _config() -> ValidationConfig:
    return ValidationConfig(
        commands=CommandsConfig(
            test=CommandConfig(command="uv run pytest", timeout=60),
            lint=CommandConfig(command="uvx ruff check .", timeout=30),
            custom_commands={
                "security": CustomCommandConfig(command="npm audit", timeout=45)
            },
        )
    )


def test_resolve_trigger_resolves_builtin_commands() -> None:
    trigger_config = SessionEndTriggerConfig(
        failure_mode=FailureMode.CONTINUE,
        commands=(
            TriggerCommandRef(ref="test"),
            TriggerCommandRef(ref="lint"),
        ),
    )

    outcome = resolve_trigger(_config(), trigger_config, TriggerType.SESSION_END)

    assert outcome.trigger_type == TriggerType.SESSION_END
    assert outcome.commands == (
        TriggerPlan("test", "uv run pytest", 60),
        TriggerPlan("lint", "uvx ruff check .", 30),
    )


def test_resolve_trigger_includes_custom_commands() -> None:
    trigger_config = SessionEndTriggerConfig(
        failure_mode=FailureMode.CONTINUE,
        commands=(TriggerCommandRef(ref="security"),),
    )

    outcome = resolve_trigger(_config(), trigger_config, TriggerType.SESSION_END)

    assert outcome.commands == (TriggerPlan("security", "npm audit", 45),)


def test_resolve_trigger_applies_command_and_timeout_overrides() -> None:
    trigger_config = SessionEndTriggerConfig(
        failure_mode=FailureMode.CONTINUE,
        commands=(TriggerCommandRef(ref="test", command="pytest -q", timeout=10),),
    )

    outcome = resolve_trigger(_config(), trigger_config, TriggerType.SESSION_END)

    assert outcome.commands == (TriggerPlan("test", "pytest -q", 10),)


def test_resolve_trigger_preserves_zero_timeout_override() -> None:
    trigger_config = SessionEndTriggerConfig(
        failure_mode=FailureMode.CONTINUE,
        commands=(TriggerCommandRef(ref="test", timeout=0),),
    )

    outcome = resolve_trigger(_config(), trigger_config, TriggerType.SESSION_END)

    assert outcome.commands == (TriggerPlan("test", "uv run pytest", 0),)


def test_resolve_trigger_raises_for_unknown_ref() -> None:
    trigger_config = SessionEndTriggerConfig(
        failure_mode=FailureMode.CONTINUE,
        commands=(TriggerCommandRef(ref="missing"),),
    )

    with pytest.raises(ConfigError, match="references unknown command 'missing'"):
        resolve_trigger(_config(), trigger_config, TriggerType.SESSION_END)


def test_resolve_trigger_can_resolve_one_command_ref() -> None:
    trigger_config = SessionEndTriggerConfig(
        failure_mode=FailureMode.CONTINUE,
        commands=(
            TriggerCommandRef(ref="test"),
            TriggerCommandRef(ref="lint", command="ruff check src", timeout=5),
        ),
    )

    outcome = resolve_trigger(
        _config(),
        trigger_config,
        TriggerType.SESSION_END,
        command_ref=trigger_config.commands[1],
    )

    assert outcome.commands == (TriggerPlan("lint", "ruff check src", 5),)
