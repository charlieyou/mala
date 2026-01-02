"""Config merger for combining preset and user configurations.

This module provides the merge_configs function to merge a preset configuration
with user overrides. User values take precedence over preset values.

Merge rules:
- If no preset, return user config as-is
- Command fields: user replaces preset if present; explicit null disables; omitted inherits
- Coverage: user replaces preset entirely if present; explicit null disables; omitted inherits
- List fields (code_patterns, config_files, setup_files): user replaces preset entirely

To distinguish "field not set" from "explicitly disabled", this module defines
a DISABLED sentinel. When a command or coverage is set to DISABLED, it means the user
explicitly wants to disable that feature even if the preset defines it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from src.domain.validation.config import (
    CommandsConfig,
    ValidationConfig,
)

if TYPE_CHECKING:
    from src.domain.validation.config import (
        CommandConfig,
        YamlCoverageConfig,
    )


@dataclass(frozen=True)
class _DisabledSentinel:
    """Sentinel value indicating a command or coverage is explicitly disabled.

    This is used to distinguish between:
    - None: field not specified (inherit from preset)
    - DISABLED: field explicitly set to null (disable even if preset defines it)
    """

    pass


DISABLED: Final[_DisabledSentinel] = _DisabledSentinel()
"""Sentinel value for explicitly disabled commands or coverage.

Use this when parsing user config where a command or coverage is explicitly set to null,
as opposed to being omitted entirely.

Example:
    # User YAML with explicit null:
    # commands:
    #   lint: null  # <-- explicitly disabled
    #   test: "pytest"  # <-- override
    #   # format not mentioned, will inherit from preset
    # coverage: null  # <-- explicitly disabled

    # In the parser:
    user_commands = CommandsConfig(
        lint=DISABLED,  # type: ignore[arg-type]  # explicit null
        test=CommandConfig(command="pytest"),  # override
        # format is omitted, defaults to None (inherit)
    )
    user_config = ValidationConfig(
        commands=user_commands,
        coverage=DISABLED,  # type: ignore[arg-type]  # explicit null disables coverage
    )
"""


def merge_configs(
    preset: ValidationConfig | None,
    user: ValidationConfig,
) -> ValidationConfig:
    """Merge preset configuration with user overrides.

    User values take precedence over preset values. When a user field is None
    (not specified), the preset value is inherited. When a user field is
    explicitly set to DISABLED sentinel, the feature is disabled even if
    the preset defines it.

    Args:
        preset: Base preset configuration to merge with, or None.
        user: User configuration with overrides.

    Returns:
        Merged ValidationConfig with user values taking precedence.

    Examples:
        >>> from src.domain.validation.config import CommandConfig, CommandsConfig
        >>> preset_cfg = ValidationConfig(
        ...     commands=CommandsConfig(
        ...         test=CommandConfig(command="pytest"),
        ...         lint=CommandConfig(command="ruff check"),
        ...     ),
        ...     code_patterns=("**/*.py",),
        ... )
        >>> user_cfg = ValidationConfig(
        ...     commands=CommandsConfig(
        ...         test=CommandConfig(command="pytest -v"),  # override
        ...     ),
        ... )
        >>> result = merge_configs(preset_cfg, user_cfg)
        >>> result.commands.test.command
        'pytest -v'
        >>> result.commands.lint.command  # inherited
        'ruff check'
    """
    # If no preset, return user config as-is
    if preset is None:
        return user

    # Merge commands
    merged_commands = _merge_commands(preset.commands, user.commands)

    # Coverage: user replaces entirely if present, DISABLED disables, None inherits
    merged_coverage = _merge_coverage(preset.coverage, user.coverage)

    # List fields: user replaces entirely if non-empty
    merged_code_patterns = (
        user.code_patterns if user.code_patterns else preset.code_patterns
    )
    merged_config_files = (
        user.config_files if user.config_files else preset.config_files
    )
    merged_setup_files = user.setup_files if user.setup_files else preset.setup_files

    return ValidationConfig(
        preset=user.preset,  # Keep user's preset reference
        commands=merged_commands,
        coverage=merged_coverage,
        code_patterns=merged_code_patterns,
        config_files=merged_config_files,
        setup_files=merged_setup_files,
    )


def _merge_commands(
    preset: CommandsConfig,
    user: CommandsConfig,
) -> CommandsConfig:
    """Merge preset and user command configurations.

    For each command field:
    - If user value is DISABLED sentinel: result is None (disabled)
    - If user value is a CommandConfig: use user value (override)
    - If user value is None: use preset value (inherit)
    """
    return CommandsConfig(
        setup=_merge_command_field(preset.setup, user.setup),
        test=_merge_command_field(preset.test, user.test),
        lint=_merge_command_field(preset.lint, user.lint),
        format=_merge_command_field(preset.format, user.format),
        typecheck=_merge_command_field(preset.typecheck, user.typecheck),
        e2e=_merge_command_field(preset.e2e, user.e2e),
    )


def _merge_command_field(
    preset_cmd: CommandConfig | None,
    user_cmd: CommandConfig | None,
) -> CommandConfig | None:
    """Merge a single command field.

    The user_cmd may be:
    - DISABLED sentinel: explicitly disabled, return None
    - CommandConfig instance: user override, return it
    - None: not specified, inherit from preset
    """
    # Check if user explicitly disabled this command
    # The sentinel is passed through the type system as CommandConfig | None
    # but at runtime it's actually DISABLED
    if user_cmd is DISABLED:  # type: ignore[comparison-overlap]
        return None

    # If user specified a command, use it (override)
    if user_cmd is not None:
        return user_cmd

    # Otherwise inherit from preset
    return preset_cmd


def _merge_coverage(
    preset_cov: YamlCoverageConfig | None,
    user_cov: YamlCoverageConfig | None,
) -> YamlCoverageConfig | None:
    """Merge coverage configurations.

    The user_cov may be:
    - DISABLED sentinel: explicitly disabled, return None
    - YamlCoverageConfig instance: user override, return it
    - None: not specified, inherit from preset
    """
    # Check if user explicitly disabled coverage
    # The sentinel is passed through the type system as YamlCoverageConfig | None
    # but at runtime it's actually DISABLED
    if user_cov is DISABLED:  # type: ignore[comparison-overlap]
        return None

    # If user specified coverage, use it (override)
    if user_cov is not None:
        return user_cov

    # Otherwise inherit from preset
    return preset_cov
