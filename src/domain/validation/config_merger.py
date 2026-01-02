"""Config merger for combining preset and user configurations.

This module provides the merge_configs function to merge a preset configuration
with user overrides. User values take precedence over preset values.

Merge rules:
- If no preset, return user config as-is
- Command fields: user replaces preset if explicitly set; omitted inherits
- Coverage: user replaces preset if explicitly set; omitted inherits
- List fields (code_patterns, config_files, setup_files): user replaces if explicitly set

Field presence is tracked via the `_fields_set` attribute on configs, which
records which fields were explicitly provided in the source YAML (even if
the value was null/empty). This allows distinguishing "not set" from
"explicitly set to null/empty".

For programmatic configs (where _fields_set is empty), non-None values are
treated as explicit overrides. This ensures programmatic callers can override
preset values without needing to manually populate _fields_set.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.domain.validation.config import (
    CommandsConfig,
    ValidationConfig,
)

if TYPE_CHECKING:
    from src.domain.validation.config import (
        CommandConfig,
        YamlCoverageConfig,
    )


def _is_field_explicitly_set(
    field_name: str,
    fields_set: frozenset[str],
    user_value: object,
    is_default: bool,
) -> bool:
    """Determine if a field should be treated as explicitly set.

    For YAML configs (fields_set is populated), use fields_set membership.
    For programmatic configs (fields_set is empty), treat non-None values
    that differ from the default as explicit overrides.

    Args:
        field_name: Name of the field to check.
        fields_set: The _fields_set from the config.
        user_value: The current value of the field.
        is_default: Whether user_value equals the default value.

    Returns:
        True if the field should be treated as explicitly set.
    """
    # If fields_set is populated, use it (YAML config via from_dict)
    if fields_set:
        return field_name in fields_set

    # For programmatic configs (empty fields_set):
    # Treat non-default values as explicit overrides
    return not is_default


def merge_configs(
    preset: ValidationConfig | None,
    user: ValidationConfig,
) -> ValidationConfig:
    """Merge preset configuration with user overrides.

    User values take precedence over preset values. When a user field is not
    explicitly set (not in _fields_set), the preset value is inherited. When
    a user field is explicitly set (in _fields_set), the user value is used
    even if it's None or empty.

    For programmatic configs (created via constructor, where _fields_set is
    empty), non-default values are treated as explicit overrides. This ensures
    both YAML and programmatic configs work correctly.

    Args:
        preset: Base preset configuration to merge with, or None.
        user: User configuration with overrides.

    Returns:
        Merged ValidationConfig with user values taking precedence.

    Examples:
        >>> # Use from_dict to create configs - this populates _fields_set
        >>> # which is required for overrides to work correctly
        >>> preset_cfg = ValidationConfig.from_dict({
        ...     "commands": {
        ...         "test": "pytest",
        ...         "lint": "ruff check",
        ...     },
        ...     "code_patterns": ["**/*.py"],
        ... })
        >>> user_cfg = ValidationConfig.from_dict({
        ...     "commands": {
        ...         "test": "pytest -v",  # override
        ...     },
        ... })
        >>> result = merge_configs(preset_cfg, user_cfg)
        >>> result.commands.test.command
        'pytest -v'
        >>> result.commands.lint.command  # inherited
        'ruff check'
    """
    # If no preset, return user config as-is
    if preset is None:
        return user

    # Merge commands - check if user explicitly set commands
    user_commands_explicitly_set = _is_field_explicitly_set(
        "commands",
        user._fields_set,
        user.commands,
        # Default is an empty CommandsConfig with empty _fields_set
        user.commands == CommandsConfig(),
    )
    merged_commands = _merge_commands(
        preset.commands, user.commands, user_commands_explicitly_set
    )

    # Coverage: user replaces if explicitly set, otherwise inherit
    user_coverage_explicitly_set = _is_field_explicitly_set(
        "coverage",
        user._fields_set,
        user.coverage,
        user.coverage is None,  # Default is None
    )
    merged_coverage = _merge_coverage(
        preset.coverage, user.coverage, user_coverage_explicitly_set
    )

    # List fields: user replaces if explicitly set, otherwise inherit
    code_patterns_explicitly_set = _is_field_explicitly_set(
        "code_patterns",
        user._fields_set,
        user.code_patterns,
        user.code_patterns == (),  # Default is empty tuple
    )
    merged_code_patterns = (
        user.code_patterns if code_patterns_explicitly_set else preset.code_patterns
    )

    config_files_explicitly_set = _is_field_explicitly_set(
        "config_files",
        user._fields_set,
        user.config_files,
        user.config_files == (),  # Default is empty tuple
    )
    merged_config_files = (
        user.config_files if config_files_explicitly_set else preset.config_files
    )

    setup_files_explicitly_set = _is_field_explicitly_set(
        "setup_files",
        user._fields_set,
        user.setup_files,
        user.setup_files == (),  # Default is empty tuple
    )
    merged_setup_files = (
        user.setup_files if setup_files_explicitly_set else preset.setup_files
    )

    return ValidationConfig(
        preset=user.preset,  # Keep user's preset reference
        commands=merged_commands,
        coverage=merged_coverage,
        code_patterns=merged_code_patterns,
        config_files=merged_config_files,
        setup_files=merged_setup_files,
        _fields_set=user._fields_set,  # Preserve user's fields_set
    )


def _merge_commands(
    preset: CommandsConfig,
    user: CommandsConfig,
    user_commands_explicitly_set: bool,
) -> CommandsConfig:
    """Merge preset and user command configurations.

    For each command field:
    - If field is explicitly set by user: use user value (even if None)
    - If field is not explicitly set: inherit from preset

    Special case: If the user explicitly set the commands field to null or
    an empty object (i.e., user_commands_explicitly_set is True but
    user._fields_set is empty and all command values are None), this clears
    all preset commands.
    """
    # Short-circuit: If user explicitly set commands to null/empty (no individual
    # command fields set, and we're using YAML-style tracking), this clears all
    # preset commands
    if user_commands_explicitly_set and not user._fields_set:
        # Check if this is a "clear all" case - user explicitly set commands
        # but all values are None (from YAML: commands: null or commands: {})
        all_none = (
            user.setup is None
            and user.test is None
            and user.lint is None
            and user.format is None
            and user.typecheck is None
            and user.e2e is None
        )
        if all_none:
            return user

    return CommandsConfig(
        setup=_merge_command_field(preset.setup, user.setup, "setup", user._fields_set),
        test=_merge_command_field(preset.test, user.test, "test", user._fields_set),
        lint=_merge_command_field(preset.lint, user.lint, "lint", user._fields_set),
        format=_merge_command_field(
            preset.format, user.format, "format", user._fields_set
        ),
        typecheck=_merge_command_field(
            preset.typecheck, user.typecheck, "typecheck", user._fields_set
        ),
        e2e=_merge_command_field(preset.e2e, user.e2e, "e2e", user._fields_set),
        _fields_set=user._fields_set,  # Preserve user's fields_set
    )


def _merge_command_field(
    preset_cmd: CommandConfig | None,
    user_cmd: CommandConfig | None,
    field_name: str,
    user_fields_set: frozenset[str],
) -> CommandConfig | None:
    """Merge a single command field.

    If the field was explicitly set by the user (in user_fields_set or
    non-None for programmatic configs), use the user value. Otherwise,
    inherit from preset.
    """
    # Check if field is explicitly set
    is_explicit = _is_field_explicitly_set(
        field_name,
        user_fields_set,
        user_cmd,
        user_cmd is None,  # Default is None
    )

    if is_explicit:
        return user_cmd

    # Otherwise inherit from preset
    return preset_cmd


def _merge_coverage(
    preset_cov: YamlCoverageConfig | None,
    user_cov: YamlCoverageConfig | None,
    user_explicitly_set: bool,
) -> YamlCoverageConfig | None:
    """Merge coverage configurations.

    If user explicitly set coverage (even to null), use user value.
    Otherwise inherit from preset.
    """
    # If user explicitly set coverage, use their value (even if None)
    if user_explicitly_set:
        return user_cov

    # Otherwise inherit from preset
    return preset_cov
