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
    CommandConfig,
    CommandsConfig,
    ConfigError,
    CustomOverrideMode,
    ValidationConfig,
)

if TYPE_CHECKING:
    from src.domain.validation.config import (
        CustomCommandConfig,
        YamlCoverageConfig,
    )


def _validate_no_partial_commands(commands: CommandsConfig) -> None:
    """Validate that no commands have empty command strings.

    Partial configs (command="") are only valid during preset merge.
    When there's no preset, all commands must have a command string.

    Raises:
        ConfigError: If any command has an empty command string.
    """
    command_fields = ["setup", "test", "lint", "format", "typecheck", "e2e"]
    for field_name in command_fields:
        cmd = getattr(commands, field_name)
        if cmd is not None and not cmd.command:
            raise ConfigError(
                f"Command '{field_name}' in global_validation_commands has no "
                "'command' field. Partial overrides (e.g., timeout-only) require "
                "a preset to inherit the command string from."
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
    # If no preset, return user config as-is but validate no partial configs
    if preset is None:
        _validate_no_partial_commands(user.global_validation_commands)
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

    # Merge global command overrides with field-level deep merge
    # Per T005: project commands merge field-by-field with preset commands
    # e.g., project {test: {timeout: 120}} inherits command from preset
    # Special handling:
    # - Empty {} uses preset pool (inherits)
    # - Explicit null clears preset pool (via _is_null_override flag)
    user_global_validation_commands_explicitly_set = _is_field_explicitly_set(
        "global_validation_commands",
        user._fields_set,
        user.global_validation_commands,
        user.global_validation_commands == CommandsConfig(),
    )
    # Check if user explicitly set to null (clear preset)
    if (
        user_global_validation_commands_explicitly_set
        and user.global_validation_commands._is_null_override
    ):
        # Explicit null clears preset - return empty config
        merged_global_validation_commands = user.global_validation_commands
    else:
        merged_global_validation_commands = _merge_commands(
            preset.global_validation_commands,
            user.global_validation_commands,
            user_global_validation_commands_explicitly_set,
            clear_on_explicit_empty=False,  # Empty {} inherits preset pool per T005
            deep_merge_fields=True,  # Field-level merge within CommandConfig
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

    # custom_commands: user always takes precedence (presets cannot define them)
    # No merging needed - copy user's custom_commands to avoid aliasing
    merged_custom_commands = dict(user.custom_commands)

    # global_custom_commands: user always takes precedence (presets cannot define)
    # Copy if set, otherwise None (not set = use repo-level at runtime)
    merged_global_custom_commands = (
        dict(user.global_custom_commands)
        if user.global_custom_commands is not None
        else None
    )

    # Reviewer fields: user overrides if explicitly set, otherwise inherit from preset
    reviewer_type_explicitly_set = _is_field_explicitly_set(
        "reviewer_type",
        user._fields_set,
        user.reviewer_type,
        user.reviewer_type == "agent_sdk",  # Default
    )
    merged_reviewer_type = (
        user.reviewer_type if reviewer_type_explicitly_set else preset.reviewer_type
    )

    timeout_explicitly_set = _is_field_explicitly_set(
        "agent_sdk_review_timeout",
        user._fields_set,
        user.agent_sdk_review_timeout,
        user.agent_sdk_review_timeout == 600,  # Default
    )
    merged_timeout = (
        user.agent_sdk_review_timeout
        if timeout_explicitly_set
        else preset.agent_sdk_review_timeout
    )

    model_explicitly_set = _is_field_explicitly_set(
        "agent_sdk_reviewer_model",
        user._fields_set,
        user.agent_sdk_reviewer_model,
        user.agent_sdk_reviewer_model == "sonnet",  # Default
    )
    merged_model = (
        user.agent_sdk_reviewer_model
        if model_explicitly_set
        else preset.agent_sdk_reviewer_model
    )

    return ValidationConfig(
        preset=user.preset,  # Keep user's preset reference
        commands=merged_commands,
        global_validation_commands=merged_global_validation_commands,
        coverage=merged_coverage,
        code_patterns=merged_code_patterns,
        config_files=merged_config_files,
        setup_files=merged_setup_files,
        custom_commands=merged_custom_commands,
        global_custom_commands=merged_global_custom_commands,
        reviewer_type=merged_reviewer_type,
        agent_sdk_review_timeout=merged_timeout,
        agent_sdk_reviewer_model=merged_model,
        validation_triggers=user.validation_triggers,  # User's triggers (presets don't define)
        _fields_set=user._fields_set,  # Preserve user's fields_set
    )


def _merge_commands(
    preset: CommandsConfig,
    user: CommandsConfig,
    user_commands_explicitly_set: bool,
    *,
    clear_on_explicit_empty: bool = False,
    deep_merge_fields: bool = False,
) -> CommandsConfig:
    """Merge preset and user command configurations.

    For each command field:
    - If field is explicitly set by user: use user value (even if None)
    - If field is not explicitly set: inherit from preset

    When deep_merge_fields=True (for global_validation_commands), individual fields
    within CommandConfig are merged:
    - command: user overrides preset if user explicitly set it
    - timeout: user overrides preset if user explicitly set it
    This allows users to override just timeout while inheriting command from preset.

    Args:
        preset: Preset CommandsConfig to merge with.
        user: User CommandsConfig with overrides.
        user_commands_explicitly_set: Whether the user explicitly set the commands
            field at the parent level (commands or global_validation_commands).
        clear_on_explicit_empty: If True and user explicitly set the field to null
            or empty ({}) at the top level, return empty CommandsConfig without
            inheriting from preset. This is used for global_validation_commands to allow
            users to clear all preset overrides.
        deep_merge_fields: If True, merge individual fields within CommandConfig
            rather than replacing the entire CommandConfig. Used for
            global_validation_commands field-level deep merge.

    """
    # If clear_on_explicit_empty is enabled and user explicitly set commands to
    # null/empty (no individual fields set), short-circuit to return the user's
    # empty config without inheriting. This allows users to clear all preset
    # global_validation_commands overrides.
    # Note: _clear_customs sets custom_override_mode=CLEAR but doesn't set built-in
    # fields, so we must also check that no custom command directive was given.
    # Similarly, if user has custom_commands (but no built-ins), that's not "empty".
    if (
        clear_on_explicit_empty
        and user_commands_explicitly_set
        and not user._fields_set
        and not user.custom_commands
        and user.custom_override_mode == CustomOverrideMode.INHERIT
    ):
        return user

    # Choose merge function based on deep_merge_fields flag
    merge_fn = _merge_command_field_deep if deep_merge_fields else _merge_command_field

    # Determine custom_override_mode: user overrides preset only if user set a
    # non-default mode (CLEAR, REPLACE, ADDITIVE) or has custom_commands
    if user.custom_override_mode != CustomOverrideMode.INHERIT or user.custom_commands:
        merged_custom_override_mode = user.custom_override_mode
    else:
        merged_custom_override_mode = preset.custom_override_mode

    # Merge custom_commands based on mode and deep_merge_fields flag
    if merged_custom_override_mode == CustomOverrideMode.CLEAR:
        # _clear_customs: true - clear all custom commands (both preset and user)
        merged_custom_commands: dict[str, CustomCommandConfig] = {}
    elif deep_merge_fields:
        # For global_validation_commands: merge preset + user (user overrides by name)
        merged_custom_commands = {**preset.custom_commands, **user.custom_commands}
    else:
        # For regular commands: user replaces entirely
        merged_custom_commands = user.custom_commands

    return CommandsConfig(
        setup=merge_fn(preset.setup, user.setup, "setup", user._fields_set),
        test=merge_fn(preset.test, user.test, "test", user._fields_set),
        lint=merge_fn(preset.lint, user.lint, "lint", user._fields_set),
        format=merge_fn(preset.format, user.format, "format", user._fields_set),
        typecheck=merge_fn(
            preset.typecheck, user.typecheck, "typecheck", user._fields_set
        ),
        e2e=merge_fn(preset.e2e, user.e2e, "e2e", user._fields_set),
        custom_commands=merged_custom_commands,
        custom_override_mode=merged_custom_override_mode,
        _fields_set=user._fields_set,  # Preserve user's fields_set
    )


def _merge_command_field(
    preset_cmd: CommandConfig | None,
    user_cmd: CommandConfig | None,
    field_name: str,
    user_fields_set: frozenset[str],
) -> CommandConfig | None:
    """Merge a single command field (replace entire CommandConfig).

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


def _merge_command_field_deep(
    preset_cmd: CommandConfig | None,
    user_cmd: CommandConfig | None,
    field_name: str,
    user_fields_set: frozenset[str],
) -> CommandConfig | None:
    """Merge a single command field with field-level deep merge.

    For global_validation_commands, individual fields within CommandConfig are merged:
    - If user explicitly set 'command', use user's command; else inherit preset's
    - If user explicitly set 'timeout', use user's timeout; else inherit preset's

    This allows users to override just the timeout while inheriting the command
    from the preset, or vice versa.

    Args:
        preset_cmd: Preset command config (may be None).
        user_cmd: User command config (may be None or partial).
        field_name: Name of the command field (e.g., "test", "lint").
        user_fields_set: Fields explicitly set in the parent CommandsConfig.

    Returns:
        Merged CommandConfig, or None if both preset and user are None/disabled.
    """
    # Check if this command field is explicitly set by user
    is_explicit = _is_field_explicitly_set(
        field_name,
        user_fields_set,
        user_cmd,
        user_cmd is None,  # Default is None
    )

    if not is_explicit:
        # User didn't set this command - inherit from preset entirely
        return preset_cmd

    if user_cmd is None:
        # User explicitly set to None/null - disable this command
        return None

    if preset_cmd is None:
        # User set a command but no preset exists
        # If user_cmd.command is empty (""), it's a partial config without preset
        # to inherit from - this is an error
        if not user_cmd.command:
            raise ConfigError(
                f"Cannot specify partial override for '{field_name}' in "
                "global_validation_commands: no preset command to inherit from. "
                "Either provide a 'command' string or ensure the preset defines this command."
            )
        return user_cmd

    # Both preset and user have configs - do field-level merge
    # Determine which fields user explicitly set within CommandConfig
    user_cmd_fields_set = user_cmd._fields_set

    # For programmatic configs (_fields_set is empty), treat non-default values
    # as explicitly set to preserve caller's intent
    is_programmatic = not user_cmd_fields_set

    # Merge 'command' field
    # For YAML: check _fields_set; for programmatic: non-empty command is explicit
    command_is_explicit = "command" in user_cmd_fields_set or (
        is_programmatic and user_cmd.command
    )
    if command_is_explicit:
        merged_command = user_cmd.command
    else:
        # User didn't set command - inherit from preset
        merged_command = preset_cmd.command

    # Merge 'timeout' field
    # For YAML: check _fields_set; for programmatic: non-None timeout is explicit
    timeout_is_explicit = "timeout" in user_cmd_fields_set or (
        is_programmatic and user_cmd.timeout is not None
    )
    if timeout_is_explicit:
        merged_timeout = user_cmd.timeout
    else:
        # User didn't set timeout - inherit from preset
        merged_timeout = preset_cmd.timeout

    # Create merged CommandConfig
    # Preserve user's _fields_set for downstream tracking
    return CommandConfig(
        command=merged_command,
        timeout=merged_timeout,
        _fields_set=user_cmd_fields_set,
    )


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
