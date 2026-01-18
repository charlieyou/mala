"""YAML configuration loader for mala.yaml.

This module provides functionality to load, parse, and validate the mala.yaml
configuration file. It enforces strict schema validation and provides clear
error messages for common misconfigurations.

Key functions:
- load_config: Load and validate mala.yaml from a repository path
- parse_yaml: Parse YAML content with error handling
- validate_schema: Validate against expected schema
- build_config: Convert parsed dict to ValidationConfig dataclass
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import yaml

from src.domain.validation.config import (
    ConfigError,
    EpicCompletionTriggerConfig,
    EpicDepth,
    EvidenceCheckConfig,
    FailureMode,
    FireOn,
    PeriodicTriggerConfig,
    RunEndTriggerConfig,
    SessionEndTriggerConfig,
    TriggerCommandRef,
    ValidationConfig,
    ValidationTriggersConfig,
)

if TYPE_CHECKING:
    from pathlib import Path

    from src.domain.validation.config import (
        BaseTriggerConfig,
        CerberusConfig,
        CodeReviewConfig,
        EpicVerifierConfig,
        VerificationRetryPolicy,
    )


class ConfigMissingError(ConfigError):
    """Raised when the mala.yaml configuration file is not found.

    This is a subclass of ConfigError to allow callers to catch either:
    - ConfigMissingError: Only handle missing file case
    - ConfigError: Handle all config errors (missing, invalid syntax, etc.)

    Example:
        >>> raise ConfigMissingError(Path("/path/to/repo"))
        ConfigMissingError: mala.yaml not found in /path/to/repo.
    """

    repo_path: Path  # Explicit class-level annotation for type checkers

    def __init__(self, repo_path: Path) -> None:
        self.repo_path = repo_path
        message = f"mala.yaml not found in {repo_path}. Mala requires a configuration file to run."
        super().__init__(message)


# Fields allowed at the top level of mala.yaml
_ALLOWED_TOP_LEVEL_FIELDS = frozenset(
    {
        "preset",
        "commands",
        "coverage",
        "code_patterns",
        "config_files",
        "setup_files",
        "validation_triggers",
        "claude_settings_sources",
        "timeout_minutes",
        "max_idle_retries",
        "idle_timeout_seconds",
        "max_diff_size_kb",
        "evidence_check",
        "epic_verification",
        "per_issue_review",
    }
)


def load_config(repo_path: Path) -> ValidationConfig:
    """Load and validate mala.yaml from the repository root.

    This is the main entry point for loading configuration. It reads the file,
    parses YAML, validates the schema, builds the config dataclass, and runs
    post-build validation.

    Args:
        repo_path: Path to the repository root directory.

    Returns:
        ValidationConfig instance with all configuration loaded.

    Raises:
        ConfigError: If the file is missing, has invalid YAML syntax,
            contains unknown fields, has invalid types, or fails validation.

    Example:
        >>> config = load_config(Path("/path/to/repo"))
        >>> print(config.preset)
        'python-uv'
    """
    config_file = repo_path / "mala.yaml"

    if not config_file.exists():
        raise ConfigMissingError(repo_path)

    try:
        content = config_file.read_text(encoding="utf-8")
    except OSError as e:
        raise ConfigError(f"Failed to read {config_file}: {e}") from e
    except UnicodeDecodeError as e:
        raise ConfigError(f"Failed to decode {config_file}: {e}") from e
    data = _parse_yaml(content)
    _validate_schema(data)
    config = _build_config(data)
    _validate_config(config)
    return config


def _parse_yaml(content: str) -> dict[str, Any]:
    """Parse YAML content into a dictionary.

    Args:
        content: Raw YAML string content.

    Returns:
        Parsed dictionary. Returns empty dict for empty/null YAML.

    Raises:
        ConfigError: If YAML syntax is invalid.
    """
    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as e:
        # Extract useful error details from the exception
        details = str(e)
        raise ConfigError(f"Invalid YAML syntax in mala.yaml: {details}") from e

    # Handle empty file or file with only comments
    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ConfigError(
            f"mala.yaml must be a YAML mapping, got {type(data).__name__}"
        )

    return data


def _validate_schema(data: dict[str, Any]) -> None:
    """Validate the parsed YAML against the expected schema.

    This function checks for unknown fields at the top level. Field type
    validation is handled by the dataclass constructors.

    Args:
        data: Parsed YAML dictionary.

    Raises:
        ConfigError: If unknown fields are present.
    """
    # Check for removed global_validation_commands with helpful migration message
    if "global_validation_commands" in data:
        raise ConfigError(
            "global_validation_commands is not supported. "
            "Migrate to validation_triggers with commands:\n\n"
            "validation_triggers:\n"
            "  run_end:\n"
            "    failure_mode: continue\n"
            "    commands:\n"
            "      - ref: test\n"
            "      - ref: lint\n\n"
            "See migration guide at "
            "https://docs.mala.ai/migration/validation-triggers"
        )

    # Check for removed custom_commands top-level field
    if "custom_commands" in data:
        raise ConfigError(
            "custom_commands (top-level) is not supported. "
            "Define custom commands under the 'commands' key:\n\n"
            "commands:\n"
            "  my_custom_cmd:\n"
            "    command: 'my-tool --check'\n"
            "    timeout: 120\n"
        )

    # Check for removed validate_every field with helpful migration message
    if "validate_every" in data:
        raise ConfigError(
            "validate_every is not supported. Use validation_triggers.periodic with "
            "interval field. See migration guide at "
            "https://docs.mala.ai/migration/validation-triggers"
        )

    unknown_fields = set(data.keys()) - _ALLOWED_TOP_LEVEL_FIELDS
    if unknown_fields:
        # Sort for consistent error messages; convert to str to handle
        # non-string YAML keys (e.g., null, integers) without TypeError
        unknown_as_strs = sorted(str(k) for k in unknown_fields)
        first_unknown = unknown_as_strs[0]
        raise ConfigError(f"Unknown field '{first_unknown}' in mala.yaml")


_TRIGGER_COMMAND_REF_FIELDS = frozenset({"ref", "command", "timeout"})


def _parse_trigger_command_ref(
    data: dict[str, Any], trigger_name: str, cmd_index: int
) -> TriggerCommandRef:
    """Parse a single command reference from a trigger's commands list.

    Args:
        data: Dict with 'ref' (required), optional 'command' and 'timeout' overrides.
        trigger_name: Name of the trigger for error messages.
        cmd_index: Index of the command in the list for error messages.

    Returns:
        TriggerCommandRef instance.

    Raises:
        ConfigError: If 'ref' is missing, has wrong type, or unknown fields present.
    """
    # Validate unknown fields
    unknown = set(data.keys()) - _TRIGGER_COMMAND_REF_FIELDS
    if unknown:
        first = sorted(str(k) for k in unknown)[0]
        raise ConfigError(
            f"Unknown field '{first}' in command {cmd_index} of trigger {trigger_name}"
        )

    if "ref" not in data:
        raise ConfigError(
            f"'ref' is required for command {cmd_index} in trigger {trigger_name}"
        )
    ref = data["ref"]
    if not isinstance(ref, str):
        raise ConfigError(
            f"'ref' must be a string in command {cmd_index} of trigger {trigger_name}, "
            f"got {type(ref).__name__}"
        )
    if not ref.strip():
        raise ConfigError(
            f"'ref' cannot be empty in command {cmd_index} of trigger {trigger_name}"
        )

    command = data.get("command")
    if command is not None:
        if not isinstance(command, str):
            raise ConfigError(
                f"'command' must be a string in command {cmd_index} of trigger {trigger_name}, "
                f"got {type(command).__name__}"
            )
        if not command.strip():
            raise ConfigError(
                f"'command' cannot be empty in command {cmd_index} of trigger {trigger_name}"
            )

    timeout = data.get("timeout")
    if timeout is not None:
        if isinstance(timeout, bool) or not isinstance(timeout, int):
            raise ConfigError(
                f"'timeout' must be an integer in command {cmd_index} of trigger {trigger_name}, "
                f"got {type(timeout).__name__}"
            )

    return TriggerCommandRef(ref=ref, command=command, timeout=timeout)


def _parse_commands_list(
    data: dict[str, Any], trigger_name: str
) -> tuple[TriggerCommandRef, ...]:
    """Parse the commands list for a trigger.

    Args:
        data: The trigger config dict.
        trigger_name: Name of the trigger for error messages.

    Returns:
        Tuple of TriggerCommandRef instances.

    Raises:
        ConfigError: If commands has wrong type.
    """
    commands_data = data.get("commands", [])
    if not isinstance(commands_data, list):
        raise ConfigError(
            f"'commands' must be a list for trigger {trigger_name}, "
            f"got {type(commands_data).__name__}"
        )

    commands: list[TriggerCommandRef] = []
    for i, cmd_data in enumerate(commands_data):
        if isinstance(cmd_data, str):
            # Allow shorthand: just the ref string
            if not cmd_data.strip():
                raise ConfigError(
                    f"Command {i} in trigger {trigger_name} cannot be an empty string"
                )
            commands.append(TriggerCommandRef(ref=cmd_data))
        elif isinstance(cmd_data, dict):
            commands.append(_parse_trigger_command_ref(cmd_data, trigger_name, i))
        else:
            raise ConfigError(
                f"Command {i} in trigger {trigger_name} must be a string or object, "
                f"got {type(cmd_data).__name__}"
            )

    return tuple(commands)


def _parse_failure_mode(data: dict[str, Any], trigger_name: str) -> FailureMode:
    """Parse and validate failure_mode for a trigger.

    Args:
        data: The trigger config dict.
        trigger_name: Name of the trigger for error messages.

    Returns:
        FailureMode enum value.

    Raises:
        ConfigError: If failure_mode is missing or invalid.
    """
    if "failure_mode" not in data:
        raise ConfigError(f"failure_mode required for trigger {trigger_name}")

    mode_str = data["failure_mode"]
    if not isinstance(mode_str, str):
        raise ConfigError(
            f"failure_mode must be a string for trigger {trigger_name}, "
            f"got {type(mode_str).__name__}"
        )

    try:
        return FailureMode(mode_str)
    except ValueError:
        valid = ", ".join(m.value for m in FailureMode)
        raise ConfigError(
            f"Invalid failure_mode '{mode_str}' for trigger {trigger_name}. "
            f"Valid values: {valid}"
        ) from None


def _parse_max_retries(
    data: dict[str, Any], trigger_name: str, failure_mode: FailureMode
) -> int | None:
    """Parse and validate max_retries for a trigger.

    Args:
        data: The trigger config dict.
        trigger_name: Name of the trigger for error messages.
        failure_mode: The parsed failure mode.

    Returns:
        max_retries value or None.

    Raises:
        ConfigError: If max_retries is required but missing, or has wrong type.
    """
    max_retries = data.get("max_retries")

    if failure_mode == FailureMode.REMEDIATE and max_retries is None:
        raise ConfigError(
            f"max_retries required when failure_mode=remediate for trigger {trigger_name}"
        )

    if max_retries is not None:
        if isinstance(max_retries, bool) or not isinstance(max_retries, int):
            raise ConfigError(
                f"max_retries must be an integer for trigger {trigger_name}, "
                f"got {type(max_retries).__name__}"
            )
        if max_retries < 0:
            raise ConfigError(
                f"max_retries must be >= 0 for trigger {trigger_name}, got {max_retries}"
            )

    return max_retries


_CODE_REVIEW_FIELDS = frozenset(
    {
        "enabled",
        "reviewer_type",
        "failure_mode",
        "max_retries",
        "finding_threshold",
        "baseline",
        "cerberus",
        "agent_sdk_timeout",
        "agent_sdk_model",
        "track_review_issues",
    }
)

_CERBERUS_FIELDS = frozenset({"timeout", "spawn_args", "wait_args", "env"})


def _parse_cerberus_config(data: dict[str, Any]) -> CerberusConfig:
    """Parse cerberus-specific configuration block.

    Args:
        data: The cerberus config dict.

    Returns:
        CerberusConfig instance.

    Raises:
        ConfigError: If data has invalid types or unknown fields.
    """
    # Import here to avoid circular import at module level
    from src.domain.validation.config import CerberusConfig as CerberusConfigClass

    # Validate unknown fields
    unknown = set(data.keys()) - _CERBERUS_FIELDS
    if unknown:
        first = sorted(str(k) for k in unknown)[0]
        raise ConfigError(f"Unknown field '{first}' in code_review.cerberus")

    # Parse timeout (optional, defaults to 300)
    timeout = 300
    if "timeout" in data:
        timeout_val = data["timeout"]
        if isinstance(timeout_val, bool) or not isinstance(timeout_val, int):
            raise ConfigError(
                f"cerberus.timeout must be an integer, got {type(timeout_val).__name__}"
            )
        timeout = timeout_val

    # Parse spawn_args (optional)
    spawn_args: tuple[str, ...] = ()
    if "spawn_args" in data:
        spawn_args_val = data["spawn_args"]
        if not isinstance(spawn_args_val, list):
            raise ConfigError(
                f"cerberus.spawn_args must be a list, got {type(spawn_args_val).__name__}"
            )
        for i, arg in enumerate(spawn_args_val):
            if not isinstance(arg, str):
                raise ConfigError(
                    f"cerberus.spawn_args[{i}] must be a string, "
                    f"got {type(arg).__name__}"
                )
        spawn_args = tuple(spawn_args_val)

    # Parse wait_args (optional)
    wait_args: tuple[str, ...] = ()
    if "wait_args" in data:
        wait_args_val = data["wait_args"]
        if not isinstance(wait_args_val, list):
            raise ConfigError(
                f"cerberus.wait_args must be a list, got {type(wait_args_val).__name__}"
            )
        for i, arg in enumerate(wait_args_val):
            if not isinstance(arg, str):
                raise ConfigError(
                    f"cerberus.wait_args[{i}] must be a string, "
                    f"got {type(arg).__name__}"
                )
        wait_args = tuple(wait_args_val)

    # Parse env (optional)
    env: tuple[tuple[str, str], ...] = ()
    if "env" in data:
        env_val = data["env"]
        if not isinstance(env_val, dict):
            raise ConfigError(
                f"cerberus.env must be an object, got {type(env_val).__name__}"
            )
        env_list: list[tuple[str, str]] = []
        for key, value in env_val.items():
            if not isinstance(key, str):
                raise ConfigError(
                    f"cerberus.env key must be a string, got {type(key).__name__}"
                )
            if not isinstance(value, str):
                raise ConfigError(
                    f"cerberus.env['{key}'] must be a string, "
                    f"got {type(value).__name__}"
                )
            env_list.append((key, value))
        env = tuple(sorted(env_list))

    return CerberusConfigClass(
        timeout=timeout,
        spawn_args=spawn_args,
        wait_args=wait_args,
        env=env,
    )


_EPIC_VERIFICATION_FIELDS = frozenset(
    {
        "enabled",
        "reviewer_type",
        "timeout",
        "max_retries",
        "failure_mode",
        "cerberus",
        "agent_sdk_timeout",
        "agent_sdk_model",
        "retry_policy",
    }
)

_RETRY_POLICY_FIELDS = frozenset(
    {"timeout_retries", "execution_retries", "parse_retries"}
)


def _parse_retry_policy(
    data: dict[str, Any] | None,
) -> VerificationRetryPolicy:
    """Parse retry_policy configuration block.

    Args:
        data: The retry_policy config dict from epic_verification, or None.

    Returns:
        VerificationRetryPolicy with parsed settings (defaults if data is None).

    Raises:
        ConfigError: If fields are invalid or unknown fields present.
    """
    from src.domain.validation.config import (
        VerificationRetryPolicy as VerificationRetryPolicyClass,
    )

    if data is None:
        return VerificationRetryPolicyClass()

    if not isinstance(data, dict):
        raise ConfigError(
            f"epic_verification.retry_policy must be an object, "
            f"got {type(data).__name__}"
        )

    # Validate unknown fields - fail fast
    unknown = set(data.keys()) - _RETRY_POLICY_FIELDS
    if unknown:
        first = sorted(str(k) for k in unknown)[0]
        raise ConfigError(f"Unknown field '{first}' in epic_verification.retry_policy")

    # Parse timeout_retries (optional, defaults to 3)
    timeout_retries = 3
    if "timeout_retries" in data:
        val = data["timeout_retries"]
        if isinstance(val, bool) or not isinstance(val, int):
            raise ConfigError(
                f"epic_verification.retry_policy.timeout_retries must be an integer, "
                f"got {type(val).__name__}"
            )
        if val < 0:
            raise ConfigError(
                f"epic_verification.retry_policy.timeout_retries must be non-negative, "
                f"got {val}"
            )
        timeout_retries = val

    # Parse execution_retries (optional, defaults to 2)
    execution_retries = 2
    if "execution_retries" in data:
        val = data["execution_retries"]
        if isinstance(val, bool) or not isinstance(val, int):
            raise ConfigError(
                f"epic_verification.retry_policy.execution_retries must be an integer, "
                f"got {type(val).__name__}"
            )
        if val < 0:
            raise ConfigError(
                f"epic_verification.retry_policy.execution_retries must be non-negative, "
                f"got {val}"
            )
        execution_retries = val

    # Parse parse_retries (optional, defaults to 1)
    parse_retries = 1
    if "parse_retries" in data:
        val = data["parse_retries"]
        if isinstance(val, bool) or not isinstance(val, int):
            raise ConfigError(
                f"epic_verification.retry_policy.parse_retries must be an integer, "
                f"got {type(val).__name__}"
            )
        if val < 0:
            raise ConfigError(
                f"epic_verification.retry_policy.parse_retries must be non-negative, "
                f"got {val}"
            )
        parse_retries = val

    return VerificationRetryPolicyClass(
        timeout_retries=timeout_retries,
        execution_retries=execution_retries,
        parse_retries=parse_retries,
    )


def _parse_epic_verification_config(
    data: dict[str, Any] | None,
) -> EpicVerifierConfig:
    """Parse epic_verification configuration block.

    Args:
        data: The epic_verification config dict from mala.yaml, or None.

    Returns:
        EpicVerifierConfig with parsed settings (defaults if data is None).

    Raises:
        ConfigError: If fields are invalid or unknown fields present.
    """
    from src.domain.validation.config import (
        EpicVerifierConfig as EpicVerifierConfigClass,
    )

    if data is None:
        return EpicVerifierConfigClass()

    if not isinstance(data, dict):
        raise ConfigError(
            f"epic_verification must be an object, got {type(data).__name__}"
        )

    # Validate unknown fields - fail fast
    unknown = set(data.keys()) - _EPIC_VERIFICATION_FIELDS
    if unknown:
        first = sorted(str(k) for k in unknown)[0]
        raise ConfigError(f"Unknown field '{first}' in epic_verification")

    # Parse enabled (optional, defaults to True)
    enabled = True
    if "enabled" in data:
        enabled_val = data["enabled"]
        if not isinstance(enabled_val, bool):
            raise ConfigError(
                f"epic_verification.enabled must be a boolean, "
                f"got {type(enabled_val).__name__}"
            )
        enabled = enabled_val

    # Parse reviewer_type (optional, defaults to "agent_sdk")
    reviewer_type: Literal["cerberus", "agent_sdk"] = "agent_sdk"
    if "reviewer_type" in data:
        val = data["reviewer_type"]
        if val not in ("cerberus", "agent_sdk"):
            raise ConfigError(
                f"epic_verification.reviewer_type must be 'cerberus' or 'agent_sdk', "
                f"got '{val}'"
            )
        reviewer_type = val

    # Parse timeout (optional, defaults to 600)
    timeout = 600
    if "timeout" in data:
        timeout_val = data["timeout"]
        if isinstance(timeout_val, bool) or not isinstance(timeout_val, int):
            raise ConfigError(
                f"epic_verification.timeout must be an integer, "
                f"got {type(timeout_val).__name__}"
            )
        if timeout_val < 0:
            raise ConfigError(
                f"epic_verification.timeout must be non-negative, got {timeout_val}"
            )
        timeout = timeout_val

    # Parse max_retries (optional, defaults to 3)
    max_retries = 3
    if "max_retries" in data:
        max_retries_val = data["max_retries"]
        if isinstance(max_retries_val, bool) or not isinstance(max_retries_val, int):
            raise ConfigError(
                f"epic_verification.max_retries must be an integer, "
                f"got {type(max_retries_val).__name__}"
            )
        if max_retries_val < 0:
            raise ConfigError(
                f"epic_verification.max_retries must be non-negative, "
                f"got {max_retries_val}"
            )
        max_retries = max_retries_val

    # Parse failure_mode (optional, defaults to CONTINUE)
    failure_mode = FailureMode.CONTINUE
    if "failure_mode" in data:
        fm_val = data["failure_mode"]
        if not isinstance(fm_val, str):
            raise ConfigError(
                f"epic_verification.failure_mode must be a string, "
                f"got {type(fm_val).__name__}"
            )
        try:
            failure_mode = FailureMode(fm_val)
        except ValueError:
            valid = ", ".join(f.value for f in FailureMode)
            raise ConfigError(
                f"epic_verification.failure_mode must be one of {valid}, got '{fm_val}'"
            ) from None

    # Parse cerberus (optional nested block)
    cerberus: CerberusConfig | None = None
    if "cerberus" in data:
        cerberus_val = data["cerberus"]
        if cerberus_val is not None:
            if not isinstance(cerberus_val, dict):
                raise ConfigError(
                    f"epic_verification.cerberus must be an object, "
                    f"got {type(cerberus_val).__name__}"
                )
            cerberus = _parse_cerberus_config(cerberus_val)

    # Parse agent_sdk_timeout (optional, defaults to 600)
    agent_sdk_timeout = 600
    if "agent_sdk_timeout" in data:
        timeout_val = data["agent_sdk_timeout"]
        if isinstance(timeout_val, bool) or not isinstance(timeout_val, int):
            raise ConfigError(
                f"epic_verification.agent_sdk_timeout must be an integer, "
                f"got {type(timeout_val).__name__}"
            )
        if timeout_val < 0:
            raise ConfigError(
                f"epic_verification.agent_sdk_timeout must be non-negative, "
                f"got {timeout_val}"
            )
        agent_sdk_timeout = timeout_val

    # Parse agent_sdk_model (optional, defaults to "sonnet")
    agent_sdk_model: Literal["sonnet", "opus", "haiku"] = "sonnet"
    if "agent_sdk_model" in data:
        model_val = data["agent_sdk_model"]
        if model_val not in ("sonnet", "opus", "haiku"):
            raise ConfigError(
                f"epic_verification.agent_sdk_model must be 'sonnet', 'opus', or 'haiku', "
                f"got '{model_val}'"
            )
        agent_sdk_model = model_val

    # Parse retry_policy (optional nested block for per-category retry limits)
    retry_policy = _parse_retry_policy(data.get("retry_policy"))

    return EpicVerifierConfigClass(
        enabled=enabled,
        reviewer_type=reviewer_type,
        timeout=timeout,
        max_retries=max_retries,
        failure_mode=failure_mode,
        cerberus=cerberus,
        agent_sdk_timeout=agent_sdk_timeout,
        agent_sdk_model=agent_sdk_model,
        retry_policy=retry_policy,
    )


def _parse_code_review_config(
    data: dict[str, Any],
    trigger_name: str,
    *,
    is_per_issue_review: bool = False,
) -> CodeReviewConfig | None:
    """Parse code_review configuration block.

    Args:
        data: The code_review config dict from the trigger or per_issue_review section.
        trigger_name: Name of the parent trigger (for validation warnings/errors).
            When is_per_issue_review=True, this is used only in error messages.
        is_per_issue_review: If True, parse as per_issue_review config rather than
            a trigger's code_review config. This affects validation: baseline is
            not applicable for per_issue_review and will emit a warning if set.

    Returns:
        CodeReviewConfig if data is provided, None otherwise.

    Raises:
        ConfigError: If required fields missing, invalid, or unknown fields present.
    """
    import logging

    # Import here to avoid circular import at module level
    from src.domain.validation.config import CodeReviewConfig as CodeReviewConfigClass

    logger = logging.getLogger(__name__)

    if data is None:
        return None

    if not isinstance(data, dict):
        raise ConfigError(
            f"code_review must be an object for trigger {trigger_name}, "
            f"got {type(data).__name__}"
        )

    # Validate unknown fields - fail fast
    unknown = set(data.keys()) - _CODE_REVIEW_FIELDS
    if unknown:
        first = sorted(str(k) for k in unknown)[0]
        raise ConfigError(f"Unknown field '{first}' in code_review for {trigger_name}")

    # Parse enabled (optional, defaults to False)
    enabled = False
    if "enabled" in data:
        enabled_val = data["enabled"]
        if not isinstance(enabled_val, bool):
            raise ConfigError(
                f"code_review.enabled must be a boolean for trigger {trigger_name}, "
                f"got {type(enabled_val).__name__}"
            )
        enabled = enabled_val

    # Parse reviewer_type (optional, defaults to "cerberus")
    reviewer_type: Literal["cerberus", "agent_sdk"] = "cerberus"
    if "reviewer_type" in data:
        rt_val = data["reviewer_type"]
        if not isinstance(rt_val, str):
            raise ConfigError(
                f"code_review.reviewer_type must be a string for trigger {trigger_name}, "
                f"got {type(rt_val).__name__}"
            )
        if rt_val not in ("cerberus", "agent_sdk"):
            # ERROR only if enabled: true with invalid reviewer_type
            # When disabled, allow invalid value (per spec)
            if enabled:
                raise ConfigError(
                    f"code_review.reviewer_type must be 'cerberus' or 'agent_sdk' "
                    f"for trigger {trigger_name}, got '{rt_val}'"
                )
        reviewer_type = rt_val  # type: ignore[assignment]

    # Parse failure_mode (optional, defaults to CONTINUE)
    failure_mode = FailureMode.CONTINUE
    if "failure_mode" in data:
        fm_val = data["failure_mode"]
        if not isinstance(fm_val, str):
            raise ConfigError(
                f"code_review.failure_mode must be a string for trigger {trigger_name}, "
                f"got {type(fm_val).__name__}"
            )
        try:
            failure_mode = FailureMode(fm_val)
        except ValueError:
            valid = ", ".join(m.value for m in FailureMode)
            raise ConfigError(
                f"Invalid code_review.failure_mode '{fm_val}' for trigger {trigger_name}. "
                f"Valid values: {valid}"
            ) from None

    # Parse max_retries (optional, defaults to 3)
    max_retries = 3
    if "max_retries" in data:
        mr_val = data["max_retries"]
        if isinstance(mr_val, bool) or not isinstance(mr_val, int):
            raise ConfigError(
                f"code_review.max_retries must be an integer for trigger {trigger_name}, "
                f"got {type(mr_val).__name__}"
            )
        # ERROR if max_retries < 0
        if mr_val < 0:
            raise ConfigError(
                f"code_review.max_retries must be >= 0 for trigger {trigger_name}, "
                f"got {mr_val}"
            )
        max_retries = mr_val

    # Parse finding_threshold (optional, defaults to "none")
    finding_threshold: Literal["P0", "P1", "P2", "P3", "none"] = "none"
    valid_thresholds = ("P0", "P1", "P2", "P3", "none")
    if "finding_threshold" in data:
        ft_val = data["finding_threshold"]
        if not isinstance(ft_val, str):
            raise ConfigError(
                f"code_review.finding_threshold must be a string for trigger {trigger_name}, "
                f"got {type(ft_val).__name__}"
            )
        if ft_val not in valid_thresholds:
            raise ConfigError(
                f"Invalid code_review.finding_threshold '{ft_val}' for trigger {trigger_name}. "
                f"Valid values: {', '.join(valid_thresholds)}"
            )
        finding_threshold = ft_val  # type: ignore[assignment]

    # Parse baseline with trigger-specific validation
    baseline: Literal["since_run_start", "since_last_review"] | None = None
    valid_baselines = ("since_run_start", "since_last_review")
    if "baseline" in data:
        bl_val = data["baseline"]
        if bl_val is not None:
            if not isinstance(bl_val, str):
                raise ConfigError(
                    f"code_review.baseline must be a string for trigger {trigger_name}, "
                    f"got {type(bl_val).__name__}"
                )
            if bl_val not in valid_baselines:
                raise ConfigError(
                    f"Invalid code_review.baseline '{bl_val}' for trigger {trigger_name}. "
                    f"Valid values: {', '.join(valid_baselines)}"
                )
            # WARN if baseline set for per_issue_review - ignore field
            # per_issue_review reviews individual commits, not a range of code
            if is_per_issue_review:
                logger.warning(
                    "baseline is not applicable for per_issue_review; "
                    "ignoring baseline='%s'",
                    bl_val,
                )
                baseline = None
            # WARN if baseline set for session_end - ignore field
            elif trigger_name == "session_end":
                logger.warning(
                    "code_review.baseline is not applicable for session_end trigger; "
                    "ignoring baseline='%s'",
                    bl_val,
                )
                baseline = None
            else:
                baseline = bl_val  # type: ignore[assignment]

    # WARN if baseline missing/null for epic_completion/run_end - default to since_run_start
    # Explicit null counts as missing since baseline is required for these triggers
    # This does NOT apply to per_issue_review (baseline is not applicable there)
    if (
        not is_per_issue_review
        and trigger_name in ("epic_completion", "run_end")
        and baseline is None
    ):
        logger.warning(
            "code_review.baseline not specified for %s trigger; "
            "defaulting to 'since_run_start'",
            trigger_name,
        )
        baseline = "since_run_start"

    # Parse cerberus (optional)
    cerberus = None
    if "cerberus" in data:
        cerberus_val = data["cerberus"]
        if cerberus_val is not None:
            if not isinstance(cerberus_val, dict):
                raise ConfigError(
                    f"code_review.cerberus must be an object for trigger {trigger_name}, "
                    f"got {type(cerberus_val).__name__}"
                )
            cerberus = _parse_cerberus_config(cerberus_val)

    # Parse agent_sdk_timeout (optional, defaults to 600)
    agent_sdk_timeout = 600
    if "agent_sdk_timeout" in data:
        timeout_val = data["agent_sdk_timeout"]
        if isinstance(timeout_val, bool) or not isinstance(timeout_val, int):
            raise ConfigError(
                f"code_review.agent_sdk_timeout must be an integer for trigger {trigger_name}, "
                f"got {type(timeout_val).__name__}"
            )
        if timeout_val <= 0:
            raise ConfigError(
                f"code_review.agent_sdk_timeout must be positive for trigger {trigger_name}, "
                f"got {timeout_val}"
            )
        agent_sdk_timeout = timeout_val

    # Parse agent_sdk_model (optional, defaults to "sonnet")
    agent_sdk_model: Literal["sonnet", "opus", "haiku"] = "sonnet"
    if "agent_sdk_model" in data:
        model_val = data["agent_sdk_model"]
        if not isinstance(model_val, str):
            raise ConfigError(
                f"code_review.agent_sdk_model must be a string for trigger {trigger_name}, "
                f"got {type(model_val).__name__}"
            )
        if model_val not in ("sonnet", "opus", "haiku"):
            raise ConfigError(
                f"code_review.agent_sdk_model must be 'sonnet', 'opus', or 'haiku' "
                f"for trigger {trigger_name}, got '{model_val}'"
            )
        agent_sdk_model = model_val  # type: ignore[assignment]

    # Parse track_review_issues (optional, defaults to True)
    track_review_issues = True
    if "track_review_issues" in data:
        tri_val = data["track_review_issues"]
        if not isinstance(tri_val, bool):
            raise ConfigError(
                f"code_review.track_review_issues must be a boolean for trigger {trigger_name}, "
                f"got {type(tri_val).__name__}"
            )
        track_review_issues = tri_val

    return CodeReviewConfigClass(
        enabled=enabled,
        reviewer_type=reviewer_type,
        failure_mode=failure_mode,
        max_retries=max_retries,
        finding_threshold=finding_threshold,
        baseline=baseline,
        cerberus=cerberus,
        agent_sdk_timeout=agent_sdk_timeout,
        agent_sdk_model=agent_sdk_model,
        track_review_issues=track_review_issues,
    )


_EVIDENCE_CHECK_FIELDS = frozenset({"required"})


def _parse_evidence_check_config(
    data: dict[str, Any] | None,
) -> EvidenceCheckConfig | None:
    """Parse evidence_check configuration block.

    Args:
        data: The evidence_check config dict from mala.yaml.

    Returns:
        EvidenceCheckConfig if data is provided, None otherwise.

    Raises:
        ConfigError: If data has invalid types or unknown fields.
    """
    if data is None:
        return None

    if not isinstance(data, dict):
        raise ConfigError(
            f"evidence_check must be an object, got {type(data).__name__}"
        )

    # Validate unknown fields
    unknown = set(data.keys()) - _EVIDENCE_CHECK_FIELDS
    if unknown:
        first = sorted(str(k) for k in unknown)[0]
        raise ConfigError(f"Unknown field '{first}' in evidence_check")

    # Parse required (optional, defaults to empty tuple)
    required: tuple[str, ...] = ()
    if "required" in data:
        req_val = data["required"]
        if req_val is None:
            # null required → empty tuple
            required = ()
        elif not isinstance(req_val, list):
            raise ConfigError(
                f"evidence_check.required must be a list, got {type(req_val).__name__}"
            )
        else:
            # Validate each element is a string
            for i, item in enumerate(req_val):
                if not isinstance(item, str):
                    raise ConfigError(
                        f"evidence_check.required[{i}] must be a string, "
                        f"got {type(item).__name__}"
                    )
            required = tuple(req_val)

    return EvidenceCheckConfig(required=required)


_EPIC_COMPLETION_FIELDS = frozenset(
    {
        "failure_mode",
        "max_retries",
        "commands",
        "epic_depth",
        "fire_on",
        "code_review",
        "max_epic_verification_retries",
        "epic_verify_lock_timeout_seconds",
    }
)


def _parse_epic_completion_trigger(
    data: dict[str, Any],
) -> EpicCompletionTriggerConfig:
    """Parse epic_completion trigger config.

    Args:
        data: The epic_completion config dict.

    Returns:
        EpicCompletionTriggerConfig instance.

    Raises:
        ConfigError: If required fields missing, invalid, or unknown fields present.
    """
    trigger_name = "epic_completion"

    # Validate unknown fields
    unknown = set(data.keys()) - _EPIC_COMPLETION_FIELDS
    if unknown:
        first = sorted(str(k) for k in unknown)[0]
        raise ConfigError(f"Unknown field '{first}' in trigger {trigger_name}")

    failure_mode = _parse_failure_mode(data, trigger_name)
    max_retries = _parse_max_retries(data, trigger_name, failure_mode)
    commands = _parse_commands_list(data, trigger_name)

    # Parse epic_depth (required for epic_completion)
    if "epic_depth" not in data:
        raise ConfigError(f"epic_depth required for trigger {trigger_name}")
    depth_str = data["epic_depth"]
    if not isinstance(depth_str, str):
        raise ConfigError(
            f"epic_depth must be a string for trigger {trigger_name}, "
            f"got {type(depth_str).__name__}"
        )
    try:
        epic_depth = EpicDepth(depth_str)
    except ValueError:
        valid = ", ".join(d.value for d in EpicDepth)
        raise ConfigError(
            f"Invalid epic_depth '{depth_str}' for trigger {trigger_name}. "
            f"Valid values: {valid}"
        ) from None

    # Parse fire_on (required for epic_completion)
    if "fire_on" not in data:
        raise ConfigError(f"fire_on required for trigger {trigger_name}")
    fire_on_str = data["fire_on"]
    if not isinstance(fire_on_str, str):
        raise ConfigError(
            f"fire_on must be a string for trigger {trigger_name}, "
            f"got {type(fire_on_str).__name__}"
        )
    try:
        fire_on = FireOn(fire_on_str)
    except ValueError:
        valid = ", ".join(f.value for f in FireOn)
        raise ConfigError(
            f"Invalid fire_on '{fire_on_str}' for trigger {trigger_name}. "
            f"Valid values: {valid}"
        ) from None

    # Parse code_review (optional)
    code_review = None
    if "code_review" in data:
        code_review_data = data["code_review"]
        if code_review_data is not None:
            code_review = _parse_code_review_config(code_review_data, trigger_name)

    # Parse max_epic_verification_retries (optional)
    max_epic_verification_retries: int | None = None
    if "max_epic_verification_retries" in data:
        val = data["max_epic_verification_retries"]
        if val is not None:
            if not isinstance(val, int) or isinstance(val, bool):
                raise ConfigError(
                    f"max_epic_verification_retries must be an integer for trigger "
                    f"{trigger_name}, got {type(val).__name__}"
                )
            if val < 0:
                raise ConfigError(
                    f"max_epic_verification_retries must be non-negative for trigger "
                    f"{trigger_name}, got {val}"
                )
            max_epic_verification_retries = val

    # Parse epic_verify_lock_timeout_seconds (optional)
    epic_verify_lock_timeout_seconds: int | None = None
    if "epic_verify_lock_timeout_seconds" in data:
        val = data["epic_verify_lock_timeout_seconds"]
        if val is not None:
            if not isinstance(val, int) or isinstance(val, bool):
                raise ConfigError(
                    f"epic_verify_lock_timeout_seconds must be an integer for trigger "
                    f"{trigger_name}, got {type(val).__name__}"
                )
            if val < 0:
                raise ConfigError(
                    f"epic_verify_lock_timeout_seconds must be non-negative for trigger "
                    f"{trigger_name}, got {val}"
                )
            epic_verify_lock_timeout_seconds = val

    return EpicCompletionTriggerConfig(
        failure_mode=failure_mode,
        commands=commands,
        max_retries=max_retries,
        epic_depth=epic_depth,
        fire_on=fire_on,
        code_review=code_review,
        max_epic_verification_retries=max_epic_verification_retries,
        epic_verify_lock_timeout_seconds=epic_verify_lock_timeout_seconds,
    )


_SESSION_END_FIELDS = frozenset(
    {"failure_mode", "max_retries", "commands", "code_review"}
)


def _parse_session_end_trigger(data: dict[str, Any]) -> SessionEndTriggerConfig:
    """Parse session_end trigger config.

    Args:
        data: The session_end config dict.

    Returns:
        SessionEndTriggerConfig instance.

    Raises:
        ConfigError: If required fields missing, invalid, or unknown fields present.
    """
    trigger_name = "session_end"

    # Validate unknown fields
    unknown = set(data.keys()) - _SESSION_END_FIELDS
    if unknown:
        first = sorted(str(k) for k in unknown)[0]
        raise ConfigError(f"Unknown field '{first}' in trigger {trigger_name}")

    failure_mode = _parse_failure_mode(data, trigger_name)
    max_retries = _parse_max_retries(data, trigger_name, failure_mode)
    commands = _parse_commands_list(data, trigger_name)

    # Parse code_review (optional)
    code_review = None
    if "code_review" in data:
        code_review_data = data["code_review"]
        if code_review_data is not None:
            code_review = _parse_code_review_config(code_review_data, trigger_name)

    return SessionEndTriggerConfig(
        failure_mode=failure_mode,
        commands=commands,
        max_retries=max_retries,
        code_review=code_review,
    )


_PERIODIC_FIELDS = frozenset(
    {"failure_mode", "max_retries", "commands", "interval", "code_review"}
)


def _parse_periodic_trigger(data: dict[str, Any]) -> PeriodicTriggerConfig:
    """Parse periodic trigger config.

    Args:
        data: The periodic config dict.

    Returns:
        PeriodicTriggerConfig instance.

    Raises:
        ConfigError: If required fields missing, invalid, or unknown fields present.
    """
    trigger_name = "periodic"

    # Validate unknown fields
    unknown = set(data.keys()) - _PERIODIC_FIELDS
    if unknown:
        first = sorted(str(k) for k in unknown)[0]
        raise ConfigError(f"Unknown field '{first}' in trigger {trigger_name}")

    failure_mode = _parse_failure_mode(data, trigger_name)
    max_retries = _parse_max_retries(data, trigger_name, failure_mode)
    commands = _parse_commands_list(data, trigger_name)

    # Parse interval (required for periodic)
    if "interval" not in data:
        raise ConfigError(f"interval required for trigger {trigger_name}")
    interval = data["interval"]
    if isinstance(interval, bool) or not isinstance(interval, int):
        raise ConfigError(
            f"interval must be an integer for trigger {trigger_name}, "
            f"got {type(interval).__name__}"
        )
    if interval < 1:
        raise ConfigError(
            f"interval must be >= 1 for trigger {trigger_name}, got {interval}"
        )

    # Parse optional code_review config
    code_review = None
    if "code_review" in data:
        code_review_data = data["code_review"]
        if code_review_data is not None:
            code_review = _parse_code_review_config(code_review_data, trigger_name)

    return PeriodicTriggerConfig(
        failure_mode=failure_mode,
        commands=commands,
        max_retries=max_retries,
        interval=interval,
        code_review=code_review,
    )


_RUN_END_FIELDS = frozenset(
    {"failure_mode", "max_retries", "commands", "fire_on", "code_review"}
)


def _parse_run_end_trigger(data: dict[str, Any]) -> RunEndTriggerConfig:
    """Parse run_end trigger config.

    Args:
        data: The run_end config dict.

    Returns:
        RunEndTriggerConfig instance.

    Raises:
        ConfigError: If required fields missing, invalid, or unknown fields present.
    """
    trigger_name = "run_end"

    # Validate unknown fields
    unknown = set(data.keys()) - _RUN_END_FIELDS
    if unknown:
        first = sorted(str(k) for k in unknown)[0]
        raise ConfigError(f"Unknown field '{first}' in trigger {trigger_name}")

    failure_mode = _parse_failure_mode(data, trigger_name)
    max_retries = _parse_max_retries(data, trigger_name, failure_mode)
    commands = _parse_commands_list(data, trigger_name)

    # Parse fire_on (optional, defaults to success)
    fire_on = FireOn.SUCCESS
    if "fire_on" in data:
        fire_on_str = data["fire_on"]
        if not isinstance(fire_on_str, str):
            raise ConfigError(
                f"fire_on must be a string for trigger {trigger_name}, "
                f"got {type(fire_on_str).__name__}"
            )
        try:
            fire_on = FireOn(fire_on_str)
        except ValueError:
            valid = ", ".join(f.value for f in FireOn)
            raise ConfigError(
                f"Invalid fire_on '{fire_on_str}' for trigger {trigger_name}. "
                f"Valid values: {valid}"
            ) from None

    # Parse code_review (optional)
    code_review = None
    if "code_review" in data:
        code_review_data = data["code_review"]
        if code_review_data is not None:
            code_review = _parse_code_review_config(code_review_data, trigger_name)

    return RunEndTriggerConfig(
        failure_mode=failure_mode,
        commands=commands,
        max_retries=max_retries,
        fire_on=fire_on,
        code_review=code_review,
    )


_VALIDATION_TRIGGERS_FIELDS = frozenset(
    {"epic_completion", "session_end", "periodic", "run_end"}
)


def _parse_validation_triggers(
    data: dict[str, Any] | None,
) -> ValidationTriggersConfig | None:
    """Parse the validation_triggers section from mala.yaml.

    Args:
        data: The validation_triggers dict from the YAML.

    Returns:
        ValidationTriggersConfig if data is provided, None otherwise.

    Raises:
        ConfigError: If data is not a dict, has unknown keys, or any trigger is invalid.
    """
    if data is None:
        return None

    if not isinstance(data, dict):
        raise ConfigError(
            f"validation_triggers must be an object, got {type(data).__name__}"
        )

    # Validate unknown fields
    unknown = set(data.keys()) - _VALIDATION_TRIGGERS_FIELDS
    if unknown:
        first = sorted(str(k) for k in unknown)[0]
        raise ConfigError(f"Unknown trigger '{first}' in validation_triggers")

    epic_completion = None
    session_end = None
    periodic = None

    if "epic_completion" in data:
        epic_data = data["epic_completion"]
        if epic_data is not None:
            if not isinstance(epic_data, dict):
                raise ConfigError(
                    f"epic_completion must be an object, got {type(epic_data).__name__}"
                )
            epic_completion = _parse_epic_completion_trigger(epic_data)

    if "session_end" in data:
        session_data = data["session_end"]
        if session_data is not None:
            if not isinstance(session_data, dict):
                raise ConfigError(
                    f"session_end must be an object, got {type(session_data).__name__}"
                )
            session_end = _parse_session_end_trigger(session_data)

    if "periodic" in data:
        periodic_data = data["periodic"]
        if periodic_data is not None:
            if not isinstance(periodic_data, dict):
                raise ConfigError(
                    f"periodic must be an object, got {type(periodic_data).__name__}"
                )
            periodic = _parse_periodic_trigger(periodic_data)

    run_end = None
    if "run_end" in data:
        run_end_data = data["run_end"]
        if run_end_data is not None:
            if not isinstance(run_end_data, dict):
                raise ConfigError(
                    f"run_end must be an object, got {type(run_end_data).__name__}"
                )
            run_end = _parse_run_end_trigger(run_end_data)

    return ValidationTriggersConfig(
        epic_completion=epic_completion,
        session_end=session_end,
        periodic=periodic,
        run_end=run_end,
    )


def _build_config(data: dict[str, Any]) -> ValidationConfig:
    """Convert a validated YAML dict to a ValidationConfig dataclass.

    This function delegates to ValidationConfig.from_dict which handles
    parsing of nested structures (commands, coverage, validation_triggers, etc.).

    Args:
        data: Validated YAML dictionary.

    Returns:
        ValidationConfig instance.

    Raises:
        ConfigError: If any field has an invalid type or value.
    """
    return ValidationConfig.from_dict(data)


def _validate_config(config: ValidationConfig) -> None:
    """Perform post-build validation on the configuration.

    This validates semantic constraints that can't be checked during
    parsing, such as ensuring at least one command is defined (when
    no preset is specified).

    Args:
        config: Built ValidationConfig instance.

    Raises:
        ConfigError: If configuration is semantically invalid.
    """
    # If no preset is specified, at least one command must be defined
    if config.preset is None and not config.has_any_command():
        raise ConfigError(
            "At least one command must be defined. "
            "Specify a preset or define commands directly."
        )


def _validate_trigger_command_refs(config: ValidationConfig) -> None:
    """Validate that all trigger command refs exist in the effective base pool.

    The base pool is constructed from the merged config commands,
    following the same logic as run_coordinator._build_base_pool. This validation ensures
    invalid refs are caught at startup rather than at runtime when triggers fire.

    Args:
        config: The effective merged ValidationConfig (after preset merge).

    Raises:
        ConfigError: If any trigger references a command that doesn't exist in the base pool.
    """
    from src.domain.validation.config import BUILTIN_COMMAND_NAMES, TriggerType

    triggers = config.validation_triggers
    if triggers is None:
        return  # No triggers configured

    # Build base pool following run_coordinator._build_base_pool logic:
    # - Built-in commands: from commands
    # - Custom commands: from commands.custom_commands
    base_pool: set[str] = set()
    base_cmds = config.commands

    # Add built-in commands
    for cmd_name in BUILTIN_COMMAND_NAMES:
        base_cmd = getattr(base_cmds, cmd_name, None)
        if base_cmd is not None:
            base_pool.add(cmd_name)

    # Add custom commands from commands section
    for name in base_cmds.custom_commands:
        base_pool.add(name)

    # Validate each configured trigger's command refs
    trigger_configs: list[tuple[TriggerType, BaseTriggerConfig | None]] = [
        (TriggerType.EPIC_COMPLETION, triggers.epic_completion),
        (TriggerType.SESSION_END, triggers.session_end),
        (TriggerType.PERIODIC, triggers.periodic),
        (TriggerType.RUN_END, triggers.run_end),
    ]

    for trigger_type, maybe_config in trigger_configs:
        if maybe_config is None:
            continue
        trigger_config: BaseTriggerConfig = maybe_config
        for cmd_ref in trigger_config.commands:
            if cmd_ref.ref not in base_pool:
                available = ", ".join(sorted(base_pool)) if base_pool else "(none)"
                raise ConfigError(
                    f"trigger {trigger_type.value} references unknown command "
                    f"'{cmd_ref.ref}'. Available: {available}"
                )


def validate_generated_config(data: dict[str, Any]) -> None:
    """Validate a dictionary against the config schema and semantic rules.

    This is a convenience wrapper for validating programmatically-generated
    config data (e.g., from `mala init`).

    Args:
        data: Dictionary containing mala.yaml configuration.

    Raises:
        ConfigError: If schema validation or semantic validation fails.
    """
    _validate_schema(data)
    config = ValidationConfig.from_dict(data)  # raises ConfigError
    _validate_config(config)  # raises ConfigError


def dump_config_yaml(data: dict[str, Any]) -> str:
    """Dump config data to YAML string.

    Args:
        data: Dictionary containing mala.yaml configuration.

    Returns:
        YAML-formatted string.
    """
    return yaml.dump(data, default_flow_style=False, sort_keys=False)
