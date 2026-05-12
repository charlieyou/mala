"""Section-level parser for mala.yaml validation configuration.

This module owns all per-section parsing logic for the validation config. It
exposes one function per top-level YAML key plus an aggregate
``parse_validation_config`` so that:

- Each section can be unit-tested in isolation.
- ``ValidationConfig`` is a pure dataclass; its ``from_dict`` is a thin
  delegate to ``parse_validation_config``.

Section parsers return ``(value, was_present)`` so callers can rebuild the
``_fields_set`` tracking used for preset/CLI override merging.

This module imports only from ``config_types`` (pure dataclass definitions);
it never imports from ``config_loader``. All parser helpers — including those
for the heavyweight sections (validation_triggers / evidence_check /
epic_verification / per_issue_review) — live here directly.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, cast

from src.core.constants import (
    DEFAULT_AGENT_SDK_REVIEW_TIMEOUT_SECONDS,
    DEFAULT_CERBERUS_REVIEW_TIMEOUT_SECONDS,
    VALID_CLAUDE_SETTINGS_SOURCES,
    VALID_CODEX_APPROVAL_POLICIES,
    VALID_CODEX_SANDBOXES,
)
from src.domain.validation.config_types import (
    CerberusConfig,
    CodeReviewConfig,
    CodexOptionsConfig,
    CommandsConfig,
    ConfigError,
    EpicCompletionTriggerConfig,
    EpicDepth,
    EpicVerifierConfig,
    EvidenceCheckConfig,
    FailureMode,
    FireOn,
    PeriodicTriggerConfig,
    RunEndTriggerConfig,
    SessionEndTriggerConfig,
    TriggerCommandRef,
    ValidationConfig,
    ValidationTriggersConfig,
    VerificationRetryPolicy,
    YamlCoverageConfig,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


_CODER_OPTIONS_KEYS: frozenset[str] = frozenset({"codex"})
_CODEX_OPTIONS_KEYS: frozenset[str] = frozenset(
    {"approval_policy", "sandbox", "mcp_servers"}
)


def _parse_codex_options_block(raw: object) -> CodexOptionsConfig:
    """Parse the inner ``coder_options.codex`` block.

    Strict-validates ``approval_policy``, ``sandbox`` and shape-validates
    ``mcp_servers``. Raises :class:`ConfigError` on any
    unknown key, wrong type, or invalid enum value (AC #13).
    """
    if raw is None:
        return CodexOptionsConfig()
    if not isinstance(raw, dict):
        raise ConfigError(
            f"coder_options.codex must be an object, got {type(raw).__name__}"
        )
    value = cast("dict[str, object]", raw)
    unknown = set(value.keys()) - _CODEX_OPTIONS_KEYS
    if unknown:
        first = sorted(str(k) for k in unknown)[0]
        raise ConfigError(f"Unknown field '{first}' in coder_options.codex")

    approval_policy: (
        Literal["never", "on-request", "on-failure", "untrusted"] | None
    ) = None
    if "approval_policy" in value:
        ap_val = value["approval_policy"]
        if ap_val is not None:
            if (
                not isinstance(ap_val, str)
                or ap_val not in VALID_CODEX_APPROVAL_POLICIES
            ):
                valid = ", ".join(sorted(VALID_CODEX_APPROVAL_POLICIES))
                raise ConfigError(
                    "coder_options.codex.approval_policy must be one of "
                    f"{valid}, got {ap_val!r}"
                )
            # Narrow Literal type for the type checker.
            if ap_val == "on-request":
                approval_policy = "on-request"
            elif ap_val == "on-failure":
                approval_policy = "on-failure"
            elif ap_val == "untrusted":
                approval_policy = "untrusted"
            else:
                approval_policy = "never"

    sandbox: Literal["read-only", "workspace-write", "danger-full-access"] | None = None
    if "sandbox" in value:
        sb_val = value["sandbox"]
        if sb_val is not None:
            if not isinstance(sb_val, str) or sb_val not in VALID_CODEX_SANDBOXES:
                valid = ", ".join(sorted(VALID_CODEX_SANDBOXES))
                raise ConfigError(
                    "coder_options.codex.sandbox must be one of "
                    f"{valid}, got {sb_val!r}"
                )
            if sb_val == "read-only":
                sandbox = "read-only"
            elif sb_val == "workspace-write":
                sandbox = "workspace-write"
            else:
                sandbox = "danger-full-access"

    mcp_servers: tuple[tuple[str, object], ...] = ()
    if "mcp_servers" in value:
        ms_val = value["mcp_servers"]
        if ms_val is not None:
            if not isinstance(ms_val, dict):
                raise ConfigError(
                    "coder_options.codex.mcp_servers must be an object, got "
                    f"{type(ms_val).__name__}"
                )
            ms_dict = cast("dict[str, object]", ms_val)
            entries: list[tuple[str, object]] = []
            for key in ms_dict:
                if not isinstance(key, str):
                    raise ConfigError(
                        "coder_options.codex.mcp_servers keys must be strings, "
                        f"got {type(key).__name__}"
                    )
                entries.append((key, ms_dict[key]))
            mcp_servers = tuple(entries)

    return CodexOptionsConfig(
        approval_policy=approval_policy,
        sandbox=sandbox,
        mcp_servers=mcp_servers,
    )


def _parse_coder_options(raw: object) -> CodexOptionsConfig | None:
    """Parse the top-level ``coder_options`` block.

    Only ``coder_options.codex`` is supported; ``coder_options.amp`` was
    flattened into top-level ``amp_mode`` and is rejected here so the
    legacy YAML shape fails fast (regression coverage in
    ``tests/unit/domain/validation/test_coder_schema.py``).
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ConfigError(f"coder_options must be an object, got {type(raw).__name__}")
    value = cast("dict[str, object]", raw)
    unknown = set(value.keys()) - _CODER_OPTIONS_KEYS
    if unknown:
        first = sorted(str(k) for k in unknown)[0]
        raise ConfigError(
            f"Unknown field '{first}' in coder_options. "
            "Only 'codex' is supported under coder_options; amp options live at "
            "top-level 'amp_mode'."
        )
    if "codex" not in value:
        return CodexOptionsConfig()
    return _parse_codex_options_block(value["codex"])


def parse_preset(data: dict[str, Any]) -> tuple[str | None, bool]:
    """Parse the ``preset`` key from a top-level mala.yaml dict.

    Returns ``(value, was_present)``. Raises :class:`ConfigError` for non-string
    values.
    """
    if "preset" not in data:
        return None, False
    preset = data["preset"]
    if preset is not None and not isinstance(preset, str):
        raise ConfigError(f"preset must be a string, got {type(preset).__name__}")
    return preset, True


def parse_commands(data: dict[str, Any]) -> tuple[CommandsConfig, bool]:
    """Parse the ``commands`` section into a :class:`CommandsConfig`.

    Absent or null ``commands`` returns an empty ``CommandsConfig``; the
    ``was_present`` flag tracks whether the key appeared in ``data`` so
    callers can populate ``_fields_set``.
    """
    if "commands" not in data:
        return CommandsConfig.from_dict(None), False
    commands_data = data["commands"]
    if commands_data is not None and not isinstance(commands_data, dict):
        raise ConfigError(
            f"commands must be an object, got {type(commands_data).__name__}"
        )
    return (
        CommandsConfig.from_dict(cast("dict[str, object] | None", commands_data)),
        True,
    )


def parse_coverage(data: dict[str, Any]) -> tuple[YamlCoverageConfig | None, bool]:
    """Parse the ``coverage`` section into a :class:`YamlCoverageConfig`.

    Null or absent values yield ``None``. Non-object values raise
    :class:`ConfigError`.
    """
    if "coverage" not in data:
        return None, False
    coverage_data = data["coverage"]
    if coverage_data is None:
        return None, True
    if not isinstance(coverage_data, dict):
        raise ConfigError(
            f"coverage must be an object, got {type(coverage_data).__name__}"
        )
    return (
        YamlCoverageConfig.from_dict(cast("dict[str, object]", coverage_data)),
        True,
    )


def parse_string_list_field(
    data: dict[str, Any], key: str
) -> tuple[tuple[str, ...], bool]:
    """Parse a top-level list-of-strings field (e.g. ``code_patterns``).

    Used for ``code_patterns``, ``config_files``, and ``setup_files``. Absent or
    null values yield ``()``; non-list values raise :class:`ConfigError`.
    """
    if key not in data:
        return (), False
    value = data[key]
    if value is None:
        return (), True
    if not isinstance(value, list):
        raise ConfigError(f"{key} must be a list, got {type(value).__name__}")
    result: list[str] = []
    for i, item in enumerate(value):
        if not isinstance(item, str):
            raise ConfigError(f"{key}[{i}] must be a string, got {type(item).__name__}")
        result.append(item)
    return tuple(result), True


def parse_claude_settings_sources(
    data: dict[str, Any],
) -> tuple[tuple[str, ...] | None, bool]:
    """Parse the ``claude_settings_sources`` list.

    Each entry is whitespace-stripped and validated against
    :data:`VALID_CLAUDE_SETTINGS_SOURCES`. ``None`` is preserved separately
    from absence so the merger can distinguish "use default" from "use empty".
    """
    if "claude_settings_sources" not in data:
        return None, False
    css_value = data["claude_settings_sources"]
    if css_value is None:
        return None, True
    if not isinstance(css_value, list):
        raise ConfigError(
            f"claude_settings_sources must be a list, got {type(css_value).__name__}"
        )
    sources: list[str] = []
    for i, item in enumerate(css_value):
        if not isinstance(item, str):
            raise ConfigError(
                f"claude_settings_sources[{i}] must be a string, "
                f"got {type(item).__name__}"
            )
        source = item.strip()
        if source not in VALID_CLAUDE_SETTINGS_SOURCES:
            valid_str = ", ".join(sorted(VALID_CLAUDE_SETTINGS_SOURCES))
            raise ConfigError(
                f"Invalid Claude settings source '{source}'. Valid sources: {valid_str}"
            )
        sources.append(source)
    return tuple(sources), True


def parse_validation_triggers(
    data: dict[str, Any],
) -> tuple[ValidationTriggersConfig | None, bool]:
    """Parse the ``validation_triggers`` section."""
    if "validation_triggers" not in data:
        return None, False
    triggers_data = data["validation_triggers"]
    if triggers_data is None:
        return None, True
    if not isinstance(triggers_data, dict):
        raise ConfigError(
            f"validation_triggers must be an object, got {type(triggers_data).__name__}"
        )
    return (
        _parse_validation_triggers(cast("dict[str, object]", triggers_data)),
        True,
    )


def _parse_int_field(
    data: dict[str, Any],
    key: str,
    *,
    allow_zero: bool,
) -> tuple[int | None, bool]:
    """Shared parser for ``timeout_minutes`` / ``max_idle_retries`` /
    ``max_diff_size_kb``.

    ``allow_zero``:
        - False → field must be strictly positive (``timeout_minutes``).
        - True  → field must be non-negative (``max_idle_retries``,
          ``max_diff_size_kb``).
    """
    if key not in data:
        return None, False
    value = data[key]
    if value is None:
        return None, True
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{key} must be an integer, got {type(value).__name__}")
    if allow_zero:
        if value < 0:
            raise ConfigError(f"{key} must be non-negative, got {value}")
    elif value <= 0:
        raise ConfigError(f"{key} must be positive, got {value}")
    return value, True


def parse_timeout_minutes(data: dict[str, Any]) -> tuple[int | None, bool]:
    """Parse ``timeout_minutes``: strictly positive integer or null."""
    return _parse_int_field(data, "timeout_minutes", allow_zero=False)


def parse_max_idle_retries(data: dict[str, Any]) -> tuple[int | None, bool]:
    """Parse ``max_idle_retries``: non-negative integer or null."""
    return _parse_int_field(data, "max_idle_retries", allow_zero=True)


def parse_max_diff_size_kb(data: dict[str, Any]) -> tuple[int | None, bool]:
    """Parse ``max_diff_size_kb``: non-negative integer or null."""
    return _parse_int_field(data, "max_diff_size_kb", allow_zero=True)


def parse_idle_timeout_seconds(
    data: dict[str, Any],
) -> tuple[float | None, bool]:
    """Parse ``idle_timeout_seconds``: non-negative number (int or float) or null."""
    if "idle_timeout_seconds" not in data:
        return None, False
    value = data["idle_timeout_seconds"]
    if value is None:
        return None, True
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(
            f"idle_timeout_seconds must be a number, got {type(value).__name__}"
        )
    if value < 0:
        raise ConfigError(f"idle_timeout_seconds must be non-negative, got {value}")
    return float(value), True


def parse_evidence_check(
    data: dict[str, Any],
) -> tuple[EvidenceCheckConfig | None, bool]:
    """Parse the ``evidence_check`` section."""
    if "evidence_check" not in data:
        return None, False
    ec_data = data["evidence_check"]
    if ec_data is None:
        return None, True
    if not isinstance(ec_data, dict):
        raise ConfigError(
            f"evidence_check must be an object, got {type(ec_data).__name__}"
        )
    return (
        _parse_evidence_check_config(cast("dict[str, object]", ec_data)),
        True,
    )


def parse_epic_verification(
    data: dict[str, Any],
) -> tuple[EpicVerifierConfig | None, bool]:
    """Parse the ``epic_verification`` section."""
    if "epic_verification" not in data:
        return None, False
    ev_data = data["epic_verification"]
    if ev_data is None:
        return None, True
    if not isinstance(ev_data, dict):
        raise ConfigError(
            f"epic_verification must be an object, got {type(ev_data).__name__}"
        )
    return (
        _parse_epic_verification_config(cast("dict[str, object]", ev_data)),
        True,
    )


def parse_per_issue_review(
    data: dict[str, Any],
) -> tuple[CodeReviewConfig | None, bool]:
    """Parse the ``per_issue_review`` section."""
    if "per_issue_review" not in data:
        return None, False
    pir_data = data["per_issue_review"]
    if pir_data is None:
        return None, True
    if not isinstance(pir_data, dict):
        raise ConfigError(
            f"per_issue_review must be an object, got {type(pir_data).__name__}"
        )
    return (
        _parse_code_review_config(
            cast("dict[str, object]", pir_data),
            "per_issue_review",
            is_per_issue_review=True,
        ),
        True,
    )


def parse_coder(
    data: dict[str, Any],
) -> tuple[Literal["claude", "amp", "codex"] | None, bool]:
    """Parse the top-level ``coder`` enum (``claude`` | ``amp`` | ``codex``)."""
    if "coder" not in data:
        return None, False
    coder_val = data["coder"]
    if coder_val is None:
        return None, True
    if not isinstance(coder_val, str) or coder_val not in ("claude", "amp", "codex"):
        raise ConfigError(
            f"coder must be 'claude', 'amp', or 'codex', got {coder_val!r}"
        )
    if coder_val == "amp":
        return "amp", True
    if coder_val == "codex":
        return "codex", True
    return "claude", True


def parse_amp_mode(
    data: dict[str, Any],
) -> tuple[Literal["smart", "rush", "deep"] | None, bool]:
    """Parse the top-level ``amp_mode`` enum (``smart`` | ``rush`` | ``deep``)."""
    if "amp_mode" not in data:
        return None, False
    mode_val = data["amp_mode"]
    if mode_val is None:
        return None, True
    if not isinstance(mode_val, str) or mode_val not in ("smart", "rush", "deep"):
        raise ConfigError(
            f"amp_mode must be 'smart', 'rush', or 'deep', got {mode_val!r}"
        )
    if mode_val == "rush":
        return "rush", True
    if mode_val == "deep":
        return "deep", True
    return "smart", True


def parse_effort(data: dict[str, Any]) -> tuple[str | None, bool]:
    """Parse the top-level ``effort`` enum (``low``..``max``)."""
    if "effort" not in data:
        return None, False
    effort_val = data["effort"]
    if effort_val is None:
        return None, True
    if not isinstance(effort_val, str) or effort_val not in (
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    ):
        raise ConfigError(
            "effort must be 'low', 'medium', 'high', 'xhigh', "
            f"or 'max', got {effort_val!r}"
        )
    return effort_val, True


def parse_model(data: dict[str, Any]) -> tuple[str | None, bool]:
    """Parse the top-level ``model`` option."""
    if "model" not in data:
        return None, False
    model_val = data["model"]
    if model_val is None:
        return None, True
    if not isinstance(model_val, str) or not model_val.strip():
        raise ConfigError(f"model must be a non-empty string, got {model_val!r}")
    return model_val.strip(), True


def parse_coder_options(
    data: dict[str, Any],
) -> tuple[CodexOptionsConfig | None, bool]:
    """Parse the ``coder_options`` block (only ``coder_options.codex`` allowed)."""
    if "coder_options" not in data:
        return None, False
    return _parse_coder_options(data["coder_options"]), True


def _validate_effort_compatibility(
    coder: Literal["claude", "amp", "codex"] | None,
    amp_mode: Literal["smart", "rush", "deep"] | None,
    effort: str | None,
) -> None:
    """Reject the ``coder=amp + amp_mode=deep + effort in {high, max}`` combo.

    Default ``amp_mode`` is ``deep`` when unset.
    """
    effective_amp_mode = amp_mode if amp_mode is not None else "deep"
    if coder == "amp" and effective_amp_mode == "deep" and effort in {"high", "max"}:
        raise ConfigError(
            "effort must be 'low', 'medium', or 'xhigh' when "
            f"coder is 'amp' and amp_mode is 'deep', got {effort!r}"
        )


def parse_validation_config(data: dict[str, Any]) -> ValidationConfig:
    """Parse a top-level mala.yaml dict into a :class:`ValidationConfig`.

    Aggregates every section parser, threads ``_fields_set`` tracking from the
    ``was_present`` flags, and runs the coder/amp_mode/effort cross-section
    validation.
    """
    fields_set: set[str] = set()

    preset, present = parse_preset(data)
    if present:
        fields_set.add("preset")

    commands, present = parse_commands(data)
    if present:
        fields_set.add("commands")

    coverage, present = parse_coverage(data)
    if present:
        fields_set.add("coverage")

    code_patterns, present = parse_string_list_field(data, "code_patterns")
    if present:
        fields_set.add("code_patterns")

    config_files, present = parse_string_list_field(data, "config_files")
    if present:
        fields_set.add("config_files")

    setup_files, present = parse_string_list_field(data, "setup_files")
    if present:
        fields_set.add("setup_files")

    claude_settings_sources, present = parse_claude_settings_sources(data)
    if present:
        fields_set.add("claude_settings_sources")

    validation_triggers, present = parse_validation_triggers(data)
    if present:
        fields_set.add("validation_triggers")

    timeout_minutes, present = parse_timeout_minutes(data)
    if present:
        fields_set.add("timeout_minutes")

    max_idle_retries, present = parse_max_idle_retries(data)
    if present:
        fields_set.add("max_idle_retries")

    idle_timeout_seconds, present = parse_idle_timeout_seconds(data)
    if present:
        fields_set.add("idle_timeout_seconds")

    max_diff_size_kb, present = parse_max_diff_size_kb(data)
    if present:
        fields_set.add("max_diff_size_kb")

    evidence_check, present = parse_evidence_check(data)
    if present:
        fields_set.add("evidence_check")

    epic_verification, present = parse_epic_verification(data)
    if present:
        fields_set.add("epic_verification")

    per_issue_review, present = parse_per_issue_review(data)
    if present:
        fields_set.add("per_issue_review")

    coder, present = parse_coder(data)
    if present:
        fields_set.add("coder")

    amp_mode, present = parse_amp_mode(data)
    if present:
        fields_set.add("amp_mode")

    model, present = parse_model(data)
    if present:
        fields_set.add("model")

    effort, present = parse_effort(data)
    if present:
        fields_set.add("effort")

    _validate_effort_compatibility(coder, amp_mode, effort)

    codex_options, present = parse_coder_options(data)
    if present:
        fields_set.add("coder_options")

    return ValidationConfig(
        preset=preset,
        commands=commands,
        coverage=coverage,
        code_patterns=code_patterns,
        config_files=config_files,
        setup_files=setup_files,
        claude_settings_sources=claude_settings_sources,
        validation_triggers=validation_triggers,
        timeout_minutes=timeout_minutes,
        max_idle_retries=max_idle_retries,
        idle_timeout_seconds=idle_timeout_seconds,
        max_diff_size_kb=max_diff_size_kb,
        evidence_check=evidence_check,
        epic_verification=epic_verification
        if epic_verification is not None
        else EpicVerifierConfig(),
        per_issue_review=per_issue_review
        if per_issue_review is not None
        else CodeReviewConfig(enabled=False),
        coder=coder,
        amp_mode=amp_mode,
        model=model,
        effort=effort,
        codex_options=codex_options,
        _fields_set=frozenset(fields_set),
    )


# ---------------------------------------------------------------------------
# Heavyweight-section helpers (validation_triggers / evidence_check /
# epic_verification / per_issue_review). Moved here from config_loader.py so
# the parser owns all parsing logic and ValidationConfig stays a pure
# dataclass boundary.
# ---------------------------------------------------------------------------


_TRIGGER_COMMAND_REF_FIELDS = frozenset({"ref", "command", "timeout"})


def _parse_trigger_command_ref(
    data: Mapping[object, object], trigger_name: str, cmd_index: int
) -> TriggerCommandRef:
    """Parse a single command reference from a trigger's commands list."""
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
    """Parse the commands list for a trigger."""
    commands_data = data.get("commands", [])
    if not isinstance(commands_data, list):
        raise ConfigError(
            f"'commands' must be a list for trigger {trigger_name}, "
            f"got {type(commands_data).__name__}"
        )

    commands: list[TriggerCommandRef] = []
    for i, cmd_data in enumerate(commands_data):
        if isinstance(cmd_data, str):
            if not cmd_data.strip():
                raise ConfigError(
                    f"Command {i} in trigger {trigger_name} cannot be an empty string"
                )
            commands.append(TriggerCommandRef(ref=cmd_data))
        elif isinstance(cmd_data, dict):
            command_data = cast("Mapping[object, object]", cmd_data)
            commands.append(_parse_trigger_command_ref(command_data, trigger_name, i))
        else:
            raise ConfigError(
                f"Command {i} in trigger {trigger_name} must be a string or object, "
                f"got {type(cmd_data).__name__}"
            )

    return tuple(commands)


def _parse_failure_mode(data: dict[str, Any], trigger_name: str) -> FailureMode:
    """Parse and validate failure_mode for a trigger."""
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
    """Parse and validate max_retries for a trigger."""
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
    """Parse cerberus-specific configuration block."""
    unknown = set(data.keys()) - _CERBERUS_FIELDS
    if unknown:
        first = sorted(str(k) for k in unknown)[0]
        raise ConfigError(f"Unknown field '{first}' in code_review.cerberus")

    timeout = DEFAULT_CERBERUS_REVIEW_TIMEOUT_SECONDS
    if "timeout" in data:
        timeout_val = data["timeout"]
        if isinstance(timeout_val, bool) or not isinstance(timeout_val, int):
            raise ConfigError(
                f"cerberus.timeout must be an integer, got {type(timeout_val).__name__}"
            )
        timeout = timeout_val

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

    return CerberusConfig(
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
    """Parse retry_policy configuration block."""
    if data is None:
        return VerificationRetryPolicy()

    if not isinstance(data, dict):
        raise ConfigError(
            f"epic_verification.retry_policy must be an object, "
            f"got {type(data).__name__}"
        )

    unknown = set(data.keys()) - _RETRY_POLICY_FIELDS
    if unknown:
        first = sorted(str(k) for k in unknown)[0]
        raise ConfigError(f"Unknown field '{first}' in epic_verification.retry_policy")

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

    return VerificationRetryPolicy(
        timeout_retries=timeout_retries,
        execution_retries=execution_retries,
        parse_retries=parse_retries,
    )


def _parse_epic_verification_config(
    data: dict[str, Any] | None,
) -> EpicVerifierConfig:
    """Parse epic_verification configuration block."""
    if data is None:
        return EpicVerifierConfig()

    if not isinstance(data, dict):
        raise ConfigError(
            f"epic_verification must be an object, got {type(data).__name__}"
        )

    unknown = set(data.keys()) - _EPIC_VERIFICATION_FIELDS
    if unknown:
        first = sorted(str(k) for k in unknown)[0]
        raise ConfigError(f"Unknown field '{first}' in epic_verification")

    enabled = True
    if "enabled" in data:
        enabled_val = data["enabled"]
        if not isinstance(enabled_val, bool):
            raise ConfigError(
                f"epic_verification.enabled must be a boolean, "
                f"got {type(enabled_val).__name__}"
            )
        enabled = enabled_val

    reviewer_type: Literal["cerberus", "agent_sdk"] = "agent_sdk"
    if "reviewer_type" in data:
        val = data["reviewer_type"]
        if val not in ("cerberus", "agent_sdk"):
            raise ConfigError(
                f"epic_verification.reviewer_type must be 'cerberus' or 'agent_sdk', "
                f"got '{val}'"
            )
        reviewer_type = val

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

    agent_sdk_timeout = DEFAULT_AGENT_SDK_REVIEW_TIMEOUT_SECONDS
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

    agent_sdk_model: Literal["sonnet", "opus", "haiku"] = "opus"
    if "agent_sdk_model" in data:
        model_val = data["agent_sdk_model"]
        if model_val not in ("sonnet", "opus", "haiku"):
            raise ConfigError(
                f"epic_verification.agent_sdk_model must be 'sonnet', 'opus', or 'haiku', "
                f"got '{model_val}'"
            )
        agent_sdk_model = model_val

    retry_policy = _parse_retry_policy(data.get("retry_policy"))

    return EpicVerifierConfig(
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
    data: dict[str, Any] | None,
    trigger_name: str,
    *,
    is_per_issue_review: bool = False,
) -> CodeReviewConfig | None:
    """Parse code_review configuration block."""
    logger = logging.getLogger(__name__)

    if data is None:
        return None

    if not isinstance(data, dict):
        raise ConfigError(
            f"code_review must be an object for trigger {trigger_name}, "
            f"got {type(data).__name__}"
        )

    unknown = set(data.keys()) - _CODE_REVIEW_FIELDS
    if unknown:
        first = sorted(str(k) for k in unknown)[0]
        raise ConfigError(f"Unknown field '{first}' in code_review for {trigger_name}")

    enabled = False
    if "enabled" in data:
        enabled_val = data["enabled"]
        if not isinstance(enabled_val, bool):
            raise ConfigError(
                f"code_review.enabled must be a boolean for trigger {trigger_name}, "
                f"got {type(enabled_val).__name__}"
            )
        enabled = enabled_val

    reviewer_type: Literal["cerberus", "agent_sdk"] = "cerberus"
    if "reviewer_type" in data:
        rt_val = data["reviewer_type"]
        if not isinstance(rt_val, str):
            raise ConfigError(
                f"code_review.reviewer_type must be a string for trigger {trigger_name}, "
                f"got {type(rt_val).__name__}"
            )
        if rt_val not in ("cerberus", "agent_sdk"):
            if enabled:
                raise ConfigError(
                    f"code_review.reviewer_type must be 'cerberus' or 'agent_sdk' "
                    f"for trigger {trigger_name}, got '{rt_val}'"
                )
        reviewer_type = cast("Literal['cerberus', 'agent_sdk']", rt_val)

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

    max_retries = 3
    if "max_retries" in data:
        mr_val = data["max_retries"]
        if isinstance(mr_val, bool) or not isinstance(mr_val, int):
            raise ConfigError(
                f"code_review.max_retries must be an integer for trigger {trigger_name}, "
                f"got {type(mr_val).__name__}"
            )
        if mr_val < 0:
            raise ConfigError(
                f"code_review.max_retries must be >= 0 for trigger {trigger_name}, "
                f"got {mr_val}"
            )
        max_retries = mr_val

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
        finding_threshold = cast("Literal['P0', 'P1', 'P2', 'P3', 'none']", ft_val)

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
            if is_per_issue_review:
                logger.warning(
                    "baseline is not applicable for per_issue_review; "
                    "ignoring baseline='%s'",
                    bl_val,
                )
                baseline = None
            elif trigger_name == "session_end":
                logger.warning(
                    "code_review.baseline is not applicable for session_end trigger; "
                    "ignoring baseline='%s'",
                    bl_val,
                )
                baseline = None
            else:
                baseline = cast(
                    "Literal['since_run_start', 'since_last_review']", bl_val
                )

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

    agent_sdk_timeout = DEFAULT_AGENT_SDK_REVIEW_TIMEOUT_SECONDS
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

    agent_sdk_model: Literal["sonnet", "opus", "haiku"] = "opus"
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
        agent_sdk_model = cast("Literal['sonnet', 'opus', 'haiku']", model_val)

    track_review_issues = True
    if "track_review_issues" in data:
        tri_val = data["track_review_issues"]
        if not isinstance(tri_val, bool):
            raise ConfigError(
                f"code_review.track_review_issues must be a boolean for trigger {trigger_name}, "
                f"got {type(tri_val).__name__}"
            )
        track_review_issues = tri_val

    return CodeReviewConfig(
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
    """Parse evidence_check configuration block."""
    if data is None:
        return None

    if not isinstance(data, dict):
        raise ConfigError(
            f"evidence_check must be an object, got {type(data).__name__}"
        )

    unknown = set(data.keys()) - _EVIDENCE_CHECK_FIELDS
    if unknown:
        first = sorted(str(k) for k in unknown)[0]
        raise ConfigError(f"Unknown field '{first}' in evidence_check")

    required: tuple[str, ...] = ()
    if "required" in data:
        req_val = data["required"]
        if req_val is None:
            required = ()
        elif not isinstance(req_val, list):
            raise ConfigError(
                f"evidence_check.required must be a list, got {type(req_val).__name__}"
            )
        else:
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
    """Parse epic_completion trigger config."""
    trigger_name = "epic_completion"

    unknown = set(data.keys()) - _EPIC_COMPLETION_FIELDS
    if unknown:
        first = sorted(str(k) for k in unknown)[0]
        raise ConfigError(f"Unknown field '{first}' in trigger {trigger_name}")

    failure_mode = _parse_failure_mode(data, trigger_name)
    max_retries = _parse_max_retries(data, trigger_name, failure_mode)
    commands = _parse_commands_list(data, trigger_name)

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

    code_review = None
    if "code_review" in data:
        code_review_data = data["code_review"]
        if code_review_data is not None:
            code_review = _parse_code_review_config(code_review_data, trigger_name)

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
    """Parse session_end trigger config."""
    trigger_name = "session_end"

    unknown = set(data.keys()) - _SESSION_END_FIELDS
    if unknown:
        first = sorted(str(k) for k in unknown)[0]
        raise ConfigError(f"Unknown field '{first}' in trigger {trigger_name}")

    failure_mode = _parse_failure_mode(data, trigger_name)
    max_retries = _parse_max_retries(data, trigger_name, failure_mode)
    commands = _parse_commands_list(data, trigger_name)

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
    """Parse periodic trigger config."""
    trigger_name = "periodic"

    unknown = set(data.keys()) - _PERIODIC_FIELDS
    if unknown:
        first = sorted(str(k) for k in unknown)[0]
        raise ConfigError(f"Unknown field '{first}' in trigger {trigger_name}")

    failure_mode = _parse_failure_mode(data, trigger_name)
    max_retries = _parse_max_retries(data, trigger_name, failure_mode)
    commands = _parse_commands_list(data, trigger_name)

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
    """Parse run_end trigger config."""
    trigger_name = "run_end"

    unknown = set(data.keys()) - _RUN_END_FIELDS
    if unknown:
        first = sorted(str(k) for k in unknown)[0]
        raise ConfigError(f"Unknown field '{first}' in trigger {trigger_name}")

    failure_mode = _parse_failure_mode(data, trigger_name)
    max_retries = _parse_max_retries(data, trigger_name, failure_mode)
    commands = _parse_commands_list(data, trigger_name)

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
    """Parse the validation_triggers section from mala.yaml."""
    if data is None:
        return None

    if not isinstance(data, dict):
        raise ConfigError(
            f"validation_triggers must be an object, got {type(data).__name__}"
        )

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
