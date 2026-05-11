"""Section-level parser for mala.yaml validation configuration.

This module hosts the per-section parsing logic that ``ValidationConfig.from_dict``
currently inlines. It exposes one function per top-level YAML key plus an aggregate
``parse_validation_config`` so that:

- Each section can be unit-tested in isolation.
- ``ValidationConfig`` can later (T_C2) become a pure dataclass whose
  ``from_dict`` simply delegates to ``parse_validation_config``.

Section parsers return ``(value, was_present)`` so callers can rebuild the
``_fields_set`` tracking used for preset/CLI override merging.

This module imports only from ``config_types`` (pure dataclass definitions);
it never imports from ``config``. Parser helpers (``_parse_coder_options`` and
the inner ``_parse_codex_options_block``) live here directly rather than in
``config_types`` so the type module stays free of parsing logic. For the
heavyweight sections (validation_triggers / evidence_check / epic_verification /
per_issue_review) it currently delegates to private helpers in
``config_loader.py``; T_C2 will move those into this module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from src.core.constants import (
    VALID_CLAUDE_SETTINGS_SOURCES,
    VALID_CODEX_APPROVAL_POLICIES,
    VALID_CODEX_SANDBOXES,
    validate_codex_effort,
)
from src.domain.validation.config_types import (
    CodeReviewConfig,
    CodexOptionsConfig,
    CommandsConfig,
    ConfigError,
    EpicVerifierConfig,
    ValidationConfig,
    YamlCoverageConfig,
)

if TYPE_CHECKING:
    from src.domain.validation.config_types import (
        EvidenceCheckConfig,
        ValidationTriggersConfig,
    )


_CODER_OPTIONS_KEYS: frozenset[str] = frozenset({"codex"})
_CODEX_OPTIONS_KEYS: frozenset[str] = frozenset(
    {"model", "effort", "approval_policy", "sandbox", "mcp_servers"}
)


def _parse_codex_options_block(raw: object) -> CodexOptionsConfig:
    """Parse the inner ``coder_options.codex`` block.

    Strict-validates ``model``, ``effort``, ``approval_policy``, ``sandbox``
    and (shape-only) ``mcp_servers``. Raises :class:`ConfigError` on any
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

    model: str | None = None
    if "model" in value:
        model_val = value["model"]
        if model_val is not None:
            if not isinstance(model_val, str) or not model_val.strip():
                raise ConfigError(
                    "coder_options.codex.model must be a non-empty string"
                )
            model = model_val

    effort: str | None = None
    if "effort" in value:
        effort_val = value["effort"]
        if effort_val is not None:
            if not isinstance(effort_val, str):
                raise ConfigError(
                    "coder_options.codex.effort must be a string, got "
                    f"{type(effort_val).__name__}"
                )
            try:
                validate_codex_effort(effort_val, source="coder_options.codex.effort")
            except ValueError as exc:
                raise ConfigError(str(exc)) from exc
            effort = effort_val

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
        model=model,
        effort=effort,
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
    values (matching the existing ``ValidationConfig.from_dict`` message).
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
    """Parse the ``validation_triggers`` section.

    Delegates to ``config_loader._parse_validation_triggers`` for the actual
    trigger-tree parsing (T_C2 will fold that helper into this module).
    """
    if "validation_triggers" not in data:
        return None, False
    triggers_data = data["validation_triggers"]
    if triggers_data is None:
        return None, True
    if not isinstance(triggers_data, dict):
        raise ConfigError(
            f"validation_triggers must be an object, got {type(triggers_data).__name__}"
        )
    from src.domain.validation.config_loader import _parse_validation_triggers

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
    """Parse the ``evidence_check`` section.

    Delegates to ``config_loader._parse_evidence_check_config`` (T_C2 will
    fold that helper here).
    """
    if "evidence_check" not in data:
        return None, False
    ec_data = data["evidence_check"]
    if ec_data is None:
        return None, True
    if not isinstance(ec_data, dict):
        raise ConfigError(
            f"evidence_check must be an object, got {type(ec_data).__name__}"
        )
    from src.domain.validation.config_loader import _parse_evidence_check_config

    return (
        _parse_evidence_check_config(cast("dict[str, object]", ec_data)),
        True,
    )


def parse_epic_verification(
    data: dict[str, Any],
) -> tuple[EpicVerifierConfig | None, bool]:
    """Parse the ``epic_verification`` section.

    Delegates to ``config_loader._parse_epic_verification_config`` (T_C2 will
    fold that helper here).
    """
    if "epic_verification" not in data:
        return None, False
    ev_data = data["epic_verification"]
    if ev_data is None:
        return None, True
    if not isinstance(ev_data, dict):
        raise ConfigError(
            f"epic_verification must be an object, got {type(ev_data).__name__}"
        )
    from src.domain.validation.config_loader import _parse_epic_verification_config

    return (
        _parse_epic_verification_config(cast("dict[str, object]", ev_data)),
        True,
    )


def parse_per_issue_review(
    data: dict[str, Any],
) -> tuple[CodeReviewConfig | None, bool]:
    """Parse the ``per_issue_review`` section.

    Delegates to ``config_loader._parse_code_review_config`` (T_C2 will fold
    that helper here).
    """
    if "per_issue_review" not in data:
        return None, False
    pir_data = data["per_issue_review"]
    if pir_data is None:
        return None, True
    if not isinstance(pir_data, dict):
        raise ConfigError(
            f"per_issue_review must be an object, got {type(pir_data).__name__}"
        )
    from src.domain.validation.config_loader import _parse_code_review_config

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

    Mirrors the cross-section check previously inlined in
    ``ValidationConfig.from_dict``. Default ``amp_mode`` is ``deep`` when unset.
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
        effort=effort,
        codex_options=codex_options,
        _fields_set=frozenset(fields_set),
    )
