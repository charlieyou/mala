"""Init config generation helpers.

Pure helpers for ``mala init`` YAML generation. The CLI owns the prompt
backend (questionary) and the final file I/O; this module owns the
state machine that drives the workflow: preset selection, evidence /
trigger / per-issue-review section construction, and validation
retry/recovery. Prompts are invoked through the :class:`InitPrompts`
protocol so the helper stays free of Typer or questionary
(import-linter contract ``Only CLI imports typer``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from src.domain.validation.config_types import ConfigError

if TYPE_CHECKING:
    from collections.abc import Callable

_EVIDENCE_PRESET_CANDIDATES = frozenset({"test", "lint"})
_TRIGGER_PRESET_EXCLUDED = frozenset({"setup", "e2e"})

# Recovery choices the prompts port returns on validation failure.
RECOVERY_REVISE = "Revise selections"
RECOVERY_SKIP = "Skip section"
RECOVERY_ABORT = "Abort init"


def compute_evidence_defaults(commands: list[str], is_preset: bool) -> list[str]:
    """Compute default evidence check commands.

    Args:
        commands: List of command names available.
        is_preset: True if using a preset, False for custom.

    Returns:
        List of command names that should be evidence checks by default.
        Preset path returns the intersection of available commands with
        ``{test, lint}``; custom path returns an empty list.
    """
    if not is_preset:
        return []
    return [cmd for cmd in commands if cmd in _EVIDENCE_PRESET_CANDIDATES]


def compute_trigger_defaults(commands: list[str], is_preset: bool) -> list[str]:
    """Compute default validation trigger commands.

    Args:
        commands: List of command names available.
        is_preset: True if using a preset, False for custom.

    Returns:
        List of command names that should be triggers by default. Preset
        path excludes ``setup`` and ``e2e``; custom path returns an empty
        list.
    """
    if not is_preset:
        return []
    return [cmd for cmd in commands if cmd not in _TRIGGER_PRESET_EXCLUDED]


def build_evidence_check_dict(required: list[str]) -> dict[str, list[str]]:
    """Build the ``evidence_check`` config section.

    Args:
        required: List of command names that are required evidence.

    Returns:
        Dict with ``required`` mapping to the command list.
    """
    return {"required": required}


def build_per_issue_review_dict(config: dict[str, Any]) -> dict[str, Any]:
    """Build the ``per_issue_review`` config section.

    Args:
        config: Dict with ``enabled``, ``reviewer_type``, ``max_retries``,
            ``finding_threshold`` keys collected from prompts.

    Returns:
        Dict suitable for YAML output.
    """
    return config


def build_validation_triggers_dict(commands: list[str]) -> dict[str, Any]:
    """Build the ``validation_triggers`` config section.

    Args:
        commands: List of command names for run-end triggers.

    Returns:
        Dict with ``run_end.failure_mode`` and ``run_end.commands`` using
        ``ref`` syntax.
    """
    return {
        "run_end": {
            "failure_mode": "continue",
            "commands": [{"ref": cmd} for cmd in commands],
        }
    }


@dataclass(frozen=True)
class InitInvocationCheck:
    """Result of validating ``mala init`` flag combinations.

    Attributes:
        error_message: User-facing error message when an invalid
            combination is detected. ``None`` means the invocation is
            valid; the CLI proceeds with the configured flow.
    """

    error_message: str | None = None

    @property
    def is_valid(self) -> bool:
        """Convenience predicate: True when no error message is set."""
        return self.error_message is None


def validate_init_invocation(
    *,
    yes: bool,
    preset: str | None,
    skip_evidence: bool,
    skip_triggers: bool,
    is_tty: bool,
) -> InitInvocationCheck:
    """Validate the combination of ``mala init`` flags and TTY state.

    Mirrors the rules previously embedded at the top of the Typer command:

    - ``--yes`` requires ``--preset``.
    - Non-interactive (no TTY) mode requires either ``--yes`` with
      ``--preset`` or ``--preset`` with both ``--skip-evidence`` and
      ``--skip-triggers`` (so no prompts are needed).

    Returns:
        :class:`InitInvocationCheck` carrying the error message the CLI
        should print, or an empty check when the invocation is valid.
    """
    if yes and not preset:
        return InitInvocationCheck(error_message="--yes requires --preset")

    prompts_needed = not (skip_evidence and skip_triggers and preset)
    if not is_tty and not yes and prompts_needed:
        return InitInvocationCheck(
            error_message=(
                "Non-interactive mode requires --yes with --preset, "
                "or --preset with --skip-evidence and --skip-triggers"
            )
        )
    return InitInvocationCheck()


@dataclass(frozen=True)
class PresetSelection:
    """Resolved preset selection state for ``mala init``.

    Attributes:
        is_preset: True when a known preset was chosen.
        commands: Command names defined by the preset (empty when no
            preset).
        config_data: Initial config dict (``{"preset": preset_name}`` for
            preset selections, otherwise an empty dict).
        error_message: Set when the preset name is not recognized; the
            CLI renders it before exiting.
    """

    is_preset: bool = False
    commands: list[str] = field(default_factory=list)
    config_data: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None


def resolve_preset_selection(
    preset: str,
    presets: list[str],
    commands_lookup: Callable[[str], list[str]],
) -> PresetSelection:
    """Resolve a preset name into the initial init-state for the CLI.

    The CLI passes the candidate name, the list of known preset names,
    and a callable that returns the command list for a given preset
    (typically ``get_preset_config_commands``). This helper validates
    the preset is known and returns the initial ``config_data`` shape
    and commands list the rest of the init flow operates on.

    Args:
        preset: Preset name provided via ``--preset`` or selected by the
            user.
        presets: List of valid preset names.
        commands_lookup: Callable returning the list of command names a
            preset defines. Only invoked when ``preset`` is in
            ``presets``.

    Returns:
        :class:`PresetSelection` carrying the resolved state. When the
        preset is unknown, ``error_message`` is set and ``commands``/
        ``config_data`` are empty; ``commands_lookup`` is not invoked.
    """
    if preset not in presets:
        return PresetSelection(error_message=f"Unknown preset '{preset}'")
    return PresetSelection(
        is_preset=True,
        commands=list(commands_lookup(preset)),
        config_data={"preset": preset},
    )


class InitPrompts(Protocol):
    """Prompt port for ``mala init`` interactive flows.

    The CLI implements this via questionary; tests provide an in-memory
    stub. All ``prompt_*`` methods return ``None`` to mean "user
    cancelled / skip this section" (the workflow treats cancellation at
    a required step as exit 1).
    """

    def prompt_preset(self, presets: list[str]) -> str | None:
        """Ask the user to pick a preset, or ``None`` to choose custom."""

    def prompt_custom_commands(self) -> dict[str, str] | None:
        """Collect custom command definitions; ``None`` to cancel."""

    def prompt_evidence_check(
        self, commands: list[str], is_preset: bool
    ) -> list[str] | None:
        """Pick required evidence commands; ``None`` to skip section."""

    def prompt_per_issue_review(self) -> dict[str, Any] | None:
        """Configure per-issue review; ``None`` to skip section."""

    def prompt_run_end_trigger(
        self, commands: list[str], is_preset: bool
    ) -> list[str] | None:
        """Pick run-end trigger commands; ``None`` to skip section."""

    def prompt_recovery_choice(self, error_message: str) -> str | None:
        """Pick recovery action on validation error.

        Returns one of :data:`RECOVERY_REVISE`, :data:`RECOVERY_SKIP`,
        :data:`RECOVERY_ABORT`, or ``None`` to abort (cancelled prompt).
        """

    def notify(self, message: str) -> None:
        """Informational message routed to user-facing stdout."""


@dataclass(frozen=True)
class InitWorkflowResult:
    """Result of executing the ``mala init`` workflow.

    Attributes:
        config_data: Final config dict ready for YAML dump. ``None``
            when the workflow aborted (cancelled or exhausted retries).
        exit_code: Exit code the CLI should propagate. Zero on success.
        error_message: Optional user-facing error message; the CLI
            formats and prints it before exiting on non-zero codes.
    """

    config_data: dict[str, Any] | None = None
    exit_code: int = 0
    error_message: str | None = None


def _apply_section_decisions(
    config_data: dict[str, Any],
    *,
    commands: list[str],
    is_preset: bool,
    skip_evidence: bool,
    skip_triggers: bool,
    yes: bool,
    is_tty: bool,
    prompts: InitPrompts,
) -> None:
    """Populate evidence/per-issue-review/triggers in ``config_data``.

    Mirrors the section-by-section logic the CLI previously embedded.
    The ``--yes`` path uses computed defaults; the interactive path
    delegates to the prompts port. Returns nothing; mutates
    ``config_data`` in place so callers can reuse it across the
    retry/recovery loop.
    """
    evidence_required: list[str] | None = None
    if not skip_evidence:
        if yes:
            evidence_required = compute_evidence_defaults(commands, is_preset)
        elif is_tty:
            evidence_required = prompts.prompt_evidence_check(commands, is_preset)
    if evidence_required is not None:
        config_data["evidence_check"] = build_evidence_check_dict(evidence_required)

    if is_tty and not yes:
        per_issue_review_config = prompts.prompt_per_issue_review()
        if per_issue_review_config is not None:
            config_data["per_issue_review"] = build_per_issue_review_dict(
                per_issue_review_config
            )

    trigger_commands: list[str] | None = None
    if not skip_triggers:
        if yes:
            trigger_commands = compute_trigger_defaults(commands, is_preset)
        elif is_tty:
            trigger_commands = prompts.prompt_run_end_trigger(commands, is_preset)
    if trigger_commands is not None:
        config_data["validation_triggers"] = build_validation_triggers_dict(
            trigger_commands
        )


def _reprompt_sections_for_revise(
    config_data: dict[str, Any],
    *,
    commands: list[str],
    is_preset: bool,
    skip_evidence: bool,
    skip_triggers: bool,
    prompts: InitPrompts,
) -> None:
    """Re-prompt evidence / per-issue review / triggers during recovery.

    Each prompt's result either replaces or removes its config section,
    matching the original retry loop behavior.
    """
    if not skip_evidence and commands:
        evidence_required = prompts.prompt_evidence_check(commands, is_preset)
        if evidence_required is not None:
            config_data["evidence_check"] = build_evidence_check_dict(evidence_required)
        else:
            config_data.pop("evidence_check", None)
    if commands:
        per_issue_review_config = prompts.prompt_per_issue_review()
        if per_issue_review_config is not None:
            config_data["per_issue_review"] = build_per_issue_review_dict(
                per_issue_review_config
            )
        else:
            config_data.pop("per_issue_review", None)
    if not skip_triggers and commands:
        trigger_commands = prompts.prompt_run_end_trigger(commands, is_preset)
        if trigger_commands is not None:
            config_data["validation_triggers"] = build_validation_triggers_dict(
                trigger_commands
            )
        else:
            config_data.pop("validation_triggers", None)


def execute_init_workflow(
    *,
    preset: str | None,
    skip_evidence: bool,
    skip_triggers: bool,
    yes: bool,
    is_tty: bool,
    presets: list[str],
    commands_lookup: Callable[[str], list[str]],
    validate_config: Callable[[dict[str, Any]], None],
    prompts: InitPrompts,
) -> InitWorkflowResult:
    """Drive the ``mala init`` workflow end-to-end.

    Owns preset/custom branching, section-by-section construction of
    ``config_data``, and the validate/retry/recovery loop. Prompts are
    issued through :class:`InitPrompts`; the CLI provides a questionary
    implementation. Flag/TTY validation happens before this call via
    :func:`validate_init_invocation`; this helper trusts its inputs.

    Args:
        preset: Preset name from ``--preset`` (or ``None``).
        skip_evidence: ``--skip-evidence`` flag.
        skip_triggers: ``--skip-triggers`` flag.
        yes: ``--yes`` flag (preset path uses computed defaults).
        is_tty: Whether stdin is a TTY (interactive mode enabled).
        presets: List of known preset names.
        commands_lookup: Callable returning the command names a preset
            defines (typically ``get_preset_config_commands``).
        validate_config: Callable that validates a config dict and
            raises :class:`ConfigError` on failure (typically
            ``validate_init_config``).
        prompts: Prompt port for interactive flows.

    Returns:
        :class:`InitWorkflowResult`. ``config_data`` is populated on
        success; ``exit_code`` is non-zero and ``error_message`` may be
        set on failure.
    """
    config_data: dict[str, Any]
    if preset:
        selection = resolve_preset_selection(preset, presets, commands_lookup)
        if selection.error_message:
            return InitWorkflowResult(
                exit_code=1, error_message=selection.error_message
            )
        is_preset = selection.is_preset
        commands = selection.commands
        config_data = dict(selection.config_data)
    elif is_tty:
        selected_preset = prompts.prompt_preset(presets)
        if selected_preset:
            is_preset = True
            commands = list(commands_lookup(selected_preset))
            config_data = {"preset": selected_preset}
        else:
            is_preset = False
            commands = []
            custom_commands = prompts.prompt_custom_commands()
            if custom_commands is None:
                return InitWorkflowResult(exit_code=1)
            if custom_commands:
                commands = list(custom_commands.keys())
            config_data = {"commands": custom_commands}
    else:
        # Unreachable in practice: validate_init_invocation gates this.
        return InitWorkflowResult(
            exit_code=1,
            error_message="Cannot run interactively in non-TTY mode",
        )

    if not commands:
        prompts.notify("No commands defined; skipping evidence and trigger setup.")
    else:
        _apply_section_decisions(
            config_data,
            commands=commands,
            is_preset=is_preset,
            skip_evidence=skip_evidence,
            skip_triggers=skip_triggers,
            yes=yes,
            is_tty=is_tty,
            prompts=prompts,
        )

    max_retries = 3
    for attempt in range(max_retries + 1):
        try:
            validate_config(config_data)
            break
        except ConfigError as e:
            if not is_tty:
                return InitWorkflowResult(exit_code=1, error_message=str(e))
            if attempt >= max_retries:
                return InitWorkflowResult(
                    exit_code=1, error_message=f"{e} (max retries exceeded)"
                )
            choice = prompts.prompt_recovery_choice(str(e))
            if choice is None or choice == RECOVERY_ABORT:
                return InitWorkflowResult(exit_code=1)
            if choice == RECOVERY_SKIP:
                if not commands:
                    return InitWorkflowResult(
                        exit_code=1,
                        error_message="Cannot skip - no commands defined",
                    )
                config_data.pop("evidence_check", None)
                config_data.pop("per_issue_review", None)
                config_data.pop("validation_triggers", None)
            else:
                if not commands:
                    custom_commands = prompts.prompt_custom_commands()
                    if custom_commands is None:
                        return InitWorkflowResult(exit_code=1)
                    if custom_commands:
                        commands = list(custom_commands.keys())
                        config_data["commands"] = custom_commands
                _reprompt_sections_for_revise(
                    config_data,
                    commands=commands,
                    is_preset=is_preset,
                    skip_evidence=skip_evidence,
                    skip_triggers=skip_triggers,
                    prompts=prompts,
                )

    return InitWorkflowResult(config_data=config_data)


__all__ = [
    "RECOVERY_ABORT",
    "RECOVERY_REVISE",
    "RECOVERY_SKIP",
    "InitInvocationCheck",
    "InitPrompts",
    "InitWorkflowResult",
    "PresetSelection",
    "build_evidence_check_dict",
    "build_per_issue_review_dict",
    "build_validation_triggers_dict",
    "compute_evidence_defaults",
    "compute_trigger_defaults",
    "execute_init_workflow",
    "resolve_preset_selection",
    "validate_init_invocation",
]
