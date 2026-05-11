"""Init config generation helpers.

Pure helpers for ``mala init`` YAML generation. The CLI owns prompts and
file I/O; this module holds the deterministic transformations from user
selections to ``mala.yaml`` config dict pieces so they can be exercised
without Typer or questionary.

No prompts, no file I/O, no Typer dependency (enforced by import-linter
contract ``Only CLI imports typer``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

_EVIDENCE_PRESET_CANDIDATES = frozenset({"test", "lint"})
_TRIGGER_PRESET_EXCLUDED = frozenset({"setup", "e2e"})


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


__all__ = [
    "InitInvocationCheck",
    "PresetSelection",
    "build_evidence_check_dict",
    "build_per_issue_review_dict",
    "build_validation_triggers_dict",
    "compute_evidence_defaults",
    "compute_trigger_defaults",
    "resolve_preset_selection",
    "validate_init_invocation",
]
