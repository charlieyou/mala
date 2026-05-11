"""Init config generation helpers.

Pure helpers for ``mala init`` YAML generation. The CLI owns prompts and
file I/O; this module holds the deterministic transformations from user
selections to ``mala.yaml`` config dict pieces so they can be exercised
without Typer or questionary.

No prompts, no file I/O, no Typer dependency (enforced by import-linter
contract ``Only CLI imports typer``).
"""

from __future__ import annotations

from typing import Any

_EVIDENCE_PRESET_CANDIDATES = frozenset({"test", "lint"})
_TRIGGER_PRESET_EXCLUDED = frozenset({"setup", "e2e"})


def get_preset_command_names(preset_name: str) -> list[str]:
    """Return command names defined in a preset.

    Args:
        preset_name: Name of the preset (e.g., 'python-uv', 'go').

    Returns:
        Sorted list of command names defined in the preset.

    Raises:
        PresetNotFoundError: If the preset name is not recognized.
    """
    from src.orchestration.cli_support import get_preset_config_commands

    return get_preset_config_commands(preset_name)


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


__all__ = [
    "build_evidence_check_dict",
    "build_per_issue_review_dict",
    "build_validation_triggers_dict",
    "compute_evidence_defaults",
    "compute_trigger_defaults",
    "get_preset_command_names",
]
