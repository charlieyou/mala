"""CLI option parsing and validation helpers.

Moves the validators and scope parsing out of the Typer CLI so they can be
exercised in isolation. Helpers raise plain ``ValueError`` on bad input so
this module stays free of any CLI framework dependency (import-linter
contract ``Only CLI imports typer``). The Typer callbacks in ``src.cli.cli``
remain thin and only convert ``ValueError`` into Typer-flavored errors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from src.infra.io.config import (
    parse_amp_mode,
    parse_codex_approval_policy,
    parse_codex_effort,
    parse_codex_model,
    parse_codex_sandbox,
    parse_coder,
    parse_effort,
)

_SCOPE_FORMATS = "Valid formats: all, epic:<id>, orphans, ids:<id,...>"


@dataclass(frozen=True)
class ScopeConfig:
    """Parsed scope configuration from the ``--scope`` option.

    Attributes:
        scope_type: The type of scope ('all', 'epic', 'orphans', 'ids').
        ids: List of issue IDs to process (from 'ids:' prefix). Order-preserving.
        epic_id: Epic ID to filter by (from 'epic:' prefix).
        duplicate_ids: IDs that were repeated in an ``ids:`` scope and removed
            during deduplication. Empty for any other scope_type. The CLI emits
            a warning when this is non-empty so users see which entries were
            collapsed.
    """

    scope_type: str = "all"
    ids: list[str] | None = None
    epic_id: str | None = None
    duplicate_ids: tuple[str, ...] = field(default_factory=tuple)


def parse_scope(scope: str) -> ScopeConfig:
    """Parse the ``--scope`` option value into a ``ScopeConfig``.

    Supported formats:
        - 'all' (default): Process all issues
        - 'epic:<id>': Filter to issues under the specified epic
        - 'orphans': Process only orphan issues (no parent epic)
        - 'ids:<id1>,<id2>,...': Process specific IDs in order; duplicates are
          collapsed and surfaced via ``ScopeConfig.duplicate_ids``.

    Args:
        scope: The scope string (e.g., ``ids:T-1,T-2``).

    Returns:
        ``ScopeConfig`` with parsed values.

    Raises:
        ValueError: If scope format is invalid. The CLI converts this into a
            ``typer.Exit(1)`` after printing the message.
    """
    scope = scope.strip()

    if scope == "all":
        return ScopeConfig(scope_type="all")

    if scope == "orphans":
        return ScopeConfig(scope_type="orphans")

    if scope.startswith("epic:"):
        epic_id = scope[5:].strip()
        if not epic_id:
            raise ValueError(
                "Invalid --scope: 'epic:' requires an epic ID. " + _SCOPE_FORMATS
            )
        return ScopeConfig(scope_type="epic", epic_id=epic_id)

    if scope.startswith("ids:"):
        ids_str = scope[4:].strip()
        if not ids_str:
            raise ValueError(
                "Invalid --scope: 'ids:' requires at least one ID. " + _SCOPE_FORMATS
            )

        raw_ids = [id_.strip() for id_ in ids_str.split(",") if id_.strip()]
        if not raw_ids:
            raise ValueError(
                "Invalid --scope: 'ids:' requires at least one ID. " + _SCOPE_FORMATS
            )

        seen: set[str] = set()
        unique_ids: list[str] = []
        duplicates: list[str] = []
        for id_ in raw_ids:
            if id_ in seen:
                duplicates.append(id_)
            else:
                seen.add(id_)
                unique_ids.append(id_)

        return ScopeConfig(
            scope_type="ids",
            ids=unique_ids,
            duplicate_ids=tuple(duplicates),
        )

    raise ValueError(f"Invalid --scope value: '{scope}'. " + _SCOPE_FORMATS)


def validate_coder_option(
    value: str | None,
) -> Literal["claude", "amp", "codex"] | None:
    """Validate the ``--coder`` option value.

    Raises:
        ValueError: If ``value`` is not a recognized coder. Propagated from
            ``parse_coder``.
    """
    return parse_coder(value, source="--coder")


def validate_amp_mode_option(
    value: str | None,
) -> Literal["smart", "rush", "deep"] | None:
    """Validate the ``--amp-mode`` option value.

    Raises:
        ValueError: If ``value`` is not a recognized Amp mode.
    """
    return parse_amp_mode(value, source="--amp-mode")


def validate_effort_option(value: str | None) -> str | None:
    """Validate the ``--effort`` option value.

    Raises:
        ValueError: If ``value`` is not a recognized effort level.
    """
    return parse_effort(value, source="--effort")


def validate_codex_model_option(value: str | None) -> str | None:
    """Validate the ``--codex-model`` option value (shape-only).

    Raises:
        ValueError: Reserved for future stricter validation; the underlying
            parser currently performs shape-only validation.
    """
    return parse_codex_model(value, source="--codex-model")


def validate_codex_effort_option(value: str | None) -> str | None:
    """Validate the ``--codex-effort`` option value against the Codex SDK enum.

    Raises:
        ValueError: If ``value`` is not accepted by the Codex ``ReasoningEffort``
            enum (or its documented fallback).
    """
    return parse_codex_effort(value, source="--codex-effort")


def validate_codex_approval_policy_option(
    value: str | None,
) -> Literal["never", "on-request", "on-failure", "untrusted"] | None:
    """Validate the ``--codex-approval-policy`` option value.

    Raises:
        ValueError: If ``value`` is not a recognized Codex approval policy.
    """
    return parse_codex_approval_policy(value, source="--codex-approval-policy")


def validate_codex_sandbox_option(
    value: str | None,
) -> Literal["read-only", "workspace-write", "danger-full-access"] | None:
    """Validate the ``--codex-sandbox`` option value.

    Raises:
        ValueError: If ``value`` is not a recognized Codex sandbox mode.
    """
    return parse_codex_sandbox(value, source="--codex-sandbox")
