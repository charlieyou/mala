"""CLI override application for MalaConfig.

Moves the `_apply_cli_overrides` logic out of the Typer CLI so it can be
exercised in isolation. The CLI is responsible for parsing flag values; this
module receives parsed string options and returns a new `MalaConfig` whose
fields reflect the CLI > env > yaml > default precedence chain.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

from src.infra.io.config import CLIOverrides, MalaConfig, build_resolved_config

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class CLIOverrideOptions:
    """Parsed CLI override options forwarded from Typer commands."""

    claude_settings_sources: str | None = None
    coder: str | None = None
    amp_mode: str | None = None
    model: str | None = None
    effort: str | None = None
    codex_approval_policy: str | None = None
    codex_sandbox: str | None = None

    def has_any(self) -> bool:
        """Return True when at least one override value is set."""
        return any(
            value is not None
            for value in (
                self.claude_settings_sources,
                self.coder,
                self.amp_mode,
                self.model,
                self.effort,
                self.codex_approval_policy,
                self.codex_sandbox,
            )
        )


def apply_cli_overrides(
    config: MalaConfig,
    options: CLIOverrideOptions,
) -> MalaConfig:
    """Apply CLI overrides to MalaConfig.

    Uses build_resolved_config so CLI > env > yaml > default precedence and
    cross-coder ignored-flag info logging stay in one place.

    Args:
        config: Base MalaConfig (typically returned by ``MalaConfig.from_env``).
        options: Parsed CLI override options.

    Returns:
        A new MalaConfig reflecting the merged overrides. Returns the input
        unchanged when no overrides are set.

    Raises:
        ValueError: Propagated from ``build_resolved_config`` when override
            values cannot be parsed.
    """
    if not options.has_any():
        return config

    overrides = CLIOverrides(
        claude_settings_sources=options.claude_settings_sources,
        coder=options.coder,
        amp_mode=options.amp_mode,
        model=options.model,
        effort=options.effort,
        codex_approval_policy=options.codex_approval_policy,
        codex_sandbox=options.codex_sandbox,
    )
    resolved = build_resolved_config(config, overrides)

    init_kwargs = {
        field.name: getattr(config, field.name)
        for field in fields(config)
        if field.init
        and field.name
        not in {
            "claude_settings_sources_init",
            "coder",
            "coder_options",
            "model",
            "_model_is_default",
            "effort",
            "_effort_is_default",
        }
    }
    return MalaConfig(
        **init_kwargs,
        claude_settings_sources_init=resolved.claude_settings_sources,
        coder=resolved.coder,
        coder_options=resolved.coder_options,
        model=resolved.model,
        _model_is_default=resolved.model_is_default,
        effort=resolved.effort,
        _effort_is_default=resolved.effort_is_default,
    )


def build_resolved_mala_config(
    repo_path: Path,
    options: CLIOverrideOptions,
) -> MalaConfig:
    """Build a :class:`MalaConfig` with CLI > env > yaml > default precedence.

    Encapsulates the yaml coder resolution, ``MalaConfig.from_env`` build,
    and CLI override application so the Typer CLI is left with argument
    parsing and output formatting only.

    Args:
        repo_path: Repository path; used to locate ``mala.yaml``.
        options: Parsed CLI override options.

    Returns:
        A :class:`MalaConfig` reflecting the merged values.

    Raises:
        ConfigError: Propagated from the yaml loader when ``mala.yaml`` is
            present but malformed.
        ValueError: Propagated from ``apply_cli_overrides`` when an
            override value cannot be parsed.
    """
    # Local imports keep the heavy factory/config_resolution modules out of
    # the import graph until a CLI command actually needs them.
    from src.orchestration.config_resolution import _resolve_yaml_codex_options
    from src.orchestration.factory import load_yaml_coder_resolution

    (
        yaml_css,
        yaml_coder,
        yaml_amp_mode,
        yaml_model,
        yaml_effort,
        yaml_codex_options,
    ) = load_yaml_coder_resolution(repo_path)
    config = MalaConfig.from_env(
        validate=False,
        yaml_claude_settings_sources=yaml_css,
        yaml_coder=yaml_coder,
        yaml_amp_mode=yaml_amp_mode,
        yaml_model=yaml_model,
        yaml_effort=yaml_effort,
        yaml_codex_options=_resolve_yaml_codex_options(yaml_codex_options),
    )
    return apply_cli_overrides(config, options)
