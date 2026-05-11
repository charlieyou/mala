"""CLI override application for MalaConfig.

Moves the `_apply_cli_overrides` logic out of the Typer CLI so it can be
exercised in isolation. The CLI is responsible for parsing flag values; this
module receives parsed string options and returns a new `MalaConfig` whose
fields reflect the CLI > env > yaml > default precedence chain.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from src.infra.io.config import CLIOverrides, MalaConfig, build_resolved_config


@dataclass(frozen=True)
class CLIOverrideOptions:
    """Parsed CLI override options forwarded from Typer commands."""

    claude_settings_sources: str | None = None
    coder: str | None = None
    amp_mode: str | None = None
    effort: str | None = None
    codex_model: str | None = None
    codex_effort: str | None = None
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
                self.effort,
                self.codex_model,
                self.codex_effort,
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
        effort=options.effort,
        codex_model=options.codex_model,
        codex_effort=options.codex_effort,
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
            "effort",
        }
    }
    return MalaConfig(
        **init_kwargs,
        claude_settings_sources_init=resolved.claude_settings_sources,
        coder=resolved.coder,
        coder_options=resolved.coder_options,
        effort=resolved.effort,
    )
