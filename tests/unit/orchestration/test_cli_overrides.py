"""Unit tests for apply_cli_overrides.

Exercise the CLI override application helper without Typer. The CLI itself
is a thin adapter: it parses Typer flags, builds a CLIOverrideOptions, and
delegates here. These tests assert override values flow through to the
resulting MalaConfig with CLI > yaml > default precedence.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.infra.io.config import MalaConfig
from src.orchestration.cli_overrides import (
    CLIOverrideOptions,
    apply_cli_overrides,
    build_resolved_mala_config,
)


pytestmark = pytest.mark.unit


class TestCLIOverrideOptions:
    """Behavior of the CLIOverrideOptions value object."""

    def test_has_any_false_when_all_none(self) -> None:
        assert CLIOverrideOptions().has_any() is False

    def test_has_any_true_when_any_string_field_set(self) -> None:
        assert CLIOverrideOptions(claude_settings_sources="value").has_any() is True
        assert CLIOverrideOptions(coder="value").has_any() is True
        assert CLIOverrideOptions(amp_mode="value").has_any() is True
        assert CLIOverrideOptions(model="value").has_any() is True
        assert CLIOverrideOptions(effort="value").has_any() is True
        assert CLIOverrideOptions(codex_approval_policy="value").has_any() is True
        assert CLIOverrideOptions(codex_sandbox="value").has_any() is True

    def test_has_any_true_when_fast_set(self) -> None:
        assert CLIOverrideOptions(fast=True).has_any() is True


class TestApplyCLIOverrides:
    """Apply CLI overrides preserves CLI > env > yaml > default precedence."""

    def test_returns_input_when_no_overrides(self) -> None:
        """No override values means no work: return the input config unchanged."""
        config = MalaConfig.from_env(validate=False)
        result = apply_cli_overrides(config, CLIOverrideOptions())
        assert result is config

    def test_coder_override_replaces_yaml_coder(self) -> None:
        """CLI --coder beats the yaml selection."""
        config = MalaConfig.from_env(validate=False, yaml_coder="amp")
        assert config.coder == "amp"

        result = apply_cli_overrides(
            config,
            CLIOverrideOptions(coder="claude"),
        )
        assert result.coder == "claude"

    def test_amp_mode_override_threads_through_to_coder_options(self) -> None:
        """--amp-mode lands on coder_options.amp.mode for amp coder."""
        config = MalaConfig.from_env(validate=False, yaml_coder="amp")
        result = apply_cli_overrides(
            config,
            CLIOverrideOptions(coder="amp", amp_mode="rush"),
        )
        assert result.coder == "amp"
        assert result.coder_options.amp.mode == "rush"

    def test_effort_override_replaces_default(self) -> None:
        """--effort is reflected on the returned config."""
        config = MalaConfig.from_env(validate=False, yaml_coder="claude")
        result = apply_cli_overrides(
            config,
            CLIOverrideOptions(coder="claude", effort="high"),
        )
        assert result.effort == "high"

    def test_claude_settings_sources_override_parses_csv(self) -> None:
        """--claude-settings-sources csv parses into the typed tuple."""
        config = MalaConfig.from_env(validate=False, yaml_coder="claude")
        result = apply_cli_overrides(
            config,
            CLIOverrideOptions(claude_settings_sources="local,user"),
        )
        assert result.claude_settings_sources == ("local", "user")

    def test_coder_overrides_thread_through_to_model_and_codex_options(self) -> None:
        """Coder CLI options reach model/effort and Codex-only options."""
        config = MalaConfig.from_env(validate=False, yaml_coder="codex")
        result = apply_cli_overrides(
            config,
            CLIOverrideOptions(
                coder="codex",
                model="gpt-5.5",
                effort="high",
                codex_approval_policy="never",
                codex_sandbox="workspace-write",
            ),
        )
        assert result.coder == "codex"
        assert result.model == "gpt-5.5"
        assert result.effort == "high"
        codex = result.coder_options.codex
        assert codex.approval_policy == "never"
        assert codex.sandbox == "workspace-write"

    def test_fast_override_enables_codex_fast_mode(self) -> None:
        """--fast lands on Codex options without selecting another coder."""
        config = MalaConfig.from_env(validate=False, yaml_coder="codex")

        result = apply_cli_overrides(config, CLIOverrideOptions(fast=True))

        assert result.coder == "codex"
        assert result.coder_options.codex.fast_mode is True

    def test_invalid_amp_mode_raises_value_error(self) -> None:
        """build_resolved_config raises ValueError on invalid override values."""
        config = MalaConfig.from_env(validate=False, yaml_coder="amp")
        with pytest.raises(ValueError):
            apply_cli_overrides(
                config,
                CLIOverrideOptions(coder="amp", amp_mode="not-a-mode"),
            )

    def test_preserves_unrelated_fields(self) -> None:
        """Fields not touched by CLI overrides survive the round-trip."""
        config = MalaConfig(
            runs_dir=Path("/tmp/runs-fixed"),
            lock_dir=Path("/tmp/locks-fixed"),
            coder="claude",
        )

        result = apply_cli_overrides(
            config,
            CLIOverrideOptions(effort="medium"),
        )
        assert result.runs_dir == Path("/tmp/runs-fixed")
        assert result.lock_dir == Path("/tmp/locks-fixed")
        assert result.effort == "medium"

    def test_returns_new_instance_when_overrides_apply(self) -> None:
        """When any override is set the helper returns a fresh MalaConfig."""
        config = MalaConfig.from_env(validate=False, yaml_coder="claude")
        result = apply_cli_overrides(
            config,
            CLIOverrideOptions(coder="claude"),
        )
        assert result is not config


class TestBuildResolvedMalaConfig:
    """``build_resolved_mala_config`` combines yaml + env + CLI overrides."""

    def test_no_mala_yaml_no_overrides_uses_env_defaults(self, tmp_path: Path) -> None:
        """Without mala.yaml or overrides, returns a MalaConfig.from_env-like value."""
        config = build_resolved_mala_config(tmp_path, CLIOverrideOptions())
        # No overrides set means apply_cli_overrides returns the input unchanged;
        # a coder default is present regardless.
        assert config.coder in ("claude", "amp", "codex")

    def test_cli_overrides_beat_defaults(self, tmp_path: Path) -> None:
        """CLI overrides flow through the combined helper."""
        config = build_resolved_mala_config(
            tmp_path,
            CLIOverrideOptions(coder="amp", amp_mode="smart", effort="high"),
        )
        assert config.coder == "amp"
        assert config.coder_options.amp.mode == "smart"
        assert config.effort == "high"

    def test_invalid_override_raises_value_error(self, tmp_path: Path) -> None:
        """Invalid CLI override propagates as ValueError from apply_cli_overrides."""
        with pytest.raises(ValueError):
            build_resolved_mala_config(
                tmp_path,
                CLIOverrideOptions(coder="amp", amp_mode="not-a-mode"),
            )

    def test_yaml_coder_resolution_feeds_from_env(self, tmp_path: Path) -> None:
        """A valid mala.yaml ``coder:`` value reaches MalaConfig through from_env."""
        # Preset provides the required commands so the yaml validator does
        # not abort before reaching the coder fields under test.
        (tmp_path / "mala.yaml").write_text("preset: go\ncoder: amp\namp_mode: rush\n")
        config = build_resolved_mala_config(tmp_path, CLIOverrideOptions())
        assert config.coder == "amp"
        assert config.coder_options.amp.mode == "rush"

    def test_explicit_config_path_feeds_yaml_coder_resolution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An explicit non-mala.yaml file supplies yaml coder values."""
        for name in ("MALA_CODER", "MALA_AMP_MODE", "MALA_EFFORT"):
            monkeypatch.delenv(name, raising=False)
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        (repo_path / "mala.yaml").write_text("preset: go\ncoder: claude\n")
        alt_config = tmp_path / "mala.codex.yaml"
        alt_config.write_text("preset: go\ncoder: amp\namp_mode: deep\n")

        config = build_resolved_mala_config(
            repo_path,
            CLIOverrideOptions(),
            config_path=alt_config,
        )

        assert config.coder == "amp"
        assert config.coder_options.amp.mode == "deep"
