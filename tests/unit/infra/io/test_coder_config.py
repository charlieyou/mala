"""Unit tests for the coder/coder_options resolver in src/infra/io/config.py.

Covers AC#3: CLI > env > yaml > default precedence for both `coder` and
`coder_options.amp.mode`, plus the regression that older `mala.yaml` files
without `coder:` continue to load with the Claude default.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from src.infra.io.config import (
    AmpOptions,
    CLIOverrides,
    CoderOptions,
    ConfigurationError,
    MalaConfig,
    build_resolved_config,
    parse_amp_mode,
    parse_coder,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip relevant env vars so tests are deterministic."""
    for var in (
        "MALA_CODER",
        "MALA_AMP_MODE",
        "MALA_CLAUDE_SETTINGS_SOURCES",
    ):
        monkeypatch.delenv(var, raising=False)


class TestDefaults:
    def test_malaconfig_defaults_to_claude(self) -> None:
        config = MalaConfig()
        assert config.coder == "claude"
        assert config.coder_options.amp.mode == "deep"

    def test_from_env_no_overrides_uses_defaults(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = MalaConfig.from_env(validate=False)
        assert config.coder == "claude"
        assert config.coder_options.amp.mode == "deep"

    def test_amp_options_and_coder_options_are_frozen(self) -> None:
        amp = AmpOptions()
        with pytest.raises(Exception):
            amp.mode = "deep"  # type: ignore[misc]  # ty:ignore[invalid-assignment]
        opts = CoderOptions()
        with pytest.raises(Exception):
            opts.amp = AmpOptions(mode="rush")  # type: ignore[misc]  # ty:ignore[invalid-assignment]


class TestEnvLayer:
    def test_env_coder_amp_honored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MALA_CODER", "amp")
        config = MalaConfig.from_env(validate=False)
        assert config.coder == "amp"

    def test_env_amp_mode_honored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MALA_AMP_MODE", "deep")
        config = MalaConfig.from_env(validate=False)
        assert config.coder_options.amp.mode == "deep"

    def test_env_amp_mode_strips_whitespace(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MALA_AMP_MODE", "  rush  ")
        config = MalaConfig.from_env(validate=False)
        assert config.coder_options.amp.mode == "rush"

    def test_env_coder_invalid_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MALA_CODER", "opus")
        with pytest.raises(ConfigurationError) as exc_info:
            MalaConfig.from_env(validate=False)
        assert "MALA_CODER" in str(exc_info.value)
        assert "opus" in str(exc_info.value)

    def test_env_amp_mode_invalid_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MALA_AMP_MODE", "blast")
        with pytest.raises(ConfigurationError) as exc_info:
            MalaConfig.from_env(validate=False)
        assert "MALA_AMP_MODE" in str(exc_info.value)
        assert "blast" in str(exc_info.value)


class TestYamlLayer:
    def test_yaml_coder_used_when_env_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = MalaConfig.from_env(validate=False, yaml_coder="amp")
        assert config.coder == "amp"

    def test_yaml_amp_mode_used_when_env_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = MalaConfig.from_env(validate=False, yaml_amp_mode="deep")
        assert config.coder_options.amp.mode == "deep"

    def test_env_beats_yaml(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MALA_CODER", "amp")
        monkeypatch.setenv("MALA_AMP_MODE", "rush")
        config = MalaConfig.from_env(
            validate=False, yaml_coder="claude", yaml_amp_mode="deep"
        )
        assert config.coder == "amp"
        assert config.coder_options.amp.mode == "rush"


class TestRegressionLegacyYaml:
    def test_legacy_yaml_without_coder_field_loads_with_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Older mala.yaml files without `coder:` still load and default to claude."""
        # yaml_coder=None simulates a yaml file without the `coder:` field.
        config = MalaConfig.from_env(
            validate=False, yaml_coder=None, yaml_amp_mode=None
        )
        assert config.coder == "claude"
        assert config.coder_options.amp.mode == "deep"


class TestYamlEndToEnd:
    """End-to-end yaml-to-MalaConfig flow (loads a real mala.yaml on disk)."""

    def test_real_yaml_with_coder_amp_reaches_malaconfig(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Loading a real mala.yaml with `coder: amp` flows into MalaConfig.

        Regression for the gap where ValidationConfig validated `coder:` but
        did not store it, leaving the yaml resolver layer dead in production
        (factory.create_orchestrator had nothing to forward to from_env).
        """
        from src.domain.validation.config_loader import load_config

        yaml_path = tmp_path / "mala.yaml"
        yaml_path.write_text(
            "preset: python-uv\ncoder: amp\ncoder_options:\n  amp:\n    mode: deep\n"
        )
        validation_config = load_config(tmp_path)
        assert validation_config.coder == "amp"
        assert validation_config.amp_mode == "deep"

        # Ensure env vars don't pre-empt the yaml layer.
        monkeypatch.delenv("MALA_CODER", raising=False)
        monkeypatch.delenv("MALA_AMP_MODE", raising=False)
        config = MalaConfig.from_env(
            validate=False,
            yaml_coder=validation_config.coder,
            yaml_amp_mode=validation_config.amp_mode,
        )
        assert config.coder == "amp"
        assert config.coder_options.amp.mode == "deep"

    def test_real_yaml_without_coder_falls_through_to_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Real mala.yaml without `coder:` still loads and defaults to claude."""
        from src.domain.validation.config_loader import load_config

        yaml_path = tmp_path / "mala.yaml"
        yaml_path.write_text("preset: python-uv\n")
        validation_config = load_config(tmp_path)
        assert validation_config.coder is None
        assert validation_config.amp_mode is None

        config = MalaConfig.from_env(
            validate=False,
            yaml_coder=validation_config.coder,
            yaml_amp_mode=validation_config.amp_mode,
        )
        assert config.coder == "claude"
        assert config.coder_options.amp.mode == "deep"


class TestBuildResolvedConfigPrecedence:
    def _base(self) -> MalaConfig:
        return MalaConfig(
            runs_dir=Path("/tmp/runs"),
            lock_dir=Path("/tmp/locks"),
            claude_config_dir=Path("/tmp/claude"),
        )

    def test_default_when_no_layers_set(self) -> None:
        resolved = build_resolved_config(self._base(), CLIOverrides())
        assert resolved.coder == "claude"
        assert resolved.coder_options.amp.mode == "deep"

    def test_yaml_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        config = MalaConfig.from_env(
            validate=False, yaml_coder="amp", yaml_amp_mode="deep"
        )
        resolved = build_resolved_config(config, CLIOverrides())
        assert resolved.coder == "amp"
        assert resolved.coder_options.amp.mode == "deep"

    def test_env_beats_yaml(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MALA_CODER", "claude")
        monkeypatch.setenv("MALA_AMP_MODE", "rush")
        config = MalaConfig.from_env(
            validate=False, yaml_coder="amp", yaml_amp_mode="deep"
        )
        resolved = build_resolved_config(config, CLIOverrides())
        assert resolved.coder == "claude"
        assert resolved.coder_options.amp.mode == "rush"

    def test_cli_beats_env_beats_yaml_beats_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CLI wins when all four layers are set."""
        monkeypatch.setenv("MALA_CODER", "claude")
        monkeypatch.setenv("MALA_AMP_MODE", "rush")
        config = MalaConfig.from_env(
            validate=False, yaml_coder="claude", yaml_amp_mode="deep"
        )
        overrides = CLIOverrides(coder="amp", amp_mode="smart")
        resolved = build_resolved_config(config, overrides)
        assert resolved.coder == "amp"
        assert resolved.coder_options.amp.mode == "smart"

    def test_cli_invalid_coder_raises(self) -> None:
        with pytest.raises(ValueError, match="CLI"):
            build_resolved_config(self._base(), CLIOverrides(coder="opus"))

    def test_cli_invalid_amp_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="CLI"):
            build_resolved_config(self._base(), CLIOverrides(amp_mode="blast"))

    def test_partial_cli_only_overrides_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CLI override on one field must not affect the other."""
        monkeypatch.setenv("MALA_AMP_MODE", "deep")
        config = MalaConfig.from_env(validate=False, yaml_coder="amp")
        # CLI overrides only coder back to claude; amp_mode comes from env (deep).
        resolved = build_resolved_config(config, CLIOverrides(coder="claude"))
        assert resolved.coder == "claude"
        assert resolved.coder_options.amp.mode == "deep"


class TestIgnoredOptionsLog:
    def test_amp_mode_ignored_when_coder_claude(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """`--amp-mode rush --coder claude` logs an info-level ignored message."""
        config = MalaConfig()  # default coder=claude
        overrides = CLIOverrides(coder="claude", amp_mode="rush")
        with caplog.at_level(logging.INFO, logger="src.infra.io.config"):
            build_resolved_config(config, overrides)
        assert any(
            record.levelno == logging.INFO
            and "ignored" in record.getMessage().lower()
            and "claude" in record.getMessage().lower()
            and "rush" in record.getMessage().lower()
            for record in caplog.records
        )

    def test_claude_settings_sources_ignored_when_coder_amp(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        config = MalaConfig(claude_settings_sources_init=("user",))
        overrides = CLIOverrides(coder="amp")
        with caplog.at_level(logging.INFO, logger="src.infra.io.config"):
            build_resolved_config(config, overrides)
        assert any(
            record.levelno == logging.INFO
            and "ignored" in record.getMessage().lower()
            and "amp" in record.getMessage().lower()
            for record in caplog.records
        )

    def test_no_ignored_log_when_coder_matches(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        config = MalaConfig()
        overrides = CLIOverrides(coder="amp", amp_mode="deep")
        with caplog.at_level(logging.INFO, logger="src.infra.io.config"):
            build_resolved_config(config, overrides)
        assert not any(
            "ignored" in record.getMessage().lower() for record in caplog.records
        )


class TestParseHelpers:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("claude", "claude"), ("amp", "amp"), (None, None), ("", None), ("  ", None)],
    )
    def test_parse_coder_valid_and_empty(
        self, raw: str | None, expected: str | None
    ) -> None:
        assert parse_coder(raw, source="test") == expected

    def test_parse_coder_rejects_invalid(self) -> None:
        with pytest.raises(ValueError, match="Invalid coder 'opus'"):
            parse_coder("opus", source="MALA_CODER")

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("smart", "smart"),
            ("rush", "rush"),
            ("deep", "deep"),
            (None, None),
            ("", None),
        ],
    )
    def test_parse_amp_mode_valid_and_empty(
        self, raw: str | None, expected: str | None
    ) -> None:
        assert parse_amp_mode(raw, source="test") == expected

    def test_parse_amp_mode_rejects_invalid(self) -> None:
        with pytest.raises(ValueError, match="Invalid amp mode 'blast'"):
            parse_amp_mode("blast", source="MALA_AMP_MODE")
