"""Unit tests for MalaConfig in src/config.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import ConfigurationError, MalaConfig


class TestMalaConfigDefaults:
    """Tests for MalaConfig default values."""

    def test_default_runs_dir(self) -> None:
        """Default runs_dir is ~/.config/mala/runs."""
        config = MalaConfig()
        assert config.runs_dir == Path.home() / ".config" / "mala" / "runs"

    def test_default_lock_dir(self) -> None:
        """Default lock_dir is /tmp/mala-locks."""
        config = MalaConfig()
        assert config.lock_dir == Path("/tmp/mala-locks")

    def test_default_claude_config_dir(self) -> None:
        """Default claude_config_dir is ~/.claude."""
        config = MalaConfig()
        assert config.claude_config_dir == Path.home() / ".claude"

    def test_default_api_keys_are_none(self) -> None:
        """API keys default to None."""
        config = MalaConfig()
        assert config.braintrust_api_key is None
        assert config.morph_api_key is None

    def test_default_feature_flags_disabled(self) -> None:
        """Feature flags are disabled when API keys are not provided."""
        config = MalaConfig()
        assert config.braintrust_enabled is False
        assert config.morph_enabled is False


class TestMalaConfigFeatureFlags:
    """Tests for feature flag derivation."""

    def test_braintrust_enabled_when_api_key_provided(self) -> None:
        """braintrust_enabled is True when braintrust_api_key is provided."""
        config = MalaConfig(braintrust_api_key="test-api-key")
        assert config.braintrust_enabled is True

    def test_morph_enabled_when_api_key_provided(self) -> None:
        """morph_enabled is True when morph_api_key is provided."""
        config = MalaConfig(morph_api_key="test-morph-key")
        assert config.morph_enabled is True

    def test_both_features_enabled(self) -> None:
        """Both features can be enabled simultaneously."""
        config = MalaConfig(
            braintrust_api_key="bt-key",
            morph_api_key="morph-key",
        )
        assert config.braintrust_enabled is True
        assert config.morph_enabled is True

    def test_feature_flag_explicit_override_preserved(self) -> None:
        """Explicit feature flag setting is preserved."""
        # With explicit flag=True but no API key, validation will fail
        # So we test that the flag is set, not that it's valid
        config = MalaConfig(
            braintrust_enabled=True,
            morph_enabled=True,
        )
        assert config.braintrust_enabled is True
        assert config.morph_enabled is True
        # But validation should fail due to missing API keys
        errors = config.validate()
        assert len(errors) == 2
        assert any("BRAINTRUST_API_KEY" in e for e in errors)
        assert any("MORPH_API_KEY" in e for e in errors)


class TestMalaConfigFromEnv:
    """Tests for from_env() classmethod."""

    def test_from_env_with_no_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """from_env() uses defaults when no env vars are set."""
        # Clear relevant env vars
        monkeypatch.delenv("MALA_RUNS_DIR", raising=False)
        monkeypatch.delenv("MALA_LOCK_DIR", raising=False)
        monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
        monkeypatch.delenv("BRAINTRUST_API_KEY", raising=False)
        monkeypatch.delenv("MORPH_API_KEY", raising=False)

        config = MalaConfig.from_env()

        assert config.runs_dir == Path.home() / ".config" / "mala" / "runs"
        assert config.lock_dir == Path("/tmp/mala-locks")
        assert config.claude_config_dir == Path.home() / ".claude"
        assert config.braintrust_api_key is None
        assert config.morph_api_key is None

    def test_from_env_reads_mala_runs_dir(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """from_env() reads MALA_RUNS_DIR."""
        monkeypatch.setenv("MALA_RUNS_DIR", "/custom/runs")
        # Use validate=False since /custom doesn't exist
        config = MalaConfig.from_env(validate=False)
        assert config.runs_dir == Path("/custom/runs")

    def test_from_env_reads_mala_lock_dir(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """from_env() reads MALA_LOCK_DIR."""
        monkeypatch.setenv("MALA_LOCK_DIR", "/custom/locks")
        # Use validate=False since /custom doesn't exist
        config = MalaConfig.from_env(validate=False)
        assert config.lock_dir == Path("/custom/locks")

    def test_from_env_reads_claude_config_dir(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """from_env() reads CLAUDE_CONFIG_DIR."""
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/custom/claude")
        # Use validate=False since /custom doesn't exist
        config = MalaConfig.from_env(validate=False)
        assert config.claude_config_dir == Path("/custom/claude")

    def test_from_env_reads_braintrust_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """from_env() reads BRAINTRUST_API_KEY."""
        monkeypatch.setenv("BRAINTRUST_API_KEY", "bt-test-key")
        config = MalaConfig.from_env()
        assert config.braintrust_api_key == "bt-test-key"
        assert config.braintrust_enabled is True

    def test_from_env_reads_morph_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """from_env() reads MORPH_API_KEY."""
        monkeypatch.setenv("MORPH_API_KEY", "morph-test-key")
        config = MalaConfig.from_env()
        assert config.morph_api_key == "morph-test-key"
        assert config.morph_enabled is True

    def test_from_env_treats_empty_string_as_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """from_env() treats empty string API keys as None."""
        monkeypatch.setenv("BRAINTRUST_API_KEY", "")
        monkeypatch.setenv("MORPH_API_KEY", "")
        config = MalaConfig.from_env()
        assert config.braintrust_api_key is None
        assert config.morph_api_key is None
        assert config.braintrust_enabled is False
        assert config.morph_enabled is False

    def test_from_env_validates_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """from_env() runs validation by default and raises on errors."""
        # Set relative path which will fail validation
        monkeypatch.setenv("MALA_RUNS_DIR", "relative/path")
        monkeypatch.delenv("BRAINTRUST_API_KEY", raising=False)
        monkeypatch.delenv("MORPH_API_KEY", raising=False)

        with pytest.raises(ConfigurationError) as exc_info:
            MalaConfig.from_env()

        assert "runs_dir should be an absolute path" in str(exc_info.value)
        assert len(exc_info.value.errors) >= 1

    def test_from_env_skip_validation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """from_env(validate=False) skips validation."""
        # Set relative path which would fail validation
        monkeypatch.setenv("MALA_RUNS_DIR", "relative/path")

        # Should not raise with validate=False
        config = MalaConfig.from_env(validate=False)
        assert config.runs_dir == Path("relative/path")


class TestMalaConfigValidate:
    """Tests for validate() method."""

    def test_validate_default_config(self) -> None:
        """Default config passes validation (assuming home dir exists)."""
        config = MalaConfig()
        errors = config.validate()
        # Default config should be valid since home dir exists
        assert len(errors) == 0

    def test_validate_missing_braintrust_api_key(self) -> None:
        """validate() reports missing BRAINTRUST_API_KEY when enabled."""
        config = MalaConfig(braintrust_enabled=True)
        errors = config.validate()
        assert any("BRAINTRUST_API_KEY" in e for e in errors)

    def test_validate_missing_morph_api_key(self) -> None:
        """validate() reports missing MORPH_API_KEY when enabled."""
        config = MalaConfig(morph_enabled=True)
        errors = config.validate()
        assert any("MORPH_API_KEY" in e for e in errors)

    def test_validate_api_key_present_passes(self) -> None:
        """validate() passes when API key is present for enabled feature."""
        config = MalaConfig(
            braintrust_api_key="test-key",
            morph_api_key="test-key",
        )
        errors = config.validate()
        # Should pass - API keys are present
        assert not any("BRAINTRUST_API_KEY" in e for e in errors)
        assert not any("MORPH_API_KEY" in e for e in errors)

    def test_validate_custom_absolute_paths(self, tmp_path: Path) -> None:
        """Custom absolute paths pass validation."""
        config = MalaConfig(
            runs_dir=tmp_path / "runs",
            lock_dir=tmp_path / "locks",
            claude_config_dir=tmp_path / "claude",
        )
        errors = config.validate()
        assert len(errors) == 0

    def test_validate_relative_paths_produce_errors(self) -> None:
        """Relative paths produce validation errors."""
        config = MalaConfig(
            runs_dir=Path("relative/runs"),
            lock_dir=Path("relative/locks"),
            claude_config_dir=Path("relative/claude"),
        )
        errors = config.validate()
        # Relative paths produce "should be absolute" errors
        assert len(errors) == 3
        assert any("runs_dir" in e and "absolute" in e for e in errors)
        assert any("lock_dir" in e and "absolute" in e for e in errors)
        assert any("claude_config_dir" in e and "absolute" in e for e in errors)


class TestMalaConfigEnsureDirectories:
    """Tests for ensure_directories() method."""

    def test_ensure_directories_creates_dirs(self, tmp_path: Path) -> None:
        """ensure_directories() creates runs_dir and lock_dir."""
        runs = tmp_path / "runs"
        locks = tmp_path / "locks"

        config = MalaConfig(
            runs_dir=runs,
            lock_dir=locks,
        )

        assert not runs.exists()
        assert not locks.exists()

        config.ensure_directories()

        assert runs.exists()
        assert runs.is_dir()
        assert locks.exists()
        assert locks.is_dir()

    def test_ensure_directories_is_idempotent(self, tmp_path: Path) -> None:
        """ensure_directories() can be called multiple times."""
        runs = tmp_path / "runs"
        locks = tmp_path / "locks"

        config = MalaConfig(
            runs_dir=runs,
            lock_dir=locks,
        )

        config.ensure_directories()
        config.ensure_directories()  # Should not raise

        assert runs.exists()
        assert locks.exists()


class TestMalaConfigImmutability:
    """Tests for frozen dataclass behavior."""

    def test_config_is_frozen(self) -> None:
        """MalaConfig is immutable after creation."""
        config = MalaConfig()
        with pytest.raises(AttributeError):
            config.runs_dir = Path("/new/path")  # type: ignore[misc]

    def test_config_is_hashable(self) -> None:
        """Frozen MalaConfig is hashable and can be used in sets."""
        config1 = MalaConfig(morph_api_key="key1")
        config2 = MalaConfig(morph_api_key="key2")

        # Should be hashable
        config_set = {config1, config2}
        assert len(config_set) == 2
