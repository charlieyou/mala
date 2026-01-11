"""Unit tests for MalaConfig in src/config.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.core.constants import DEFAULT_CLAUDE_SETTINGS_SOURCES
from src.infra.io.config import (
    CLIOverrides,
    ConfigurationError,
    MalaConfig,
    build_resolved_config,
)


class TestMalaConfigFromEnv:
    """Tests for from_env() classmethod."""

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

    def test_from_env_ignores_review_timeout_deprecated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """from_env() ignores MALA_REVIEW_TIMEOUT with deprecation warning."""
        monkeypatch.setenv("MALA_REVIEW_TIMEOUT", "450")
        with pytest.warns(
            DeprecationWarning, match="MALA_REVIEW_TIMEOUT is deprecated"
        ):
            config = MalaConfig.from_env(validate=False)
        # Env var is ignored; default is used
        assert config.review_timeout == 1200

    def test_from_env_ignores_cerberus_spawn_args_deprecated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """from_env() ignores MALA_CERBERUS_SPAWN_ARGS with deprecation warning."""
        monkeypatch.setenv("MALA_CERBERUS_SPAWN_ARGS", "--foo bar --flag")
        with pytest.warns(
            DeprecationWarning, match="MALA_CERBERUS_SPAWN_ARGS is deprecated"
        ):
            config = MalaConfig.from_env(validate=False)
        # Env var is ignored; default is used
        assert config.cerberus_spawn_args == ()

    def test_from_env_ignores_cerberus_wait_args_deprecated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """from_env() ignores MALA_CERBERUS_WAIT_ARGS with deprecation warning."""
        monkeypatch.setenv("MALA_CERBERUS_WAIT_ARGS", "--baz qux")
        with pytest.warns(
            DeprecationWarning, match="MALA_CERBERUS_WAIT_ARGS is deprecated"
        ):
            config = MalaConfig.from_env(validate=False)
        # Env var is ignored; default is used
        assert config.cerberus_wait_args == ()

    def test_from_env_ignores_cerberus_env_deprecated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """from_env() ignores MALA_CERBERUS_ENV with deprecation warning."""
        monkeypatch.setenv("MALA_CERBERUS_ENV", '{"FOO":"bar","NUM":1}')
        with pytest.warns(DeprecationWarning, match="MALA_CERBERUS_ENV is deprecated"):
            config = MalaConfig.from_env(validate=False)
        # Env var is ignored; default is used
        assert config.cerberus_env == ()

    def test_from_env_ignores_max_diff_size_kb_deprecated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """from_env() ignores MALA_MAX_DIFF_SIZE_KB with deprecation warning."""
        monkeypatch.setenv("MALA_MAX_DIFF_SIZE_KB", "100")
        with pytest.warns(
            DeprecationWarning, match="MALA_MAX_DIFF_SIZE_KB is deprecated"
        ):
            # No field for this in MalaConfig; just verify warning is emitted
            MalaConfig.from_env(validate=False)

    def test_from_env_detects_cerberus_bin_path_schema_v2(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """from_env() detects cerberus bin path with v2 plugins schema."""
        plugins_dir = tmp_path / "plugins"
        install_dir = plugins_dir / "cache" / "cerberus" / "cerberus" / "1.1.5"
        bin_dir = install_dir / "bin"
        bin_dir.mkdir(parents=True)
        (bin_dir / "review-gate").write_text("#!/usr/bin/env bash\n")

        installed = {
            "version": 2,
            "plugins": {
                "cerberus@cerberus": [
                    {
                        "installPath": str(install_dir),
                        "version": "1.1.5",
                    }
                ]
            },
        }
        (plugins_dir / "installed_plugins.json").write_text(json.dumps(installed))

        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
        config = MalaConfig.from_env(validate=False)
        assert config.cerberus_bin_path == bin_dir

    def test_from_env_validates_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """from_env() runs validation by default and raises on errors."""
        # Set relative path which will fail validation
        monkeypatch.setenv("MALA_RUNS_DIR", "relative/path")

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


class TestMalaConfigClaudeSettingsSources:
    """Tests for claude_settings_sources field and parsing."""

    def test_from_env_parses_claude_settings_sources(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """from_env() parses MALA_CLAUDE_SETTINGS_SOURCES."""
        monkeypatch.setenv("MALA_CLAUDE_SETTINGS_SOURCES", "local,project")
        config = MalaConfig.from_env(validate=False)
        assert config.claude_settings_sources == ("local", "project")

    def test_from_env_filters_empty_parts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """from_env() filters empty parts from comma-separated string."""
        monkeypatch.setenv("MALA_CLAUDE_SETTINGS_SOURCES", "local,,project")
        config = MalaConfig.from_env(validate=False)
        assert config.claude_settings_sources == ("local", "project")

    def test_from_env_strips_whitespace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """from_env() strips whitespace from source names."""
        monkeypatch.setenv("MALA_CLAUDE_SETTINGS_SOURCES", " local , project , user ")
        config = MalaConfig.from_env(validate=False)
        assert config.claude_settings_sources == ("local", "project", "user")

    def test_from_env_rejects_invalid_source(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """from_env() rejects invalid source names."""
        monkeypatch.setenv("MALA_CLAUDE_SETTINGS_SOURCES", "foo")
        with pytest.raises(ConfigurationError) as exc_info:
            MalaConfig.from_env(validate=False)
        assert "MALA_CLAUDE_SETTINGS_SOURCES" in str(exc_info.value)
        assert "Invalid source 'foo'" in str(exc_info.value)
        assert "local" in str(exc_info.value)  # Valid sources listed

    def test_from_env_default_when_not_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """from_env() uses default when env var not set."""
        monkeypatch.delenv("MALA_CLAUDE_SETTINGS_SOURCES", raising=False)
        config = MalaConfig.from_env(validate=False)
        assert config.claude_settings_sources == DEFAULT_CLAUDE_SETTINGS_SOURCES

    def test_constructor_accepts_override(self) -> None:
        """MalaConfig constructor accepts claude_settings_sources_init parameter."""
        config = MalaConfig(claude_settings_sources_init=("user",))
        assert config.claude_settings_sources == ("user",)

    def test_constructor_default_value(self) -> None:
        """MalaConfig uses default when claude_settings_sources not provided."""
        config = MalaConfig()
        assert config.claude_settings_sources == DEFAULT_CLAUDE_SETTINGS_SOURCES

    def test_from_env_yaml_fallback_used_when_env_not_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """from_env() uses yaml fallback when env var not set."""
        monkeypatch.delenv("MALA_CLAUDE_SETTINGS_SOURCES", raising=False)
        config = MalaConfig.from_env(
            validate=False, yaml_claude_settings_sources=("user",)
        )
        assert config.claude_settings_sources == ("user",)

    def test_from_env_env_var_wins_over_yaml(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """from_env() prefers env var over yaml fallback."""
        monkeypatch.setenv("MALA_CLAUDE_SETTINGS_SOURCES", "local,project")
        config = MalaConfig.from_env(
            validate=False, yaml_claude_settings_sources=("user",)
        )
        assert config.claude_settings_sources == ("local", "project")

    def test_from_env_default_when_both_not_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """from_env() uses default when neither env nor yaml set."""
        monkeypatch.delenv("MALA_CLAUDE_SETTINGS_SOURCES", raising=False)
        config = MalaConfig.from_env(validate=False, yaml_claude_settings_sources=None)
        assert config.claude_settings_sources == DEFAULT_CLAUDE_SETTINGS_SOURCES


class TestMalaConfigValidate:
    """Tests for validate() method."""

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

    def test_validate_negative_review_timeout(self) -> None:
        """Negative review_timeout produces validation error."""
        config = MalaConfig(review_timeout=-1)
        errors = config.validate()
        assert any("review_timeout" in e for e in errors)


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
        config1 = MalaConfig(review_timeout=100)
        config2 = MalaConfig(review_timeout=200)

        # Should be hashable
        config_set = {config1, config2}
        assert len(config_set) == 2


class TestBuildResolvedConfig:
    """Tests for build_resolved_config() function."""

    def test_base_config_only_no_overrides(self, tmp_path: Path) -> None:
        """build_resolved_config with only base config, no CLI overrides."""
        base = MalaConfig(
            runs_dir=tmp_path / "runs",
            lock_dir=tmp_path / "locks",
            claude_config_dir=tmp_path / "claude",
        )
        resolved = build_resolved_config(base, None)

        assert resolved.runs_dir == tmp_path / "runs"
        assert resolved.lock_dir == tmp_path / "locks"
        assert resolved.claude_config_dir == tmp_path / "claude"

    def test_cli_overrides_only_default_base(self) -> None:
        """build_resolved_config with default base and CLI overrides."""
        base = MalaConfig()
        overrides = CLIOverrides(
            cerberus_spawn_args="--mode fast",
            review_timeout=600,
        )
        resolved = build_resolved_config(base, overrides)

        assert resolved.cerberus_spawn_args == ("--mode", "fast")
        assert resolved.review_timeout == 600

    def test_cli_overrides_take_precedence(self, tmp_path: Path) -> None:
        """CLI overrides win over base config values."""
        base = MalaConfig(
            runs_dir=tmp_path,
            cerberus_spawn_args=("--base", "arg"),
            cerberus_env=(("BASE_KEY", "base_value"),),
            review_timeout=300,
        )
        overrides = CLIOverrides(
            cerberus_spawn_args="--cli arg",
            cerberus_env='{"CLI_KEY": "cli_value"}',
            review_timeout=900,
        )
        resolved = build_resolved_config(base, overrides)

        assert resolved.cerberus_spawn_args == ("--cli", "arg")
        assert dict(resolved.cerberus_env) == {"CLI_KEY": "cli_value"}
        assert resolved.review_timeout == 900

    def test_base_config_used_when_override_is_none(self, tmp_path: Path) -> None:
        """Base config values used when corresponding override is None."""
        base = MalaConfig(
            runs_dir=tmp_path,
            cerberus_spawn_args=("--base",),
            cerberus_wait_args=("--wait-base",),
            cerberus_env=(("KEY", "val"),),
            review_timeout=450,
            max_epic_verification_retries=5,
        )
        overrides = CLIOverrides()  # All None
        resolved = build_resolved_config(base, overrides)

        assert resolved.cerberus_spawn_args == ("--base",)
        assert resolved.cerberus_wait_args == ("--wait-base",)
        assert resolved.cerberus_env == (("KEY", "val"),)
        assert resolved.review_timeout == 450
        assert resolved.max_epic_verification_retries == 5


class TestCerberusArgsParsing:
    """Tests for cerberus_spawn_args and cerberus_wait_args parsing."""

    def test_simple_args(self) -> None:
        """Parse simple space-separated args."""
        base = MalaConfig()
        overrides = CLIOverrides(cerberus_spawn_args="--foo bar --flag")
        resolved = build_resolved_config(base, overrides)
        assert resolved.cerberus_spawn_args == ("--foo", "bar", "--flag")

    def test_quoted_args_with_spaces(self) -> None:
        """Parse args with quoted strings containing spaces."""
        base = MalaConfig()
        overrides = CLIOverrides(cerberus_spawn_args='--message "hello world"')
        resolved = build_resolved_config(base, overrides)
        assert resolved.cerberus_spawn_args == ("--message", "hello world")

    def test_escaped_quotes(self) -> None:
        """Parse args with escaped quotes."""
        base = MalaConfig()
        overrides = CLIOverrides(cerberus_spawn_args='--msg "it\'s fine"')
        resolved = build_resolved_config(base, overrides)
        assert resolved.cerberus_spawn_args == ("--msg", "it's fine")

    def test_empty_string_gives_empty_tuple(self) -> None:
        """Empty string produces empty tuple."""
        base = MalaConfig()
        overrides = CLIOverrides(cerberus_spawn_args="")
        resolved = build_resolved_config(base, overrides)
        assert resolved.cerberus_spawn_args == ()

    def test_whitespace_only_gives_empty_tuple(self) -> None:
        """Whitespace-only string produces empty tuple."""
        base = MalaConfig()
        overrides = CLIOverrides(cerberus_spawn_args="   ")
        resolved = build_resolved_config(base, overrides)
        assert resolved.cerberus_spawn_args == ()

    def test_wait_args_parsed_same_as_spawn_args(self) -> None:
        """cerberus_wait_args uses same parsing as spawn_args."""
        base = MalaConfig()
        overrides = CLIOverrides(cerberus_wait_args='--timeout 60 --msg "wait"')
        resolved = build_resolved_config(base, overrides)
        assert resolved.cerberus_wait_args == ("--timeout", "60", "--msg", "wait")

    def test_invalid_quotes_raises_value_error(self) -> None:
        """Unbalanced quotes raise ValueError with CLI source."""
        base = MalaConfig()
        overrides = CLIOverrides(cerberus_spawn_args='"unclosed')
        with pytest.raises(ValueError) as exc_info:
            build_resolved_config(base, overrides)
        assert "CLI" in str(exc_info.value)


class TestCerberusEnvParsing:
    """Tests for cerberus_env parsing (JSON and KEY=VALUE formats)."""

    def test_json_format_simple(self) -> None:
        """Parse JSON object format."""
        base = MalaConfig()
        overrides = CLIOverrides(cerberus_env='{"FOO": "bar", "BAZ": "qux"}')
        resolved = build_resolved_config(base, overrides)
        assert dict(resolved.cerberus_env) == {"BAZ": "qux", "FOO": "bar"}

    def test_json_format_with_numbers(self) -> None:
        """JSON values converted to strings."""
        base = MalaConfig()
        overrides = CLIOverrides(cerberus_env='{"NUM": 42, "BOOL": true}')
        resolved = build_resolved_config(base, overrides)
        assert dict(resolved.cerberus_env) == {"BOOL": "True", "NUM": "42"}

    def test_comma_separated_kv_format(self) -> None:
        """Parse comma-separated KEY=VALUE format."""
        base = MalaConfig()
        overrides = CLIOverrides(cerberus_env="FOO=bar,BAZ=qux")
        resolved = build_resolved_config(base, overrides)
        assert dict(resolved.cerberus_env) == {"BAZ": "qux", "FOO": "bar"}

    def test_kv_format_with_equals_in_value(self) -> None:
        """Values can contain equals signs."""
        base = MalaConfig()
        overrides = CLIOverrides(cerberus_env="URL=http://host?a=1")
        resolved = build_resolved_config(base, overrides)
        assert dict(resolved.cerberus_env) == {"URL": "http://host?a=1"}

    def test_empty_string_gives_empty_tuple(self) -> None:
        """Empty string produces empty tuple."""
        base = MalaConfig()
        overrides = CLIOverrides(cerberus_env="")
        resolved = build_resolved_config(base, overrides)
        assert resolved.cerberus_env == ()

    def test_whitespace_only_gives_empty_tuple(self) -> None:
        """Whitespace-only string produces empty tuple."""
        base = MalaConfig()
        overrides = CLIOverrides(cerberus_env="   ")
        resolved = build_resolved_config(base, overrides)
        assert resolved.cerberus_env == ()

    def test_invalid_json_raises_value_error(self) -> None:
        """Invalid JSON raises ValueError with clear message."""
        base = MalaConfig()
        overrides = CLIOverrides(cerberus_env='{"invalid": }')
        with pytest.raises(ValueError) as exc_info:
            build_resolved_config(base, overrides)
        assert "CLI" in str(exc_info.value)
        assert "JSON" in str(exc_info.value)

    def test_json_non_object_raises_value_error(self) -> None:
        """JSON array raises ValueError."""
        base = MalaConfig()
        overrides = CLIOverrides(cerberus_env='["array", "not", "object"]')
        with pytest.raises(ValueError) as exc_info:
            build_resolved_config(base, overrides)
        # Array starting with [ is parsed as KEY=VALUE format, not JSON
        assert "KEY=VALUE" in str(exc_info.value)

    def test_kv_missing_equals_raises_value_error(self) -> None:
        """KEY=VALUE format without equals raises ValueError."""
        base = MalaConfig()
        overrides = CLIOverrides(cerberus_env="INVALID_ENTRY")
        with pytest.raises(ValueError) as exc_info:
            build_resolved_config(base, overrides)
        assert "KEY=VALUE" in str(exc_info.value)

    def test_kv_empty_key_raises_value_error(self) -> None:
        """Empty key in KEY=VALUE raises ValueError."""
        base = MalaConfig()
        overrides = CLIOverrides(cerberus_env="=value")
        with pytest.raises(ValueError) as exc_info:
            build_resolved_config(base, overrides)
        assert "empty key" in str(exc_info.value).lower()


class TestResolvedConfigImmutability:
    """Tests for ResolvedConfig frozen dataclass behavior."""

    def test_resolved_config_is_frozen(self) -> None:
        """ResolvedConfig is immutable after creation."""
        base = MalaConfig()
        resolved = build_resolved_config(base, None)

        with pytest.raises(AttributeError):
            resolved.runs_dir = Path("/new")  # type: ignore[misc]

    def test_resolved_config_is_hashable(self) -> None:
        """ResolvedConfig can be used in sets."""
        base1 = MalaConfig(review_timeout=100)
        base2 = MalaConfig(review_timeout=200)
        resolved1 = build_resolved_config(base1, None)
        resolved2 = build_resolved_config(base2, None)

        config_set = {resolved1, resolved2}
        assert len(config_set) == 2


class TestBuildResolvedConfigIdempotency:
    """Tests for idempotent and deterministic behavior."""

    def test_same_inputs_same_outputs(self, tmp_path: Path) -> None:
        """Same inputs produce identical outputs."""
        base = MalaConfig(
            runs_dir=tmp_path,
            cerberus_spawn_args=("--arg",),
        )
        overrides = CLIOverrides(cerberus_env="FOO=bar")

        resolved1 = build_resolved_config(base, overrides)
        resolved2 = build_resolved_config(base, overrides)

        assert resolved1 == resolved2

    def test_env_ordering_is_stable(self) -> None:
        """cerberus_env is sorted for stable ordering."""
        base = MalaConfig()
        overrides = CLIOverrides(cerberus_env="Z=1,A=2,M=3")
        resolved = build_resolved_config(base, overrides)

        # Should be sorted alphabetically
        assert resolved.cerberus_env == (("A", "2"), ("M", "3"), ("Z", "1"))


class TestBuildResolvedConfigReviewEnabled:
    """Tests for review_enabled derivation."""

    def test_review_disabled_by_cli_flag(self) -> None:
        """disable_review override disables review."""
        base = MalaConfig()  # review_enabled=True by default
        overrides = CLIOverrides(disable_review=True)
        resolved = build_resolved_config(base, overrides)

        assert resolved.review_enabled is False

    def test_review_enabled_when_not_disabled(self) -> None:
        """review remains enabled when not disabled."""
        base = MalaConfig()
        overrides = CLIOverrides(disable_review=False)
        resolved = build_resolved_config(base, overrides)

        assert resolved.review_enabled is True


class TestResolvedConfigClaudeSettingsSources:
    """Tests for claude_settings_sources in ResolvedConfig via build_resolved_config()."""

    def test_cli_overrides_env_and_yaml(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CLI overrides env var which overrides yaml (CLI wins)."""
        # Set env var (would be picked up by MalaConfig.from_env)
        monkeypatch.setenv("MALA_CLAUDE_SETTINGS_SOURCES", "project")
        base = MalaConfig.from_env(
            validate=False, yaml_claude_settings_sources=("user",)
        )
        # base now has ("project",) from env
        assert base.claude_settings_sources == ("project",)

        # CLI override should win
        overrides = CLIOverrides(claude_settings_sources="local")
        resolved = build_resolved_config(base, overrides)

        assert resolved.claude_settings_sources == ("local",)

    def test_env_overrides_yaml(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Env var overrides yaml when CLI not set (Env wins)."""
        monkeypatch.setenv("MALA_CLAUDE_SETTINGS_SOURCES", "project")
        base = MalaConfig.from_env(
            validate=False, yaml_claude_settings_sources=("user",)
        )
        # base has ("project",) from env
        assert base.claude_settings_sources == ("project",)

        overrides = CLIOverrides()  # No CLI override
        resolved = build_resolved_config(base, overrides)

        assert resolved.claude_settings_sources == ("project",)

    def test_yaml_used_when_env_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """YAML value used when env var not set and CLI not set (YAML wins)."""
        monkeypatch.delenv("MALA_CLAUDE_SETTINGS_SOURCES", raising=False)
        base = MalaConfig.from_env(
            validate=False, yaml_claude_settings_sources=("user",)
        )
        # base has ("user",) from yaml
        assert base.claude_settings_sources == ("user",)

        overrides = CLIOverrides()  # No CLI override
        resolved = build_resolved_config(base, overrides)

        assert resolved.claude_settings_sources == ("user",)

    def test_default_when_nothing_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default value used when CLI, env, and yaml are all not set."""
        monkeypatch.delenv("MALA_CLAUDE_SETTINGS_SOURCES", raising=False)
        base = MalaConfig.from_env(validate=False, yaml_claude_settings_sources=None)
        # base has default
        assert base.claude_settings_sources == DEFAULT_CLAUDE_SETTINGS_SOURCES

        overrides = CLIOverrides()  # No CLI override
        resolved = build_resolved_config(base, overrides)

        assert resolved.claude_settings_sources == ("local", "project")

    def test_parse_cli_multiple_sources(self) -> None:
        """CLI value with multiple sources is parsed correctly."""
        base = MalaConfig()
        overrides = CLIOverrides(claude_settings_sources="local,project,user")
        resolved = build_resolved_config(base, overrides)

        assert resolved.claude_settings_sources == ("local", "project", "user")

    def test_cli_whitespace_stripped(self) -> None:
        """CLI value with whitespace is stripped correctly."""
        base = MalaConfig()
        overrides = CLIOverrides(claude_settings_sources=" local , project ")
        resolved = build_resolved_config(base, overrides)

        assert resolved.claude_settings_sources == ("local", "project")

    def test_invalid_cli_source_raises(self) -> None:
        """Invalid source in CLI raises ValueError with clear message."""
        base = MalaConfig()
        overrides = CLIOverrides(claude_settings_sources="foo")

        with pytest.raises(ValueError) as exc_info:
            build_resolved_config(base, overrides)

        assert "CLI: Invalid source 'foo'" in str(exc_info.value)
        assert "Valid sources: local, project, user" in str(exc_info.value)

    def test_empty_cli_falls_back_to_base(self) -> None:
        """Empty CLI value falls back to base config."""
        base = MalaConfig(claude_settings_sources_init=("user",))
        overrides = CLIOverrides(claude_settings_sources="")
        resolved = build_resolved_config(base, overrides)

        assert resolved.claude_settings_sources == ("user",)
