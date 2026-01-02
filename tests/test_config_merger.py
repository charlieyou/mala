"""Unit tests for config_merger.py."""

from __future__ import annotations

import pytest

from src.domain.validation.config import (
    CommandConfig,
    CommandsConfig,
    ValidationConfig,
    YamlCoverageConfig,
)
from src.domain.validation.config_merger import DISABLED, merge_configs


class TestMergeConfigsNoPreset:
    """Tests for merge_configs when no preset is provided."""

    def test_no_preset_returns_user_config_unchanged(self) -> None:
        """When preset is None, user config is returned as-is."""
        user = ValidationConfig(
            commands=CommandsConfig(
                test=CommandConfig(command="pytest"),
            ),
            code_patterns=("**/*.py",),
        )
        result = merge_configs(None, user)
        assert result is user

    def test_no_preset_with_full_user_config(self) -> None:
        """User config with all fields is returned unchanged when no preset."""
        user = ValidationConfig(
            preset="python-uv",
            commands=CommandsConfig(
                setup=CommandConfig(command="uv sync"),
                test=CommandConfig(command="pytest", timeout=300),
                lint=CommandConfig(command="ruff check ."),
                format=CommandConfig(command="ruff format --check ."),
                typecheck=CommandConfig(command="ty check"),
                e2e=CommandConfig(command="pytest -m e2e"),
            ),
            coverage=YamlCoverageConfig(
                format="xml",
                file="coverage.xml",
                threshold=80.0,
            ),
            code_patterns=("**/*.py",),
            config_files=("pyproject.toml",),
            setup_files=("uv.lock",),
        )
        result = merge_configs(None, user)
        assert result is user


class TestMergeConfigsPresetWithNoUserOverrides:
    """Tests for merge_configs with preset but no user overrides."""

    def test_preset_commands_inherited(self) -> None:
        """Preset commands are used when user doesn't override."""
        preset = ValidationConfig(
            commands=CommandsConfig(
                test=CommandConfig(command="pytest"),
                lint=CommandConfig(command="ruff check ."),
            ),
        )
        user = ValidationConfig()  # No overrides
        result = merge_configs(preset, user)

        assert result.commands.test is not None
        assert result.commands.test.command == "pytest"
        assert result.commands.lint is not None
        assert result.commands.lint.command == "ruff check ."

    def test_preset_coverage_inherited(self) -> None:
        """Preset coverage is used when user doesn't provide it."""
        preset = ValidationConfig(
            coverage=YamlCoverageConfig(
                format="xml",
                file="coverage.xml",
                threshold=85.0,
            ),
        )
        user = ValidationConfig()
        result = merge_configs(preset, user)

        assert result.coverage is not None
        assert result.coverage.threshold == 85.0

    def test_preset_list_fields_inherited(self) -> None:
        """Preset list fields are used when user doesn't provide them."""
        preset = ValidationConfig(
            code_patterns=("**/*.py", "pyproject.toml"),
            config_files=("ruff.toml",),
            setup_files=("uv.lock",),
        )
        user = ValidationConfig()
        result = merge_configs(preset, user)

        assert result.code_patterns == ("**/*.py", "pyproject.toml")
        assert result.config_files == ("ruff.toml",)
        assert result.setup_files == ("uv.lock",)


class TestMergeConfigsUserOverridesPreset:
    """Tests for user overrides replacing preset values."""

    def test_user_command_replaces_preset(self) -> None:
        """User command replaces preset command."""
        preset = ValidationConfig(
            commands=CommandsConfig(
                test=CommandConfig(command="pytest"),
            ),
        )
        user = ValidationConfig(
            commands=CommandsConfig(
                test=CommandConfig(command="pytest -v --tb=short"),
            ),
        )
        result = merge_configs(preset, user)

        assert result.commands.test is not None
        assert result.commands.test.command == "pytest -v --tb=short"

    def test_user_command_with_timeout_replaces_preset(self) -> None:
        """User command with timeout replaces preset command."""
        preset = ValidationConfig(
            commands=CommandsConfig(
                test=CommandConfig(command="pytest"),
            ),
        )
        user = ValidationConfig(
            commands=CommandsConfig(
                test=CommandConfig(command="pytest", timeout=600),
            ),
        )
        result = merge_configs(preset, user)

        assert result.commands.test is not None
        assert result.commands.test.command == "pytest"
        assert result.commands.test.timeout == 600

    def test_user_coverage_replaces_preset_entirely(self) -> None:
        """User coverage configuration replaces preset entirely."""
        preset = ValidationConfig(
            coverage=YamlCoverageConfig(
                format="xml",
                file="preset-coverage.xml",
                threshold=85.0,
                command="pytest --cov",
            ),
        )
        user = ValidationConfig(
            coverage=YamlCoverageConfig(
                format="xml",
                file="user-coverage.xml",
                threshold=90.0,
            ),
        )
        result = merge_configs(preset, user)

        assert result.coverage is not None
        assert result.coverage.file == "user-coverage.xml"
        assert result.coverage.threshold == 90.0
        # User didn't specify command, but coverage is replaced entirely
        assert result.coverage.command is None

    def test_user_list_fields_replace_preset(self) -> None:
        """User list fields replace preset lists entirely (not extend)."""
        preset = ValidationConfig(
            code_patterns=("**/*.py", "**/*.pyx"),
            config_files=("pyproject.toml", "ruff.toml"),
            setup_files=("uv.lock", "requirements.txt"),
        )
        user = ValidationConfig(
            code_patterns=("src/**/*.py",),  # More specific
            config_files=("mypy.ini",),  # Different config
            setup_files=("poetry.lock",),  # Different lockfile
        )
        result = merge_configs(preset, user)

        # User lists replace preset lists - no merging
        assert result.code_patterns == ("src/**/*.py",)
        assert result.config_files == ("mypy.ini",)
        assert result.setup_files == ("poetry.lock",)


class TestMergeConfigsExplicitDisable:
    """Tests for explicitly disabling preset commands with DISABLED sentinel."""

    def test_disabled_sentinel_disables_preset_command(self) -> None:
        """DISABLED sentinel disables a command even if preset defines it."""
        preset = ValidationConfig(
            commands=CommandsConfig(
                lint=CommandConfig(command="ruff check ."),
            ),
        )
        # User explicitly disables lint
        user = ValidationConfig(
            commands=CommandsConfig(
                lint=DISABLED,  # type: ignore[arg-type]
            ),
        )
        result = merge_configs(preset, user)

        assert result.commands.lint is None

    def test_disabled_sentinel_for_multiple_commands(self) -> None:
        """Multiple commands can be disabled with DISABLED sentinel."""
        preset = ValidationConfig(
            commands=CommandsConfig(
                test=CommandConfig(command="pytest"),
                lint=CommandConfig(command="ruff check ."),
                typecheck=CommandConfig(command="mypy ."),
            ),
        )
        user = ValidationConfig(
            commands=CommandsConfig(
                lint=DISABLED,  # type: ignore[arg-type]
                typecheck=DISABLED,  # type: ignore[arg-type]
            ),
        )
        result = merge_configs(preset, user)

        # test is inherited
        assert result.commands.test is not None
        assert result.commands.test.command == "pytest"
        # lint and typecheck are disabled
        assert result.commands.lint is None
        assert result.commands.typecheck is None

    def test_disabled_sentinel_with_no_preset_command(self) -> None:
        """DISABLED on non-existent preset command results in None."""
        preset = ValidationConfig(
            commands=CommandsConfig(
                test=CommandConfig(command="pytest"),
            ),
        )
        user = ValidationConfig(
            commands=CommandsConfig(
                lint=DISABLED,  # type: ignore[arg-type]  # preset doesn't define lint
            ),
        )
        result = merge_configs(preset, user)

        assert result.commands.lint is None


class TestMergeConfigsExplicitDisableCoverage:
    """Tests for explicitly disabling preset coverage with DISABLED sentinel."""

    def test_disabled_sentinel_disables_preset_coverage(self) -> None:
        """DISABLED sentinel disables coverage even if preset defines it."""
        preset = ValidationConfig(
            coverage=YamlCoverageConfig(
                format="xml",
                file="coverage.xml",
                threshold=85.0,
            ),
        )
        # User explicitly disables coverage
        user = ValidationConfig(
            coverage=DISABLED,  # type: ignore[arg-type]
        )
        result = merge_configs(preset, user)

        assert result.coverage is None

    def test_disabled_coverage_with_no_preset_coverage(self) -> None:
        """DISABLED on non-existent preset coverage results in None."""
        preset = ValidationConfig(
            commands=CommandsConfig(
                test=CommandConfig(command="pytest"),
            ),
            # No coverage defined in preset
        )
        user = ValidationConfig(
            coverage=DISABLED,  # type: ignore[arg-type]
        )
        result = merge_configs(preset, user)

        assert result.coverage is None

    def test_disabled_coverage_with_other_overrides(self) -> None:
        """DISABLED coverage can coexist with other user overrides."""
        preset = ValidationConfig(
            commands=CommandsConfig(
                test=CommandConfig(command="pytest"),
                lint=CommandConfig(command="ruff check ."),
            ),
            coverage=YamlCoverageConfig(
                format="xml",
                file="coverage.xml",
                threshold=85.0,
            ),
            code_patterns=("**/*.py",),
        )
        user = ValidationConfig(
            commands=CommandsConfig(
                test=CommandConfig(command="pytest -v"),  # Override test
            ),
            coverage=DISABLED,  # type: ignore[arg-type]  # Disable coverage
            code_patterns=("src/**/*.py",),  # Override patterns
        )
        result = merge_configs(preset, user)

        # test overridden
        assert result.commands.test is not None
        assert result.commands.test.command == "pytest -v"
        # lint inherited
        assert result.commands.lint is not None
        assert result.commands.lint.command == "ruff check ."
        # coverage disabled
        assert result.coverage is None
        # patterns overridden
        assert result.code_patterns == ("src/**/*.py",)


class TestMergeConfigsOmittedInheritsPreset:
    """Tests for omitted fields inheriting preset values."""

    def test_omitted_command_inherits_from_preset(self) -> None:
        """When user omits a command, preset command is inherited."""
        preset = ValidationConfig(
            commands=CommandsConfig(
                setup=CommandConfig(command="uv sync"),
                test=CommandConfig(command="pytest"),
                lint=CommandConfig(command="ruff check ."),
            ),
        )
        user = ValidationConfig(
            commands=CommandsConfig(
                test=CommandConfig(command="pytest -v"),  # Only override test
            ),
        )
        result = merge_configs(preset, user)

        # setup and lint inherited from preset
        assert result.commands.setup is not None
        assert result.commands.setup.command == "uv sync"
        assert result.commands.lint is not None
        assert result.commands.lint.command == "ruff check ."
        # test overridden by user
        assert result.commands.test is not None
        assert result.commands.test.command == "pytest -v"

    def test_partial_user_config_preserves_preset_values(self) -> None:
        """Partial user config preserves all unspecified preset values."""
        preset = ValidationConfig(
            commands=CommandsConfig(
                setup=CommandConfig(command="npm install"),
                test=CommandConfig(command="npm test"),
                lint=CommandConfig(command="npm run lint"),
                format=CommandConfig(command="npm run format:check"),
                typecheck=CommandConfig(command="tsc --noEmit"),
                e2e=CommandConfig(command="npm run e2e"),
            ),
            coverage=YamlCoverageConfig(
                format="xml",
                file="coverage.xml",
                threshold=80.0,
            ),
            code_patterns=("**/*.ts", "**/*.tsx"),
            config_files=(".eslintrc.js",),
            setup_files=("package-lock.json",),
        )
        # User only overrides test command
        user = ValidationConfig(
            commands=CommandsConfig(
                test=CommandConfig(command="npm test -- --coverage"),
            ),
        )
        result = merge_configs(preset, user)

        # All preset values preserved except test
        assert result.commands.setup is not None
        assert result.commands.setup.command == "npm install"
        assert result.commands.test is not None
        assert result.commands.test.command == "npm test -- --coverage"
        assert result.commands.lint is not None
        assert result.commands.lint.command == "npm run lint"
        assert result.coverage is not None
        assert result.coverage.threshold == 80.0
        assert result.code_patterns == ("**/*.ts", "**/*.tsx")


class TestMergeConfigsComplexScenarios:
    """Complex merge scenarios combining multiple behaviors."""

    def test_mixed_inherit_override_disable(self) -> None:
        """Mix of inherited, overridden, and disabled commands."""
        preset = ValidationConfig(
            commands=CommandsConfig(
                setup=CommandConfig(command="make setup"),
                test=CommandConfig(command="make test"),
                lint=CommandConfig(command="make lint"),
                format=CommandConfig(command="make format-check"),
                typecheck=CommandConfig(command="make typecheck"),
            ),
        )
        user = ValidationConfig(
            commands=CommandsConfig(
                test=CommandConfig(command="pytest"),  # Override
                lint=DISABLED,  # type: ignore[arg-type]  # Disable
                # setup, format, typecheck: inherit
            ),
        )
        result = merge_configs(preset, user)

        # setup inherited
        assert result.commands.setup is not None
        assert result.commands.setup.command == "make setup"
        # test overridden
        assert result.commands.test is not None
        assert result.commands.test.command == "pytest"
        # lint disabled
        assert result.commands.lint is None
        # format inherited
        assert result.commands.format is not None
        assert result.commands.format.command == "make format-check"
        # typecheck inherited
        assert result.commands.typecheck is not None
        assert result.commands.typecheck.command == "make typecheck"

    def test_user_adds_command_preset_doesnt_have(self) -> None:
        """User can add commands that preset doesn't define."""
        preset = ValidationConfig(
            commands=CommandsConfig(
                test=CommandConfig(command="pytest"),
            ),
        )
        user = ValidationConfig(
            commands=CommandsConfig(
                e2e=CommandConfig(command="pytest -m e2e"),
            ),
        )
        result = merge_configs(preset, user)

        # test from preset
        assert result.commands.test is not None
        assert result.commands.test.command == "pytest"
        # e2e from user
        assert result.commands.e2e is not None
        assert result.commands.e2e.command == "pytest -m e2e"

    def test_user_preset_reference_preserved(self) -> None:
        """User's preset field is preserved in merged config."""
        preset = ValidationConfig(
            commands=CommandsConfig(
                test=CommandConfig(command="pytest"),
            ),
        )
        user = ValidationConfig(
            preset="python-uv",
        )
        result = merge_configs(preset, user)

        assert result.preset == "python-uv"


class TestMergeConfigsEdgeCases:
    """Edge cases and boundary conditions."""

    def test_both_empty_configs(self) -> None:
        """Merging two empty configs results in empty config."""
        preset = ValidationConfig()
        user = ValidationConfig()
        result = merge_configs(preset, user)

        assert result.commands.setup is None
        assert result.commands.test is None
        assert result.commands.lint is None
        assert result.commands.format is None
        assert result.commands.typecheck is None
        assert result.commands.e2e is None
        assert result.coverage is None
        assert result.code_patterns == ()
        assert result.config_files == ()
        assert result.setup_files == ()

    def test_preset_with_timeout_user_without_inherits_timeout(self) -> None:
        """When user overrides command, they must specify timeout if wanted."""
        preset = ValidationConfig(
            commands=CommandsConfig(
                test=CommandConfig(command="pytest", timeout=300),
            ),
        )
        user = ValidationConfig(
            commands=CommandsConfig(
                test=CommandConfig(command="pytest -v"),  # No timeout
            ),
        )
        result = merge_configs(preset, user)

        # User's command replaces entirely - timeout not inherited
        assert result.commands.test is not None
        assert result.commands.test.command == "pytest -v"
        assert result.commands.test.timeout is None

    def test_empty_tuple_list_fields_dont_replace(self) -> None:
        """Empty user list fields don't replace preset values."""
        preset = ValidationConfig(
            code_patterns=("**/*.py",),
            config_files=("pyproject.toml",),
            setup_files=("uv.lock",),
        )
        user = ValidationConfig(
            code_patterns=(),  # Empty
            config_files=(),  # Empty
            setup_files=(),  # Empty
        )
        result = merge_configs(preset, user)

        # Preset values preserved when user has empty tuples
        assert result.code_patterns == ("**/*.py",)
        assert result.config_files == ("pyproject.toml",)
        assert result.setup_files == ("uv.lock",)


class TestDisabledSentinel:
    """Tests for the DISABLED sentinel value itself."""

    def test_disabled_is_singleton(self) -> None:
        """DISABLED is a singleton instance."""
        from src.domain.validation.config_merger import DISABLED as DISABLED2

        assert DISABLED is DISABLED2

    def test_disabled_is_frozen(self) -> None:
        """DISABLED sentinel is immutable."""
        with pytest.raises(AttributeError):
            DISABLED.foo = "bar"  # type: ignore[attr-defined]

    def test_disabled_is_not_none(self) -> None:
        """DISABLED sentinel is distinct from None."""
        assert DISABLED is not None
        assert DISABLED != None  # noqa: E711
