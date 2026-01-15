"""Unit tests for config_merger.py."""

from __future__ import annotations


from src.domain.validation.config import (
    CommandConfig,
    CommandsConfig,
    ValidationConfig,
    YamlCoverageConfig,
)
from src.domain.validation.config_merger import merge_configs


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
                build=CommandConfig(command="python -m build"),
                test=CommandConfig(command="pytest"),
                lint=CommandConfig(command="ruff check ."),
            ),
        )
        user = ValidationConfig()  # No overrides
        result = merge_configs(preset, user)

        assert result.commands.build is not None
        assert result.commands.build.command == "python -m build"
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
        # User explicitly sets test command (tracked via _fields_set)
        user = ValidationConfig.from_dict(
            {"commands": {"test": "pytest -v --tb=short"}}
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
        user = ValidationConfig.from_dict(
            {"commands": {"test": {"command": "pytest", "timeout": 600}}}
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
        user = ValidationConfig.from_dict(
            {
                "coverage": {
                    "format": "xml",
                    "file": "user-coverage.xml",
                    "threshold": 90.0,
                }
            }
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
        user = ValidationConfig.from_dict(
            {
                "code_patterns": ["src/**/*.py"],
                "config_files": ["mypy.ini"],
                "setup_files": ["poetry.lock"],
            }
        )
        result = merge_configs(preset, user)

        # User lists replace preset lists - no merging
        assert result.code_patterns == ("src/**/*.py",)
        assert result.config_files == ("mypy.ini",)
        assert result.setup_files == ("poetry.lock",)


class TestMergeConfigsExplicitDisable:
    """Tests for explicitly disabling preset values with null."""

    def test_explicit_null_disables_preset_command(self) -> None:
        """Explicit null disables a command even if preset defines it."""
        preset = ValidationConfig(
            commands=CommandsConfig(
                lint=CommandConfig(command="ruff check ."),
            ),
        )
        # User explicitly sets lint to null (tracked in _fields_set)
        user = ValidationConfig.from_dict({"commands": {"lint": None}})
        result = merge_configs(preset, user)

        assert result.commands.lint is None

    def test_explicit_null_for_multiple_commands(self) -> None:
        """Multiple commands can be disabled with explicit null."""
        preset = ValidationConfig(
            commands=CommandsConfig(
                test=CommandConfig(command="pytest"),
                lint=CommandConfig(command="ruff check ."),
                typecheck=CommandConfig(command="mypy ."),
            ),
        )
        user = ValidationConfig.from_dict(
            {"commands": {"lint": None, "typecheck": None}}
        )
        result = merge_configs(preset, user)

        # test is inherited
        assert result.commands.test is not None
        assert result.commands.test.command == "pytest"
        # lint and typecheck are disabled
        assert result.commands.lint is None
        assert result.commands.typecheck is None

    def test_explicit_null_with_no_preset_command(self) -> None:
        """Explicit null on non-existent preset command results in None."""
        preset = ValidationConfig(
            commands=CommandsConfig(
                test=CommandConfig(command="pytest"),
            ),
        )
        user = ValidationConfig.from_dict(
            {"commands": {"lint": None}}  # preset doesn't define lint
        )
        result = merge_configs(preset, user)

        assert result.commands.lint is None

    def test_explicit_null_disables_preset_coverage(self) -> None:
        """Explicit null disables coverage even if preset defines it."""
        preset = ValidationConfig(
            coverage=YamlCoverageConfig(
                format="xml",
                file="coverage.xml",
                threshold=80.0,
            ),
        )
        # User explicitly sets coverage to null (tracked in _fields_set)
        user = ValidationConfig.from_dict({"coverage": None})
        result = merge_configs(preset, user)

        assert result.coverage is None

    def test_explicit_empty_list_clears_preset_patterns(self) -> None:
        """Explicit empty list clears preset patterns."""
        preset = ValidationConfig(
            code_patterns=("**/*.py",),
            config_files=("pyproject.toml",),
            setup_files=("uv.lock",),
        )
        # User explicitly sets to empty lists
        user = ValidationConfig.from_dict(
            {
                "code_patterns": [],
                "config_files": [],
                "setup_files": [],
            }
        )
        result = merge_configs(preset, user)

        # Empty lists are used because user explicitly set them
        assert result.code_patterns == ()
        assert result.config_files == ()
        assert result.setup_files == ()


class TestMergeConfigsExplicitDisableCoverage:
    """Tests for explicitly disabling preset coverage with null."""

    def test_explicit_null_disables_preset_coverage(self) -> None:
        """Explicit null disables coverage even if preset defines it."""
        preset = ValidationConfig(
            coverage=YamlCoverageConfig(
                format="xml",
                file="coverage.xml",
                threshold=85.0,
            ),
        )
        # User explicitly sets coverage to null
        user = ValidationConfig.from_dict({"coverage": None})
        result = merge_configs(preset, user)

        assert result.coverage is None

    def test_explicit_null_coverage_with_no_preset_coverage(self) -> None:
        """Explicit null on non-existent preset coverage results in None."""
        preset = ValidationConfig(
            commands=CommandsConfig(
                test=CommandConfig(command="pytest"),
            ),
            # No coverage defined in preset
        )
        user = ValidationConfig.from_dict({"coverage": None})
        result = merge_configs(preset, user)

        assert result.coverage is None

    def test_explicit_null_coverage_with_other_overrides(self) -> None:
        """Explicit null coverage can coexist with other user overrides."""
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
        user = ValidationConfig.from_dict(
            {
                "commands": {"test": "pytest -v"},  # Override test
                "coverage": None,  # Disable coverage
                "code_patterns": ["src/**/*.py"],  # Override patterns
            }
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
        user = ValidationConfig.from_dict(
            {"commands": {"test": "pytest -v"}}  # Only override test
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
        user = ValidationConfig.from_dict(
            {"commands": {"test": "npm test -- --coverage"}}
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
        user = ValidationConfig.from_dict(
            {
                "commands": {
                    "test": "pytest",  # Override
                    "lint": None,  # Disable
                    # setup, format, typecheck: inherit
                }
            }
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
        user = ValidationConfig.from_dict({"commands": {"e2e": "pytest -m e2e"}})
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
        user = ValidationConfig.from_dict({"preset": "python-uv"})
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
        user = ValidationConfig.from_dict(
            {"commands": {"test": "pytest -v"}}  # No timeout
        )
        result = merge_configs(preset, user)

        # User's command replaces entirely - timeout not inherited
        assert result.commands.test is not None
        assert result.commands.test.command == "pytest -v"
        assert result.commands.test.timeout is None

    def test_empty_tuple_list_fields_without_explicit_set_dont_replace(self) -> None:
        """Empty user list fields without explicit set don't replace preset values."""
        preset = ValidationConfig(
            code_patterns=("**/*.py",),
            config_files=("pyproject.toml",),
            setup_files=("uv.lock",),
        )
        # User config created without from_dict, so _fields_set is empty
        user = ValidationConfig(
            code_patterns=(),  # Empty
            config_files=(),  # Empty
            setup_files=(),  # Empty
        )
        result = merge_configs(preset, user)

        # Preset values preserved when user has empty tuples but didn't explicitly set them
        assert result.code_patterns == ("**/*.py",)
        assert result.config_files == ("pyproject.toml",)
        assert result.setup_files == ("uv.lock",)


class TestFieldsSetTracking:
    """Tests for _fields_set tracking in from_dict."""

    def test_from_dict_tracks_present_fields(self) -> None:
        """from_dict tracks which fields were explicitly present."""
        config = ValidationConfig.from_dict(
            {
                "preset": "python-uv",
                "commands": {"test": "pytest"},
                "code_patterns": ["**/*.py"],
            }
        )

        assert "preset" in config._fields_set
        assert "commands" in config._fields_set
        assert "code_patterns" in config._fields_set
        assert "coverage" not in config._fields_set
        assert "config_files" not in config._fields_set
        assert "setup_files" not in config._fields_set

    def test_from_dict_tracks_null_fields(self) -> None:
        """from_dict tracks fields explicitly set to null."""
        config = ValidationConfig.from_dict(
            {
                "coverage": None,  # Explicit null
            }
        )

        assert "coverage" in config._fields_set
        assert config.coverage is None

    def test_from_dict_tracks_empty_list_fields(self) -> None:
        """from_dict tracks fields explicitly set to empty list."""
        config = ValidationConfig.from_dict(
            {
                "code_patterns": [],  # Explicit empty list
            }
        )

        assert "code_patterns" in config._fields_set
        assert config.code_patterns == ()

    def test_commands_from_dict_tracks_present_fields(self) -> None:
        """CommandsConfig.from_dict tracks which fields were explicitly present."""
        commands = CommandsConfig.from_dict(
            {
                "test": "pytest",
                "lint": None,  # Explicit null
            }
        )

        assert "test" in commands._fields_set
        assert "lint" in commands._fields_set
        assert "setup" not in commands._fields_set
        assert "format" not in commands._fields_set

    def test_empty_dict_has_empty_fields_set(self) -> None:
        """from_dict with empty dict has empty _fields_set."""
        config = ValidationConfig.from_dict({})

        assert config._fields_set == frozenset()


class TestMergeConfigsFieldsSetPreservation:
    """Tests that merged configs preserve _fields_set correctly."""

    def test_merged_config_preserves_user_fields_set(self) -> None:
        """Merged config preserves user's _fields_set."""
        preset = ValidationConfig(
            commands=CommandsConfig(
                test=CommandConfig(command="pytest"),
            ),
        )
        user = ValidationConfig.from_dict(
            {
                "commands": {"lint": "ruff check ."},
                "code_patterns": ["src/**/*.py"],
            }
        )

        result = merge_configs(preset, user)

        # Merged config has user's _fields_set
        assert "commands" in result._fields_set
        assert "code_patterns" in result._fields_set
        assert "coverage" not in result._fields_set

    def test_merged_commands_preserves_user_fields_set(self) -> None:
        """Merged commands preserve user's _fields_set."""
        preset = ValidationConfig(
            commands=CommandsConfig(
                test=CommandConfig(command="pytest"),
                lint=CommandConfig(command="pylint"),
            ),
        )
        user = ValidationConfig.from_dict({"commands": {"lint": "ruff check ."}})

        result = merge_configs(preset, user)

        # Merged commands has user's _fields_set
        assert "lint" in result.commands._fields_set
        assert "test" not in result.commands._fields_set


class TestExplicitCommandsNullOrEmptyInheritsPreset:
    """Tests for commands: null or commands: {} inheriting preset commands."""

    def test_explicit_commands_null_inherits_all_preset_commands(self) -> None:
        """Setting commands: null inherits preset commands."""
        preset = ValidationConfig(
            commands=CommandsConfig(
                setup=CommandConfig(command="uv sync"),
                test=CommandConfig(command="pytest"),
                lint=CommandConfig(command="ruff check ."),
                format=CommandConfig(command="ruff format --check"),
                typecheck=CommandConfig(command="ty check"),
                e2e=CommandConfig(command="pytest -m e2e"),
            ),
        )
        # User explicitly sets commands to null (simulates `commands: null` in YAML)
        user = ValidationConfig.from_dict({"commands": None})
        result = merge_configs(preset, user)

        # All commands should be inherited
        assert result.commands.setup is not None
        assert result.commands.setup.command == "uv sync"
        assert result.commands.test is not None
        assert result.commands.test.command == "pytest"
        assert result.commands.lint is not None
        assert result.commands.lint.command == "ruff check ."
        assert result.commands.format is not None
        assert result.commands.format.command == "ruff format --check"
        assert result.commands.typecheck is not None
        assert result.commands.typecheck.command == "ty check"
        assert result.commands.e2e is not None
        assert result.commands.e2e.command == "pytest -m e2e"

    def test_explicit_commands_empty_object_inherits_all_preset_commands(self) -> None:
        """Setting commands: {} inherits preset commands."""
        preset = ValidationConfig(
            commands=CommandsConfig(
                setup=CommandConfig(command="npm install"),
                test=CommandConfig(command="npm test"),
                lint=CommandConfig(command="npm run lint"),
            ),
        )
        # User explicitly sets commands to empty object (simulates `commands: {}`)
        user = ValidationConfig.from_dict({"commands": {}})
        result = merge_configs(preset, user)

        # All commands should be inherited
        assert result.commands.setup is not None
        assert result.commands.setup.command == "npm install"
        assert result.commands.test is not None
        assert result.commands.test.command == "npm test"
        assert result.commands.lint is not None
        assert result.commands.lint.command == "npm run lint"
        assert result.commands.format is None
        assert result.commands.typecheck is None
        assert result.commands.e2e is None

    def test_explicit_commands_empty_with_other_fields_still_works(self) -> None:
        """commands: {} with other fields still inherits commands while preserving others."""
        preset = ValidationConfig(
            commands=CommandsConfig(
                test=CommandConfig(command="pytest"),
                lint=CommandConfig(command="ruff check ."),
            ),
            code_patterns=("**/*.py",),
            coverage=YamlCoverageConfig(
                format="xml",
                file="coverage.xml",
                threshold=80.0,
            ),
        )
        # User explicitly sets commands to empty but provides other overrides
        user = ValidationConfig.from_dict(
            {
                "commands": {},  # Clear all commands
                "code_patterns": ["src/**/*.py"],  # Override patterns
            }
        )
        result = merge_configs(preset, user)

        # Commands should be inherited
        assert result.commands.setup is None
        assert result.commands.test is not None
        assert result.commands.test.command == "pytest"
        assert result.commands.lint is not None
        assert result.commands.lint.command == "ruff check ."

        # Other fields work normally
        assert result.code_patterns == ("src/**/*.py",)  # User override
        assert result.coverage is not None  # Inherited from preset
        assert result.coverage.threshold == 80.0

    def test_commands_with_only_one_field_still_inherits_others(self) -> None:
        """commands with at least one field still inherits unspecified commands.

        This verifies the short-circuit only applies when NO command fields are set.
        """
        preset = ValidationConfig(
            commands=CommandsConfig(
                setup=CommandConfig(command="uv sync"),
                test=CommandConfig(command="pytest"),
                lint=CommandConfig(command="ruff check ."),
            ),
        )
        # User sets only test - other commands should still inherit
        user = ValidationConfig.from_dict({"commands": {"test": "pytest -v"}})
        result = merge_configs(preset, user)

        # Only test is overridden; others inherit
        assert result.commands.setup is not None
        assert result.commands.setup.command == "uv sync"
        assert result.commands.test is not None
        assert result.commands.test.command == "pytest -v"
        assert result.commands.lint is not None
        assert result.commands.lint.command == "ruff check ."

    def test_commands_null_does_not_affect_preset_coverage(self) -> None:
        """Setting commands: null does not affect preset coverage inheritance."""
        preset = ValidationConfig(
            commands=CommandsConfig(
                test=CommandConfig(command="pytest"),
            ),
            coverage=YamlCoverageConfig(
                format="xml",
                file="coverage.xml",
                threshold=85.0,
            ),
        )
        user = ValidationConfig.from_dict({"commands": None})
        result = merge_configs(preset, user)

        # Commands inherited
        assert result.commands.test is not None
        assert result.commands.test.command == "pytest"
        # Coverage still inherited
        assert result.coverage is not None
        assert result.coverage.threshold == 85.0


class TestProgrammaticConfigOverrides:
    """Tests for programmatic (non-YAML) config overrides.

    When configs are created programmatically (via constructor instead of from_dict),
    _fields_set is empty. Non-None values should still override preset values.
    This ensures programmatic configs behave correctly and don't silently inherit.
    """

    def test_programmatic_coverage_overrides_preset(self) -> None:
        """Programmatic coverage config should override preset coverage."""
        preset = ValidationConfig(
            coverage=YamlCoverageConfig(
                format="xml",
                file="preset-coverage.xml",
                threshold=85.0,
            ),
        )
        # Programmatic config - _fields_set is empty but coverage is non-None
        user = ValidationConfig(
            coverage=YamlCoverageConfig(
                format="xml",
                file="user-coverage.xml",
                threshold=90.0,
            ),
        )
        result = merge_configs(preset, user)

        # User's programmatic coverage should override preset
        assert result.coverage is not None
        assert result.coverage.file == "user-coverage.xml"
        assert result.coverage.threshold == 90.0

    def test_programmatic_commands_override_preset(self) -> None:
        """Programmatic commands should override preset commands."""
        preset = ValidationConfig(
            commands=CommandsConfig(
                test=CommandConfig(command="pytest"),
                lint=CommandConfig(command="pylint"),
            ),
        )
        # Programmatic config with different test command
        user = ValidationConfig(
            commands=CommandsConfig(
                test=CommandConfig(command="pytest -v"),
            ),
        )
        result = merge_configs(preset, user)

        # User's test command overrides preset
        assert result.commands.test is not None
        assert result.commands.test.command == "pytest -v"
        # lint should inherit from preset (user didn't set it)
        assert result.commands.lint is not None
        assert result.commands.lint.command == "pylint"

    def test_programmatic_code_patterns_override_preset(self) -> None:
        """Programmatic code_patterns should override preset."""
        preset = ValidationConfig(
            code_patterns=("**/*.py", "**/*.pyx"),
        )
        # Programmatic config with different patterns
        user = ValidationConfig(
            code_patterns=("src/**/*.py",),
        )
        result = merge_configs(preset, user)

        # User's patterns should override preset
        assert result.code_patterns == ("src/**/*.py",)

    def test_programmatic_config_with_multiple_fields(self) -> None:
        """Programmatic config with multiple fields all override preset."""
        preset = ValidationConfig(
            commands=CommandsConfig(
                test=CommandConfig(command="pytest"),
            ),
            coverage=YamlCoverageConfig(
                format="xml",
                file="coverage.xml",
                threshold=80.0,
            ),
            code_patterns=("**/*.py",),
        )
        # Programmatic config overriding multiple fields
        user = ValidationConfig(
            commands=CommandsConfig(
                test=CommandConfig(command="pytest -v"),
                lint=CommandConfig(command="ruff check ."),
            ),
            coverage=YamlCoverageConfig(
                format="xml",
                file="new-coverage.xml",
                threshold=90.0,
            ),
            code_patterns=("src/**/*.py",),
        )
        result = merge_configs(preset, user)

        # All user values override preset
        assert result.commands.test is not None
        assert result.commands.test.command == "pytest -v"
        assert result.commands.lint is not None
        assert result.commands.lint.command == "ruff check ."
        assert result.coverage is not None
        assert result.coverage.file == "new-coverage.xml"
        assert result.coverage.threshold == 90.0
        assert result.code_patterns == ("src/**/*.py",)

    def test_programmatic_config_merge_works(self) -> None:
        """Verify programmatic configs (direct constructor) merge correctly.

        When configs are created programmatically rather than from_dict,
        non-default values are treated as explicit overrides.
        """
        preset_cfg = ValidationConfig(
            commands=CommandsConfig(
                test=CommandConfig(command="pytest"),
                lint=CommandConfig(command="ruff check"),
            ),
            code_patterns=("**/*.py",),
        )
        user_cfg = ValidationConfig(
            commands=CommandsConfig(
                test=CommandConfig(command="pytest -v"),  # override
            ),
        )
        result = merge_configs(preset_cfg, user_cfg)

        assert result.commands.test is not None
        assert result.commands.test.command == "pytest -v"
        assert result.commands.lint is not None
        assert result.commands.lint.command == "ruff check"

    def test_programmatic_empty_commands_inherits_preset(self) -> None:
        """Empty programmatic CommandsConfig should inherit all preset commands.

        Unlike from_dict({}) which clears commands, an empty constructor
        means "no fields set" and should inherit.
        """
        preset = ValidationConfig(
            commands=CommandsConfig(
                test=CommandConfig(command="pytest"),
                lint=CommandConfig(command="ruff check ."),
            ),
        )
        # Empty CommandsConfig - should inherit all commands
        user = ValidationConfig(
            commands=CommandsConfig(),  # No commands set
        )
        result = merge_configs(preset, user)

        # All commands inherited
        assert result.commands.test is not None
        assert result.commands.test.command == "pytest"
        assert result.commands.lint is not None
        assert result.commands.lint.command == "ruff check ."

    def test_programmatic_mixed_with_yaml_config(self) -> None:
        """Programmatic preset with YAML-parsed user config."""
        # Preset could be loaded programmatically (e.g., built-in presets)
        preset = ValidationConfig(
            commands=CommandsConfig(
                setup=CommandConfig(command="uv sync"),
                test=CommandConfig(command="pytest"),
                lint=CommandConfig(command="ruff check ."),
                typecheck=CommandConfig(command="ty check"),
            ),
            code_patterns=("**/*.py",),
        )
        # User config from YAML
        user = ValidationConfig.from_dict(
            {
                "commands": {"test": "pytest -v"},
                "coverage": {
                    "format": "xml",
                    "file": "coverage.xml",
                    "threshold": 85.0,
                },
            }
        )
        result = merge_configs(preset, user)

        # Test overridden
        assert result.commands.test is not None
        assert result.commands.test.command == "pytest -v"
        # Others inherited
        assert result.commands.setup is not None
        assert result.commands.setup.command == "uv sync"
        assert result.commands.lint is not None
        assert result.commands.lint.command == "ruff check ."
        # User's coverage
        assert result.coverage is not None
        assert result.coverage.threshold == 85.0


class TestValidationTriggersNotInheritedFromPreset:
    """Tests that validation_triggers are NOT inherited from preset.

    Presets define commands, NOT validation_triggers. Users must configure
    triggers themselves.

    This ensures triggers are always project-defined, never inherited from presets.
    """

    def test_preset_triggers_not_inherited_when_user_has_none(self) -> None:
        """User without validation_triggers does NOT inherit preset triggers."""
        from src.domain.validation.config import (
            FailureMode,
            SessionEndTriggerConfig,
            ValidationTriggersConfig,
        )

        # Preset has triggers defined (hypothetically - presets shouldn't but we test defense)
        preset = ValidationConfig(
            commands=CommandsConfig(test=CommandConfig(command="pytest")),
            validation_triggers=ValidationTriggersConfig(
                session_end=SessionEndTriggerConfig(
                    failure_mode=FailureMode.CONTINUE,
                    commands=(),
                ),
            ),
        )
        # User does not set validation_triggers
        user = ValidationConfig(
            commands=CommandsConfig(),
            _fields_set=frozenset({"commands"}),
        )

        result = merge_configs(preset, user)

        # Result should NOT have triggers - they are never inherited
        assert result.validation_triggers is None

    def test_user_triggers_preserved_when_preset_has_none(self) -> None:
        """User's validation_triggers preserved when preset has none."""
        from src.domain.validation.config import (
            FailureMode,
            SessionEndTriggerConfig,
            ValidationTriggersConfig,
        )

        preset = ValidationConfig(
            commands=CommandsConfig(test=CommandConfig(command="pytest")),
            validation_triggers=None,  # Preset has no triggers (correct behavior)
        )
        user = ValidationConfig(
            validation_triggers=ValidationTriggersConfig(
                session_end=SessionEndTriggerConfig(
                    failure_mode=FailureMode.ABORT,
                    commands=(),
                ),
            ),
            _fields_set=frozenset({"validation_triggers"}),
        )

        result = merge_configs(preset, user)

        # User's triggers should be preserved
        assert result.validation_triggers is not None
        assert result.validation_triggers.session_end is not None
        assert result.validation_triggers.session_end.failure_mode == FailureMode.ABORT

    def test_user_triggers_override_preset_triggers(self) -> None:
        """User's validation_triggers always take precedence over preset."""
        from src.domain.validation.config import (
            FailureMode,
            SessionEndTriggerConfig,
            ValidationTriggersConfig,
        )

        # Preset has triggers (hypothetically)
        preset = ValidationConfig(
            commands=CommandsConfig(test=CommandConfig(command="pytest")),
            validation_triggers=ValidationTriggersConfig(
                session_end=SessionEndTriggerConfig(
                    failure_mode=FailureMode.CONTINUE,
                    commands=(),
                ),
            ),
        )
        # User has different triggers
        user = ValidationConfig(
            validation_triggers=ValidationTriggersConfig(
                session_end=SessionEndTriggerConfig(
                    failure_mode=FailureMode.ABORT,
                    commands=(),
                ),
            ),
            _fields_set=frozenset({"validation_triggers"}),
        )

        result = merge_configs(preset, user)

        # User's triggers should be used
        assert result.validation_triggers is not None
        assert result.validation_triggers.session_end is not None
        assert result.validation_triggers.session_end.failure_mode == FailureMode.ABORT

    def test_user_explicit_none_triggers_not_overridden_by_preset(self) -> None:
        """User's explicit null for validation_triggers is not overridden by preset."""
        from src.domain.validation.config import (
            FailureMode,
            SessionEndTriggerConfig,
            ValidationTriggersConfig,
        )

        # Preset has triggers (hypothetically)
        preset = ValidationConfig(
            commands=CommandsConfig(test=CommandConfig(command="pytest")),
            validation_triggers=ValidationTriggersConfig(
                session_end=SessionEndTriggerConfig(
                    failure_mode=FailureMode.CONTINUE,
                    commands=(),
                ),
            ),
        )
        # User explicitly sets validation_triggers to None
        user = ValidationConfig(
            validation_triggers=None,
            _fields_set=frozenset({"validation_triggers"}),  # Explicitly set
        )

        result = merge_configs(preset, user)

        # User's explicit None should be preserved
        assert result.validation_triggers is None


class TestTimeoutMinutesPreservedInMerge:
    """Tests that timeout_minutes is preserved during preset merge."""

    def test_user_timeout_minutes_preserved_in_merge(self) -> None:
        """User's timeout_minutes is preserved when merging with preset."""
        preset = ValidationConfig(
            commands=CommandsConfig(test=CommandConfig(command="pytest")),
        )
        user = ValidationConfig(
            timeout_minutes=30,
            _fields_set=frozenset({"timeout_minutes"}),
        )

        result = merge_configs(preset, user)

        assert result.timeout_minutes == 30

    def test_timeout_minutes_none_when_user_does_not_set(self) -> None:
        """timeout_minutes is None when user does not set it."""
        preset = ValidationConfig(
            commands=CommandsConfig(test=CommandConfig(command="pytest")),
        )
        user = ValidationConfig(
            _fields_set=frozenset(),
        )

        result = merge_configs(preset, user)

        assert result.timeout_minutes is None
