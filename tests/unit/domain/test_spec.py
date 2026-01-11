"""Unit tests for src/validation/spec.py - ValidationSpec and related types.

TDD tests for:
- ValidationSpec construction from config files
- Code vs docs classification
- Disable list handling
- Config-driven spec building
"""

import shutil
from pathlib import Path


from src.domain.validation.config import (
    CommandConfig,
    CommandsConfig,
    CustomCommandConfig,
)
from src.domain.validation.spec import (
    DEFAULT_COMMAND_TIMEOUT,
    CommandKind,
    CoverageConfig,
    E2EConfig,
    ValidationCommand,
    ValidationScope,
    ValidationSpec,
    _build_commands_from_config,
    build_validation_spec,
    classify_change,
)


class TestValidationCommand:
    """Test ValidationCommand dataclass."""

    def test_basic_command(self) -> None:
        cmd = ValidationCommand(
            name="pytest",
            command="uv run pytest",
            kind=CommandKind.TEST,
        )
        assert cmd.name == "pytest"
        assert cmd.command == "uv run pytest"
        assert cmd.kind == CommandKind.TEST
        assert cmd.shell is True  # default
        assert cmd.timeout == DEFAULT_COMMAND_TIMEOUT  # default 120
        assert cmd.use_test_mutex is False  # default
        assert cmd.allow_fail is False  # default

    def test_command_with_mutex(self) -> None:
        cmd = ValidationCommand(
            name="pytest",
            command="uv run pytest",
            kind=CommandKind.TEST,
            use_test_mutex=True,
        )
        assert cmd.use_test_mutex is True

    def test_command_allow_fail(self) -> None:
        cmd = ValidationCommand(
            name="ty check",
            command="uvx ty check",
            kind=CommandKind.TYPECHECK,
            allow_fail=True,
        )
        assert cmd.allow_fail is True

    def test_command_with_custom_timeout(self) -> None:
        cmd = ValidationCommand(
            name="pytest",
            command="uv run pytest",
            kind=CommandKind.TEST,
            timeout=300,
        )
        assert cmd.timeout == 300


class TestValidationSpec:
    """Test ValidationSpec dataclass."""

    def test_minimal_spec(self) -> None:
        spec = ValidationSpec(
            commands=[],
            scope=ValidationScope.PER_SESSION,
        )
        assert spec.commands == []
        assert spec.require_clean_git is True  # default
        assert spec.require_pytest_for_code_changes is True  # default
        assert spec.scope == ValidationScope.PER_SESSION
        assert spec.coverage is not None
        assert spec.e2e is not None
        assert spec.code_patterns == []
        assert spec.config_files == []
        assert spec.setup_files == []

    def test_full_spec(self) -> None:
        cmd = ValidationCommand(
            name="pytest",
            command="uv run pytest",
            kind=CommandKind.TEST,
        )
        spec = ValidationSpec(
            commands=[cmd],
            require_clean_git=True,
            require_pytest_for_code_changes=True,
            coverage=CoverageConfig(enabled=True, min_percent=90.0),
            e2e=E2EConfig(enabled=False),
            scope=ValidationScope.GLOBAL,
            code_patterns=["**/*.py"],
            config_files=["pyproject.toml"],
            setup_files=["uv.lock"],
        )
        assert len(spec.commands) == 1
        assert spec.coverage.min_percent == 90.0
        assert spec.e2e.enabled is False
        assert spec.code_patterns == ["**/*.py"]
        assert spec.config_files == ["pyproject.toml"]
        assert spec.setup_files == ["uv.lock"]

    def test_commands_by_kind(self) -> None:
        lint_cmd = ValidationCommand(
            name="ruff check", command="uvx ruff check", kind=CommandKind.LINT
        )
        test_cmd = ValidationCommand(
            name="pytest", command="uv run pytest", kind=CommandKind.TEST
        )
        spec = ValidationSpec(
            commands=[lint_cmd, test_cmd],
            scope=ValidationScope.PER_SESSION,
        )

        lint_cmds = spec.commands_by_kind(CommandKind.LINT)
        assert len(lint_cmds) == 1
        assert lint_cmds[0].name == "ruff check"

        test_cmds = spec.commands_by_kind(CommandKind.TEST)
        assert len(test_cmds) == 1
        assert test_cmds[0].name == "pytest"

        e2e_cmds = spec.commands_by_kind(CommandKind.E2E)
        assert len(e2e_cmds) == 0


class TestClassifyChange:
    """Test code vs docs classification helper."""

    def test_python_files_are_code(self) -> None:
        assert classify_change("src/app.py") == "code"
        assert classify_change("tests/test_app.py") == "code"

    def test_shell_scripts_are_code(self) -> None:
        assert classify_change("scripts/deploy.sh") == "code"

    def test_config_files_are_code(self) -> None:
        assert classify_change("pyproject.toml") == "code"
        assert classify_change("config/settings.toml") == "code"
        assert classify_change(".env.template") == "code"

    def test_yaml_files_are_code(self) -> None:
        assert classify_change(".github/workflows/ci.yml") == "code"
        assert classify_change("config/settings.yaml") == "code"

    def test_json_files_are_code(self) -> None:
        assert classify_change("package.json") == "code"
        assert classify_change("config.json") == "code"

    def test_markdown_files_are_docs(self) -> None:
        assert classify_change("README.md") == "docs"
        assert classify_change("docs/guide.md") == "docs"

    def test_rst_files_are_docs(self) -> None:
        assert classify_change("docs/index.rst") == "docs"

    def test_txt_files_are_docs(self) -> None:
        assert classify_change("CHANGELOG.txt") == "docs"

    def test_code_paths_are_code(self) -> None:
        # Paths under src/, tests/, commands/, src/scripts/ are code
        assert classify_change("src/anything.xyz") == "code"
        assert classify_change("tests/conftest.py") == "code"
        assert classify_change("commands/deploy.py") == "code"

    def test_unknown_extension_defaults_to_docs(self) -> None:
        # Files with unknown extensions outside code paths default to docs
        assert classify_change("data/records.csv") == "docs"
        assert classify_change("notes.unknown") == "docs"

    def test_uv_lock_is_code(self) -> None:
        assert classify_change("uv.lock") == "code"


class TestBuildValidationSpec:
    """Test building ValidationSpec from config files."""

    def test_no_config_returns_default_spec(self, tmp_path: Path) -> None:
        """Without mala.yaml, returns default spec with standard Python/uv commands.

        This aligns with build_prompt_validation_commands() which also returns
        defaults when config is missing.
        """
        spec = build_validation_spec(tmp_path)

        # Should have default commands (not empty)
        command_names = [cmd.name for cmd in spec.commands]
        assert "format" in command_names
        assert "lint" in command_names
        assert "typecheck" in command_names
        assert "test" in command_names

        # Default scope
        assert spec.scope == ValidationScope.PER_SESSION

        # Coverage and E2E disabled by default (require explicit config)
        assert spec.coverage.enabled is False
        assert spec.e2e.enabled is False

    def test_loads_go_project_config(self, tmp_path: Path) -> None:
        """Test loading Go project configuration."""
        config_src = Path("tests/fixtures/mala-configs/go-project.yaml")
        config_dst = tmp_path / "mala.yaml"
        shutil.copy(config_src, config_dst)

        spec = build_validation_spec(tmp_path)

        # Check commands were loaded
        command_names = [cmd.name for cmd in spec.commands]
        assert "setup" in command_names
        assert "test" in command_names
        assert "lint" in command_names
        assert "format" in command_names

        # Check code patterns
        assert "**/*.go" in spec.code_patterns
        assert "**/go.mod" in spec.code_patterns

        # Check config files
        assert ".golangci.yml" in spec.config_files

        # Check setup files
        assert "go.mod" in spec.setup_files
        assert "go.sum" in spec.setup_files

    def test_loads_node_project_config(self, tmp_path: Path) -> None:
        """Test loading Node.js project configuration."""
        config_src = Path("tests/fixtures/mala-configs/node-project.yaml")
        config_dst = tmp_path / "mala.yaml"
        shutil.copy(config_src, config_dst)

        spec = build_validation_spec(tmp_path)

        # Check commands
        command_names = [cmd.name for cmd in spec.commands]
        assert "setup" in command_names
        assert "test" in command_names
        assert "lint" in command_names
        assert "format" in command_names
        assert "typecheck" in command_names

        # Check coverage is enabled with threshold
        assert spec.coverage.enabled is True
        assert spec.coverage.min_percent == 80
        assert spec.coverage.report_path == Path("coverage/coverage.xml")

    def test_partial_config_only_defines_specified_commands(
        self, tmp_path: Path
    ) -> None:
        """Config with only some commands should only have those commands."""
        config_src = Path("tests/fixtures/mala-configs/partial-config.yaml")
        config_dst = tmp_path / "mala.yaml"
        shutil.copy(config_src, config_dst)

        spec = build_validation_spec(tmp_path)

        command_names = [cmd.name for cmd in spec.commands]
        assert "test" in command_names
        assert "lint" in command_names
        # These should not be present
        assert "setup" not in command_names
        assert "format" not in command_names
        assert "typecheck" not in command_names

    def test_command_with_custom_timeout(self, tmp_path: Path) -> None:
        """Test that custom timeout values are applied."""
        config_src = Path("tests/fixtures/mala-configs/command-with-timeout.yaml")
        config_dst = tmp_path / "mala.yaml"
        shutil.copy(config_src, config_dst)

        spec = build_validation_spec(tmp_path)

        # Find setup command and check timeout
        setup_cmd = next((cmd for cmd in spec.commands if cmd.name == "setup"), None)
        assert setup_cmd is not None
        assert setup_cmd.timeout == 300

        # Find test command and check timeout
        test_cmd = next((cmd for cmd in spec.commands if cmd.name == "test"), None)
        assert test_cmd is not None
        assert test_cmd.timeout == 600

        # Lint should have default timeout
        lint_cmd = next((cmd for cmd in spec.commands if cmd.name == "lint"), None)
        assert lint_cmd is not None
        assert lint_cmd.timeout == DEFAULT_COMMAND_TIMEOUT

    def test_disable_post_validate(self, tmp_path: Path) -> None:
        """disable_validations post-validate removes all commands."""
        config_src = Path("tests/fixtures/mala-configs/go-project.yaml")
        config_dst = tmp_path / "mala.yaml"
        shutil.copy(config_src, config_dst)

        spec = build_validation_spec(
            tmp_path,
            disable_validations={"post-validate"},
        )

        assert spec.commands == []

    def test_disable_coverage(self, tmp_path: Path) -> None:
        """disable_validations coverage disables coverage."""
        config_src = Path("tests/fixtures/mala-configs/node-project.yaml")
        config_dst = tmp_path / "mala.yaml"
        shutil.copy(config_src, config_dst)

        spec = build_validation_spec(
            tmp_path,
            disable_validations={"coverage"},
        )

        assert spec.coverage.enabled is False

    def test_disable_e2e(self, tmp_path: Path) -> None:
        """disable_validations e2e disables E2E."""
        config_src = Path("tests/fixtures/mala-configs/go-project.yaml")
        config_dst = tmp_path / "mala.yaml"
        shutil.copy(config_src, config_dst)

        spec = build_validation_spec(
            tmp_path,
            scope=ValidationScope.GLOBAL,
            disable_validations={"e2e"},
        )

        assert spec.e2e.enabled is False

    def test_scope_defaults_to_per_session(self, tmp_path: Path) -> None:
        """When scope is not specified, defaults to PER_SESSION."""
        config_src = Path("tests/fixtures/mala-configs/partial-config.yaml")
        config_dst = tmp_path / "mala.yaml"
        shutil.copy(config_src, config_dst)

        spec = build_validation_spec(tmp_path)

        assert spec.scope == ValidationScope.PER_SESSION

    def test_global_scope_can_enable_e2e(self, tmp_path: Path) -> None:
        """Global scope enables E2E if e2e command is defined."""
        # Create config with e2e command
        config_content = """
commands:
  test: "pytest"
  e2e: "pytest -m e2e"
"""
        (tmp_path / "mala.yaml").write_text(config_content)

        spec = build_validation_spec(tmp_path, scope=ValidationScope.GLOBAL)

        assert spec.e2e.enabled is True

    def test_coverage_enabled_for_both_scopes_without_overrides(
        self, tmp_path: Path
    ) -> None:
        """Coverage should be enabled for both scopes when same test command is used.

        When the same test command is used for both scopes, coverage should be enabled for both.
        """
        config_content = """
commands:
  test: "pytest --cov=src --cov-report=xml"
coverage:
  format: xml
  file: coverage.xml
  threshold: 80
"""
        (tmp_path / "mala.yaml").write_text(config_content)

        # Both scopes should have coverage enabled
        per_session_spec = build_validation_spec(
            tmp_path, scope=ValidationScope.PER_SESSION
        )
        assert per_session_spec.coverage.enabled is True

        global_spec = build_validation_spec(tmp_path, scope=ValidationScope.GLOBAL)
        assert global_spec.coverage.enabled is True

    def test_no_test_command_anywhere_with_coverage_raises_error(
        self, tmp_path: Path
    ) -> None:
        """Coverage without any test command should raise ConfigError.

        If there's no commands.test, but coverage is enabled, we should raise an
        error because there's no way to generate coverage data.
        """
        import pytest

        from src.domain.validation.config import ConfigError

        config_content = """
commands:
  lint: "uvx ruff check ."
coverage:
  format: xml
  file: coverage.xml
  threshold: 80
"""
        (tmp_path / "mala.yaml").write_text(config_content)

        # Should raise ConfigError because no test command exists anywhere
        with pytest.raises(ConfigError) as exc_info:
            build_validation_spec(tmp_path, scope=ValidationScope.PER_SESSION)

        assert "coverage" in str(exc_info.value).lower()
        assert "test" in str(exc_info.value).lower()

    def test_command_shell_is_true_by_default(self, tmp_path: Path) -> None:
        """All commands should have shell=True by default."""
        config_src = Path("tests/fixtures/mala-configs/partial-config.yaml")
        config_dst = tmp_path / "mala.yaml"
        shutil.copy(config_src, config_dst)

        spec = build_validation_spec(tmp_path)

        for cmd in spec.commands:
            assert cmd.shell is True

    def test_command_is_shell_string(self, tmp_path: Path) -> None:
        """Commands should be shell strings, not lists."""
        config_src = Path("tests/fixtures/mala-configs/partial-config.yaml")
        config_dst = tmp_path / "mala.yaml"
        shutil.copy(config_src, config_dst)

        spec = build_validation_spec(tmp_path)

        for cmd in spec.commands:
            assert isinstance(cmd.command, str)


class TestClassifyChangesMultiple:
    """Test classification with multiple changed files."""

    def test_all_docs_returns_docs(self) -> None:
        files = ["README.md", "docs/api.md", "CHANGELOG.txt"]
        results = [classify_change(f) for f in files]
        assert all(r == "docs" for r in results)

    def test_any_code_means_code(self) -> None:
        files = ["README.md", "src/app.py"]
        # If any file is code, the overall change should be treated as code
        has_code = any(classify_change(f) == "code" for f in files)
        assert has_code is True

    def test_empty_files_list(self) -> None:
        # No files changed - could be docs-only or code depending on context
        files: list[str] = []
        has_code = any(classify_change(f) == "code" for f in files)
        assert has_code is False


class TestBuildValidationSpecWithPreset:
    """Test building ValidationSpec with preset inheritance."""

    def test_preset_override_config(self, tmp_path: Path) -> None:
        """Test preset override behavior."""
        config_src = Path("tests/fixtures/mala-configs/preset-override.yaml")
        config_dst = tmp_path / "mala.yaml"
        shutil.copy(config_src, config_dst)

        spec = build_validation_spec(tmp_path)

        # Test command should be overridden
        test_cmd = next((cmd for cmd in spec.commands if cmd.name == "test"), None)
        assert test_cmd is not None
        assert "pytest -v --slow" in test_cmd.command

        # Coverage threshold should be overridden
        assert spec.coverage.min_percent == 95

        # Other commands should come from preset (python-uv)
        # setup, lint, format, typecheck should be inherited
        command_names = [cmd.name for cmd in spec.commands]
        assert "setup" in command_names
        assert "lint" in command_names
        assert "format" in command_names
        assert "typecheck" in command_names


class TestCoverageScopes:
    """Tests for coverage behavior with scopes."""

    def test_coverage_enabled_for_both_scopes_with_same_test(
        self, tmp_path: Path
    ) -> None:
        """Coverage should be enabled for both scopes when same test command is used."""
        config_content = """
commands:
  test: "pytest"
coverage:
  format: xml
  file: coverage.xml
  threshold: 80
"""
        (tmp_path / "mala.yaml").write_text(config_content)

        per_session_spec = build_validation_spec(
            tmp_path, scope=ValidationScope.PER_SESSION
        )
        global_spec = build_validation_spec(tmp_path, scope=ValidationScope.GLOBAL)

        # Both scopes should have coverage enabled
        assert per_session_spec.coverage.enabled is True
        assert global_spec.coverage.enabled is True


class TestBuildValidationSpecCustomCommands:
    """Test custom commands in build_validation_spec."""

    def test_build_validation_spec_custom_commands_pipeline_order(self) -> None:
        """Custom commands appear after typecheck, before test in pipeline order."""
        commands_config = CommandsConfig(
            format=CommandConfig(command="uvx ruff format --check ."),
            lint=CommandConfig(command="uvx ruff check ."),
            typecheck=CommandConfig(command="uvx ty check"),
            test=CommandConfig(command="uv run pytest"),
        )
        custom_commands = {
            "security_scan": CustomCommandConfig(command="bandit -r src/"),
            "docs_check": CustomCommandConfig(command="mkdocs build --strict"),
        }

        commands = _build_commands_from_config(commands_config, custom_commands)

        # Get command names in order
        cmd_names = [cmd.name for cmd in commands]

        # Verify pipeline order: format → lint → typecheck → custom → test
        format_idx = cmd_names.index("format")
        lint_idx = cmd_names.index("lint")
        typecheck_idx = cmd_names.index("typecheck")
        security_idx = cmd_names.index("security_scan")
        docs_idx = cmd_names.index("docs_check")
        test_idx = cmd_names.index("test")

        assert format_idx < lint_idx < typecheck_idx
        assert typecheck_idx < security_idx < test_idx
        assert typecheck_idx < docs_idx < test_idx

    def test_build_validation_spec_custom_commands_insertion_order(self) -> None:
        """Custom commands preserve dict insertion order."""
        commands_config = CommandsConfig(test=CommandConfig(command="pytest"))
        custom_commands = {
            "cmd_a": CustomCommandConfig(command="echo a"),
            "cmd_b": CustomCommandConfig(command="echo b"),
            "cmd_c": CustomCommandConfig(command="echo c"),
        }

        commands = _build_commands_from_config(commands_config, custom_commands)

        # Find custom commands in order
        custom_cmds = [cmd for cmd in commands if cmd.kind == CommandKind.CUSTOM]

        assert len(custom_cmds) == 3
        assert custom_cmds[0].name == "cmd_a"
        assert custom_cmds[1].name == "cmd_b"
        assert custom_cmds[2].name == "cmd_c"

    def test_build_validation_spec_custom_commands_attributes(self) -> None:
        """Custom commands have correct attributes (kind, allow_fail, timeout)."""
        commands_config = CommandsConfig(test=CommandConfig(command="pytest"))
        custom_commands = {
            "security_scan": CustomCommandConfig(
                command="bandit -r src/", allow_fail=True, timeout=300
            ),
            "docs_check": CustomCommandConfig(command="mkdocs build --strict"),
        }

        commands = _build_commands_from_config(commands_config, custom_commands)

        custom_cmds = {
            cmd.name: cmd for cmd in commands if cmd.kind == CommandKind.CUSTOM
        }

        assert len(custom_cmds) == 2

        security = custom_cmds["security_scan"]
        assert security.kind == CommandKind.CUSTOM
        assert security.command == "bandit -r src/"
        assert security.allow_fail is True
        assert security.timeout == 300

        docs = custom_cmds["docs_check"]
        assert docs.kind == CommandKind.CUSTOM
        assert docs.command == "mkdocs build --strict"
        assert docs.allow_fail is False
        assert docs.timeout == DEFAULT_COMMAND_TIMEOUT


class TestCustomCommandsYamlOrderPreservation:
    """Regression tests for dict order preservation in custom_commands.

    Python 3.7+ guarantees dict insertion order. These tests ensure
    custom_commands dict order is preserved through spec building.
    """

    def test_custom_commands_order_preserved(self) -> None:
        """Custom commands preserve dict insertion order through spec building.

        This is a regression test to ensure that:
        1. Dict insertion order is preserved in custom_commands
        2. _build_commands_from_config maintains that order in output
        """
        commands_config = CommandsConfig(test=CommandConfig(command="pytest"))
        # Dict with specific insertion order
        custom_commands = {
            "cmd_alpha": CustomCommandConfig(command="echo alpha"),
            "cmd_beta": CustomCommandConfig(command="echo beta"),
            "cmd_gamma": CustomCommandConfig(command="echo gamma"),
        }

        # Build commands and extract custom commands
        commands = _build_commands_from_config(commands_config, custom_commands)
        custom_cmds = [cmd for cmd in commands if cmd.kind == CommandKind.CUSTOM]

        # Verify order matches insertion order
        assert len(custom_cmds) == 3, "Expected exactly 3 custom commands"
        assert custom_cmds[0].name == "cmd_alpha"
        assert custom_cmds[1].name == "cmd_beta"
        assert custom_cmds[2].name == "cmd_gamma"


class TestInlineCustomCommandsIntegration:
    """Integration tests: YAML with inline customs → ValidationSpec."""

    def test_inline_customs_in_commands_yaml_to_spec(self, tmp_path: Path) -> None:
        """YAML with inline custom commands in commands section → ValidationSpec.

        Tests full path: YAML file → ValidationConfig → build_validation_spec → ValidationSpec
        with custom commands appearing in the commands list.
        """
        # Create mala.yaml with inline custom command in commands section
        yaml_content = """\
preset: python-uv
commands:
  lint: uvx ruff check .
  typecheck: uvx ty check
  test: uv run pytest
  security: bandit -r src/
"""
        (tmp_path / "mala.yaml").write_text(yaml_content)

        spec = build_validation_spec(tmp_path)

        # Verify custom command is in spec
        cmd_names = [cmd.name for cmd in spec.commands]
        assert "security" in cmd_names

        # Verify custom command has correct attributes
        security_cmd = next(cmd for cmd in spec.commands if cmd.name == "security")
        assert security_cmd.kind == CommandKind.CUSTOM
        assert security_cmd.command == "bandit -r src/"

        # Verify pipeline order: standard commands before test, custom after typecheck
        typecheck_idx = cmd_names.index("typecheck")
        security_idx = cmd_names.index("security")
        test_idx = cmd_names.index("test")
        assert typecheck_idx < security_idx < test_idx


class TestConfigMissingSemanticsConsistency:
    """Tests for consistent config_missing behavior between spec and prompts.

    When config is missing (no mala.yaml), both build_validation_spec() and
    build_prompt_validation_commands() should return defaults, not disable
    everything. This ensures agents get sensible commands AND the orchestrator
    validates them.
    """

    def test_config_missing_flag_returns_defaults_for_spec(
        self, tmp_path: Path
    ) -> None:
        """config_missing=True should return default spec, not empty spec.

        This aligns with build_prompt_validation_commands() behavior.
        """
        spec = build_validation_spec(tmp_path, config_missing=True)

        # Should have default commands
        command_names = [cmd.name for cmd in spec.commands]
        assert "format" in command_names
        assert "lint" in command_names
        assert "typecheck" in command_names
        assert "test" in command_names

    def test_spec_and_prompts_consistent_when_config_missing(
        self, tmp_path: Path
    ) -> None:
        """build_validation_spec and build_prompt_validation_commands should be consistent.

        Both functions should use defaults when config is missing. This is a
        regression test for the issue where spec returned empty (disabled
        validations) while prompts returned defaults (enabled commands).
        """
        from src.domain.prompts import build_prompt_validation_commands

        # Get defaults from prompts
        prompt_commands = build_prompt_validation_commands(
            tmp_path, config_missing=True
        )

        # Get defaults from spec
        spec = build_validation_spec(tmp_path, config_missing=True)

        # Verify spec commands match prompt commands
        spec_test_cmd = next(cmd for cmd in spec.commands if cmd.name == "test")
        assert spec_test_cmd.command == prompt_commands.test

        spec_lint_cmd = next(cmd for cmd in spec.commands if cmd.name == "lint")
        assert spec_lint_cmd.command == prompt_commands.lint

        spec_format_cmd = next(cmd for cmd in spec.commands if cmd.name == "format")
        assert spec_format_cmd.command == prompt_commands.format

        spec_typecheck_cmd = next(
            cmd for cmd in spec.commands if cmd.name == "typecheck"
        )
        assert spec_typecheck_cmd.command == prompt_commands.typecheck


class TestTriggerCommandRefValidation:
    """Tests for trigger command ref validation at startup.

    Per mala-553g: Invalid trigger command refs must be caught at config load /
    spec build time, not at runtime when triggers fire.
    """

    def test_invalid_trigger_ref_fails_at_spec_build(self, tmp_path: Path) -> None:
        """build_validation_spec raises ConfigError for invalid trigger command ref."""
        import pytest

        from src.domain.validation.config import ConfigError

        # Create mala.yaml with trigger referencing non-existent command
        yaml_content = """\
preset: python-uv
commands:
  test: uv run pytest
  lint: uvx ruff check .
validation_triggers:
  session_end:
    failure_mode: continue
    commands:
      - ref: typo_command
"""
        (tmp_path / "mala.yaml").write_text(yaml_content)

        with pytest.raises(
            ConfigError,
            match=r"trigger session_end references unknown command 'typo_command'",
        ):
            build_validation_spec(tmp_path)

    def test_valid_trigger_refs_pass_validation(self, tmp_path: Path) -> None:
        """build_validation_spec succeeds when all trigger refs exist in base pool."""
        # Create mala.yaml with valid trigger refs
        yaml_content = """\
preset: python-uv
commands:
  test: uv run pytest
  lint: uvx ruff check .
validation_triggers:
  session_end:
    failure_mode: continue
    commands:
      - ref: test
      - ref: lint
"""
        (tmp_path / "mala.yaml").write_text(yaml_content)

        # Should not raise - all refs exist
        spec = build_validation_spec(tmp_path)
        assert spec is not None

    def test_built_in_command_ref_in_trigger_validated(self, tmp_path: Path) -> None:
        """Trigger can reference built-in commands from commands section."""
        yaml_content = """\
preset: python-uv
commands:
  test: uv run pytest
  lint: uvx ruff check .
validation_triggers:
  session_end:
    failure_mode: continue
    commands:
      - ref: lint
"""
        (tmp_path / "mala.yaml").write_text(yaml_content)

        # Should not raise - lint is a built-in command
        spec = build_validation_spec(tmp_path)
        assert spec is not None

    def test_custom_command_ref_in_trigger_validated(self, tmp_path: Path) -> None:
        """Trigger can reference inline custom commands from commands section."""
        yaml_content = """\
preset: python-uv
commands:
  test: uv run pytest
  custom_check: my-custom-command
validation_triggers:
  session_end:
    failure_mode: continue
    commands:
      - ref: custom_check
"""
        (tmp_path / "mala.yaml").write_text(yaml_content)

        # Should not raise - custom_check is defined in commands
        spec = build_validation_spec(tmp_path)
        assert spec is not None

    def test_invalid_ref_error_lists_available_commands(self, tmp_path: Path) -> None:
        """Error message includes list of available commands including customs."""
        import pytest

        from src.domain.validation.config import ConfigError

        yaml_content = """\
preset: python-uv
commands:
  test: uv run pytest
  lint: uvx ruff check .
  my_custom: echo custom
validation_triggers:
  epic_completion:
    failure_mode: abort
    epic_depth: top_level
    fire_on: success
    commands:
      - ref: nonexistent
"""
        (tmp_path / "mala.yaml").write_text(yaml_content)

        with pytest.raises(ConfigError) as exc_info:
            build_validation_spec(tmp_path)

        error_msg = str(exc_info.value)
        # Should list available commands from base pool (built-ins and customs)
        assert "lint" in error_msg
        assert "test" in error_msg
        assert "my_custom" in error_msg


class TestEvidenceCheckValidation:
    """Tests for evidence_check.required validation against resolved command map."""

    def test_valid_key_from_preset_only(self, tmp_path: Path) -> None:
        """evidence_check.required can reference preset-only commands (e.g., 'test')."""
        # Uses python-uv preset which provides test command
        yaml_content = """\
preset: python-uv
evidence_check:
  required:
    - test
"""
        (tmp_path / "mala.yaml").write_text(yaml_content)

        spec = build_validation_spec(tmp_path)
        assert spec.evidence_required == ("test",)

    def test_valid_key_from_project_commands(self, tmp_path: Path) -> None:
        """evidence_check.required can reference project-defined commands."""
        yaml_content = """\
commands:
  lint: uvx ruff check .
evidence_check:
  required:
    - lint
"""
        (tmp_path / "mala.yaml").write_text(yaml_content)

        spec = build_validation_spec(tmp_path)
        assert spec.evidence_required == ("lint",)

    def test_invalid_key_produces_error_with_available_keys(
        self, tmp_path: Path
    ) -> None:
        """Invalid evidence_check key produces error listing available commands."""
        import pytest

        from src.domain.validation.config import ConfigError

        yaml_content = """\
preset: python-uv
commands:
  test: uv run pytest
  lint: uvx ruff check .
evidence_check:
  required:
    - nonexistent
"""
        (tmp_path / "mala.yaml").write_text(yaml_content)

        with pytest.raises(ConfigError) as exc_info:
            build_validation_spec(tmp_path)

        error_msg = str(exc_info.value)
        assert (
            "evidence_check.required references unknown command 'nonexistent'"
            in error_msg
        )
        assert "lint" in error_msg
        assert "test" in error_msg

    def test_global_only_command_validates_pre_scope(self, tmp_path: Path) -> None:
        """Commands that run only at global scope still validate in evidence_check."""
        yaml_content = """\
preset: python-uv
commands:
  test: uv run pytest
  e2e: uv run pytest tests/e2e
evidence_check:
  required:
    - e2e
"""
        (tmp_path / "mala.yaml").write_text(yaml_content)

        spec = build_validation_spec(tmp_path)
        assert spec.evidence_required == ("e2e",)

    def test_project_override_of_preset_key(self, tmp_path: Path) -> None:
        """Project can override preset command and still validate in evidence_check."""
        yaml_content = """\
preset: python-uv
commands:
  test: my-custom-test-runner
evidence_check:
  required:
    - test
"""
        (tmp_path / "mala.yaml").write_text(yaml_content)

        spec = build_validation_spec(tmp_path)
        assert spec.evidence_required == ("test",)
