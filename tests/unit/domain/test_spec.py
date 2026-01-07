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
    CustomOverrideMode,
)
from src.domain.validation.spec import (
    DEFAULT_COMMAND_TIMEOUT,
    CommandKind,
    CoverageConfig,
    E2EConfig,
    ValidationCommand,
    ValidationScope,
    ValidationSpec,
    _apply_custom_commands_override,
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

    def test_no_config_returns_empty_spec(self, tmp_path: Path) -> None:
        """Without mala.yaml, returns empty spec."""
        spec = build_validation_spec(tmp_path)

        assert spec.commands == []
        assert spec.scope == ValidationScope.PER_SESSION
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

    def test_global_validation_commands_override_base(self, tmp_path: Path) -> None:
        """Global commands should override base commands when provided."""
        config_dst = tmp_path / "mala.yaml"
        config_dst.write_text(
            "\n".join(
                [
                    "commands:",
                    '  test: "pytest issue"',
                    "global_validation_commands:",
                    '  test: "pytest global"',
                ]
            )
            + "\n"
        )

        issue_spec = build_validation_spec(tmp_path, scope=ValidationScope.PER_SESSION)
        run_spec = build_validation_spec(tmp_path, scope=ValidationScope.GLOBAL)

        issue_test = next(cmd for cmd in issue_spec.commands if cmd.name == "test")
        run_test = next(cmd for cmd in run_spec.commands if cmd.name == "test")

        assert issue_test.command == "pytest issue"
        assert run_test.command == "pytest global"

    def test_global_validation_commands_can_disable_base(self, tmp_path: Path) -> None:
        """Global overrides can explicitly disable a base command."""
        config_dst = tmp_path / "mala.yaml"
        config_dst.write_text(
            "\n".join(
                [
                    "commands:",
                    '  test: "pytest issue"',
                    "global_validation_commands:",
                    "  test: null",
                ]
            )
            + "\n"
        )

        issue_spec = build_validation_spec(tmp_path, scope=ValidationScope.PER_SESSION)
        run_spec = build_validation_spec(tmp_path, scope=ValidationScope.GLOBAL)

        assert any(cmd.name == "test" for cmd in issue_spec.commands)
        assert not any(cmd.name == "test" for cmd in run_spec.commands)

    def test_coverage_with_global_null_test_raises_error(
        self, tmp_path: Path
    ) -> None:
        """Coverage enabled + global test=null should raise ConfigError.

        If commands.test is set but global_validation_commands.test is explicitly null,
        and coverage is enabled, building a GLOBAL spec should fail because
        coverage requires a test command to generate coverage data.
        """
        import pytest

        from src.domain.validation.config import ConfigError

        config_dst = tmp_path / "mala.yaml"
        config_dst.write_text(
            "\n".join(
                [
                    "commands:",
                    '  test: "uv run pytest"',
                    "global_validation_commands:",
                    "  test: null",
                    "coverage:",
                    "  format: xml",
                    "  file: coverage.xml",
                    "  threshold: 80",
                ]
            )
            + "\n"
        )

        # PER_SESSION scope should work fine (test command is present)
        issue_spec = build_validation_spec(tmp_path, scope=ValidationScope.PER_SESSION)
        assert issue_spec.coverage.enabled is True
        assert any(cmd.name == "test" for cmd in issue_spec.commands)

        # GLOBAL scope should fail because test is disabled but coverage is enabled
        with pytest.raises(ConfigError) as exc_info:
            build_validation_spec(tmp_path, scope=ValidationScope.GLOBAL)

        assert "coverage" in str(exc_info.value).lower()
        assert "test" in str(exc_info.value).lower()

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
        """--disable post-validate removes all commands."""
        config_src = Path("tests/fixtures/mala-configs/go-project.yaml")
        config_dst = tmp_path / "mala.yaml"
        shutil.copy(config_src, config_dst)

        spec = build_validation_spec(
            tmp_path,
            disable_validations={"post-validate"},
        )

        assert spec.commands == []

    def test_disable_coverage(self, tmp_path: Path) -> None:
        """--disable coverage disables coverage."""
        config_src = Path("tests/fixtures/mala-configs/node-project.yaml")
        config_dst = tmp_path / "mala.yaml"
        shutil.copy(config_src, config_dst)

        spec = build_validation_spec(
            tmp_path,
            disable_validations={"coverage"},
        )

        assert spec.coverage.enabled is False

    def test_disable_e2e(self, tmp_path: Path) -> None:
        """--disable e2e disables E2E."""
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

    def test_global_validation_commands_e2e_null_disables_e2e(self, tmp_path: Path) -> None:
        """global_validation_commands.e2e: null disables E2E even if base e2e is defined."""
        # Create config with e2e command but global_validation_commands.e2e: null
        config_content = """
commands:
  test: "pytest"
  e2e: "pytest -m e2e"
global_validation_commands:
  e2e: null
"""
        (tmp_path / "mala.yaml").write_text(config_content)

        spec = build_validation_spec(tmp_path, scope=ValidationScope.GLOBAL)

        # E2E should be disabled because global_validation_commands.e2e: null overrides
        assert spec.e2e.enabled is False

    def test_global_test_disables_coverage_for_per_session(
        self, tmp_path: Path
    ) -> None:
        """Coverage should be disabled for PER_SESSION when global_validation_commands.test is set.

        When global_validation_commands.test provides a different test command (e.g., with
        --cov flags), the base commands.test won't generate coverage.xml, so
        per-session validation should not check coverage.
        """
        config_content = """
commands:
  test: "pytest"
global_validation_commands:
  test: "pytest --cov=src --cov-report=xml"
coverage:
  format: xml
  file: coverage.xml
  threshold: 80
"""
        (tmp_path / "mala.yaml").write_text(config_content)

        # PER_SESSION scope should have coverage DISABLED
        # because global_validation_commands.test is set (meaning only global generates coverage)
        per_session_spec = build_validation_spec(
            tmp_path, scope=ValidationScope.PER_SESSION
        )
        assert per_session_spec.coverage.enabled is False

        # GLOBAL scope should have coverage ENABLED
        global_spec = build_validation_spec(
            tmp_path, scope=ValidationScope.GLOBAL
        )
        assert global_spec.coverage.enabled is True

    def test_no_global_test_enables_coverage_for_both_scopes(
        self, tmp_path: Path
    ) -> None:
        """Coverage should be enabled for both scopes when global_validation_commands.test is not set.

        When there's no global_validation_commands.test override, the same test command
        is used for both scopes, so coverage should be enabled for both.
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

        global_spec = build_validation_spec(
            tmp_path, scope=ValidationScope.GLOBAL
        )
        assert global_spec.coverage.enabled is True

    def test_global_test_null_preserves_coverage_for_per_session(
        self, tmp_path: Path
    ) -> None:
        """Coverage should stay enabled for PER_SESSION when global_validation_commands.test is null.

        When global_validation_commands.test is explicitly null (disabling test at global),
        per-session should still run with coverage since it has a test command. The null
        value indicates "skip test at global" - not "move coverage to global".
        """
        config_content = """
commands:
  test: "pytest --cov=src --cov-report=xml"
global_validation_commands:
  test: null
coverage:
  format: xml
  file: coverage.xml
  threshold: 80
"""
        (tmp_path / "mala.yaml").write_text(config_content)

        # PER_SESSION scope should have coverage enabled (base test has --cov)
        # global_validation_commands.test is null, which means "skip test at global",
        # not "move coverage to global"
        per_session_spec = build_validation_spec(
            tmp_path, scope=ValidationScope.PER_SESSION
        )
        assert per_session_spec.coverage.enabled is True

    def test_global_only_test_with_coverage_no_base_test(
        self, tmp_path: Path
    ) -> None:
        """Coverage with global_validation_commands.test only (no base test) should work.

        When a config has only global_validation_commands.test (no base commands.test)
        plus coverage settings, building a PER_SESSION spec should not raise an error.
        Coverage will be generated at global where the test command exists.
        """
        config_content = """
commands:
  lint: "uvx ruff check ."
global_validation_commands:
  test: "uv run pytest --cov=src --cov-report=xml"
coverage:
  format: xml
  file: coverage.xml
  threshold: 80
"""
        (tmp_path / "mala.yaml").write_text(config_content)

        # PER_SESSION scope should NOT raise ConfigError
        # Coverage is disabled for PER_SESSION since there's no test command
        # but the error should not fire because global_validation_commands.test exists
        per_session_spec = build_validation_spec(
            tmp_path, scope=ValidationScope.PER_SESSION
        )
        # Coverage should be disabled for PER_SESSION (no test command to run)
        assert per_session_spec.coverage.enabled is False
        # No test command in per-session spec
        assert not any(cmd.name == "test" for cmd in per_session_spec.commands)

        # GLOBAL scope should have coverage enabled
        global_spec = build_validation_spec(
            tmp_path, scope=ValidationScope.GLOBAL
        )
        assert global_spec.coverage.enabled is True
        # Global should have test command
        assert any(cmd.name == "test" for cmd in global_spec.commands)

    def test_no_test_command_anywhere_with_coverage_raises_error(
        self, tmp_path: Path
    ) -> None:
        """Coverage without any test command should raise ConfigError.

        If there's no commands.test and no global_validation_commands.test, but coverage
        is enabled, we should raise an error because there's no way to generate
        coverage data.
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


class TestCoverageOnlyAtGlobal:
    """Tests for coverage-only-at-global behavior based on global_validation_commands.test.

    When a user explicitly sets global_validation_commands.test (e.g., with --cov flags),
    coverage should only be checked at global, not per-session. This prevents
    per-session validation from failing due to missing coverage.xml that only
    global generates.
    """

    def test_user_global_test_disables_per_session_coverage(
        self, tmp_path: Path
    ) -> None:
        """User's global_validation_commands.test should disable coverage for per-session.

        When the user explicitly sets global_validation_commands.test with coverage flags,
        per-session scope should have coverage disabled since only global
        generates coverage.xml.
        """
        config_content = """
preset: python-uv
global_validation_commands:
  test: "pytest --cov=src --cov-report=xml"
coverage:
  format: xml
  file: coverage.xml
  threshold: 80
"""
        (tmp_path / "mala.yaml").write_text(config_content)

        per_session_spec = build_validation_spec(
            tmp_path, scope=ValidationScope.PER_SESSION
        )
        global_spec = build_validation_spec(
            tmp_path, scope=ValidationScope.GLOBAL
        )

        # Per-session should NOT check coverage (global generates it)
        assert per_session_spec.coverage.enabled is False
        # Global should check coverage
        assert global_spec.coverage.enabled is True

    def test_preset_global_test_does_not_disable_per_session_coverage(
        self, tmp_path: Path
    ) -> None:
        """Preset's global_validation_commands.test should NOT disable per-session coverage.

        When global_validation_commands.test comes from preset (not user), it shouldn't
        affect coverage behavior. This is because the user didn't explicitly
        opt into the coverage-only-at-global pattern.

        Note: Currently python-uv doesn't have global_validation_commands, so we use
        a custom mala.yaml to simulate this scenario.
        """
        # Write a preset-like config directly (simulating a preset with global_validation_commands)
        # In real usage, this would come from a preset
        config_content = """
commands:
  test: "pytest"
coverage:
  format: xml
  file: coverage.xml
  threshold: 80
"""
        (tmp_path / "mala.yaml").write_text(config_content)

        # Without explicit global_validation_commands.test, coverage should be enabled for both
        per_session_spec = build_validation_spec(
            tmp_path, scope=ValidationScope.PER_SESSION
        )
        global_spec = build_validation_spec(
            tmp_path, scope=ValidationScope.GLOBAL
        )

        # Both scopes should have coverage enabled
        assert per_session_spec.coverage.enabled is True
        assert global_spec.coverage.enabled is True

    def test_global_test_null_preserves_per_session_coverage(
        self, tmp_path: Path
    ) -> None:
        """global_validation_commands.test: null should NOT disable per-session coverage.

        When user sets global_validation_commands.test to null (to skip tests at global),
        this is different from setting a test command. The intent is to skip
        testing at global, not to move coverage there.
        """
        config_content = """
commands:
  test: "pytest --cov=src --cov-report=xml"
global_validation_commands:
  test: null
coverage:
  format: xml
  file: coverage.xml
  threshold: 80
"""
        (tmp_path / "mala.yaml").write_text(config_content)

        per_session_spec = build_validation_spec(
            tmp_path, scope=ValidationScope.PER_SESSION
        )

        # Per-session should still check coverage (global test is disabled, not moved)
        assert per_session_spec.coverage.enabled is True
        assert any(cmd.name == "test" for cmd in per_session_spec.commands)


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


class TestApplyCommandOverridesCustomCommands:
    """Test global custom_commands override with mode-based semantics."""

    def test_global_replace_mode_fully_replaces_repo_customs(self) -> None:
        """GLOBAL + REPLACE mode returns only global customs."""
        repo_customs = {
            "cmd_a": CustomCommandConfig(command="echo a"),
            "cmd_b": CustomCommandConfig(command="echo b"),
        }
        global_validation_commands = CommandsConfig(
            custom_commands={"cmd_a": CustomCommandConfig(command="echo new_a")},
            custom_override_mode=CustomOverrideMode.REPLACE,
        )

        # PER_SESSION ignores global mode, returns repo customs
        per_session_result = _apply_custom_commands_override(
            repo_customs, global_validation_commands, ValidationScope.PER_SESSION
        )
        assert len(per_session_result) == 2
        assert set(per_session_result.keys()) == {"cmd_a", "cmd_b"}

        # GLOBAL with REPLACE returns only global customs
        global_result = _apply_custom_commands_override(
            repo_customs, global_validation_commands, ValidationScope.GLOBAL
        )
        assert len(global_result) == 1
        assert "cmd_a" in global_result
        assert global_result["cmd_a"].command == "echo new_a"

    def test_global_clear_mode_returns_empty_dict(self) -> None:
        """GLOBAL + CLEAR mode returns empty dict (no customs)."""
        repo_customs = {
            "cmd_a": CustomCommandConfig(command="echo a"),
            "cmd_b": CustomCommandConfig(command="echo b"),
        }
        global_validation_commands = CommandsConfig(
            custom_override_mode=CustomOverrideMode.CLEAR,
        )

        # PER_SESSION ignores global mode, returns repo customs
        per_session_result = _apply_custom_commands_override(
            repo_customs, global_validation_commands, ValidationScope.PER_SESSION
        )
        assert len(per_session_result) == 2

        # GLOBAL with CLEAR returns empty dict
        global_result = _apply_custom_commands_override(
            repo_customs, global_validation_commands, ValidationScope.GLOBAL
        )
        assert len(global_result) == 0

    def test_global_inherit_mode_returns_repo_customs(self) -> None:
        """GLOBAL + INHERIT mode returns repo-level customs unchanged."""
        repo_customs = {
            "cmd_a": CustomCommandConfig(command="echo a"),
            "cmd_b": CustomCommandConfig(command="echo b"),
        }
        global_validation_commands = CommandsConfig(
            custom_override_mode=CustomOverrideMode.INHERIT,
        )

        # Both scopes return repo customs with INHERIT
        per_session_result = _apply_custom_commands_override(
            repo_customs, global_validation_commands, ValidationScope.PER_SESSION
        )
        global_result = _apply_custom_commands_override(
            repo_customs, global_validation_commands, ValidationScope.GLOBAL
        )

        assert len(per_session_result) == 2
        assert len(global_result) == 2
        assert set(per_session_result.keys()) == {"cmd_a", "cmd_b"}
        assert set(global_result.keys()) == {"cmd_a", "cmd_b"}

    def test_no_global_validation_commands_uses_repo_customs(self) -> None:
        """When global_validation_commands is None, use repo-level customs."""
        repo_customs = {
            "cmd_a": CustomCommandConfig(command="echo a"),
            "cmd_b": CustomCommandConfig(command="echo b"),
        }

        # Both scopes return repo customs when global_validation_commands is None
        per_session_result = _apply_custom_commands_override(
            repo_customs, None, ValidationScope.PER_SESSION
        )
        global_result = _apply_custom_commands_override(
            repo_customs, None, ValidationScope.GLOBAL
        )

        assert len(per_session_result) == 2
        assert len(global_result) == 2


class TestApplyCustomCommandsOverride:
    """Tests for _apply_custom_commands_override with mode-based API."""

    def test_additive_mode_merges_customs(self) -> None:
        """GLOBAL + ADDITIVE mode merges global into repo-level."""
        repo_customs = {"cmd_a": CustomCommandConfig(command="echo a")}
        global_validation_commands = CommandsConfig(
            custom_commands={"cmd_b": CustomCommandConfig(command="echo b")},
            custom_override_mode=CustomOverrideMode.ADDITIVE,
        )

        # ADDITIVE merges: repo + run
        result = _apply_custom_commands_override(
            repo_customs=repo_customs,
            global_validation_commands=global_validation_commands,
            scope=ValidationScope.GLOBAL,
        )

        assert len(result) == 2
        assert "cmd_a" in result
        assert "cmd_b" in result
        assert result["cmd_a"].command == "echo a"
        assert result["cmd_b"].command == "echo b"

    def test_additive_mode_global_overrides_same_key(self) -> None:
        """ADDITIVE mode: global value overrides repo-level for same key."""
        repo_customs = {"cmd_a": CustomCommandConfig(command="echo old")}
        global_validation_commands = CommandsConfig(
            custom_commands={"cmd_a": CustomCommandConfig(command="echo new")},
            custom_override_mode=CustomOverrideMode.ADDITIVE,
        )

        result = _apply_custom_commands_override(
            repo_customs=repo_customs,
            global_validation_commands=global_validation_commands,
            scope=ValidationScope.GLOBAL,
        )

        assert len(result) == 1
        assert result["cmd_a"].command == "echo new"

    def test_per_session_scope_ignores_all_global_modes(self) -> None:
        """PER_SESSION scope always uses repo customs regardless of mode."""
        repo_customs = {"cmd_a": CustomCommandConfig(command="echo a")}

        # Test all modes - PER_SESSION should always return repo_customs
        for mode in CustomOverrideMode:
            global_validation_commands = CommandsConfig(
                custom_commands={"cmd_b": CustomCommandConfig(command="echo b")},
                custom_override_mode=mode,
            )
            result = _apply_custom_commands_override(
                repo_customs=repo_customs,
                global_validation_commands=global_validation_commands,
                scope=ValidationScope.PER_SESSION,
            )
            assert result == repo_customs, f"PER_SESSION should ignore {mode}"

    def test_replace_mode_replaces_with_empty_customs(self) -> None:
        """REPLACE mode with empty custom_commands returns empty dict."""
        repo_customs = {"cmd_a": CustomCommandConfig(command="echo a")}
        global_validation_commands = CommandsConfig(
            custom_commands={},  # Empty
            custom_override_mode=CustomOverrideMode.REPLACE,
        )

        result = _apply_custom_commands_override(
            repo_customs=repo_customs,
            global_validation_commands=global_validation_commands,
            scope=ValidationScope.GLOBAL,
        )

        assert result == {}


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

    def test_global_additive_customs_yaml_to_spec(self, tmp_path: Path) -> None:
        """YAML with +prefixed customs in global_validation_commands → merged ValidationSpec.

        Tests ADDITIVE mode: global customs are merged with repo-level customs.
        """
        yaml_content = """\
preset: python-uv
commands:
  lint: uvx ruff check .
  test: uv run pytest
  security: bandit -r src/
global_validation_commands:
  +integration: uv run pytest -m integration
"""
        (tmp_path / "mala.yaml").write_text(yaml_content)

        # PER_SESSION scope: only repo-level customs
        per_session_spec = build_validation_spec(
            tmp_path, scope=ValidationScope.PER_SESSION
        )
        per_session_names = [cmd.name for cmd in per_session_spec.commands]
        assert "security" in per_session_names
        assert "integration" not in per_session_names

        # GLOBAL scope: merged customs (repo + global additive)
        global_spec = build_validation_spec(
            tmp_path, scope=ValidationScope.GLOBAL
        )
        global_names = [cmd.name for cmd in global_spec.commands]
        assert "security" in global_names
        assert "integration" in global_names

    def test_global_replace_customs_clears_repo_customs(
        self, tmp_path: Path
    ) -> None:
        """YAML with unprefixed custom in global_validation_commands → repo customs cleared.

        Tests REPLACE mode with actual custom command at global.
        """
        yaml_content = """\
preset: python-uv
commands:
  lint: uvx ruff check .
  test: uv run pytest
  security: bandit -r src/
global_validation_commands:
  integration: uv run pytest -m integration
"""
        (tmp_path / "mala.yaml").write_text(yaml_content)

        # PER_SESSION scope: only repo-level customs
        per_session_spec = build_validation_spec(
            tmp_path, scope=ValidationScope.PER_SESSION
        )
        per_session_names = [cmd.name for cmd in per_session_spec.commands]
        assert "security" in per_session_names

        # GLOBAL scope: REPLACE mode - only global customs
        global_spec = build_validation_spec(
            tmp_path, scope=ValidationScope.GLOBAL
        )
        global_names = [cmd.name for cmd in global_spec.commands]
        assert "security" not in global_names  # Repo custom cleared
        assert "integration" in global_names  # Global custom present
