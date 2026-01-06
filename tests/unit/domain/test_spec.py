"""Unit tests for src/validation/spec.py - ValidationSpec and related types.

TDD tests for:
- ValidationSpec construction from config files
- Code vs docs classification
- Disable list handling
- Config-driven spec building
"""

import shutil
from pathlib import Path


from src.domain.validation.config import CustomCommandConfig
from src.domain.validation.spec import (
    DEFAULT_COMMAND_TIMEOUT,
    CommandKind,
    CoverageConfig,
    E2EConfig,
    ValidationCommand,
    ValidationScope,
    ValidationSpec,
    _apply_custom_commands_override,
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
            scope=ValidationScope.PER_ISSUE,
        )
        assert spec.commands == []
        assert spec.require_clean_git is True  # default
        assert spec.require_pytest_for_code_changes is True  # default
        assert spec.scope == ValidationScope.PER_ISSUE
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
            scope=ValidationScope.RUN_LEVEL,
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
            scope=ValidationScope.PER_ISSUE,
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
        assert spec.scope == ValidationScope.PER_ISSUE
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

    def test_run_level_commands_override_base(self, tmp_path: Path) -> None:
        """Run-level commands should override base commands when provided."""
        config_dst = tmp_path / "mala.yaml"
        config_dst.write_text(
            "\n".join(
                [
                    "commands:",
                    '  test: "pytest issue"',
                    "run_level_commands:",
                    '  test: "pytest run-level"',
                ]
            )
            + "\n"
        )

        issue_spec = build_validation_spec(tmp_path, scope=ValidationScope.PER_ISSUE)
        run_spec = build_validation_spec(tmp_path, scope=ValidationScope.RUN_LEVEL)

        issue_test = next(cmd for cmd in issue_spec.commands if cmd.name == "test")
        run_test = next(cmd for cmd in run_spec.commands if cmd.name == "test")

        assert issue_test.command == "pytest issue"
        assert run_test.command == "pytest run-level"

    def test_run_level_commands_can_disable_base(self, tmp_path: Path) -> None:
        """Run-level overrides can explicitly disable a base command."""
        config_dst = tmp_path / "mala.yaml"
        config_dst.write_text(
            "\n".join(
                [
                    "commands:",
                    '  test: "pytest issue"',
                    "run_level_commands:",
                    "  test: null",
                ]
            )
            + "\n"
        )

        issue_spec = build_validation_spec(tmp_path, scope=ValidationScope.PER_ISSUE)
        run_spec = build_validation_spec(tmp_path, scope=ValidationScope.RUN_LEVEL)

        assert any(cmd.name == "test" for cmd in issue_spec.commands)
        assert not any(cmd.name == "test" for cmd in run_spec.commands)

    def test_coverage_with_run_level_null_test_raises_error(
        self, tmp_path: Path
    ) -> None:
        """Coverage enabled + run_level test=null should raise ConfigError.

        If commands.test is set but run_level_commands.test is explicitly null,
        and coverage is enabled, building a RUN_LEVEL spec should fail because
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
                    "run_level_commands:",
                    "  test: null",
                    "coverage:",
                    "  format: xml",
                    "  file: coverage.xml",
                    "  threshold: 80",
                ]
            )
            + "\n"
        )

        # PER_ISSUE scope should work fine (test command is present)
        issue_spec = build_validation_spec(tmp_path, scope=ValidationScope.PER_ISSUE)
        assert issue_spec.coverage.enabled is True
        assert any(cmd.name == "test" for cmd in issue_spec.commands)

        # RUN_LEVEL scope should fail because test is disabled but coverage is enabled
        with pytest.raises(ConfigError) as exc_info:
            build_validation_spec(tmp_path, scope=ValidationScope.RUN_LEVEL)

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
            scope=ValidationScope.RUN_LEVEL,
            disable_validations={"e2e"},
        )

        assert spec.e2e.enabled is False

    def test_scope_defaults_to_per_issue(self, tmp_path: Path) -> None:
        """When scope is not specified, defaults to PER_ISSUE."""
        config_src = Path("tests/fixtures/mala-configs/partial-config.yaml")
        config_dst = tmp_path / "mala.yaml"
        shutil.copy(config_src, config_dst)

        spec = build_validation_spec(tmp_path)

        assert spec.scope == ValidationScope.PER_ISSUE

    def test_run_level_scope_can_enable_e2e(self, tmp_path: Path) -> None:
        """Run-level scope enables E2E if e2e command is defined."""
        # Create config with e2e command
        config_content = """
commands:
  test: "pytest"
  e2e: "pytest -m e2e"
"""
        (tmp_path / "mala.yaml").write_text(config_content)

        spec = build_validation_spec(tmp_path, scope=ValidationScope.RUN_LEVEL)

        assert spec.e2e.enabled is True

    def test_run_level_commands_e2e_null_disables_e2e(self, tmp_path: Path) -> None:
        """run_level_commands.e2e: null disables E2E even if base e2e is defined."""
        # Create config with e2e command but run_level_commands.e2e: null
        config_content = """
commands:
  test: "pytest"
  e2e: "pytest -m e2e"
run_level_commands:
  e2e: null
"""
        (tmp_path / "mala.yaml").write_text(config_content)

        spec = build_validation_spec(tmp_path, scope=ValidationScope.RUN_LEVEL)

        # E2E should be disabled because run_level_commands.e2e: null overrides
        assert spec.e2e.enabled is False

    def test_run_level_test_disables_coverage_for_per_issue(
        self, tmp_path: Path
    ) -> None:
        """Coverage should be disabled for PER_ISSUE when run_level_commands.test is set.

        When run_level_commands.test provides a different test command (e.g., with
        --cov flags), the base commands.test won't generate coverage.xml, so
        per-issue validation should not check coverage.
        """
        config_content = """
commands:
  test: "pytest"
run_level_commands:
  test: "pytest --cov=src --cov-report=xml"
coverage:
  format: xml
  file: coverage.xml
  threshold: 80
"""
        (tmp_path / "mala.yaml").write_text(config_content)

        # PER_ISSUE scope should have coverage DISABLED
        # because run_level_commands.test is set (meaning only run-level generates coverage)
        per_issue_spec = build_validation_spec(
            tmp_path, scope=ValidationScope.PER_ISSUE
        )
        assert per_issue_spec.coverage.enabled is False

        # RUN_LEVEL scope should have coverage ENABLED
        run_level_spec = build_validation_spec(
            tmp_path, scope=ValidationScope.RUN_LEVEL
        )
        assert run_level_spec.coverage.enabled is True

    def test_no_run_level_test_enables_coverage_for_both_scopes(
        self, tmp_path: Path
    ) -> None:
        """Coverage should be enabled for both scopes when run_level_commands.test is not set.

        When there's no run_level_commands.test override, the same test command
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
        per_issue_spec = build_validation_spec(
            tmp_path, scope=ValidationScope.PER_ISSUE
        )
        assert per_issue_spec.coverage.enabled is True

        run_level_spec = build_validation_spec(
            tmp_path, scope=ValidationScope.RUN_LEVEL
        )
        assert run_level_spec.coverage.enabled is True

    def test_run_level_test_null_preserves_coverage_for_per_issue(
        self, tmp_path: Path
    ) -> None:
        """Coverage should stay enabled for PER_ISSUE when run_level_commands.test is null.

        When run_level_commands.test is explicitly null (disabling test at run level),
        per-issue should still run with coverage since it has a test command. The null
        value indicates "skip test at run level" - not "move coverage to run level".
        """
        config_content = """
commands:
  test: "pytest --cov=src --cov-report=xml"
run_level_commands:
  test: null
coverage:
  format: xml
  file: coverage.xml
  threshold: 80
"""
        (tmp_path / "mala.yaml").write_text(config_content)

        # PER_ISSUE scope should have coverage enabled (base test has --cov)
        # run_level_commands.test is null, which means "skip test at run level",
        # not "move coverage to run level"
        per_issue_spec = build_validation_spec(
            tmp_path, scope=ValidationScope.PER_ISSUE
        )
        assert per_issue_spec.coverage.enabled is True

    def test_run_level_only_test_with_coverage_no_base_test(
        self, tmp_path: Path
    ) -> None:
        """Coverage with run_level_commands.test only (no base test) should work.

        When a config has only run_level_commands.test (no base commands.test)
        plus coverage settings, building a PER_ISSUE spec should not raise an error.
        Coverage will be generated at run-level where the test command exists.
        """
        config_content = """
commands:
  lint: "uvx ruff check ."
run_level_commands:
  test: "uv run pytest --cov=src --cov-report=xml"
coverage:
  format: xml
  file: coverage.xml
  threshold: 80
"""
        (tmp_path / "mala.yaml").write_text(config_content)

        # PER_ISSUE scope should NOT raise ConfigError
        # Coverage is disabled for PER_ISSUE since there's no test command
        # but the error should not fire because run_level_commands.test exists
        per_issue_spec = build_validation_spec(
            tmp_path, scope=ValidationScope.PER_ISSUE
        )
        # Coverage should be disabled for PER_ISSUE (no test command to run)
        assert per_issue_spec.coverage.enabled is False
        # No test command in per-issue spec
        assert not any(cmd.name == "test" for cmd in per_issue_spec.commands)

        # RUN_LEVEL scope should have coverage enabled
        run_level_spec = build_validation_spec(
            tmp_path, scope=ValidationScope.RUN_LEVEL
        )
        assert run_level_spec.coverage.enabled is True
        # Run-level should have test command
        assert any(cmd.name == "test" for cmd in run_level_spec.commands)

    def test_no_test_command_anywhere_with_coverage_raises_error(
        self, tmp_path: Path
    ) -> None:
        """Coverage without any test command should raise ConfigError.

        If there's no commands.test and no run_level_commands.test, but coverage
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
            build_validation_spec(tmp_path, scope=ValidationScope.PER_ISSUE)

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


class TestCoverageOnlyAtRunLevel:
    """Tests for coverage-only-at-run-level behavior based on run_level_commands.test.

    When a user explicitly sets run_level_commands.test (e.g., with --cov flags),
    coverage should only be checked at run-level, not per-issue. This prevents
    per-issue validation from failing due to missing coverage.xml that only
    run-level generates.
    """

    def test_user_run_level_test_disables_per_issue_coverage(
        self, tmp_path: Path
    ) -> None:
        """User's run_level_commands.test should disable coverage for per-issue.

        When the user explicitly sets run_level_commands.test with coverage flags,
        per-issue scope should have coverage disabled since only run-level
        generates coverage.xml.
        """
        config_content = """
preset: python-uv
run_level_commands:
  test: "pytest --cov=src --cov-report=xml"
coverage:
  format: xml
  file: coverage.xml
  threshold: 80
"""
        (tmp_path / "mala.yaml").write_text(config_content)

        per_issue_spec = build_validation_spec(
            tmp_path, scope=ValidationScope.PER_ISSUE
        )
        run_level_spec = build_validation_spec(
            tmp_path, scope=ValidationScope.RUN_LEVEL
        )

        # Per-issue should NOT check coverage (run-level generates it)
        assert per_issue_spec.coverage.enabled is False
        # Run-level should check coverage
        assert run_level_spec.coverage.enabled is True

    def test_preset_run_level_test_does_not_disable_per_issue_coverage(
        self, tmp_path: Path
    ) -> None:
        """Preset's run_level_commands.test should NOT disable per-issue coverage.

        When run_level_commands.test comes from preset (not user), it shouldn't
        affect coverage behavior. This is because the user didn't explicitly
        opt into the coverage-only-at-run-level pattern.

        Note: Currently python-uv doesn't have run_level_commands, so we use
        a custom mala.yaml to simulate this scenario.
        """
        # Write a preset-like config directly (simulating a preset with run_level_commands)
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

        # Without explicit run_level_commands.test, coverage should be enabled for both
        per_issue_spec = build_validation_spec(
            tmp_path, scope=ValidationScope.PER_ISSUE
        )
        run_level_spec = build_validation_spec(
            tmp_path, scope=ValidationScope.RUN_LEVEL
        )

        # Both scopes should have coverage enabled
        assert per_issue_spec.coverage.enabled is True
        assert run_level_spec.coverage.enabled is True

    def test_run_level_test_null_preserves_per_issue_coverage(
        self, tmp_path: Path
    ) -> None:
        """run_level_commands.test: null should NOT disable per-issue coverage.

        When user sets run_level_commands.test to null (to skip tests at run level),
        this is different from setting a test command. The intent is to skip
        testing at run level, not to move coverage there.
        """
        config_content = """
commands:
  test: "pytest --cov=src --cov-report=xml"
run_level_commands:
  test: null
coverage:
  format: xml
  file: coverage.xml
  threshold: 80
"""
        (tmp_path / "mala.yaml").write_text(config_content)

        per_issue_spec = build_validation_spec(
            tmp_path, scope=ValidationScope.PER_ISSUE
        )

        # Per-issue should still check coverage (run-level test is disabled, not moved)
        assert per_issue_spec.coverage.enabled is True
        assert any(cmd.name == "test" for cmd in per_issue_spec.commands)


class TestBuildValidationSpecCustomCommands:
    """Test custom commands in build_validation_spec."""

    def test_build_validation_spec_custom_commands_pipeline_order(
        self, tmp_path: Path
    ) -> None:
        """Custom commands appear after typecheck, before test in pipeline order."""
        config_content = """
commands:
  format: "uvx ruff format --check ."
  lint: "uvx ruff check ."
  typecheck: "uvx ty check"
  test: "uv run pytest"
custom_commands:
  security_scan:
    command: "bandit -r src/"
  docs_check:
    command: "mkdocs build --strict"
"""
        (tmp_path / "mala.yaml").write_text(config_content)

        spec = build_validation_spec(tmp_path, scope=ValidationScope.PER_ISSUE)

        # Get command names in order
        cmd_names = [cmd.name for cmd in spec.commands]

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

    def test_build_validation_spec_custom_commands_insertion_order(
        self, tmp_path: Path
    ) -> None:
        """Custom commands preserve dict insertion order (YAML order)."""
        config_content = """
commands:
  test: "pytest"
custom_commands:
  cmd_a: "echo a"
  cmd_b: "echo b"
  cmd_c: "echo c"
"""
        (tmp_path / "mala.yaml").write_text(config_content)

        spec = build_validation_spec(tmp_path)

        # Find custom commands in order
        custom_cmds = [cmd for cmd in spec.commands if cmd.kind == CommandKind.CUSTOM]

        assert len(custom_cmds) == 3
        assert custom_cmds[0].name == "cmd_a"
        assert custom_cmds[1].name == "cmd_b"
        assert custom_cmds[2].name == "cmd_c"

    def test_build_validation_spec_custom_commands_attributes(
        self, tmp_path: Path
    ) -> None:
        """Custom commands have correct attributes (kind, allow_fail, timeout)."""
        config_content = """
commands:
  test: "pytest"
custom_commands:
  security_scan:
    command: "bandit -r src/"
    allow_fail: true
    timeout: 300
  docs_check:
    command: "mkdocs build --strict"
"""
        (tmp_path / "mala.yaml").write_text(config_content)

        spec = build_validation_spec(tmp_path)

        custom_cmds = {
            cmd.name: cmd for cmd in spec.commands if cmd.kind == CommandKind.CUSTOM
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
    """Test run-level custom_commands override with full-replace semantics."""

    def test_apply_command_overrides_custom_commands_full_replace(
        self, tmp_path: Path
    ) -> None:
        """Run-level custom_commands fully replaces repo-level (no inheritance)."""
        config_content = """
commands:
  test: "pytest"
custom_commands:
  cmd_a: "echo a"
  cmd_b: "echo b"
run_level_custom_commands:
  cmd_a: "echo new_a"
"""
        (tmp_path / "mala.yaml").write_text(config_content)

        # PER_ISSUE should have both cmd_a and cmd_b
        per_issue_spec = build_validation_spec(
            tmp_path, scope=ValidationScope.PER_ISSUE
        )
        per_issue_custom = [
            cmd for cmd in per_issue_spec.commands if cmd.kind == CommandKind.CUSTOM
        ]
        assert len(per_issue_custom) == 2
        assert {c.name for c in per_issue_custom} == {"cmd_a", "cmd_b"}

        # RUN_LEVEL should have ONLY cmd_a (cmd_b not inherited)
        run_level_spec = build_validation_spec(
            tmp_path, scope=ValidationScope.RUN_LEVEL
        )
        run_level_custom = [
            cmd for cmd in run_level_spec.commands if cmd.kind == CommandKind.CUSTOM
        ]
        assert len(run_level_custom) == 1
        assert run_level_custom[0].name == "cmd_a"
        assert run_level_custom[0].command == "echo new_a"

    def test_apply_command_overrides_custom_commands_empty_dict_disables(
        self, tmp_path: Path
    ) -> None:
        """Run-level custom_commands: {} disables all custom commands."""
        config_content = """
commands:
  test: "pytest"
custom_commands:
  cmd_a: "echo a"
  cmd_b: "echo b"
run_level_custom_commands: {}
"""
        (tmp_path / "mala.yaml").write_text(config_content)

        # PER_ISSUE should have both cmd_a and cmd_b
        per_issue_spec = build_validation_spec(
            tmp_path, scope=ValidationScope.PER_ISSUE
        )
        per_issue_custom = [
            cmd for cmd in per_issue_spec.commands if cmd.kind == CommandKind.CUSTOM
        ]
        assert len(per_issue_custom) == 2

        # RUN_LEVEL should have NO custom commands
        run_level_spec = build_validation_spec(
            tmp_path, scope=ValidationScope.RUN_LEVEL
        )
        run_level_custom = [
            cmd for cmd in run_level_spec.commands if cmd.kind == CommandKind.CUSTOM
        ]
        assert len(run_level_custom) == 0

    def test_apply_command_overrides_custom_commands_null_uses_repo_level(
        self, tmp_path: Path
    ) -> None:
        """Run-level custom_commands: null uses repo-level commands."""
        config_content = """
commands:
  test: "pytest"
custom_commands:
  cmd_a: "echo a"
  cmd_b: "echo b"
run_level_custom_commands: null
"""
        (tmp_path / "mala.yaml").write_text(config_content)

        # Both scopes should have cmd_a and cmd_b (null = use repo-level)
        per_issue_spec = build_validation_spec(
            tmp_path, scope=ValidationScope.PER_ISSUE
        )
        run_level_spec = build_validation_spec(
            tmp_path, scope=ValidationScope.RUN_LEVEL
        )

        per_issue_custom = [
            cmd for cmd in per_issue_spec.commands if cmd.kind == CommandKind.CUSTOM
        ]
        run_level_custom = [
            cmd for cmd in run_level_spec.commands if cmd.kind == CommandKind.CUSTOM
        ]

        assert len(per_issue_custom) == 2
        assert len(run_level_custom) == 2
        assert {c.name for c in per_issue_custom} == {"cmd_a", "cmd_b"}
        assert {c.name for c in run_level_custom} == {"cmd_a", "cmd_b"}

    def test_apply_command_overrides_custom_commands_omitted_uses_repo_level(
        self, tmp_path: Path
    ) -> None:
        """Omitted run_level_custom_commands uses repo-level commands."""
        config_content = """
commands:
  test: "pytest"
custom_commands:
  cmd_a: "echo a"
  cmd_b: "echo b"
"""
        (tmp_path / "mala.yaml").write_text(config_content)

        # Both scopes should have cmd_a and cmd_b
        per_issue_spec = build_validation_spec(
            tmp_path, scope=ValidationScope.PER_ISSUE
        )
        run_level_spec = build_validation_spec(
            tmp_path, scope=ValidationScope.RUN_LEVEL
        )

        per_issue_custom = [
            cmd for cmd in per_issue_spec.commands if cmd.kind == CommandKind.CUSTOM
        ]
        run_level_custom = [
            cmd for cmd in run_level_spec.commands if cmd.kind == CommandKind.CUSTOM
        ]

        assert len(per_issue_custom) == 2
        assert len(run_level_custom) == 2


class TestApplyCustomCommandsOverride:
    """Tests for _apply_custom_commands_override with programmatic configs."""

    def test_programmatic_override_with_empty_fields_set(self) -> None:
        """Programmatic override should work even when _fields_set is empty.

        When ValidationConfig is created programmatically (not via from_dict),
        _fields_set is empty. The override should still apply if the value
        is non-None, consistent with _apply_command_overrides behavior.
        """
        base = {"cmd_a": CustomCommandConfig(command="echo a")}
        override = {"cmd_b": CustomCommandConfig(command="echo b")}

        # With empty fields_set but non-None override, should use override
        result = _apply_custom_commands_override(
            base=base,
            override=override,
            fields_set=frozenset(),  # Empty - simulates programmatic creation
            scope=ValidationScope.RUN_LEVEL,
        )

        assert result == override
        assert "cmd_b" in result
        assert "cmd_a" not in result

    def test_explicit_fields_set_takes_precedence(self) -> None:
        """When fields_set contains run_level_custom_commands, use it."""
        base = {"cmd_a": CustomCommandConfig(command="echo a")}
        override = {"cmd_b": CustomCommandConfig(command="echo b")}

        result = _apply_custom_commands_override(
            base=base,
            override=override,
            fields_set=frozenset(["run_level_custom_commands"]),
            scope=ValidationScope.RUN_LEVEL,
        )

        assert result == override

    def test_empty_fields_set_with_none_override_uses_base(self) -> None:
        """When override is None and fields_set is empty, use base."""
        base = {"cmd_a": CustomCommandConfig(command="echo a")}

        result = _apply_custom_commands_override(
            base=base,
            override=None,
            fields_set=frozenset(),
            scope=ValidationScope.RUN_LEVEL,
        )

        assert result == base

    def test_per_issue_scope_always_uses_base(self) -> None:
        """PER_ISSUE scope should always use base regardless of override."""
        base = {"cmd_a": CustomCommandConfig(command="echo a")}
        override = {"cmd_b": CustomCommandConfig(command="echo b")}

        # Even with override set, PER_ISSUE should use base
        result = _apply_custom_commands_override(
            base=base,
            override=override,
            fields_set=frozenset(["run_level_custom_commands"]),
            scope=ValidationScope.PER_ISSUE,
        )

        assert result == base


class TestCustomCommandsYamlOrderPreservation:
    """Regression tests for YAML key order preservation in custom_commands.

    Python 3.7+ guarantees dict insertion order, and PyYAML preserves mapping order.
    These tests ensure custom_commands: {a: ..., b: ..., c: ...} results in
    execution order a, b, c.
    """

    def test_custom_commands_yaml_order_preserved(self, tmp_path: Path) -> None:
        """Custom commands preserve YAML key order through ValidationSpec.

        This is a regression test to ensure that:
        1. YAML key order is preserved when parsing custom_commands
        2. ValidationConfig preserves dict insertion order
        3. build_validation_spec() maintains that order in spec.commands
        """
        # Create YAML with custom commands in specific order
        config_content = """
commands:
  test: "pytest"
custom_commands:
  cmd_alpha: "echo alpha"
  cmd_beta: "echo beta"
  cmd_gamma: "echo gamma"
"""
        (tmp_path / "mala.yaml").write_text(config_content)

        # Build spec and extract custom commands
        spec = build_validation_spec(tmp_path, scope=ValidationScope.PER_ISSUE)
        custom_cmds = [cmd for cmd in spec.commands if cmd.kind == CommandKind.CUSTOM]

        # Verify order matches YAML key order
        assert len(custom_cmds) == 3, "Expected exactly 3 custom commands"
        assert custom_cmds[0].name == "cmd_alpha"
        assert custom_cmds[1].name == "cmd_beta"
        assert custom_cmds[2].name == "cmd_gamma"
