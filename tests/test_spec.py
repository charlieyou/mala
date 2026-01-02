"""Unit tests for src/validation/spec.py - ValidationSpec and related types.

TDD tests for:
- ValidationSpec construction from config files
- Code vs docs classification
- Disable list handling
- Config-driven spec building
"""

import shutil
from pathlib import Path


from src.domain.validation.spec import (
    DEFAULT_COMMAND_TIMEOUT,
    CommandKind,
    CoverageConfig,
    E2EConfig,
    IssueResolution,
    ResolutionOutcome,
    ValidationArtifacts,
    ValidationCommand,
    ValidationContext,
    ValidationScope,
    ValidationSpec,
    build_validation_spec,
    classify_change,
)


class TestValidationScope:
    """Test ValidationScope enum."""

    def test_scope_values(self) -> None:
        assert ValidationScope.PER_ISSUE.value == "per_issue"
        assert ValidationScope.RUN_LEVEL.value == "run_level"


class TestCommandKind:
    """Test CommandKind enum."""

    def test_kind_values(self) -> None:
        assert CommandKind.LINT.value == "lint"
        assert CommandKind.FORMAT.value == "format"
        assert CommandKind.TYPECHECK.value == "typecheck"
        assert CommandKind.TEST.value == "test"
        assert CommandKind.E2E.value == "e2e"


class TestResolutionOutcome:
    """Test ResolutionOutcome enum."""

    def test_outcome_values(self) -> None:
        assert ResolutionOutcome.SUCCESS.value == "success"
        assert ResolutionOutcome.NO_CHANGE.value == "no_change"
        assert ResolutionOutcome.OBSOLETE.value == "obsolete"


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

    def test_command_shell_default_true(self) -> None:
        cmd = ValidationCommand(
            name="test",
            command="echo hello",
            kind=CommandKind.TEST,
        )
        assert cmd.shell is True

    def test_command_shell_explicit_false(self) -> None:
        cmd = ValidationCommand(
            name="test",
            command="echo hello",
            kind=CommandKind.TEST,
            shell=False,
        )
        assert cmd.shell is False


class TestCoverageConfig:
    """Test CoverageConfig dataclass."""

    def test_default_values(self) -> None:
        config = CoverageConfig()
        assert config.enabled is True
        assert config.min_percent is None  # Default is None (no-decrease mode)
        assert config.branch is True
        assert config.report_path is None

    def test_custom_values(self) -> None:
        config = CoverageConfig(
            enabled=True,
            min_percent=90.0,
            branch=False,
            report_path=Path("/tmp/coverage.xml"),
        )
        assert config.min_percent == 90.0
        assert config.branch is False
        assert config.report_path == Path("/tmp/coverage.xml")

    def test_none_min_percent_is_no_decrease_mode(self) -> None:
        """When min_percent is None, coverage uses no-decrease baseline mode."""
        config = CoverageConfig(enabled=True, min_percent=None)
        assert config.min_percent is None


class TestE2EConfig:
    """Test E2EConfig dataclass."""

    def test_default_values(self) -> None:
        config = E2EConfig()
        assert config.enabled is True
        assert config.fixture_root is None
        assert config.command == []
        assert config.required_env == []

    def test_custom_values(self) -> None:
        config = E2EConfig(
            enabled=True,
            fixture_root=Path("/tmp/fixtures"),
            command=["mala", "run"],
            required_env=["MORPH_API_KEY"],
        )
        assert config.fixture_root == Path("/tmp/fixtures")
        assert config.command == ["mala", "run"]
        assert config.required_env == ["MORPH_API_KEY"]


class TestValidationArtifacts:
    """Test ValidationArtifacts dataclass."""

    def test_default_values(self) -> None:
        artifacts = ValidationArtifacts(log_dir=Path("/tmp/logs"))
        assert artifacts.log_dir == Path("/tmp/logs")
        assert artifacts.worktree_path is None
        assert artifacts.worktree_state is None
        assert artifacts.coverage_report is None
        assert artifacts.e2e_fixture_path is None

    def test_custom_values(self) -> None:
        artifacts = ValidationArtifacts(
            log_dir=Path("/tmp/logs"),
            worktree_path=Path("/tmp/worktree"),
            worktree_state="kept",
            coverage_report=Path("/tmp/coverage.xml"),
            e2e_fixture_path=Path("/tmp/e2e"),
        )
        assert artifacts.worktree_path == Path("/tmp/worktree")
        assert artifacts.worktree_state == "kept"


class TestIssueResolution:
    """Test IssueResolution dataclass."""

    def test_success_resolution(self) -> None:
        resolution = IssueResolution(
            outcome=ResolutionOutcome.SUCCESS,
            rationale="All tests pass",
        )
        assert resolution.outcome == ResolutionOutcome.SUCCESS
        assert resolution.rationale == "All tests pass"

    def test_no_change_resolution(self) -> None:
        resolution = IssueResolution(
            outcome=ResolutionOutcome.NO_CHANGE,
            rationale="Issue already addressed in previous commit",
        )
        assert resolution.outcome == ResolutionOutcome.NO_CHANGE

    def test_obsolete_resolution(self) -> None:
        resolution = IssueResolution(
            outcome=ResolutionOutcome.OBSOLETE,
            rationale="Feature removed from scope",
        )
        assert resolution.outcome == ResolutionOutcome.OBSOLETE


class TestValidationContext:
    """Test ValidationContext dataclass."""

    def test_per_issue_context(self) -> None:
        ctx = ValidationContext(
            issue_id="test-123",
            repo_path=Path("/tmp/repo"),
            commit_hash="abc123",
            changed_files=["src/app.py"],
            scope=ValidationScope.PER_ISSUE,
        )
        assert ctx.issue_id == "test-123"
        assert ctx.repo_path == Path("/tmp/repo")
        assert ctx.commit_hash == "abc123"
        assert ctx.log_path is None  # default
        assert ctx.changed_files == ["src/app.py"]
        assert ctx.scope == ValidationScope.PER_ISSUE

    def test_run_level_context(self) -> None:
        ctx = ValidationContext(
            issue_id=None,
            repo_path=Path("/tmp/repo"),
            commit_hash="def456",
            changed_files=[],
            scope=ValidationScope.RUN_LEVEL,
        )
        assert ctx.issue_id is None
        assert ctx.scope == ValidationScope.RUN_LEVEL


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
        """--disable-validations=post-validate removes all commands."""
        config_src = Path("tests/fixtures/mala-configs/go-project.yaml")
        config_dst = tmp_path / "mala.yaml"
        shutil.copy(config_src, config_dst)

        spec = build_validation_spec(
            tmp_path,
            disable_validations={"post-validate"},
        )

        assert spec.commands == []

    def test_disable_coverage(self, tmp_path: Path) -> None:
        """--disable-validations=coverage disables coverage."""
        config_src = Path("tests/fixtures/mala-configs/node-project.yaml")
        config_dst = tmp_path / "mala.yaml"
        shutil.copy(config_src, config_dst)

        spec = build_validation_spec(
            tmp_path,
            disable_validations={"coverage"},
        )

        assert spec.coverage.enabled is False

    def test_disable_e2e(self, tmp_path: Path) -> None:
        """--disable-validations=e2e disables E2E."""
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
