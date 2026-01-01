"""Unit tests for src/validation/spec.py - ValidationSpec and related types.

TDD tests for:
- ValidationSpec construction from CLI inputs
- Code vs docs classification
- Disable list handling
"""

from pathlib import Path

from src.domain.validation.spec import (
    CommandKind,
    CoverageConfig,
    E2EConfig,
    IssueResolution,
    RepoType,
    ResolutionOutcome,
    ValidationArtifacts,
    ValidationCommand,
    ValidationContext,
    ValidationScope,
    ValidationSpec,
    build_validation_spec,
    classify_change,
    detect_repo_type,
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
            command=["uv", "run", "pytest"],
            kind=CommandKind.TEST,
        )
        assert cmd.name == "pytest"
        assert cmd.command == ["uv", "run", "pytest"]
        assert cmd.kind == CommandKind.TEST
        assert cmd.use_test_mutex is False  # default
        assert cmd.allow_fail is False  # default

    def test_command_with_mutex(self) -> None:
        cmd = ValidationCommand(
            name="pytest",
            command=["uv", "run", "pytest"],
            kind=CommandKind.TEST,
            use_test_mutex=True,
        )
        assert cmd.use_test_mutex is True

    def test_command_allow_fail(self) -> None:
        cmd = ValidationCommand(
            name="ty check",
            command=["uvx", "ty", "check"],
            kind=CommandKind.TYPECHECK,
            allow_fail=True,
        )
        assert cmd.allow_fail is True


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

    def test_full_spec(self) -> None:
        cmd = ValidationCommand(
            name="pytest",
            command=["uv", "run", "pytest"],
            kind=CommandKind.TEST,
        )
        spec = ValidationSpec(
            commands=[cmd],
            require_clean_git=True,
            require_pytest_for_code_changes=True,
            coverage=CoverageConfig(enabled=True, min_percent=90.0),
            e2e=E2EConfig(enabled=False),
            scope=ValidationScope.RUN_LEVEL,
        )
        assert len(spec.commands) == 1
        assert spec.coverage.min_percent == 90.0
        assert spec.e2e.enabled is False

    def test_commands_by_kind(self) -> None:
        lint_cmd = ValidationCommand(
            name="ruff check", command=["uvx", "ruff", "check"], kind=CommandKind.LINT
        )
        test_cmd = ValidationCommand(
            name="pytest", command=["uv", "run", "pytest"], kind=CommandKind.TEST
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
    """Test building ValidationSpec from inputs."""

    def test_default_spec_has_all_commands(self) -> None:
        """Default spec includes all validation commands."""
        spec = build_validation_spec(scope=ValidationScope.PER_ISSUE)

        command_names = [cmd.name for cmd in spec.commands]
        assert "uv sync" in command_names
        assert "ruff format" in command_names
        assert "ruff check" in command_names
        assert "ty check" in command_names
        assert "pytest" in command_names

    def test_uv_sync_is_first_command(self) -> None:
        """uv sync must be the first command to set up dependencies."""
        spec = build_validation_spec(scope=ValidationScope.PER_ISSUE)

        assert len(spec.commands) >= 1
        first_cmd = spec.commands[0]
        assert first_cmd.name == "uv sync"
        assert first_cmd.command == ["uv", "sync", "--all-extras"]
        assert first_cmd.kind == CommandKind.SETUP

    def test_uv_sync_has_setup_kind(self) -> None:
        """uv sync command has SETUP kind for proper classification."""
        spec = build_validation_spec(scope=ValidationScope.PER_ISSUE)

        setup_cmds = spec.commands_by_kind(CommandKind.SETUP)
        assert len(setup_cmds) == 1
        assert setup_cmds[0].name == "uv sync"

    def test_command_order_is_setup_then_lint_then_test(self) -> None:
        """Commands should run in order: setup -> format -> lint -> typecheck -> test."""
        spec = build_validation_spec(scope=ValidationScope.PER_ISSUE)

        command_names = [cmd.name for cmd in spec.commands]
        # Verify order
        assert command_names.index("uv sync") < command_names.index("ruff format")
        assert command_names.index("ruff format") < command_names.index("ruff check")
        assert command_names.index("ruff check") < command_names.index("ty check")
        assert command_names.index("ty check") < command_names.index("pytest")

    def test_default_spec_coverage_enabled(self) -> None:
        spec = build_validation_spec(scope=ValidationScope.PER_ISSUE)
        assert spec.coverage.enabled is True
        assert spec.coverage.min_percent is None  # Default is None (no-decrease mode)

    def test_default_spec_e2e_enabled_for_run_level(self) -> None:
        spec = build_validation_spec(scope=ValidationScope.RUN_LEVEL)
        assert spec.e2e.enabled is True

    def test_e2e_does_not_require_morph_api_key(self) -> None:
        """E2E should not require MORPH_API_KEY since MorphLLM is optional."""
        spec = build_validation_spec(scope=ValidationScope.RUN_LEVEL)
        assert spec.e2e.enabled is True
        # MORPH_API_KEY should not be in required_env since Morph is optional
        assert "MORPH_API_KEY" not in spec.e2e.required_env

    def test_per_issue_spec_e2e_disabled(self) -> None:
        """E2E is only for run-level, not per-issue."""
        spec = build_validation_spec(scope=ValidationScope.PER_ISSUE)
        assert spec.e2e.enabled is False

    def test_disable_integration_tests(self) -> None:
        """--disable-validations=integration-tests keeps pytest to unit tests only."""
        spec = build_validation_spec(
            scope=ValidationScope.PER_ISSUE,
            disable_validations={"integration-tests"},
        )

        pytest_cmd = next((cmd for cmd in spec.commands if cmd.name == "pytest"), None)
        assert pytest_cmd is not None
        # Should explicitly select unit tests only
        assert "-m" in pytest_cmd.command
        m_idx = pytest_cmd.command.index("-m")
        assert pytest_cmd.command[m_idx + 1] == "unit"

    def test_disable_coverage(self) -> None:
        """--disable-validations=coverage disables coverage."""
        spec = build_validation_spec(
            scope=ValidationScope.PER_ISSUE,
            disable_validations={"coverage"},
        )
        assert spec.coverage.enabled is False

    def test_disable_e2e(self) -> None:
        """--disable-validations=e2e disables E2E even at run-level."""
        spec = build_validation_spec(
            scope=ValidationScope.RUN_LEVEL,
            disable_validations={"e2e"},
        )
        assert spec.e2e.enabled is False

    def test_disable_post_validate(self) -> None:
        """--disable-validations=post-validate removes all commands including uv sync."""
        spec = build_validation_spec(
            scope=ValidationScope.PER_ISSUE,
            disable_validations={"post-validate"},
        )
        # Should have no validation commands when post-validate is disabled
        command_names = [cmd.name for cmd in spec.commands]
        assert "uv sync" not in command_names
        assert "pytest" not in command_names
        assert "ruff format" not in command_names
        assert "ruff check" not in command_names
        assert "ty check" not in command_names
        assert len(spec.commands) == 0

    def test_custom_coverage_threshold(self) -> None:
        spec = build_validation_spec(
            scope=ValidationScope.PER_ISSUE,
            coverage_threshold=90.0,
        )
        assert spec.coverage.min_percent == 90.0

    def test_multiple_disable_validations(self) -> None:
        """Multiple comma-separated disable values work."""
        spec = build_validation_spec(
            scope=ValidationScope.RUN_LEVEL,
            disable_validations={"coverage", "e2e", "integration-tests"},
        )
        assert spec.coverage.enabled is False
        assert spec.e2e.enabled is False

    def test_command_kinds_are_correct(self) -> None:
        """Commands have appropriate kinds assigned."""
        spec = build_validation_spec(scope=ValidationScope.PER_ISSUE)

        for cmd in spec.commands:
            if cmd.name == "uv sync":
                assert cmd.kind == CommandKind.SETUP
            elif "format" in cmd.name:
                assert cmd.kind == CommandKind.FORMAT
            elif "ruff check" in cmd.name:
                assert cmd.kind == CommandKind.LINT
            elif "ty check" in cmd.name:
                assert cmd.kind == CommandKind.TYPECHECK
            elif "pytest" in cmd.name:
                assert cmd.kind == CommandKind.TEST

    def test_pytest_includes_coverage_flags_when_coverage_enabled(self) -> None:
        """When coverage is enabled, pytest command includes --cov flags."""
        spec = build_validation_spec(scope=ValidationScope.PER_ISSUE)

        pytest_cmd = next((cmd for cmd in spec.commands if cmd.name == "pytest"), None)
        assert pytest_cmd is not None
        assert spec.coverage.enabled is True

        # The pytest command should include coverage flags
        cmd_str = " ".join(pytest_cmd.command)
        assert "--cov=src" in cmd_str
        assert "--cov-report=xml" in cmd_str
        assert "--cov-branch" in cmd_str
        # Default threshold is None, so --cov-fail-under=0 is used
        assert "--cov-fail-under=0" in cmd_str

    def test_pytest_excludes_coverage_flags_when_coverage_disabled(self) -> None:
        """When coverage is disabled, pytest command should not include --cov flags."""
        spec = build_validation_spec(
            scope=ValidationScope.PER_ISSUE,
            disable_validations={"coverage"},
        )

        pytest_cmd = next((cmd for cmd in spec.commands if cmd.name == "pytest"), None)
        assert pytest_cmd is not None
        assert spec.coverage.enabled is False

        # The pytest command should NOT include coverage flags
        cmd_str = " ".join(pytest_cmd.command)
        assert "--cov" not in cmd_str
        assert "--cov-report" not in cmd_str

    def test_coverage_threshold_in_pytest_command(self) -> None:
        """Coverage threshold is correctly passed to pytest --cov-fail-under."""
        spec = build_validation_spec(
            scope=ValidationScope.PER_ISSUE,
            coverage_threshold=92.5,
        )

        pytest_cmd = next((cmd for cmd in spec.commands if cmd.name == "pytest"), None)
        assert pytest_cmd is not None

        cmd_str = " ".join(pytest_cmd.command)
        assert "--cov-fail-under=92.5" in cmd_str

    def test_none_threshold_uses_cov_fail_under_zero(self) -> None:
        """When threshold is None (no-decrease mode), pytest uses --cov-fail-under=0."""
        spec = build_validation_spec(
            scope=ValidationScope.PER_ISSUE,
            coverage_threshold=None,
        )

        pytest_cmd = next((cmd for cmd in spec.commands if cmd.name == "pytest"), None)
        assert pytest_cmd is not None
        assert spec.coverage.min_percent is None

        cmd_str = " ".join(pytest_cmd.command)
        # --cov-fail-under=0 is used to override pyproject.toml threshold
        assert "--cov-fail-under=0" in cmd_str
        # Should NOT have any other threshold value
        assert "--cov-fail-under=85" not in cmd_str

    def test_explicit_threshold_uses_that_value(self) -> None:
        """When threshold is explicitly provided, pytest uses that value."""
        spec = build_validation_spec(
            scope=ValidationScope.PER_ISSUE,
            coverage_threshold=80.0,
        )

        pytest_cmd = next((cmd for cmd in spec.commands if cmd.name == "pytest"), None)
        assert pytest_cmd is not None
        assert spec.coverage.min_percent == 80.0

        cmd_str = " ".join(pytest_cmd.command)
        assert "--cov-fail-under=80.0" in cmd_str
        assert "--cov-fail-under=0" not in cmd_str


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


class TestRepoType:
    """Test RepoType enum."""

    def test_repo_type_values(self) -> None:
        assert RepoType.PYTHON.value == "python"
        assert RepoType.GENERIC.value == "generic"


class TestDetectRepoType:
    """Test detect_repo_type function."""

    def test_detects_python_from_pyproject_toml(self, tmp_path: Path) -> None:
        """Repo with pyproject.toml is detected as Python."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")
        assert detect_repo_type(tmp_path) == RepoType.PYTHON

    def test_detects_python_from_uv_lock(self, tmp_path: Path) -> None:
        """Repo with uv.lock is detected as Python."""
        (tmp_path / "uv.lock").write_text("version = 1\n")
        assert detect_repo_type(tmp_path) == RepoType.PYTHON

    def test_detects_python_from_requirements_txt(self, tmp_path: Path) -> None:
        """Repo with requirements.txt is detected as Python."""
        (tmp_path / "requirements.txt").write_text("requests==2.28.0\n")
        assert detect_repo_type(tmp_path) == RepoType.PYTHON

    def test_generic_for_empty_repo(self, tmp_path: Path) -> None:
        """Empty repo is detected as generic."""
        assert detect_repo_type(tmp_path) == RepoType.GENERIC

    def test_generic_for_non_python_repo(self, tmp_path: Path) -> None:
        """Repo without Python markers is detected as generic."""
        (tmp_path / "package.json").write_text('{"name": "test"}\n')
        (tmp_path / "index.js").write_text("console.log('hello');\n")
        assert detect_repo_type(tmp_path) == RepoType.GENERIC

    def test_pyproject_toml_takes_priority(self, tmp_path: Path) -> None:
        """pyproject.toml is checked first for detection."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")
        (tmp_path / "package.json").write_text('{"name": "test"}\n')
        assert detect_repo_type(tmp_path) == RepoType.PYTHON


class TestBuildValidationSpecWithRepoPath:
    """Test build_validation_spec with repo_path parameter."""

    def test_python_repo_includes_all_commands(self, tmp_path: Path) -> None:
        """Python repo gets all Python validation commands."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")

        spec = build_validation_spec(
            scope=ValidationScope.PER_ISSUE, repo_path=tmp_path
        )

        command_names = [cmd.name for cmd in spec.commands]
        assert "pytest" in command_names
        assert "ruff check" in command_names
        assert "ruff format" in command_names
        assert "ty check" in command_names

    def test_generic_repo_has_no_python_commands(self, tmp_path: Path) -> None:
        """Generic (non-Python) repo gets no Python validation commands."""
        # Empty tmp_path = generic repo
        spec = build_validation_spec(
            scope=ValidationScope.PER_ISSUE, repo_path=tmp_path
        )

        command_names = [cmd.name for cmd in spec.commands]
        assert "pytest" not in command_names
        assert "ruff check" not in command_names
        assert "ruff format" not in command_names
        assert "ty check" not in command_names

    def test_generic_repo_has_empty_commands_list(self, tmp_path: Path) -> None:
        """Generic repo with no Python markers has empty commands list."""
        spec = build_validation_spec(
            scope=ValidationScope.PER_ISSUE, repo_path=tmp_path
        )

        # No Python commands, and E2E is only for run-level
        assert len(spec.commands) == 0

    def test_no_repo_path_defaults_to_python(self) -> None:
        """When repo_path is not provided, default to Python (backward compat)."""
        spec = build_validation_spec(scope=ValidationScope.PER_ISSUE)

        command_names = [cmd.name for cmd in spec.commands]
        assert "pytest" in command_names

    def test_generic_repo_still_respects_disable_validations(
        self, tmp_path: Path
    ) -> None:
        """Disable validations work even for generic repos."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")

        spec = build_validation_spec(
            scope=ValidationScope.PER_ISSUE,
            repo_path=tmp_path,
            disable_validations={"post-validate"},
        )

        command_names = [cmd.name for cmd in spec.commands]
        # pytest/ruff/ty are disabled by post-validate
        assert "pytest" not in command_names
        assert "ruff check" not in command_names

    def test_e2e_enabled_for_run_level_python_repo(self, tmp_path: Path) -> None:
        """E2E is enabled for run-level scope with Python repo."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")

        spec = build_validation_spec(
            scope=ValidationScope.RUN_LEVEL, repo_path=tmp_path
        )

        assert spec.e2e.enabled is True

    def test_e2e_enabled_for_run_level_generic_repo(self, tmp_path: Path) -> None:
        """E2E is still enabled for run-level scope even with generic repo."""
        spec = build_validation_spec(
            scope=ValidationScope.RUN_LEVEL, repo_path=tmp_path
        )

        # E2E is controlled by scope, not repo type
        assert spec.e2e.enabled is True
