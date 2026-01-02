"""Tests for validation gating module.

Tests cover:
- AC 9: custom code_patterns filter files correctly
- Only matching files trigger validation when code_patterns specified
- Non-matching files (e.g., .md when patterns is ["*.py"]) don't trigger
- Empty code_patterns matches all files
- mala.yaml change always triggers validation
- config_files change triggers lint cache invalidation
- setup_files change triggers setup re-run
"""

from dataclasses import dataclass, field

from src.domain.validation.validation_gating import (
    get_config_files_changed,
    get_matching_code_files,
    get_setup_files_changed,
    should_invalidate_lint_cache,
    should_invalidate_setup_cache,
    should_trigger_validation,
)


@dataclass
class MockValidationSpec:
    """Mock ValidationSpec for testing gating logic."""

    code_patterns: list[str] = field(default_factory=list)
    config_files: list[str] = field(default_factory=list)
    setup_files: list[str] = field(default_factory=list)


class TestShouldTriggerValidation:
    """Tests for should_trigger_validation function."""

    def test_empty_changed_files_returns_false(self) -> None:
        """No changes = no validation needed."""
        spec = MockValidationSpec(code_patterns=["*.py"])
        assert should_trigger_validation([], spec) is False

    def test_mala_yaml_always_triggers(self) -> None:
        """mala.yaml change always triggers validation regardless of patterns."""
        spec = MockValidationSpec(code_patterns=["*.py"])
        assert should_trigger_validation(["mala.yaml"], spec) is True
        assert should_trigger_validation(["README.md", "mala.yaml"], spec) is True

    def test_mala_yaml_in_path_triggers(self) -> None:
        """mala.yaml in a path still triggers (basename check)."""
        spec = MockValidationSpec(code_patterns=["*.py"])
        # Full path should still match by basename
        assert should_trigger_validation(["/repo/mala.yaml"], spec) is True

    def test_empty_patterns_matches_all(self) -> None:
        """Empty code_patterns matches all files."""
        spec = MockValidationSpec(code_patterns=[])
        assert should_trigger_validation(["foo.py"], spec) is True
        assert should_trigger_validation(["README.md"], spec) is True
        assert should_trigger_validation(["any/path/file.txt"], spec) is True

    def test_py_pattern_matches_py_files(self) -> None:
        """*.py pattern only matches .py files."""
        spec = MockValidationSpec(code_patterns=["*.py"])
        assert should_trigger_validation(["foo.py"], spec) is True
        assert should_trigger_validation(["src/bar.py"], spec) is True

    def test_py_pattern_does_not_match_other_files(self) -> None:
        """*.py pattern does not match non-.py files."""
        spec = MockValidationSpec(code_patterns=["*.py"])
        assert should_trigger_validation(["README.md"], spec) is False
        assert should_trigger_validation(["config.yaml"], spec) is False
        assert should_trigger_validation(["image.png"], spec) is False

    def test_mixed_files_triggers_if_any_match(self) -> None:
        """Validation triggers if ANY file matches patterns."""
        spec = MockValidationSpec(code_patterns=["*.py"])
        assert should_trigger_validation(["README.md", "foo.py"], spec) is True
        assert should_trigger_validation(["foo.py", "README.md"], spec) is True

    def test_path_pattern_src_star_py(self) -> None:
        """src/*.py pattern matches only direct children of src/."""
        spec = MockValidationSpec(code_patterns=["src/*.py"])
        assert should_trigger_validation(["src/foo.py"], spec) is True
        assert should_trigger_validation(["src/sub/foo.py"], spec) is False
        assert should_trigger_validation(["tests/foo.py"], spec) is False

    def test_recursive_pattern_src_doublestar_py(self) -> None:
        """src/**/*.py pattern matches any depth under src/."""
        spec = MockValidationSpec(code_patterns=["src/**/*.py"])
        assert should_trigger_validation(["src/foo.py"], spec) is True
        assert should_trigger_validation(["src/sub/bar.py"], spec) is True
        assert should_trigger_validation(["src/a/b/c/deep.py"], spec) is True
        assert should_trigger_validation(["tests/test.py"], spec) is False

    def test_multiple_patterns_or_logic(self) -> None:
        """Multiple patterns use OR logic - any match triggers."""
        spec = MockValidationSpec(code_patterns=["src/**/*.py", "tests/**/*.py"])
        assert should_trigger_validation(["src/main.py"], spec) is True
        assert should_trigger_validation(["tests/test_main.py"], spec) is True
        assert should_trigger_validation(["README.md"], spec) is False

    def test_config_files_trigger_validation(self) -> None:
        """Config file changes trigger validation even if code_patterns don't match."""
        spec = MockValidationSpec(
            code_patterns=["*.py"],
            config_files=["pyproject.toml", ".ruff.toml"],
        )
        assert should_trigger_validation(["pyproject.toml"], spec) is True
        assert should_trigger_validation([".ruff.toml"], spec) is True

    def test_setup_files_trigger_validation(self) -> None:
        """Setup file changes trigger validation even if code_patterns don't match."""
        spec = MockValidationSpec(code_patterns=["*.py"], setup_files=["uv.lock"])
        assert should_trigger_validation(["uv.lock"], spec) is True


class TestGetMatchingCodeFiles:
    """Tests for get_matching_code_files function."""

    def test_empty_patterns_returns_all(self) -> None:
        """Empty patterns returns all files."""
        spec = MockValidationSpec(code_patterns=[])
        files = ["a.py", "b.md", "c.txt"]
        assert get_matching_code_files(files, spec) == files

    def test_filters_to_matching_files(self) -> None:
        """Returns only files matching patterns."""
        spec = MockValidationSpec(code_patterns=["*.py"])
        files = ["a.py", "b.md", "c.py"]
        result = get_matching_code_files(files, spec)
        assert result == ["a.py", "c.py"]

    def test_preserves_order(self) -> None:
        """Preserves order of matched files."""
        spec = MockValidationSpec(code_patterns=["*.py"])
        files = ["z.py", "a.py", "m.py"]
        result = get_matching_code_files(files, spec)
        assert result == ["z.py", "a.py", "m.py"]


class TestShouldInvalidateLintCache:
    """Tests for should_invalidate_lint_cache function."""

    def test_empty_changed_files_returns_false(self) -> None:
        """No changes = no invalidation."""
        spec = MockValidationSpec(config_files=["pyproject.toml"])
        assert should_invalidate_lint_cache([], spec) is False

    def test_empty_config_files_returns_false(self) -> None:
        """No config_files patterns = no invalidation."""
        spec = MockValidationSpec(config_files=[])
        assert should_invalidate_lint_cache(["pyproject.toml"], spec) is False

    def test_mala_yaml_invalidates_cache(self) -> None:
        """mala.yaml change invalidates lint cache."""
        spec = MockValidationSpec(config_files=["pyproject.toml"])
        assert should_invalidate_lint_cache(["mala.yaml"], spec) is True

    def test_config_file_change_invalidates(self) -> None:
        """Matching config file change invalidates cache."""
        spec = MockValidationSpec(config_files=["pyproject.toml", ".ruff.toml"])
        assert should_invalidate_lint_cache(["pyproject.toml"], spec) is True
        assert should_invalidate_lint_cache([".ruff.toml"], spec) is True

    def test_non_config_file_does_not_invalidate(self) -> None:
        """Non-matching file does not invalidate cache."""
        spec = MockValidationSpec(config_files=["pyproject.toml"])
        assert should_invalidate_lint_cache(["src/main.py"], spec) is False
        assert should_invalidate_lint_cache(["README.md"], spec) is False

    def test_pattern_matching_for_config_files(self) -> None:
        """Config files support glob patterns."""
        spec = MockValidationSpec(config_files=["*.toml", ".ruff*"])
        assert should_invalidate_lint_cache(["pyproject.toml"], spec) is True
        assert should_invalidate_lint_cache([".ruff.toml"], spec) is True
        assert should_invalidate_lint_cache(["ruff.toml"], spec) is True


class TestShouldInvalidateSetupCache:
    """Tests for should_invalidate_setup_cache function."""

    def test_empty_changed_files_returns_false(self) -> None:
        """No changes = no invalidation."""
        spec = MockValidationSpec(setup_files=["uv.lock"])
        assert should_invalidate_setup_cache([], spec) is False

    def test_empty_setup_files_returns_false(self) -> None:
        """No setup_files patterns = no invalidation."""
        spec = MockValidationSpec(setup_files=[])
        assert should_invalidate_setup_cache(["uv.lock"], spec) is False

    def test_setup_file_change_invalidates(self) -> None:
        """Matching setup file change invalidates cache."""
        spec = MockValidationSpec(setup_files=["uv.lock", "requirements*.txt"])
        assert should_invalidate_setup_cache(["uv.lock"], spec) is True

    def test_requirements_pattern(self) -> None:
        """requirements*.txt pattern matches variants."""
        spec = MockValidationSpec(setup_files=["requirements*.txt"])
        assert should_invalidate_setup_cache(["requirements.txt"], spec) is True
        assert should_invalidate_setup_cache(["requirements-dev.txt"], spec) is True
        assert should_invalidate_setup_cache(["requirements_test.txt"], spec) is True

    def test_non_setup_file_does_not_invalidate(self) -> None:
        """Non-matching file does not invalidate cache."""
        spec = MockValidationSpec(setup_files=["uv.lock"])
        assert should_invalidate_setup_cache(["src/main.py"], spec) is False


class TestGetConfigFilesChanged:
    """Tests for get_config_files_changed function."""

    def test_empty_config_files_returns_empty(self) -> None:
        """No config_files patterns = empty result."""
        spec = MockValidationSpec(config_files=[])
        assert get_config_files_changed(["pyproject.toml"], spec) == []

    def test_returns_matching_config_files(self) -> None:
        """Returns files matching config_files patterns."""
        spec = MockValidationSpec(config_files=["*.toml"])
        files = ["pyproject.toml", "src/main.py", ".ruff.toml"]
        result = get_config_files_changed(files, spec)
        assert set(result) == {"pyproject.toml", ".ruff.toml"}

    def test_includes_mala_yaml_if_changed(self) -> None:
        """Includes mala.yaml in result if it was changed."""
        spec = MockValidationSpec(config_files=["*.toml"])
        files = ["mala.yaml", "pyproject.toml"]
        result = get_config_files_changed(files, spec)
        assert "mala.yaml" in result
        assert "pyproject.toml" in result


class TestGetSetupFilesChanged:
    """Tests for get_setup_files_changed function."""

    def test_empty_setup_files_returns_empty(self) -> None:
        """No setup_files patterns = empty result."""
        spec = MockValidationSpec(setup_files=[])
        assert get_setup_files_changed(["uv.lock"], spec) == []

    def test_returns_matching_setup_files(self) -> None:
        """Returns files matching setup_files patterns."""
        spec = MockValidationSpec(setup_files=["*.lock", "requirements*.txt"])
        files = ["uv.lock", "src/main.py", "requirements.txt"]
        result = get_setup_files_changed(files, spec)
        assert set(result) == {"uv.lock", "requirements.txt"}


class TestIntegrationScenarios:
    """Integration tests for real-world scenarios."""

    def test_python_project_py_only(self) -> None:
        """Python project with *.py patterns - only .py files trigger."""
        spec = MockValidationSpec(
            code_patterns=["*.py"],
            config_files=["pyproject.toml", ".ruff.toml"],
            setup_files=["uv.lock"],
        )

        # Python file changes trigger validation
        assert should_trigger_validation(["src/main.py"], spec) is True

        # Doc changes don't trigger
        assert should_trigger_validation(["README.md"], spec) is False
        assert should_trigger_validation(["docs/guide.md"], spec) is False

        # Config changes trigger validation and invalidate lint cache
        assert should_trigger_validation(["pyproject.toml"], spec) is True
        assert should_invalidate_lint_cache(["pyproject.toml"], spec) is True

    def test_full_stack_project_multiple_patterns(self) -> None:
        """Full-stack project with multiple code patterns."""
        spec = MockValidationSpec(
            code_patterns=["src/**/*.py", "src/**/*.ts", "tests/**/*.py"],
            config_files=["pyproject.toml", "tsconfig.json"],
            setup_files=["uv.lock", "package-lock.json"],
        )

        # Various code files trigger
        assert should_trigger_validation(["src/backend/main.py"], spec) is True
        assert should_trigger_validation(["src/frontend/app.ts"], spec) is True
        assert should_trigger_validation(["tests/test_main.py"], spec) is True

        # Other files don't trigger
        assert should_trigger_validation(["README.md"], spec) is False
        assert should_trigger_validation(["src/frontend/styles.css"], spec) is False

    def test_docs_only_change_with_py_patterns(self) -> None:
        """Docs-only change with *.py patterns skips validation."""
        spec = MockValidationSpec(
            code_patterns=["*.py"],
            config_files=["pyproject.toml"],
            setup_files=["uv.lock"],
        )

        # Pure doc changes
        doc_changes = ["README.md", "docs/api.md", "CHANGELOG.md"]
        assert should_trigger_validation(doc_changes, spec) is False
        assert should_invalidate_lint_cache(doc_changes, spec) is False
        assert should_invalidate_setup_cache(doc_changes, spec) is False

    def test_mala_yaml_override(self) -> None:
        """mala.yaml change overrides all pattern checks."""
        spec = MockValidationSpec(
            code_patterns=["*.py"],  # Would normally skip .md files
            config_files=[],  # No config patterns
            setup_files=[],  # No setup patterns
        )

        # mala.yaml always triggers even with restrictive patterns
        assert should_trigger_validation(["mala.yaml"], spec) is True
        assert should_trigger_validation(["README.md", "mala.yaml"], spec) is True

        # mala.yaml also invalidates lint cache
        assert should_invalidate_lint_cache(["mala.yaml"], spec) is True
