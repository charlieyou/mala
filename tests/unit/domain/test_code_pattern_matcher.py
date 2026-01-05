"""Tests for code_pattern_matcher module."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

from src.domain.validation.code_pattern_matcher import (
    filter_matching_files,
    glob_to_regex,
    matches_pattern,
)


class TestGlobToRegex:
    """Tests for glob_to_regex function."""

    def test_literal_string(self) -> None:
        """Test that literal strings match exactly."""
        regex = glob_to_regex("foo.py")
        assert regex.match("foo.py")
        assert not regex.match("bar.py")
        assert not regex.match("foo.pyc")

    def test_single_star_matches_non_slash(self) -> None:
        """Test that * matches any character except /."""
        regex = glob_to_regex("*.py")
        assert regex.match("foo.py")
        assert regex.match("bar.py")
        assert not regex.match("foo.js")
        assert not regex.match("foo/bar.py")

    def test_double_star_matches_anything(self) -> None:
        """Test that ** matches any character including /."""
        regex = glob_to_regex("**/*.py")
        assert regex.match("foo.py")
        assert regex.match("src/foo.py")
        assert regex.match("src/sub/deep/file.py")

    def test_question_mark(self) -> None:
        """Test that ? matches a single non-slash character."""
        regex = glob_to_regex("test_?.py")
        assert regex.match("test_a.py")
        assert not regex.match("test_ab.py")
        assert not regex.match("test_.py")

    def test_escapes_regex_special_chars(self) -> None:
        """Test that regex special characters are escaped."""
        regex = glob_to_regex("file[1].py")
        assert regex.match("file[1].py")
        assert not regex.match("file1.py")

    def test_doublestar_slash_requires_directory_boundary(self) -> None:
        """Test that **/ only matches at directory boundaries, not within filenames.

        This is a regression test for the bug where **/test_*.py would
        incorrectly match 'mytest_main.py' because ** was translated to .*
        which would match the 'my' prefix.
        """
        regex = glob_to_regex("**/test_*.py")
        # Should match: files starting with test_ at any directory depth
        assert regex.match("test_main.py")
        assert regex.match("tests/test_utils.py")
        assert regex.match("src/tests/test_foo.py")
        # Should NOT match: files where 'test_' is not at start of filename
        assert not regex.match("mytest_main.py")
        assert not regex.match("src/mytest_foo.py")
        assert not regex.match("abctest_bar.py")

    def test_doublestar_slash_regex_pattern(self) -> None:
        """Test that **/ is translated to the correct regex pattern."""
        regex = glob_to_regex("**/foo.py")
        # Pattern should be (?:.*/)?foo\.py
        # This means: optionally match anything ending with /, then foo.py
        assert regex.pattern == "^(?:.*/)?foo\\.py$"

        regex2 = glob_to_regex("src/**/bar.py")
        assert regex2.pattern == "^src/(?:.*/)?bar\\.py$"

    def test_invalid_pattern_treated_as_literal(
        self, monkeypatch: "pytest.MonkeyPatch", caplog: "pytest.LogCaptureFixture"
    ) -> None:
        """Invalid regex patterns fall back to literal matching with warning.

        When the generated regex would be invalid, glob_to_regex logs a warning
        and returns a pattern that matches the input literally.
        """
        import re

        from src.domain.validation import code_pattern_matcher

        # Force re.compile to fail on first call only (the glob-generated pattern),
        # then succeed on second call (the literal fallback pattern)
        original_compile = re.compile
        call_count = 0

        def mock_compile(
            pattern: str, *args: object, **kwargs: object
        ) -> re.Pattern[str]:
            nonlocal call_count
            call_count += 1
            # Only fail on first call (the glob pattern), not the fallback
            if call_count == 1:
                raise re.error("mock error")
            return original_compile(pattern, *args, **kwargs)

        monkeypatch.setattr(re, "compile", mock_compile)

        # Pattern should trigger the fallback
        regex = code_pattern_matcher.glob_to_regex("test*.py")

        # Fallback should escape the pattern and match literally
        assert regex.match("test*.py")
        assert not regex.match("testfoo.py")
        # Warning should be logged
        assert "Invalid glob pattern" in caplog.text


class TestMatchesPattern:
    """Tests for matches_pattern function."""

    def test_star_py_matches_py_files(self) -> None:
        """AC: *.py matches foo.py, not foo.js."""
        assert matches_pattern("foo.py", "*.py")
        assert not matches_pattern("foo.js", "*.py")
        assert matches_pattern("bar.py", "*.py")
        assert matches_pattern("test_utils.py", "*.py")

    def test_star_py_matches_nested_basename(self) -> None:
        """Filename-only pattern matches basename of nested files."""
        assert matches_pattern("src/foo.py", "*.py")
        assert matches_pattern("src/sub/deep/file.py", "*.py")

    def test_src_star_py_single_level(self) -> None:
        """AC: src/*.py matches src/foo.py, not src/sub/foo.py."""
        assert matches_pattern("src/foo.py", "src/*.py")
        assert not matches_pattern("src/sub/foo.py", "src/*.py")
        assert matches_pattern("src/bar.py", "src/*.py")

    def test_src_doublestar_py_recursive(self) -> None:
        """AC: src/**/*.py matches src/foo.py and src/sub/deep/file.py."""
        assert matches_pattern("src/foo.py", "src/**/*.py")
        assert matches_pattern("src/sub/deep/file.py", "src/**/*.py")
        assert matches_pattern("src/a/b/c/d.py", "src/**/*.py")

    def test_doublestar_test_star_py(self) -> None:
        """AC: **/test_*.py matches test_main.py and tests/test_utils.py."""
        assert matches_pattern("test_main.py", "**/test_*.py")
        assert matches_pattern("tests/test_utils.py", "**/test_*.py")
        assert matches_pattern("src/tests/test_foo.py", "**/test_*.py")

    def test_doublestar_slash_does_not_match_partial_filename(self) -> None:
        """Test that **/ does not match within a filename (only at directory boundaries).

        Regression test: **/test_*.py should NOT match 'mytest_main.py' or 'src/abctest.py'
        because ** should only match complete directory segments, not partial prefixes.
        """
        # These should NOT match - 'test_' is not at start of filename
        assert not matches_pattern("mytest_main.py", "**/test_*.py")
        assert not matches_pattern("src/abctest_foo.py", "**/test_*.py")
        assert not matches_pattern("lib/prefix_test_bar.py", "**/test_*.py")

        # These SHOULD match - 'test_' is at start of filename
        assert matches_pattern("test_main.py", "**/test_*.py")
        assert matches_pattern("tests/test_utils.py", "**/test_*.py")
        assert matches_pattern("a/b/c/test_deep.py", "**/test_*.py")

    def test_path_separator_normalization(self) -> None:
        """Test that backslashes are normalized to forward slashes."""
        assert matches_pattern("src\\foo.py", "src/*.py")
        assert matches_pattern("src/foo.py", "src\\*.py")

    def test_leading_slash_stripped(self) -> None:
        """Test that leading slashes are handled correctly."""
        assert matches_pattern("/src/foo.py", "src/*.py")


class TestFilterMatchingFiles:
    """Tests for filter_matching_files function."""

    def test_empty_patterns_matches_everything(self) -> None:
        """AC: Empty patterns list matches everything."""
        files = ["foo.py", "bar.js", "README.md"]
        result = filter_matching_files(files, [])
        assert result == files

    def test_single_pattern(self) -> None:
        """Test filtering with a single pattern."""
        files = ["foo.py", "bar.js", "baz.py"]
        result = filter_matching_files(files, ["*.py"])
        assert result == ["foo.py", "baz.py"]

    def test_multiple_patterns(self) -> None:
        """Test filtering with multiple patterns (OR logic)."""
        files = ["foo.py", "bar.js", "baz.ts", "README.md"]
        result = filter_matching_files(files, ["*.py", "*.js"])
        assert result == ["foo.py", "bar.js"]

    def test_path_patterns(self) -> None:
        """Test filtering with path patterns."""
        files = [
            "src/foo.py",
            "src/sub/bar.py",
            "tests/test_foo.py",
            "README.md",
        ]
        result = filter_matching_files(files, ["src/*.py"])
        assert result == ["src/foo.py"]

    def test_recursive_patterns(self) -> None:
        """Test filtering with recursive patterns."""
        files = [
            "src/foo.py",
            "src/sub/bar.py",
            "src/sub/deep/baz.py",
            "tests/test_foo.py",
        ]
        result = filter_matching_files(files, ["src/**/*.py"])
        assert result == ["src/foo.py", "src/sub/bar.py", "src/sub/deep/baz.py"]

    def test_no_matches(self) -> None:
        """Test when no files match the patterns."""
        files = ["foo.py", "bar.py"]
        result = filter_matching_files(files, ["*.js"])
        assert result == []

    def test_empty_files_list(self) -> None:
        """Test with empty files list."""
        result = filter_matching_files([], ["*.py"])
        assert result == []

    def test_returns_correct_subset(self) -> None:
        """AC: filter_matching_files returns correct subset."""
        files = [
            "src/main.py",
            "src/utils.py",
            "src/sub/helper.py",
            "tests/test_main.py",
            "tests/test_utils.py",
            "README.md",
            "pyproject.toml",
        ]
        # Match all Python files in src/ (any depth) and all test files
        patterns = ["src/**/*.py", "**/test_*.py"]
        result = filter_matching_files(files, patterns)
        assert set(result) == {
            "src/main.py",
            "src/utils.py",
            "src/sub/helper.py",
            "tests/test_main.py",
            "tests/test_utils.py",
        }

    def test_doublestar_pattern_excludes_partial_matches(self) -> None:
        """Test that **/ patterns don't match files with partial name matches."""
        files = [
            "test_main.py",
            "mytest_main.py",  # Should NOT match
            "tests/test_utils.py",
            "src/abctest_foo.py",  # Should NOT match
            "lib/test_helper.py",
        ]
        result = filter_matching_files(files, ["**/test_*.py"])
        assert result == ["test_main.py", "tests/test_utils.py", "lib/test_helper.py"]


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_pattern_with_dots(self) -> None:
        """Test patterns with dots are handled correctly."""
        assert matches_pattern("file.test.py", "*.py")
        assert matches_pattern("file.test.py", "file.test.py")

    def test_pattern_with_special_regex_chars(self) -> None:
        """Test patterns with regex special characters."""
        assert matches_pattern("file(1).py", "file(1).py")
        assert matches_pattern("file[1].py", "file[1].py")
        assert matches_pattern("file+test.py", "file+test.py")

    def test_doublestar_at_start(self) -> None:
        """Test ** at the start of pattern."""
        assert matches_pattern("foo.py", "**/*.py")
        assert matches_pattern("a/b/c.py", "**/*.py")

    def test_doublestar_in_middle(self) -> None:
        """Test ** in the middle of pattern."""
        assert matches_pattern("src/a/b/c/test.py", "src/**/test.py")
        assert matches_pattern("src/test.py", "src/**/test.py")

    def test_multiple_stars(self) -> None:
        """Test patterns with multiple * characters."""
        assert matches_pattern("test_foo_bar.py", "test_*_*.py")
        assert not matches_pattern("test_foo.py", "test_*_*.py")

    def test_empty_pattern(self) -> None:
        """Test empty pattern only matches empty string."""
        regex = glob_to_regex("")
        assert regex.match("")
        assert not regex.match("foo")

    def test_star_only(self) -> None:
        """Test pattern that is just *.

        Since * is a filename-only pattern (no /), it matches against
        the basename. So path/file.txt matches * because basename file.txt
        matches *.
        """
        assert matches_pattern("anything.txt", "*")
        # Filename-only pattern matches against basename
        assert matches_pattern("path/file.txt", "*")

    def test_doublestar_only(self) -> None:
        """Test pattern that is just **."""
        assert matches_pattern("anything.txt", "**")
        assert matches_pattern("path/file.txt", "**")
        assert matches_pattern("a/b/c/d/e.txt", "**")
