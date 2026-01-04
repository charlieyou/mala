"""Tests for tool_name_extractor module."""

import logging

import pytest

from src.core.tool_name_extractor import extract_tool_name


class TestExtractToolName:
    """Tests for extract_tool_name function."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("command", "expected"),
        [
            # Basic commands
            ("eslint .", "eslint"),
            ("pytest tests/", "pytest"),
            ("ruff check .", "ruff"),
            # npx wrapper (AC: "npx eslint" → "eslint")
            ("npx eslint .", "eslint"),
            ("npx prettier --write .", "prettier"),
            # uvx wrapper (AC: "uvx ruff check ." → "ruff")
            ("uvx ruff check .", "ruff"),
            ("uvx mypy src/", "mypy"),
            # uv run wrapper (AC: "uv run pytest" → "pytest")
            ("uv run pytest", "pytest"),
            ("uv run ruff check .", "ruff"),
            # bunx wrapper
            ("bunx eslint .", "eslint"),
            # pipx wrappers
            ("pipx black .", "black"),
            ("pipx run mypy .", "mypy"),
            # python -m wrapper
            ("python -m pytest", "pytest"),
            ("python3 -m mypy src/", "mypy"),
            # poetry run wrapper
            ("poetry run pytest", "pytest"),
            # Compound commands (AC: "go test ./..." → "go test")
            ("go test ./...", "go test"),
            ("go build .", "go build"),
            ("go vet ./...", "go vet"),
            ("go fmt .", "go fmt"),
            # Cargo compound commands (AC: "cargo clippy" → "cargo clippy")
            ("cargo clippy", "cargo clippy"),
            ("cargo test", "cargo test"),
            ("cargo build", "cargo build"),
            ("cargo check", "cargo check"),
            # npm test compound
            ("npm test", "npm test"),
            # npm run script (AC: "npm run lint" → "npm run:lint")
            ("npm run lint", "npm run:lint"),
            ("npm run build", "npm run:build"),
            ("npm run test:unit", "npm run:test:unit"),
            # pnpm and yarn equivalents
            ("pnpm run lint", "pnpm run:lint"),
            ("yarn run test", "yarn run:test"),
            ("pnpm test", "pnpm test"),
            ("yarn test", "yarn test"),
        ],
    )
    def test_spec_examples(self, command: str, expected: str) -> None:
        """Test all spec examples from the issue."""
        assert extract_tool_name(command) == expected

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("command", "expected"),
        [
            # Path stripping (AC: /usr/bin/eslint → eslint)
            ("/usr/bin/eslint .", "eslint"),
            ("/usr/local/bin/pytest", "pytest"),
            ("./node_modules/.bin/eslint", "eslint"),
            ("/home/user/.local/bin/ruff check", "ruff"),
        ],
    )
    def test_path_stripping(self, command: str, expected: str) -> None:
        """Test path prefix stripping works."""
        assert extract_tool_name(command) == expected

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("command", "expected"),
        [
            # Environment variable assignments
            ("NODE_ENV=test npx eslint .", "eslint"),
            ("FOO=bar BAZ=qux pytest", "pytest"),
            ("CI=true npm test", "npm test"),
        ],
    )
    def test_env_var_assignments(self, command: str, expected: str) -> None:
        """Test that env var assignments are skipped."""
        assert extract_tool_name(command) == expected

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("command", "expected"),
        [
            # Shell built-ins skipped
            ("export FOO=bar && eslint .", "eslint"),
            ("cd /tmp && pytest", "pytest"),
            ("unset FOO && pytest", "pytest"),
        ],
    )
    def test_shell_builtins(self, command: str, expected: str) -> None:
        """Test that shell built-ins are skipped."""
        assert extract_tool_name(command) == expected

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("command", "expected"),
        [
            # Shell operators
            ("npm install && npm test", "npm test"),
            ("eslint . || exit 1", "eslint"),
            ("echo starting; pytest", "pytest"),
            ("cat file.txt | grep pattern", "cat"),
        ],
    )
    def test_shell_operators(self, command: str, expected: str) -> None:
        """Test handling of shell operators."""
        assert extract_tool_name(command) == expected

    @pytest.mark.unit
    def test_unknown_wrapper_fallback(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that unknown commands fall back gracefully."""
        # This is not a known wrapper pattern, should just return the command
        result = extract_tool_name("some-custom-tool --flag arg")
        assert result == "some-custom-tool"

    @pytest.mark.unit
    def test_empty_command_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test empty command returns empty string with warning."""
        with caplog.at_level(logging.WARNING):
            result = extract_tool_name("")
            assert result == ""
            assert "Empty command" in caplog.text

    @pytest.mark.unit
    def test_whitespace_only_command(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test whitespace-only command returns empty string."""
        with caplog.at_level(logging.WARNING):
            result = extract_tool_name("   ")
            assert result == ""

    @pytest.mark.unit
    def test_malformed_quotes(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test malformed command with unclosed quotes falls back to split."""
        with caplog.at_level(logging.WARNING):
            # Unclosed quote should trigger shlex fallback
            result = extract_tool_name('eslint "unclosed')
            # Should still extract the tool name
            assert result == "eslint"
            assert "shlex.split failed" in caplog.text

    @pytest.mark.unit
    def test_wrapper_without_command(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test wrapper without following command returns wrapper name."""
        with caplog.at_level(logging.WARNING):
            result = extract_tool_name("npx")
            assert result == "npx"
            assert "without following command" in caplog.text

    @pytest.mark.unit
    def test_multi_token_wrapper_without_command(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test multi-token wrapper without following command."""
        with caplog.at_level(logging.WARNING):
            result = extract_tool_name("uv run")
            assert result == "uv"

    @pytest.mark.unit
    def test_only_env_assignments(self) -> None:
        """Test command with only env assignments returns empty."""
        result = extract_tool_name("FOO=bar BAZ=qux")
        assert result == ""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("command", "expected"),
        [
            # Edge cases with flags
            ("npx -y eslint .", "eslint"),
            ("uvx --quiet ruff check", "ruff"),
            ("npx -p eslint eslint .", "eslint"),
            ("uvx --from ruff ruff check", "ruff"),
            ("uv run --extra dev pytest", "pytest"),
            # uv run with --group flag (issue mala-drqh)
            ("uv run --group dev pytest", "pytest"),
            ("uv run --only-group test pytest", "pytest"),
            ("uv run --no-group dev pytest", "pytest"),
            # uv run with other value flags
            ("uv run --python 3.12 pytest", "pytest"),
            ("uv run -C /tmp pytest", "pytest"),
            ("uv run --directory /tmp pytest", "pytest"),
            # poetry run with --quiet (standalone flag)
            ("poetry run --quiet pytest", "pytest"),
        ],
    )
    def test_wrapper_with_flags(self, command: str, expected: str) -> None:
        """Test wrappers with flags before the tool name."""
        assert extract_tool_name(command) == expected

    @pytest.mark.unit
    def test_npm_run_without_script(self) -> None:
        """Test npm run without script name."""
        result = extract_tool_name("npm run")
        assert result == "npm run"

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("command", "expected"),
        [
            # Uppercase compound commands (issue mala-tqi4)
            ("CARGO clippy", "cargo clippy"),
            ("CARGO CLIPPY", "cargo clippy"),
            ("GO test ./...", "go test"),
            ("GO TEST ./...", "go test"),
            ("NPM test", "npm test"),
            ("NPM TEST", "npm test"),
            # Uppercase wrappers
            ("NPX eslint .", "eslint"),
            ("UVX ruff check .", "ruff"),
            ("BUNX eslint .", "eslint"),
            # Uppercase multi-token wrappers
            ("UV RUN pytest", "pytest"),
            ("PYTHON -M pytest", "pytest"),
            ("POETRY RUN pytest", "pytest"),
            # Mixed case
            ("Cargo Clippy", "cargo clippy"),
            ("Npm Run lint", "npm run:lint"),
            ("Npx Eslint .", "eslint"),
        ],
    )
    def test_case_insensitive_matching(self, command: str, expected: str) -> None:
        """Test case-insensitive matching for wrappers and compound commands.

        Uppercase versions of commands like CARGO clippy, NPX eslint should be
        recognized correctly and return lowercase normalized tool names for
        consistent lint_type identification.
        """
        assert extract_tool_name(command) == expected

    @pytest.mark.unit
    def test_complex_real_world_command(self) -> None:
        """Test a complex real-world command."""
        result = extract_tool_name(
            "CI=true NODE_ENV=test npx --yes eslint --ext .ts,.tsx src/"
        )
        assert result == "eslint"

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("command", "expected"),
        [
            # set with flags only, followed by real command via &&
            ("set -e && pytest", "pytest"),
            ("set -ex && ruff check .", "ruff"),
            ("set -o pipefail && npm test", "npm test"),
            # set -- skips ALL remaining tokens (they're positional params)
            # The real command comes after &&
            ("set -- pytest -q && echo done", "echo"),
            ("set -- foo bar baz && ruff check", "ruff"),
            # set -- with no following command segment returns empty
            ("set -- pytest -q", ""),
            ("set -- foo bar", ""),
            # Combined: set flags then set -- in same segment
            ("set -e && set -- foo && pytest", "pytest"),
            # set +e (unset errexit) followed by command
            ("set +e && eslint .", "eslint"),
            ("set +o pipefail && pytest", "pytest"),
        ],
    )
    def test_set_builtin_handling(self, command: str, expected: str) -> None:
        """Test that set builtin and its arguments are handled correctly.

        The set builtin has special argument patterns:
        - Flags like -e, -x, +e, -o pipefail
        - -- marks end of options; all remaining tokens are positional
          parameters (not commands to execute)
        """
        assert extract_tool_name(command) == expected
