"""Tests for domain/prompts.py module.

Tests the prompt loading utilities and build_prompt_validation_commands function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.domain.prompts import (
    build_continuation_prompt,
    build_prompt_validation_commands,
    extract_checkpoint,
    format_implementer_prompt,
    load_prompt,
    load_prompts,
)
from src.domain.validation.config import PromptValidationCommands
from src.infra.tools.env import PROMPTS_DIR

if TYPE_CHECKING:
    from pathlib import Path


class TestLoadPrompts:
    """Tests for load_prompts function."""

    def test_gate_followup_contains_template_placeholders(self) -> None:
        """Gate followup prompt contains expected placeholders."""
        result = load_prompts(PROMPTS_DIR)
        # Check for placeholders that should be in the template
        assert "{attempt}" in result.gate_followup_prompt
        assert "{max_attempts}" in result.gate_followup_prompt
        assert "{failure_reasons}" in result.gate_followup_prompt
        assert "{issue_id}" in result.gate_followup_prompt
        # Canonical-wrapper substitution replaces the per-command placeholders
        assert "{missing_command_wrappers}" in result.gate_followup_prompt
        assert "{lint_command}" not in result.gate_followup_prompt
        assert "{format_command}" not in result.gate_followup_prompt
        assert "{typecheck_command}" not in result.gate_followup_prompt
        assert "{custom_commands_section}" not in result.gate_followup_prompt
        assert "{test_command}" not in result.gate_followup_prompt

    def test_raises_on_missing_prompt_dir(self, tmp_path: Path) -> None:
        """load_prompts raises FileNotFoundError for missing directory."""
        with pytest.raises(FileNotFoundError):
            load_prompts(tmp_path / "nonexistent")

    def test_review_agent_prompt_defaults_empty_when_missing(
        self, tmp_path: Path
    ) -> None:
        """review_agent_prompt defaults to empty string when file is missing."""
        # Create minimal prompt dir with required files but no review_agent.md
        (tmp_path / "implementer_prompt.md").write_text("impl")
        (tmp_path / "review_followup.md").write_text("review")
        (tmp_path / "gate_followup.md").write_text("gate")
        (tmp_path / "fixer.md").write_text("fixer")
        (tmp_path / "idle_resume.md").write_text("idle")
        (tmp_path / "checkpoint_request.md").write_text("checkpoint")
        (tmp_path / "continuation.md").write_text("continuation")

        result = load_prompts(tmp_path)
        assert result.review_agent_prompt == ""

    def test_review_agent_prompt_loaded_when_exists(self, tmp_path: Path) -> None:
        """review_agent_prompt is loaded when review_agent.md exists."""
        # Create minimal prompt dir with all files including review_agent.md
        (tmp_path / "implementer_prompt.md").write_text("impl")
        (tmp_path / "review_followup.md").write_text("review")
        (tmp_path / "gate_followup.md").write_text("gate")
        (tmp_path / "fixer.md").write_text("fixer")
        (tmp_path / "idle_resume.md").write_text("idle")
        (tmp_path / "checkpoint_request.md").write_text("checkpoint")
        (tmp_path / "continuation.md").write_text("continuation")
        (tmp_path / "review_agent.md").write_text("Review agent system prompt content")

        result = load_prompts(tmp_path)
        assert result.review_agent_prompt == "Review agent system prompt content"


class TestLoadPrompt:
    """Tests for load_prompt function."""

    def test_loads_checkpoint_request(self) -> None:
        """load_prompt loads checkpoint_request.md correctly."""
        content = load_prompt("checkpoint_request")
        assert "<checkpoint>" in content
        assert "## Goal" in content
        assert "## Completed Work" in content
        assert "## Remaining Tasks" in content

    def test_loads_continuation(self) -> None:
        """load_prompt loads continuation.md correctly."""
        content = load_prompt("continuation")
        assert "{checkpoint}" in content
        assert "continuation" in content.lower()

    def test_raises_on_missing_prompt(self) -> None:
        """load_prompt raises FileNotFoundError for missing prompt."""
        with pytest.raises(FileNotFoundError):
            load_prompt("nonexistent_prompt")

    def test_uses_custom_prompt_dir(self, tmp_path: Path) -> None:
        """load_prompt uses custom prompt_dir when provided."""
        prompt_file = tmp_path / "custom_prompt.md"
        prompt_file.write_text("Custom prompt content")
        result = load_prompt("custom_prompt", prompt_dir=tmp_path)
        assert result == "Custom prompt content"


class TestBuildContinuationPrompt:
    """Tests for build_continuation_prompt function."""

    def test_formats_checkpoint_into_template(self) -> None:
        """build_continuation_prompt inserts checkpoint into template."""
        template = "Continue from checkpoint:\n{checkpoint}\n\nEnd continuation."
        checkpoint = """<checkpoint>
## Goal
Complete the feature implementation.

## Completed Work
- Added main.py:10 function
</checkpoint>"""
        result = build_continuation_prompt(template, checkpoint)
        assert checkpoint in result
        assert "{checkpoint}" not in result
        assert "continuation" in result.lower()

    def test_handles_curly_braces_in_checkpoint(self) -> None:
        """build_continuation_prompt handles curly braces in checkpoint text."""
        template = "Checkpoint: {checkpoint}"
        checkpoint = """## Code snippet
```python
data = {"key": "value", "nested": {}}
```"""
        result = build_continuation_prompt(template, checkpoint)
        assert checkpoint in result
        assert "{checkpoint}" not in result
        # Verify the braces are preserved exactly
        assert '{"key": "value"' in result

    def test_replaces_placeholder_in_template(self) -> None:
        """build_continuation_prompt replaces {checkpoint} with content."""
        template = "Custom template: {checkpoint}"
        result = build_continuation_prompt(template, "test content")
        assert result == "Custom template: test content"


class TestExtractCheckpoint:
    """Tests for extract_checkpoint function."""

    def test_extracts_well_formed_checkpoint(self) -> None:
        """Extracts content between checkpoint tags."""
        text = "<checkpoint>some content</checkpoint>"
        assert extract_checkpoint(text) == "some content"

    def test_extracts_multiline_checkpoint(self) -> None:
        """Extracts multiline content from checkpoint."""
        text = """<checkpoint>
## Goal
Complete feature

## Completed Work
- Added main.py:10
</checkpoint>"""
        result = extract_checkpoint(text)
        assert "## Goal" in result
        assert "## Completed Work" in result
        assert "Added main.py:10" in result

    def test_extracts_from_xml_code_block_wrapper(self) -> None:
        """Extracts checkpoint from content wrapped in ```xml block."""
        text = """```xml
<checkpoint>content inside</checkpoint>
```"""
        assert extract_checkpoint(text) == "content inside"

    def test_extracts_from_markdown_code_block_wrapper(self) -> None:
        """Extracts checkpoint from content wrapped in ```markdown block."""
        text = """```markdown
<checkpoint>wrapped content</checkpoint>
```"""
        assert extract_checkpoint(text) == "wrapped content"

    def test_extracts_from_plain_code_block_wrapper(self) -> None:
        """Extracts checkpoint from content wrapped in ``` block."""
        text = """```
<checkpoint>plain wrapped</checkpoint>
```"""
        assert extract_checkpoint(text) == "plain wrapped"

    def test_returns_full_text_when_no_tags(self) -> None:
        """Returns full text as fallback when no checkpoint tags found."""
        text = "No checkpoint here, just plain text."
        assert extract_checkpoint(text) == text

    def test_fallback_strips_code_block_wrapper(self) -> None:
        """Strips code block wrapper in fallback when no checkpoint tags."""
        text = """```xml
Just some content without tags
```"""
        assert extract_checkpoint(text) == "Just some content without tags"

    def test_fallback_strips_indented_closing_fence(self) -> None:
        """Strips indented closing code fence in fallback mode."""
        text = "```xml\nJust some content\n  ```"
        assert extract_checkpoint(text) == "Just some content"

    def test_fallback_strips_compact_code_block(self) -> None:
        """Strips closing fence in compact blocks without trailing newline."""
        text = "```text\ncontent```"
        assert extract_checkpoint(text) == "content"

    def test_returns_full_text_for_empty_input(self) -> None:
        """Returns empty string for empty input."""
        assert extract_checkpoint("") == ""

    def test_returns_full_text_for_whitespace_only(self) -> None:
        """Returns whitespace for whitespace-only input."""
        text = "   \n\t  "
        assert extract_checkpoint(text) == text

    def test_handles_nested_tags_returns_outermost(self) -> None:
        """Returns outermost checkpoint content when tags are nested."""
        text = "<checkpoint>outer <checkpoint>inner</checkpoint> end</checkpoint>"
        result = extract_checkpoint(text)
        # Depth tracking returns full outermost block
        assert result == "outer <checkpoint>inner</checkpoint> end"

    def test_multiple_blocks_returns_first(self) -> None:
        """Returns content of first checkpoint block when multiple exist."""
        text = "<checkpoint>first</checkpoint> text <checkpoint>second</checkpoint>"
        result = extract_checkpoint(text)
        assert result == "first"

    def test_preserves_whitespace_inside_checkpoint(self) -> None:
        """Preserves leading/trailing whitespace inside checkpoint content."""
        text = "<checkpoint>  spaced content  </checkpoint>"
        assert extract_checkpoint(text) == "  spaced content  "

    def test_handles_malformed_opening_tag_only(self) -> None:
        """Returns content after opening tag if no closing tag present."""
        text = "<checkpoint>unclosed content"
        assert extract_checkpoint(text) == "unclosed content"

    def test_handles_malformed_closing_tag_only(self) -> None:
        """Returns full text if only closing tag present."""
        text = "content without opening</checkpoint>"
        assert extract_checkpoint(text) == text


class TestBuildPromptValidationCommands:
    """Tests for build_prompt_validation_commands function."""

    def test_with_python_uv_preset(self, tmp_path: Path) -> None:
        """Python-uv preset returns correct commands."""
        # Create mala.yaml with python-uv preset
        mala_yaml = tmp_path / "mala.yaml"
        mala_yaml.write_text("preset: python-uv\n")

        result = build_prompt_validation_commands(tmp_path)

        assert "ruff check" in result.lint
        assert "ruff format" in result.format
        assert "ty check" in result.typecheck
        assert "pytest" in result.test

    def test_with_go_preset(self, tmp_path: Path) -> None:
        """Go preset returns correct commands."""
        # Create mala.yaml with go preset
        mala_yaml = tmp_path / "mala.yaml"
        mala_yaml.write_text("preset: go\n")

        result = build_prompt_validation_commands(tmp_path)

        assert result.lint == "golangci-lint run"
        assert "gofmt" in result.format
        assert result.test == "go test ./..."

    def test_with_custom_commands(self, tmp_path: Path) -> None:
        """Custom commands override preset defaults."""
        mala_yaml = tmp_path / "mala.yaml"
        mala_yaml.write_text(
            """
preset: python-uv
commands:
  lint: "custom-lint"
  test: "custom-test"
"""
        )

        result = build_prompt_validation_commands(tmp_path)

        assert result.lint == "custom-lint"
        assert result.test == "custom-test"
        # format and typecheck should still come from preset
        assert "ruff format" in result.format
        assert "ty check" in result.typecheck

    def test_missing_config_returns_defaults(self, tmp_path: Path) -> None:
        """Missing mala.yaml returns default Python/uv commands with isolation flags."""
        # No mala.yaml file exists
        result = build_prompt_validation_commands(tmp_path)

        # Default commands include isolation flags for parallel agent runs
        # Note: RUFF_CACHE_DIR env var for ruff, -o cache_dir for pytest
        assert (
            result.lint
            == "RUFF_CACHE_DIR=/tmp/ruff-${AGENT_ID:-default} uvx ruff check ."
        )
        assert (
            result.format
            == "RUFF_CACHE_DIR=/tmp/ruff-${AGENT_ID:-default} uvx ruff format ."
        )
        assert result.typecheck == "uvx ty check"
        assert (
            result.test == "uv run pytest -o cache_dir=/tmp/pytest-${AGENT_ID:-default}"
        )

    def test_partial_commands_use_fallbacks(self, tmp_path: Path) -> None:
        """Partial commands config uses fallbacks for missing ones."""
        mala_yaml = tmp_path / "mala.yaml"
        mala_yaml.write_text(
            """
commands:
  test: "pytest"
  lint: "ruff check ."
"""
        )

        result = build_prompt_validation_commands(tmp_path)

        assert result.test == "pytest"
        assert result.lint == "ruff check ."
        # format and typecheck should use fallbacks
        assert "No format command configured" in result.format
        assert "No typecheck command configured" in result.typecheck


class TestPromptTemplateIntegration:
    """Integration tests for prompt template rendering."""

    def test_implementer_prompt_renders_with_python_commands(
        self, tmp_path: Path
    ) -> None:
        """Implementer prompt renders correctly with Python validation commands."""
        prompts = load_prompts(PROMPTS_DIR)

        mala_yaml = tmp_path / "mala.yaml"
        mala_yaml.write_text("preset: python-uv\n")

        cmds = build_prompt_validation_commands(tmp_path)
        prompt = prompts.implementer_prompt.format(
            issue_id="test-123",
            repo_path=tmp_path,
            lock_dir="/tmp/locks",
            validation_log_dir="/tmp/validation-logs",
            agent_id="test-agent",
            lint_command=cmds.lint,
            format_command=cmds.format,
            typecheck_command=cmds.typecheck,
            custom_commands_section="",
            test_command=cmds.test,
            issue_description="Test issue description",
        )

        # Verify Python commands appear in rendered prompt
        assert "ruff check" in prompt
        assert "ruff format" in prompt
        assert "ty check" in prompt
        assert "pytest" in prompt
        # Verify no unsubstituted placeholders remain
        assert "{lint_command}" not in prompt
        assert "{format_command}" not in prompt
        assert "{typecheck_command}" not in prompt
        assert "{test_command}" not in prompt
        assert "{validation_log_dir}" not in prompt
        assert "Validation Log Directory" in prompt

    def test_implementer_prompt_renders_with_go_commands(self, tmp_path: Path) -> None:
        """Implementer prompt renders correctly with Go validation commands."""
        prompts = load_prompts(PROMPTS_DIR)

        mala_yaml = tmp_path / "mala.yaml"
        mala_yaml.write_text("preset: go\n")

        cmds = build_prompt_validation_commands(tmp_path)
        prompt = prompts.implementer_prompt.format(
            issue_id="test-123",
            repo_path=tmp_path,
            lock_dir="/tmp/locks",
            validation_log_dir="/tmp/validation-logs",
            agent_id="test-agent",
            lint_command=cmds.lint,
            format_command=cmds.format,
            typecheck_command=cmds.typecheck,
            custom_commands_section="",
            test_command=cmds.test,
            issue_description="Test issue description",
        )

        # Verify Go commands appear in rendered prompt
        assert "golangci-lint" in prompt
        assert "gofmt" in prompt
        assert "go test" in prompt
        # Verify no Python commands
        assert "ruff" not in prompt
        assert "uvx" not in prompt

    def test_gate_followup_renders_with_commands(self, tmp_path: Path) -> None:
        """Gate followup prompt renders canonical wrappers for missing commands."""
        from src.domain.prompts import (
            commands_for_gate_followup,
            format_gate_followup_prompt,
        )

        prompts = load_prompts(PROMPTS_DIR)

        mala_yaml = tmp_path / "mala.yaml"
        mala_yaml.write_text("preset: go\n")

        cmds = build_prompt_validation_commands(tmp_path)
        validation_log_dir = tmp_path / "validation-logs"

        prompt = format_gate_followup_prompt(
            prompts.gate_followup_prompt,
            attempt=1,
            max_attempts=3,
            failure_reasons="- lint failed\n- tests failed",
            issue_id="test-123",
            missing_commands=commands_for_gate_followup(cmds),
            validation_log_dir=validation_log_dir,
        )

        # Verify Go commands appear inside canonical wrappers
        assert "golangci-lint" in prompt
        assert "go test" in prompt
        # Canonical wrapper signals
        assert "printf 'MALA_EVIDENCE name=%s" in prompt
        assert " 'lint' " in prompt
        assert " 'test' " in prompt
        # Verify no unsubstituted placeholders
        assert "{missing_command_wrappers}" not in prompt
        assert "{lint_command}" not in prompt

    def test_implementer_prompt_directs_custom_commands_to_named_logs(
        self,
    ) -> None:
        """Implementer prompt discourages generic custom.log evidence."""
        prompts = load_prompts(PROMPTS_DIR)

        assert "{issue_id}.<evidence_key>.log" in prompts.implementer_prompt
        assert "Do not combine multiple custom validation commands" in (
            prompts.implementer_prompt
        )
        # Custom command keys are referenced in narrative as evidence-key examples.
        assert "python_test" in prompts.implementer_prompt
        assert "python_lint" in prompts.implementer_prompt


class TestFormatImplementerPrompt:
    """Tests for format_implementer_prompt function."""

    def test_escapes_braces_in_issue_description(self, tmp_path: Path) -> None:
        """Issue descriptions with curly braces are escaped to prevent format errors."""
        prompts = load_prompts(PROMPTS_DIR)
        cmds = build_prompt_validation_commands(tmp_path, config_missing=True)

        description_with_braces = 'Config example: {"key": "value"}'
        prompt = format_implementer_prompt(
            prompts.implementer_prompt,
            issue_id="test-123",
            repo_path=tmp_path,
            agent_id="test-agent",
            validation_commands=cmds,
            lock_dir=tmp_path / "locks",
            validation_log_dir=tmp_path / "validation-logs",
            issue_description=description_with_braces,
        )

        # Braces should be escaped (doubled) in the output
        assert '{{"key": "value"}}' in prompt
        # Original unescaped braces should not cause KeyError

    def test_handles_none_issue_description(self, tmp_path: Path) -> None:
        """None issue_description falls back to default message."""
        prompts = load_prompts(PROMPTS_DIR)
        cmds = build_prompt_validation_commands(tmp_path, config_missing=True)

        prompt = format_implementer_prompt(
            prompts.implementer_prompt,
            issue_id="test-123",
            repo_path=tmp_path,
            agent_id="test-agent",
            validation_commands=cmds,
            lock_dir=tmp_path / "locks",
            validation_log_dir=tmp_path / "validation-logs",
            issue_description=None,
        )

        assert "No description available" in prompt

    def test_uses_validation_log_dir_for_quality_logs(self, tmp_path: Path) -> None:
        """Implementer prompt keeps lock coordination separate from log output."""
        prompts = load_prompts(PROMPTS_DIR)
        cmds = build_prompt_validation_commands(tmp_path, config_missing=True)
        lock_dir = tmp_path / "locks"
        validation_log_dir = tmp_path / "validation-logs"

        prompt = format_implementer_prompt(
            prompts.implementer_prompt,
            issue_id="test-123",
            repo_path=tmp_path,
            agent_id="test-agent",
            validation_commands=cmds,
            lock_dir=lock_dir,
            validation_log_dir=validation_log_dir,
            issue_description="Issue details",
        )

        assert f"**Lock Directory:** {lock_dir}" in prompt
        assert f"**Validation Log Directory:** {validation_log_dir}" in prompt
        assert f"mkdir -p {validation_log_dir}" in prompt
        assert prompt.count(f"mkdir -p {validation_log_dir}") == 1
        assert f"{validation_log_dir}/test-123.test.log" in prompt
        assert f"{lock_dir}/test-123.test.log" not in prompt


class TestCanonicalWrapperPrompts:
    """Prompt-tests pinning the canonical-wrapper substitution contract."""

    @staticmethod
    def _make_validation_commands(
        *,
        custom: tuple[tuple[str, str, int, bool], ...] = (),
    ) -> PromptValidationCommands:
        return PromptValidationCommands(
            lint="uvx ruff check .",
            format="uvx ruff format .",
            typecheck="uvx ty check",
            test="uv run pytest",
            custom_commands=custom,
        )

    def test_implementer_prompt_renders_one_canonical_wrapper_per_command(
        self, tmp_path: Path
    ) -> None:
        """AC1 + AC3: built-ins + advisory custom render five canonical wrappers."""
        prompts = load_prompts(PROMPTS_DIR)
        validation_log_dir = tmp_path / "validation-logs"
        cmds = self._make_validation_commands(
            custom=(("security-scan", "trivy fs .", 60, True),),
        )

        prompt = format_implementer_prompt(
            prompts.implementer_prompt,
            issue_id="bd-mala-abc",
            repo_path=tmp_path,
            agent_id="test-agent",
            validation_commands=cmds,
            lock_dir=tmp_path / "locks",
            validation_log_dir=validation_log_dir,
            issue_description="Test issue",
        )

        # The canonical wrapper format string `MALA_EVIDENCE name=%s ...`
        # appears once per generated wrapper.
        assert prompt.count("MALA_EVIDENCE name=%s exit=%s log=%s") == 5
        assert prompt.count("printf 'MALA_EVIDENCE name=%s") == 5

        # Each evidence key is supplied as the printf argument `'<name>'`
        for evidence_key in ("format", "lint", "typecheck", "security-scan", "test"):
            assert f" '{evidence_key}' " in prompt, evidence_key
            log_path = validation_log_dir / f"bd-mala-abc.{evidence_key}.log"
            assert f'__mala_log="{log_path}"' in prompt

        # Strict wrappers propagate exit; advisory wrappers exit 0
        strict_exits = prompt.count('exit "$__mala_status"')
        advisory_exits = prompt.count("\n  exit 0\n")
        assert strict_exits == 4
        assert advisory_exits == 1

        # AC3: no marker syntax remains anywhere in the prompt
        assert "[custom:" not in prompt

    def test_implementer_prompt_embeds_exact_configured_commands(
        self, tmp_path: Path
    ) -> None:
        """Generated wrappers include the exact configured command via escape_for_bash_lc."""
        from src.domain.validation_wrapper import escape_for_bash_lc

        prompts = load_prompts(PROMPTS_DIR)
        cmds = self._make_validation_commands(
            custom=(("python_test", "uv run pytest -k 'happy path'", 120, False),),
        )

        prompt = format_implementer_prompt(
            prompts.implementer_prompt,
            issue_id="bd-mala-abc",
            repo_path=tmp_path,
            agent_id="test-agent",
            validation_commands=cmds,
            lock_dir=tmp_path / "locks",
            validation_log_dir=tmp_path / "validation-logs",
            issue_description="Test issue",
        )

        for command in (
            cmds.lint,
            cmds.format,
            cmds.typecheck,
            cmds.test,
            "uv run pytest -k 'happy path'",
        ):
            assert escape_for_bash_lc(command) in prompt, command

    def test_implementer_prompt_renders_with_no_custom_commands(
        self, tmp_path: Path
    ) -> None:
        """Without custom commands, the section is empty but four built-ins still render."""
        prompts = load_prompts(PROMPTS_DIR)
        cmds = self._make_validation_commands()

        prompt = format_implementer_prompt(
            prompts.implementer_prompt,
            issue_id="bd-mala-abc",
            repo_path=tmp_path,
            agent_id="test-agent",
            validation_commands=cmds,
            lock_dir=tmp_path / "locks",
            validation_log_dir=tmp_path / "validation-logs",
            issue_description="Test",
        )

        strict_exits = prompt.count('exit "$__mala_status"')
        assert strict_exits == 4
        assert "[custom:" not in prompt
        assert "{custom_commands_section}" not in prompt

    def test_gate_followup_renders_one_wrapper_per_missing_command(
        self, tmp_path: Path
    ) -> None:
        """AC2: gate_followup with two missing commands yields exactly two wrappers."""
        from src.domain.prompts import format_gate_followup_prompt
        from src.domain.validation.spec import CommandKind, ValidationCommand

        prompts = load_prompts(PROMPTS_DIR)
        validation_log_dir = tmp_path / "validation-logs"
        lint_cmd = ValidationCommand(
            name="lint",
            command="uvx ruff check .",
            kind=CommandKind.LINT,
        )
        security_scan_cmd = ValidationCommand(
            name="security-scan",
            command="trivy fs .",
            kind=CommandKind.CUSTOM,
            allow_fail=True,
        )

        prompt = format_gate_followup_prompt(
            prompts.gate_followup_prompt,
            attempt=2,
            max_attempts=3,
            failure_reasons="- missing lint evidence\n- missing security-scan evidence",
            issue_id="bd-mala-abc",
            missing_commands=[lint_cmd, security_scan_cmd],
            validation_log_dir=validation_log_dir,
        )

        # Exactly two canonical wrappers
        assert prompt.count("printf 'MALA_EVIDENCE name=%s") == 2
        assert "uvx ruff check ." in prompt
        assert "trivy fs ." in prompt
        # Strict wrapper for lint, advisory for security-scan
        assert prompt.count('exit "$__mala_status"') == 1
        assert prompt.count("\n  exit 0\n") == 1
        # Rerun-rule sentence and unsubstituted placeholder absence
        assert "Do not manually echo `MALA_EVIDENCE`" in prompt
        assert "{missing_command_wrappers}" not in prompt
        assert "{format_command}" not in prompt
        assert "{test_command}" not in prompt
        assert "[custom:" not in prompt

    def test_gate_followup_with_no_missing_commands_omits_wrappers(
        self, tmp_path: Path
    ) -> None:
        """Empty missing_commands renders no wrappers, only the rule sentence."""
        from src.domain.prompts import format_gate_followup_prompt

        prompts = load_prompts(PROMPTS_DIR)
        prompt = format_gate_followup_prompt(
            prompts.gate_followup_prompt,
            attempt=1,
            max_attempts=3,
            failure_reasons="- some failure",
            issue_id="bd-mala-abc",
            missing_commands=[],
            validation_log_dir=tmp_path / "validation-logs",
        )

        assert "printf 'MALA_EVIDENCE name=%s" not in prompt
        assert "Do not manually echo `MALA_EVIDENCE`" in prompt

    def test_canonical_wrapper_round_trips_command_with_single_quotes(
        self, tmp_path: Path
    ) -> None:
        """Negative case: configured command with `'` is escaped via escape_for_bash_lc."""
        from src.domain.validation_wrapper import escape_for_bash_lc

        prompts = load_prompts(PROMPTS_DIR)
        cmds = self._make_validation_commands(
            custom=(
                (
                    "python_e2e",
                    'bash -c "echo it\'s fine"',
                    120,
                    False,
                ),
            ),
        )

        prompt = format_implementer_prompt(
            prompts.implementer_prompt,
            issue_id="bd-mala-abc",
            repo_path=tmp_path,
            agent_id="test-agent",
            validation_commands=cmds,
            lock_dir=tmp_path / "locks",
            validation_log_dir=tmp_path / "validation-logs",
            issue_description="Test",
        )

        embedded = escape_for_bash_lc('bash -c "echo it\'s fine"')
        assert embedded in prompt
        # And the unescaped form is NOT present
        assert "echo it's fine" not in prompt or "it'\\''s" in prompt

    def test_no_marker_syntax_in_any_rendered_prompt(self, tmp_path: Path) -> None:
        """AC3: `[custom:` does not appear in any rendered prompt for any spec."""
        from src.domain.prompts import format_gate_followup_prompt
        from src.domain.validation.spec import CommandKind, ValidationCommand

        prompts = load_prompts(PROMPTS_DIR)
        validation_log_dir = tmp_path / "validation-logs"

        for custom in (
            (),
            (("scan", "trivy fs .", 60, True),),
            (("strict_one", "do-strict", 30, False), ("adv_one", "do-adv", 30, True)),
        ):
            cmds = self._make_validation_commands(custom=custom)
            impl_prompt = format_implementer_prompt(
                prompts.implementer_prompt,
                issue_id="bd-mala-abc",
                repo_path=tmp_path,
                agent_id="test-agent",
                validation_commands=cmds,
                lock_dir=tmp_path / "locks",
                validation_log_dir=validation_log_dir,
                issue_description="Test",
            )
            assert "[custom:" not in impl_prompt

            missing = [
                ValidationCommand(
                    name="lint",
                    command="uvx ruff check .",
                    kind=CommandKind.LINT,
                ),
            ]
            for name, command, timeout, allow_fail in custom:
                missing.append(
                    ValidationCommand(
                        name=name,
                        command=command,
                        kind=CommandKind.CUSTOM,
                        timeout=timeout,
                        allow_fail=allow_fail,
                    )
                )
            gate_prompt = format_gate_followup_prompt(
                prompts.gate_followup_prompt,
                attempt=1,
                max_attempts=3,
                failure_reasons="-",
                issue_id="bd-mala-abc",
                missing_commands=missing,
                validation_log_dir=validation_log_dir,
            )
            assert "[custom:" not in gate_prompt
