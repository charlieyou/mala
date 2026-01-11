"""Integration tests for evidence detection of custom validation commands.

These tests validate the end-to-end evidence detection path:
JSONL log → SessionLogParser → parse_validation_evidence_with_spec() → ValidationEvidence

The tests exercise custom command marker parsing as specified in R5 (Evidence Detection).
Integration marker is applied automatically via path-based pytest configuration.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from src.domain.evidence_check import EvidenceCheck
from src.domain.validation.spec import (
    CommandKind,
    ValidationCommand,
    ValidationScope,
    ValidationSpec,
)
from src.infra.tools.command_runner import CommandRunner
from src.infra.io.session_log_parser import FileSystemLogProvider

if TYPE_CHECKING:
    from pathlib import Path


def _create_jsonl_log(log_path: Path, entries: list[dict]) -> None:
    """Create a JSONL log file from a list of entries."""
    log_path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")


def _make_bash_tool_use_entry(tool_id: str, command: str) -> dict:
    """Create an assistant message with a Bash tool_use block."""
    return {
        "type": "assistant",
        "message": {
            "content": [
                {
                    "type": "tool_use",
                    "id": tool_id,
                    "name": "Bash",
                    "input": {"command": command},
                }
            ]
        },
    }


def _make_tool_result_entry(
    tool_use_id: str, content: str, is_error: bool = False
) -> dict:
    """Create a user message with a tool_result block containing output content."""
    return {
        "type": "user",
        "message": {
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": content,
                    "is_error": is_error,
                }
            ]
        },
    }


def _make_validation_spec_with_custom_commands(
    custom_commands: list[tuple[str, str, bool]],
) -> ValidationSpec:
    """Create a ValidationSpec with custom commands.

    Args:
        custom_commands: List of (name, command, allow_fail) tuples.

    Returns:
        ValidationSpec with custom commands configured.
    """
    commands = [
        ValidationCommand(
            name=name,
            command=cmd,
            kind=CommandKind.CUSTOM,
            allow_fail=allow_fail,
        )
        for name, cmd, allow_fail in custom_commands
    ]
    return ValidationSpec(
        scope=ValidationScope.PER_SESSION,
        commands=commands,
    )


class TestEvidenceDetectionCustomCommandsIntegration:
    """Integration test for custom command evidence detection (R5).

    This test exercises the full log parse → evidence → gate check path.
    """

    def test_detects_custom_command_pass_marker(self, tmp_path: Path) -> None:
        """Custom command pass marker populates custom_commands_ran.

        This test creates a log with:
        1. Bash tool_use running a custom command
        2. tool_result with success marker [custom:import_lint:pass]

        Expected behavior (after T007):
        - custom_commands_ran["import_lint"] == True
        - custom_commands_failed["import_lint"] == False

        Current behavior (stub):
        - custom_commands_ran is empty (test FAILS)
        """
        log_path = tmp_path / "session.jsonl"
        entries = [
            _make_bash_tool_use_entry("toolu_1", "python scripts/import_lint.py"),
            _make_tool_result_entry(
                "toolu_1",
                "Running import lint...\n[custom:import_lint:pass]\nDone.",
                is_error=False,
            ),
        ]
        _create_jsonl_log(log_path, entries)

        spec = _make_validation_spec_with_custom_commands(
            [
                ("import_lint", "python scripts/import_lint.py", False),
            ]
        )

        log_provider = FileSystemLogProvider()
        command_runner = CommandRunner(cwd=tmp_path)
        gate = EvidenceCheck(tmp_path, log_provider, command_runner)

        evidence = gate.parse_validation_evidence_with_spec(log_path, spec)

        # This assertion will FAIL until T007 implements marker parsing
        assert "import_lint" in evidence.custom_commands_ran
        assert evidence.custom_commands_ran["import_lint"] is True
        assert evidence.custom_commands_failed.get("import_lint", False) is False

    def test_detects_custom_command_fail_marker(self, tmp_path: Path) -> None:
        """Custom command fail marker populates custom_commands_failed.

        This test creates a log with:
        1. Bash tool_use running a custom command
        2. tool_result with failure marker [custom:import_lint:fail exit=1]

        Expected behavior (after T007):
        - custom_commands_ran["import_lint"] == True
        - custom_commands_failed["import_lint"] == True
        """
        log_path = tmp_path / "session.jsonl"
        entries = [
            _make_bash_tool_use_entry("toolu_1", "python scripts/import_lint.py"),
            _make_tool_result_entry(
                "toolu_1",
                "Running import lint...\n[custom:import_lint:fail exit=1]\nError found.",
                is_error=True,
            ),
        ]
        _create_jsonl_log(log_path, entries)

        spec = _make_validation_spec_with_custom_commands(
            [
                ("import_lint", "python scripts/import_lint.py", False),
            ]
        )

        log_provider = FileSystemLogProvider()
        command_runner = CommandRunner(cwd=tmp_path)
        gate = EvidenceCheck(tmp_path, log_provider, command_runner)

        evidence = gate.parse_validation_evidence_with_spec(log_path, spec)

        # This assertion will FAIL until T007 implements marker parsing
        assert "import_lint" in evidence.custom_commands_ran
        assert evidence.custom_commands_ran["import_lint"] is True
        assert evidence.custom_commands_failed["import_lint"] is True

    def test_detects_multiple_custom_commands(self, tmp_path: Path) -> None:
        """Multiple custom commands are tracked independently.

        This test creates a log with two custom commands:
        - import_lint: passes
        - security_check: fails

        Expected behavior (after T007):
        - Both commands tracked in custom_commands_ran
        - Only security_check marked as failed
        """
        log_path = tmp_path / "session.jsonl"
        entries = [
            # First command: import_lint passes
            _make_bash_tool_use_entry("toolu_1", "python scripts/import_lint.py"),
            _make_tool_result_entry(
                "toolu_1",
                "[custom:import_lint:pass]",
                is_error=False,
            ),
            # Second command: security_check fails
            _make_bash_tool_use_entry("toolu_2", "python scripts/security_check.py"),
            _make_tool_result_entry(
                "toolu_2",
                "[custom:security_check:fail exit=1]",
                is_error=True,
            ),
        ]
        _create_jsonl_log(log_path, entries)

        spec = _make_validation_spec_with_custom_commands(
            [
                ("import_lint", "python scripts/import_lint.py", False),
                ("security_check", "python scripts/security_check.py", False),
            ]
        )

        log_provider = FileSystemLogProvider()
        command_runner = CommandRunner(cwd=tmp_path)
        gate = EvidenceCheck(tmp_path, log_provider, command_runner)

        evidence = gate.parse_validation_evidence_with_spec(log_path, spec)

        # This assertion will FAIL until T007 implements marker parsing
        assert "import_lint" in evidence.custom_commands_ran
        assert "security_check" in evidence.custom_commands_ran
        assert evidence.custom_commands_ran["import_lint"] is True
        assert evidence.custom_commands_ran["security_check"] is True
        assert evidence.custom_commands_failed.get("import_lint", False) is False
        assert evidence.custom_commands_failed["security_check"] is True


class TestEvidenceCheckConfigIntegration:
    """Integration test for evidence_check configuration (T001).

    This test exercises the full config parse → preset merge → evidence check flow.
    """

    def test_evidence_check_config_parsing_end_to_end(self, tmp_path: Path) -> None:
        """Evidence check config is parsed, merged, and surfaces in ValidationSpec.

        This test creates a mala.yaml with:
        1. evidence_check.required: [test]
        2. Minimal commands config

        Expected behavior (after T002-T004):
        - ValidationSpec.evidence_required contains "test"
        - Evidence filtering respects the required list

        Current behavior (stub):
        - evidence_check parsing returns None (test FAILS)

        Note: This test MUST fail with "assertion error" on evidence_required,
        NOT with import errors or syntax errors. The skeleton infrastructure
        must be wired correctly for downstream tasks.
        """
        from src.domain.validation.config_loader import load_config
        from src.domain.validation.spec import ValidationScope, build_validation_spec

        # Create minimal mala.yaml with evidence_check
        mala_yaml = tmp_path / "mala.yaml"
        mala_yaml.write_text(
            """\
commands:
  test: "echo test"

evidence_check:
  required:
    - test
"""
        )

        # Load the config from mala.yaml - exercises the full parsing path
        config = load_config(tmp_path)

        # Verify evidence_check field exists on ValidationConfig
        # This will be None because _parse_evidence_check_config() stub returns None
        assert hasattr(config, "evidence_check")

        # Build validation spec passing the loaded config explicitly
        # This ensures we test the wiring from config → spec
        spec = build_validation_spec(
            tmp_path,
            scope=ValidationScope.PER_SESSION,
            validation_config=config,
        )

        # This assertion FAILS because:
        # 1. _parse_evidence_check_config() returns None (stub)
        # 2. build_validation_spec() doesn't yet propagate evidence_required from config
        # T002 will implement parsing, T004 will wire build_validation_spec()
        assert spec.evidence_required == ("test",), (
            f"Expected evidence_required=('test',) but got {spec.evidence_required!r}. "
            "This failure is expected until T002 implements evidence_check parsing."
        )

    def test_full_flow_with_filtering(self, tmp_path: Path) -> None:
        """Full flow: config parsing → build_validation_spec → evidence check with filtering.

        This test verifies integration test 22 from the plan:
        - Config with evidence_check.required filters which commands are checked
        - Commands not in required list don't cause failures even if not run
        """
        from src.domain.validation.config_loader import load_config
        from src.domain.validation.spec import ValidationScope, build_validation_spec

        from src.domain.evidence_check import (
            check_evidence_against_spec,
            ValidationEvidence,
        )

        # Create mala.yaml with multiple commands but only one required
        mala_yaml = tmp_path / "mala.yaml"
        mala_yaml.write_text(
            """\
commands:
  lint: "echo lint"
  format: "echo format"
  test: "echo test"

evidence_check:
  required:
    - lint
"""
        )

        # Load config and build spec
        config = load_config(tmp_path)
        spec = build_validation_spec(
            tmp_path,
            scope=ValidationScope.PER_SESSION,
            validation_config=config,
        )

        # Verify only "lint" is required
        assert spec.evidence_required == ("lint",)

        # Create evidence where lint ran but test/format did not
        # Since only lint is required, this should pass
        from src.domain.validation.spec import CommandKind

        evidence = ValidationEvidence(
            commands_ran={CommandKind.LINT: True},
        )

        passed, missing, failed_strict = check_evidence_against_spec(evidence, spec)

        # Should pass because lint ran (format/test not required)
        assert passed is True
        assert missing == []
        assert failed_strict == []

    def test_preset_merge_with_project_evidence_check(self, tmp_path: Path) -> None:
        """Preset + project merge: project evidence_check overrides/extends preset.

        This test verifies integration test 23 from the plan:
        - Project mala.yaml can specify preset (e.g., python-uv)
        - Project evidence_check.required takes precedence
        - The resolved commands from preset are available in spec
        """
        from src.domain.validation.config_loader import load_config
        from src.domain.validation.spec import ValidationScope, build_validation_spec

        # Create mala.yaml extending a preset with custom evidence_check
        mala_yaml = tmp_path / "mala.yaml"
        mala_yaml.write_text(
            """\
preset: python-uv

evidence_check:
  required:
    - lint
    - format
"""
        )

        # Load config - this exercises preset merging
        config = load_config(tmp_path)

        # Verify evidence_check was parsed from project config
        assert config.evidence_check is not None
        assert config.evidence_check.required == ("lint", "format")

        # Build validation spec - exercises full wiring
        spec = build_validation_spec(
            tmp_path,
            scope=ValidationScope.PER_SESSION,
            validation_config=config,
        )

        # Verify evidence_required is set from project config
        assert spec.evidence_required == ("lint", "format")

        # Verify commands from preset are available
        # python-uv preset includes lint, format, typecheck, test
        command_names = [cmd.name for cmd in spec.commands]
        assert "lint" in command_names
        assert "format" in command_names
