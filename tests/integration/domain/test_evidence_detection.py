"""Integration tests for evidence detection of custom validation commands.

These tests validate the end-to-end evidence detection path:
JSONL log → SessionLogParser → parse_validation_evidence_with_spec() → ValidationEvidence

The tests exercise custom command marker parsing as specified in R5 (Evidence Detection).
Integration marker is applied automatically via path-based pytest configuration.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from src.domain.quality_gate import QualityGate
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
        scope=ValidationScope.PER_ISSUE,
        commands=commands,
    )


class TestEvidenceDetectionCustomCommandsIntegration:
    """Integration test for custom command evidence detection (R5).

    This test exercises the full log parse → evidence → gate check path.
    Tests are marked xfail until T007 implements marker parsing.
    """

    @pytest.mark.xfail(reason="T007: marker parsing not yet implemented")
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
        gate = QualityGate(tmp_path, log_provider, command_runner)

        evidence = gate.parse_validation_evidence_with_spec(log_path, spec)

        # This assertion will FAIL until T007 implements marker parsing
        assert "import_lint" in evidence.custom_commands_ran
        assert evidence.custom_commands_ran["import_lint"] is True
        assert evidence.custom_commands_failed.get("import_lint", False) is False

    @pytest.mark.xfail(reason="T007: marker parsing not yet implemented")
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
        gate = QualityGate(tmp_path, log_provider, command_runner)

        evidence = gate.parse_validation_evidence_with_spec(log_path, spec)

        # This assertion will FAIL until T007 implements marker parsing
        assert "import_lint" in evidence.custom_commands_ran
        assert evidence.custom_commands_ran["import_lint"] is True
        assert evidence.custom_commands_failed["import_lint"] is True

    @pytest.mark.xfail(reason="T007: marker parsing not yet implemented")
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
        gate = QualityGate(tmp_path, log_provider, command_runner)

        evidence = gate.parse_validation_evidence_with_spec(log_path, spec)

        # This assertion will FAIL until T007 implements marker parsing
        assert "import_lint" in evidence.custom_commands_ran
        assert "security_check" in evidence.custom_commands_ran
        assert evidence.custom_commands_ran["import_lint"] is True
        assert evidence.custom_commands_ran["security_check"] is True
        assert evidence.custom_commands_failed.get("import_lint", False) is False
        assert evidence.custom_commands_failed["security_check"] is True
