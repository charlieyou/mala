"""Integration tests for evidence detection of custom validation commands.

These tests validate the end-to-end evidence detection path:
JSONL log → SessionLogParser → parse_validation_evidence_with_spec() → ValidationEvidence

The tests exercise custom command marker parsing as specified in R5 (Evidence Detection).
Integration marker is applied automatically via path-based pytest configuration.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from src.domain.evidence_check import EvidenceCheck, check_evidence_against_spec
from src.domain.validation.spec import (
    CommandKind,
    ValidationCommand,
    ValidationScope,
    ValidationSpec,
)
from src.domain.validation_wrapper import build_canonical_wrapper
from src.infra.tools.command_runner import CommandRunner
from src.infra.io.session_log_parser import FileSystemLogProvider

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


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

    def test_detects_custom_command_from_exact_configured_command(
        self, tmp_path: Path
    ) -> None:
        """Exact configured custom command invocation counts as named evidence."""
        log_path = tmp_path / "session.jsonl"
        entries = [
            _make_bash_tool_use_entry(
                "toolu_1",
                "make -C shadow-write-consumer test > /tmp/issue.python_test.log 2>&1",
            ),
            _make_tool_result_entry(
                "toolu_1",
                "python_test exit=0 log=/tmp/issue.python_test.log",
                is_error=False,
            ),
        ]
        _create_jsonl_log(log_path, entries)

        spec = _make_validation_spec_with_custom_commands(
            [("python_test", "make -C shadow-write-consumer test", False)]
        )

        gate = EvidenceCheck(
            tmp_path, FileSystemLogProvider(), CommandRunner(cwd=tmp_path)
        )

        evidence = gate.parse_validation_evidence_with_spec(log_path, spec)

        assert evidence.custom_commands_ran["python_test"] is True
        assert evidence.custom_commands_failed.get("python_test", False) is False

    def test_named_exit_line_overrides_single_command_shell_success(
        self, tmp_path: Path
    ) -> None:
        """Single custom fallback uses named exit evidence before shell status."""
        log_path = tmp_path / "session.jsonl"
        entries = [
            _make_bash_tool_use_entry(
                "toolu_1",
                "make -C shadow-write-consumer test > /tmp/issue.python_test.log 2>&1; "
                "echo 'python_test exit=1 log=/tmp/issue.python_test.log'",
            ),
            _make_tool_result_entry(
                "toolu_1",
                "python_test exit=1 log=/tmp/issue.python_test.log",
                is_error=False,
            ),
        ]
        _create_jsonl_log(log_path, entries)

        spec = _make_validation_spec_with_custom_commands(
            [("python_test", "make -C shadow-write-consumer test", False)]
        )

        gate = EvidenceCheck(
            tmp_path, FileSystemLogProvider(), CommandRunner(cwd=tmp_path)
        )

        evidence = gate.parse_validation_evidence_with_spec(log_path, spec)

        assert evidence.custom_commands_ran["python_test"] is True
        assert evidence.custom_commands_failed["python_test"] is True

    def test_named_exit_line_overrides_single_command_shell_failure(
        self, tmp_path: Path
    ) -> None:
        """A named zero exit line can credit the custom command despite later shell failure."""
        log_path = tmp_path / "session.jsonl"
        entries = [
            _make_bash_tool_use_entry(
                "toolu_1",
                "make -C shadow-write-consumer test > /tmp/issue.python_test.log 2>&1; "
                "echo 'python_test exit=0 log=/tmp/issue.python_test.log'; false",
            ),
            _make_tool_result_entry(
                "toolu_1",
                "python_test exit=0 log=/tmp/issue.python_test.log",
                is_error=True,
            ),
        ]
        _create_jsonl_log(log_path, entries)

        spec = _make_validation_spec_with_custom_commands(
            [("python_test", "make -C shadow-write-consumer test", False)]
        )

        gate = EvidenceCheck(
            tmp_path, FileSystemLogProvider(), CommandRunner(cwd=tmp_path)
        )

        evidence = gate.parse_validation_evidence_with_spec(log_path, spec)

        assert evidence.custom_commands_ran["python_test"] is True
        assert evidence.custom_commands_failed.get("python_test", False) is False

    def test_detects_multiple_custom_commands_with_per_command_exit_lines(
        self, tmp_path: Path
    ) -> None:
        """A combined shell command needs per-custom exit lines for attribution."""
        log_path = tmp_path / "session.jsonl"
        entries = [
            _make_bash_tool_use_entry(
                "toolu_1",
                "make -C shadow-write-consumer test > /tmp/issue.python_test.log 2>&1; "
                "echo 'python_test exit=0 log=/tmp/issue.python_test.log'; "
                "make -C shadow-write-consumer lint > /tmp/issue.python_lint.log 2>&1; "
                "echo 'python_lint exit=0 log=/tmp/issue.python_lint.log'",
            ),
            _make_tool_result_entry(
                "toolu_1",
                "python_test exit=0 log=/tmp/issue.python_test.log\n"
                "python_lint exit=0 log=/tmp/issue.python_lint.log",
                is_error=False,
            ),
        ]
        _create_jsonl_log(log_path, entries)

        spec = _make_validation_spec_with_custom_commands(
            [
                ("python_test", "make -C shadow-write-consumer test", False),
                ("python_lint", "make -C shadow-write-consumer lint", False),
            ]
        )

        gate = EvidenceCheck(
            tmp_path, FileSystemLogProvider(), CommandRunner(cwd=tmp_path)
        )

        evidence = gate.parse_validation_evidence_with_spec(log_path, spec)

        assert evidence.custom_commands_ran["python_test"] is True
        assert evidence.custom_commands_ran["python_lint"] is True
        assert evidence.custom_commands_failed.get("python_test", False) is False
        assert evidence.custom_commands_failed.get("python_lint", False) is False

    def test_does_not_credit_unlabeled_combined_custom_commands(
        self, tmp_path: Path
    ) -> None:
        """One aggregate shell success is not enough for multiple custom checks."""
        log_path = tmp_path / "session.jsonl"
        entries = [
            _make_bash_tool_use_entry(
                "toolu_1",
                "make -C shadow-write-consumer test; make -C shadow-write-consumer lint",
            ),
            _make_tool_result_entry("toolu_1", "custom exit=0", is_error=False),
        ]
        _create_jsonl_log(log_path, entries)

        spec = _make_validation_spec_with_custom_commands(
            [
                ("python_test", "make -C shadow-write-consumer test", False),
                ("python_lint", "make -C shadow-write-consumer lint", False),
            ]
        )

        gate = EvidenceCheck(
            tmp_path, FileSystemLogProvider(), CommandRunner(cwd=tmp_path)
        )

        evidence = gate.parse_validation_evidence_with_spec(log_path, spec)

        assert "python_test" not in evidence.custom_commands_ran
        assert "python_lint" not in evidence.custom_commands_ran

    def test_marker_failure_overrides_raw_custom_command_success(
        self, tmp_path: Path
    ) -> None:
        """Explicit custom markers remain authoritative over raw command status."""
        log_path = tmp_path / "session.jsonl"
        entries = [
            _make_bash_tool_use_entry("toolu_1", "make -C shadow-write-consumer test"),
            _make_tool_result_entry(
                "toolu_1",
                "[custom:python_test:start]\n[custom:python_test:fail exit=2]",
                is_error=False,
            ),
        ]
        _create_jsonl_log(log_path, entries)

        spec = _make_validation_spec_with_custom_commands(
            [("python_test", "make -C shadow-write-consumer test", False)]
        )

        gate = EvidenceCheck(
            tmp_path, FileSystemLogProvider(), CommandRunner(cwd=tmp_path)
        )

        evidence = gate.parse_validation_evidence_with_spec(log_path, spec)

        assert evidence.custom_commands_ran["python_test"] is True
        assert evidence.custom_commands_failed["python_test"] is True

    def test_generic_make_command_does_not_credit_custom_command(
        self, tmp_path: Path
    ) -> None:
        """Only exact configured custom commands are credited by fallback."""
        log_path = tmp_path / "session.jsonl"
        entries = [
            _make_bash_tool_use_entry("toolu_1", "make build"),
            _make_tool_result_entry("toolu_1", "build ok", is_error=False),
        ]
        _create_jsonl_log(log_path, entries)

        spec = _make_validation_spec_with_custom_commands(
            [("python_test", "make -C shadow-write-consumer test", False)]
        )

        gate = EvidenceCheck(
            tmp_path, FileSystemLogProvider(), CommandRunner(cwd=tmp_path)
        )

        evidence = gate.parse_validation_evidence_with_spec(log_path, spec)

        assert "python_test" not in evidence.custom_commands_ran

    def test_quoted_custom_command_does_not_credit_custom_command(
        self, tmp_path: Path
    ) -> None:
        """Mentioning a configured command in output text is not evidence."""
        log_path = tmp_path / "session.jsonl"
        entries = [
            _make_bash_tool_use_entry(
                "toolu_1", "echo 'make -C shadow-write-consumer test'"
            ),
            _make_tool_result_entry("toolu_1", "make -C shadow-write-consumer test"),
        ]
        _create_jsonl_log(log_path, entries)

        spec = _make_validation_spec_with_custom_commands(
            [("python_test", "make -C shadow-write-consumer test", False)]
        )

        gate = EvidenceCheck(
            tmp_path, FileSystemLogProvider(), CommandRunner(cwd=tmp_path)
        )

        evidence = gate.parse_validation_evidence_with_spec(log_path, spec)

        assert "python_test" not in evidence.custom_commands_ran

    def test_compound_custom_command_without_named_exit_line_does_not_credit_pass(
        self, tmp_path: Path
    ) -> None:
        """Compound shell success is not enough to credit a custom command."""
        log_path = tmp_path / "session.jsonl"
        entries = [
            _make_bash_tool_use_entry(
                "toolu_1", "make -C shadow-write-consumer test || true"
            ),
            _make_tool_result_entry("toolu_1", "done", is_error=False),
        ]
        _create_jsonl_log(log_path, entries)

        spec = _make_validation_spec_with_custom_commands(
            [("python_test", "make -C shadow-write-consumer test", False)]
        )

        gate = EvidenceCheck(
            tmp_path, FileSystemLogProvider(), CommandRunner(cwd=tmp_path)
        )

        evidence = gate.parse_validation_evidence_with_spec(log_path, spec)

        assert "python_test" not in evidence.custom_commands_ran

    def test_marker_failure_is_not_overridden_by_later_raw_custom_command(
        self, tmp_path: Path
    ) -> None:
        """Explicit marker state remains authoritative over later raw fallback."""
        log_path = tmp_path / "session.jsonl"
        entries = [
            _make_bash_tool_use_entry("toolu_1", "make -C shadow-write-consumer test"),
            _make_tool_result_entry(
                "toolu_1",
                "[custom:python_test:start]\n[custom:python_test:fail exit=2]",
                is_error=True,
            ),
            _make_bash_tool_use_entry("toolu_2", "make -C shadow-write-consumer test"),
            _make_tool_result_entry("toolu_2", "python_test exit=0", is_error=False),
        ]
        _create_jsonl_log(log_path, entries)

        spec = _make_validation_spec_with_custom_commands(
            [("python_test", "make -C shadow-write-consumer test", False)]
        )

        gate = EvidenceCheck(
            tmp_path, FileSystemLogProvider(), CommandRunner(cwd=tmp_path)
        )

        evidence = gate.parse_validation_evidence_with_spec(log_path, spec)

        assert evidence.custom_commands_ran["python_test"] is True
        assert evidence.custom_commands_failed["python_test"] is True

    def test_similar_custom_command_prefix_does_not_credit_custom_command(
        self, tmp_path: Path
    ) -> None:
        """Configured custom command matches must end on a shell token boundary."""
        log_path = tmp_path / "session.jsonl"
        entries = [
            _make_bash_tool_use_entry(
                "toolu_1", "make -C shadow-write-consumer test-extra"
            ),
            _make_tool_result_entry("toolu_1", "python_test exit=0", is_error=False),
        ]
        _create_jsonl_log(log_path, entries)

        spec = _make_validation_spec_with_custom_commands(
            [("python_test", "make -C shadow-write-consumer test", False)]
        )

        gate = EvidenceCheck(
            tmp_path, FileSystemLogProvider(), CommandRunner(cwd=tmp_path)
        )

        evidence = gate.parse_validation_evidence_with_spec(log_path, spec)

        assert "python_test" not in evidence.custom_commands_ran

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

        evidence_provider = FileSystemLogProvider()
        command_runner = CommandRunner(cwd=tmp_path)
        gate = EvidenceCheck(tmp_path, evidence_provider, command_runner)

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

        evidence_provider = FileSystemLogProvider()
        command_runner = CommandRunner(cwd=tmp_path)
        gate = EvidenceCheck(tmp_path, evidence_provider, command_runner)

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

        evidence_provider = FileSystemLogProvider()
        command_runner = CommandRunner(cwd=tmp_path)
        gate = EvidenceCheck(tmp_path, evidence_provider, command_runner)

        evidence = gate.parse_validation_evidence_with_spec(log_path, spec)

        # This assertion will FAIL until T007 implements marker parsing
        assert "import_lint" in evidence.custom_commands_ran
        assert "security_check" in evidence.custom_commands_ran
        assert evidence.custom_commands_ran["import_lint"] is True
        assert evidence.custom_commands_ran["security_check"] is True
        assert evidence.custom_commands_failed.get("import_lint", False) is False
        assert evidence.custom_commands_failed["security_check"] is True


class TestCanonicalEvidenceDetectionIntegration:
    """Integration tests for generated-wrapper evidence recognition."""

    def _make_gate(self, tmp_path: Path) -> EvidenceCheck:
        return EvidenceCheck(
            tmp_path, FileSystemLogProvider(), CommandRunner(cwd=tmp_path)
        )

    def test_builtin_canonical_wrapper_records_summary_evidence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MALA_VALIDATION_LOG_DIR", str(tmp_path))
        command = ValidationCommand(
            name="lint",
            command="uvx ruff check .",
            kind=CommandKind.LINT,
        )
        wrapper = build_canonical_wrapper(
            command,
            issue_id="mala-3gbpn.3",
            validation_log_dir=tmp_path,
        )
        log_path = tmp_path / "session.jsonl"
        entries = [
            _make_bash_tool_use_entry("toolu_lint", wrapper),
            _make_tool_result_entry(
                "toolu_lint",
                f"MALA_EVIDENCE name=lint exit=0 log={tmp_path / 'mala-3gbpn.3.lint.log'}",
            ),
        ]
        _create_jsonl_log(log_path, entries)
        spec = ValidationSpec(
            commands=[command],
            scope=ValidationScope.PER_SESSION,
            evidence_required=("lint",),
        )

        evidence = self._make_gate(tmp_path).parse_validation_evidence_with_spec(
            log_path, spec
        )

        record = evidence.commands["lint"]
        assert record.seen is True
        assert record.status == "passed"
        assert record.source == "command+summary"

    def test_custom_canonical_wrapper_records_failure_evidence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MALA_VALIDATION_LOG_DIR", str(tmp_path))
        command = ValidationCommand(
            name="python_test",
            command="make -C shadow-write-consumer test",
            kind=CommandKind.CUSTOM,
        )
        wrapper = build_canonical_wrapper(
            command,
            issue_id="mala-3gbpn.3",
            validation_log_dir=tmp_path,
        )
        log_path = tmp_path / "session.jsonl"
        entries = [
            _make_bash_tool_use_entry("toolu_python_test", wrapper),
            _make_tool_result_entry(
                "toolu_python_test",
                "MALA_EVIDENCE name=python_test exit=1 "
                f"log={tmp_path / 'mala-3gbpn.3.python_test.log'}",
                is_error=True,
            ),
        ]
        _create_jsonl_log(log_path, entries)
        spec = ValidationSpec(
            commands=[command],
            scope=ValidationScope.PER_SESSION,
            evidence_required=("python_test",),
        )

        evidence = self._make_gate(tmp_path).parse_validation_evidence_with_spec(
            log_path, spec
        )

        record = evidence.commands["python_test"]
        assert record.status == "failed"
        assert record.exit_code == 1
        assert record.source == "command+summary"

    def test_advisory_custom_failure_does_not_fail_gate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MALA_VALIDATION_LOG_DIR", str(tmp_path))
        command = ValidationCommand(
            name="python_test",
            command="make -C shadow-write-consumer test",
            kind=CommandKind.CUSTOM,
            allow_fail=True,
        )
        wrapper = build_canonical_wrapper(
            command,
            issue_id="mala-3gbpn.3",
            validation_log_dir=tmp_path,
        )
        log_path = tmp_path / "session.jsonl"
        entries = [
            _make_bash_tool_use_entry("toolu_python_test", wrapper),
            _make_tool_result_entry(
                "toolu_python_test",
                "MALA_EVIDENCE name=python_test exit=1 "
                f"log={tmp_path / 'mala-3gbpn.3.python_test.log'}",
                is_error=False,
            ),
        ]
        _create_jsonl_log(log_path, entries)
        spec = ValidationSpec(
            commands=[command],
            scope=ValidationScope.PER_SESSION,
            evidence_required=("python_test",),
        )
        evidence = self._make_gate(tmp_path).parse_validation_evidence_with_spec(
            log_path, spec
        )

        passed, missing, failed_strict = check_evidence_against_spec(evidence, spec)

        assert passed is True
        assert missing == []
        assert failed_strict == []

    def test_strict_custom_failure_fails_gate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MALA_VALIDATION_LOG_DIR", str(tmp_path))
        command = ValidationCommand(
            name="python_test",
            command="make -C shadow-write-consumer test",
            kind=CommandKind.CUSTOM,
            allow_fail=False,
        )
        wrapper = build_canonical_wrapper(
            command,
            issue_id="mala-3gbpn.3",
            validation_log_dir=tmp_path,
        )
        log_path = tmp_path / "session.jsonl"
        entries = [
            _make_bash_tool_use_entry("toolu_python_test", wrapper),
            _make_tool_result_entry(
                "toolu_python_test",
                "MALA_EVIDENCE name=python_test exit=1 "
                f"log={tmp_path / 'mala-3gbpn.3.python_test.log'}",
                is_error=True,
            ),
        ]
        _create_jsonl_log(log_path, entries)
        spec = ValidationSpec(
            commands=[command],
            scope=ValidationScope.PER_SESSION,
            evidence_required=("python_test",),
        )
        evidence = self._make_gate(tmp_path).parse_validation_evidence_with_spec(
            log_path, spec
        )

        passed, missing, failed_strict = check_evidence_against_spec(evidence, spec)

        assert passed is False
        assert missing == []
        assert failed_strict == ["python_test"]


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
        from src.domain.evidence_check import CommandEvidence
        from src.domain.validation.spec import CommandKind

        evidence = ValidationEvidence(
            commands={
                "lint": CommandEvidence(
                    name="lint",
                    kind=CommandKind.LINT,
                    seen=True,
                    status="passed",
                ),
            },
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
