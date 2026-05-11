"""Unit tests for ``src.orchestration.init_config`` helpers.

These cover the pure config-generation helpers used by ``mala init``.
They contain no prompts and no file I/O.
"""

from __future__ import annotations

import pytest

from src.orchestration.init_config import (
    build_evidence_check_dict,
    build_per_issue_review_dict,
    build_validation_triggers_dict,
    compute_evidence_defaults,
    compute_trigger_defaults,
    resolve_preset_selection,
    validate_init_invocation,
)

pytestmark = pytest.mark.unit


class TestComputeEvidenceDefaults:
    def test_preset_path_intersects_test_lint(self) -> None:
        result = compute_evidence_defaults(["test", "lint", "format"], is_preset=True)
        assert result == ["test", "lint"]

    def test_preset_path_subset(self) -> None:
        result = compute_evidence_defaults(["test", "format"], is_preset=True)
        assert result == ["test"]

    def test_custom_path_returns_empty(self) -> None:
        result = compute_evidence_defaults(["build", "test", "lint"], is_preset=False)
        assert result == []

    def test_preserves_input_order(self) -> None:
        # Order in commands list should be preserved in result.
        result = compute_evidence_defaults(["lint", "format", "test"], is_preset=True)
        assert result == ["lint", "test"]


class TestComputeTriggerDefaults:
    def test_preset_path_excludes_setup_and_e2e(self) -> None:
        result = compute_trigger_defaults(
            ["setup", "test", "lint", "e2e"], is_preset=True
        )
        assert result == ["test", "lint"]

    def test_custom_path_returns_empty(self) -> None:
        result = compute_trigger_defaults(
            ["setup", "build", "test", "lint"], is_preset=False
        )
        assert result == []

    def test_preset_with_only_excluded_returns_empty(self) -> None:
        result = compute_trigger_defaults(["setup", "e2e"], is_preset=True)
        assert result == []


class TestBuildEvidenceCheckDict:
    def test_with_commands(self) -> None:
        assert build_evidence_check_dict(["test", "lint"]) == {
            "required": ["test", "lint"]
        }

    def test_empty(self) -> None:
        assert build_evidence_check_dict([]) == {"required": []}


class TestBuildPerIssueReviewDict:
    def test_passthrough(self) -> None:
        config = {
            "enabled": True,
            "reviewer_type": "cerberus",
            "max_retries": 3,
            "finding_threshold": "none",
        }
        assert build_per_issue_review_dict(config) == config


class TestBuildValidationTriggersDict:
    def test_ref_syntax_and_failure_mode(self) -> None:
        assert build_validation_triggers_dict(["test", "lint"]) == {
            "run_end": {
                "failure_mode": "continue",
                "commands": [{"ref": "test"}, {"ref": "lint"}],
            }
        }

    def test_empty_commands(self) -> None:
        assert build_validation_triggers_dict([]) == {
            "run_end": {"failure_mode": "continue", "commands": []}
        }


class TestValidateInitInvocation:
    def test_valid_yes_with_preset(self) -> None:
        check = validate_init_invocation(
            yes=True, preset="go", skip_evidence=False, skip_triggers=False, is_tty=True
        )
        assert check.error_message is None
        assert check.is_valid

    def test_yes_without_preset_errors(self) -> None:
        check = validate_init_invocation(
            yes=True,
            preset=None,
            skip_evidence=False,
            skip_triggers=False,
            is_tty=True,
        )
        assert check.error_message == "--yes requires --preset"
        assert not check.is_valid

    def test_non_tty_without_yes_or_skip_flags_errors(self) -> None:
        check = validate_init_invocation(
            yes=False,
            preset="go",
            skip_evidence=False,
            skip_triggers=False,
            is_tty=False,
        )
        assert check.error_message is not None
        assert "non-interactive" in check.error_message.lower()

    def test_non_tty_preset_with_skip_evidence_and_triggers_valid(self) -> None:
        check = validate_init_invocation(
            yes=False,
            preset="go",
            skip_evidence=True,
            skip_triggers=True,
            is_tty=False,
        )
        assert check.is_valid

    def test_interactive_default_valid(self) -> None:
        # TTY with no other flags is allowed; CLI will prompt.
        check = validate_init_invocation(
            yes=False,
            preset=None,
            skip_evidence=False,
            skip_triggers=False,
            is_tty=True,
        )
        assert check.is_valid


class TestResolvePresetSelection:
    def test_unknown_preset_returns_error(self) -> None:
        called: list[str] = []

        def lookup(name: str) -> list[str]:
            called.append(name)
            return []

        selection = resolve_preset_selection("nonexistent", ["go", "rust"], lookup)
        assert selection.error_message == "Unknown preset 'nonexistent'"
        assert not selection.is_preset
        assert selection.commands == []
        assert selection.config_data == {}
        assert called == []  # commands_lookup not invoked on invalid preset

    def test_known_preset_returns_commands_and_config(self) -> None:
        selection = resolve_preset_selection(
            "go", ["go", "rust"], lambda _: ["test", "lint", "format"]
        )
        assert selection.is_preset is True
        assert selection.commands == ["test", "lint", "format"]
        assert selection.config_data == {"preset": "go"}
        assert selection.error_message is None

    def test_returned_commands_list_is_independent(self) -> None:
        source = ["test", "lint"]
        selection = resolve_preset_selection("go", ["go"], lambda _: source)
        selection.commands.append("extra")
        assert source == ["test", "lint"]  # caller's list unaffected
