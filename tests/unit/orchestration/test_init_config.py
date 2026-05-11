"""Unit tests for ``src.orchestration.init_config`` helpers.

These cover the pure config-generation helpers used by ``mala init``.
They contain no prompts and no file I/O.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.domain.validation.config_types import ConfigError
from src.orchestration.init_config import (
    RECOVERY_ABORT,
    RECOVERY_REVISE,
    RECOVERY_SKIP,
    build_evidence_check_dict,
    build_per_issue_review_dict,
    build_validation_triggers_dict,
    compute_evidence_defaults,
    compute_trigger_defaults,
    execute_init_workflow,
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


class FakePrompts:
    """In-memory ``InitPrompts`` stub for driving the workflow."""

    def __init__(
        self,
        *,
        preset: str | None = None,
        custom_commands: dict[str, str] | None = None,
        evidence_check: list[str] | None = None,
        per_issue_review: dict[str, Any] | None = None,
        run_end_trigger: list[str] | None = None,
        recovery_choice: str | None = None,
    ) -> None:
        self._preset = preset
        self._custom_commands = custom_commands
        self._evidence_check = evidence_check
        self._per_issue_review = per_issue_review
        self._run_end_trigger = run_end_trigger
        self._recovery_choice = recovery_choice
        self.notifications: list[str] = []
        self.preset_calls = 0
        self.custom_calls = 0
        self.evidence_calls = 0
        self.review_calls = 0
        self.trigger_calls = 0
        self.recovery_calls: list[str] = []

    def prompt_preset(self, presets: list[str]) -> str | None:
        self.preset_calls += 1
        return self._preset

    def prompt_custom_commands(self) -> dict[str, str] | None:
        self.custom_calls += 1
        return self._custom_commands

    def prompt_evidence_check(
        self, commands: list[str], is_preset: bool
    ) -> list[str] | None:
        self.evidence_calls += 1
        return self._evidence_check

    def prompt_per_issue_review(self) -> dict[str, Any] | None:
        self.review_calls += 1
        return self._per_issue_review

    def prompt_run_end_trigger(
        self, commands: list[str], is_preset: bool
    ) -> list[str] | None:
        self.trigger_calls += 1
        return self._run_end_trigger

    def prompt_recovery_choice(self, error_message: str) -> str | None:
        self.recovery_calls.append(error_message)
        return self._recovery_choice

    def notify(self, message: str) -> None:
        self.notifications.append(message)


def _ok_validator(data: dict[str, Any]) -> None:
    return None


class TestExecuteInitWorkflowPresetYes:
    """``--preset --yes`` path: defaults are applied without prompts."""

    def test_preset_yes_applies_evidence_and_trigger_defaults(self) -> None:
        prompts = FakePrompts()
        result = execute_init_workflow(
            preset="go",
            skip_evidence=False,
            skip_triggers=False,
            yes=True,
            is_tty=False,
            presets=["go", "rust"],
            commands_lookup=lambda _: ["test", "lint", "setup"],
            validate_config=_ok_validator,
            prompts=prompts,
        )
        assert result.exit_code == 0
        assert result.config_data is not None
        assert result.config_data["preset"] == "go"
        # compute_evidence_defaults preset path returns intersection with {test, lint}
        assert result.config_data["evidence_check"] == {"required": ["test", "lint"]}
        # compute_trigger_defaults preset path excludes setup/e2e
        assert result.config_data["validation_triggers"]["run_end"]["commands"] == [
            {"ref": "test"},
            {"ref": "lint"},
        ]
        # No prompts should be issued in --yes mode.
        assert prompts.preset_calls == 0
        assert prompts.evidence_calls == 0
        assert prompts.review_calls == 0
        assert prompts.trigger_calls == 0

    def test_preset_yes_with_skip_flags_omits_sections(self) -> None:
        prompts = FakePrompts()
        result = execute_init_workflow(
            preset="go",
            skip_evidence=True,
            skip_triggers=True,
            yes=True,
            is_tty=False,
            presets=["go"],
            commands_lookup=lambda _: ["test", "lint"],
            validate_config=_ok_validator,
            prompts=prompts,
        )
        assert result.config_data == {"preset": "go"}

    def test_unknown_preset_returns_error(self) -> None:
        result = execute_init_workflow(
            preset="nonexistent",
            skip_evidence=True,
            skip_triggers=True,
            yes=True,
            is_tty=False,
            presets=["go"],
            commands_lookup=lambda _: [],
            validate_config=_ok_validator,
            prompts=FakePrompts(),
        )
        assert result.exit_code == 1
        assert result.error_message == "Unknown preset 'nonexistent'"
        assert result.config_data is None


class TestExecuteInitWorkflowInteractive:
    """Interactive flows: preset/custom branching via the prompts port."""

    def test_interactive_preset_selection_builds_config(self) -> None:
        prompts = FakePrompts(
            preset="go",
            evidence_check=["test"],
            per_issue_review={"enabled": False},
            run_end_trigger=["test"],
        )
        result = execute_init_workflow(
            preset=None,
            skip_evidence=False,
            skip_triggers=False,
            yes=False,
            is_tty=True,
            presets=["go"],
            commands_lookup=lambda _: ["test", "lint"],
            validate_config=_ok_validator,
            prompts=prompts,
        )
        assert result.exit_code == 0
        assert result.config_data is not None
        assert result.config_data["preset"] == "go"
        assert result.config_data["evidence_check"] == {"required": ["test"]}
        assert result.config_data["per_issue_review"] == {"enabled": False}
        assert result.config_data["validation_triggers"]["run_end"]["commands"] == [
            {"ref": "test"}
        ]
        assert prompts.preset_calls == 1
        assert prompts.evidence_calls == 1
        assert prompts.review_calls == 1
        assert prompts.trigger_calls == 1

    def test_interactive_custom_flow_builds_commands_dict(self) -> None:
        prompts = FakePrompts(
            preset=None,
            custom_commands={"build": "cargo build", "test": "cargo test"},
            evidence_check=None,
            per_issue_review=None,
            run_end_trigger=None,
        )
        result = execute_init_workflow(
            preset=None,
            skip_evidence=False,
            skip_triggers=False,
            yes=False,
            is_tty=True,
            presets=["go"],
            commands_lookup=lambda _: [],
            validate_config=_ok_validator,
            prompts=prompts,
        )
        assert result.exit_code == 0
        assert result.config_data is not None
        assert result.config_data["commands"] == {
            "build": "cargo build",
            "test": "cargo test",
        }
        assert "evidence_check" not in result.config_data
        assert "per_issue_review" not in result.config_data
        assert "validation_triggers" not in result.config_data

    def test_interactive_custom_cancelled_returns_exit_one(self) -> None:
        prompts = FakePrompts(preset=None, custom_commands=None)
        result = execute_init_workflow(
            preset=None,
            skip_evidence=False,
            skip_triggers=False,
            yes=False,
            is_tty=True,
            presets=["go"],
            commands_lookup=lambda _: [],
            validate_config=_ok_validator,
            prompts=prompts,
        )
        assert result.exit_code == 1
        assert result.config_data is None

    def test_empty_commands_emits_notification_and_skips_sections(self) -> None:
        prompts = FakePrompts(preset=None, custom_commands={})
        execute_init_workflow(
            preset=None,
            skip_evidence=False,
            skip_triggers=False,
            yes=False,
            is_tty=True,
            presets=["go"],
            commands_lookup=lambda _: [],
            validate_config=_ok_validator,
            prompts=prompts,
        )
        assert any("skipping evidence" in msg for msg in prompts.notifications)
        assert prompts.evidence_calls == 0
        assert prompts.trigger_calls == 0
        assert prompts.review_calls == 0


class TestExecuteInitWorkflowValidationRecovery:
    """Validation retry/recovery branching."""

    def test_non_tty_validation_error_returns_error_message(self) -> None:
        def failing_validator(data: dict[str, Any]) -> None:
            raise ConfigError("bad config")

        result = execute_init_workflow(
            preset="go",
            skip_evidence=True,
            skip_triggers=True,
            yes=True,
            is_tty=False,
            presets=["go"],
            commands_lookup=lambda _: [],
            validate_config=failing_validator,
            prompts=FakePrompts(),
        )
        assert result.exit_code == 1
        assert result.error_message == "bad config"

    def test_abort_recovery_returns_exit_one(self) -> None:
        def failing_validator(data: dict[str, Any]) -> None:
            raise ConfigError("oops")

        prompts = FakePrompts(recovery_choice=RECOVERY_ABORT)
        result = execute_init_workflow(
            preset="go",
            skip_evidence=False,
            skip_triggers=False,
            yes=True,
            is_tty=True,
            presets=["go"],
            commands_lookup=lambda _: ["test"],
            validate_config=failing_validator,
            prompts=prompts,
        )
        assert result.exit_code == 1
        assert result.config_data is None
        assert len(prompts.recovery_calls) == 1

    def test_cancelled_recovery_returns_exit_one(self) -> None:
        """A None choice (cancellation) also aborts."""

        def failing_validator(data: dict[str, Any]) -> None:
            raise ConfigError("oops")

        prompts = FakePrompts(recovery_choice=None)
        result = execute_init_workflow(
            preset="go",
            skip_evidence=False,
            skip_triggers=False,
            yes=True,
            is_tty=True,
            presets=["go"],
            commands_lookup=lambda _: ["test"],
            validate_config=failing_validator,
            prompts=prompts,
        )
        assert result.exit_code == 1

    def test_skip_recovery_drops_problem_sections_and_revalidates(self) -> None:
        """SKIP removes sections; second validate call succeeds."""
        call_count = {"n": 0}

        def validator(data: dict[str, Any]) -> None:
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise ConfigError("first attempt fails")

        prompts = FakePrompts(recovery_choice=RECOVERY_SKIP)
        result = execute_init_workflow(
            preset="go",
            skip_evidence=False,
            skip_triggers=False,
            yes=True,
            is_tty=True,
            presets=["go"],
            commands_lookup=lambda _: ["test", "lint"],
            validate_config=validator,
            prompts=prompts,
        )
        assert result.exit_code == 0
        assert result.config_data is not None
        assert "evidence_check" not in result.config_data
        assert "per_issue_review" not in result.config_data
        assert "validation_triggers" not in result.config_data

    def test_skip_with_empty_commands_returns_error(self) -> None:
        def failing_validator(data: dict[str, Any]) -> None:
            raise ConfigError("oops")

        prompts = FakePrompts(
            preset=None, custom_commands={}, recovery_choice=RECOVERY_SKIP
        )
        result = execute_init_workflow(
            preset=None,
            skip_evidence=False,
            skip_triggers=False,
            yes=False,
            is_tty=True,
            presets=["go"],
            commands_lookup=lambda _: [],
            validate_config=failing_validator,
            prompts=prompts,
        )
        assert result.exit_code == 1
        assert result.error_message == "Cannot skip - no commands defined"

    def test_revise_recovery_reprompts_and_revalidates(self) -> None:
        """REVISE re-prompts sections; second validate call succeeds."""
        call_count = {"n": 0}

        def validator(data: dict[str, Any]) -> None:
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise ConfigError("first attempt fails")

        prompts = FakePrompts(
            recovery_choice=RECOVERY_REVISE,
            evidence_check=["lint"],
            per_issue_review=None,
            run_end_trigger=["lint"],
        )
        result = execute_init_workflow(
            preset="go",
            skip_evidence=False,
            skip_triggers=False,
            yes=True,
            is_tty=True,
            presets=["go"],
            commands_lookup=lambda _: ["test", "lint"],
            validate_config=validator,
            prompts=prompts,
        )
        assert result.exit_code == 0
        assert result.config_data is not None
        assert result.config_data["evidence_check"] == {"required": ["lint"]}

    def test_exhausting_retries_returns_error(self) -> None:
        def always_fails(data: dict[str, Any]) -> None:
            raise ConfigError("persistent error")

        prompts = FakePrompts(recovery_choice=RECOVERY_SKIP)
        result = execute_init_workflow(
            preset="go",
            skip_evidence=False,
            skip_triggers=False,
            yes=True,
            is_tty=True,
            presets=["go"],
            commands_lookup=lambda _: ["test"],
            validate_config=always_fails,
            prompts=prompts,
        )
        assert result.exit_code == 1
        assert result.error_message is not None
        assert "persistent error" in result.error_message
        assert "max retries exceeded" in result.error_message
