"""Unit tests for TriggerEngine.

Tests for trigger policy evaluation:
- on_issue_completed (session_end trigger)
- on_epic_closed (epic_completion trigger with depth/fire_on filters)
- on_run_end (run_end trigger with fire_on filter)
- Command resolution and overrides
"""

from __future__ import annotations

import pytest

from src.domain.validation.config import (
    CommandConfig,
    CommandsConfig,
    ConfigError,
    EpicCompletionTriggerConfig,
    EpicDepth,
    FailureMode,
    FireOn,
    RunEndTriggerConfig,
    SessionEndTriggerConfig,
    TriggerCommandRef,
    TriggerType,
    ValidationConfig,
    ValidationTriggersConfig,
)
from src.pipeline.issue_result import IssueResult
from src.pipeline.trigger_engine import TriggerActions, TriggerEngine


def make_validation_config(
    *,
    commands: dict[str, str | None] | None = None,
    triggers: ValidationTriggersConfig | None = None,
) -> ValidationConfig:
    """Create a ValidationConfig with specified commands and triggers."""
    cmd_configs: dict[str, CommandConfig | None] = {}
    if commands:
        for name, cmd_str in commands.items():
            if cmd_str is not None:
                cmd_configs[name] = CommandConfig(command=cmd_str)
            else:
                cmd_configs[name] = None

    commands_config = CommandsConfig(
        test=cmd_configs.get("test"),
        lint=cmd_configs.get("lint"),
        format=cmd_configs.get("format"),
        typecheck=cmd_configs.get("typecheck"),
    )

    return ValidationConfig(
        commands=commands_config,
        validation_triggers=triggers,
    )


def make_issue_result(
    issue_id: str = "test-1",
    success: bool = True,
) -> IssueResult:
    """Create a minimal IssueResult for testing."""
    return IssueResult(
        issue_id=issue_id,
        agent_id="agent-1",
        success=success,
        summary="Test summary",
        duration_seconds=1.0,
    )


class TestTriggerEngineInit:
    """Tests for TriggerEngine initialization."""

    def test_none_config_creates_empty_engine(self) -> None:
        """TriggerEngine with None config returns None for all events."""
        engine = TriggerEngine(None)

        result = make_issue_result()
        assert engine.on_issue_completed("test-1", result) is None
        assert engine.on_epic_closed("epic-1", is_top_level=True, success=True) is None
        assert engine.on_run_end(1, 1) is None

    def test_no_triggers_configured(self) -> None:
        """TriggerEngine with no triggers configured returns None for all events."""
        config = make_validation_config(
            commands={"test": "pytest"},
            triggers=None,
        )
        engine = TriggerEngine(config)

        result = make_issue_result()
        assert engine.on_issue_completed("test-1", result) is None
        assert engine.on_epic_closed("epic-1", is_top_level=True, success=True) is None
        assert engine.on_run_end(1, 1) is None


class TestOnIssueCompleted:
    """Tests for on_issue_completed (session_end trigger)."""

    def test_session_end_fires_on_issue_completion(self) -> None:
        """session_end trigger fires on every issue completion."""
        config = make_validation_config(
            commands={"test": "pytest", "lint": "ruff check ."},
            triggers=ValidationTriggersConfig(
                session_end=SessionEndTriggerConfig(
                    failure_mode=FailureMode.CONTINUE,
                    commands=(TriggerCommandRef(ref="test"),),
                ),
            ),
        )
        engine = TriggerEngine(config)

        result = make_issue_result(success=True)
        actions = engine.on_issue_completed("test-1", result)

        assert actions is not None
        assert actions.should_run is True
        assert len(actions.commands) == 1
        assert actions.commands[0].ref == "test"
        assert actions.commands[0].effective_command == "pytest"
        assert actions.failure_mode == FailureMode.CONTINUE
        assert actions.trigger_type == TriggerType.SESSION_END

    def test_session_end_fires_on_failure_too(self) -> None:
        """session_end trigger fires regardless of issue success/failure."""
        config = make_validation_config(
            commands={"test": "pytest"},
            triggers=ValidationTriggersConfig(
                session_end=SessionEndTriggerConfig(
                    failure_mode=FailureMode.ABORT,
                    commands=(TriggerCommandRef(ref="test"),),
                ),
            ),
        )
        engine = TriggerEngine(config)

        result = make_issue_result(success=False)
        actions = engine.on_issue_completed("test-1", result)

        assert actions is not None
        assert actions.should_run is True

    def test_no_session_end_configured(self) -> None:
        """Returns None when session_end trigger is not configured."""
        config = make_validation_config(
            commands={"test": "pytest"},
            triggers=ValidationTriggersConfig(
                epic_completion=EpicCompletionTriggerConfig(
                    failure_mode=FailureMode.CONTINUE,
                    commands=(TriggerCommandRef(ref="test"),),
                    epic_depth=EpicDepth.TOP_LEVEL,
                    fire_on=FireOn.SUCCESS,
                ),
            ),
        )
        engine = TriggerEngine(config)

        result = make_issue_result()
        assert engine.on_issue_completed("test-1", result) is None


class TestOnEpicClosed:
    """Tests for on_epic_closed (epic_completion trigger)."""

    def test_epic_completion_fires_on_success(self) -> None:
        """epic_completion fires when fire_on=SUCCESS and epic succeeds."""
        config = make_validation_config(
            commands={"lint": "ruff check ."},
            triggers=ValidationTriggersConfig(
                epic_completion=EpicCompletionTriggerConfig(
                    failure_mode=FailureMode.REMEDIATE,
                    commands=(TriggerCommandRef(ref="lint"),),
                    epic_depth=EpicDepth.ALL,
                    fire_on=FireOn.SUCCESS,
                ),
            ),
        )
        engine = TriggerEngine(config)

        actions = engine.on_epic_closed("epic-1", is_top_level=False, success=True)

        assert actions is not None
        assert actions.should_run is True
        assert len(actions.commands) == 1
        assert actions.commands[0].ref == "lint"
        assert actions.failure_mode == FailureMode.REMEDIATE
        assert actions.trigger_type == TriggerType.EPIC_COMPLETION

    def test_epic_completion_skips_on_failure_when_fire_on_success(self) -> None:
        """epic_completion skips when fire_on=SUCCESS and epic fails."""
        config = make_validation_config(
            commands={"lint": "ruff check ."},
            triggers=ValidationTriggersConfig(
                epic_completion=EpicCompletionTriggerConfig(
                    failure_mode=FailureMode.CONTINUE,
                    commands=(TriggerCommandRef(ref="lint"),),
                    epic_depth=EpicDepth.ALL,
                    fire_on=FireOn.SUCCESS,
                ),
            ),
        )
        engine = TriggerEngine(config)

        actions = engine.on_epic_closed("epic-1", is_top_level=True, success=False)
        assert actions is None

    def test_epic_completion_fires_on_failure(self) -> None:
        """epic_completion fires when fire_on=FAILURE and epic fails."""
        config = make_validation_config(
            commands={"lint": "ruff check ."},
            triggers=ValidationTriggersConfig(
                epic_completion=EpicCompletionTriggerConfig(
                    failure_mode=FailureMode.ABORT,
                    commands=(TriggerCommandRef(ref="lint"),),
                    epic_depth=EpicDepth.ALL,
                    fire_on=FireOn.FAILURE,
                ),
            ),
        )
        engine = TriggerEngine(config)

        actions = engine.on_epic_closed("epic-1", is_top_level=True, success=False)

        assert actions is not None
        assert actions.should_run is True

    def test_epic_completion_skips_on_success_when_fire_on_failure(self) -> None:
        """epic_completion skips when fire_on=FAILURE and epic succeeds."""
        config = make_validation_config(
            commands={"lint": "ruff check ."},
            triggers=ValidationTriggersConfig(
                epic_completion=EpicCompletionTriggerConfig(
                    failure_mode=FailureMode.CONTINUE,
                    commands=(TriggerCommandRef(ref="lint"),),
                    epic_depth=EpicDepth.ALL,
                    fire_on=FireOn.FAILURE,
                ),
            ),
        )
        engine = TriggerEngine(config)

        actions = engine.on_epic_closed("epic-1", is_top_level=True, success=True)
        assert actions is None

    def test_epic_completion_fires_on_both(self) -> None:
        """epic_completion fires when fire_on=BOTH regardless of success."""
        config = make_validation_config(
            commands={"test": "pytest"},
            triggers=ValidationTriggersConfig(
                epic_completion=EpicCompletionTriggerConfig(
                    failure_mode=FailureMode.CONTINUE,
                    commands=(TriggerCommandRef(ref="test"),),
                    epic_depth=EpicDepth.ALL,
                    fire_on=FireOn.BOTH,
                ),
            ),
        )
        engine = TriggerEngine(config)

        # Fires on success
        actions = engine.on_epic_closed("epic-1", is_top_level=True, success=True)
        assert actions is not None

        # Fires on failure
        actions = engine.on_epic_closed("epic-1", is_top_level=True, success=False)
        assert actions is not None

    def test_epic_depth_top_level_only(self) -> None:
        """epic_depth=TOP_LEVEL only fires for top-level epics."""
        config = make_validation_config(
            commands={"test": "pytest"},
            triggers=ValidationTriggersConfig(
                epic_completion=EpicCompletionTriggerConfig(
                    failure_mode=FailureMode.CONTINUE,
                    commands=(TriggerCommandRef(ref="test"),),
                    epic_depth=EpicDepth.TOP_LEVEL,
                    fire_on=FireOn.SUCCESS,
                ),
            ),
        )
        engine = TriggerEngine(config)

        # Top-level fires
        actions = engine.on_epic_closed("epic-1", is_top_level=True, success=True)
        assert actions is not None

        # Nested skipped
        actions = engine.on_epic_closed("epic-1.1", is_top_level=False, success=True)
        assert actions is None

    def test_epic_depth_all(self) -> None:
        """epic_depth=ALL fires for all epics including nested."""
        config = make_validation_config(
            commands={"test": "pytest"},
            triggers=ValidationTriggersConfig(
                epic_completion=EpicCompletionTriggerConfig(
                    failure_mode=FailureMode.CONTINUE,
                    commands=(TriggerCommandRef(ref="test"),),
                    epic_depth=EpicDepth.ALL,
                    fire_on=FireOn.SUCCESS,
                ),
            ),
        )
        engine = TriggerEngine(config)

        # Top-level fires
        actions = engine.on_epic_closed("epic-1", is_top_level=True, success=True)
        assert actions is not None

        # Nested fires too
        actions = engine.on_epic_closed("epic-1.1", is_top_level=False, success=True)
        assert actions is not None

    def test_no_epic_completion_configured(self) -> None:
        """Returns None when epic_completion trigger is not configured."""
        config = make_validation_config(
            commands={"test": "pytest"},
            triggers=ValidationTriggersConfig(
                session_end=SessionEndTriggerConfig(
                    failure_mode=FailureMode.CONTINUE,
                    commands=(TriggerCommandRef(ref="test"),),
                ),
            ),
        )
        engine = TriggerEngine(config)

        assert engine.on_epic_closed("epic-1", is_top_level=True, success=True) is None


class TestOnRunEnd:
    """Tests for on_run_end (run_end trigger)."""

    def test_run_end_fires_on_success(self) -> None:
        """run_end fires when fire_on=SUCCESS and all issues succeed."""
        config = make_validation_config(
            commands={"test": "pytest"},
            triggers=ValidationTriggersConfig(
                run_end=RunEndTriggerConfig(
                    failure_mode=FailureMode.CONTINUE,
                    commands=(TriggerCommandRef(ref="test"),),
                    fire_on=FireOn.SUCCESS,
                ),
            ),
        )
        engine = TriggerEngine(config)

        actions = engine.on_run_end(5, 5)

        assert actions is not None
        assert actions.should_run is True
        assert len(actions.commands) == 1
        assert actions.commands[0].ref == "test"
        assert actions.trigger_type == TriggerType.RUN_END

    def test_run_end_skips_on_partial_failure(self) -> None:
        """run_end skips when fire_on=SUCCESS and some issues failed."""
        config = make_validation_config(
            commands={"test": "pytest"},
            triggers=ValidationTriggersConfig(
                run_end=RunEndTriggerConfig(
                    failure_mode=FailureMode.CONTINUE,
                    commands=(TriggerCommandRef(ref="test"),),
                    fire_on=FireOn.SUCCESS,
                ),
            ),
        )
        engine = TriggerEngine(config)

        actions = engine.on_run_end(3, 5)
        assert actions is None

    def test_run_end_fires_on_failure(self) -> None:
        """run_end fires when fire_on=FAILURE and some issues failed."""
        config = make_validation_config(
            commands={"test": "pytest"},
            triggers=ValidationTriggersConfig(
                run_end=RunEndTriggerConfig(
                    failure_mode=FailureMode.ABORT,
                    commands=(TriggerCommandRef(ref="test"),),
                    fire_on=FireOn.FAILURE,
                ),
            ),
        )
        engine = TriggerEngine(config)

        actions = engine.on_run_end(3, 5)

        assert actions is not None
        assert actions.should_run is True

    def test_run_end_skips_on_all_success_when_fire_on_failure(self) -> None:
        """run_end skips when fire_on=FAILURE and all issues succeed."""
        config = make_validation_config(
            commands={"test": "pytest"},
            triggers=ValidationTriggersConfig(
                run_end=RunEndTriggerConfig(
                    failure_mode=FailureMode.CONTINUE,
                    commands=(TriggerCommandRef(ref="test"),),
                    fire_on=FireOn.FAILURE,
                ),
            ),
        )
        engine = TriggerEngine(config)

        actions = engine.on_run_end(5, 5)
        assert actions is None

    def test_run_end_fires_on_both(self) -> None:
        """run_end fires when fire_on=BOTH regardless of success."""
        config = make_validation_config(
            commands={"test": "pytest"},
            triggers=ValidationTriggersConfig(
                run_end=RunEndTriggerConfig(
                    failure_mode=FailureMode.CONTINUE,
                    commands=(TriggerCommandRef(ref="test"),),
                    fire_on=FireOn.BOTH,
                ),
            ),
        )
        engine = TriggerEngine(config)

        # Fires on all success
        actions = engine.on_run_end(5, 5)
        assert actions is not None

        # Fires on partial failure
        actions = engine.on_run_end(3, 5)
        assert actions is not None

    def test_run_end_zero_issues_is_failure(self) -> None:
        """Zero total issues is treated as failure for run_end."""
        config = make_validation_config(
            commands={"test": "pytest"},
            triggers=ValidationTriggersConfig(
                run_end=RunEndTriggerConfig(
                    failure_mode=FailureMode.CONTINUE,
                    commands=(TriggerCommandRef(ref="test"),),
                    fire_on=FireOn.SUCCESS,
                ),
            ),
        )
        engine = TriggerEngine(config)

        # 0/0 is treated as failure
        actions = engine.on_run_end(0, 0)
        assert actions is None

    def test_no_run_end_configured(self) -> None:
        """Returns None when run_end trigger is not configured."""
        config = make_validation_config(
            commands={"test": "pytest"},
            triggers=ValidationTriggersConfig(
                session_end=SessionEndTriggerConfig(
                    failure_mode=FailureMode.CONTINUE,
                    commands=(TriggerCommandRef(ref="test"),),
                ),
            ),
        )
        engine = TriggerEngine(config)

        assert engine.on_run_end(5, 5) is None


class TestCommandResolution:
    """Tests for command resolution and overrides."""

    def test_resolve_command_from_base_pool(self) -> None:
        """Commands are resolved from base pool."""
        config = make_validation_config(
            commands={"test": "uv run pytest", "lint": "ruff check ."},
            triggers=ValidationTriggersConfig(
                session_end=SessionEndTriggerConfig(
                    failure_mode=FailureMode.CONTINUE,
                    commands=(
                        TriggerCommandRef(ref="test"),
                        TriggerCommandRef(ref="lint"),
                    ),
                ),
            ),
        )
        engine = TriggerEngine(config)

        result = make_issue_result()
        actions = engine.on_issue_completed("test-1", result)

        assert actions is not None
        assert len(actions.commands) == 2
        assert actions.commands[0].ref == "test"
        assert actions.commands[0].effective_command == "uv run pytest"
        assert actions.commands[1].ref == "lint"
        assert actions.commands[1].effective_command == "ruff check ."

    def test_command_override(self) -> None:
        """Command string can be overridden in trigger ref."""
        config = make_validation_config(
            commands={"test": "pytest"},
            triggers=ValidationTriggersConfig(
                session_end=SessionEndTriggerConfig(
                    failure_mode=FailureMode.CONTINUE,
                    commands=(
                        TriggerCommandRef(ref="test", command="pytest -v --tb=short"),
                    ),
                ),
            ),
        )
        engine = TriggerEngine(config)

        result = make_issue_result()
        actions = engine.on_issue_completed("test-1", result)

        assert actions is not None
        assert actions.commands[0].effective_command == "pytest -v --tb=short"

    def test_timeout_override(self) -> None:
        """Timeout can be overridden in trigger ref."""
        config = make_validation_config(
            commands={"test": "pytest"},
            triggers=ValidationTriggersConfig(
                session_end=SessionEndTriggerConfig(
                    failure_mode=FailureMode.CONTINUE,
                    commands=(TriggerCommandRef(ref="test", timeout=120),),
                ),
            ),
        )
        engine = TriggerEngine(config)

        result = make_issue_result()
        actions = engine.on_issue_completed("test-1", result)

        assert actions is not None
        assert actions.commands[0].effective_timeout == 120

    def test_unknown_command_ref_raises_error(self) -> None:
        """ConfigError raised when command ref not in base pool."""
        config = make_validation_config(
            commands={"test": "pytest"},
            triggers=ValidationTriggersConfig(
                session_end=SessionEndTriggerConfig(
                    failure_mode=FailureMode.CONTINUE,
                    commands=(TriggerCommandRef(ref="unknown"),),
                ),
            ),
        )
        engine = TriggerEngine(config)

        result = make_issue_result()
        with pytest.raises(ConfigError, match="references unknown command 'unknown'"):
            engine.on_issue_completed("test-1", result)

    def test_resolve_commands_public_interface(self) -> None:
        """resolve_commands() provides public interface for command resolution."""
        config = make_validation_config(
            commands={"test": "pytest"},
            triggers=ValidationTriggersConfig(
                session_end=SessionEndTriggerConfig(
                    failure_mode=FailureMode.CONTINUE,
                    commands=(TriggerCommandRef(ref="test"),),
                ),
            ),
        )
        engine = TriggerEngine(config)

        trigger_config = config.validation_triggers.session_end  # type: ignore[union-attr]
        assert trigger_config is not None
        resolved = engine.resolve_commands(trigger_config, TriggerType.SESSION_END)

        assert len(resolved) == 1
        assert resolved[0].ref == "test"
        assert resolved[0].effective_command == "pytest"


class TestTriggerActionsDataclass:
    """Tests for TriggerActions dataclass."""

    def test_trigger_actions_immutable(self) -> None:
        """TriggerActions is immutable (frozen)."""
        actions = TriggerActions(
            should_run=True,
            commands=[],
            failure_mode=FailureMode.CONTINUE,
        )

        with pytest.raises(AttributeError):
            actions.should_run = False  # type: ignore[misc]

    def test_trigger_actions_defaults(self) -> None:
        """TriggerActions has sensible defaults."""
        actions = TriggerActions(
            should_run=False,
            commands=[],
            failure_mode=FailureMode.ABORT,
        )

        assert actions.trigger_type is None
        assert actions.trigger_config is None
