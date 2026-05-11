"""Section-level tests for :mod:`src.domain.validation.config_parser`.

These tests cover each top-level YAML/dict section that
``parse_validation_config`` parses, asserting:

- Parser returns the expected typed config for each shape.
- ``was_present`` is True when the key appears in the dict (even with null),
  False otherwise — the ``_fields_set`` distinction the merger relies on.
- User-visible error messages are stable.

The aggregate ``parse_validation_config`` is also tested end-to-end including
the ``coder=amp + amp_mode=deep + effort in {high, max}`` cross-section rule.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from src.domain.validation.config_types import (
    CodeReviewConfig,
    CodexOptionsConfig,
    CommandsConfig,
    ConfigError,
    EpicVerifierConfig,
    EvidenceCheckConfig,
    FailureMode,
    ValidationConfig,
    ValidationTriggersConfig,
    YamlCoverageConfig,
)
from src.domain.validation.config_parser import (
    parse_amp_mode,
    parse_claude_settings_sources,
    parse_coder,
    parse_coder_options,
    parse_commands,
    parse_coverage,
    parse_effort,
    parse_epic_verification,
    parse_evidence_check,
    parse_idle_timeout_seconds,
    parse_max_diff_size_kb,
    parse_max_idle_retries,
    parse_per_issue_review,
    parse_preset,
    parse_string_list_field,
    parse_timeout_minutes,
    parse_validation_config,
    parse_validation_triggers,
)


# ---------------------------------------------------------------------------
# parse_preset
# ---------------------------------------------------------------------------


class TestParsePreset:
    def test_missing(self) -> None:
        assert parse_preset({}) == (None, False)

    def test_string(self) -> None:
        assert parse_preset({"preset": "python-uv"}) == ("python-uv", True)

    def test_explicit_null(self) -> None:
        assert parse_preset({"preset": None}) == (None, True)

    def test_invalid_type(self) -> None:
        with pytest.raises(ConfigError, match="preset must be a string"):
            parse_preset({"preset": 123})


# ---------------------------------------------------------------------------
# parse_commands
# ---------------------------------------------------------------------------


class TestParseCommands:
    def test_missing_returns_empty_commands(self) -> None:
        commands, present = parse_commands({})
        assert isinstance(commands, CommandsConfig)
        assert commands.setup is None
        assert commands.test is None
        assert present is False

    def test_null_returns_empty_commands(self) -> None:
        commands, present = parse_commands({"commands": None})
        assert isinstance(commands, CommandsConfig)
        assert commands.test is None
        assert present is True

    def test_dict_parsed(self) -> None:
        commands, present = parse_commands(
            {"commands": {"test": "pytest", "lint": "ruff check"}}
        )
        assert present is True
        assert commands.test is not None
        assert commands.test.command == "pytest"
        assert commands.lint is not None
        assert commands.lint.command == "ruff check"

    def test_custom_command_parsed(self) -> None:
        commands, _ = parse_commands({"commands": {"my_check": "echo ok"}})
        assert "my_check" in commands.custom_commands
        assert commands.custom_commands["my_check"].command == "echo ok"

    def test_invalid_type(self) -> None:
        with pytest.raises(ConfigError, match="commands must be an object"):
            parse_commands({"commands": "pytest"})

    def test_invalid_inner_command_message_preserved(self) -> None:
        with pytest.raises(ConfigError, match="cannot be empty string"):
            parse_commands({"commands": {"test": ""}})


# ---------------------------------------------------------------------------
# parse_coverage
# ---------------------------------------------------------------------------


class TestParseCoverage:
    def test_missing(self) -> None:
        assert parse_coverage({}) == (None, False)

    def test_explicit_null(self) -> None:
        assert parse_coverage({"coverage": None}) == (None, True)

    def test_full(self) -> None:
        coverage, present = parse_coverage(
            {
                "coverage": {
                    "format": "xml",
                    "file": "coverage.xml",
                    "threshold": 80,
                }
            }
        )
        assert present is True
        assert isinstance(coverage, YamlCoverageConfig)
        assert coverage.format == "xml"
        assert coverage.threshold == 80.0

    def test_invalid_type(self) -> None:
        with pytest.raises(ConfigError, match="coverage must be an object"):
            parse_coverage({"coverage": "bad"})

    def test_invalid_inner_message_preserved(self) -> None:
        with pytest.raises(ConfigError, match="Unsupported coverage format 'lcov'"):
            parse_coverage(
                {
                    "coverage": {
                        "format": "lcov",
                        "file": "lcov.info",
                        "threshold": 80,
                    }
                }
            )


# ---------------------------------------------------------------------------
# parse_string_list_field (code_patterns / config_files / setup_files)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["code_patterns", "config_files", "setup_files"])
class TestParseStringListField:
    def test_missing(self, key: str) -> None:
        assert parse_string_list_field({}, key) == ((), False)

    def test_explicit_null(self, key: str) -> None:
        assert parse_string_list_field({key: None}, key) == ((), True)

    def test_empty_list(self, key: str) -> None:
        assert parse_string_list_field({key: []}, key) == ((), True)

    def test_list_of_strings(self, key: str) -> None:
        value, present = parse_string_list_field({key: ["a", "b", "c"]}, key)
        assert value == ("a", "b", "c")
        assert present is True

    def test_non_list_type(self, key: str) -> None:
        with pytest.raises(ConfigError, match=f"{key} must be a list"):
            parse_string_list_field({key: "a,b"}, key)

    def test_non_string_item(self, key: str) -> None:
        with pytest.raises(ConfigError, match=rf"{key}\[1\] must be a string"):
            parse_string_list_field({key: ["ok", 42]}, key)


# ---------------------------------------------------------------------------
# parse_claude_settings_sources
# ---------------------------------------------------------------------------


class TestParseClaudeSettingsSources:
    def test_missing(self) -> None:
        assert parse_claude_settings_sources({}) == (None, False)

    def test_explicit_null(self) -> None:
        assert parse_claude_settings_sources({"claude_settings_sources": None}) == (
            None,
            True,
        )

    def test_empty_list(self) -> None:
        assert parse_claude_settings_sources({"claude_settings_sources": []}) == (
            (),
            True,
        )

    def test_valid_sources(self) -> None:
        value, present = parse_claude_settings_sources(
            {"claude_settings_sources": ["local", "project", "user"]}
        )
        assert value == ("local", "project", "user")
        assert present is True

    def test_whitespace_stripped(self) -> None:
        value, _ = parse_claude_settings_sources(
            {"claude_settings_sources": ["  local  ", " project"]}
        )
        assert value == ("local", "project")

    def test_invalid_source(self) -> None:
        with pytest.raises(
            ConfigError,
            match=r"Invalid Claude settings source 'foo'\. Valid sources: local, project, user",
        ):
            parse_claude_settings_sources({"claude_settings_sources": ["foo"]})

    def test_non_list_type(self) -> None:
        with pytest.raises(ConfigError, match="claude_settings_sources must be a list"):
            parse_claude_settings_sources({"claude_settings_sources": "local"})

    def test_non_string_item(self) -> None:
        with pytest.raises(
            ConfigError, match=r"claude_settings_sources\[0\] must be a string"
        ):
            parse_claude_settings_sources({"claude_settings_sources": [123]})


# ---------------------------------------------------------------------------
# parse_validation_triggers
# ---------------------------------------------------------------------------


class TestParseValidationTriggers:
    def test_missing(self) -> None:
        assert parse_validation_triggers({}) == (None, False)

    def test_explicit_null(self) -> None:
        assert parse_validation_triggers({"validation_triggers": None}) == (
            None,
            True,
        )

    def test_empty_dict(self) -> None:
        triggers, present = parse_validation_triggers({"validation_triggers": {}})
        assert present is True
        assert isinstance(triggers, ValidationTriggersConfig)
        assert triggers.epic_completion is None
        assert triggers.periodic is None

    def test_periodic_parsed(self) -> None:
        triggers, present = parse_validation_triggers(
            {
                "validation_triggers": {
                    "periodic": {
                        "interval": 3600,
                        "failure_mode": "continue",
                        "commands": [{"ref": "test"}],
                    }
                }
            }
        )
        assert present is True
        assert triggers is not None
        assert triggers.periodic is not None
        assert triggers.periodic.interval == 3600

    def test_invalid_type(self) -> None:
        with pytest.raises(ConfigError, match="validation_triggers must be an object"):
            parse_validation_triggers({"validation_triggers": "nope"})


# ---------------------------------------------------------------------------
# Numeric scalar fields
# ---------------------------------------------------------------------------


class TestParseTimeoutMinutes:
    def test_missing(self) -> None:
        assert parse_timeout_minutes({}) == (None, False)

    def test_explicit_null(self) -> None:
        assert parse_timeout_minutes({"timeout_minutes": None}) == (None, True)

    def test_positive_int(self) -> None:
        assert parse_timeout_minutes({"timeout_minutes": 30}) == (30, True)

    @pytest.mark.parametrize("value", ["60", 60.5, True, False])
    def test_invalid_type(self, value: object) -> None:
        with pytest.raises(ConfigError, match="timeout_minutes must be an integer"):
            parse_timeout_minutes({"timeout_minutes": value})

    @pytest.mark.parametrize("value", [0, -1])
    def test_non_positive(self, value: int) -> None:
        with pytest.raises(ConfigError, match="timeout_minutes must be positive"):
            parse_timeout_minutes({"timeout_minutes": value})


class TestParseMaxIdleRetries:
    def test_missing(self) -> None:
        assert parse_max_idle_retries({}) == (None, False)

    def test_explicit_null(self) -> None:
        assert parse_max_idle_retries({"max_idle_retries": None}) == (None, True)

    @pytest.mark.parametrize("value", [0, 3])
    def test_non_negative(self, value: int) -> None:
        assert parse_max_idle_retries({"max_idle_retries": value}) == (value, True)

    def test_negative_rejected(self) -> None:
        with pytest.raises(ConfigError, match="max_idle_retries must be non-negative"):
            parse_max_idle_retries({"max_idle_retries": -1})

    @pytest.mark.parametrize("value", [True, False, "2", 1.5])
    def test_invalid_type(self, value: object) -> None:
        with pytest.raises(ConfigError, match="max_idle_retries must be an integer"):
            parse_max_idle_retries({"max_idle_retries": value})


class TestParseMaxDiffSizeKb:
    def test_missing(self) -> None:
        assert parse_max_diff_size_kb({}) == (None, False)

    @pytest.mark.parametrize("value", [0, 256])
    def test_non_negative(self, value: int) -> None:
        assert parse_max_diff_size_kb({"max_diff_size_kb": value}) == (value, True)

    def test_negative_rejected(self) -> None:
        with pytest.raises(ConfigError, match="max_diff_size_kb must be non-negative"):
            parse_max_diff_size_kb({"max_diff_size_kb": -5})

    @pytest.mark.parametrize("value", [True, False, "100"])
    def test_invalid_type(self, value: object) -> None:
        with pytest.raises(ConfigError, match="max_diff_size_kb must be an integer"):
            parse_max_diff_size_kb({"max_diff_size_kb": value})


class TestParseIdleTimeoutSeconds:
    def test_missing(self) -> None:
        assert parse_idle_timeout_seconds({}) == (None, False)

    def test_explicit_null(self) -> None:
        assert parse_idle_timeout_seconds({"idle_timeout_seconds": None}) == (
            None,
            True,
        )

    @pytest.mark.parametrize("value,expected", [(0, 0.0), (10, 10.0), (1.5, 1.5)])
    def test_non_negative_number(self, value: float, expected: float) -> None:
        assert parse_idle_timeout_seconds({"idle_timeout_seconds": value}) == (
            expected,
            True,
        )

    def test_negative_rejected(self) -> None:
        with pytest.raises(
            ConfigError, match="idle_timeout_seconds must be non-negative"
        ):
            parse_idle_timeout_seconds({"idle_timeout_seconds": -0.1})

    @pytest.mark.parametrize("value", [True, False, "10"])
    def test_invalid_type(self, value: object) -> None:
        with pytest.raises(ConfigError, match="idle_timeout_seconds must be a number"):
            parse_idle_timeout_seconds({"idle_timeout_seconds": value})


# ---------------------------------------------------------------------------
# parse_evidence_check / parse_epic_verification / parse_per_issue_review
# ---------------------------------------------------------------------------


class TestParseEvidenceCheck:
    def test_missing(self) -> None:
        assert parse_evidence_check({}) == (None, False)

    def test_explicit_null(self) -> None:
        assert parse_evidence_check({"evidence_check": None}) == (None, True)

    def test_dict_parsed(self) -> None:
        ec, present = parse_evidence_check(
            {"evidence_check": {"required": ["lint", "test"]}}
        )
        assert present is True
        assert isinstance(ec, EvidenceCheckConfig)
        assert ec.required == ("lint", "test")

    def test_invalid_type(self) -> None:
        with pytest.raises(ConfigError, match="evidence_check must be an object"):
            parse_evidence_check({"evidence_check": "no"})


class TestParseEpicVerification:
    def test_missing(self) -> None:
        assert parse_epic_verification({}) == (None, False)

    def test_explicit_null(self) -> None:
        assert parse_epic_verification({"epic_verification": None}) == (None, True)

    def test_dict_parsed(self) -> None:
        ev, present = parse_epic_verification(
            {"epic_verification": {"enabled": False, "max_retries": 5}}
        )
        assert present is True
        assert isinstance(ev, EpicVerifierConfig)
        assert ev.enabled is False
        assert ev.max_retries == 5

    def test_invalid_type(self) -> None:
        with pytest.raises(ConfigError, match="epic_verification must be an object"):
            parse_epic_verification({"epic_verification": []})


class TestParsePerIssueReview:
    def test_missing(self) -> None:
        assert parse_per_issue_review({}) == (None, False)

    def test_explicit_null(self) -> None:
        assert parse_per_issue_review({"per_issue_review": None}) == (None, True)

    def test_dict_parsed(self) -> None:
        review, present = parse_per_issue_review(
            {"per_issue_review": {"enabled": True}}
        )
        assert present is True
        assert isinstance(review, CodeReviewConfig)
        assert review.enabled is True

    def test_invalid_type(self) -> None:
        with pytest.raises(ConfigError, match="per_issue_review must be an object"):
            parse_per_issue_review({"per_issue_review": "yes"})


# ---------------------------------------------------------------------------
# Coder / amp_mode / effort / coder_options
# ---------------------------------------------------------------------------


class TestParseCoder:
    def test_missing(self) -> None:
        assert parse_coder({}) == (None, False)

    def test_explicit_null(self) -> None:
        assert parse_coder({"coder": None}) == (None, True)

    @pytest.mark.parametrize("name", ["claude", "amp", "codex"])
    def test_valid_values(self, name: str) -> None:
        value, present = parse_coder({"coder": name})
        assert value == name
        assert present is True

    @pytest.mark.parametrize("bad", ["openai", "GPT", "", 1])
    def test_invalid_value(self, bad: object) -> None:
        with pytest.raises(
            ConfigError,
            match=r"coder must be 'claude', 'amp', or 'codex'",
        ):
            parse_coder({"coder": bad})


class TestParseAmpMode:
    def test_missing(self) -> None:
        assert parse_amp_mode({}) == (None, False)

    @pytest.mark.parametrize("name", ["smart", "rush", "deep"])
    def test_valid_values(self, name: str) -> None:
        value, present = parse_amp_mode({"amp_mode": name})
        assert value == name
        assert present is True

    def test_invalid_value(self) -> None:
        with pytest.raises(
            ConfigError, match=r"amp_mode must be 'smart', 'rush', or 'deep'"
        ):
            parse_amp_mode({"amp_mode": "fast"})


class TestParseEffort:
    def test_missing(self) -> None:
        assert parse_effort({}) == (None, False)

    @pytest.mark.parametrize("name", ["low", "medium", "high", "xhigh", "max"])
    def test_valid_values(self, name: str) -> None:
        value, present = parse_effort({"effort": name})
        assert value == name
        assert present is True

    def test_invalid_value(self) -> None:
        with pytest.raises(
            ConfigError,
            match=r"effort must be 'low', 'medium', 'high', 'xhigh', or 'max'",
        ):
            parse_effort({"effort": "extreme"})


class TestParseCoderOptions:
    def test_missing(self) -> None:
        assert parse_coder_options({}) == (None, False)

    def test_empty_block_yields_defaults(self) -> None:
        opts, present = parse_coder_options({"coder_options": {"codex": {}}})
        assert present is True
        assert isinstance(opts, CodexOptionsConfig)
        assert opts == CodexOptionsConfig()

    def test_codex_options_round_trip(self) -> None:
        opts, _ = parse_coder_options(
            {
                "coder_options": {
                    "codex": {
                        "model": "gpt-5.5",
                        "effort": "high",
                        "approval_policy": "never",
                        "sandbox": "danger-full-access",
                    }
                }
            }
        )
        assert opts is not None
        assert opts.model == "gpt-5.5"
        assert opts.effort == "high"
        assert opts.approval_policy == "never"
        assert opts.sandbox == "danger-full-access"

    def test_unknown_top_level_rejected(self) -> None:
        with pytest.raises(ConfigError, match=r"Unknown field 'amp' in coder_options"):
            parse_coder_options({"coder_options": {"amp": {"mode": "deep"}}})

    def test_non_object_rejected(self) -> None:
        with pytest.raises(ConfigError, match="coder_options must be an object"):
            parse_coder_options({"coder_options": "codex"})


# ---------------------------------------------------------------------------
# Aggregate: parse_validation_config
# ---------------------------------------------------------------------------


class TestParseValidationConfigAggregate:
    def test_empty_dict_yields_defaults(self) -> None:
        cfg = parse_validation_config({})
        assert isinstance(cfg, ValidationConfig)
        assert cfg.preset is None
        assert cfg.commands.test is None
        assert cfg.code_patterns == ()
        assert cfg.epic_verification == EpicVerifierConfig()
        assert cfg.per_issue_review == CodeReviewConfig(enabled=False)
        assert cfg._fields_set == frozenset()

    def test_parse_validation_config_minimal(self) -> None:
        cfg = parse_validation_config({})
        assert isinstance(cfg, ValidationConfig)
        assert cfg.preset is None
        assert cfg._fields_set == frozenset()

    def test_parse_validation_config_full(self) -> None:
        data: dict[str, Any] = {
            "preset": "python-uv",
            "commands": {"test": "pytest", "lint": "ruff check"},
            "coverage": {
                "format": "xml",
                "file": "coverage.xml",
                "threshold": 80,
            },
            "code_patterns": ["*.py"],
            "config_files": ["pyproject.toml"],
            "setup_files": ["uv.lock"],
            "claude_settings_sources": ["local", "project"],
            "validation_triggers": {
                "periodic": {
                    "interval": 3600,
                    "failure_mode": "continue",
                    "commands": [{"ref": "test"}],
                }
            },
            "timeout_minutes": 30,
            "max_idle_retries": 2,
            "idle_timeout_seconds": 90.0,
            "max_diff_size_kb": 200,
            "evidence_check": {"required": ["lint"]},
            "epic_verification": {"enabled": False, "max_retries": 4},
            "per_issue_review": {"enabled": True, "max_retries": 1},
            "coder": "claude",
            "effort": "medium",
            "coder_options": {"codex": {"model": "gpt-5.5"}},
        }
        cfg = parse_validation_config(data)
        assert cfg.preset == "python-uv"
        assert cfg.coder == "claude"
        assert cfg.effort == "medium"
        assert cfg.epic_verification.enabled is False
        assert cfg.epic_verification.max_retries == 4
        assert cfg.per_issue_review.enabled is True
        assert cfg.per_issue_review.max_retries == 1
        assert cfg.evidence_check is not None
        assert cfg.evidence_check.required == ("lint",)
        assert cfg.validation_triggers is not None
        assert cfg.validation_triggers.periodic is not None
        assert cfg.validation_triggers.periodic.interval == 3600
        assert cfg.codex_options is not None
        assert cfg.codex_options.model == "gpt-5.5"

    def test_fields_set_tracks_explicit_keys(self) -> None:
        cfg = parse_validation_config(
            {
                "preset": "python-uv",
                "timeout_minutes": None,
                "code_patterns": [],
            }
        )
        assert "preset" in cfg._fields_set
        assert "timeout_minutes" in cfg._fields_set
        assert "code_patterns" in cfg._fields_set
        assert "commands" not in cfg._fields_set
        assert "coverage" not in cfg._fields_set

    def test_amp_deep_high_effort_rejected(self) -> None:
        with pytest.raises(
            ConfigError,
            match=(
                r"effort must be 'low', 'medium', or 'xhigh' when "
                r"coder is 'amp' and amp_mode is 'deep'"
            ),
        ):
            parse_validation_config(
                {"coder": "amp", "amp_mode": "deep", "effort": "high"}
            )

    def test_amp_deep_max_effort_rejected_default_amp_mode(self) -> None:
        # ``amp_mode`` defaults to ``deep`` when omitted, so ``max`` still trips.
        with pytest.raises(
            ConfigError,
            match=r"coder is 'amp' and amp_mode is 'deep'",
        ):
            parse_validation_config({"coder": "amp", "effort": "max"})

    def test_amp_rush_high_effort_allowed(self) -> None:
        cfg = parse_validation_config(
            {"coder": "amp", "amp_mode": "rush", "effort": "high"}
        )
        assert cfg.coder == "amp"
        assert cfg.amp_mode == "rush"
        assert cfg.effort == "high"

    def test_failure_mode_round_trip_via_aggregate(self) -> None:
        """Trigger nested enums survive parser → ValidationConfig path."""
        cfg = parse_validation_config(
            {
                "validation_triggers": {
                    "session_end": {
                        "failure_mode": "abort",
                        "commands": [],
                    }
                }
            }
        )
        triggers = cast("ValidationTriggersConfig", cfg.validation_triggers)
        assert triggers.session_end is not None
        assert triggers.session_end.failure_mode == FailureMode.ABORT

    def test_invalid_preset_message_preserved(self) -> None:
        with pytest.raises(ConfigError, match="preset must be a string"):
            parse_validation_config({"preset": 123})

    def test_invalid_coder_message_preserved(self) -> None:
        with pytest.raises(
            ConfigError, match=r"coder must be 'claude', 'amp', or 'codex'"
        ):
            parse_validation_config({"coder": "openai"})


# ---------------------------------------------------------------------------
# Module hygiene: ValidationConfig is a pure dataclass boundary; the parser
# owns all parsing logic.
# ---------------------------------------------------------------------------


class TestModuleSurface:
    def test_parser_does_not_call_validationconfig_from_dict(self) -> None:
        """The parser must not call ``ValidationConfig.from_dict`` itself.

        ``from_dict`` has been removed from ``ValidationConfig`` as part of
        T_C2; this check guards against accidental reintroduction (e.g. via a
        new convenience method on the dataclass). Docstring/comment mentions
        are allowed; we grep for the call pattern with an open paren so prose
        references don't trip the check.
        """
        import inspect

        from src.domain.validation import config_parser

        source = inspect.getsource(config_parser)
        assert "ValidationConfig.from_dict(" not in source

    def test_validationconfig_has_no_from_dict(self) -> None:
        """``ValidationConfig`` must not expose ``from_dict``.

        AC-1 of the Workstream C remediation: ``ValidationConfig`` is a pure
        dataclass boundary, so all parsing entry points live on
        :func:`config_parser.parse_validation_config`. Reintroducing
        ``ValidationConfig.from_dict`` would put a parsing seam back on the
        type module via a lazy import.
        """
        from src.domain.validation.config_types import ValidationConfig

        assert not hasattr(ValidationConfig, "from_dict")
