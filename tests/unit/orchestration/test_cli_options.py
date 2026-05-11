"""Unit tests for the orchestration cli_options helpers.

Exercise the validators and scope parser without Typer. The CLI converts the
``ValueError`` these helpers raise into ``typer.BadParameter`` / ``typer.Exit``;
the framework-free helpers themselves only need to assert input validation
and parsing branches.
"""

from __future__ import annotations

import pytest

from src.orchestration.cli_options import (
    ScopeConfig,
    parse_scope,
    validate_amp_mode_option,
    validate_codex_approval_policy_option,
    validate_codex_effort_option,
    validate_codex_model_option,
    validate_codex_sandbox_option,
    validate_coder_option,
    validate_effort_option,
)


pytestmark = pytest.mark.unit


class TestParseScope:
    """Branch coverage for ``parse_scope``."""

    def test_all(self) -> None:
        result = parse_scope("all")
        assert result == ScopeConfig(scope_type="all")

    def test_all_with_whitespace(self) -> None:
        assert parse_scope("  all  ").scope_type == "all"

    def test_orphans(self) -> None:
        result = parse_scope("orphans")
        assert result.scope_type == "orphans"
        assert result.ids is None
        assert result.epic_id is None

    def test_epic(self) -> None:
        result = parse_scope("epic:E-123")
        assert result.scope_type == "epic"
        assert result.epic_id == "E-123"
        assert result.ids is None

    def test_epic_with_whitespace(self) -> None:
        result = parse_scope("epic:  E-123  ")
        assert result.scope_type == "epic"
        assert result.epic_id == "E-123"

    def test_epic_empty_id_raises(self) -> None:
        with pytest.raises(ValueError, match="'epic:' requires an epic ID"):
            parse_scope("epic:")

    def test_epic_whitespace_only_id_raises(self) -> None:
        with pytest.raises(ValueError, match="'epic:' requires an epic ID"):
            parse_scope("epic:   ")

    def test_ids_single(self) -> None:
        result = parse_scope("ids:T-1")
        assert result.scope_type == "ids"
        assert result.ids == ["T-1"]
        assert result.duplicate_ids == ()

    def test_ids_multiple_preserves_order(self) -> None:
        result = parse_scope("ids:T-1,T-2,T-3")
        assert result.ids == ["T-1", "T-2", "T-3"]
        assert result.duplicate_ids == ()

    def test_ids_with_whitespace(self) -> None:
        result = parse_scope("ids:  T-1  ,  T-2  ,  T-3  ")
        assert result.ids == ["T-1", "T-2", "T-3"]

    def test_ids_deduplicates_and_records_duplicates(self) -> None:
        """Dedup is the helper's job; the CLI displays ``duplicate_ids``."""
        result = parse_scope("ids:T-1,T-1,T-2")
        assert result.ids == ["T-1", "T-2"]
        assert result.duplicate_ids == ("T-1",)

    def test_ids_deduplicates_preserves_first_occurrence(self) -> None:
        result = parse_scope("ids:T-3,T-1,T-2,T-1,T-3")
        assert result.ids == ["T-3", "T-1", "T-2"]
        assert result.duplicate_ids == ("T-1", "T-3")

    def test_ids_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="'ids:' requires at least one ID"):
            parse_scope("ids:")

    def test_ids_whitespace_only_raises(self) -> None:
        with pytest.raises(ValueError, match="'ids:' requires at least one ID"):
            parse_scope("ids:   ")

    def test_ids_empty_between_commas_raises(self) -> None:
        with pytest.raises(ValueError, match="'ids:' requires at least one ID"):
            parse_scope("ids:, , ,")

    def test_invalid_prefix_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid --scope value: 'invalid:xyz'"):
            parse_scope("invalid:xyz")

    def test_unknown_keyword_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid --scope value: 'unknown'"):
            parse_scope("unknown")


class TestValidateCoderOption:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (None, None),
            ("claude", "claude"),
            ("amp", "amp"),
            ("codex", "codex"),
        ],
    )
    def test_accepts_valid_and_none(
        self, value: str | None, expected: str | None
    ) -> None:
        assert validate_coder_option(value) == expected

    def test_rejects_unknown(self) -> None:
        with pytest.raises(ValueError, match="--coder: Invalid coder 'bogus'"):
            validate_coder_option("bogus")


class TestValidateAmpModeOption:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (None, None),
            ("smart", "smart"),
            ("rush", "rush"),
            ("deep", "deep"),
        ],
    )
    def test_accepts_valid_and_none(
        self, value: str | None, expected: str | None
    ) -> None:
        assert validate_amp_mode_option(value) == expected

    def test_rejects_unknown(self) -> None:
        with pytest.raises(ValueError, match="--amp-mode: Invalid amp mode 'wild'"):
            validate_amp_mode_option("wild")


class TestValidateEffortOption:
    @pytest.mark.parametrize("value", ["low", "medium", "high", "xhigh", "max"])
    def test_accepts_valid(self, value: str) -> None:
        assert validate_effort_option(value) == value

    def test_none_passthrough(self) -> None:
        assert validate_effort_option(None) is None

    def test_rejects_unknown(self) -> None:
        with pytest.raises(ValueError, match="--effort: Invalid effort 'turbo'"):
            validate_effort_option("turbo")


class TestValidateCodexModelOption:
    """Codex model parsing is shape-only: any non-empty value passes through."""

    def test_passes_through_non_empty(self) -> None:
        assert validate_codex_model_option("gpt-5.5") == "gpt-5.5"

    def test_strips_whitespace(self) -> None:
        assert validate_codex_model_option("  gpt-5.5  ") == "gpt-5.5"

    def test_empty_and_whitespace_treated_as_none(self) -> None:
        assert validate_codex_model_option(None) is None
        assert validate_codex_model_option("") is None
        assert validate_codex_model_option("   ") is None


class TestValidateCodexEffortOption:
    def test_none_passthrough(self) -> None:
        assert validate_codex_effort_option(None) is None

    def test_rejects_unknown(self) -> None:
        with pytest.raises(ValueError):
            validate_codex_effort_option("nope-not-a-real-effort")


class TestValidateCodexApprovalPolicyOption:
    @pytest.mark.parametrize(
        "value", ["never", "on-request", "on-failure", "untrusted"]
    )
    def test_accepts_valid(self, value: str) -> None:
        assert validate_codex_approval_policy_option(value) == value

    def test_none_passthrough(self) -> None:
        assert validate_codex_approval_policy_option(None) is None

    def test_rejects_unknown(self) -> None:
        with pytest.raises(
            ValueError, match="--codex-approval-policy: Invalid Codex approval_policy"
        ):
            validate_codex_approval_policy_option("always")


class TestValidateCodexSandboxOption:
    @pytest.mark.parametrize(
        "value", ["read-only", "workspace-write", "danger-full-access"]
    )
    def test_accepts_valid(self, value: str) -> None:
        assert validate_codex_sandbox_option(value) == value

    def test_none_passthrough(self) -> None:
        assert validate_codex_sandbox_option(None) is None

    def test_rejects_unknown(self) -> None:
        with pytest.raises(
            ValueError, match="--codex-sandbox: Invalid Codex sandbox 'jailed'"
        ):
            validate_codex_sandbox_option("jailed")
