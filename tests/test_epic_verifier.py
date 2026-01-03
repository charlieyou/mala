"""Unit tests for epic verification functionality.

Tests the EpicVerifier class and ClaudeEpicVerificationModel including:
- Spec path extraction from descriptions
- Remediation issue creation with deduplication
- Human review issue creation
- Lock usage for sequential epic processing
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.orchestration.orchestrator import MalaOrchestrator

from src.domain.epic.scope import ScopedCommits
from src.infra.epic_verifier import (
    ClaudeEpicVerificationModel,
    EpicVerifier,
    _compute_criterion_hash,
    _extract_json_from_code_blocks,
    extract_spec_paths,
)
from src.core.models import EpicVerdict, RetryConfig, UnmetCriterion
from src.infra.tools.command_runner import CommandResult


# ============================================================================
# Test fixtures
# ============================================================================


@pytest.fixture
def mock_beads() -> MagicMock:
    """Create a mock BeadsClient."""
    beads = MagicMock()
    beads.get_issue_description_async = AsyncMock(return_value="Epic description")
    beads.get_epic_children_async = AsyncMock(return_value={"child-1", "child-2"})
    beads.get_epic_blockers_async = AsyncMock(return_value=set())
    beads.close_async = AsyncMock(return_value=True)
    beads.find_issue_by_tag_async = AsyncMock(return_value=None)
    beads.create_issue_async = AsyncMock(return_value="issue-123")
    return beads


@pytest.fixture
def mock_model() -> MagicMock:
    """Create a mock EpicVerificationModel."""
    model = MagicMock()
    model.verify = AsyncMock(
        return_value=EpicVerdict(
            passed=True,
            unmet_criteria=[],
            confidence=0.9,
            reasoning="All criteria met",
        )
    )
    return model


@pytest.fixture
def verifier(
    tmp_path: Path, mock_beads: MagicMock, mock_model: MagicMock
) -> EpicVerifier:
    """Create an EpicVerifier with mock dependencies."""
    return EpicVerifier(
        beads=mock_beads,
        model=mock_model,
        repo_path=tmp_path,
    )


def _stub_commit_helpers(verifier: EpicVerifier, sha: str = "abc123") -> None:
    """Stub scope_analyzer to avoid hitting git in tests."""
    mock_scope_analyzer = MagicMock()
    mock_scope_analyzer.compute_scoped_commits = AsyncMock(
        return_value=ScopedCommits(
            commit_shas=[sha],
            commit_range=sha,
            commit_summary=f"- {sha} summary",
        )
    )
    verifier.scope_analyzer = mock_scope_analyzer


# ============================================================================
# Test extract_spec_paths
# ============================================================================


class TestExtractSpecPaths:
    """Tests for spec path extraction from text."""

    def test_extracts_see_pattern(self) -> None:
        """Should extract 'See specs/...' pattern."""
        text = "Implements auth. See specs/auth/login.md for details."
        paths = extract_spec_paths(text)
        assert paths == ["specs/auth/login.md"]

    def test_extracts_spec_colon_pattern(self) -> None:
        """Should extract 'Spec: specs/...' pattern."""
        text = "Spec: specs/api/v2.md"
        paths = extract_spec_paths(text)
        assert paths == ["specs/api/v2.md"]

    def test_extracts_bracket_pattern(self) -> None:
        """Should extract '[specs/...]' pattern."""
        text = "Reference: [specs/design.md]"
        paths = extract_spec_paths(text)
        assert paths == ["specs/design.md"]

    def test_extracts_bare_pattern(self) -> None:
        """Should extract bare 'specs/...' pattern."""
        text = "Details in specs/overview.md documentation."
        paths = extract_spec_paths(text)
        assert paths == ["specs/overview.md"]

    def test_extracts_nested_directory(self) -> None:
        """Should handle deeply nested spec paths."""
        text = "See specs/auth/v2/oauth/login.md"
        paths = extract_spec_paths(text)
        assert paths == ["specs/auth/v2/oauth/login.md"]

    def test_handles_case_insensitive_extension(self) -> None:
        """Should handle .MD extension."""
        text = "See specs/README.MD for more."
        paths = extract_spec_paths(text)
        assert paths == ["specs/README.MD"]

    def test_deduplicates_paths(self) -> None:
        """Should deduplicate multiple references to same spec."""
        text = "See specs/foo.md. Also specs/foo.md again."
        paths = extract_spec_paths(text)
        assert paths == ["specs/foo.md"]

    def test_returns_empty_for_no_specs(self) -> None:
        """Should return empty list when no specs found."""
        text = "No spec references here."
        paths = extract_spec_paths(text)
        assert paths == []

    def test_extracts_multiple_specs(self) -> None:
        """Should extract multiple different spec paths."""
        text = "See specs/a.md and specs/b.md for details."
        paths = extract_spec_paths(text)
        assert "specs/a.md" in paths
        assert "specs/b.md" in paths


# ============================================================================
# Test remediation issue creation
# ============================================================================


class TestRemediationIssueCreation:
    """Tests for remediation issue creation and deduplication."""

    @pytest.mark.asyncio
    async def test_creates_issue_for_unmet_criterion(
        self, verifier: EpicVerifier, mock_beads: MagicMock
    ) -> None:
        """Should create an issue for each unmet criterion."""
        # Mock beads to return None (no existing issue) and then create new one
        mock_beads.find_issue_by_tag_async = AsyncMock(return_value=None)
        mock_beads.create_issue_async = AsyncMock(return_value="remediation-1")

        verdict = EpicVerdict(
            passed=False,
            unmet_criteria=[
                UnmetCriterion(
                    criterion="Must have error handling",
                    evidence="No try/catch found",
                    priority=1,
                    criterion_hash=_compute_criterion_hash("Must have error handling"),
                )
            ],
            confidence=0.8,
            reasoning="Missing error handling",
        )

        blocking_ids, informational_ids = await verifier.create_remediation_issues(
            "epic-1", verdict
        )
        assert len(blocking_ids) == 1
        assert blocking_ids[0] == "remediation-1"
        assert len(informational_ids) == 0
        # Verify create was called with parent_id (P1 is blocking)
        mock_beads.create_issue_async.assert_called_once()
        call_kwargs = mock_beads.create_issue_async.call_args[1]
        assert call_kwargs["parent_id"] == "epic-1"

    @pytest.mark.asyncio
    async def test_deduplicates_by_tag(
        self, verifier: EpicVerifier, mock_beads: MagicMock
    ) -> None:
        """Should reuse existing issue with matching dedup tag."""
        # Mock beads to return existing issue
        mock_beads.find_issue_by_tag_async = AsyncMock(return_value="existing-issue")

        verdict = EpicVerdict(
            passed=False,
            unmet_criteria=[
                UnmetCriterion(
                    criterion="Criterion text",
                    evidence="Evidence",
                    priority=1,
                    criterion_hash=_compute_criterion_hash("Criterion text"),
                )
            ],
            confidence=0.8,
            reasoning="Test",
        )

        blocking_ids, informational_ids = await verifier.create_remediation_issues(
            "epic-1", verdict
        )
        assert blocking_ids == ["existing-issue"]
        assert informational_ids == []

    @pytest.mark.asyncio
    async def test_remediation_issue_format(
        self, verifier: EpicVerifier, mock_beads: MagicMock
    ) -> None:
        """Should create issue with correct title/body/tags format."""
        # Mock beads methods
        mock_beads.find_issue_by_tag_async = AsyncMock(return_value=None)
        mock_beads.create_issue_async = AsyncMock(return_value="new-1")

        verdict = EpicVerdict(
            passed=False,
            unmet_criteria=[
                UnmetCriterion(
                    criterion="API must return 400 for bad input",
                    evidence="Returns 500 instead",
                    priority=0,
                    criterion_hash=_compute_criterion_hash(
                        "API must return 400 for bad input"
                    ),
                )
            ],
            confidence=0.8,
            reasoning="Test",
        )

        await verifier.create_remediation_issues("epic-1", verdict)

        # Verify create was called with expected arguments
        mock_beads.create_issue_async.assert_called_once()
        call_kwargs = mock_beads.create_issue_async.call_args[1]
        assert call_kwargs["title"].startswith("[Remediation]")
        assert "epic_remediation:" in call_kwargs["tags"][0]
        assert "auto_generated" in call_kwargs["tags"]
        assert call_kwargs["parent_id"] == "epic-1"

    @pytest.mark.asyncio
    async def test_creates_advisory_issue_for_p2_criterion(
        self, verifier: EpicVerifier, mock_beads: MagicMock
    ) -> None:
        """P2/P3 criteria should create standalone advisory issues, not parented."""
        mock_beads.find_issue_by_tag_async = AsyncMock(return_value=None)
        mock_beads.create_issue_async = AsyncMock(return_value="advisory-1")

        verdict = EpicVerdict(
            passed=True,  # P2/P3 don't block
            unmet_criteria=[
                UnmetCriterion(
                    criterion="Method should be under 60 lines",
                    evidence="Method is 84 lines but works correctly",
                    priority=3,  # P3 - advisory
                    criterion_hash=_compute_criterion_hash(
                        "Method should be under 60 lines"
                    ),
                )
            ],
            confidence=0.9,
            reasoning="All functional criteria met, only style preference remains",
        )

        blocking_ids, informational_ids = await verifier.create_remediation_issues(
            "epic-1", verdict
        )

        # P3 is informational, not blocking
        assert blocking_ids == []
        assert informational_ids == ["advisory-1"]

        # Verify issue created with advisory prefix and no parent
        mock_beads.create_issue_async.assert_called_once()
        call_kwargs = mock_beads.create_issue_async.call_args[1]
        assert call_kwargs["title"].startswith("[Advisory]")
        assert call_kwargs["priority"] == "P3"
        assert call_kwargs["parent_id"] is None  # Not parented to epic

    @pytest.mark.asyncio
    async def test_mixed_blocking_and_advisory_criteria(
        self, verifier: EpicVerifier, mock_beads: MagicMock
    ) -> None:
        """Mixed P1 and P3 criteria should create both blocking and advisory issues."""
        mock_beads.find_issue_by_tag_async = AsyncMock(return_value=None)
        issue_ids = iter(["blocking-1", "advisory-1"])
        mock_beads.create_issue_async = AsyncMock(
            side_effect=lambda **_: next(issue_ids)
        )

        verdict = EpicVerdict(
            passed=False,
            unmet_criteria=[
                UnmetCriterion(
                    criterion="API must return 400",
                    evidence="Returns 500",
                    priority=1,  # P1 - blocking
                    criterion_hash=_compute_criterion_hash("API must return 400"),
                ),
                UnmetCriterion(
                    criterion="Prefer shorter methods",
                    evidence="Method is long",
                    priority=3,  # P3 - advisory
                    criterion_hash=_compute_criterion_hash("Prefer shorter methods"),
                ),
            ],
            confidence=0.85,
            reasoning="Functional issue and style suggestion",
        )

        blocking_ids, informational_ids = await verifier.create_remediation_issues(
            "epic-1", verdict
        )

        assert blocking_ids == ["blocking-1"]
        assert informational_ids == ["advisory-1"]

        # Check both calls
        calls = mock_beads.create_issue_async.call_args_list
        assert len(calls) == 2

        # First call (P1) should be blocking with parent
        assert calls[0][1]["title"].startswith("[Remediation]")
        assert calls[0][1]["parent_id"] == "epic-1"

        # Second call (P3) should be advisory without parent
        assert calls[1][1]["title"].startswith("[Advisory]")
        assert calls[1][1]["parent_id"] is None


# ============================================================================
# Test verify_epic
# ============================================================================


class TestVerifyEpic:
    """Tests for verify_epic method."""

    @pytest.mark.asyncio
    async def test_returns_verdict_for_missing_criteria(
        self, verifier: EpicVerifier, mock_beads: MagicMock
    ) -> None:
        """Should return low-confidence verdict when no criteria found."""
        mock_beads.get_issue_description_async.return_value = None

        verdict = await verifier.verify_epic("epic-1")

        assert verdict.passed is False
        assert verdict.confidence == 0.0
        assert "No acceptance criteria" in verdict.reasoning

    @pytest.mark.asyncio
    async def test_returns_verdict_for_no_children(
        self, verifier: EpicVerifier, mock_beads: MagicMock
    ) -> None:
        """Should return low-confidence verdict when no children found."""
        mock_beads.get_epic_children_async.return_value = set()

        verdict = await verifier.verify_epic("epic-1")

        assert verdict.passed is False
        assert verdict.confidence == 0.0
        assert "No child issues" in verdict.reasoning

    @pytest.mark.asyncio
    async def test_returns_verdict_for_no_commits(self, verifier: EpicVerifier) -> None:
        """Should return low-confidence verdict when no commits found."""

        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            return CommandResult(command=cmd, returncode=0, stdout="")

        verifier._runner.run_async = mock_run_async  # type: ignore[method-assign]

        verdict = await verifier.verify_epic("epic-1")

        assert verdict.passed is False
        assert verdict.confidence == 0.0
        assert "No commits found" in verdict.reasoning

    @pytest.mark.asyncio
    async def test_loads_spec_content_if_found(
        self,
        tmp_path: Path,
        mock_beads: MagicMock,
        mock_model: MagicMock,
    ) -> None:
        """Should load spec content when referenced and file exists."""
        # Create spec file
        spec_dir = tmp_path / "specs"
        spec_dir.mkdir()
        (spec_dir / "auth.md").write_text("# Auth Spec\nDetails here.")

        mock_beads.get_issue_description_async.return_value = (
            "Implement auth. See specs/auth.md"
        )

        verifier = EpicVerifier(
            beads=mock_beads,
            model=mock_model,
            repo_path=tmp_path,
        )

        _stub_commit_helpers(verifier)

        await verifier.verify_epic("epic-1")

        # Verify model was called with spec content
        mock_model.verify.assert_called_once()
        call_args = mock_model.verify.call_args
        assert call_args[0][3] == "# Auth Spec\nDetails here."


# ============================================================================
# Test verify_and_close_eligible
# ============================================================================


class TestVerifyAndCloseEligible:
    """Tests for verify_and_close_eligible method."""

    @pytest.mark.asyncio
    async def test_closes_passed_epics(
        self, verifier: EpicVerifier, mock_beads: MagicMock, mock_model: MagicMock
    ) -> None:
        """Should close epics that pass verification."""

        # Mock eligible epics
        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            if "epic" in cmd and "status" in cmd:
                return CommandResult(
                    command=cmd,
                    returncode=0,
                    stdout='[{"eligible_for_close": true, "epic": {"id": "epic-1"}}]',
                )
            return CommandResult(command=cmd, returncode=0, stdout="")

        verifier._runner.run_async = mock_run_async  # type: ignore[method-assign]
        _stub_commit_helpers(verifier)

        result = await verifier.verify_and_close_eligible()

        assert result.passed_count == 1
        mock_beads.close_async.assert_called_with("epic-1")

    @pytest.mark.asyncio
    async def test_human_override_bypasses_verification(
        self, verifier: EpicVerifier, mock_beads: MagicMock, mock_model: MagicMock
    ) -> None:
        """Should close overridden epics without verification."""

        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            if "epic" in cmd and "status" in cmd:
                return CommandResult(
                    command=cmd,
                    returncode=0,
                    stdout='[{"eligible_for_close": true, "epic": {"id": "epic-1"}}]',
                )
            return CommandResult(command=cmd, returncode=0, stdout="")

        verifier._runner.run_async = mock_run_async  # type: ignore[method-assign]

        result = await verifier.verify_and_close_eligible(
            human_override_epic_ids={"epic-1"}
        )

        assert result.passed_count == 1
        # Model should NOT be called for overridden epics
        mock_model.verify.assert_not_called()
        mock_beads.close_async.assert_called_with("epic-1")

    @pytest.mark.asyncio
    async def test_creates_remediation_for_failed(
        self, verifier: EpicVerifier, mock_beads: MagicMock, mock_model: MagicMock
    ) -> None:
        """Should create remediation issues for failed verification."""
        mock_model.verify.return_value = EpicVerdict(
            passed=False,
            unmet_criteria=[
                UnmetCriterion(
                    criterion="API must return 400 for bad input",
                    evidence="Returns 500 instead",
                    priority=1,
                    criterion_hash=_compute_criterion_hash(
                        "API must return 400 for bad input"
                    ),
                )
            ],
            confidence=0.85,
            reasoning="Incorrect error handling",
        )

        created_issues: list[str] = []

        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            if "epic" in cmd and "status" in cmd:
                return CommandResult(
                    command=cmd,
                    returncode=0,
                    stdout='[{"eligible_for_close": true, "epic": {"id": "epic-1"}}]',
                )
            if "show" in cmd and "epic-1" in cmd:
                return CommandResult(
                    command=cmd,
                    returncode=0,
                    stdout='{"priority": "P2"}',
                )
            if "list" in cmd and "--label" in cmd:
                return CommandResult(command=cmd, returncode=0, stdout="[]")
            if cmd[:2] == ["bd", "create"]:
                created_issues.append("rem-1")
                return CommandResult(
                    command=cmd,
                    returncode=0,
                    stdout="Created issue: rem-1",
                )
            if "dep" in cmd:
                return CommandResult(command=cmd, returncode=0, stdout="")
            return CommandResult(command=cmd, returncode=0, stdout="")

        verifier._runner.run_async = mock_run_async  # type: ignore[method-assign]
        _stub_commit_helpers(verifier)

        result = await verifier.verify_and_close_eligible()

        assert result.failed_count == 1
        assert len(result.remediation_issues_created) == 1
        # Epic should NOT be closed
        mock_beads.close_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_low_confidence_still_passes_if_verdict_passed(
        self, verifier: EpicVerifier, mock_beads: MagicMock, mock_model: MagicMock
    ) -> None:
        """Low confidence verdict with passed=True should still close epic."""
        mock_model.verify.return_value = EpicVerdict(
            passed=True,
            unmet_criteria=[],
            confidence=0.3,  # Low confidence no longer triggers special handling
            reasoning="Uncertain",
        )

        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            if "epic" in cmd and "status" in cmd:
                return CommandResult(
                    command=cmd,
                    returncode=0,
                    stdout='[{"eligible_for_close": true, "epic": {"id": "epic-1"}}]',
                )
            if "dep" in cmd:
                return CommandResult(command=cmd, returncode=0, stdout="")
            return CommandResult(command=cmd, returncode=0, stdout="")

        verifier._runner.run_async = mock_run_async  # type: ignore[method-assign]
        _stub_commit_helpers(verifier)

        result = await verifier.verify_and_close_eligible()

        # Low confidence no longer triggers human review - just passes/closes
        assert result.human_review_count == 0
        assert result.passed_count == 1
        mock_beads.close_async.assert_called_with("epic-1")


# ============================================================================
# Test verify_and_close_epic (single epic verification)
# ============================================================================


class TestVerifyAndCloseEpic:
    """Tests for verify_and_close_epic method (single epic verification)."""

    @pytest.mark.asyncio
    async def test_verifies_and_closes_eligible_epic(
        self, verifier: EpicVerifier, mock_beads: MagicMock, mock_model: MagicMock
    ) -> None:
        """Should verify and close an eligible epic."""

        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            if "epic" in cmd and "status" in cmd:
                return CommandResult(
                    command=cmd,
                    returncode=0,
                    stdout='[{"eligible_for_close": true, "epic": {"id": "epic-1"}}]',
                )
            return CommandResult(command=cmd, returncode=0, stdout="")

        verifier._runner.run_async = mock_run_async  # type: ignore[method-assign]
        _stub_commit_helpers(verifier)

        result = await verifier.verify_and_close_epic("epic-1")

        assert result.verified_count == 1
        assert result.passed_count == 1
        mock_beads.close_async.assert_called_with("epic-1")

    @pytest.mark.asyncio
    async def test_skips_non_eligible_epic(
        self, verifier: EpicVerifier, mock_beads: MagicMock, mock_model: MagicMock
    ) -> None:
        """Should not verify an epic that is not eligible."""

        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            if "epic" in cmd and "status" in cmd:
                # Return empty list - no eligible epics
                return CommandResult(
                    command=cmd,
                    returncode=0,
                    stdout="[]",
                )
            return CommandResult(command=cmd, returncode=0, stdout="")

        verifier._runner.run_async = mock_run_async  # type: ignore[method-assign]

        result = await verifier.verify_and_close_epic("epic-1")

        # Should not verify or close
        assert result.verified_count == 0
        assert result.passed_count == 0
        mock_model.verify.assert_not_called()
        mock_beads.close_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_human_override_bypasses_verification(
        self, verifier: EpicVerifier, mock_beads: MagicMock, mock_model: MagicMock
    ) -> None:
        """Should close with human override without verification."""

        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            if "epic" in cmd and "status" in cmd:
                return CommandResult(
                    command=cmd,
                    returncode=0,
                    stdout='[{"eligible_for_close": true, "epic": {"id": "epic-1"}}]',
                )
            return CommandResult(command=cmd, returncode=0, stdout="")

        verifier._runner.run_async = mock_run_async  # type: ignore[method-assign]

        result = await verifier.verify_and_close_epic("epic-1", human_override=True)

        assert result.passed_count == 1
        mock_model.verify.assert_not_called()
        mock_beads.close_async.assert_called_with("epic-1")

    @pytest.mark.asyncio
    async def test_creates_remediation_for_failed(
        self, verifier: EpicVerifier, mock_beads: MagicMock, mock_model: MagicMock
    ) -> None:
        """Should create remediation issues for failed verification."""
        mock_model.verify.return_value = EpicVerdict(
            passed=False,
            unmet_criteria=[
                UnmetCriterion(
                    criterion="API must return 400 for bad input",
                    evidence="Returns 500 instead",
                    priority=1,
                    criterion_hash=_compute_criterion_hash(
                        "API must return 400 for bad input"
                    ),
                )
            ],
            confidence=0.85,
            reasoning="Incorrect error handling",
        )

        created_issues: list[str] = []

        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            if "epic" in cmd and "status" in cmd:
                return CommandResult(
                    command=cmd,
                    returncode=0,
                    stdout='[{"eligible_for_close": true, "epic": {"id": "epic-1"}}]',
                )
            if "show" in cmd and "epic-1" in cmd:
                return CommandResult(
                    command=cmd,
                    returncode=0,
                    stdout='{"priority": "P2"}',
                )
            if "list" in cmd and "--label" in cmd:
                return CommandResult(command=cmd, returncode=0, stdout="[]")
            if cmd[:2] == ["bd", "create"]:
                created_issues.append("rem-1")
                return CommandResult(
                    command=cmd,
                    returncode=0,
                    stdout="Created issue: rem-1",
                )
            if "dep" in cmd:
                return CommandResult(command=cmd, returncode=0, stdout="")
            return CommandResult(command=cmd, returncode=0, stdout="")

        verifier._runner.run_async = mock_run_async  # type: ignore[method-assign]
        _stub_commit_helpers(verifier)

        result = await verifier.verify_and_close_epic("epic-1")

        assert result.failed_count == 1
        assert len(result.remediation_issues_created) == 1
        # Epic should NOT be closed when verification fails
        mock_beads.close_async.assert_not_called()


# ============================================================================
# Test ClaudeEpicVerificationModel
# ============================================================================


class TestClaudeEpicVerificationModel:
    """Tests for Claude-based verification model."""

    def test_parses_valid_json_response(self) -> None:
        """Should parse valid JSON verdict."""
        model = ClaudeEpicVerificationModel()
        response = json.dumps(
            {
                "passed": True,
                "confidence": 0.92,
                "reasoning": "All criteria met",
                "unmet_criteria": [],
            }
        )
        verdict = model._parse_verdict(response)
        assert verdict.passed is True
        assert verdict.confidence == 0.92

    def test_parses_json_in_code_block(self) -> None:
        """Should extract JSON from markdown code block."""
        model = ClaudeEpicVerificationModel()
        response = """Here is my analysis:
```json
{
    "passed": false,
    "confidence": 0.85,
    "reasoning": "Missing error handling",
    "unmet_criteria": [
        {
            "criterion": "Must handle errors",
            "evidence": "No try/catch",
            "severity": "major"
        }
    ]
}
```
"""
        verdict = model._parse_verdict(response)
        assert verdict.passed is False
        assert len(verdict.unmet_criteria) == 1
        assert verdict.unmet_criteria[0].criterion == "Must handle errors"

    def test_returns_low_confidence_on_parse_failure(self) -> None:
        """Should return low-confidence failure when parsing fails."""
        model = ClaudeEpicVerificationModel()
        verdict = model._parse_verdict("Invalid response with no JSON")
        assert verdict.passed is False
        assert verdict.confidence == 0.0
        assert "Failed to parse" in verdict.reasoning

    def test_computes_criterion_hash(self) -> None:
        """Should compute consistent hash for criterion."""
        model = ClaudeEpicVerificationModel()
        response = json.dumps(
            {
                "passed": False,
                "confidence": 0.8,
                "reasoning": "Test",
                "unmet_criteria": [
                    {
                        "criterion": "Test criterion",
                        "evidence": "Evidence",
                        "severity": "minor",
                    }
                ],
            }
        )
        verdict = model._parse_verdict(response)
        assert verdict.unmet_criteria[0].criterion_hash == _compute_criterion_hash(
            "Test criterion"
        )

    def test_parses_json_with_multiple_code_blocks(self) -> None:
        """Should extract JSON from first code block when multiple blocks exist."""
        model = ClaudeEpicVerificationModel()
        response = """Here is my analysis:

First, let me show you the code that was reviewed:
```python
def example():
    return 42
```

Now here is the verdict:
```json
{
    "passed": true,
    "confidence": 0.88,
    "reasoning": "Tests exist",
    "unmet_criteria": []
}
```

And here's another block for reference:
```
some other content
```
"""
        verdict = model._parse_verdict(response)
        assert verdict.passed is True
        assert verdict.confidence == 0.88

    def test_parses_json_when_json_block_is_not_first(self) -> None:
        """Should find JSON block even if other code blocks come first."""
        model = ClaudeEpicVerificationModel()
        response = """Analysis:

Code snippet:
```bash
echo "hello world"
```

Result:
```json
{
    "passed": false,
    "confidence": 0.75,
    "reasoning": "Missing tests",
    "unmet_criteria": [
        {"criterion": "Test coverage", "evidence": "No tests", "severity": "major"}
    ]
}
```
"""
        verdict = model._parse_verdict(response)
        assert verdict.passed is False
        assert verdict.confidence == 0.75
        assert len(verdict.unmet_criteria) == 1


# ============================================================================
# Test _extract_json_from_code_blocks helper
# ============================================================================


class TestExtractJsonFromCodeBlocks:
    """Tests for the JSON extraction helper function."""

    def test_extracts_from_single_json_block(self) -> None:
        """Should extract JSON from a single json code block."""
        text = '```json\n{"key": "value"}\n```'
        result = _extract_json_from_code_blocks(text)
        assert result == '{"key": "value"}'

    def test_extracts_from_plain_code_block(self) -> None:
        """Should extract JSON from a code block without language specifier."""
        text = '```\n{"key": "value"}\n```'
        result = _extract_json_from_code_blocks(text)
        assert result == '{"key": "value"}'

    def test_returns_none_when_no_code_blocks(self) -> None:
        """Should return None when there are no code blocks."""
        text = "Just plain text without any code blocks."
        result = _extract_json_from_code_blocks(text)
        assert result is None

    def test_returns_none_when_no_json_in_code_blocks(self) -> None:
        """Should return None when code blocks don't contain JSON."""
        text = """```python
def foo():
    pass
```"""
        result = _extract_json_from_code_blocks(text)
        assert result is None

    def test_extracts_first_json_block_from_multiple(self) -> None:
        """Should return first JSON code block when multiple exist."""
        text = """```python
code = "not json"
```

```json
{"first": true}
```

```json
{"second": true}
```"""
        result = _extract_json_from_code_blocks(text)
        assert result == '{"first": true}'

    def test_handles_code_block_with_backticks_spanning_multiple_lines(self) -> None:
        """Should handle multiline JSON in code blocks."""
        text = """```json
{
    "passed": true,
    "confidence": 0.9,
    "reasoning": "All good",
    "unmet_criteria": []
}
```"""
        result = _extract_json_from_code_blocks(text)
        assert result is not None
        import json

        data = json.loads(result)
        assert data["passed"] is True
        assert data["confidence"] == 0.9

    def test_skips_non_json_blocks_to_find_json(self) -> None:
        """Should skip non-JSON blocks to find one that starts with '{'."""
        text = """Here is the diff:
```diff
- old line
+ new line
```

And the result:
```json
{"result": "success"}
```"""
        result = _extract_json_from_code_blocks(text)
        assert result == '{"result": "success"}'

    def test_handles_empty_code_block(self) -> None:
        """Should handle empty code blocks without crashing."""
        text = """```
```

```json
{"valid": true}
```"""
        result = _extract_json_from_code_blocks(text)
        assert result == '{"valid": true}'


# ============================================================================
# Test lock usage
# ============================================================================


class TestLockUsage:
    """Tests for sequential epic verification with locking."""

    @pytest.mark.asyncio
    async def test_acquires_lock_before_verification(
        self, verifier: EpicVerifier, mock_beads: MagicMock, mock_model: MagicMock
    ) -> None:
        """Should acquire lock before verifying an epic."""
        # Set up lock_manager
        verifier.lock_manager = MagicMock()

        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            if "epic" in cmd and "status" in cmd:
                return CommandResult(
                    command=cmd,
                    returncode=0,
                    stdout='[{"eligible_for_close": true, "epic": {"id": "epic-1"}}]',
                )
            return CommandResult(command=cmd, returncode=0, stdout="")

        verifier._runner.run_async = mock_run_async  # type: ignore[method-assign]
        _stub_commit_helpers(verifier)

        # Mock the locking module
        import os

        expected_agent_id = f"epic_verifier_{os.getpid()}"

        with patch("src.infra.tools.locking.wait_for_lock") as mock_wait:
            mock_wait.return_value = True

            with patch("src.infra.tools.locking.get_lock_holder") as mock_holder:
                mock_holder.return_value = expected_agent_id

                with patch("src.infra.tools.locking.lock_path") as mock_path:
                    mock_lock_path = MagicMock()
                    mock_lock_path.exists.return_value = True
                    mock_path.return_value = mock_lock_path

                    await verifier.verify_and_close_eligible()

                    # Lock should have been attempted since lock_manager is set
                    # (the actual call happens inside a try/except ImportError)
                    # Since we patched the module, the import succeeds
                    mock_wait.assert_called_once()
                    call_args = mock_wait.call_args[0]
                    assert call_args[0] == "epic_verify:epic-1"
                    assert call_args[1] == expected_agent_id
                    assert call_args[2] == str(verifier.repo_path)  # repo_namespace


# ============================================================================
# Test priority adjustment
# ============================================================================


# ============================================================================
# Test model error handling
# ============================================================================


class TestModelErrorHandling:
    """Tests for model timeout/error handling."""

    @pytest.mark.asyncio
    async def test_model_timeout_returns_low_confidence_verdict(
        self, verifier: EpicVerifier, mock_beads: MagicMock, mock_model: MagicMock
    ) -> None:
        """Should return low-confidence verdict when model times out."""
        # Make model raise timeout error
        mock_model.verify.side_effect = TimeoutError("Model call timed out")

        _stub_commit_helpers(verifier)

        verdict = await verifier.verify_epic("epic-1")

        assert verdict.passed is False
        assert verdict.confidence == 0.0
        assert "Model verification failed" in verdict.reasoning

    @pytest.mark.asyncio
    async def test_model_error_returns_low_confidence_verdict(
        self, verifier: EpicVerifier, mock_beads: MagicMock, mock_model: MagicMock
    ) -> None:
        """Should return low-confidence verdict when model raises error."""
        # Make model raise generic error
        mock_model.verify.side_effect = RuntimeError("API connection failed")

        _stub_commit_helpers(verifier)

        verdict = await verifier.verify_epic("epic-1")

        assert verdict.passed is False
        assert verdict.confidence == 0.0
        assert "API connection failed" in verdict.reasoning

    @pytest.mark.asyncio
    async def test_model_error_triggers_failure_in_eligible_flow(
        self, verifier: EpicVerifier, mock_beads: MagicMock, mock_model: MagicMock
    ) -> None:
        """Model errors should trigger failure (not human review) in verify_and_close_eligible."""
        mock_model.verify.side_effect = TimeoutError("timeout")

        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            if "epic" in cmd and "status" in cmd:
                return CommandResult(
                    command=cmd,
                    returncode=0,
                    stdout='[{"eligible_for_close": true, "epic": {"id": "epic-1"}}]',
                )
            if "dep" in cmd:
                return CommandResult(command=cmd, returncode=0, stdout="")
            return CommandResult(command=cmd, returncode=0, stdout="")

        verifier._runner.run_async = mock_run_async  # type: ignore[method-assign]
        _stub_commit_helpers(verifier)

        result = await verifier.verify_and_close_eligible()

        # Model error results in passed=False verdict, which triggers failure
        assert result.human_review_count == 0
        assert result.failed_count == 1
        mock_beads.close_async.assert_not_called()


# ============================================================================
# Integration Tests for EpicVerifier Orchestrator Wiring
# ============================================================================


@pytest.mark.integration
class TestEpicVerifierOrchestratorIntegration:
    """Integration tests for EpicVerifier wiring in MalaOrchestrator."""

    @pytest.mark.asyncio
    async def test_orchestrator_uses_epic_verifier_for_closure(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """Orchestrator should use EpicVerifier instead of close_eligible_epics_async."""
        from src.infra.io.config import MalaConfig

        # Create orchestrator with mock beads
        config = MalaConfig()
        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            config=config,
        )

        # Verify EpicVerifier was instantiated
        assert orchestrator.epic_verifier is not None

    @pytest.mark.asyncio
    async def test_orchestrator_respects_epic_override_ids(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """Orchestrator should pass epic_override_ids to verify_and_close_eligible."""
        from src.infra.io.config import MalaConfig

        override_ids = {"epic-1", "epic-2"}
        config = MalaConfig()
        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            config=config,
            epic_override_ids=override_ids,
        )

        # Verify override IDs were stored
        assert orchestrator.epic_override_ids == override_ids

    @pytest.mark.asyncio
    async def test_verify_and_close_with_override_bypasses_verification(
        self,
    ) -> None:
        """Epics in override set should close without model verification."""
        mock_beads = MagicMock()
        mock_beads.get_issue_description_async = AsyncMock(return_value="Epic desc")
        mock_beads.get_epic_children_async = AsyncMock(return_value={"child-1"})
        mock_beads.close_async = AsyncMock(return_value=True)

        mock_model = MagicMock()
        mock_model.verify = AsyncMock()  # Should NOT be called

        verifier = EpicVerifier(
            beads=mock_beads,
            model=mock_model,
            repo_path=Path("/tmp"),
        )

        # Mock _get_eligible_epics to return epic-1
        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            if "epic" in cmd and "status" in cmd:
                return CommandResult(
                    command=cmd,
                    returncode=0,
                    stdout='[{"eligible_for_close": true, "epic": {"id": "epic-1"}}]',
                )
            return CommandResult(command=cmd, returncode=0, stdout="")

        verifier._runner.run_async = mock_run_async  # type: ignore[method-assign]

        result = await verifier.verify_and_close_eligible(
            human_override_epic_ids={"epic-1"}
        )

        # Epic should be closed without calling model
        mock_model.verify.assert_not_called()
        mock_beads.close_async.assert_called_with("epic-1")
        assert result.passed_count == 1
        assert (
            result.verdicts["epic-1"].reasoning
            == "Human override - bypassed verification"
        )

    @pytest.mark.asyncio
    async def test_verify_and_close_creates_remediation_on_fail(self) -> None:
        """Failed verification should create remediation issues, not close epic."""
        mock_beads = MagicMock()
        mock_beads.get_issue_description_async = AsyncMock(return_value="Epic desc")
        mock_beads.get_epic_children_async = AsyncMock(return_value={"child-1"})
        mock_beads.get_epic_blockers_async = AsyncMock(return_value=set())
        mock_beads.close_async = AsyncMock(return_value=True)
        mock_beads.find_issue_by_tag_async = AsyncMock(return_value=None)
        mock_beads.create_issue_async = AsyncMock(return_value="rem-1")

        mock_model = MagicMock()
        mock_model.verify = AsyncMock(
            return_value=EpicVerdict(
                passed=False,
                unmet_criteria=[
                    UnmetCriterion(
                        criterion="API must return 400 for bad input",
                        evidence="Returns 500 instead",
                        priority=1,
                        criterion_hash=_compute_criterion_hash(
                            "API must return 400 for bad input"
                        ),
                    )
                ],
                confidence=0.9,
                reasoning="Incorrect error handling",
            )
        )

        verifier = EpicVerifier(
            beads=mock_beads,
            model=mock_model,
            repo_path=Path("/tmp"),
        )

        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            if "epic" in cmd and "status" in cmd:
                return CommandResult(
                    command=cmd,
                    returncode=0,
                    stdout='[{"eligible_for_close": true, "epic": {"id": "epic-1"}}]',
                )
            if "show" in cmd and "epic-1" in cmd:
                return CommandResult(
                    command=cmd,
                    returncode=0,
                    stdout='{"priority": "P2"}',
                )
            if "dep" in cmd:
                return CommandResult(command=cmd, returncode=0, stdout="")
            return CommandResult(command=cmd, returncode=0, stdout="")

        verifier._runner.run_async = mock_run_async  # type: ignore[method-assign]
        _stub_commit_helpers(verifier)

        result = await verifier.verify_and_close_eligible()

        # Epic should NOT be closed (verification failed)
        mock_beads.close_async.assert_not_called()
        assert result.failed_count == 1
        assert result.passed_count == 0
        assert len(result.remediation_issues_created) == 1

    @pytest.mark.asyncio
    async def test_verify_passes_for_met_criteria(self) -> None:
        """Verification should pass and close epic when all criteria met."""
        mock_beads = MagicMock()
        mock_beads.get_issue_description_async = AsyncMock(return_value="Epic desc")
        mock_beads.get_epic_children_async = AsyncMock(return_value={"child-1"})
        mock_beads.get_epic_blockers_async = AsyncMock(return_value=set())
        mock_beads.close_async = AsyncMock(return_value=True)

        mock_model = MagicMock()
        mock_model.verify = AsyncMock(
            return_value=EpicVerdict(
                passed=True,
                unmet_criteria=[],
                confidence=0.95,
                reasoning="All criteria met",
            )
        )

        verifier = EpicVerifier(
            beads=mock_beads,
            model=mock_model,
            repo_path=Path("/tmp"),
        )

        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            if "epic" in cmd and "status" in cmd:
                return CommandResult(
                    command=cmd,
                    returncode=0,
                    stdout='[{"eligible_for_close": true, "epic": {"id": "epic-1"}}]',
                )
            return CommandResult(command=cmd, returncode=0, stdout="")

        verifier._runner.run_async = mock_run_async  # type: ignore[method-assign]
        _stub_commit_helpers(verifier)

        result = await verifier.verify_and_close_eligible()

        # Epic should be closed (verification passed)
        mock_beads.close_async.assert_called_with("epic-1")
        assert result.passed_count == 1
        assert result.failed_count == 0


# ============================================================================
# Test SDK-backed verify behavior (without real SDK calls)
# ============================================================================


class TestVerifyWithSDK:
    """Tests for verify() using patched SDK responses."""

    @pytest.mark.asyncio
    async def test_verify_parses_sdk_response(self) -> None:
        model = ClaudeEpicVerificationModel()
        model._prompt_template = (
            "criteria: {epic_criteria}, spec: {spec_content}, "
            "range: {commit_range}, list: {commit_list}"
        )

        async def fake_verify(_prompt: str) -> str:
            return (
                '{"passed": true, "confidence": 0.9, "reasoning": "ok", '
                '"unmet_criteria": []}'
            )

        model._verify_with_agent_sdk = fake_verify  # type: ignore[method-assign]

        verdict = await model.verify("criteria", "range", "- abc123", None)

        assert verdict.passed is True
        assert verdict.confidence == 0.9

    @pytest.mark.asyncio
    async def test_verify_handles_non_json_response(self) -> None:
        model = ClaudeEpicVerificationModel()
        model._prompt_template = (
            "criteria: {epic_criteria}, spec: {spec_content}, "
            "range: {commit_range}, list: {commit_list}"
        )

        async def fake_verify(_prompt: str) -> str:
            return "not json"

        model._verify_with_agent_sdk = fake_verify  # type: ignore[method-assign]

        verdict = await model.verify("criteria", "range", "- abc123", None)

        assert verdict.passed is False
        assert verdict.confidence == 0.0
        assert "Failed to parse" in verdict.reasoning


class TestRetryConfig:
    """Tests for retry_config storage (used by callers for consistency)."""

    def test_retry_config_is_stored(self) -> None:
        """Should store retry_config when provided."""
        custom_config = RetryConfig(
            max_retries=5,
            initial_delay_ms=500,
            backoff_multiplier=3.0,
            max_delay_ms=60000,
        )
        model = ClaudeEpicVerificationModel(retry_config=custom_config)

        assert model.retry_config.max_retries == 5
        assert model.retry_config.initial_delay_ms == 500
        assert model.retry_config.backoff_multiplier == 3.0
        assert model.retry_config.max_delay_ms == 60000

    def test_retry_config_defaults(self) -> None:
        """Should use default RetryConfig when not provided."""
        model = ClaudeEpicVerificationModel()

        # Check defaults from RetryConfig
        assert model.retry_config.max_retries == 2
        assert model.retry_config.initial_delay_ms == 1000
        assert model.retry_config.backoff_multiplier == 2.0
        assert model.retry_config.max_delay_ms == 30000
