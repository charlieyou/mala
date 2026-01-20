"""Unit tests for epic verification functionality.

Tests the EpicVerifier class and ClaudeEpicVerificationModel including:
- Spec path extraction from descriptions
- Remediation issue creation with deduplication
- Lock usage for sequential epic processing
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.orchestration.orchestrator import MalaOrchestrator

from src.infra.epic_scope import ScopedCommits
from src.infra.epic_verifier import (
    ClaudeEpicVerificationModel,
    EpicVerifier,
    _compute_criterion_hash,
    _extract_json_from_code_blocks,
)
from src.core.models import EpicVerdict, UnmetCriterion
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
            reasoning="All criteria met",
        )
    )
    return model


@pytest.fixture
def mock_command_runner() -> MagicMock:
    """Create a mock CommandRunner."""
    runner = MagicMock()
    runner.run.return_value = CommandResult(
        command=["mock"], returncode=0, stdout="", stderr=""
    )
    runner.run_async = AsyncMock(
        return_value=CommandResult(command=["mock"], returncode=0, stdout="", stderr="")
    )
    return runner


@pytest.fixture
def verifier(
    tmp_path: Path,
    mock_beads: MagicMock,
    mock_model: MagicMock,
    mock_command_runner: MagicMock,
) -> EpicVerifier:
    """Create an EpicVerifier with mock dependencies."""
    return EpicVerifier(
        beads=mock_beads,
        model=mock_model,
        repo_path=tmp_path,
        command_runner=mock_command_runner,
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
            reasoning="Test",
        )

        await verifier.create_remediation_issues("epic-1", verdict)

        # Verify create was called with expected arguments
        mock_beads.create_issue_async.assert_called_once()
        call_kwargs = mock_beads.create_issue_async.call_args[1]
        assert call_kwargs["title"].startswith("[Remediation]")
        assert call_kwargs["tags"][0].startswith("er:")
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
        mock_beads.add_dependency_async = AsyncMock(return_value=True)

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

    @pytest.mark.asyncio
    async def test_creates_sequential_dependencies_between_remediation_issues(
        self, verifier: EpicVerifier, mock_beads: MagicMock
    ) -> None:
        """Should add sequential dependencies so remediation issues are not picked up in parallel."""
        issue_ids = iter(["issue-1", "issue-2", "issue-3"])
        mock_beads.find_issue_by_tag_async = AsyncMock(return_value=None)
        mock_beads.create_issue_async = AsyncMock(side_effect=lambda **_: next(issue_ids))
        mock_beads.add_dependency_async = AsyncMock(return_value=True)

        verdict = EpicVerdict(
            passed=False,
            unmet_criteria=[
                UnmetCriterion(
                    criterion="First criterion",
                    evidence="Evidence 1",
                    priority=0,  # blocking
                    criterion_hash=_compute_criterion_hash("First criterion"),
                ),
                UnmetCriterion(
                    criterion="Second criterion",
                    evidence="Evidence 2",
                    priority=3,  # informational
                    criterion_hash=_compute_criterion_hash("Second criterion"),
                ),
                UnmetCriterion(
                    criterion="Third criterion",
                    evidence="Evidence 3",
                    priority=1,  # blocking
                    criterion_hash=_compute_criterion_hash("Third criterion"),
                ),
            ],
            reasoning="Mixed blocking and informational issues",
        )

        blocking_ids, informational_ids = await verifier.create_remediation_issues(
            "epic-1", verdict
        )

        assert blocking_ids == ["issue-1", "issue-3"]
        assert informational_ids == ["issue-2"]

        # Verify sequential dependencies across all issues
        dep_calls = mock_beads.add_dependency_async.call_args_list
        assert len(dep_calls) == 2
        assert dep_calls[0].args == ("issue-2", "issue-1")
        assert dep_calls[1].args == ("issue-3", "issue-2")


# ============================================================================
# Test verify_epic
# ============================================================================


class TestVerifyEpic:
    """Tests for verify_epic method."""

    @pytest.mark.asyncio
    async def test_returns_verdict_for_missing_criteria(
        self, verifier: EpicVerifier, mock_beads: MagicMock
    ) -> None:
        """Should return failure verdict when no criteria found."""
        mock_beads.get_issue_description_async.return_value = None

        verdict = await verifier.verify_epic("epic-1")

        assert verdict.passed is False
        assert "No acceptance criteria" in verdict.reasoning

    @pytest.mark.asyncio
    async def test_returns_verdict_for_no_children(
        self, verifier: EpicVerifier, mock_beads: MagicMock
    ) -> None:
        """Should return failure verdict when no children found."""
        mock_beads.get_epic_children_async.return_value = set()

        verdict = await verifier.verify_epic("epic-1")

        assert verdict.passed is False
        assert "No child issues" in verdict.reasoning

    @pytest.mark.asyncio
    async def test_returns_verdict_for_no_commits(self, verifier: EpicVerifier) -> None:
        """Should return failure verdict when no commits found."""

        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            return CommandResult(command=cmd, returncode=0, stdout="")

        verifier._runner.run_async = mock_run_async  # type: ignore[method-assign]

        verdict = await verifier.verify_epic("epic-1")

        assert verdict.passed is False
        assert "No commits found" in verdict.reasoning


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
            if cmd[:2] == ["br", "create"]:
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
    async def test_passing_verdict_closes_epic(
        self, verifier: EpicVerifier, mock_beads: MagicMock, mock_model: MagicMock
    ) -> None:
        """Passing verdict should close epic."""
        mock_model.verify.return_value = EpicVerdict(
            passed=True,
            unmet_criteria=[],
            reasoning="All criteria met",
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
    async def test_ineligibility_reason_shows_open_children(
        self, verifier: EpicVerifier, mock_beads: MagicMock, mock_model: MagicMock
    ) -> None:
        """Should return reason with open children count when epic not eligible."""

        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            if "epic" in cmd and "status" in cmd:
                # Return epic with 3 open children out of 5
                return CommandResult(
                    command=cmd,
                    returncode=0,
                    stdout=json.dumps(
                        [
                            {
                                "epic": {"id": "epic-1"},
                                "eligible_for_close": False,
                                "total_children": 5,
                                "closed_children": 2,
                            }
                        ]
                    ),
                )
            return CommandResult(command=cmd, returncode=0, stdout="")

        verifier._runner.run_async = mock_run_async  # type: ignore[method-assign]

        result = await verifier.verify_and_close_epic("epic-1")

        assert result.verified_count == 0
        assert result.ineligibility_reason == "3 of 5 child issues still open"

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
            if cmd[:2] == ["br", "create"]:
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
                "findings": [],
                "verdict": "PASS",
                "summary": "All criteria met",
            }
        )
        verdict = model._parse_verdict(response)
        assert verdict.passed is True

    def test_parses_json_in_code_block(self) -> None:
        """Should extract JSON from markdown code block."""
        model = ClaudeEpicVerificationModel()
        response = """Here is my analysis:
```json
{
    "findings": [
        {
            "title": "[P1] Must handle errors",
            "body": "No try/catch found in error paths",
            "priority": 1,
            "file_path": null,
            "line_start": null,
            "line_end": null
        }
    ],
    "verdict": "FAIL",
    "summary": "Missing error handling"
}
```
"""
        verdict = model._parse_verdict(response)
        assert verdict.passed is False
        assert len(verdict.unmet_criteria) == 1
        assert verdict.unmet_criteria[0].criterion == "[P1] Must handle errors"

    def test_returns_failure_on_parse_failure(self) -> None:
        """Should return failure verdict when parsing fails."""
        model = ClaudeEpicVerificationModel()
        verdict = model._parse_verdict("Invalid response with no JSON")
        assert verdict.passed is False
        assert "Failed to parse" in verdict.reasoning

    def test_computes_criterion_hash(self) -> None:
        """Should compute consistent hash for criterion."""
        model = ClaudeEpicVerificationModel()
        response = json.dumps(
            {
                "findings": [
                    {
                        "title": "Test criterion",
                        "body": "Evidence",
                        "priority": 2,
                        "file_path": None,
                        "line_start": None,
                        "line_end": None,
                    }
                ],
                "verdict": "NEEDS_WORK",
                "summary": "Test",
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
    "findings": [],
    "verdict": "PASS",
    "summary": "Tests exist"
}
```

And here's another block for reference:
```
some other content
```
"""
        verdict = model._parse_verdict(response)
        assert verdict.passed is True

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
    "findings": [
        {"title": "Test coverage", "body": "No tests", "priority": 1, "file_path": null, "line_start": null, "line_end": null}
    ],
    "verdict": "FAIL",
    "summary": "Missing tests"
}
```
"""
        verdict = model._parse_verdict(response)
        assert verdict.passed is False
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
    "findings": [],
    "verdict": "PASS",
    "summary": "All good"
}
```"""
        result = _extract_json_from_code_blocks(text)
        assert result is not None
        import json

        data = json.loads(result)
        assert data["verdict"] == "PASS"
        assert data["findings"] == []

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
        import os

        expected_agent_id = f"epic_verifier_{os.getpid()}"

        # Set up lock_manager with mock methods
        mock_lock_manager = MagicMock()
        mock_lock_manager.wait_for_lock.return_value = True
        verifier.lock_manager = mock_lock_manager

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

        await verifier.verify_and_close_eligible()

        # Lock should have been acquired via lock_manager
        mock_lock_manager.wait_for_lock.assert_called_once()
        call_args = mock_lock_manager.wait_for_lock.call_args[0]
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
        assert result.failed_count == 1
        mock_beads.close_async.assert_not_called()


# ============================================================================
# Integration Tests for EpicVerifier Orchestrator Wiring
# ============================================================================


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

        mock_runner = MagicMock()

        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            if "epic" in cmd and "status" in cmd:
                return CommandResult(
                    command=cmd,
                    returncode=0,
                    stdout='[{"eligible_for_close": true, "epic": {"id": "epic-1"}}]',
                )
            return CommandResult(command=cmd, returncode=0, stdout="")

        mock_runner.run_async = mock_run_async

        verifier = EpicVerifier(
            beads=mock_beads,
            model=mock_model,
            repo_path=Path("/tmp"),
            command_runner=mock_runner,
        )

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
                reasoning="Incorrect error handling",
            )
        )

        mock_runner = MagicMock()

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

        mock_runner.run_async = mock_run_async

        verifier = EpicVerifier(
            beads=mock_beads,
            model=mock_model,
            repo_path=Path("/tmp"),
            command_runner=mock_runner,
        )
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
                reasoning="All criteria met",
            )
        )

        mock_runner = MagicMock()

        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            if "epic" in cmd and "status" in cmd:
                return CommandResult(
                    command=cmd,
                    returncode=0,
                    stdout='[{"eligible_for_close": true, "epic": {"id": "epic-1"}}]',
                )
            return CommandResult(command=cmd, returncode=0, stdout="")

        mock_runner.run_async = mock_run_async

        verifier = EpicVerifier(
            beads=mock_beads,
            model=mock_model,
            repo_path=Path("/tmp"),
            command_runner=mock_runner,
        )
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
        model._prompt_template = "context: {epic_context}"

        async def fake_verify(_prompt: str) -> str:
            return '{"findings": [], "verdict": "PASS", "summary": "ok"}'

        model._verify_with_agent_sdk = fake_verify  # type: ignore[method-assign]

        verdict = await model.verify("criteria")

        assert verdict.passed is True

    @pytest.mark.asyncio
    async def test_verify_handles_non_json_response(self) -> None:
        model = ClaudeEpicVerificationModel()
        model._prompt_template = "context: {epic_context}"

        async def fake_verify(_prompt: str) -> str:
            return "not json"

        model._verify_with_agent_sdk = fake_verify  # type: ignore[method-assign]

        verdict = await model.verify("criteria")

        assert verdict.passed is False
        assert "Failed to parse" in verdict.reasoning


# ============================================================================
# Test max_diff_size_kb limit
# ============================================================================


class TestMaxDiffSizeKb:
    """Tests for max_diff_size_kb configuration."""

    @pytest.mark.asyncio
    async def test_skips_verification_when_diff_exceeds_limit(
        self,
        tmp_path: Path,
        mock_beads: MagicMock,
        mock_model: MagicMock,
    ) -> None:
        """Should skip verification and return requires human review when diff too large."""
        mock_runner = MagicMock()

        # Create verifier with max_diff_size_kb limit
        verifier = EpicVerifier(
            beads=mock_beads,
            model=mock_model,
            repo_path=tmp_path,
            command_runner=mock_runner,
            max_diff_size_kb=10,  # 10 KB limit
        )

        # Stub scope analyzer to return commit range
        mock_scope_analyzer = MagicMock()
        mock_scope_analyzer.compute_scoped_commits = AsyncMock(
            return_value=ScopedCommits(
                commit_shas=["abc123"],
                commit_range="abc123^..def456",
                commit_summary="- abc123 summary",
            )
        )
        verifier.scope_analyzer = mock_scope_analyzer

        # Mock git diff --numstat to return large diff (1000 lines = ~80KB)
        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            if "diff" in cmd and "--numstat" in cmd:
                # 1000 added + 500 removed = 1500 lines * 80 bytes = 120KB
                return CommandResult(
                    command=cmd,
                    returncode=0,
                    stdout="1000\t500\tfile.py",
                )
            return CommandResult(command=cmd, returncode=0, stdout="")

        mock_runner.run_async = mock_run_async

        verdict = await verifier.verify_epic("epic-1")

        # Verification should be skipped
        mock_model.verify.assert_not_called()
        assert verdict.passed is False
        assert "exceeds limit" in verdict.reasoning
        assert "Requires human review" in verdict.reasoning
        # Should have a synthetic unmet criterion for remediation issue creation
        assert len(verdict.unmet_criteria) == 1
        unmet = verdict.unmet_criteria[0]
        assert "Diff size must not exceed 10 KB" in unmet.criterion
        assert "exceeds limit" in unmet.evidence
        assert unmet.priority == 1  # P1 blocking

    @pytest.mark.asyncio
    async def test_proceeds_when_diff_within_limit(
        self,
        tmp_path: Path,
        mock_beads: MagicMock,
        mock_model: MagicMock,
    ) -> None:
        """Should proceed with verification when diff is within limit."""
        mock_runner = MagicMock()

        verifier = EpicVerifier(
            beads=mock_beads,
            model=mock_model,
            repo_path=tmp_path,
            command_runner=mock_runner,
            max_diff_size_kb=100,  # 100 KB limit
        )

        mock_scope_analyzer = MagicMock()
        mock_scope_analyzer.compute_scoped_commits = AsyncMock(
            return_value=ScopedCommits(
                commit_shas=["abc123"],
                commit_range="abc123^..def456",
                commit_summary="- abc123 summary",
            )
        )
        verifier.scope_analyzer = mock_scope_analyzer

        # Mock small diff (10 lines = ~0.8KB)
        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            if "diff" in cmd and "--numstat" in cmd:
                return CommandResult(
                    command=cmd,
                    returncode=0,
                    stdout="5\t5\tfile.py",
                )
            return CommandResult(command=cmd, returncode=0, stdout="")

        mock_runner.run_async = mock_run_async

        verdict = await verifier.verify_epic("epic-1")

        # Verification should proceed
        mock_model.verify.assert_called_once()
        assert verdict.passed is True

    @pytest.mark.asyncio
    async def test_no_limit_when_max_diff_size_kb_none(
        self,
        tmp_path: Path,
        mock_beads: MagicMock,
        mock_model: MagicMock,
    ) -> None:
        """Should not check diff size when max_diff_size_kb is None."""
        mock_runner = MagicMock()

        verifier = EpicVerifier(
            beads=mock_beads,
            model=mock_model,
            repo_path=tmp_path,
            command_runner=mock_runner,
            max_diff_size_kb=None,  # No limit
        )

        mock_scope_analyzer = MagicMock()
        mock_scope_analyzer.compute_scoped_commits = AsyncMock(
            return_value=ScopedCommits(
                commit_shas=["abc123"],
                commit_range="abc123^..def456",
                commit_summary="- abc123 summary",
            )
        )
        verifier.scope_analyzer = mock_scope_analyzer

        # Mock large diff - should not matter when limit is None
        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            # Don't expect diff --numstat to be called when max_diff_size_kb is None
            return CommandResult(command=cmd, returncode=0, stdout="")

        mock_runner.run_async = mock_run_async

        await verifier.verify_epic("epic-1")

        # Verification should proceed regardless of diff size
        mock_model.verify.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_verification_for_single_commit_exceeding_limit(
        self,
        tmp_path: Path,
        mock_beads: MagicMock,
        mock_model: MagicMock,
    ) -> None:
        """Should skip verification for single commit (no ..) that exceeds limit."""
        mock_runner = MagicMock()

        verifier = EpicVerifier(
            beads=mock_beads,
            model=mock_model,
            repo_path=tmp_path,
            command_runner=mock_runner,
            max_diff_size_kb=10,  # 10 KB limit
        )

        # Single commit range (no ..) - common case for single-issue epics
        mock_scope_analyzer = MagicMock()
        mock_scope_analyzer.compute_scoped_commits = AsyncMock(
            return_value=ScopedCommits(
                commit_shas=["abc123"],
                commit_range="abc123",  # Single SHA, no ..
                commit_summary="- abc123 summary",
            )
        )
        verifier.scope_analyzer = mock_scope_analyzer

        # Mock git diff to return large diff for abc123^..abc123
        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            if "diff" in cmd and "--numstat" in cmd:
                # Verify it's called with SHA^..SHA format
                assert "abc123^" in cmd, f"Expected abc123^ in {cmd}"
                assert "abc123" in cmd
                # Return large diff
                return CommandResult(
                    command=cmd,
                    returncode=0,
                    stdout="1000\t500\tfile.py",
                )
            return CommandResult(command=cmd, returncode=0, stdout="")

        mock_runner.run_async = mock_run_async

        verdict = await verifier.verify_epic("epic-1")

        # Verification should be skipped due to large diff
        mock_model.verify.assert_not_called()
        assert verdict.passed is False
        assert "exceeds limit" in verdict.reasoning
