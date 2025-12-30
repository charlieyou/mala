"""Unit tests for epic verification functionality.

Tests the EpicVerifier class and ClaudeEpicVerificationModel including:
- Scoped diff computation (child commits only, skip merges)
- Spec path extraction from descriptions
- Large diff handling (tiered truncation)
- Remediation issue creation with deduplication
- Human review issue creation
- Lock usage for sequential epic processing
"""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.epic_verifier import (
    ClaudeEpicVerificationModel,
    EpicVerifier,
    _compute_criterion_hash,
    extract_spec_paths,
)
from src.models import EpicVerdict, UnmetCriterion
from src.tools.command_runner import CommandResult


# ============================================================================
# Test fixtures
# ============================================================================


@pytest.fixture
def mock_beads() -> MagicMock:
    """Create a mock BeadsClient."""
    beads = MagicMock()
    beads.get_issue_description_async = AsyncMock(return_value="Epic description")
    beads.get_epic_children_async = AsyncMock(return_value={"child-1", "child-2"})
    beads.close_async = AsyncMock(return_value=True)
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
# Test scoped diff computation
# ============================================================================


class TestScopedDiffComputation:
    """Tests for scoped diff computation from child commits."""

    @pytest.mark.asyncio
    async def test_collects_commits_by_prefix(self, verifier: EpicVerifier) -> None:
        """Should collect commits matching bd-<child_id>: prefix."""

        # Mock git log to return commits
        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            if "git" in cmd and "log" in cmd:
                if "--grep=^bd-child-1:" in cmd:
                    return CommandResult(
                        command=cmd,
                        returncode=0,
                        stdout="abc123\ndef456",
                    )
                elif "--grep=^bd-child-2:" in cmd:
                    return CommandResult(
                        command=cmd,
                        returncode=0,
                        stdout="ghi789",
                    )
            if "git" in cmd and "show" in cmd:
                return CommandResult(
                    command=cmd,
                    returncode=0,
                    stdout="diff content for commit",
                )
            return CommandResult(command=cmd, returncode=1, stdout="", stderr="")

        verifier._runner.run_async = mock_run_async  # type: ignore[method-assign]

        diff = await verifier._compute_scoped_diff({"child-1", "child-2"})
        assert "diff content for commit" in diff

    @pytest.mark.asyncio
    async def test_skips_merge_commits(self, verifier: EpicVerifier) -> None:
        """Should skip merge commits via --no-merges flag."""
        commands_run: list[list[str]] = []

        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            commands_run.append(cmd)
            return CommandResult(command=cmd, returncode=0, stdout="")

        verifier._runner.run_async = mock_run_async  # type: ignore[method-assign]

        await verifier._compute_scoped_diff({"child-1"})

        # Verify --no-merges was used
        log_cmds = [c for c in commands_run if "log" in c]
        assert len(log_cmds) > 0
        assert "--no-merges" in log_cmds[0]

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_commits(self, verifier: EpicVerifier) -> None:
        """Should return empty string when no commits found."""

        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            return CommandResult(command=cmd, returncode=0, stdout="")

        verifier._runner.run_async = mock_run_async  # type: ignore[method-assign]

        diff = await verifier._compute_scoped_diff({"child-1"})
        assert diff == ""

    @pytest.mark.asyncio
    async def test_handles_multiple_commits_per_issue(
        self, verifier: EpicVerifier
    ) -> None:
        """Should include all commits matching an issue prefix."""
        commits_shown: list[str] = []

        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            if "log" in cmd:
                return CommandResult(
                    command=cmd,
                    returncode=0,
                    stdout="commit1\ncommit2\ncommit3",
                )
            if "show" in cmd and len(cmd) > 2:
                commits_shown.append(cmd[-1])
                return CommandResult(
                    command=cmd,
                    returncode=0,
                    stdout=f"diff for {cmd[-1]}",
                )
            return CommandResult(command=cmd, returncode=0, stdout="")

        verifier._runner.run_async = mock_run_async  # type: ignore[method-assign]

        diff = await verifier._compute_scoped_diff({"child-1"})
        assert len(commits_shown) == 3
        assert "diff for commit1" in diff
        assert "diff for commit2" in diff
        assert "diff for commit3" in diff


# ============================================================================
# Test large diff handling
# ============================================================================


class TestLargeDiffHandling:
    """Tests for tiered large diff handling."""

    def test_returns_unchanged_under_limit(self, verifier: EpicVerifier) -> None:
        """Should return diff unchanged if under size limit."""
        small_diff = "diff --git a/file.py b/file.py\n+some change"
        result = verifier._handle_large_diff(small_diff)
        assert result == small_diff

    def test_file_summary_mode_for_medium_diff(self, verifier: EpicVerifier) -> None:
        """Should use file-summary mode for diffs between limits."""
        # Create a diff larger than 100KB but under 500KB
        verifier.max_diff_size_kb = 1  # Lower limit for testing
        medium_diff = "diff --git a/file.py b/file.py\n" + ("+" * 2000)
        result = verifier._handle_large_diff(medium_diff)
        assert "file-summary mode" in result

    def test_file_list_mode_for_large_diff(self, verifier: EpicVerifier) -> None:
        """Should use file-list mode for very large diffs."""
        verifier.max_diff_size_kb = 1  # Lower limit
        # Create very large diff (over 500KB equivalent at scale)
        large_diff = "diff --git a/file.py b/file.py\n" + ("x" * 600000)
        result = verifier._handle_large_diff(large_diff)
        assert "file-list mode" in result

    def test_file_summary_truncates_per_file(self, verifier: EpicVerifier) -> None:
        """Should truncate each file to 50 lines in file-summary mode."""
        diff = "diff --git a/file.py b/file.py\n"
        diff += "\n".join([f"+line{i}" for i in range(100)])  # 100 lines
        result = verifier._to_file_summary_mode(diff)
        assert "more lines" in result  # Count includes the diff header line

    def test_file_list_includes_small_files(
        self, tmp_path: Path, mock_beads: MagicMock, mock_model: MagicMock
    ) -> None:
        """Should include contents of small files in file-list mode."""
        # Create a small file
        (tmp_path / "small.py").write_text("print('hello')")

        verifier = EpicVerifier(
            beads=mock_beads,
            model=mock_model,
            repo_path=tmp_path,
        )

        diff = "diff --git a/small.py b/small.py\n+content"
        result = verifier._to_file_list_mode(diff)
        assert "small.py" in result
        assert "print('hello')" in result


# ============================================================================
# Test remediation issue creation
# ============================================================================


class TestRemediationIssueCreation:
    """Tests for remediation issue creation and deduplication."""

    @pytest.mark.asyncio
    async def test_creates_issue_for_unmet_criterion(
        self, verifier: EpicVerifier
    ) -> None:
        """Should create an issue for each unmet criterion."""
        created_issues: list[dict[str, object]] = []

        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            if "issue" in cmd and "create" in cmd:
                created_issues.append({"cmd": cmd})
                return CommandResult(
                    command=cmd,
                    returncode=0,
                    stdout="Created issue: remediation-1",
                )
            if "list" in cmd and "--tag" in cmd:
                # No existing issue
                return CommandResult(
                    command=cmd,
                    returncode=0,
                    stdout="[]",
                )
            return CommandResult(command=cmd, returncode=0, stdout="")

        verifier._runner.run_async = mock_run_async  # type: ignore[method-assign]

        verdict = EpicVerdict(
            passed=False,
            unmet_criteria=[
                UnmetCriterion(
                    criterion="Must have error handling",
                    evidence="No try/catch found",
                    severity="major",
                    criterion_hash=_compute_criterion_hash("Must have error handling"),
                )
            ],
            confidence=0.8,
            reasoning="Missing error handling",
        )

        issue_ids = await verifier.create_remediation_issues("epic-1", verdict)
        assert len(issue_ids) == 1
        assert issue_ids[0] == "remediation-1"
        assert len(created_issues) == 1

    @pytest.mark.asyncio
    async def test_deduplicates_by_tag(self, verifier: EpicVerifier) -> None:
        """Should reuse existing issue with matching dedup tag."""

        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            if "list" in cmd and "--tag" in cmd:
                # Return existing issue
                return CommandResult(
                    command=cmd,
                    returncode=0,
                    stdout='[{"id": "existing-issue"}]',
                )
            return CommandResult(command=cmd, returncode=0, stdout="")

        verifier._runner.run_async = mock_run_async  # type: ignore[method-assign]

        verdict = EpicVerdict(
            passed=False,
            unmet_criteria=[
                UnmetCriterion(
                    criterion="Criterion text",
                    evidence="Evidence",
                    severity="major",
                    criterion_hash=_compute_criterion_hash("Criterion text"),
                )
            ],
            confidence=0.8,
            reasoning="Test",
        )

        issue_ids = await verifier.create_remediation_issues("epic-1", verdict)
        assert issue_ids == ["existing-issue"]

    @pytest.mark.asyncio
    async def test_remediation_issue_format(self, verifier: EpicVerifier) -> None:
        """Should create issue with correct title/body/tags format."""
        created_cmd: list[str] = []

        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            if "issue" in cmd and "create" in cmd:
                created_cmd.extend(cmd)
                return CommandResult(
                    command=cmd,
                    returncode=0,
                    stdout="Created issue: new-1",
                )
            if "list" in cmd:
                return CommandResult(command=cmd, returncode=0, stdout="[]")
            if "show" in cmd:
                return CommandResult(
                    command=cmd,
                    returncode=0,
                    stdout='{"priority": "P2"}',
                )
            return CommandResult(command=cmd, returncode=0, stdout="")

        verifier._runner.run_async = mock_run_async  # type: ignore[method-assign]

        verdict = EpicVerdict(
            passed=False,
            unmet_criteria=[
                UnmetCriterion(
                    criterion="API must return 400 for bad input",
                    evidence="Returns 500 instead",
                    severity="critical",
                    criterion_hash=_compute_criterion_hash(
                        "API must return 400 for bad input"
                    ),
                )
            ],
            confidence=0.8,
            reasoning="Test",
        )

        await verifier.create_remediation_issues("epic-1", verdict)

        # Check title format
        title_idx = created_cmd.index("--title") + 1
        assert created_cmd[title_idx].startswith("[Remediation]")

        # Check tags include dedup tag and auto_generated
        tag_indices = [i for i, x in enumerate(created_cmd) if x == "--tag"]
        tags = [created_cmd[i + 1] for i in tag_indices]
        assert any("epic_remediation:" in t for t in tags)
        assert "auto_generated" in tags


# ============================================================================
# Test human review issue creation
# ============================================================================


class TestHumanReviewCreation:
    """Tests for human review issue creation."""

    @pytest.mark.asyncio
    async def test_creates_human_review_issue(self, verifier: EpicVerifier) -> None:
        """Should create human review issue with correct format."""
        created_cmd: list[str] = []

        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            if "issue" in cmd and "create" in cmd:
                created_cmd.extend(cmd)
                return CommandResult(
                    command=cmd,
                    returncode=0,
                    stdout="Created issue: review-1",
                )
            if "dep" in cmd and "add" in cmd:
                return CommandResult(command=cmd, returncode=0, stdout="")
            return CommandResult(command=cmd, returncode=0, stdout="")

        verifier._runner.run_async = mock_run_async  # type: ignore[method-assign]

        issue_id = await verifier.request_human_review(
            "epic-1",
            "Missing acceptance criteria",
            None,
        )

        assert issue_id == "review-1"
        title_idx = created_cmd.index("--title") + 1
        assert "[Human Review]" in created_cmd[title_idx]
        assert "epic-1" in created_cmd[title_idx]

    @pytest.mark.asyncio
    async def test_human_review_adds_blocker(self, verifier: EpicVerifier) -> None:
        """Should add review issue as epic blocker."""
        dep_cmds: list[list[str]] = []

        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            if "issue" in cmd and "create" in cmd:
                return CommandResult(
                    command=cmd,
                    returncode=0,
                    stdout="Created issue: review-1",
                )
            if "dep" in cmd and "add" in cmd:
                dep_cmds.append(list(cmd))
                return CommandResult(command=cmd, returncode=0, stdout="")
            return CommandResult(command=cmd, returncode=0, stdout="")

        verifier._runner.run_async = mock_run_async  # type: ignore[method-assign]

        await verifier.request_human_review("epic-1", "Low confidence", None)

        assert len(dep_cmds) == 1
        assert "epic-1" in dep_cmds[0]
        assert "--blocked-by" in dep_cmds[0]
        assert "review-1" in dep_cmds[0]


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

        # Mock diff computation
        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            if "log" in cmd:
                return CommandResult(command=cmd, returncode=0, stdout="abc123")
            if "show" in cmd:
                return CommandResult(command=cmd, returncode=0, stdout="diff content")
            return CommandResult(command=cmd, returncode=0, stdout="")

        verifier._runner.run_async = mock_run_async  # type: ignore[method-assign]

        await verifier.verify_epic("epic-1")

        # Verify model was called with spec content
        mock_model.verify.assert_called_once()
        call_args = mock_model.verify.call_args
        assert call_args[0][2] == "# Auth Spec\nDetails here."


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
            if "epic" in cmd and "list" in cmd and "--eligible" in cmd:
                return CommandResult(
                    command=cmd,
                    returncode=0,
                    stdout='[{"id": "epic-1"}]',
                )
            if "log" in cmd:
                return CommandResult(command=cmd, returncode=0, stdout="abc123")
            if "show" in cmd:
                return CommandResult(command=cmd, returncode=0, stdout="diff")
            return CommandResult(command=cmd, returncode=0, stdout="")

        verifier._runner.run_async = mock_run_async  # type: ignore[method-assign]

        result = await verifier.verify_and_close_eligible()

        assert result.passed_count == 1
        mock_beads.close_async.assert_called_with("epic-1")

    @pytest.mark.asyncio
    async def test_human_override_bypasses_verification(
        self, verifier: EpicVerifier, mock_beads: MagicMock, mock_model: MagicMock
    ) -> None:
        """Should close overridden epics without verification."""

        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            if "epic" in cmd and "list" in cmd:
                return CommandResult(
                    command=cmd,
                    returncode=0,
                    stdout='[{"id": "epic-1"}]',
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
                    criterion="Must have tests",
                    evidence="No tests found",
                    severity="major",
                    criterion_hash=_compute_criterion_hash("Must have tests"),
                )
            ],
            confidence=0.85,
            reasoning="Missing tests",
        )

        created_issues: list[str] = []

        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            if "epic" in cmd and "list" in cmd:
                return CommandResult(
                    command=cmd,
                    returncode=0,
                    stdout='[{"id": "epic-1"}]',
                )
            if "log" in cmd:
                return CommandResult(command=cmd, returncode=0, stdout="abc123")
            if "show" in cmd and "epic-1" in cmd:
                return CommandResult(
                    command=cmd,
                    returncode=0,
                    stdout='{"priority": "P2"}',
                )
            if "show" in cmd:
                return CommandResult(command=cmd, returncode=0, stdout="diff")
            if "list" in cmd and "--tag" in cmd:
                return CommandResult(command=cmd, returncode=0, stdout="[]")
            if "issue" in cmd and "create" in cmd:
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

        result = await verifier.verify_and_close_eligible()

        assert result.failed_count == 1
        assert len(result.remediation_issues_created) == 1
        # Epic should NOT be closed
        mock_beads.close_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_low_confidence_triggers_review(
        self, verifier: EpicVerifier, mock_beads: MagicMock, mock_model: MagicMock
    ) -> None:
        """Should create human review for low confidence verdicts."""
        mock_model.verify.return_value = EpicVerdict(
            passed=True,
            unmet_criteria=[],
            confidence=0.3,  # Below 0.5 threshold
            reasoning="Uncertain",
        )

        review_created = False

        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            nonlocal review_created
            if "epic" in cmd and "list" in cmd:
                return CommandResult(
                    command=cmd,
                    returncode=0,
                    stdout='[{"id": "epic-1"}]',
                )
            if "log" in cmd:
                return CommandResult(command=cmd, returncode=0, stdout="abc123")
            if "show" in cmd:
                return CommandResult(command=cmd, returncode=0, stdout="diff")
            if "issue" in cmd and "create" in cmd:
                review_created = True
                return CommandResult(
                    command=cmd,
                    returncode=0,
                    stdout="Created issue: review-1",
                )
            if "dep" in cmd:
                return CommandResult(command=cmd, returncode=0, stdout="")
            return CommandResult(command=cmd, returncode=0, stdout="")

        verifier._runner.run_async = mock_run_async  # type: ignore[method-assign]

        result = await verifier.verify_and_close_eligible()

        assert result.human_review_count == 1
        assert review_created


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
            if "epic" in cmd and "list" in cmd:
                return CommandResult(
                    command=cmd,
                    returncode=0,
                    stdout='[{"id": "epic-1"}]',
                )
            if "log" in cmd:
                return CommandResult(command=cmd, returncode=0, stdout="abc123")
            if "show" in cmd:
                return CommandResult(command=cmd, returncode=0, stdout="diff")
            return CommandResult(command=cmd, returncode=0, stdout="")

        verifier._runner.run_async = mock_run_async  # type: ignore[method-assign]

        # Mock the locking module
        with patch("src.tools.locking.wait_for_lock") as mock_wait:
            mock_wait.return_value = True

            with patch("src.tools.locking.get_lock_holder") as mock_holder:
                mock_holder.return_value = "epic_verifier"

                with patch("src.tools.locking.lock_path") as mock_path:
                    mock_lock_path = MagicMock()
                    mock_lock_path.exists.return_value = True
                    mock_path.return_value = mock_lock_path

                    await verifier.verify_and_close_eligible()

                    # Lock should have been attempted since lock_manager is set
                    # (the actual call happens inside a try/except ImportError)
                    # Since we patched the module, the import succeeds
                    mock_wait.assert_called_once()
                    call_args = mock_wait.call_args[0]
                    assert "epic_verify:epic-1" == call_args[0]


# ============================================================================
# Test priority adjustment
# ============================================================================


class TestPriorityAdjustment:
    """Tests for severity-based priority adjustment."""

    @pytest.mark.asyncio
    async def test_critical_keeps_epic_priority(self, verifier: EpicVerifier) -> None:
        """Critical severity should keep same priority as epic."""

        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            if "show" in cmd:
                return CommandResult(
                    command=cmd,
                    returncode=0,
                    stdout='{"priority": "P2"}',
                )
            return CommandResult(command=cmd, returncode=0, stdout="")

        verifier._runner.run_async = mock_run_async  # type: ignore[method-assign]

        priority = await verifier._get_adjusted_priority("epic-1", "critical")
        assert priority == "P2"

    @pytest.mark.asyncio
    async def test_major_lowers_priority(self, verifier: EpicVerifier) -> None:
        """Major severity should lower priority by 1."""

        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            if "show" in cmd:
                return CommandResult(
                    command=cmd,
                    returncode=0,
                    stdout='{"priority": "P2"}',
                )
            return CommandResult(command=cmd, returncode=0, stdout="")

        verifier._runner.run_async = mock_run_async  # type: ignore[method-assign]

        priority = await verifier._get_adjusted_priority("epic-1", "major")
        assert priority == "P3"

    @pytest.mark.asyncio
    async def test_minor_lowers_priority_by_two(self, verifier: EpicVerifier) -> None:
        """Minor severity should lower priority by 2."""

        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            if "show" in cmd:
                return CommandResult(
                    command=cmd,
                    returncode=0,
                    stdout='{"priority": "P2"}',
                )
            return CommandResult(command=cmd, returncode=0, stdout="")

        verifier._runner.run_async = mock_run_async  # type: ignore[method-assign]

        priority = await verifier._get_adjusted_priority("epic-1", "minor")
        assert priority == "P4"

    @pytest.mark.asyncio
    async def test_priority_caps_at_p5(self, verifier: EpicVerifier) -> None:
        """Priority should cap at P5."""

        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            if "show" in cmd:
                return CommandResult(
                    command=cmd,
                    returncode=0,
                    stdout='{"priority": "P4"}',
                )
            return CommandResult(command=cmd, returncode=0, stdout="")

        verifier._runner.run_async = mock_run_async  # type: ignore[method-assign]

        priority = await verifier._get_adjusted_priority("epic-1", "minor")
        assert priority == "P5"  # P4 + 2 = P6, capped to P5


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

        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            if "log" in cmd:
                return CommandResult(command=cmd, returncode=0, stdout="abc123")
            if "show" in cmd:
                return CommandResult(command=cmd, returncode=0, stdout="diff")
            return CommandResult(command=cmd, returncode=0, stdout="")

        verifier._runner.run_async = mock_run_async  # type: ignore[method-assign]

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

        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            if "log" in cmd:
                return CommandResult(command=cmd, returncode=0, stdout="abc123")
            if "show" in cmd:
                return CommandResult(command=cmd, returncode=0, stdout="diff")
            return CommandResult(command=cmd, returncode=0, stdout="")

        verifier._runner.run_async = mock_run_async  # type: ignore[method-assign]

        verdict = await verifier.verify_epic("epic-1")

        assert verdict.passed is False
        assert verdict.confidence == 0.0
        assert "API connection failed" in verdict.reasoning

    @pytest.mark.asyncio
    async def test_model_error_triggers_human_review_in_eligible_flow(
        self, verifier: EpicVerifier, mock_beads: MagicMock, mock_model: MagicMock
    ) -> None:
        """Model errors should trigger human review in verify_and_close_eligible."""
        mock_model.verify.side_effect = TimeoutError("timeout")

        review_created = False

        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            nonlocal review_created
            if "epic" in cmd and "list" in cmd:
                return CommandResult(
                    command=cmd,
                    returncode=0,
                    stdout='[{"id": "epic-1"}]',
                )
            if "log" in cmd:
                return CommandResult(command=cmd, returncode=0, stdout="abc123")
            if "show" in cmd:
                return CommandResult(command=cmd, returncode=0, stdout="diff")
            if "issue" in cmd and "create" in cmd:
                review_created = True
                return CommandResult(
                    command=cmd,
                    returncode=0,
                    stdout="Created issue: review-1",
                )
            if "dep" in cmd:
                return CommandResult(command=cmd, returncode=0, stdout="")
            return CommandResult(command=cmd, returncode=0, stdout="")

        verifier._runner.run_async = mock_run_async  # type: ignore[method-assign]

        result = await verifier.verify_and_close_eligible()

        # Low confidence (0.0) should trigger human review
        assert result.human_review_count == 1
        assert review_created


# ============================================================================
# Test binary file handling
# ============================================================================


class TestBinaryFileHandling:
    """Tests for handling binary/non-UTF8 files in file-list mode."""

    def test_file_list_skips_binary_files(
        self, tmp_path: Path, mock_beads: MagicMock, mock_model: MagicMock
    ) -> None:
        """Should skip binary files without crashing."""
        # Create a binary file
        (tmp_path / "binary.bin").write_bytes(b"\x00\x01\x02\xff\xfe")
        # Create a text file
        (tmp_path / "text.py").write_text("print('hello')")

        verifier = EpicVerifier(
            beads=mock_beads,
            model=mock_model,
            repo_path=tmp_path,
        )

        diff = "diff --git a/binary.bin b/binary.bin\n+content\ndiff --git a/text.py b/text.py\n+code"
        result = verifier._to_file_list_mode(diff)

        # Should include text file but not crash on binary
        assert "text.py" in result
        assert "print('hello')" in result
        # Binary file should be listed but content not included
        assert "binary.bin" in result

    def test_file_list_skips_non_utf8_files(
        self, tmp_path: Path, mock_beads: MagicMock, mock_model: MagicMock
    ) -> None:
        """Should skip files with invalid UTF-8 without crashing."""
        # Create a file with invalid UTF-8
        (tmp_path / "invalid.txt").write_bytes(b"Hello \xff\xfe World")

        verifier = EpicVerifier(
            beads=mock_beads,
            model=mock_model,
            repo_path=tmp_path,
        )

        diff = "diff --git a/invalid.txt b/invalid.txt\n+content"
        # Should not raise UnicodeDecodeError
        result = verifier._to_file_list_mode(diff)

        # File should be listed but content not included
        assert "invalid.txt" in result
        assert "Hello" not in result  # Content should not be present
