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
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.epic_verifier import (
    ClaudeEpicVerificationModel,
    EpicVerifier,
    _compute_criterion_hash,
    _extract_json_from_code_blocks,
    extract_spec_paths,
)
from src.models import EpicVerdict, RetryConfig, UnmetCriterion
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
    beads.get_epic_blockers_async = AsyncMock(return_value=set())
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

    @pytest.mark.asyncio
    async def test_returns_unchanged_under_limit(self, verifier: EpicVerifier) -> None:
        """Should return diff unchanged if under size limit."""
        small_diff = "diff --git a/file.py b/file.py\n+some change"
        result = await verifier._handle_large_diff(small_diff)
        assert result == small_diff

    @pytest.mark.asyncio
    async def test_file_summary_mode_for_medium_diff(
        self, verifier: EpicVerifier
    ) -> None:
        """Should use file-summary mode for diffs between limits."""
        # Create a diff larger than 100KB but under 500KB
        verifier.max_diff_size_kb = 1  # Lower limit for testing
        medium_diff = "diff --git a/file.py b/file.py\n" + ("+" * 2000)
        result = await verifier._handle_large_diff(medium_diff)
        assert "file-summary mode" in result

    @pytest.mark.asyncio
    async def test_file_list_mode_for_large_diff(self, verifier: EpicVerifier) -> None:
        """Should use file-list mode for very large diffs."""
        verifier.max_diff_size_kb = 1  # Lower limit
        # Create very large diff (over 500KB equivalent at scale)
        large_diff = "diff --git a/file.py b/file.py\n" + ("x" * 600000)
        result = await verifier._handle_large_diff(large_diff)
        assert "file-list mode" in result

    def test_file_summary_truncates_per_file(self, verifier: EpicVerifier) -> None:
        """Should truncate each file to 50 lines in file-summary mode."""
        diff = "diff --git a/file.py b/file.py\n"
        diff += "\n".join([f"+line{i}" for i in range(100)])  # 100 lines
        result = verifier._to_file_summary_mode(diff)
        assert "more lines" in result  # Count includes the diff header line

    @pytest.mark.asyncio
    async def test_file_list_includes_small_files(
        self, tmp_path: Path, mock_beads: MagicMock, mock_model: MagicMock
    ) -> None:
        """Should include contents of small files in file-list mode.

        Files are read from HEAD commit to ensure consistency with the scoped diff.
        """
        import subprocess

        # Initialize a git repo in tmp_path
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

        # Create and commit a small file
        (tmp_path / "small.py").write_text("print('hello')")
        subprocess.run(
            ["git", "add", "small.py"], cwd=tmp_path, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "initial"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

        verifier = EpicVerifier(
            beads=mock_beads,
            model=mock_model,
            repo_path=tmp_path,
        )

        diff = "diff --git a/small.py b/small.py\n+content"
        result = await verifier._to_file_list_mode(diff)
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
            if cmd[:2] == ["bd", "create"]:
                created_issues.append({"cmd": cmd})
                return CommandResult(
                    command=cmd,
                    returncode=0,
                    stdout="Created issue: remediation-1",
                )
            if "list" in cmd and "--label" in cmd:
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
        assert "--parent" in created_issues[0]["cmd"]
        parent_idx = created_issues[0]["cmd"].index("--parent") + 1
        assert created_issues[0]["cmd"][parent_idx] == "epic-1"

    @pytest.mark.asyncio
    async def test_deduplicates_by_tag(self, verifier: EpicVerifier) -> None:
        """Should reuse existing issue with matching dedup tag."""

        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            if "list" in cmd and "--label" in cmd:
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
            if cmd[:2] == ["bd", "create"]:
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
        labels_idx = created_cmd.index("--labels") + 1
        labels = created_cmd[labels_idx].split(",")
        assert any("epic_remediation:" in label for label in labels)
        assert "auto_generated" in labels
        parent_idx = created_cmd.index("--parent") + 1
        assert created_cmd[parent_idx] == "epic-1"


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
            if cmd[:2] == ["bd", "create"]:
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
            if cmd[:2] == ["bd", "create"]:
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
            if cmd[:2] == ["bd", "create"]:
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
            if "epic" in cmd and "list" in cmd and "--eligible" in cmd:
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
            if "epic" in cmd and "list" in cmd and "--eligible" in cmd:
                return CommandResult(
                    command=cmd,
                    returncode=0,
                    stdout='[{"id": "epic-1"}]',
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
            if "epic" in cmd and "list" in cmd and "--eligible" in cmd:
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
        import os

        expected_agent_id = f"epic_verifier_{os.getpid()}"

        with patch("src.tools.locking.wait_for_lock") as mock_wait:
            mock_wait.return_value = True

            with patch("src.tools.locking.get_lock_holder") as mock_holder:
                mock_holder.return_value = expected_agent_id

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
                    assert call_args[0] == "epic_verify:epic-1"
                    assert call_args[1] == expected_agent_id
                    assert call_args[2] == str(verifier.repo_path)  # repo_namespace


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
            if cmd[:2] == ["bd", "create"]:
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

    @pytest.mark.asyncio
    async def test_file_list_skips_binary_files(
        self, tmp_path: Path, mock_beads: MagicMock, mock_model: MagicMock
    ) -> None:
        """Should skip binary files without crashing.

        Files are read from HEAD commit to ensure consistency with the scoped diff.
        """
        import subprocess

        # Initialize a git repo in tmp_path
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

        # Create and commit files
        # Create a binary file
        (tmp_path / "binary.bin").write_bytes(b"\x00\x01\x02\xff\xfe")
        # Create a text file
        (tmp_path / "text.py").write_text("print('hello')")
        subprocess.run(
            ["git", "add", "."], cwd=tmp_path, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "initial"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

        verifier = EpicVerifier(
            beads=mock_beads,
            model=mock_model,
            repo_path=tmp_path,
        )

        diff = "diff --git a/binary.bin b/binary.bin\n+content\ndiff --git a/text.py b/text.py\n+code"
        result = await verifier._to_file_list_mode(diff)

        # Should include text file but not crash on binary
        assert "text.py" in result
        assert "print('hello')" in result
        # Binary file should be listed but content not included
        assert "binary.bin" in result

    @pytest.mark.asyncio
    async def test_file_list_skips_non_utf8_files(
        self, tmp_path: Path, mock_beads: MagicMock, mock_model: MagicMock
    ) -> None:
        """Should skip files with invalid UTF-8 without crashing.

        Files are read from HEAD commit to ensure consistency with the scoped diff.
        """
        import subprocess

        # Initialize a git repo in tmp_path
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

        # Create a file with invalid UTF-8 and commit it
        (tmp_path / "invalid.txt").write_bytes(b"Hello \xff\xfe World")
        subprocess.run(
            ["git", "add", "."], cwd=tmp_path, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "initial"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

        verifier = EpicVerifier(
            beads=mock_beads,
            model=mock_model,
            repo_path=tmp_path,
        )

        diff = "diff --git a/invalid.txt b/invalid.txt\n+content"
        # Should not raise UnicodeDecodeError
        result = await verifier._to_file_list_mode(diff)

        # File should be listed but content not included
        assert "invalid.txt" in result
        assert "Hello" not in result  # Content should not be present


# ============================================================================
# Integration Tests for EpicVerifier Orchestrator Wiring
# ============================================================================


@pytest.mark.integration
class TestEpicVerifierOrchestratorIntegration:
    """Integration tests for EpicVerifier wiring in MalaOrchestrator."""

    @pytest.mark.asyncio
    async def test_orchestrator_uses_epic_verifier_for_closure(
        self, tmp_path: Path
    ) -> None:
        """Orchestrator should use EpicVerifier instead of close_eligible_epics_async."""
        from src.orchestrator import MalaOrchestrator
        from src.config import MalaConfig

        # Create orchestrator with mock beads
        config = MalaConfig(max_diff_size_kb=100)
        orchestrator = MalaOrchestrator(
            repo_path=tmp_path,
            max_agents=1,
            config=config,
        )

        # Verify EpicVerifier was instantiated
        assert orchestrator.epic_verifier is not None
        assert orchestrator.epic_verifier.max_diff_size_kb == 100

    @pytest.mark.asyncio
    async def test_orchestrator_respects_epic_override_ids(
        self, tmp_path: Path
    ) -> None:
        """Orchestrator should pass epic_override_ids to verify_and_close_eligible."""
        from src.orchestrator import MalaOrchestrator
        from src.config import MalaConfig

        override_ids = {"epic-1", "epic-2"}
        config = MalaConfig(max_diff_size_kb=100)
        orchestrator = MalaOrchestrator(
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

        mock_model = MagicMock()
        mock_model.verify = AsyncMock(
            return_value=EpicVerdict(
                passed=False,
                unmet_criteria=[
                    UnmetCriterion(
                        criterion="Must have tests",
                        evidence="No tests",
                        severity="major",
                        criterion_hash=_compute_criterion_hash("Must have tests"),
                    )
                ],
                confidence=0.9,
                reasoning="Tests missing",
            )
        )

        verifier = EpicVerifier(
            beads=mock_beads,
            model=mock_model,
            repo_path=Path("/tmp"),
        )

        created_issues: list[str] = []

        async def mock_run_async(cmd: list[str], **kwargs: object) -> CommandResult:
            if "epic" in cmd and "list" in cmd and "--eligible" in cmd:
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

        result = await verifier.verify_and_close_eligible()

        # Epic should NOT be closed (verification failed)
        mock_beads.close_async.assert_not_called()
        assert result.failed_count == 1
        assert result.passed_count == 0
        assert len(result.remediation_issues_created) == 1

    @pytest.mark.asyncio
    async def test_config_max_diff_size_kb_is_respected(self) -> None:
        """EpicVerifier should use max_diff_size_kb from config."""
        from src.config import MalaConfig

        config = MalaConfig(max_diff_size_kb=50)
        assert config.max_diff_size_kb == 50

        mock_beads = MagicMock()
        mock_model = MagicMock()

        verifier = EpicVerifier(
            beads=mock_beads,
            model=mock_model,
            repo_path=Path("/tmp"),
            max_diff_size_kb=config.max_diff_size_kb,
        )

        assert verifier.max_diff_size_kb == 50

    @pytest.mark.asyncio
    async def test_config_default_max_diff_size_kb_is_100(self) -> None:
        """Default max_diff_size_kb should be 100."""
        from src.config import MalaConfig

        config = MalaConfig()
        assert config.max_diff_size_kb == 100

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
            "criteria: {epic_criteria}, spec: {spec_content}, diff: {diff_content}"
        )

        async def fake_verify(_prompt: str) -> str:
            return (
                '{"passed": true, "confidence": 0.9, "reasoning": "ok", '
                '"unmet_criteria": []}'
            )

        model._verify_with_agent_sdk = fake_verify  # type: ignore[method-assign]

        verdict = await model.verify("criteria", "diff", None)

        assert verdict.passed is True
        assert verdict.confidence == 0.9

    @pytest.mark.asyncio
    async def test_verify_handles_non_json_response(self) -> None:
        model = ClaudeEpicVerificationModel()
        model._prompt_template = (
            "criteria: {epic_criteria}, spec: {spec_content}, diff: {diff_content}"
        )

        async def fake_verify(_prompt: str) -> str:
            return "not json"

        model._verify_with_agent_sdk = fake_verify  # type: ignore[method-assign]

        verdict = await model.verify("criteria", "diff", None)

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
