"""Unit tests for extracted orchestration helpers.

Tests for:
- gate_metadata: GateMetadata building from gate results and logs
- review_tracking: Creating tracking issues from P2/P3 findings
- run_config: Building EventRunConfig and RunMetadata
- prompts: Loading and caching prompt templates
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock, patch

import pytest

from src.domain.quality_gate import GateResult, ValidationEvidence
from src.domain.validation.spec import CommandKind
from src.orchestration.gate_metadata import (
    GateMetadata,
    build_gate_metadata,
    build_gate_metadata_from_logs,
)
from src.orchestration.prompts import (
    get_fixer_prompt,
    get_implementer_prompt,
    get_review_followup_prompt,
)
from src.orchestration.review_tracking import (
    _extract_existing_fingerprints,
    _get_finding_fingerprint,
    _update_header_count,
    create_review_tracking_issues,
)
from src.orchestration.run_config import build_event_run_config, build_run_metadata
from src.infra.tools.env import USER_CONFIG_DIR

if TYPE_CHECKING:
    from src.core.protocols import IssueProvider
    from src.infra.io.event_protocol import MalaEventSink


# ============================================================================
# gate_metadata tests
# ============================================================================


class TestGateMetadata:
    """Test GateMetadata dataclass."""

    def test_empty_metadata(self) -> None:
        metadata = GateMetadata()
        assert metadata.quality_gate_result is None
        assert metadata.validation_result is None


class TestBuildGateMetadata:
    """Test build_gate_metadata function."""

    def test_none_gate_result(self) -> None:
        """Should return empty metadata when gate_result is None."""
        metadata = build_gate_metadata(None, passed=True)
        assert metadata.quality_gate_result is None
        assert metadata.validation_result is None

    def test_passed_gate_result(self) -> None:
        """Should extract metadata from a passed gate result."""
        evidence = ValidationEvidence(
            commands_ran={CommandKind.TEST: True, CommandKind.LINT: True},
            failed_commands=[],
        )
        gate_result = GateResult(
            passed=True,
            failure_reasons=[],
            validation_evidence=evidence,
            commit_hash="abc123",
        )

        metadata = build_gate_metadata(gate_result, passed=True)

        assert metadata.quality_gate_result is not None
        assert metadata.quality_gate_result.passed is True
        assert metadata.quality_gate_result.evidence["commit_found"] is True
        assert metadata.quality_gate_result.failure_reasons == []

        assert metadata.validation_result is not None
        assert metadata.validation_result.passed is True
        # Commands run uses kind.value (e.g., "test", "lint")
        assert "test" in metadata.validation_result.commands_run
        assert "lint" in metadata.validation_result.commands_run
        assert metadata.validation_result.commands_failed == []

    def test_failed_gate_result(self) -> None:
        """Should extract metadata from a failed gate result."""
        evidence = ValidationEvidence(
            commands_ran={CommandKind.TEST: True, CommandKind.LINT: True},
            failed_commands=["ruff check"],
        )
        gate_result = GateResult(
            passed=False,
            failure_reasons=["ruff check failed"],
            validation_evidence=evidence,
            commit_hash="abc123",
        )

        metadata = build_gate_metadata(gate_result, passed=False)

        assert metadata.quality_gate_result is not None
        assert metadata.quality_gate_result.passed is False
        assert "ruff check failed" in metadata.quality_gate_result.failure_reasons

        assert metadata.validation_result is not None
        assert metadata.validation_result.passed is False
        assert "ruff check" in metadata.validation_result.commands_failed

    def test_passed_override(self) -> None:
        """Should override passed status when overall run passed."""
        evidence = ValidationEvidence(
            commands_ran={CommandKind.TEST: True},
            failed_commands=[],
        )
        gate_result = GateResult(
            passed=False,  # Gate failed but overall run passed
            failure_reasons=["some reason"],
            validation_evidence=evidence,
            commit_hash="abc123",
        )

        metadata = build_gate_metadata(gate_result, passed=True)

        # When passed=True, it should override gate_result.passed
        assert metadata.quality_gate_result is not None
        assert metadata.quality_gate_result.passed is True
        assert metadata.validation_result is not None
        assert metadata.validation_result.passed is True

    def test_no_commit_hash(self) -> None:
        """Should handle missing commit hash."""
        gate_result = GateResult(
            passed=True,
            failure_reasons=[],
            validation_evidence=None,
            commit_hash=None,
        )

        metadata = build_gate_metadata(gate_result, passed=True)

        assert metadata.quality_gate_result is not None
        assert metadata.quality_gate_result.evidence["commit_found"] is False

    def test_no_validation_evidence(self) -> None:
        """Should handle missing validation evidence."""
        gate_result = GateResult(
            passed=True,
            failure_reasons=[],
            validation_evidence=None,
            commit_hash="abc123",
        )

        metadata = build_gate_metadata(gate_result, passed=True)

        assert metadata.quality_gate_result is not None
        # No validation result when evidence is missing
        assert metadata.validation_result is None


class TestBuildGateMetadataFromLogs:
    """Test build_gate_metadata_from_logs fallback function."""

    def test_none_spec(self, tmp_path: Path) -> None:
        """Should return empty metadata when spec is None."""
        log_path = tmp_path / "session.log"
        log_path.touch()

        mock_gate = MagicMock()
        metadata = build_gate_metadata_from_logs(
            log_path=log_path,
            result_summary="Success",
            result_success=True,
            quality_gate=mock_gate,
            per_issue_spec=None,
        )

        assert metadata.quality_gate_result is None
        assert metadata.validation_result is None

    def test_with_spec(self, tmp_path: Path) -> None:
        """Should parse logs and extract metadata when spec is provided."""
        log_path = tmp_path / "session.log"
        log_path.write_text("mock log content")

        evidence = ValidationEvidence(
            commands_ran={CommandKind.TEST: True},
            failed_commands=[],
        )

        mock_gate = MagicMock()
        mock_gate.parse_validation_evidence_with_spec.return_value = evidence

        mock_spec = MagicMock()

        metadata = build_gate_metadata_from_logs(
            log_path=log_path,
            result_summary="Success",
            result_success=True,
            quality_gate=mock_gate,
            per_issue_spec=mock_spec,
        )

        assert metadata.quality_gate_result is not None
        assert metadata.quality_gate_result.passed is True
        assert metadata.validation_result is not None
        # Commands run uses kind.value (e.g., "test")
        assert "test" in metadata.validation_result.commands_run

    def test_failure_reason_extraction(self, tmp_path: Path) -> None:
        """Should extract failure reasons from result summary."""
        log_path = tmp_path / "session.log"
        log_path.write_text("mock log content")

        evidence = ValidationEvidence(
            commands_ran={CommandKind.TEST: True},
            failed_commands=["pytest"],
        )

        mock_gate = MagicMock()
        mock_gate.parse_validation_evidence_with_spec.return_value = evidence

        mock_spec = MagicMock()

        metadata = build_gate_metadata_from_logs(
            log_path=log_path,
            result_summary="Quality gate failed: tests failed; lint failed",
            result_success=False,
            quality_gate=mock_gate,
            per_issue_spec=mock_spec,
        )

        assert metadata.quality_gate_result is not None
        assert "tests failed" in metadata.quality_gate_result.failure_reasons
        assert "lint failed" in metadata.quality_gate_result.failure_reasons


# ============================================================================
# review_tracking tests
# ============================================================================


@dataclass
class FakeReviewIssue:
    """Fake ReviewIssue implementing the protocol."""

    file: str
    line_start: int
    line_end: int
    priority: int | None
    title: str
    body: str
    reviewer: str


class FakeIssueProvider:
    """Fake IssueProvider for testing review tracking."""

    def __init__(self) -> None:
        self.created_issues: list[dict[str, Any]] = []
        self.existing_tags: dict[str, str] = {}
        self.issue_descriptions: dict[str, str] = {}
        self.updated_descriptions: list[tuple[str, str]] = []
        self.updated_issues: list[tuple[str, str | None, str | None]] = []

    async def find_issue_by_tag_async(self, tag: str) -> str | None:
        return self.existing_tags.get(tag)

    async def get_issue_description_async(self, issue_id: str) -> str | None:
        return self.issue_descriptions.get(issue_id)

    async def update_issue_description_async(
        self, issue_id: str, description: str
    ) -> bool:
        self.updated_descriptions.append((issue_id, description))
        self.issue_descriptions[issue_id] = description
        return True

    async def update_issue_async(
        self,
        issue_id: str,
        *,
        title: str | None = None,
        priority: str | None = None,
    ) -> bool:
        self.updated_issues.append((issue_id, title, priority))
        return True

    async def create_issue_async(
        self,
        title: str,
        description: str,
        priority: str,
        tags: list[str],
    ) -> str | None:
        issue_id = f"new-{len(self.created_issues) + 1}"
        self.created_issues.append(
            {
                "id": issue_id,
                "title": title,
                "description": description,
                "priority": priority,
                "tags": tags,
            }
        )
        return issue_id


class FakeEventSink:
    """Fake event sink for testing."""

    def __init__(self) -> None:
        self.warnings: list[str] = []

    def on_warning(self, message: str, agent_id: str) -> None:
        self.warnings.append(message)


class TestGetFindingFingerprint:
    """Test _get_finding_fingerprint function."""

    def test_returns_hex_hash(self) -> None:
        """Should return a 16-character hex hash."""
        issue = FakeReviewIssue(
            file="src/foo.py",
            line_start=10,
            line_end=20,
            priority=2,
            title="Test finding",
            body="Body text",
            reviewer="claude",
        )
        result = _get_finding_fingerprint(issue)
        assert len(result) == 16
        assert all(c in "0123456789abcdef" for c in result)

    def test_same_input_same_hash(self) -> None:
        """Same issue data should produce same fingerprint."""
        issue1 = FakeReviewIssue(
            file="src/foo.py",
            line_start=10,
            line_end=20,
            priority=2,
            title="Test finding",
            body="Body text",
            reviewer="claude",
        )
        issue2 = FakeReviewIssue(
            file="src/foo.py",
            line_start=10,
            line_end=20,
            priority=3,  # Different priority shouldn't affect fingerprint
            title="Test finding",
            body="Different body",  # Different body shouldn't affect fingerprint
            reviewer="gemini",  # Different reviewer shouldn't affect fingerprint
        )
        assert _get_finding_fingerprint(issue1) == _get_finding_fingerprint(issue2)

    def test_different_location_different_hash(self) -> None:
        """Different file/line should produce different fingerprint."""
        issue1 = FakeReviewIssue(
            file="src/foo.py",
            line_start=10,
            line_end=20,
            priority=2,
            title="Test finding",
            body="Body",
            reviewer="claude",
        )
        issue2 = FakeReviewIssue(
            file="src/bar.py",
            line_start=10,
            line_end=20,
            priority=2,
            title="Test finding",
            body="Body",
            reviewer="claude",
        )
        assert _get_finding_fingerprint(issue1) != _get_finding_fingerprint(issue2)

    def test_handles_special_characters_in_title(self) -> None:
        """Should handle special characters like --> in title."""
        issue = FakeReviewIssue(
            file="src/foo.py",
            line_start=10,
            line_end=20,
            priority=2,
            title="Code contains --> which could break regex",
            body="Body",
            reviewer="claude",
        )
        result = _get_finding_fingerprint(issue)
        # Should still produce valid hex hash
        assert len(result) == 16
        assert all(c in "0123456789abcdef" for c in result)


class TestExtractExistingFingerprints:
    """Test _extract_existing_fingerprints function."""

    def test_extracts_hex_fingerprints(self) -> None:
        """Should extract hex fingerprints from HTML comments."""
        description = """## Review Findings
        
Some content here.

<!-- fp:abc123def456789a -->
<!-- fp:1234567890abcdef -->
"""
        result = _extract_existing_fingerprints(description)
        assert result == {"abc123def456789a", "1234567890abcdef"}

    def test_empty_description(self) -> None:
        """Should return empty set for description without fingerprints."""
        result = _extract_existing_fingerprints("No fingerprints here")
        assert result == set()

    def test_extracts_legacy_fingerprints(self) -> None:
        """Should extract legacy format fingerprints as hashed values."""
        description = """
<!-- fp:abc123def456789a -->
<!-- fp:file.py:10:20:title -->
<!-- not a fingerprint -->
"""
        result = _extract_existing_fingerprints(description)
        # Hex fingerprints returned as-is, legacy fingerprints are hashed
        # hash of "file.py:10:20:title" = 1b405df404ffe502
        assert result == {"abc123def456789a", "1b405df404ffe502"}

    def test_ignores_arrow_in_content(self) -> None:
        """Should not be confused by --> appearing in content."""
        description = """
Some text with --> arrows.
<!-- fp:abc123def456789a -->
More text with --> here.
"""
        result = _extract_existing_fingerprints(description)
        assert result == {"abc123def456789a"}


class TestUpdateHeaderCount:
    """Test _update_header_count function."""

    def test_updates_plural_count(self) -> None:
        """Should update count in header with plural form."""
        description = (
            "This issue consolidates 5 non-blocking findings from code review."
        )
        result = _update_header_count(description, 7)
        assert (
            result
            == "This issue consolidates 7 non-blocking findings from code review."
        )

    def test_updates_singular_count(self) -> None:
        """Should handle singular form correctly."""
        description = (
            "This issue consolidates 5 non-blocking findings from code review."
        )
        result = _update_header_count(description, 1)
        assert (
            result == "This issue consolidates 1 non-blocking finding from code review."
        )

    def test_singular_to_plural(self) -> None:
        """Should convert from singular to plural."""
        description = "This issue consolidates 1 non-blocking finding from code review."
        result = _update_header_count(description, 3)
        assert (
            result
            == "This issue consolidates 3 non-blocking findings from code review."
        )

    def test_does_not_match_similar_text_in_body(self) -> None:
        """Should not replace similar patterns in finding body."""
        description = """This issue consolidates 2 non-blocking findings from code review.

### Finding 1: Found 3 non-blocking findings in this file

This finding reports that there are 3 non-blocking findings elsewhere.
"""
        result = _update_header_count(description, 5)
        # Only the header should change, not the body
        assert "consolidates 5 non-blocking findings" in result
        assert "Found 3 non-blocking findings in this file" in result

    def test_no_match_returns_unchanged(self) -> None:
        """Should return unchanged if pattern not found."""
        description = "No matching pattern here."
        result = _update_header_count(description, 5)
        assert result == description


class TestCreateReviewTrackingIssues:
    """Test create_review_tracking_issues function."""

    @pytest.mark.asyncio
    async def test_creates_single_consolidated_issue(self) -> None:
        """Should create a single tracking issue consolidating all findings."""
        beads = FakeIssueProvider()
        event_sink = FakeEventSink()

        review_issues = [
            FakeReviewIssue(
                file="src/foo.py",
                line_start=10,
                line_end=10,
                priority=2,
                title="Consider refactoring",
                body="This function is too long",
                reviewer="gemini",
            ),
            FakeReviewIssue(
                file="src/bar.py",
                line_start=20,
                line_end=30,
                priority=3,
                title="Code smell",
                body="Details here",
                reviewer="claude",
            ),
        ]

        await create_review_tracking_issues(
            beads=cast("IssueProvider", beads),
            event_sink=cast("MalaEventSink", event_sink),
            source_issue_id="bd-test-1",
            review_issues=review_issues,
        )

        # Should create exactly one consolidated issue
        assert len(beads.created_issues) == 1
        issue = beads.created_issues[0]
        assert "[Review] 2 non-blocking findings from bd-test-1" in issue["title"]
        # Priority should be the highest (lowest number) - P2
        assert issue["priority"] == "P2"
        assert "auto_generated" in issue["tags"]
        # Description should contain both findings
        assert "src/foo.py:10" in issue["description"]
        assert "src/bar.py:20-30" in issue["description"]
        assert "Consider refactoring" in issue["description"]
        assert "Code smell" in issue["description"]
        assert len(event_sink.warnings) == 1

    @pytest.mark.asyncio
    async def test_creates_issue_with_line_range_in_description(self) -> None:
        """Should format line range correctly in description."""
        beads = FakeIssueProvider()
        event_sink = FakeEventSink()

        review_issues = [
            FakeReviewIssue(
                file="src/bar.py",
                line_start=10,
                line_end=20,
                priority=3,
                title="Code smell",
                body="Details here",
                reviewer="claude",
            )
        ]

        await create_review_tracking_issues(
            beads=cast("IssueProvider", beads),
            event_sink=cast("MalaEventSink", event_sink),
            source_issue_id="bd-test-2",
            review_issues=review_issues,
        )

        assert len(beads.created_issues) == 1
        issue = beads.created_issues[0]
        assert "src/bar.py:10-20" in issue["description"]
        assert issue["priority"] == "P3"
        # Single finding uses singular
        assert "[Review] 1 non-blocking finding from bd-test-2" in issue["title"]

    @pytest.mark.asyncio
    async def test_skips_duplicate_findings_for_same_source(self) -> None:
        """Should skip if exact findings already exist in source issue's tracker."""
        beads = FakeIssueProvider()
        event_sink = FakeEventSink()

        review_issues = [
            FakeReviewIssue(
                file="src/foo.py",
                line_start=10,
                line_end=10,
                priority=2,
                title="Consider refactoring",
                body="This function is too long",
                reviewer="gemini",
            )
        ]

        # Pre-populate existing tracking issue for this source
        import hashlib

        # Content-based dedup tag - fingerprints are now hex hashes
        content = "src/foo.py:10:10:Consider refactoring"
        individual_fp = hashlib.sha256(content.encode()).hexdigest()[:16]
        # For a single finding, pipe-join of one element equals the element itself.
        # The production code uses "|".join(sorted([fp])) which equals just fp.
        content_hash = hashlib.sha256(individual_fp.encode()).hexdigest()[:12]
        dedup_tag = f"review_finding:{content_hash}"

        beads.existing_tags["source:bd-test-3"] = "existing-issue-1"
        beads.issue_descriptions["existing-issue-1"] = (
            f"## Review Findings\n<!-- {dedup_tag} -->"
        )

        await create_review_tracking_issues(
            beads=cast("IssueProvider", beads),
            event_sink=cast("MalaEventSink", event_sink),
            source_issue_id="bd-test-3",
            review_issues=review_issues,
        )

        # Should not create any issues or update (findings already exist)
        assert len(beads.created_issues) == 0
        assert len(beads.updated_descriptions) == 0
        assert len(event_sink.warnings) == 0

    @pytest.mark.asyncio
    async def test_appends_new_findings_to_existing_issue(self) -> None:
        """Should append new findings to existing tracking issue for same source."""
        beads = FakeIssueProvider()
        event_sink = FakeEventSink()

        # First set of findings (already in the existing issue)
        existing_desc = """## Review Findings

This issue consolidates 1 non-blocking finding from code review.

**Source issue:** bd-test-6
**Highest priority:** P2

---

### Finding 1: Old finding

**Priority:** P2
**Reviewer:** claude
**Location:** src/old.py:5

---

<!-- review_finding:abc123 -->
"""
        beads.existing_tags["source:bd-test-6"] = "existing-issue-1"
        beads.issue_descriptions["existing-issue-1"] = existing_desc

        # New findings to append
        new_issues = [
            FakeReviewIssue(
                file="src/new.py",
                line_start=20,
                line_end=20,
                priority=3,
                title="New finding",
                body="Something new",
                reviewer="gemini",
            )
        ]

        await create_review_tracking_issues(
            beads=cast("IssueProvider", beads),
            event_sink=cast("MalaEventSink", event_sink),
            source_issue_id="bd-test-6",
            review_issues=new_issues,
        )

        # Should update, not create
        assert len(beads.created_issues) == 0
        assert len(beads.updated_descriptions) == 1

        updated_id, updated_desc = beads.updated_descriptions[0]
        assert updated_id == "existing-issue-1"
        # Should update count from 1 to 2
        assert "2 non-blocking finding" in updated_desc
        # Should contain the new finding numbered as Finding 2
        assert "### Finding 2: New finding" in updated_desc
        assert "src/new.py:20" in updated_desc
        assert "Something new" in updated_desc
        # Warning should mention appending
        assert len(event_sink.warnings) == 1
        assert "Appended" in event_sink.warnings[0]

    @pytest.mark.asyncio
    async def test_handles_none_priority(self) -> None:
        """Should default to P3 when priority is None."""
        beads = FakeIssueProvider()
        event_sink = FakeEventSink()

        review_issues = [
            FakeReviewIssue(
                file="src/foo.py",
                line_start=1,
                line_end=1,
                priority=None,
                title="Minor issue",
                body="",
                reviewer="test",
            )
        ]

        await create_review_tracking_issues(
            beads=cast("IssueProvider", beads),
            event_sink=cast("MalaEventSink", event_sink),
            source_issue_id="bd-test-4",
            review_issues=review_issues,
        )

        assert len(beads.created_issues) == 1
        assert beads.created_issues[0]["priority"] == "P3"

    @pytest.mark.asyncio
    async def test_empty_issues_list(self) -> None:
        """Should not create issue when no review issues provided."""
        beads = FakeIssueProvider()
        event_sink = FakeEventSink()

        await create_review_tracking_issues(
            beads=cast("IssueProvider", beads),
            event_sink=cast("MalaEventSink", event_sink),
            source_issue_id="bd-test-5",
            review_issues=[],
        )

        assert len(beads.created_issues) == 0
        assert len(event_sink.warnings) == 0


# ============================================================================
# run_config tests
# ============================================================================


class TestBuildEventRunConfig:
    """Test build_event_run_config function."""

    def test_basic_config(self, tmp_path: Path) -> None:
        """Should build basic EventRunConfig."""
        config = build_event_run_config(
            repo_path=tmp_path,
            max_agents=4,
            timeout_seconds=1800,
            max_issues=10,
            max_gate_retries=3,
            max_review_retries=2,
            epic_id="test-epic",
            only_ids={"bd-1", "bd-2"},
            braintrust_enabled=True,
            review_enabled=True,
            review_disabled_reason=None,
            morph_enabled=True,
            prioritize_wip=False,
            orphans_only=False,
            cli_args={"verbose": True},
            morph_disabled_reason=None,
            braintrust_disabled_reason=None,
        )

        assert config.repo_path == str(tmp_path)
        assert config.max_agents == 4
        assert config.timeout_minutes == 30  # 1800 / 60
        assert config.max_issues == 10
        assert config.max_gate_retries == 3
        assert config.max_review_retries == 2
        assert config.epic_id == "test-epic"
        assert config.only_ids is not None
        assert set(config.only_ids) == {"bd-1", "bd-2"}
        assert config.braintrust_enabled is True
        assert config.review_enabled is True
        assert config.morph_enabled is True
        assert config.orphans_only is False

    def test_morph_disabled_no_key(self, tmp_path: Path) -> None:
        """Should set morph_disabled_reason when key missing."""
        config = build_event_run_config(
            repo_path=tmp_path,
            max_agents=4,
            timeout_seconds=1800,
            max_issues=10,
            max_gate_retries=3,
            max_review_retries=2,
            epic_id=None,
            only_ids=None,
            braintrust_enabled=False,
            review_enabled=True,
            review_disabled_reason=None,
            morph_enabled=False,
            prioritize_wip=False,
            orphans_only=False,
            cli_args=None,
            morph_disabled_reason=f"add MORPH_API_KEY to {USER_CONFIG_DIR}/.env",
            braintrust_disabled_reason=None,
        )

        assert config.morph_enabled is False
        assert (
            config.morph_disabled_reason
            == f"add MORPH_API_KEY to {USER_CONFIG_DIR}/.env"
        )

    def test_morph_disabled_by_cli(self, tmp_path: Path) -> None:
        """Should set morph_disabled_reason when --no-morph used."""
        config = build_event_run_config(
            repo_path=tmp_path,
            max_agents=4,
            timeout_seconds=1800,
            max_issues=10,
            max_gate_retries=3,
            max_review_retries=2,
            epic_id=None,
            only_ids=None,
            braintrust_enabled=False,
            review_enabled=True,
            review_disabled_reason=None,
            morph_enabled=False,
            prioritize_wip=False,
            orphans_only=False,
            cli_args={"no_morph": True},
            morph_disabled_reason="--no-morph",
            braintrust_disabled_reason=None,
        )

        assert config.morph_enabled is False
        assert config.morph_disabled_reason == "--no-morph"


class TestBuildRunMetadata:
    """Test build_run_metadata function."""

    def test_basic_metadata(self, tmp_path: Path) -> None:
        """Should build RunMetadata with correct config."""
        with patch(
            "src.infra.io.log_output.run_metadata.configure_debug_logging"
        ) as mock_debug:
            mock_debug.return_value = tmp_path / "debug.log"

            metadata = build_run_metadata(
                repo_path=tmp_path,
                max_agents=4,
                timeout_seconds=1800,
                max_issues=10,
                epic_id="test-epic",
                only_ids={"bd-1"},
                braintrust_enabled=True,
                max_gate_retries=3,
                max_review_retries=2,
                review_enabled=True,
                orphans_only=False,
                cli_args={"verbose": True},
                version="1.0.0",
            )

        assert metadata.repo_path == tmp_path
        assert metadata.version == "1.0.0"
        assert metadata.config.max_agents == 4
        assert metadata.config.timeout_minutes == 30
        assert metadata.config.max_issues == 10
        assert metadata.config.epic_id == "test-epic"
        assert metadata.config.braintrust_enabled is True
        assert metadata.config.max_gate_retries == 3
        assert metadata.config.max_review_retries == 2
        assert metadata.config.review_enabled is True
        assert metadata.config.orphans_only is False

    def test_none_timeout(self, tmp_path: Path) -> None:
        """Should handle None timeout."""
        with patch(
            "src.infra.io.log_output.run_metadata.configure_debug_logging"
        ) as mock_debug:
            mock_debug.return_value = None

            metadata = build_run_metadata(
                repo_path=tmp_path,
                max_agents=None,
                timeout_seconds=None,
                max_issues=None,
                epic_id=None,
                only_ids=None,
                braintrust_enabled=False,
                max_gate_retries=3,
                max_review_retries=2,
                review_enabled=False,
                orphans_only=False,
                cli_args=None,
                version="1.0.0",
            )

        assert metadata.config.timeout_minutes is None
        assert metadata.config.max_agents is None
        assert metadata.config.max_issues is None


# ============================================================================
# prompts tests
# ============================================================================


class TestPromptLoading:
    """Test prompt loading utilities."""

    def test_get_implementer_prompt(self) -> None:
        """Should load implementer prompt."""
        prompt = get_implementer_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        # Verify it contains expected placeholders
        assert "{issue_id}" in prompt or "issue" in prompt.lower()

    def test_get_review_followup_prompt(self) -> None:
        """Should load review followup prompt."""
        prompt = get_review_followup_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_get_fixer_prompt(self) -> None:
        """Should load fixer prompt."""
        prompt = get_fixer_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_prompts_are_cached(self) -> None:
        """Prompts should be cached after first load."""
        # Get prompts twice
        prompt1 = get_implementer_prompt()
        prompt2 = get_implementer_prompt()

        # Should be the exact same object due to caching
        assert prompt1 is prompt2
