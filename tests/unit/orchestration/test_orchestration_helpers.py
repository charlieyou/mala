"""Unit tests for extracted orchestration helpers.

Tests for:
- gate_metadata: GateMetadata building from gate results and logs
- review_tracking: Creating tracking issues from P2/P3 findings
- run_config: Building EventRunConfig and RunMetadata
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from src.domain.evidence_check import GateResult, ValidationEvidence
from src.domain.validation.spec import CommandKind
from src.pipeline.gate_metadata import (
    build_gate_metadata,
    build_gate_metadata_from_logs,
)
from src.orchestration.review_tracking import (
    _extract_existing_fingerprints,
    _get_finding_fingerprint,
    _update_header_count,
    create_review_tracking_issues,
)
from src.orchestration.run_config import build_event_run_config, build_run_metadata
from src.core.protocols.events import MalaEventSink
from tests.fakes.gate_checker import FakeGateChecker
from tests.fakes.issue_provider import FakeIssue, FakeIssueProvider

if TYPE_CHECKING:
    from src.core.protocols.issue import IssueProvider
    from src.core.protocols.validation import GateChecker


# ============================================================================
# gate_metadata tests
# ============================================================================


class TestBuildGateMetadata:
    """Test build_gate_metadata function."""

    def test_none_gate_result(self) -> None:
        """Should return empty metadata when gate_result is None."""
        metadata = build_gate_metadata(None, passed=True)
        assert metadata.evidence_check_result is None
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

        assert metadata.evidence_check_result is not None
        assert metadata.evidence_check_result.passed is True
        assert metadata.evidence_check_result.evidence["commit_found"] is True
        assert metadata.evidence_check_result.failure_reasons == []

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

        assert metadata.evidence_check_result is not None
        assert metadata.evidence_check_result.passed is False
        assert "ruff check failed" in metadata.evidence_check_result.failure_reasons

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
        assert metadata.evidence_check_result is not None
        assert metadata.evidence_check_result.passed is True
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

        assert metadata.evidence_check_result is not None
        assert metadata.evidence_check_result.evidence["commit_found"] is False

    def test_no_validation_evidence(self) -> None:
        """Should handle missing validation evidence."""
        gate_result = GateResult(
            passed=True,
            failure_reasons=[],
            validation_evidence=None,
            commit_hash="abc123",
        )

        metadata = build_gate_metadata(gate_result, passed=True)

        assert metadata.evidence_check_result is not None
        # No validation result when evidence is missing
        assert metadata.validation_result is None


class TestBuildGateMetadataFromLogs:
    """Test build_gate_metadata_from_logs fallback function."""

    def test_none_spec(self, tmp_path: Path) -> None:
        """Should return empty metadata when spec is None."""
        log_path = tmp_path / "session.log"
        log_path.touch()

        gate = FakeGateChecker()
        metadata = build_gate_metadata_from_logs(
            log_path=log_path,
            result_summary="Success",
            result_success=True,
            evidence_check=cast("GateChecker", gate),
            per_session_spec=None,
        )

        assert metadata.evidence_check_result is None
        assert metadata.validation_result is None

    def test_with_spec(self, tmp_path: Path) -> None:
        """Should parse logs and extract metadata when spec is provided."""
        from src.domain.validation.spec import ValidationScope, ValidationSpec

        log_path = tmp_path / "session.log"
        log_path.write_text("mock log content")

        evidence = ValidationEvidence(
            commands_ran={CommandKind.TEST: True},
            failed_commands=[],
        )

        gate = FakeGateChecker(validation_evidence=evidence)
        spec = ValidationSpec(commands=[], scope=ValidationScope.PER_SESSION)

        metadata = build_gate_metadata_from_logs(
            log_path=log_path,
            result_summary="Success",
            result_success=True,
            evidence_check=cast("GateChecker", gate),
            per_session_spec=spec,
        )

        assert metadata.evidence_check_result is not None
        assert metadata.evidence_check_result.passed is True
        assert metadata.validation_result is not None
        # Commands run uses kind.value (e.g., "test")
        assert "test" in metadata.validation_result.commands_run

    def test_failure_reason_extraction(self, tmp_path: Path) -> None:
        """Should extract failure reasons from result summary."""
        from src.domain.validation.spec import ValidationScope, ValidationSpec

        log_path = tmp_path / "session.log"
        log_path.write_text("mock log content")

        evidence = ValidationEvidence(
            commands_ran={CommandKind.TEST: True},
            failed_commands=["pytest"],
        )

        gate = FakeGateChecker(validation_evidence=evidence)
        spec = ValidationSpec(commands=[], scope=ValidationScope.PER_SESSION)

        metadata = build_gate_metadata_from_logs(
            log_path=log_path,
            result_summary="Quality gate failed: tests failed; lint failed",
            result_success=False,
            evidence_check=cast("GateChecker", gate),
            per_session_spec=spec,
        )

        assert metadata.evidence_check_result is not None
        assert "tests failed" in metadata.evidence_check_result.failure_reasons
        assert "lint failed" in metadata.evidence_check_result.failure_reasons


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


# FakeIssueProvider moved to tests/fakes/issue_provider.py
# Import from there: from tests.fakes.issue_provider import FakeIssueProvider


class FakeEventSink(MalaEventSink):
    """Fake event sink for testing."""

    def __init__(self) -> None:
        self.warnings: list[str] = []

    def on_warning(self, message: str, agent_id: str | None = None) -> None:
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
        title = cast("str", issue["title"])
        description = cast("str", issue["description"])
        tags = cast("list[str]", issue["tags"])
        assert "[Review] 2 non-blocking findings from bd-test-1" in title
        # Priority should be the highest (lowest number) - P2
        assert issue["priority"] == "P2"
        assert "auto_generated" in tags
        # Description should contain both findings
        assert "src/foo.py:10" in description
        assert "src/bar.py:20-30" in description
        assert "Consider refactoring" in description
        assert "Code smell" in description
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
        title = cast("str", issue["title"])
        description = cast("str", issue["description"])
        assert "src/bar.py:10-20" in description
        assert issue["priority"] == "P3"
        # Single finding uses singular
        assert "[Review] 1 non-blocking finding from bd-test-2" in title

    @pytest.mark.asyncio
    async def test_skips_duplicate_findings_for_same_source(self) -> None:
        """Should skip if exact findings already exist in source issue's tracker."""
        # Pre-populate existing tracking issue for this source
        import hashlib

        # Content-based dedup tag - fingerprints are now hex hashes
        content = "src/foo.py:10:10:Consider refactoring"
        individual_fp = hashlib.sha256(content.encode()).hexdigest()[:16]
        # For a single finding, pipe-join of one element equals the element itself.
        # The production code uses "|".join(sorted([fp])) which equals just fp.
        content_hash = hashlib.sha256(individual_fp.encode()).hexdigest()[:12]
        dedup_tag = f"review_finding:{content_hash}"

        beads = FakeIssueProvider(
            {
                "existing-issue-1": FakeIssue(
                    id="existing-issue-1",
                    description=f"## Review Findings\n<!-- {dedup_tag} -->",
                    tags=["source:bd-test-3"],
                )
            }
        )
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
        beads = FakeIssueProvider(
            {
                "existing-issue-1": FakeIssue(
                    id="existing-issue-1",
                    description=existing_desc,
                    tags=["source:bd-test-6"],
                )
            }
        )
        event_sink = FakeEventSink()

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

    @pytest.mark.asyncio
    async def test_passes_parent_epic_id_to_create_issue(self) -> None:
        """Should pass parent_epic_id to create_issue_async when provided."""
        beads = FakeIssueProvider()
        event_sink = FakeEventSink()

        review_issues = [
            FakeReviewIssue(
                file="src/foo.py",
                line_start=10,
                line_end=10,
                priority=2,
                title="Test finding",
                body="Test body",
                reviewer="test",
            )
        ]

        await create_review_tracking_issues(
            beads=cast("IssueProvider", beads),
            event_sink=cast("MalaEventSink", event_sink),
            source_issue_id="bd-test-parent",
            review_issues=review_issues,
            parent_epic_id="bd-epic-123",
        )

        assert len(beads.created_issues) == 1
        assert beads.created_issues[0]["parent_id"] == "bd-epic-123"

    @pytest.mark.asyncio
    async def test_parent_epic_id_defaults_to_none(self) -> None:
        """Should not pass parent_id when parent_epic_id not provided."""
        beads = FakeIssueProvider()
        event_sink = FakeEventSink()

        review_issues = [
            FakeReviewIssue(
                file="src/foo.py",
                line_start=10,
                line_end=10,
                priority=2,
                title="Test finding",
                body="Test body",
                reviewer="test",
            )
        ]

        await create_review_tracking_issues(
            beads=cast("IssueProvider", beads),
            event_sink=cast("MalaEventSink", event_sink),
            source_issue_id="bd-test-no-parent",
            review_issues=review_issues,
        )

        assert len(beads.created_issues) == 1
        assert beads.created_issues[0]["parent_id"] is None


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
            only_ids=["bd-1", "bd-2"],
            review_enabled=True,
            review_disabled_reason=None,
            include_wip=False,
            orphans_only=False,
            cli_args={"verbose": True},
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
        assert config.review_enabled is True
        assert config.orphans_only is False
        # Default validation_triggers is None
        assert config.validation_triggers is None

    def test_with_validation_triggers(self, tmp_path: Path) -> None:
        """Should build EventRunConfig with validation triggers summary."""
        from src.domain.validation.config import (
            FailureMode,
            SessionEndTriggerConfig,
            TriggerCommandRef,
            ValidationTriggersConfig,
        )

        triggers = ValidationTriggersConfig(
            session_end=SessionEndTriggerConfig(
                failure_mode=FailureMode.REMEDIATE,
                commands=(
                    TriggerCommandRef(ref="test"),
                    TriggerCommandRef(ref="lint"),
                ),
            ),
        )

        config = build_event_run_config(
            repo_path=tmp_path,
            max_agents=2,
            timeout_seconds=600,
            max_issues=5,
            max_gate_retries=3,
            max_review_retries=2,
            epic_id=None,
            only_ids=None,
            review_enabled=True,
            review_disabled_reason=None,
            include_wip=False,
            orphans_only=False,
            cli_args=None,
            validation_triggers=triggers,
        )

        assert config.validation_triggers is not None
        assert config.validation_triggers.session_end is not None
        assert config.validation_triggers.session_end.enabled is True
        assert config.validation_triggers.session_end.failure_mode == "remediate"
        assert config.validation_triggers.session_end.command_count == 2
        # Other triggers should be None
        assert config.validation_triggers.epic_completion is None
        assert config.validation_triggers.periodic is None


class TestBuildRunMetadata:
    """Test build_run_metadata function."""

    def test_basic_metadata(self, tmp_path: Path) -> None:
        """Should build RunMetadata with correct config."""
        # build_run_metadata creates RunMetadata which may configure debug logging.
        # We test the resulting config values, not whether internal logging was called.
        metadata = build_run_metadata(
            repo_path=tmp_path,
            max_agents=4,
            timeout_seconds=1800,
            max_issues=10,
            epic_id="test-epic",
            only_ids=["bd-1"],
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
        assert metadata.config.max_gate_retries == 3
        assert metadata.config.max_review_retries == 2
        assert metadata.config.review_enabled is True
        assert metadata.config.orphans_only is False

    def test_none_timeout(self, tmp_path: Path) -> None:
        """Should handle None timeout."""
        metadata = build_run_metadata(
            repo_path=tmp_path,
            max_agents=None,
            timeout_seconds=None,
            max_issues=None,
            epic_id=None,
            only_ids=None,
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
