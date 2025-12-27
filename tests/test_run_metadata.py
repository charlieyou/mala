"""Unit tests for src/logging/run_metadata.py - RunMetadata and related types.

Tests for:
- RunMetadata serialization/deserialization
- ValidationResult and IssueResolution integration
- Backward compatibility with existing fields
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.logging.run_metadata import (
    IssueRun,
    QualityGateResult,
    RunConfig,
    RunMetadata,
    ValidationResult,
)
from src.validation.spec import (
    IssueResolution,
    ResolutionOutcome,
    ValidationArtifacts,
)


class TestQualityGateResult:
    """Test QualityGateResult dataclass."""

    def test_basic_result(self) -> None:
        result = QualityGateResult(passed=True)
        assert result.passed is True
        assert result.evidence == {}
        assert result.failure_reasons == []

    def test_failed_result(self) -> None:
        result = QualityGateResult(
            passed=False,
            evidence={"pytest_ran": True, "ruff_check_ran": False},
            failure_reasons=["ruff check not run"],
        )
        assert result.passed is False
        assert result.evidence["pytest_ran"] is True
        assert result.evidence["ruff_check_ran"] is False
        assert "ruff check not run" in result.failure_reasons


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_basic_result(self) -> None:
        result = ValidationResult(passed=True)
        assert result.passed is True
        assert result.commands_run == []
        assert result.commands_failed == []
        assert result.artifacts is None
        assert result.coverage_percent is None
        assert result.e2e_passed is None

    def test_result_with_commands(self) -> None:
        result = ValidationResult(
            passed=False,
            commands_run=["pytest", "ruff check"],
            commands_failed=["ruff check"],
            coverage_percent=82.5,
            e2e_passed=True,
        )
        assert result.passed is False
        assert "pytest" in result.commands_run
        assert "ruff check" in result.commands_failed
        assert result.coverage_percent == 82.5
        assert result.e2e_passed is True

    def test_result_with_artifacts(self) -> None:
        artifacts = ValidationArtifacts(
            log_dir=Path("/tmp/logs"),
            worktree_path=Path("/tmp/worktree"),
            worktree_state="kept",
            coverage_report=Path("/tmp/coverage.json"),
        )
        result = ValidationResult(
            passed=True,
            commands_run=["pytest"],
            artifacts=artifacts,
        )
        assert result.artifacts is not None
        assert result.artifacts.log_dir == Path("/tmp/logs")
        assert result.artifacts.worktree_state == "kept"


class TestIssueRun:
    """Test IssueRun dataclass."""

    def test_basic_issue_run(self) -> None:
        issue = IssueRun(
            issue_id="test-1",
            agent_id="agent-123",
            status="success",
            duration_seconds=120.5,
        )
        assert issue.issue_id == "test-1"
        assert issue.agent_id == "agent-123"
        assert issue.status == "success"
        assert issue.duration_seconds == 120.5
        assert issue.session_id is None
        assert issue.validation is None
        assert issue.resolution is None

    def test_issue_run_with_validation(self) -> None:
        validation = ValidationResult(
            passed=True,
            commands_run=["pytest", "ruff check"],
        )
        issue = IssueRun(
            issue_id="test-2",
            agent_id="agent-456",
            status="success",
            duration_seconds=60.0,
            validation=validation,
        )
        assert issue.validation is not None
        assert issue.validation.passed is True

    def test_issue_run_with_resolution(self) -> None:
        resolution = IssueResolution(
            outcome=ResolutionOutcome.NO_CHANGE,
            rationale="Already fixed in previous commit",
        )
        issue = IssueRun(
            issue_id="test-3",
            agent_id="agent-789",
            status="success",
            duration_seconds=30.0,
            resolution=resolution,
        )
        assert issue.resolution is not None
        assert issue.resolution.outcome == ResolutionOutcome.NO_CHANGE


class TestRunConfig:
    """Test RunConfig dataclass."""

    def test_basic_config(self) -> None:
        config = RunConfig(
            max_agents=4,
            timeout_minutes=30,
            max_issues=10,
            epic_id="epic-1",
            only_ids=["issue-1", "issue-2"],
            braintrust_enabled=True,
        )
        assert config.max_agents == 4
        assert config.braintrust_enabled is True

    def test_optional_fields(self) -> None:
        config = RunConfig(
            max_agents=None,
            timeout_minutes=None,
            max_issues=None,
            epic_id=None,
            only_ids=None,
            braintrust_enabled=False,
        )
        assert config.max_gate_retries is None
        assert config.max_review_retries is None
        assert config.codex_review is None


class TestRunMetadata:
    """Test RunMetadata class."""

    @pytest.fixture
    def basic_config(self) -> RunConfig:
        return RunConfig(
            max_agents=2,
            timeout_minutes=60,
            max_issues=None,
            epic_id=None,
            only_ids=None,
            braintrust_enabled=False,
        )

    @pytest.fixture
    def metadata(self, basic_config: RunConfig) -> RunMetadata:
        return RunMetadata(
            repo_path=Path("/tmp/test-repo"),
            config=basic_config,
            version="1.0.0",
        )

    def test_init(self, metadata: RunMetadata) -> None:
        assert metadata.run_id is not None
        assert metadata.started_at is not None
        assert metadata.completed_at is None
        assert metadata.version == "1.0.0"
        assert metadata.issues == {}
        assert metadata.run_validation is None

    def test_record_issue(self, metadata: RunMetadata) -> None:
        issue = IssueRun(
            issue_id="test-1",
            agent_id="agent-1",
            status="success",
            duration_seconds=100.0,
        )
        metadata.record_issue(issue)
        assert "test-1" in metadata.issues
        assert metadata.issues["test-1"].agent_id == "agent-1"

    def test_record_run_validation(self, metadata: RunMetadata) -> None:
        result = ValidationResult(
            passed=True,
            commands_run=["pytest", "ruff check"],
        )
        metadata.record_run_validation(result)
        assert metadata.run_validation is not None
        assert metadata.run_validation.passed is True


class TestRunMetadataSerialization:
    """Test RunMetadata serialization and deserialization."""

    @pytest.fixture
    def config(self) -> RunConfig:
        return RunConfig(
            max_agents=4,
            timeout_minutes=30,
            max_issues=10,
            epic_id="epic-1",
            only_ids=["issue-1"],
            braintrust_enabled=True,
            max_gate_retries=3,
            max_review_retries=2,
            codex_review=True,
        )

    @pytest.fixture
    def metadata_with_issues(self, config: RunConfig) -> RunMetadata:
        metadata = RunMetadata(
            repo_path=Path("/tmp/test-repo"),
            config=config,
            version="1.0.0",
        )

        # Add issue with validation
        validation = ValidationResult(
            passed=True,
            commands_run=["pytest", "ruff check", "ruff format"],
            commands_failed=[],
            artifacts=ValidationArtifacts(
                log_dir=Path("/tmp/logs"),
                worktree_path=Path("/tmp/worktree"),
                worktree_state="removed",
            ),
            coverage_percent=87.5,
            e2e_passed=True,
        )
        issue1 = IssueRun(
            issue_id="test-1",
            agent_id="agent-1",
            status="success",
            duration_seconds=120.0,
            session_id="session-abc",
            log_path="/tmp/logs/agent-1.jsonl",
            quality_gate=QualityGateResult(
                passed=True,
                evidence={"pytest_ran": True, "commit_found": True},
            ),
            validation=validation,
        )
        metadata.record_issue(issue1)

        # Add issue with resolution
        resolution = IssueResolution(
            outcome=ResolutionOutcome.OBSOLETE,
            rationale="Feature was removed in earlier commit",
        )
        issue2 = IssueRun(
            issue_id="test-2",
            agent_id="agent-2",
            status="success",
            duration_seconds=15.0,
            resolution=resolution,
        )
        metadata.record_issue(issue2)

        # Add run-level validation
        run_validation = ValidationResult(
            passed=True,
            commands_run=["e2e tests"],
            e2e_passed=True,
        )
        metadata.record_run_validation(run_validation)

        return metadata

    def test_to_dict_basic(self, config: RunConfig) -> None:
        metadata = RunMetadata(
            repo_path=Path("/tmp/repo"),
            config=config,
            version="1.0.0",
        )
        data = metadata._to_dict()

        assert data["run_id"] == metadata.run_id
        assert data["version"] == "1.0.0"
        assert data["repo_path"] == "/tmp/repo"
        assert data["config"]["max_agents"] == 4
        assert data["issues"] == {}
        assert data["run_validation"] is None

    def test_to_dict_with_issues(self, metadata_with_issues: RunMetadata) -> None:
        data = metadata_with_issues._to_dict()

        # Check issue with validation
        issue1_data = data["issues"]["test-1"]
        assert issue1_data["status"] == "success"
        assert issue1_data["validation"]["passed"] is True
        assert "pytest" in issue1_data["validation"]["commands_run"]
        assert issue1_data["validation"]["coverage_percent"] == 87.5
        assert issue1_data["validation"]["artifacts"]["log_dir"] == "/tmp/logs"

        # Check issue with resolution
        issue2_data = data["issues"]["test-2"]
        assert issue2_data["resolution"]["outcome"] == "obsolete"
        assert "removed" in issue2_data["resolution"]["rationale"]

        # Check run-level validation
        assert data["run_validation"]["passed"] is True
        assert data["run_validation"]["e2e_passed"] is True

    def test_save_and_load_roundtrip(self, metadata_with_issues: RunMetadata) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Override RUNS_DIR for test
            from src.logging import run_metadata

            original_runs_dir = run_metadata.RUNS_DIR
            run_metadata.RUNS_DIR = Path(tmpdir)

            try:
                # Save
                path = metadata_with_issues.save()
                assert path.exists()

                # Load
                loaded = RunMetadata.load(path)

                # Verify basic fields
                assert loaded.run_id == metadata_with_issues.run_id
                assert loaded.version == metadata_with_issues.version
                assert loaded.repo_path == metadata_with_issues.repo_path
                assert loaded.completed_at is not None

                # Verify config
                assert loaded.config.max_agents == 4
                assert loaded.config.braintrust_enabled is True
                assert loaded.config.codex_review is True

                # Verify issue with validation
                assert "test-1" in loaded.issues
                issue1 = loaded.issues["test-1"]
                assert issue1.validation is not None
                assert issue1.validation.passed is True
                assert issue1.validation.coverage_percent == 87.5
                assert issue1.validation.artifacts is not None
                assert issue1.validation.artifacts.log_dir == Path("/tmp/logs")

                # Verify issue with resolution
                assert "test-2" in loaded.issues
                issue2 = loaded.issues["test-2"]
                assert issue2.resolution is not None
                assert issue2.resolution.outcome == ResolutionOutcome.OBSOLETE

                # Verify run-level validation
                assert loaded.run_validation is not None
                assert loaded.run_validation.passed is True
                assert loaded.run_validation.e2e_passed is True

            finally:
                run_metadata.RUNS_DIR = original_runs_dir

    def test_load_handles_missing_optional_fields(self) -> None:
        """Test that load handles files without new optional fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a minimal JSON file (simulating older format)
            minimal_data = {
                "run_id": "test-run-id",
                "started_at": "2025-01-01T00:00:00+00:00",
                "completed_at": "2025-01-01T01:00:00+00:00",
                "version": "0.9.0",
                "repo_path": "/tmp/repo",
                "config": {
                    "max_agents": 2,
                    "timeout_minutes": 30,
                    "max_issues": None,
                    "epic_id": None,
                    "only_ids": None,
                    "braintrust_enabled": False,
                },
                "issues": {
                    "old-issue": {
                        "issue_id": "old-issue",
                        "agent_id": "old-agent",
                        "status": "success",
                        "duration_seconds": 60.0,
                        # No validation, resolution, quality_gate
                    }
                },
                # No run_validation
            }

            path = Path(tmpdir) / "test.json"
            with open(path, "w") as f:
                json.dump(minimal_data, f)

            # Load should work without errors
            loaded = RunMetadata.load(path)

            assert loaded.run_id == "test-run-id"
            assert loaded.version == "0.9.0"
            assert "old-issue" in loaded.issues

            issue = loaded.issues["old-issue"]
            assert issue.validation is None
            assert issue.resolution is None
            assert issue.quality_gate is None

            assert loaded.run_validation is None

    def test_serialization_handles_none_artifacts(self) -> None:
        """Test that serialization handles None artifacts cleanly."""
        config = RunConfig(
            max_agents=1,
            timeout_minutes=10,
            max_issues=None,
            epic_id=None,
            only_ids=None,
            braintrust_enabled=False,
        )
        metadata = RunMetadata(
            repo_path=Path("/tmp/repo"),
            config=config,
            version="1.0.0",
        )

        # Add validation result with None artifacts
        validation = ValidationResult(
            passed=True,
            commands_run=["pytest"],
            artifacts=None,
        )
        issue = IssueRun(
            issue_id="test-1",
            agent_id="agent-1",
            status="success",
            duration_seconds=30.0,
            validation=validation,
        )
        metadata.record_issue(issue)

        data = metadata._to_dict()
        assert data["issues"]["test-1"]["validation"]["artifacts"] is None

    def test_resolution_outcome_serialization(self) -> None:
        """Test all ResolutionOutcome values serialize/deserialize correctly."""
        outcomes = [
            ResolutionOutcome.SUCCESS,
            ResolutionOutcome.NO_CHANGE,
            ResolutionOutcome.OBSOLETE,
        ]

        config = RunConfig(
            max_agents=1,
            timeout_minutes=1,
            max_issues=None,
            epic_id=None,
            only_ids=None,
            braintrust_enabled=False,
        )
        metadata = RunMetadata(
            repo_path=Path("/tmp"),
            config=config,
            version="1.0.0",
        )

        for outcome in outcomes:
            resolution = IssueResolution(
                outcome=outcome,
                rationale=f"Test {outcome.value}",
            )

            # Serialize
            data = metadata._serialize_issue_resolution(resolution)

            # Deserialize
            loaded = RunMetadata._deserialize_issue_resolution(data)

            assert loaded is not None
            assert loaded.outcome == outcome
            assert loaded.rationale == f"Test {outcome.value}"
