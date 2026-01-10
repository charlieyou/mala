"""Unit tests for RunMetadata and related types.

Tests for:
- RunMetadata serialization/deserialization
- ValidationResult and IssueResolution integration
- Backward compatibility with existing fields
- Running instance tracking (RunningInstance, markers, filtering)
"""

import json
import os
import tempfile
from datetime import datetime, UTC
from pathlib import Path
from unittest.mock import patch

import pytest

from src.infra.io.log_output.run_metadata import (
    IssueRun,
    EvidenceCheckResult,
    RunConfig,
    RunMetadata,
    RunningInstance,
    ValidationResult,
    cleanup_debug_logging,
    configure_debug_logging,
    extract_session_from_run,
    get_running_instances,
    get_running_instances_for_dir,
    lookup_prior_session,
    lookup_prior_session_info,
    parse_timestamp,
    remove_run_marker,
    write_run_marker,
)
from src.core.models import (
    IssueResolution,
    ResolutionOutcome,
    ValidationArtifacts,
)


class TestEvidenceCheckResult:
    """Test EvidenceCheckResult dataclass."""

    def test_failed_result(self) -> None:
        # New spec-driven evidence uses CommandKind values as keys
        result = EvidenceCheckResult(
            passed=False,
            evidence={"test": True, "lint": False},
            failure_reasons=["ruff check not run"],
        )
        assert result.passed is False
        assert result.evidence["test"] is True
        assert result.evidence["lint"] is False
        assert "ruff check not run" in result.failure_reasons


class TestValidationResult:
    """Test ValidationResult dataclass."""

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
            baseline_timestamp=1234567890,
        )
        assert issue.validation is not None
        assert issue.validation.passed is True
        assert issue.baseline_timestamp == 1234567890

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

    def test_issue_run_with_last_review_issues(self) -> None:
        """Test IssueRun with last_review_issues field."""
        review_issues = [
            {"issue": "Missing test coverage", "severity": "warning"},
            {"issue": "Consider adding docstring", "severity": "info"},
        ]
        issue = IssueRun(
            issue_id="test-4",
            agent_id="agent-review",
            status="success",
            duration_seconds=45.0,
            last_review_issues=review_issues,
        )
        assert issue.last_review_issues is not None
        assert len(issue.last_review_issues) == 2
        assert issue.last_review_issues[0]["issue"] == "Missing test coverage"

    def test_issue_run_without_last_review_issues_defaults_to_none(self) -> None:
        """Test IssueRun without last_review_issues has None default."""
        issue = IssueRun(
            issue_id="test-5",
            agent_id="agent-no-review",
            status="success",
            duration_seconds=20.0,
        )
        assert issue.last_review_issues is None


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
        assert metadata.run_start_commit is None
        assert metadata.last_cumulative_review_commits == {}

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
            max_gate_retries=3,
            max_review_retries=2,
            review_enabled=True,
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
            evidence_check=EvidenceCheckResult(
                passed=True,
                evidence={"test": True, "commit_found": True},
            ),
            validation=validation,
            baseline_timestamp=1700000000,
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

        # Add global validation
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
        assert data["run_start_commit"] is None
        assert data["last_cumulative_review_commits"] == {}

    def test_to_dict_with_issues(self, metadata_with_issues: RunMetadata) -> None:
        data = metadata_with_issues._to_dict()

        # Check issue with validation
        issue1_data = data["issues"]["test-1"]
        assert issue1_data["status"] == "success"
        assert issue1_data["validation"]["passed"] is True
        assert "pytest" in issue1_data["validation"]["commands_run"]
        assert issue1_data["validation"]["coverage_percent"] == 87.5
        assert issue1_data["validation"]["artifacts"]["log_dir"] == "/tmp/logs"
        assert issue1_data["baseline_timestamp"] == 1700000000

        # Check issue with resolution
        issue2_data = data["issues"]["test-2"]
        assert issue2_data["resolution"]["outcome"] == "obsolete"
        assert "removed" in issue2_data["resolution"]["rationale"]

        # Check global validation
        assert data["run_validation"]["passed"] is True
        assert data["run_validation"]["e2e_passed"] is True

    def test_save_and_load_roundtrip(self, config: RunConfig) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create metadata with DI for runs_dir
            runs_dir = Path(tmpdir) / "-tmp-test-repo"
            metadata = RunMetadata(
                repo_path=Path("/tmp/test-repo"),
                config=config,
                version="1.0.0",
                runs_dir=runs_dir,
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
                evidence_check=EvidenceCheckResult(
                    passed=True,
                    evidence={"test": True, "commit_found": True},
                ),
                validation=validation,
                baseline_timestamp=1700000001,
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

            # Add global validation
            run_validation = ValidationResult(
                passed=True,
                commands_run=["e2e tests"],
                e2e_passed=True,
            )
            metadata.record_run_validation(run_validation)

            # Save using DI
            path = metadata.save()
            assert path.exists()
            # Verify new filename format: timestamp_shortid.json
            assert "_" in path.name
            assert path.name.endswith(".json")
            # Verify it's in the expected subdirectory
            assert path.parent.name == "-tmp-test-repo"

            # Load
            loaded = RunMetadata.load(path)

            # Verify basic fields
            assert loaded.run_id == metadata.run_id
            assert loaded.version == metadata.version
            assert loaded.repo_path == metadata.repo_path
            assert loaded.completed_at is not None

            # Verify config
            assert loaded.config.max_agents == 4
            assert loaded.config.review_enabled is True

            # Verify issue with validation
            assert "test-1" in loaded.issues
            issue1_loaded = loaded.issues["test-1"]
            assert issue1_loaded.validation is not None
            assert issue1_loaded.validation.passed is True
            assert issue1_loaded.validation.coverage_percent == 87.5
            assert issue1_loaded.validation.artifacts is not None
            assert issue1_loaded.validation.artifacts.log_dir == Path("/tmp/logs")
            assert issue1_loaded.baseline_timestamp == 1700000001

            # Verify issue with resolution
            assert "test-2" in loaded.issues
            issue2_loaded = loaded.issues["test-2"]
            assert issue2_loaded.resolution is not None
            assert issue2_loaded.resolution.outcome == ResolutionOutcome.OBSOLETE

            # Verify global validation
            assert loaded.run_validation is not None
            assert loaded.run_validation.passed is True
            assert loaded.run_validation.e2e_passed is True

    def test_last_review_issues_roundtrip(self, config: RunConfig) -> None:
        """Test that last_review_issues is preserved through save/load round-trip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir) / "-tmp-test-repo"
            metadata = RunMetadata(
                repo_path=Path("/tmp/test-repo"),
                config=config,
                version="1.0.0",
                runs_dir=runs_dir,
            )

            # Create review issues to test round-trip
            review_issues = [
                {"file": "src/foo.py", "issue": "Missing test", "severity": "warning"},
                {"file": "src/bar.py", "issue": "Type error", "severity": "error"},
            ]

            # Add issue with last_review_issues
            issue = IssueRun(
                issue_id="review-test-1",
                agent_id="agent-review",
                status="failed",
                duration_seconds=60.0,
                last_review_issues=review_issues,
            )
            metadata.record_issue(issue)

            # Save and load
            path = metadata.save()
            loaded = RunMetadata.load(path)

            # Verify last_review_issues is preserved
            assert "review-test-1" in loaded.issues
            loaded_issue = loaded.issues["review-test-1"]
            assert loaded_issue.last_review_issues is not None
            assert len(loaded_issue.last_review_issues) == 2
            assert loaded_issue.last_review_issues[0]["file"] == "src/foo.py"
            assert loaded_issue.last_review_issues[0]["issue"] == "Missing test"
            assert loaded_issue.last_review_issues[1]["file"] == "src/bar.py"
            assert loaded_issue.last_review_issues[1]["severity"] == "error"

    def test_last_review_issues_none_roundtrip(self, config: RunConfig) -> None:
        """Test that None last_review_issues is preserved through save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir) / "-tmp-test-repo"
            metadata = RunMetadata(
                repo_path=Path("/tmp/test-repo"),
                config=config,
                version="1.0.0",
                runs_dir=runs_dir,
            )

            # Add issue without last_review_issues
            issue = IssueRun(
                issue_id="no-review-1",
                agent_id="agent-no-review",
                status="success",
                duration_seconds=30.0,
                last_review_issues=None,
            )
            metadata.record_issue(issue)

            # Save and load
            path = metadata.save()
            loaded = RunMetadata.load(path)

            # Verify None is preserved
            assert "no-review-1" in loaded.issues
            loaded_issue = loaded.issues["no-review-1"]
            assert loaded_issue.last_review_issues is None

    def test_save_creates_repo_segmented_directory(self) -> None:
        """Test that save creates files in repo-segmented subdirectories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RunConfig(
                max_agents=1,
                timeout_minutes=10,
                max_issues=None,
                epic_id=None,
                only_ids=None,
            )

            # Test the full save with DI via runs_dir
            runs_dir = Path(tmpdir) / "-home-user-my-project"
            metadata = RunMetadata(
                repo_path=Path("/home/user/my-project"),
                config=config,
                version="1.0.0",
                runs_dir=runs_dir,
            )

            path = metadata.save()

            # Verify file is in correct subdirectory
            assert path.parent.name == "-home-user-my-project"
            assert path.parent.parent == Path(tmpdir)

            # Verify filename format: YYYY-MM-DDTHH-MM-SS_shortid.json
            import re

            pattern = r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}_[a-f0-9]{8}\.json"
            assert re.match(pattern, path.name), (
                f"Filename {path.name} doesn't match expected pattern"
            )

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
                },
                "issues": {
                    "old-issue": {
                        "issue_id": "old-issue",
                        "agent_id": "old-agent",
                        "status": "success",
                        "duration_seconds": 60.0,
                        # No validation, resolution, evidence_check
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
            assert issue.evidence_check is None

            assert loaded.run_validation is None

    def test_serialization_handles_none_artifacts(self) -> None:
        """Test that serialization handles None artifacts cleanly."""
        config = RunConfig(
            max_agents=1,
            timeout_minutes=10,
            max_issues=None,
            epic_id=None,
            only_ids=None,
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

    def test_cumulative_review_fields_serialization(self) -> None:
        """Test run_start_commit and last_cumulative_review_commits round-trip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir) / "-tmp-test-repo"
            config = RunConfig(
                max_agents=1,
                timeout_minutes=10,
                max_issues=None,
                epic_id=None,
                only_ids=None,
            )
            metadata = RunMetadata(
                repo_path=Path("/tmp/test-repo"),
                config=config,
                version="1.0.0",
                runs_dir=runs_dir,
            )

            # Set cumulative review fields
            metadata.run_start_commit = "abc123def456"
            metadata.last_cumulative_review_commits = {
                "run_end": "def789abc012",
                "epic_completion:epic-1": "fed321cba654",
            }

            # Verify _to_dict serializes correctly
            data = metadata._to_dict()
            assert data["run_start_commit"] == "abc123def456"
            assert data["last_cumulative_review_commits"] == {
                "run_end": "def789abc012",
                "epic_completion:epic-1": "fed321cba654",
            }

            # Save and load
            path = metadata.save()
            loaded = RunMetadata.load(path)

            # Verify loaded values match
            assert loaded.run_start_commit == "abc123def456"
            assert loaded.last_cumulative_review_commits == {
                "run_end": "def789abc012",
                "epic_completion:epic-1": "fed321cba654",
            }

    def test_cumulative_review_fields_backward_compat(self) -> None:
        """Test loading old metadata files without cumulative review fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create JSON without cumulative review fields (old format)
            old_data = {
                "run_id": "old-run-id",
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
                },
                "issues": {},
                # No run_start_commit or last_cumulative_review_commits
            }

            path = Path(tmpdir) / "old_format.json"
            with open(path, "w") as f:
                json.dump(old_data, f)

            # Load should succeed with defaults
            loaded = RunMetadata.load(path)

            assert loaded.run_start_commit is None
            assert loaded.last_cumulative_review_commits == {}


class TestRunningInstance:
    """Test RunningInstance dataclass and related functions."""

    def test_running_instance_creation(self) -> None:
        """Test creating a RunningInstance."""
        instance = RunningInstance(
            run_id="test-run-123",
            repo_path=Path("/home/user/repo"),
            started_at=datetime.now(UTC),
            pid=12345,
            max_agents=4,
        )
        assert instance.run_id == "test-run-123"
        assert instance.repo_path == Path("/home/user/repo")
        assert instance.pid == 12345
        assert instance.max_agents == 4
        assert instance.issues_in_progress == 0

    def test_running_instance_defaults(self) -> None:
        """Test RunningInstance default values."""
        instance = RunningInstance(
            run_id="test",
            repo_path=Path("/tmp"),
            started_at=datetime.now(UTC),
            pid=1,
        )
        assert instance.max_agents is None
        assert instance.issues_in_progress == 0


class TestRunMarkers:
    """Test run marker file operations."""

    def test_write_and_remove_marker(self) -> None:
        """Test writing and removing a run marker."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_dir = Path(tmpdir)

            # Write marker using DI
            path = write_run_marker(
                run_id="test-run-1",
                repo_path=Path("/home/user/project"),
                max_agents=3,
                lock_dir=lock_dir,
            )

            assert path.exists()
            assert path.name == "run-test-run-1.marker"

            # Verify contents
            with open(path) as f:
                data = json.load(f)
            assert data["run_id"] == "test-run-1"
            assert data["repo_path"] == "/home/user/project"
            assert data["max_agents"] == 3
            assert "started_at" in data
            assert "pid" in data

            # Remove marker using DI
            removed = remove_run_marker("test-run-1", lock_dir=lock_dir)
            assert removed is True
            assert not path.exists()

            # Remove non-existent marker
            removed_again = remove_run_marker("test-run-1", lock_dir=lock_dir)
            assert removed_again is False

    def test_marker_path_format(self) -> None:
        """Test that markers are created with expected path format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_dir = Path(tmpdir)
            # Write marker and verify path format through public API
            path = write_run_marker(
                run_id="my-run-id",
                repo_path=Path("/tmp/repo"),
                lock_dir=lock_dir,
            )
            assert path == lock_dir / "run-my-run-id.marker"
            assert path.exists()


class TestGetRunningInstances:
    """Test get_running_instances and get_running_instances_for_dir."""

    def test_get_running_instances_empty(self) -> None:
        """Test with no markers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_dir = Path(tmpdir)
            # Use DI instead of patch
            instances = get_running_instances(lock_dir=lock_dir)
            assert instances == []

    def test_get_running_instances_nonexistent_dir(self) -> None:
        """Test with non-existent lock directory."""
        # Use DI instead of patch
        instances = get_running_instances(lock_dir=Path("/nonexistent/path"))
        assert instances == []

    def test_get_running_instances_with_markers(self) -> None:
        """Test reading valid markers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_dir = Path(tmpdir)

            # Create marker files
            marker1 = lock_dir / "run-test-1.marker"
            marker1.write_text(
                json.dumps(
                    {
                        "run_id": "test-1",
                        "repo_path": "/home/user/repo1",
                        "started_at": datetime.now(UTC).isoformat(),
                        "pid": os.getpid(),  # Use current PID so it's "running"
                        "max_agents": 2,
                    }
                )
            )

            marker2 = lock_dir / "run-test-2.marker"
            marker2.write_text(
                json.dumps(
                    {
                        "run_id": "test-2",
                        "repo_path": "/home/user/repo2",
                        "started_at": datetime.now(UTC).isoformat(),
                        "pid": os.getpid(),
                        "max_agents": None,
                    }
                )
            )

            # Use DI instead of patch
            instances = get_running_instances(lock_dir=lock_dir)

            assert len(instances) == 2
            run_ids = {i.run_id for i in instances}
            assert run_ids == {"test-1", "test-2"}

    def test_get_running_instances_cleans_stale_markers(self) -> None:
        """Test that stale markers (dead PIDs) are cleaned up."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_dir = Path(tmpdir)

            # Create marker with non-existent PID
            marker = lock_dir / "run-stale.marker"
            marker.write_text(
                json.dumps(
                    {
                        "run_id": "stale",
                        "repo_path": "/tmp/stale",
                        "started_at": datetime.now(UTC).isoformat(),
                        "pid": 99999999,  # Very unlikely to exist
                        "max_agents": 1,
                    }
                )
            )

            # Use DI for both lock_dir and is_process_running
            instances = get_running_instances(
                lock_dir=lock_dir,
                is_process_running=lambda pid: False,  # All processes are "dead"
            )

            # Should return no instances and clean up the marker
            assert instances == []
            assert not marker.exists()

    def test_get_running_instances_handles_corrupted_markers(self) -> None:
        """Test that corrupted markers are cleaned up."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_dir = Path(tmpdir)

            # Create invalid JSON marker
            bad_marker = lock_dir / "run-bad.marker"
            bad_marker.write_text("not valid json")

            # Create valid marker
            good_marker = lock_dir / "run-good.marker"
            good_marker.write_text(
                json.dumps(
                    {
                        "run_id": "good",
                        "repo_path": "/tmp/good",
                        "started_at": datetime.now(UTC).isoformat(),
                        "pid": os.getpid(),
                        "max_agents": 1,
                    }
                )
            )

            # Use DI instead of patch
            instances = get_running_instances(lock_dir=lock_dir)

            # Should return only the good instance
            assert len(instances) == 1
            assert instances[0].run_id == "good"
            # Bad marker should be cleaned up
            assert not bad_marker.exists()

    def test_get_running_instances_for_dir_filters_correctly(self) -> None:
        """Test directory filtering logic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_dir = Path(tmpdir)
            target_dir = Path(tmpdir) / "target-repo"
            other_dir = Path(tmpdir) / "other-repo"
            target_dir.mkdir()
            other_dir.mkdir()

            # Create markers for different directories
            marker1 = lock_dir / "run-target.marker"
            marker1.write_text(
                json.dumps(
                    {
                        "run_id": "target",
                        "repo_path": str(target_dir),
                        "started_at": datetime.now(UTC).isoformat(),
                        "pid": os.getpid(),
                        "max_agents": 2,
                    }
                )
            )

            marker2 = lock_dir / "run-other.marker"
            marker2.write_text(
                json.dumps(
                    {
                        "run_id": "other",
                        "repo_path": str(other_dir),
                        "started_at": datetime.now(UTC).isoformat(),
                        "pid": os.getpid(),
                        "max_agents": 1,
                    }
                )
            )

            # Use DI instead of patch - filter for target directory
            target_instances = get_running_instances_for_dir(
                target_dir, lock_dir=lock_dir
            )
            assert len(target_instances) == 1
            assert target_instances[0].run_id == "target"

            # Filter for other directory
            other_instances = get_running_instances_for_dir(
                other_dir, lock_dir=lock_dir
            )
            assert len(other_instances) == 1
            assert other_instances[0].run_id == "other"

            # Filter for non-matching directory
            no_instances = get_running_instances_for_dir(
                Path("/nonexistent"), lock_dir=lock_dir
            )
            assert no_instances == []


class TestProcessDetection:
    """Test process liveness detection through public API."""

    def test_current_process_detected_as_running(self) -> None:
        """Test that markers with current PID are returned as running instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_dir = Path(tmpdir)

            # Create marker with current PID
            marker = lock_dir / "run-current.marker"
            marker.write_text(
                json.dumps(
                    {
                        "run_id": "current",
                        "repo_path": "/tmp/repo",
                        "started_at": datetime.now(UTC).isoformat(),
                        "pid": os.getpid(),  # Current process
                        "max_agents": 1,
                    }
                )
            )

            # Should detect current process as running
            instances = get_running_instances(lock_dir=lock_dir)
            assert len(instances) == 1
            assert instances[0].run_id == "current"

    def test_nonexistent_pid_detected_as_stale(self) -> None:
        """Test that markers with non-existent PID are cleaned up."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_dir = Path(tmpdir)

            # Create marker with non-existent PID
            marker = lock_dir / "run-stale.marker"
            marker.write_text(
                json.dumps(
                    {
                        "run_id": "stale",
                        "repo_path": "/tmp/repo",
                        "started_at": datetime.now(UTC).isoformat(),
                        "pid": 99999999,  # Very unlikely to exist
                        "max_agents": 1,
                    }
                )
            )

            # Should detect as stale and clean up
            instances = get_running_instances(lock_dir=lock_dir)
            assert instances == []
            assert not marker.exists()

    def test_custom_process_checker_via_di(self) -> None:
        """Test that custom is_process_running can be injected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_dir = Path(tmpdir)

            # Create marker with arbitrary PID
            marker = lock_dir / "run-test.marker"
            marker.write_text(
                json.dumps(
                    {
                        "run_id": "test",
                        "repo_path": "/tmp/repo",
                        "started_at": datetime.now(UTC).isoformat(),
                        "pid": 12345,
                        "max_agents": 1,
                    }
                )
            )

            # With custom checker that says all processes are alive
            instances = get_running_instances(
                lock_dir=lock_dir,
                is_process_running=lambda pid: True,
            )
            assert len(instances) == 1
            assert instances[0].pid == 12345

            # With custom checker that says all processes are dead
            instances = get_running_instances(
                lock_dir=lock_dir,
                is_process_running=lambda pid: False,
            )
            assert instances == []


class TestDebugLogging:
    """Test debug logging configuration and cleanup."""

    def test_configure_and_cleanup_debug_logging(self) -> None:
        """Test that configure_debug_logging adds handler and cleanup removes it."""
        import logging

        with tempfile.TemporaryDirectory() as tmpdir:
            # Temporarily enable debug logging for this test
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("MALA_DISABLE_DEBUG_LOG", None)

                run_id = "test-run-12345678"

                # Configure debug logging using DI (runs_dir parameter)
                log_path = configure_debug_logging(
                    Path("/tmp/repo"), run_id, runs_dir=Path(tmpdir)
                )

                # Verify handler was added
                src_logger = logging.getLogger("src")
                handler_names = [getattr(h, "name", "") for h in src_logger.handlers]
                assert f"mala_debug_{run_id}" in handler_names
                assert log_path is not None
                assert log_path.exists()

                # Cleanup debug logging
                cleaned = cleanup_debug_logging(run_id)
                assert cleaned is True

                # Verify handler was removed
                handler_names_after = [
                    getattr(h, "name", "") for h in src_logger.handlers
                ]
                assert f"mala_debug_{run_id}" not in handler_names_after

    def test_cleanup_nonexistent_handler_returns_false(self) -> None:
        """Test that cleanup returns False when handler doesn't exist."""
        cleaned = cleanup_debug_logging("nonexistent-run-id")
        assert cleaned is False

    def test_save_cleans_up_debug_handler(self) -> None:
        """Test that RunMetadata.save() cleans up the debug handler."""
        import logging

        with tempfile.TemporaryDirectory() as tmpdir:
            # Temporarily enable debug logging for this test
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("MALA_DISABLE_DEBUG_LOG", None)

                config = RunConfig(
                    max_agents=1,
                    timeout_minutes=10,
                    max_issues=None,
                    epic_id=None,
                    only_ids=None,
                )
                # Use DI via runs_dir parameter
                metadata = RunMetadata(
                    repo_path=Path("/tmp/test-repo"),
                    config=config,
                    version="1.0.0",
                    runs_dir=Path(tmpdir),
                )

                # Verify handler was added
                src_logger = logging.getLogger("src")
                handler_names = [getattr(h, "name", "") for h in src_logger.handlers]
                assert any(name.startswith("mala_debug_") for name in handler_names)

                # Save should clean up the handler
                metadata.save()

                # Verify handler was removed
                handler_names_after = [
                    getattr(h, "name", "") for h in src_logger.handlers
                ]
                assert not any(
                    name == f"mala_debug_{metadata.run_id}"
                    for name in handler_names_after
                )

    def test_configure_removes_previous_handlers(self) -> None:
        """Test that configuring a new handler removes old ones."""
        import logging

        with tempfile.TemporaryDirectory() as tmpdir:
            # Temporarily enable debug logging for this test
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("MALA_DISABLE_DEBUG_LOG", None)

                # Configure first handler using DI
                run_id_1 = "first-run-12345678"
                configure_debug_logging(
                    Path("/tmp/repo"), run_id_1, runs_dir=Path(tmpdir)
                )

                src_logger = logging.getLogger("src")
                handler_names = [getattr(h, "name", "") for h in src_logger.handlers]
                assert f"mala_debug_{run_id_1}" in handler_names

                # Configure second handler - should remove first
                run_id_2 = "second-run-87654321"
                configure_debug_logging(
                    Path("/tmp/repo"), run_id_2, runs_dir=Path(tmpdir)
                )

                handler_names_after = [
                    getattr(h, "name", "") for h in src_logger.handlers
                ]
                assert f"mala_debug_{run_id_1}" not in handler_names_after
                assert f"mala_debug_{run_id_2}" in handler_names_after

                # Clean up
                cleanup_debug_logging(run_id_2)

    def test_configure_disabled_by_env_var(self) -> None:
        """Test that MALA_DISABLE_DEBUG_LOG=1 disables debug logging."""
        with patch.dict(os.environ, {"MALA_DISABLE_DEBUG_LOG": "1"}):
            log_path = configure_debug_logging(Path("/tmp/repo"), "test-run-id")
            assert log_path is None

    def test_configure_handles_permission_error(self) -> None:
        """Test that configure_debug_logging handles permission errors gracefully."""
        with patch(
            "src.infra.io.log_output.run_metadata.get_repo_runs_dir",
            side_effect=PermissionError("Access denied"),
        ):
            log_path = configure_debug_logging(Path("/tmp/repo"), "test-run-id")
            assert log_path is None

    def test_configure_handles_readonly_filesystem(self) -> None:
        """Test that configure_debug_logging handles read-only filesystem errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create the readonly directory
            readonly_path = Path(tmpdir) / "readonly"
            readonly_path.mkdir(exist_ok=True)

            # Use DI to specify runs_dir, but mock FileHandler to simulate read-only fs
            import logging

            with patch.object(
                logging,
                "FileHandler",
                side_effect=OSError("Read-only file system"),
            ):
                log_path = configure_debug_logging(
                    Path("/tmp/repo"), "test-run", runs_dir=readonly_path
                )
                assert log_path is None

    def test_cleanup_is_idempotent(self) -> None:
        """Test that RunMetadata.cleanup() is idempotent (safe to call multiple times)."""
        import logging

        with tempfile.TemporaryDirectory() as tmpdir:
            # Temporarily enable debug logging for this test
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("MALA_DISABLE_DEBUG_LOG", None)

                config = RunConfig(
                    max_agents=1,
                    timeout_minutes=10,
                    max_issues=None,
                    epic_id=None,
                    only_ids=None,
                )
                # Use DI via runs_dir parameter
                metadata = RunMetadata(
                    repo_path=Path("/tmp/test-repo"),
                    config=config,
                    version="1.0.0",
                    runs_dir=Path(tmpdir),
                )

                try:
                    # Verify debug logging was configured
                    assert metadata.debug_log_path is not None
                    src_logger = logging.getLogger("src")
                    handler_name = f"mala_debug_{metadata.run_id}"
                    handler_names = [
                        getattr(h, "name", "") for h in src_logger.handlers
                    ]
                    assert handler_name in handler_names

                    # First cleanup removes the handler
                    metadata.cleanup()
                    handler_names_after = [
                        getattr(h, "name", "") for h in src_logger.handlers
                    ]
                    assert handler_name not in handler_names_after

                    # Subsequent cleanups are safe (no-op, no errors)
                    metadata.cleanup()
                    metadata.cleanup()

                    # And save still works after cleanup
                    path = metadata.save()
                    assert path.exists()
                finally:
                    # Ensure handler is always cleaned up, even if assertions fail
                    metadata.cleanup()


class TestLookupPriorSession:
    """Test lookup_prior_session function."""

    def test_lookup_finds_recent_session(self) -> None:
        """Test that lookup returns the most recent session_id for an issue."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            runs_dir = repo_path / ".mala" / "runs"
            runs_dir.mkdir(parents=True)

            # Create two run files with different timestamps
            older_run = {
                "run_id": "run-1",
                "started_at": "2024-01-01T10:00:00Z",
                "issues": {"issue-123": {"session_id": "old-session-abc"}},
            }
            newer_run = {
                "run_id": "run-2",
                "started_at": "2024-01-02T10:00:00Z",
                "issues": {"issue-123": {"session_id": "new-session-xyz"}},
            }

            (runs_dir / "2024-01-01T10-00-00_run1.json").write_text(
                json.dumps(older_run)
            )
            (runs_dir / "2024-01-02T10-00-00_run2.json").write_text(
                json.dumps(newer_run)
            )

            with patch(
                "src.infra.io.log_output.run_metadata.get_repo_runs_dir",
                return_value=runs_dir,
            ):
                result = lookup_prior_session(repo_path, "issue-123")

            assert result == "new-session-xyz"

    def test_lookup_info_includes_baseline_timestamp(self) -> None:
        """Test that lookup_prior_session_info returns baseline_timestamp."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            runs_dir = repo_path / ".mala" / "runs"
            runs_dir.mkdir(parents=True)

            run_data = {
                "run_id": "run-1",
                "started_at": "2024-01-01T10:00:00Z",
                "issues": {
                    "issue-123": {
                        "session_id": "session-abc",
                        "baseline_timestamp": 1700000000,
                    }
                },
            }

            (runs_dir / "2024-01-01T10-00-00_run1.json").write_text(
                json.dumps(run_data)
            )

            with patch(
                "src.infra.io.log_output.run_metadata.get_repo_runs_dir",
                return_value=runs_dir,
            ):
                result = lookup_prior_session_info(repo_path, "issue-123")

            assert result is not None
            assert result.session_id == "session-abc"
            assert result.baseline_timestamp == 1700000000

    def test_lookup_returns_none_if_missing(self) -> None:
        """Test that lookup returns None when no matching issue found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            runs_dir = repo_path / ".mala" / "runs"
            runs_dir.mkdir(parents=True)

            # Create a run file with a different issue
            run_data = {
                "run_id": "run-1",
                "started_at": "2024-01-01T10:00:00Z",
                "issues": {"other-issue": {"session_id": "session-abc"}},
            }
            (runs_dir / "2024-01-01T10-00-00_run1.json").write_text(
                json.dumps(run_data)
            )

            with patch(
                "src.infra.io.log_output.run_metadata.get_repo_runs_dir",
                return_value=runs_dir,
            ):
                result = lookup_prior_session(repo_path, "issue-123")

            assert result is None

    def test_lookup_returns_none_if_dir_missing(self) -> None:
        """Test that lookup returns None when runs directory doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            nonexistent_dir = repo_path / ".mala" / "runs"
            # Don't create the directory

            with patch(
                "src.infra.io.log_output.run_metadata.get_repo_runs_dir",
                return_value=nonexistent_dir,
            ):
                result = lookup_prior_session(repo_path, "issue-123")

            assert result is None

    def test_lookup_ignores_runs_without_session_id(self) -> None:
        """Test that runs without session_id are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            runs_dir = repo_path / ".mala" / "runs"
            runs_dir.mkdir(parents=True)

            # Run with session_id
            with_session = {
                "run_id": "run-1",
                "started_at": "2024-01-01T10:00:00Z",
                "issues": {"issue-123": {"session_id": "valid-session"}},
            }
            # Run without session_id (newer)
            without_session = {
                "run_id": "run-2",
                "started_at": "2024-01-02T10:00:00Z",
                "issues": {"issue-123": {"status": "done"}},  # No session_id
            }
            # Run with null session_id (even newer)
            null_session = {
                "run_id": "run-3",
                "started_at": "2024-01-03T10:00:00Z",
                "issues": {"issue-123": {"session_id": None}},
            }
            # Run with empty session_id
            empty_session = {
                "run_id": "run-4",
                "started_at": "2024-01-04T10:00:00Z",
                "issues": {"issue-123": {"session_id": ""}},
            }

            (runs_dir / "run1.json").write_text(json.dumps(with_session))
            (runs_dir / "run2.json").write_text(json.dumps(without_session))
            (runs_dir / "run3.json").write_text(json.dumps(null_session))
            (runs_dir / "run4.json").write_text(json.dumps(empty_session))

            with patch(
                "src.infra.io.log_output.run_metadata.get_repo_runs_dir",
                return_value=runs_dir,
            ):
                result = lookup_prior_session(repo_path, "issue-123")

            # Should return the only valid session_id
            assert result == "valid-session"

    def test_lookup_handles_corrupt_json(self) -> None:
        """Test that corrupt JSON files are skipped gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            runs_dir = repo_path / ".mala" / "runs"
            runs_dir.mkdir(parents=True)

            # Create a corrupt JSON file
            (runs_dir / "corrupt.json").write_text("not valid json {{{")

            # Create a valid run file
            valid_run = {
                "run_id": "run-1",
                "started_at": "2024-01-01T10:00:00Z",
                "issues": {"issue-123": {"session_id": "valid-session"}},
            }
            (runs_dir / "valid.json").write_text(json.dumps(valid_run))

            with patch(
                "src.infra.io.log_output.run_metadata.get_repo_runs_dir",
                return_value=runs_dir,
            ):
                # Should not raise, and should return the valid session
                result = lookup_prior_session(repo_path, "issue-123")

            assert result == "valid-session"

    def test_lookup_sorts_by_started_at_timestamp(self) -> None:
        """Test that sessions are sorted by started_at, not filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            runs_dir = repo_path / ".mala" / "runs"
            runs_dir.mkdir(parents=True)

            # Create runs with timestamps in opposite order of filenames
            # File z.json has older timestamp
            older_run = {
                "run_id": "run-old",
                "started_at": "2024-01-01T10:00:00Z",
                "issues": {"issue-123": {"session_id": "old-session"}},
            }
            # File a.json has newer timestamp
            newer_run = {
                "run_id": "run-new",
                "started_at": "2024-01-02T10:00:00Z",
                "issues": {"issue-123": {"session_id": "new-session"}},
            }

            # Intentionally name files opposite to timestamp order
            (runs_dir / "z_run.json").write_text(json.dumps(older_run))
            (runs_dir / "a_run.json").write_text(json.dumps(newer_run))

            with patch(
                "src.infra.io.log_output.run_metadata.get_repo_runs_dir",
                return_value=runs_dir,
            ):
                result = lookup_prior_session(repo_path, "issue-123")

            # Should return newer session based on timestamp, not filename
            assert result == "new-session"

    def test_lookup_handles_timezone_formats(self) -> None:
        """Test that both Z and +00:00 timezone formats work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            runs_dir = repo_path / ".mala" / "runs"
            runs_dir.mkdir(parents=True)

            # Run with Z suffix
            run_z = {
                "run_id": "run-z",
                "started_at": "2024-01-01T10:00:00Z",
                "issues": {"issue-123": {"session_id": "session-z"}},
            }
            # Run with +00:00 suffix (newer)
            run_offset = {
                "run_id": "run-offset",
                "started_at": "2024-01-02T10:00:00+00:00",
                "issues": {"issue-123": {"session_id": "session-offset"}},
            }

            (runs_dir / "run_z.json").write_text(json.dumps(run_z))
            (runs_dir / "run_offset.json").write_text(json.dumps(run_offset))

            with patch(
                "src.infra.io.log_output.run_metadata.get_repo_runs_dir",
                return_value=runs_dir,
            ):
                result = lookup_prior_session(repo_path, "issue-123")

            assert result == "session-offset"

    def test_lookup_handles_malformed_data_structures(self) -> None:
        """Test that malformed data structures are handled gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            runs_dir = repo_path / ".mala" / "runs"
            runs_dir.mkdir(parents=True)

            # issues is a list instead of dict
            bad_issues_type = {
                "run_id": "run-1",
                "started_at": "2024-01-01T10:00:00Z",
                "issues": ["issue-123"],
            }
            # issue_data is a string instead of dict
            bad_issue_data = {
                "run_id": "run-2",
                "started_at": "2024-01-02T10:00:00Z",
                "issues": {"issue-123": "not-a-dict"},
            }
            # Top-level is an array instead of dict
            (runs_dir / "array.json").write_text('[{"run_id": "x"}]')

            # Valid run
            valid_run = {
                "run_id": "run-3",
                "started_at": "2024-01-03T10:00:00Z",
                "issues": {"issue-123": {"session_id": "valid-session"}},
            }

            (runs_dir / "bad1.json").write_text(json.dumps(bad_issues_type))
            (runs_dir / "bad2.json").write_text(json.dumps(bad_issue_data))
            (runs_dir / "valid.json").write_text(json.dumps(valid_run))

            with patch(
                "src.infra.io.log_output.run_metadata.get_repo_runs_dir",
                return_value=runs_dir,
            ):
                result = lookup_prior_session(repo_path, "issue-123")

            assert result == "valid-session"

    def test_lookup_deterministic_on_timestamp_ties(self) -> None:
        """Test that lookup uses run_id as tiebreaker for same timestamps."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            runs_dir = repo_path / ".mala" / "runs"
            runs_dir.mkdir(parents=True)

            # Create two runs with identical timestamps but different run_ids
            run_a = {
                "run_id": "aaa-run",
                "started_at": "2024-01-01T10:00:00Z",
                "issues": {"issue-123": {"session_id": "session-aaa"}},
            }
            run_z = {
                "run_id": "zzz-run",
                "started_at": "2024-01-01T10:00:00Z",  # Same timestamp
                "issues": {"issue-123": {"session_id": "session-zzz"}},
            }

            (runs_dir / "run_a.json").write_text(json.dumps(run_a))
            (runs_dir / "run_z.json").write_text(json.dumps(run_z))

            with patch(
                "src.infra.io.log_output.run_metadata.get_repo_runs_dir",
                return_value=runs_dir,
            ):
                result = lookup_prior_session(repo_path, "issue-123")

            # Should return session from run_id "aaa-run" (sorts first alphabetically)
            assert result == "session-aaa"

    def test_lookup_logs_warning_for_corrupt_files(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that corrupt JSON files are logged as warnings."""
        import logging

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            runs_dir = repo_path / ".mala" / "runs"
            runs_dir.mkdir(parents=True)

            # Create a corrupt JSON file
            corrupt_path = runs_dir / "corrupt.json"
            corrupt_path.write_text("not valid json {{{")

            # Create a valid run file
            valid_run = {
                "run_id": "run-1",
                "started_at": "2024-01-01T10:00:00Z",
                "issues": {"issue-123": {"session_id": "valid-session"}},
            }
            (runs_dir / "valid.json").write_text(json.dumps(valid_run))

            with patch(
                "src.infra.io.log_output.run_metadata.get_repo_runs_dir",
                return_value=runs_dir,
            ):
                caplog.set_level(logging.WARNING)
                result = lookup_prior_session(repo_path, "issue-123")

            assert result == "valid-session"
            # Check that a warning was logged for the corrupt file
            assert any(
                "Skipping corrupt file" in record.message
                and "corrupt.json" in record.message
                for record in caplog.records
            )


class TestParseTimestamp:
    """Test parse_timestamp function (canonical timestamp parsing)."""

    def test_parses_z_suffix(self) -> None:
        """Test parsing timestamp with Z suffix."""
        result = parse_timestamp("2024-01-01T10:00:00Z")
        assert result > 0

    def test_parses_offset_suffix(self) -> None:
        """Test parsing timestamp with +00:00 suffix."""
        result = parse_timestamp("2024-01-01T10:00:00+00:00")
        assert result > 0

    def test_z_and_offset_are_equal(self) -> None:
        """Test that Z and +00:00 produce the same result."""
        z_result = parse_timestamp("2024-01-01T10:00:00Z")
        offset_result = parse_timestamp("2024-01-01T10:00:00+00:00")
        assert z_result == offset_result

    def test_returns_zero_for_invalid(self) -> None:
        """Test that invalid timestamps return 0.0."""
        assert parse_timestamp("not a timestamp") == 0.0
        assert parse_timestamp("") == 0.0

    def test_handles_none_gracefully(self) -> None:
        """Test that None input returns 0.0 (via TypeError)."""
        # The function catches TypeError for this case
        assert parse_timestamp(None) == 0.0  # type: ignore[arg-type]


class TestExtractSessionFromRun:
    """Test extract_session_from_run function for SessionInfo extraction."""

    def test_extracts_last_review_issues(self, tmp_path: Path) -> None:
        """Test that last_review_issues is extracted into SessionInfo."""
        review_issues = [
            {"issue": "Missing error handling", "severity": "error"},
            {"issue": "Add docstring", "severity": "info"},
        ]
        data = {
            "run_id": "run-123",
            "started_at": "2024-01-15T10:00:00Z",
            "issues": {
                "issue-abc": {
                    "session_id": "session-xyz",
                    "status": "success",
                    "last_review_issues": review_issues,
                }
            },
        }
        path = tmp_path / "run.json"

        result = extract_session_from_run(data, path, "issue-abc")

        assert result is not None
        assert result.last_review_issues is not None
        assert len(result.last_review_issues) == 2
        assert result.last_review_issues[0]["issue"] == "Missing error handling"

    def test_old_format_without_last_review_issues_returns_none(
        self, tmp_path: Path
    ) -> None:
        """Test backward compat: old format without field returns None."""
        data = {
            "run_id": "run-old",
            "started_at": "2024-01-01T10:00:00Z",
            "issues": {
                "issue-legacy": {
                    "session_id": "session-old",
                    "status": "done",
                    # No last_review_issues field
                }
            },
        }
        path = tmp_path / "old_run.json"

        result = extract_session_from_run(data, path, "issue-legacy")

        assert result is not None
        assert result.session_id == "session-old"
        assert result.last_review_issues is None

    def test_handles_invalid_last_review_issues_type(self, tmp_path: Path) -> None:
        """Test that non-list last_review_issues is treated as None."""
        data = {
            "run_id": "run-invalid",
            "started_at": "2024-01-01T10:00:00Z",
            "issues": {
                "issue-bad": {
                    "session_id": "session-bad",
                    "last_review_issues": "not a list",  # Invalid type
                }
            },
        }
        path = tmp_path / "invalid_run.json"

        result = extract_session_from_run(data, path, "issue-bad")

        assert result is not None
        assert result.last_review_issues is None

    def test_filters_non_dict_items_in_last_review_issues(self, tmp_path: Path) -> None:
        """Test that non-dict items in last_review_issues list are filtered out."""
        data = {
            "run_id": "run-mixed",
            "started_at": "2024-01-01T10:00:00Z",
            "issues": {
                "issue-mixed": {
                    "session_id": "session-mixed",
                    "last_review_issues": [
                        {"file": "valid.py", "title": "Valid issue"},  # Valid
                        None,  # Invalid - should be filtered
                        "string item",  # Invalid - should be filtered
                        {"file": "another.py", "title": "Also valid"},  # Valid
                        123,  # Invalid - should be filtered
                    ],
                }
            },
        }
        path = tmp_path / "mixed_run.json"

        result = extract_session_from_run(data, path, "issue-mixed")

        assert result is not None
        assert result.last_review_issues == [
            {"file": "valid.py", "title": "Valid issue"},
            {"file": "another.py", "title": "Also valid"},
        ]

    def test_returns_none_when_all_items_invalid(self, tmp_path: Path) -> None:
        """Test that list with only invalid items returns None."""
        data = {
            "run_id": "run-all-invalid",
            "started_at": "2024-01-01T10:00:00Z",
            "issues": {
                "issue-all-invalid": {
                    "session_id": "session-all-invalid",
                    "last_review_issues": [None, "string", 123, []],  # All invalid
                }
            },
        }
        path = tmp_path / "all_invalid_run.json"

        result = extract_session_from_run(data, path, "issue-all-invalid")

        assert result is not None
        assert result.last_review_issues is None

    def test_handles_null_last_review_issues(self, tmp_path: Path) -> None:
        """Test that null last_review_issues is handled correctly."""
        data = {
            "run_id": "run-null",
            "started_at": "2024-01-01T10:00:00Z",
            "issues": {
                "issue-null": {
                    "session_id": "session-null",
                    "last_review_issues": None,
                }
            },
        }
        path = tmp_path / "null_run.json"

        result = extract_session_from_run(data, path, "issue-null")

        assert result is not None
        assert result.last_review_issues is None

    def test_forward_compat_extra_keys_ignored(self, tmp_path: Path) -> None:
        """Test forward compat: extra unknown keys don't crash."""
        data = {
            "run_id": "run-future",
            "started_at": "2024-01-01T10:00:00Z",
            "future_field": "unknown_value",  # Future field at run level
            "issues": {
                "issue-future": {
                    "session_id": "session-future",
                    "status": "success",
                    "another_future_field": {
                        "nested": "data"
                    },  # Future field at issue level
                    "last_review_issues": [{"issue": "Valid issue"}],
                }
            },
        }
        path = tmp_path / "future_run.json"

        result = extract_session_from_run(data, path, "issue-future")

        assert result is not None
        assert result.session_id == "session-future"
        assert result.last_review_issues == [{"issue": "Valid issue"}]

    def test_returns_none_for_missing_issue(self, tmp_path: Path) -> None:
        """Test that missing issue returns None."""
        data = {
            "run_id": "run-1",
            "started_at": "2024-01-01T10:00:00Z",
            "issues": {"other-issue": {"session_id": "session-1"}},
        }
        path = tmp_path / "run.json"

        result = extract_session_from_run(data, path, "nonexistent-issue")

        assert result is None

    def test_returns_none_for_invalid_issues_dict(self, tmp_path: Path) -> None:
        """Test that non-dict issues field returns None."""
        data = {
            "run_id": "run-1",
            "started_at": "2024-01-01T10:00:00Z",
            "issues": "not a dict",
        }
        path = tmp_path / "run.json"

        result = extract_session_from_run(data, path, "any-issue")

        assert result is None


class TestLookupPriorSessionInfoWithReviewIssues:
    """Test lookup_prior_session_info includes last_review_issues."""

    def test_lookup_info_includes_last_review_issues(self) -> None:
        """Test that lookup_prior_session_info returns last_review_issues."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            runs_dir = repo_path / ".mala" / "runs"
            runs_dir.mkdir(parents=True)

            review_issues = [
                {"issue": "Fix linting error", "line": 42},
            ]
            run_data = {
                "run_id": "run-review",
                "started_at": "2024-01-15T10:00:00Z",
                "issues": {
                    "issue-with-review": {
                        "session_id": "session-reviewed",
                        "last_review_issues": review_issues,
                    }
                },
            }

            (runs_dir / "2024-01-15T10-00-00_run.json").write_text(
                json.dumps(run_data), encoding="utf-8"
            )

            with patch(
                "src.infra.io.log_output.run_metadata.get_repo_runs_dir",
                return_value=runs_dir,
            ):
                result = lookup_prior_session_info(repo_path, "issue-with-review")

            assert result is not None
            assert result.session_id == "session-reviewed"
            assert result.last_review_issues is not None
            assert len(result.last_review_issues) == 1
            assert result.last_review_issues[0]["issue"] == "Fix linting error"

    def test_lookup_info_without_review_issues_returns_none_field(self) -> None:
        """Test lookup_prior_session_info without last_review_issues."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            runs_dir = repo_path / ".mala" / "runs"
            runs_dir.mkdir(parents=True)

            run_data = {
                "run_id": "run-no-review",
                "started_at": "2024-01-10T10:00:00Z",
                "issues": {
                    "issue-no-review": {
                        "session_id": "session-no-review",
                        # No last_review_issues
                    }
                },
            }

            (runs_dir / "2024-01-10T10-00-00_run.json").write_text(
                json.dumps(run_data), encoding="utf-8"
            )

            with patch(
                "src.infra.io.log_output.run_metadata.get_repo_runs_dir",
                return_value=runs_dir,
            ):
                result = lookup_prior_session_info(repo_path, "issue-no-review")

            assert result is not None
            assert result.session_id == "session-no-review"
            assert result.last_review_issues is None
