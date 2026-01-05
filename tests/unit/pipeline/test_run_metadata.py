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
    QualityGateResult,
    RunConfig,
    RunMetadata,
    RunningInstance,
    ValidationResult,
    cleanup_debug_logging,
    configure_debug_logging,
    get_running_instances,
    get_running_instances_for_dir,
    remove_run_marker,
    write_run_marker,
)
from src.core.models import (
    IssueResolution,
    ResolutionOutcome,
    ValidationArtifacts,
)


class TestQualityGateResult:
    """Test QualityGateResult dataclass."""

    def test_failed_result(self) -> None:
        # New spec-driven evidence uses CommandKind values as keys
        result = QualityGateResult(
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
            quality_gate=QualityGateResult(
                passed=True,
                evidence={"test": True, "commit_found": True},
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
                quality_gate=QualityGateResult(
                    passed=True,
                    evidence={"test": True, "commit_found": True},
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
            assert loaded.config.braintrust_enabled is True
            assert loaded.config.review_enabled is True

            # Verify issue with validation
            assert "test-1" in loaded.issues
            issue1_loaded = loaded.issues["test-1"]
            assert issue1_loaded.validation is not None
            assert issue1_loaded.validation.passed is True
            assert issue1_loaded.validation.coverage_percent == 87.5
            assert issue1_loaded.validation.artifacts is not None
            assert issue1_loaded.validation.artifacts.log_dir == Path("/tmp/logs")

            # Verify issue with resolution
            assert "test-2" in loaded.issues
            issue2_loaded = loaded.issues["test-2"]
            assert issue2_loaded.resolution is not None
            assert issue2_loaded.resolution.outcome == ResolutionOutcome.OBSOLETE

            # Verify run-level validation
            assert loaded.run_validation is not None
            assert loaded.run_validation.passed is True
            assert loaded.run_validation.e2e_passed is True

    def test_save_creates_repo_segmented_directory(self) -> None:
        """Test that save creates files in repo-segmented subdirectories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RunConfig(
                max_agents=1,
                timeout_minutes=10,
                max_issues=None,
                epic_id=None,
                only_ids=None,
                braintrust_enabled=False,
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
                handler_names = [
                    getattr(h, "name", "") for h in src_logger.handlers
                ]
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
                    braintrust_enabled=False,
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
                handler_names = [
                    getattr(h, "name", "") for h in src_logger.handlers
                ]
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
                handler_names = [
                    getattr(h, "name", "") for h in src_logger.handlers
                ]
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
                    braintrust_enabled=False,
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
