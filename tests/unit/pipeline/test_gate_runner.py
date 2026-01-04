"""Unit tests for GateRunner pipeline stage.

Tests the extracted gate checking logic using in-memory fakes,
without SDK or subprocess dependencies.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import pytest

from src.domain.lifecycle import RetryState
from src.pipeline.gate_runner import (
    GateRunner,
    GateRunnerConfig,
    PerIssueGateInput,
)
from src.core.protocols import GateChecker  # noqa: TC001 - needed at runtime for cast()
from src.domain.quality_gate import CommitResult, GateResult, ValidationEvidence
from src.domain.validation.spec import (
    CommandKind,
    ValidationCommand,
    ValidationScope,
    ValidationSpec,
)


@dataclass
class FakeGateChecker:
    """In-memory fake for GateChecker protocol.

    Allows tests to configure gate behavior without filesystem or git.
    """

    # Pre-configured result to return from check_with_resolution
    gate_result: GateResult = field(
        default_factory=lambda: GateResult(passed=True, failure_reasons=[])
    )
    # Pre-configured result for check_no_progress
    no_progress_result: bool = False
    # Pre-configured log end offset
    log_end_offset: int = 1000
    # Pre-configured commit result
    commit_result: CommitResult = field(
        default_factory=lambda: CommitResult(exists=True, commit_hash="abc123")
    )
    # Pre-configured validation evidence
    validation_evidence: ValidationEvidence = field(default_factory=ValidationEvidence)

    # Track calls for verification
    check_with_resolution_calls: list[dict] = field(default_factory=list)
    check_no_progress_calls: list[dict] = field(default_factory=list)
    get_log_end_offset_calls: list[dict] = field(default_factory=list)

    def check_with_resolution(
        self,
        issue_id: str,
        log_path: Path,
        baseline_timestamp: int | None = None,
        log_offset: int = 0,
        spec: ValidationSpec | None = None,
    ) -> GateResult:
        """Return pre-configured gate result."""
        self.check_with_resolution_calls.append(
            {
                "issue_id": issue_id,
                "log_path": log_path,
                "baseline_timestamp": baseline_timestamp,
                "log_offset": log_offset,
                "spec": spec,
            }
        )
        return self.gate_result

    def get_log_end_offset(self, log_path: Path, start_offset: int = 0) -> int:
        """Return pre-configured offset."""
        self.get_log_end_offset_calls.append(
            {"log_path": log_path, "start_offset": start_offset}
        )
        return self.log_end_offset

    def check_no_progress(
        self,
        log_path: Path,
        log_offset: int,
        previous_commit_hash: str | None,
        current_commit_hash: str | None,
        spec: ValidationSpec | None = None,
        check_validation_evidence: bool = True,
    ) -> bool:
        """Return pre-configured no_progress result."""
        self.check_no_progress_calls.append(
            {
                "log_path": log_path,
                "log_offset": log_offset,
                "previous_commit_hash": previous_commit_hash,
                "current_commit_hash": current_commit_hash,
                "spec": spec,
                "check_validation_evidence": check_validation_evidence,
            }
        )
        return self.no_progress_result

    def parse_validation_evidence_with_spec(
        self, log_path: Path, spec: ValidationSpec, offset: int = 0
    ) -> ValidationEvidence:
        """Return pre-configured validation evidence."""
        return self.validation_evidence

    def check_commit_exists(
        self, issue_id: str, baseline_timestamp: int | None = None
    ) -> CommitResult:
        """Return pre-configured commit result."""
        return self.commit_result


class TestPerIssueGate:
    """Test per-issue quality gate checking."""

    @pytest.fixture
    def tmp_log_path(self, tmp_path: Path) -> Path:
        """Create a temporary log file path."""
        log_path = tmp_path / "session.jsonl"
        log_path.write_text("")  # Create empty file
        return log_path

    @pytest.fixture
    def fake_checker(self) -> FakeGateChecker:
        """Create a fake gate checker."""
        return FakeGateChecker()

    @pytest.fixture
    def minimal_spec(self) -> ValidationSpec:
        """Create a minimal ValidationSpec for tests."""
        return ValidationSpec(
            commands=[],
            scope=ValidationScope.PER_ISSUE,
        )

    @pytest.fixture
    def runner(self, fake_checker: FakeGateChecker, tmp_path: Path) -> GateRunner:
        """Create a GateRunner with fake dependencies."""
        return GateRunner(
            gate_checker=cast("GateChecker", fake_checker),
            repo_path=tmp_path,
            config=GateRunnerConfig(max_gate_retries=3),
        )

    def test_run_per_issue_gate_returns_gate_result(
        self,
        runner: GateRunner,
        fake_checker: FakeGateChecker,
        tmp_log_path: Path,
        minimal_spec: ValidationSpec,
    ) -> None:
        """Gate runner should return the gate result from checker."""
        fake_checker.gate_result = GateResult(
            passed=True,
            failure_reasons=[],
            commit_hash="def456",
        )

        input = PerIssueGateInput(
            issue_id="test-123",
            log_path=tmp_log_path,
            retry_state=RetryState(),
            spec=minimal_spec,
        )
        output = runner.run_per_issue_gate(input)

        assert output.gate_result.passed is True
        assert output.gate_result.commit_hash == "def456"

    def test_run_per_issue_gate_returns_new_offset(
        self,
        runner: GateRunner,
        fake_checker: FakeGateChecker,
        tmp_log_path: Path,
        minimal_spec: ValidationSpec,
    ) -> None:
        """Gate runner should return the new log offset."""
        fake_checker.log_end_offset = 5000

        input = PerIssueGateInput(
            issue_id="test-123",
            log_path=tmp_log_path,
            retry_state=RetryState(),
            spec=minimal_spec,
        )
        output = runner.run_per_issue_gate(input)

        assert output.new_log_offset == 5000

    def test_run_per_issue_gate_passes_retry_state(
        self,
        runner: GateRunner,
        fake_checker: FakeGateChecker,
        tmp_log_path: Path,
        minimal_spec: ValidationSpec,
    ) -> None:
        """Gate runner should pass retry state to checker."""
        retry_state = RetryState(
            gate_attempt=2,
            log_offset=500,
            baseline_timestamp=1234567890,
        )

        input = PerIssueGateInput(
            issue_id="test-123",
            log_path=tmp_log_path,
            retry_state=retry_state,
            spec=minimal_spec,
        )
        runner.run_per_issue_gate(input)

        assert len(fake_checker.check_with_resolution_calls) == 1
        call = fake_checker.check_with_resolution_calls[0]
        assert call["issue_id"] == "test-123"
        assert call["baseline_timestamp"] == 1234567890
        assert call["log_offset"] == 500

    def test_run_per_issue_gate_checks_no_progress_on_retry(
        self,
        runner: GateRunner,
        fake_checker: FakeGateChecker,
        tmp_log_path: Path,
        minimal_spec: ValidationSpec,
    ) -> None:
        """Gate runner should check no_progress on retry attempts."""
        fake_checker.gate_result = GateResult(
            passed=False,
            failure_reasons=["Missing commit"],
            commit_hash="abc123",
        )
        fake_checker.no_progress_result = False

        # First attempt - should NOT check no_progress
        input = PerIssueGateInput(
            issue_id="test-123",
            log_path=tmp_log_path,
            retry_state=RetryState(gate_attempt=1),
            spec=minimal_spec,
        )
        runner.run_per_issue_gate(input)
        assert len(fake_checker.check_no_progress_calls) == 0

        # Second attempt - should check no_progress
        input = PerIssueGateInput(
            issue_id="test-123",
            log_path=tmp_log_path,
            retry_state=RetryState(gate_attempt=2),
            spec=minimal_spec,
        )
        runner.run_per_issue_gate(input)
        assert len(fake_checker.check_no_progress_calls) == 1

    def test_run_per_issue_gate_adds_no_progress_failure(
        self,
        runner: GateRunner,
        fake_checker: FakeGateChecker,
        tmp_log_path: Path,
        minimal_spec: ValidationSpec,
    ) -> None:
        """Gate runner should add no_progress to failure reasons when detected."""
        fake_checker.gate_result = GateResult(
            passed=False,
            failure_reasons=["Missing commit"],
            commit_hash="abc123",
        )
        fake_checker.no_progress_result = True

        input = PerIssueGateInput(
            issue_id="test-123",
            log_path=tmp_log_path,
            retry_state=RetryState(gate_attempt=2),
            spec=minimal_spec,
        )
        output = runner.run_per_issue_gate(input)

        assert output.gate_result.passed is False
        assert output.gate_result.no_progress is True
        assert "No progress" in output.gate_result.failure_reasons[-1]
        assert len(output.gate_result.failure_reasons) == 2

    def test_run_per_issue_gate_skips_no_progress_when_passed(
        self,
        runner: GateRunner,
        fake_checker: FakeGateChecker,
        tmp_log_path: Path,
        minimal_spec: ValidationSpec,
    ) -> None:
        """Gate runner should not check no_progress when gate passes."""
        fake_checker.gate_result = GateResult(
            passed=True,
            failure_reasons=[],
            commit_hash="abc123",
        )

        input = PerIssueGateInput(
            issue_id="test-123",
            log_path=tmp_log_path,
            retry_state=RetryState(gate_attempt=2),
            spec=minimal_spec,
        )
        runner.run_per_issue_gate(input)

        # Should not check no_progress when gate passed
        assert len(fake_checker.check_no_progress_calls) == 0


class TestSpecCaching:
    """Test ValidationSpec caching behavior."""

    @pytest.fixture
    def fake_checker(self) -> FakeGateChecker:
        """Create a fake gate checker."""
        return FakeGateChecker()

    @pytest.fixture
    def runner(self, fake_checker: FakeGateChecker, tmp_path: Path) -> GateRunner:
        """Create a GateRunner with fake dependencies."""
        return GateRunner(
            gate_checker=cast("GateChecker", fake_checker),
            repo_path=tmp_path,
            config=GateRunnerConfig(max_gate_retries=3),
        )

    def test_builds_spec_when_not_provided(
        self,
        runner: GateRunner,
        fake_checker: FakeGateChecker,
        tmp_path: Path,
    ) -> None:
        """Gate runner should build spec when not provided in input."""
        log_path = tmp_path / "session.jsonl"
        log_path.write_text("")

        # Create mala.yaml so build_validation_spec works
        (tmp_path / "mala.yaml").write_text("commands:\n  test: echo test\n")

        input = PerIssueGateInput(
            issue_id="test-123",
            log_path=log_path,
            retry_state=RetryState(),
            spec=None,  # Not provided
        )
        runner.run_per_issue_gate(input)

        # Spec should have been built and cached
        cached_spec = runner.get_cached_spec()
        assert cached_spec is not None
        assert cached_spec.scope == ValidationScope.PER_ISSUE

    def test_uses_provided_spec(
        self,
        runner: GateRunner,
        fake_checker: FakeGateChecker,
        tmp_path: Path,
    ) -> None:
        """Gate runner should use provided spec instead of building."""
        log_path = tmp_path / "session.jsonl"
        log_path.write_text("")

        custom_spec = ValidationSpec(
            commands=[
                ValidationCommand(
                    name="custom-test",
                    command="echo test",
                    kind=CommandKind.TEST,
                )
            ],
            scope=ValidationScope.PER_ISSUE,
        )

        input = PerIssueGateInput(
            issue_id="test-123",
            log_path=log_path,
            retry_state=RetryState(),
            spec=custom_spec,
        )
        runner.run_per_issue_gate(input)

        # Should have passed custom spec to checker
        assert len(fake_checker.check_with_resolution_calls) == 1
        call = fake_checker.check_with_resolution_calls[0]
        assert call["spec"] is custom_spec

    def test_set_cached_spec(
        self,
        runner: GateRunner,
        fake_checker: FakeGateChecker,
        tmp_path: Path,
    ) -> None:
        """set_cached_spec should pre-populate the cache."""
        custom_spec = ValidationSpec(
            commands=[],
            scope=ValidationScope.PER_ISSUE,
        )

        runner.set_cached_spec(custom_spec)

        assert runner.get_cached_spec() is custom_spec

    def test_get_cached_spec_returns_none_initially(
        self,
        runner: GateRunner,
    ) -> None:
        """get_cached_spec should return None before any gate runs."""
        assert runner.get_cached_spec() is None


class TestGateRunnerConfig:
    """Test GateRunnerConfig behavior."""

    def test_default_config_values(self) -> None:
        """Config should have sensible defaults."""
        config = GateRunnerConfig()

        assert config.max_gate_retries == 3
        assert config.disable_validations is None
        assert config.coverage_threshold is None

    def test_config_with_custom_values(self) -> None:
        """Config should accept custom values."""
        config = GateRunnerConfig(
            max_gate_retries=5,
            disable_validations={"pytest", "ruff"},
            coverage_threshold=80.0,
        )

        assert config.max_gate_retries == 5
        assert config.disable_validations == {"pytest", "ruff"}
        assert config.coverage_threshold == 80.0
