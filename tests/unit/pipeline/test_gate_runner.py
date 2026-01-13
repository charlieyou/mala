"""Unit tests for GateRunner pipeline stage.

Tests the extracted gate checking logic using in-memory fakes,
without SDK or subprocess dependencies.
"""

from pathlib import Path
from typing import cast

import pytest

from src.domain.lifecycle import RetryState
from src.pipeline.gate_runner import (
    GateRunner,
    GateRunnerConfig,
    PerSessionGateInput,
)
from src.core.protocols.validation import GateChecker  # noqa: TC001 - needed at runtime for cast()
from src.domain.evidence_check import GateResult
from src.domain.validation.spec import (
    CommandKind,
    ValidationCommand,
    ValidationScope,
    ValidationSpec,
)
from tests.fakes.gate_checker import FakeGateChecker


class TestPerSessionGate:
    """Test per-session quality gate checking."""

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
            scope=ValidationScope.PER_SESSION,
        )

    @pytest.fixture
    def runner(self, fake_checker: FakeGateChecker, tmp_path: Path) -> GateRunner:
        """Create a GateRunner with fake dependencies."""
        return GateRunner(
            gate_checker=cast("GateChecker", fake_checker),
            repo_path=tmp_path,
            config=GateRunnerConfig(max_gate_retries=3),
        )

    def test_run_per_session_gate_returns_gate_result(
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

        input = PerSessionGateInput(
            issue_id="test-123",
            log_path=tmp_log_path,
            retry_state=RetryState(),
            spec=minimal_spec,
        )
        output = runner.run_per_session_gate(input)

        assert output.gate_result.passed is True
        assert output.gate_result.commit_hash == "def456"

    def test_run_per_session_gate_returns_new_offset(
        self,
        runner: GateRunner,
        fake_checker: FakeGateChecker,
        tmp_log_path: Path,
        minimal_spec: ValidationSpec,
    ) -> None:
        """Gate runner should return the new log offset."""
        fake_checker.log_end_offset = 5000

        input = PerSessionGateInput(
            issue_id="test-123",
            log_path=tmp_log_path,
            retry_state=RetryState(),
            spec=minimal_spec,
        )
        output = runner.run_per_session_gate(input)

        assert output.new_log_offset == 5000

    def test_run_per_session_gate_passes_retry_state(
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

        input = PerSessionGateInput(
            issue_id="test-123",
            log_path=tmp_log_path,
            retry_state=retry_state,
            spec=minimal_spec,
        )
        runner.run_per_session_gate(input)

        assert len(fake_checker.check_with_resolution_calls) == 1
        call = fake_checker.check_with_resolution_calls[0]
        assert call["issue_id"] == "test-123"
        assert call["baseline_timestamp"] == 1234567890
        assert call["log_offset"] == 500

    def test_run_per_session_gate_checks_no_progress_on_retry(
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
        input = PerSessionGateInput(
            issue_id="test-123",
            log_path=tmp_log_path,
            retry_state=RetryState(gate_attempt=1),
            spec=minimal_spec,
        )
        runner.run_per_session_gate(input)
        assert len(fake_checker.check_no_progress_calls) == 0

        # Second attempt - should check no_progress
        input = PerSessionGateInput(
            issue_id="test-123",
            log_path=tmp_log_path,
            retry_state=RetryState(gate_attempt=2),
            spec=minimal_spec,
        )
        runner.run_per_session_gate(input)
        assert len(fake_checker.check_no_progress_calls) == 1

    def test_run_per_session_gate_adds_no_progress_failure(
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

        input = PerSessionGateInput(
            issue_id="test-123",
            log_path=tmp_log_path,
            retry_state=RetryState(gate_attempt=2),
            spec=minimal_spec,
        )
        output = runner.run_per_session_gate(input)

        assert output.gate_result.passed is False
        assert output.gate_result.no_progress is True
        assert "No progress" in output.gate_result.failure_reasons[-1]
        assert len(output.gate_result.failure_reasons) == 2

    def test_run_per_session_gate_skips_no_progress_when_passed(
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

        input = PerSessionGateInput(
            issue_id="test-123",
            log_path=tmp_log_path,
            retry_state=RetryState(gate_attempt=2),
            spec=minimal_spec,
        )
        runner.run_per_session_gate(input)

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

        input = PerSessionGateInput(
            issue_id="test-123",
            log_path=log_path,
            retry_state=RetryState(),
            spec=None,  # Not provided
        )
        runner.run_per_session_gate(input)

        # Spec should have been built and cached
        cached_spec = runner.get_cached_spec()
        assert cached_spec is not None
        assert cached_spec.scope == ValidationScope.PER_SESSION

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
            scope=ValidationScope.PER_SESSION,
        )

        input = PerSessionGateInput(
            issue_id="test-123",
            log_path=log_path,
            retry_state=RetryState(),
            spec=custom_spec,
        )
        runner.run_per_session_gate(input)

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
            scope=ValidationScope.PER_SESSION,
        )

        runner.set_cached_spec(custom_spec)

        assert runner.get_cached_spec() is custom_spec

    def test_get_cached_spec_returns_none_initially(
        self,
        runner: GateRunner,
    ) -> None:
        """get_cached_spec should return None before any gate runs."""
        assert runner.get_cached_spec() is None
