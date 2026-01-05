"""In-memory fake implementations for testing.

This module provides fake implementations of mala protocols and interfaces
for use in unit and integration tests. Fakes are preferred over mocks because
they:

1. Implement real protocol contracts, catching interface mismatches at test time
2. Provide deterministic, predictable behavior without call-order dependencies
3. Enable behavior-based testing (assert outputs/state) over interaction testing

See CLAUDE.md "Testing Philosophy" and "Fakes over mocks" for guidelines.
"""

from tests.fakes.command_runner import FakeCommandRunner, UnregisteredCommandError
from tests.fakes.coordinator_callbacks import (
    FakeAbortCallback,
    FakeFinalizeCallback,
    FakeSpawnCallback,
)
from tests.fakes.deadlock_monitor import FakeDeadlockMonitor
from tests.fakes.env_config import FakeEnvConfig
from tests.fakes.epic_model import (
    FakeEpicVerificationModel,
    VerificationAttempt,
    make_failing_verdict,
    make_passing_verdict,
)
from tests.fakes.epic_verification import (
    CoordinatorVerificationAttempt,
    FakeVerificationResults,
    make_failing_verification_result,
    make_not_eligible_verification_result,
    make_passing_verification_result,
)
from tests.fakes.event_sink import FakeEventSink, RecordedEvent
from tests.fakes.gate_checker import FakeGateChecker
from tests.fakes.issue_provider import FakeIssue, FakeIssueProvider
from tests.fakes.lint_cache import FakeLintCache
from tests.fakes.lock_manager import FakeLockManager, LockAcquireCall
from tests.fakes.sdk_client import FakeSDKClient, FakeSDKClientFactory
from tests.fakes.stream_processor import FakeStreamProcessor

__all__ = [
    "CoordinatorVerificationAttempt",
    "FakeAbortCallback",
    "FakeCommandRunner",
    "FakeDeadlockMonitor",
    "FakeEnvConfig",
    "FakeEpicVerificationModel",
    "FakeEventSink",
    "FakeFinalizeCallback",
    "FakeGateChecker",
    "FakeIssue",
    "FakeIssueProvider",
    "FakeLintCache",
    "FakeLockManager",
    "FakeSDKClient",
    "FakeSDKClientFactory",
    "FakeSpawnCallback",
    "FakeStreamProcessor",
    "FakeVerificationResults",
    "LockAcquireCall",
    "RecordedEvent",
    "UnregisteredCommandError",
    "VerificationAttempt",
    "make_failing_verdict",
    "make_failing_verification_result",
    "make_not_eligible_verification_result",
    "make_passing_verdict",
    "make_passing_verification_result",
]
