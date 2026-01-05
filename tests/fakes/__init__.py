"""In-memory fake implementations for testing.

This module provides fake implementations of mala protocols and interfaces
for use in unit and integration tests. Fakes are preferred over mocks because
they:

1. Implement real protocol contracts, catching interface mismatches at test time
2. Provide deterministic, predictable behavior without call-order dependencies
3. Enable behavior-based testing (assert outputs/state) over interaction testing

See CLAUDE.md "Testing Philosophy" and "Fakes over mocks" for guidelines.

Planned fakes (to be implemented):
- FakeIssueProvider: In-memory issue storage implementing IssueProvider
"""

from tests.fakes.command_runner import FakeCommandRunner, UnregisteredCommandError
from tests.fakes.epic_model import (
    FakeEpicVerificationModel,
    VerificationAttempt,
    make_failing_verdict,
    make_passing_verdict,
)
from tests.fakes.event_sink import FakeEventSink, RecordedEvent
from tests.fakes.lock_manager import FakeLockManager, LockAcquireCall

__all__ = [
    "FakeCommandRunner",
    "FakeEpicVerificationModel",
    "FakeEventSink",
    "FakeLockManager",
    "LockAcquireCall",
    "RecordedEvent",
    "UnregisteredCommandError",
    "VerificationAttempt",
    "make_failing_verdict",
    "make_passing_verdict",
]
