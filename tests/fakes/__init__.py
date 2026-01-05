"""In-memory fake implementations for testing.

This module will provide fake implementations of mala protocols and interfaces
for use in unit and integration tests. Fakes are preferred over mocks because
they:

1. Implement real protocol contracts, catching interface mismatches at test time
2. Provide deterministic, predictable behavior without call-order dependencies
3. Enable behavior-based testing (assert outputs/state) over interaction testing

See CLAUDE.md "Testing Philosophy" and "Fakes over mocks" for guidelines.

Planned fakes (to be implemented):
- FakeIssueProvider: In-memory issue storage implementing IssueProvider protocol
- FakeCommandRunner: Deterministic command execution implementing CommandRunnerPort
- FakeLockManager: In-memory lock coordination implementing LockManagerPort
- FakeEventSink: Event capture implementing MalaEventSink
- FakeEpicVerificationModel: Controlled epic verification responses
"""
