"""In-memory fake implementations for testing.

This module provides fake implementations of mala protocols and interfaces
for use in unit and integration tests. Fakes are preferred over mocks because
they:

1. Implement real protocol contracts, catching interface mismatches at test time
2. Provide deterministic, predictable behavior without call-order dependencies
3. Enable behavior-based testing (assert outputs/state) over interaction testing

See CLAUDE.md "Testing Philosophy" and "Fakes over mocks" for guidelines.

Available fakes:
- FakeIssueProvider: In-memory issue storage implementing IssueProvider protocol
- FakeCommandRunner: Deterministic command execution with fail-closed semantics
- FakeLockManager: In-memory lock coordination for parallel test scenarios
- FakeEventSink: Event capture with completeness verification
- FakeEpicVerificationModel: Controlled epic verification responses

Usage:
    from tests.fakes import FakeIssueProvider, FakeCommandRunner

    def test_something():
        provider = FakeIssueProvider()
        provider.add_issue(...)
        # test code that uses provider
"""
