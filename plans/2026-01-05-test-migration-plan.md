# Test Suite Migration Plan: Mocks → Fakes

**Date**: 2026-01-05
**Status**: Draft
**Goal**: Align test suite with testing philosophy (CLAUDE.md / AGENTS.md)

---

## Context & Goals

The mala test suite has grown to ~2,289 tests with heavy reliance on `MagicMock`, `patch()`, and `assert_called*` patterns. Per the testing philosophy in CLAUDE.md and AGENTS.md, these patterns couple tests to implementation details rather than behavior, making refactoring risky and tests brittle.

**Goals**:
1. Replace mock-heavy tests with fakes-based behavioral tests
2. Delete tests that don't catch meaningful bugs (dataclass defaults, stdlib behavior)
3. Reduce test maintenance burden while preserving defect-catching capability

**Success looks like**: Tests that assert on observable outcomes (state changes, events emitted, return values) rather than internal call sequences.

---

## Scope & Non-Goals

### In Scope
- Creating missing fakes for core protocols (`IssueProvider`, `CommandRunnerPort`, `LockManagerPort`, `EpicVerificationModel`)
- Consolidating scattered `FakeEventSink` implementations
- Rewriting mock-heavy test files to use fakes and behavioral assertions
- Deleting zero-value tests (dataclass defaults, stdlib behavior)
- Adding contract tests that validate fake-vs-real parity

### Non-Goals
- Modifying production code beyond adding constructor parameters for DI (no behavioral changes)
- Achieving 100% mock elimination (acceptable uses remain for third-party SDK internals)
- Rewriting tests that already follow good patterns (e.g., `test_deadlock.py`, `test_log_events.py`)
- Performance optimization of the test suite

---

## Assumptions & Constraints

### Existing DI Seams (No Production Changes Required)
- `make_orchestrator` factory in `tests/conftest.py` accepts protocol implementations
- `FakeSDKClient`, `FakeGateChecker`, `FakeCodeReviewer` already exist and are injected via fixtures
- `Orchestrator.__init__` accepts `issue_provider`, `event_sink`, `sdk_client` parameters

### Production Code Changes Required
The following production files need constructor modifications to accept injected dependencies.

**Important**: All new parameters must be **keyword-only** (after `*,`) to avoid breaking existing positional call sites. Before landing changes, verify no positional constructions exist:
```bash
rg 'Orchestrator\([^)]*[^=,]\s*\)' src/ tests/  # Check for positional args
```

| File | Current State | Required Change |
|------|---------------|-----------------|
| `src/orchestration/orchestrator.py` | Hardcodes `get_runs_dir()`, `release_run_locks()` | Add keyword-only `runs_dir: Path = None` and `lock_releaser: Callable = None` with defaults |
| `src/domain/validation/spec_executor.py` | Already accepts `CommandRunnerPort` via `ExecutorConfig` | No changes needed - DI seam exists |
| `src/infra/epic_verifier.py` (EpicVerifier) | Already accepts `model: EpicVerificationModel` in constructor | No changes needed - DI seam exists |

### Constraints
- Coverage must stay ≥72% (`--cov-fail-under=72`)
- CI runs unit tests only (no external services); contract tests for real providers require separate integration job
- Parallel test execution (`-n auto`) means fakes must be thread-safe or tests must be isolated

### Concurrency/Async Safety for Fakes
Fakes will be used under async orchestrator flows. Requirements:
- **Single-threaded async**: Fakes assume single event loop; `asyncio.Lock` not required for state mutations since await points are explicit
- **Parallel pytest**: Each test gets its own fake instances via fixtures (no shared state across workers)
- **FakeIssueProvider.claim_async**: Returns synchronously (no real I/O), safe for concurrent awaits within single test

If future parallelism within tests requires concurrent fake access, fakes should use `asyncio.Lock` for state mutations.

---

## Prerequisites

1. **Baseline metrics captured**: Run tracking commands to establish current anti-pattern counts:
   ```bash
   rg -c 'MagicMock|AsyncMock' tests/ --type py | awk -F: '{sum+=$2} END {print sum}'
   rg -c 'with patch|@patch' tests/ --type py | awk -F: '{sum+=$2} END {print sum}'
   rg -c '\.assert_called' tests/ --type py | awk -F: '{sum+=$2} END {print sum}'
   ```
2. **Team alignment**: Testing philosophy documented in `tests/README.md`, linking to CLAUDE.md/AGENTS.md
3. **CI job for contract tests**: Separate job with `bd` CLI and real workspace for running real provider tests (can start locally; CI job can be added later)

---

## High-Level Approach

### Target Architecture

```
tests/
├── fakes/                    # Canonical fake implementations
│   ├── __init__.py
│   ├── issue_provider.py     # FakeIssueProvider (implements IssueProvider protocol)
│   ├── command_runner.py     # FakeCommandRunner (implements CommandRunnerPort)
│   ├── lock_manager.py       # FakeLockManager (implements LockManagerPort)
│   ├── epic_model.py         # FakeEpicVerificationModel
│   └── event_sink.py         # FakeEventSink (consolidated)
├── contracts/                # Protocol contract tests
│   └── test_issue_provider_contract.py
├── unit/                     # Behavioral unit tests (no patches, no assert_called)
├── integration/              # Cross-layer tests with real file I/O
└── e2e/                      # End-to-end tests against real services
```

### Pytest Marker Configuration

Markers are already registered in `pyproject.toml`. Ensure `tests/contracts/` is included in integration runs:

```toml
# pyproject.toml (existing)
addopts = "--strict-markers -m 'unit or integration' ..."
markers = [
    "unit: fast, isolated tests (default)",
    "integration: tests that exercise multiple components or real tools locally",
    "e2e: end-to-end tests that require real CLI/API auth",
]
```

CI jobs:
- **Default/PR**: `uv run pytest -m unit` (excludes contracts with real providers)
- **Integration**: `uv run pytest -m integration` (includes `tests/contracts/`)

### Fake Design Principles

1. **Protocol compliance**: Each fake implements the full protocol interface (verified by `uvx ty check`)
2. **Observable state**: Fakes expose internal state for assertions (`claimed`, `closed`, `calls`)
3. **Deterministic**: No randomness; use configurable sequences for multi-call scenarios
4. **Minimal logic**: Just enough to track state transitions; no business logic
5. **Fail-closed**: Unregistered inputs raise errors rather than returning defaults

### FakeCommandRunner Matching Semantics

**Fail-closed design** to preserve defect-catching capability:
```python
class UnregisteredCommandError(Exception):
    """Raised when FakeCommandRunner receives an unregistered command."""
    pass

class FakeCommandRunner:
    def __init__(
        self,
        results: dict[tuple[str, ...], CommandResult] | None = None,
        *,
        allow_unregistered: bool = False,  # Explicit escape hatch
    ):
        self.results = results or {}  # cmd tuple -> result
        self.calls: list[tuple[tuple[str, ...], dict]] = []  # (cmd, kwargs)
        self._allow_unregistered = allow_unregistered

    def run(self, cmd: list[str], **kwargs) -> CommandResult:
        cmd_tuple = tuple(cmd)
        self.calls.append((cmd_tuple, kwargs))
        # Exact tuple match
        if cmd_tuple in self.results:
            return self.results[cmd_tuple]
        # Fail-closed: unregistered commands raise by default
        if not self._allow_unregistered:
            raise UnregisteredCommandError(
                f"FakeCommandRunner received unregistered command: {cmd}. "
                f"Register it in results or set allow_unregistered=True."
            )
        return CommandResult(cmd, 0, "", "")

    # Helper methods for flexible assertions
    def has_call_with_prefix(self, prefix: tuple[str, ...]) -> bool:
        """Check if any call starts with the given prefix."""
        return any(cmd[:len(prefix)] == prefix for cmd, _ in self.calls)

    def get_calls_with_prefix(self, prefix: tuple[str, ...]) -> list[tuple[tuple[str, ...], dict]]:
        """Get all calls starting with the given prefix."""
        return [(cmd, kw) for cmd, kw in self.calls if cmd[:len(prefix)] == prefix]
```

Assertions use helper methods to avoid brittleness:
```python
# Flexible: passes regardless of extra flags
assert fake_runner.has_call_with_prefix(("pytest",))

# When cwd matters:
pytest_calls = fake_runner.get_calls_with_prefix(("pytest",))
assert any(kw.get("cwd") == repo_path for _, kw in pytest_calls)
```

---

## Technical Design

### Fake Implementations

**FakeIssueProvider** (implements `IssueProvider` - 15 methods):
```python
@dataclass
class FakeIssue:
    id: str
    status: str = "ready"
    priority: int = 5
    parent_epic: str | None = None
    description: str = ""
    tags: list[str] = field(default_factory=list)

class FakeIssueProvider:
    def __init__(self, issues: dict[str, FakeIssue] | None = None):
        self.issues = issues or {}
        self.claimed: set[str] = set()
        self.closed: set[str] = set()
        self.reset_calls: list[str] = []
        self.created_issues: list[dict] = []

    async def get_ready_async(self, exclude_ids=None, ...) -> list[str]:
        return [id for id, i in self.issues.items()
                if i.status == "ready" and id not in (exclude_ids or set())]

    async def claim_async(self, issue_id: str) -> bool:
        self.claimed.add(issue_id)
        self.issues[issue_id].status = "in_progress"
        return True
    # ... other methods with simple in-memory logic
```

**FakeLockManager** (implements `LockManagerPort`):
```python
class FakeLockManager:
    def __init__(self):
        self.locks: dict[str, str] = {}  # path -> holder_agent_id
        self.acquire_calls: list[tuple[str, str]] = []

    def try_lock(self, path: str, agent_id: str) -> bool:
        if path in self.locks:
            return False
        self.locks[path] = agent_id
        self.acquire_calls.append((path, agent_id))
        return True
```

**FakeEpicVerificationModel** (implements `EpicVerificationModel`):
```python
@dataclass
class VerificationAttempt:
    """Observable record of a verification attempt."""
    epic_id: str
    verdict: EpicVerdict

class FakeEpicVerificationModel:
    def __init__(self, verdicts: list[EpicVerdict] | None = None):
        self.verdicts = verdicts or []
        self.attempts: list[VerificationAttempt] = []  # Observable history
        self._verdict_index = 0

    async def verify(self, epic_id: str, ...) -> EpicVerdict:
        if self._verdict_index < len(self.verdicts):
            verdict = self.verdicts[self._verdict_index]
            self._verdict_index += 1
        else:
            verdict = EpicVerdict(passed=True, unmet_criteria=[], confidence=1.0, reasoning="")
        # Record attempt for observable assertions
        self.attempts.append(VerificationAttempt(epic_id=epic_id, verdict=verdict))
        return verdict
```

**FakeEventSink** (implements `MalaEventSink` ~50 methods):

All methods explicitly implemented for type safety and IDE discoverability. No `__getattr__` fallback—missing methods cause type errors at check time rather than silent no-ops at runtime.

```python
class FakeEventSink:
    """Explicitly implements all MalaEventSink methods for type safety."""

    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def _record(self, event_type: str, **kwargs):
        self.events.append((event_type, kwargs))

    # Explicit implementations for each protocol method
    def on_issue_started(self, issue_id: str, agent_id: str) -> None:
        self._record("issue_started", issue_id=issue_id, agent_id=agent_id)

    def on_issue_completed(self, issue_id: str, agent_id: str, resolution: str) -> None:
        self._record("issue_completed", issue_id=issue_id, agent_id=agent_id, resolution=resolution)

    def on_issue_failed(self, issue_id: str, agent_id: str, error: str) -> None:
        self._record("issue_failed", issue_id=issue_id, agent_id=agent_id, error=error)

    # ... all ~50 methods explicitly implemented via _record
    # Can be generated from protocol definition to reduce boilerplate

    # Assertion helpers
    def has_event(self, event_type: str, **match) -> bool:
        return any(t == event_type and all(d.get(k) == v for k, v in match.items())
                   for t, d in self.events)

    def get_events(self, event_type: str) -> list[dict]:
        return [d for t, d in self.events if t == event_type]
```

### Keeping FakeEventSink in Sync with Protocol

Add a contract test that fails when `MalaEventSink` protocol methods are missing from `FakeEventSink`:

```python
# tests/contracts/test_event_sink_completeness.py
import inspect
from src.core.protocols import MalaEventSink
from tests.fakes.event_sink import FakeEventSink

def test_fake_event_sink_implements_all_protocol_methods():
    """Ensure FakeEventSink stays in sync with MalaEventSink protocol."""
    protocol_methods = {
        name for name, _ in inspect.getmembers(MalaEventSink, predicate=inspect.isfunction)
        if not name.startswith("_")
    }
    fake_methods = {
        name for name, _ in inspect.getmembers(FakeEventSink, predicate=inspect.isfunction)
        if not name.startswith("_")
    }

    missing = protocol_methods - fake_methods
    assert not missing, f"FakeEventSink missing protocol methods: {missing}"
```

This test fails immediately when protocol evolves, before any runtime issues occur.

### Contract Tests for Fake-Real Parity

Contract tests validate that fakes behave consistently with real implementations:

```python
# tests/contracts/test_issue_provider_contract.py
import os
import shutil
import pytest

# Skip real provider tests when prerequisites are missing
def _has_bd_cli() -> bool:
    return shutil.which("bd") is not None

def _has_beads_workspace() -> bool:
    return os.environ.get("BEADS_TEST_WORKSPACE") is not None

class TestIssueProviderContract:
    """Contract tests that both FakeIssueProvider and BeadsClient must pass."""

    @pytest.fixture(params=["fake", "real"])
    def provider(self, request, tmp_path):
        if request.param == "fake":
            yield FakeIssueProvider({"issue-1": FakeIssue("issue-1")})
        else:
            if not _has_bd_cli():
                pytest.skip("Real provider requires bd CLI (not found in PATH)")
            if not _has_beads_workspace():
                pytest.skip("Real provider requires BEADS_TEST_WORKSPACE env var")
            yield make_real_beads_client(tmp_path)

    @pytest.mark.integration  # Skipped in unit-only CI; runs in integration job
    async def test_claim_marks_in_progress(self, provider):
        await provider.claim_async("issue-1")
        ready = await provider.get_ready_async()
        assert "issue-1" not in ready

    @pytest.mark.integration
    async def test_close_removes_from_ready(self, provider):
        await provider.close_async("issue-1", resolution="completed")
        ready = await provider.get_ready_async()
        assert "issue-1" not in ready
```

**CI Configuration**:
- `uv run pytest -m unit` - Runs fake-only tests (default CI job)
- `uv run pytest -m integration` - Runs contract tests including real providers (dedicated job with prerequisites)

This ensures contract tests actually validate parity, not just the fake in isolation.

---

## File Impact Summary

### Test Files (Primary Changes)

| File | Action | Change Type |
|------|--------|-------------|
| `tests/fakes/*.py` | Create | New fake implementations |
| `tests/contracts/*.py` | Create | Contract tests for fakes |
| `tests/unit/test_orchestrator.py` | Split + rewrite | Behavioral tests with fakes |
| `tests/unit/test_validation.py` | Rewrite | Inject FakeCommandRunner |
| `tests/unit/test_epic_verification_retry.py` | Rewrite | Use FakeEpicVerificationModel |
| `tests/unit/test_config.py` | Delete tests | Remove dataclass default tests |

### Source Files (DI Modifications)

| File | Change |
|------|--------|
| `src/orchestration/orchestrator.py` | Add keyword-only `runs_dir`, `lock_releaser` params |
| `src/domain/validation/spec_executor.py` | No changes - DI via `ExecutorConfig.command_runner` exists |
| `src/infra/epic_verifier.py` | No changes - DI via `EpicVerifier(model=...)` exists |

---

## Risks, Edge Cases & Breaking Changes

### Risk: Coverage Drops Below Threshold
- **Mitigation**: Run `pytest --cov-fail-under=72` after each file change
- **Rollback**: If coverage drops, revert and identify which deleted tests provided unique coverage; create behavioral replacements first

### Risk: Fake Diverges from Real Implementation
- **Mitigation**: Contract tests run against both fake and real in integration CI
- **Detection**: Contract test failure on real provider indicates divergence

### Risk: Tests Become Flaky Under Parallelism
- **Mitigation**: Each test gets isolated fake instances via fixtures
- **Detection**: Run with `-n auto --reruns 2` to catch flakes early

### Risk: Fake Silently Accepts Wrong Commands
- **Mitigation**: FakeCommandRunner is fail-closed; unregistered commands raise `UnregisteredCommandError`
- **Escape hatch**: Tests that intentionally ignore commands can use `allow_unregistered=True`

### Risk: FakeEventSink Drifts from Protocol
- **Mitigation**: Contract test `test_fake_event_sink_implements_all_protocol_methods` fails on missing methods
- **Detection**: `uvx ty check` catches type mismatches

### Edge Case: Third-Party SDK Internals
- **Decision**: `MagicMock` acceptable for Braintrust SDK internals where no fake exists
- **Tracking**: Document remaining mocks in `tests/README.md` with rationale

### No Breaking Changes
- Production behavior unchanged; only constructor signatures gain optional keyword-only parameters
- Existing tests continue to work during incremental migration

---

## Testing & Verification

### Acceptance Criteria (Behavioral Outcomes)

1. **Orchestrator issue lifecycle**: Given a ready issue, when orchestrator runs, then:
   - `fake_issues.issues["issue-1"].status == "in_progress"` after claim
   - `fake_events.has_event("issue_completed", issue_id="issue-1")` after success
   - `fake_events.has_event("issue_failed", issue_id="issue-1")` after failure

2. **Validation command execution**: Given a validation spec with pytest + ruff, when runner executes:
   - `fake_runner.has_call_with_prefix(("pytest",))` is True
   - `fake_runner.has_call_with_prefix(("uvx", "ruff"))` is True
   - Result reflects first failing command's exit code

3. **Epic verification retry**: Given an epic with 2 failing then 1 passing verdict:
   - `coordinator.check_epic_closure()` returns `final_passed=True`
   - `len(fake_model.attempts) == 3` (3 verification attempts recorded)
   - `fake_model.attempts[0].verdict.passed is False`
   - `fake_model.attempts[2].verdict.passed is True`

4. **Contract parity**: For each fake, contract tests pass against both fake and real implementations in integration CI

5. **Type safety**: `uvx ty check` passes with all fakes implementing full protocol interfaces

6. **FakeEventSink completeness**: `test_fake_event_sink_implements_all_protocol_methods` passes

### Verification Process

After each major change:
1. `uvx ty check` - Fakes implement full protocol interfaces
2. `uv run pytest -m unit` - All unit tests pass
3. `uv run pytest --cov-fail-under=72` - Coverage maintained

### Progress Indicators (Not Success Criteria)

These metrics indicate migration progress but do not define success:
- MagicMock/AsyncMock count trending down
- patch() uses in unit tests trending toward zero
- assert_called* uses trending toward zero

---

## Open Questions

1. **Contract test frequency**: Run on every PR or only nightly?
   - Every PR: Catches divergence early but slower CI
   - Nightly: Faster PRs but delayed feedback

2. **Migration order**: Should we prioritize highest mock-count files (test_orchestrator) or highest-value files first?
