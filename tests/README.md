# Mala Test Suite

## Testing Philosophy

This test suite follows the testing philosophy documented in [CLAUDE.md](../CLAUDE.md#testing-philosophy) and [AGENTS.md](../AGENTS.md#testing-philosophy):

- **Behavior over implementation**: Test externally visible behavior (outputs, state changes, emitted events), not implementation details
- **Fakes over mocks**: Use in-memory fakes implementing protocols. Use `MagicMock` only as last resort
- **Don't test the obvious**: Skip tests for dataclass defaults, Python stdlib, or third-party library behavior
- **Keep it small**: Prefer fewer high-signal tests over exhaustive low-value coverage

## Directory Structure

```
tests/
├── __init__.py     # Package marker (enables `import tests.fakes`)
├── unit/           # Isolated unit tests (no network, no subprocess)
├── integration/    # Cross-layer tests validating component composition
├── e2e/            # End-to-end smoke tests (requires CLI auth)
├── fakes/          # In-memory fake implementations of protocols
├── fixtures/       # Shared test data and fixtures
└── conftest.py     # Pytest configuration and shared fixtures
```

## Fakes Module (`tests/fakes/`)

The `fakes/` module provides in-memory implementations of mala protocols for testing.

**Available fakes**:

| Fake | Protocol | Purpose |
|------|----------|---------|
| `FakeIssueProvider` | `IssueProvider` | In-memory issue storage |
| `FakeCommandRunner` | `CommandRunnerPort` | Deterministic command execution |
| `FakeLockManager` | `LockManagerPort` | In-memory lock coordination |
| `FakeEventSink` | `MalaEventSink` | Event capture with verification |
| `FakeEpicVerificationModel` | `EpicVerificationModel` | Controlled verification responses |

### Why Fakes Over Mocks

Fakes implement real protocol contracts, which means:
1. Interface mismatches are caught at test time (not production)
2. Tests assert on behavior (outputs, state) not interactions (call counts)
3. Contract tests can verify both fake and real implementations

### Usage

```python
import pytest
from tests.fakes import FakeIssue, FakeIssueProvider

@pytest.mark.asyncio
async def test_issue_processing():
    provider = FakeIssueProvider(
        {"test-1": FakeIssue(id="test-1", status="open", title="Test")}
    )

    # Test code that uses provider (IssueProvider is an async protocol)
    issue_ids = await provider.get_ready_async()

    # Assert on outcomes, not calls
    assert len(issue_ids) == 1
    assert issue_ids[0] == "test-1"
```

## Running Tests

```bash
uv run pytest                              # Unit + integration (default)
uv run pytest -m unit                      # Unit tests only
uv run pytest -m integration -n auto       # Integration in parallel
uv run pytest -m e2e                       # E2E tests (requires auth)
```

## Documented Mock/Patch Exceptions

The following mock and patch usages are intentional and documented exceptions to the "fakes over mocks" guideline:

### Patches for External Dependencies

| File | Patch Target | Rationale |
|------|--------------|-----------|
| `test_locking_mcp.py` | `try_lock`, `get_lock_holder`, `wait_for_lock_async`, `release_lock`, `cleanup_agent_locks` | MCP tool handlers use module-level functions from `locking.py`. Patches isolate handler logic from filesystem lock operations. The module architecture doesn't support dependency injection without significant refactoring. |
| `test_idle_retry_policy.py` | `asyncio.sleep` | Standard pattern for testing async timing/backoff without actual delays. |
| `test_validation.py` | `SpecResultBuilder._run_e2e` | E2E tests require complex fixture setup. Patching allows testing E2E triggering logic without running actual E2E. |
| `test_validation.py` | `create_worktree` | Tests worktree failure handling without git worktree operations. |
| `test_validation.py` | `is_baseline_stale` | Controls coverage baseline staleness for deterministic tests. |
| `test_run_metadata.py` | `get_runs_dir`, env vars | Tests run metadata persistence with controlled paths. |
| `test_watch_mode.py` | `time.monotonic` | Controls time progression for deterministic timing tests. |

### MagicMock/AsyncMock Usage

| Pattern | Rationale |
|---------|-----------|
| `MagicMock(spec=LintCache)` | `LintCache` from `src/infra/hooks` satisfies `LintCacheProtocol`. Mock with spec ensures interface compliance. Creating a fake would duplicate the real implementation. |
| `AsyncMock` for callbacks | Callback functions (e.g., `spawn_callback`, `finalize_callback`) are simple callables. AsyncMock tracks invocation without needing a full fake. |
| `MagicMock` for SDK internals | Claude SDK types like `SdkMcpTool` have complex initialization. Mocking with spec validates interface usage. |

### call_count Assertions

Some tests use `.call_count` assertions. These are **not** implementation-detail tests but rather behavioral assertions:

- **Caching behavior** (`test_beads_client.py`): Verifies cache prevents redundant subprocess calls
- **Validation scheduling** (`test_watch_mode.py`): Verifies validation triggers at correct thresholds
- **Short-circuit logic** (`test_quality_gate.py`): Verifies no git commands for obsolete resolution

These assertions test observable optimization behaviors that are part of the contract.

## Running Tests

See [CLAUDE.md](../CLAUDE.md#testing) for test commands and markers.
