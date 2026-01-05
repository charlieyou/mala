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
2. Tests assert on behavior (outputs, state) not interactions
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

See [CLAUDE.md](../CLAUDE.md#testing) for full test commands.

```bash
uv run pytest                              # Unit + integration (default)
uv run pytest -m unit                      # Unit tests only
uv run pytest -m integration -n auto       # Integration in parallel
uv run pytest -m e2e                       # E2E tests (requires auth)
```

## Documented Mock/Patch Exceptions

The following mock and patch usages are intentional exceptions to the "fakes over mocks" guideline. Each is documented with rationale.

### Patches for External Dependencies

| File | Patch Target | Rationale |
|------|--------------|-----------|
| `tests/unit/infra/tools/test_locking_mcp.py` | `try_lock`, `get_lock_holder`, `wait_for_lock_async`, `release_lock`, `cleanup_agent_locks` | MCP tool handlers use module-level functions. Patches isolate handler logic from filesystem operations. |
| `tests/unit/pipeline/test_idle_retry_policy.py` | `asyncio.sleep` | Standard pattern for testing async timing/backoff without actual delays. |
| `tests/unit/domain/test_validation.py` | `SpecResultBuilder._run_e2e`, `create_worktree`, `is_baseline_stale`, `LintCache.should_skip` | E2E/worktree/coverage internals that require complex setup or filesystem access. |
| `tests/unit/pipeline/test_run_metadata.py` | `get_runs_dir`, env vars | Tests run metadata persistence with controlled paths. |
| `tests/unit/pipeline/test_watch_mode.py` | `time.monotonic` | Controls time progression for deterministic timing tests. |
| `tests/unit/domain/test_spec_workspace.py` | `create_worktree`, `remove_worktree` | Tests worktree lifecycle without git worktree operations. |
| `tests/unit/infra/test_cerberus_review.py` | `CommandRunner`, `os.environ` | Tests review orchestration with controlled subprocess behavior. |
| `tests/unit/infra/test_cerberus_gate_cli.py` | `os.environ` | Tests CLI behavior with controlled environment. |
| `tests/unit/orchestration/test_orchestrator.py` | `AgentSessionConfig.__init__` | Tests orchestrator wiring without full SDK initialization. |
| `tests/unit/cli/test_cli.py` | `USER_CONFIG_DIR` env | Tests CLI with controlled config paths. |
| `tests/unit/infra/test_beads_client.py` | `monkeypatch.setattr` for internal methods | Tests beads client caching and subprocess behavior. |

### MagicMock/AsyncMock Usage

| File/Pattern | Rationale |
|--------------|-----------|
| `tests/unit/domain/test_quality_gate.py` | Uses `MagicMock` for `CommandRunnerPort` to satisfy constructor. Tests focus on log parsing, not command execution. Migration to `FakeCommandRunner` tracked separately. |
| `MagicMock(spec=LintCache)` | `LintCache` satisfies `LintCacheProtocol`. Mock with spec ensures interface compliance. |
| `AsyncMock` for callbacks | Callback functions are simple callables. AsyncMock tracks invocation without needing a full fake. |
| `MagicMock` for SDK internals | Claude SDK types have complex initialization. Mocking with spec validates interface usage. |

### Observable State vs call_count

Tests should prefer observable state assertions over `call_count`. However, some tests use `call_count` or track invocations to verify optimization behaviors that are part of the contract:

- **Caching** (`test_beads_client.py`): Verifies cache prevents redundant subprocess calls by tracking call lists
- **Validation scheduling** (`test_watch_mode.py`): Verifies validation triggers at correct thresholds via callback invocation tracking
- **Short-circuit logic** (`test_quality_gate.py`): Verifies no commands run for obsolete resolution via `FakeCommandRunner.calls`

These patterns test **optimization guarantees**, not implementation details. Prefer tracking calls in a list (observable state) over `AsyncMock.call_count` where practical.
