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
├── unit/           # Isolated unit tests (no network, no subprocess)
├── integration/    # Cross-layer tests validating component composition
├── e2e/            # End-to-end smoke tests (requires CLI auth)
├── fakes/          # In-memory fake implementations of protocols
├── fixtures/       # Shared test data and fixtures
└── conftest.py     # Pytest configuration and shared fixtures
```

## Fakes Module (`tests/fakes/`)

The `fakes/` module will provide in-memory implementations of mala protocols for testing.

**Planned fakes** (to be implemented):

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

### Usage (once fakes are implemented)

```python
# Example usage pattern (FakeIssueProvider not yet available)
from tests.fakes import FakeIssueProvider

def test_issue_processing():
    provider = FakeIssueProvider()
    provider.add_issue(Issue(id="test-1", title="Test"))

    # Test code that uses provider
    result = process_issues(provider)

    # Assert on outcomes, not calls
    assert result.processed_count == 1
```

## Running Tests

```bash
uv run pytest                              # Unit + integration (default)
uv run pytest -m unit                      # Unit tests only
uv run pytest -m integration -n auto       # Integration in parallel
uv run pytest -m e2e                       # E2E tests (requires auth)
```

## Baseline Metrics (2026-01-05)

Current mock/patch usage to be reduced through fakes migration:

| Metric | Count |
|--------|-------|
| `MagicMock`/`AsyncMock` | 854 |
| `with patch`/`@patch` | 196 |
| `.assert_called*` | 64 |

Goal: Reduce these counts by migrating tests to use fakes from `tests/fakes/`.
