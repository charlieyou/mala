# Testing

These instructions apply to all tests under `tests/` in this repo.

## Running Tests

```bash
uv run pytest                              # Unit + integration tests (default, excludes e2e)
uv run pytest -m unit                      # Unit tests only
uv run pytest -m integration -n auto       # Integration tests in parallel
uv run pytest -m e2e                       # End-to-end tests (requires CLI auth)
uv run pytest -m "unit or integration or e2e"  # All tests
uv run pytest --reruns 2                   # Auto-retry flaky tests (2 retries)
uv run pytest -m integration -n auto --reruns 2   # Parallel + auto-retry
```

- **Coverage threshold**: 72% (enforced via `--cov-fail-under=72`)
- **Parallel execution**: Use `-n auto` for parallel test runs (pytest-xdist)
- **Flaky test retries**: Use `--reruns N` to auto-retry failed tests (pytest-rerunfailures)
- **Unit/Integration/E2E**: Use markers `unit`, `integration`, `e2e` to select categories

## Testing Philosophy

- **Behavior over implementation**: Test externally visible behavior (outputs, state changes, emitted events), not implementation details. Avoid `call_count`, exact call ordering, and "was called" assertions.
- **Fakes over mocks**: Use in-memory fakes implementing protocols (`FakeSDKClient`, `FakeGateChecker`, `FakeIssueProvider`). Use `MagicMock` only as last resort for hard-to-fake edges.
- **Don't test the obvious**: Skip tests for dataclass defaults, Python stdlib, or third-party library behavior. Only test mala-specific rules and protocol contracts.
- **Keep it small**: If a test doesn't catch a meaningful bug or protect a contract, delete it. Prefer fewer high-signal tests over exhaustive low-value coverage.

## Unit Tests (`unit/`)

- **Isolated**: No network, no real git, no subprocess, no global state. Inject fakes via constructor.
- **Single behavior**: Arrange minimal inputs → Act once → Assert on domain objects and state transitions.
- **Table-driven**: Use `@pytest.mark.parametrize` for behavior matrices instead of duplicate tests.
- **Assert decisions**: Test "what the component chose to do" given inputs, not incidental details.

## Integration Tests (`integration/`)

- **Cross-layer**: Validate real components compose correctly across layer boundaries (core ↔ infra).
- **Deterministic**: Use `tmp_path` and real file parsing, but fake network/subprocess via ports.
- **Contract tests**: When adding a new protocol/adapter, add a contract test that validates both the fake and real implementation.

## E2E Tests (`e2e/`)

- **User-visible outcomes**: Assert exit status, produced artifacts, final verdicts. No internal call assertions.
- **Few and stable**: Treat as smoke tests, not edge-case coverage. If flaky, move logic to integration/unit tests.
