# Agent Instructions

## Python Environment

- **Package manager**: uv
- **Linting & formatting**: ruff
- **Type checking**: ty

```bash
uv sync                # Install dependencies
uv run <script>        # Run scripts
uvx ruff check .       # Lint
uvx ruff format .      # Format
uvx ty check           # Type check
```

## Testing

```bash
uv run pytest                              # Unit tests only (default, excludes slow)
uv run pytest -m slow -n auto              # Slow/integration tests in parallel
uv run pytest -m "slow or not slow"        # All tests
uv run pytest --reruns 2                   # Auto-retry flaky tests (2 retries)
uv run pytest -m slow -n auto --reruns 2   # Parallel + auto-retry
```

- **Coverage threshold**: 72% (enforced via `--cov-fail-under=72`)
- **Parallel execution**: Use `-n auto` for parallel test runs (pytest-xdist)
- **Flaky test retries**: Use `--reruns N` to auto-retry failed tests (pytest-rerunfailures)
- **Slow tests**: Marked with `@pytest.mark.slow`, excluded by default
