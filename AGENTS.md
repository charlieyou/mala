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

## Code Migration Rules

- **No backward-compatibility shims**: When moving/renaming modules, update all imports directly. Never create re-export shims.
- **No re-exports**: Don't create modules that just import and re-export from elsewhere. If code moves, fix the imports.
