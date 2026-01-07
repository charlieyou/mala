# Development

## Dependencies

```bash
uv sync
```

## Type Checking

mala uses strict type checking with both `ty` and `ruff`:

```bash
uvx ty check         # Type check with strict rules
uvx ruff check .     # Lint with type annotation rules
uvx ruff format .    # Format code
```

All ty rules are set to `error` level in `pyproject.toml` for maximum strictness.

## Test Coverage

Tests require 72% minimum coverage (enforced via `--cov-fail-under=72`):

```bash
uv run pytest                              # Unit + integration tests (default, excludes e2e, no coverage)
uv run pytest -m unit                      # Unit tests only
uv run pytest -m integration -n auto       # Integration tests in parallel
uv run pytest -m e2e                       # End-to-end tests (requires CLI auth)
uv run pytest -m "unit or integration or e2e"  # All tests
uv run pytest --reruns 2                   # Auto-retry flaky tests (2 retries)
uv run pytest -m integration -n auto --reruns 2   # Parallel + auto-retry
uv run pytest --cov=src --cov-fail-under=72       # Manual coverage check
```

- **Parallel execution**: Use `-n auto` for parallel test runs (pytest-xdist)
- **Flaky test retries**: Use `--reruns N` to auto-retry failed tests (pytest-rerunfailures)
- **Unit/Integration/E2E**: Use markers `unit`, `integration`, `e2e` to select categories
- **Coverage**: Not run by default; quality gate adds coverage flags automatically

## Package Structure

The codebase is organized into layered packages with enforced import boundaries (via import-linter):

```
src/
├── core/           # Models, protocols, log events (no internal dependencies)
├── domain/         # Business logic: lifecycle, evidence_check, validation, prompts
├── infra/          # Infrastructure: clients/, io/, tools/, hooks/
├── pipeline/       # Agent session pipeline, gate/review runners, run coordinator
├── orchestration/  # Orchestrator, factory, CLI support
├── cli/            # CLI entry point
├── prompts/        # Prompt templates
└── scripts/        # Utility scripts
```

**Layer dependencies** (enforced by import-linter contracts):
- `core` → (none)
- `domain` → `core`
- `infra` → `core`
- `pipeline` → `core`, `domain`, `infra`
- `orchestration` → `core`, `domain`, `infra`, `pipeline`
- `cli` → all layers

For detailed architecture documentation, see [architecture.md](architecture.md).
