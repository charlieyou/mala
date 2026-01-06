# Agent Instructions

<!-- Keep in sync with CLAUDE.md -->

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

All code changes should include appropriate tests. See [tests/AGENTS.md](tests/AGENTS.md) for testing commands, philosophy, and guidelines.

## Code Migration Rules

- **No backward-compatibility shims**: When moving/renaming modules, update all imports directly. Never create re-export shims.
- **No re-exports**: Don't create modules that just import and re-export from elsewhere. If code moves, fix the imports.
