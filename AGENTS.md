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

## Documentation

- [docs/architecture.md](docs/architecture.md) - System architecture and design
- [docs/cli-reference.md](docs/cli-reference.md) - CLI commands and usage
- [docs/development.md](docs/development.md) - Development setup and workflow
- [docs/project-config.md](docs/project-config.md) - Project configuration (mala.yaml)
- [docs/validation.md](docs/validation.md) - Validation rules and schemas
