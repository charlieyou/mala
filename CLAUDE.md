# Agent Instructions

## Python Environment

- **Package manager**: uv
- **Linting & formatting**: ruff
- **Type checking**: ty

```bash
uv sync                # Install dependencies
uv run <script>        # Run scripts
uv run pytest          # Run tests
uvx ruff check .       # Lint
uvx ruff format .      # Format
uvx ty check           # Type check
```
