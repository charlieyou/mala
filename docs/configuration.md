# Configuration

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--max-agents`, `-n` | unlimited | Maximum concurrent agents |
| `--timeout`, `-t` | 60 | Timeout per agent in minutes |
| `--max-issues`, `-i` | unlimited | Maximum total issues to process |
| `--epic`, `-e` | - | Only process tasks that are children of this epic |
| `--only`, `-o` | - | Comma-separated list of issue IDs to process exclusively |
| `--max-gate-retries` | 3 | Maximum quality gate retry attempts per issue |
| `--max-review-retries` | 3 | Maximum external review retry attempts per issue |
| `--review-timeout` | 1200 | Timeout in seconds for Cerberus review operations |
| `--cerberus-spawn-args` | - | Extra args appended to `review-gate spawn-code-review` |
| `--cerberus-wait-args` | - | Extra args appended to `review-gate wait` |
| `--cerberus-env` | - | Extra env for review-gate (JSON object or comma KEY=VALUE list) |
| `--disable-validations` | - | Comma-separated list (see below) |
| `--coverage-threshold` | 85 | Minimum coverage percentage (0-100); if not set, uses 'no decrease' mode |
| `--wip` | false | Prioritize in_progress issues before open issues |
| `--focus/--no-focus` | focus | Group tasks by epic for focused work; `--no-focus` uses priority-only ordering |
| `--dry-run` | false | Preview task order without processing |
| `--verbose`, `-v` | false | Enable verbose output; shows full tool arguments |
| `--no-morph` | false | Disable MorphLLM routing; use built-in tools directly |
| `--epic-override` | - | Comma-separated epic IDs to close without verification (human bypass) |
| `--debug-log` | false | Write debug logs to `~/.config/mala/runs/{repo-name}/{timestamp}.debug.log` |

### Disable Validation Flags

| Flag | Description |
|------|-------------|
| `post-validate` | Skip all per-issue validation (tests, lint, typecheck) |
| `run-level-validate` | Skip run-level validation |
| `integration-tests` | Exclude integration tests from pytest |
| `coverage` | Disable coverage checking |
| `e2e` | Disable E2E fixture repo test |
| `review` | Disable external review (Cerberus review-gate) |
| `followup-on-run-validate-fail` | Don't mark issues with `needs-followup` on run-level validation failure |

## Global Configuration

mala uses a global config directory at `~/.config/mala/`:

```
~/.config/mala/
├── .env          # API keys (ANTHROPIC_API_KEY, MORPH_API_KEY, BRAINTRUST_API_KEY)
└── logs/         # JSONL session logs
```

Environment variables are loaded from `~/.config/mala/.env` (global config).
Precedence: CLI flags override global config, which overrides program defaults.

### Directory Overrides

| Variable | Default | Description |
|----------|---------|-------------|
| `MALA_RUNS_DIR` | `~/.config/mala/runs` | Directory for run metadata |
| `MALA_LOCK_DIR` | `/tmp/mala-locks` | Directory for filesystem locks |
| `MALA_REVIEW_TIMEOUT` | `1200` | Review-gate wait timeout in seconds |

### Epic Verification

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_API_KEY` | - | API key for LLM calls (falls back to `ANTHROPIC_API_KEY`) |
| `LLM_BASE_URL` | - | Base URL for LLM API (for proxy/routing) |

### Cerberus Overrides

| Variable | Description |
|----------|-------------|
| `MALA_CERBERUS_SPAWN_ARGS` | Extra args for `review-gate spawn-code-review` |
| `MALA_CERBERUS_WAIT_ARGS` | Extra args for `review-gate wait` |
| `MALA_CERBERUS_ENV` | Extra env for review-gate (JSON object or comma KEY=VALUE list) |

Note: The repo's `.env` file is for testing only and is not loaded by the program.

## Braintrust Tracing

To enable Braintrust tracing for agent sessions, add your API key to the global config:

```bash
echo "BRAINTRUST_API_KEY=your-key" >> ~/.config/mala/.env
```

## MorphLLM MCP (Optional)

MorphLLM MCP provides enhanced editing and search tools (`edit_file`, `warpgrep_codebase_search`)
for agents. It is **optional** and enabled when `MORPH_API_KEY` is present.

**When enabled** (MORPH_API_KEY set and `--no-morph` not specified):
- MCP server is launched via `npx -y @morphllm/morphmcp`
- Agents use MCP tools (prefixed as `mcp__morphllm__*`)
- Built-in `Edit` and `Grep` tools are blocked to enforce MCP usage

**When disabled** (MORPH_API_KEY not set or `--no-morph` flag used):
- No MCP server is configured
- Agents use built-in `Edit` and `Grep` tools (both lock-enforced)
- Startup logs show "morph: disabled"

To enable MorphLLM:
```bash
echo "MORPH_API_KEY=your-key" >> ~/.config/mala/.env
```

To temporarily disable MorphLLM (useful for debugging or cost control):
```bash
mala run --no-morph /path/to/repo
```

## Logs

Agent logs are written in JSONL format to `~/.config/mala/logs/`:

```
<session-uuid>.jsonl
```

Check log status with:
```bash
mala status    # Shows recent logs and their timestamps
```

### Output Verbosity

mala supports two output modes controlled by `--verbose`:

| Mode | Flag | Description |
|------|------|-------------|
| **Normal** | (default) | Single line per tool call |
| **Verbose** | `--verbose` / `-v` | Full tool arguments in key=value format |

```bash
mala run /path/to/repo          # Normal output (default)
mala run -v /path/to/repo       # Verbose mode - full tool args
```

## Terminal Output

Agent output uses color-coded prefixes to distinguish concurrent agents:
- Each agent gets a unique bright color (cyan, yellow, magenta, green, blue, white)
- Log lines are prefixed with `[issue-id]` in the agent's color
- Tool usage, text output, and completion status are all color-coded
