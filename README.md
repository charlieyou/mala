# mala

**M**ulti-**A**gent **L**oop **A**rchitecture

A multi-agent system for processing beads issues in parallel using the Claude Agent SDK.

The name also derives from Sanskrit, where *mala* means "garland" or "string of beads" - fitting for a system that orchestrates beads issues in a continuous loop, like counting prayer beads.

## Installation

```bash
uv tool install . --reinstall
```

## Usage

```bash
# Run the parallel worker (from repo with beads issues)
mala run /path/to/repo

# Or with options
mala run --max-agents 3 --timeout 30 --max-issues 5 /path/to/repo

# Check status (locks, config, logs)
mala status

# Clean up locks and logs
mala clean
```

## Architecture

```
mala (Python Orchestrator)
├── Spawns: N agents in parallel (asyncio tasks)
├── Each agent: ClaudeSDKClient session implementing one issue
├── Coordination: Filesystem locks prevent edit conflicts
└── Cleanup: Locks released on completion, timeout, or failure
```

## Coordination Layers

| Layer | Tool | Purpose |
|-------|------|---------|
| Issue-level | Beads (`bd`) | Prevents duplicate claims via status updates |
| File-level | Filesystem locks | Prevents edit conflicts between agents |

## How It Works

1. **Orchestrator** queries `bd ready --json` for available issues
2. **Filtering**: Epics (`issue_type: "epic"`) are automatically skipped - only tasks/bugs are processed
3. **Spawning**: Up to N parallel agent tasks (default: 3)
4. **Each agent**:
   - Gets assigned an issue (already claimed by orchestrator)
   - Acquires filesystem locks before editing any files
   - Implements the issue, runs quality checks, commits
   - Releases locks, closes issue
5. **On file conflict**: Agent polls every 1s for lock (up to 60s timeout), returns BLOCKED if unavailable
6. **Quality gate**: After completion, verifies commit exists and validation commands ran
7. **On failure/timeout**: Orchestrator releases orphaned locks, writes handoff file, resets issue status

### Epics and Parent-Child Issues

The orchestrator handles epics as follows:

- **Epics are skipped**: Issues with `issue_type: "epic"` are never assigned to agents
- **Parent-child is non-blocking**: Use `bd dep add <child> <epic> --type parent-child` to link tasks to epics without blocking
- **Auto-close**: After each run, `bd epic close-eligible` is called to auto-close epics where all children are complete

**Workflow:**
```bash
# Create epic
bd create "Feature X" -t epic -p 1  # Returns: proj-abc

# Create child tasks and link to epic
bd create "Implement part 1" -p 2
bd dep add proj-def proj-abc --type parent-child

# Check epic progress
bd epic status

# Orchestrator auto-closes epics when all children complete
```

## Agent Workflow

Each spawned agent follows this workflow:

1. **Understand**: Read issue with `bd show <id>`
2. **Lock files**: Acquire filesystem locks before editing
3. **Implement**: Write code following project conventions
4. **Quality checks**: Run linters, formatters, type checkers
5. **Self-review**: Verify implementation meets requirements
6. **Commit**: Stage and commit changes locally
7. **Cleanup**: Release locks, close issue with `bd close <id>`

## Quality Gate

After an agent completes an issue, the orchestrator runs a quality gate that verifies:

1. **Commit exists**: A git commit with `bd-<issue_id>` in the message
2. **Validation evidence**: The agent ran required checks (parsed from JSONL logs):
   - `pytest` - tests
   - `ruff check` and `ruff format` - linting/formatting
   - `ty check` - type checking
   - `uv sync` - dependency installation

If the gate fails, the issue is marked with `needs-followup` label and reset for retry.

## Lock Enforcement

File locks are enforced at two levels:

1. **Advisory locks**: Agents acquire locks before editing files via `lock-try.sh`
2. **PreToolUse hook**: A hook blocks file-write tool calls (`Write`, `NotebookEdit`, `mcp__morphllm__edit_file`) unless the agent holds the lock for that file

Lock keys are canonicalized so equivalent paths (absolute/relative, with `./..` segments) produce identical locks. When `REPO_NAMESPACE` is set, paths become repo-relative.

## Test Mutex

Repo-wide commands that could interfere between agents are serialized via `test-mutex.sh`:

```bash
./src/scripts/test-mutex.sh pytest           # Run tests with mutex
./src/scripts/test-mutex.sh ruff check .     # Lint with mutex
./src/scripts/test-mutex.sh ty check         # Type check with mutex
./src/scripts/test-mutex.sh uv sync          # Sync deps with mutex
```

The mutex uses a fixed key (`__test_mutex__`) and is released on exit, even on failure or signals.

## Failure Handoff

When an agent fails (including quality gate failures), a handoff file is written to preserve context for debugging or retry:

```
.mala/handoff/<issue_id>.md
```

Contains:
- Error summary
- Last tool error (if any)
- Last 10 Bash commands from the session log

## Key Design Decisions

- **Filesystem locks** via atomic hardlink (sandbox-compatible, no external deps)
- **Canonical lock keys** with path normalization and repo namespace support
- **60-second lock timeout** with 1s polling (fail fast on conflicts)
- **Lock enforcement hook** blocks writes to unlocked files
- **Orchestrator claims issues** before spawning (agents don't claim)
- **Quality gate** verifies commits and validation before accepting work
- **JSONL logs** in `~/.config/mala/logs/` for debugging
- **asyncio.wait** for efficient parallel task management

## Configuration

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--max-agents`, `-n` | 3 | Maximum concurrent agents |
| `--timeout`, `-t` | 30 | Timeout per agent in minutes |
| `--max-issues`, `-i` | unlimited | Maximum total issues to process |

### Global Configuration

mala uses a global config directory at `~/.config/mala/`:

```
~/.config/mala/
├── .env          # API keys (ANTHROPIC_API_KEY, MORPH_API_KEY, BRAINTRUST_API_KEY)
└── logs/         # JSONL session logs
```

Environment variables are loaded in order (later overrides earlier):
1. `~/.config/mala/.env` (global config)
2. `<repo>/.env` (repo-specific overrides)

### Braintrust Tracing

To enable Braintrust tracing for agent sessions, add your API key to the global config:

```bash
echo "BRAINTRUST_API_KEY=your-key" >> ~/.config/mala/.env
```

### MorphLLM MCP

mala configures the MorphLLM MCP server for all agents and requires `MORPH_API_KEY` to run.
The server is launched via `npx -y @morphllm/morphmcp`, and agents use MCP tools like
`edit_file` and `warpgrep_codebase_search` (prefixed as `mcp__morphllm__*`). The built-in
`Edit` and `Grep` tools are blocked to enforce MCP usage.

```bash
echo "MORPH_API_KEY=your-key" >> ~/.config/mala/.env
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

### Log Truncation

By default, log output is truncated for readability. Use `--no-truncate` to see full output:

```bash
mala run --no-truncate /path/to/repo    # Full tool arguments and summaries
```

Tool calls display abbreviated `key=value` pairs by default, or pretty-printed JSON with `--no-truncate`.

## Terminal Output

Agent output uses color-coded prefixes to distinguish concurrent agents:
- Each agent gets a unique bright color (cyan, yellow, magenta, green, blue, white)
- Log lines are prefixed with `[issue-id]` in the agent's color
- Tool usage, text output, and completion status are all color-coded

