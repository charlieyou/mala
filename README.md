# mala

A multi-agent system for processing beads issues in parallel using the Claude Agent SDK.

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
5. **On file conflict**: Agent waits up to 60s for lock, returns BLOCKED if unavailable
6. **On failure/timeout**: Orchestrator releases orphaned locks, resets issue status

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

## Key Design Decisions

- **Filesystem locks** via atomic mkdir (sandbox-compatible, no external deps)
- **60-second lock timeout** (fail fast on conflicts)
- **Orchestrator claims issues** before spawning (agents don't claim)
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

## Terminal Output

Agent output uses color-coded prefixes to distinguish concurrent agents:
- Each agent gets a unique bright color (cyan, yellow, magenta, green, blue, white)
- Log lines are prefixed with `[issue-id]` in the agent's color
- Tool usage, text output, and completion status are all color-coded

## TODOs

Braintrust LLM tracing is now implemented - see "Braintrust Tracing" section above.

Ideas
* Add back claude orchestrator, manually managing context
* agent mail using filesystem?
* use codex for code review?

### Context Exhaustion Handling
When agents run out of context mid-implementation:
- [ ] Detect context exhaustion (agent returns incomplete)
- [ ] Save progress state (files modified, locks held)
- [ ] Spawn continuation agent with summary of prior work
- [ ] Or: commit partial progress, mark issue as needs-continuation

### Other Improvements
- [ ] Deadlock detection (two agents waiting on each other)
- [ ] Metrics/logging for multi-agent sessions
- [ ] Graceful shutdown (signal all agents to wrap up)
