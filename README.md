# mala

A multi-agent system for processing beads issues in parallel using the Claude Agent SDK.

## Installation

```bash
cd mala
uv sync
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

1. **Orchestrator** spawns up to N parallel agent tasks (default: 3)
2. **Each agent**:
   - Gets assigned an issue (already claimed by orchestrator)
   - Acquires filesystem locks before editing any files
   - Implements the issue, runs quality checks, commits
   - Releases locks, closes issue
3. **On file conflict**: Agent waits up to 60s for lock, returns BLOCKED if unavailable
4. **On failure/timeout**: Orchestrator releases orphaned locks, resets issue status

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
├── .env          # API keys (ANTHROPIC_API_KEY, BRAINTRUST_API_KEY)
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
