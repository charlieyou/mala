# mala

**M**ulti-**A**gent **L**oop **A**rchitecture

A multi-agent system for processing beads issues in parallel using the Claude Agent SDK.

The name also derives from Sanskrit, where *mala* means "garland" or "string of beads" - fitting for a system that orchestrates beads issues in a continuous loop, like counting prayer beads.

## Prerequisites

### Beads (`bd`)

Mala requires [Beads](https://github.com/cyouAI/beads) to be installed and configured. Beads is the issue tracking system that agents pull work from.

```bash
# Install beads
uv tool install beads

# Initialize beads in your target repo
cd /path/to/repo
bd init
```

### Creating Issues Correctly

Mala's effectiveness depends entirely on well-structured beads issues. Agents work autonomously without human guidance, so each issue must be **self-contained and unambiguous**.

**Issue requirements for parallel agent execution:**

| Principle | Description |
|-----------|-------------|
| **Atomic** | One issue = one clear outcome. Avoid bundling unrelated fixes. |
| **Sized for agents** | Each issue must be completable within ~100k tokens. Split larger work into epics with child issues. |
| **Minimal file overlap** | Issues that touch the same files cannot run in parallel. Use dependencies to serialize overlapping work. |
| **Actionable** | Every issue needs clear acceptance criteria and a test plan. |
| **Grounded** | Only create issues with sufficient context. Include exact file/line pointers when available. |

**Issue structure:**

```
Title: Clear, specific description

Context:
- Background explaining why this matters
- File/line pointers to relevant code

Scope:
- In: what will be changed
- Out: explicit non-goals

Acceptance Criteria:
- Testable outcomes

Test Plan:
- How to verify the fix
```

**Dependencies:**

- Use `bd dep add <child> <parent>` only when two issues must touch the same file
- Use `bd dep add <child> <epic> --type parent-child` to group related tasks under an epic without blocking

See `commands/bd-breakdown.md` for the full issue creation workflow.

## Installation

```bash
uv tool install . --reinstall
```

## Usage

```bash
# Run the parallel worker (from repo with beads issues)
mala run /path/to/repo

# Or with options
mala run --max-agents 5 --timeout 30 --max-issues 10 /path/to/repo

# Process only children of a specific epic
mala run --epic proj-abc /path/to/repo

# Process specific issues only
mala run --only issue-1,issue-2 /path/to/repo

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
3. **Spawning**: Up to N parallel agent tasks (unlimited by default)
4. **Each agent**:
   - Gets assigned an issue (already claimed by orchestrator)
   - Acquires filesystem locks before editing any files
   - Implements the issue, runs quality checks, commits
   - Releases locks (orchestrator handles issue closing)
5. **On file conflict**: Agent polls every 1s for lock (up to 60s timeout), returns BLOCKED if unavailable
6. **Quality gate**: After agent completes, orchestrator verifies commit exists and validation commands ran
7. **Same-session re-entry**: If gate fails, orchestrator resumes the SAME Claude session with failure context
8. **Codex review** (optional): After gate passes, automated code review with fix cycle
9. **On success**: Orchestrator closes the issue via `bd close`
10. **On failure**: After retries exhausted, orchestrator marks issue with `needs-followup` label and records log path in notes

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
7. **Cleanup**: Release locks (orchestrator closes issue after gate passes)

Note: Agents do NOT close issues directly. The orchestrator closes issues only after the quality gate (and optional Codex review) passes.

## Quality Gate

After an agent completes an issue, the orchestrator runs a quality gate that verifies:

1. **Commit exists**: A git commit with `bd-<issue_id>` in the message
2. **Validation evidence**: The agent ran required checks (parsed from JSONL logs):
   - `pytest` - tests
   - `ruff check` and `ruff format` - linting/formatting
   - `ty check` - type checking
   - `uv sync` - dependency installation

### Same-Session Re-entry

If the gate fails, the orchestrator **resumes the same Claude session** with a follow-up prompt containing:
- The specific failure reasons (missing commit, missing validations)
- Instructions to fix and re-run validations
- The current attempt number (e.g., "Attempt 2/3")

This continues for up to `max_gate_retries` attempts (default: 3). The orchestrator tracks:
- **Log offset**: Only evidence from the current attempt is considered
- **Previous commit hash**: Detects "no progress" when commit is unchanged
- **No-progress detection**: Stops retries early if agent makes no meaningful changes

### Codex Review (Optional)

When `--codex-review` is enabled, after the deterministic gate passes:

1. **Codex review runs**: Invokes `codex review --commit <sha>` with a strict JSON schema prompt
2. **JSON parsing**: Output must be valid JSON with `passed` boolean and `issues` array
3. **Parse retry**: If JSON is invalid, retries once with stricter prompt (fail-closed behavior)
4. **Review failure handling**: If review finds errors, orchestrator resumes the SAME session with:
   - List of issues (file, line, severity, message)
   - Instructions to fix errors and re-run validations
5. **Re-gating**: After fixes, runs both deterministic gate AND Codex review again

Review retries are capped at `max_review_retries` (default: 2).

### Failure Handling

After all retries are exhausted (gate or review), the orchestrator:
- Marks the issue with `needs-followup` label
- Records error summary and log path in issue notes
- Does NOT close the issue (leaves it for manual intervention)

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

When an agent fails (including quality gate failures after all retries), the orchestrator records context in the beads issue notes:
- Error summary (gate failures, review issues, timeout, etc.)
- Path to the JSONL session log (in `~/.mala/logs/`)
- Attempt counts (gate attempts, review attempts)

The next agent (or human) can read the issue notes with `bd show <issue_id>` and grep the log file for context.

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
| `--max-agents`, `-n` | unlimited | Maximum concurrent agents |
| `--timeout`, `-t` | 30 | Timeout per agent in minutes |
| `--max-issues`, `-i` | unlimited | Maximum total issues to process |
| `--epic`, `-e` | - | Only process tasks that are children of this epic |
| `--only`, `-o` | - | Comma-separated list of issue IDs to process exclusively |
| `--max-gate-retries` | 3 | Maximum quality gate retry attempts per issue |
| `--max-review-retries` | 2 | Maximum Codex review retry attempts per issue |
| `--codex-review` | disabled | Enable automated Codex code review after gate passes |
| `--verbose`, `-v` | false | Enable verbose output with full tool arguments |

### Global Configuration

mala uses a global config directory at `~/.config/mala/`:

```
~/.config/mala/
├── .env          # API keys (ANTHROPIC_API_KEY, MORPH_API_KEY, BRAINTRUST_API_KEY)
└── logs/         # JSONL session logs
```

Environment variables are loaded from `~/.config/mala/.env` (global config).

Note: The repo's `.env` file is for testing only and is not loaded by the program.

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

