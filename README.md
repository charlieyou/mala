# mala

**M**ulti-**A**gent **L**oop **A**rchitecture

A multi-agent system for processing beads issues in parallel using the Claude Agent SDK.

The name also derives from Sanskrit, where *mala* means "garland" or "string of beads" - fitting for a system that orchestrates beads issues in a continuous loop, like counting prayer beads.

## Why Mala?

**The core insight: agents degrade as context grows.**

LLM agents become unreliable as their context window fills up. Early in a session, an agent follows instructions precisely, catches edge cases, and produces clean code. But as context accumulates—tool outputs, file contents, previous attempts—performance degrades.

**The solution: small tasks, fresh context, automated verification.**

1. **Breaking work into atomic issues** — Each issue is sized to complete within ~100k tokens
2. **Starting each agent with cleared context** — Every issue gets a fresh agent session
3. **Running automated checks after completion** — Linting, tests, type checking, and code review
4. **Looping until done** — The orchestrator continuously spawns agents for ready issues

### Limitations

**This project is a work in progress.** Expect rough edges, breaking changes, and incomplete features.

- **Python-only**: Tailored for Python projects (pytest, ruff, ty)
- **Opinionated stack**: Assumes uv, ruff, and ty
- **Run in a sandbox**: Agent permissions are permissive
- **Garbage in, garbage out**: Output quality depends on issue quality

## Prerequisites

### Beads (`bd`)

[Beads](https://github.com/steveyegge/beads) is the issue tracking system that agents pull work from.

```bash
# Install (pick one)
npm install -g @beads/bd
# or: brew install steveyegge/beads/bd
# or: go install github.com/steveyegge/beads/cmd/bd@latest

# Initialize in your repo
cd /path/to/repo && bd init
```

### Claude Code

[Claude Code](https://docs.anthropic.com/en/docs/claude-code) CLI is the agent runtime.

```bash
npm install -g @anthropic-ai/claude-code
claude login
```

### Cerberus Review-Gate (Optional)

[Cerberus](https://github.com/charlieyou/cerberus) provides automated code review. Disable with `--disable review`.

```bash
claude /plugin marketplace add charlieyou/cerberus
claude /plugin install cerberus
```

## Installation

```bash
uv tool install . --reinstall
```

## Usage

```bash
mala run /path/to/repo                    # Run the parallel worker
mala run --max-agents 5 /path/to/repo     # Limit concurrent agents
mala run --scope epic:proj-abc /path/to/repo    # Process children of epic
mala run --scope ids:issue-1,issue-2 --order input /path/to/repo  # Specific issues in order
mala run --resume /path/to/repo            # Prioritize in_progress issues
mala run --watch --validate-every 10 /path/to/repo  # Keep polling and validate every N issues
mala status                               # Check locks, config, logs
mala clean                                # Clean up locks and logs
mala epic-verify proj-abc /path/to/repo   # Verify and close an epic
```

## How It Works

1. **Orchestrator** queries `bd ready --json` for available issues
2. **Filtering**: Epics are skipped - only tasks/bugs are processed
3. **Spawning**: Up to N parallel agent tasks (unlimited by default)
4. **Per-session pipeline**: Agent implements → evidence check → external review → close
5. **Global validation**: Final validation pass catches cross-issue regressions
6. **Epic verification**: When all children close, verifies acceptance criteria

### Agent Workflow

1. **Understand**: Read issue with `bd show <id>`
2. **Lock files**: Acquire filesystem locks before editing
3. **Implement**: Write code following project conventions
4. **Quality checks**: Run linters, formatters, type checkers
5. **Commit**: Stage and commit changes locally
6. **Cleanup**: Release locks (orchestrator closes issue after gate passes)

### Resolution Markers

Agents can signal non-implementation resolutions:

| Marker | Meaning |
|--------|---------|
| `ISSUE_NO_CHANGE` | Issue requires no code changes |
| `ISSUE_OBSOLETE` | Issue is no longer relevant |
| `ISSUE_ALREADY_COMPLETE` | Work was already done in a prior commit |

### Epics and Parent-Child Issues

- **Epics are skipped**: Issues with `issue_type: "epic"` are never assigned to agents
- **Parent-child is non-blocking**: Use `bd dep add <child> <epic> --type parent-child`
- **Verification before close**: When all children complete, the epic is verified against its acceptance criteria

## Coordination

| Layer | Tool | Purpose |
|-------|------|---------|
| Issue-level | Beads (`bd`) | Prevents duplicate claims via status updates |
| File-level | Filesystem locks | Prevents edit conflicts between agents |

### Lock Enforcement

File locks are enforced at two levels:

1. **MCP locking tools**: Agents acquire locks before editing files via `lock_acquire`/`lock_release` MCP tools
2. **PreToolUse hook**: Blocks file-write tool calls unless the agent holds the lock

### Git Safety

Dangerous git operations that can cause conflicts between concurrent agents are blocked:
`git stash`, `git reset --hard`, `git rebase`, `git checkout -f`, `git restore`, `git clean -f`, `git merge --abort`

## Creating Issues

Mala's effectiveness depends on well-structured beads issues. Each issue must be **self-contained and unambiguous**.

| Principle | Description |
|-----------|-------------|
| **Atomic** | One issue = one clear outcome |
| **Sized for agents** | Completable within ~100k tokens |
| **Minimal file overlap** | Issues touching same files cannot run in parallel |
| **Actionable** | Clear acceptance criteria and test plan |
| **Grounded** | Include exact file/line pointers when available |

See `commands/bd-breakdown.md` for the full issue creation workflow.

## Documentation

- [Architecture](docs/architecture.md) — Layered architecture, module responsibilities, key flows
- [CLI Reference](docs/cli-reference.md) — CLI options, environment variables, integrations
- [Project Configuration](docs/project-config.md) — mala.yaml schema, presets, coverage settings
- [Validation](docs/validation.md) — Evidence check, external review, global validation
- [Development](docs/development.md) — Type checking, testing, package structure
- `plans/` — Historical design documents (not actively maintained)

## Key Design Decisions

- **Event sink architecture** decouples orchestration from presentation
- **Filesystem locks** via atomic hardlink (sandbox-compatible)
- **Orchestrator claims issues** before spawning (agents don't claim)
- **Evidence check** verifies commits and validation before accepting work
- **Epic verification** uses AI to validate collective work against acceptance criteria
- **JSONL logs** in `~/.config/mala/logs/` for debugging
