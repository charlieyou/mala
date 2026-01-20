# mala

[![PyPI version](https://img.shields.io/pypi/v/mala-agent.svg)](https://pypi.org/project/mala-agent/)

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

## Prerequisites

### Beads

[Beads](https://github.com/Dicklesworthstone/beads_rust) is the issue tracking system that agents pull work from. See the repo for installation instructions.

### Claude Code

[Claude Code](https://code.claude.com/docs/en/setup) CLI is the agent runtime. See the docs for installation instructions.

### Cerberus Review-Gate (Optional)

[Cerberus](https://github.com/charlieyou/cerberus) provides automated code review when `reviewer_type: cerberus`
is enabled in `mala.yaml`. If you use `reviewer_type: agent_sdk`, no Cerberus install is required.

```bash
claude /plugin marketplace add charlieyou/cerberus
claude /plugin install cerberus
```

## Installation

```bash
uv tool install mala-agent
```

## Usage

```bash
mala init                                 # Interactively create mala.yaml
mala init --yes --preset python-uv         # Non-interactive init with defaults
mala run /path/to/repo                    # Run the parallel worker
mala run --max-agents 5 /path/to/repo     # Limit concurrent agents
mala run --scope epic:proj-abc /path/to/repo    # Process children of epic
mala run --scope ids:issue-1,issue-2 --order input /path/to/repo  # Specific issues in order
mala run --resume /path/to/repo            # Include in_progress issues and resume sessions
mala run --strict --resume /path/to/repo   # Fail if a resumed issue has no session
mala run --watch /path/to/repo             # Keep polling for new issues
mala status                               # Check locks, config, logs
mala status --all                          # Show running instances across directories
mala logs list                            # List recent runs
mala logs sessions --issue ISSUE-123      # Find sessions for an issue
mala logs show <run_id_prefix>            # Show run metadata
mala clean                                # Clean up locks
mala clean --force                         # Clean even if mala is running
mala epic-verify proj-abc /path/to/repo   # Verify and close an epic
```

## How It Works

1. **Orchestrator** queries `bd ready --json` for available issues
2. **Filtering**: Epics are skipped - only tasks/bugs are processed
3. **Spawning**: Up to N parallel agent tasks (unlimited by default)
4. **Per-session pipeline**: Agent implements → quality gate (commit + evidence) → session_end trigger (optional) → external review → close
5. **Trigger validation**: `periodic`, `epic_completion`, and `run_end` triggers run configured commands with optional fixer remediation
6. **Epic verification**: When all children close, verifies acceptance criteria

### Agent Workflow

1. **Understand**: Read issue details (injected into prompt)
2. **Lock files**: Acquire filesystem locks before editing
3. **Implement**: Write code following project conventions
4. **Quality checks**: Run the required validations for evidence (see `evidence_check` in `mala.yaml`)
5. **Commit**: Stage and commit changes locally
6. **Session-end validation**: Orchestrator may run additional commands after gate passes
7. **Cleanup**: Release locks (orchestrator closes issue after gate + review)

### Resolution Markers

Agents can signal non-implementation resolutions:

| Marker | Meaning |
|--------|---------|
| `ISSUE_NO_CHANGE` | Issue requires no code changes |
| `ISSUE_OBSOLETE` | Issue is no longer relevant |
| `ISSUE_ALREADY_COMPLETE` | Work was already done in a prior commit |
| `ISSUE_DOCS_ONLY` | Documentation-only changes; skip validation evidence |

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

Dangerous commands are blocked to avoid destructive or conflicting actions:

- **Destructive git operations**: `git reset --hard|--soft|--mixed`, `git reset HEAD`, `git checkout -f|--force|--`, `git restore`,
  `git clean -f|-fd`, `git rebase`, `git commit --amend`, `git branch -D`, `git merge --abort`, `git rebase --abort`,
  `git cherry-pick --abort`, `git worktree remove`, `git submodule deinit -f`, `git stash`
- **Dangerous shell patterns**: `rm -rf /`, `rm -rf ~`, fork bombs, `mkfs.*`, raw disk writes, `curl|wget | bash/sh`

The hook errors include safe alternatives where possible.

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
- [Validation](docs/validation.md) — Evidence check, session_end, review gates, trigger validation
- [Validation Triggers](docs/validation-triggers.md) — Trigger-based validation and code review
- [Development](docs/development.md) — Type checking, testing, package structure
- `plans/` — Historical design documents (not actively maintained)

## Running in a Sandbox

Mala spawns AI agents with permissive tool access. **Running in a container is strongly recommended** to limit blast radius if an agent misbehaves.

### DevContainer (Recommended)

This repo includes a DevContainer configuration for developing mala:

```bash
devcontainer up --workspace-folder .
devcontainer exec --workspace-folder . mala run /workspaces/mala
```

The DevContainer mounts:
- `/workspaces/mala` — the mala source code
- `/.claude` — Claude Code auth and plugins (including Cerberus)
- `/.codex` — Codex CLI config
- `/.gemini` — Gemini CLI config
- `/.config/mala` — mala logs and run state

Pre-installed tools: Claude Code, Codex CLI, Gemini CLI, bd (Beads), uv, Python 3.12, Node.js

### What DevContainers Protect Against

| Risk | Protected? |
|------|------------|
| Modifying files outside mounted dirs | ✅ Yes |
| Accessing host processes | ✅ Yes |
| Persisting malware on host | ✅ Yes |
| Reading mounted sensitive files | ❌ No |
| Network exfiltration | ❌ No (full network access) |

DevContainers provide **process isolation** (prevent accidents) not **security isolation** (prevent malice).
