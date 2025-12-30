# mala

**M**ulti-**A**gent **L**oop **A**rchitecture

A multi-agent system for processing beads issues in parallel using the Claude Agent SDK.

The name also derives from Sanskrit, where *mala* means "garland" or "string of beads" - fitting for a system that orchestrates beads issues in a continuous loop, like counting prayer beads.

## Why Mala?

**The core insight: agents degrade as context grows.**

LLM agents become unreliable as their context window fills up. Early in a session, an agent follows instructions precisely, catches edge cases, and produces clean code. But as context accumulates—tool outputs, file contents, previous attempts—performance degrades. The agent starts missing requirements, making sloppy mistakes, or getting stuck in loops.

This creates a fundamental tension: complex features require sustained work, but agents can't sustain quality over long sessions.

**The solution: small tasks, fresh context, automated verification.**

Mala addresses this by:

1. **Breaking work into atomic issues** — Each issue is sized to complete within ~100k tokens, small enough that agents finish before context degradation kicks in
2. **Starting each agent with cleared context** — Every issue gets a fresh agent session with only the issue description and codebase, no accumulated baggage
3. **Running automated checks after completion** — Linting, tests, type checking, and code review catch mistakes before they compound
4. **Looping until done** — The orchestrator continuously spawns agents for ready issues, with quality gates ensuring each piece of work is solid before moving on

This architecture turns unreliable long-context agents into reliable short-context workers, each doing one thing well before handing off to the next.

### Limitations

**This project is a work in progress.** Expect rough edges, breaking changes, and incomplete features.

Current constraints:
- **Python-only**: The validation system, quality gates, and tooling are tailored for Python projects (pytest, ruff, ty). Other languages aren't supported yet.
- **Opinionated stack**: Assumes uv for package management, ruff for linting/formatting, and ty for type checking
- **Single-repo focus**: Designed for monorepo workflows; multi-repo coordination isn't implemented
- **Run in a sandbox**: Agent permissions are permissive; run in an isolated environment (container, VM) to limit blast radius
- **Garbage in, garbage out**: Agent output quality depends entirely on issue quality. Vague or poorly-scoped issues produce poor results.

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

### Claude Code

Mala uses the [Claude Code](https://docs.anthropic.com/en/docs/claude-code) CLI as the agent runtime. Install and authenticate:

```bash
npm install -g @anthropic-ai/claude-code
claude login
```

### Cerberus Review-Gate (Optional)

[Cerberus](https://github.com/cyouAI/cerberus) provides automated code review via external reviewers (Codex, Gemini, Claude) after agents complete their work. Install if you want automated review:

```bash
# Cerberus CLI (optional - for automated code review)
pip install cerberus-review
```

Review can be disabled with `--disable-validations=review`.

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
mala run --max-agents 5 --max-issues 10 /path/to/repo

# With a timeout
mala run --timeout 30 /path/to/repo

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
8. **External review**: After gate passes, automated code review via Cerberus review-gate (disable with `--disable-validations=review`)
9. **On success**: Orchestrator closes the issue via `bd close`
10. **On failure**: After retries exhausted, orchestrator marks issue with `needs-followup` label and records log path in notes
11. **Run-level validation**: After all issues complete, a final validation pass catches cross-issue regressions (see [Run-Level Validation](#run-level-validation))

### Epics and Parent-Child Issues

The orchestrator handles epics as follows:

- **Epics are skipped**: Issues with `issue_type: "epic"` are never assigned to agents
- **Parent-child is non-blocking**: Use `bd dep add <child> <epic> --type parent-child` to link tasks to epics without blocking
- **Immediate auto-close**: Epics are closed immediately after each child completes (not deferred to end of run)

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

Note: Agents do NOT close issues directly. The orchestrator closes issues only after the quality gate (and optional external review) passes.

### Resolution Markers

Agents can signal non-implementation resolutions by printing these markers:

| Marker | Meaning | Outcome |
|--------|---------|---------|
| `ISSUE_NO_CHANGE` | Issue requires no code changes (already fixed, docs-only, etc.) | Orchestrator closes issue without requiring commit |
| `ISSUE_OBSOLETE` | Issue is no longer relevant (superseded, invalid, etc.) | Orchestrator closes issue without requiring commit |
| `ISSUE_ALREADY_COMPLETE` | Work was already done in a prior commit (agent found existing solution) | Orchestrator closes issue, referencing the prior commit |

These markers allow agents to handle issues that don't need implementation without failing the quality gate. All three skip external review since there's no new code to review.

## Quality Gate

After an agent completes an issue, the orchestrator runs a quality gate that verifies:

1. **Commit exists**: A git commit with `bd-<issue_id>` in the message, created during the current run (stale commits from previous runs are rejected via baseline timestamp)
2. **Validation evidence**: The agent ran ALL required checks (parsed from JSONL logs):
   - `pytest` - tests
   - `ruff check` - linting
   - `ruff format` - formatting
   - `ty check` - type checking
3. **Commands succeeded**: All validation commands must exit with zero status (non-zero exits fail the gate)

All validation commands must run AND pass for the gate to pass. Partial validation (e.g., only tests) is rejected. The required commands are spec-driven via `ValidationSpec`.

### Same-Session Re-entry

If the gate fails, the orchestrator **resumes the same Claude session** with a follow-up prompt containing:
- The specific failure reasons (missing commit, missing validations)
- Instructions to fix and re-run validations
- The current attempt number (e.g., "Attempt 2/3")

This continues for up to `max_gate_retries` attempts (default: 3). The orchestrator tracks:
- **Log offset**: Only evidence from the current attempt is considered
- **Previous commit hash**: Detects "no progress" when commit is unchanged
- **No-progress detection**: Stops retries early if agent makes no meaningful changes

### External Review (Cerberus Review-Gate)

External review via Cerberus is enabled by default. After the deterministic gate passes:

1. **Review spawns**: Cerberus `review-gate` spawns external reviewers (Codex, Gemini, Claude) to review the diff
2. **Scope verification**: Reviewers check the diff against the issue description and acceptance criteria to catch incomplete implementations
3. **Consensus**: All available reviewers must unanimously pass for the review to pass
4. **Review failure handling**: If any reviewer finds errors, orchestrator resumes the SAME session with:
   - List of issues (file, line, priority, message) from all reviewers
   - Instructions to fix errors and re-run validations
   - Cumulative diff from baseline (includes all work across retry attempts)
5. **Re-gating**: After fixes, runs both deterministic gate AND external review again

Review retries are capped at `max_review_retries` (default: 3). Use `--disable-validations=review` to disable.

**Skipped for no-work resolutions**: Issues resolved with `ISSUE_NO_CHANGE`, `ISSUE_OBSOLETE`, or `ISSUE_ALREADY_COMPLETE` skip external review entirely since there's no new code to review.

### Failure Handling

After all retries are exhausted (gate or review), the orchestrator:
- Marks the issue with `needs-followup` label
- Records error summary and log path in issue notes
- Does NOT close the issue (leaves it for manual intervention)

## Run-Level Validation

After all issues complete, the orchestrator runs a final validation pass. This catches issues that only manifest when all changes are combined:

1. **Triggers**: After all per-issue work completes (with at least one success)
2. **Worktree validation**: Runs tests in isolated worktree at HEAD commit
3. **Fixer agent**: On failure, spawns a dedicated fixer agent with the failure output
4. **Retry loop**: Continues up to `max_gate_retries` attempts

**Validation scopes:**

| Scope | When | What runs |
|-------|------|-----------|
| **Per-issue** | After each issue completes | pytest, ruff, ty |
| **Run-level** | After all issues complete | pytest, ruff, ty, + E2E fixture test |

**Test flags (apply to both scopes):**

| Flag | Default | Effect |
|------|---------|--------|
| `integration-tests` | included | Pytest tests marked `@pytest.mark.integration` are skipped when this flag is in the disable list |
| `e2e` | enabled (run-level only) | E2E fixture test runs only during run-level validation |

**Disable flags:**
- `--disable-validations=run-level-validate`: Skip run-level validation entirely
- `--disable-validations=integration-tests`: Exclude integration-marked pytest tests
- `--disable-validations=e2e`: Disable E2E fixture test (only affects run-level)
- `--disable-validations=followup-on-run-validate-fail`: Don't mark issues on run-level validation failure

## Lock Enforcement

File locks are enforced at two levels:

1. **Advisory locks**: Agents acquire locks before editing files via `lock-try.sh`
2. **PreToolUse hook**: A hook blocks file-write tool calls (`Write`, `Edit`, `NotebookEdit`, `mcp__morphllm__edit_file`) unless the agent holds the lock for that file

Lock keys are canonicalized so equivalent paths (absolute/relative, with `./..` segments) produce identical locks. When `REPO_NAMESPACE` is set, paths become repo-relative.

## Git Safety

Dangerous git operations that can cause conflicts between concurrent agents are blocked:

| Blocked Operation | Reason | Safe Alternative |
|-------------------|--------|------------------|
| `git stash` | Hides changes other agents cannot see | Commit changes: `git commit -m 'WIP: ...'` |
| `git reset --hard` | Discards uncommitted changes silently | Use `git checkout <file>` for specific files |
| `git rebase` | Rewrites history, requires human input | Use `git merge` instead |
| `git checkout -f` | Discards local changes | Commit changes first |
| `git restore` | Discards uncommitted changes without confirmation | Commit changes first, or use `git diff` to review before discarding |
| `git clean -f` | Removes untracked files | Use `rm` for specific files |
| `git merge --abort` | May discard other agents' work | Resolve conflicts instead |

Each blocked operation includes a safe alternative in the error message.

## Parallel Validation

Agents run validation commands in parallel using **isolated cache directories** to prevent conflicts:

```bash
pytest -o cache_dir=/tmp/pytest-$AGENT_ID        # Isolated pytest cache
ruff check . --cache-dir=/tmp/ruff-$AGENT_ID     # Isolated ruff cache
ruff format .                                     # No cache conflicts
ty check                                          # Type check (read-only)
uv sync                                           # Has internal locking
```

This approach avoids deadlocks that occurred when agents held file locks while waiting for a global test mutex. File locks prevent concurrent edits; isolated caches prevent validation conflicts.

**Legacy `test-mutex.sh`**: A wrapper script exists for serializing commands if needed, but agents no longer require it for standard validation.

## Validation System

The validation module (`src/validation/`) provides structured validation with policy-based configuration.

### Repo Type Detection

mala automatically detects the repository type and adjusts validation accordingly:

| Repo Type | Detection | Validation |
|-----------|-----------|------------|
| **Python** | Has `pyproject.toml`, `uv.lock`, or `requirements.txt` | Full Python toolchain (pytest, ruff, ty) |
| **Generic** | No Python project markers | Minimal validation (no Python-specific tools) |

This allows mala to process issues in non-Python repositories without failing on missing Python tooling.

### ValidationSpec

Defines what validations run per scope (per-issue vs run-level):

```python
from src.validation import build_validation_spec, ValidationScope

spec = build_validation_spec(
    scope=ValidationScope.PER_ISSUE,
    disable_validations={"integration-tests"},  # Optional disable flags
    coverage_threshold=85.0,
)
```

**Disable flags:**
- `post-validate`: Skip test commands entirely
- `integration-tests`: Exclude integration tests from pytest
- `coverage`: Disable coverage checking
- `e2e`: Disable E2E fixture repo test
- `review`: Disable external review (Cerberus review-gate)

### Code vs Docs Classification

Changes are classified to determine validation requirements:

| Category | Paths/Files | Validation |
|----------|-------------|------------|
| **Code** | `src/**`, `tests/**`, `commands/**`, `.py`, `.sh`, `.toml` | Full suite (tests + coverage) |
| **Docs** | `.md`, `.rst`, `.txt` outside code paths | Full suite (tests still run) |

Note: For docs-only issues that need no changes, agents can use `ISSUE_NO_CHANGE` to skip validation entirely.

### Worktree Validation

Clean-room validation runs in isolated git worktrees:

```
/tmp/mala-worktrees/{run_id}/{issue_id}/{attempt}/
```

- Commits are tested in isolation from the working tree
- Failed validations can keep worktrees for debugging (`--keep-worktrees`)
- Stale worktrees from crashed runs are auto-cleaned

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
- **Lock ownership tracking**: Each run only releases its own locks on shutdown (supports concurrent mala instances)
- **Orchestrator claims issues** before spawning (agents don't claim)
- **Quality gate** verifies commits and validation before accepting work
- **Process group management**: Subprocesses (bd/git) run in new sessions for clean termination on timeout
- **JSONL logs** in `~/.config/mala/logs/` for debugging
- **asyncio.wait** for efficient parallel task management

## Configuration

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--max-agents`, `-n` | unlimited | Maximum concurrent agents |
| `--timeout`, `-t` | none | Timeout per agent in minutes |
| `--max-issues`, `-i` | unlimited | Maximum total issues to process |
| `--epic`, `-e` | - | Only process tasks that are children of this epic |
| `--only`, `-o` | - | Comma-separated list of issue IDs to process exclusively |
| `--max-gate-retries` | 3 | Maximum quality gate retry attempts per issue |
| `--max-review-retries` | 3 | Maximum external review retry attempts per issue |
| `--disable-validations` | - | Comma-separated list (see below) |
| `--coverage-threshold` | 85.0 | Minimum coverage percentage (0-100) |
| `--wip` | false | Include and prioritize in_progress issues (without this flag, only open issues are fetched) |
| `--verbose/-q` | verbose | Verbose shows full tool args; quiet (`-q`) shows single line per tool call |
| `--no-morph` | false | Disable MorphLLM routing; use built-in tools directly |

**Disable validation flags:**

| Flag | Description |
|------|-------------|
| `post-validate` | Skip all per-issue validation (tests, lint, typecheck) |
| `run-level-validate` | Skip run-level validation |
| `integration-tests` | Exclude integration tests from pytest |
| `coverage` | Disable coverage checking |
| `e2e` | Disable E2E fixture repo test |
| `review` | Disable external review (Cerberus review-gate) |
| `followup-on-run-validate-fail` | Don't mark issues with `needs-followup` on run-level validation failure |

### Global Configuration

mala uses a global config directory at `~/.config/mala/`:

```
~/.config/mala/
├── .env          # API keys (ANTHROPIC_API_KEY, MORPH_API_KEY, BRAINTRUST_API_KEY)
└── logs/         # JSONL session logs
```

Environment variables are loaded from `~/.config/mala/.env` (global config).

**Directory overrides** (set in `.env` or environment):

| Variable | Default | Description |
|----------|---------|-------------|
| `MALA_RUNS_DIR` | `~/.config/mala/runs` | Directory for run metadata |
| `MALA_LOCK_DIR` | `~/.config/mala/locks` | Directory for filesystem locks |

Note: The repo's `.env` file is for testing only and is not loaded by the program.

### Braintrust Tracing

To enable Braintrust tracing for agent sessions, add your API key to the global config:

```bash
echo "BRAINTRUST_API_KEY=your-key" >> ~/.config/mala/.env
```

### MorphLLM MCP (Optional)

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

mala supports three output modes controlled by `--verbose/-q`:

| Mode | Flag | Description |
|------|------|-------------|
| **Verbose** | `--verbose` (default) | Full tool arguments in key=value format |
| **Quiet** | `-q` | Single line per tool call (tool name + abbreviated args) |

```bash
mala run /path/to/repo          # Verbose output (default)
mala run -q /path/to/repo       # Quiet mode - single line per tool
```

## Terminal Output

Agent output uses color-coded prefixes to distinguish concurrent agents:
- Each agent gets a unique bright color (cyan, yellow, magenta, green, blue, white)
- Log lines are prefixed with `[issue-id]` in the agent's color
- Tool usage, text output, and completion status are all color-coded

## Development

### Type Checking

mala uses strict type checking with both `ty` and `ruff`:

```bash
uvx ty check         # Type check with strict rules
uvx ruff check .     # Lint with type annotation rules
uvx ruff format .    # Format code
```

All ty rules are set to `error` level in `pyproject.toml` for maximum strictness.

### Test Coverage

Tests require 85% minimum coverage (enforced at quality gate, not during default test runs):

```bash
uv run pytest                              # Unit + integration tests (default, excludes e2e, no coverage)
uv run pytest -m unit                      # Unit tests only
uv run pytest -m integration -n auto       # Integration tests in parallel
uv run pytest -m e2e                       # End-to-end tests (requires CLI auth)
uv run pytest -m "unit or integration or e2e"  # All tests
uv run pytest --reruns 2                   # Auto-retry flaky tests (2 retries)
uv run pytest -m integration -n auto --reruns 2   # Parallel + auto-retry
uv run pytest --cov=src --cov-fail-under=85       # Manual coverage check
```

- **Parallel execution**: Use `-n auto` for parallel test runs (pytest-xdist)
- **Flaky test retries**: Use `--reruns N` to auto-retry failed tests (pytest-rerunfailures)
- **Unit/Integration/E2E**: Use markers `unit`, `integration`, `e2e` to select categories
- **Coverage**: Not run by default; quality gate adds coverage flags automatically
