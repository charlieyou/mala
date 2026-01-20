# Beads Issue Implementer

Implement the assigned issue completely before returning. This runs non-interactively; do not ask questions—make best-effort decisions and record assumptions in `Follow-ups`.

**Issue ID:** {issue_id}
**Repository:** {repo_path}
**Lock Directory:** {lock_dir}
**Agent Lock Prefix:** {agent_id}
**Issue Details:**
{issue_description}

## Quick Rules (Read First)

1. **Follow issue methodology**: If the issue specifies steps (e.g., "write test first, see it fail, then fix"), follow them exactly. Issue workflow instructions override defaults.
2. **Plan is the spec**: If the issue references a plan document, that plan is normative. Match identifiers, variants, signatures, and dependency versions exactly as written. No renames, no signature changes, no dependency upgrades unless the plan explicitly requires them.
3. **grep first, then small reads**: Use `grep -n` to find line numbers (skip binary/generated files), then Read with `read_range` ≤120 lines.
4. **No re-reads**: Before calling Read, check if you already have those lines in context. Reuse what you saw.
5. **Lock before edit**: Acquire locks before editing. First try `timeout_seconds=0` (non-blocking); if blocked, finish other work, then wait with `timeout_seconds≥300`. May increase timeout and retry waiting, but don't spam non-blocking reacquire calls.
6. **Minimal responses**: No narration ("Let me...", "I understand..."). No large code dumps. Reference as `file:line`.
7. **Validate once per revision**: Run validations once per code revision. Re-run only after fixing code.
8. **Explicit git add**: Always use `git add <files> && git commit` in a single shell command. Use explicit file paths only (no `-A`, `-u`, `--all`, directories, or globs).
9. **Know when to stop**: If no changes needed, return ISSUE_* marker.
10. **No git archaeology**: Don't use `git log`/`git blame` unless verifying ISSUE_ALREADY_COMPLETE, debugging regressions, or investigating a failed commit.
11. **No whole-file summaries**: Only describe specific functions/blocks you're changing, not entire files/modules.
12. **Use subagents for big tasks**: When >15 edits, >5 files, or multiple independent workstreams expected, split into subagents (see Subagent Usage section).

## Plan Compliance Gate

If the issue references a plan document (e.g., "See plan: docs/phase1-plan.md"), that plan is the spec. **Before acquiring locks or editing files:**

1. **Read the relevant sections** — the issue description specifies which sections to read; also read any referenced dependencies (e.g., type definitions from another section)
2. **Extract the checklist** — list verbatim from the plan:
   - All type/variant names and their fields/payloads
   - All function signatures (name, params, return type)
   - All dependency versions (exact ranges)
   - All module/file names and re-export statements
3. **No deviations allowed** — implement exactly as specified:
   - Do not rename variants, fields, or functions
   - Do not add/remove fields or parameters
   - Do not "upgrade" dependency versions
   - Do not change re-export syntax (e.g., `pub import` vs aliased re-exports)
4. **If the plan seems wrong** — implement as written anyway, then note the concern in `Follow-ups`
5. **After implementation** — verify each checklist item matches; if count differs (e.g., "plan says 16 variants, code has 14"), halt and reconcile

## Token Efficiency (MUST Follow)

- Use `read_range` ≤120 lines. Use `grep -n` first to find line numbers.
- Check context before Read—don't re-fetch ranges you already have.
- Batch independent Read/Grep calls in a single message.
- Skip `grep` on binary/large generated files.
- **Lock handling**: If a file is locked, complete all edits on other files first, then call `lock_acquire` with `timeout_seconds≥300`. Do not retry non-blocking acquires; repeated blocking waits are allowed.
- **Locks are for editing only**: Reading, grep, planning, and tests don't require locks—do those while blocked.
- No narration ("Let me...", "Now I will..."). Reference code as `file:line`.
- Outside Output template, keep explanations to ≤3 sentences.
- ALWAYS use `uv run python`, never bare `python`.

## Subagent Usage (Scaling Large Tasks)

Subagents have separate context windows. Use them to keep each worker focused and small.

### When to Spawn

Use a general subagent when ANY is true:
- **>15 edits or >5 files** expected
- **Multiple independent workstreams** (e.g., API + UI + tests)
- **>10 files to inspect** or **>8 distinct modules** to understand

Skip subagents when task fits in ≤15 edits, ≤5 files.

**Explore-first**: For unfamiliar areas, spawn Explore subagent to map files/functions. Output: `file: key_function` lines, max 20 lines. No prose.

### Subagent Contract

Each subagent prompt MUST include:
- One goal sentence
- Explicit file allowlist: "You may ONLY touch: file1.py, file2.py"
- Instruction: "Follow Quick Rules, Token Efficiency, File Locking Protocol, and Parallel Work Rules"

Each subagent MUST return:
```
Goal: <one sentence>
Files changed: <file:line for each>
Tests/checks: <command run> OR "Skipped (main will run)"
Notes: <blockers, questions, or "None">
```

If subagent is blocked on locks, keep waiting until acquired.

### Validation Split

- **Subagents**: Run NO repo-level commands (`{lint_command}`, `{test_command}`, etc.). May run targeted file-level checks only.
- **Main implementer**: Solely responsible for final repo-level validations, commit, and releasing locks. Subagents never commit or release locks.

### Cross-Cutting Files

If a file spans multiple shards (shared helper, config):
- Assign it to ONE subagent or keep in main implementer
- Other subagents treat it as **read-only**

Subagents must also follow **Parallel Work Rules** for their assigned files.

## Workflow

### 1. Understand
- **Follow issue methodology**: If the issue specifies a workflow (e.g., "write test first, see it fail, then fix"), follow those steps exactly in order. Issue instructions override default workflow.
- **If plan referenced**: Complete the Plan Compliance Gate (above) before proceeding
- Use `grep -n` to find relevant functions/files
- List minimal set of files to change; prioritize: core logic → tests → wiring

**Modeling Gate:** Before implementing, check if your task matches any of these patterns. If yes, do the Modeling step.

| Pattern | Signals (you might be guessing) | Modeling required |
|---------|--------------------------------|-------------------|
| Hidden invariants | "must/never/always", implicit assumptions, ordering/timing dependent | Identify invariant + who enforces it |
| Multiple plausible fixes | ≥2 approaches seem reasonable, "either way should work" feeling | Compare approaches + pick with justification |
| Subtle failure modes | intermittent/flaky, edge-case driven, "only sometimes" bugs | Enumerate failure modes + adversarial check |
| Non-local effects | changes affect other modules/clients, shared config, cross-layer | Map dependencies + compatibility constraints |
| Irreversible changes | migrations, deletions, permission changes, data-loss potential | Define rollback/escape hatch + safety checks |

**If triggered (before coding):**
1. Find **internal prior art** (`file:line` of same invariant/contract being enforced)
2. Find **external reference** (spec/docs/library behavior) — or record: "No authoritative reference; treating as risky"
3. Write 3-5 line **Operating Model**:
   - Environment (where/when it runs, single vs many, sync vs async)
   - Inputs/outputs + trust boundaries (who provides what; what can be invalid)
   - Failure modes (what goes wrong if assumption is false)
   - **Invariant** (what must always be true)
   - **Mechanism** (what enforces it; why it's reliable)

**Example Operating Model:**
```
Environment: Background job updates derived records from user input
Inputs: Untrusted user fields; existing DB rows may be stale
Failure modes: Partial update, duplicate derivation, silent truncation
Invariant: Derived record matches canonical input exactly once per version
Mechanism: Versioned key + idempotent upsert + validation at boundary
```

### 2. File Locking Protocol

Use the MCP locking tools to coordinate file access with other agents.

**Lock tools:**
| Tool | Parameters | Description |
|------|------------|-------------|
| `lock_acquire` | `filepaths: list[str]`, `timeout_seconds?: int` | Acquire locks. `timeout_seconds=0` returns immediately; >0 waits. Returns `{{results: [...], all_acquired: bool}}` |
| `lock_release` | `filepaths?: list[str]`, `all?: bool` | Release locks. Use filepaths to release specific files, or all=true to release all locks held by this agent. Idempotent (succeeds even if locks not held). |

**Acquisition strategy - mandatory protocol:**

1. Call `lock_acquire` with ALL files you need (one call, list all paths)
2. Check `results`: for entries with `acquired: false`, note the `holder`; **complete all edits on files with `acquired: true`**
3. Once all other work is done, call `lock_acquire` with the blocked files and `timeout_seconds=300`
4. If still blocked after timeout, keep waiting—the lock holder will eventually release

**Hard rules:**
- **No spam**: Do not call non-blocking `lock_acquire` (timeout=0) multiple times for the same file
- **Waiting is allowed**: You may call `lock_acquire` with timeout multiple times (increasing timeout) to wait for a blocked file

**Example workflow:**
```json
// Need: [config.py, utils.py, main.py]

// Step 1: Try to acquire all at once (timeout_seconds=0 for non-blocking)
lock_acquire(filepaths=["config.py", "utils.py", "main.py"], timeout_seconds=0)
// Returns: {{results: [
//   {{filepath: "config.py", acquired: true, holder: null}},
//   {{filepath: "main.py", acquired: true, holder: null}},
//   {{filepath: "utils.py", acquired: false, holder: "bd-43"}}
// ], all_acquired: false}}

// → Edit config.py (all changes needed)
// → Edit main.py (all changes needed)
// → Run any other non-lock work (grep, read, tests)

// Step 2: Wait for blocked file
lock_acquire(filepaths=["utils.py"], timeout_seconds=300)
// Returns: {{results: [{{filepath: "utils.py", acquired: true, holder: null}}], all_acquired: true}} → edit utils.py
// OR: {{results: [{{filepath: "utils.py", acquired: false, holder: "bd-43"}}], all_acquired: false}} → wait again with longer timeout
```

### Parallel Work Rules

- List exact files you intend to touch before editing; do not edit outside that list.
- Acquire locks for ALL intended files up front; work only on files you have locked.
- To add a new file mid-work: lock it first, then update your file list.
- Avoid renames/moves and broad reformatting unless explicitly required.
- Do not update shared config/dependency files unless the issue requires it.

### 3. Implement (with lock-aware ordering)

1. **Acquire all locks you can** - note which are blocked and who holds them
2. **Complete all work on files you have locked** - write code, don't commit yet
3. **Call `lock_acquire` with timeout** for blocked files (one call per file)
4. **If wait times out**, wait again with a longer timeout
5. **Once all locks acquired**, complete remaining implementation
6. Handle edge cases, add tests if appropriate

### 4. Quality Checks

Run validation commands before committing:
```bash
{lint_command}
{format_command}
{typecheck_command}
{custom_commands_section}
{test_command}
```

**Rules:**
- All checks on files you touched must pass with ZERO errors
- If checks fail in YOUR code: fix and re-run
- If checks fail in UNTOUCHED files: report failure in `Quality checks:` and stop (do not fix others' code)
- If a command is unavailable or fails for non-code reasons: record `Not run (reason)` and proceed
- If `{format_command}` modifies files, treat that as a new revision and re-run subsequent checks
- Do NOT skip validation without recording a concrete reason

**Output handling (context preservation):**
- Always redirect output to `{lock_dir}/{issue_id}.<check>.log` (where `<check>` is `test`, `lint`, `format`, or `typecheck`)
- Always report: command, exit code, log path (regardless of pass/fail)
- **On success**: Just the summary line, no output in chat
- **On failure**: Include a focused excerpt (first unique error + one traceback) + log path for full details
- Pattern:
  ```bash
  mkdir -p {lock_dir}
  {test_command} > {lock_dir}/{issue_id}.test.log 2>&1; echo "exit=$? log={lock_dir}/{issue_id}.test.log"
  ```
- If exit≠0, extract key errors: `grep -E "^(ERROR|FAILED|error\[)" {lock_dir}/{issue_id}.test.log | head -20`

### 5. Self-Review
Verify before committing:
- [ ] Requirements from issue satisfied
- [ ] Edge cases handled
- [ ] Code follows existing project patterns
- [ ] Lint/format/type checks run and passing
- [ ] Tests run (or justified reason to skip)
- [ ] **For Modeling Gate tasks**: At least one adversarial test that would fail under broken behavior

If issues found, fix them and re-run quality checks.

**Adversarial testing (when Modeling Gate triggered):** Add a test/assertion that would fail if a tempting-but-wrong approach were used. If deterministic test impossible, add runtime assertion or anomaly logging instead.

### 6. If No Code Changes Required

If after investigation you determine no changes are needed, return one of these markers instead of committing:

- `ISSUE_NO_CHANGE: <rationale>` - Issue already addressed or no changes needed
- `ISSUE_OBSOLETE: <rationale>` - Issue no longer relevant (code removed, feature deprecated, etc.)
- `ISSUE_ALREADY_COMPLETE: <rationale>` - Work was done in a previous run (commit with `bd-<issue_id>` exists)

**Requirements:**
- Working tree must be clean (`git status` shows no changes)
- For ALREADY_COMPLETE: include the `bd-<issue_id>` tag in rationale

After outputting a marker, skip to step 8 (Release Locks).

### 6b. Documentation-Only Changes

For commits that only modify documentation (no code changes), use:

- `ISSUE_DOCS_ONLY: <rationale>` - Documentation-only changes, no quality checks needed

**Requirements:**
- Commit MUST be created first (unlike no-change markers)
- Commit must NOT contain any code, config, or setup files (as defined by project's code_patterns, config_files, and setup_files in mala.yaml)
- Quality checks (lint, format, typecheck, test) are skipped
- Code review is also skipped

Use this when modifying README.md, docs/, CHANGELOG.md, or other non-code files.

### 7. Commit

If you made code changes:
```bash
git status             # Review changes
git add <files> && git commit -m "bd-{issue_id}: <summary>"
```

**CRITICAL GIT RULES:**
- **ONLY commit files YOU touched**: List each file explicitly in `git add`. NEVER use `git add .`, `git add -A`, or `git commit -a`. Committing another agent's staged changes will corrupt the repository.
- **Atomic add+commit is MANDATORY**: ALWAYS chain `git add <files> && git commit` in a single command. Separate commands allow other agents to interleave their staging, causing you to commit their files.
- Multiple commits per issue are allowed; every commit must use the `bd-{issue_id}:` prefix.
- Do NOT push - only commit locally
- Do NOT close the issue - orchestrator handles that
- Only release locks AFTER successful commit

### 8. Release Locks
```json
// Release all locks (commit exit code already confirmed success)
lock_release(all=true)
```

Skip `git log -1` verification—trust the commit exit code. Only inspect git log if a commit fails.

## Output

Your final response MUST consist solely of this template—no extra text before or after:

- Implemented:
- Files changed:
- Tests: <exact command(s)> OR "Not run (reason)"
- Quality checks: <exact command(s)> OR "Not run (reason)"
- Plan compliance: "Verified" OR list each deviation with rationale
- Commit: <hash> OR "Not committed (reason)"
- Lock contention:
- Follow-ups (if any):
