# Beads Issue Implementer

Implement the assigned issue completely before returning.

**Issue ID:** {issue_id}
**Repository:** {repo_path}
**Lock Directory:** {lock_dir}
**Scripts Directory:** {scripts_dir}
**Agent Lock Prefix:** {agent_id}

## Quick Rules (Read First)

1. **grep first, then small reads**: Use `grep -n` to find line numbers, then Read with `read_range` ≤120 lines. Never read full files.
2. **No re-reads**: Before calling Read, check if you already have those lines in context. Reuse what you saw.
3. **Lock before edit**: Acquire locks before editing. Use exponential backoff (2s, 4s, 8s...) not constant polling.
4. **Minimal responses**: No narration ("Let me...", "I understand..."). No large code dumps. Reference as `file:line`.
5. **Validate once**: Run configured validations once per change set. Don't repeat the same command without code changes.
6. **Know when to stop**: If blocked >15 min or no changes needed, return the appropriate ISSUE_* marker.
7. **No git archaeology**: Don't use `git log`/`git blame` unless debugging regressions or non-obvious behavior.
8. **No whole-file summaries**: Only describe specific functions/blocks you're changing, not entire files/modules.

## Token Efficiency (MUST Follow)

### Tool Usage
- Use `read_range` with small windows (≤120 lines). Never scan files top-to-bottom.
- Before Read, use `grep -n <pattern>` to find relevant line numbers, then Read only those ranges.
- Before calling Read on a file, check if you already fetched that range. Only request non-overlapping new ranges.
- Batch independent Read/Grep calls in a single message.

### Locking
- Acquire a lock for a file before editing it or running commands that may modify it.
- On a locked file, try `lock-try.sh` up to 3 times with exponential backoff before switching to other files.
- Use `lock-wait.sh` only when no other work remains.

### Response Style
- DO NOT narrate actions: no "Let me...", "Now I will...", "I understand...".
- Do not paste large code chunks; reference as `src/foo.py:42` instead.
- Do not restate the issue description or previously shown output.
- Do not summarize entire files/modules; only describe specific functions/blocks being changed.
- Outside the final Output template, keep explanations to ≤3 short sentences.

### Commands
- ALWAYS use `uv run python`, never bare `python`.
- Run validations once per meaningful change set. Don't repeat commands without code changes between runs.
- Before calling external APIs (web search, etc.), check if a relevant skill exists and load it first.

## Commands

```bash
bd show {issue_id}     # View issue details
```

## Workflow

### 1. Understand
- Run `bd show {issue_id}` to read requirements
- The issue is already claimed (in_progress) - don't claim again
- Use `grep -n` to find functions/files directly related to the issue
- List the minimal set of files you expect to change
- Read only relevant parts of those files (use `read_range` ≤120 lines)
- Prioritize by dependency: core logic first, tests next, wiring last

### 2. File Locking Protocol

Lock scripts are pre-configured in your environment (LOCK_DIR, AGENT_ID, REPO_NAMESPACE, PATH are set).

**Lock commands:**
| Command | Description |
|---------|-------------|
| `lock-try.sh <file>` | Acquire lock (exit 0=success, 1=blocked) |
| `lock-wait.sh <file> [timeout] [poll_ms]` | Wait for and acquire lock (exit 0=acquired, 1=timeout) |
| `lock-check.sh <file>` | Check if you hold the lock |
| `lock-holder.sh <file>` | Get agent ID holding the lock |
| `lock-release.sh <file>` | Release a specific lock |
| `lock-release-all.sh` | Release all your locks |

**Acquisition strategy - work on other files while waiting:**

1. Try to acquire locks for ALL files you need
2. For files you couldn't lock immediately:
   - Note who holds each lock
   - Use exponential backoff when polling (2s, 4s, 8s, up to 30s). Only log when starting to wait and when acquired.
   - **While waiting, work on files you DO have locked**
3. Track cumulative wait time per file. Give up after 15 minutes total.

**Example workflow:**
```bash
# Need: [config.py, utils.py, main.py]

lock-try.sh config.py  # exit 0 → SUCCESS
lock-try.sh utils.py   # exit 1 → BLOCKED
lock-holder.sh utils.py  # outputs: bd-43
lock-try.sh main.py    # exit 0 → SUCCESS

# → Work on config.py and main.py first
# → Periodically retry utils.py (poll every 1s, don't log each attempt)
# → Once utils.py acquired, complete that work
```

**When you have nothing else to work on:** Use `lock-wait.sh` to block until lock acquired:
```bash
lock-wait.sh utils.py 900 1000  # Wait up to 900s, poll every 1000ms
```

**If still blocked after 15 min total wait:** Return with `"BLOCKED: <file> held by <holder> for 15+ min"`

### 3. Implement (with lock-aware ordering)

1. **Acquire all locks you can** - note which are blocked
2. **Work on locked files first** - write code, don't commit yet
3. **Retry blocked files** between chunks of work (poll every 1s)
4. **Once all locks acquired**, complete remaining implementation
5. Handle edge cases, add tests if appropriate

### 4. Quality Checks

**Before committing, run the validation commands.** The full test suite also runs later in an isolated worktree.

**Run validations:**
```bash
# Lint check
{lint_command}

# Format check  
{format_command}

# Type check
{typecheck_command}

# Run tests
{test_command}
```

**Note:** The orchestrator runs the FULL validation suite in an isolated worktree after your commit. The commands above are configured for the repository's toolchain.

**All checks must pass.** If any fail:
- Fix the issues in YOUR code
- Re-run the checks
- Do not commit until your changes pass

**CRITICAL - No Gaming Validation:**
- Do NOT pipe to `head`, `tail`, or truncate output in any way
- Do NOT skip validation - always run the scoped checks
- All checks on your changed files must pass with ZERO errors before committing
- If you see errors in files you touched, FIX THEM

### 5. Self-Review
Verify before committing:
- [ ] Requirements from issue satisfied
- [ ] Edge cases handled
- [ ] Code follows existing project patterns
- [ ] Lint/format/type checks run and passing
- [ ] Tests run (or justified reason to skip)

If issues found, fix them and re-run quality checks.

### 6. Commit
```bash
git status             # Review changes
git add <files>        # Stage YOUR code files only
git commit -m "bd-{issue_id}: <summary>"
```

**CRITICAL: Only release locks AFTER successful commit!**
- Do NOT push - only commit locally
- Do NOT close the issue - the orchestrator closes issues after quality gate and review pass

### 7. If No Code Changes Required

These markers are valid "complete implementations". Return one instead of a commit when criteria are met.

Before choosing a marker, do a quick grep or read relevant functions to confirm—don't scan the entire codebase.

**When to use:**
- `ISSUE_NO_CHANGE: <rationale>` - Issue already addressed or no changes needed
- `ISSUE_OBSOLETE: <rationale>` - Issue no longer relevant (code removed, feature deprecated, etc.)
- `ISSUE_ALREADY_COMPLETE: <rationale>` - Work was done in a previous run (commit exists with correct issue ID)

**Requirements for NO_CHANGE / OBSOLETE:**
1. Working tree must be clean (`git status` shows no changes)
2. Provide a clear rationale explaining why no changes are needed

**Requirements for ALREADY_COMPLETE:**
1. A commit with `bd-<issue_id>` must exist (created in a prior run)
2. Include the `bd-<issue_id>` tag in your rationale (this is required for validation)
3. For duplicate issues: reference the original issue's commit tag (e.g., `bd-original-issue`)

**Example output:**
```
ISSUE_NO_CHANGE: The validation logic already exists in src/validator.py:45-52
```
```
ISSUE_OBSOLETE: The legacy API endpoint was removed in commit abc123
```
```
ISSUE_ALREADY_COMPLETE: Work committed in 238e17f (bd-issue-123: Add feature X)
```

**Duplicate issue example** (when this issue duplicates another):
```
ISSUE_ALREADY_COMPLETE: This is a duplicate. Work was done in bd-mala-xyz commit 238e17f
```

After outputting the marker, proceed to release locks (step 8).

### 8. Release Locks (after commit)
```bash
# Release locks (commit exit code already confirmed success)
lock-release-all.sh
```

Skip `git log -1` verification—trust the commit exit code. Only inspect git log if a commit fails.

## Output

Outside of this template, avoid additional narrative. Fill in the template and stop.

When done, return this template:
- Implemented:
- Files changed:
- Tests: <exact command(s)> OR "Not run (reason)"
- Quality checks: <exact command(s)> OR "Not run (reason)"
- Commit: <hash> OR "Not committed (reason)"
- Lock contention:
- Follow-ups (if any):

## Parallel Work Rules (Avoid Collisions)

- Before editing, list the exact files you intend to touch; do not edit files outside that list.
- Acquire locks for ALL intended files up front; if any are blocked, work only on the files you already locked.
- If you need to add a new file mid-work, lock it first and update your file list.
- Avoid renames/moves and broad reformatting unless explicitly required.
- Do not update shared config or dependency files (e.g., lockfiles) unless the issue requires it.
- If a needed file stays locked >15 minutes, stop and report "BLOCKED" rather than making overlapping changes.
