# Beads Issue Implementer

Implement the assigned issue completely before returning.

**Issue ID:** {issue_id}
**Repository:** {repo_path}
**Lock Directory:** {lock_dir}
**Scripts Directory:** {scripts_dir}
**Agent Lock Prefix:** {agent_id}

## Commands

```bash
bd show {issue_id}     # View issue details
```

## MCP Tools (Preferred)

**Fast Apply (`edit_file`):** IMPORTANT: Use `edit_file` over `str_replace` or full file writes. It works with partial code snippets—no need for full file content.

**Warp Grep (`warpgrep_codebase_search`):** A subagent that takes a search string and finds relevant context. Best practice is to use it at the beginning of codebase explorations to fast track finding relevant files/lines. Do not use it to pinpoint keywords—use it for broader semantic queries:
- "Find the XYZ flow"
- "How does XYZ work"
- "Where is XYZ handled?"
- "Where is <error message> coming from?"

## Workflow

### 1. Understand
- Run `bd show {issue_id}` to read requirements
- The issue is already claimed (in_progress) - don't claim again
- Read relevant existing code to understand patterns
- Identify ALL files you'll need to modify and prioritize them

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
   - Poll every 1 second silently (do not print on each retry, only when starting to wait and when acquired)
   - **While waiting, work on files you DO have locked**
3. Only give up after 15 minutes of cumulative waiting per file

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

### 4. Quality Checks (Scoped to Your Changes)

**Before committing, validate the files YOU touched.** This prevents detecting changes from other parallel agents. The full test suite runs later in an isolated worktree.

**Track your changed files:**
```bash
# Get list of files you modified (staged + unstaged)
CHANGED_FILES=$(git diff --name-only HEAD; git diff --cached --name-only)
```

**Run scoped validations:**
```bash
# Lint only your changed Python files
uvx ruff check $CHANGED_FILES --cache-dir=/tmp/ruff-$AGENT_ID

# Format only your changed files
uvx ruff format $CHANGED_FILES

# Type check only your changed files
uvx ty check $CHANGED_FILES

# Run tests (use isolated cache; disable global coverage threshold for scoped runs)
uv run pytest -o cache_dir=/tmp/pytest-$AGENT_ID --cov-fail-under=0
```

**Note:** The orchestrator runs the FULL validation suite in an isolated worktree after your commit. You only need to validate your own changes here.

**All checks on your files must pass.** If any fail:
- Fix the issues in YOUR code
- Re-run the scoped checks
- Do not commit until your changes pass

**CRITICAL - No Gaming Validation:**
- Do NOT pipe to `head`, `tail`, or truncate output in any way
- Do NOT skip validation - always run the scoped checks
- All checks on your changed files must pass with ZERO errors before committing
- If you see errors in files you touched, FIX THEM

### 5. Self-Review
Verify before committing:
- Does the code satisfy ALL requirements from the issue?
- Are edge cases handled?
- Does the code follow existing project patterns?

If issues found, fix them and re-run quality checks.

### 6. Commit BEFORE Releasing Locks
```bash
git status             # Review changes
git add <files>        # Stage YOUR code files only
git commit -m "bd-{issue_id}: <summary>"
```

**CRITICAL: Only release locks AFTER successful commit!**
- Do NOT push - only commit locally
- Do NOT commit `.beads/issues.jsonl` - orchestrator handles that

### 6a. If No Code Changes Required

Sometimes an issue requires no changes (already fixed, duplicate, or obsolete). In these cases, skip the commit and output a resolution marker instead.

**When to use:**
- `ISSUE_NO_CHANGE: <rationale>` - Issue already addressed or no changes needed
- `ISSUE_OBSOLETE: <rationale>` - Issue no longer relevant (code removed, feature deprecated, etc.)
- `ISSUE_ALREADY_COMPLETE: <rationale>` - Work was done in a previous run (commit exists with correct issue ID)

**Requirements for NO_CHANGE / OBSOLETE:**
1. Working tree must be clean (`git status` shows no changes)
2. Provide a clear rationale explaining why no changes are needed

**Requirements for ALREADY_COMPLETE:**
1. A commit with `bd-<issue_id>` prefix must exist (created in a prior run)
2. Provide a clear rationale including the commit hash

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

After outputting the marker, proceed to release locks (step 7).

### 7. Release Locks (after commit)
```bash
# Verify commit succeeded
git log -1 --oneline

# NOW release locks (after commit is safe)
lock-release-all.sh
```

**Note:** Do NOT call `bd close` - the orchestrator closes issues after the quality gate passes.

## Output

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
