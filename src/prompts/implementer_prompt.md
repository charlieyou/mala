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

### 4. Quality Checks (Full Validation Required)

**Before committing, run the full validation suite.** Use isolated cache directories to allow parallel validation:

```bash
# Sync dependencies only if needed (skip if pyproject.toml/uv.lock unchanged and .venv exists)
if [ ! -d .venv ] || git diff --name-only HEAD~10 2>/dev/null | grep -qE '^(pyproject\.toml|uv\.lock)$'; then
  uv sync --all-extras
fi

uv run pytest --cache-dir=/tmp/pytest-$AGENT_ID         # Isolated pytest cache
uvx ruff check . --cache-dir=/tmp/ruff-$AGENT_ID        # Isolated ruff cache
uvx ruff format .                                       # Format (no cache concerns)
uvx ty check                                            # Type check
```

**Note:** No test mutex needed. Isolated caches prevent conflicts between parallel agents.

**All checks must pass.** If any fail:
- Fix the issues in your code
- Re-run the full suite
- Do not commit until all checks pass

**CRITICAL - No Gaming Validation:**
- Run `uvx ty check` with NO path arguments (checks entire codebase)
- Do NOT pipe to `head`, `tail`, or truncate output in any way
- Do NOT assume errors are "pre-existing" - verify with `git blame` first
- If you modified a test file, type errors in that file are YOUR responsibility
- All checks must pass with ZERO errors before committing
- If you see errors, FIX THEM - do not claim they are someone else's problem

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

**Requirements:**
1. Working tree must be clean (`git status` shows no changes)
2. Provide a clear rationale explaining why no changes are needed

**Example output:**
```
ISSUE_NO_CHANGE: The validation logic already exists in src/validator.py:45-52
```
```
ISSUE_OBSOLETE: The legacy API endpoint was removed in commit abc123
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
