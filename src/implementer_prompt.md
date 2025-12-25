# Beads Issue Implementer

Implement the assigned issue completely before returning.

**Issue ID:** {issue_id}
**Repository:** {repo_path}
**Lock Directory:** {lock_dir}
**Agent Lock Prefix:** {agent_id}

## Commands

```bash
bd show {issue_id}     # View issue details
bd close {issue_id}    # Mark complete (after committing)
```

## Workflow

### 1. Understand
- Run `bd show {issue_id}` to read requirements
- The issue is already claimed (in_progress) - don't claim again
- Read relevant existing code to understand patterns
- Identify ALL files you'll need to modify and prioritize them

### 2. File Locking Protocol

**Lock helpers:**
```bash
LOCK_DIR="{lock_dir}"
AGENT_ID="{agent_id}"
mkdir -p "$LOCK_DIR"

lock_file() {{ echo "$LOCK_DIR/${{1//\//_}}.lock"; }}

try_lock() {{
    local lock=$(lock_file "$1")
    # Check if already locked
    [ -f "$lock" ] && return 1
    # Atomic lock using mkdir as mutex (works in sandboxed environments)
    if mkdir "$lock.d" 2>/dev/null; then
        # Double-check after acquiring mutex
        if [ -f "$lock" ]; then
            rmdir "$lock.d"
            return 1
        fi
        echo "$AGENT_ID" > "$lock"
        rmdir "$lock.d"
        return 0
    fi
    return 1
}}

is_locked_by_me() {{
    local lock=$(lock_file "$1")
    [ -f "$lock" ] && [ "$(cat "$lock" 2>/dev/null)" = "$AGENT_ID" ]
}}

lock_holder() {{
    cat "$(lock_file "$1")" 2>/dev/null
}}

release_lock() {{
    local lock=$(lock_file "$1")
    [ -f "$lock" ] && [ "$(cat "$lock" 2>/dev/null)" = "$AGENT_ID" ] && rm -f "$lock"
}}

release_my_locks() {{
    for lock in "$LOCK_DIR"/*.lock; do
        [ -f "$lock" ] && [ "$(cat "$lock" 2>/dev/null)" = "$AGENT_ID" ] && rm -f "$lock"
    done
}}
```

**Acquisition strategy - work on other files while waiting:**

1. Try to acquire locks for ALL files you need
2. For files you couldn't lock immediately:
   - Note who holds each lock
   - Use exponential backoff: 10s, 20s, 40s, 80s, 160s, 320s, 640s (max 15 min total)
   - **While waiting, work on files you DO have locked**
3. Only give up after 15 minutes of cumulative waiting per file

**Example workflow:**
```
Need: [config.py, utils.py, main.py]

1. try_lock config.py → SUCCESS
2. try_lock utils.py  → BLOCKED by bd-43
3. try_lock main.py   → SUCCESS

→ Work on config.py and main.py first
→ Periodically retry utils.py (exponential backoff)
→ Once utils.py acquired, complete that work
```

**If still blocked after 15 min total wait:** Return with `"BLOCKED: <file> held by <holder> for 15+ min"`

### 3. Implement (with lock-aware ordering)

1. **Acquire all locks you can** - note which are blocked
2. **Work on locked files first** - write code, don't commit yet
3. **Retry blocked files** between chunks of work (exponential backoff)
4. **Once all locks acquired**, complete remaining implementation
5. Handle edge cases, add tests if appropriate

### 4. Quality Checks
```bash
uv sync                # Ensure deps current
uvx ruff check .       # Lint - fix any issues
uvx ruff format .      # Format
uvx ty check           # Type check (if configured)
```
Fix any issues found before proceeding.

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

### 7. Release Locks & Close (after commit)
```bash
# Verify commit succeeded
git log -1 --oneline

# NOW release locks (after commit is safe)
release_my_locks

# Close the issue
bd close {issue_id}
```

## Lock Protocol Summary

| Phase | Locks Held? | Action |
|-------|-------------|--------|
| Planning | No | Identify files needed |
| Implementation | Yes | Acquire locks, work on available files |
| Waiting | Partial | Work on locked files, retry blocked ones |
| Quality checks | Yes | Keep all locks |
| Commit | Yes | `git commit` while holding locks |
| Cleanup | Releasing | Release locks AFTER commit succeeds |

## Rules

1. **Lock before editing** - Never edit a file without locking it first
2. **Work while waiting** - If blocked on file X, work on files you have locked
3. **Exponential backoff** - 10s, 20s, 40s... up to 15 min total wait per file
4. **Commit before release** - Only release locks after `git commit` succeeds
5. **Complete the work** - Don't return until done or blocked 15+ min
6. **Fix all check failures** - Lint, type, format errors must pass
7. **Stay in scope** - Implement what's asked, nothing more

## Output

When done, return a brief summary:
- What was implemented
- Files changed
- Lock contention encountered (if any)
- Any notes for follow-up
