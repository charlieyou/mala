# Coordination Plan (codex) — Final Synthesis

## Executive Summary
Multi-agent runs fail because shared state makes changes indistinguishable, locking is advisory, prompts encourage shortcuts, and there is no validation gate before accepting work. The best solution is a two-track approach:
- **Track A**: harden the current shared-worktree system (short, timeboxed). **STATUS: COMPLETE**
- **Track B**: migrate to per-agent worktrees with a merge queue and strict validation (end state). **STATUS: NOT STARTED**

## Root Causes (confirmed)
- Shared worktree means `git status` and tests see combined uncommitted changes. *(Track B will address)*
- File locks prevent concurrent writes but not semantic conflicts across files. *(Track B will address)*
- ~~Lock keys are raw paths without repo namespace; relative vs absolute can bypass locks.~~ ✅ *Fixed: A3*
- ~~Tool calls can edit files without verifying lock ownership.~~ ✅ *Fixed: A2*
- ~~Prompt guidance explicitly allows skipping full validation and blame-shifting.~~ ✅ *Fixed: A1*
- ~~No external validation gate; agents self-certify and commit directly.~~ ✅ *Fixed: A4*
- ~~Failures reset issues without preserving context.~~ ✅ *Fixed: A6*

## Non-negotiables
- **Isolation**: an agent's tests and `git status` must reflect only its own changes.
- **Validation gate**: tests + lint + type checks must pass before merge.
- **Ownership**: one agent owns an issue until resolved or explicitly reassigned.
- **Evidence**: failures must record context (commands + errors).

## Track A: Shared-Worktree Hardening (COMPLETE)
A stop-gap to reduce damage while Track B is built.

### A1. Prompt Cleanup ✅
- Removed "Speed/Scope Guardrails" that allowed skipping tests.
- Removed "run only touched files" guidance.
- Added full validation suite requirement with test mutex before commits.
- Added `REPO_NAMESPACE` to environment variables.

### A2. Enforced Lock Ownership ✅
- Added `PreToolUse` hook that blocks file writes (`Write`, `NotebookEdit`, `mcp__morphllm__edit_file`) unless the agent holds the lock.
- Implemented in `src/hooks.py` with `make_lock_enforcement_hook()`.

### A3. Canonical Lock Keys ✅
- Added `_canonicalize_path()` helper for path normalization.
- Lock keys now resolve symlinks, normalize `./..` segments, and use repo-relative paths when `REPO_NAMESPACE` is set.
- Updated SDK/integration tests for hash-based lock key naming.

### A4. Quality Gate ✅
- Added `QualityGate` class in `src/quality_gate.py`.
- Verifies commit exists with `bd-<issue_id>` in message.
- Parses JSONL logs for validation commands (pytest, ruff check/format, ty check, uv sync).
- On failure: marks issue with `needs-followup` label and records failure context.

### A5. Global Test Mutex ✅
- Added `test-mutex.sh` script that acquires `__test_mutex__` before repo-wide commands.
- Mutex released on exit, even on failure or signals.
- Serializes `pytest`, `ruff`, `ty`, `uv sync`.

### A6. Failure Handoff ✅
- Writes `.mala/handoff/<issue_id>.md` on agent failure.
- Contains error summary, last tool error, and last 10 Bash commands from session log.

## Track B: Worktrees + Merge Queue (2-4 weeks, end state)
Full isolation removes the core coordination failures.

### B1. Worktree per Agent
`git worktree add /tmp/mala-worktrees/<issue>-<session> -b agent/<issue>/<session>`

### B2. Agent Runs in Isolation
- No file locking needed for normal edits.
- Optional lock only for shared external resources.

### B3. Merge Queue (Sequential)
- Process completed branches in order:
  - Rebase onto main
  - Run full validation
  - Merge if pass

### B4. Validation Gate (Hard)
Run full suite in worktree:
`uv run pytest`, `uvx ty check`, `uvx ruff check .`, `uvx ruff format . --check`
If any fail, return to the agent for fixes (keep worktree).

### B5. Rejection Loop
- Agent fixes in its worktree and re-enters the queue.
- After N failures, mark issue blocked.

### B6. Cleanup
- On success: remove worktree + branch.
- On permanent failure: archive handoff and clean worktree.

## Prompt Changes

### Shared-Worktree Mode
- Remove "Speed/Scope Guardrails" and any language that permits skipping tests.
- Add explicit validation requirements (full suite) and the test mutex rule.

### Worktree Mode
- Remove the file-locking protocol section entirely.
- Add an isolation acknowledgement:
  - "You are in an isolated worktree; any failure is caused by your changes."
- Require full validation before completion.

## Optional Safety Valve
- **Sequential mode**: process one issue at a time for high-risk runs.

## Success Criteria
- 0 lock collisions (Track A).
- 95%+ of issues have validated checks recorded (Track B).
- 0 merges without passing the validation gate.
- Median time-to-repair for failures decreases week-over-week.

## Rollout Order
1) ~~A1 Prompt cleanup~~ ✅
2) ~~A2 Lock enforcement + A3 Canonical keys~~ ✅
3) ~~A4 Quality gate~~ ✅
4) ~~A5 Test mutex + A6 Handoff files~~ ✅
5) B1-B6 Worktrees + merge queue (next)
6) Retire file locking after worktrees stabilize
