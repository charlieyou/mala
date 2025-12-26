# Coordination Plan (codex) — Final Synthesis

## Executive Summary
Multi-agent runs fail because shared state makes changes indistinguishable, locking is advisory, prompts encourage shortcuts, and there is no validation gate before accepting work. The best solution is a two-track approach:
- **Track A**: harden the current shared-worktree system (short, timeboxed).
- **Track B**: migrate to per-agent worktrees with a merge queue and strict validation (end state).

## Root Causes (confirmed)
- Shared worktree means `git status` and tests see combined uncommitted changes.
- File locks prevent concurrent writes but not semantic conflicts across files.
- Lock keys are raw paths without repo namespace; relative vs absolute can bypass locks.
- Tool calls can edit files without verifying lock ownership.
- Prompt guidance explicitly allows skipping full validation and blame-shifting.
- No external validation gate; agents self-certify and commit directly.
- Failures reset issues without preserving context.

## Non-negotiables
- **Isolation**: an agent's tests and `git status` must reflect only its own changes.
- **Validation gate**: tests + lint + type checks must pass before merge.
- **Ownership**: one agent owns an issue until resolved or explicitly reassigned.
- **Evidence**: failures must record context (commands + errors).

## Track A: Shared-Worktree Hardening (1-2 weeks, timeboxed)
A stop-gap to reduce damage while Track B is built.

### A1. Prompt Cleanup (Immediate)
- Remove any guidance that allows skipping tests or blaming pre-existing failures.
- Replace with a single, clear testing policy (see "Prompt Changes" below).

### A2. Enforced Lock Ownership (High)
- Add a PreToolUse hook that blocks file writes unless the agent holds the lock.

### A3. Canonical Lock Keys (High)
- Set `REPO_NAMESPACE` in the prompt to repo root.
- Normalize file paths (repo-relative + resolve symlinks) before hashing.
- Update slow SDK lock tests and helpers to match hashed naming.

### A4. Quality Gate (High)
- After an agent completes, verify:
  - A commit exists with `bd-<issue_id>` in the message.
  - Validation checks ran (parse JSONL logs or re-run in orchestrator).
- On failure: mark `needs-followup` and attach failure context.

### A5. Global Test Mutex (Medium)
- Serialize repo-wide commands: `uv sync`, `pytest`, `ruff`, `ty`.

### A6. Failure Handoff (Medium)
- Write `.mala/handoff/<issue_id>.md` with the last commands and error summary.

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

## Rollout Order (Recommended)
1) A1 Prompt cleanup
2) A2 Lock enforcement + A3 Canonical keys
3) A4 Quality gate
4) A5 Test mutex + A6 Handoff files
5) B1-B6 Worktrees + merge queue
6) Retire file locking after worktrees stabilize
