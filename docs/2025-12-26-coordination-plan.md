# Coordination Plan - Reference (2025-12-26)

Status: archived. We are no longer executing against this plan; it is kept for
future reference. The current coordination mechanics, results, and contingency
work are documented below.

## Current coordination (active today)

### System mechanics
- Shared worktree with per-file locks for write access.
- Lock ownership is enforced: `PreToolUse` blocks file writes unless the agent
  holds the lock (Write, NotebookEdit, morph edit).
- Lock keys are canonicalized (resolve symlinks, normalize paths, use repo
  namespace) to prevent path alias bypass.
- Global test mutex serializes repo-wide commands
  (`pytest`, `ruff`, `ty`, `uv sync`) via `__test_mutex__`.
- Quality gate (`src/quality_gate.py`) requires:
  - Commit message contains `bd-<issue_id>`.
  - JSONL logs show full validation commands executed.
  - On failure: mark `needs-followup` and record evidence.
- Failure handoff writes error context + JSONL log path into beads issue notes.
- Prompts require full validation; no "skip tests" guidance remains.
- Observability via Braintrust spans and JSONL logs
  (`~/.config/mala/logs/*.jsonl`); `mala status` surfaces locks/logs.

### Typical run flow
1) Agent acquires locks for all files it plans to modify.
2) Agent edits files; lock enforcement blocks unowned writes.
3) Agent runs full validation (serialized by test mutex).
4) Agent commits with `bd-<issue_id>` in the message.
5) Quality gate validates commit + logs and marks success/followup.
6) On failure: beads notes include evidence; locks released on exit.

## Results (recent)

As of 2025-12-26 (last 24 hours, Braintrust logs):
- Total log entries: 3,899
- Lock tries: 294
- Lock waits: 6 (about 2% of tries, ~0.15% of total logs)
- Lock-holder checks: 57
- Waits were clustered between 18:39-19:09 UTC and were mostly the test mutex or
  `tests/test_orchestrator.py`.

These results indicate low contention and stable coordination under the current
shared-worktree hardening.

## Future work if coordination degrades

Trigger examples (any of these should prompt review):
- Lock-wait rate rises above ~5% of lock tries or >20 waits/day.
- Test mutex waits exceed 10 minutes routinely.
- Increased failures that cite shared-state contamination.
- Higher agent concurrency (e.g., >3 active agents) without stable pass rates.

Mitigation steps (in order):
1) Enable sequential mode for high-risk runs.
2) Tighten the quality gate to require a clean `git status` and re-run the full
   suite before acceptance.
3) Revisit Phase 2 (worktrees + merge queue) if shared-state failures persist.
4) Add stricter per-agent isolation or a dedicated validation worker.

## Historical plan (inactive, retained for reference)

### Root causes (confirmed)
- Shared worktree means `git status` and tests see combined uncommitted changes.
- File locks prevent concurrent writes but not semantic conflicts across files.
- Lock keys had raw path collisions (fixed).
- Tool calls could edit files without lock ownership (fixed).
- Prompt guidance allowed skipping tests (fixed).
- No external validation gate before accepting work (fixed).
- Failures reset issues without preserving context (fixed).

### Track A: Shared-worktree hardening (complete)
- A1: Prompt cleanup and full validation requirement.
- A2: Lock ownership enforcement.
- A3: Canonical lock keys.
- A4: Quality gate (commit + JSONL validation evidence).
- A5: Global test mutex.
- A6: Failure handoff to beads notes.

### Track B: Worktrees + merge queue (not pursued)
- B1: Worktree per agent.
- B2: Agent runs in isolation (no locking).
- B3: Merge queue with rebase + full validation.
- B4: Hard validation gate.
- B5: Rejection loop with bounded retries.
- B6: Cleanup (worktree + branch removal).
