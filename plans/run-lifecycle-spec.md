# 2026-01-10 — Run Lifecycle Spec (Proposed)

Date: 2026-01-10
Owner: user request (run lifecycle intent)
Status: Draft

## Intent
- Session end validation should run **after each issue completes** and **before code review**.
- Run-end validation and global validation should be **merged into a single stage**.

## Current Behavior (as observed in code)
- `session_end` trigger runs once after all issues complete (orchestrator finalization).
- Per-issue review runs after the per-issue gate.
- `run_end` trigger is conditional on `fire_on` and runs before global validation.
- Global validation runs after all issues complete and is independent of `run_end`.

## Proposed Lifecycle (Target)

Per-issue (each issue completes):
1) Agent work + commit
2) **Session-end trigger validation** (commands + optional code_review if enabled)
3) Gate + Review
   - Gate uses validation evidence and command results for the issue.
   - Review runs only after session-end validation completes.

Run-end (once after all issues complete):
4) **Unified run validation stage**
   - Merge `run_end` trigger + global validation into a single stage
   - Runs one command pipeline using the same command pool
   - Optional cumulative code_review runs after commands (if enabled)

## Configuration Semantics
- `validation_triggers.session_end` is now per-issue and executes before review.
- `validation_triggers.run_end` becomes the sole definition of the final run-level validation.
- `global_validation_commands` are merged into the run-end command pool (no separate stage).
- If `run_end.fire_on` blocks, the unified stage is skipped (no global validation fallback).

## Logging Expectations
- Per-issue logs should show:
  - `[trigger] ◦ [session_end] queued: issue_id=<id>`
  - `[trigger] → [session_end] validation_started: ...`
  - `[review]` should only start after session_end completes.
- End-of-run logs should show:
  - `[trigger] ◦ [run_end] queued: success_count=...`
  - `[trigger] → [run_end] validation_started: ...`
  - No separate `[run] → GATE` stage.

## Migration/Compatibility Notes
- Existing configs that rely on global validation should continue to work by
  mapping `global_validation_commands` into the unified run_end stage.
- If `run_end` is not configured, either:
  - (Option A) skip the unified stage entirely, or
  - (Option B) create a default run_end stage from preset commands.
  Decision needed.

## Open Questions
- Should per-issue session_end validation replace the current gate commands, or
  run in addition to them?
- Should `run_end` cumulative code_review be allowed to run without commands?
- What should happen if no issues succeed (success_count == 0)?
