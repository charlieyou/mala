# Persistent Run Mode (`--watch`)

**Tier:** S
**Owner:** [TBD]
**Target ship:** [TBD]
**Links:** [TBD]

## 1. Outcome & Scope

**Problem / context**
When the orchestrator completes all ready tasks and finds no more ready issues, it immediately exits. This blocks workflows where external processes (humans, CI, tools) add issues to the queue while mala is running, or where multi-stage dependencies mean later issues become ready after earlier ones complete. Users are forced to restart the tool repeatedly to check for new work.

**Change summary**
Add a `--watch` flag that keeps the orchestrator running after all ready tasks complete, polling for new issues every 60 seconds instead of exiting. Also add `--validate-every N` (default 10) to trigger periodic run-level validation during long-running sessions.

**Scope boundary**
Only affects run loop exit behavior in the main issue execution loop. Does not change:
- Issue readiness detection logic
- Agent spawning or finalization
- Gate/review retry mechanisms
- Existing `--max-issues` behavior (watch mode respects this limit)

## 2. User Experience & Flows

**UX impact**
- User-visible? yes
- When `mala run --watch` is used and no ready tasks remain, mala logs "Idle: no ready issues. Polling in 60s..." and continues running instead of exiting
- Idle logging: logs once on transition to idle, then every 5 minutes while idle (not every poll) to reduce noise
- Log message format: "Idle: no ready issues" when queue is empty; "Idle: N issues exist but none ready" when issues are blocked by dependencies

## 3. Requirements + Verification

**Acceptance criteria**

*Core watch behavior:*
- When `--watch` is NOT set (default), orchestrator exits immediately when no ready tasks remain (current behavior preserved)
- When `--watch` IS set and no ready tasks remain, orchestrator waits 60 seconds before re-querying for ready issues
- When new issues become ready during a wait period, they are picked up on the next poll and processed normally
- When `--watch` IS set with `--max-issues N`, orchestrator stops watching and exits after N issues complete
- **Idle state definition**: The orchestrator enters idle/sleep state only when BOTH conditions are true: (1) no ready issues in queue, AND (2) no active agents running

*Completion counting:*
- An issue counts as "complete" when it reaches a terminal state (success OR failure). Retries within an issue don't increment the counter; only final disposition does
- Infrastructure failures (agent spawn failure, orchestrator errors) do NOT count as completions—only issues that started and reached a terminal state count
- `--max-issues` and `--validate-every` both use this same completion counter
- Counter is cumulative from session start (not reset after validation)

*Periodic validation (`--validate-every N`):*
- Triggers run-level validation after every N issues complete (default N=10)
- Validation is **blocking**: pause new agent spawning, wait for all active agents to finish, run validation, then resume
- If validation fails (after fixer retries exhausted), abort the watch session with exit code 1
- Only applies in watch mode; in normal mode, validation runs once at end as today
- If session ends before N issues complete, validation runs on exit (same as normal mode)
- **Parallel completion handling**: Validation triggers once when `completed_count >= next_threshold`. If counter jumps from 9→12, validation runs once (not 3 times). After validation, `next_threshold` advances by N (e.g., 10→20)

*Argument validation:*
- `--validate-every N`: N must be a positive integer (≥1). N=0 is rejected with error: "Error: --validate-every must be at least 1"
- `--max-issues N`: N must be a positive integer (≥1). N=0 is rejected with error: "Error: --max-issues must be at least 1"
- Invalid arguments exit immediately with code 2 (before any processing)

*Signal handling (Ctrl+C):*
- During active processing: stop spawning new agents, wait for active agents to finish (no timeout), then run final validation and exit
- During idle/sleep: break sleep immediately, run final validation if any issues completed this session, then exit
- During blocking validation: SIGINT is deferred until validation completes, then exit
- "Finish active tasks" means let current agent iterations complete but don't start new issues

*Polling error handling:*
- If `bd ready` query fails (CLI error, parse error, IO error): log error, wait 60s, retry
- After 3 consecutive poll failures: run final validation if any issues completed, then abort with exit code 3

*Final validation on exit:*
- Run final validation if: (1) at least one issue completed this session, AND (2) completion count has increased since last validation
- Skip final validation if: validation just ran at a `--validate-every` threshold (avoids redundant runs)
- Skip final validation if: aborting due to validation failure (already failed)

*Exit codes:*
- 0: Success (all issues passed, validation passed)
- 1: Validation failure (periodic or final validation failed after retries)
- 2: Invalid arguments
- 3: Poll failure (3 consecutive bd CLI failures)
- 130: User interrupt (SIGINT/Ctrl+C) after graceful shutdown

## 4. Instrumentation & Release Checks

**Validation after release**
- How to confirm: Run `mala run --watch`, add issues externally via `bd create` while running, verify they get picked up and processed within ~60 seconds
- Known risks: Low - watch mode is opt-in. Poll interval of 60s mitigates excessive bd CLI load

**Testing approach**
- Inject clock/sleep function to test polling logic without real delays
- Unit tests assert "wait called with 60s" and "re-query invoked after wait"
- Integration test for Ctrl+C handling with short/mocked sleep

**Open questions**
- None remaining
