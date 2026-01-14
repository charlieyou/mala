## Quality Gate Failed (Attempt {attempt}/{max_attempts})

**Token Efficiency:** Use `read_range` ≤120 lines. No narration ("Let me..."). No git archaeology. No whole-file summaries. Fix directly.

The quality gate check failed with the following issues:
{failure_reasons}

**Stale commit detection (previous runs)**

If the failure reason mentions "stale commits from previous runs are rejected", the orchestrator is telling you that any existing `bd-{issue_id}` commit was created *before* this run's baseline timestamp. In this case:

1. **If the work is already complete from a prior run** (the commit fully implements the issue and your working tree is clean):
   - Verify a `bd-{issue_id}` commit exists with `git log --oneline --grep="bd-{issue_id}"`
   - Return `ISSUE_ALREADY_COMPLETE: <rationale>` as your final output
   - Your rationale MUST include the commit hash (e.g., "Work completed in commit abc1234 with message bd-{issue_id}: ...")
   - This skips the baseline check and validation evidence

2. **If the work is not complete** (code changes still needed), treat this as a normal failure: make changes, run validations, and create a new commit.

**Required actions (for all other failures):**
1. Fix ALL issues causing validation failures - including pre-existing errors in files you didn't touch
2. Re-run the full validation suite on the ENTIRE codebase:
   - `{test_command}`
   - `{lint_command}`
   - `{format_command}`
   - `{typecheck_command}`
3. Commit your changes with message: `bd-{issue_id}: <description>` (multiple commits allowed; use the prefix on each). Use `git add <files>` with explicit file paths only (no `-A`, `-u`, `--all`, directories, or globs) and commit in the same command.

**CRITICAL:** Do NOT scope checks to only your modified files. The validation runs on the entire codebase. Fix ALL errors you see, even if you didn't introduce them. Do NOT use `git blame` to decide whether to fix an error.

Note: The orchestrator requires NEW validation evidence - re-run all validations even if you ran them before.
