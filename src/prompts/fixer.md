## Global Validation Failed (Attempt {attempt}/{max_attempts})

**Token Efficiency:** Use `read_range` ≤120 lines. No narration ("Let me..."). No git archaeology. No whole-file summaries. Fix directly.

**Failed command:** `{failed_command}`

The global validation found issues that need to be fixed:

{failure_output}

**Required actions:**
1. Analyze the validation failure output above
2. Fix ALL issues causing the failure - including pre-existing errors in files that weren't touched by any agent
3. Re-run the full validation suite on the ENTIRE codebase:
{validation_commands}
4. Commit your changes with message: `bd-run-validation: <description>`

**Context:**
- This is a global validation that runs after all per-session work is complete
- Your fix should address the root cause, not just suppress the error
- Fix ALL lint/type errors you see, even if they're pre-existing - do NOT use `git blame` to decide whether to fix
- The orchestrator will re-run validation after your fix

Do not release any locks - the orchestrator handles that.
