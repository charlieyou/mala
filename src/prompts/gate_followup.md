## Quality Gate Failed (Attempt {attempt}/{max_attempts})

**Token Efficiency:** Use `read_range` ≤120 lines. No narration ("Let me..."). No git archaeology. No whole-file summaries. Fix directly.

The quality gate check failed with the following issues:
{failure_reasons}

**Required actions:**
1. Fix ALL issues causing validation failures - including pre-existing errors in files you didn't touch
2. Re-run the full validation suite on the ENTIRE codebase:
   - `{test_command}`
   - `{lint_command}`
   - `{format_command}`
   - `{typecheck_command}`
3. Commit your changes with message: `bd-{issue_id}: <description>` (multiple commits allowed; use the prefix on each). Use `git add <files>` with explicit file paths only (no `-A`, `-u`, `--all`, directories, or globs) and commit in the same command.

**CRITICAL:** Do NOT scope checks to only your modified files. The validation runs on the entire codebase. Fix ALL errors you see, even if you didn't introduce them. Do NOT use `git blame` to decide whether to fix an error.

Note: The orchestrator requires NEW validation evidence - re-run all validations even if you ran them before.
