## Quality Gate Failed (Attempt {attempt}/{max_attempts})

The quality gate check failed with the following issues:
{failure_reasons}

**Required actions:**
1. Fix ALL issues causing validation failures - including pre-existing errors in files you didn't touch
2. Re-run the full validation suite on the ENTIRE codebase:
   - `uv run pytest`
   - `uvx ruff check .`
   - `uvx ruff format .`
   - `uvx ty check`
3. Commit your changes with message: `bd-{issue_id}: <description>`

**CRITICAL:** Do NOT scope checks to only your modified files. The validation runs on the entire codebase. Fix ALL errors you see, even if you didn't introduce them. Do NOT use `git blame` to decide whether to fix an error.

Note: The orchestrator requires NEW validation evidence - re-run all validations even if you ran them before.
