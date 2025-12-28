## Run-Level Validation Failed (Attempt {attempt}/{max_attempts})

The run-level validation (Gate 4) found issues that need to be fixed:

{failure_output}

**Required actions:**
1. Analyze the validation failure output above
2. Identify and fix all issues causing the failure
3. Re-run the full validation suite:
   - `uv run pytest`
   - `uvx ruff check .`
   - `uvx ruff format .`
   - `uvx ty check`
4. Commit your changes with message: `bd-run-validation: <description>`

**Context:**
- This is a run-level validation that runs after all per-issue work is complete
- Your fix should address the root cause, not just suppress the error
- The orchestrator will re-run validation after your fix

Do not release any locks - the orchestrator handles that.
