## Quality Gate Failed (Attempt {attempt}/{max_attempts})

The quality gate check failed with the following issues:
{failure_reasons}

**Required actions:**
1. Fix the issues listed above
2. Re-run the full validation suite:
   - `uv run pytest`
   - `uvx ruff check .`
   - `uvx ruff format .`
   - `uvx ty check`
3. Commit your changes with message: `bd-{issue_id}: <description>`

Note: The orchestrator requires NEW validation evidence - re-run all validations even if you ran them before.
